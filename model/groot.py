import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.configs.policies import PolicyFeature, FeatureType
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_IMAGES, OBS_STATE
from lerobot.policies.utils import get_device_from_parameters

from transformers.feature_extraction_utils import BatchFeature

from utils.general import printable_params, rprint_architecture
import dataclasses
print([f.name for f in dataclasses.fields(GrootConfig)])
def _groot_features(
    state_dim: int, action_dim: int
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )

@printable_params
@rprint_architecture
class GROOTWrapper(nn.Module):
    def __init__(self, config, dataset_stats, device=torch.device("cuda")):
        super(GROOTWrapper, self).__init__()
        self.config = config
        self.device = device

        model_id = config.get("model_id", "nvidia/gr00t17-lerobot-libero_object-640")
        if model_id is None:
            print("model_id set to None explicitly, loading without weights.")
            input_features, output_features = _groot_features(state_dim=8, action_dim=7)
            groot_config = GrootConfig(
                        input_features=input_features,
                        output_features=output_features,
                        device=device)
            self.model = GrootPolicy(groot_config)  # constructs via GR00TN17(config) internally
        else:
            print(f"Loading GR00T N1.7 model from {model_id}")
            local_dir = self._fix_pretrained_config(model_id)
            self.model = GrootPolicy.from_pretrained(local_dir,
                                                     device_map=device)  # constructs via GR00TN17(config) internally

        self.model.to(device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config, pretrained_path=model_id, dataset_stats=dataset_stats
        )

        self.fixed_time = self._init_fixed_time()

        self._init_tracing_layers()
        self._freeze_non_tracing_parameters()

    def _fix_pretrained_config(self, model_id):
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=model_id)
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        print("Before:", cfg.get("base_model_path"))
        cfg["base_model_path"] = "nvidia/GR00T-N1.7-3B"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("After:", cfg["base_model_path"])
        return local_dir


    def _init_tracing_layers(self):
        all_layer_names = [name for name, _ in self.model.named_modules() if "attn" in name]
        # NNsight wraps PI05Wrapper -> model (PI05Policy) -> model (PI05Pytorch), so paths that reach
        # paligemma_with_expert need the extra `model.` prefix before that branch.
        self.tracing_layers = [f"model.{name}" for name in all_layer_names if "action_head" in name and name.endswith("to_out.0") and "vl_self_attention" not in name]
        print(f"Tracing layers found: {self.tracing_layers}")

        if not self.tracing_layers:
            raise ValueError(
                f"No tracing layers found matching 'action_head.*to_out.0'. "
                f"Available attention layers: {all_layer_names[:5]}..."
            )

    def _freeze_non_tracing_parameters(self):
        # freeze all non action_head parameters
        self.model._groot_model.backbone.requires_grad_(False)

    def _init_fixed_time(self):
        if (value := self.config.get("fixed_time", None)) is not None:
            assert 0.0 <= value <= 1.0
            return torch.tensor(value, device=self.device)
        return None

    def preprocess_batch(self, batch):
        """Preprocess a raw batch for model input.

        Args:
            batch: Dictionary containing raw batch data

        Returns:
            Preprocessed batch ready for model forward pass
        """
        return self.preprocessor(batch)

    def select_action(self, processed_batch):
        return self.model.select_action(processed_batch)  # TODO: untested

    def forward(self, batch, loss_reduction="mean"):
        """Forward pass through the model.

        Args:
            batch: Preprocessed batch data
            loss_reduction: How to reduce the loss

        Returns:
            Model output
        """
        groot_inputs = self.model._filter_groot_inputs(batch, include_action=True)

        # Get device from model parameters
        device = get_device_from_parameters(self)

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.model.config.use_bf16):
            outputs = self.groot_model_forward(groot_inputs)

        action_loss = outputs["action_loss"]
        action_mask = outputs["action_mask"]

        # Isaac-GR00T returns a BatchFeature;
        if loss_reduction == "mean":
            loss = action_loss.sum(dim=tuple(range(1, action_loss.dim()))) / (action_mask.sum(dim=tuple(range(1, action_mask.dim()))) + 1e-6)
            return loss.mean()
        elif loss_reduction == "sample_mean":
            loss = action_loss.sum(dim=tuple(range(1, action_loss.dim()))) / (action_mask.sum(dim=tuple(range(1, action_mask.dim()))) + 1e-6)
            return loss
        else:
            return action_loss.sum() / (action_mask.sum() + 1e-6)

    def groot_model_forward(self, groot_inputs):
        """Forward pass through the GR00T model.

        Args:
            groot_inputs: Dictionary containing preprocessed inputs for the GR00T model

        Returns:
            Model output
        """
        groot = self.model._groot_model
        backbone_inputs, action_inputs = groot.prepare_input(groot_inputs)
        backbone_outputs = groot.backbone(backbone_inputs)
        return self.action_head(backbone_outputs, action_inputs)

    def action_head(self, backbone_output, action_input):
        """Forward pass through the action head of the GR00T model.
        """
        action_head = self.model._groot_model.action_head
        action_head.set_frozen_modules_to_eval_mode()
        backbone_output = action_head.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        if action_input.state.shape[1] != action_head.config.state_history_length:
            raise ValueError("state history length does not match GR00T N1.7 config.")
        state = action_input.state.view(action_input.state.shape[0], 1, -1)
        state_features = action_head.state_encoder(state, embodiment_id)

        if action_head.training and action_head.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device) < action_head.state_dropout_prob
            )
            state_features = state_features * (1 - do_dropout[:, None, None].to(dtype=state_features.dtype))

        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if self.fixed_time is not None:
            t = self.fixed_time.expand(actions.shape[0]).to(device=actions.device, dtype=actions.dtype)
        else:
            t = action_head.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * action_head.num_timestep_buckets).long()
        action_features = action_head.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        if action_head.config.use_alternate_vl_dit:
            model_output, _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output, _ = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        return BatchFeature(
            data={
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    def get_tracing_layers(self):
        return self.tracing_layers
