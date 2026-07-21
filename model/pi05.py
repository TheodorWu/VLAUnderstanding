import torch
import torch.nn as nn
import re

from lerobot.policies.pi05.modeling_pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.configs.policies import PolicyFeature, FeatureType, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from utils.general import printable_params, rprint_architecture

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32


@printable_params
@rprint_architecture
class PI05Wrapper(nn.Module):
    def __init__(self, config, dataset_stats, device=torch.device("cuda")):
        super(PI05Wrapper, self).__init__()
        self.config = config
        model_id = config.get("model_id", "lerobot/pi05_base")
        self.device = device

        assert dataset_stats is not None, "dataset_stats must be provided to PI05Wrapper for image feature registration."

        if model_id is None:
            print("model_id set to None explicitly, loading without weights.")
            pi_config = PI05Config(
                max_action_dim=DUMMY_ACTION_DIM,
                max_state_dim=DUMMY_STATE_DIM,
                dtype="bfloat16",
                image_resolution=(224, 224)
            )
            self.model = PI05Policy(pi_config)
            self.model.to(device)
        else:
            print(f"Loading PI05 model from {model_id}")
            pi_config = PreTrainedConfig.from_pretrained(model_id)
            pi_config.compile_model = False  # Disable torch.compile for now due to issues with nnsight tracing
            self.model = PI05Policy.from_pretrained(model_id, config=pi_config, device_map=device)

        self._patch_rmsnorm_layers()

        self._register_image_features(dataset_stats)

        # todo: check if I also have to register other features like state and action too.

        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config=self.model.config, dataset_stats=dataset_stats
        )

        self._init_tracing_layers()

        self.fixed_time = self._init_fixed_time()
        self.fixed_noise = None

        self._freeze_non_tracing_parameters()

    def _freeze_non_tracing_parameters(self):
        # freeze all non gemma_expert parameters
        self.model.model.paligemma_with_expert.paligemma.requires_grad_(False)

    def _get_fixed_noise(self, shape, device):
        if self.fixed_noise is None:
            batch_independent_shape = 1, *shape[1:]  # Exclude batch dimension
            self.fixed_noise = self.model.model.sample_noise(batch_independent_shape, device)

        return self.fixed_noise.expand(shape[0], *self.fixed_noise.shape[1:]).to(device)

    def _init_fixed_time(self):
        if (value := self.config.get("fixed_time", None)) is not None:
            assert 0.0 <= value <= 1.0
            return torch.tensor(value, device=self.device)
        return None

    def _patch_rmsnorm_layers(self):
        for module in self.model.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight"):
                    module._nn_weight = module.weight
                    continue

                if hasattr(module, "scale"):
                    module.weight = module.scale
                    module._nn_weight = module.scale
                    continue

                dense_weight = getattr(getattr(module, "dense", None), "weight", None)
                fallback_dtype = dense_weight.dtype if dense_weight is not None else torch.float16
                fallback_device = dense_weight.device if dense_weight is not None else self.device
                module.weight = torch.zeros(module.dim, dtype=fallback_dtype, device=fallback_device)
                module._nn_weight = module.weight

    def _register_image_features(self, dataset_stats):
        """
        Ensure config.input_features contains VISUAL entries for every
        observation.images.* key present in dataset_stats.

        from_pretrained sometimes strips input_features during JSON round-tripping,
        and the manual-construction path never sets them at all.
        """
        h, w = self.model.config.image_resolution
        registered = []

        for key in dataset_stats:
            if key.startswith("observation.images.") and key not in self.model.config.input_features:
                self.model.config.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, h, w),
                )
                registered.append(key)

        if not registered:
            raise ValueError(
                "No 'observation.images.*' keys found in dataset_stats, so image features "
                f"cannot be inferred. dataset_stats keys: {list(dataset_stats.keys())}"
            )

        print(f"Registered {len(registered)} image feature(s) from dataset_stats: {registered}")

    def forward(self, processed_batch, loss_reduction="mean"):
        # batch should already be preprocessed before calling forward
        # Wrapper forward replaces model forward for attribution patching to work and get rid of unnecessary statements that crash nnsights tracing
        # Prepare inputs
        images, img_masks = self.model._preprocess_images(processed_batch)
        tokens, masks = processed_batch[f"{OBS_LANGUAGE_TOKENS}"], processed_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.prepare_action(processed_batch)

        if self.config.get("use_fixed_noise", False):
            noise = self._get_fixed_noise(actions.shape, actions.device)
        else:
            noise = self.model.model.sample_noise(actions.shape, actions.device)

        if self.fixed_time is not None:
            time = self.fixed_time.expand(actions.shape[0])
        else:
            time = self.model.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        # Truncate losses to actual action dimensions
        original_action_dim = self.model.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        # Default: return scalar mean loss
        if loss_reduction == "mean":
            loss = losses.mean()
            return loss
        elif loss_reduction == "sample_mean":
            loss = losses.mean(dim=tuple(range(1, losses.dim())))
            return loss
        else:
            return losses

    def preprocess_batch(self, batch):
        """Preprocess a raw batch for model input.

        Args:
            batch: Dictionary containing raw batch data

        Returns:
            Preprocessed batch ready for model forward pass
        """
        return self.preprocessor(batch)

    def postprocess_output(self, raw_output):
        output = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        return self.postprocessor(output)

    def get_tokenizer(self):
        return self.preprocessor

    def _init_tracing_layers(self):
        all_layer_names = [name for name, _ in self.model.named_modules() if "self_attn" in name]
        # NNsight wraps PI05Wrapper -> model (PI05Policy) -> model (PI05Pytorch), so paths that reach
        # paligemma_with_expert need the extra `model.` prefix before that branch.
        self.tracing_layers = [f"model.{name}" for name in all_layer_names if "gemma_expert" in name and "o_proj" in name]
        if not self.tracing_layers:
            raise ValueError(
                f"No tracing layers found matching 'gemma_expert'. "
                f"Available attention layers: {all_layer_names[:5]}..."
            )

    def get_tracing_layers(self):
        return self.tracing_layers

    @staticmethod
    def sort_layers(layer_names: list[str]) -> list[str]:
        return sorted(layer_names, key=lambda name: tuple(int(n) for n in re.findall(r'\d+', name)))
