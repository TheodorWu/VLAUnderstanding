import torch
import torch.nn as nn

from lerobot.policies.pi05.modeling_pi05 import PI05Config, PI05Policy
import lerobot.policies.pi05.modeling_pi05 as pi_module
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.configs.policies import PolicyFeature, FeatureType

from utils.general import printable_params, rprint_architecture

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32

def _patch_policy_init_for_meta(original_init):
    """Wraps PI05Policy.__init__ to disable .to() calls during meta device init,
    preventing meta tensor materialization errors."""
    def patched_init(self_model, config, **kwargs):
        original_to = torch.nn.Module.to
        torch.nn.Module.to = lambda self, *args, **kwargs: self  # no-op
        try:
            original_init(self_model, config, **kwargs)
        finally:
            torch.nn.Module.to = original_to  # always restore, even on exception
    return patched_init


@printable_params
@rprint_architecture
class PI05Wrapper(nn.Module):
    def __init__(self, config, dataset_stats, device=torch.device("cuda")):
        super(PI05Wrapper, self).__init__()
        self.config = config
        model_id = config.get("model_id", "lerobot/pi05_base")

        if model_id is None:
            print("model_id set to None explicitly, loading without weights.")
            pi_config = PI05Config(
                max_action_dim=DUMMY_ACTION_DIM,
                max_state_dim=DUMMY_STATE_DIM,
                dtype="bfloat16"
            )
            pi_module.PI05Policy.__init__ = _patch_policy_init_for_meta(
                pi_module.PI05Policy.__init__
            )
            with torch.device("meta"):
                self.model = PI05Policy(pi_config)
            self.device = torch.device("meta")  # Model starts on CPU since it's uninitialized
        else:
            print(f"Loading PI05 model from {model_id}")
            self.model = PI05Policy.from_pretrained(model_id, device_map=device)

        # self.model.to(device)

        self._patch_rmsnorm_layers()

        self._register_image_features(dataset_stats)

        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config=self.model.config, dataset_stats=dataset_stats
        )

        self._init_tracing_layers()

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
        if self.model.config.image_features:
            return  # Already populated — nothing to do

        h, w = self.model.config.image_resolution
        registered = []

        for key in dataset_stats:
            if key.startswith("observation.images."):
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

    def forward(self, batch):
        # batch should already be preprocessed before calling forward
        return self.model(batch)

    def preprocess_batch(self, batch):
        """Preprocess a raw batch for model input.

        Args:
            batch: Dictionary containing raw batch data

        Returns:
            Preprocessed batch ready for model forward pass
        """
        return self.preprocessor(batch)

    def postprocess_output(self, output):
        """Postprocess model output.

        Args:
            output: Raw model output (possibly a tuple)

        Returns:
            Postprocessed output
        """
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_tokenizer(self):
        return self.preprocessor

    def _init_tracing_layers(self):
        all_layer_names = [name for name, _ in self.model.named_modules() if "self_attn" in name]
        # NNsight wraps PI05Wrapper -> model (PI05Policy) -> model (PI05Pytorch), so paths that reach
        # paligemma_with_expert need the extra `model.` prefix before that branch.
        self.tracing_layers = [f"model.{name}" for name in all_layer_names if "gemma_expert" in name]
        self.logits_layer = "lm_head"

    def get_tracing_layers(self):
        return self.tracing_layers

    def get_output_proxy(self):
        # The traced wrapper exposes the forward return value as model.output.
        return self.model.output

    def metric(self, output):
        # scalar value for backward()
        # L2 Norm for velocity from flow matching expert
        # Logit difference for VL backbone tokens
        return output.sum()
