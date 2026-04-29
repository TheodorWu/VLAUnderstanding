import torch
import torch.nn as nn

from lerobot.policies.pi05.modeling_pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.processor_pi05 import PI05Config, make_pi05_pre_post_processors

from utils.general import printable_params, rprint_architecture

DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32

@printable_params
@rprint_architecture
class PI05Wrapper(nn.Module):
    def __init__(self, config, dataset_stats, device=torch.device("cuda")):
        super().__init__()
        super(PI05Wrapper, self).__init__()
        self.config = config
        model_id = config.get("model_id", "lerobot/pi05_base")

        if model_id is None:
            print("model_id set to None explicitly, loading without weights.")
            config = PI05Config(max_action_dim=DUMMY_ACTION_DIM, max_state_dim=DUMMY_STATE_DIM, dtype="float32")
            self.model = PI05Policy(config)
        else:
            print(f"Loading PI05 model from {model_id}")
            self.model = PI05Policy.from_pretrained(model_id)

        self.model.to(device)

        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config=self.model.config, dataset_stats=dataset_stats
        )

        self._init_tracing_layers()

    def get_tokenizer(self):
        return self.preprocessor

    def _init_tracing_layers(self):
        all_model_layers = [(name, layer) for name, layer in self.model.named_modules() if "self_attn" in name]

        self.tracing_layers = [(name, layer) for name, layer in all_model_layers if "gemma_expert" in name]
        self.logits_layer = "lm_head"

    def get_tracing_target(self):
        # which activation within a layer
        pass

    def get_output_proxy(self):
        # which output to save for the layer
        pass

    def metric(self):
        # scalar value for backward()
        pass

'''
backbone uses paligemma with expert to get sth called past_key_values which is passed to the action expert for flow matched denoising.

'''
