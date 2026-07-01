import re
import torch
import torch.nn as nn

from model.pi05 import PI05Wrapper
from lerobot.policies.pi05.modeling_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


class DummyAttention(nn.Module):
    """Named `self_attn` on the parent layer. `o_proj` is the hook target."""

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)  # <-- hook onto this

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / x.shape[-1] ** 0.5, dim=-1)
        return self.o_proj(attn @ v)


class DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = DummyAttention(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class TinyVLA(nn.Module):
    def __init__(self, state_dim=8, dim=32, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, dim)
        self.layers = nn.ModuleList([DummyLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, state_dim] -> [B, T, dim]
        for layer in self.layers:
            x = layer(x)
        return x


class DummyVLAWrapper(nn.Module):
    def __init__(self, config, dataset_stats, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.model = TinyVLA(dim=config.get("dim", 32), num_layers=config.get("num_layers", 3)).to(device)
        self._init_tracing_layers()

        pi_config = PI05Config(
            max_action_dim=32,
            max_state_dim=32,
            dtype="bfloat16",
            image_resolution=(224, 224)
        )
        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config=pi_config, dataset_stats=dataset_stats
        )

    def _init_tracing_layers(self):
        self.tracing_layers = [name for name, _ in self.model.named_modules() if name.endswith("o_proj")]
        if not self.tracing_layers:
            raise ValueError("No tracing layers found (expected modules named 'o_proj').")

    def preprocess_batch(self, batch):
        return self.preprocessor(batch)

    def get_tracing_layers(self):
        return self.tracing_layers

    @staticmethod
    def sort_layers(layer_names):
        return sorted(layer_names, key=lambda name: tuple(int(n) for n in re.findall(r"\d+", name)))

    @staticmethod
    def transform_obs_to_batch(obs, task):
        return PI05Wrapper.transform_obs_to_batch(obs, task)  # Reuse the same transformation logic as PI05Wrapper

    def forward(self, batch):
        x = batch["observation.state"] # [B, state_dim] e.g. [1, 8]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.unsqueeze(1)               # [B, 1, state_dim] -> one token
        out = self.model(x)
        return out.mean()

    def select_action(self, processed_batch):
        """Select an action based on the processed batch. For the dummy model, we can just return a random action."""
        return [0.] * 7
