import torch

from eval.logger import Logger
from data.activation_reader import ActivationReader
import numpy as np
from dataclasses import dataclass
import einops
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from utils.general import pad_to_length

@dataclass
class AttributionResult:
    perturbation_type: str
    layer_names: list[str]
    matrix: np.ndarray                  # shape: (n_layers, n_heads) — already aggregated
    scalar_scores: np.ndarray = None              # Optional overall score for each layer

class AttributionPatchingEvaluator():
    def __init__(self, config, layer_sort_fn=None):
        self.logger = Logger()
        self.config = config
        self.evaluator_config = config.get("evaluator", {})
        self.activation_reader = ActivationReader(config)
        self.layer_sort_fn = layer_sort_fn or (lambda x: x)

    def _add_batch_dim(self, batch):
        for k in ["clean", "corrupt", "gradients"]:
            if hasattr(batch, k):
                tensor = getattr(batch, k)
                if torch.is_tensor(tensor) and tensor.dim() == 2:
                    setattr(batch, k, tensor.unsqueeze(0))
        return batch

    def _accumulate_token_attr(
        self,
        residual_attr: np.ndarray | torch.Tensor,  # (batch, seq)
        running_sum: np.ndarray | torch.Tensor | None,
    ) -> np.ndarray | torch.Tensor:
        """
        Accumulates residual_attr into running_sum, extending the seq dimension if necessary.
        Pads the shorter array with zeros along axis=1 before adding.
        """
        if running_sum is None:
            return residual_attr.copy()

        current_len = running_sum.shape[0]
        new_len = residual_attr.shape[0]

        if new_len > current_len:
            running_sum = pad_to_length(running_sum, new_len)
        elif current_len > new_len:
            residual_attr = pad_to_length(residual_attr, current_len)

        return running_sum + residual_attr

    def compute_layer_attributions(self) -> AttributionResult:
        running_sum = {}
        sample_count = {}
        running_token_attr = {}

        for batch in self.activation_reader.iter_data():
            batch = self._add_batch_dim(batch)
            layer = batch.layer
            residual_attr = einops.reduce(
                batch.gradients * (batch.clean - batch.corrupt),
                "batch seq d_model -> batch seq",
                "sum",
            )  # (batch, seq)

            print(f"residual_attr shape: {residual_attr.shape}")
            batch_sum = residual_attr.sum(axis=0)  # scalar
            batch_n = residual_attr.shape[0]

            token_attr = self._accumulate_token_attr(residual_attr, running_token_attr.get(layer, None))
            running_token_attr[layer] = token_attr

            if layer not in running_sum:
                running_sum[layer] = batch_sum
                sample_count[layer] = batch_n
            else:
                running_sum[layer] += batch_sum
                sample_count[layer] += batch_n


        layer_names = list(running_sum.keys())
        layer_names = self.layer_sort_fn(layer_names)
        scores = np.array([running_sum[l] / sample_count[l] for l in layer_names])  # (n_layers,)
        matrix = np.array([token_attr[l] / sample_count[l] for l in layer_names])  # (n_layers,)

        return AttributionResult(
            perturbation_type=self.activation_reader.metadata.get("perturbation_type"),
            layer_names=layer_names,
            matrix=matrix,
            scalar_scores=scores,
        )

    def plot_heatmap(
        self,
        result: AttributionResult,
    ):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            result.matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"T{i}" for i in range(result.matrix.shape[1])],
            yticklabels=result.layer_names,
            linewidths=0.3,
            cbar_kws={"label": "Attribution Score"},
        )
        ax.set_title(f"Attribution — {result.perturbation_type}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        plt.tight_layout()

        if self.evaluator_config.get("save_to_wandb"):
            print("Logging heatmap to Weights & Biases...")
            wandb.log({f"attribution_heatmap_{result.perturbation_type}": wandb.Image(fig)})
        if self.evaluator_config.get("save_path"):
            plt.savefig(self.evaluator_config.get("save_path"), dpi=150, bbox_inches="tight")
        if self.evaluator_config.get("show"):
            plt.show()
        plt.close(fig)
        return fig
