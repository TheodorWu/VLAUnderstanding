import torch

from eval.logger import Logger
from data.activation_reader import ActivationReader
import numpy as np
from dataclasses import dataclass
import einops
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

@dataclass
class AttributionResult:
    perturbation_type: str
    layer_names: list[str]
    matrix: np.ndarray                  # shape: (n_layers, n_heads) — already aggregated

class AttributionPatchingEvaluator():
    def __init__(self, config):
        self.logger = Logger()
        self.config = config
        self.activation_reader = ActivationReader(config)

    def _add_batch_dim(self, batch):
        for k in ["clean", "corrupt", "gradients"]:
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dim() == 2:
                setattr(batch, k, getattr(batch, k).unsqueeze(0))
        return batch

    def compute_layer_attributions(self) -> AttributionResult:
        running_sum = {}
        sample_count = {}

        for batch in self.activation_reader.iter_data():
            batch = self._add_batch_dim(batch)
            layer = batch.layer
            residual_attr = einops.reduce(
                batch.gradients * (batch.clean - batch.corrupt),
                "batch seq d_model -> batch seq",
                "sum",
            )  # (batch, seq)

            batch_sum = residual_attr.sum(axis=0)  # scalar
            batch_n = residual_attr.shape[0]

            if layer not in running_sum:
                running_sum[layer] = batch_sum
                sample_count[layer] = batch_n
            else:
                running_sum[layer] += batch_sum
                sample_count[layer] += batch_n

        layer_names = list(running_sum.keys())
        scores = np.array([running_sum[l] / sample_count[l] for l in layer_names])  # (n_layers,)

        return AttributionResult(
            perturbation_type=self.activation_reader.metadata.get("perturbation_type"),
            layer_names=layer_names,
            matrix=scores,
        )

    def plot_heatmap(
        self,
        result: AttributionResult,
        save_path: str = None,
        show: bool = False,
        save_to_wandb: bool = False
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

        if save_to_wandb:
            wandb.log({f"attribution_heatmap_{result.perturbation_type}": wandb.Image(fig)})
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig
