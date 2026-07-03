import torch

from pathlib import Path
from eval.logger import Logger
from data.activation_reader import ActivationReader
import numpy as np
from dataclasses import dataclass
import einops
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from utils.general import pad_to_length, add_batch_dim
from utils.display import layer_display_name

@dataclass
class AttributionResult:
    perturbation_type: str
    layer_names: list[str]
    matrix: np.ndarray                  # shape: (n_layers, seq)
    std_matrix: np.ndarray                  # (n_layers, seq) — std dev attribution
    scalar_scores: np.ndarray = None              # Optional overall score for each layer
    norm_matrix: np.ndarray = None  # (n_layers, seq) — mean residual stream norm
    layer_samples: dict[str, list[float]] = None  # Optional per-layer list of per-sample scores

class AttributionPatchingEvaluator():
    def __init__(self, config, layer_sort_fn=None):
        self.logger = Logger()
        self.config = config
        self.evaluator_config = config.get("evaluator", {})
        self.activation_reader = ActivationReader(config)
        self.layer_sort_fn = layer_sort_fn or (lambda x: x)
        self.save_path = Path(self.evaluator_config.get("save_path", None)) if self.evaluator_config.get("save_path", None) else None

    def _accumulate_token_sq_attr(
        self,
        residual_attr: np.ndarray | torch.Tensor,  # (batch, seq)
        running_sq_sum: np.ndarray | torch.Tensor | None,
    ) -> np.ndarray | torch.Tensor:
        sq = (residual_attr ** 2).sum(axis=0)  # (seq,)

        if running_sq_sum is None:
            if isinstance(sq, torch.Tensor):
                return sq.clone()
            else:
                return sq.copy()

        current_len = running_sq_sum.shape[0]
        new_len = sq.shape[0]

        if new_len > current_len:
            running_sq_sum = pad_to_length(running_sq_sum, new_len)
        elif current_len > new_len:
            sq = pad_to_length(sq, current_len)

        return running_sq_sum + sq

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
            if isinstance(residual_attr, torch.Tensor):
                return residual_attr.clone()
            else:
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
        token_count = {}
        running_token_attr = {}
        running_token_sq_attr = {}
        running_token_norm = {}
        layer_samples = {}  # layer -> list of per-sample scalars

        for batch in self.activation_reader.iter_data():
            batch = add_batch_dim(batch)
            layer = batch.layer
            if batch.gradients is None:
                print(f"Skipping layer '{batch.layer}': no gradients available")
                continue
            residual_attr = einops.reduce(
                batch.gradients * (batch.clean - batch.corrupt),
                "batch seq d_model -> batch seq",
                "sum",
            )  # (batch, seq)

            print(f"residual_attr shape: {residual_attr.shape}")
            batch_sum = residual_attr.sum(axis=0)
            batch_n = residual_attr.shape[0]

            token_attr = self._accumulate_token_attr(batch_sum, running_token_attr.get(layer, None)) # (seq,)
            running_token_attr[layer] = token_attr

            sq_attr = self._accumulate_token_sq_attr(residual_attr, running_token_sq_attr.get(layer, None))
            running_token_sq_attr[layer] = sq_attr

            delta = batch.clean - batch.corrupt  # (batch, seq, d_model)
            delta_norm = torch.norm(delta, dim=-1)  # (batch, seq) — L2 norm over d_model

            norm_attr = self._accumulate_token_attr(
                delta_norm.sum(axis=0),
                running_token_norm.get(layer, None)
            )
            running_token_norm[layer] = norm_attr

            scalar_sum = batch_sum.sum()
            token_n = batch_sum.shape[0]

            if layer not in running_sum:
                running_sum[layer] = scalar_sum
                sample_count[layer] = batch_n
                token_count[layer] = batch_n * token_n
            else:
                running_sum[layer] += scalar_sum
                sample_count[layer] += batch_n
                token_count[layer] += batch_n * token_n

            if layer not in layer_samples:
                layer_samples[layer] = []
            per_sample_scores = residual_attr.mean(axis=1)  # (batch,)
            layer_samples[layer].extend(per_sample_scores.tolist())

        layer_names = list(running_sum.keys())
        layer_names = self.layer_sort_fn(layer_names)
        scores = np.array([running_sum[l] / token_count[l] for l in layer_names])  # (n_layers,)
        matrix = np.array([running_token_attr[l] / sample_count[l] for l in layer_names])  # (n_layers, seq)
        norm_matrix = np.array([running_token_norm[l] / sample_count[l] for l in layer_names])  # (n_layers, seq)

        # for i, l in enumerate(layer_names):
        #     mean = running_token_attr[l] / sample_count[l]          # E[x]
        #     mean_sq = running_token_sq_attr[l] / sample_count[l]    # E[x²]
        #     variance = mean_sq - mean ** 2                           # Var[x]
        #     print(f"mean range: {mean.min():.3f} to {mean.max():.3f}")
        #     print(f"mean_sq range: {mean_sq.min():.3f} to {mean_sq.max():.3f}")
        #     print(f"variance range: {variance.min():.3f} to {variance.max():.3f}")
        #     print(f"matrix min: {matrix[i].min():.3f}, matrix_max: {matrix[i].max():.3f}, mean: {mean.mean():.3f}, std: {mean.std():.3f}")

        std_matrix = np.array([
            np.sqrt(np.maximum(
                running_token_sq_attr[l] / sample_count[l] - (running_token_attr[l] / sample_count[l]) ** 2,
                0
            ))
            for l in layer_names
        ])  # (n_layers, seq)

        return AttributionResult(
            perturbation_type=self.activation_reader.metadata.get("perturbation_type"),
            layer_names=layer_names,
            matrix=matrix,
            scalar_scores=scores,
            std_matrix=std_matrix,
            norm_matrix=norm_matrix,
            layer_samples=layer_samples
        )

    def plot_heatmap(
        self,
        result: AttributionResult,
        std: bool = False,
    ):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            result.matrix if not std else result.std_matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=[f"T{i}" for i in range(result.matrix.shape[1])],
            yticklabels=[layer_display_name(l) for l in result.layer_names],
            linewidths=0.3,
            cbar_kws={"label": "Attribution Score" if not std else "Std Dev of Attribution Score"},
        )
        ax.set_title(f"Attribution — {result.perturbation_type}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        plt.tight_layout()

        name = f"attribution_heatmap_{result.perturbation_type}" if not std else f"attribution_std_heatmap_{result.perturbation_type}"
        self._save_and_show(fig, name)

    def plot_layer_scores(
        self,
        result: AttributionResult,
    ):
        fig, ax = plt.subplots(figsize=(8, len(result.layer_names) * 0.4 + 1))

        scores = result.scalar_scores
        colors = ["#d73027" if s > 0 else "#4575b4" for s in scores]

        ax.barh([layer_display_name(l) for l in result.layer_names], scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Attribution Score")
        ax.set_title(f"Per-Layer Attribution — {result.perturbation_type}")
        ax.invert_yaxis()  # layer 0 at top, matches heatmap orientation
        plt.tight_layout()

        self._save_and_show(fig, f"layer_scores_{result.perturbation_type}")

    def plot_norm_heatmap(
        self,
        result: AttributionResult,
    ):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            result.norm_matrix,
            ax=ax,
            cmap="Reds",
            center=None,
            xticklabels=[f"T{i}" for i in range(result.norm_matrix.shape[1])],
            yticklabels=[layer_display_name(l) for l in result.layer_names],
            linewidths=0.3,
            cbar_kws={"label": "Residual Stream Norm"},
        )
        ax.set_title(f"Residual Stream Norm — {result.perturbation_type}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        plt.tight_layout()

        self._save_and_show(fig, f"norm_heatmap_{result.perturbation_type}")

    def plot_layer_distributions(
        self,
        result: AttributionResult,
        kind: str = "violin",  # "violin" | "box"
    ):
        fig, ax = plt.subplots(figsize=(8, len(result.layer_names) * 0.4 + 1))

        data = [result.layer_samples[l] for l in result.layer_names]

        if kind == "violin":
            parts = ax.violinplot(data, vert=False, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("#4575b4")
                pc.set_alpha(0.7)
        else:
            ax.boxplot(data, vert=False, patch_artist=True)

        ax.set_yticks(range(1, len(result.layer_names) + 1))
        ax.set_yticklabels([layer_display_name(l) for l in result.layer_names])
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Attribution Score")
        ax.set_title(f"Per-Layer Attribution Distribution — {result.perturbation_type}")
        ax.invert_yaxis()
        plt.tight_layout()

        self._save_and_show(fig, f"layer_distributions_{result.perturbation_type}")

    def plot_sample_metadata_dist(self):
        sample_metadata = self.activation_reader.get_all_sample_metadata()
        perturbed_indices = []
        for i, m in sample_metadata.items():
            for idx in m.get("perturbed_token_idxs", []):
                perturbed_indices.append(idx)

        if not perturbed_indices:
            print("No perturbed indices found in sample metadata.")
            return

        perturbed_indices = np.array(perturbed_indices)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(perturbed_indices, bins=range(perturbed_indices.min(), perturbed_indices.max() + 2),
                    color="#4575b4", edgecolor="white", linewidth=0.5)
        axes[0].set_xlabel("Token Position")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Perturbed Token Indices")

        # CDF — useful for seeing if perturbations are front/back loaded
        sorted_indices = np.sort(perturbed_indices)
        cdf = np.arange(1, len(sorted_indices) + 1) / len(sorted_indices)
        axes[1].plot(sorted_indices, cdf, color="#4575b4")
        axes[1].set_xlabel("Token Position")
        axes[1].set_ylabel("Cumulative Fraction")
        axes[1].set_title("CDF of Perturbed Token Indices")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Perturbed Index Distribution — {len(sample_metadata)} samples, "
                    f"{len(perturbed_indices)} total perturbations", fontsize=11)
        plt.tight_layout()
        self._save_and_show(fig, "perturbed_index_dist")
        return fig

    def _save_and_show(self, fig, name):
        if self.evaluator_config.get("save_to_wandb"):
            wandb.log({name: wandb.Image(fig)})
        if self.save_path:
            plt.savefig(self.save_path / f"{name}.svg", dpi=150, bbox_inches="tight")
        if self.evaluator_config.get("show"):
            plt.show()
        plt.close(fig)
