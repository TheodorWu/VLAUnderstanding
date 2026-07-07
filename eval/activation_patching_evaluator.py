from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wandb

from eval.logger import Logger
from data.activation_reader import ActivationReader
from utils.display import layer_display_name

class PatchingResult:
    def __init__(self, perturbation_type, layer_names, scalar_scores, layer_samples):
        self.perturbation_type = perturbation_type
        self.layer_names = layer_names
        self.scalar_scores = scalar_scores
        self.layer_samples = layer_samples

class ActivationPatchingEvaluator:
    def __init__(self, config, layer_sort_fn=None):
        self.logger = Logger()
        self.config = config
        self.evaluator_config = config.get("evaluator", {}).get("activation_patching_evaluator", {})
        self.activation_reader = ActivationReader(self.evaluator_config or config)
        self.layer_sort_fn = layer_sort_fn or (lambda x: x)
        self.save_path = Path(self.evaluator_config.get("save_path", None)) if self.evaluator_config.get("save_path", None) else None

    def compute_layer_patching_effects(self) -> PatchingResult:
        running_sum = {}
        sample_count = {}
        layer_samples = {}

        for batch in self.activation_reader.iter_data():
            layer = batch.layer

            if batch.patching_effect is None:
                print(f"Skipping layer '{layer}': no patching effect available")
                continue

            effect = np.asarray(batch.patching_effect)  # (batch,)

            if layer not in running_sum:
                running_sum[layer] = effect.sum()
                sample_count[layer] = effect.shape[0]
                layer_samples[layer] = []
            else:
                running_sum[layer] += effect.sum()
                sample_count[layer] += effect.shape[0]

            layer_samples[layer].extend(effect.tolist())

        layer_names = self.layer_sort_fn(list(running_sum.keys()))
        scores = np.array([running_sum[l] / sample_count[l] for l in layer_names])

        return PatchingResult(
            perturbation_type=self.activation_reader.metadata.get("perturbation_type"),
            layer_names=layer_names,
            scalar_scores=scores,
            layer_samples=layer_samples,
        )

    def plot_patching_heatmap(self, result: PatchingResult, invert=True):
        layer_names = result.layer_names
        scores = result.scalar_scores.copy()

        if invert:
            scores = 1 - scores
            cbar_label = "Causal importance (1 - normalized patching effect)"
        else:
            cbar_label = "Normalized patching effect (0=clean-like, 1=corrupted-like)"

        matrix = scores[np.newaxis, :]  # (1, n_layers)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)

        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=90, fontsize=7)
        ax.set_yticks([0])
        ax.set_yticklabels([result.perturbation_type])
        ax.set_xlabel("Layer")
        ax.set_title(f"Activation patching effect per layer ({result.perturbation_type})")
        fig.colorbar(im, ax=ax, label=cbar_label)
        plt.tight_layout()
        self._save_and_show(fig, f"patching_heatmap_{result.perturbation_type}")

    def plot_patching_distribution(self, result: PatchingResult, invert=True):
        fig, ax = plt.subplots(figsize=(12, 4))
        data = [
            (1 - np.array(result.layer_samples[l])) if invert else np.array(result.layer_samples[l])
            for l in result.layer_names
        ]
        ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, len(result.layer_names) + 1))
        ax.set_xticklabels(result.layer_names, rotation=90, fontsize=7)
        ax.set_ylabel("Causal importance" if invert else "Normalized patching effect")
        ax.set_title(f"Per-sample patching effect distribution ({result.perturbation_type})")
        plt.tight_layout()
        self._save_and_show(fig, f"patching_distribution_{result.perturbation_type}")

    def plot_atp_vs_patching(self, atp_result, patching_result, invert_patching=True):
        fig, ax = plt.subplots(figsize=(6, 6))

        atp_scores = dict(zip(atp_result.layer_names, atp_result.scalar_scores))
        patch_scores = dict(zip(patching_result.layer_names, patching_result.scalar_scores))

        common_layers = [l for l in atp_result.layer_names if l in patch_scores]
        x = np.array([atp_scores[l] for l in common_layers])
        y = np.array([patch_scores[l] for l in common_layers])
        if invert_patching:
            y = 1 - y

        ax.scatter(x, y)
        for i, l in enumerate(common_layers):
            ax.annotate(layer_display_name(l), (x[i], y[i]), fontsize=6)  # short layer label

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (perfect approximation)")
        ax.set_xlabel("AtP score")
        ax.set_ylabel("Activation patching (causal importance)")
        ax.legend()
        plt.tight_layout()
        self._save_and_show(fig, f"atp_vs_patching_{atp_result.perturbation_type}_{patching_result.perturbation_type}")

    def _save_and_show(self, fig, name: str):
        if self.evaluator_config.get("save_to_wandb"):
            wandb.log({name: wandb.Image(fig)})
        if self.save_path:
            plt.savefig(self.save_path / f"{name}.svg", dpi=150, bbox_inches="tight")
        if self.evaluator_config.get("show"):
            plt.show()
        plt.close(fig)
