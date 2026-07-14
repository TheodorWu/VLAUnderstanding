from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wandb

from eval.logger import Logger
from data.activation_reader import ActivationReader
from utils.display import layer_display_name
from utils.general import pretty_print_config

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
        print(f"ActivationPatchingEvaluator config: {self.evaluator_config}")
        pretty_print_config(self.evaluator_config)
        self.activation_reader = ActivationReader(self.evaluator_config or config)
        self.layer_sort_fn = layer_sort_fn or (lambda x: x)
        self.save_path = Path(self.evaluator_config.get("save_path", None)) if self.evaluator_config.get("save_path", None) else None

    def compute_layer_patching_effects(self, denom_quantile: float = 0.1) -> PatchingResult:
        # Pass 1: accumulate raw sums (for the aggregate/point-estimate ratio)
        # and collect raw per-sample losses (for the per-sample ratio distribution).
        clean_sum = {}
        corrupted_sum = {}
        patched_sum = {}
        sample_count = {}

        layer_clean = {}
        layer_corrupted = {}
        layer_patched = {}

        for batch in self.activation_reader.iter_data():
            layer = batch.layer

            if batch.patched_loss is None or batch.clean_loss is None or batch.corrupted_loss is None:
                print(f"Skipping layer '{layer}': missing loss data")
                continue

            clean = np.asarray(batch.clean_loss)      # (batch,)
            corrupted = np.asarray(batch.corrupted_loss)  # (batch,)
            patched = np.asarray(batch.patched_loss)   # (batch,)

            if layer not in clean_sum:
                clean_sum[layer] = 0.0
                corrupted_sum[layer] = 0.0
                patched_sum[layer] = 0.0
                sample_count[layer] = 0
                layer_clean[layer] = []
                layer_corrupted[layer] = []
                layer_patched[layer] = []

            clean_sum[layer] += clean.sum()
            corrupted_sum[layer] += corrupted.sum()
            patched_sum[layer] += patched.sum()
            sample_count[layer] += clean.shape[0]

            layer_clean[layer].extend(clean.tolist())
            layer_corrupted[layer].extend(corrupted.tolist())
            layer_patched[layer].extend(patched.tolist())

        layer_names = self.layer_sort_fn(list(clean_sum.keys()))

        # Aggregate ratio: computed from summed losses, i.e. ratio(means), not mean(ratios).
        # This is the stable, headline point-estimate per layer.
        scores = np.array([
            (patched_sum[l] - clean_sum[l]) / (corrupted_sum[l] - clean_sum[l])
            for l in layer_names
        ])

        # Per-sample ratios: computed for distribution/variance inspection,
        # with unstable (near-zero-denominator) samples masked out rather than
        # blown up or silently clamped.
        layer_samples = {}
        for l in layer_names:
            clean_arr = np.asarray(layer_clean[l])
            corrupted_arr = np.asarray(layer_corrupted[l])
            patched_arr = np.asarray(layer_patched[l])

            denom = corrupted_arr - clean_arr
            threshold = np.quantile(np.abs(denom), denom_quantile) if denom.size else 0.0
            stable_mask = np.abs(denom) > threshold

            ratio = np.full_like(denom, np.nan, dtype=np.float64)
            ratio[stable_mask] = (patched_arr[stable_mask] - clean_arr[stable_mask]) / denom[stable_mask]

            n_dropped = (~stable_mask).sum()
            if n_dropped > 0:
                print(f"Layer '{l}': dropped {n_dropped}/{denom.size} samples with unstable denominator")

            layer_samples[l] = ratio.tolist()  # NaNs preserved; filter downstream as needed

        return PatchingResult(
            perturbation_type=self.activation_reader.metadata.get("perturbation_type"),
            layer_names=layer_names,
            scalar_scores=scores,
            layer_samples=layer_samples,
        )

    def plot_patching_heatmap(self, result: PatchingResult, invert=True):
        layer_names = result.layer_names
        scores = self._filter_nan_inf(np.array(result.scalar_scores))

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
            (1 - self._filter_nan_inf(np.array(result.layer_samples[l]))) if invert else self._filter_nan_inf(np.array(result.layer_samples[l]))
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

    def _filter_nan_inf(self, arr):
        arr = np.asarray(arr)
        return arr[~np.isnan(arr) & ~np.isinf(arr)]
