from dataclasses import dataclass
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from eval.logger import Logger
from data.activation_reader import ActivationReader
from utils.general import add_batch_dim


@dataclass
class LayerReservoir:
    layer: str
    n_samples: int
    fields: list[str]
    data: dict[str, np.ndarray]   # field -> (n_samples, d_model)
    sample_ids: list[str]
    n_total_seen: int


@dataclass
class PCAResult:
    layer: str
    clean: np.ndarray              # (n_samples, n_components)
    corrupt: np.ndarray | None
    explained_variance: np.ndarray
    pca: PCA
    sample_ids: list[str]


@dataclass
class CKAResult:
    layer_a: str
    layer_b: str
    score: float
    n_samples: int


class ReservoirEvaluator:
    def __init__(self, config, layer_sort_fn=None):
        self.logger = Logger()
        self.config = config
        self.evaluator_config = config.get("evaluator", {})
        self.activation_reader = ActivationReader(config)
        self.layer_sort_fn = layer_sort_fn or (lambda x: x)
        self.save_path = (
            Path(self.evaluator_config["save_path"])
            if self.evaluator_config.get("save_path")
            else None
        )
        self.n_pca_components = self.evaluator_config.get("pca_components", 2)
        self.n_samples = self.evaluator_config.get("n_samples", 1000)

    def build_reservoir(
        self,
        layer: str,
        fields: list[str] = ["clean", "corrupt"],
        pool_mode: str = "mean",
        seed: int = 42,
    ) -> LayerReservoir:
        """Reservoir sample n_samples from a single layer."""
        rng = np.random.default_rng(seed)
        reservoir = {f: [] for f in fields}
        sample_ids = []
        i = 0

        for batch in self.activation_reader.iter_data(layer=layer):
            batch = add_batch_dim(batch)
            pooled = {
                f: self._pool_sequence(getattr(batch, f), mode=pool_mode)
                for f in fields
                if getattr(batch, f) is not None
            }
            batch_size = next(iter(pooled.values())).shape[0]

            for b in range(batch_size):
                row = {f: pooled[f][b] for f in pooled}
                sid = batch.sample_id if isinstance(batch.sample_id, str) else batch.sample_id[b]

                if i < self.n_samples:
                    for f in row:
                        reservoir[f].append(row[f])
                    sample_ids.append(sid)
                else:
                    j = int(rng.integers(0, i + 1))
                    if j < self.n_samples:
                        for f in row:
                            reservoir[f][j] = row[f]
                        sample_ids[j] = sid
                i += 1

        if i < self.n_samples:
            print(f"Warning: only {i} samples available for layer '{layer}', fewer than requested {self.n_samples}")

        return LayerReservoir(
            layer=layer,
            n_samples=min(i, self.n_samples),
            fields=fields,
            data={f: np.stack(reservoir[f]) for f in reservoir if reservoir[f]},
            sample_ids=sample_ids,
            n_total_seen=i,
        )

    def compute_pca(
        self,
        reservoir: LayerReservoir,
        project_dims: int | None = None,
    ) -> PCAResult:
        """
        Fit PCA on clean activations, project clean + corrupt.
        Optionally random-project to project_dims first for very high-d models.
        """
        clean = reservoir.data["clean"]
        corrupt = reservoir.data.get("corrupt", None)

        if project_dims is not None:
            projector = GaussianRandomProjection(n_components=project_dims)
            clean = projector.fit_transform(clean)
            if corrupt is not None:
                corrupt = projector.transform(corrupt)

        pca = PCA(n_components=self.n_pca_components)
        clean_proj = pca.fit_transform(clean)
        corrupt_proj = pca.transform(corrupt) if corrupt is not None else None

        return PCAResult(
            layer=reservoir.layer,
            clean=clean_proj,
            corrupt=corrupt_proj,
            explained_variance=pca.explained_variance_ratio_,
            pca=pca,
            sample_ids=reservoir.sample_ids,
        )

    def compute_cka_from_reservoir(
        self,
        reservoir: LayerReservoir,
    ) -> CKAResult:
        layer = reservoir.layer
        clean = reservoir.data["clean"]
        corrupt = reservoir.data.get("corrupt", None)
        score = self._linear_cka(clean, corrupt)
        return CKAResult(layer_a=f"{layer}/clean", layer_b=f"{layer}/corrupt", score=score, n_samples=self.n_samples)

    def compute_perturbation_cka(
        self,
        layer: str,
    ) -> CKAResult:
        """CKA between clean and corrupt activations at the same layer."""
        reservoir = self.build_reservoir(layer, fields=["clean", "corrupt"])
        return self.compute_cka_from_reservoir(reservoir)

    def compute_all_perturbation_cka(
        self,
        layer_names: list[str],
    ) -> list[CKAResult]:
        results = []
        for layer in layer_names:
            result = self.compute_perturbation_cka(layer=layer)
            print(f"CKA {layer}: {result.score:.3f}")
            results.append(result)
        return results

    def plot_pca(self, result: PCAResult):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(
            result.clean[:, 0], result.clean[:, 1],
            label="clean", alpha=0.5, color="#4575b4", s=15,
        )
        if result.corrupt is not None:
            ax.scatter(
                result.corrupt[:, 0], result.corrupt[:, 1],
                label="corrupt", alpha=0.5, color="#d73027", s=15,
            )

        ev = result.explained_variance
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"PCA — {result.layer}")
        ax.legend()
        plt.tight_layout()
        self._save_and_show(fig, f"pca_{result.layer}")

    def plot_perturbation_cka(self, results: list[CKAResult]):
        """
        Bar chart of clean/corrupt CKA score per layer.
        Score near 1.0 → perturbation had little effect.
        Score near 0.0 → perturbation significantly changed the representation.
        """
        layer_names = [r.layer_a.replace("/clean", "") for r in results]
        scores = [r.score for r in results]

        fig, ax = plt.subplots(figsize=(8, len(layer_names) * 0.4 + 1))
        colors = ["#d73027" if s < 0.5 else "#4575b4" for s in scores]

        ax.barh(layer_names, scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("CKA(clean, corrupt)")
        ax.set_title("Perturbation Effect per Layer\n(lower = more affected by perturbation)")
        ax.invert_yaxis()
        plt.tight_layout()

        self._save_and_show(fig, "perturbation_cka")

    def _pool_sequence(self, tensor: torch.Tensor, mode: str = "mean") -> np.ndarray:
        if mode == "mean":
            pooled = tensor.mean(dim=1)
        elif mode == "last":
            pooled = tensor[:, -1, :]
        elif mode == "first":
            pooled = tensor[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        return pooled.detach().cpu().numpy() if isinstance(pooled, torch.Tensor) else pooled

    @staticmethod
    def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        hsic_xy = np.linalg.norm(X.T @ Y) ** 2
        hsic_xx = np.linalg.norm(X.T @ X) ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y) ** 2
        return float(hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy)))

    def _save_and_show(self, fig, name: str):
        if self.evaluator_config.get("save_to_wandb"):
            wandb.log({name: wandb.Image(fig)})
        if self.save_path:
            plt.savefig(self.save_path / f"{name}.svg", dpi=150, bbox_inches="tight")
        if self.evaluator_config.get("show"):
            plt.show()
        plt.close(fig)
