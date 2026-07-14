import io
from pathlib import Path
import json
import torch
import wandb
import webdataset as wds
from dataclasses import dataclass
from typing import Optional

from utils.perturbator import build_perturbation_type_key

@dataclass
class SampleMetadata:
    sample_id: int
    instruction: Optional[str] = None
    corrupt_instruction: Optional[str] = None
    perturbed_token_idxs: Optional[list[int]] = None

class ActivationDataBatch:
    # might be useful for grouping activations/gradients from multiple samples together before writing to disk in attribution patching loop
    def __init__(self, layer, sample_ids, clean=None, corrupt=None, gradients=None, patched_loss=None, clean_loss=None, corrupted_loss=None):
        self.data_points = [
            ActivationDataPoint(
                layer=layer,
                sample_id=sid,
                clean=clean[i] if clean is not None else None,
                corrupt=corrupt[i] if corrupt is not None else None,
                gradients=gradients[i] if gradients is not None else None,
                patched_loss=patched_loss[i] if patched_loss is not None else None,
                clean_loss=clean_loss[i] if clean_loss is not None else None,
                corrupted_loss=corrupted_loss[i] if corrupted_loss is not None else None,
            )
            for i, sid in enumerate(sample_ids)
        ]

    def __iter__(self):
        yield from self.data_points

@dataclass
class ActivationDataPoint:
    layer: str
    sample_id: int
    clean: Optional[torch.Tensor] = None
    corrupt: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    patched_loss: Optional[torch.Tensor] = None
    clean_loss: Optional[torch.Tensor] = None
    corrupted_loss: Optional[torch.Tensor] = None

    def __iter__(self):
        yield self

class ActivationWriter():
    def __init__(self, config):
        self.config = config.get("activation_writer", {})
        self.chunk_size = self.config.get("chunk_size", 16)
        self.max_shard_size = self.config.get("max_shard_size", 1e9)  # 1GB per shard
        self.run_name = wandb.run.name if wandb.run else "test_run"
        self.sinks = {}  # keyed by (layer, data_type)

        self.metadata = {
            "chunk_size": self.chunk_size,
            "max_shard_size": self.max_shard_size,
            "run_name": self.run_name,
            "perturbation_type": build_perturbation_type_key(config.get("perturbator", {})),
            "num_samples": {}
        }

        self._init_directory()

    def _init_directory(self):
        # Implementation for initializing the directory structure for saving results
        project_root = Path(__file__).parent.parent
        self.data_root = Path(f"{project_root}/{self.config.get('output_dir', 'results')}/{self.run_name}/data")
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._write_metadata()

    def _write_metadata(self):
        self.metadata_path = self.data_root / "metadata.json"
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def write_config(self, config):
        config_path = self.data_root / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def __exit__(self, *args):
        self._write_metadata()  # update metadata with final sample count

        for sink in self.sinks.values():
            sink.close()

    def _get_sink(self, layer):
        """Lazily create a ShardWriter for a given layer."""
        key = f"{layer}"
        if key not in self.sinks:
            shard_dir = self.data_root / str(layer)
            shard_dir.mkdir(parents=True, exist_ok=True)
            pattern = str(shard_dir) + "/%06d.tar"
            self.sinks[key] = wds.ShardWriter(
                pattern,
                maxcount=self.chunk_size,
                maxsize=self.max_shard_size,
            )
        return self.sinks[key]

    def _tensor_to_bytes(self, tensor):
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def add_data(self, data: ActivationDataPoint | ActivationDataBatch):
        for activation_data_point in data:
            layer = activation_data_point.layer
            sample_id = activation_data_point.sample_id
            clean = activation_data_point.clean
            corrupt = activation_data_point.corrupt
            gradients = activation_data_point.gradients
            patched_loss = activation_data_point.patched_loss
            clean_loss = activation_data_point.clean_loss
            corrupted_loss = activation_data_point.corrupted_loss

            sample = {"__key__": str(sample_id)}
            if clean is not None:
                sample["clean.pth"] = self._tensor_to_bytes(clean)
            if corrupt is not None:
                sample["corrupt.pth"] = self._tensor_to_bytes(corrupt)
            if gradients is not None:
                sample["gradients.pth"] = self._tensor_to_bytes(gradients)
            if patched_loss is not None:
                sample["patched_loss.pth"] = self._tensor_to_bytes(patched_loss)
            if clean_loss is not None:
                sample["clean_loss.pth"] = self._tensor_to_bytes(clean_loss)
            if corrupted_loss is not None:
                sample["corrupted_loss.pth"] = self._tensor_to_bytes(corrupted_loss)

            if len(sample) > 1:  # more than just __key__
                self._get_sink(layer).write(sample)
                # update metadata count
                if layer not in self.metadata["num_samples"]:
                    self.metadata["num_samples"][layer] = 0
                self.metadata["num_samples"][layer] += 1

    def add_sample_metadata(self, metadata: SampleMetadata):
        if "sample_metadata" not in self.sinks:
            shard_dir = self.data_root / "sample_metadata"
            shard_dir.mkdir(parents=True, exist_ok=True)
            pattern = str(shard_dir) + "/%06d.tar"
            self.sinks["sample_metadata"] = wds.ShardWriter(
                pattern,
                maxcount=self.chunk_size,
                maxsize=self.max_shard_size,
            )

        sample = {
            "__key__": str(metadata.sample_id),
            "metadata.json": json.dumps({
                "instruction": metadata.instruction,
                "corrupt_instruction": metadata.corrupt_instruction,
                "perturbed_token_idxs": metadata.perturbed_token_idxs
            }).encode(),
        }

        self.sinks["sample_metadata"].write(sample)
