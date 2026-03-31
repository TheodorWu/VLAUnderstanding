import io
from pathlib import Path
import json
import torch
import wandb
import webdataset as wds

class ActivationDataBatch:
    # might be useful for grouping activations/gradients from multiple samples together before writing to disk in attribution patching loop
    def __init__(self, layer, sample_ids, activations=None, gradients=None):
        data_points = []
        for sample_id in sample_ids:
            data_points.append(ActivationDataPoint(layer, sample_id, activations, gradients))
        self.data_points = data_points

class ActivationDataPoint:
    def __init__(self, layer, sample_id, activations=None, gradients=None):
        self.layer = layer
        self.sample_id = sample_id
        self.activations = activations
        self.gradients = gradients

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
            "run_name": self.run_name
        }

        self._init_directory()

    def _init_directory(self):
        # Implementation for initializing the directory structure for saving results
        project_root = Path(__file__).parent.parent
        self.data_root = Path(f"{project_root}/{self.config.get('output_dir', 'results')}/{self.run_name}/data")
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.data_root / "metadata.json"
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def __exit__(self, *args):
        for sink in self.sinks.values():
            sink.close()

    def _get_sink(self, layer, data_type):
        """Lazily create a ShardWriter for a given (layer, data_type) pair."""
        assert data_type in ["activations", "gradients"], "data_type must be 'activations' or 'gradients'"
        key = f"{layer}_{data_type}"
        if key not in self.sinks:
            shard_dir = self.data_root / str(layer) / data_type
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

    def add_data(self, activation_data_point: ActivationDataPoint):
        layer = activation_data_point.layer
        sample_id = activation_data_point.sample_id
        activations = activation_data_point.activations
        gradients = activation_data_point.gradients

        for data_type, tensor in [("activations", activations), ("gradients", gradients)]:
            if tensor is None:
                continue
            sink = self._get_sink(layer, data_type)
            sink.write({
                "__key__": str(sample_id),
                "tensor.pth": self._tensor_to_bytes(tensor),
            })
