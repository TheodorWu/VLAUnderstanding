import io
import json
from pathlib import Path

import torch
import wandb
import webdataset as wds

from data.activation_writer import ActivationDataBatch, ActivationDataPoint


class ActivationReader:
    def __init__(self, config):
        self.config = config.get("activation_reader", config.get("activation_writer", {}))
        self.output_dir = self.config.get("output_dir", "results")
        self.run_name = self.config.get("run_name") or (wandb.run.name if wandb.run else "test_run")

        project_root = Path(__file__).parent.parent
        self.data_root = Path(f"{project_root}/{self.output_dir}/{self.run_name}/data")
        self.metadata_path = self.data_root / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _get_shard_dir(self, layer, data_type):
        if data_type not in ["activations", "gradients"]:
            raise ValueError("data_type must be 'activations' or 'gradients'")
        return self.data_root / str(layer) / data_type

    def _iter_shard_paths(self, layer=None, data_type=None):
        if layer is not None and data_type is not None:
            yield from sorted(self._get_shard_dir(layer, data_type).glob("*.tar"))
            return

        if layer is not None:
            for shard_dir in sorted((self.data_root / str(layer)).glob("*")):
                if shard_dir.is_dir():
                    yield from sorted(shard_dir.glob("*.tar"))
            return

        for shard_dir in sorted(self.data_root.glob("*/*")):
            if shard_dir.is_dir():
                yield from sorted(shard_dir.glob("*.tar"))

    def _iter_samples(self, layer=None, data_type=None):
        shard_paths = [str(path) for path in self._iter_shard_paths(layer=layer, data_type=data_type)]
        if not shard_paths:
            return

        dataset = wds.WebDataset(shard_paths, shardshuffle=False)
        try:
            for sample in dataset:
                yield sample
        finally:
            dataset.close()

    def _tensor_from_bytes(self, tensor_bytes):
        buffer = io.BytesIO(tensor_bytes)
        return torch.load(buffer, map_location="cpu")

    def iter_data(self, layer=None, data_type=None):
        if layer is None and data_type is None:
            for layer_dir in sorted(self.data_root.glob("*")):
                if not layer_dir.is_dir():
                    continue
                for dtype_dir in sorted(layer_dir.glob("*")):
                    if dtype_dir.is_dir():
                        yield from self.iter_data(layer_dir.name, dtype_dir.name)
            return

        resolved_layer = layer
        resolved_data_type = data_type
        for sample in self._iter_samples(layer=layer, data_type=data_type):
            if resolved_layer is None or resolved_data_type is None:
                source_path = Path(sample["__url__"])
                resolved_data_type = source_path.parent.name
                resolved_layer = source_path.parent.parent.name

            sample_id = sample["__key__"]
            tensor = self._tensor_from_bytes(sample["tensor.pth"])
            if resolved_data_type == "activations":
                yield ActivationDataPoint(resolved_layer, sample_id, activations=tensor)
            else:
                yield ActivationDataPoint(resolved_layer, sample_id, gradients=tensor)

    def read_layer(self, layer, data_type="activations"):
        return list(self.iter_data(layer=layer, data_type=data_type))

    def read_all(self):
        return list(self.iter_data())

    def get_tensor(self, layer, sample_id, data_type="activations"):
        for activation_data_point in self.iter_data(layer=layer, data_type=data_type):
            if activation_data_point.sample_id == sample_id:
                if data_type == "activations":
                    return activation_data_point.activations
                return activation_data_point.gradients
        return None

    def get_metadata(self):
        return self.metadata

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


__all__ = ["ActivationReader", "ActivationDataPoint", "ActivationDataBatch"]
