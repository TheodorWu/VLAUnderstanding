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
        metadata = {}
        if not self.metadata_path.exists():
            return metadata
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["layer_names"] = [str(d) for d in self.data_root.glob("*") if d.is_dir()]
        return metadata

    def _get_shard_dir(self, layer, data_type):
        if data_type not in ["clean", "gradients", "corrupt"]:
            raise ValueError("data_type must be 'clean', 'gradients', or 'corrupt'")
        return self.data_root / str(layer)

    def _iter_shard_paths(self, layer=None):
        if layer is not None:
            yield from sorted((self.data_root / str(layer)).glob("*.tar"))
            return
        for layer_dir in sorted(self.data_root.glob("*")):
            if layer_dir.is_dir():
                yield from sorted(layer_dir.glob("*.tar"))

    def _tensor_from_bytes(self, tensor_bytes):
        buffer = io.BytesIO(tensor_bytes)
        return torch.load(buffer, map_location="cpu")

    def iter_data(self, layer=None):
        shard_paths = [str(p) for p in self._iter_shard_paths(layer=layer) if not "sample_metadata" in str(p)]
        if not shard_paths:
            print(f"No data found for layer '{layer}' in run '{self.run_name}'. Searched paths: {[str(p) for p in self.data_root.glob('*/*.tar')]}")
            return

        dataset =  wds.WebDataset(shard_paths, shardshuffle=False)
        print(f"Number of samples for perturbation type '{self.metadata.get('perturbation_type', 'unknown')}' and layer '{layer if layer else 'all'}': {self.metadata.get('num_samples', 'unknown')}")
        try:
            for sample in dataset:
                source_layer = layer or Path(sample["__url__"]).parent.name
                yield ActivationDataPoint(
                    layer=source_layer,
                    sample_id=sample["__key__"],
                    clean=self._tensor_from_bytes(sample["clean.pth"]) if "clean.pth" in sample else None,
                    corrupt=self._tensor_from_bytes(sample["corrupt.pth"]) if "corrupt.pth" in sample else None,
                    gradients=self._tensor_from_bytes(sample["gradients.pth"]) if "gradients.pth" in sample else None,
                )
        except Exception as e:
            print(f"Error reading shards: {e}")
        finally:
            dataset.__exit__()
            del dataset

    def read_layer(self, layer):
        return list(self.iter_data(layer=layer))

    def read_all(self):
        return list(self.iter_data())

    def get_metadata(self):
        return self.metadata



__all__ = ["ActivationReader", "ActivationDataPoint", "ActivationDataBatch"]
