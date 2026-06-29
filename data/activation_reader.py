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
        self.sample_metadata = {}

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
        shard_paths = [str(p) for p in self._iter_shard_paths(layer=layer) if "sample_metadata" not in str(p)]
        if not shard_paths:
            print(f"No data found for layer '{layer}'")
            return

        skipped_shards = 0
        skipped_samples = 0

        for shard_path in shard_paths:
            try:
                dataset = wds.WebDataset(shard_path, shardshuffle=False)
                iterator = iter(dataset)
            except Exception as e:
                print(f"Skipping unreadable shard {shard_path}: {e}")
                skipped_shards += 1
                continue

            while True:
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Skipping corrupted entry in {shard_path}: {e}")
                    skipped_shards += 1
                    break  # shard is dead, move to next one

                try:
                    yield ActivationDataPoint(
                        layer=layer or Path(sample["__url__"]).parent.name,
                        sample_id=sample["__key__"],
                        clean=self._tensor_from_bytes(sample["clean.pth"]) if "clean.pth" in sample else None,
                        corrupt=self._tensor_from_bytes(sample["corrupt.pth"]) if "corrupt.pth" in sample else None,
                        gradients=self._tensor_from_bytes(sample["gradients.pth"]) if "gradients.pth" in sample else None,
                    )
                except Exception as e:
                    print(f"Skipping bad sample '{sample.get('__key__', 'unknown')}': {e}")
                    skipped_samples += 1

            dataset.__exit__()

        if skipped_shards or skipped_samples:
            print(f"Finished with {skipped_shards} skipped shards, {skipped_samples} skipped samples")

    def read_layer(self, layer):
        return list(self.iter_data(layer=layer))

    def read_all(self):
        return list(self.iter_data())

    def get_metadata(self):
        return self.metadata

    def _load_sample_metadata(self):
        sample_metadata_shards = [str(p) for p in self.data_root.glob("sample_metadata/*.tar")]
        dataset =  wds.WebDataset(sample_metadata_shards, shardshuffle=False)
        iterator = iter(dataset)
        skipped = 0
        while True:
            try:
                sample = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                skipped += 1
                print(f"Skipping corrupted shard entry: {e}")
                continue

            sample_id = sample.get("__key__")

            try:
                self.sample_metadata[sample_id] = json.loads(sample["metadata.json"].decode("utf-8"))
            except Exception as e:
                skipped += 1
                print(f"Skipping corrupted sample metadata for '{sample_id}': {e}")

    def get_sample_metadata(self, sample_id):
        if not getattr(self, "sample_metadata", None):
            self._load_sample_metadata()
        return self.sample_metadata.get(sample_id, None)

    def get_all_sample_metadata(self):
        if not getattr(self, "sample_metadata", None):
            self._load_sample_metadata()
        return self.sample_metadata


__all__ = ["ActivationReader", "ActivationDataPoint", "ActivationDataBatch"]
