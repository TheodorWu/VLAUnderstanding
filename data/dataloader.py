import torch
from torch.utils.data import DataLoader, Subset

def get_dataloader(dataset_name, batch_size, shuffle=True, max_samples=None, **kwargs):
    if dataset_name == "libero":
        from data.libero import get_libero_dataset
        dataset = get_libero_dataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")

    if max_samples is not None:
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if kwargs.get("single_batch", False):
        return SingleBatchDataloader(dataloader)
    return dataloader

class SingleBatchDataloader:
    """Wraps an existing dataloader and yields only the first batch once."""

    def __init__(self, dataloader):
        self._batch = next(iter(dataloader))
        self.dataset = dataloader.dataset

    def __iter__(self):
        yield self._batch

    def __len__(self):
        return 1
