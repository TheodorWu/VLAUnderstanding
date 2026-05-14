from torch.utils.data import DataLoader

def get_dataloader(dataset_name, batch_size, shuffle=True, **kwargs):
    if dataset_name == "libero":
        from data.libero import get_libero_dataset
        dataset = get_libero_dataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
