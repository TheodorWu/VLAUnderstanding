from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def get_libero_dataset(**kwargs):
    # Login using e.g. `huggingface-cli login` to access this dataset
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "libero"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "lerobot/libero_10_image"
    fps = kwargs.get("fps", None)
    if not fps:
        raise ValueError("FPS must be provided to get_libero_dataset since it is needed for timestamp calculation.")
    chunk_size = kwargs.get("chunk_size", None)
    if not chunk_size:
        raise ValueError("Chunk size must be provided to get_libero_dataset since it is needed for timestamp calculation.")
    # meta = LeRobotDatasetMetadata(repo_id, root=data_dir)
    ds = LeRobotDataset(repo_id, root=data_dir, delta_timestamps={
            "action": [i / fps for i in range(chunk_size)],  # chunk_size=50 steps
        })
    return ds
