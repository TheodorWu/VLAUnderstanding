from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def get_libero_dataset():
    # Login using e.g. `huggingface-cli login` to access this dataset
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "libero"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "lerobot/libero_10_image"
    # meta = LeRobotDatasetMetadata(repo_id, root=data_dir)
    ds = LeRobotDataset(repo_id, root=data_dir)
    return ds
