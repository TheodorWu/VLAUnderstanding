from pathlib import Path
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "libero"
data_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset("physical-intelligence/libero", cache_dir=str(data_dir))
