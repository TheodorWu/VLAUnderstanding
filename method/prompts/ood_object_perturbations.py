

import json
from pathlib import Path
import random


class OODObjectPerturbator():
    def __init__(self, objects_path: str | Path | None = None):
        if objects_path is None:
            objects_path = Path(__file__).parent.parent.parent / "data" / "random_objects.json"
        self.objects = self._load_objects(Path(objects_path))
        assert len(self.objects) > 0, f"No objects found in {objects_path}"

    def _load_objects(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Objects not found at {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("objects", [])

    def __call__(self) -> str:
        return random.choice(self.objects)

