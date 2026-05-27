import requests
import json
from pathlib import Path

class ConceptNetPerturbator:
    def __init__(self):
        self.conceptnet_base = "http://api.conceptnet.io"
        self.cache_file = "conceptnet_cache.json"
        self.cache = self._init_cache()

    def _init_cache(self):
        project_root = Path(__file__).parent.parent
        self.cache_path = project_root / self.cache_file
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def query_conceptnet(self, url):
        if url not in self.cache:
            self.cache[url] = requests.get(url, timeout=10).json()
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        return self.cache[url]

class SemanticScaler(ConceptNetPerturbator):
    def __call__(self, *args, **kwds):
        return self.scale_up(*args, **kwds)

    def scale_up(self, word):
        normalized = word.lower().replace(" ", "_")
        url = f"{self.conceptnet_base}/query?node=/c/en/{normalized}&rel=/r/IsA&start=/c/en/{normalized}"
        response = self.query_conceptnet(url)

        edges = response.get("edges", [])
        if not edges:
            return word

        # Traverse one level up the IsA hierarchy
        return edges[0]["end"]["label"]


class SynonymReplacer(ConceptNetPerturbator):
    def __call__(self, *args, **kwds):
        return self.replace_with_synonym(*args, **kwds)

    def replace_with_synonym(self, word):
        normalized = word.lower().replace(" ", "_")
        url = f"{self.conceptnet_base}/query?node=/c/en/{normalized}&rel=/r/Synonym&start=/c/en/{normalized}"
        response = self.query_conceptnet(url)

        edges = response.get("edges", [])
        for edge in edges:
            candidate = edge["end"]["label"]
            if candidate.lower() != word.lower() and word.lower() not in candidate.lower():
                return candidate

        # Fall back to SimilarTo if no Synonym relation found
        url = f"{self.conceptnet_base}/query?node=/c/en/{normalized}&rel=/r/SimilarTo&start=/c/en/{normalized}"
        response = self.query_conceptnet(url)

        edges = response.get("edges", [])
        for edge in edges:
            candidate = edge["end"]["label"]
            if candidate.lower() != word.lower() and word.lower() not in candidate.lower():
                return candidate

        return word
