import json
from pathlib import Path


class LexiconPerturbator:
    """Base class for perturbators backed by a local lexicon file."""

    def __init__(self, lexicon_path: str | Path | None = None):
        if lexicon_path is None:
            lexicon_path = Path(__file__).parent.parent / "data" / "lexicon.json"
        self.lexicon = self._load_lexicon(Path(lexicon_path))

    def _load_lexicon(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Lexicon not found at {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _lookup(self, word: str) -> dict:
        """Return the lexicon entry for word, or an empty dict if missing."""
        return self.lexicon.get(word.lower().strip(), {})

class SemanticScaler(LexiconPerturbator):
    """Replaces a word with its hypernym (one level up the IS-A hierarchy)."""

    def __call__(self, word: str) -> str:
        return self.scale_up(word)

    def scale_up(self, word: str) -> str:
        entry = self._lookup(word)
        return entry.get("hypernym", word)  # fall back to original word

class SynonymReplacer(LexiconPerturbator):
    """Replaces a word with the first synonym that isn't the word itself."""

    def __call__(self, word: str) -> str:
        return self.replace_with_synonym(word)

    def replace_with_synonym(self, word: str) -> str:
        entry = self._lookup(word)
        synonyms = entry.get("synonyms", [])
        for candidate in synonyms:
            if candidate.lower() != word.lower():
                return candidate
        return word  # fall back to original
