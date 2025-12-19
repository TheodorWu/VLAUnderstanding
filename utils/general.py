class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'") from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'") from exc
