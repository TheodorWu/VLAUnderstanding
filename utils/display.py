import re


def layer_block_index(layer_name: str) -> int | None:
    """Extract the transformer block index from a layer name, if present."""
    match = re.search(r'(?:transformer_blocks|layers)\.(\d+)', layer_name)
    return int(match.group(1)) if match else None


def layer_display_name(layer_name: str) -> str:
    idx = layer_block_index(layer_name)
    return str(idx) if idx is not None else layer_name
