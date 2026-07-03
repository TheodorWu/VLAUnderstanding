import re

def layer_display_name(layer_name: str) -> str:
    """Convert a layer name to a more human-readable format."""
    # Example: "model.transformer.layers.0.self_attn.gemma_expert.o_proj" -> "0"
    match = re.search(r'layers\.(\d+)', layer_name)
    if match:
        return f"{match.group(1)}"
    else:
        return layer_name
