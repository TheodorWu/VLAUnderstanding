import os
import sys
import random
from rich import json
from rich.tree import Tree
from rich.console import Console
import numpy as np
import torch
from torch.nn.parameter import is_lazy
import yaml
from pathlib import Path

HF_CACHE_DEFAULT = str(Path(__file__).parent.parent / ".hfcache")

def set_hf_cache_dir():
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = HF_CACHE_DEFAULT
        print(f"HF_HOME not set, using default: {HF_CACHE_DEFAULT}")
    else:
        print(f"HF_HOME already set to: {os.environ['HF_HOME']}, keeping.")


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

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def test_gpu_availability():
    print(f"Using torch {torch.__version__}", file=sys.stdout)
    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)

    if  torch.cuda.is_available():
        print("Using GPU", file=sys.stdout)
        print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
        print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Using CPU", file=sys.stdout)
        device = torch.device("cpu")

    return device

def pad_to_length(arr, target_len: int):
    pad_width = target_len - arr.shape[0]
    if pad_width == 0:
        return arr
    if isinstance(arr, np.ndarray):
        return np.pad(arr, (0, pad_width))
    else:  # torch.Tensor
        return torch.nn.functional.pad(arr, (0, pad_width))

### decorators ###
def printable_params(cls):
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad and not is_lazy(p))
        total = sum(p.numel() for p in self.parameters() if not is_lazy(p))
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/(total+1e-8)*100:.4f}")
    cls.print_trainable_parameters = print_trainable_parameters
    return cls

def rprint_architecture(cls):
    def print_architecture(self, output_dir):
        console = Console(record=True)
        tree = Tree(f"[bold blue]{self.__class__.__name__} Architecture[/bold blue]")

        # Also build dict for YAML
        def add_module_tree(parent_node, module, parent_dict=None):
            result = {} if parent_dict is None else parent_dict

            for name, child_module in module.named_children():
                child_node = parent_node.add(f"[cyan]{name}[/cyan]: {child_module.__class__.__name__}")

                # Get children first to check if there are any
                children = {}
                add_module_tree(child_node, child_module, children)

                # Only add children key if there are actual children
                if children:
                    result[name] = {
                        "type": child_module.__class__.__name__,
                        "children": children
                    }
                else:
                    # For leaf nodes, just store the type as a string
                    result[name] = child_module.__class__.__name__

            return result

        architecture_dict = add_module_tree(tree, self.model)

        # Print and save visual
        console.print(tree)
        filename = f"{self.__class__.__name__}_architecture"
        svg_path = output_dir / f"{filename}.svg"
        yaml_path = output_dir / f"{filename}.yaml"
        console.save_svg(svg_path)

        # Capture and save PyTorch's standard print output
        pytorch_console = Console(record=True, width=120)
        pytorch_console.print(self.model)
        pytorch_svg_path = output_dir / f"{filename}_pytorch.svg"
        pytorch_console.save_svg(pytorch_svg_path)

        # Save YAML
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(architecture_dict, f, default_flow_style=False, sort_keys=False)

    cls.print_architecture = print_architecture
    return cls

def config_to_dict(cfg) -> dict:
    """Convert OmegaConf config or plain dict/object to a serializable dict."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    except ImportError:
        pass
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return dict(cfg)

def add_batch_dim(batch):
        for k in ["clean", "corrupt", "gradients"]:
            if hasattr(batch, k):
                tensor = getattr(batch, k)
                if torch.is_tensor(tensor) and tensor.dim() == 2:
                    setattr(batch, k, tensor.unsqueeze(0))
        return batch

def pretty_print_config(cfg) -> None:
    """Pretty print a config, with optional logger support."""
    d = config_to_dict(cfg)
    formatted = json.dumps(d, indent=2, default=str)
    msg = f"Initializing with config:\n{formatted}"
    print(msg)

def get_result_layer_names(resultdir):
    """Get layer names from a result directory."""
    resultdir = Path(resultdir)
    if not resultdir.exists():
        raise FileNotFoundError(f"Result directory {resultdir} does not exist.")
    layer_names = [d.name for d in resultdir.iterdir() if d.is_dir() and not "sample_metadata" in d.name]
    return sorted(layer_names)
