import sys
import random

import numpy as np
import torch
from torch.nn.parameter import is_lazy

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

def test_gpu_availability(cfg=None):
    print(f"Using torch {torch.__version__}", file=sys.stdout)
    print(f"Cuda available: {torch.cuda.is_available()}", file=sys.stdout)
    print('__CUDNN VERSION:', torch.backends.cudnn.version(), file=sys.stdout)
    print('Available devices ', torch.cuda.device_count(), file=sys.stdout)
    print('Current cuda device ', torch.cuda.current_device(), file=sys.stdout)
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if cfg and cfg.training.device == "gpu" and torch.cuda.is_available():
        print("Using GPU", file=sys.stdout)
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Using CPU", file=sys.stdout)
        device = torch.device("cpu")

    return device

def printable_params(cls):
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad and not is_lazy(p))
        total = sum(p.numel() for p in self.parameters() if not is_lazy(p))
        print(f"trainable: {trainable:,} || all params: {total:,} || trainable%: {trainable/(total+1e-8)*100:.4f}")
    cls.print_trainable_parameters = print_trainable_parameters
    return cls
