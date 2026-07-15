from pathlib import Path
from nnsight import NNsight
import torch

from utils.general import set_hf_cache_dir

class ModelInitializer:
    def __init__(self, config, dataset_stats=None, device=torch.device("cpu")):
        self.config = config
        self.dataset_stats = dataset_stats
        self.device = device

        set_hf_cache_dir()

    def initialize(self):
        # Initialization logic based on the configuration
        print(f"Initializing model with config: {self.config}")
        return self._initialize_model()

    def _initialize_model(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "pi05")
        print(f"Model Type: {model_type}")
        if model_type == "pi05":
            from model.pi05 import PI05Wrapper
            model = PI05Wrapper(model_config, self.dataset_stats, self.device)
        elif model_type == "groot":
            from model.groot import GROOTWrapper
            model = GROOTWrapper(model_config, self.dataset_stats, self.device)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

        try:
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters() # pylint: disable=no-member
            if hasattr(model, "print_architecture") and model_config.get("print_architecture", False):
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "output" / "model_architectures"
                output_dir.mkdir(parents=True, exist_ok=True)
                model.print_architecture(output_dir) # pylint: disable=no-member
        except Exception as e:
            print(f"Error while printing model details: {e}")

        if model_config.get("wrap_with_nnsight", True):
            model = self.wrap_nnsight(model)
        return model

    def wrap_nnsight(self, model):
        print("Wrapping model with NNsight for tracing.")
        return NNsight(model)
