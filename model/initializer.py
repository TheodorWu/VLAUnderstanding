from model.pi0 import PI0Wrapper
from pathlib import Path
from nnsight import NNsight

class ModelInitializer:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        # Initialization logic based on the configuration
        print(f"Initializing with config: {self.config}")
        return self._initialize_model()

    def _initialize_model(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "pi0")
        print(f"Model Type: {model_type}")
        if model_type == "pi0":
            model = PI0Wrapper(model_config)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

        try:
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters() # pylint: disable=no-member
            if hasattr(model, "print_architecture"):
                project_root = Path(__file__).parent.parent
                output_dir = project_root / "output" / "model_architectures"
                output_dir.mkdir(parents=True, exist_ok=True)
                model.print_architecture(output_dir) # pylint: disable=no-member
        except Exception as e:
            print(f"Error while printing model details: {e}")

        model = self.wrap_nnsight(model)
        return model

    def wrap_nnsight(self, model):
        print("Wrapping model with NNsight for tracing.")
        return NNsight(model)
