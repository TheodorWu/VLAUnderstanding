from model.pi0 import PI0Wrapper

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

        model.print_trainable_parameters() # pylint: disable=no-member
        return model

