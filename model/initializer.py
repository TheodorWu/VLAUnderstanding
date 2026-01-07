class ModelInitializer:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        # Initialization logic based on the configuration
        print(f"Initializing with config: {self.config}")

    def _initialize_model(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "default_model")
        model_params = model_config.get("params", {})
        print(f"Model Type: {model_type}, Model Params: {model_params}")
