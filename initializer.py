from method.initializer import MethodInitializer
from model.initializer import ModelInitializer

class Initializer:
    def __init__(self, config):
        self.config = config
        self.method_initializer = MethodInitializer(config.get("method"))
        self.model_initializer = ModelInitializer(config.get("model"))

    def initialize(self):
        # Initialization logic based on the configuration
        print(f"Initializing with config: {self.config}")
        self.method_initializer.initialize()
        self.model_initializer.initialize()
