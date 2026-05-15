from method.initializer import MethodInitializer
from model.initializer import ModelInitializer
from utils.general import seed_all

from data.dataloader import get_dataloader

class Initializer:
    def __init__(self, config):
        self.config = config
        self.method_initializer = MethodInitializer(config.get("method"))
        self.model_initializer = ModelInitializer(config.get("model"))

    def initialize(self):
        # Initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        print(f"Initializing with config: {self.config}")
        model = self.model_initializer.initialize()
        dataset = get_dataloader(**self.config.get("dataset", {}))
        self.method_initializer.initialize(model=model, dataset=dataset, device=model.get_device())
