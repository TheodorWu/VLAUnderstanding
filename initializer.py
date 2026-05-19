import torch

from method.initializer import MethodInitializer
from model.initializer import ModelInitializer
from eval.attribution_patching_evaluator import AttributionPatchingEvaluator
from utils.general import seed_all, test_gpu_availability

from data.dataloader import get_dataloader

class Initializer:
    def __init__(self, config):
        self.config = config
        device = config.get("device", None)
        if not device:
            self.device = test_gpu_availability()
        else:
            self.device = torch.device(device)

        self.method_initializer = MethodInitializer(config)
        self.model_initializer = ModelInitializer(config.get("model"), device=self.device)

    def method(self):
        # Initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        print(f"Initializing with config: {self.config}")
        model = self.model_initializer.initialize()
        dataset = get_dataloader(**self.config.get("dataset", {}))
        method = self.method_initializer.initialize(model=model, dataset=dataset, device=self.device)
        return method

    def evaluate(self):
        # Evaluation initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        print(f"Initializing evaluation with config: {self.config}")
        evaluator = AttributionPatchingEvaluator(self.config)
        return evaluator
