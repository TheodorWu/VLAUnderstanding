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


    def method(self):
        # Initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        print(f"Initializing with config: {self.config}")
        dataset = get_dataloader(**self.config.get("dataset", {}))
        dataset_stats = dataset.dataset.meta.stats
        self.model_initializer = ModelInitializer(self.config.get("model"), dataset_stats=dataset_stats, device=self.device)
        model = self.model_initializer.initialize()

        self.method_initializer = MethodInitializer(self.config)
        method = self.method_initializer.initialize(model=model, dataset=dataset, device=self.device)
        return method

    def evaluate(self):
        # Evaluation initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        print(f"Initializing evaluation with config: {self.config}")
        evaluator = AttributionPatchingEvaluator(self.config)
        return evaluator
