import wandb
from coolname import generate_slug

import torch

from method.initializer import MethodInitializer
from model.initializer import ModelInitializer
from eval.attribution_patching_evaluator import AttributionPatchingEvaluator
from eval.reservoir_evaluator import ReservoirEvaluator
from eval.pipeline import EvaluatorPipeline
from utils.general import seed_all, test_gpu_availability, pretty_print_config

from data.dataloader import get_dataloader

class Initializer:
    def __init__(self, config):
        self.config = config
        device = config.get("device", None)
        if not device:
            self.device = test_gpu_availability()
        else:
            self.device = torch.device(device)
        self._init_wandb()

    def _init_wandb(self):
        wandb_config = self.config.get("wandb", {})
        self.run_name = f"Attribution-Patching-{generate_slug(2)}"
        wandb.init(project=wandb_config.get("wandb_project", "default_vla_understanding"),
                    name=self.run_name,
                    config=self.config)

    def method(self):
        # Initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        pretty_print_config(self.config)
        dataset = get_dataloader(**self.config.get("dataset", {}))
        dataset_stats = dataset.dataset.meta.stats
        self.model_initializer = ModelInitializer(self.config.get("model"), dataset_stats=dataset_stats, device=self.device)
        model = self.model_initializer.initialize()

        self.method_initializer = MethodInitializer(self.config)
        method = self.method_initializer.initialize(model=model, dataset=dataset, device=self.device)

        self.config["activation_reader"] = self.config.get("activation_reader", {})
        self.config["activation_reader"]["run_name"] = self.run_name

        return method

    def evaluate(self):
        # Evaluation initialization logic based on the configuration
        seed_all(self.config.get("seed", 42))
        pretty_print_config(self.config)
        evaluator_pipeline = EvaluatorPipeline()

        evaluator_pipeline.add_evaluator(AttributionPatchingEvaluator(self.config, layer_sort_fn=self.get_layer_sort_fn()))
        evaluator_pipeline.add_evaluator(ReservoirEvaluator(self.config, layer_sort_fn=self.get_layer_sort_fn()))
        return evaluator_pipeline

    def get_layer_sort_fn(self):
        # Example of how to get a layer sorting function based on the model type
        model_type = self.config.get("model", {}).get("type", "")
        if model_type == "pi05":
            from model.pi05 import PI05Wrapper
            return  PI05Wrapper.sort_layers
        else:
            return None
