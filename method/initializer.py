import wandb
from coolname import generate_slug

from method.attribution_patching import AttributionPatching
from method.prompts.perturbator import PromptPerturbator

class MethodInitializer:
    def __init__(self, config):
        self.config = config
        self._init_wandb()

    def _init_wandb(self):
        wandb_config = self.config.get("wandb", {})
        wandb.init(project=wandb_config.get("wandb_project", "default_vla_understanding"),
                    name=f"Attribution-Patching-{generate_slug(2)}",
                    config=self.config)

    def initialize(self, model, dataset, device='cuda'):
        # Initialization logic based on the configuration
        print(f"Initializing Method with config: {self.config}")
        perturbator = self.initialize_perturbator()
        method =  AttributionPatching(self.config, model, perturbator, dataset, device=device)
        return method

    def initialize_perturbator(self):
        print(f"Initializing perturbator with config: {self.config.get('perturbator', {})}")
        perturbator = PromptPerturbator(self.config.get('perturbator', {}))
        return perturbator
