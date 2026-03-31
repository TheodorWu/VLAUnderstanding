import wandb
from coolname import generate_slug

from method.attribution_patching import AttributionPatching
from method.prompts.perturbator import PromptPerturbator

class MethodInitializer:
    def __init__(self, config):
        self.config = config
        self._init_wandb()

    def _init_wandb(self):
        wandb.init(project=self.config.get("wandb_project", "default_vla_understanding"),
                    name=f"Attribution-Patching-{generate_slug(2)}",
                    config=self.config)

    def initialize(self, model, dataset, device='cuda'):
        # Initialization logic based on the configuration
        print(f"Initializing with config: {self.config}")
        perturbator = self.initialize_perturbator()
        metric = self.initialize_metric()
        tokenizer = model.get_tokenizer()  # Assuming the model has a method to get the tokenizer
        method =  AttributionPatching(self.config, model, tokenizer, perturbator, dataset, metric, device=device)
        return method

    def initialize_perturbator(self):
        print(f"Initializing perturbator with config: {self.config.get('perturbator', {})}")
        perturbator = PromptPerturbator(self.config.get('perturbator', {}))
        return perturbator

    def initialize_metric(self):
        # Placeholder for metric initialization logic
        if self.config.get('metric') == 'logit_diff':
            from method.losses.logit_diff import LogitDifference
            return LogitDifference()
        else:
            raise NotImplementedError(f"Metric {self.config.get('metric')} not implemented.")
