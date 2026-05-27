from method.attribution_patching import AttributionPatching
from method.prompts.perturbator import PromptPerturbator

class MethodInitializer:
    def __init__(self, config):
        self.config = config

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
