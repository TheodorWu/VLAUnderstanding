from method.activation_patching import ActivationPatching
from method.attribution_patching import AttributionPatching
from method.prompts.perturbator import PromptPerturbator

class MethodInitializer:
    def __init__(self, config):
        self.config = config

    def initialize(self, model, dataset, device='cuda'):
        # Initialization logic based on the configuration
        print(f"Initializing Method with config: {self.config}")
        perturbator = self.initialize_perturbator()
        if self.config.get("method", {}).get("name") == "activation_patching":
            print("Using Activation Patching method.")
            method = ActivationPatching(self.config, model, dataset, perturbator, device=device)
        else:
            method = AttributionPatching(self.config, model, dataset, perturbator, device=device)
        return method

    def initialize_perturbator(self):
        print(f"Initializing perturbator with config: {self.config.get('perturbator', {})}")
        perturbator = PromptPerturbator(self.config.get('perturbator', {}))
        return perturbator
