from method.attribution_patching import AttributionPatching
from method.attribution_patching_inference import AttributionPatchingInference
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

    def initialize_inference(self, model, env, device='cuda'):
        # Initialization logic for inference based on the configuration
        print(f"Initializing Method for inference with config: {self.config}")
        perturbator = self.initialize_perturbator()
        method = AttributionPatchingInference(self.config, model, perturbator, env, device=device)
        return method
