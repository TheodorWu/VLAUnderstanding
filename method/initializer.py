from method.attribution_patching import AttributionPatching

class MethodInitializer:
    def __init__(self, config):
        self.config = config

    def initialize(self, model, tokenizer, dataset, metric, device='cuda'):
        # Initialization logic based on the configuration
        print(f"Initializing with config: {self.config}")
        method =  AttributionPatching(self.config, model, tokenizer, dataset, metric, device=device)
        return method


