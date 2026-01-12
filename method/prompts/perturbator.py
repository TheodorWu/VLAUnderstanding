from semantic_perturbations import SemanticScaler, SynonymReplacer

class PerturbedPromptOutput:
    """
    Docstring for PerturbedPromptOutput

    Class to handle the output of perturbed prompts.
    """
    def __init__(self, original_prompt, perturbed_prompt):
        self.original_prompt = original_prompt
        self.perturbed_prompt = perturbed_prompt


class PromptPerturbator:
    """
    Docstring for PromptPerturbator

    Class to handle perturbation of prompts based on given configuration and method calls.
    """
    def __init__(self, config):
        self.config = config
        self.semantic_scaler = SemanticScaler()
        self.synonym_replacer = SynonymReplacer()

    def directional_perturbation(self, prompt):
        if "left" in prompt:
            perturbed_prompt = prompt.replace("left", "right")
        elif "right" in prompt:
            perturbed_prompt = prompt.replace("right", "left")
        elif "up" in prompt:
            perturbed_prompt = prompt.replace("up", "down")
        elif "down" in prompt:
            perturbed_prompt = prompt.replace("down", "up")
        else:
            perturbed_prompt = prompt
        return PerturbedPromptOutput(prompt, perturbed_prompt)

    def perturb(self, data):
        # Implement perturbation logic based on config
        pass
