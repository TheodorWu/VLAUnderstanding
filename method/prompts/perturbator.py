from method.prompts.semantic_perturbations import SemanticScaler, SynonymReplacer

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

    def synonym_perturbation(self, prompt, target_word):
        # Placeholder for semantic perturbation logic
        synonym  = self.synonym_replacer(target_word)
        perturbed_prompt = prompt.replace(target_word, synonym)
        return PerturbedPromptOutput(prompt, perturbed_prompt)

    def semantic_scaling_perturbation(self, prompt, target_word):
        # Placeholder for semantic scaling logic
        scaled_word  = self.semantic_scaler(target_word)
        perturbed_prompt = prompt.replace(target_word, scaled_word)
        return PerturbedPromptOutput(prompt, perturbed_prompt)

    def perturb(self, data):
        # TODO: Implement perturbation logic based on config
        pass
