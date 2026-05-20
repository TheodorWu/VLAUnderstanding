from method.prompts.semantic_perturbations import SemanticScaler, SynonymReplacer

class PerturbedPromptOutput:
    """
    Docstring for PerturbedPromptOutput

    Class to handle the output of perturbed prompts.
    """
    def __init__(self, original_prompt, perturbed_prompt):
        self.original_prompt = original_prompt
        self.perturbed_prompt = perturbed_prompt

class BatchPerturbedPromptOutput:
    """
    Docstring for BatchPerturbedPromptOutput

    Class to handle the output of perturbed prompts for a batch of data.
    """
    def __init__(self, original_prompts, perturbed_prompts):
        self.original_prompts = original_prompts
        self.perturbed_prompts = perturbed_prompts

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


    def perturb_single(self, prompt):

        if self.config.get("directional"):
            prompt = self.directional_perturbation(prompt)

        if self.config.get("synonym"):
            prompt = self.synonym_perturbation(prompt, target_word="put")  # Example target word

        if self.config.get("semantic_scaling"):
            prompt = self.semantic_scaling_perturbation(prompt, target_word="bowl")  # Example target word

        return prompt

    def perturb(self, data):
        prompt = data["prompt"]
        if isinstance(prompt, list):
            perturbed_prompts = []
            for p in prompt:
                perturbed_output = self.perturb_single(p)
                perturbed_prompts.append(perturbed_output.perturbed_prompt)
            return BatchPerturbedPromptOutput(prompt, perturbed_prompts)
        else:
            return self.perturb_single(prompt)
