from method.prompts.semantic_perturbations import SemanticScaler, SynonymReplacer
import re

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

        self.BLOCKLIST = {"pick up", "up to", "up for"}

    def directional_perturbation(self, prompt):
        replacements = {"left": "right", "right": "left", "up": "down", "down": "up"}

        # 1. Mask blocklisted phrases
        masked = prompt
        placeholders = {}
        for i, phrase in enumerate(self.BLOCKLIST):
            placeholder = f"__BLOCKED_{i}__"
            if phrase in masked.lower():
                placeholders[placeholder] = phrase
                masked = re.sub(re.escape(phrase), placeholder, masked, flags=re.IGNORECASE)

        # 2. Replace directions in one pass
        pattern = re.compile(r'\b(' + '|'.join(replacements.keys()) + r')\b', flags=re.IGNORECASE)
        replaced = pattern.sub(lambda m: replacements[m.group().lower()], masked)

        # 3. Restore masked phrases
        for placeholder, original in placeholders.items():
            replaced = replaced.replace(placeholder, original)

        return PerturbedPromptOutput(prompt, replaced)

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
