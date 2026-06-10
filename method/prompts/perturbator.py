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

        self.blocklist = set(self.config.get("blocklist", ["pick up"]))  # Default blocklist with "pick up"
        self._init_target_words()

    def _init_target_words(self):
        target_words_config = self.config.get("target_words")
        if isinstance(target_words_config, list):
            self.target_words = target_words_config
        elif target_words_config in ["noun", "verb"]:
            lexicon = self.semantic_scaler.lexicon
            self.target_words = [word for word, entry in lexicon.items() if entry.get("pos") == target_words_config]
            print(f"Initialized target words for POS '{target_words_config}': {len(self.target_words)} words found.")
        else:
            raise ValueError("Invalid target_words configuration. Must be a list or one of 'noun', 'verb'.")

    def directional_perturbation(self, prompt):
        replacements = {"left": "right", "right": "left", "up": "down", "down": "up"}

        # 1. Mask blocklisted phrases
        masked = prompt
        placeholders = {}
        for i, phrase in enumerate(self.blocklist):
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

    def _handle_str_or_output(self, prompt):
        if isinstance(prompt, PerturbedPromptOutput):
            return prompt.original_prompt, prompt.perturbed_prompt
        else:
            return prompt, prompt

    def synonym_perturbation(self, prompt, target_word):
        original_prompt, text = self._handle_str_or_output(prompt)

        synonym  = self.synonym_replacer(target_word)
        perturbed_prompt = text.replace(target_word, synonym)
        return PerturbedPromptOutput(original_prompt, perturbed_prompt)

    def semantic_scaling_perturbation(self, prompt, target_word):

        original_prompt, text = self._handle_str_or_output(prompt)
        scaled_word  = self.semantic_scaler(target_word)
        perturbed_prompt = text.replace(target_word, scaled_word)
        return PerturbedPromptOutput(original_prompt, perturbed_prompt)

    def perturb_single_prompt(self, prompt: str):
        if self.config.get("directional"):
            prompt = self.directional_perturbation(prompt)

        if self.config.get("synonym") or self.config.get("semantic_scaling"):
            for target_word in self.target_words:
                if self.config.get("synonym"):
                    prompt = self.synonym_perturbation(prompt, target_word=target_word)
                if self.config.get("semantic_scaling"):
                    prompt = self.semantic_scaling_perturbation(prompt, target_word=target_word)

        return prompt

    def perturb(self, data):
        prompt = data["prompt"]
        if isinstance(prompt, list):
            perturbed_prompts = []
            for p in prompt:
                perturbed_output = self.perturb_single_prompt(p)
                perturbed_prompts.append(perturbed_output.perturbed_prompt)
            return BatchPerturbedPromptOutput(prompt, perturbed_prompts)
        else:
            return self.perturb_single_prompt(prompt)
