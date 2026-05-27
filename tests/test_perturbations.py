import unittest

from method.prompts.perturbator import PromptPerturbator, PerturbedPromptOutput


class TestDirectionalPromptPerturbator(unittest.TestCase):
    def setUp(self):
        self.config = {"method": "directional", "blocklist": ["pick up"], "target_pos": "noun", "num_targets": 1}
        self.perturbator = PromptPerturbator(self.config)


    ### Test Cases for Directional Perturbation ###
    def test_directional_perturbation_left_to_right(self):
        prompt = "Move the object to the left."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Move the object to the right.")

    def test_directional_perturbation_right_to_left(self):
        prompt = "Move the object to the right."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Move the object to the left.")

    def test_directional_perturbation_up_to_down(self):
        prompt = "Lift the box up."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Lift the box down.")

    def test_directional_perturbation_down_to_up(self):
        prompt = "Push the button down."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Push the button up.")

    def test_directional_perturbation_no_change(self):
        prompt = "This is a neutral statement."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, prompt)

    # Word boundary tests (should NOT replace substrings)
    def test_directional_perturbation_no_replace_uppercase(self):
        prompt = "Move the object to the left."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Move the object to the right.")

    def test_directional_perturbation_no_replace_in_update(self):
        prompt = "Update the configuration."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Update the configuration.")

    def test_directional_perturbation_no_replace_in_upper(self):
        prompt = "Move to the upper shelf."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Move to the upper shelf.")

    def test_directional_perturbation_no_replace_in_downtown(self):
        prompt = "Go to the downtown area."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Go to the downtown area.")

    def test_directional_perturbation_no_replace_in_leftover(self):
        prompt = "Store the leftover items."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Store the leftover items.")

    # Blocklist tests (phrasal verbs should NOT be replaced)
    def test_directional_perturbation_no_replace_pick_up(self):
        prompt = "Pick up the object."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Pick up the object.".lower())

    # Blocklist + direction in same prompt
    def test_directional_perturbation_pick_up_with_direction(self):
        prompt = "Pick up the object and move left."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Pick up the object and move right.".lower())

    # Single-pass replacement (no double replacement)
    def test_directional_perturbation_no_double_replace_up_down(self):
        prompt = "Move up then down."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "Move down then up.")

    def test_directional_perturbation_no_double_replace_left_right(self):
        prompt = "First go left then right."
        result = self.perturbator.directional_perturbation(prompt)
        self.assertEqual(result.perturbed_prompt, "First go right then left.")

    ### Test Cases for Semantic Perturbation ###
    def test_semantic_perturbation_scaling(self):
        prompt = "put the bowl on the stove"
        result = self.perturbator.semantic_scaling_perturbation(prompt, "bowl")
        self.assertNotIn("bowl", result.perturbed_prompt)

    def test_semantic_perturbation_synonym_replacement(self):
        prompt = "put the bowl on the stove"
        result = self.perturbator.synonym_perturbation(prompt, "put")
        self.assertNotEqual(result.perturbed_prompt, prompt)


if __name__ == '__main__':
    unittest.main()
