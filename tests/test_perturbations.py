import unittest

from method.prompts.perturbator import PromptPerturbator, PerturbedPromptOutput


class TestDirectionalPromptPerturbator(unittest.TestCase):
    def setUp(self):
        self.config = {"method": "directional"}
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

    ### Test Cases for Semantic Perturbation ###
    def test_semantic_perturbation_scaling(self):
        # TODO: Implement test for semantic scaling perturbation
        prompt = "Increase the brightness of the image."
        result = self.perturbator.semantic_scaler.scale(prompt, factor=1.5)
        self.assertIn("brighter", result)

    def test_semantic_perturbation_synonym_replacement(self):
        # TODO: Implement test for semantic synonym replacement perturbation
        prompt = "The quick brown fox jumps over the lazy dog."
        result = self.perturbator.semantic_synonym_replacer.replace(prompt)
        self.assertNotEqual(result, prompt)



if __name__ == '__main__':
    unittest.main()
