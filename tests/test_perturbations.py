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
        prompt = "put the bowl on the stove"
        result = self.perturbator.semantic_scaling_perturbation(prompt, "bowl")
        self.assertNotIn("bowl", result.perturbed_prompt)

    def test_semantic_perturbation_synonym_replacement(self):
        prompt = "put the bowl on the stove"
        result = self.perturbator.synonym_perturbation(prompt, "put")
        self.assertNotEqual(result.perturbed_prompt, prompt)



if __name__ == '__main__':
    unittest.main()
