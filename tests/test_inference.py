import unittest

from data.dataloader import get_dataloader
from method.initializer import MethodInitializer
from model.initializer import ModelInitializer
from model.pi05 import PI05Wrapper
from utils.general import test_gpu_availability
from envs.libero import get_libero

class TestMethod(unittest.TestCase):
    def setUp(self):
        self.env, _ = get_libero(task_id=0)

        self.dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50, single_batch=True)
        self.dataset_stats = self.dataloader.dataset.meta.stats

    def test_env_and_dataset_observations_equal(self):
        obs = self.env.reset()
        obs_batch = PI05Wrapper.transform_obs_to_batch(obs, self.env.language_instruction)
        batch = next(iter(self.dataloader))
        obs_keys = [ k  for k in batch.keys() if "observation" in k or "task" == k]
        for k in obs_keys:
            self.assertIn(k, obs_batch, f"Initial observation from env does not contain key: {k}")

    def test_attribution_patching_inference_dummy(self):
        config = {
            "model": {
                "type": "dummy"
            },
            "perturbator": {
                "directional": True,
                "target_words": "noun"
            }
        }
        modelInitializer = ModelInitializer(config, dataset_stats=self.dataset_stats)
        model = modelInitializer.initialize()

        methodInitializer = MethodInitializer(config)
        method = methodInitializer.initialize_inference(
            model=model,
            env=self.env,
            device=model.device
        )
        method.main(unit_test=True)

    def test_attribution_patching_inference(self):

        config = {
            "model": {
                "type": "pi05",
                "model_id": None # Use None to load the model without pretrained weights for testing purposes
            },
            "perturbator": {
                "directional": True,
                "target_words": "noun"
            }
        }

        device = test_gpu_availability()
        modelInitializer = ModelInitializer(config, dataset_stats=self.dataset_stats, device=device)
        model = modelInitializer.initialize()

        methodInitializer = MethodInitializer(config)
        method = methodInitializer.initialize_inference(
            model=model,
            env=self.env,    # Replace with actual dataset
            device=device
        )
        method.main(unit_test=True)  # Run in unit test mode to process only one batch and avoid early exit due to no perturbations
        # self.assertEqual(model.config, config["model"])

if __name__ == "__main__":
    t = TestMethod()
    t.setUp()
    t.test_attribution_patching_inference_dummy()
    t.tearDown()
    # unittest.main()
