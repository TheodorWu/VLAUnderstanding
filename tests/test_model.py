import unittest

from lerobot.policies import PI05Config
import torch

from model.initializer import ModelInitializer
from utils.general import test_gpu_availability

class TestModels(unittest.TestCase):
    def test_pi05_initialization(self):
        config = {
            "model": {
                "type": "pi05",
                "model_id": None # Use None to load the model without pretrained weights for testing purposes
            }
        }
        initializer = ModelInitializer(config)
        model = initializer.initialize()
        self.assertEqual(model.config, config["model"])

    def test_pi05_call(self):
        # Load actual sample from libero dataset
        from data.dataloader import get_dataloader
        dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50)
        batch = next(iter(dataloader))

        config = {
            "model": {
                "type": "pi05",
                "model_id": None,
                # "model_id": "lerobot/pi05_libero", # Use pretrained weights for this test to ensure forward pass works
                "wrap_with_nnsight": False
            }
        }
        dataset_stats = dataloader.dataset.meta.stats
        device = test_gpu_availability()
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=device)
        model = initializer.initialize()

        # Preprocess batch before passing to forward
        processed_batch = model.preprocess_batch(batch)
        output = model(processed_batch)
        print(f"Output sample: {output}")
        print(f"Output shape: {output.shape}")

        self.assertIsNotNone(output)

    def test_pi05_call_with_nnsight(self):
        # Load actual sample from libero dataset
        from data.dataloader import get_dataloader
        dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50)
        batch = next(iter(dataloader))

        config = {
            "model": {
                "type": "pi05",
                # "model_id": "lerobot/pi05_libero", # Use pretrained weights for this test to ensure forward pass works
                "model_id": None,
                "wrap_with_nnsight": True
            }
        }
        device = test_gpu_availability()
        dataset_stats = dataloader.dataset.meta.stats
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=device)
        model = initializer.initialize()

        # Preprocess batch before passing to forward
        processed_batch = model.preprocess_batch(batch)
        output = model(processed_batch)
        print(f"Output sample: {output}")
        print(f"Output shape: {output.shape}")

        self.assertIsNotNone(output)

    def test_pi05_preprocessor(self):
        from data.dataloader import get_dataloader
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50)
        dataset_stats = dataloader.dataset.meta.stats
        pi_config = PI05Config(
            max_action_dim=32,
            max_state_dim=32,
            dtype="bfloat16",
            image_resolution=(224, 224)
        )
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config=pi_config, dataset_stats=dataset_stats
        )
        batch = next(iter(dataloader))
        processed_batch = preprocessor(batch)


if __name__ == "__main__":
    t = TestModels()
    t.setUp()
    # t.test_pi05_call()
    t.test_pi05_call_with_nnsight()
    # t.test_pi05_preprocessor()
    t.tearDown()
    # suite = unittest.TestSuite()
    # # suite.addTest(TestModels('test_pi05_initialization'))
    # suite.addTest(TestModels('test_pi05_call'))
    # unittest.TextTestRunner().run(suite)

    # unittest.main()
