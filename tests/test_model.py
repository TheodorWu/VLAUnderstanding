import unittest

import torch

from model.initializer import ModelInitializer

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
        dataloader = get_dataloader("libero", batch_size=2)
        batch = next(iter(dataloader))

        config = {
            "model": {
                "type": "pi05",
                "model_id": "lerobot/pi05_base", # Use pretrained weights for this test to ensure forward pass works
                "wrap_with_nnsight": False
            }
        }
        dataset_stats = dataloader.dataset.meta.stats
        initializer = ModelInitializer(config, dataset_stats=dataset_stats)
        model = initializer.initialize()


        # Preprocess batch before passing to forward
        processed_batch = model.preprocess_batch(batch)
        output = model(processed_batch)

        # Check output shape is reasonable (batch_size, action_dim)
        self.assertEqual(output.shape[0], 2)  # batch size
        self.assertGreater(output.shape[1], 0)  # action dimension should be positive

if __name__ == "__main__":
    t = TestModels()
    t.setUp()
    t.test_pi05_call()
    t.tearDown()
    # suite = unittest.TestSuite()
    # # suite.addTest(TestModels('test_pi05_initialization'))
    # suite.addTest(TestModels('test_pi05_call'))
    # unittest.TextTestRunner().run(suite)

    unittest.main()
