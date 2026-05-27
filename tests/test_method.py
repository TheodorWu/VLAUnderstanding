import unittest

from method.initializer import MethodInitializer
from model.initializer import ModelInitializer
from utils.general import test_gpu_availability

class TestMethod(unittest.TestCase):
    def test_attribution_patching(self):
        from data.dataloader import get_dataloader
        dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50, single_batch=True)

        config = {
            "model": {
                "type": "pi05",
                "model_id": None # Use None to load the model without pretrained weights for testing purposes
            },
            "perturbator": {
                "directional": True
            }
        }

        dataset_stats = dataloader.dataset.meta.stats
        device = test_gpu_availability()
        modelInitializer = ModelInitializer(config, dataset_stats=dataset_stats, device=device)
        model = modelInitializer.initialize()


        methodInitializer = MethodInitializer(config)
        method = methodInitializer.initialize(
            model=model,
            dataset=dataloader,    # Replace with actual dataset
            device=device
        )
        method.main(unit_test=True)  # Run in unit test mode to process only one batch and avoid early exit due to no perturbations
        # self.assertEqual(model.config, config["model"])

if __name__ == "__main__":

    t = TestMethod()
    t.setUp()
    t.test_attribution_patching()
    t.tearDown()
    # unittest.main()
