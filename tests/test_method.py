import unittest

from method.initializer import MethodInitializer
from model.initializer import ModelInitializer

class TestMethod(unittest.TestCase):
    def test_attribution_patching(self):

        config = {
            "model": {
                "type": "pi0"
            }
        }
        modelInitializer = ModelInitializer(config)
        model = modelInitializer.initialize()

        from data.dataloader import get_dataloader
        ds = get_dataloader("libero", batch_size=4)

        methodInitializer = MethodInitializer(config)
        method = methodInitializer.initialize(
            model=model,
            dataset=ds,    # Replace with actual dataset
            device='cuda'
        )
        method.main()
        # self.assertEqual(model.config, config["model"])

if __name__ == "__main__":
    unittest.main()
