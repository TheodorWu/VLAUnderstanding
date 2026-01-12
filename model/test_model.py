import unittest

from model.initializer import ModelInitializer

class TestModels(unittest.TestCase):
    def test_pi0_initialization(self):
        # TODO: Implement test for Pi0 model initialization
        config = {"model_name": "pi0", "param1": 5}
        initializer = ModelInitializer(config)
        model = initializer.initialize()
        self.assertEqual(model.config, config)

if __name__ == "__main__":
    unittest.main()
