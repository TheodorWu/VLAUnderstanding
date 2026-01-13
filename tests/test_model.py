import unittest

from model.initializer import ModelInitializer

class TestModels(unittest.TestCase):
    def test_pi0_initialization(self):
        config = {
            "model": {
                "type": "pi0"
            }
        }
        initializer = ModelInitializer(config)
        model = initializer.initialize()
        self.assertEqual(model.config, config["model"])

if __name__ == "__main__":
    unittest.main()
