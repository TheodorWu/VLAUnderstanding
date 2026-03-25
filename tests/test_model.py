import unittest

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

if __name__ == "__main__":
    t = TestModels()
    t.setUp()
    t.test_pi05_initialization()
    t.tearDown()
    # suite = unittest.TestSuite()
    # suite.addTest(TestModels('test_pi05_initialization'))
    # unittest.TextTestRunner().run(suite)

    # unittest.main()
