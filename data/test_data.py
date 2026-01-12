import unittest

class TestDataProcessing(unittest.TestCase):
    def test_libero_dataset_loading(self):
        from data.dataloader import get_dataloader
        dataloader = get_dataloader("libero", batch_size=4)
        dataset = dataloader.dataset
        self.assertIsNotNone(dataset)
        self.assertIn('train', dataset)
        self.assertIn('validation', dataset)
        self.assertIn('test', dataset)

if __name__ == "__main__":
    unittest.main()
