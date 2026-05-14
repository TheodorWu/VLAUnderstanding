import unittest

class TestDataProcessing(unittest.TestCase):
    def test_libero_dataset_loading(self):
        from data.dataloader import get_dataloader
        dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50)
        dataset = dataloader.dataset
        self.assertIsNotNone(dataset)
        batch = next(iter(dataloader))
        self.assertIn("observation.images.image", batch)
        self.assertIn("observation.state", batch)
        self.assertIn("action", batch)

if __name__ == "__main__":
    # t = TestDataProcessing()
    # t.setUp()
    # t.test_libero_dataset_loading()
    # t.tearDown()
    unittest.main()
