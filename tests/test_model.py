import unittest

from lerobot.policies import PI05Config
import torch

from model.initializer import ModelInitializer
from utils.general import test_gpu_availability
from data.dataloader import get_dataloader

class TestModels(unittest.TestCase):
    def setUp(self):
        self.dataloader = get_dataloader("libero", batch_size=2, fps=10, chunk_size=50)
        self.dataset_stats = self.dataloader.dataset.meta.stats
        self.dataloader_groot = get_dataloader("libero", batch_size=2, fps=10, chunk_size=40)

    def test_groot_initialization(self):
        config = {
                "type": "groot",
                "model_id": None # Use None to load the model without pretrained weights for testing purposes
        }
        dataset_stats = self.dataset_stats
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=torch.device("cpu"))
        model = initializer.initialize()
        self.assertEqual(model.config, config["model"])

    def test_groot_call(self):
        batch = next(iter(self.dataloader_groot))

        config = {
                "type": "groot",
                # "model_id": None,
                "model_id": "nvidia/gr00t17-lerobot-libero_10-640", # Use pretrained weights for this test to ensure forward pass works
                "wrap_with_nnsight": False,
                # "print_architecture": True
                "fixed_time": 0.6
        }
        dataset_stats = self.dataset_stats
        device = test_gpu_availability()
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=torch.device(device))
        model = initializer.initialize()
        called_modules = []
        def make_hook(name):
            def hook(module, args, kwargs, output):
                called_modules.append(name)
            return hook

        handles = []
        for name, module in model.model._groot_model.action_head.named_modules():
            handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))

        print([n for n in called_modules if "vl_self_attention" in n])
        # Preprocess batch before passing to forward
        with torch.no_grad():
            processed_batch = model.preprocess_batch(batch)
            output = model(processed_batch)
        print(f"Output sample: {output}")
        print(f"Output shape: {output.shape}")
        for h in handles:
            h.remove()

        self.assertIsNotNone(output)

    def test_pi05_initialization(self):
        config = {
            "type": "pi05",
            "model_id": None # Use None to load the model without pretrained weights for testing purposes
        }
        dataset_stats = self.dataset_stats
        initializer = ModelInitializer(config, dataset_stats=dataset_stats)
        model = initializer.initialize()
        self.assertEqual(model.config, config["model"])

    def test_pi05_call(self):
        batch = next(iter(self.dataloader))

        config = {
            "type": "pi05",
            "model_id": None,
            # "model_id": "lerobot/pi05_libero", # Use pretrained weights for this test to ensure forward pass works
            "wrap_with_nnsight": False
        }
        dataset_stats = self.dataset_stats
        device = test_gpu_availability()
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=device)
        model = initializer.initialize()

        # Preprocess batch before passing to forward
        processed_batch = model.preprocess_batch(batch)
        output = model(processed_batch)
        print(f"Output sample: {output}")
        print(f"Output shape: {output.shape}")

        self.assertIsNotNone(output)

    def test_pi05_call_pretrained(self):
        batch = next(iter(self.dataloader))

        config = {
                "type": "pi05",
                # "model_id": None,
                "model_id": "lerobot/pi05_libero_finetuned_v044", # Use pretrained weights for this test to ensure forward pass works
                "wrap_with_nnsight": False
        }
        dataset_stats = self.dataset_stats
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
        batch = next(iter(self.dataloader))

        config = {
            "type": "pi05",
            "model_id": None,
            "wrap_with_nnsight": True
        }
        device = test_gpu_availability()
        dataset_stats = self.dataset_stats
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=device)
        model = initializer.initialize()

        # Preprocess batch before passing to forward
        processed_batch = model.preprocess_batch(batch)
        output = model(processed_batch)
        print(f"Output sample: {output}")
        print(f"Output shape: {output.shape}")

        self.assertIsNotNone(output)

    def test_pi05_preprocessor(self):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        dataloader = self.dataloader
        dataset_stats = self.dataset_stats
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

    def test_groot_preprocessor(self):
        config = {
                "type": "groot",
                "model_id": None,
                # "model_id": "lerobot/pi05_libero", # Use pretrained weights for this test to ensure forward pass works
                "wrap_with_nnsight": False,
                # "print_architecture": True
                "fixed_time": 0.6
        }
        dataset_stats = self.dataset_stats
        device = test_gpu_availability()
        initializer = ModelInitializer(config, dataset_stats=dataset_stats, device=torch.device(device))
        model = initializer.initialize()
        preprocessor, postprocessor = model.preprocessor, model.postprocessor
        batch = next(iter(self.dataloader_groot))
        processed_batch = preprocessor(batch)



if __name__ == "__main__":
    t = TestModels()
    t.setUp()
    # t.test_groot_initialization()
    # t.test_groot_call()
    # t.test_groot_preprocessor()
    t.test_pi05_call_pretrained()
    # t.test_pi05_call_with_nnsight()
    # t.test_pi05_preprocessor()
    t.tearDown()
    # suite = unittest.TestSuite()
    # # suite.addTest(TestModels('test_pi05_initialization'))
    # suite.addTest(TestModels('test_pi05_call'))
    # unittest.TextTestRunner().run(suite)

    # unittest.main()
