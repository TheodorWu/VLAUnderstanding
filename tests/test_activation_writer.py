import unittest
from unittest.mock import MagicMock
import shutil
from pathlib import Path
import torch

from data.activation_writer import ActivationWriter, ActivationDataPoint


class TestActivationWriter(unittest.TestCase):
    def setUp(self):
        self.output_dir_name = "test_activation_writer_dir"
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / self.output_dir_name
        self.test_dir = self.output_dir / "test_run"
        self.test_data_dir = self.test_dir / "data"

    def tearDown(self):
        shutil.rmtree(Path(self.output_dir), ignore_errors=True)

    def _make_writer(self):
        config = {
            "activation_writer": {
                "chunk_size": 4,
                "max_shard_size": 1e9,
                "output_dir": str(self.output_dir_name)
            }
        }
        writer = ActivationWriter(config)
        return writer

    # --- directory initialisation ---

    def test_init_directory_creates_data_root(self):
        self._make_writer()
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(self.test_data_dir.exists())

    # --- sink creation ---

    def test_get_sink_creates_directory(self):
        writer = self._make_writer()
        writer._get_sink("layer_0", "activations")
        self.assertTrue((self.test_data_dir / "layer_0" / "activations").exists())
        writer.__exit__(None, None, None)

    def test_get_sink_reuses_existing_sink(self):
        writer = self._make_writer()
        sink_a = writer._get_sink("layer_0", "activations")
        sink_b = writer._get_sink("layer_0", "activations")
        self.assertIs(sink_a, sink_b)
        writer.__exit__(None, None, None)

    def test_separate_sinks_for_activations_and_gradients(self):
        writer = self._make_writer()
        act_sink = writer._get_sink("layer_0", "activations")
        grad_sink = writer._get_sink("layer_0", "gradients")
        self.assertIsNot(act_sink, grad_sink)
        writer.__exit__(None, None, None)

    def test_separate_sinks_per_layer(self):
        writer = self._make_writer()
        sink_l0 = writer._get_sink("layer_0", "activations")
        sink_l1 = writer._get_sink("layer_1", "activations")
        self.assertIsNot(sink_l0, sink_l1)
        writer.__exit__(None, None, None)

    # --- add_data ---

    def test_add_activations_only_creates_activation_sink(self):
        writer = self._make_writer()
        point = ActivationDataPoint("layer_0", "s1", activations=torch.randn(4, 8))
        writer.add_data(point)
        self.assertIn("layer_0_activations", writer.sinks)
        self.assertNotIn("layer_0_gradients", writer.sinks)
        writer.__exit__(None, None, None)

    def test_add_gradients_only_creates_gradient_sink(self):
        writer = self._make_writer()
        point = ActivationDataPoint("layer_0", "s1", gradients=torch.randn(4, 8))
        writer.add_data(point)
        self.assertIn("layer_0_gradients", writer.sinks)
        self.assertNotIn("layer_0_activations", writer.sinks)
        writer.__exit__(None, None, None)

    def test_add_both_creates_both_sinks(self):
        writer = self._make_writer()
        point = ActivationDataPoint(
            "layer_0", "s1",
            activations=torch.randn(4, 8),
            gradients=torch.randn(4, 8),
        )
        writer.add_data(point)
        self.assertIn("layer_0_activations", writer.sinks)
        self.assertIn("layer_0_gradients", writer.sinks)
        writer.__exit__(None, None, None)

    def test_add_none_fields_creates_no_sinks(self):
        writer = self._make_writer()
        point = ActivationDataPoint("layer_0", "s1")
        writer.add_data(point)
        self.assertEqual(len(writer.sinks), 0)

    def test_data_written_to_correct_layer_dir(self):
        writer = self._make_writer()
        point = ActivationDataPoint("layer_2", "s1", activations=torch.randn(4, 8))
        writer.add_data(point)
        writer.__exit__(None, None, None)
        tar_files = list((self.test_data_dir / "layer_2" / "activations").glob("*.tar"))
        self.assertGreater(len(tar_files), 0)

    def test_multiple_layers_write_to_separate_dirs(self):
        writer = self._make_writer()
        writer.add_data(ActivationDataPoint("layer_0", "s1", activations=torch.randn(4, 8)))
        writer.add_data(ActivationDataPoint("layer_1", "s1", activations=torch.randn(4, 8)))
        writer.__exit__(None, None, None)
        self.assertTrue((self.test_data_dir / "layer_0" / "activations").exists())
        self.assertTrue((self.test_data_dir / "layer_1" / "activations").exists())

    # --- exit ---

    def test_exit_closes_all_sinks(self):
        writer = self._make_writer()
        writer._get_sink("layer_0", "activations")
        writer._get_sink("layer_1", "gradients")
        for sink in writer.sinks.values():
            sink.close = MagicMock()
        writer.__exit__(None, None, None)
        for sink in writer.sinks.values():
            sink.close.assert_called_once()

if __name__ == "__main__":
    # test = TestActivationWriter()
    # test.setUp()
    # test.test_get_sink_creates_directory()
    # test.tearDown()
    unittest.main()
