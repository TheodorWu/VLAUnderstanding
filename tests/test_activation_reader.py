import shutil
from pathlib import Path
import unittest

import torch

from data.activation_reader import ActivationReader
from data.activation_writer import ActivationDataPoint, ActivationWriter


class TestActivationReader(unittest.TestCase):
    def setUp(self):
        self.output_dir_name = "test_activation_reader_dir"
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / self.output_dir_name
        self.test_dir = self.output_dir / "test_run"
        self.test_data_dir = self.test_dir / "data"

    def tearDown(self):
        shutil.rmtree(Path(self.output_dir), ignore_errors=True)

    def _make_config(self):
        return {
            "activation_writer": {
                "chunk_size": 4,
                "max_shard_size": 1e9,
                "output_dir": str(self.output_dir_name),
            }
        }

    def test_round_trip_reads_back_written_tensors(self):
        # resource allocation warning triggered only within test context.
        config = self._make_config()
        writer = ActivationWriter(config)

        activation_tensor = torch.randn(4, 8)
        corrupt_tensor = torch.randn(4, 8)
        gradient_tensor = torch.randn(4, 8)
        writer.add_data(
            ActivationDataPoint(
                "layer_0",
                "sample_1",
                clean=activation_tensor,
                corrupt=corrupt_tensor,
                gradients=gradient_tensor,
            )
        )
        writer.__exit__(None, None, None)

        reader = ActivationReader(config)

        datapoints = reader.read_layer("layer_0")

        self.assertEqual(len(datapoints), 1)
        self.assertTrue(torch.equal(datapoints[0].clean, activation_tensor))
        self.assertTrue(torch.equal(datapoints[0].corrupt, corrupt_tensor))
        self.assertTrue(torch.equal(datapoints[0].gradients, gradient_tensor))


if __name__ == "__main__":
    # test = TestActivationReader()
    # test.setUp()
    # test.test_round_trip_reads_back_written_tensors()
    # test.tearDown()
    unittest.main()
