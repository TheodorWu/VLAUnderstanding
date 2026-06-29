import unittest
import shutil
from pathlib import Path
import numpy as np
import torch

from data.activation_writer import ActivationWriter, ActivationDataPoint, SampleMetadata
from eval.attribution_patching_evaluator import AttributionPatchingEvaluator, AttributionResult
from eval.reservoir_evaluator import ReservoirEvaluator

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.output_dir_name = "test_activation_reader_dir"
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / self.output_dir_name
        self.test_dir = self.output_dir / "test_run"
        self.test_data_dir = self.test_dir / "data"
        self._create_dummy_layer_activation_data()

    def _make_config(self):
        return {
                "chunk_size": 4,
                "max_shard_size": 1e9,
                "output_dir": str(self.output_dir_name),
            }

    def _create_dummy_layer_activation_data(self):
        writer_config = {
            "activation_writer": self._make_config(),
            "perturbator": {
                "directional": True
            }
        }
        writer = ActivationWriter(writer_config)
        # Create dummy activation data for testing
        clean_tensor = [torch.randn(4, 10, 768),torch.randn(4, 8, 768)]  # (batch, seq, d_model)
        corrupt_tensor = [torch.randn(4, 10, 768),torch.randn(4, 8, 768)]
        gradient_tensor = [torch.randn(4, 10, 768),torch.randn(4, 8, 768)]
        for layer in ["layer_0", "layer_1"]:
            for i in range(10):
                writer.add_data(
                    ActivationDataPoint(
                        layer=layer,
                        sample_id=f"sample_{i}",
                        clean=clean_tensor[i%2],
                        corrupt=corrupt_tensor[i%2],
                        gradients=gradient_tensor[i%2],
                    )
                )
                writer.add_sample_metadata(SampleMetadata(
                        sample_id=f"sample_{i}",
                        instruction=f"Instruction {i}",
                        corrupt_instruction=f"Corrupt Instruction {i}",
                        perturbed_token_idxs=[i % 5]
                    )
                )
        writer.__exit__(None, None, None)

    def tearDown(self):
        shutil.rmtree(Path(self.output_dir), ignore_errors=True)

    def test_compute_layer_attributions(self):
        evaluator = AttributionPatchingEvaluator(
            config={
                "activation_reader": self._make_config()
            }
        )
        self.assertIsNotNone(evaluator)
        result = evaluator.compute_layer_attributions()

        self.assertIsInstance(result, AttributionResult)
        self.assertEqual(result.matrix.shape, (2, 10))
        self.assertEqual(result.layer_names, ["layer_0", "layer_1"])
        self.assertFalse(np.isnan(result.matrix).any())     # no NaNs from bad reduction
        self.assertEqual(result.perturbation_type, "directional")

    def test_plot_heatmap(self):
        evaluator = AttributionPatchingEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {
                    "show": True
                }
            }
        )
        result = AttributionResult(
            perturbation_type="directional",
            layer_names=["layer_0", "layer_1"],
            matrix=np.random.randn(2, 10),
            std_matrix=np.random.randn(2, 10)
        )
        evaluator.plot_heatmap(result)

    def test_compute_into_plot(self):
        evaluator = AttributionPatchingEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {
                    "show": True
                }
            }
        )
        result = evaluator.compute_layer_attributions()
        evaluator.plot_heatmap(result)
        evaluator.plot_heatmap(result, std=True)
        evaluator.plot_layer_scores(result)
        evaluator.plot_norm_heatmap(result)
        evaluator.plot_layer_distributions(result)

    def test_plot_sample_metadata_dist(self):
        evaluator = AttributionPatchingEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {
                    "show": True
                }
            }
        )
        evaluator.plot_sample_metadata_dist()

    def test_pca(self):
        evaluator = ReservoirEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {"show": True}
            }
        )

        reservoir = evaluator.build_reservoir(layer="layer_0", n_samples=32, fields=["clean", "corrupt"])

        pca_result= evaluator.compute_pca(reservoir)

        evaluator.plot_pca(pca_result)

    def test_cka(self):
        evaluator = ReservoirEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {"show": True}
            }
        )

        results = evaluator.compute_all_perturbation_cka(
            layer_names=["layer_0", "layer_1"],
            n_samples=32,
        )
        evaluator.plot_perturbation_cka(results)

if __name__ == "__main__":
    # t = TestEvaluation()
    # t.setUp()
    # t.test_pca()
    # t.test_cka()
    # t.tearDown()
    unittest.main()
