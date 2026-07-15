import unittest
import shutil
from pathlib import Path
import torch
import wandb
import numpy as np

from data.activation_writer import ActivationDataBatch, ActivationWriter, SampleMetadata
from eval.activation_patching_evaluator import ActivationPatchingEvaluator
from eval.attribution_patching_evaluator import AttributionResult

class TestEvaluationActivationPatching(unittest.TestCase):
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
        for layer in ["layer_0", "layer_1"]:
            for i in range(10):
                size = 10 if i % 2 == 0 else 8
                writer.add_data(
                    ActivationDataBatch(
                        layer=layer,
                        sample_ids=[f"sample_{j + i*10}" for j in range(size)],
                        patched_loss=torch.randn(size),   # fresh draw each time
                        clean_loss=torch.randn(size),
                        corrupted_loss=torch.randn(size)
                    )
                )
                # writer.add_sample_metadata(SampleMetadata(
                #         sample_id=[f"sample_{j + i*10}" for j in range(10)],
                #         instruction=f"Instruction {i}",
                #         corrupt_instruction=f"Corrupt Instruction {i}",
                #         perturbed_token_idxs=[i % 5]
                #     )
                # )
        writer.__exit__(None, None, None)

    def tearDown(self):
        shutil.rmtree(Path(self.output_dir), ignore_errors=True)

    def test_activation_patching_plots(self):
        evaluator = ActivationPatchingEvaluator(
            config={
                "activation_reader": self._make_config(),
                "evaluator": {
                    "show": True
                }
            }
        )
        result = evaluator.compute_layer_patching_effects()
        for l in result.layer_samples:
            arr = np.array(result.layer_samples[l])
            print(l, len(arr), arr.std(), np.isnan(arr).any(), np.isinf(arr).any())
        evaluator.log_layer_scores(result)
        evaluator.plot_patching_distribution(result, invert=True)
        evaluator.plot_patching_heatmap(result, invert=True)
        atp_result = AttributionResult(
            perturbation_type=result.perturbation_type,
            layer_names=result.layer_names,
            scalar_scores=result.scalar_scores -1,
            layer_samples=result.layer_samples,
            matrix=None,
            std_matrix=None
        )
        evaluator.plot_atp_vs_patching(atp_result, result)




if __name__ == "__main__":
    t = TestEvaluationActivationPatching()
    t.setUp()
    t.test_activation_patching_plots()
    t.tearDown()
    # unittest.main()
