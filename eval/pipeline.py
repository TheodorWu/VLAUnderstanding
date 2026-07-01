
from eval.attribution_patching_evaluator import AttributionPatchingEvaluator
from eval.reservoir_evaluator import ReservoirEvaluator
from utils.general import get_result_layer_names

class EvaluatorPipeline:
    def __init__(self):
        self.evaluators = []

    def add_evaluator(self, evaluator):
        self.evaluators.append(evaluator)

    def run_evaluation(self):
        for evaluator in self.evaluators:
            if isinstance(evaluator, AttributionPatchingEvaluator):
                result = evaluator.compute_layer_attributions()
                self._safe_plot(evaluator.plot_heatmap, result)
                self._safe_plot(evaluator.plot_heatmap, result, std=True)
                self._safe_plot(evaluator.plot_layer_scores, result)
                self._safe_plot(evaluator.plot_norm_heatmap, result)
                self._safe_plot(evaluator.plot_layer_distributions, result)
                self._safe_plot(evaluator.plot_sample_metadata_dist)
            elif isinstance(evaluator, ReservoirEvaluator):
                data_root = evaluator.activation_reader.data_root
                layer_names = get_result_layer_names(data_root)
                for layer in layer_names:
                    reservoir = evaluator.build_reservoir(layer=layer, fields=["clean", "corrupt"])
                    pca_result = evaluator.compute_pca(reservoir)
                    self._safe_plot(evaluator.plot_pca, pca_result)

                results = evaluator.compute_all_perturbation_cka(
                    layer_names=layer_names,
                )
                self._safe_plot(evaluator.plot_perturbation_cka, results)

    def _safe_plot(self, plot_fn, *args, **kwargs):
        try:
            plot_fn(*args, **kwargs)
        except Exception as e:
            print(f"Error plotting: {e}")
