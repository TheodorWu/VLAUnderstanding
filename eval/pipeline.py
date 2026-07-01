
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
                self._attribution_patching_evaluation(evaluator)
            elif isinstance(evaluator, ReservoirEvaluator):
                self._reservoir_evaluation(evaluator)

    def _attribution_patching_evaluation(self, evaluator: AttributionPatchingEvaluator):
        result = evaluator.compute_layer_attributions()
        self._safe_plot(evaluator.plot_heatmap, result)
        self._safe_plot(evaluator.plot_heatmap, result, std=True)
        self._safe_plot(evaluator.plot_layer_scores, result)
        self._safe_plot(evaluator.plot_norm_heatmap, result)
        self._safe_plot(evaluator.plot_layer_distributions, result)
        self._safe_plot(evaluator.plot_sample_metadata_dist)

    def _reservoir_evaluation(self, evaluator: ReservoirEvaluator):
        data_root = evaluator.activation_reader.data_root
        layer_names = get_result_layer_names(data_root)
        cka_results = []
        for layer in layer_names:
            reservoir = evaluator.build_reservoir(layer=layer, fields=["clean", "corrupt"])
            try:
                pca_result = evaluator.compute_pca(reservoir)
                self._safe_plot(evaluator.plot_pca, pca_result)
            except Exception as e:
                print(f"Error computing PCA for layer {layer}: {e}")

            try:
                cka_result = evaluator.compute_cka_from_reservoir(reservoir)
                cka_results.append(cka_result)
            except Exception as e:
                print(f"Error computing CKA for layer {layer}: {e}")

        self._safe_plot(evaluator.plot_perturbation_cka, cka_results)


    def _safe_plot(self, plot_fn, *args, **kwargs):
        try:
            plot_fn(*args, **kwargs)
        except Exception as e:
            print(f"Error plotting: {e}")
