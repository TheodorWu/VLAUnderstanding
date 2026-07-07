
from eval.activation_patching_evaluator import ActivationPatchingEvaluator
from eval.attribution_patching_evaluator import AttributionPatchingEvaluator
from eval.reservoir_evaluator import ReservoirEvaluator
from utils.general import get_result_layer_names

class EvaluatorPipeline:
    def __init__(self):
        self.evaluators = []
        self.results = {}

    def add_evaluator(self, evaluator):
        if isinstance(evaluator, ActivationPatchingEvaluator) and not any(isinstance(e, AttributionPatchingEvaluator) for e in self.evaluators):
            print("Warning: Adding ActivationPatchingEvaluator without AttributionPatchingEvaluator may lead to incomplete analysis. Make sure to add AttributionPatchingEvaluator before ActivationPatchingEvaluator if you want to compare results.")
        self.evaluators.append(evaluator)

    def run_evaluation(self):
        for evaluator in self.evaluators:
            if isinstance(evaluator, AttributionPatchingEvaluator):
                self._attribution_patching_evaluation(evaluator)
            elif isinstance(evaluator, ReservoirEvaluator):
                self._reservoir_evaluation(evaluator)
            elif isinstance(evaluator, ActivationPatchingEvaluator):
                self._activation_patching_evaluation(evaluator)

    def _attribution_patching_evaluation(self, evaluator: AttributionPatchingEvaluator):
        result = evaluator.compute_layer_attributions()
        self.results[evaluator.__class__.__name__] = result
        self._safe_plot(evaluator.plot_heatmap, result)
        self._safe_plot(evaluator.plot_heatmap, result, std=True)
        self._safe_plot(evaluator.plot_layer_scores, result)
        self._safe_plot(evaluator.plot_norm_heatmap, result)
        self._safe_plot(evaluator.plot_layer_distributions, result)
        self._safe_plot(evaluator.plot_sample_metadata_dist)

    def _reservoir_evaluation(self, evaluator: ReservoirEvaluator):
        data_root = evaluator.activation_reader.data_root
        layer_names = get_result_layer_names(data_root)
        layer_names = evaluator.layer_sort_fn(layer_names)
        cka_results = []
        self.results[evaluator.__class__.__name__] = {
            'pca_results': {},
            'cka_results': {}
        }
        for layer in layer_names:
            reservoir = evaluator.build_reservoir(layer=layer, fields=["clean", "corrupt"])
            try:
                pca_result = evaluator.compute_pca(reservoir)
                self._safe_plot(evaluator.plot_pca, pca_result)
                self.results[evaluator.__class__.__name__]['pca_results'][layer] = pca_result
            except Exception as e:
                print(f"Error computing PCA for layer {layer}: {e}")

            try:
                cka_result = evaluator.compute_cka_from_reservoir(reservoir)
                self.results[evaluator.__class__.__name__]['cka_results'][layer] = cka_result
                cka_results.append(cka_result)
            except Exception as e:
                print(f"Error computing CKA for layer {layer}: {e}")

        self._safe_plot(evaluator.plot_perturbation_cka, cka_results)

    def _activation_patching_evaluation(self, evaluator: ActivationPatchingEvaluator):
        result = evaluator.compute_layer_patching_effects()
        self.results[evaluator.__class__.__name__] = result
        self._safe_plot(evaluator.plot_patching_distribution, result)
        self._safe_plot(evaluator.plot_patching_heatmap, result)

        if AttributionPatchingEvaluator.__name__ in self.results:
            atp_result = self.results[AttributionPatchingEvaluator.__name__]
            self._safe_plot(evaluator.plot_atp_vs_patching, atp_result, result)


    def _safe_plot(self, plot_fn, *args, **kwargs):
        try:
            plot_fn(*args, **kwargs)
        except Exception as e:
            print(f"Error plotting: {e}")
