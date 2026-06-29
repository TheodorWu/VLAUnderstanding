
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
                evaluator.plot_heatmap(result)
                evaluator.plot_heatmap(result, std=True)
                evaluator.plot_layer_scores(result)
                evaluator.plot_norm_heatmap(result)
                evaluator.plot_layer_distributions(result)
            elif isinstance(evaluator, ReservoirEvaluator):
                data_root = evaluator.activation_reader.data_root
                layer_names = get_result_layer_names(data_root)
                for layer in layer_names:
                    reservoir = evaluator.build_reservoir(layer=layer, fields=["clean", "corrupt"])
                    pca_result = evaluator.compute_pca(reservoir)
                    evaluator.plot_pca(pca_result)

                results = evaluator.compute_all_perturbation_cka(
                    layer_names=layer_names,
                )
                evaluator.plot_perturbation_cka(results)
