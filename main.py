from initializer import Initializer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="dev")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    initializer = Initializer(config)

    if cfg.mode == "run" or cfg.mode == "full":
        with initializer.method() as method:
            method.main()

    if cfg.mode == "evaluate" or cfg.mode == "full":
        evaluator = initializer.evaluate()
        result = evaluator.compute_layer_attributions()
        evaluator.plot_heatmap(result)
        evaluator.plot_heatmap(result, std=True)
        evaluator.plot_layer_scores(result)
        evaluator.plot_norm_heatmap(result)
        evaluator.plot_layer_distributions(result)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
