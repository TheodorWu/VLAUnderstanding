from initializer import Initializer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    initializer = Initializer(config)

    if cfg.mode == "run" or cfg.mode == "full":
        method = initializer.method()
        method.main()
    elif cfg.mode == "evaluate" or cfg.mode == "full":
        evaluator = initializer.evaluate()
        result = evaluator.compute_layer_attributions()
        evaluator.plot_heatmap(result, **config["evaluator"]["params"])

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
