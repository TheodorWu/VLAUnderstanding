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
        evaluator.run_evaluation()

    if cfg.mode == "inference":
        with initializer.method_inference() as method:
            method.main()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
