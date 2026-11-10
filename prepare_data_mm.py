import hydra
import os
# from src.preprocessing import preprocess
from src.qwenvilencode import preprocess
from omegaconf import DictConfig, OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="config", config_name="r2vq", version_base=None)
def main(cfg: DictConfig):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.output_subdir = None
    preprocess(cfg)


if __name__ == "__main__":
    main()