import torch
import hydra
import os
import lightning as L
from src.utils import items_count
from src.dataset import SemIDDataModule
from src.rqblock import RQ_VAEPretrainer
from loguru import logger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="config", config_name="r2vq", version_base=None)
def main(cfg: DictConfig):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.output_subdir = None
    OmegaConf.update(cfg, "item_count", items_count(cfg), merge=True)
    # pretrain the rq-vae is a self-supervised task, so there is no evaluation and testing
    ckpt = ModelCheckpoint(save_top_k=10, save_on_train_epoch_end=True, monitor='val_r2vq/collision', mode='min')
    tb_logger = pl_loggers.TensorBoardLogger('log_drvq', f'{cfg.dataset}')
    loggers = [tb_logger]
    seed_everything(seed=cfg.seed, workers=True)
    trainer = L.Trainer(accelerator='gpu', 
                        devices=[0],
                        strategy=DDPStrategy(find_unused_parameters=True), 
                        precision='bf16-mixed',
                        max_epochs=cfg.rq_epoch,
                        sync_batchnorm=True,
                        logger=loggers,
                        check_val_every_n_epoch=2,
                        callbacks=[ckpt])
    # * prepare data and model
    dm = SemIDDataModule(cfg)
    model = RQ_VAEPretrainer(cfg)
    trainer.fit(model, dm)
    trainer.predict(model, ckpt_path='last', datamodule=dm)


if __name__ == "__main__":
    main()