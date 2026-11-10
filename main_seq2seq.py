import torch
import hydra
import lightning as L
from src.dataset import SBRDataModule
from src.model import INTEGERTrainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision('high')


@hydra.main(config_path="config", config_name="seq2seqrecommender", version_base=None)
def main(cfg: DictConfig):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.output_subdir = None
    ckpt = ModelCheckpoint(save_top_k=5, save_on_train_epoch_end=True, monitor='eval/R@5', mode='max')
    early_stopping = EarlyStopping(monitor='eval/R@5', patience=10, mode='max')
    tb_logger = pl_loggers.TensorBoardLogger('log_vilrec', f'{cfg.dataset}')
    loggers = [tb_logger]
    seed_everything(seed=cfg.seed, workers=True)
    trainer = L.Trainer(accelerator='gpu', 
                        devices=[0],
                        strategy=DDPStrategy(find_unused_parameters=False), 
                        precision='bf16-mixed',
                        max_epochs=cfg.epochs,
                        sync_batchnorm=True,
                        logger=loggers,
                        check_val_every_n_epoch=2,
                        callbacks=[ckpt, early_stopping])
    # * prepare data and model
    dm = SBRDataModule(cfg)
    model = INTEGERTrainer(cfg)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, ckpt_path='best', datamodule=dm)


if __name__ == "__main__":
    main()