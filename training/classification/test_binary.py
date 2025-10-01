import os
import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import HeadDataModule
from pytorch_lightning import seed_everything
from src.ctrate_binary_model import LightningMLP


@hydra.main(version_base=None, config_path='configs', config_name='config_binary_test.yaml')
def test_vit_mlp_pipeline(cfg):
    # fix seed
    seed_everything(42)
    tb_logger = TensorBoardLogger(save_dir='tb_test')
    # init datamodule
    dm = HeadDataModule(config=cfg)
    # init model
    model = LightningMLP(config=cfg, num_classes=2)
    model.model.eval()
    # train
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator='gpu',
        devices=[0],
        precision=16
    )
    trainer.test(
        model,
        datamodule=dm,
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/logs/tb_logs_binary_oversample/lightning_logs/version_5/checkpoints/vit_mlp-epoch=93-val_f1=0.983160138130188.ckpt'
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_binary_ctrate_mosmed/2025-09-26/19-44-22/14/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=49-val_f1=0.9046242833137512.ckpt'
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_binary_ctrate_mosmed/2025-09-26/20-04-11/11/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=05-val_f1=0.8788679242134094.ckpt'
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_binary_ctrate_mosmed/2025-09-25/20-26-23/18/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=99-val_f1=0.9795153737068176.ckpt'
        ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_binary_ctrate_mosmed/2025-09-26/20-53-49/9/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=17-val_f1=0.8754298686981201.ckpt'
    )

if __name__ == '__main__':
    test_vit_mlp_pipeline()
