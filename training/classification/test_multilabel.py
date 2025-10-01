import os
import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import HeadDataModule
from pytorch_lightning import seed_everything
from src.ctrate_multilabel_model import LightningMLP


@hydra.main(version_base=None, config_path='configs', config_name='config_multilabel_test.yaml')
def test_vit_mlp_pipeline(cfg):
    # fix seed
    seed_everything(42)
    tb_logger = TensorBoardLogger(save_dir='tb_test')
    # init datamodule
    dm = HeadDataModule(config=cfg)
    # init model
    model = LightningMLP(config=cfg, num_classes=dm.num_classes)
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
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-07-45/27/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=50-val_f1_macro=0.5806165337562561.ckpt'
        # ckpt_path='/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-30-30/4/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=195-val_f1_macro=0.8323163986206055.ckpt'
        # ckpt_path = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-07-45/0/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=12-val_f1_macro=0.43588829040527344.ckpt'
        # ckpt_path = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/13-07-45/12/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=27-val_f1_macro=0.48999685049057007.ckpt'
        # ckpt_path = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/18-38-35/10/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=12-val_f1_macro=0.44971513748168945.ckpt'
        # ckpt_path = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/18-56-21/10/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=10-val_f1_macro=0.4467194080352783.ckpt' # good
        ckpt_path = '/home/free4ky/projects/chest-diseases/training/classification/runs/output/outputs_multibel/2025-09-26/19-16-52/1/logs/lightning_logs/version_0/checkpoints/vit_mlp-epoch=49-val_f1_macro=0.45349034667015076.ckpt'
        
    )

if __name__ == '__main__':
    test_vit_mlp_pipeline()
