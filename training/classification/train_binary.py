import pytorch_lightning as pl
import hydra
from src.utils import get_metric_value
from src.datamodule import HeadDataModule
# from src.ctrate_datamodule import CTRateDataModule
from datasets import load_dataset
# from src.model import LightningMLP
# from training.classification.src.ctrate_multilabel_model import LightningMLP
from training.classification.src.ctrate_binary_model import LightningMLP
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(version_base=None, config_path='configs', config_name='config_binary.yaml')
def train_pipeline(cfg):
    # fix seed
    seed_everything(42)
    # init logger
    tb_logger = TensorBoardLogger(cfg.save_dir)
    # init datamodule
    dm = HeadDataModule(cfg)
    # init model
    model = LightningMLP(config=cfg, num_classes=2)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=cfg.optimized_metric,
        mode="max",
        filename="vit_mlp-{epoch:02d}-{val_f1}",
    )
    # checkpoint_callback2 = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="train_f1",
    #     mode="max",
    #     filename="vit_mlp-{epoch:02d}-{train_f1}",
    # )

    # train
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_f1", mode="max"), 
            checkpoint_callback,
            # checkpoint_callback2
        ],
        logger=tb_logger,
        accelerator='gpu',
        devices=[0],
        min_epochs=1,
        max_epochs=cfg.max_epochs,
        precision=16,
    )
    trainer.fit(model, dm)
    train_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics}

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == '__main__':
    train_pipeline()