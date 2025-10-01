from typing import Any, List
import torch
from torch import nn
from hydra.utils import instantiate
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelAccuracy,
)
from collections import defaultdict
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, auc
from .mlp import MLP


class LightningMLP(pl.LightningModule):
    def __init__(self, config=None, num_classes=20):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.model = MLP(
            input_size=config.model.input_size,
            activation=config.model.activation,
            dropout=config.model.dropout,
            num_classes=num_classes,
            hidden_sizes=config.model.hidden_sizes
        )

        weights = torch.tensor([
            9.211362733, 2.384068466, 8.295479204, 32.8629776, 2.992233613,
            6.064870808, 3.176470588, 4.187083754, 3.022222222, 1.216071737,
            1.677849552, 3.152851834, 7.123261694, 18.16629381, 13.8480647,
            6.335045662, 10.81701149, 13.40695067, 52.9287, 83.6382
        ]).cuda()

        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=weights)

        # Metrics
        self.auroc_macro = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.auroc_per_class = MultilabelAUROC(num_labels=num_classes, average=None)
        self.f1_macro = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.f1_per_class = MultilabelF1Score(num_labels=num_classes, average=None)
        self.precision = MultilabelPrecision(num_labels=num_classes, average="macro")
        self.recall = MultilabelRecall(num_labels=num_classes, average="macro")
        self.accuracy = MultilabelAccuracy(num_labels=num_classes, average='macro')

        self.accum = defaultdict(lambda: torch.tensor([]))
        self.aupr_macro = None
        self.aupr_per_class = None

    def forward(self, x) -> Any:
        return self.model(x)

    def on_training_epoch_start(self):
        self.accum = defaultdict(lambda: torch.tensor([]))

    def _common_step(self, batch, batch_idx, mode: str):
        x, y = batch
        x = torch.nn.functional.normalize(x, dim=-1)
        logits = self.forward(x)
        loss = self.loss_func(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.accum[f'{mode}_preds'] = torch.cat([self.accum[f'{mode}_preds'], preds.detach().cpu()])
        self.accum[f'{mode}_y'] = torch.cat([self.accum[f'{mode}_y'], y.detach().cpu()])
        self.accum[f'{mode}_probs'] = torch.cat([self.accum[f'{mode}_probs'], probs.detach().cpu()])
        return x, y, preds, logits, loss

    def _calc_metric(self, mode):
        y, preds, probs = self.accum[f'{mode}_y'], self.accum[f'{mode}_preds'], self.accum[f'{mode}_probs']
        device = probs.device if probs.is_cuda else "cpu"

        self.auroc_macro = self.auroc_macro.to(device)
        self.auroc_per_class = self.auroc_per_class.to(device)
        self.f1_macro = self.f1_macro.to(device)
        self.f1_per_class = self.f1_per_class.to(device)
        self.precision = self.precision.to(device)
        self.recall = self.recall.to(device)
        self.accuracy = self.accuracy.to(device)

        # Macro metrics
        auroc_macro = self.auroc_macro(probs, y.int())
        precision = self.precision(probs, y.int())
        recall = self.recall(probs, y.int())
        f1_macro = self.f1_macro(probs, y.int())
        accuracy = self.accuracy(probs, y.int())

        # Per-class metrics only for test
        if mode == "test":
            auroc_per_class = self.auroc_per_class(probs, y.int())
            f1_per_class = self.f1_per_class(probs, y.int())

            # Compute AU-PR per class
            self.aupr_per_class = []
            for i in range(probs.shape[1]):
                p, r, _ = precision_recall_curve(y[:, i].numpy(), probs[:, i].numpy())
                self.aupr_per_class.append(auc(r, p))
            self.aupr_per_class = torch.tensor(self.aupr_per_class)

            # Macro AU-PR
            y_flat = y.flatten()
            probs_flat = probs.flatten()
            p, r, _ = precision_recall_curve(y_flat.numpy(), probs_flat.numpy())
            self.aupr_macro = auc(r, p)
        else:
            auroc_per_class = None
            f1_per_class = None
            self.aupr_per_class = None
            self.aupr_macro = None

        # Reset accum
        self.accum[f'{mode}_y'] = torch.tensor([])
        self.accum[f'{mode}_preds'] = torch.tensor([])
        self.accum[f'{mode}_probs'] = torch.tensor([])

        return auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs.cpu(), y.cpu()

    def _log_step(self, mode, auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y):
        self.log(f"{mode}_auroc_macro", auroc_macro, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{mode}_precision", precision, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{mode}_recall", recall, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{mode}_f1_macro", f1_macro, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(f"{mode}_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if mode == "test":
            # if f1_per_class is not None:
            #     for i, val in enumerate(f1_per_class):
            #         self.log(f"{mode}_f1_class_{i}", val, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            # if auroc_per_class is not None:
            #     for i, val in enumerate(auroc_per_class):
            #         self.log(f"{mode}_auroc_class_{i}", val, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

            # # Log AU-PR
            # if self.aupr_per_class is not None:
            #     for i, val in enumerate(self.aupr_per_class):
            #         self.log(f"{mode}_aupr_class_{i}", val.item(), prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.aupr_macro is not None:
                self.log(f"{mode}_aupr_macro", self.aupr_macro, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        x, y, preds, logits, loss = self._common_step(batch, batch_idx, mode='train')
        lr = self.lr_schedulers().get_last_lr()[-1]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, mode='val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, mode='test')

    # def _make_line_plot(self, x:List[Any], y:List[Any], x_label:str, y_label:str, plot_name:str) -> None:
    #     pr_fig = sns.lineplot(x=x, y=y).get_figure()
    #     pr_ax = pr_fig.gca()
    #     pr_ax.set_xlabel(x_label)
    #     pr_ax.set_ylabel(y_label)
    #     plt.close(pr_fig)
    #     self.logger.experiment.add_figure(plot_name, pr_fig, self.global_step)

    def on_test_epoch_end(self) -> None:
        auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y = self._calc_metric(mode='test')
        self._log_step('test', auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y)
    
    def on_validation_epoch_end(self):
        auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y = self._calc_metric(mode='val')
        self._log_step('val', auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y)
    
    def on_train_epoch_end(self):
        auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y = self._calc_metric(mode='train')
        self._log_step('train', auroc_macro, auroc_per_class, precision, recall, f1_macro, f1_per_class, accuracy, probs, y)

    def configure_optimizers(self) -> Any:
        optimizer = instantiate(self.config.optimizer, self.parameters())
        scheduler = instantiate(self.config.scheduler.body, optimizer=optimizer)
        config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': self.config.scheduler.pl_cfg.interval
            }
        }
        return config