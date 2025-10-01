import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import hydra
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelConfusionMatrix,
    MultilabelPrecision
)
from torchmetrics.wrappers import BootStrapper
from mlp import MLP
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from tqdm import tqdm

def _load_concat_dataset(embed_paths, label_paths):
    X_list = [torch.load(p) for p in embed_paths]
    y_list = [torch.load(p) for p in label_paths]
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return TensorDataset(X, y)

def sensitivity_macro(y_true, y_pred, num_labels):
    cm = MultilabelConfusionMatrix(num_labels=num_labels).to("cpu")
    conf = cm(torch.tensor(y_pred), torch.tensor(y_true))
    sens = conf[:, 1, 1] / (conf[:, 1, 1] + conf[:, 1, 0] + 1e-8)
    return sens.mean().item()

def specificity_macro(y_true, y_pred, num_labels):
    cm = MultilabelConfusionMatrix(num_labels=num_labels).to("cpu")
    conf = cm(torch.tensor(y_pred), torch.tensor(y_true))
    spec = conf[:, 0, 0] / (conf[:, 0, 0] + conf[:, 0, 1] + 1e-8)
    return spec.mean().item()


MODEL_CKPT = '/home/free4ky/projects/chest-diseases/model_multilabel20_pr034_rec0777.pth'


@hydra.main(version_base=None, config_path="../configs", config_name="config_multilabel_test.yaml")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = 20


    logger.info("Loading model...")
    model = MLP(
        input_size=config.model.input_size,
        activation=config.model.activation,
        dropout=config.model.dropout,
        hidden_sizes=config.model.hidden_sizes,
        num_classes=num_labels,
    )
    state_dict = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    val_ds = _load_concat_dataset(config.dataset.test.embeds, config.dataset.test.labels)
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check embed_paths and label_paths in config.")

    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    logger.info("Computing pathology probabilities...")
    probs, labels = [], []
    with torch.no_grad():
        for emb, y in tqdm(val_dl):
            emb = torch.nn.functional.normalize(emb, dim=-1).to(device)
            logits = model(emb)
            prob = torch.sigmoid(logits).cpu()
            probs.append(prob)
            labels.append(y)

    probs = torch.cat(probs, dim=0)
    labels = torch.cat(labels, dim=0).long()
    preds_05 = (probs > 0.5).int()

    # --- Wrap metrics in BootStrapper ---
    auroc_bs = BootStrapper(
        MultilabelAUROC(num_labels=num_labels, average="macro"),
        num_bootstraps=1000, quantile=.95
    )
    auprc_bs = BootStrapper(
        MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        num_bootstraps=1000, quantile=.95
    )
    f1_bs = BootStrapper(
        MultilabelF1Score(num_labels=num_labels, average="macro"),
        num_bootstraps=1000, quantile=.95
    )
    prec_bs = BootStrapper(
        MultilabelPrecision(num_labels=num_labels, average="macro"),
        num_bootstraps=1000, quantile=.95
    )

    # --- Update with predictions ---
    auroc_bs.update(probs, labels)
    auprc_bs.update(probs, labels)
    f1_bs.update(preds_05, labels)
    prec_bs.update(preds_05, labels)


    # --- Compute outputs ---
    auroc_res = auroc_bs.compute()
    auprc_res = auprc_bs.compute()
    f1_res = f1_bs.compute()
    prec_res  = prec_bs.compute()

    auroc_mean, auroc_std = auroc_res['mean'].item(), auroc_res['std'].item()
    auprc_mean, auprc_std = auprc_res['mean'].item(), auprc_res['std'].item()
    f1_mean, f1_std = f1_res['mean'].item(), f1_res['std'].item()
    prec_mean, prec_std   = prec_res['mean'].item(), prec_res['std'].item()

    # Для CI можно использовать mean ± 1.96 * std (приблизительно 95%)
    auroc_ci = (auroc_mean - 1.96*auroc_std, auroc_mean + 1.96*auroc_std)
    auprc_ci = (auprc_mean - 1.96*auprc_std, auprc_mean + 1.96*auprc_std)
    f1_ci    = (f1_mean - 1.96*f1_std, f1_mean + 1.96*f1_std)
    prec_ci  = (prec_mean - 1.96*prec_std, prec_mean + 1.96*prec_std)

    # Sensitivity/Specificity через bootstrap руками (torchmetrics пока не умеет)
    sens_values, spec_values = [], []
    n = len(labels)
    rng = np.random.default_rng(seed=42)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        sens_values.append(sensitivity_macro(labels[idx].numpy(), preds_05[idx].numpy(), num_labels))
        spec_values.append(specificity_macro(labels[idx].numpy(), preds_05[idx].numpy(), num_labels))
    sens_mean, sens_ci = np.mean(sens_values), (np.percentile(sens_values, 2.5), np.percentile(sens_values, 97.5))
    spec_mean, spec_ci = np.mean(spec_values), (np.percentile(spec_values, 2.5), np.percentile(spec_values, 97.5))

    # --- Collect into table ---
    metrics = {
        "AUROC":      (auroc_mean, *auroc_ci),
        "AUPRC":      (auprc_mean, *auprc_ci),
        "F1":         (f1_mean, *f1_ci),
        "Precision":  (prec_mean, *prec_ci),
        "Sensitivity":(sens_mean, *sens_ci),
        "Specificity":(spec_mean, *spec_ci),
    }

    metrics_df = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["Mean", "Lower", "Upper"]
    ).reset_index().rename(columns={"index": "Metric"})

    logger.info("\n" + metrics_df.to_string(index=False))
    metrics_df.to_excel("mosmed_macro_bootstrapper.xlsx", index=False)


if __name__ == "__main__":
    main()
