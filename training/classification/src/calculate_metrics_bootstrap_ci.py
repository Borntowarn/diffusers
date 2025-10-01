import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import hydra
from torchmetrics import PrecisionRecallCurve, F1Score, ConfusionMatrix
from typing import List, Any
from torcheval.metrics.functional import binary_auprc, binary_auroc
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from collections import defaultdict
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


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """
    Бутстреп для доверительного интервала метрики
    """
    n = len(y_true)
    stats = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
        lower = np.percentile(stats, 100 * alpha / 2)
        upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return np.mean(stats), (lower, upper)


def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn + 1e-8)


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)


MODEL_CKPT = '/home/free4ky/projects/chest-diseases/model_binary_2025-09-29_10-34-57_3.pth'


@hydra.main(version_base=None, config_path="../configs", config_name="config_binary_test.yaml")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from checkpoint: {MODEL_CKPT}")
    model = MLP(
        input_size=config.model.input_size,
        activation=config.model.activation,
        dropout=config.model.dropout,
        hidden_sizes=config.model.hidden_sizes,
        num_classes=2,
    )
    state_dict = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    val_ds = _load_concat_dataset(
        config.dataset.test.embeds, config.dataset.test.labels)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    logger.info("Computing pathology probabilities...")
    probs, labels = [], val_ds.tensors[-1].any(dim=-1).long()
    with torch.no_grad():
        for emb, _ in tqdm(val_dl):
            emb = torch.nn.functional.normalize(emb, dim=-1)
            logits = model(emb.to(device))
            prob = nn.functional.softmax(logits, dim=-1)[0, -1].cpu().item()
            probs.append(prob)

    probs_tensor = torch.tensor(probs)
    preds_05 = (probs_tensor > 0.5).int().numpy()
    y_true = labels.numpy()

    # --- Метрики с CI ---
    auc_mean, auc_ci = bootstrap_ci(
        lambda yt, yp: roc_auc_score(yt, yp), y_true, probs_tensor.numpy()
    )
    sens_mean, sens_ci = bootstrap_ci(sensitivity, y_true, preds_05)

    spec_mean, spec_ci = bootstrap_ci(specificity, y_true, preds_05)

    auprc_mean, auprc_ci = bootstrap_ci(
        lambda yt, yp: average_precision_score(yt, yp), y_true, probs_tensor.numpy())
    
    f1_mean, f1_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp), y_true, preds_05)

    prec_mean, prec_ci = bootstrap_ci(
        lambda yt, yp: precision_score(yt, yp), y_true, preds_05
    )
    
    # --- Собираем в таблицу ---
    metrics = {
        "AUROC":      (auc_mean, *auc_ci),
        "AUPRC":      (auprc_mean, *auprc_ci),
        "Precision":  (prec_mean, *prec_ci),
        "F1":         (f1_mean, *f1_ci),
        "Sensitivity":(sens_mean, *sens_ci),
        "Specificity":(spec_mean, *spec_ci),
    }

    metrics_df = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["Mean", "Lower", "Upper"]
    ).reset_index().rename(columns={"index": "Metric"})
    # metrics_df = pd.DataFrame([metrics_data])
    logger.info("\n" + metrics_df.to_string(index=False))
    metrics_df.to_excel('mosmed.xlsx', index=False)


if __name__ == "__main__":
    main()
