import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from loguru import logger
from typing import Callable
from collections import defaultdict
import os

from src.ensamble import EnsembleModel  
from src.vae import BetaVAE             
from src.mlp import MLP                 


# -------------------------------
# 📦 Dataset loader
# -------------------------------
def _load_concat_dataset(embed_paths, label_paths):
    X_list = [torch.load(p) for p in embed_paths]
    y_list = [torch.load(p) for p in label_paths]
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return TensorDataset(X, y)


# -------------------------------
# 🧮 Utility functions
# -------------------------------
def bootstrap_ci(metric_fn: Callable, y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """Bootstrap confidence interval for metrics"""
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn + 1e-8)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-8)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load models from config ---
    logger.info(f"Loading VAE from {cfg.model.vae.vae_ckpt}")
    vae = BetaVAE(
        input_dim=cfg.model.mlp.input_size,
        latent_dim=cfg.model.vae.latent_dim,
        beta=cfg.model.vae.beta,
    )
    vae.load_state_dict(torch.load(cfg.model.vae.vae_ckpt, map_location=device))
    vae.to(device).eval()

    logger.info(f"Loading MLP from {cfg.model.mlp.mlp_ckpt}")
    mlp = MLP(
        input_size=cfg.model.mlp.input_size,
        num_classes=cfg.model.mlp.num_classes,
        hidden_sizes=cfg.model.mlp.hidden_sizes,
        activation=cfg.model.mlp.activation,
        dropout=cfg.model.mlp.dropout,
    )
    mlp.load_state_dict(torch.load(cfg.model.mlp.mlp_ckpt, map_location=device))
    mlp.to(device).eval()

    # --- Initialize ensemble model ---
    ensemble = EnsembleModel(mlp=mlp, vae=vae)
    ensemble.classifier.load_state_dict(state_dict=torch.load(cfg.save_path))
    ensemble.to(device).eval()

    # --- Load test dataset ---
    embed_paths = cfg.data.val.embeds
    label_paths = cfg.data.val.labels
    ds = _load_concat_dataset(embed_paths, label_paths)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    logger.info("Running inference...")
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dl):
            y = y.any(dim=-1).long()
            x = x.to(device)
            out = ensemble(x)
            probs = torch.sigmoid(out)
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels).astype(int)
    preds = (scores > 0.5).astype(int)

    # --- Compute metrics ---
    auc_mean, auc_ci = bootstrap_ci(roc_auc_score, labels, scores)
    auprc_mean, auprc_ci = bootstrap_ci(average_precision_score, labels, scores)
    f1_mean, f1_ci = bootstrap_ci(f1_score, labels, preds)
    prec_mean, prec_ci = bootstrap_ci(precision_score, labels, preds)
    sens_mean, sens_ci = bootstrap_ci(sensitivity, labels, preds)
    spec_mean, spec_ci = bootstrap_ci(specificity, labels, preds)

    metrics = {
        "AUROC": (auc_mean, *auc_ci),
        "AUPRC": (auprc_mean, *auprc_ci),
        "Precision": (prec_mean, *prec_ci),
        "F1": (f1_mean, *f1_ci),
        "Sensitivity": (sens_mean, *sens_ci),
        "Specificity": (spec_mean, *spec_ci),
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Mean", "Lower", "Upper"]).reset_index()
    metrics_df.rename(columns={"index": "Metric"}, inplace=True)
    logger.info("\n" + metrics_df.to_string(index=False))

    output_path = 'ensemble.xlsx'
    metrics_df.to_excel(output_path, index=False)
    logger.info(f"✅ Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
