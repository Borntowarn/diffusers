import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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

from src.vae import BetaVAE  # import your trained VAE class


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
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn + 1e-8)


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)


# -------------------------------
# 🚀 Inference
# -------------------------------
def run_inference(model_ckpt, embed_paths, label_paths, threshold=None, batch_size=64, output_xlsx="vae_metrics.xlsx"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading VAE checkpoint from {model_ckpt}")
    # vae = BetaVAE(
    #     input_dim=512,
    #     latent_dim=32,
    #     hidden_dims=[256, 128],
    #     beta=4.0
    # )
    vae = BetaVAE(
        input_dim=512,
        latent_dim=16,
        hidden_dims=[],
        beta=10.0,
        use_bn=True,
        activation='leaky_relu'
    )
    # vae = BetaVAE(
    #     input_dim=512,
    #     latent_dim=64,
    #     hidden_dims=[512, 256, 128],
    #     beta=4.0,
    #     use_bn=True,
    #     activation='gelu'
    # )
    vae.load_state_dict(torch.load(model_ckpt, map_location=device))
    vae.to(device).eval()

    logger.info("Loading test dataset...")
    ds = _load_concat_dataset(embed_paths, label_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logger.info("Computing reconstruction errors...")
    errors, labels = [], []

    # with torch.no_grad():
    #     for x, y in tqdm(dl):
    #         x = x.to(device)
    #         recon, mu, logvar = vae(x)
    #         recon_error = torch.mean((x - recon) ** 2, dim=1)
    #         errors.extend(recon_error.cpu().numpy())
    #         labels.extend(y.cpu().numpy())
    with torch.no_grad():
        for x, y in tqdm(dl):
            x = torch.nn.functional.normalize(x, dim=-1)
            x = x.to(device)
            mu, _ = vae.encode(x)
            recon = vae.decode(mu)        # deterministic reconstruction
            recon_error = torch.mean((x - recon) ** 2, dim=1)
            errors.extend(recon_error.cpu().numpy())
            labels.extend(y.any(dim=-1).cpu().numpy())

    errors = np.array(errors)
    labels = np.array(labels).astype(int)

    scores = errors

    # Compute threshold
    if threshold is None:
        threshold = np.percentile(scores[labels == 0], 95)  # 95th percentile of normals
        logger.info(f"Auto threshold set to {threshold:.4f}")

    preds = (scores > threshold).astype(int)  # 1 = pathology

    # --- Metrics with bootstrap ---
    auc_mean, auc_ci = bootstrap_ci(lambda yt, yp: roc_auc_score(yt, yp), labels, scores)
    auprc_mean, auprc_ci = bootstrap_ci(lambda yt, yp: average_precision_score(yt, yp), labels, scores)
    sens_mean, sens_ci = bootstrap_ci(sensitivity, labels, preds)
    spec_mean, spec_ci = bootstrap_ci(specificity, labels, preds)
    f1_mean, f1_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp), labels, preds)
    prec_mean, prec_ci = bootstrap_ci(lambda yt, yp: precision_score(yt, yp), labels, preds)

    metrics = {
        "AUROC": (auc_mean, *auc_ci),
        "AUPRC": (auprc_mean, *auprc_ci),
        "Precision": (prec_mean, *prec_ci),
        "F1": (f1_mean, *f1_ci),
        "Sensitivity": (sens_mean, *sens_ci),
        "Specificity": (spec_mean, *spec_ci),
        "Threshold": (threshold, np.nan, np.nan),
    }

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Mean", "Lower", "Upper"]).reset_index()
    metrics_df.rename(columns={"index": "Metric"}, inplace=True)
    logger.info("\n" + metrics_df.to_string(index=False))

    metrics_df.to_excel(output_xlsx, index=False)
    logger.info(f"Metrics saved to {output_xlsx}")

    return metrics_df, scores, labels, preds


# -------------------------------
# 🧭 Example usage
# -------------------------------
if __name__ == "__main__":
    MODEL_CKPT = "./runs/checkpoints/vae_checkpoints14/vae_epoch_100.pth"
    # MODEL_CKPT = "./runs/grid_search_vae/vae_latent_dim=16_beta=10_lr=3e-05_hidden_dims=()_use_bn=True_activation=leaky_relu.pth"
    # MODEL_CKPT = "./runs/grid_search_vae/vae_latent_dim=64_beta=4_lr=0.0001_hidden_dims=(512, 256, 128)_use_bn=True_activation=gelu.pth"

    EMBED_PATHS = ["./data/validation_data.pt"]
    LABEL_PATHS = ["./data/validation_labels.pt"]
    # EMBED_PATHS = ["./data/test_data.pt"]
    # LABEL_PATHS = ["./data/test_labels.pt"]

    run_inference(
        model_ckpt=MODEL_CKPT,
        embed_paths=EMBED_PATHS,
        label_paths=LABEL_PATHS,
        threshold=0.155,
        batch_size=64,
        output_xlsx="vae_metrics.xlsx",
    )


# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     roc_auc_score,
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
# )
# from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm
# from loguru import logger
# from typing import Callable
# import os

# from src.vae import BetaVAE  # import your trained VAE class


# # -------------------------------
# # 📦 Dataset loader
# # -------------------------------
# def _load_concat_dataset(embed_paths, label_paths):
#     X_list = [torch.load(p) for p in embed_paths]
#     y_list = [torch.load(p) for p in label_paths]
#     X = torch.cat(X_list, dim=0)
#     y = torch.cat(y_list, dim=0)
#     return TensorDataset(X, y)


# # -------------------------------
# # 🧮 Utility functions
# # -------------------------------
# def bootstrap_ci(metric_fn: Callable, y_true, y_pred, n_bootstrap=1000, alpha=0.05):
#     """Bootstrap confidence interval for metrics"""
#     n = len(y_true)
#     stats = []
#     rng = np.random.default_rng(seed=42)
#     for _ in range(n_bootstrap):
#         idx = rng.integers(0, n, n)
#         stats.append(metric_fn(y_true[idx], y_pred[idx]))
#     lower = np.percentile(stats, 100 * alpha / 2)
#     upper = np.percentile(stats, 100 * (1 - alpha / 2))
#     return np.mean(stats), (lower, upper)


# def sensitivity(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     return tp / (tp + fn + 1e-8)


# def specificity(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     return tn / (tn + fp + 1e-8)


# # -------------------------------
# # 🔍 Grid search thresholds
# # -------------------------------
# def gridsearch_thresholds(scores, labels, n_thresholds=100, max_imbalance=0.05):
#     """
#     Grid-search thresholds to maximize AUROC while keeping sensitivity and specificity balanced.
#     """
#     thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
#     best_auc = -1
#     best_threshold = None
#     best_metrics = None

#     for t in thresholds:
#         preds = (scores > t).astype(int)
#         sens = sensitivity(labels, preds)
#         spec = specificity(labels, preds)
#         auc = roc_auc_score(labels, scores)

#         if abs(sens - spec) > max_imbalance:
#             continue  # skip thresholds that are too imbalanced

#         if auc > best_auc:
#             best_auc = auc
#             best_threshold = t
#             best_metrics = {
#                 "Threshold": t,
#                 "Sensitivity": sens,
#                 "Specificity": spec,
#                 "AUROC": auc,
#                 "AUPRC": average_precision_score(labels, scores),
#                 "Precision": precision_score(labels, preds),
#                 "F1": f1_score(labels, preds),
#             }

#     if best_threshold is None:
#         raise ValueError("No threshold found satisfying the balance constraint.")

#     return best_threshold, best_metrics


# # -------------------------------
# # 🚀 Inference
# # -------------------------------
# def run_inference(
#     model_ckpt,
#     embed_paths,
#     label_paths,
#     threshold=None,
#     batch_size=64,
#     output_xlsx="vae_metrics.xlsx",
#     grid_n_thresholds=200,
#     max_imbalance=0.05,
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     logger.info(f"Loading VAE checkpoint from {model_ckpt}")
#     vae = BetaVAE(
#         input_dim=512,
#         latent_dim=64,
#         beta=4.0
#     )
#     vae.load_state_dict(torch.load(model_ckpt, map_location=device))
#     vae.to(device).eval()

#     logger.info("Loading test dataset...")
#     ds = _load_concat_dataset(embed_paths, label_paths)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

#     logger.info("Computing reconstruction errors...")
#     errors, labels = [], []

#     with torch.no_grad():
#         for x, y in tqdm(dl):
#             x = x.to(device)
#             mu, _ = vae.encode(x)
#             recon = vae.decode(mu)
#             recon_error = torch.mean((x - recon) ** 2, dim=1)
#             errors.extend(recon_error.cpu().numpy())
#             labels.extend(y.any(dim=-1).cpu().numpy())

#     errors = np.array(errors)
#     labels = np.array(labels).astype(int)

#     scores = errors  # anomaly score

#     # -------------------------------
#     # Threshold selection
#     # -------------------------------
#     if threshold is None:
#         threshold, _ = gridsearch_thresholds(scores, labels, n_thresholds=grid_n_thresholds, max_imbalance=max_imbalance)
#         logger.info(f"Grid-searched threshold: {threshold:.4f}")

#     preds = (scores > threshold).astype(int)

#     # --- Metrics with bootstrap ---
#     auc_mean, auc_ci = bootstrap_ci(lambda yt, yp: roc_auc_score(yt, yp), labels, scores)
#     auprc_mean, auprc_ci = bootstrap_ci(lambda yt, yp: average_precision_score(yt, yp), labels, scores)
#     sens_mean, sens_ci = bootstrap_ci(sensitivity, labels, preds)
#     spec_mean, spec_ci = bootstrap_ci(specificity, labels, preds)
#     f1_mean, f1_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp), labels, preds)
#     prec_mean, prec_ci = bootstrap_ci(lambda yt, yp: precision_score(yt, yp), labels, preds)

#     metrics = {
#         "AUROC": (auc_mean, *auc_ci),
#         "AUPRC": (auprc_mean, *auprc_ci),
#         "Precision": (prec_mean, *prec_ci),
#         "F1": (f1_mean, *f1_ci),
#         "Sensitivity": (sens_mean, *sens_ci),
#         "Specificity": (spec_mean, *spec_ci),
#         "Threshold": (threshold, np.nan, np.nan),
#     }

#     metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Mean", "Lower", "Upper"]).reset_index()
#     metrics_df.rename(columns={"index": "Metric"}, inplace=True)
#     logger.info("\n" + metrics_df.to_string(index=False))

#     metrics_df.to_excel(output_xlsx, index=False)
#     logger.info(f"Metrics saved to {output_xlsx}")

#     return metrics_df, scores, labels, preds


# # -------------------------------
# # 🧭 Example usage
# # -------------------------------
# if __name__ == "__main__":
#     MODEL_CKPT = "./runs/checkpoints/vae_checkpoints5/vae_epoch_100.pth"
#     EMBED_PATHS = ["./data/validation_data.pt"]
#     LABEL_PATHS = ["./data/validation_labels.pt"]

#     run_inference(
#         model_ckpt=MODEL_CKPT,
#         embed_paths=EMBED_PATHS,
#         label_paths=LABEL_PATHS,
#         threshold=None,  # set to None to grid search
#         batch_size=64,
#         output_xlsx="vae_metrics.xlsx",
#         grid_n_thresholds=200,
#         max_imbalance=0.05,
#     )
