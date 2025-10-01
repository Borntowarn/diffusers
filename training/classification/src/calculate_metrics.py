import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import hydra
from torchmetrics import PrecisionRecallCurve, F1Score, ConfusionMatrix, Specificity, SensitivityAtSpecificity
from typing import List, Any
from torcheval.metrics.functional import binary_auprc, binary_auroc
from collections import defaultdict
from mlp import MLP
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from tqdm import tqdm

def _load_concat_dataset(embed_paths, label_paths):
    """
    Load multiple .pt files and concatenate along dim 0
    """
    X_list = [torch.load(p) for p in embed_paths]
    y_list = [torch.load(p) for p in label_paths]
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    return TensorDataset(X, y)

def make_line_plot(x: List[Any], y: List[Any], x_label: str, y_label: str,
                   plot_name: str, label: str, color: str) -> None:
    sns.lineplot(x=x, y=y, label=label, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_name)
    plt.legend()
    plt.tight_layout()


def plot_confusion_matrix(y_true: torch.Tensor, y_probs: torch.Tensor,
                          threshold: float, title: str, save_path: str) -> None:
    preds = (y_probs > threshold).long()
    cm = ConfusionMatrix(task="binary", num_classes=2)(preds, y_true)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


MODEL_CKPT = '/home/free4ky/projects/chest-diseases/model_binary_ctrate_mosmed2.pth'  # specify path


@hydra.main(version_base=None, config_path="../configs", config_name="config_binary_test.yaml")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info(f"Loading model from checkpoint: {MODEL_CKPT}")
    model = MLP(
        input_size=config.model.input_size,
        activation=config.model.activation,
        dropout=config.model.dropout,
        num_classes=2,
    )
    state_dict = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.success(f"Model loaded successfully")
    # 2. Dataset and DataLoader
    logger.info("Preparing dataset and dataloader...")
    val_ds = _load_concat_dataset(config.dataset.test.embeds, config.dataset.test.labels)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 3. Inference
    logger.info("Computing pathology probabilities...")
    probs, labels = [], val_ds.tensors[-1].any(dim=-1).long()
    with torch.no_grad():
        for emb, _ in tqdm(val_dl):
            emb = torch.nn.functional.normalize(emb, dim=-1)
            logits = model(emb.to(device))
            prob = nn.functional.softmax(logits, dim=-1)[0, -1].cpu().item()
            probs.append(prob)

    probs_tensor = torch.tensor(probs)

    # 4. Precision-Recall Curve & Optimal Threshold
    pr_curve = PrecisionRecallCurve(task="binary")
    precisions, recalls, thresholds = pr_curve(probs_tensor, labels)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = torch.argmax(f1_scores)
    best_threshold = thresholds[best_idx].item()
    best_f1 = f1_scores[best_idx].item()
    logger.info(f"Best F1 threshold: {best_threshold:.4f}, F1={best_f1:.4f}")

    # 5. Metrics
    auprc = binary_auprc(probs_tensor, labels).item()
    auroc = binary_auroc(probs_tensor, labels).item()
    f1_at_05 = F1Score(task="binary")( (probs_tensor > 0.5).long(), labels ).item()

    # 6. Confusion Matrices
    plot_confusion_matrix(labels, probs_tensor, 0.5,
                          title="Confusion Matrix (Threshold=0.5)",
                          save_path="confmat_thr0.5.png")

    plot_confusion_matrix(labels, probs_tensor, best_threshold,
                          title="Confusion Matrix (Best F1 Threshold)",
                          save_path="confmat_bestF1.png")

# Collect metrics into a single-row DataFrame
    metrics_data = {
        "threshold_best": best_threshold,
        "F1_05": f1_at_05,
        "F1_best": best_f1,
        "AUPRC": auprc,
        "AUROC": auroc,
    }
    metrics_df = pd.DataFrame([metrics_data])

    logger.info("\n" + metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
