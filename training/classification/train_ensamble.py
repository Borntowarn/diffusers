import torch
import torch.nn as nn
import torch.optim as optim
import os
import hydra
import numpy as np
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from loguru import logger
from omegaconf import DictConfig
from src.mlp import MLP
from src.vae import BetaVAE
from src.ensamble import EnsembleModel
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from diffusers.optimization import get_scheduler


def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn + 1e-8)

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)

def _load_concat_dataset(
        embed_paths,
        label_paths,
        with_ids=False
    ):
    """
    Load multiple .pt files and concatenate along dim 0.
    Optionally keep track of dataset_id for each sample (for training).

    Args:
        embed_paths (list[str])
        label_paths (list[str])
        with_ids (bool): whether to return dataset_ids

    Returns:
        If with_ids=True:
            TensorDataset(X, y), dataset_ids
        Else:
            TensorDataset(X, y)
    """
    X_list, y_list, ids_list = [], [], []
    dataset_id = 0

    for i, (e_path, l_path) in enumerate(zip(embed_paths, label_paths)):
        if i % 2 == 0 and i > 0:
            dataset_id += 1
        X = torch.load(e_path)
        y = torch.load(l_path)

        X_list.append(X)
        y_list.append(y)

        if with_ids:
            ids_list.append(torch.full((y.size(0),), dataset_id, dtype=torch.long))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    if with_ids:
        dataset_ids = torch.cat(ids_list, dim=0)
        _, count = dataset_ids.unique(return_counts=True)
        print(count)
        return TensorDataset(X, y), dataset_ids
    else:
        return TensorDataset(X, y)

def get_train_sampler(labels, dataset_ids=None, dataset_boost=None):
    """
    Build WeightedRandomSampler with class balancing and optional dataset boosting.
    """
    class_vals, class_counts = np.unique(labels, return_counts=True)
    class_weights = {cls: 1.0 / count for cls, count in zip(class_vals, class_counts)}

    sample_weights = np.zeros(len(labels), dtype=np.float32)
    for i, label in enumerate(tqdm(labels, desc="Building sample weights")):
        w = class_weights[label]
        if dataset_ids is not None and dataset_boost is not None:
            w *= dataset_boost.get(int(dataset_ids[i]), 1.0)
        sample_weights[i] = w

    N = int(max(class_counts) * len(class_counts))
    return WeightedRandomSampler(sample_weights, num_samples=N, replacement=True)


# -------------------------
# Main training function
# -------------------------
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device

    # TensorBoard setup
    if accelerator.is_local_main_process:
        os.makedirs(cfg.log_dir, exist_ok=True)
        writer = SummaryWriter(cfg.log_dir)
        logger.info(f"🧭 TensorBoard logging at {cfg.log_dir}")
    else:
        writer = None

    # --- Load models ---
    mlp = MLP(cfg.model.mlp.input_size, cfg.model.mlp.num_classes,
              cfg.model.mlp.activation, cfg.model.mlp.hidden_sizes, cfg.model.mlp.dropout)
    mlp.load_state_dict(torch.load(cfg.model.mlp.mlp_ckpt, map_location=device))
    mlp.requires_grad_(False)

    vae = BetaVAE(latent_dim=cfg.model.vae.latent_dim)
    vae.load_state_dict(torch.load(cfg.model.vae.vae_ckpt, map_location=device))
    vae.requires_grad_(False)

    ensemble_model = EnsembleModel(mlp, vae)
    ensemble_model.mlp.eval()
    ensemble_model.vae.eval()
    ensemble_model.classifier.train()

    # --- Load datasets ---
    train_dataset, train_ids = _load_concat_dataset(cfg.data.train.embeds, cfg.data.train.labels, with_ids=True)
    val_dataset = _load_concat_dataset(cfg.data.val.embeds, cfg.data.val.labels)

    labels = train_dataset.tensors[1].any(dim=-1).long().numpy()
    dataset_ids = train_ids.numpy() if train_ids is not None else None
    sampler = get_train_sampler(labels, dataset_ids, cfg.data.train.get("dataset_boost", None))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(ensemble_model.classifier.parameters(), lr=cfg.lr)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.num_epochs * np.ceil(len(train_loader))
    )
    criterion = nn.BCEWithLogitsLoss()

    ensemble_model, optimizer, train_loader, val_loader = accelerator.prepare(
        ensemble_model, optimizer, train_loader, val_loader
    )

    logger.info(f"🚀 Starting ensemble training for {cfg.num_epochs} epochs...")
    global_step = 0
    for epoch in range(cfg.num_epochs):
        # -------------------------
        # Training
        # -------------------------
        ensemble_model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            y = y.any(dim=-1).float()
            preds = ensemble_model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item() * x.size(0)
            global_step += 1

            if writer and global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        ensemble_model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                y = y.any(dim=-1).float()
                preds = ensemble_model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)

        # Metrics
        try:
            val_auroc = roc_auc_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, binary_preds)
            val_precision = precision_score(all_labels, binary_preds)
            val_recall = recall_score(all_labels, binary_preds)
            val_sens = sensitivity(all_labels, binary_preds)
            val_spec = specificity(all_labels, binary_preds)
        except Exception as e:
            accelerator.print(f"⚠️ Metric computation failed: {e}")
            val_auroc = val_f1 = val_precision = val_recall = val_sens = val_spec = 0.0

        # TensorBoard logging
        if writer:
            writer.add_scalars("loss_summary", {"train": avg_train_loss, "val": val_loss}, epoch + 1)
            writer.add_scalar("metrics/AUROC", val_auroc, epoch + 1)
            writer.add_scalar("metrics/F1", val_f1, epoch + 1)
            writer.add_scalar("metrics/Precision", val_precision, epoch + 1)
            writer.add_scalar("metrics/Recall", val_recall, epoch + 1)
            writer.add_scalar("metrics/Sensitivity", val_sens, epoch + 1)
            writer.add_scalar("metrics/Specificity", val_spec, epoch + 1)

        accelerator.print(
            f"Epoch [{epoch+1}/{cfg.num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | "
            f"Sensitivity: {val_sens:.4f} | Specificity: {val_spec:.4f}"
        )

    # -------------------------
    # Save model
    # -------------------------
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(ensemble_model)
    torch.save(unwrapped.classifier.state_dict(), cfg.save_path)
    accelerator.print(f"💾 Saved ensemble classifier to {cfg.save_path}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()