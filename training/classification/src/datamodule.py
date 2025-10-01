# 

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pytorch_lightning as pl
from tqdm import tqdm


def get_train_sampler(labels, dataset_ids=None, dataset_boost=None) -> WeightedRandomSampler:
    """
    Build WeightedRandomSampler with both class balancing and dataset boosting.

    Args:
        labels (np.ndarray): class labels per sample
        dataset_ids (np.ndarray): dataset index per sample
        dataset_boost (dict): {dataset_id: multiplier}
    """
    # class weights (inverse frequency)
    class_count = np.unique(labels, return_counts=True)
    class_dict = dict(zip(*class_count))
    class_weights = {k: 1 / v for k, v in class_dict.items()}

    sample_weights = np.zeros(len(labels), dtype=np.float32)

    for i, label in enumerate(tqdm(labels, desc="Building sample weights")):
        w = class_weights[label]
        if dataset_ids is not None and dataset_boost is not None:
            w *= dataset_boost.get(int(dataset_ids[i]), 1.0)
        sample_weights[i] = w

    N = int(max(class_dict.values()) * len(class_dict))  # target sample size
    return WeightedRandomSampler(sample_weights, num_samples=N, replacement=True)


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


class HeadDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = 20

    def setup(self, stage: str) -> None:
        # Training with dataset IDs
        self.train_ds, self.dataset_ids = _load_concat_dataset(
            self.config.dataset.train.embeds,
            self.config.dataset.train.labels,
            with_ids=True,
        )

        # Validation (no dataset IDs)
        self.val_ds = _load_concat_dataset(
            self.config.dataset.val.embeds,
            self.config.dataset.val.labels,
            with_ids=False,
        )

        # Test (no dataset IDs)
        self.test_ds = _load_concat_dataset(
            self.config.dataset.test.embeds,
            self.config.dataset.test.labels,
            with_ids=False,
        )

    def train_dataloader(self):
        shuffle = True
        sampler = None
        if self.config.oversample:
            labels = self.train_ds.tensors[-1].any(dim=-1).long().numpy()
            sampler = get_train_sampler(
                labels,
                dataset_ids=self.dataset_ids,
                dataset_boost=self.config.get("dataset_boost", None),
            )
            shuffle = False

        return DataLoader(
            self.train_ds,
            batch_size=self.config.model.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=4,
        )
