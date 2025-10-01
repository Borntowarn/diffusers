import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def main(args):
    # Load dataset from HuggingFace
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        token=args.hf_token,
        cache_dir=args.cache_dir
    )['train'].to_pandas()

    # Prepare save directory
    os.makedirs(args.save_dir, exist_ok=True)

    label_cols = dataset.columns[1:]

    all_tensors = []
    all_labels = []
    all_names = []

    # Traverse files and collect
    for root, _, files in os.walk(args.root_dir):
        for fname in files:
            if fname.endswith('.pt'):
                volname = fname.replace('.nii.pt', '.nii.gz')
                row = dataset[dataset['VolumeName'] == volname]
                if row.empty:
                    continue

                tensor = torch.load(os.path.join(root, fname))
                label = torch.tensor(row[label_cols].values.squeeze(), dtype=torch.float32)

                all_tensors.append(tensor)
                all_labels.append(label)
                all_names.append(volname)

    # Stack tensors and labels
    X = torch.stack(all_tensors)
    Y = torch.stack(all_labels)
    print(f"Full dataset shape: {X.shape}, labels shape: {Y.shape}")

    # Train/Val/Test split
    train_idx, temp_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)

    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for split_name, indices in splits.items():
        split_X = X[indices]
        split_Y = Y[indices]
        torch.save(split_X, os.path.join(args.save_dir, f"{split_name}_data.pt"))
        torch.save(split_Y, os.path.join(args.save_dir, f"{split_name}_labels.pt"))
        print(f"Saved {split_name}: {split_X.shape}, {split_Y.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CT-RATE dataset and save tensors.')
    parser.add_argument('--dataset_name', type=str, default='ibrahimhamamci/CT-RATE')
    parser.add_argument('--dataset_config', type=str, default='labels')
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./data')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing .pt files')
    parser.add_argument('--save_dir', type=str, default='preprocessed', help='Directory to save stacked tensors')

    args = parser.parse_args()
    main(args)
