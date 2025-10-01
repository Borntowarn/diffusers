# viz_embeddings_fiftyone.py
import time
import torch
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
from fiftyone import Classification, Classifications
from torch.utils.data import TensorDataset
from tqdm import tqdm

# --1) Load multiple .pt files into a single TensorDataset ---
def _load_concat_dataset(embed_paths, label_paths):
    X_list = [torch.load(p) for p in embed_paths]
    y_list = [torch.load(p) for p in label_paths]
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return TensorDataset(X, y)

# Paths to your embeddings and labels
embed_paths = [
    "/app/data/preprocessed_train_20/train_data.pt",
    "/app/data/preprocessed_train_20/val_data.pt",
    "/app/data/preprocessed_mosmed/train_data.pt",
    "/app/data/preprocessed_mosmed/val_data.pt",
]
label_paths = [
    "/app/data/preprocessed_train_20/train_labels.pt",
    "/app/data/preprocessed_train_20/val_labels.pt",
    "/app/data/preprocessed_mosmed/train_labels.pt",
    "/app/data/preprocessed_mosmed/val_labels.pt",
]

val_ds = _load_concat_dataset(embed_paths, label_paths)
print(f"Loaded dataset {val_ds.tensors[0].shape}")
emb = val_ds.tensors[0].detach().cpu().numpy()
labels = val_ds.tensors[1].detach().cpu().numpy()

# Validate shapes
assert emb.shape[0] == labels.shape[0]
n_samples, emb_dim = emb.shape
n_classes = labels.shape[1]

# --2) Convert labels to list-of-class-names ---
class_names = [f"class_{i}" for i in range(n_classes)]
label_mask = labels >= 0.5 if np.issubdtype(labels.dtype, np.floating) else labels.astype(bool)
label_lists = [
    [class_names[j] for j, present in enumerate(row) if present]
    for row in label_mask
]
top_labels = [lst[0] if lst else "NONE" for lst in label_lists]

# --3) Source labels for coloring ---
source_labels = ["VAL20"] * (torch.load(label_paths[0]).shape[0] + torch.load(label_paths[1]).shape[0]) + \
                ["MOSMED"] * (torch.load(label_paths[2]).shape[0] + torch.load(label_paths[3]).shape[0])

# --4) Normal vs Pathology labels ---
normal_vs_pathology = labels.any(axis=1).astype(int)  # 0 = normal, 1 = pathology
normal_vs_pathology_str = [str(x) for x in normal_vs_pathology]  # as string for coloring in App

# --5) Create FiftyOne dataset ---
dataset_name = f"embeddings_viz_{int(time.time())}"
dataset = fo.Dataset(dataset_name)

assert len(label_lists) == len(source_labels) == len(normal_vs_pathology_str)

samples = []
for i, (labs, src, np_label) in tqdm(enumerate(zip(label_lists, source_labels, normal_vs_pathology_str))):
    sample = fo.Sample(filepath=f"sample_{i}.jpg")
    sample["labels_mult"] = Classifications(
        classifications=[Classification(label=l) for l in labs]
    )
    sample["embedding"] = emb[i].tolist()
    sample["source"] = src
    sample["normal_vs_pathology"] = np_label
    samples.append(sample)
    

dataset.add_samples(samples)
print(f"Added {len(samples)} samples to dataset '{dataset_name}'")

# --6) Compute 2D or 3D UMAP embeddings ---
results = fob.compute_visualization(
    dataset,
    embeddings=emb,
    method="umap",
    num_dims=2,  # or 3 if you want 3D visualization
    brain_key="embeddings_umap"
)

# Store embeddings for scatterplot in the App
results.index_points(points_field="umap_points", create_index=True, progress=True)

# --7) Launch FiftyOne App (remote for Docker) ---
session = fo.launch_app(
    dataset,
    remote=True,
    address="0.0.0.0",
    port=5151
)

print(f"Dataset: {dataset_name} — open FiftyOne App at http://localhost:5151")

# Keep container alive until user exits
try:
    print("FiftyOne App running. Press Ctrl+C to exit.")
    session.wait()
except KeyboardInterrupt:
    print("Exiting FiftyOne App...")