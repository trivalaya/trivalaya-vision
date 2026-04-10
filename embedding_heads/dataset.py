"""Load precomputed DINOv2 768-d obverse embeddings + authority labels.

Extracts the confusion subset coins from the full prod embedding matrix,
aligns with the existing train/val/test splits from legend_cnn.
"""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from embedding_heads.config import (
    CONFUSION_SUBSET_PATH,
    DATA_DIR,
    OBV_FEATURES_PATH,
    PROD_META_PATH,
    SPLITS_PATH,
)


def extract_subset_embeddings():
    """Extract 768-d embeddings for confusion subset coins.

    Returns (embeddings_dict, label_map) where embeddings_dict maps
    coin_id -> 768-d numpy vector, and label_map maps authority -> int.
    """
    # Load prod metadata to get row indices
    meta = pd.read_csv(PROD_META_PATH)
    subset = pd.read_csv(CONFUSION_SUBSET_PATH)

    # Build coin_id -> row index mapping
    meta_idx = {cid: i for i, cid in enumerate(meta["coin_id"])}

    # Find which subset coins exist in prod
    subset_ids = subset["coin_id"].tolist()
    found = [(cid, meta_idx[cid]) for cid in subset_ids if cid in meta_idx]
    print(f"Subset: {len(subset_ids)}, found in prod: {len(found)}")

    # Load only the rows we need
    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")
    indices = [idx for _, idx in found]
    embeddings = obv[indices].copy()  # [N, 768]

    # Build coin_id -> embedding dict
    embed_dict = {}
    for (cid, _), emb in zip(found, embeddings):
        embed_dict[cid] = emb

    # Label map from subset
    label_map = dict(
        zip(subset["authority_label_int"], subset["authority"])
    )
    label_map = {int(k): v for k, v in label_map.items()}

    return embed_dict, label_map


class EmbeddingDataset(Dataset):
    """Dataset of (embedding, label) pairs."""

    def __init__(self, coin_ids, embed_dict, coin_to_label):
        self.coin_ids = [c for c in coin_ids if c in embed_dict and c in coin_to_label]
        self.embed_dict = embed_dict
        self.coin_to_label = coin_to_label

    def __len__(self):
        return len(self.coin_ids)

    def __getitem__(self, idx):
        cid = self.coin_ids[idx]
        emb = torch.from_numpy(self.embed_dict[cid]).float()
        label = self.coin_to_label[cid]
        return emb, label


def build_dataloaders(batch_size=1024):
    """Build train/val/test dataloaders from existing splits.

    Returns (train_loader, val_loader, test_loader, num_classes, class_map).
    """
    embed_dict, class_map = extract_subset_embeddings()

    # Load splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    # Build coin_id -> label mapping
    subset = pd.read_csv(CONFUSION_SUBSET_PATH)
    coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    train_ds = EmbeddingDataset(train_ids, embed_dict, coin_to_label)
    val_ds = EmbeddingDataset(val_ids, embed_dict, coin_to_label)
    test_ds = EmbeddingDataset(test_ids, embed_dict, coin_to_label)

    print(f"Datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Classes: {len(class_map)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(class_map), class_map
