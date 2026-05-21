"""Load precomputed DINOv2 768-d obverse embeddings + labels.

Extracts subset coins from the full prod embedding matrix,
aligns with train/val/test splits.

Supports:
  v1: 10-class confusion subset (legend_cnn splits)
  v2: 36-class expanded subset (Imperial + Provincial)
  scope-based: ri_authority (81-class), ri_denomination (17-class), etc.
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
    SCOPE_DATA_DIR,
    SPLITS_PATH,
)

# v2 expanded paths
EXPANDED_SUBSET_PATH = DATA_DIR / "expanded_subset.csv"
EXPANDED_SPLITS_PATH = DATA_DIR / "splits_v2.json"
EXPANDED_CLASS_MAP_PATH = DATA_DIR / "class_map_v2.json"


def extract_subset_embeddings(version="v1"):
    """Extract 768-d embeddings for subset coins.

    Args:
        version: "v1" for 10-class confusion subset, "v2" for expanded 36-class

    Returns (embeddings_dict, label_map) where embeddings_dict maps
    coin_id -> 768-d numpy vector, and label_map maps int -> authority name.
    """
    meta = pd.read_csv(PROD_META_PATH)
    meta_idx = {cid: i for i, cid in enumerate(meta["id"])}

    if version == "v2":
        subset = pd.read_csv(EXPANDED_SUBSET_PATH)
        id_col = "coin_id"
        label_col = "authority_label_int"
        auth_col = "authority_normalized"
    else:
        subset = pd.read_csv(CONFUSION_SUBSET_PATH)
        id_col = "coin_id"
        label_col = "authority_label_int"
        auth_col = "authority"

    subset_ids = subset[id_col].tolist()
    found = [(cid, meta_idx[cid]) for cid in subset_ids if cid in meta_idx]
    print(f"Subset ({version}): {len(subset_ids)}, found in prod: {len(found)}")

    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")
    indices = [idx for _, idx in found]
    embeddings = obv[indices].copy()

    embed_dict = {}
    for (cid, _), emb in zip(found, embeddings):
        embed_dict[cid] = emb

    label_map = dict(zip(subset[label_col], subset[auth_col]))
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


def build_dataloaders(batch_size=1024, version="v1"):
    """Build train/val/test dataloaders from saved splits.

    Args:
        version: "v1" for 10-class, "v2" for 36-class expanded

    Returns (train_loader, val_loader, test_loader, num_classes, class_map).
    """
    embed_dict, class_map = extract_subset_embeddings(version=version)

    if version == "v2":
        splits_path = EXPANDED_SPLITS_PATH
        subset = pd.read_csv(EXPANDED_SUBSET_PATH)
        coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))
    else:
        splits_path = SPLITS_PATH
        subset = pd.read_csv(CONFUSION_SUBSET_PATH)
        coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))

    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    train_ds = EmbeddingDataset(train_ids, embed_dict, coin_to_label)
    val_ds = EmbeddingDataset(val_ids, embed_dict, coin_to_label)
    test_ds = EmbeddingDataset(test_ids, embed_dict, coin_to_label)

    print(f"Datasets ({version}): train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Classes: {len(class_map)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(class_map), class_map


def build_dataloaders_from_scope(scope, batch_size=1024):
    """Build train/val/test dataloaders from a scope dataset.

    Args:
        scope: scope name, e.g. "ri_authority", "ri_denomination"

    Returns (train_loader, val_loader, test_loader, num_classes, class_map).
    """
    subset_path = SCOPE_DATA_DIR / f"{scope}_subset.csv"
    splits_path = SCOPE_DATA_DIR / f"{scope}_splits.json"
    class_map_path = SCOPE_DATA_DIR / f"{scope}_class_map.json"

    # Load class map
    with open(class_map_path) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    # Load subset and build coin_id -> label mapping
    subset = pd.read_csv(subset_path)
    coin_to_label = dict(zip(subset["coin_id"], subset["label_int"]))

    # Load embeddings for all coins in the subset
    meta = pd.read_csv(PROD_META_PATH)
    meta_idx = {cid: i for i, cid in enumerate(meta["id"])}

    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")
    embed_dict = {}
    for cid in subset["coin_id"]:
        if cid in meta_idx:
            embed_dict[cid] = obv[meta_idx[cid]].copy()

    print(f"Scope {scope}: {len(embed_dict)} embeddings loaded, {len(class_map)} classes")

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    train_ds = EmbeddingDataset(splits["train"], embed_dict, coin_to_label)
    val_ds = EmbeddingDataset(splits["val"], embed_dict, coin_to_label)
    test_ds = EmbeddingDataset(splits["test"], embed_dict, coin_to_label)

    print(f"Datasets ({scope}): train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(class_map), class_map
