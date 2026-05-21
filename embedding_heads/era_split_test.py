#!/usr/bin/env python3
"""Test whether era-specific CosFace models outperform the combined 36-class model.

Trains two separate models:
  Pre-Severan:  13 authorities (Augustus through Commodus)
  Post-Severan: 23 authorities (Septimius Severus through Licinius I)

Compares KNN-10 in projected space against the combined model on the same test coins.

    python -m embedding_heads.era_split_test
"""

import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from embedding_heads.config import CHECKPOINT_DIR, OBV_FEATURES_PATH, PROD_META_PATH
from embedding_heads.dataset import (
    EmbeddingDataset, EXPANDED_SUBSET_PATH, EXPANDED_SPLITS_PATH,
    extract_subset_embeddings,
)
from embedding_heads.heads import CosFaceHead

PRE_SEVERAN = [
    "Augustus", "Tiberius", "Claudius", "Nero", "Otho",
    "Vespasian", "Domitian", "Nerva", "Trajan", "Hadrian",
    "Antoninus Pius", "Marcus Aurelius", "Commodus",
]

POST_SEVERAN = [
    "Septimius Severus", "Julia Domna", "Caracalla", "Geta",
    "Macrinus", "Elagabalus", "Severus Alexander",
    "Maximinus I", "Gordian III", "Philip I", "Trajan Decius",
    "Valerian I", "Gallienus", "Macrianus", "Postumus", "Probus",
    "Diocletian", "Maximian", "Constantius I",
    "Constantine I", "Constantine II", "Constantius II", "Licinius I",
]

SEVERAN_CRISIS = [
    "Septimius Severus", "Julia Domna", "Caracalla", "Geta",
    "Macrinus", "Elagabalus", "Severus Alexander",
    "Maximinus I", "Gordian III", "Philip I", "Trajan Decius",
    "Valerian I", "Gallienus", "Macrianus", "Postumus",
]

ILLYRIAN_TETRARCHY = [
    "Probus",
    "Diocletian", "Maximian", "Constantius I",
    "Constantine I", "Constantine II", "Constantius II", "Licinius I",
]

# Use the sweep winner config
LR = 5e-3
S = 30
M = 0.50
DIM = 128
EPOCHS = 200
PATIENCE = 7
SEED = 42


def train_era_model(era_name, authorities, embed_dict, full_subset, full_splits):
    """Train a CosFace model on a subset of authorities."""

    # Filter to this era's authorities
    era_subset = full_subset[full_subset["authority_normalized"].isin(authorities)].copy()

    # Re-index labels 0..N-1 for this era
    era_auths = sorted(era_subset["authority_normalized"].unique())
    auth_to_int = {a: i for i, a in enumerate(era_auths)}
    era_subset["era_label"] = era_subset["authority_normalized"].map(auth_to_int)
    coin_to_label = dict(zip(era_subset["coin_id"], era_subset["era_label"]))
    num_classes = len(era_auths)
    class_map = {i: a for a, i in auth_to_int.items()}

    # Use the same train/val/test split coin_ids, filtered to this era
    era_coin_ids = set(era_subset["coin_id"])
    train_ids = [c for c in full_splits["train"] if c in era_coin_ids]
    val_ids = [c for c in full_splits["val"] if c in era_coin_ids]
    test_ids = [c for c in full_splits["test"] if c in era_coin_ids]

    train_ds = EmbeddingDataset(train_ids, embed_dict, coin_to_label)
    val_ds = EmbeddingDataset(val_ids, embed_dict, coin_to_label)
    test_ds = EmbeddingDataset(test_ids, embed_dict, coin_to_label)

    print(f"\n{'='*60}")
    print(f"ERA: {era_name} ({num_classes} classes)")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    for i, a in sorted(class_map.items()):
        n = (era_subset["era_label"] == i).sum()
        print(f"    [{i:>2}] {a:<25} {n:>5}")

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # Train
    model = CosFaceHead(num_classes, s=S, m=M, embed_dim_out=DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for embs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(embs, labels), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for embs, labels in val_loader:
                loss = criterion(model(embs, labels), labels)
                val_loss += loss.item() * embs.size(0)
                val_n += embs.size(0)
        val_loss /= val_n
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Trained: {best_epoch} epochs, val_loss={best_val_loss:.4f}")

    # Evaluate: KNN in projected space
    def collect(ids):
        raw, proj, labels = [], [], []
        with torch.no_grad():
            for cid in ids:
                if cid in embed_dict and cid in coin_to_label:
                    r = embed_dict[cid]
                    t = torch.from_numpy(r).float().unsqueeze(0)
                    p = model.embed(t).squeeze(0).numpy()
                    raw.append(r)
                    proj.append(p)
                    labels.append(coin_to_label[cid])
        return np.array(raw), np.array(proj), np.array(labels)

    train_raw, train_proj, train_labels = collect(train_ids)
    test_raw, test_proj, test_labels = collect(test_ids)

    results = {"era": era_name, "num_classes": num_classes,
               "n_train": len(train_labels), "n_test": len(test_labels)}

    for k in [1, 5, 10]:
        knn_proj = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_proj.fit(train_proj, train_labels)
        results[f"knn{k}_proj"] = knn_proj.score(test_proj, test_labels)

        knn_raw = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_raw.fit(train_raw, train_labels)
        results[f"knn{k}_raw"] = knn_raw.score(test_raw, test_labels)

    # Prototype accuracy
    prototypes = F.normalize(model.margin.weight.data, dim=1).numpy()
    cos_sim = test_proj @ prototypes.T
    results["proto_top1"] = float((cos_sim.argmax(axis=1) == test_labels).mean())

    # Head top-5
    with torch.no_grad():
        logits = model(torch.from_numpy(test_raw).float())
        probs = F.softmax(logits, dim=1).numpy()
    results["head_top1"] = float((probs.argmax(axis=1) == test_labels).mean())
    results["head_top5"] = float(np.mean([
        test_labels[i] in np.argsort(-probs[i])[:5] for i in range(len(test_labels))
    ]))

    return results


def get_combined_results(embed_dict, full_subset, full_splits, era_authorities, era_name):
    """Get KNN results from the combined 36-class model on just this era's test coins."""
    ck = torch.load(CHECKPOINT_DIR / "cosface_v2_best.pt", map_location="cpu", weights_only=False)
    model = CosFaceHead(ck["num_classes"], s=30, m=0.50, embed_dim_out=128)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    combined_class_map = {int(k): v for k, v in ck["class_map"].items()}
    # Reverse: authority -> combined label
    auth_to_combined = {v: k for k, v in combined_class_map.items()}

    era_coin_ids = set(full_subset[full_subset["authority_normalized"].isin(era_authorities)]["coin_id"])
    coin_to_combined = dict(zip(full_subset["coin_id"], full_subset["authority_label_int"]))

    train_ids = [c for c in full_splits["train"] if c in era_coin_ids]
    test_ids = [c for c in full_splits["test"] if c in era_coin_ids]

    # Use ALL training data (not just era) for the combined model's KNN
    all_coin_to_label = dict(zip(full_subset["coin_id"], full_subset["authority_label_int"]))
    all_train_ids = [c for c in full_splits["train"] if c in embed_dict and c in all_coin_to_label]

    def collect(ids, label_map):
        raw, proj, labels = [], [], []
        with torch.no_grad():
            for cid in ids:
                if cid in embed_dict and cid in label_map:
                    r = embed_dict[cid]
                    t = torch.from_numpy(r).float().unsqueeze(0)
                    p = model.embed(t).squeeze(0).numpy()
                    raw.append(r)
                    proj.append(p)
                    labels.append(label_map[cid])
        return np.array(raw), np.array(proj), np.array(labels)

    # Train on all 36 classes, test on this era only
    train_raw, train_proj, train_labels = collect(all_train_ids, all_coin_to_label)
    test_raw, test_proj, test_labels = collect(test_ids, coin_to_combined)

    results = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_proj, train_labels)
        results[f"knn{k}_combined"] = knn.score(test_proj, test_labels)

    return results


def main():
    # Load shared data
    embed_dict, _ = extract_subset_embeddings(version="v2")
    full_subset = pd.read_csv(EXPANDED_SUBSET_PATH)
    with open(EXPANDED_SPLITS_PATH) as f:
        full_splits = json.load(f)

    # Train era-specific models
    pre_results = train_era_model("Pre-Severan", PRE_SEVERAN, embed_dict, full_subset, full_splits)
    sev_results = train_era_model("Severan+Crisis", SEVERAN_CRISIS, embed_dict, full_subset, full_splits)
    illy_results = train_era_model("Illyrian+Tetrarchy", ILLYRIAN_TETRARCHY, embed_dict, full_subset, full_splits)

    # Get combined model results on same test coins
    pre_combined = get_combined_results(embed_dict, full_subset, full_splits, PRE_SEVERAN, "Pre-Severan")
    sev_combined = get_combined_results(embed_dict, full_subset, full_splits, SEVERAN_CRISIS, "Severan+Crisis")
    illy_combined = get_combined_results(embed_dict, full_subset, full_splits, ILLYRIAN_TETRARCHY, "Illyrian+Tetrarchy")

    # Report
    print(f"\n{'='*80}")
    print("THREE-ERA SPLIT COMPARISON")
    print(f"{'='*80}")

    for era_r, comb_r, era_name in [
        (pre_results, pre_combined, "Pre-Severan"),
        (sev_results, sev_combined, "Severan+Crisis"),
        (illy_results, illy_combined, "Illyrian+Tetrarchy"),
    ]:
        print(f"\n{era_name} ({era_r['num_classes']} classes, {era_r['n_test']} test coins)")
        print(f"  {'Method':<30} {'KNN-1':>7} {'KNN-5':>7} {'KNN-10':>7}")
        print(f"  {'-'*53}")
        print(f"  {'Era model (projected)':<30} {era_r['knn1_proj']:>7.1%} {era_r['knn5_proj']:>7.1%} {era_r['knn10_proj']:>7.1%}")
        print(f"  {'Combined 36-class (projected)':<30} {comb_r['knn1_combined']:>7.1%} {comb_r['knn5_combined']:>7.1%} {comb_r['knn10_combined']:>7.1%}")
        print(f"  {'Era model (raw 768-d)':<30} {era_r['knn1_raw']:>7.1%} {era_r['knn5_raw']:>7.1%} {era_r['knn10_raw']:>7.1%}")
        delta = era_r["knn10_proj"] - comb_r["knn10_combined"]
        print(f"  Era vs Combined KNN-10 delta: {delta:+.1%}")
        print(f"  Era head top-1: {era_r['head_top1']:.1%}  top-5: {era_r['head_top5']:.1%}")


if __name__ == "__main__":
    main()
