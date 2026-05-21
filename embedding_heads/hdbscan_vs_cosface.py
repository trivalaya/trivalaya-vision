#!/usr/bin/env python3
"""Compare HDBSCAN cluster-vote vs CosFace learned embeddings for coin labeling.

Three-level purity comparison:
  Part 1 — Period level  (~11 classes, ~125k coins)
  Part 2 — Authority level (36 classes, ~24k coins, reuse existing checkpoint)
  Part 3 — Cross-period bleed (low-purity HDBSCAN clusters)

CPU-safe: all features are precomputed, model is a single Linear(768,128).

    python -m embedding_heads.hdbscan_vs_cosface
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from embedding_heads.config import CHECKPOINT_DIR, OBV_FEATURES_PATH, OUTPUT_DIR, PROD_META_PATH
from embedding_heads.dataset import (
    EXPANDED_SPLITS_PATH, EXPANDED_SUBSET_PATH, EmbeddingDataset,
    extract_subset_embeddings,
)
from embedding_heads.heads import CosFaceHead

# ── Period normalization ────────────────────────────────────────────

PERIOD_NORMALIZE = {
    "Roman Imperial": "roman_imperial",
    "Greek": "greek",
    "Roman Provincial": "roman_provincial",
    "Romam Provincial": "roman_provincial",
    "Roman Republican": "roman_republican",
    "Byzantine": "byzantine",
    "Medieval": "medieval",
    "Celtic": "celtic",
    "Islamic": "islamic",
    "Central Asian": "central_asian",
    "Oriental Greek": "oriental_greek",
    "Persian": "persian",
    "India": "india",
    "Modern": "modern",
    "Tessera": "tessera",
}

EXCLUDE_PERIODS = {"unknown", "non_coin", "modern", "india", "tessera"}
MIN_PERIOD_COINS = 400

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def load_all_coins():
    """Load enriched CSV + features, normalize periods, build index."""
    print("Loading enriched metadata...")
    meta = pd.read_csv(PROD_META_PATH, low_memory=False)
    print(f"  Rows: {len(meta):,}")

    # Normalize period labels
    meta["period_clean"] = meta["parser_period"].replace(PERIOD_NORMALIZE)
    meta["period_clean"] = meta["period_clean"].str.lower().str.strip()

    # One more pass to catch remaining case variants
    extra = {
        "roman imperial": "roman_imperial",
        "roman provincial": "roman_provincial",
        "roman republican": "roman_republican",
        "central asian": "central_asian",
        "oriental greek": "oriental_greek",
    }
    meta["period_clean"] = meta["period_clean"].replace(extra)

    # Exclude
    meta = meta[~meta["period_clean"].isin(EXCLUDE_PERIODS)].copy()

    # Filter to periods with enough coins
    counts = meta["period_clean"].value_counts()
    viable_periods = counts[counts >= MIN_PERIOD_COINS].index.tolist()
    meta = meta[meta["period_clean"].isin(viable_periods)].copy()

    print(f"  After filtering: {len(meta):,} coins, {len(viable_periods)} periods")
    for p in sorted(viable_periods):
        n = (meta["period_clean"] == p).sum()
        print(f"    {p:<25} {n:>6}")

    # Load features (row index = CSV row)
    print("\nLoading obv_features_768.npy (mmap)...")
    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")
    print(f"  Shape: {obv.shape}")

    return meta, obv, viable_periods


def is_noise(cluster_id):
    """Check if a cluster_id is noise."""
    s = str(cluster_id)
    return s == "-1" or s == "nan" or s.endswith("-noise")


# ═══════════════════════════════════════════════════════════════════
# PART 1: Period-level comparison
# ═══════════════════════════════════════════════════════════════════

def part1_period(meta, obv, viable_periods):
    """Train period-level CosFace, compare against HDBSCAN cluster majority vote."""
    print("\n" + "=" * 70)
    print("PART 1: PERIOD-LEVEL CLASSIFICATION")
    print("=" * 70)

    # Assign integer labels
    period_list = sorted(viable_periods)
    period_to_int = {p: i for i, p in enumerate(period_list)}
    num_classes = len(period_list)
    meta = meta.copy()
    meta["period_label"] = meta["period_clean"].map(period_to_int)

    print(f"\n{num_classes} period classes, {len(meta):,} coins")

    # Build row-index-based embedding lookup
    # CSV row i -> obv[i]
    row_to_idx = {row: row for row in meta.index}

    # Stratified split
    rng = random.Random(SEED)
    train_rows, val_rows, test_rows = [], [], []

    for period in period_list:
        rows = meta[meta["period_clean"] == period].index.tolist()
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, int(n * VAL_FRAC))
        n_test = max(1, int(n * TEST_FRAC))
        train_rows.extend(rows[: n - n_val - n_test])
        val_rows.extend(rows[n - n_val - n_test : n - n_test])
        test_rows.extend(rows[n - n_test :])

    print(f"Split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Extract embeddings into dict keyed by row index
    embed_dict = {}
    all_rows = train_rows + val_rows + test_rows
    # Batch-load from mmap
    indices = np.array(all_rows)
    embeddings = obv[indices]
    for row, emb in zip(all_rows, embeddings):
        embed_dict[row] = emb

    row_to_label = dict(zip(meta.index, meta["period_label"]))

    # Build dataloaders
    train_ds = EmbeddingDataset(train_rows, embed_dict, row_to_label)
    val_ds = EmbeddingDataset(val_rows, embed_dict, row_to_label)
    test_ds = EmbeddingDataset(test_rows, embed_dict, row_to_label)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    # ── Train CosFace for period ──
    print(f"\nTraining CosFace period head ({num_classes} classes)...")
    model = CosFaceHead(num_classes, s=30.0, m=0.35, embed_dim_out=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 7
    epochs_no_improve = 0

    for epoch in range(1, 201):
        model.train()
        for embs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(embs, labels), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_n = 0.0, 0
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

        if epoch % 10 == 0 or epochs_no_improve >= patience:
            print(f"  Epoch {epoch:>3d}  val_loss={val_loss:.4f}  best_ep={best_epoch}")

        if epochs_no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # ── Evaluate CosFace on test set ──
    def collect_embeddings(rows, model):
        raw_list, proj_list, labels = [], [], []
        with torch.no_grad():
            for row in rows:
                if row in embed_dict and row in row_to_label:
                    r = embed_dict[row]
                    t = torch.from_numpy(r).float().unsqueeze(0)
                    p = model.embed(t).squeeze(0).numpy()
                    raw_list.append(r)
                    proj_list.append(p)
                    labels.append(row_to_label[row])
        return np.array(raw_list), np.array(proj_list), np.array(labels)

    train_raw, train_proj, train_labels = collect_embeddings(train_rows, model)
    test_raw, test_proj, test_labels = collect_embeddings(test_rows, model)

    # KNN in projected space
    cosface_knn = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_proj, train_labels)
        cosface_knn[k] = knn.score(test_proj, test_labels)

    # Head classification
    with torch.no_grad():
        logits = model(torch.from_numpy(test_raw).float())
        probs = F.softmax(logits, dim=1).numpy()
    cosface_top1 = float((probs.argmax(axis=1) == test_labels).mean())
    cosface_top3 = float(np.mean([
        test_labels[i] in np.argsort(-probs[i])[:3]
        for i in range(len(test_labels))
    ]))

    # Also KNN in raw 768-d space (baseline for DINOv2 alone)
    raw_knn = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_raw, train_labels)
        raw_knn[k] = knn.score(test_raw, test_labels)

    # ── HDBSCAN cluster majority-vote baseline ──
    # For each cluster, compute majority period
    cluster_ids = meta["cluster_id"].astype(str)
    clustered_mask = ~meta.index.to_series().apply(
        lambda row: is_noise(cluster_ids.iloc[meta.index.get_loc(row)]
                             if row in meta.index else "-1")
    )
    # Simpler: build cluster -> majority period from ALL data (train+val+test)
    cluster_majority = {}
    for cid in meta["cluster_id"].astype(str).unique():
        if is_noise(cid):
            continue
        sub = meta[meta["cluster_id"].astype(str) == cid]
        if len(sub) > 0:
            majority = sub["period_clean"].mode().iloc[0]
            cluster_majority[cid] = period_to_int.get(majority, -1)

    # Evaluate on test set
    hdbscan_correct = 0
    hdbscan_total = 0
    hdbscan_noise_skipped = 0
    test_meta = meta.loc[test_rows]

    for row in test_rows:
        true_label = row_to_label[row]
        cid = str(meta.at[row, "cluster_id"])
        if is_noise(cid):
            hdbscan_noise_skipped += 1
            continue
        pred = cluster_majority.get(cid, -1)
        if pred == true_label:
            hdbscan_correct += 1
        hdbscan_total += 1

    hdbscan_acc = hdbscan_correct / hdbscan_total if hdbscan_total > 0 else 0
    noise_pct = hdbscan_noise_skipped / len(test_rows) * 100

    # ── Report ──
    results = {
        "num_classes": num_classes,
        "periods": period_list,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "cosface": {
            "knn1": cosface_knn[1], "knn5": cosface_knn[5], "knn10": cosface_knn[10],
            "head_top1": cosface_top1, "head_top3": cosface_top3,
            "best_epoch": best_epoch,
        },
        "raw_dino": {
            "knn1": raw_knn[1], "knn5": raw_knn[5], "knn10": raw_knn[10],
        },
        "hdbscan": {
            "cluster_vote_acc": hdbscan_acc,
            "evaluated_on": hdbscan_total,
            "noise_skipped": hdbscan_noise_skipped,
            "noise_pct": noise_pct,
        },
    }

    print(f"\n{'─'*60}")
    print(f"PERIOD CLASSIFICATION ({num_classes} classes, {len(test_rows):,} test coins)")
    print(f"{'─'*60}")
    print(f"  {'Method':<35} {'Acc':>7}")
    print(f"  {'─'*44}")
    print(f"  {'HDBSCAN cluster majority vote':<35} {hdbscan_acc:>7.1%}  (on {hdbscan_total:,} clustered, {noise_pct:.1f}% noise skipped)")
    print(f"  {'Raw DINOv2 KNN-1 (768-d)':<35} {raw_knn[1]:>7.1%}")
    print(f"  {'Raw DINOv2 KNN-10 (768-d)':<35} {raw_knn[10]:>7.1%}")
    print(f"  {'CosFace KNN-1 (128-d)':<35} {cosface_knn[1]:>7.1%}")
    print(f"  {'CosFace KNN-10 (128-d)':<35} {cosface_knn[10]:>7.1%}")
    print(f"  {'CosFace head top-1':<35} {cosface_top1:>7.1%}")
    print(f"  {'CosFace head top-3':<35} {cosface_top3:>7.1%}")

    return results, model, embed_dict, row_to_label, train_rows, test_rows, meta, period_to_int


# ═══════════════════════════════════════════════════════════════════
# PART 2: Authority-level comparison (36-class, reuse checkpoint)
# ═══════════════════════════════════════════════════════════════════

def part2_authority(meta_full):
    """Compare existing 36-class CosFace checkpoint vs HDBSCAN for authority."""
    print("\n" + "=" * 70)
    print("PART 2: AUTHORITY-LEVEL IDENTIFICATION (36 classes)")
    print("=" * 70)

    # Load existing v2 checkpoint
    ck_path = CHECKPOINT_DIR / "cosface_v2_best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = CosFaceHead(ck["num_classes"], s=30, m=0.50, embed_dim_out=128)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    class_map = {int(k): v for k, v in ck["class_map"].items()}

    # Load expanded subset data
    embed_dict, _ = extract_subset_embeddings(version="v2")
    subset = pd.read_csv(EXPANDED_SUBSET_PATH)
    coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))
    coin_to_auth = dict(zip(subset["coin_id"], subset["authority_normalized"]))

    with open(EXPANDED_SPLITS_PATH) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    # ── CosFace evaluation ──
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

    cosface_knn = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_proj, train_labels)
        cosface_knn[k] = knn.score(test_proj, test_labels)

    with torch.no_grad():
        logits = model(torch.from_numpy(test_raw).float())
        probs = F.softmax(logits, dim=1).numpy()
    cosface_top1 = float((probs.argmax(axis=1) == test_labels).mean())
    cosface_top5 = float(np.mean([
        test_labels[i] in np.argsort(-probs[i])[:5]
        for i in range(len(test_labels))
    ]))

    raw_knn = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_raw, train_labels)
        raw_knn[k] = knn.score(test_raw, test_labels)

    # ── HDBSCAN cluster majority-vote for authority ──
    # Map coin_id -> cluster_id from full metadata
    coin_to_cluster = dict(zip(meta_full["coin_id"], meta_full["cluster_id"].astype(str)))

    # For each cluster, find majority authority (among the 36 known)
    # Only count coins from the expanded subset that are in the training set
    cluster_auth_votes = {}
    for cid in train_ids:
        if cid not in coin_to_label or cid not in coin_to_cluster:
            continue
        cluster = coin_to_cluster[cid]
        if is_noise(cluster):
            continue
        if cluster not in cluster_auth_votes:
            cluster_auth_votes[cluster] = Counter()
        cluster_auth_votes[cluster][coin_to_label[cid]] += 1

    cluster_majority_auth = {}
    for cluster, votes in cluster_auth_votes.items():
        cluster_majority_auth[cluster] = votes.most_common(1)[0][0]

    # Evaluate on test set
    hdbscan_correct = 0
    hdbscan_total = 0
    hdbscan_noise = 0

    for cid in test_ids:
        if cid not in coin_to_label or cid not in coin_to_cluster:
            continue
        true_label = coin_to_label[cid]
        cluster = coin_to_cluster[cid]
        if is_noise(cluster):
            hdbscan_noise += 1
            continue
        pred = cluster_majority_auth.get(cluster, -1)
        if pred == true_label:
            hdbscan_correct += 1
        hdbscan_total += 1

    hdbscan_acc = hdbscan_correct / hdbscan_total if hdbscan_total > 0 else 0
    n_test_total = len([c for c in test_ids if c in coin_to_label])
    noise_pct = hdbscan_noise / n_test_total * 100 if n_test_total > 0 else 0

    results = {
        "num_classes": 36,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "cosface": {
            "knn1": cosface_knn[1], "knn5": cosface_knn[5], "knn10": cosface_knn[10],
            "head_top1": cosface_top1, "head_top5": cosface_top5,
        },
        "raw_dino": {
            "knn1": raw_knn[1], "knn5": raw_knn[5], "knn10": raw_knn[10],
        },
        "hdbscan": {
            "cluster_vote_acc": hdbscan_acc,
            "evaluated_on": hdbscan_total,
            "noise_skipped": hdbscan_noise,
            "noise_pct": noise_pct,
        },
    }

    print(f"\n{'─'*60}")
    print(f"AUTHORITY IDENTIFICATION (36 classes, {len(test_labels):,} test coins)")
    print(f"{'─'*60}")
    print(f"  {'Method':<35} {'Acc':>7}")
    print(f"  {'─'*44}")
    print(f"  {'HDBSCAN cluster majority vote':<35} {hdbscan_acc:>7.1%}  (on {hdbscan_total:,} clustered, {noise_pct:.1f}% noise skipped)")
    print(f"  {'Raw DINOv2 KNN-1 (768-d)':<35} {raw_knn[1]:>7.1%}")
    print(f"  {'Raw DINOv2 KNN-10 (768-d)':<35} {raw_knn[10]:>7.1%}")
    print(f"  {'CosFace KNN-1 (128-d)':<35} {cosface_knn[1]:>7.1%}")
    print(f"  {'CosFace KNN-10 (128-d)':<35} {cosface_knn[10]:>7.1%}")
    print(f"  {'CosFace head top-1':<35} {cosface_top1:>7.1%}")
    print(f"  {'CosFace head top-5':<35} {cosface_top5:>7.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════
# PART 3: Cross-period bleed analysis
# ═══════════════════════════════════════════════════════════════════

def part3_bleed(meta, period_model, embed_dict, row_to_label, test_rows, period_to_int):
    """On low-purity HDBSCAN clusters, can CosFace correctly separate mixed-period coins?"""
    print("\n" + "=" * 70)
    print("PART 3: CROSS-PERIOD BLEED ANALYSIS")
    print("=" * 70)

    int_to_period = {v: k for k, v in period_to_int.items()}

    # Compute per-cluster period purity
    cluster_ids = meta["cluster_id"].astype(str)
    unique_clusters = [c for c in cluster_ids.unique() if not is_noise(c)]

    cluster_stats = []
    for cid in unique_clusters:
        mask = cluster_ids == cid
        sub = meta[mask]
        if len(sub) < 10:
            continue
        period_counts = sub["period_clean"].value_counts()
        majority_period = period_counts.index[0]
        purity = period_counts.iloc[0] / len(sub)
        cluster_stats.append({
            "cluster_id": cid,
            "size": len(sub),
            "majority_period": majority_period,
            "purity": purity,
            "n_periods": len(period_counts),
        })

    cluster_df = pd.DataFrame(cluster_stats)
    print(f"\nClusters analyzed: {len(cluster_df):,}")
    print(f"  Purity >= 90%: {(cluster_df['purity'] >= 0.90).sum()}")
    print(f"  Purity 75-90%: {((cluster_df['purity'] >= 0.75) & (cluster_df['purity'] < 0.90)).sum()}")
    print(f"  Purity 50-75%: {((cluster_df['purity'] >= 0.50) & (cluster_df['purity'] < 0.75)).sum()}")
    print(f"  Purity < 50%:  {(cluster_df['purity'] < 0.50).sum()}")

    # Focus on low-purity clusters (< 75% period purity, size >= 50)
    low_purity = cluster_df[(cluster_df["purity"] < 0.75) & (cluster_df["size"] >= 50)]
    print(f"\nLow-purity clusters (<75%, size>=50): {len(low_purity)}")
    lp_coin_count = low_purity["size"].sum()
    print(f"  Total coins in these clusters: {lp_coin_count:,}")

    if len(low_purity) == 0:
        print("  No low-purity clusters found, skipping bleed analysis.")
        return {"low_purity_clusters": 0}

    # For coins in low-purity clusters that are also in test set, compare predictions
    test_set = set(test_rows)
    lp_clusters = set(low_purity["cluster_id"])

    cosface_correct = 0
    cosface_total = 0
    hdbscan_correct = 0
    hdbscan_total = 0

    # Build cluster -> majority period label (int)
    cluster_majority_period = {}
    for _, row in low_purity.iterrows():
        mp = row["majority_period"]
        cluster_majority_period[row["cluster_id"]] = period_to_int.get(mp, -1)

    for idx in meta.index:
        cid = str(meta.at[idx, "cluster_id"])
        if cid not in lp_clusters:
            continue
        if idx not in test_set:
            continue
        if idx not in embed_dict or idx not in row_to_label:
            continue

        true_label = row_to_label[idx]

        # HDBSCAN prediction: cluster majority
        hdbscan_pred = cluster_majority_period.get(cid, -1)
        if hdbscan_pred == true_label:
            hdbscan_correct += 1
        hdbscan_total += 1

        # CosFace prediction: head classification
        r = embed_dict[idx]
        with torch.no_grad():
            t = torch.from_numpy(r).float().unsqueeze(0)
            logits = period_model(t)
            pred = logits.argmax(dim=1).item()
        if pred == true_label:
            cosface_correct += 1
        cosface_total += 1

    cosface_acc = cosface_correct / cosface_total if cosface_total > 0 else 0
    hdbscan_acc = hdbscan_correct / hdbscan_total if hdbscan_total > 0 else 0

    # Also compute weighted average HDBSCAN purity across these clusters
    weighted_purity = (low_purity["size"] * low_purity["purity"]).sum() / low_purity["size"].sum()

    results = {
        "low_purity_clusters": len(low_purity),
        "total_coins_in_lp": int(lp_coin_count),
        "test_coins_evaluated": cosface_total,
        "hdbscan_weighted_purity": float(weighted_purity),
        "hdbscan_majority_acc": hdbscan_acc,
        "cosface_head_acc": cosface_acc,
    }

    # Per-cluster breakdown for the worst offenders
    worst = low_purity.nsmallest(10, "purity")
    print(f"\n{'─'*60}")
    print(f"CROSS-PERIOD BLEED ({len(low_purity)} low-purity clusters, {cosface_total} test coins)")
    print(f"{'─'*60}")
    print(f"  HDBSCAN weighted purity:    {weighted_purity:>7.1%}")
    print(f"  HDBSCAN majority-vote acc:  {hdbscan_acc:>7.1%}")
    print(f"  CosFace head acc:           {cosface_acc:>7.1%}")
    print(f"  Delta (CosFace - HDBSCAN):  {cosface_acc - hdbscan_acc:>+7.1%}")

    print(f"\n  10 worst-purity clusters:")
    print(f"  {'Cluster':<15} {'Size':>6} {'Purity':>7} {'Majority':<20} {'#Periods':>8}")
    for _, row in worst.iterrows():
        print(f"  {row['cluster_id']:<15} {row['size']:>6} {row['purity']:>7.1%} "
              f"{row['majority_period']:<20} {row['n_periods']:>8}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    meta, obv, viable_periods = load_all_coins()

    # Part 1: Period-level
    p1, period_model, embed_dict, row_to_label, train_rows, test_rows, meta_filtered, period_to_int = \
        part1_period(meta, obv, viable_periods)

    # Part 2: Authority-level
    p2 = part2_authority(meta)

    # Part 3: Cross-period bleed
    p3 = part3_bleed(meta_filtered, period_model, embed_dict, row_to_label,
                     test_rows, period_to_int)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: HDBSCAN vs CosFace")
    print("=" * 70)

    print(f"\n  {'Level':<20} {'HDBSCAN':>10} {'Raw KNN-10':>12} {'CosFace KNN-10':>16} {'CosFace Head':>14}")
    print(f"  {'─'*74}")
    print(f"  {'Period (~11 cls)':<20} "
          f"{p1['hdbscan']['cluster_vote_acc']:>10.1%} "
          f"{p1['raw_dino']['knn10']:>12.1%} "
          f"{p1['cosface']['knn10']:>16.1%} "
          f"{p1['cosface']['head_top1']:>14.1%}")
    print(f"  {'Authority (36 cls)':<20} "
          f"{p2['hdbscan']['cluster_vote_acc']:>10.1%} "
          f"{p2['raw_dino']['knn10']:>12.1%} "
          f"{p2['cosface']['knn10']:>16.1%} "
          f"{p2['cosface']['head_top1']:>14.1%}")
    if p3.get("test_coins_evaluated", 0) > 0:
        print(f"  {'Bleed clusters':<20} "
              f"{p3['hdbscan_majority_acc']:>10.1%} "
              f"{'—':>12} "
              f"{'—':>16} "
              f"{p3['cosface_head_acc']:>14.1%}")

    print(f"\n  Notes:")
    print(f"  - HDBSCAN cannot predict for noise coins ({p1['hdbscan']['noise_pct']:.1f}% period, {p2['hdbscan']['noise_pct']:.1f}% authority)")
    print(f"  - CosFace evaluates ALL test coins (no noise gap)")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {"period": p1, "authority": p2, "bleed": p3}
    out_path = OUTPUT_DIR / "hdbscan_vs_cosface.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
