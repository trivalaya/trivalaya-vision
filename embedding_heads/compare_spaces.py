#!/usr/bin/env python3
"""Apples-to-apples comparison of three embedding spaces for authority retrieval.

Uses ONE consistent protocol across all three spaces:
  - Reference pool: ri_authority training split (23,310 coins)
  - Queries: ri_authority test split (2,869 coins)
  - Labels: 83-class authority labels
  - Distance metric: cosine
  - Voting: uniform majority vote
  - k: 1, 5, 10

Three spaces:
  1. Raw DINOv2 768-d (no projection)
  2. 36-class CosFace projected 128-d (cosface_v2_best.pt)
  3. 83-class CosFace projected 128-d (ri_authority_cosface_best.pt)

Also reports classifier-head accuracy (top-1, top-5) for the two CosFace models.

Usage:
    python -m embedding_heads.compare_spaces
"""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

from embedding_heads.config import CHECKPOINT_DIR, OBV_FEATURES_PATH, OUTPUT_DIR, PROD_META_PATH
from embedding_heads.dataset import DATA_DIR
from embedding_heads.heads import CosFaceHead


def load_83class_splits():
    """Load ri_authority dataset: subset, splits, class_map."""
    subset = pd.read_csv(DATA_DIR / "ri_authority_subset.csv")
    with open(DATA_DIR / "ri_authority_splits.json") as f:
        splits = json.load(f)
    with open(DATA_DIR / "ri_authority_class_map.json") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    coin_to_label = dict(zip(subset["coin_id"], subset["label_int"]))
    return splits, coin_to_label, class_map


def load_embeddings(coin_ids, coin_to_label):
    """Load raw 768-d embeddings for a list of coin_ids."""
    meta = pd.read_csv(PROD_META_PATH)
    meta_idx = {cid: i for i, cid in enumerate(meta["coin_id"])}

    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")

    raw_embs = []
    labels = []
    valid_ids = []
    for cid in coin_ids:
        if cid in meta_idx and cid in coin_to_label:
            raw_embs.append(obv[meta_idx[cid]].copy())
            labels.append(coin_to_label[cid])
            valid_ids.append(cid)

    return np.array(raw_embs), np.array(labels), valid_ids


def project_through_model(raw_embs, model):
    """Project raw 768-d embeddings through a CosFace model's projection layer."""
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(raw_embs).float()
        projected = model.embed(t).numpy()
    return projected


def knn_evaluate(train_embs, train_labels, test_embs, test_labels, ks=(1, 5, 10)):
    """Run KNN evaluation at multiple k values. Returns dict of k -> accuracy."""
    results = {}
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_embs, train_labels)
        results[k] = float(knn.score(test_embs, test_labels))
    return results


def head_evaluate(raw_embs, labels, model, num_classes):
    """Run classifier-head evaluation (top-1, top-5)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(raw_embs).float())
        probs = F.softmax(logits, dim=1).numpy()

    top1 = float((probs.argmax(axis=1) == labels).mean())

    k5 = min(5, num_classes)
    top5 = float(np.mean([
        labels[i] in np.argsort(-probs[i])[:k5]
        for i in range(len(labels))
    ]))

    return top1, top5


def per_class_knn(train_embs, train_labels, test_embs, test_labels, class_map, k=10):
    """Per-class KNN-10 recall breakdown."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embs, train_labels)
    predictions = knn.predict(test_embs)

    per_class = {}
    for cls_idx, cls_name in class_map.items():
        mask = test_labels == cls_idx
        if mask.sum() == 0:
            continue
        cls_preds = predictions[mask]
        cls_true = test_labels[mask]
        acc = float((cls_preds == cls_true).mean())
        per_class[cls_name] = {"accuracy": acc, "count": int(mask.sum())}

    return per_class


def main():
    print("=" * 70)
    print("EMBEDDING SPACE COMPARISON: Raw 768-d vs 36-class vs 83-class")
    print("=" * 70)

    # Load data
    print("\nLoading 83-class ri_authority dataset...")
    splits, coin_to_label, class_map = load_83class_splits()

    print("Loading raw embeddings for train split...")
    train_raw, train_labels, train_ids = load_embeddings(splits["train"], coin_to_label)
    print(f"  Train: {len(train_raw)} coins")

    print("Loading raw embeddings for test split...")
    test_raw, test_labels, test_ids = load_embeddings(splits["test"], coin_to_label)
    print(f"  Test: {len(test_raw)} coins")

    num_classes = len(class_map)
    print(f"  Classes: {num_classes}")

    # Load models
    print("\nLoading 36-class CosFace model...")
    ck36 = torch.load(CHECKPOINT_DIR / "cosface_v2_best.pt", map_location="cpu", weights_only=False)
    model_36 = CosFaceHead(ck36["num_classes"], s=30, m=0.50, embed_dim_out=128)
    model_36.load_state_dict(ck36["model_state_dict"])

    print("Loading 83-class CosFace model...")
    ck83 = torch.load(CHECKPOINT_DIR / "ri_authority_cosface_best.pt", map_location="cpu", weights_only=False)
    model_83 = CosFaceHead(ck83["num_classes"], s=30, m=0.35)
    model_83.load_state_dict(ck83["model_state_dict"])

    # Project through both models
    print("\nProjecting embeddings...")
    train_proj36 = project_through_model(train_raw, model_36)
    test_proj36 = project_through_model(test_raw, model_36)
    train_proj83 = project_through_model(train_raw, model_83)
    test_proj83 = project_through_model(test_raw, model_83)

    # ── KNN evaluation across all three spaces ──
    print("\nRunning KNN evaluation (k=1, 5, 10)...")

    print("  Raw 768-d...")
    raw_knn = knn_evaluate(train_raw, train_labels, test_raw, test_labels)

    print("  36-class projected 128-d...")
    proj36_knn = knn_evaluate(train_proj36, train_labels, test_proj36, test_labels)

    print("  83-class projected 128-d...")
    proj83_knn = knn_evaluate(train_proj83, train_labels, test_proj83, test_labels)

    # ── Classifier head accuracy ──
    print("\nRunning classifier head evaluation...")
    # 83-class head has the right class set
    head83_top1, head83_top5 = head_evaluate(test_raw, test_labels, model_83, num_classes)

    # 36-class head can't classify 83 classes — skip head eval, only retrieval matters

    # ── Per-class KNN-10 breakdown (83-class space) ──
    print("\nComputing per-class KNN-10 breakdowns...")
    pc_raw = per_class_knn(train_raw, train_labels, test_raw, test_labels, class_map)
    pc_83 = per_class_knn(train_proj83, train_labels, test_proj83, test_labels, class_map)

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"RESULTS: 83-class authority labels, {len(test_labels)} test / {len(train_labels)} ref")
    print(f"{'='*70}")
    print(f"\n  Protocol: train-reference, test-query, cosine, uniform majority vote")
    print(f"  Reference pool: {len(train_labels):,} coins")
    print(f"  Query set: {len(test_labels):,} coins")
    print(f"  Classes: {num_classes}")

    print(f"\n  {'Space':<30} {'KNN-1':>7} {'KNN-5':>7} {'KNN-10':>7}")
    print(f"  {'─'*53}")
    print(f"  {'Raw DINOv2 768-d':<30} {raw_knn[1]:>7.1%} {raw_knn[5]:>7.1%} {raw_knn[10]:>7.1%}")
    print(f"  {'36-class CosFace 128-d':<30} {proj36_knn[1]:>7.1%} {proj36_knn[5]:>7.1%} {proj36_knn[10]:>7.1%}")
    print(f"  {'83-class CosFace 128-d':<30} {proj83_knn[1]:>7.1%} {proj83_knn[5]:>7.1%} {proj83_knn[10]:>7.1%}")

    print(f"\n  83-class classifier head:")
    print(f"    Top-1: {head83_top1:.1%}    Top-5: {head83_top5:.1%}")

    # Delta analysis
    print(f"\n  Deltas vs raw 768-d (KNN-10):")
    print(f"    36-class projection: {proj36_knn[10] - raw_knn[10]:>+7.1%}")
    print(f"    83-class projection: {proj83_knn[10] - raw_knn[10]:>+7.1%}")

    print(f"\n  Delta 83-class vs 36-class (KNN-10): {proj83_knn[10] - proj36_knn[10]:>+7.1%}")

    # Per-class: biggest gainers and losers (83-class vs raw)
    deltas = {}
    for name in pc_83:
        if name in pc_raw and pc_raw[name]["count"] >= 5:
            deltas[name] = {
                "raw": pc_raw[name]["accuracy"],
                "proj83": pc_83[name]["accuracy"],
                "delta": pc_83[name]["accuracy"] - pc_raw[name]["accuracy"],
                "count": pc_83[name]["count"],
            }

    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1]["delta"])

    print(f"\n  Biggest losers (83-class proj vs raw, KNN-10):")
    for name, d in sorted_deltas[:10]:
        print(f"    {name:<30} raw={d['raw']:.0%} proj={d['proj83']:.0%} delta={d['delta']:>+.0%}  (n={d['count']})")

    print(f"\n  Biggest gainers (83-class proj vs raw, KNN-10):")
    for name, d in sorted_deltas[-10:]:
        print(f"    {name:<30} raw={d['raw']:.0%} proj={d['proj83']:.0%} delta={d['delta']:>+.0%}  (n={d['count']})")

    # Save full results
    results = {
        "protocol": {
            "reference": "train split",
            "queries": "test split",
            "distance": "cosine",
            "voting": "uniform majority",
            "n_train": len(train_labels),
            "n_test": len(test_labels),
            "num_classes": num_classes,
        },
        "knn": {
            "raw_768d": {f"knn{k}": v for k, v in raw_knn.items()},
            "proj_36class_128d": {f"knn{k}": v for k, v in proj36_knn.items()},
            "proj_83class_128d": {f"knn{k}": v for k, v in proj83_knn.items()},
        },
        "head_83class": {"top1": head83_top1, "top5": head83_top5},
        "per_class_knn10": {
            "raw_768d": pc_raw,
            "proj_83class_128d": pc_83,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "compare_spaces.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()
