#!/usr/bin/env python3
"""Sweep CosFace hyperparameters for 36-class expanded dataset.

    python -m embedding_heads.sweep_cosface
"""

import csv
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embedding_heads.config import CHECKPOINT_DIR, OUTPUT_DIR
from embedding_heads.dataset import build_dataloaders
from embedding_heads.heads import CosFaceHead


SWEEP_GRID = {
    "lr": [1e-2, 5e-3],
    "m": [0.35, 0.50],
    "s": [30, 64],
    "dim": [128, 256],
}

EPOCHS = 200
PATIENCE = 7
BATCH_SIZE = 1024
VERSION = "v2"


def train_and_eval(lr, s, m, dim, train_loader, val_loader, test_loader,
                   num_classes, embed_dict, coin_to_label, splits):
    """Train one CosFace config and return metrics."""

    model = CosFaceHead(num_classes, s=s, m=m, embed_dim_out=dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            logits = model(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                margin_logits = model(embeddings, labels)
                loss = criterion(margin_logits, labels)
                val_loss += loss.item() * embeddings.size(0)
                val_total += embeddings.size(0)

        val_loss /= val_total
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

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # Collect projected embeddings for KNN
    def collect(ids):
        raw_list, proj_list, label_list = [], [], []
        with torch.no_grad():
            for cid in ids:
                if cid in embed_dict and cid in coin_to_label:
                    r = embed_dict[cid]
                    t = torch.from_numpy(r).float().unsqueeze(0)
                    p = model.embed(t).squeeze(0).numpy()
                    raw_list.append(r)
                    proj_list.append(p)
                    label_list.append(coin_to_label[cid])
        return np.array(raw_list), np.array(proj_list), np.array(label_list)

    train_raw, train_proj, train_labels = collect(splits["train"])
    test_raw, test_proj, test_labels = collect(splits["test"])

    # KNN metrics (primary)
    knn_accs = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_proj, train_labels)
        knn_accs[k] = knn.score(test_proj, test_labels)

    # Prototype cosine (margin weights as centroids)
    prototypes = F.normalize(model.margin.weight.data, dim=1).numpy()
    cos_sim = test_proj @ prototypes.T
    proto_top1 = (cos_sim.argmax(axis=1) == test_labels).mean()

    # Head classification
    with torch.no_grad():
        logits = model(torch.from_numpy(test_raw).float())
        probs = F.softmax(logits, dim=1).numpy()

    head_top1 = (probs.argmax(axis=1) == test_labels).mean()
    head_top5 = np.mean([test_labels[i] in np.argsort(-probs[i])[:5] for i in range(len(test_labels))])
    head_top10 = np.mean([test_labels[i] in np.argsort(-probs[i])[:10] for i in range(len(test_labels))])

    # Quick calibration: mean confidence of correct vs wrong
    top1_confs = probs.max(axis=1)
    top1_preds = probs.argmax(axis=1)
    correct_mask = top1_preds == test_labels
    cal_gap = top1_confs[correct_mask].mean() - top1_confs[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

    return {
        "lr": lr, "s": s, "m": m, "dim": dim,
        "best_epoch": best_epoch,
        "knn1": knn_accs[1], "knn5": knn_accs[5], "knn10": knn_accs[10],
        "proto_top1": proto_top1,
        "head_top1": head_top1, "head_top5": head_top5, "head_top10": head_top10,
        "cal_gap": cal_gap,
    }


def main():
    import json
    import pandas as pd
    from embedding_heads.dataset import extract_subset_embeddings, EXPANDED_SUBSET_PATH, EXPANDED_SPLITS_PATH

    # Load data once
    train_loader, val_loader, test_loader, num_classes, class_map = build_dataloaders(
        BATCH_SIZE, version=VERSION
    )

    embed_dict, _ = extract_subset_embeddings(version=VERSION)
    subset = pd.read_csv(EXPANDED_SUBSET_PATH)
    coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))
    with open(EXPANDED_SPLITS_PATH) as f:
        splits = json.load(f)

    combos = list(itertools.product(
        SWEEP_GRID["lr"], SWEEP_GRID["s"], SWEEP_GRID["m"], SWEEP_GRID["dim"]
    ))
    print(f"Sweep: {len(combos)} configurations")
    print()

    results = []
    for i, (lr, s, m, dim) in enumerate(combos, 1):
        tag = f"lr={lr} s={s} m={m:.2f} dim={dim}"
        t0 = time.time()
        print(f"[{i}/{len(combos)}] {tag} ...", end=" ", flush=True)

        r = train_and_eval(lr, s, m, dim, train_loader, val_loader, test_loader,
                           num_classes, embed_dict, coin_to_label, splits)
        elapsed = time.time() - t0
        r["time"] = round(elapsed, 1)

        print(f"KNN-10={r['knn10']:.1%}  proto={r['proto_top1']:.1%}  "
              f"top5={r['head_top5']:.1%}  ep={r['best_epoch']}  {elapsed:.0f}s")

        results.append(r)

    # Sort by KNN-10 (primary metric)
    results.sort(key=lambda r: -r["knn10"])

    print("\n" + "=" * 100)
    print("SWEEP RESULTS (sorted by KNN-10)")
    print("=" * 100)
    header = (f"{'lr':>6} {'s':>3} {'m':>5} {'dim':>4}  "
              f"{'KNN-1':>6} {'KNN-5':>6} {'KNN-10':>6}  "
              f"{'Proto':>6} {'Top-5':>6} {'Top-10':>6}  "
              f"{'CalGap':>6} {'Ep':>3}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['lr']:>6.0e} {r['s']:>3.0f} {r['m']:>5.2f} {r['dim']:>4}  "
              f"{r['knn1']:>6.1%} {r['knn5']:>6.1%} {r['knn10']:>6.1%}  "
              f"{r['proto_top1']:>6.1%} {r['head_top5']:>6.1%} {r['head_top10']:>6.1%}  "
              f"{r['cal_gap']:>6.3f} {r['best_epoch']:>3}")

    # Save best checkpoint
    best = results[0]
    print(f"\nBest: lr={best['lr']} s={best['s']} m={best['m']} dim={best['dim']}")
    print(f"  KNN-10={best['knn10']:.1%}  Proto={best['proto_top1']:.1%}  "
          f"Top-5={best['head_top5']:.1%}  Top-10={best['head_top10']:.1%}")

    # Retrain best config and save checkpoint
    print("\nRetraining best config for checkpoint save...")
    model = CosFaceHead(num_classes, s=best["s"], m=best["m"], embed_dim_out=best["dim"])
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=best["lr"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            logits = model(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                margin_logits = model(embeddings, labels)
                loss = criterion(margin_logits, labels)
                val_loss += loss.item() * embeddings.size(0)
                val_total += embeddings.size(0)
        val_loss /= val_total
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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_map": class_map,
        "head_name": "cosface",
        "lr": best["lr"], "s": best["s"], "m": best["m"],
        "embed_dim_out": best["dim"],
        "epoch": best_epoch,
        "version": VERSION,
    }, CHECKPOINT_DIR / "cosface_v2_best.pt")
    print(f"Saved: {CHECKPOINT_DIR / 'cosface_v2_best.pt'}")

    # Save sweep results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "sweep_cosface_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sweep log: {OUTPUT_DIR / 'sweep_cosface_v2.json'}")


if __name__ == "__main__":
    main()
