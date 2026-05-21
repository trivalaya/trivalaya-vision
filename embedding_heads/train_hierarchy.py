#!/usr/bin/env python3
"""Train CosFace heads for hierarchical identification.

Trains a scoped CosFace head (768 -> 128-d projection + margin layer),
evaluates classification (top-1, top-5) and retrieval (KNN-10),
exports production assets.

Usage:
    python -m embedding_heads.train_hierarchy --scope ri_authority
    python -m embedding_heads.train_hierarchy --scope ri_denomination
    python -m embedding_heads.train_hierarchy --scope ri_authority --lr 1e-3 --s 30 --m 0.35
"""

import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embedding_heads.config import CHECKPOINT_DIR, OUTPUT_DIR
from embedding_heads.dataset import build_dataloaders_from_scope
from embedding_heads.heads import CosFaceHead


# ── Spec-compliant scope -> export name mapping ────────────────────

EXPORT_NAMES = {
    "period": "period",
    "ri_authority": "roman_imperial_authority",
    "ri_denomination": "roman_imperial_denomination",
}


def train(scope, lr=1e-3, epochs=200, patience=7, batch_size=1024, s=30.0, m=0.35):
    print(f"=== Training CosFace head: {scope} ===")
    print(f"  lr={lr}  s={s}  m={m}  epochs={epochs}  patience={patience}")

    train_loader, val_loader, test_loader, num_classes, class_map = \
        build_dataloaders_from_scope(scope, batch_size)

    model = CosFaceHead(num_classes, s=s, m=m)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Classes: {num_classes}  Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ck_name = f"{scope}_cosface_best.pt"
    log_path = OUTPUT_DIR / f"train_log_{scope}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            logits = model(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * embeddings.size(0)
            with torch.no_grad():
                clean_logits = model(embeddings)
                _, predicted = clean_logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += embeddings.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                margin_logits = model(embeddings, labels)
                loss = criterion(margin_logits, labels)
                clean_logits = model(embeddings)

                val_loss_sum += loss.item() * embeddings.size(0)
                _, predicted = clean_logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += embeddings.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}",
                             f"{current_lr:.6f}", f"{elapsed:.2f}"])

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}  "
              f"lr={current_lr:.1e}  {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "scope": scope,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_map": class_map,
                "config": {"lr": lr, "s": s, "m": m, "batch_size": batch_size},
            }
            torch.save(checkpoint, CHECKPOINT_DIR / ck_name)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest: epoch {best_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / ck_name}")

    # Evaluate on test set (train set as KNN reference pool)
    evaluate(scope, CHECKPOINT_DIR / ck_name, train_loader, test_loader, class_map)

    return best_epoch, best_val_loss, best_val_acc


@torch.no_grad()
def evaluate(scope, checkpoint_path, train_loader, test_loader, class_map):
    """Evaluate: top-1, top-5 classification + KNN-10 retrieval accuracy.

    KNN-10 uses the training set as the reference pool and queries with
    the test set, matching the methodology used in hdbscan_vs_cosface.py.
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = ck["num_classes"]
    s = ck["config"]["s"]
    m = ck["config"]["m"]

    model = CosFaceHead(num_classes, s=s, m=m)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Collect test embeddings + labels + projected embeddings
    all_logits = []
    all_labels = []
    all_projected = []

    for embeddings, labels in test_loader:
        logits = model(embeddings)
        projected = model.embed(embeddings)

        all_logits.append(logits)
        all_labels.append(labels)
        all_projected.append(projected)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    test_projected = torch.cat(all_projected)

    n = all_labels.size(0)

    # Top-1 and top-5 classification accuracy
    _, top1_pred = all_logits.topk(1, dim=1)
    top1_acc = top1_pred.squeeze(1).eq(all_labels).float().mean().item()

    k5 = min(5, num_classes)
    _, top5_pred = all_logits.topk(k5, dim=1)
    top5_acc = top5_pred.eq(all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Build train reference pool for KNN
    train_projected = []
    train_labels = []
    for embeddings, labels in train_loader:
        projected = model.embed(embeddings)
        train_projected.append(projected)
        train_labels.append(labels)

    train_projected = torch.cat(train_projected)  # (N_train, 128)
    train_labels = torch.cat(train_labels)         # (N_train,)

    # KNN-10: query test against train reference pool
    # Process in batches to avoid OOM on large sets
    k_nn = 10
    knn_correct = 0
    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        query_batch = test_projected[start:end]       # (B, 128)
        label_batch = all_labels[start:end]            # (B,)

        sim = query_batch @ train_projected.T          # (B, N_train)
        _, knn_idx = sim.topk(k_nn, dim=1)             # (B, 10)
        knn_lab = train_labels[knn_idx]                 # (B, 10)

        # Majority vote
        for i in range(end - start):
            neighbor_labels = knn_lab[i].tolist()
            label_counts = {}
            for lab in neighbor_labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            predicted = max(label_counts, key=label_counts.get)
            if predicted == label_batch[i].item():
                knn_correct += 1

    knn_acc = knn_correct / n

    print(f"\n{'='*60}")
    print(f"Test evaluation: {scope} ({n} coins, {num_classes} classes)")
    print(f"{'='*60}")
    print(f"  Top-1 accuracy:  {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-5 accuracy:  {top5_acc:.4f} ({top5_acc*100:.1f}%)")
    print(f"  KNN-10 accuracy: {knn_acc:.4f} ({knn_acc*100:.1f}%)")

    # Per-class breakdown for bottom-10 (hardest classes)
    per_class = {}
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        class_logits = all_logits[mask]
        class_labels = all_labels[mask]
        _, pred = class_logits.topk(1, dim=1)
        acc = pred.squeeze(1).eq(class_labels).float().mean().item()
        per_class[class_map[i]] = (acc, int(mask.sum()))

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1][0])
    print(f"\nHardest 10 classes (by top-1):")
    for name, (acc, count) in sorted_classes[:10]:
        print(f"  {name:<35} {acc*100:5.1f}%  (n={count})")

    print(f"\nEasiest 10 classes (by top-1):")
    for name, (acc, count) in sorted_classes[-10:]:
        print(f"  {name:<35} {acc*100:5.1f}%  (n={count})")

    # Save evaluation results
    results = {
        "scope": scope,
        "n_test": n,
        "num_classes": num_classes,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "knn10_accuracy": knn_acc,
        "per_class": {name: {"accuracy": acc, "count": count}
                      for name, (acc, count) in per_class.items()},
    }
    results_path = OUTPUT_DIR / f"eval_{scope}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return results


def eval_only(scope, batch_size=1024):
    """Re-evaluate an existing checkpoint without retraining."""
    checkpoint_path = CHECKPOINT_DIR / f"{scope}_cosface_best.pt"
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_map = {int(k): v for k, v in ck["class_map"].items()}

    train_loader, _, test_loader, num_classes, _ = \
        build_dataloaders_from_scope(scope, batch_size)

    return evaluate(scope, checkpoint_path, train_loader, test_loader, class_map)


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical CosFace head")
    parser.add_argument("--scope", required=True, choices=["period", "ri_authority", "ri_denomination"])
    parser.add_argument("--eval-only", action="store_true", help="Re-evaluate existing checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--s", type=float, default=30.0, help="CosFace scale")
    parser.add_argument("--m", type=float, default=0.35, help="CosFace margin")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args.scope, batch_size=args.batch_size)
    else:
        train(args.scope, lr=args.lr, epochs=args.epochs, patience=args.patience,
              batch_size=args.batch_size, s=args.s, m=args.m)


if __name__ == "__main__":
    main()
