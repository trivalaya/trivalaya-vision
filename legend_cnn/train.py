#!/usr/bin/env python3
"""Step 3D: Train LegendCNN on promoted legend ribbons.

    python -m legend_cnn.train \
        --ribbon-stats runs/run_.../ribbon_stats.csv \
        [--epochs 50] [--batch-size 256] [--lr 1e-3]
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from legend_cnn.config import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR
from legend_cnn.dataset import build_datasets, load_splits
from legend_cnn.model import LegendCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def meta_collate(batch):
    """Custom collate that handles the meta dict."""
    tensors, labels, metas = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long), list(metas)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_training_curves(log_path, out_path):
    """Plot training curves from train_log.csv."""
    import pandas as pd
    df = pd.read_csv(log_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["train_loss"], label="Train")
    ax1.plot(df["epoch"], df["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(ribbon_stats_path, source_csv_path, epochs=50, batch_size=256,
          lr=1e-3, patience=5, seed=42):
    """Full training pipeline."""
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        print(f"  VRAM: {vram / 1e9:.1f} GB")

    # Datasets
    train_ds, val_ds, test_ds, class_map = build_datasets(
        ribbon_stats_path, source_csv_path
    )
    num_classes = len(class_map)
    print(f"\nDataset: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"Classes: {num_classes}")
    for i, auth in sorted(class_map.items()):
        print(f"  [{i}] {auth}")

    # Loaders
    use_cuda = device.type == "cuda"
    n_workers = 4 if use_cuda else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=use_cuda, collate_fn=meta_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=use_cuda, collate_fn=meta_collate)

    # Model
    model = LegendCNN(num_classes=num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Checkpointing
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    # Logging
    log_path = OUTPUT_DIR / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr", "epoch_time"])

    splits = load_splits()

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{val_acc:.4f}", f"{current_lr:.6f}", f"{elapsed:.1f}"])

        # Print
        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={current_lr:.1e}  {elapsed:.1f}s")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_map": class_map,
                "num_classes": num_classes,
                "promoted_run_id": splits.get("promoted_run_id", ""),
                "split_version": splits.get("version", ""),
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest: epoch {best_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / 'best.pt'}")

    # Training curves
    curves_path = OUTPUT_DIR / "training_curves.png"
    save_training_curves(log_path, curves_path)
    print(f"Training curves: {curves_path}")

    return best_epoch, best_val_loss, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train LegendCNN")
    parser.add_argument("--ribbon-stats", required=True, help="Promoted ribbon_stats.csv")
    parser.add_argument("--source-csv", default=str(DATA_DIR / "confusion_subset.csv"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.ribbon_stats, args.source_csv,
          epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, patience=args.patience, seed=args.seed)


if __name__ == "__main__":
    main()
