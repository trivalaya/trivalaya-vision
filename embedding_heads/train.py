#!/usr/bin/env python3
"""Train one embedding head.

    python -m embedding_heads.train --head linear
    python -m embedding_heads.train --head mlp
    python -m embedding_heads.train --head arcface
    python -m embedding_heads.train --head cosface
"""

import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embedding_heads.config import CHECKPOINT_DIR, OUTPUT_DIR
from embedding_heads.dataset import build_dataloaders
from embedding_heads.heads import build_head


DEFAULTS = {
    "linear":  {"lr": 1e-3, "epochs": 50, "patience": 7},
    "mlp":     {"lr": 1e-3, "epochs": 50, "patience": 7},
    "arcface": {"lr": 1e-3, "epochs": 200, "patience": 7},
    "cosface": {"lr": 1e-3, "epochs": 200, "patience": 7},
}


def train_one_epoch(model, loader, criterion, optimizer, head_name):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        optimizer.zero_grad()
        if head_name in ("arcface", "cosface"):
            logits = model(embeddings, labels)
        else:
            logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        with torch.no_grad():
            if head_name in ("arcface", "cosface"):
                clean_logits = model(embeddings)
            else:
                clean_logits = logits
            _, predicted = clean_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += embeddings.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, head_name):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for embeddings, labels in loader:
        if head_name in ("arcface", "cosface"):
            # Loss uses margin-applied logits
            margin_logits = model(embeddings, labels)
            loss = criterion(margin_logits, labels)
            # Accuracy uses clean logits (no margin)
            clean_logits = model(embeddings)
        else:
            clean_logits = model(embeddings)
            loss = criterion(clean_logits, labels)

        total_loss += loss.item() * embeddings.size(0)
        _, predicted = clean_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += embeddings.size(0)

    return total_loss / total, correct / total


def train(head_name, lr=None, epochs=None, patience=None, batch_size=1024):
    defaults = DEFAULTS[head_name]
    lr = lr or defaults["lr"]
    epochs = epochs or defaults["epochs"]
    patience = patience or defaults["patience"]

    print(f"=== Training: {head_name} ===")
    print(f"  lr={lr}  epochs={epochs}  patience={patience}  batch_size={batch_size}")

    train_loader, val_loader, test_loader, num_classes, class_map = build_dataloaders(batch_size)
    model = build_head(head_name, num_classes)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    log_path = OUTPUT_DIR / f"train_log_{head_name}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, head_name)
        val_loss, val_acc = validate(model, val_loader, criterion, head_name)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}",
                             f"{current_lr:.6f}", f"{elapsed:.2f}"])

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"lr={current_lr:.1e}  {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "head_name": head_name,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_map": class_map,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f"{head_name}_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nBest: epoch {best_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / f'{head_name}_best.pt'}")

    return best_epoch, best_val_loss, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train embedding head")
    parser.add_argument("--head", required=True, choices=["linear", "mlp", "arcface", "cosface"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    train(args.head, lr=args.lr, epochs=args.epochs,
          patience=args.patience, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
