#!/usr/bin/env python3
"""Step 3E: Evaluate trained LegendCNN on held-out test set.

    python -m legend_cnn.evaluate \
        --ribbon-stats runs/run_.../ribbon_stats.csv \
        [--checkpoint checkpoints/best.pt]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from legend_cnn.config import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR
from legend_cnn.dataset import build_datasets
from legend_cnn.model import LegendCNN
from legend_cnn.train import meta_collate


@torch.no_grad()
def predict_all(model, loader, device):
    """Run inference on all samples, collecting predictions and metadata."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_metas = []

    for images, labels, metas in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_metas.extend(metas)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_metas


def build_predictions_df(preds, labels, probs, metas, class_map):
    """Build test_predictions.csv DataFrame."""
    rows = []
    for pred, true, prob, meta in zip(preds, labels, probs, metas):
        rows.append({
            "coin_id": meta["coin_id"],
            "true_label": int(true),
            "true_authority": class_map[int(true)],
            "pred_label": int(pred),
            "pred_authority": class_map[int(pred)],
            "correct": int(pred == true),
            "confidence": float(prob[pred]),
            "quality_bucket": meta.get("quality_bucket", ""),
            "quality_score": meta.get("quality_score"),
            "used_fallback": meta.get("used_fallback", False),
            "center_method": meta.get("center_method", ""),
        })
    return pd.DataFrame(rows)


def compute_classification_report(preds_df, class_map):
    """Compute per-class precision, recall, F1."""
    classes = sorted(class_map.keys())
    report = {}

    for cls_idx in classes:
        auth = class_map[cls_idx]
        true_mask = preds_df["true_label"] == cls_idx
        pred_mask = preds_df["pred_label"] == cls_idx

        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[auth] = {
            "label": cls_idx,
            "n": int(true_mask.sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    overall_acc = float(preds_df["correct"].mean())
    report["_overall"] = {
        "accuracy": round(overall_acc, 4),
        "n": len(preds_df),
    }

    return report


def plot_confusion_matrix(preds_df, class_map, out_path):
    """Plot and save confusion matrix."""
    n = len(class_map)
    labels = [class_map[i] for i in range(n)]
    cm = np.zeros((n, n), dtype=int)

    for _, row in preds_df.iterrows():
        cm[row["true_label"], row["pred_label"]] += 1

    # Normalize by row (true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (counts)")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    # Normalized
    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (row-normalized)")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_analysis(preds_df, class_map):
    """Run post-eval error analysis slices."""
    print(f"\n{'=' * 60}")
    print("ERROR ANALYSIS")
    print(f"{'=' * 60}")

    # 1. Accuracy by quality_bucket
    print("\n1. Accuracy by quality_bucket:")
    for bucket, g in preds_df.groupby("quality_bucket"):
        acc = g["correct"].mean()
        print(f"  {bucket:15s}  n={len(g):>4d}  acc={acc:.3f}")

    # 2. Accuracy by fallback status
    print("\n2. Accuracy by fallback status:")
    for fb, g in preds_df.groupby("used_fallback"):
        acc = g["correct"].mean()
        label = "fallback" if fb else "no fallback"
        print(f"  {label:15s}  n={len(g):>4d}  acc={acc:.3f}")

    # 3. High cross-confusion pairs (>20%)
    n_classes = len(class_map)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for _, row in preds_df.iterrows():
        cm[row["true_label"], row["pred_label"]] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    print("\n3. Authority pairs with >20% cross-confusion:")
    found = False
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_norm[i, j] > 0.20:
                print(f"  {class_map[i]:25s} -> {class_map[j]:25s}  {cm_norm[i, j]:.1%}  ({cm[i, j]}/{cm.sum(axis=1)[i]})")
                found = True
    if not found:
        print("  None")

    # 4. Top confident wrong predictions
    wrong = preds_df[preds_df["correct"] == 0].sort_values("confidence", ascending=False)
    print(f"\n4. Top 10 confident wrong predictions:")
    for _, row in wrong.head(10).iterrows():
        print(f"  coin={row['coin_id']}  true={row['true_authority']:20s}  "
              f"pred={row['pred_authority']:20s}  conf={row['confidence']:.3f}  "
              f"q={row['quality_score']:.3f}  fb={row['used_fallback']}")

    # 5. Weak recall classes
    print(f"\n5. Classes with recall < 0.70:")
    report = compute_classification_report(preds_df, class_map)
    for auth, stats in sorted(report.items()):
        if auth == "_overall":
            continue
        if stats["recall"] < 0.70:
            print(f"  {auth:25s}  recall={stats['recall']:.3f}  f1={stats['f1']:.3f}  n={stats['n']}")


def evaluate(ribbon_stats_path, source_csv_path, checkpoint_path=None):
    """Full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = checkpoint_path or (CHECKPOINT_DIR / "best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_map = checkpoint["class_map"]
    num_classes = checkpoint["num_classes"]
    print(f"Checkpoint: epoch {checkpoint['epoch']}  "
          f"val_loss={checkpoint['val_loss']:.4f}  val_acc={checkpoint['val_acc']:.4f}")

    # Model
    model = LegendCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test dataset
    _, _, test_ds, _ = build_datasets(ribbon_stats_path, source_csv_path)
    use_cuda = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0 if not use_cuda else 4,
                             pin_memory=use_cuda, collate_fn=meta_collate)
    print(f"Test set: {len(test_ds)} samples")

    # Predict
    preds, labels, probs, metas = predict_all(model, test_loader, device)

    # Build predictions DataFrame
    preds_df = build_predictions_df(preds, labels, probs, metas, class_map)

    # Overall accuracy
    overall_acc = float(preds_df["correct"].mean())
    print(f"\nTest accuracy: {overall_acc:.4f} ({int(preds_df['correct'].sum())}/{len(preds_df)})")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preds_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"Predictions: {OUTPUT_DIR / 'test_predictions.csv'}")

    report = compute_classification_report(preds_df, class_map)
    with open(OUTPUT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report: {OUTPUT_DIR / 'classification_report.json'}")

    # Per-class summary
    print(f"\nPer-class results:")
    print(f"  {'Authority':25s} {'N':>4s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s}")
    for auth in sorted(report):
        if auth == "_overall":
            continue
        s = report[auth]
        print(f"  {auth:25s} {s['n']:>4d} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}")

    # Confusion matrix
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(preds_df, class_map, cm_path)
    print(f"Confusion matrix: {cm_path}")

    # Inference timing
    model.eval()
    dummy = torch.randn(1, 1, 64, 512).to(device)
    times = []
    for _ in range(100):
        t0 = time.time()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    mean_ms = np.mean(times[10:]) * 1000  # skip warmup
    print(f"\nInference latency: {mean_ms:.2f} ms/sample")

    # Error analysis
    error_analysis(preds_df, class_map)

    return overall_acc, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate LegendCNN")
    parser.add_argument("--ribbon-stats", required=True, help="Promoted ribbon_stats.csv")
    parser.add_argument("--source-csv", default=str(DATA_DIR / "confusion_subset.csv"))
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: best.pt)")
    args = parser.parse_args()

    evaluate(args.ribbon_stats, args.source_csv, args.checkpoint)


if __name__ == "__main__":
    main()
