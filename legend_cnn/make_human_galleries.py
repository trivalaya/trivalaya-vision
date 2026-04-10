#!/usr/bin/env python3
"""Generate human-eye comparison galleries for visual plausibility check.

Three galleries:
  1. Same authority, different coins — shows intra-class variation
  2. Different authority, visually similar — shows inter-class ambiguity
  3. Model confidence: top correct vs top wrong predictions

Usage:
    python -m legend_cnn.make_human_galleries \
        --ribbon-stats runs/run_.../ribbon_stats.csv \
        --checkpoint checkpoints/best.pt \
        --source-csv data/confusion_subset.csv
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from legend_cnn.config import DATA_DIR, OUTPUT_DIR
from legend_cnn.dataset import load_splits
from legend_cnn.model import LegendCNN


def load_ribbon(path):
    return np.load(path).astype(np.float32)


def ribbon_grid(ribbons, titles, suptitle, out_path, cols=4):
    """Plot a grid of ribbons with titles."""
    n = len(ribbons)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 1.8))
    if rows == 1:
        axes = [axes] if cols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, (ribbon, title) in enumerate(zip(ribbons, titles)):
        ax = axes_flat[i]
        ax.imshow(ribbon, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def gallery_same_authority(stats_df, out_dir, n_per_auth=6, seed=42):
    """Gallery 1: Same authority, different coins."""
    rng = random.Random(seed)
    success = stats_df[stats_df["status"] == "success"].copy()
    authorities = sorted(success["authority"].unique())

    for auth in authorities:
        group = success[success["authority"] == auth]
        sample = group.sample(n=min(n_per_auth, len(group)), random_state=seed)

        ribbons = []
        titles = []
        for _, row in sample.iterrows():
            r = load_ribbon(row["ribbon_path"])
            ribbons.append(r)
            qb = row.get("quality_bucket", "")
            titles.append(f"coin {row['coin_id']}  [{qb}]")

        safe_name = auth.replace(" ", "_").lower()
        ribbon_grid(
            ribbons, titles,
            f"Same Authority: {auth} ({len(group)} total coins)",
            out_dir / f"same_auth_{safe_name}.png",
            cols=3,
        )


def gallery_cross_authority(stats_df, out_dir, n_per_auth=3, seed=42):
    """Gallery 2: Different authorities side by side for visual comparison."""
    success = stats_df[stats_df["status"] == "success"].copy()
    authorities = sorted(success["authority"].unique())

    # Pick n_per_auth good-quality samples from each authority
    ribbons = []
    titles = []
    for auth in authorities:
        group = success[success["authority"] == auth]
        good = group[group["quality_bucket"] == "good"]
        if len(good) < n_per_auth:
            good = group
        sample = good.sample(n=min(n_per_auth, len(good)), random_state=seed)
        for _, row in sample.iterrows():
            r = load_ribbon(row["ribbon_path"])
            ribbons.append(r)
            titles.append(f"{auth}\ncoin {row['coin_id']}")

    ribbon_grid(
        ribbons, titles,
        "Cross-Authority Comparison (good quality samples)",
        out_dir / "cross_authority.png",
        cols=n_per_auth,
    )


def gallery_model_confidence(stats_df, checkpoint_path, source_csv_path,
                             out_dir, n_show=8, seed=42):
    """Gallery 3: Model's top confident correct and top confident wrong."""
    # Load model
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_map = ck["class_map"]
    num_classes = ck["num_classes"]

    model = LegendCNN(num_classes=num_classes)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Load splits, use test set
    splits = load_splits()
    test_ids = set(splits["test"])

    source = pd.read_csv(source_csv_path)
    success = stats_df[stats_df["status"] == "success"].copy()
    if "authority_label_int" not in success.columns:
        label_map = source[["coin_id", "authority_label_int"]].drop_duplicates("coin_id")
        success = success.merge(label_map, on="coin_id", how="left")

    test_df = success[success["coin_id"].isin(test_ids)].copy()
    print(f"  Test set: {len(test_df)} ribbons")

    # Build int->auth map from class_map (keys might be strings from JSON)
    int_to_auth = {int(k): v for k, v in class_map.items()}

    # Run inference
    results = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            ribbon = load_ribbon(row["ribbon_path"])
            tensor = torch.from_numpy(ribbon).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_class = probs.argmax().item()
            pred_conf = probs[pred_class].item()
            true_class = int(row["authority_label_int"])

            results.append({
                "coin_id": row["coin_id"],
                "ribbon_path": row["ribbon_path"],
                "true_class": true_class,
                "true_auth": int_to_auth.get(true_class, f"?{true_class}"),
                "pred_class": pred_class,
                "pred_auth": int_to_auth.get(pred_class, f"?{pred_class}"),
                "pred_conf": pred_conf,
                "correct": pred_class == true_class,
                "quality_bucket": row.get("quality_bucket", ""),
            })

    res_df = pd.DataFrame(results)

    correct = res_df[res_df["correct"]].sort_values("pred_conf", ascending=False)
    wrong = res_df[~res_df["correct"]].sort_values("pred_conf", ascending=False)

    print(f"  Correct: {len(correct)}  Wrong: {len(wrong)}")
    print(f"  Test accuracy: {len(correct)/len(res_df):.1%}")

    # Top confident correct
    top_correct = correct.head(n_show)
    ribbons_c, titles_c = [], []
    for _, row in top_correct.iterrows():
        ribbons_c.append(load_ribbon(row["ribbon_path"]))
        titles_c.append(
            f"coin {row['coin_id']}\n"
            f"TRUE: {row['true_auth']}\n"
            f"conf: {row['pred_conf']:.1%}"
        )

    if ribbons_c:
        ribbon_grid(
            ribbons_c, titles_c,
            f"Top {n_show} Most Confident CORRECT Predictions",
            out_dir / "confidence_correct.png",
            cols=4,
        )

    # Top confident wrong
    top_wrong = wrong.head(n_show)
    ribbons_w, titles_w = [], []
    for _, row in top_wrong.iterrows():
        ribbons_w.append(load_ribbon(row["ribbon_path"]))
        titles_w.append(
            f"coin {row['coin_id']}\n"
            f"TRUE: {row['true_auth']}\n"
            f"PRED: {row['pred_auth']}\n"
            f"conf: {row['pred_conf']:.1%}"
        )

    if ribbons_w:
        ribbon_grid(
            ribbons_w, titles_w,
            f"Top {n_show} Most Confident WRONG Predictions",
            out_dir / "confidence_wrong.png",
            cols=4,
        )

    # Confusion summary
    print(f"\n  Most common misclassifications:")
    if len(wrong) > 0:
        wrong["pair"] = wrong["true_auth"] + " -> " + wrong["pred_auth"]
        top_pairs = wrong["pair"].value_counts().head(10)
        for pair, count in top_pairs.items():
            print(f"    {pair}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Human-eye comparison galleries")
    parser.add_argument("--ribbon-stats", required=True)
    parser.add_argument("--checkpoint", default="legend_cnn/checkpoints/best.pt")
    parser.add_argument("--source-csv", default=str(DATA_DIR / "confusion_subset.csv"))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR / "human_galleries"))
    args = parser.parse_args()

    stats_df = pd.read_csv(args.ribbon_stats)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Gallery 1: Same authority, different coins")
    gallery_same_authority(stats_df, out_dir)

    print("\nGallery 2: Cross-authority comparison")
    gallery_cross_authority(stats_df, out_dir)

    print("\nGallery 3: Model confidence (correct vs wrong)")
    gallery_model_confidence(stats_df, args.checkpoint, args.source_csv, out_dir)

    print(f"\nAll galleries saved to {out_dir}")


if __name__ == "__main__":
    main()
