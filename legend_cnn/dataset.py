"""Step 3A/3B: Dataset freeze, splits, and LegendRibbonDataset.

Builds train/val/test splits from a promoted run's ribbon_stats.csv,
then provides a PyTorch Dataset for training and evaluation.

    # Freeze splits
    python -m legend_cnn.dataset --freeze \
        --ribbon-stats runs/run_.../ribbon_stats.csv \
        --source-csv data/confusion_subset.csv

    # Verify
    python -m legend_cnn.dataset --verify
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from legend_cnn.config import DATA_DIR


SPLITS_FILE = DATA_DIR / "splits_v1.json"

# ── Split Construction ───────────────────────────────────────────────


def build_splits(ribbon_stats_path, source_csv_path, seed=42,
                 train_frac=0.8, val_frac=0.1, test_frac=0.1,
                 min_class_size=10):
    """Build stratified train/val/test splits from promoted ribbon_stats.

    Joins ribbon_stats with source CSV to get authority_label_int.
    Filters to status=success with loadable ribbon files.

    Returns (splits_dict, metadata_df).
    """
    stats = pd.read_csv(ribbon_stats_path)
    source = pd.read_csv(source_csv_path)

    # Join to get authority_label_int
    if "authority_label_int" not in stats.columns:
        label_map = source[["coin_id", "authority_label_int"]].drop_duplicates("coin_id")
        stats = stats.merge(label_map, on="coin_id", how="left")

    # Filter to usable rows
    usable = stats[
        (stats["status"] == "success") &
        stats["ribbon_path"].notna() &
        (stats["ribbon_path"] != "") &
        stats["authority_label_int"].notna()
    ].copy()

    # Verify ribbon files exist
    valid_mask = usable["ribbon_path"].apply(lambda p: Path(p).exists())
    usable = usable[valid_mask]

    # Check class sizes
    class_counts = usable["authority"].value_counts()
    small_classes = class_counts[class_counts < min_class_size]
    if len(small_classes) > 0:
        print(f"WARNING: Dropping {len(small_classes)} classes with < {min_class_size} samples:")
        for auth, cnt in small_classes.items():
            print(f"  {auth}: {cnt}")
        usable = usable[~usable["authority"].isin(small_classes.index)]

    # Stratified split
    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []

    for auth, group in usable.groupby("authority"):
        ids = group["coin_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        n_train = n - n_val - n_test

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    splits = {
        "version": "v1",
        "promoted_run_id": Path(ribbon_stats_path).parent.name,
        "seed": seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    return splits, usable


def save_splits(splits, path=None):
    path = path or SPLITS_FILE
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {path}")
    print(f"  train: {splits['n_train']}  val: {splits['n_val']}  test: {splits['n_test']}")


def load_splits(path=None):
    path = path or SPLITS_FILE
    with open(path) as f:
        return json.load(f)


# ── Dataset Class ────────────────────────────────────────────────────


class LegendRibbonDataset(Dataset):
    """PyTorch Dataset for promoted legend ribbons.

    Returns:
        image_tensor: float32 [1, 64, 512]
        label: int authority class
        meta: dict with coin_id, authority, quality_bucket, etc.
    """

    def __init__(self, df, augment=False):
        """
        Args:
            df: DataFrame with ribbon_path, authority_label_int, and metadata columns.
            augment: if True, apply training augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load ribbon
        ribbon = np.load(row["ribbon_path"]).astype(np.float32)
        tensor = torch.from_numpy(ribbon).unsqueeze(0)  # [1, H, W]

        # Augmentations (training only)
        if self.augment:
            # Horizontal circular shift (simulates coin rotation)
            shift = torch.randint(0, tensor.shape[-1], (1,)).item()
            tensor = torch.roll(tensor, shifts=shift, dims=-1)

            # Brightness jitter ±10%
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            tensor = tensor * brightness

            # Contrast jitter ±10%
            mean_val = tensor.mean()
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            tensor = (tensor - mean_val) * contrast + mean_val

            tensor = tensor.clamp(0.0, 1.0)

        label = int(row["authority_label_int"])

        meta = {
            "coin_id": int(row["coin_id"]),
            "authority": row.get("authority", ""),
            "quality_bucket": row.get("quality_bucket", ""),
            "quality_score": row.get("quality_score", None),
            "used_fallback": bool(row.get("used_fallback", False)),
            "center_method": row.get("center_method", ""),
        }

        return tensor, label, meta


def build_datasets(ribbon_stats_path, source_csv_path, splits_path=None):
    """Build train/val/test datasets from saved splits.

    Returns (train_ds, val_ds, test_ds, class_map).
    """
    splits = load_splits(splits_path)
    stats = pd.read_csv(ribbon_stats_path)
    source = pd.read_csv(source_csv_path)

    # Join to get authority_label_int
    if "authority_label_int" not in stats.columns:
        label_map = source[["coin_id", "authority_label_int"]].drop_duplicates("coin_id")
        stats = stats.merge(label_map, on="coin_id", how="left")

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    train_df = stats[stats["coin_id"].isin(train_set)]
    val_df = stats[stats["coin_id"].isin(val_set)]
    test_df = stats[stats["coin_id"].isin(test_set)]

    # Build class map
    all_auths = sorted(stats["authority"].unique())
    auth_to_int = {a: i for i, a in enumerate(all_auths)}
    class_map = {i: a for a, i in auth_to_int.items()}

    train_ds = LegendRibbonDataset(train_df, augment=True)
    val_ds = LegendRibbonDataset(val_df, augment=False)
    test_ds = LegendRibbonDataset(test_df, augment=False)

    return train_ds, val_ds, test_ds, class_map


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Dataset freeze and verification")
    parser.add_argument("--freeze", action="store_true", help="Build and save splits")
    parser.add_argument("--verify", action="store_true", help="Verify saved splits")
    parser.add_argument("--ribbon-stats", type=str, help="Path to ribbon_stats.csv")
    parser.add_argument("--source-csv", type=str,
                        default=str(DATA_DIR / "confusion_subset.csv"))
    args = parser.parse_args()

    if args.freeze:
        if not args.ribbon_stats:
            print("ERROR: --ribbon-stats required for --freeze")
            return
        splits, usable = build_splits(args.ribbon_stats, args.source_csv)
        save_splits(splits)

        # Print per-authority breakdown
        for split_name in ["train", "val", "test"]:
            ids = set(splits[split_name])
            sub = usable[usable["coin_id"].isin(ids)]
            print(f"\n{split_name} ({len(sub)} coins):")
            for auth, g in sub.groupby("authority"):
                print(f"  {auth:25s} {len(g):>4d}")

    elif args.verify:
        splits = load_splits()
        print(f"Split version: {splits['version']}")
        print(f"Promoted run: {splits['promoted_run_id']}")
        print(f"Seed: {splits['seed']}")
        print(f"Train: {splits['n_train']}  Val: {splits['n_val']}  Test: {splits['n_test']}")

        # Check no overlap
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set), "Train/val overlap!"
        assert train_set.isdisjoint(test_set), "Train/test overlap!"
        assert val_set.isdisjoint(test_set), "Val/test overlap!"
        print("No split overlap: OK")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
