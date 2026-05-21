#!/usr/bin/env python3
"""Build scoped datasets for hierarchical CosFace training.

Generates subsets with label normalization, stratified splits, and class maps.

Scopes:
  period           — All coins, parser_period label, >= 400 coins
  ri_authority     — Roman Imperial + Provincial, authority >= 50 coins
  ri_denomination  — Roman Imperial, denomination >= 50 coins

Usage:
    python -m embedding_heads.build_hierarchy_data
    python -m embedding_heads.build_hierarchy_data --scope period
    python -m embedding_heads.build_hierarchy_data --scope ri_authority
    python -m embedding_heads.build_hierarchy_data --scope ri_denomination
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_heads.config import DATA_DIR, OBV_FEATURES_PATH, PROD_META_PATH

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# ── Roman Imperial Authority scope ──────────────────────────────────

RI_AUTHORITY_PERIODS = {"Roman Imperial", "roman_imperial", "Roman Provincial"}
RI_AUTHORITY_MIN_COINS = 50

RI_AUTHORITY_MERGES = {
    "Julia Domna. Augusta": "Julia Domna",
    "Divus Antoninus Pius. Died": "Antoninus Pius",
    "Divus Antoninus Pius": "Antoninus Pius",
    "EGYPT, Alexandria. Antoninus Pius": "Antoninus Pius",
    "ALEXANDRIA. Antoninus Pius": "Antoninus Pius",
    "Maximianus": "Maximian",
    "Valerian": "Valerian I",
    "Constantinus": "Constantine I",
    "Faustina Senior": "Diva Faustina Senior",
}

RI_AUTHORITY_EXCLUDES = {
    "Various",           # catch-all, not a real authority
    "Illyrian Dynasty",  # dynasty, not an individual
    "Denier",            # denomination leaked into authority column
    "Faustina",          # ambiguous (71/93 paired with "Ae" denom, unclear Senior vs Junior)
}

# ── Period scope ────────────────────────────────────────────────────

PERIOD_ALL_PERIODS = None  # special: no period filter, use all coins
PERIOD_MIN_COINS = 400

PERIOD_MERGES = {
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
    # lowercase variants
    "roman imperial": "roman_imperial",
    "roman provincial": "roman_provincial",
    "roman republican": "roman_republican",
    "central asian": "central_asian",
    "oriental greek": "oriental_greek",
}

PERIOD_EXCLUDES = {
    "unknown", "non_coin", "modern", "india", "tessera",
}

# ── Roman Imperial Denomination scope ───────────────────────────────

RI_DENOMINATION_PERIODS = {"Roman Imperial", "roman_imperial"}
RI_DENOMINATION_MIN_COINS = 50

RI_DENOMINATION_MERGES = {
    # No merges needed yet — Ae/Bronze handled via exclusion
}

RI_DENOMINATION_EXCLUDES = {
    "Ae",      # vague catch-all for bronze
    "Crown",   # scraper contamination (not a Roman denomination)
    "Denier",  # medieval French terminology leaked in
}

# ── Scope definitions ───────────────────────────────────────────────

SCOPES = {
    "period": {
        "periods": PERIOD_ALL_PERIODS,   # None = no period filter
        "label_col": "parser_period",
        "min_coins": PERIOD_MIN_COINS,
        "merges": PERIOD_MERGES,
        "excludes": PERIOD_EXCLUDES,
    },
    "ri_authority": {
        "periods": RI_AUTHORITY_PERIODS,
        "label_col": "authority",
        "min_coins": RI_AUTHORITY_MIN_COINS,
        "merges": RI_AUTHORITY_MERGES,
        "excludes": RI_AUTHORITY_EXCLUDES,
    },
    "ri_denomination": {
        "periods": RI_DENOMINATION_PERIODS,
        "label_col": "denomination",
        "min_coins": RI_DENOMINATION_MIN_COINS,
        "merges": RI_DENOMINATION_MERGES,
        "excludes": RI_DENOMINATION_EXCLUDES,
    },
}


def build_scope(scope_name):
    """Build dataset for a single scope."""
    cfg = SCOPES[scope_name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(PROD_META_PATH)
    print(f"Loaded {len(meta):,} coins from prod metadata")

    # Filter to scope periods (None = all coins)
    if cfg["periods"] is not None:
        df = meta[meta["period"].isin(cfg["periods"])].copy()
    else:
        df = meta.copy()
    print(f"Period filter ({scope_name}): {len(df):,} coins")

    # Drop rows with missing labels
    label_col = cfg["label_col"]
    df = df[df[label_col].notna() & (df[label_col] != "")]
    print(f"With {label_col} labels: {len(df):,}")

    # Normalize labels
    df["label_clean"] = df[label_col].replace(cfg["merges"])

    # Exclude bad labels
    df = df[~df["label_clean"].isin(cfg["excludes"])]

    # Count per label
    counts = df["label_clean"].value_counts()
    viable = counts[counts >= cfg["min_coins"]]
    print(f"Viable classes (>= {cfg['min_coins']} coins): {len(viable)}")

    # Check which coins have embeddings
    meta_idx = {cid: i for i, cid in enumerate(meta["coin_id"])}

    df_viable = df[df["label_clean"].isin(viable.index)].copy()
    has_embedding = df_viable["coin_id"].apply(lambda c: c in meta_idx)
    df_viable = df_viable[has_embedding]

    # Assign integer labels
    labels_sorted = sorted(df_viable["label_clean"].unique())
    label_to_int = {lab: i for i, lab in enumerate(labels_sorted)}
    df_viable["label_int"] = df_viable["label_clean"].map(label_to_int)

    print(f"Total coins with embeddings: {len(df_viable):,}")
    print(f"Classes: {len(labels_sorted)}")

    # Print per-class breakdown
    print(f"\n{'#':>3} {'Label':<35} {'Count':>6}")
    print("-" * 50)
    for lab in labels_sorted:
        n = (df_viable["label_clean"] == lab).sum()
        idx = label_to_int[lab]
        print(f"{idx:>3} {lab:<35} {n:>6}")

    # Stratified split
    rng = random.Random(SEED)
    train_ids, val_ids, test_ids = [], [], []

    for lab in labels_sorted:
        ids = df_viable[df_viable["label_clean"] == lab]["coin_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(n * VAL_FRAC))
        n_test = max(1, int(n * TEST_FRAC))
        n_train = n - n_val - n_test

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    splits = {
        "scope": scope_name,
        "seed": SEED,
        "n_classes": len(labels_sorted),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Save subset CSV
    subset_df = df_viable[["coin_id", "label_clean", "label_int", "period", label_col]].copy()
    subset_df = subset_df.rename(columns={"label_clean": "label_normalized", label_col: "label_raw"})
    subset_path = DATA_DIR / f"{scope_name}_subset.csv"
    subset_df.to_csv(subset_path, index=False)
    print(f"\nSubset CSV: {subset_path} ({len(subset_df):,} rows)")

    # Save splits
    splits_path = DATA_DIR / f"{scope_name}_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits: {splits_path}")
    print(f"  train: {splits['n_train']}  val: {splits['n_val']}  test: {splits['n_test']}")

    # Save class map
    class_map = {i: lab for lab, i in label_to_int.items()}
    class_map_path = DATA_DIR / f"{scope_name}_class_map.json"
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Class map: {class_map_path}")

    return subset_df, splits, class_map


def main():
    parser = argparse.ArgumentParser(description="Build hierarchy datasets")
    parser.add_argument("--scope", default=None, choices=list(SCOPES.keys()),
                        help="Build one scope (default: all)")
    args = parser.parse_args()

    scopes = [args.scope] if args.scope else list(SCOPES.keys())
    for scope in scopes:
        print(f"\n{'='*60}")
        print(f"Building scope: {scope}")
        print(f"{'='*60}\n")
        build_scope(scope)


if __name__ == "__main__":
    main()
