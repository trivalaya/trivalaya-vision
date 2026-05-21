#!/usr/bin/env python3
"""Build expanded confusion subset: top authorities from Imperial + Provincial.

Merges label variants, excludes 'Various', builds stratified splits.

    python -m embedding_heads.build_expanded_subset
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_heads.config import (
    DATA_DIR, OBV_FEATURES_PATH, PROD_META_PATH,
)

# ── Label normalization ─────────────────────────────────────────────

AUTHORITY_MERGES = {
    # Julia Domna variants -> Julia Domna
    "Julia Domna. Augusta": "Julia Domna",
    # Antoninus Pius variants -> Antoninus Pius
    "EGYPT, Alexandria. Antoninus Pius": "Antoninus Pius",
    "ALEXANDRIA. Antoninus Pius": "Antoninus Pius",
    "Divus Antoninus Pius. Died": "Antoninus Pius",
    "Divus Antoninus Pius": "Antoninus Pius",
    # Maximian variants
    "Maximianus": "Maximian",
    # Valerian variants
    "Valerian": "Valerian I",
}

EXCLUDE_AUTHORITIES = {
    "Various",
}

INCLUDE_PERIODS = {
    "Roman Imperial",
    "roman_imperial",
    "Roman Provincial",
}

MIN_COINS = 200
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def build():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(PROD_META_PATH)
    print(f"Loaded {len(meta):,} coins from prod metadata")

    # Filter to Roman Imperial + Provincial
    ri = meta[meta["period"].isin(INCLUDE_PERIODS)].copy()
    print(f"Roman Imperial + Provincial: {len(ri):,}")

    # Normalize authority labels
    ri["authority_clean"] = ri["authority"].replace(AUTHORITY_MERGES)

    # Exclude catch-all labels
    ri = ri[~ri["authority_clean"].isin(EXCLUDE_AUTHORITIES)]

    # Count per authority
    auth_counts = ri["authority_clean"].value_counts()
    viable = auth_counts[auth_counts >= MIN_COINS]
    print(f"\nViable authorities (>= {MIN_COINS} coins): {len(viable)}")

    # Check which coin_ids have embeddings
    obv = np.load(OBV_FEATURES_PATH, mmap_mode="r")
    meta_idx = {cid: i for i, cid in enumerate(meta["coin_id"])}

    # Filter to viable authorities and coins with embeddings
    ri_viable = ri[ri["authority_clean"].isin(viable.index)].copy()
    has_embedding = ri_viable["coin_id"].apply(lambda c: c in meta_idx)
    ri_viable = ri_viable[has_embedding]

    # Assign integer labels
    authorities = sorted(ri_viable["authority_clean"].unique())
    auth_to_int = {a: i for i, a in enumerate(authorities)}
    ri_viable["authority_label_int"] = ri_viable["authority_clean"].map(auth_to_int)

    print(f"Total coins with embeddings: {len(ri_viable):,}")
    print(f"Classes: {len(authorities)}")

    # Print per-authority breakdown
    print(f"\n{'#':>3} {'Authority':<35} {'Total':>6} {'Imp':>5} {'Prov':>5} {'Label':>5}")
    print("-" * 65)
    for auth in authorities:
        sub = ri_viable[ri_viable["authority_clean"] == auth]
        n_imp = sub[sub["period"].isin(["Roman Imperial", "roman_imperial"])].shape[0]
        n_prov = sub[sub["period"] == "Roman Provincial"].shape[0]
        label = auth_to_int[auth]
        print(f"{label:>3} {auth:<35} {len(sub):>6} {n_imp:>5} {n_prov:>5} {label:>5}")

    # Stratified split
    rng = random.Random(SEED)
    train_ids, val_ids, test_ids = [], [], []

    for auth in authorities:
        ids = ri_viable[ri_viable["authority_clean"] == auth]["coin_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(n * VAL_FRAC))
        n_test = max(1, int(n * TEST_FRAC))
        n_train = n - n_val - n_test

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    splits = {
        "version": "v2_expanded",
        "seed": SEED,
        "n_classes": len(authorities),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Save subset CSV
    subset_df = ri_viable[["coin_id", "authority_clean", "authority_label_int",
                           "period", "authority"]].copy()
    subset_df = subset_df.rename(columns={"authority_clean": "authority_normalized",
                                          "authority": "authority_raw"})
    subset_path = DATA_DIR / "expanded_subset.csv"
    subset_df.to_csv(subset_path, index=False)
    print(f"\nSubset CSV: {subset_path} ({len(subset_df):,} rows)")

    # Save splits
    splits_path = DATA_DIR / "splits_v2.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits: {splits_path}")
    print(f"  train: {splits['n_train']}  val: {splits['n_val']}  test: {splits['n_test']}")

    # Save class map
    class_map = {i: a for a, i in auth_to_int.items()}
    class_map_path = DATA_DIR / "class_map_v2.json"
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Class map: {class_map_path}")

    return subset_df, splits, class_map


if __name__ == "__main__":
    build()
