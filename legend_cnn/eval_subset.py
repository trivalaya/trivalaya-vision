#!/usr/bin/env python3
"""Build a frozen evaluation subset by stratified random sampling.

Samples from confusion_subset.csv, balancing across authorities.
Supports manual injection of known hard cases.

    python -m legend_cnn.eval_subset [--target 300] [--seed 42]
    python -m legend_cnn.eval_subset --add-hard-cases hard_cases.txt
"""

import argparse
import sys

import pandas as pd

from legend_cnn.config import CONFUSION_SUBSET_CSV, DATA_DIR, EVAL_SUBSET_CSV


def stratified_sample(df, target_n, seed=42):
    """Stratified random sample balanced across authorities.

    Each authority gets at least floor(target_n / n_authorities) coins,
    then remaining slots are filled from authorities with the most headroom.
    """
    authorities = sorted(df["authority"].unique())
    n_auth = len(authorities)
    per_auth = target_n // n_auth
    remainder = target_n - per_auth * n_auth

    rng = pd.np if hasattr(pd, "np") else __import__("numpy")
    samples = []
    auth_pools = {}

    for auth in authorities:
        pool = df[df["authority"] == auth]
        auth_pools[auth] = pool
        n = min(per_auth, len(pool))
        samples.append(pool.sample(n=n, random_state=seed))

    # Distribute remainder to authorities with most available coins
    headroom = [
        (auth, len(auth_pools[auth]) - min(per_auth, len(auth_pools[auth])))
        for auth in authorities
    ]
    headroom.sort(key=lambda x: -x[1])

    already_sampled = pd.concat(samples, ignore_index=True)
    sampled_ids = set(already_sampled["coin_id"])

    for auth, _ in headroom:
        if remainder <= 0:
            break
        pool = auth_pools[auth]
        remaining = pool[~pool["coin_id"].isin(sampled_ids)]
        if remaining.empty:
            continue
        n = min(1, len(remaining), remainder)
        extra = remaining.sample(n=n, random_state=seed + hash(auth) % 10000)
        samples.append(extra)
        sampled_ids.update(extra["coin_id"])
        remainder -= n

    return pd.concat(samples, ignore_index=True)


def add_hard_cases(df, hard_case_file, source_df):
    """Inject known hard cases from a text file (one coin_id per line)."""
    with open(hard_case_file) as f:
        hard_ids = {int(line.strip()) for line in f if line.strip().isdigit()}

    existing_ids = set(df["coin_id"])
    to_add = hard_ids - existing_ids

    if not to_add:
        print(f"All {len(hard_ids)} hard cases already in subset.")
        return df

    new_rows = source_df[source_df["coin_id"].isin(to_add)].copy()
    if len(new_rows) < len(to_add):
        missing = to_add - set(new_rows["coin_id"])
        print(f"WARNING: {len(missing)} hard case coin_ids not found in source CSV: {missing}")

    new_rows["known_hard_case"] = True
    # Mark existing hard cases too
    df.loc[df["coin_id"].isin(hard_ids), "known_hard_case"] = True

    result = pd.concat([df, new_rows], ignore_index=True)
    print(f"Added {len(new_rows)} hard cases (total now {len(result)})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build eval subset from confusion_subset.csv")
    parser.add_argument("--target", type=int, default=300,
                        help="Target number of coins (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--add-hard-cases", type=str, default=None,
                        help="Text file with coin_ids to force-include as hard cases")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/eval_subset_v1.csv)")
    args = parser.parse_args()

    if not CONFUSION_SUBSET_CSV.exists():
        print(f"ERROR: {CONFUSION_SUBSET_CSV} not found. Run select_subset.py first.")
        sys.exit(1)

    source_df = pd.read_csv(CONFUSION_SUBSET_CSV)
    # Only keep rows with image URLs
    source_df = source_df[
        source_df["highres_url"].notna() & (source_df["highres_url"] != "") &
        source_df["transparent_url"].notna() & (source_df["transparent_url"] != "")
    ].copy()
    print(f"Source: {len(source_df)} coins across {source_df['authority'].nunique()} authorities")

    out_path = args.output or str(EVAL_SUBSET_CSV)

    if args.add_hard_cases:
        # Load existing subset and add hard cases
        existing = pd.read_csv(out_path)
        result = add_hard_cases(existing, args.add_hard_cases, source_df)
    else:
        # Fresh stratified sample
        result = stratified_sample(source_df, args.target, args.seed)
        if "known_hard_case" not in result.columns:
            result["known_hard_case"] = False

    # Keep only the columns we need
    keep_cols = ["coin_id", "authority", "highres_url", "transparent_url", "known_hard_case"]
    for col in keep_cols:
        if col not in result.columns:
            result[col] = False if col == "known_hard_case" else ""
    result = result[keep_cols]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nEval subset: {len(result)} coins")
    print(f"Per-authority breakdown:")
    for auth, group in result.groupby("authority"):
        hard = group["known_hard_case"].sum()
        hard_str = f" ({hard} hard)" if hard else ""
        print(f"  {auth:30s} {len(group):>4d}{hard_str}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
