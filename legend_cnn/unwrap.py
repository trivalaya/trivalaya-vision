#!/usr/bin/env python3
"""Step 2: Polar unwrap pipeline for legend ribbons.

Downloads 512x512 coin images and transparent masks from DO Spaces,
finds the coin center/radius from the mask, unwraps the legend annulus
into a grayscale ribbon, and saves as .npy.

This is a thin CLI wrapper over unwrap_core. For parameterized runs,
use run_experiment.py instead.

    python -m legend_cnn.unwrap [--workers 16]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from legend_cnn.config import (
    CONFUSION_SUBSET_CSV,
    DATA_DIR,
    MIN_CONTENT_FRACTION,
    RIBBON_DIR,
)
from legend_cnn.unwrap_core import (
    UnwrapConfig,
    get_s3_client,
    process_coin,
)


def main():
    parser = argparse.ArgumentParser(description="Polar unwrap legend ribbons")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of download/process threads")
    args = parser.parse_args()

    if not CONFUSION_SUBSET_CSV.exists():
        print(f"ERROR: {CONFUSION_SUBSET_CSV} not found. Run select_subset.py first.")
        sys.exit(1)

    df = pd.read_csv(CONFUSION_SUBSET_CSV)
    print(f"Loaded {len(df)} coins from {CONFUSION_SUBSET_CSV}")

    RIBBON_DIR.mkdir(parents=True, exist_ok=True)

    # Skip already-processed coins
    existing = {p.stem for p in RIBBON_DIR.glob("*.npy")}
    to_process = df[~df["coin_id"].astype(str).isin(existing)]
    print(f"Already processed: {len(existing)}, remaining: {len(to_process)}")

    if len(to_process) == 0:
        print("All coins already processed.")
        _update_csv(df)
        return

    # Use default config (matches original hardcoded values)
    cfg = UnwrapConfig()

    s3 = get_s3_client()
    rows = to_process.to_dict("records")

    successes = 0
    failures = 0
    errors = []
    radii = []
    offsets = []
    content_fractions = []
    fallback_counts = {}
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_coin, s3, row, cfg, RIBBON_DIR): row
            for row in rows
        }
        for i, future in enumerate(as_completed(futures), 1):
            coin_id, ribbon_path, meta = future.result()
            if ribbon_path and meta.get("status") == "success":
                # Apply content fraction rejection (backward compat)
                ribbon = np.load(ribbon_path)
                cf = float((ribbon > 0.02).mean())
                if cf < MIN_CONTENT_FRACTION:
                    failures += 1
                    import os
                    os.remove(ribbon_path)
                    errors.append((coin_id, f"low_content ({cf:.1%})"))
                    continue

                successes += 1
                radii.append(meta.get("radius", 0))
                offsets.append(meta.get("center_offset", 0))
                content_fractions.append(cf)
                fb = meta.get("fallback_reason")
                if fb:
                    fallback_counts[fb] = fallback_counts.get(fb, 0) + 1
            else:
                failures += 1
                errors.append((coin_id, meta.get("failure_reason", "unknown")))

            if i % 500 == 0 or i == len(futures):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(futures)}] {rate:.1f} img/s  "
                      f"ok={successes} fail={failures}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({len(rows) / elapsed:.1f} img/s)")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")

    if radii:
        print(f"\n  Geometry:")
        print(f"    Mean radius:        {np.mean(radii):.1f} px")
        print(f"    Mean center offset: {np.mean(offsets):.1f} px")

    if content_fractions:
        cf = np.array(content_fractions)
        print(f"\n  Ribbon quality:")
        print(f"    Content fraction:  mean={cf.mean():.3f}  min={cf.min():.3f}  "
              f"p5={np.percentile(cf, 5):.3f}  p25={np.percentile(cf, 25):.3f}")
        suspect = int((cf < 0.30).sum())
        print(f"    Suspect (<30% content): {suspect} ribbons")

    if fallback_counts:
        print(f"\n  Fallback usage ({sum(fallback_counts.values())} total):")
        for reason, count in sorted(fallback_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    else:
        print(f"\n  Fallback usage: 0 (all contours valid)")

    if errors:
        print(f"\nFirst 20 errors:")
        for cid, err in errors[:20]:
            print(f"  coin_id={cid}: {err}")

    _update_csv(df)


def _update_csv(df):
    """Add ribbon_path column to the CSV for coins that have .npy files."""
    ribbon_paths = {}
    for p in RIBBON_DIR.glob("*.npy"):
        ribbon_paths[p.stem] = str(p)

    df["ribbon_path"] = df["coin_id"].astype(str).map(ribbon_paths).fillna("")
    valid = (df["ribbon_path"] != "").sum()
    df.to_csv(CONFUSION_SUBSET_CSV, index=False)
    print(f"\nUpdated CSV: {valid}/{len(df)} coins have ribbon_path")


if __name__ == "__main__":
    main()
