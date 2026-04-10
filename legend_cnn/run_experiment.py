#!/usr/bin/env python3
"""Run one unwrap experiment: config + subset -> ribbons + stats + summary.

    python -m legend_cnn.run_experiment \
        --config configs/unwrap_v1_ir072_or090_bbox.json \
        --subset data/eval_subset_v1.csv \
        [--baseline runs/baseline_eval_v1] \
        [--calibrate]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from legend_cnn.config import NORM_RANGES_FILE, RUNS_DIR
from legend_cnn.unwrap_core import UnwrapConfig, get_s3_client, process_coin
from legend_cnn.score_ribbons import (
    NormRanges,
    assign_geometry_flags,
    assign_quality_bucket,
    calibrate_norm_ranges,
    compute_quality_score,
    compute_ribbon_metrics,
)


def load_config(config_path):
    with open(config_path) as f:
        d = json.load(f)
    return UnwrapConfig.from_dict(d)


def make_run_dir(cfg):
    """Create timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}_{cfg.config_id}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_summary(run_id, cfg, stats_df, wall_clock, cache_hits, cache_total):
    """Build summary.json dict from ribbon_stats DataFrame."""
    n_total = len(stats_df)
    n_success = int((stats_df["status"] == "success").sum())
    n_failed = int((stats_df["status"] == "failed").sum())

    bucket_counts = stats_df[stats_df["status"] == "success"]["quality_bucket"].value_counts()

    scores = stats_df.loc[stats_df["status"] == "success", "quality_score"].dropna()

    summary = {
        "run_id": run_id,
        "config_id": cfg.config_id,
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_good": int(bucket_counts.get("good", 0)),
        "n_borderline": int(bucket_counts.get("borderline", 0)),
        "n_bad_blank": int(bucket_counts.get("bad_blank", 0)),
        "n_bad_low_signal": int(bucket_counts.get("bad_low_signal", 0)),
        "fallback_rate": round(float(stats_df["used_fallback"].sum()) / max(n_total, 1), 4),
        "n_with_geometry_flags": int(
            (stats_df.get("geometry_flags", pd.Series(dtype=str)).fillna("") != "").sum()
        ),
        "mean_quality_score": round(float(scores.mean()), 4) if len(scores) else 0.0,
        "median_quality_score": round(float(scores.median()), 4) if len(scores) else 0.0,
        "wall_clock_seconds": round(wall_clock, 2),
        "mean_seconds_per_coin": round(wall_clock / max(n_total, 1), 3),
        "cache_hit_rate": round(cache_hits / max(cache_total, 1), 4),
    }

    # Per-authority stats
    per_auth = {}
    for auth, group in stats_df.groupby("authority"):
        auth_success = group[group["status"] == "success"]
        auth_scores = auth_success["quality_score"].dropna()
        per_auth[auth] = {
            "n": len(group),
            "mean_quality_score": round(float(auth_scores.mean()), 4) if len(auth_scores) else 0.0,
            "good_rate": round(
                float((auth_success["quality_bucket"] == "good").sum()) / max(len(group), 1), 4
            ),
            "fallback_rate": round(float(group["used_fallback"].sum()) / max(len(group), 1), 4),
            "failure_rate": round(float((group["status"] == "failed").sum()) / max(len(group), 1), 4),
        }
    summary["per_authority_stats"] = per_auth

    # Hard case stats
    hard = stats_df[stats_df.get("known_hard_case", False) == True]
    if len(hard) > 0:
        hard_success = hard[hard["status"] == "success"]
        hard_scores = hard_success["quality_score"].dropna()
        summary["hard_case_stats"] = {
            "n": len(hard),
            "mean_quality_score": round(float(hard_scores.mean()), 4) if len(hard_scores) else 0.0,
            "good_rate": round(
                float((hard_success["quality_bucket"] == "good").sum()) / max(len(hard), 1), 4
            ),
            "fallback_rate": round(float(hard["used_fallback"].sum()) / max(len(hard), 1), 4),
        }
    else:
        summary["hard_case_stats"] = {"n": 0}

    return summary


def run_experiment(cfg, subset_path, norm_ranges=None, baseline_dir=None,
                   workers=8, save_previews=False):
    """Execute one experiment run.

    Returns (run_dir, summary_dict).
    """
    df = pd.read_csv(subset_path)
    print(f"Loaded {len(df)} coins from {subset_path}")

    run_dir = make_run_dir(cfg)
    ribbon_dir = run_dir / "ribbons"
    ribbon_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Load or create norm ranges
    if norm_ranges is None:
        if NORM_RANGES_FILE.exists():
            norm_ranges = NormRanges.load(NORM_RANGES_FILE)
            print(f"Loaded normalization ranges from {NORM_RANGES_FILE}")
        else:
            norm_ranges = NormRanges()
            print("WARNING: Using placeholder normalization ranges. Run with --calibrate first.")

    s3 = get_s3_client()
    rows = df.to_dict("records")

    # Process all coins
    results = []
    cache_hits = 0
    cache_total = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_coin, s3, row, cfg, ribbon_dir, save_previews): row
            for row in rows
        }
        for i, future in enumerate(as_completed(futures), 1):
            coin_id, ribbon_path, meta = future.result()
            row = futures[future]

            # Track cache hits
            if meta.get("cache_hit_highres"):
                cache_hits += 1
            if meta.get("cache_hit_transparent"):
                cache_hits += 1
            cache_total += 2

            # Compute metrics and score for successful ribbons
            if ribbon_path and meta.get("status") == "success":
                ribbon = np.load(ribbon_path)
                metrics = compute_ribbon_metrics(ribbon)
                qscore = compute_quality_score(metrics, meta, norm_ranges, cfg.max_center_offset)
                bucket = assign_quality_bucket(metrics, meta, qscore)
                geo_flags = assign_geometry_flags(meta)
            else:
                metrics = {
                    "nonblack_fraction": None, "nonzero_fraction": None,
                    "mean_intensity": None, "std_intensity": None,
                    "gradient_energy": None, "edge_density": None,
                    "row_variance_mean": None,
                }
                qscore = None
                bucket = None
                geo_flags = []

            results.append({
                "coin_id": coin_id,
                "authority": row.get("authority", ""),
                "ribbon_path": ribbon_path or "",
                "preview_path": "",
                "status": meta.get("status", "failed"),
                "failure_reason": meta.get("failure_reason"),
                "quality_bucket": bucket,
                "quality_score": qscore,
                "geometry_flags": "|".join(geo_flags) if geo_flags else "",
                **metrics,
                "cx": meta.get("cx"),
                "cy": meta.get("cy"),
                "radius": meta.get("radius"),
                "center_offset": meta.get("center_offset"),
                "mask_area_fraction": meta.get("mask_area_fraction"),
                "used_fallback": meta.get("used_fallback", False),
                "fallback_reason": meta.get("fallback_reason"),
                "center_method": cfg.center_method,
                "config_id": cfg.config_id,
                "known_hard_case": row.get("known_hard_case", False),
            })

            if i % 50 == 0 or i == len(futures):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                n_ok = sum(1 for r in results if r["status"] == "success")
                print(f"  [{i}/{len(futures)}] {rate:.1f} coin/s  ok={n_ok}")

    wall_clock = time.time() - start

    # Validate status/bucket invariant
    stats_df = pd.DataFrame(results)
    failed_with_bucket = stats_df[
        (stats_df["status"] == "failed") & stats_df["quality_bucket"].notna()
    ]
    success_no_bucket = stats_df[
        (stats_df["status"] == "success") & stats_df["quality_bucket"].isna()
    ]
    if len(failed_with_bucket) > 0:
        print(f"WARNING: {len(failed_with_bucket)} failed ribbons have a quality bucket (bug)")
    if len(success_no_bucket) > 0:
        print(f"WARNING: {len(success_no_bucket)} successful ribbons missing quality bucket (bug)")

    # Write ribbon_stats.csv
    stats_df.to_csv(run_dir / "ribbon_stats.csv", index=False)

    # Build and write summary
    run_id = run_dir.name
    summary = build_summary(run_id, cfg, stats_df, wall_clock, cache_hits, cache_total)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Run: {run_id}")
    print(f"  Total: {summary['n_total']}  Success: {summary['n_success']}  "
          f"Failed: {summary['n_failed']}")
    print(f"  Good: {summary['n_good']}  Borderline: {summary['n_borderline']}  "
          f"Bad blank: {summary['n_bad_blank']}  Bad signal: {summary['n_bad_low_signal']}")
    print(f"  Geometry flags: {summary['n_with_geometry_flags']}  "
          f"Fallback rate: {summary['fallback_rate']:.1%}")
    print(f"  Mean score: {summary['mean_quality_score']}  "
          f"Median: {summary['median_quality_score']}")
    print(f"  Fallback rate: {summary['fallback_rate']:.1%}")
    print(f"  Cache hit rate: {summary['cache_hit_rate']:.1%}")
    print(f"  Wall clock: {wall_clock:.1f}s ({summary['mean_seconds_per_coin']:.2f}s/coin)")
    print(f"  Output: {run_dir}")

    return run_dir, summary


def calibrate_and_save(run_dir, cfg=None):
    """Calibrate normalization ranges, then re-score all ribbons.

    The initial run used placeholder norms, so recorded scores are not
    comparable to future runs. This function:
    1. Calibrates p5/p95 ranges from raw metrics
    2. Re-scores every ribbon with calibrated norms
    3. Rewrites ribbon_stats.csv and summary.json

    This ensures the baseline's recorded scores use the same norms
    as all subsequent candidate runs.
    """
    stats_df = pd.read_csv(run_dir / "ribbon_stats.csv")
    success = stats_df[stats_df["status"] == "success"]

    metrics_list = []
    for _, row in success.iterrows():
        metrics_list.append({
            "gradient_energy": row["gradient_energy"],
            "row_variance_mean": row["row_variance_mean"],
        })

    ranges = calibrate_norm_ranges(metrics_list)
    ranges.save(NORM_RANGES_FILE)
    print(f"Calibrated normalization ranges saved to {NORM_RANGES_FILE}")
    print(f"  gradient_energy: [{ranges.gradient_energy_lo:.6f}, {ranges.gradient_energy_hi:.6f}]")
    print(f"  row_variance:    [{ranges.row_variance_lo:.6f}, {ranges.row_variance_hi:.6f}]")

    # Re-score all ribbons with calibrated norms
    max_offset = cfg.max_center_offset if cfg else 150.0
    print("Re-scoring ribbons with calibrated norms...")
    for idx, row in stats_df.iterrows():
        if row["status"] != "success":
            continue
        metrics = {
            "nonblack_fraction": row["nonblack_fraction"],
            "nonzero_fraction": row["nonzero_fraction"],
            "mean_intensity": row["mean_intensity"],
            "std_intensity": row["std_intensity"],
            "gradient_energy": row["gradient_energy"],
            "edge_density": row["edge_density"],
            "row_variance_mean": row["row_variance_mean"],
        }
        geo_meta = {
            "used_fallback": row["used_fallback"],
            "center_offset": row["center_offset"],
            "fallback_reason": row.get("fallback_reason"),
            "radius": row.get("radius"),
        }
        qscore = compute_quality_score(metrics, geo_meta, ranges, max_offset)
        bucket = assign_quality_bucket(metrics, geo_meta, qscore)
        geo_flags = assign_geometry_flags(geo_meta)
        stats_df.at[idx, "quality_score"] = qscore
        stats_df.at[idx, "quality_bucket"] = bucket
        stats_df.at[idx, "geometry_flags"] = "|".join(geo_flags) if geo_flags else ""

    # Rewrite stats and summary
    stats_df.to_csv(run_dir / "ribbon_stats.csv", index=False)

    run_id = run_dir.name
    config_id = stats_df["config_id"].iloc[0] if len(stats_df) else "unknown"

    # Rebuild summary with a minimal cfg-like object for build_summary
    class _CfgStub:
        pass
    stub = _CfgStub()
    stub.config_id = config_id
    summary = build_summary(run_id, stub, stats_df, 0, 0, 0)
    # Preserve original timing from existing summary
    try:
        with open(run_dir / "summary.json") as f:
            old_summary = json.load(f)
        summary["wall_clock_seconds"] = old_summary.get("wall_clock_seconds", 0)
        summary["mean_seconds_per_coin"] = old_summary.get("mean_seconds_per_coin", 0)
        summary["cache_hit_rate"] = old_summary.get("cache_hit_rate", 0)
    except Exception:
        pass

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Re-scored: mean_quality_score={summary['mean_quality_score']:.4f}  "
          f"good={summary['n_good']}  borderline={summary['n_borderline']}")

    return ranges


def main():
    parser = argparse.ArgumentParser(description="Run one unwrap experiment")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("--subset", required=True, help="Eval subset CSV")
    parser.add_argument("--baseline", default=None,
                        help="Baseline run directory (for comparison galleries)")
    parser.add_argument("--calibrate", action="store_true",
                        help="After run, calibrate norm ranges from this run's metrics")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--previews", action="store_true",
                        help="Save ribbon preview PNGs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {cfg.config_id}")

    run_dir, summary = run_experiment(
        cfg,
        args.subset,
        baseline_dir=args.baseline,
        workers=args.workers,
        save_previews=args.previews,
    )

    if args.calibrate:
        calibrate_and_save(run_dir, cfg=cfg)

    # Generate galleries
    try:
        from legend_cnn.make_galleries import generate_all_galleries
        galleries_dir = run_dir / "galleries"
        galleries_dir.mkdir(parents=True, exist_ok=True)
        stats_df = pd.read_csv(run_dir / "ribbon_stats.csv")
        generate_all_galleries(stats_df, galleries_dir, baseline_dir=args.baseline)
    except Exception as e:
        print(f"WARNING: Gallery generation failed: {e}")


if __name__ == "__main__":
    main()
