#!/usr/bin/env python3
"""Run a controlled parameter sweep across a grid of unwrap configs.

    python -m legend_cnn.sweep_experiments \
        --subset data/eval_subset_v1.csv \
        [--baseline runs/baseline_eval_v1] \
        [--workers 8]

Generates one config per grid cell, runs each experiment, and produces
a ranked summary across all runs.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

from legend_cnn.config import CONFIGS_DIR, RUNS_DIR
from legend_cnn.run_experiment import load_config, run_experiment
from legend_cnn.score_ribbons import NormRanges
from legend_cnn.unwrap_core import UnwrapConfig


# ── Default v1 Sweep Grid ────────────────────────────────────────────

DEFAULT_GRID = {
    "inner_r_ratio": [0.68, 0.72, 0.76],
    "outer_r_ratio": [0.88, 0.90, 0.92],
    "center_method": ["bbox"],
}


def generate_sweep_configs(grid=None, base_config=None):
    """Generate UnwrapConfig instances from a parameter grid.

    Args:
        grid: dict of param_name -> list of values to sweep
        base_config: UnwrapConfig with defaults for non-swept params

    Returns:
        list of UnwrapConfig
    """
    if grid is None:
        grid = DEFAULT_GRID
    if base_config is None:
        base_config = UnwrapConfig()

    param_names = sorted(grid.keys())
    param_values = [grid[k] for k in param_names]

    configs = []
    for combo in itertools.product(*param_values):
        overrides = dict(zip(param_names, combo))

        # Build config_id from varied params
        id_parts = []
        for k, v in sorted(overrides.items()):
            if isinstance(v, float):
                id_parts.append(f"{k[:3]}{v:.2f}".replace(".", ""))
            else:
                id_parts.append(f"{k[:3]}_{v}")
        config_id = "sweep_" + "_".join(id_parts)

        d = base_config.to_dict()
        d.update(overrides)
        d["config_id"] = config_id
        configs.append(UnwrapConfig.from_dict(d))

    return configs


def save_sweep_configs(configs):
    """Save each config as a JSON file under configs/."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in configs:
        path = CONFIGS_DIR / f"{cfg.config_id}.json"
        with open(path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        paths.append(path)
    return paths


def run_sweep(configs, subset_path, baseline_dir=None, workers=8):
    """Run all configs in sequence, collecting summaries.

    Returns list of (run_dir, summary) tuples.
    """
    results = []
    n = len(configs)

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"Sweep run {i}/{n}: {cfg.config_id}")
        print(f"{'=' * 60}")

        run_dir, summary = run_experiment(
            cfg, subset_path,
            baseline_dir=baseline_dir,
            workers=workers,
        )

        # Generate galleries
        try:
            from legend_cnn.make_galleries import generate_all_galleries
            import pandas as pd
            stats_df = pd.read_csv(run_dir / "ribbon_stats.csv")
            galleries_dir = run_dir / "galleries"
            generate_all_galleries(stats_df, galleries_dir, baseline_dir=baseline_dir)
        except Exception as e:
            print(f"WARNING: Gallery generation failed: {e}")

        results.append((run_dir, summary))

    return results


def print_sweep_summary(results):
    """Print a ranked summary table of all sweep runs."""
    print(f"\n{'=' * 80}")
    print("SWEEP SUMMARY — Ranked by good_rate")
    print(f"{'=' * 80}")

    # Primary ranking: mean_quality_score (visual quality), then fallback_rate (process reliability)
    ranked = sorted(results, key=lambda x: (-x[1].get("mean_quality_score", 0), x[1].get("fallback_rate", 1)))

    header = f"{'Config':<35s} {'MeanQ':>6s} {'Good%':>6s} {'Blank%':>7s} {'FB%':>5s} {'GeoFlg%':>7s} {'Time':>6s}"
    print(header)
    print("-" * len(header))

    for run_dir, s in ranked:
        n = max(s["n_total"], 1)
        good_pct = s["n_good"] / n * 100
        blank_pct = s["n_bad_blank"] / n * 100
        fb_pct = s["fallback_rate"] * 100
        geo_pct = s.get("n_with_geometry_flags", 0) / n * 100

        print(f"{s['config_id']:<35s} {s['mean_quality_score']:>6.3f} {good_pct:>5.1f}% "
              f"{blank_pct:>6.1f}% {fb_pct:>4.1f}% {geo_pct:>6.1f}% "
              f"{s['wall_clock_seconds']:>5.0f}s")

    # Save sweep summary
    summary_path = RUNS_DIR / "sweep_summary.json"
    sweep_data = [
        {"run_dir": str(rd), "summary": s}
        for rd, s in ranked
    ]
    with open(summary_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"\nSweep summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep")
    parser.add_argument("--subset", required=True, help="Eval subset CSV")
    parser.add_argument("--baseline", default=None, help="Baseline run directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grid", default=None,
                        help="JSON file with parameter grid (default: 3x3 radial sweep)")
    parser.add_argument("--base-config", default=None,
                        help="JSON config for non-swept parameters")
    args = parser.parse_args()

    # Load grid
    if args.grid:
        with open(args.grid) as f:
            grid = json.load(f)
    else:
        grid = DEFAULT_GRID

    # Load base config
    base_cfg = None
    if args.base_config:
        base_cfg = load_config(args.base_config)

    configs = generate_sweep_configs(grid, base_cfg)
    print(f"Generated {len(configs)} sweep configs:")
    for cfg in configs:
        print(f"  {cfg.config_id}")

    save_sweep_configs(configs)

    results = run_sweep(configs, args.subset, args.baseline, args.workers)
    print_sweep_summary(results)


if __name__ == "__main__":
    main()
