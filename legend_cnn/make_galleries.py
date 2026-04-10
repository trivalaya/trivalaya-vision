#!/usr/bin/env python3
"""Generate ranked review galleries from a completed experiment run.

    python -m legend_cnn.make_galleries --run-dir runs/run_YYYYMMDD_HHMMSS_config_id
    python -m legend_cnn.make_galleries --run-dir runs/run_... --baseline runs/baseline_eval_v1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_ribbon(path):
    """Load a ribbon .npy file, return ndarray or None."""
    try:
        return np.load(path)
    except Exception:
        return None


def _render_gallery(ribbons_with_labels, title, out_path, max_rows=20):
    """Render a list of (ribbon, label) pairs into a gallery PNG."""
    items = [(r, l) for r, l in ribbons_with_labels if r is not None][:max_rows]
    if not items:
        return

    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(14, n * 0.7 + 1), dpi=100)
    if n == 1:
        axes = [axes]

    for ax, (ribbon, label) in zip(axes, items):
        ax.imshow(ribbon, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _render_comparison_gallery(pairs, title, out_path, max_rows=20):
    """Render side-by-side comparison: baseline vs candidate.

    pairs: list of (ribbon_baseline, ribbon_candidate, label)
    """
    items = [(rb, rc, l) for rb, rc, l in pairs if rb is not None and rc is not None][:max_rows]
    if not items:
        return

    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 0.7 + 1), dpi=100)
    if n == 1:
        axes = [axes]

    for i, (rb, rc, label) in enumerate(items):
        ax_b = axes[i][0] if n > 1 else axes[0]
        ax_c = axes[i][1] if n > 1 else axes[1]

        ax_b.imshow(rb, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax_b.set_xticks([])
        ax_b.set_yticks([])

        ax_c.imshow(rc, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax_c.set_xticks([])
        ax_c.set_yticks([])

        ax_b.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")

        if i == 0:
            ax_b.set_title("Baseline", fontsize=9)
            ax_c.set_title("Candidate", fontsize=9)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def gallery_best(stats_df, out_path, n=20):
    """Top N ribbons by quality score."""
    df = stats_df[stats_df["status"] == "success"].nlargest(n, "quality_score")
    items = []
    for _, row in df.iterrows():
        r = _load_ribbon(row["ribbon_path"])
        label = f"{row['coin_id']}  q={row['quality_score']:.3f}  {row['authority']}"
        items.append((r, label))
    _render_gallery(items, f"Best {n} by Quality Score", out_path, max_rows=n)


def gallery_worst(stats_df, out_path, n=20):
    """Bottom N ribbons by quality score."""
    success = stats_df[stats_df["status"] == "success"].dropna(subset=["quality_score"])
    df = success.nsmallest(n, "quality_score")
    items = []
    for _, row in df.iterrows():
        r = _load_ribbon(row["ribbon_path"])
        label = f"{row['coin_id']}  q={row['quality_score']:.3f}  [{row['quality_bucket']}]"
        items.append((r, label))
    _render_gallery(items, f"Worst {n} by Quality Score", out_path, max_rows=n)


def gallery_random(stats_df, out_path, n=20, seed=42):
    """Random sample of N ribbons."""
    success = stats_df[stats_df["status"] == "success"]
    df = success.sample(n=min(n, len(success)), random_state=seed)
    items = []
    for _, row in df.iterrows():
        r = _load_ribbon(row["ribbon_path"])
        label = f"{row['coin_id']}  q={row['quality_score']:.3f}  {row['authority']}"
        items.append((r, label))
    _render_gallery(items, f"Random Sample (n={len(df)})", out_path, max_rows=n)


def gallery_by_authority(stats_df, out_path, per_auth=5, seed=42):
    """Sample per authority."""
    success = stats_df[stats_df["status"] == "success"]
    items = []
    for auth in sorted(success["authority"].unique()):
        auth_df = success[success["authority"] == auth]
        sample = auth_df.sample(n=min(per_auth, len(auth_df)), random_state=seed)
        for _, row in sample.iterrows():
            r = _load_ribbon(row["ribbon_path"])
            label = f"{auth}  {row['coin_id']}  q={row['quality_score']:.3f}"
            items.append((r, label))
    _render_gallery(items, "By Authority", out_path, max_rows=len(items))


def gallery_fallbacks(stats_df, out_path, n=30):
    """All or sampled fallback cases."""
    fb = stats_df[stats_df["used_fallback"] == True]
    fb_success = fb[fb["status"] == "success"]
    if len(fb_success) > n:
        fb_success = fb_success.sample(n=n, random_state=42)
    items = []
    for _, row in fb_success.iterrows():
        r = _load_ribbon(row["ribbon_path"])
        label = f"{row['coin_id']}  {row['fallback_reason']}  q={row.get('quality_score', '?')}"
        items.append((r, label))
    title = f"Fallback Cases ({len(fb)} total, showing {len(items)})"
    _render_gallery(items, title, out_path, max_rows=len(items))


def gallery_hard_cases(stats_df, out_path):
    """All coins marked known_hard_case."""
    hard = stats_df[stats_df.get("known_hard_case", False) == True]
    if hard.empty:
        return
    items = []
    for _, row in hard.iterrows():
        if row["status"] == "success" and row.get("ribbon_path"):
            r = _load_ribbon(row["ribbon_path"])
            label = f"{row['coin_id']}  q={row.get('quality_score', '?')}  [{row.get('quality_bucket', '?')}]"
        else:
            r = None
            label = f"{row['coin_id']}  FAILED: {row.get('failure_reason', '?')}"
        items.append((r, label))
    _render_gallery(items, f"Hard Cases ({len(hard)})", out_path, max_rows=len(items))


def gallery_vs_baseline(stats_df, galleries_dir, baseline_dir, n=10):
    """Side-by-side comparison: top N improved + top N regressed vs baseline."""
    baseline_dir = Path(baseline_dir)
    baseline_stats = pd.read_csv(baseline_dir / "ribbon_stats.csv")

    # Merge on coin_id
    merged = stats_df.merge(
        baseline_stats[["coin_id", "quality_score", "ribbon_path"]],
        on="coin_id",
        suffixes=("_cand", "_base"),
    )
    merged = merged.dropna(subset=["quality_score_cand", "quality_score_base"])
    merged["score_delta"] = merged["quality_score_cand"] - merged["quality_score_base"]

    # Top improvements
    improved = merged.nlargest(n, "score_delta")
    # Top regressions
    regressed = merged.nsmallest(n, "score_delta")

    pairs = []
    for _, row in improved.iterrows():
        rb = _load_ribbon(row["ribbon_path_base"])
        rc = _load_ribbon(row["ribbon_path_cand"])
        delta = row["score_delta"]
        label = f"{row['coin_id']}  +{delta:.3f}"
        pairs.append((rb, rc, label))

    for _, row in regressed.iterrows():
        rb = _load_ribbon(row["ribbon_path_base"])
        rc = _load_ribbon(row["ribbon_path_cand"])
        delta = row["score_delta"]
        label = f"{row['coin_id']}  {delta:.3f}"
        pairs.append((rb, rc, label))

    out_path = galleries_dir / "vs_baseline.png"
    _render_comparison_gallery(
        pairs,
        f"Baseline vs Candidate (top {n} improved / top {n} regressed)",
        out_path,
        max_rows=len(pairs),
    )


def generate_all_galleries(stats_df, galleries_dir, baseline_dir=None):
    """Generate all standard galleries for a run."""
    galleries_dir = Path(galleries_dir)
    galleries_dir.mkdir(parents=True, exist_ok=True)

    print("Generating galleries...")
    gallery_best(stats_df, galleries_dir / "best.png")
    gallery_worst(stats_df, galleries_dir / "worst.png")
    gallery_random(stats_df, galleries_dir / "random.png")
    gallery_by_authority(stats_df, galleries_dir / "by_authority.png")
    gallery_fallbacks(stats_df, galleries_dir / "fallbacks.png")
    gallery_hard_cases(stats_df, galleries_dir / "hard_cases.png")

    if baseline_dir:
        gallery_vs_baseline(stats_df, galleries_dir, baseline_dir)

    print(f"Galleries saved to {galleries_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate review galleries")
    parser.add_argument("--run-dir", required=True, help="Completed run directory")
    parser.add_argument("--baseline", default=None, help="Baseline run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stats_df = pd.read_csv(run_dir / "ribbon_stats.csv")
    galleries_dir = run_dir / "galleries"
    generate_all_galleries(stats_df, galleries_dir, baseline_dir=args.baseline)


if __name__ == "__main__":
    main()
