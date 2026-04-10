#!/usr/bin/env python3
"""Step 2 validation: generate a visual gallery of unwrapped legend ribbons.

Randomly samples ribbons per authority and produces a grid image for
visual inspection before proceeding to training.

    python -m legend_cnn.validate_ribbons [--per-authority 20]
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from legend_cnn.config import CONFUSION_SUBSET_CSV, DATA_DIR, RIBBON_DIR


def main():
    parser = argparse.ArgumentParser(description="Validate ribbon gallery")
    parser.add_argument("--per-authority", type=int, default=20,
                        help="Number of sample ribbons per authority")
    args = parser.parse_args()

    if not CONFUSION_SUBSET_CSV.exists():
        print(f"ERROR: {CONFUSION_SUBSET_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(CONFUSION_SUBSET_CSV)
    if "ribbon_path" not in df.columns or df["ribbon_path"].isna().all():
        print("ERROR: No ribbon_path in CSV. Run unwrap.py first.")
        sys.exit(1)

    df = df[df["ribbon_path"].notna() & (df["ribbon_path"] != "")]
    print(f"Coins with ribbons: {len(df)}")

    authorities = sorted(df["authority"].unique())
    n_auth = len(authorities)
    n_per = args.per_authority

    fig, axes = plt.subplots(
        n_auth * n_per, 1,
        figsize=(16, n_auth * n_per * 0.5),
        dpi=100,
    )
    if n_auth * n_per == 1:
        axes = [axes]

    ax_idx = 0
    stats = {"total_ribbons": 0, "radii": [], "failed_loads": 0}

    for auth in authorities:
        auth_df = df[df["authority"] == auth]
        sample = auth_df.sample(n=min(n_per, len(auth_df)), random_state=42)

        for i, (_, row) in enumerate(sample.iterrows()):
            ribbon_path = row["ribbon_path"]
            try:
                ribbon = np.load(ribbon_path)
                stats["total_ribbons"] += 1
            except Exception:
                stats["failed_loads"] += 1
                axes[ax_idx].set_visible(False)
                ax_idx += 1
                continue

            ax = axes[ax_idx]
            ax.imshow(ribbon, cmap="gray", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_ylabel(
                    f"{auth}\n({len(auth_df)})",
                    fontsize=8, rotation=0, ha="right", va="center",
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")

            ax_idx += 1

    # Hide any unused axes
    while ax_idx < len(axes):
        axes[ax_idx].set_visible(False)
        ax_idx += 1

    plt.suptitle(
        f"Legend Ribbon Gallery — {n_per} samples per authority",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()

    out_path = DATA_DIR / "ribbon_gallery.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)

    print(f"\nGallery saved to {out_path}")
    print(f"  Total ribbons rendered: {stats['total_ribbons']}")
    print(f"  Failed loads: {stats['failed_loads']}")
    print(f"  Authorities: {n_auth}")
    print(f"  Samples per authority: {n_per}")


if __name__ == "__main__":
    main()
