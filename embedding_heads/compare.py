#!/usr/bin/env python3
"""Compare all trained heads and produce summary tables.

    python -m embedding_heads.compare
"""

import json
from pathlib import Path

from embedding_heads.config import CHECKPOINT_DIR, OUTPUT_DIR
from embedding_heads.evaluate import evaluate


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    heads = []
    for name in ["linear", "mlp", "arcface", "cosface"]:
        ck_path = CHECKPOINT_DIR / f"{name}_best.pt"
        if ck_path.exists():
            heads.append(name)
        else:
            print(f"Skipping {name} (no checkpoint)")

    all_results = []
    for head in heads:
        results = evaluate(head)
        all_results.append(results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    header = f"{'Head':<10} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7}  {'Worst Pair':<30} {'Err':>4}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['head']:<10} {r['top1']:>7.1%} {r['top3']:>7.1%} {r['top5']:>7.1%}  "
              f"{r['worst_pair']:<30} {r['worst_pair_count']:>4}")

    # ArcFace embedding table
    margin_heads = [r for r in all_results if "projected_knn1" in r]
    if margin_heads:
        print(f"\n{'Embedding Space Metrics':<30} {'Raw 768-d':>10} {'Projected':>10}")
        print("-" * 52)
        for r in margin_heads:
            print(f"{r['head'] + ' KNN-1':<30} {r['raw_knn1']:>10.1%} {r['projected_knn1']:>10.1%}")
            print(f"{r['head'] + ' KNN-10':<30} {r['raw_knn10']:>10.1%} {r['projected_knn10']:>10.1%}")

    # Save combined
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
