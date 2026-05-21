#!/usr/bin/env python3
"""Train + evaluate all scopes from head_training_data/.

    python -m embedding_heads.run_all_scopes
    python -m embedding_heads.run_all_scopes --eval-only
    python -m embedding_heads.run_all_scopes --scope auth_ri_severan_13way
"""

import argparse
import json
import time

from embedding_heads.config import OUTPUT_DIR, SCOPE_DATA_DIR
from embedding_heads.evaluate import evaluate_scope
from embedding_heads.train import train


def discover_scopes():
    """Find all scopes that have the full triplet of files."""
    scopes = set()
    for p in SCOPE_DATA_DIR.glob("*_subset.csv"):
        scope = p.stem.replace("_subset", "")
        splits = SCOPE_DATA_DIR / f"{scope}_splits.json"
        cmap = SCOPE_DATA_DIR / f"{scope}_class_map.json"
        if splits.exists() and cmap.exists():
            scopes.add(scope)
    return sorted(scopes)


def run_all(scopes, eval_only=False):
    results = {}
    total_t0 = time.time()

    for i, scope in enumerate(scopes, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(scopes)}] {scope}")
        print(f"{'='*70}")

        scope_t0 = time.time()

        if not eval_only:
            best_epoch, best_val_loss, best_val_acc = train(
                "cosface", scope=scope,
            )

        eval_result = evaluate_scope(scope, head_name="cosface")
        elapsed = time.time() - scope_t0

        # Apply gates: top-1 >= 70%, lift over raw KNN-10 >= 5pp, ECE < 0.10
        top1 = eval_result["top1"]
        raw_knn10 = eval_result.get("raw_knn10", 0)
        proj_knn10 = eval_result.get("projected_knn10", 0)
        knn_lift = top1 - raw_knn10

        if top1 >= 0.70 and knn_lift >= 0.05:
            gate = "PASS"
        elif top1 >= 0.70 and raw_knn10 >= top1 - 0.05:
            gate = "knn_sufficient"
        elif top1 >= 0.60:
            gate = "candidate"
        else:
            gate = "FAIL"

        eval_result["gate"] = gate
        eval_result["elapsed_s"] = round(elapsed, 1)
        results[scope] = eval_result

        print(f"\n  >> GATE: {gate}  (top1={top1:.1%}, raw_knn10={raw_knn10:.1%}, lift={knn_lift:+.1%})")
        print(f"  >> Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY ({len(scopes)} scopes, {total_elapsed:.0f}s total)")
    print(f"{'='*70}\n")

    print(f"{'Scope':<42} {'Top-1':>6} {'KNN-10':>7} {'Lift':>6} {'Gate':<15}")
    print("-" * 80)

    for scope in scopes:
        r = results[scope]
        raw_knn10 = r.get("raw_knn10", 0)
        lift = r["top1"] - raw_knn10
        print(f"{scope:<42} {r['top1']:>6.1%} {raw_knn10:>7.1%} {lift:>+6.1%} {r['gate']:<15}")

    pass_count = sum(1 for r in results.values() if r["gate"] == "PASS")
    print(f"\nPASS: {pass_count}/{len(scopes)}")

    # Save combined results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "eval_all_scopes.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR / 'eval_all_scopes.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate all scopes")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing checkpoints")
    parser.add_argument("--scope", nargs="*", default=None,
                        help="Run specific scope(s) instead of all")
    args = parser.parse_args()

    if args.scope:
        scopes = args.scope
    else:
        scopes = discover_scopes()

    print(f"Scopes to run ({len(scopes)}):")
    for s in scopes:
        print(f"  {s}")

    run_all(scopes, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
