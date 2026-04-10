#!/usr/bin/env python3
"""Compare candidate runs against a baseline and emit a ranked report.

    python -m legend_cnn.compare_runs \
        --baseline runs/baseline_eval_v1 \
        --candidates runs/run_20260410_* \
        [--top 3]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_summary(run_dir):
    """Load summary.json from a run directory."""
    path = Path(run_dir) / "summary.json"
    with open(path) as f:
        return json.load(f)


def compute_deltas(baseline, candidate):
    """Compute improvement deltas between baseline and candidate."""
    b_n = max(baseline["n_total"], 1)
    c_n = max(candidate["n_total"], 1)

    b_good_rate = baseline["n_good"] / b_n
    c_good_rate = candidate["n_good"] / c_n
    b_blank_rate = baseline["n_bad_blank"] / b_n
    c_blank_rate = candidate["n_bad_blank"] / c_n

    good_rate_relative = (c_good_rate - b_good_rate) / max(b_good_rate, 0.001)
    blank_rate_relative = (c_blank_rate - b_blank_rate) / max(b_blank_rate, 0.001)

    return {
        "good_rate_baseline": round(b_good_rate, 4),
        "good_rate_candidate": round(c_good_rate, 4),
        "good_rate_delta": round(c_good_rate - b_good_rate, 4),
        "good_rate_relative": round(good_rate_relative, 4),
        "blank_rate_baseline": round(b_blank_rate, 4),
        "blank_rate_candidate": round(c_blank_rate, 4),
        "blank_rate_delta": round(c_blank_rate - b_blank_rate, 4),
        "blank_rate_relative": round(blank_rate_relative, 4),
        "fallback_rate_baseline": baseline["fallback_rate"],
        "fallback_rate_candidate": candidate["fallback_rate"],
        "fallback_rate_delta": round(candidate["fallback_rate"] - baseline["fallback_rate"], 4),
        "mean_score_baseline": baseline["mean_quality_score"],
        "mean_score_candidate": candidate["mean_quality_score"],
        "mean_score_delta": round(
            candidate["mean_quality_score"] - baseline["mean_quality_score"], 4
        ),
    }


def check_acceptance(deltas):
    """Check if candidate meets heuristic acceptance thresholds.

    Thresholds are heuristic — acceptance decisions must be validated
    by paired visual review via the vs_baseline.png gallery.
    """
    checks = {
        "good_rate_improved_10pct": deltas["good_rate_relative"] >= 0.10,
        "blank_rate_reduced_20pct": deltas["blank_rate_relative"] <= -0.20,
        "fallback_rate_not_worse": deltas["fallback_rate_delta"] <= 0.02,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def check_authority_regressions(baseline, candidate, min_n=10):
    """Check for authority-level regressions.

    Only flags authorities with at least min_n coins to avoid
    noisy false alarms from small samples.
    """
    b_auth = baseline.get("per_authority_stats", {})
    c_auth = candidate.get("per_authority_stats", {})

    regressions = []
    for auth in b_auth:
        if auth not in c_auth:
            continue
        b_n = b_auth[auth].get("n", 0)
        if b_n < min_n:
            continue
        b_good = b_auth[auth].get("good_rate", 0)
        c_good = c_auth[auth].get("good_rate", 0)
        if b_good > 0 and c_good < b_good * 0.5:
            regressions.append({
                "authority": auth,
                "n": b_n,
                "baseline_good_rate": b_good,
                "candidate_good_rate": c_good,
                "drop": round(b_good - c_good, 4),
            })

    return regressions


def compare_runs(baseline_dir, candidate_dirs, top_n=3):
    """Compare candidates to baseline and print ranked report."""
    baseline = load_summary(baseline_dir)
    print(f"Baseline: {baseline['config_id']}  "
          f"(good={baseline['n_good']}/{baseline['n_total']}, "
          f"score={baseline['mean_quality_score']:.3f})")

    comparisons = []
    for cdir in candidate_dirs:
        cdir = Path(cdir)
        try:
            candidate = load_summary(cdir)
        except Exception as e:
            print(f"  WARNING: Could not load {cdir}: {e}")
            continue

        deltas = compute_deltas(baseline, candidate)
        checks = check_acceptance(deltas)
        regressions = check_authority_regressions(baseline, candidate)

        comparisons.append({
            "run_dir": str(cdir),
            "config_id": candidate["config_id"],
            "summary": candidate,
            "deltas": deltas,
            "checks": checks,
            "authority_regressions": regressions,
        })

    # Rank by good_rate improvement
    comparisons.sort(key=lambda x: -x["deltas"]["good_rate_delta"])

    # Print report
    print(f"\n{'=' * 80}")
    print(f"COMPARISON REPORT — {len(comparisons)} candidates vs baseline")
    print(f"{'=' * 80}")

    for i, comp in enumerate(comparisons, 1):
        d = comp["deltas"]
        c = comp["checks"]
        verdict = "ACCEPT" if c["all_passed"] else "REVIEW"

        print(f"\n#{i}  {comp['config_id']}  [{verdict}]")
        print(f"  Good rate:     {d['good_rate_baseline']:.1%} -> {d['good_rate_candidate']:.1%}  "
              f"({d['good_rate_relative']:+.1%} relative)")
        print(f"  Blank rate:    {d['blank_rate_baseline']:.1%} -> {d['blank_rate_candidate']:.1%}  "
              f"({d['blank_rate_relative']:+.1%} relative)")
        print(f"  Fallback rate: {d['fallback_rate_baseline']:.1%} -> {d['fallback_rate_candidate']:.1%}  "
              f"(delta {d['fallback_rate_delta']:+.3f})")
        print(f"  Mean score:    {d['mean_score_baseline']:.3f} -> {d['mean_score_candidate']:.3f}  "
              f"(delta {d['mean_score_delta']:+.3f})")

        if comp["authority_regressions"]:
            print(f"  AUTHORITY REGRESSIONS:")
            for r in comp["authority_regressions"]:
                print(f"    {r['authority']}: {r['baseline_good_rate']:.1%} -> "
                      f"{r['candidate_good_rate']:.1%} (drop {r['drop']:.1%})")

        if not c["all_passed"]:
            fails = [k for k, v in c.items() if k != "all_passed" and not v]
            print(f"  Failed checks: {', '.join(fails)}")

    # Recommend top candidates for visual review
    print(f"\n{'=' * 80}")
    print(f"TOP {min(top_n, len(comparisons))} FOR VISUAL REVIEW:")
    for comp in comparisons[:top_n]:
        print(f"  {comp['config_id']}  (run: {comp['run_dir']})")
    print(f"\nReview vs_baseline.png galleries before promoting any candidate.")

    # Save report
    report = {
        "baseline": {"dir": str(baseline_dir), "summary": baseline},
        "comparisons": comparisons,
    }
    report_path = Path(baseline_dir).parent / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Compare runs to baseline")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="Candidate run directories")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top candidates to recommend for review")
    args = parser.parse_args()

    compare_runs(args.baseline, args.candidates, args.top)


if __name__ == "__main__":
    main()
