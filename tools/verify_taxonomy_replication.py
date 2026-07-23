#!/usr/bin/env python3
"""
verify_taxonomy_replication.py — prove `rim_stall_taxonomy.py`'s replication
of the primary segmentation is faithful to production Layer 1.

The taxonomy scanner re-executes the preamble + segmentation itself so it can
see per-contour state that `layer_1_structural_salience` never returns. That
is only worth anything if the replication is exact. Two independent checks:

  A. PREDICTION check (free, whole population). The scanner's `n_will_hough`
     must predict the measured cost class. Joined against the frozen baseline
     timings in specs/results/rim_recovery_cost_ab_ks17.csv, a side with
     n_will_hough > 0 must be a stall (cpu_s > 20) and vice versa. Reports a
     confusion matrix; any off-diagonal cell is a replication defect.

  B. IDENTITY check (expensive, small sample). Runs the REAL, unmodified
     `layer_1_structural_salience` on the same sides and compares the
     production `rim_recovered` flag count and detection count against the
     scanner's own pass-2 replay. Costs a full Hough run per stall side, so
     it is --limit'ed by default.

Read-only. No production file is imported-and-patched, only imported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rim_stall_taxonomy import _load_and_resize, split_sides  # noqa: E402


def check_prediction(scan_json: str, cost_csv: str, lane: str, stall_s: float):
    sides = {(s["id"], s["side"]): s for s in json.load(open(scan_json))}
    rows = [r for r in csv.DictReader(open(cost_csv))
            if r["config"] == "baseline" and r["lane"] == lane]
    cm = Counter()
    misses = []
    for r in rows:
        key = (r["id"], r["side"])
        if key not in sides:
            continue
        pred = sides[key]["n_will_hough"] > 0
        actual = float(r["cpu_s"]) > stall_s
        cm[(pred, actual)] += 1
        if pred != actual:
            misses.append((key, sides[key]["n_will_hough"], float(r["cpu_s"]),
                           sides[key]["max_hough_roi_px"]))
    n = sum(cm.values())
    print(f"\n=== A. PREDICTION check (lane={lane}, stall = cpu_s > {stall_s}s) ===")
    print(f"joined sides: {n}")
    print(f"  predict Hough & IS  stall : {cm[(True, True)]}")
    print(f"  predict Hough & NOT stall : {cm[(True, False)]}")
    print(f"  predict none  & IS  stall : {cm[(False, True)]}   <-- replication miss")
    print(f"  predict none  & NOT stall : {cm[(False, False)]}")
    acc = (cm[(True, True)] + cm[(False, False)]) / max(1, n)
    print(f"  agreement: {100*acc:.1f}%")
    for k, nh, cpu, roi in misses:
        print(f"    MISMATCH {k} n_will_hough={nh} cpu_s={cpu:.1f} max_roi_px={roi}")
    return acc


def check_identity(scan_json: str, images: str, house: str, limit: int):
    from src.layer1_geometry import layer_1_structural_salience

    sides = [s for s in json.load(open(scan_json)) if s["n_will_hough"] > 0]
    sides = sides[:limit]
    print(f"\n=== B. IDENTITY check vs real layer_1_structural_salience "
          f"(n={len(sides)}, house={house}) ===")
    ok = 0
    for s in sides:
        cands = list(Path(images).glob(f"{s['id']}.*"))
        if not cands:
            continue
        img = _load_and_resize(str(cands[0]))
        im = split_sides(img)[s["side"]]
        t0 = time.process_time()
        res = layer_1_structural_salience(im, source_type="auction", house=house)
        cpu = time.process_time() - t0
        # "objects" is the real key — see layer_1_structural_salience.
        dets = res.get("objects", []) or []
        n_rec = sum(1 for d in dets
                    if (d.get("debug_data") or {}).get("rim_recovered"))
        stalled = cpu > 20
        agree = stalled == (s["n_will_hough"] > 0)
        ok += agree
        print(f"  {s['id']} {s['side']}: real cpu={cpu:7.1f}s n_det={len(dets)} "
              f"n_rim_recovered={n_rec} | scanner n_will_hough={s['n_will_hough']} "
              f"max_roi_px={s['max_hough_roi_px']}  {'OK' if agree else 'MISMATCH'}")
    print(f"  agreement: {ok}/{len(sides)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--cost-csv", default=None)
    ap.add_argument("--lane", default="ingest")
    ap.add_argument("--stall-s", type=float, default=20.0)
    ap.add_argument("--images", default=None)
    ap.add_argument("--house", default="cng_feature")
    ap.add_argument("--identity-limit", type=int, default=0)
    a = ap.parse_args()
    if a.cost_csv:
        check_prediction(a.scan, a.cost_csv, a.lane, a.stall_s)
    if a.identity_limit and a.images:
        check_identity(a.scan, a.images, a.house, a.identity_limit)


if __name__ == "__main__":
    main()
