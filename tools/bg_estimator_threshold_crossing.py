#!/usr/bin/env python
"""Which sides can M1 actually change? -- estimator-only scan, no segmentation.

`detect_background_histogram` has exactly ONE production consumer:

    src/layer1_geometry.py:592   avg_bg, _ = detect_background_histogram(gray)
    src/layer1_geometry.py:600   thresh_type = INV if avg_bg > 110 else BINARY

`bg_type` is discarded at the call site, and the value is consumed by a single
binary comparison against `Layer1Config.BRIGHT_BACKGROUND_THRESHOLD = 110`.

So M1 can only change Layer-1 behavior on a side where the returned value
CROSSES 110.  A side whose estimate moves 31.2 -> 75.0 is more accurate but
behaviorally INERT: both values are below 110, both select the same
`thresh_type`, and every downstream pixel is identical.

This script computes the OFF and ON values for every side and reports the
crossing set -- cheap (four patch reads plus a histogram per side, no Otsu, no
contours, no Hough), so it can run alongside a heavy job without disturbing it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import cv2

VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIGHT_BACKGROUND_THRESHOLD = 110


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=f"{VISION_ROOT}/specs/results/rim_stall_taxonomy_ks17_scan.json")
    ap.add_argument("--images", default="/home/claudeuser/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["half", "full"], default="half",
                    help="half = split into obv/rev sides (what the taxonomy and "
                         "the serving lane measure); full = the WHOLE photo, "
                         "which is what the ingest lane's analyze_image "
                         "actually receives. They are different geometries and "
                         "must not be derived from one another.")
    a = ap.parse_args()

    cv2.setNumThreads(1)
    sys.path.insert(0, VISION_ROOT)
    sys.path.insert(0, os.path.join(VISION_ROOT, "tools"))
    from rim_stall_taxonomy import _load_and_resize, split_sides, backdrop_ring_mean
    import src.math_utils as mu

    scan = json.load(open(a.scan))
    by_img = {}
    for r in scan:
        by_img.setdefault(r["id"], []).append(r["side"])

    rows = []
    for i, (img_id, sides) in enumerate(sorted(by_img.items()), 1):
        img = _load_and_resize(os.path.join(a.images, f"{img_id}.jpg"))
        if img is None:
            continue
        if a.layout == "full":
            parts = {"full": img}
            sides = ["full"]
        else:
            parts = split_sides(img)
        for side in sides:
            gray = cv2.cvtColor(parts[side], cv2.COLOR_BGR2GRAY)
            os.environ.pop(mu.BG_CORNER_LOCAL_TRUST_ENV, None)
            off_v, off_t = mu.detect_background_histogram(gray)
            os.environ[mu.BG_CORNER_LOCAL_TRUST_ENV] = "1"
            on_v, on_t = mu.detect_background_histogram(gray)
            os.environ.pop(mu.BG_CORNER_LOCAL_TRUST_ENV, None)
            truth = backdrop_ring_mean(gray)
            off_pol = off_v > BRIGHT_BACKGROUND_THRESHOLD
            on_pol = on_v > BRIGHT_BACKGROUND_THRESHOLD
            rows.append({
                "id": img_id, "side": side,
                "off_value": round(float(off_v), 3), "off_bg_type": off_t,
                "on_value": round(float(on_v), 3), "on_bg_type": on_t,
                "ring_truth": round(truth, 3),
                "off_err": round(float(off_v) - truth, 3),
                "on_err": round(float(on_v) - truth, 3),
                "off_thresh": "INV" if off_pol else "BINARY",
                "on_thresh": "INV" if on_pol else "BINARY",
                "crosses_110": off_pol != on_pol,
                "value_changed": float(off_v) != float(on_v),
            })
        if i % 50 == 0:
            print(f"  [{i}/{len(by_img)}] images", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    vc = sum(1 for r in rows if r["value_changed"])
    cr = [r for r in rows if r["crosses_110"]]
    print(f"\nsides: {n}")
    print(f"  estimator VALUE changed by M1 : {vc:4d} ({100.0*vc/n:.1f}%)")
    print(f"  CROSSES the 110 threshold     : {len(cr):4d} ({100.0*len(cr)/n:.1f}%)"
          f"   <- the only sides Layer 1 can see")
    print(f"  behaviorally INERT changes    : {vc-len(cr):4d} "
          f"({100.0*(vc-len(cr))/n:.1f}%)")
    print(f"\n  crossing directions: "
          f"{Counter((r['off_thresh'], r['on_thresh']) for r in cr)}")
    off_ok = sum(1 for r in rows if abs(r["off_err"]) <= 8)
    on_ok = sum(1 for r in rows if abs(r["on_err"]) <= 8)
    print(f"\n  Bar 1 |err|<=8 : OFF {off_ok}/{n} ({100.0*off_ok/n:.1f}%)  "
          f"-> ON {on_ok}/{n} ({100.0*on_ok/n:.1f}%)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
