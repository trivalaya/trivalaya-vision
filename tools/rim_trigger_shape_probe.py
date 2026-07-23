#!/usr/bin/env python3
"""
rim_trigger_shape_probe.py — measured probe for the #1 ranked mechanism in
specs/results/rim_stall_taxonomy_2026-07-23.md.

THE CLAIM UNDER TEST
--------------------
The rim-recovery trigger's shape test is `circularity < 0.65`, with
`circularity = 4*pi*A/P^2`. P is the one contour statistic that GROWS with
resolution (the coastline effect), and the term is quadratic in it. On the
KS-17 corpus the contours that pay ~40% of the entire Hough bill are ROUND
COINS whose boundary is merely dithered: median convex-hull circularity
0.965 against median raw circularity 0.096, median radial coefficient of
variation 0.02, median largest radial dip 1 degree. The segmentation is
correct; the metric is what fails.

So: add a scale-invariant conjunct to the trigger. `cv_r` — the coefficient
of variation of the blob's radius about its centroid — answers "is this a
disc?" without ever touching perimeter, and is invariant to resolution.

  today : recover if  circ < 0.65  AND  area_ratio < 0.85
  probe : recover if  circ < 0.65  AND  area_ratio < 0.85  AND  cv_r >= CV_MIN

Adding a conjunct can only ever REMOVE recovery attempts, never add one, so
the blast radius is bounded by construction. What must be MEASURED is whether
removing them changes any final detection — that is what this probe reports.

HOW THE ARM IS APPLIED
----------------------
No production file is modified. The probe monkeypatches
`src.layer1_geometry.recover_rim` in its own process with a wrapper that
returns `(None, 0)` for blobs failing the new conjunct and delegates
otherwise. That is exactly equivalent to the trigger not firing: pass 2 of
`_segment_and_extract_candidates` keeps the seed contour and sets
`rim_recovered=False` whenever recovery yields None.

WHAT IS COMPARED (per side, against the unmodified control arm)
  - detection count
  - per-detection bbox IoU, matched greedily
  - `rim_recovered` flags
  - the alpha mask each arm would hand downstream (union of final contours),
    reported as mask IoU — the project's >= 0.995 geometry gate
  - CPU seconds

Read-only: no DB, no service, no Spaces, nothing written outside --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.layer1_geometry as L1  # noqa: E402
import src.rim_logic as RL  # noqa: E402
from tools.rim_stall_taxonomy import (  # noqa: E402
    _load_and_resize,
    radial_profile,
    split_sides,
)

CV_MIN_DEFAULT = 0.06


def _is_disc(seed_contour, shape, cv_min, area_ratio_floor):
    """Scale-invariant 'this blob is a disc' test: every ray reaches nearly the
    same radius AND the blob fills its enclosing circle. A genuinely bitten
    coin fails the second test, so the case rim recovery exists to serve is
    never caught here."""
    rp = radial_profile(seed_contour, shape)
    if rp is None:
        return False
    area = cv2.contourArea(seed_contour)
    (_, _), enc_r = cv2.minEnclosingCircle(seed_contour)
    ar = area / (np.pi * enc_r * enc_r) if enc_r > 0 else 1.0
    return rp["cv_r"] < cv_min and ar >= area_ratio_floor


def make_arm(arm: str, cv_min: float, area_ratio_floor: float):
    """
    Two candidate interventions, both gated on the same disc test.

    `recover_skip` — suppress rim recovery entirely for disc-shaped blobs.
        Patches `layer1_geometry.recover_rim`; equivalent to the trigger not
        firing (pass 2 keeps the seed contour, rim_recovered=False).

    `hough_skip` — keep the geometric fit, suppress ONLY the Hough branch.
        Patches `rim_logic.hough_rim_recovery`, so `recover_rim` falls through
        to `return geo_c, geo_conf`. This is the arm the yield data points at:
        on the expensive tier `recover_rim` already returns the GEOMETRIC fit
        17 times out of 21, and that fit costs ~20 ms against Hough's 40-200 s.
        Wherever geo would have won anyway the outcome is identical by
        construction, not by measurement.
    """
    stats = {"calls": 0, "skipped": 0}

    if arm == "recover_skip":
        real = L1.recover_rim

        def wrapped(image_bgr, seed_contour):
            stats["calls"] += 1
            if _is_disc(seed_contour, image_bgr.shape[:2], cv_min, area_ratio_floor):
                stats["skipped"] += 1
                return None, 0
            return real(image_bgr, seed_contour)

        return ("L1.recover_rim", wrapped), stats

    real_h = RL.hough_rim_recovery

    def wrapped_h(image_bgr, seed_contour):
        stats["calls"] += 1
        if _is_disc(seed_contour, image_bgr.shape[:2], cv_min, area_ratio_floor):
            stats["skipped"] += 1
            return None, 0
        return real_h(image_bgr, seed_contour)

    return ("RL.hough_rim_recovery", wrapped_h), stats


def _dets(res):
    # layer_1_structural_salience returns its candidate list under "objects"
    # (both the normal path and the two-coin-resolution path). It is NOT
    # "detections"/"candidates" — reading a missing key here would silently
    # compare [] against [] and report every side as unchanged.
    return res.get("objects", []) or []


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def _alpha(res, shape):
    m = np.zeros(shape, np.uint8)
    for d in _dets(res):
        c = d.get("contour")
        if c is not None:
            cv2.drawContours(m, [np.asarray(c)], -1, 255, -1)
    return m


def run_side(img, house, arm, cv_min, area_ratio_floor):
    saved = (L1.recover_rim, RL.hough_rim_recovery)
    try:
        t0 = time.process_time()
        ctl = L1.layer_1_structural_salience(img, source_type="auction", house=house)
        ctl_cpu = time.process_time() - t0

        (target, wrapped), stats = make_arm(arm, cv_min, area_ratio_floor)
        if target == "L1.recover_rim":
            L1.recover_rim = wrapped
        else:
            RL.hough_rim_recovery = wrapped
        t1 = time.process_time()
        prb = L1.layer_1_structural_salience(img, source_type="auction", house=house)
        prb_cpu = time.process_time() - t1
    finally:
        L1.recover_rim, RL.hough_rim_recovery = saved

    dc, dp = _dets(ctl), _dets(prb)
    shape = img.shape[:2]
    a, b = _alpha(ctl, shape), _alpha(prb, shape)
    inter = int(cv2.countNonZero(cv2.bitwise_and(a, b)))
    union = int(cv2.countNonZero(cv2.bitwise_or(a, b)))

    ious = []
    used = set()
    for d in dc:
        best, bi = 0.0, None
        for j, e in enumerate(dp):
            if j in used:
                continue
            v = _bbox_iou(d["bbox"], e["bbox"])
            if v > best:
                best, bi = v, j
        if bi is not None:
            used.add(bi)
        ious.append(best)

    # Diagnostic, NOT a pass criterion: how much the PRIMARY object moved.
    # Detection COUNT churns for a reason unrelated to the coin's crop — a
    # large recovered circle suppresses small noise blobs via NMS containment,
    # so removing the recovery lets those specks survive as extra "detections"
    # (the same mechanism the cap800 A/B documented). Tracking the largest
    # detection separately says whether the COIN's geometry moved or only the
    # speck census did.
    def _largest(ds):
        return max(ds, key=lambda d: d["geometry"]["area"], default=None) if ds else None

    lc, lp = _largest(dc), _largest(dp)
    primary_iou = _bbox_iou(lc["bbox"], lp["bbox"]) if (lc and lp) else 0.0

    return dict(
        ctl_n=len(dc), prb_n=len(dp),
        primary_bbox_iou=round(primary_iou, 4),
        ctl_primary_area=int(lc["geometry"]["area"]) if lc else 0,
        prb_primary_area=int(lp["geometry"]["area"]) if lp else 0,
        ctl_primary_circ=lc["geometry"]["circularity"] if lc else None,
        prb_primary_circ=lp["geometry"]["circularity"] if lp else None,
        ctl_recovered=sum(1 for d in dc if (d.get("debug_data") or {}).get("rim_recovered")),
        prb_recovered=sum(1 for d in dp if (d.get("debug_data") or {}).get("rim_recovered")),
        hough_calls_ctl=stats["calls"], hough_skipped=stats["skipped"],
        ctl_cpu_s=round(ctl_cpu, 2), prb_cpu_s=round(prb_cpu, 2),
        cpu_saved_s=round(ctl_cpu - prb_cpu, 2),
        alpha_iou=round(inter / union, 5) if union else 1.0,
        min_bbox_iou=round(min(ious), 4) if ious else 1.0,
        outcome_unchanged=bool(len(dc) == len(dp)
                               and (min(ious) if ious else 1.0) >= 0.99),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--classified", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--house", default=None)
    ap.add_argument("--layout", choices=["half", "single"], default="half")
    ap.add_argument("--mechanism", default="low_contrast_coastline",
                    help="worst class to probe; 'HEALTHY' = sides with no trigger at all")
    ap.add_argument("--arm", choices=["recover_skip", "hough_skip"],
                    default="hough_skip")
    ap.add_argument("--cv-min", type=float, default=CV_MIN_DEFAULT)
    ap.add_argument("--area-ratio-floor", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scan = json.load(open(a.scan))
    if a.mechanism == "HEALTHY":
        keys = [(s["id"], s["side"]) for s in scan if s["n_will_hough"] == 0]
    else:
        rows = [r for r in csv.DictReader(open(a.classified))
                if r["mechanism"] == a.mechanism and r["cost_tier"] == "expensive"]
        keys = list(dict.fromkeys((r["id"], r["side"]) for r in rows))
    keys = keys[: a.limit]
    print(f"[trigger] arm={a.arm} mechanism={a.mechanism} sides={len(keys)} "
          f"cv_min={a.cv_min} area_ratio_floor={a.area_ratio_floor}", flush=True)

    out = []
    for i, (sid, side) in enumerate(keys):
        cands = list(Path(a.images).glob(f"{sid}.*"))
        if not cands:
            continue
        full = _load_and_resize(str(cands[0]))
        im = split_sides(full)[side] if a.layout == "half" else full
        r = run_side(im, a.house, a.arm, a.cv_min, a.area_ratio_floor)
        r.update(id=sid, side=side, mechanism=a.mechanism, arm=a.arm)
        out.append(r)
        print(f"  [{i+1}/{len(keys)}] {sid} {side}: "
              f"n {r['ctl_n']}->{r['prb_n']}  alpha_iou={r['alpha_iou']}  "
              f"cpu {r['ctl_cpu_s']}->{r['prb_cpu_s']}s  "
              f"skipped={r['hough_skipped']}/{r['hough_calls_ctl']}  "
              f"primary_iou={r['primary_bbox_iou']} "
              f"circ {r['ctl_primary_circ']}->{r['prb_primary_circ']}  "
              f"{'UNCHANGED' if r['outcome_unchanged'] else 'CHANGED'}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wtr.writeheader()
        wtr.writerows(out)

    unch = sum(1 for r in out if r["outcome_unchanged"])
    ci = min((r["alpha_iou"] for r in out), default=1.0)
    print(f"\n[trigger] {unch}/{len(out)} sides outcome-unchanged; "
          f"worst alpha_iou={ci}")
    print(f"  CPU: control {sum(r['ctl_cpu_s'] for r in out):.1f}s -> "
          f"probe {sum(r['prb_cpu_s'] for r in out):.1f}s "
          f"({100*(1-sum(r['prb_cpu_s'] for r in out)/max(1e-9,sum(r['ctl_cpu_s'] for r in out))):.1f}% saved)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
