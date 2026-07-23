#!/usr/bin/env python3
"""
rim_scale_sensitivity_probe.py — is `circularity < 0.65` measuring SHAPE or
measuring BOUNDARY ROUGHNESS AT THIS RESOLUTION?

Motivation (specs/results/rim_stall_taxonomy_2026-07-23.md): the contours
that pay the whole Hough bill are compact and disc-like by every measure
except circularity -- median solidity 0.862, median area_ratio 0.731, median
largest radial dip 1 degree, median largest radial spike 4 degrees -- yet
their median circularity is 0.088, because their perimeter is 3.4x that of
an equal-area circle. `4*pi*A/P^2` is quadratic in perimeter, and perimeter
is the one contour statistic that GROWS as you resolve more of the boundary
(the coastline effect). A coin's rim beading, dentate border, micro-chipping
and JPEG mosquito noise are all real boundary length at 1500px and invisible
at 500px.

If that is what is happening, then the same physical coin, segmented at a
lower resolution, gets a materially higher circularity -- and the "ambiguous
contour" framing is wrong: the contour is right, the metric is scale-coupled.

This probe measures it three ways, per contour, with no Hough anywhere:

  A. RESOLUTION sweep -- re-run the real preamble + segmentation on the image
     downscaled by 1.0 / 0.5 / 0.33 / 0.25, and report circularity of the
     contour whose centroid matches the full-res one.
  B. SMOOTHING alternatives at native resolution -- circularity computed on
     the convex hull, on an approxPolyDP simplification, and on a
     morphologically-opened mask. These isolate "boundary roughness" from
     "shape" without changing resolution.
  C. TRIGGER counterfactual -- under each alternative, would this contour
     still satisfy `circ < 0.65 AND area_ratio < 0.85`?

Read-only, no production file modified, no Hough call made.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Layer1Config  # noqa: E402
from src.math_utils import compute_circularity_safe, is_contour_valid  # noqa: E402
from tools.rim_stall_taxonomy import (  # noqa: E402
    _load_and_resize,
    preamble,
    segment,
    split_sides,
)

SCALES = (1.0, 0.5, 0.33, 0.25)


def _metrics(c):
    area = float(cv2.contourArea(c))
    circ = float(compute_circularity_safe(c))
    (_, _), enc_r = cv2.minEnclosingCircle(c)
    enc_area = np.pi * enc_r * enc_r
    ar = area / enc_area if enc_area > 0 else 1.0
    return circ, float(ar), area, float(enc_r)


def _match_by_centroid(contours, cx, cy, scale, tol_frac=0.06, shape=None):
    """Find the contour at the downscaled resolution that corresponds to the
    full-res one: nearest centroid within tol of the frame diagonal."""
    best, bestd = None, 1e18
    tx, ty = cx * scale, cy * scale
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        d = (M["m10"] / M["m00"] - tx) ** 2 + (M["m01"] / M["m00"] - ty) ** 2
        if d < bestd:
            best, bestd = c, d
    if best is None or shape is None:
        return best
    diag = np.hypot(*shape)
    return best if np.sqrt(bestd) <= tol_frac * diag else None


def _smoothed_variants(c, shape):
    """Boundary-roughness controls at native resolution."""
    out = {}
    hull = cv2.convexHull(c)
    out["hull"] = hull
    peri = cv2.arcLength(c, True)
    out["approx1pct"] = cv2.approxPolyDP(c, 0.01 * peri, True)
    # Morphological open+close on the filled mask, radius scaled to the blob
    (_, _), r = cv2.minEnclosingCircle(c)
    k = max(3, int(round(r * 0.04)) | 1)
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [c], -1, 255, -1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m2 = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, se), cv2.MORPH_CLOSE, se)
    cs, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        out[f"morph_k{k}"] = max(cs, key=cv2.contourArea)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--house", default=None)
    ap.add_argument("--layout", choices=["half", "single"], default="half")
    ap.add_argument("--min-roi-px", type=int, default=600_000,
                    help="restrict to the expensive class")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sides = [s for s in json.load(open(a.scan))
             if any(r.get("will_hough") and r["roi_px"] >= a.min_roi_px
                    for r in s["contours"])]
    sides = sides[:: a.stride]
    if a.limit:
        sides = sides[: a.limit]
    print(f"[scale] {len(sides)} sides with an expensive trigger contour", flush=True)

    rows = []
    for i, s in enumerate(sides):
        cands = list(Path(a.images).glob(f"{s['id']}.*"))
        if not cands:
            continue
        full = _load_and_resize(str(cands[0]))
        im = split_sides(full)[s["side"]] if a.layout == "half" else full

        per_scale = {}
        for sc in SCALES:
            img = im if sc == 1.0 else cv2.resize(
                im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))),
                interpolation=cv2.INTER_AREA)
            pre = preamble(img)
            _, _, _, cs = segment(pre, a.house)
            per_scale[sc] = (cs, (pre["h"], pre["w"]))

        for r in s["contours"]:
            if not (r.get("will_hough") and r["roi_px"] >= a.min_roi_px):
                continue
            c0 = per_scale[1.0][0][r["contour_idx"]]
            M = cv2.moments(c0)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            row = dict(id=s["id"], side=s["side"], contour_idx=r["contour_idx"],
                       roi_px=r["roi_px"], area_frac_frame=round(r["area_frac_frame"], 5))
            for sc in SCALES:
                cs, shp = per_scale[sc]
                c = c0 if sc == 1.0 else _match_by_centroid(cs, cx, cy, sc, shape=shp)
                if c is None or not is_contour_valid(
                        c, min_area=Layer1Config.Standard.MIN_AREA_PX)[0]:
                    row[f"circ@{sc}"] = row[f"ar@{sc}"] = row[f"trig@{sc}"] = None
                    continue
                circ, ar, _, _ = _metrics(c)
                row[f"circ@{sc}"] = round(circ, 4)
                row[f"ar@{sc}"] = round(ar, 4)
                row[f"trig@{sc}"] = bool(circ < Layer1Config.CIRCULARITY_RELAXED and ar < 0.85)
            for name, cv_ in _smoothed_variants(c0, (im.shape[0], im.shape[1])).items():
                circ, ar, _, _ = _metrics(cv_)
                row[f"circ_{name}"] = round(circ, 4)
                row[f"ar_{name}"] = round(ar, 4)
                row[f"trig_{name}"] = bool(circ < Layer1Config.CIRCULARITY_RELAXED and ar < 0.85)
            rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sides)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    head = ["id", "side", "contour_idx", "roi_px", "area_frac_frame"]
    with open(a.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=head + [k for k in keys if k not in head])
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n[scale] n={len(rows)} expensive trigger contours")
    print("  RESOLUTION sweep (would the trigger still fire?)")
    for sc in SCALES:
        v = [r for r in rows if r.get(f"trig@{sc}") is not None]
        t = sum(1 for r in v if r[f"trig@{sc}"])
        cc = [r[f"circ@{sc}"] for r in v]
        print(f"    scale {sc:<5}: matched={len(v):4d}  still triggers={t:4d} "
              f"({100*t/max(1,len(v)):5.1f}%)  median circ={np.median(cc):.3f}")
    print("  SMOOTHING controls at native resolution")
    for name in ("hull", "approx1pct"):
        v = [r for r in rows if f"trig_{name}" in r]
        t = sum(1 for r in v if r[f"trig_{name}"])
        cc = [r[f"circ_{name}"] for r in v]
        print(f"    {name:<12}: n={len(v):4d}  still triggers={t:4d} "
              f"({100*t/max(1,len(v)):5.1f}%)  median circ={np.median(cc):.3f}")
    morphs = sorted({k for r in rows for k in r if k.startswith("trig_morph")})
    if morphs:
        v = [(r, k) for r in rows for k in morphs if k in r]
        t = sum(1 for r, k in v if r[k])
        cc = [r[k.replace("trig_", "circ_")] for r, k in v]
        print(f"    morph_open   : n={len(v):4d}  still triggers={t:4d} "
              f"({100*t/max(1,len(v)):5.1f}%)  median circ={np.median(cc):.3f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
