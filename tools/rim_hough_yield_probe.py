#!/usr/bin/env python3
"""
rim_hough_yield_probe.py — what does the expensive Hough recovery BUY?

The Scope A cost work stalled because every lever tried so far (cap the
accumulator resolution) trades accuracy for speed: some sides get better,
some get worse. The owner ruling of 2026-07-23 rejects that trade and asks
for mechanisms "whose outcome changes are confined to the currently-
pathological tail."

That ruling only has a solution if cost and yield are DECOUPLED — i.e. if the
most expensive `hough_rim_recovery` calls are also the ones least likely to
produce an accepted rim. This probe measures exactly that, per triggering
contour:

  cost   : wall/cpu seconds, split between the geometric-fit branch and the
           Hough branch, so the attribution is unambiguous
  yield  : did recovery return a contour, and did `validate_rim_recovery`
           accept it? which branch won (geometric fit vs Hough)?
  effect : if accepted, how much did the final contour actually move?
           (IoU vs the seed blob, radius ratio, area ratio)

A pre-filter that skips recovery is FREE exactly where yield is zero. Where
yield is non-zero, skipping it is a regression — so the join of these two
columns is what tells you whether a cost fix can exist at all.

Read-only. Calls the real, unmodified `src.rim_logic.geometric_fit_recovery`,
`src.rim_logic.hough_rim_recovery` and `src.math_utils.validate_rim_recovery`,
then reproduces `recover_rim`'s tie-break in-process (calling `recover_rim`
itself would re-pay the dominant Hough cost and corrupt every timing here).
No production file is modified; nothing is written outside --out.
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

from src.math_utils import validate_rim_recovery  # noqa: E402
from src.rim_logic import (  # noqa: E402
    geometric_fit_recovery,
    hough_rim_recovery,
)
from tools.rim_stall_taxonomy import (  # noqa: E402
    _load_and_resize,
    preamble,
    segment,
    split_sides,
)


def _mask(contour, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    return m


def probe_side(img, house, scan_side, stall_only=True):
    pre = preamble(img)
    _, _, _, contours = segment(pre, house)
    h, w = pre["h"], pre["w"]
    out = []
    for r in scan_side["contours"]:
        if stall_only and not r.get("will_hough"):
            continue
        c = contours[r["contour_idx"]]

        # Split the two branches so the cost attribution is unambiguous.
        t0, c0 = time.perf_counter(), time.process_time()
        geo_c, geo_conf = geometric_fit_recovery(img, c)
        geo_wall = time.perf_counter() - t0
        geo_cpu = time.process_time() - c0

        hou_wall = hou_cpu = 0.0
        hou_c, hou_conf = None, 0.0
        if not (geo_c is not None and geo_conf > 0.65):
            t1, c1 = time.perf_counter(), time.process_time()
            hou_c, hou_conf = hough_rim_recovery(img, c)
            hou_wall = time.perf_counter() - t1
            hou_cpu = time.process_time() - c1

        # Reproduce recover_rim's own tie-break rather than calling it again:
        # a second call would pay the (dominant) Hough cost twice and inflate
        # every number in this table. Mirrors src/rim_logic.py::recover_rim.
        def _r(cc):
            return float(cv2.minEnclosingCircle(cc)[1]) if cc is not None else 0.0

        if geo_c is not None and geo_conf > 0.65:
            final_c, final_conf = geo_c, geo_conf
        elif geo_c is None:
            final_c, final_conf = hou_c, hou_conf
        elif hou_c is None:
            final_c, final_conf = geo_c, geo_conf
        elif _r(hou_c) > _r(geo_c) * 1.05 and hou_conf >= 0.12:
            final_c, final_conf = hou_c, hou_conf
        else:
            final_c, final_conf = geo_c, geo_conf
        rr_wall, rr_cpu = geo_wall + hou_wall, geo_cpu + hou_cpu

        accepted = bool(final_c is not None
                        and validate_rim_recovery(final_c, c, (h, w)))
        row = dict(
            id=scan_side["id"], side=scan_side["side"],
            contour_idx=r["contour_idx"],
            area_frac_frame=round(r["area_frac_frame"], 5),
            circularity=r["circularity"], area_ratio=r["area_ratio"],
            enc_r=r["enc_r"], roi_w=r["roi_w"], roi_h=r["roi_h"],
            roi_px=r["roi_px"], touches_border=r["touches_border"],
            geo_conf=round(float(geo_conf), 4), geo_wall_s=round(geo_wall, 3),
            geo_cpu_s=round(geo_cpu, 3),
            hough_ran=hou_c is not None or hou_wall > 0,
            hough_found=hou_c is not None,
            hough_conf=round(float(hou_conf), 4),
            hough_wall_s=round(hou_wall, 3), hough_cpu_s=round(hou_cpu, 3),
            recover_wall_s=round(rr_wall, 3), recover_cpu_s=round(rr_cpu, 3),
            final_conf=round(float(final_conf), 4),
            accepted=accepted,
        )
        if hou_c is not None and final_c is not None:
            row["branch"] = "hough" if np.array_equal(final_c, hou_c) else "geo"
        elif final_c is not None:
            row["branch"] = "geo"
        else:
            row["branch"] = "none"

        if accepted:
            a, b = _mask(c, (h, w)), _mask(final_c, (h, w))
            inter = int(cv2.countNonZero(cv2.bitwise_and(a, b)))
            union = int(cv2.countNonZero(cv2.bitwise_or(a, b)))
            row["iou_vs_seed"] = round(inter / union, 4) if union else None
            (_, _), sr = cv2.minEnclosingCircle(c)
            (_, _), fr = cv2.minEnclosingCircle(final_c)
            row["radius_ratio"] = round(float(fr / sr), 4) if sr else None
            row["area_ratio_final_vs_seed"] = round(
                float(cv2.contourArea(final_c) / max(1.0, cv2.contourArea(c))), 4)
        else:
            row["iou_vs_seed"] = row["radius_ratio"] = row["area_ratio_final_vs_seed"] = None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--house", default=None)
    ap.add_argument("--layout", choices=["half", "single"], default="half")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth stall side (cost control)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sides = [s for s in json.load(open(a.scan)) if s["n_will_hough"] > 0]
    sides = sides[:: a.stride]
    if a.limit:
        sides = sides[: a.limit]
    print(f"[yield] {len(sides)} stall sides", flush=True)

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(sides):
        cands = list(Path(a.images).glob(f"{s['id']}.*"))
        if not cands:
            continue
        img = _load_and_resize(str(cands[0]))
        im = split_sides(img)[s["side"]] if a.layout == "half" else img
        rows.extend(probe_side(im, a.house, s))
        print(f"  [{i+1}/{len(sides)}] {s['id']} {s['side']} "
              f"({time.perf_counter()-t0:.0f}s elapsed)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    acc = [r for r in rows if r["accepted"]]
    hot = sorted(rows, key=lambda r: -r["hough_cpu_s"])
    print(f"\n[yield] {len(rows)} triggering contours, "
          f"{len(acc)} accepted ({100*len(acc)/max(1,len(rows)):.0f}%)")
    print(f"  total hough cpu_s: {sum(r['hough_cpu_s'] for r in rows):.0f}")
    for lbl, sel in [("top-10% costliest", hot[: max(1, len(hot) // 10)]),
                     ("cheapest 50%", hot[len(hot) // 2:])]:
        n_a = sum(1 for r in sel if r["accepted"])
        print(f"  {lbl}: n={len(sel)} accepted={n_a} "
              f"({100*n_a/max(1,len(sel)):.0f}%) "
              f"cpu={sum(r['hough_cpu_s'] for r in sel):.0f}s")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
