#!/usr/bin/env python3
"""
Score a rim_guard_ab.py JSONL run against the RATIFIED §7 bars.

Per side, OFF vs ON:
  INGEST metric  -- all detections: ndets, greedy bbox-IoU match, rim_recovered
                    multiset, and alpha-mask IoU (union of all final contours).
  QUERY metric   -- largest-contour-wins (what _mask_query_image_meta feeds
                    DINOv2): the single largest detection's bbox IoU + its mask
                    alpha-IoU + rim_recovered.

outcome_unchanged (per lane):
  INGEST: ndet equal AND every matched bbox IoU >= 0.99 AND rim_recovered
          multiset equal.
  QUERY : same largest-contour bbox IoU >= 0.99 AND its rim_recovered equal.

Bar 1 (per lane): >= 98% of the FULL population unchanged. Non-firing sides
(not in the AB file) are unchanged BY CONSTRUCTION (guard returns False =>
recover_rim identical) -- counted as unchanged, with --population giving the
denominator. Worst alpha IoU is the MINIMUM over measured sides (non-firing
sides are alpha IoU 1.0 by construction). Changed sides are listed for
individual disposition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def _alpha(dets, shape):
    m = np.zeros(shape, np.uint8)
    for d in dets:
        c = np.asarray(d["contour"], dtype=np.int32)
        cv2.drawContours(m, [c], -1, 255, -1)
    return m


def _shape_of(rec):
    H = W = 0
    for arm in ("off", "on"):
        for d in rec[arm]["dets"]:
            c = np.asarray(d["contour"], dtype=np.int32)
            if len(c):
                W = max(W, int(c[:, 0].max()) + 2)
                H = max(H, int(c[:, 1].max()) + 2)
    return (max(H, 2), max(W, 2))


def _match_min_iou(dc, dp):
    used, ious = set(), []
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
    return min(ious) if ious else 1.0


def score_ingest(rec, shape):
    off, on = rec["off"]["dets"], rec["on"]["dets"]
    a, b = _alpha(off, shape), _alpha(on, shape)
    inter = int(cv2.countNonZero(cv2.bitwise_and(a, b)))
    union = int(cv2.countNonZero(cv2.bitwise_or(a, b)))
    alpha_iou = (inter / union) if union else 1.0
    min_bbox = _match_min_iou(off, on)
    rr_off = sorted(d["rim_recovered"] for d in off)
    rr_on = sorted(d["rim_recovered"] for d in on)
    unchanged = (len(off) == len(on) and min_bbox >= 0.99 and rr_off == rr_on)
    return dict(ndet_off=len(off), ndet_on=len(on), min_bbox_iou=round(min_bbox, 4),
                alpha_iou=round(alpha_iou, 5), rr_off=rr_off, rr_on=rr_on,
                unchanged=unchanged)


def score_query(rec, shape):
    off, on = rec["off"]["dets"], rec["on"]["dets"]
    if not off or not on:
        return dict(unchanged=(len(off) == len(on)), alpha_iou=1.0 if len(off) == len(on) else 0.0,
                    bbox_iou=1.0 if len(off) == len(on) else 0.0, note="empty-arm")
    lo = max(off, key=lambda d: d["area"])
    lp = max(on, key=lambda d: d["area"])
    a = _alpha([lo], shape)
    b = _alpha([lp], shape)
    inter = int(cv2.countNonZero(cv2.bitwise_and(a, b)))
    union = int(cv2.countNonZero(cv2.bitwise_or(a, b)))
    alpha_iou = (inter / union) if union else 1.0
    bb = _bbox_iou(lo["bbox"], lp["bbox"])
    unchanged = (bb >= 0.99 and lo["rim_recovered"] == lp["rim_recovered"])
    return dict(bbox_iou=round(bb, 4), alpha_iou=round(alpha_iou, 5),
                rr_off=lo["rim_recovered"], rr_on=lp["rim_recovered"],
                unchanged=unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--lane", required=True, choices=["ingest", "query"])
    ap.add_argument("--population", type=int, default=574)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.ab) if l.strip()]
    recs = [r for r in recs if "error" not in r]
    scorer = score_ingest if a.lane == "ingest" else score_query

    measured, changed = [], []
    worst_alpha = 1.0
    cpu_off = cpu_on = 0.0
    for r in recs:
        shape = _shape_of(r)
        s = scorer(r, shape)
        s.update(id=r["id"], side=r["side"],
                 hough_off=r["off"]["hough_calls"], hough_on=r["on"]["hough_calls"],
                 cpu_off=r["off"]["cpu_s"], cpu_on=r["on"]["cpu_s"])
        measured.append(s)
        worst_alpha = min(worst_alpha, s.get("alpha_iou", 1.0))
        cpu_off += r["off"]["cpu_s"]
        cpu_on += r["on"]["cpu_s"]
        if not s["unchanged"]:
            changed.append(s)

    n_meas = len(measured)
    n_unchanged_measured = n_meas - len(changed)
    # full population: non-measured sides unchanged by construction
    pop = a.population
    n_unchanged_total = pop - len(changed)
    pct = 100.0 * n_unchanged_total / pop

    print(f"=== Bar 1 scoring: lane={a.lane} ===")
    print(f"measured (guard-firing) sides: {n_meas}")
    print(f"  unchanged among measured   : {n_unchanged_measured}")
    print(f"  CHANGED among measured     : {len(changed)}")
    print(f"population                   : {pop}")
    print(f"  unchanged / population     : {n_unchanged_total}/{pop} = {pct:.2f}%  "
          f"(bar: >= 98%  -> {'PASS' if pct >= 98.0 else 'FAIL -> add hole-frac conjunct & re-run'})")
    print(f"  changed-rate               : {100.0*len(changed)/pop:.2f}%  "
          f"(bar: <= 2%   -> {'PASS' if len(changed) <= 0.02*pop else 'FAIL'})")
    print(f"worst alpha IoU (measured)   : {worst_alpha:.5f}  "
          f"(bar: >= 0.995 -> {'PASS' if worst_alpha >= 0.995 else 'REVIEW each changed side'})")
    p99_off = np.percentile([r['off']['cpu_s'] for r in recs], 99)
    p99_on = np.percentile([r['on']['cpu_s'] for r in recs], 99)
    print(f"p99 CPU-s (firing sides)     : OFF {p99_off:.1f}s -> ON {p99_on:.1f}s "
          f"({100*(1-p99_on/max(1e-9,p99_off)):.1f}% saved)  [REPORT-ONLY]")
    print(f"total CPU-s (firing sides)   : OFF {cpu_off:.0f}s -> ON {cpu_on:.0f}s")
    if changed:
        print(f"\nCHANGED sides (disposition each):")
        for s in sorted(changed, key=lambda x: x.get("alpha_iou", 1.0)):
            extra = (f"ndet {s.get('ndet_off')}->{s.get('ndet_on')} "
                     if a.lane == "ingest" else "")
            print(f"  {s['id']} {s['side']}: {extra}bbox_iou={s.get('min_bbox_iou', s.get('bbox_iou'))} "
                  f"alpha_iou={s.get('alpha_iou')} rr {s.get('rr_off')}->{s.get('rr_on')} "
                  f"hough {s['hough_off']}->{s['hough_on']}")
    if a.out:
        json.dump({"measured": measured, "changed": changed,
                   "pct_unchanged": pct, "worst_alpha": worst_alpha}, open(a.out, "w"))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
