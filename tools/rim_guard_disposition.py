#!/usr/bin/env python3
"""
Render an OFF-vs-ON disposition panel for each changed side of a rim_guard_ab.py
run, so each Bar-1 change can be individually classified: control-right/
guard-worse (FAIL) vs control-wrong/guard-benign vs guard-improvement.

Per side: the half image, with every OFF detection contour in GREEN and every ON
detection contour in RED, minEnclosingCircle of the largest in thin blue, and a
header with ndet, rim_recovered flags, bbox/alpha IoU. Green-only regions
(present OFF, gone/changed ON) are the guard's effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

KS17 = "/home/claudeuser/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images"


def _half(img, side):
    mid = img.shape[1] // 2
    return img[:, :mid] if side == "obv" else img[:, mid:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--images-dir", default=KS17)
    ap.add_argument("--layout", choices=["half", "single"], default="half")
    ap.add_argument("--only", default=None, help="comma id:side to force-render")
    ap.add_argument("--changed-only", action="store_true")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    force = set(a.only.split(",")) if a.only else None

    for line in open(a.ab):
        r = json.loads(line)
        if "error" in r:
            continue
        key = f"{r['id']}:{r['side']}"
        off, on = r["off"]["dets"], r["on"]["dets"]
        rr_off = sorted(d["rim_recovered"] for d in off)
        rr_on = sorted(d["rim_recovered"] for d in on)
        changed = (len(off) != len(on) or rr_off != rr_on)
        if force is not None and key not in force and r["id"] not in force:
            continue
        if force is None and a.changed_only and not changed:
            continue

        img = cv2.imread(str(Path(a.images_dir) / f"{r['id']}.jpg"))
        if img is None:
            continue
        vis = (img if a.layout == "single" else _half(img, r["side"])).copy()
        for d in off:
            cv2.drawContours(vis, [np.asarray(d["contour"], np.int32)], -1, (0, 220, 0), 3)
        for d in on:
            cv2.drawContours(vis, [np.asarray(d["contour"], np.int32)], -1, (0, 0, 255), 2)
        if off:
            lo = max(off, key=lambda d: d["area"])
            (cx, cy), rr = cv2.minEnclosingCircle(np.asarray(lo["contour"], np.int32))
            cv2.circle(vis, (int(cx), int(cy)), int(rr), (255, 180, 0), 1)
        for i, ln in enumerate([
            f"{r['id']} {r['side']}  GREEN=OFF({len(off)}) RED=ON({len(on)})",
            f"rim_recovered OFF{rr_off} -> ON{rr_on}",
            f"hough {r['off']['hough_calls']}->{r['on']['hough_calls']} "
            f"cpu {r['off']['cpu_s']}->{r['on']['cpu_s']}s",
        ]):
            y = 26 + i * 26
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        out = Path(a.outdir) / f"{r['id']}_{r['side']}.jpg"
        cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  {key} changed={changed} -> {out}")


if __name__ == "__main__":
    main()
