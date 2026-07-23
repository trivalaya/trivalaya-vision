#!/usr/bin/env python3
"""
Cheap (no-Hough) identification of the sides the shape guard would fire on, per
lane. A side "fires" iff >=1 of its triggering contours (need_recovery AND
geometric fit not confident, i.e. Hough would run) also passes the guard's disc
test. Mirrors analyze_image's preprocessing exactly; the only per-contour cost
is geometric_fit_recovery (~20ms), never Hough. ~2s/side.

lane=ingest -> raw (PNG-lossless) half, house=cng_feature
lane=query  -> temp-JPEG(q95) half (matching _mask_query_image_meta), house=None
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import os
from pathlib import Path

WT = "/home/claudeuser/wt-guard"
sys.path.insert(0, WT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import CoinConfig, Layer1Config  # noqa: E402
from src.math_utils import compute_circularity_safe, is_contour_valid  # noqa: E402
from src.rim_logic import geometric_fit_recovery  # noqa: E402
from src import rim_shape_guard as guard  # noqa: E402
from tests.conftest import _preprocess  # noqa: E402

KS17 = "/home/claudeuser/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images"


def _half(img, side):
    mid = img.shape[1] // 2
    return img[:, :mid] if side == "obv" else img[:, mid:]


def _jpeg_roundtrip(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def scan_side(img, hole_frac=None):
    ge, ez, tt, h, w = _preprocess(img)
    blur = cv2.GaussianBlur(ge, (7, 7), 0)
    _, binary = cv2.threshold(blur, 0, 255, tt)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=Layer1Config.CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_fire = 0
    fire_rois = []
    for c in contours:
        ok, _ = is_contour_valid(c, min_area=Layer1Config.Standard.MIN_AREA_PX)
        if not ok:
            continue
        circ = compute_circularity_safe(c)
        if not (CoinConfig.ENABLE_RIM_RECOVERY and circ < Layer1Config.CIRCULARITY_RELAXED):
            continue
        area = cv2.contourArea(c)
        (_, _), enc_r = cv2.minEnclosingCircle(c)
        ar = area / (np.pi * enc_r * enc_r) if enc_r > 0 else 1.0
        if not (ar < 0.85):
            continue
        gc, gconf = geometric_fit_recovery(img, c)
        will_hough = not (gc is not None and gconf > 0.65)
        if not will_hough:
            continue
        # guard disc test (env not consulted; call is_disc directly)
        if hole_frac is not None:
            os.environ[guard.ENV_MAX_HOLE_FRAC] = str(hole_frac)
        else:
            os.environ.pop(guard.ENV_MAX_HOLE_FRAC, None)
        if guard.is_disc(c, (h, w), binary=binary):
            n_fire += 1
            x, y, bw, bh = cv2.boundingRect(c)
            fire_rois.append(bw * bh)
    return n_fire, fire_rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=["ingest", "query"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hole-frac", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ids = sorted(p.stem for p in Path(KS17).glob("*.jpg"))
    if a.limit:
        ids = ids[: a.limit]
    firing = []
    n_sides = 0
    for i, ident in enumerate(ids):
        img = cv2.imread(str(Path(KS17) / f"{ident}.jpg"))
        if img is None:
            continue
        for side in ("obv", "rev"):
            half = _half(img, side)
            if a.lane == "query":
                half = _jpeg_roundtrip(half)
            n_fire, rois = scan_side(half, a.hole_frac)
            n_sides += 1
            if n_fire:
                firing.append([ident, side])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ids)} images  firing={len(firing)}", flush=True)
    json.dump(firing, open(a.out, "w"))
    print(f"[fire] lane={a.lane} hole_frac={a.hole_frac}: "
          f"{len(firing)}/{n_sides} sides fire the guard -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
