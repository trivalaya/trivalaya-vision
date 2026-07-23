#!/usr/bin/env python3
"""
rim_stall_taxonomy.py — WHY does Layer-1 primary segmentation produce
ambiguous contours on curated auction photos?

Read-only probe. Touches no production code path: it re-executes the exact
same sequence `layer_1_structural_salience` ->
`_segment_and_extract_candidates` runs (same production helpers, same
Layer1Config constants, same order), but stops before pass 2's `recover_rim`
so it can record, per contour, the metrics that decide whether the expensive
`cv2.HoughCircles` branch fires -- WITHOUT paying for it.

Why replication rather than instrumenting L1: L1 returns only finished,
NMS-suppressed candidates. The contour that actually pays the Hough cost is
frequently suppressed before it reaches the caller (the KS-17 diagnosis
measured this: `n_det` is not a stall predictor because it is counted after
suppression). Nothing short of per-contour interception answers "why".
`tools/verify_taxonomy_replication.py` proves this file's replication is
faithful against the real `layer_1_structural_salience`.

The trigger is a CONJUNCTION of three conditions, not the two usually cited:

  1. circularity          <  Layer1Config.CIRCULARITY_RELAXED (0.65)
  2. area/enclosing_area  <  0.85                    -> need_recovery
  3. geometric_fit_recovery's combined conf <= 0.65  -> Hough actually runs

(3) is the gate inside `rim_logic.recover_rim` -- a confident geometric fit
short-circuits and Hough is skipped. It is cheap to evaluate (~20ms), so this
scanner evaluates all three and reports `will_hough` exactly.

Subcommands:
  scan       per-side + per-contour metrics -> CSV/JSON  (no Hough, fast)
  classify   assign a MECHANISM to each Hough-triggering contour
  overlay    render one diagnostic panel per stall side
  montage    tile overlays into per-class contact sheets
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CoinConfig, Layer1Config, MAX_DIMENSION  # noqa: E402
from src.math_utils import (  # noqa: E402
    compute_circularity_safe,
    detect_background_histogram,
    is_contour_valid,
)
from src.layer1_geometry import _close_kernel_size  # noqa: E402
from src.rim_logic import geometric_fit_recovery  # noqa: E402


# ─────────────────────────── faithful preamble ────────────────────────────

def _load_and_resize(path: str) -> Optional[np.ndarray]:
    """Mirror of pipeline_manager._load_and_resize (MAX_DIMENSION=3200)."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        s = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def split_sides(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Mirror of corpus_match_report.load_sides for the `combined` layout."""
    w = img.shape[1]
    mid = w // 2
    return {"obv": img[:, :mid], "rev": img[:, mid:]}


def preamble(img: np.ndarray):
    """Everything layer_1_structural_salience does before segmentation."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bg, bg_type = detect_background_histogram(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    v = np.median(gray_enhanced)
    sigma = Layer1Config.Standard.CANNY_SIGMA
    edges = cv2.Canny(gray_enhanced,
                      int(max(0, (1.0 - sigma) * v)),
                      int(min(255, (1.0 + sigma) * v)))
    edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return dict(h=h, w=w, gray=gray, gray_enhanced=gray_enhanced, avg_bg=avg_bg,
                bg_type=bg_type, thresh_type=thresh_type, edge_zone=edge_zone)


def segment(pre: dict, house: Optional[str]):
    """Mirror of _segment_and_extract_candidates' blur/threshold/close/find."""
    h, w = pre["h"], pre["w"]
    blurred = cv2.GaussianBlur(pre["gray_enhanced"], (7, 7), 0)
    otsu_t, binary = cv2.threshold(blurred, 0, 255, pre["thresh_type"])
    frac_env = os.environ.get("TRIVALAYA_CLOSE_KERNEL_FRAC")
    if not frac_env:
        k = 7
    elif frac_env.strip().lower() == "auto":
        tabled = bool(house) and house.strip().lower() in Layer1Config.CLOSE_KERNEL_BY_HOUSE
        k = _close_kernel_size(h, w, house=house) if tabled else 7
    else:
        k = _close_kernel_size(h, w, frac=float(frac_env), house=house)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=Layer1Config.CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return otsu_t, k, binary, contours


# ───────────────────────── per-contour measurement ────────────────────────

def radial_profile(contour: np.ndarray, shape, n_rays: int = 360):
    """
    r(theta): distance from the blob centroid to its farthest filled pixel
    along each ray. A clean disc is flat; a BITE is a contiguous dip; an
    attached lobe (shadow/label/neighbour) is a contiguous spike.
    Returns (r, r_med, frac_dip, frac_spike, max_dip_run_deg, max_spike_run_deg).
    """
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    (_, _), enc_r = cv2.minEnclosingCircle(contour)
    steps = int(enc_r * 1.4)
    if steps < 8:
        return None
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    t = np.arange(1, steps + 1, dtype=np.float32)
    xs = np.clip((cx + np.outer(np.cos(th), t)).astype(np.int32), 0, w - 1)
    ys = np.clip((cy + np.outer(np.sin(th), t)).astype(np.int32), 0, h - 1)
    hit = mask[ys, xs] > 0
    r = np.where(hit.any(axis=1), steps - np.argmax(hit[:, ::-1], axis=1), 0).astype(np.float32)
    r_med = float(np.median(r[r > 0])) if (r > 0).any() else 0.0
    if r_med <= 0:
        return None

    def _max_run(flags):
        d = np.concatenate([flags, flags])  # circular
        best = cur = 0
        for f in d:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return min(best, n_rays) * 360.0 / n_rays

    def _n_runs(flags):
        """How many SEPARATE excursions. A klippe has 4 corner spikes; a disc
        with a noisy boundary has none. Distinguishes 'genuinely not round'
        from 'round but the boundary is dithered'."""
        f = flags.astype(np.int8)
        return int(np.count_nonzero(np.diff(np.concatenate([f, f[:1]])) == 1))

    dip, spike = r < 0.85 * r_med, r > 1.15 * r_med
    # Coefficient of variation of the radius. THE separator between a
    # low-contrast dithered rim (cv ~0.02: every ray reaches nearly the same
    # radius, the boundary just wiggles at pixel scale, inflating perimeter
    # without changing shape) and a genuinely non-circular flan (cv >~0.08:
    # klippe corners, chipped/irregular ancient flans).
    rv = r[r > 0]
    cv_r = float(np.std(rv) / np.median(rv)) if rv.size else 0.0
    return dict(r_med=r_med, frac_dip=float(dip.mean()), frac_spike=float(spike.mean()),
                max_dip_run_deg=_max_run(dip), max_spike_run_deg=_max_run(spike),
                n_dip_runs=_n_runs(dip), n_spike_runs=_n_runs(spike), cv_r=cv_r,
                cx=cx, cy=cy, enc_r=float(enc_r))


def deficit_stats(contour, gray, shape, avg_bg):
    """
    Intensity of the region the enclosing circle covers but the blob does not
    (the 'bite'), vs the blob's own interior. Names the physical cause:
      bite grey ~ background  -> the coin edge genuinely lost contrast there
      bite grey << blob       -> a dark region (toning/shadow) fell below Otsu
      bite grey >> blob       -> a blown highlight fell out of an INV threshold
    """
    h, w = shape
    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, -1)
    (ccx, ccy), enc_r = cv2.minEnclosingCircle(contour)
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (int(ccx), int(ccy)), int(enc_r), 255, -1)
    deficit = cv2.bitwise_and(disc, cv2.bitwise_not(blob))
    n_def = int(cv2.countNonZero(deficit))
    out = dict(n_deficit_px=n_def,
               deficit_frac=n_def / max(1, cv2.countNonZero(disc)))
    out["blob_mean"] = float(cv2.mean(gray, mask=blob)[0])
    out["deficit_mean"] = float(cv2.mean(gray, mask=deficit)[0]) if n_def > 100 else float("nan")
    out["bg_mean"] = float(avg_bg)
    # How much of the deficit is background-coloured (within 12 grey levels)?
    if n_def > 100:
        d = gray[deficit > 0].astype(np.float32)
        out["deficit_frac_bglike"] = float((np.abs(d - avg_bg) < 12).mean())
        out["deficit_frac_darker"] = float((d < avg_bg - 12).mean())
        out["deficit_frac_brighter"] = float((d > out["blob_mean"] - 12).mean())
    else:
        out["deficit_frac_bglike"] = out["deficit_frac_darker"] = out["deficit_frac_brighter"] = float("nan")
    return out


def hole_stats(contour, binary, shape):
    """
    How much of this blob's own interior is NOT foreground?

    Separates the two expensive mechanisms. `findContours(RETR_EXTERNAL)`
    returns only outer boundaries, so `contourArea` counts holes as filled.
    A blob that is really the BACKDROP has a coin-shaped hole punched in it
    (hole_frac large); a blob that is really a piece of the coin's RELIEF is
    solidly filled (hole_frac ~ 0).
    """
    h, w = shape
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    n_filled = int(cv2.countNonZero(filled))
    if n_filled == 0:
        return dict(hole_frac=0.0, largest_hole_frac=0.0, n_holes=0)
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(binary))
    n_holes_px = int(cv2.countNonZero(holes))
    hcs, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [cv2.contourArea(c) for c in hcs if cv2.contourArea(c) > 500]
    return dict(hole_frac=n_holes_px / n_filled,
                largest_hole_frac=(max(big) / n_filled) if big else 0.0,
                n_holes=len(big))


def backdrop_ring_mean(gray, frac: float = 0.03) -> float:
    """
    Honest background level: mean of the outer `frac` frame ring.

    Contrast this with what production computes. `detect_background_histogram`
    first tries 4x 5x5 corner patches and only trusts them if their std < 15;
    on a backdrop with a corner-to-corner luminance ramp that test always
    fails, and the histogram FALLBACK returns the mean of every pixel below
    50 -- a statistic of the scene's dark tail, not of the background.
    """
    h, w = gray.shape
    t = max(4, int(frac * min(h, w)))
    return float(np.concatenate([
        gray[:t].ravel(), gray[-t:].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel()]).mean())


def corner_ramp(gray) -> tuple:
    """(ramp, corner_std) — why the corner path is or is not trusted."""
    h, w = gray.shape
    m = 5
    cs = [gray[0:m, 0:m], gray[0:m, w - m:w], gray[h - m:h, 0:m], gray[h - m:h, w - m:w]]
    means = [float(c.mean()) for c in cs]
    allc = np.concatenate([c.flatten() for c in cs]).astype(float)
    return max(means) - min(means), float(np.std(allc))


def measure_side(img: np.ndarray, house: Optional[str], want_geo: bool = True) -> dict:
    """Full instrumented primary segmentation for one side."""
    t0, c0 = time.perf_counter(), time.process_time()
    pre = preamble(img)
    otsu_t, k, binary, contours = segment(pre, house)
    h, w, total_area = pre["h"], pre["w"], pre["h"] * pre["w"]
    ramp, cstd = corner_ramp(pre["gray"])
    ring = backdrop_ring_mean(pre["gray"])

    rows: List[dict] = []
    for ci, c in enumerate(contours):
        valid, _ = is_contour_valid(c, min_area=Layer1Config.Standard.MIN_AREA_PX)
        if not valid:
            continue
        area = float(cv2.contourArea(c))
        perimeter_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(perimeter_mask, [c], -1, 255, 1)
        perimeter_px = cv2.countNonZero(perimeter_mask)
        if perimeter_px == 0:
            continue
        edge_support = cv2.countNonZero(
            cv2.bitwise_and(perimeter_mask, pre["edge_zone"])) / perimeter_px
        circ = float(compute_circularity_safe(c))
        (_, _), enc_r = cv2.minEnclosingCircle(c)
        enc_area = np.pi * enc_r * enc_r
        area_ratio = area / enc_area if enc_area > 0 else 1.0
        need = bool(CoinConfig.ENABLE_RIM_RECOVERY
                    and circ < Layer1Config.CIRCULARITY_RELAXED
                    and area_ratio < 0.85)
        x, y, bw, bh = cv2.boundingRect(c)
        margin = int(max(bw, bh) * 0.1)
        rx1, ry1 = max(0, x - margin), max(0, y - margin)
        rx2, ry2 = min(w, x + bw + margin), min(h, y + bh + margin)
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        row = dict(
            contour_idx=ci, area=area, area_frac_frame=area / total_area,
            circularity=round(circ, 4), area_ratio=round(float(area_ratio), 4),
            solidity=round(float(area / hull_area) if hull_area > 0 else 0.0, 4),
            edge_support=round(float(edge_support), 4),
            bbox=[int(x), int(y), int(bw), int(bh)],
            enc_r=round(float(enc_r), 1),
            roi_w=roi_w, roi_h=roi_h, roi_px=roi_w * roi_h,
            touches_border=bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1),
            need_recovery=need, geo_conf=None, geo_r=None, will_hough=None,
        )
        if need and want_geo:
            gc, gconf = geometric_fit_recovery(img, c)
            row["geo_conf"] = round(float(gconf), 4)
            row["geo_r"] = round(float(cv2.minEnclosingCircle(gc)[1]), 1) if gc is not None else None
            row["will_hough"] = not (gc is not None and gconf > 0.65)
        elif need:
            row["will_hough"] = True
        rp = radial_profile(c, (h, w))
        if rp:
            row.update({f"rp_{kk}": (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in rp.items()})
        row.update({f"df_{kk}": (round(vv, 4) if isinstance(vv, float) else vv)
                    for kk, vv in deficit_stats(c, pre["gray"], (h, w), pre["avg_bg"]).items()})
        row.update({f"hs_{kk}": (round(vv, 4) if isinstance(vv, float) else vv)
                    for kk, vv in hole_stats(c, binary, (h, w)).items()})
        rows.append(row)

    return dict(
        h=h, w=w, avg_bg=round(float(pre["avg_bg"]), 2), bg_type=pre["bg_type"],
        backdrop_ring_mean=round(ring, 2), corner_ramp=round(ramp, 2),
        corner_std=round(cstd, 2), corner_path_trusted=bool(cstd < 15),
        otsu_t=float(otsu_t), close_k=k,
        polarity="INV" if pre["thresh_type"] == cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU else "BIN",
        n_contours_raw=len(contours), n_valid=len(rows),
        n_need_recovery=sum(1 for r in rows if r["need_recovery"]),
        n_will_hough=sum(1 for r in rows if r.get("will_hough")),
        max_hough_roi_px=max([r["roi_px"] for r in rows if r.get("will_hough")], default=0),
        scan_wall_s=round(time.perf_counter() - t0, 3),
        scan_cpu_s=round(time.process_time() - c0, 3),
        contours=rows,
    )


# ──────────────────────────────── driver ──────────────────────────────────

def iter_sides(paths: List[Path], layout: str):
    for p in paths:
        img = _load_and_resize(str(p))
        if img is None:
            continue
        if layout == "half":
            for side, half in split_sides(img).items():
                yield p.stem, side, half
        else:
            yield p.stem, "single", img


def cmd_scan(a):
    paths = sorted(Path(a.images).glob(a.glob))
    if a.ids:
        keep = set(a.ids.split(","))
        paths = [p for p in paths if p.stem in keep]
    if a.stride > 1:
        paths = paths[:: a.stride]
    if a.limit:
        paths = paths[: a.limit]
    print(f"[scan] {len(paths)} images, layout={a.layout}, house={a.house}", flush=True)

    out, t0 = [], time.perf_counter()
    for i, (sid, side, im) in enumerate(iter_sides(paths, a.layout)):
        m = measure_side(im, a.house)
        m["id"], m["side"], m["src"] = sid, side, str(a.images)
        out.append(m)
        if (i + 1) % 25 == 0:
            print(f"  {i+1} sides  {time.perf_counter()-t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    flat = a.out.replace(".json", ".csv")
    cols = ["id", "side", "h", "w", "avg_bg", "polarity", "otsu_t", "close_k",
            "n_valid", "n_need_recovery", "n_will_hough", "max_hough_roi_px",
            "scan_wall_s", "scan_cpu_s"]
    with open(flat, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        for m in out:
            wtr.writerow(m)
    n_stall = sum(1 for m in out if m["n_will_hough"] > 0)
    print(f"[scan] {len(out)} sides, {n_stall} ({100*n_stall/max(1,len(out)):.1f}%) "
          f"will invoke HoughCircles -> {a.out}", flush=True)


# ─────────────────────────── mechanism classifier ─────────────────────────
#
# Rules are ordered; first match wins. Every threshold below is read off the
# measured distributions in specs/results/rim_stall_taxonomy_2026-07-23.md,
# not chosen a priori.

EXPENSIVE_ROI_PX = 600_000
"""Cost tier boundary. Measured, not chosen: on the KS-17 baseline timings the
distribution of Hough ROI area is bimodal with an EMPTY gap between ~100k and
~600k px; below it median cost is 0.31 CPU-s, above 1M px it is 98 CPU-s. Any
threshold inside the gap gives the same split (97.6% agreement with measured
stalls, zero false negatives). See specs/results/rim_stall_taxonomy_2026-07-23.md."""


def classify_contour(r: dict, side: dict) -> str:
    """
    Assign a MECHANISM. Ordered rules, first match wins.

    Written after looking at the overlays, not before: the a-priori folklore
    list (attached shadow / rim-toning / specular holes / gradient background
    / holder artifact / multi-object) presupposes the blob is approximately
    the coin. On this corpus that presupposition is usually false, so the
    classes below are named for what the blob ACTUALLY is.
    """
    af = r["area_frac_frame"]
    spike = r.get("rp_max_spike_run_deg", 0.0) or 0.0
    hole = r.get("hs_largest_hole_frac", 0.0) or 0.0
    blob_mean = r.get("df_blob_mean", 0.0) or 0.0
    ring = side.get("backdrop_ring_mean", side.get("avg_bg", 0.0))
    cv_r = r.get("rp_cv_r", 1.0)
    n_spike = r.get("rp_n_spike_runs", 0) or 0
    bw, bh = r["bbox"][2], r["bbox"][3]
    aspect = max(bw, bh) / max(1, min(bw, bh))

    # 1. The blob IS the backdrop. The scene threshold landed inside the
    #    background's own luminance ramp, so the brighter part of the
    #    backdrop became foreground and the coin is a hole punched in it.
    if af > 0.18 and hole > 0.10:
        return "backdrop_vignette_blob"

    # 2. Several coins welded into one blob by MORPH_CLOSE (group-lot plates,
    #    multi-coin rows). The give-away is an elongated bbox: one coin --
    #    round OR klippe -- is near-square, a welded row is not.
    #    NB `roi_w > k*enc_r` does NOT work as a test here: roi_w is ~2.2x
    #    enc_r for every compact blob, so such a rule fires on everything.
    #
    #    Second, scale-free test: enc_r / sqrt(area/pi) — how much bigger the
    #    enclosing circle is than an equal-area disc. 1.0 for one round coin,
    #    ~1.25 for a klippe, ~1.41 for two coins side by side, ~1.73 for three
    #    in a row. That quantity is exactly 1/sqrt(area_ratio), so the test
    #    below reads area_ratio < 0.55 <=> enc_r ratio > 1.35.
    if af > 0.05 and (aspect >= 1.5 or r["area_ratio"] < 0.55):
        return "multi_object_weld"

    # 3. The contour IS the coin and the coin IS round. Every ray reaches
    #    nearly the same radius (cv_r tiny), but the boundary wiggles at pixel
    #    scale because the coin/backdrop contrast is too low for a clean
    #    threshold — so the perimeter inflates and 4*pi*A/P^2 collapses.
    #    The segmentation is CORRECT; the metric is what fails. Ordered ahead
    #    of relief because a relief fragment is never disc-shaped.
    if (cv_r < 0.06 and af > 0.02 and hole < 0.06
            and r["solidity"] > 0.80 and r["edge_support"] > 0.40):
        return "low_contrast_coastline"

    # 4. The contour is the coin, and the coin is genuinely NOT a circle:
    #    klippe/square strikes (4 corner excursions), irregular hand-struck
    #    ancient flans. Compact and convex, but the radius really does vary.
    if (cv_r >= 0.06 and n_spike >= 2 and r["solidity"] > 0.75
            and af > 0.02 and hole < 0.06):
        return "non_circular_flan"

    # 5. The blob is a piece of the coin's own RELIEF. The threshold landed
    #    inside the coin: solidly filled, brighter than the backdrop, shaped
    #    like engraved iconography (cv_r large), among many sibling fragments.
    if blob_mean > ring + 8 and hole < 0.06 and side["n_valid"] >= 4:
        return "relief_self_segmentation"

    # 6. Sub-coin debris: speckle, sensor noise, JPEG artifacts, dust, and
    #    small fragments of relief. Individually cheap (tiny ROI) but they
    #    dominate the COUNT of Hough invocations.
    if af < 0.005:
        return "sub_coin_noise_blob"

    # 7. Compact attached lobe on an otherwise coin-sized blob: cast shadow,
    #    holder edge, tag, scale bar, house watermark.
    if spike >= 25.0:
        return "attached_artifact"

    return "unclassified_ragged"


def cmd_classify(a):
    sides = json.load(open(a.scan))
    rows = []
    for s in sides:
        for r in s["contours"]:
            if not r.get("will_hough"):
                continue
            rows.append(dict(
                id=s["id"], side=s["side"], polarity=s["polarity"],
                avg_bg=s["avg_bg"],
                backdrop_ring_mean=s.get("backdrop_ring_mean"),
                corner_path_trusted=s.get("corner_path_trusted"),
                contour_idx=r["contour_idx"],
                mechanism=classify_contour(r, s),
                cost_tier=("expensive" if r["roi_px"] >= EXPENSIVE_ROI_PX else "cheap"),
                hole_frac=r.get("hs_hole_frac"),
                largest_hole_frac=r.get("hs_largest_hole_frac"),
                n_holes=r.get("hs_n_holes"),
                area_frac_frame=round(r["area_frac_frame"], 5),
                circularity=r["circularity"], area_ratio=r["area_ratio"],
                geo_conf=r.get("geo_conf"), edge_support=r["edge_support"],
                enc_r=r["enc_r"], roi_px=r["roi_px"],
                touches_border=r["touches_border"],
                max_dip_run_deg=r.get("rp_max_dip_run_deg"),
                max_spike_run_deg=r.get("rp_max_spike_run_deg"),
                n_spike_runs=r.get("rp_n_spike_runs"), cv_r=r.get("rp_cv_r"),
                solidity=r.get("solidity"), roi_px_=r.get("roi_px"),
                deficit_frac_bglike=r.get("df_deficit_frac_bglike"),
                deficit_frac_darker=r.get("df_deficit_frac_darker"),
                blob_mean=r.get("df_blob_mean"), deficit_mean=r.get("df_deficit_mean"),
            ))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        wtr.writeheader()
        wtr.writerows(rows)
    from collections import Counter
    exp = [r for r in rows if r["cost_tier"] == "expensive"]
    print(f"[classify] {len(rows)} Hough-triggering contours over "
          f"{len({(r['id'],r['side']) for r in rows})} sides "
          f"({len(exp)} expensive, roi >= {EXPENSIVE_ROI_PX:,}px)")
    for title, sel in (("ALL triggering contours", rows),
                       ("EXPENSIVE only (these are the cost)", exp)):
        print(f"  --- {title} (n={len(sel)}) ---")
        for m, n in Counter(r["mechanism"] for r in sel).most_common():
            print(f"    {n:5d}  {100*n/max(1,len(sel)):5.1f}%  {m}")
    print(f"-> {a.out}")


# ────────────────────────────── overlay panel ─────────────────────────────

def cmd_overlay(a):
    """Render image + Otsu blob + failing metrics + what recovery chose."""
    from src.rim_logic import recover_rim
    from src.math_utils import validate_rim_recovery

    sides = json.load(open(a.scan))
    if a.only:
        keep = set(a.only.split(","))
        sides = [s for s in sides if f"{s['id']}:{s['side']}" in keep or s["id"] in keep]
    sides = [s for s in sides if s["n_will_hough"] > 0][: a.limit] if a.limit else \
            [s for s in sides if s["n_will_hough"] > 0]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = {}
    if a.classify_csv and Path(a.classify_csv).exists():
        for row in csv.DictReader(open(a.classify_csv)):
            labels[(row["id"], row["side"], int(row["contour_idx"]))] = row["mechanism"]

    for s in sides:
        p = Path(a.images) / f"{s['id']}.jpg"
        if not p.exists():
            cands = list(Path(a.images).glob(f"{s['id']}.*"))
            if not cands:
                continue
            p = cands[0]
        img = _load_and_resize(str(p))
        im = split_sides(img)[s["side"]] if s["side"] in ("obv", "rev") else img
        pre = preamble(im)
        _, _, binary, contours = segment(pre, a.house)
        vis = im.copy()
        # dim non-blob area so the Otsu blob reads at a glance
        tint = vis.copy()
        tint[binary == 0] = (tint[binary == 0] * 0.45).astype(np.uint8)
        vis = tint
        y = 34
        for r in s["contours"]:
            c = contours[r["contour_idx"]]
            trig = bool(r.get("will_hough"))
            col = (0, 0, 255) if trig else (0, 200, 255)
            cv2.drawContours(vis, [c], -1, col, 3)
            (ccx, ccy), enc_r = cv2.minEnclosingCircle(c)
            cv2.circle(vis, (int(ccx), int(ccy)), int(enc_r), (255, 180, 0), 2)
            if trig:
                t0 = time.perf_counter()
                new_c, conf = recover_rim(im, c)
                dt = time.perf_counter() - t0
                ok = new_c is not None and validate_rim_recovery(new_c, c, (pre["h"], pre["w"]))
                if ok:
                    cv2.drawContours(vis, [new_c], -1, (0, 255, 0), 3)
                mech = labels.get((s["id"], s["side"], r["contour_idx"]), "?")
                for ln in [
                    f"c{r['contour_idx']} {mech}",
                    f"  circ={r['circularity']:.3f} area_ratio={r['area_ratio']:.3f} "
                    f"geo_conf={r.get('geo_conf')} edge_sup={r['edge_support']:.3f}",
                    f"  area_frac={r['area_frac_frame']:.4f} roi={r['roi_w']}x{r['roi_h']} "
                    f"dip={r.get('rp_max_dip_run_deg',0):.0f}d spike={r.get('rp_max_spike_run_deg',0):.0f}d",
                    f"  RECOVERY: {'ACCEPTED' if ok else 'REJECTED/none'} conf={conf:.3f}  {dt:.1f}s",
                ]:
                    cv2.putText(vis, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(vis, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    y += 26
                y += 6
        hdr = (f"{s['id']} {s['side']}  bg={s['avg_bg']} pol={s['polarity']} "
               f"otsu={s['otsu_t']:.0f} valid={s['n_valid']} hough={s['n_will_hough']}")
        cv2.putText(vis, hdr, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hdr, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(outdir / f"{s['id']}_{s['side']}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"  {s['id']} {s['side']}", flush=True)


def cmd_montage(a):
    """Contact sheet per mechanism class."""
    labels: Dict[str, List[str]] = {}
    for row in csv.DictReader(open(a.classify_csv)):
        labels.setdefault(row["mechanism"], []).append(f"{row['id']}_{row['side']}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for mech, keys in labels.items():
        tiles = []
        for k in dict.fromkeys(keys):
            p = Path(a.overlays) / f"{k}.jpg"
            if not p.exists():
                continue
            im = cv2.imread(str(p))
            if im is None:
                continue
            s = a.tile / max(im.shape[:2])
            tiles.append(cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s))))
            if len(tiles) >= a.max_tiles:
                break
        if not tiles:
            continue
        cols = a.cols
        th = max(t.shape[0] for t in tiles)
        tw = max(t.shape[1] for t in tiles)
        rows_n = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows_n * th, cols * tw, 3), 30, np.uint8)
        for i, t in enumerate(tiles):
            ry, cx = divmod(i, cols)
            sheet[ry * th:ry * th + t.shape[0], cx * tw:cx * tw + t.shape[1]] = t
        out = outdir / f"montage_{a.prefix}{mech}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 78])
        print(f"  {mech}: {len(tiles)} tiles -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--images", required=True)
    s.add_argument("--glob", default="*.jpg")
    s.add_argument("--layout", choices=["half", "single"], default="half")
    s.add_argument("--house", default=None)
    s.add_argument("--ids", default=None)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_scan)

    c = sub.add_parser("classify")
    c.add_argument("--scan", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_classify)

    o = sub.add_parser("overlay")
    o.add_argument("--scan", required=True)
    o.add_argument("--images", required=True)
    o.add_argument("--outdir", required=True)
    o.add_argument("--house", default=None)
    o.add_argument("--classify-csv", default=None)
    o.add_argument("--only", default=None)
    o.add_argument("--limit", type=int, default=0)
    o.set_defaults(func=cmd_overlay)

    m = sub.add_parser("montage")
    m.add_argument("--classify-csv", required=True)
    m.add_argument("--overlays", required=True)
    m.add_argument("--outdir", required=True)
    m.add_argument("--cols", type=int, default=3)
    m.add_argument("--tile", type=int, default=520)
    m.add_argument("--max-tiles", type=int, default=12)
    m.add_argument("--prefix", default="", help="disambiguate montages from different fixtures")
    m.set_defaults(func=cmd_montage)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
