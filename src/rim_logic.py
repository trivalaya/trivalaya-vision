import cv2
import numpy as np
import logging
import os
import time
from typing import Optional, Tuple

# Same gate as src/two_coin_resolver.py — set TRIVALAYA_PERF_LOG=1 to enable.
_PERF = os.environ.get("TRIVALAYA_PERF_LOG", "0") == "1"

try:
    from src.config import RimRecoveryConfig
    from src.math_utils import (
        validate_rim_recovery,
        fit_circle_to_points,
        validate_with_annulus_support
    )
except ImportError:
    logging.warning("Config/Utils import failed. Using defaults.")
    class RimRecoveryConfig:
        EDGE_SUPPORT_MIN = 0.15
        EDGE_SUPPORT_FALLBACK = 0.12
        HOUGH_ROI_CAP = 1280
    
    def validate_rim_recovery(rec, seed, shape): 
        return True
    def fit_circle_to_points(pts):
        return None, None, 0.0
    def validate_with_annulus_support(img, center, radius, band=5):
        return 0.0

logger = logging.getLogger(__name__)


def geometric_fit_recovery(image_bgr, seed_contour):
    """
    PRIMARY: Geometric circle fitting (works on fragments).
    Fits circle to existing points instead of detecting.
    """
    if seed_contour is None or len(seed_contour) < 10:
        return None, 0
    
    h, w = image_bgr.shape[:2]
    points = seed_contour.reshape(-1, 2)
    
    center, radius, fit_conf = fit_circle_to_points(points)
    
    if center is None:
        logger.debug("Geometric fit failed")
        return None, 0
    
    cx, cy = center
    
    if radius < 10 or radius > min(w, h) / 2:
        logger.debug(f"Invalid radius {radius:.0f}")
        return None, 0
    
    if not (0 <= cx < w and 0 <= cy < h):
        logger.debug(f"Center outside image")
        return None, 0
    
    band_width = max(5, int(0.02 * radius))
    edge_support = validate_with_annulus_support(image_bgr, center, radius, band_width)
    
    combined_conf = fit_conf * 0.5 + edge_support * 0.5
    
    logger.info(f"Geometric fit: r={radius:.0f}px, fit={fit_conf:.3f}, edge={edge_support:.3f}, combined={combined_conf:.3f}")
    
    if combined_conf > 0.20:
        theta = np.linspace(0, 2*np.pi, 360)
        x_pts = cx + radius * np.cos(theta)
        y_pts = cy + radius * np.sin(theta)
        contour = np.column_stack((x_pts, y_pts)).astype(np.int32).reshape((-1, 1, 2))
        logger.info(f"SUCCESS: Geometric fit conf={combined_conf:.3f}")
        return contour, combined_conf
    
    return None, 0


def hough_rim_recovery(image_bgr, seed_contour):
    """
    FALLBACK: Hough-circle recovery on the seed bbox ROI.

    Works on the failure mode where the Otsu blob has a chunk eaten out
    (toning/patina inside the flan), so the seed contour is non-circular
    and geometric fit can't find a good circle from the chunk-biased
    points. Hough votes on edge gradients across the full ROI and is
    robust to missing arcs.
    """
    if seed_contour is None or len(seed_contour) < 10:
        return None, 0

    h, w = image_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(seed_contour)
    margin = int(max(bw, bh) * 0.1)
    rx1, ry1 = max(0, x - margin), max(0, y - margin)
    rx2, ry2 = min(w, x + bw + margin), min(h, y + bh + margin)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None, 0

    expected_r = min(bw, bh) / 2.0
    if expected_r < 20:
        return None, 0

    # Perf cap: HoughCircles cost grows ~dim^4 x edge density. Run it on a
    # bounded-resolution copy of the ROI and scale the winner back — the
    # annulus band (max(5, 0.02r)) dwarfs the <=1/s px remap error.
    try:
        cap = int(os.environ.get(
            "TRIVALAYA_RIM_HOUGH_CAP",
            str(getattr(RimRecoveryConfig, "HOUGH_ROI_CAP", 1024))))
    except ValueError:
        cap = 1024
    roi_full = roi
    s = 1.0
    if cap > 0 and max(roi.shape) > cap:
        s = cap / max(roi.shape)
        if expected_r * s < 20:
            s = min(1.0, 20.0 / expected_r)
        roi = cv2.resize(
            roi,
            (max(1, int(roi.shape[1] * s)), max(1, int(roi.shape[0] * s))),
            interpolation=cv2.INTER_AREA)
    expected_r_w = expected_r * s

    # param2 is an absolute accumulator-vote bar; votes scale with the
    # circumference in pixels, so hold the bar constant RELATIVE to
    # resolution when the ROI is downscaled. Over-admission is safe — the
    # full-res edge-support gate below is the acceptance judge.
    p2 = 25 if s == 1.0 else max(12, int(round(25 * s)))
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(roi, (5, 5), 0),
        cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=int(expected_r_w * 1.5),
        param1=100, param2=p2,
        minRadius=int(expected_r_w * 0.7),
        maxRadius=int(expected_r_w * 1.3),
    )
    if circles is None:
        return None, 0

    # Score candidates on the FULL-RES ROI edges: the cap only bounds where
    # the accumulator looks for candidates; acceptance (the 0.12 edge-support
    # gate) is judged at native resolution, same as the uncapped path. When
    # downscaled, each candidate's radius is micro-refined (+/-2px) to absorb
    # the 1/s quantization of the low-res accumulator.
    edges_roi = cv2.Canny(cv2.GaussianBlur(roi_full, (5, 5), 0), 50, 150)
    roi_h, roi_w = edges_roi.shape
    cand = circles[0] if s == 1.0 else circles[0][:8]
    best = (None, 0.0)
    for ccx, ccy, ccr in cand:
        fx, fy, fr0 = ccx / s, ccy / s, ccr / s
        if not (0 <= fx < roi_w and 0 <= fy < roi_h):
            continue
        radii = (fr0,) if s == 1.0 else (fr0 - 2.0, fr0, fr0 + 2.0)
        for fr in radii:
            if fr < 10 or fr > min(roi_h, roi_w) / 2:
                continue
            ann = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.circle(ann, (int(fx), int(fy)), int(fr), 255, 3)
            ann_total = cv2.countNonZero(ann)
            if ann_total == 0:
                continue
            es = cv2.countNonZero(cv2.bitwise_and(ann, edges_roi)) / ann_total
            if es > best[1]:
                # Store full-res ROI coords; remap to image coords after
                best = ((fx, fy, fr), es)

    if best[0] is None or best[1] < 0.12:
        logger.debug(f"Hough rim recovery: best edge_support={best[1]:.3f} (need >= 0.12)")
        return None, 0

    ccx, ccy, ccr = best[0]
    gcx, gcy = ccx + rx1, ccy + ry1  # full-res ROI -> image coords
    theta = np.linspace(0, 2 * np.pi, 360)
    x_pts = gcx + ccr * np.cos(theta)
    y_pts = gcy + ccr * np.sin(theta)
    contour = np.column_stack((x_pts, y_pts)).astype(np.int32).reshape((-1, 1, 2))
    logger.info(f"Hough rim recovery SUCCESS: r={ccr:.0f}px edge_support={best[1]:.3f}")
    return contour, best[1]


def recover_rim(image_bgr, seed_contour):
    """
    Two-stage rim recovery for coin flans whose Otsu blob is non-circular.

    Runs both:
      1. Geometric circle fit on seed-contour points (good on partial-arc
         fragments).
      2. Hough-circle on the bbox ROI (robust to chunks missing inside the
         flan from toning/patina — those eat the Otsu blob but leave the
         physical rim's edge gradient intact for Hough).

    When both succeed, prefer the *larger* circle when it passes annulus
    edge-support: an under-sized fit silently clips real coin (the
    109704-style failure), while an over-sized fit just admits a few
    background pixels around the rim.
    """
    t0 = time.perf_counter() if _PERF else 0
    geo_c, geo_conf = geometric_fit_recovery(image_bgr, seed_contour)
    t_geo = time.perf_counter() - t0 if _PERF else 0

    def _radius(contour):
        if contour is None:
            return 0.0
        (_, _), r = cv2.minEnclosingCircle(contour)
        return float(r)

    # Skip the slow Hough fallback when the geometric fit is highly confident.
    # An "eaten-arc" failure (the 109704 case) shows up as geo_conf <= ~0.55:
    # the seed contour's missing arc pulls the least-squares fit small and
    # combined_conf stays moderate. A clean coin gives geo_conf > 0.65 and
    # we trust it without paying for Hough.
    if geo_c is not None and geo_conf > 0.65:
        if _PERF:
            print(f"  [rim] geo={t_geo*1000:.0f}ms conf={geo_conf:.2f} "
                  f"hough=SKIPPED total={t_geo*1000:.0f}ms")
        return geo_c, geo_conf

    t1 = time.perf_counter() if _PERF else 0
    hou_c, hou_conf = hough_rim_recovery(image_bgr, seed_contour)
    t_hou = time.perf_counter() - t1 if _PERF else 0
    if _PERF:
        total = (time.perf_counter() - t0) * 1000
        print(f"  [rim] geo={t_geo*1000:.0f}ms conf={geo_conf:.2f} "
              f"hough={t_hou*1000:.0f}ms hou_conf={hou_conf:.2f} total={total:.0f}ms")
    geo_r, hou_r = _radius(geo_c), _radius(hou_c)

    # Only one available
    if geo_c is None:
        return hou_c, hou_conf
    if hou_c is None:
        return geo_c, geo_conf

    # Both available. Prefer Hough's bigger circle if (a) it's at least 5%
    # larger than the geometric fit (signal that geo was biased small by a
    # missing arc) and (b) its edge-support cleared the Hough validator.
    if hou_r > geo_r * 1.05 and hou_conf >= 0.12:
        logger.info(
            "rim_recovery: prefer Hough r=%.0f (vs geo r=%.0f) "
            "geo_conf=%.3f hou_conf=%.3f",
            hou_r, geo_r, geo_conf, hou_conf,
        )
        return hou_c, hou_conf
    return geo_c, geo_conf
