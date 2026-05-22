import cv2
import numpy as np
import logging
from typing import Optional, Tuple

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

    circles = cv2.HoughCircles(
        cv2.GaussianBlur(roi, (5, 5), 0),
        cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=int(expected_r * 1.5),
        param1=100, param2=25,
        minRadius=int(expected_r * 0.7),
        maxRadius=int(expected_r * 1.3),
    )
    if circles is None:
        return None, 0

    # Score each candidate by edge support along its circumference
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    best = (None, 0.0)
    for ccx, ccy, ccr in circles[0]:
        gcx, gcy, gcr = ccx + rx1, ccy + ry1, ccr
        if gcr < 10 or gcr > min(w, h) / 2:
            continue
        if not (0 <= gcx < w and 0 <= gcy < h):
            continue
        ann = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ann, (int(gcx), int(gcy)), int(gcr), 255, 3)
        ann_total = cv2.countNonZero(ann)
        if ann_total == 0:
            continue
        es = cv2.countNonZero(cv2.bitwise_and(ann, edges)) / ann_total
        if es > best[1]:
            best = ((gcx, gcy, gcr), es)

    if best[0] is None or best[1] < 0.12:
        logger.debug(f"Hough rim recovery: best edge_support={best[1]:.3f} (need >= 0.12)")
        return None, 0

    gcx, gcy, gcr = best[0]
    theta = np.linspace(0, 2 * np.pi, 360)
    x_pts = gcx + gcr * np.cos(theta)
    y_pts = gcy + gcr * np.sin(theta)
    contour = np.column_stack((x_pts, y_pts)).astype(np.int32).reshape((-1, 1, 2))
    logger.info(f"Hough rim recovery SUCCESS: r={gcr:.0f}px edge_support={best[1]:.3f}")
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
    geo_c, geo_conf = geometric_fit_recovery(image_bgr, seed_contour)
    hou_c, hou_conf = hough_rim_recovery(image_bgr, seed_contour)

    def _radius(contour):
        if contour is None:
            return 0.0
        (_, _), r = cv2.minEnclosingCircle(contour)
        return float(r)

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
