"""
Rim-recovery shape guard (mechanism #1 of the 2026-07-23 stall taxonomy).

Env-gated, DEFAULT-OFF. When enabled it suppresses ONLY the expensive Hough
branch inside `rim_logic.recover_rim` -- the geometric fit always still runs --
and only for seed blobs that are *provably discs*. The motivating measurement:
on the KS-17 corpus the contours paying ~40% of the entire Hough bill are round
coins segmented correctly, whose low circularity is a resolution artefact of the
perimeter term (4*pi*A/P^2), not a shape fact. `recover_rim` already returns the
geometric fit 17 of 21 times on the expensive tier; Hough burns 40-200 CPU-s and
is then discarded. See specs/results/rim_stall_taxonomy_2026-07-23.md §5-§7.

The disc test is scale-invariant by construction:

    cv_r < CV_MIN            (coefficient of variation of the blob radius about
                              its centroid -- a disc has near-constant radius;
                              perimeter never enters, so resolution never enters)
    AND area_ratio >= AR_FLOOR   (blob fills its enclosing circle -- a genuinely
                              bitten/fragmented coin does NOT, so the case rim
                              recovery exists to serve is never caught here)

Optional third conjunct, carried as a config knob (owner ratification 2026-07-23,
§7 Bar 1 pre-approved path): when TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_MAX_HOLE_FRAC
is set, additionally require `largest_hole_frac < <value>`. This excludes the
`backdrop_vignette_blob` class (a frame-spanning background region with a
coin-shaped hole punched through it), which §6.4 measured as the one class where
Hough's answer is sometimes the one recover_rim actually returns.

cv_r and largest_hole_frac are computed here identically to
tools/rim_stall_taxonomy.py's `radial_profile` / `hole_stats`, so the shipped
guard fires on exactly the contours the taxonomy measured.

Guard OFF (env unset) => `should_skip_hough` returns False unconditionally =>
`recover_rim` is bit-identical to today. Nothing in this module runs unless the
env flag is explicitly set.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Disc-test thresholds. Frozen to the values the §6 probe measured (which
# produced 6/6 outcome-identical on the worst class); env-overridable only so a
# future sweep can move them deliberately, never as a default change.
CV_MIN_DEFAULT = 0.06
AREA_RATIO_FLOOR_DEFAULT = 0.55

ENV_GUARD = "TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD"
ENV_CV_MIN = "TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_CV_MIN"
ENV_AR_FLOOR = "TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_AR_FLOOR"
ENV_MAX_HOLE_FRAC = "TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_MAX_HOLE_FRAC"


def _enabled() -> bool:
    return os.environ.get(ENV_GUARD, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default=None):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def radial_cv(contour, shape, n_rays: int = 360):
    """
    Coefficient of variation of the blob radius, measured exactly as
    tools/rim_stall_taxonomy.py::radial_profile does it.

    r(theta) = distance from the blob centroid to its FARTHEST filled pixel
    along each ray. cv_r = std(r>0) / median(r>0). Returns None on a degenerate
    contour (moment 0, too small, or no filled pixels) -- callers treat None as
    "not a disc", so a degenerate seed keeps full recovery.
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
    rv = r[r > 0]
    if rv.size == 0:
        return None
    med = float(np.median(rv))
    if med <= 0:
        return None
    return float(np.std(rv) / med)


def largest_hole_frac(contour, binary, shape) -> float:
    """
    Fraction of the blob's filled area occupied by its single largest interior
    hole, measured exactly as tools/rim_stall_taxonomy.py::hole_stats does it.
    `binary` is the post-MORPH_CLOSE foreground mask. RETR_EXTERNAL counts holes
    as filled, so a blob that is really the backdrop (coin punched out of it)
    scores high; a piece of relief scores ~0.
    """
    h, w = shape
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    n_filled = int(cv2.countNonZero(filled))
    if n_filled == 0:
        return 0.0
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(binary))
    hcs, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [cv2.contourArea(c) for c in hcs if cv2.contourArea(c) > 500]
    return (max(big) / n_filled) if big else 0.0


def is_disc(contour, shape, binary=None) -> bool:
    """
    Pure disc test (no env reading) -- the predicate `should_skip_hough` gates on
    when the guard is enabled. Exposed for unit tests.

    binary is required only when the hole-frac conjunct is active; if that
    conjunct is active and binary is None, returns False (cannot evaluate the
    conjunct -> fail safe -> keep recovery).
    """
    cv_r = radial_cv(contour, shape)
    if cv_r is None:
        return False
    cv_min = _env_float(ENV_CV_MIN, CV_MIN_DEFAULT)
    ar_floor = _env_float(ENV_AR_FLOOR, AREA_RATIO_FLOOR_DEFAULT)

    area = float(cv2.contourArea(contour))
    (_, _), enc_r = cv2.minEnclosingCircle(contour)
    ar = area / (np.pi * enc_r * enc_r) if enc_r > 0 else 1.0
    if not (cv_r < cv_min and ar >= ar_floor):
        return False

    max_hole = _env_float(ENV_MAX_HOLE_FRAC, None)
    if max_hole is not None:
        if binary is None:
            return False
        if not (largest_hole_frac(contour, binary, shape) < max_hole):
            return False
    return True


def should_skip_hough(contour, binary, shape) -> bool:
    """
    True iff the (enabled) guard would suppress the Hough branch for this seed.
    Env unset => always False => recover_rim bit-identical to today.
    """
    if not _enabled():
        return False
    return is_disc(contour, shape, binary=binary)
