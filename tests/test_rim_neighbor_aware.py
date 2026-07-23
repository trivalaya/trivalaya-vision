"""
Scope B -- neighbor-aware rim recovery, synthetic fixture (deterministic).

Constructs two coins whose TRUE circles genuinely overlap by `overlap_px`,
then erases a strip at their shared boundary wide enough that Otsu/
MORPH_CLOSE segments them as two separate, non-circular ("bitten") blobs --
each blob independently triggers Layer 1.5 rim recovery, which (correctly,
from a single-candidate point of view) restores each coin toward its true
full-circle rim. Since the true circles overlap by construction, naive
recovery reproduces that overlap; the neighbor-aware guard
(TRIVALAYA_RIM_NEIGHBOR_GUARD) must reject it.

Mirrors this defect's real-world mechanism (specs/rim_recovery_neighbor_
aware.md "Measured evidence"): toning/patina eating a chunk out of a flan on
the side facing its neighbour lets Otsu see two clean, separate blobs, and
rim recovery -- designed to restore a coin's true rim -- undoes that
apparent separation.

(overlap_px, erase_half_width) pairs below were found by a parameter sweep
(2026-07-22, not committed -- scratch) that confirmed BOTH coins trigger
`need_recovery` (circularity < 0.65 and area_ratio < 0.85) while still
clearing MIN_AREA_PX; see specs/rim_recovery_neighbor_aware.md Scope B.
"""

import cv2
import numpy as np
import pytest

from tests.conftest import segment

BG, FG = 40, 210
ENV_GUARD = "TRIVALAYA_RIM_NEIGHBOR_GUARD"
ENV_OVERLAP_MAX = "TRIVALAYA_RIM_NEIGHBOR_OVERLAP_MAX"

# (overlap_px, erase_half_width) -- both trigger need_recovery on both coins.
OVERLAPPING_PAIRS = [(60, 40), (90, 30)]


def _render_overlapping_pair(scale: int, radius: int, overlap_px: int,
                              erase_half_width: int) -> np.ndarray:
    """
    Two coins whose TRUE circles overlap by `overlap_px`, with the shared
    boundary erased (width 2*erase_half_width) so Otsu/MORPH_CLOSE segments
    them as two separate, non-circular ("bitten") blobs pre-recovery.
    """
    r = radius
    h = max(2 * r + 40, int(scale * 0.45))
    img = np.full((h, scale, 3), BG, np.uint8)
    cy = h // 2
    d = 2 * r - overlap_px
    cx_mid = scale // 2
    cA, cB = (cx_mid - d // 2, cy), (cx_mid + d // 2, cy)
    cv2.circle(img, cA, r, (FG,) * 3, -1)
    cv2.circle(img, cB, r, (FG,) * 3, -1)
    boundary_x = (cA[0] + cB[0]) // 2
    cv2.rectangle(img, (boundary_x - erase_half_width, 0),
                  (boundary_x + erase_half_width, h), (BG,) * 3, -1)
    return img


def _fill(contour: np.ndarray, shape: tuple) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    return m


def _overlap_frac(cands: list, shape: tuple):
    """Undilated overlap as a fraction of the smaller candidate's area --
    matches tools/two_coin_weld_mask_gate.py's sliver convention."""
    if len(cands) != 2:
        return None
    m0, m1 = _fill(cands[0]["contour"], shape), _fill(cands[1]["contour"], shape)
    inter = int(np.count_nonzero(m0 & m1))
    a0, a1 = int(np.count_nonzero(m0)), int(np.count_nonzero(m1))
    return inter / min(a0, a1) if min(a0, a1) else 0.0


@pytest.mark.parametrize("overlap_px,erase_half_width", OVERLAPPING_PAIRS)
def test_naive_recovery_overlaps_the_neighbor(overlap_px, erase_half_width):
    """
    Precondition for the whole fixture: guard OFF (today's default) must
    reproduce the defect -- both coins recovered, and the recovered
    contours genuinely overlap (undilated, real alpha contamination).
    """
    img = _render_overlapping_pair(500, 60, overlap_px, erase_half_width)
    cands, _ = segment(img)
    assert len(cands) == 2
    assert all(c["debug_data"]["rim_recovered"] for c in cands)
    frac = _overlap_frac(cands, img.shape[:2])
    assert frac > 0, "fixture didn't reproduce the defect: overlap must be > 0 with the guard off"


@pytest.mark.parametrize("overlap_px,erase_half_width", OVERLAPPING_PAIRS)
def test_neighbor_guard_reverts_the_overlap_to_zero(overlap_px, erase_half_width, monkeypatch):
    """The fix, stated as one assertion: guard ON drives the same overlap to exactly zero."""
    img = _render_overlapping_pair(500, 60, overlap_px, erase_half_width)
    monkeypatch.setenv(ENV_GUARD, "1")
    cands, _ = segment(img)
    assert len(cands) == 2, "the guard must revert a candidate, not drop it"
    assert _overlap_frac(cands, img.shape[:2]) == 0.0
    assert all(c["debug_data"].get("rim_neighbor_reverted") for c in cands)


def test_guard_off_by_default_is_bit_identical_to_today():
    """No env override at all must reproduce today's production behavior exactly."""
    img = _render_overlapping_pair(500, 60, 90, 30)
    cands, _ = segment(img)
    assert all(c["debug_data"]["rim_recovered"] for c in cands)
    assert not any(c["debug_data"].get("rim_neighbor_reverted") for c in cands)
    assert _overlap_frac(cands, img.shape[:2]) > 0


def test_guard_respects_configurable_threshold(monkeypatch):
    """
    A threshold set ABOVE the fixture's actual overlap fraction must not
    revert anything -- the guard is a real threshold comparison, not an
    unconditional revert-on-any-recovery switch.
    """
    img = _render_overlapping_pair(500, 60, 90, 30)
    cands_off, _ = segment(img)
    frac = _overlap_frac(cands_off, img.shape[:2])

    monkeypatch.setenv(ENV_GUARD, "1")
    monkeypatch.setenv(ENV_OVERLAP_MAX, str(frac + 0.10))
    cands, _ = segment(img)
    assert not any(c["debug_data"].get("rim_neighbor_reverted") for c in cands)
