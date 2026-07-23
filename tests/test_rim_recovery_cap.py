"""
Scope A2 -- capping how many contours per image may invoke rim recovery.

Two well-separated (non-interacting) bitten coins, each independently
triggering `need_recovery`, with different bite depths so their pre-recovery
areas clearly rank. `TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE` unset (default)
must recover both (today's behavior, bit-identical); set to 1, only the
larger-area qualifying contour may attempt recovery -- directly addresses
the "stacks 2-5x per image" cost mechanism in specs/rim_recovery_neighbor_
aware.md Scope A.
"""

import cv2
import numpy as np
import pytest

from tests.conftest import segment

BG, FG = 40, 210
ENV_CAP = "TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE"


def _render_two_independent_bitten(scale: int = 700) -> np.ndarray:
    """Two well-separated coins (no interaction), bitten by different
    amounts so their post-bite areas clearly rank (A bigger than B) while
    both still trigger need_recovery."""
    h = 300
    img = np.full((h, scale, 3), BG, np.uint8)
    cy = h // 2
    rA, rB = 70, 70
    cA, cB = (150, cy), (550, cy)
    cv2.circle(img, cA, rA, (FG,) * 3, -1)
    cv2.circle(img, cB, rB, (FG,) * 3, -1)
    axes_a = (int(rA * 1.3), int(rA * 1.3))
    cv2.ellipse(img, cA, axes_a, 0, -0.35 * 360 / 2, 0.35 * 360 / 2, (BG,) * 3, -1)
    axes_b = (int(rB * 1.3), int(rB * 1.3))
    cv2.ellipse(img, cB, axes_b, 0, -0.40 * 360 / 2, 0.40 * 360 / 2, (BG,) * 3, -1)
    return img


def _by_x(cands):
    """Left-to-right order (A is drawn left of B)."""
    return sorted(cands, key=lambda c: c["bbox"][0])


def test_cap_unset_recovers_both_today():
    img = _render_two_independent_bitten()
    cands, _ = segment(img)
    assert len(cands) == 2
    assert all(c["debug_data"]["rim_recovered"] for c in cands)


def test_cap_one_recovers_only_the_larger_qualifying_contour(monkeypatch):
    img = _render_two_independent_bitten()
    monkeypatch.setenv(ENV_CAP, "1")
    cands, _ = segment(img)
    assert len(cands) == 2, "the cap must skip recovery, not drop a candidate"
    a, b = _by_x(cands)
    assert a["debug_data"]["rim_recovered"], "A has the larger pre-recovery area and should win the cap"
    assert not b["debug_data"]["rim_recovered"]


@pytest.mark.parametrize("cap", ["2", "5"])
def test_cap_at_or_above_qualifying_count_is_a_no_op(cap, monkeypatch):
    """A cap >= the number of qualifying contours must match the unset (today) behavior."""
    img = _render_two_independent_bitten()
    baseline, _ = segment(img)
    monkeypatch.setenv(ENV_CAP, cap)
    capped, _ = segment(img)
    assert [c["debug_data"]["rim_recovered"] for c in _by_x(capped)] == \
           [c["debug_data"]["rim_recovered"] for c in _by_x(baseline)]


def test_cap_zero_recovers_neither(monkeypatch):
    """cap=0 is the most restrictive setting -- nobody attempts recovery, distinct from unset."""
    img = _render_two_independent_bitten()
    monkeypatch.setenv(ENV_CAP, "0")
    cands, _ = segment(img)
    assert len(cands) == 2
    assert not any(c["debug_data"]["rim_recovered"] for c in cands)
