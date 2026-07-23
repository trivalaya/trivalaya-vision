"""
Mechanism #1 -- the rim-recovery shape guard (src/rim_shape_guard.py).

Env-gated, DEFAULT-OFF. When on it suppresses ONLY the Hough branch of
`rim_logic.recover_rim` (the geometric fit always still runs), and only for seed
blobs that are provable discs (cv_r < 0.06 AND area_ratio >= 0.55, plus an
optional largest_hole_frac conjunct). Doctrine + measurement:
specs/results/rim_stall_taxonomy_2026-07-23.md §5-§7.

Two layers of tests:
  - PREDICATE: is_disc / radial_cv / largest_hole_frac on directly-constructed
    contours, so the shape thresholds are pinned without fighting MORPH_CLOSE.
  - PLUMBING: recover_rim(skip_hough=...) and the pass-2 call site, with spies on
    hough_rim_recovery / geometric_fit_recovery, proving the guard suppresses
    exactly the Hough branch and nothing else, and stays inert on a non-disc
    (bitten) coin and when the env flag is unset.
"""
import cv2
import numpy as np
import pytest

import src.rim_logic as rim_logic
import src.rim_shape_guard as guard
from src.config import Layer1Config
from tests.conftest import _preprocess

ENV_GUARD = guard.ENV_GUARD
ENV_HOLE = guard.ENV_MAX_HOLE_FRAC
ENV_CV_MIN = guard.ENV_CV_MIN

BG, FG = 40, 210
SHAPE = (300, 300)


# ----------------------------- contour builders ----------------------------

def circle_contour(R=100, cx=150, cy=150, spikes=0, spike_h=1.32, spike_w=0.05, n=720):
    """A filled-disc boundary, optionally with a few narrow OUTWARD spikes.

    Narrow spikes pull minEnclosingCircle (dropping area_ratio into the
    need-recovery window) while barely moving the median radius (cv_r stays
    small) -- exactly the `low_contrast_coastline` morphology from the taxonomy.
    """
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.full_like(th, float(R))
    for k in range(spikes):
        a0 = k * 2 * np.pi / spikes
        d = np.angle(np.exp(1j * (th - a0)))
        r += (spike_h - 1) * R * np.exp(-(d / spike_w) ** 2)
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32).reshape(-1, 1, 2)


def bite_contour(R=100, cx=150, cy=150, bite_deg=120, n=720):
    """A disc with one wide inward bite -- genuinely NOT a disc: the rays into
    the bite reach the inner edge, so cv_r is large."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.full_like(th, float(R))
    d = np.angle(np.exp(1j * th))
    r[(d > -np.deg2rad(bite_deg) / 2) & (d < np.deg2rad(bite_deg) / 2)] = R * 0.45
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1).astype(np.int32).reshape(-1, 1, 2)


def render_bitten_coin(R=90, bite_frac=0.32, W=520, H=300):
    """A real image of one bitten coin -- reliably triggers need_recovery and
    reliably runs the Hough branch (geo_conf ~0.39 <= 0.65)."""
    img = np.full((H, W, 3), BG, np.uint8)
    cx, cy = W // 2, H // 2
    cv2.circle(img, (cx, cy), R, (FG,) * 3, -1)
    ax = (int(R * 1.3), int(R * 1.3))
    half = bite_frac * 360 / 2
    cv2.ellipse(img, (cx, cy), ax, 0, -half, half, (BG,) * 3, -1)
    return img


def otsu_seed(img):
    """The largest post-MORPH_CLOSE Otsu contour + the binary, matching what
    pass 1 hands to recover_rim."""
    ge, ez, tt, h, w = _preprocess(img)
    blur = cv2.GaussianBlur(ge, (7, 7), 0)
    _, binary = cv2.threshold(blur, 0, 255, tt)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=Layer1Config.CLOSE_ITERATIONS)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea), binary


def _ar(cnt):
    a = cv2.contourArea(cnt)
    (_, _), er = cv2.minEnclosingCircle(cnt)
    return a / (np.pi * er * er) if er > 0 else 1.0


# --------------------------------- predicate --------------------------------

def test_perfect_circle_is_disc():
    c = circle_contour()
    assert guard.radial_cv(c, SHAPE) < guard.CV_MIN_DEFAULT
    assert _ar(c) >= guard.AREA_RATIO_FLOOR_DEFAULT
    assert guard.is_disc(c, SHAPE)


def test_spiky_disc_is_disc():
    """3 narrow spikes: area_ratio in [0.55, 0.85), cv_r still < 0.06 -- the
    coastline case the guard is built to catch."""
    c = circle_contour(spikes=3, spike_h=1.32, spike_w=0.05)
    ar = _ar(c)
    assert 0.55 <= ar < 0.85, ar
    assert guard.radial_cv(c, SHAPE) < guard.CV_MIN_DEFAULT
    assert guard.is_disc(c, SHAPE)


def test_bitten_arc_is_not_disc():
    """A wide inward bite drives cv_r well above the threshold; the guard must
    NOT treat it as a disc, or it would disarm real rim recovery."""
    c = bite_contour(bite_deg=120)
    assert guard.radial_cv(c, SHAPE) > guard.CV_MIN_DEFAULT
    assert not guard.is_disc(c, SHAPE)


def test_degenerate_contour_is_not_disc():
    tiny = np.array([[[10, 10]], [[12, 10]], [[12, 12]], [[10, 12]]], dtype=np.int32)
    assert guard.radial_cv(tiny, SHAPE) is None
    assert not guard.is_disc(tiny, SHAPE)


def test_largest_hole_frac_measures_interior_hole():
    filled = np.zeros(SHAPE, np.uint8)
    cv2.circle(filled, (150, 150), 100, 255, -1)
    binary_holed = filled.copy()
    cv2.circle(binary_holed, (150, 150), 45, 0, -1)  # punch a hole in the fg
    c = circle_contour()
    assert guard.largest_hole_frac(c, binary_holed, SHAPE) > 0.10
    assert guard.largest_hole_frac(c, filled, SHAPE) == 0.0


# ------------------------------ env gating ---------------------------------

def test_env_off_never_skips(monkeypatch):
    monkeypatch.delenv(ENV_GUARD, raising=False)
    c = circle_contour()  # a perfect disc
    binary = np.zeros(SHAPE, np.uint8)
    cv2.drawContours(binary, [c], -1, 255, -1)
    assert not guard.should_skip_hough(c, binary, SHAPE)


def test_env_on_skips_disc_not_bitten(monkeypatch):
    monkeypatch.setenv(ENV_GUARD, "1")
    binary = np.zeros(SHAPE, np.uint8)
    disc = circle_contour(spikes=3)
    cv2.drawContours(binary, [disc], -1, 255, -1)
    assert guard.should_skip_hough(disc, binary, SHAPE)

    bitten = bite_contour(bite_deg=120)
    b2 = np.zeros(SHAPE, np.uint8)
    cv2.drawContours(b2, [bitten], -1, 255, -1)
    assert not guard.should_skip_hough(bitten, b2, SHAPE)


def test_hole_conjunct_excludes_backdrop_like_blob(monkeypatch):
    """With the MAX_HOLE_FRAC conjunct set, a disc-shaped blob with a large
    interior hole (the backdrop_vignette_blob signature) is NOT skipped; without
    the conjunct it would be. This is §7 Bar-1's pre-approved tightening."""
    monkeypatch.setenv(ENV_GUARD, "1")
    c = circle_contour()
    filled = np.zeros(SHAPE, np.uint8)
    cv2.circle(filled, (150, 150), 100, 255, -1)
    holed = filled.copy()
    cv2.circle(holed, (150, 150), 45, 0, -1)

    monkeypatch.delenv(ENV_HOLE, raising=False)
    assert guard.should_skip_hough(c, holed, SHAPE)          # base disc test skips

    monkeypatch.setenv(ENV_HOLE, "0.10")
    assert not guard.should_skip_hough(c, holed, SHAPE)      # conjunct rescues it
    assert guard.should_skip_hough(c, filled, SHAPE)         # hole-free disc still skipped


def test_thresholds_env_overridable(monkeypatch):
    monkeypatch.setenv(ENV_GUARD, "1")
    c = circle_contour(spikes=3)  # cv_r ~0.058
    binary = np.zeros(SHAPE, np.uint8)
    cv2.drawContours(binary, [c], -1, 255, -1)
    assert guard.should_skip_hough(c, binary, SHAPE)
    monkeypatch.setenv(ENV_CV_MIN, "0.02")  # tighten below this blob's cv_r
    assert not guard.should_skip_hough(c, binary, SHAPE)


# ------------------------------ recover_rim plumbing ------------------------

@pytest.fixture
def spies(monkeypatch):
    calls = {"hough": 0, "geo": 0}
    oh, og = rim_logic.hough_rim_recovery, rim_logic.geometric_fit_recovery

    def sh(*a, **k):
        calls["hough"] += 1
        return oh(*a, **k)

    def sg(*a, **k):
        calls["geo"] += 1
        return og(*a, **k)

    monkeypatch.setattr(rim_logic, "hough_rim_recovery", sh)
    monkeypatch.setattr(rim_logic, "geometric_fit_recovery", sg)
    return calls


def test_recover_rim_default_runs_hough(spies):
    img = render_bitten_coin()
    seed, _ = otsu_seed(img)
    rim_logic.recover_rim(img, seed)  # skip_hough default False
    assert spies["hough"] == 1
    assert spies["geo"] == 1


def test_recover_rim_skip_hough_suppresses_only_hough(spies):
    img = render_bitten_coin()
    seed, _ = otsu_seed(img)
    geo_c, _ = rim_logic.geometric_fit_recovery(img, seed)
    spies["hough"] = spies["geo"] = 0

    out_c, _ = rim_logic.recover_rim(img, seed, skip_hough=True)
    assert spies["hough"] == 0, "Hough branch must not run when skip_hough=True"
    assert spies["geo"] == 1, "geometric fit must still run"
    # geometric-fit-only: recover_rim returns exactly the geo branch's contour.
    assert (out_c is None) == (geo_c is None)
    if geo_c is not None:
        assert np.array_equal(out_c, geo_c)


# ------------------------------ pass-2 wiring -------------------------------

def _segment_spying(img, monkeypatch):
    """Run the real pass-2 (_segment_and_extract_candidates) counting Hough."""
    from tests.conftest import segment
    calls = {"hough": 0}
    oh = rim_logic.hough_rim_recovery

    def sh(*a, **k):
        calls["hough"] += 1
        return oh(*a, **k)

    monkeypatch.setattr(rim_logic, "hough_rim_recovery", sh)
    cands, _ = segment(img)
    return cands, calls


def test_pass2_env_off_runs_hough(monkeypatch):
    monkeypatch.delenv(ENV_GUARD, raising=False)
    cands, calls = _segment_spying(render_bitten_coin(), monkeypatch)
    assert cands, "bitten coin should still be detected"
    assert calls["hough"] >= 1, "guard off => pass 2 runs Hough exactly as today"


def test_pass2_guard_on_declines_on_nondisc(monkeypatch):
    """Bar 2 in miniature: guard ON but the blob is a bitten (non-disc) coin, so
    the guard declines and rim recovery still gets its Hough branch."""
    monkeypatch.setenv(ENV_GUARD, "1")
    cands, calls = _segment_spying(render_bitten_coin(), monkeypatch)
    assert cands
    assert calls["hough"] >= 1, "guard must not disarm recovery on a non-disc"


def test_pass2_guard_skips_hough_when_disc(monkeypatch):
    """Prove pass 2 threads skip_hough through to recover_rim: with the guard on
    and the disc predicate forced True, the Hough branch is suppressed."""
    monkeypatch.setenv(ENV_GUARD, "1")
    monkeypatch.setattr(guard, "is_disc", lambda *a, **k: True)
    cands, calls = _segment_spying(render_bitten_coin(), monkeypatch)
    assert cands, "geometric fit still recovers, so the coin is still detected"
    assert calls["hough"] == 0, "disc => Hough branch suppressed in pass 2"
