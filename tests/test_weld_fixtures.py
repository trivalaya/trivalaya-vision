"""
Tier 2 -- synthetic weld fixtures (deterministic, no network).

Pins the px-bridging model the whole spec rests on ("7x7 at 2 iterations
~= 12px").  If that model is wrong, this fails in seconds rather than after an
1800-lot sweep across nine houses.

Expected outcomes are DERIVED from the closed form in conftest.bridging_px(),
not read from a hardcoded results table: g=12 at k=7 sits exactly on the
survive side of the boundary, and a change to fixture radius or rendering must
not be able to silently flip it.

See specs/two_coin_weld_morph_close.md SS9.2.
"""

import cv2
import numpy as np
import pytest

from src.layer1_geometry import _close_kernel_size, _suppress_background_noise
from tests.conftest import (
    BG_LEVEL,
    ENV_FRAC,
    blob_count_post_close,
    blob_count_pre_close,
    bridging_px,
    render_pair,
    segment,
)

SCALES = [500, 1200, 3000]
# SS9.2's sweep, plus 4 and 11 -- the two boundary cases the closed form calls
# most tightly (k=3 survives at 4, k=7 welds at 11).
GAPS = [3, 4, 5, 7, 9, 11, 12, 15, 25]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("gap", GAPS)
def test_otsu_alone_separates_the_coins(scale, gap):
    """
    Precondition for every weld claim below: thresholding has ALREADY produced
    both coins correctly.  Anything the close then does to that is damage, not
    repair.  Also demonstrates the blur contributes zero bridging on clean
    synthetics -- Otsu re-thresholds the symmetric ramp back to the original
    boundary.
    """
    assert blob_count_pre_close(render_pair(scale, gap)) == 2


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("gap", GAPS)
def test_bridging_model_at_fixed_seven(scale, gap):
    """Today's production kernel: welds iff gap < 12, at every scale."""
    expected = 1 if gap < bridging_px(7) else 2
    assert blob_count_post_close(render_pair(scale, gap)) == expected


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("gap", GAPS)
def test_bridging_model_scale_relative(scale, gap, monkeypatch):
    """
    Same model, with the kernel sized per-image.  The weld signature must
    appear ONLY below the predicted threshold for that scale's kernel.
    """
    monkeypatch.setenv(ENV_FRAC, str(1 / 400))
    img = render_pair(scale, gap)
    k = _close_kernel_size(img.shape[0], img.shape[1])
    expected = 1 if gap < bridging_px(k) else 2
    assert blob_count_post_close(img) == expected


def test_scale_relative_fixes_the_cng_feature_case(monkeypatch):
    """
    The change, stated as one assertion.  cng_feature is 500px wide with a
    measured 5-8px inter-coin gap: welded today (7px < 12px of bridging),
    separated at k=3 (7px > 4px).
    """
    img = render_pair(500, gap=7)
    assert blob_count_pre_close(img) == 2, "Otsu should already find both coins"
    assert blob_count_post_close(img) == 1, "today's 7x7 welds them"

    monkeypatch.setenv(ENV_FRAC, str(1 / 400))
    assert blob_count_post_close(img) == 2, "k=3 must preserve the separation"


def test_kuenker_case_is_unaffected(monkeypatch):
    """The ~25px-gap control: survives under both kernels, at any scale."""
    for scale in SCALES:
        img = render_pair(scale, gap=25)
        assert blob_count_post_close(img) == 2
        monkeypatch.setenv(ENV_FRAC, str(1 / 400))
        assert blob_count_post_close(img) == 2
        monkeypatch.delenv(ENV_FRAC)


# --- SS4.3 equivalence, run in CI rather than only at the manual gate --------

@pytest.mark.parametrize("width", [2400, 3000, 3199])
@pytest.mark.parametrize("gap", [5, 12, 25])
def test_large_format_output_is_identical(width, gap, monkeypatch):
    """
    In the 2400-3199 band the kernel stays 7, so enabling the scale-relative
    path must be a no-op -- bbox-identical output.  Any diff at large scale
    means the kernel math is wrong, not that the policy changed.

    3000 is included explicitly: it is the width where the original round()
    formula diverged, and a sample drawn from ">=2400px" without it would pass
    green on a broken kernel.
    """
    img = render_pair(width, gap)

    before, _ = segment(img)
    monkeypatch.setenv(ENV_FRAC, str(1 / 400))
    after, _ = segment(img)

    assert [c["bbox"] for c in before] == [c["bbox"] for c in after]
    assert [c["geometry"]["area"] for c in before] == \
           [c["geometry"]["area"] for c in after]


# --- SS7.2 fragment cases ---------------------------------------------------

def _fragmented_pair(scale: int, gap: int, band: int) -> np.ndarray:
    """
    A pair where one coin is split in two by a background-coloured band --
    the synthetic stand-in for heavy glare or toning thresholding a coin into
    pieces.  `band` px wide, so a kernel bridging >= band heals it.
    """
    img = render_pair(scale, gap)
    h = img.shape[0]
    r = max(20, int(scale * 0.18))
    x0 = (scale - (4 * r + gap)) // 2
    cx = x0 + r  # centre of the left coin
    cv2.rectangle(img, (cx - band // 2, 0), (cx + band - band // 2, h),
                  (BG_LEVEL,) * 3, -1)
    return img


def test_under_closing_fragments_small_images(monkeypatch):
    """
    SS7.2, the cost of the fix.  A 9px fracture that k=7 heals is left open by
    k=3, turning a 2-coin lot into a 3-blob lot.  Gap is 25px so the coins are
    separate under both kernels and only the fragment effect is in play.

    This is the regression the SS4.2 sweep must track per kernel.  Asserted here
    so the tradeoff is documented in code rather than discovered in production.
    """
    img = _fragmented_pair(500, gap=25, band=9)

    assert len(segment(img)[0]) == 2, "k=7 heals a 9px fracture (9 < 12)"

    monkeypatch.setenv(ENV_FRAC, str(1 / 400))
    frags = segment(img)[0]
    assert len(frags) > 2, "k=3 leaves a 9px fracture open (9 > 4)"

    # The log signal is ndets > 2 PLUS the bbox-area distribution: a fragment
    # crop is far smaller than an intact sibling coin.
    areas = sorted(w * h for _, _, w, h in (c["bbox"] for c in frags))
    assert areas[0] < 0.6 * areas[-1], \
        f"expected a fragment-sized bbox among {areas}"


def test_small_fragment_is_silently_eaten_by_noise_suppression():
    """
    SS7.2's second variant, and the reason `ndets > 2` alone is NOT a sufficient
    dashboard signal: _suppress_background_noise can drop a small weak fragment
    sitting next to a dominant sibling, presenting as a clean-looking ndets==2
    where one detection is fragment-sized.

    Tested against the filter directly rather than through a rendered image:
    synthetic circles have perfect edge support (1.00), and the filter requires
    edge < 0.50 to drop anything, so no clean fixture can reach this branch.
    Real fragments are exactly the weak-edged case the filter targets.
    """
    def cand(area, circ, edge, bbox):
        return {
            "geometry": {"area": area, "circularity": circ},
            "classification": {"confidence": edge},
            "bbox": bbox,
        }

    dominant = cand(25000, 0.90, 0.75, (0, 0, 180, 180))
    sibling = cand(24000, 0.88, 0.72, (220, 0, 180, 180))
    fragment = cand(1500, 0.35, 0.30, (190, 60, 45, 45))  # weak + <20% of area

    kept = _suppress_background_noise([dominant, sibling, fragment])

    assert len(kept) == 2, "the fragment should have been silently dropped"
    assert fragment not in kept
    # ...and the lot now reports a perfectly ordinary-looking ndets == 2.


def test_noise_suppression_keeps_comparable_fragments():
    """
    The complement: half-coin fragments are too big to qualify for suppression,
    which is why that case surfaces as ndets > 2 instead.
    """
    def cand(area, circ, edge, bbox):
        return {
            "geometry": {"area": area, "circularity": circ},
            "classification": {"confidence": edge},
            "bbox": bbox,
        }

    dominant = cand(25000, 0.90, 0.75, (0, 0, 180, 180))
    half_a = cand(11700, 0.68, 0.40, (220, 0, 90, 180))
    half_b = cand(11700, 0.68, 0.40, (320, 0, 90, 180))

    kept = _suppress_background_noise([dominant, half_a, half_b])
    assert len(kept) == 3
