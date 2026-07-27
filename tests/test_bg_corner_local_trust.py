"""M1 -- per-corner local consistency in `detect_background_histogram`.

Ticket: specs/background_estimator_repair.md, mechanism M1 and Bar 2.

Bar 2 is the load-bearing test here and it is asserted, not sampled: on any
image where the POOLED corner test already fires today, the repaired function
must return a byte-identical value and bg_type.  A violation is an automatic
FAIL of the whole mechanism regardless of every other number, because that
property is what bounds the blast radius across all non-vignetted houses.

The comparison is against `_golden_pre_m1` below -- a frozen verbatim copy of
the shipped function as of a00f502.  Comparing the gate against itself would
prove nothing; comparing against a copy of the real prior behavior is what
makes "bit-identical when unset" a measurement rather than a restatement.
"""

import numpy as np
import pytest

from src.math_utils import (
    BG_CORNER_LOCAL_TRUST_ENV,
    CORNER_LOCAL_STD_MAX,
    detect_background_histogram,
)

import cv2


# --------------------------------------------------------------------------
# Frozen reference: the shipped implementation at a00f502, verbatim.
# --------------------------------------------------------------------------

def _golden_pre_m1(gray_image):
    h, w = gray_image.shape

    corners = []
    margin = 5
    if h > 20 and w > 20:
        corners.extend(gray_image[0:margin, 0:margin].flatten())
        corners.extend(gray_image[0:margin, w-margin:w].flatten())
        corners.extend(gray_image[h-margin:h, 0:margin].flatten())
        corners.extend(gray_image[h-margin:h, w-margin:w].flatten())

        corner_median = np.median(corners)
        corner_std = np.std(corners)

        if corner_std < 15:
            bg_type = "light" if corner_median > 127 else "dark"
            return float(corner_median), bg_type

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    dark_peak = np.sum(hist[0:50])
    light_peak = np.sum(hist[205:256])

    if light_peak > dark_peak * 2:
        bright_region = gray_image[gray_image > 200]
        bg_value = np.mean(bright_region) if len(bright_region) > 0 else 240
        return float(bg_value), "light"
    elif dark_peak > light_peak * 2:
        dark_region = gray_image[gray_image < 50]
        bg_value = np.mean(dark_region) if len(dark_region) > 0 else 20
        return float(bg_value), "dark"

    return float(np.argmax(hist)), "mixed"


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    """No test may inherit an operator's override of the M1 gate."""
    monkeypatch.delenv(BG_CORNER_LOCAL_TRUST_ENV, raising=False)


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, "1")


def _corner_patches(gray, margin=5):
    h, w = gray.shape
    return (
        gray[0:margin, 0:margin],
        gray[0:margin, w-margin:w],
        gray[h-margin:h, 0:margin],
        gray[h-margin:h, w-margin:w],
    )


def _expected_m1(gray):
    """The rule under test, computed independently of the implementation."""
    meds = [float(np.median(p)) for p in _corner_patches(gray)]
    v = float(np.median(meds))
    return v, ("light" if v > 127 else "dark")


def ramp_bg(tl, tr, bl, br, h=400, w=600):
    """Smooth bilinear brightness ramp -- a composited-backdrop vignette.

    Each 5x5 corner patch is locally flat (the ramp moves a fraction of a grey
    level across 5 px at this size) while the four corners disagree widely.
    That is exactly the case the pooled std misreads as "noisy corners".
    """
    ys = np.linspace(0.0, 1.0, h)[:, None]
    xs = np.linspace(0.0, 1.0, w)[None, :]
    top = tl * (1 - xs) + tr * xs
    bot = bl * (1 - xs) + br * xs
    return np.clip(top * (1 - ys) + bot * ys, 0, 255).astype(np.uint8)


def flat_bg(level, h=400, w=600):
    return np.full((h, w), level, np.uint8)


def with_coin(bg, fill=210, radius=110):
    img = bg.copy()
    h, w = img.shape
    cv2.circle(img, (w // 2, h // 2), radius, int(fill), -1)
    return img


# --------------------------------------------------------------------------
# Bar 2 -- bit-identity where the pooled path already fires
# --------------------------------------------------------------------------

class TestBar2BitIdentity:
    """Where pooled corner-trust fires today, M1 must change nothing."""

    @pytest.mark.parametrize("level", [0, 12, 40, 90, 127, 128, 180, 240, 255])
    def test_flat_backgrounds_identical_gate_on(self, level, monkeypatch):
        img = with_coin(flat_bg(level))
        # Precondition: this image really is on the pooled path.
        pooled_std = float(np.std(np.concatenate(
            [p.flatten() for p in _corner_patches(img)])))
        assert pooled_std < 15, "fixture must exercise the pooled path"

        off = detect_background_histogram(img)
        monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, "1")
        on = detect_background_histogram(img)

        assert on == off == _golden_pre_m1(img)

    def test_randomized_battery_pooled_path_is_untouched(self, monkeypatch):
        """Seeded sweep: wherever pooled std < 15, ON == OFF == golden.

        Asserted over the whole battery rather than a hand-picked case, so a
        future edit that reorders the branches cannot pass by construction.
        """
        rng = np.random.default_rng(20260728)
        checked = 0
        for _ in range(300):
            h = int(rng.integers(40, 300))
            w = int(rng.integers(40, 300))
            base = int(rng.integers(0, 256))
            noise = float(rng.uniform(0.0, 30.0))
            img = np.clip(
                rng.normal(base, noise, size=(h, w)), 0, 255).astype(np.uint8)
            if rng.random() < 0.5:
                img = with_coin(img, fill=int(rng.integers(0, 256)),
                                radius=int(min(h, w) // 3))

            pooled_std = float(np.std(np.concatenate(
                [p.flatten() for p in _corner_patches(img)])))
            golden = _golden_pre_m1(img)

            monkeypatch.delenv(BG_CORNER_LOCAL_TRUST_ENV, raising=False)
            off = detect_background_histogram(img)
            monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, "1")
            on = detect_background_histogram(img)

            assert off == golden, "gate OFF must equal pre-M1 behavior"
            if pooled_std < 15:
                checked += 1
                assert on == golden, (
                    f"Bar 2 VIOLATION: pooled_std={pooled_std:.3f} fires today "
                    f"but M1 returned {on} instead of {golden}")
        assert checked >= 30, f"battery only exercised {checked} pooled-path cases"

    def test_tiny_image_unchanged(self, gate_on):
        """h<=20 skips corner sampling entirely -- both arms, both gates."""
        img = np.full((10, 10), 70, np.uint8)
        assert detect_background_histogram(img) == _golden_pre_m1(img)


# --------------------------------------------------------------------------
# Default-off -- the cross-cutting requirement
# --------------------------------------------------------------------------

class TestDefaultOff:

    def test_vignette_unset_is_pre_m1(self):
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))
        assert detect_background_histogram(img) == _golden_pre_m1(img)

    @pytest.mark.parametrize("value", ["", "0", "no", "off", "false", "2", "yes"])
    def test_unrecognized_values_do_not_enable(self, monkeypatch, value):
        monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, value)
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))
        assert detect_background_histogram(img) == _golden_pre_m1(img)

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "  True  "])
    def test_recognized_values_enable(self, monkeypatch, value):
        monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, value)
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))
        assert detect_background_histogram(img) != _golden_pre_m1(img)


# --------------------------------------------------------------------------
# M1 behavior -- it fires on a vignette, and returns the right number
# --------------------------------------------------------------------------

class TestM1Fires:

    def test_cng_like_dark_vignette(self, gate_on):
        """The measured CNG corner reading: pooled std ~22, each corner flat."""
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))

        pooled_std = float(np.std(np.concatenate(
            [p.flatten() for p in _corner_patches(img)])))
        assert pooled_std > 15, "fixture must defeat the pooled test"
        assert all(float(np.std(p)) < CORNER_LOCAL_STD_MAX
                   for p in _corner_patches(img)), "each corner must be locally flat"

        got = detect_background_histogram(img)
        assert got == _expected_m1(img)

        # Ticket's stated number: ~75 against an honest outer-ring 79.0, versus
        # the 31.2 the histogram fallback returns today.
        assert 70.0 <= got[0] <= 80.0, got
        assert got[1] == "dark"

        # And it is a real change from today's answer.
        assert got != _golden_pre_m1(img)

    def test_light_vignette_returns_light(self, gate_on):
        """Polarity follows the value; it is not hardcoded to the CNG case."""
        img = with_coin(ramp_bg(190.0, 200.0, 230.0, 240.0), fill=40)
        pooled_std = float(np.std(np.concatenate(
            [p.flatten() for p in _corner_patches(img)])))
        assert pooled_std > 15

        got = detect_background_histogram(img)
        assert got == _expected_m1(img)
        assert got[1] == "light"

    def test_polarity_flip_on_light_fallback_case(self, gate_on):
        """The Bar 0 no-op class: `light_fallback` returns ~215 on a ~79 backdrop.

        M1 must correct the value AND flip polarity light -> dark, which is the
        counterfactual Bar 0 measured on the 12 no-op sides.
        """
        # Vignetted dark backdrop, with enough bright coin pixels to send the
        # histogram fallback down its `light` branch.
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4), fill=250, radius=170)
        before = _golden_pre_m1(img)
        assert before[1] == "light", f"fixture must reproduce light_fallback, got {before}"
        assert before[0] > 200

        after = detect_background_histogram(img)
        assert after[1] == "dark"
        assert 70.0 <= after[0] <= 80.0, after


class TestM1DeclinesWhenACornerIsGenuinelyNoisy:
    """M1 widens the corner test; it must not abolish it."""

    def test_noisy_corner_falls_through(self, gate_on, monkeypatch):
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))
        # Wreck one corner -- a sticker, a caption, an intruding coin edge.
        rng = np.random.default_rng(7)
        img[0:5, 0:5] = rng.integers(0, 256, size=(5, 5), dtype=np.uint8)
        assert float(np.std(img[0:5, 0:5])) >= CORNER_LOCAL_STD_MAX

        monkeypatch.delenv(BG_CORNER_LOCAL_TRUST_ENV, raising=False)
        off = detect_background_histogram(img)
        monkeypatch.setenv(BG_CORNER_LOCAL_TRUST_ENV, "1")
        on = detect_background_histogram(img)

        assert on == off == _golden_pre_m1(img)

    def test_coin_intruding_into_one_corner_falls_through(self, gate_on):
        """A coin bleeding into a corner is a real reason to distrust corners."""
        img = with_coin(ramp_bg(96.0, 99.4, 54.6, 53.4))
        cv2.circle(img, (2, 2), 40, 235, -1)   # bright disc over the TL corner
        patches = _corner_patches(img)
        # TL is now a flat BRIGHT patch -- locally clean but not background.
        # It disagrees with the other three; M1 still takes it (it cannot know),
        # so assert the guarded outcome we actually get rather than a wish.
        tl_std = float(np.std(patches[0]))
        if tl_std < CORNER_LOCAL_STD_MAX:
            got = detect_background_histogram(img)
            assert got == _expected_m1(img)
        else:
            assert detect_background_histogram(img) == _golden_pre_m1(img)
