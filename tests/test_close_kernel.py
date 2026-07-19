"""
Tier 1 -- MORPH_CLOSE kernel sizing (pure unit, no images, milliseconds).

This tier exists because the original spec's bug was pure arithmetic:
`int(round(3000/400))` is 8, and `8 | 1` is 9 -- silently giving the
42,080-coin 3000x1440 `cng` house MORE bridging than today's fixed 7, on the
one population the change promises to leave alone.  These tests run in under a
second and gate every corpus-level check.

See specs/two_coin_weld_morph_close.md SS9.1.
"""

import cv2
import pytest

from src.config import Layer1Config
from src.layer1_geometry import _close_kernel_size
from tests.conftest import ENV_FRAC, render_pair, segment

# Widths spanning every band edge in SS1's table.  1599/1600 pin the 3->5
# transition, which sits at 1600 (not 1700): int(1600/400) == 4 and 4 | 1 == 5,
# so the odd-bump rounds even quotients *up* and k reaches 5 a full 400px band
# earlier than a naive reading of the fraction suggests.
WIDTH_TABLE = [
    (370, 3),    # davissons
    (500, 3),    # cng_feature -- the fix
    (800, 3),
    (1200, 3),   # leu, gorny
    (1599, 3),   # last width in the k=3 band
    (1600, 5),   # first width in the k=5 band
    (1700, 5),   # mashops
    (1989, 5),   # obolos
    (2400, 7),   # first width in the k=7 band
    (2800, 7),
    (3000, 7),   # cng
    (3200, 9),   # MAX_DIMENSION ceiling
]

CNG_BAND = range(2400, 3200)


@pytest.mark.parametrize("width,expected", WIDTH_TABLE)
def test_width_table(width, expected):
    assert _close_kernel_size(100, width) == expected


def test_cng_band_is_uniformly_seven():
    """
    The v1-bug regression test: k must be 7 across the ENTIRE 2400-3199 band,
    which is where cng's 3000x1440 raws live (they are under MAX_DIMENSION so
    L1 sees them at full size).
    """
    off = [w for w in CNG_BAND if _close_kernel_size(1440, w) != 7]
    assert not off, f"expected k=7 across 2400-3199, deviations at {off[:10]}"


def test_cng_format_pinned():
    """Called out separately from the table so it cannot be silently edited away."""
    assert _close_kernel_size(1440, 3000) == 7


def test_floor_not_round():
    """
    The tripwire.  round(3000/400) == 8 -> 8 | 1 == 9; int(3000/400) == 7.
    If this fails, someone reintroduced round() and cng silently got more
    bridging than it has today.
    """
    assert _close_kernel_size(3000, 3000) == 7


def test_oddness():
    bad = [w for w in range(1, 10001) if _close_kernel_size(1, w) % 2 == 0]
    assert not bad, f"even kernel at widths {bad[:10]}"


def test_clamp():
    lo, hi = Layer1Config.CLOSE_KERNEL_MIN, Layer1Config.CLOSE_KERNEL_MAX
    bad = [w for w in range(1, 10001) if not lo <= _close_kernel_size(1, w) <= hi]
    assert not bad, f"kernel outside [{lo}, {hi}] at widths {bad[:10]}"


def test_max_is_odd():
    """
    Guards the `k |= 1`-after-clamp ordering: with an even MAX, k can exceed it
    by one.  config.py also asserts this at module level -- that is the copy
    that fires without a test suite; this one localises the failure.
    """
    assert Layer1Config.CLOSE_KERNEL_MAX % 2 == 1


def test_monotonic():
    prev = 0
    for w in range(1, 10001):
        k = _close_kernel_size(1, w)
        assert k >= prev, f"k decreased at width {w}: {prev} -> {k}"
        prev = k


def test_orientation_symmetry():
    """Sizing keys off max(h, w), so portrait must equal landscape."""
    for a, b in [(1440, 3000), (500, 234), (370, 200), (1200, 900)]:
        assert _close_kernel_size(a, b) == _close_kernel_size(b, a)


def test_helper_ignores_env(monkeypatch):
    """
    The helper is a pure function of its arguments -- the env gate lives at the
    call site.  If the helper ever starts reading env, the tests above stop
    testing what they claim to.
    """
    monkeypatch.setenv(ENV_FRAC, "0.5")
    assert _close_kernel_size(100, 500) == 3


def test_explicit_frac_overrides_config():
    assert _close_kernel_size(100, 500, frac=1 / 100) == 5
    assert _close_kernel_size(100, 500, frac=1 / 400) == 3


# --- The default-unchanged guard (most important test in this tier) ---------

@pytest.mark.parametrize("width", [w for w, _ in WIDTH_TABLE])
def test_default_env_unset_is_literally_seven_by_seven(width, close_kernels):
    """
    With TRIVALAYA_CLOSE_KERNEL_FRAC unset, the kernel reaching OpenCV must be
    literally (7, 7) at every width -- i.e. production behavior is byte-for-byte
    unchanged by this commit.  SS6.1 promises this; this makes it tested rather
    than asserted, and it is what lets the change land ahead of the sweep.

    NOTE: this guard inverts BY DESIGN when SS6.3 flips the default to the
    scale-relative path.  The flip commit updates it to assert the new default.
    A post-flip failure here is not a regression to revert.
    """
    segment(render_pair(width, gap=8))
    assert close_kernels, "no MORPH_CLOSE was issued"
    assert set(close_kernels) == {(7, 7)}, close_kernels


@pytest.mark.parametrize("width,expected", WIDTH_TABLE)
def test_env_set_enables_scale_relative_path(width, expected, close_kernels,
                                             monkeypatch):
    """The other half of the gate: setting the var routes through the helper."""
    monkeypatch.setenv(ENV_FRAC, str(1 / 400))
    segment(render_pair(width, gap=8))
    assert set(close_kernels) == {(expected, expected)}, close_kernels


def test_env_value_overrides_config_frac(close_kernels, monkeypatch):
    """
    The env value overrides CLOSE_KERNEL_FRAC, so the SS4.2 sweep can A/B
    without a deploy.
    """
    monkeypatch.setenv(ENV_FRAC, str(1 / 100))
    segment(render_pair(500, gap=8))
    assert set(close_kernels) == {(5, 5)}, close_kernels
