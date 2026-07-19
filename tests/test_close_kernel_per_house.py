"""
Per-house MORPH_CLOSE overrides.

The table (Layer1Config.CLOSE_KERNEL_BY_HOUSE) is EMPTY by default: the
scale-relative formula already absorbs the axis houses most obviously differ
on -- image dimensions -- so these tests mostly pin that the mechanism is
inert until someone populates it from SS4.2 sweep data, and that it validates
what they put there.

Tests patch the table rather than relying on shipped entries, so they keep
passing (and keep meaning the same thing) once real houses are added.
"""

import cv2
import pytest

from src.config import Layer1Config, validate_close_kernel_overrides
from src.layer1_geometry import _close_kernel_size
from src.pipeline_manager import analyze_image
from tests.conftest import ENV_FRAC, render_pair, segment


@pytest.fixture
def house_table(monkeypatch):
    """Install a per-house override table for the duration of one test."""
    def install(table):
        monkeypatch.setattr(Layer1Config, "CLOSE_KERNEL_BY_HOUSE", table)
        validate_close_kernel_overrides()
        return table
    return install


# --- Inert by default -------------------------------------------------------

def test_table_ships_empty():
    """
    Guards against someone adding a guessed constant.  Entries are legitimate
    only with SS4.2 measurements behind them -- and when they land, this test
    is the prompt to confirm that happened.
    """
    assert Layer1Config.CLOSE_KERNEL_BY_HOUSE == {}


@pytest.mark.parametrize("house", [None, "", "cng", "cng_feature", "kuenker",
                                   "leu", "no_such_house"])
@pytest.mark.parametrize("width", [370, 500, 1200, 1700, 3000])
def test_house_is_a_noop_with_empty_table(house, width):
    assert _close_kernel_size(100, width, house=house) == \
           _close_kernel_size(100, width)


# --- Override semantics -----------------------------------------------------

def test_frac_override_applies(house_table):
    house_table({"cng_feature": {"frac": 1 / 100}})
    # 500 * 1/100 = 5 -> 5; the global 1/400 would floor to 1 and clamp to 3.
    assert _close_kernel_size(234, 500, house="cng_feature") == 5
    assert _close_kernel_size(234, 500, house="kuenker") == 3


def test_min_override_applies(house_table):
    """SS7.3's suggested lever: hold leu at k=5 instead of dropping it to 3."""
    house_table({"leu": {"min": 5}})
    assert _close_kernel_size(604, 1200, house="leu") == 5
    assert _close_kernel_size(604, 1200, house="gorny") == 3


def test_max_override_applies(house_table):
    house_table({"tiny": {"max": 5}})
    assert _close_kernel_size(1440, 3000, house="tiny") == 5
    assert _close_kernel_size(1440, 3000, house="cng") == 7


def test_partial_override_falls_back_per_key(house_table):
    """An override names only what it changes; the rest come from the globals."""
    house_table({"h": {"min": 9}})
    assert _close_kernel_size(1440, 3000, house="h") == 9   # min lifts it
    assert _close_kernel_size(1440, 8000, house="h") == 21  # global max still caps


def test_house_matching_is_case_and_whitespace_insensitive(house_table):
    house_table({"cng_feature": {"frac": 1 / 100}})
    for spelling in ["cng_feature", "CNG_Feature", "  cng_feature  ", "CNG_FEATURE"]:
        assert _close_kernel_size(234, 500, house=spelling) == 5


def test_explicit_frac_outranks_per_house(house_table):
    """
    A sweep arm must mean one thing corpus-wide, so the env/explicit frac wins.
    Per-house min/max still apply -- they are bounds, not the quantity under test.
    """
    house_table({"cng_feature": {"frac": 1 / 100}})
    assert _close_kernel_size(234, 500, house="cng_feature", frac=1 / 400) == 3


def test_per_house_bounds_still_clamp_an_explicit_frac(house_table):
    house_table({"cng_feature": {"max": 5}})
    assert _close_kernel_size(234, 500, house="cng_feature", frac=1 / 10) == 5


def test_cng_stays_seven_under_an_unrelated_houses_override(house_table):
    """A per-house entry must not leak across houses -- the whole point of SS3."""
    house_table({"cng_feature": {"frac": 1 / 50, "max": 21}})
    assert _close_kernel_size(1440, 3000, house="cng") == 7
    assert _close_kernel_size(1440, 3000) == 7


# --- Table validation -------------------------------------------------------

@pytest.mark.parametrize("bad,reason", [
    ({"h": {"max": 20}}, "even max"),
    ({"h": {"min": 0}}, "min < 1"),
    ({"h": {"min": 9, "max": 5}}, "min > max"),
    ({"h": {"frac": 0}}, "frac <= 0"),
    ({"h": {"fraq": 1 / 400}}, "typo'd key"),
    ({"CNG": {"frac": 1 / 400}}, "non-lowercase key"),
    ({" cng": {"frac": 1 / 400}}, "unstripped key"),
])
def test_malformed_table_is_rejected_at_validation(bad, reason, monkeypatch):
    monkeypatch.setattr(Layer1Config, "CLOSE_KERNEL_BY_HOUSE", bad)
    with pytest.raises(AssertionError):
        validate_close_kernel_overrides()


def test_typoed_key_would_otherwise_be_silent(house_table):
    """
    Why the unknown-key assert earns its place: without it, {"fraq": ...} is a
    dict that reads correctly, validates nothing, and silently leaves the house
    on the global default -- a per-house tuning that quietly never applied.
    """
    with pytest.raises(AssertionError):
        house_table({"h": {"fraq": 1 / 100}})


# --- Call-site plumbing -----------------------------------------------------

def test_house_reaches_the_kernel_through_segmentation(close_kernels, monkeypatch,
                                                       house_table):
    monkeypatch.setenv(ENV_FRAC, "auto")
    house_table({"cng_feature": {"frac": 1 / 100}})

    segment(render_pair(500, gap=8))                      # no house
    assert set(close_kernels) == {(3, 3)}

    close_kernels.clear()
    from src.layer1_geometry import _segment_and_extract_candidates  # noqa: F401
    from tests.conftest import _preprocess
    img = render_pair(500, gap=8)
    ge, ez, tt, h, w = _preprocess(img)
    _segment_and_extract_candidates(img, ge, ez, tt, h, w, h * w,
                                    house="cng_feature")
    assert set(close_kernels) == {(5, 5)}


def test_house_reaches_the_kernel_through_analyze_image(close_kernels, monkeypatch,
                                                        house_table, tmp_path):
    """End-to-end through the public entry point callers actually use."""
    monkeypatch.setenv(ENV_FRAC, "auto")
    house_table({"cng_feature": {"frac": 1 / 100}})

    p = tmp_path / "lot.jpg"
    cv2.imwrite(str(p), render_pair(500, gap=8))

    analyze_image(str(p), source_type="auction", house="cng_feature")
    assert (5, 5) in set(close_kernels), close_kernels

    close_kernels.clear()
    analyze_image(str(p), source_type="auction")
    assert set(close_kernels) == {(3, 3)}, close_kernels


# --- The env gate still wins ------------------------------------------------

def test_per_house_is_dormant_while_the_default_is_fixed_seven(close_kernels,
                                                               house_table):
    """
    The landing guarantee outranks per-house config: with the env var unset,
    production is fixed 7x7 for EVERY house, however the table is populated.
    Inverts by design when SS6.3 flips the default.
    """
    house_table({"cng_feature": {"frac": 1 / 50}})
    from tests.conftest import _preprocess
    from src.layer1_geometry import _segment_and_extract_candidates

    img = render_pair(500, gap=8)
    ge, ez, tt, h, w = _preprocess(img)
    _segment_and_extract_candidates(img, ge, ez, tt, h, w, h * w,
                                    house="cng_feature")
    assert set(close_kernels) == {(7, 7)}


def test_auto_enables_configured_values(close_kernels, monkeypatch):
    """
    'auto' is the form that means "scale-relative, use configured values" --
    needed because a bare numeric env value would pin one frac corpus-wide and
    make per-house overrides untestable.
    """
    monkeypatch.setenv(ENV_FRAC, "auto")
    segment(render_pair(3000, gap=25))
    assert set(close_kernels) == {(7, 7)}

    close_kernels.clear()
    segment(render_pair(500, gap=8))
    assert set(close_kernels) == {(3, 3)}


@pytest.mark.parametrize("spelling", ["auto", "AUTO", " Auto "])
def test_auto_spelling(spelling, close_kernels, monkeypatch):
    monkeypatch.setenv(ENV_FRAC, spelling)
    segment(render_pair(500, gap=8))
    assert set(close_kernels) == {(3, 3)}
