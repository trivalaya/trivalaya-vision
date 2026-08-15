"""Bars for the per-lane mask config resolver — `src/mask_config.py`.

These are the FIRING PROOFS for the two hard failures the ruling requires
(`specs/mask_config_provenance_ruling.md` rider 2, pipeline repo). A guard that
cannot be demonstrated firing is UNMEASURED, not clean (CLAUDE.md canon #7),
so every raise here is asserted, and a positive control asserts that a
correctly-configured resolve still SUCCEEDS — without it, a resolver that
raised unconditionally would pass every negative test in this file.
"""

import json

import pytest

from src import mask_config
from src.mask_config import (
    LANE_SELECTOR_ENV,
    LEGACY_ENV_KEYS,
    MaskConfigError,
    resolve_mask_config,
)

ALL_LANES = ("serving", "ingest", "gpu_embed", "tools")


# ---------------------------------------------------------------------------
# Positive control — MUST come first in spirit: everything below is a negative
# ---------------------------------------------------------------------------

def test_positive_control_every_lane_resolves():
    """With no retired key set and a valid lane, resolution succeeds.

    This is the control that makes the negative tests mean something.
    """
    for lane in ALL_LANES:
        cfg = resolve_mask_config(lane)
        assert cfg.lane == lane
        assert len(cfg.values) == 5


def test_lane_values_are_the_ruling():
    """The ruling's table, transcribed. If this test is edited, a knob's value
    changed for a lane — which is a mask-lane generation bump needing its own
    ruling and its own bars, never a quiet edit."""
    expected = {
        "serving":   dict(close_kernel_frac="auto", rim_neighbor_guard=True,
                          bg_corner_local_trust=True, rim_recovery_max_per_image=None,
                          disable_two_coin_resolver=False),
        "ingest":    dict(close_kernel_frac="auto", rim_neighbor_guard=True,
                          bg_corner_local_trust=False, rim_recovery_max_per_image=None,
                          disable_two_coin_resolver=False),
        "gpu_embed": dict(close_kernel_frac="auto", rim_neighbor_guard=True,
                          bg_corner_local_trust=False, rim_recovery_max_per_image=None,
                          disable_two_coin_resolver=False),
        "tools":     dict(close_kernel_frac="auto", rim_neighbor_guard=True,
                          bg_corner_local_trust=False, rim_recovery_max_per_image=None,
                          disable_two_coin_resolver=False),
    }
    for lane, want in expected.items():
        assert resolve_mask_config(lane).values == want, lane


def test_gpu_embed_digests_equal_ingest_and_serving_differs():
    """Ruling 1's claim — "the corpus embeds under the INGEST arm" — is
    checkable rather than asserted, because the digest covers values only."""
    ingest = resolve_mask_config("ingest")
    gpu = resolve_mask_config("gpu_embed")
    serving = resolve_mask_config("serving")
    assert gpu.digest == ingest.digest
    assert resolve_mask_config("tools").digest == ingest.digest
    assert serving.digest != ingest.digest  # bg_corner_local_trust, M1 leg 1


# ---------------------------------------------------------------------------
# Hard failure 1 — presence of a retired key, REGARDLESS OF VALUE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", LEGACY_ENV_KEYS)
def test_any_retired_key_present_raises(monkeypatch, key):
    monkeypatch.setenv(key, "1")
    with pytest.raises(MaskConfigError) as exc:
        resolve_mask_config("ingest")
    assert key in str(exc.value)
    assert "mask_lanes.json" in str(exc.value)


@pytest.mark.parametrize("key", LEGACY_ENV_KEYS)
def test_retired_key_raises_even_when_empty(monkeypatch, key):
    """Empty string is still PRESENT. `os.environ.get(K, "")` could not tell
    these apart; `K in os.environ` can, and that is the whole point."""
    monkeypatch.setenv(key, "")
    with pytest.raises(MaskConfigError):
        resolve_mask_config("ingest")


def test_retired_key_raises_when_value_is_IDENTICAL_to_lane_config(monkeypatch):
    """★ The case a naive check would wave through.

    `serving` resolves bg_corner_local_trust=True and close_kernel_frac="auto".
    Setting the retired env keys to values that mean EXACTLY that must still
    raise: "identical today" is precisely how a fork drifts in later — the
    config moves, the stale env var does not, and nothing announces the
    disagreement. An exclusion that the defect can satisfy is not a tripwire.
    """
    serving = resolve_mask_config("serving")
    assert serving.bg_corner_local_trust is True
    assert serving.close_kernel_frac == "auto"

    monkeypatch.setenv("TRIVALAYA_BG_CORNER_LOCAL_TRUST", "1")   # == config
    monkeypatch.setenv("TRIVALAYA_CLOSE_KERNEL_FRAC", "auto")    # == config
    monkeypatch.setenv("TRIVALAYA_RIM_NEIGHBOR_GUARD", "1")      # == config
    with pytest.raises(MaskConfigError) as exc:
        resolve_mask_config("serving")
    msg = str(exc.value)
    assert "PRESENCE" in msg
    for key in ("TRIVALAYA_BG_CORNER_LOCAL_TRUST", "TRIVALAYA_CLOSE_KERNEL_FRAC",
                "TRIVALAYA_RIM_NEIGHBOR_GUARD"):
        assert key in msg


def test_guard_fires_through_active_mask_config_too(monkeypatch):
    """The guard must not be bypassable by pinning a lane first and setting the
    key afterwards — the mid-process case an import-time-only check misses."""
    with mask_config.use_lane("ingest"):
        assert mask_config.active_mask_config().lane == "ingest"
        monkeypatch.setenv("TRIVALAYA_RIM_NEIGHBOR_GUARD", "1")
        with pytest.raises(MaskConfigError):
            mask_config.active_mask_config()


# ---------------------------------------------------------------------------
# Hard failure 2 — absence of a lane
# ---------------------------------------------------------------------------

def test_no_lane_selected_raises(monkeypatch):
    monkeypatch.delenv(LANE_SELECTOR_ENV, raising=False)
    mask_config.deactivate()
    with pytest.raises(MaskConfigError) as exc:
        mask_config.active_mask_config()
    assert LANE_SELECTOR_ENV in str(exc.value)


def test_empty_lane_selector_raises(monkeypatch):
    monkeypatch.setenv(LANE_SELECTOR_ENV, "   ")
    mask_config.deactivate()
    with pytest.raises(MaskConfigError):
        mask_config.active_mask_config()


def test_unknown_lane_raises():
    with pytest.raises(MaskConfigError) as exc:
        resolve_mask_config("prod")
    assert "unknown mask lane" in str(exc.value)


def test_selector_env_var_is_honoured(monkeypatch):
    """Positive control for the selector path itself — otherwise the two
    absence tests above would pass against a resolver that never reads it."""
    mask_config.deactivate()
    monkeypatch.setenv(LANE_SELECTOR_ENV, "gpu_embed")
    assert mask_config.active_mask_config().lane == "gpu_embed"


def test_unknown_override_raises():
    """A typo'd override is otherwise a silent no-op — the same class of
    failure this module exists to close."""
    with pytest.raises(MaskConfigError):
        resolve_mask_config("tools", bg_corner_local_trust_typo=True)


# ---------------------------------------------------------------------------
# Coercion preserves the RETIRED env semantics exactly
# ---------------------------------------------------------------------------

def test_two_coin_kill_switch_matches_exactly_1_not_true():
    """The retired read was `!= "1"`, so "true" did NOT disable the resolver.
    Migrating must not quietly widen it to the 1-or-true rule the other flags
    use."""
    assert resolve_mask_config("tools", disable_two_coin_resolver="1").disable_two_coin_resolver is True
    assert resolve_mask_config("tools", disable_two_coin_resolver="true").disable_two_coin_resolver is False


def test_flag_coercion_matches_1_or_true():
    for val, want in (("1", True), ("true", True), ("TRUE", True), (" 1 ", True),
                      ("0", False), ("", False), ("yes", False), (None, False)):
        assert resolve_mask_config("tools", rim_neighbor_guard=val).rim_neighbor_guard is want, val


def test_recovery_cap_zero_is_not_unlimited():
    """The retired read was `if _max_recovery:` on a STRING, so "0" was truthy
    and meant "cap at zero". `None` vs `0` preserves that; a falsiness test
    would collapse them."""
    assert resolve_mask_config("tools", rim_recovery_max_per_image=0).rim_recovery_max_per_image == 0
    assert resolve_mask_config("tools", rim_recovery_max_per_image=None).rim_recovery_max_per_image is None
    assert resolve_mask_config("tools", rim_recovery_max_per_image="3").rim_recovery_max_per_image == 3


def test_close_kernel_frac_empty_is_none():
    """`if not _frac_env` treated "" and unset alike (fixed 7x7)."""
    assert resolve_mask_config("tools", close_kernel_frac="").close_kernel_frac is None
    assert resolve_mask_config("tools", close_kernel_frac=None).close_kernel_frac is None
    assert resolve_mask_config("tools", close_kernel_frac="0.02").close_kernel_frac == "0.02"


# ---------------------------------------------------------------------------
# Shape / immutability
# ---------------------------------------------------------------------------

def test_config_is_frozen():
    cfg = resolve_mask_config("serving")
    with pytest.raises(Exception):
        cfg.bg_corner_local_trust = False


def test_digest_is_stable_and_value_sensitive():
    a = resolve_mask_config("serving")
    b = resolve_mask_config("serving")
    assert a.digest == b.digest
    c = resolve_mask_config("serving", rim_neighbor_guard=False)
    assert c.digest != a.digest


def test_describe_prints_the_arm():
    """Bars PRINT this before trusting a result. Post-migration the arm is the
    resolved config, not an env var (CLAUDE.md env trap)."""
    text = resolve_mask_config("serving").describe()
    assert "lane=serving" in text
    assert "bg_corner_local_trust=True" in text
    assert "digest=" in text


def test_lane_file_has_no_extra_lanes():
    """The lane set is the ruling's four. A fifth lane appearing without a
    ruling is the thing this test notices."""
    with open(mask_config.LANES_PATH) as f:
        data = json.load(f)
    assert tuple(sorted(data["lanes"])) == tuple(sorted(ALL_LANES))


def test_use_lane_restores_previous_activation():
    with mask_config.use_lane("serving"):
        assert mask_config.active_mask_config().lane == "serving"
        with mask_config.use_lane("ingest"):
            assert mask_config.active_mask_config().lane == "ingest"
        assert mask_config.active_mask_config().lane == "serving"
