"""Per-lane mask configuration — the resolver for `src/mask_lanes.json`.

WHY THIS MODULE EXISTS
======================
The mask-lane knobs used to be read straight out of `os.environ`, and they were
set in the pipeline repo's `.env`. `.env` is gitignored, so `git archive HEAD`
correctly does not ship it — and a rented GPU pod therefore ran Layer 1 with
DIFFERENT PARAMETERS and produced different vectors from identical bytes.
Silently: `masked:true`, `mask_fallback_reason:null`, no no-op. The lie was in
the parameters, and the store's crop-sha256 would have claimed those vectors
were reproducible.

Full incident: `analysis/gpu_run7/c1_determinism/REPORT.md` in the pipeline
repo. Ruling: `specs/mask_config_provenance_ruling.md` (architect, 2026-08-15)
plus its five-knob brain amendment. Ticket #47.

Two more knobs of the identical class (unset in every lane, not covered by
the guard) were found and deliberately left unmigrated pending a ruling on
scope; admitted here by ticket #48 (`rim_neighbor_overlap_max`,
`rim_hough_cap`) — same file, same resolver, same guard, no code shape
change beyond `VALUE_KEYS`/`LEGACY_ENV_KEYS` growing from five to seven.
`mask_lanes.json`'s `schema_version` bumped 1→2 to mark the stamped field
SET changing; no knob's VALUE changed for the original five.

Committed config travels with `git archive`. That is the whole fix: a pod gets
the parameters BY CONSTRUCTION rather than by discipline.

HOW A LANE IS SELECTED
======================
In order, first hit wins:

  1. an explicit in-process activation — `activate()` / `use_lane()`;
  2. the `TRIVALAYA_MASK_LANE` environment variable;
  3. ERROR.

There is deliberately NO default lane and NO fallback. Absence is precisely
what caused the incident: the pod's environment was missing the knobs and the
code quietly picked its own defaults. Absence must fail loudly, not pick
something.

THE LEGACY-KEY GUARD
====================
After this migration, the presence of any of the seven retired environment
keys is an ERROR — never a silent override (ruling 2 rider 2: "the old path
dies, it does not linger").

  ★ The check fires on PRESENCE, REGARDLESS OF VALUE — including a value
    identical to the lane config's.

That is not pedantry. "Identical today" is exactly how a fork drifts in later:
the config changes, the stale environment variable does not, and the two
disagree with nothing to announce it. An exclusion that the defect can satisfy
is not a tripwire (CLAUDE.md, tripwire-suppression-rule design).

A/B harnesses and tests do not need the environment: they pass explicit
`**overrides` to `use_lane()` / `activate()`, which are recorded in the resolved
config's digest like any other value.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

__all__ = [
    "MaskConfig",
    "MaskConfigError",
    "LANE_SELECTOR_ENV",
    "LEGACY_ENV_KEYS",
    "resolve_mask_config",
    "active_mask_config",
    "activate",
    "deactivate",
    "use_lane",
    "lane_names",
    "mask_lane_version",
    "mask_config_schema_version",
]

LANES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_lanes.json")

#: The one environment variable this scheme still reads. It selects a lane by
#: NAME; it never carries a parameter value. Set it per-unit / per-invocation.
#:
#: ⚠ Never put it in the pipeline repo's `.env`: `trivalaya-runner` reads `.env`
#: via `EnvironmentFile=` and `visual_search/appv2.py` setdefaults every `.env`
#: key into `os.environ` at import, so both lanes would inherit whatever it
#: said. `appv2` enforces this.
LANE_SELECTOR_ENV = "TRIVALAYA_MASK_LANE"

#: The seven retired keys. Ordered as in the ruling's amendment table, then
#: the two ticket #48 additions (same class: mask-behaviour knobs read from
#: `os.environ`, unset in every lane before their admission). Presence of
#: ANY of these in `os.environ` is a hard error.
LEGACY_ENV_KEYS = (
    "TRIVALAYA_CLOSE_KERNEL_FRAC",
    "TRIVALAYA_RIM_NEIGHBOR_GUARD",
    "TRIVALAYA_BG_CORNER_LOCAL_TRUST",
    "TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE",
    "TRIVALAYA_DISABLE_TWO_COIN_RESOLVER",
    "TRIVALAYA_RIM_NEIGHBOR_OVERLAP_MAX",
    "TRIVALAYA_RIM_HOUGH_CAP",
)

#: The seven knobs, in stamp order. Kept in one place so the config file, the
#: resolver, the digest and the embedding-store stamp cannot drift apart.
VALUE_KEYS = (
    "close_kernel_frac",
    "rim_neighbor_guard",
    "bg_corner_local_trust",
    "rim_recovery_max_per_image",
    "disable_two_coin_resolver",
    "rim_neighbor_overlap_max",
    "rim_hough_cap",
)


class MaskConfigError(RuntimeError):
    """A lane could not be resolved, or a retired environment key is set.

    Deliberately loud and deliberately not caught anywhere in the mask lane:
    the failure this replaces was silent, and a silent successor would be no
    fix at all.
    """


# ---------------------------------------------------------------------------
# Value coercion — preserves the RETIRED env semantics exactly
# ---------------------------------------------------------------------------
# Each coercer reproduces, bit for bit, the truthiness rule the corresponding
# `os.environ.get(...)` read used before this migration, so that a config value
# and the environment value it replaces cannot mean different things. They also
# accept native JSON types, which is how the config file expresses them.

def _coerce_close_kernel_frac(v: Any) -> Optional[str]:
    """None/"" => fixed 7x7 (the old `if not _frac_env` branch). Else a string
    that layer1 interprets as "auto" or a float."""
    if v is None:
        return None
    s = str(v)
    return s if s else None


def _coerce_flag_1_or_true(v: Any) -> bool:
    """Old rule: `os.environ.get(K, "").strip().lower() in ("1", "true")`."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true")


def _coerce_flag_exactly_1(v: Any) -> bool:
    """Old rule for the two-coin kill-switch, which compared against "1"
    EXACTLY (`_os.environ.get(K) != "1"`) — "true" did NOT disable it. Kept
    distinct from `_coerce_flag_1_or_true` so the migration cannot quietly
    widen it."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v) == "1"


def _coerce_recovery_max(v: Any) -> Optional[int]:
    """None => unlimited (today's behaviour). Note the old read was
    `if _max_recovery: _cap = int(_max_recovery)` — a string "0" is truthy, so
    an explicit 0 meant "cap at zero", not "unlimited". `None` vs `0` here
    preserves that distinction, which a plain falsiness test would lose."""
    if v is None or v == "":
        return None
    return int(v)


def _coerce_overlap_max(v: Any) -> float:
    """Ticket #48. Old read: `float(os.environ.get(K, str(DEFAULT)))` — always
    a concrete number, never a None/"off" state (unlike the recovery cap
    above), so a plain `float()` reproduces it exactly."""
    return float(v)


def _coerce_hough_cap(v: Any) -> int:
    """Ticket #48. Old read: `int(os.environ.get(K, str(getattr(...))))` with a
    `ValueError` fallback to a literal — that fallback only guarded against an
    unparsable STRING reaching `int()`, which cannot happen here: the legacy-
    key guard now refuses the env path entirely, and every lane's config value
    is a real JSON number. A plain `int()` reproduces the parsed-value
    semantics; 0 is a legitimate value (uncapped / legacy full-res), not a
    sentinel, so it is not special-cased."""
    return int(v)


_COERCERS = {
    "close_kernel_frac": _coerce_close_kernel_frac,
    "rim_neighbor_guard": _coerce_flag_1_or_true,
    "bg_corner_local_trust": _coerce_flag_1_or_true,
    "rim_recovery_max_per_image": _coerce_recovery_max,
    "disable_two_coin_resolver": _coerce_flag_exactly_1,
    "rim_neighbor_overlap_max": _coerce_overlap_max,
    "rim_hough_cap": _coerce_hough_cap,
}


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MaskConfig:
    """One lane's fully resolved mask parameters.

    Frozen: a resolved config is a record of what ran, and a record that can be
    mutated after the fact is not a record.
    """

    lane: str
    mask_lane_version: str
    close_kernel_frac: Optional[str]
    rim_neighbor_guard: bool
    bg_corner_local_trust: bool
    rim_recovery_max_per_image: Optional[int]
    disable_two_coin_resolver: bool
    # Ticket #48 (2026-08-15): same class as the five above, admitted later.
    rim_neighbor_overlap_max: float
    rim_hough_cap: int
    # The stamped FIELD SET's version (mask_lanes.json "schema_version"),
    # carried alongside mask_lane_version for the identical reason rider 1
    # names for that field: a hash/version says two generations differ, not
    # how, and here specifically it is what tells a reader whether two
    # digests are even comparable (they are NOT across schema_version, since
    # the digest hashes VALUE_KEYS and that set's size is what changed).
    schema_version: int

    @property
    def values(self) -> Dict[str, Any]:
        """The seven knobs, and only those — the thing the store stamps.

        Deliberately excludes `lane` and `mask_lane_version`: the values are
        what determine the vectors, and two lanes that share them are
        genuinely the same arm.
        """
        return {k: getattr(self, k) for k in VALUE_KEYS}

    @property
    def digest(self) -> str:
        """sha256 over the seven VALUES (never the lane name).

        Convenience for cheap equality checks only. Per ruling 2 rider 1 the
        digest does NOT replace the values in the stamp: a hash says two
        generations differ, not how, which is the same detectable-but-not-
        attributable gap `n_detections` already has.

        Because the lane name is excluded, `ingest` and `gpu_embed` digest
        EQUAL — which is what makes ruling 1 ("the corpus embeds under the
        ingest arm") a checkable claim rather than an asserted one.

        ★ This digest is NOT comparable across `schema_version`: it hashes
        whatever `VALUE_KEYS` currently is, so admitting a knob (ticket #48)
        changes every digest's value even where no VALUE changed. Compare
        digests only when `schema_version` also matches.
        """
        payload = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        """One line, for a bar to PRINT before trusting its own result.

        Post-migration this — not an environment variable — is the arm.
        """
        vals = " ".join(f"{k}={self.values[k]!r}" for k in VALUE_KEYS)
        return (f"lane={self.lane} version={self.mask_lane_version} "
                f"schema={self.schema_version} digest={self.digest[:16]} {vals}")


# ---------------------------------------------------------------------------
# Lane table
# ---------------------------------------------------------------------------

_lanes_cache: Optional[dict] = None
_lanes_lock = threading.Lock()


def _load_lanes() -> dict:
    global _lanes_cache
    if _lanes_cache is None:
        with _lanes_lock:
            if _lanes_cache is None:
                try:
                    with open(LANES_PATH) as f:
                        _lanes_cache = json.load(f)
                except FileNotFoundError as exc:  # pragma: no cover - deploy fault
                    raise MaskConfigError(
                        f"mask lane config missing at {LANES_PATH}. It is COMMITTED "
                        "and must travel with the code (that is the entire point of "
                        "this file); a deploy without it is incomplete, not a reason "
                        "to fall back to defaults."
                    ) from exc
    return _lanes_cache


def lane_names() -> tuple:
    return tuple(sorted(_load_lanes()["lanes"]))


def mask_lane_version() -> str:
    return str(_load_lanes()["mask_lane_version"])


def mask_config_schema_version() -> int:
    """The stamped FIELD SET's version (ticket #48). Bumped when a knob is
    admitted or retired, independent of `mask_lane_version` (bumped for a
    code/value generation of the existing knobs). See `MaskConfig.digest`:
    digests are comparable only within one schema_version."""
    return int(_load_lanes()["schema_version"])


# ---------------------------------------------------------------------------
# The two hard failures
# ---------------------------------------------------------------------------

def _assert_no_legacy_env() -> None:
    """Raise if ANY retired key is present in `os.environ`, whatever its value.

    See the module docstring: the value is not consulted on purpose.
    """
    present = [k for k in LEGACY_ENV_KEYS if k in os.environ]
    if not present:
        return
    shown = ", ".join(f"{k}={os.environ[k]!r}" for k in present)
    raise MaskConfigError(
        "retired mask-lane environment key(s) present: " + shown + ".\n"
        "These knobs moved into committed per-lane config: " + LANES_PATH + ".\n"
        "Select a lane with " + LANE_SELECTOR_ENV + "=<" + "|".join(lane_names()) + ">, "
        "or pass explicit overrides to mask_config.use_lane(). Unset the key(s) above.\n"
        "This fires on PRESENCE, not on disagreement: a value that matches the "
        "config today is exactly how a fork drifts in later, so it is refused too. "
        "(analysis/gpu_run7/c1_determinism/REPORT.md, ticket #47)"
    )


def _selected_lane() -> str:
    lane = os.environ.get(LANE_SELECTOR_ENV, "").strip()
    if not lane:
        raise MaskConfigError(
            "no mask lane selected. Set " + LANE_SELECTOR_ENV + "=<"
            + "|".join(lane_names()) + "> for this process, or call "
            "mask_config.activate(<lane>) / use_lane(<lane>) in-process.\n"
            "There is no default lane on purpose: an environment that silently "
            "picked its own mask parameters is the Run 7 C1 incident "
            "(analysis/gpu_run7/c1_determinism/REPORT.md, ticket #47). Ad-hoc and "
            "analysis scripts should name the 'tools' lane."
        )
    return lane


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_mask_config(lane: str, **overrides: Any) -> MaskConfig:
    """Resolve one lane to a frozen `MaskConfig`.

    Raises `MaskConfigError` if the lane is absent/unknown, if any retired
    environment key is present, or if an override names a knob that does not
    exist (a typo'd override is a silent no-op otherwise, which is the same
    class of failure this module exists to close).
    """
    _assert_no_legacy_env()

    if not lane or not str(lane).strip():
        raise MaskConfigError(
            "resolve_mask_config() called with no lane. There is no default "
            "lane and no fallback — absence must fail loudly, not pick something."
        )
    lane = str(lane).strip()

    table = _load_lanes()
    if lane not in table["lanes"]:
        raise MaskConfigError(
            f"unknown mask lane {lane!r}. Known lanes: {', '.join(lane_names())} "
            f"(defined in {LANES_PATH}). A new lane is a config entry someone can "
            "read, not an argument the resolver invents."
        )

    unknown = [k for k in overrides if k not in VALUE_KEYS]
    if unknown:
        raise MaskConfigError(
            f"unknown mask config override(s): {sorted(unknown)}. "
            f"Valid knobs: {', '.join(VALUE_KEYS)}."
        )

    entry = table["lanes"][lane]
    resolved = {}
    for key in VALUE_KEYS:
        raw = overrides[key] if key in overrides else entry.get(key)
        resolved[key] = _COERCERS[key](raw)

    return MaskConfig(lane=lane, mask_lane_version=mask_lane_version(),
                      schema_version=mask_config_schema_version(), **resolved)


# ---------------------------------------------------------------------------
# In-process activation
# ---------------------------------------------------------------------------
# A plain module global, not a thread-local: the mask lane is a property of the
# PROCESS (which parameters this deployment runs), not of a thread. Serving
# resolves the same lane on every worker thread, and a thread-local would let
# two threads disagree about what produced a vector — the exact ambiguity this
# module removes.

_active: Optional[MaskConfig] = None


def activate(lane: str, **overrides: Any) -> MaskConfig:
    """Pin this process to `lane`, overriding the environment selector.

    For entrypoints that KNOW their lane in committed code (`appv2` is the
    serving process; `vision_adapter` is the ingest process) and for A/B tools
    that vary a knob explicitly instead of through the environment.
    """
    global _active
    cfg = resolve_mask_config(lane, **overrides)
    _active = cfg
    return cfg


def deactivate() -> None:
    """Drop any in-process activation, returning to the environment selector."""
    global _active
    _active = None


@contextlib.contextmanager
def use_lane(lane: str, **overrides: Any):
    """Scoped `activate()`, restoring the previous activation on exit.

    This is the sanctioned replacement for `monkeypatch.setenv(...)` in tests
    and `os.environ[K] = v` in A/B tools.
    """
    global _active
    previous = _active
    try:
        yield activate(lane, **overrides)
    finally:
        _active = previous


def active_mask_config() -> MaskConfig:
    """The config the mask lane is running under, right now.

    This is what the read sites call. It re-checks the legacy-key guard on
    every call rather than once at import, so a key set half way through a
    process (an A/B tool, a test) is caught too — an import-time-only check
    would be exec-frozen, which is the very shape of bug this migration fixes
    in `TRIVALAYA_DISABLE_TWO_COIN_RESOLVER`.
    """
    if _active is not None:
        _assert_no_legacy_env()
        return _active
    return resolve_mask_config(_selected_lane())
