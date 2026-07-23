# Scope B — neighbor-aware rim recovery, real-data sweep (2026-07-23)

Graded against `specs/rim_recovery_neighbor_aware.md`'s Scope B PRECOMMIT
bars. Tool: `tools/two_coin_weld_mask_gate.py --neighbor-guard` (extended
this session with the guard flag + a `--lots` filter; also fixed a
pre-existing crash in `PASS_d0_contour` when a lot has <2 detections, i.e.
`sliver_contour_frac` is `""` — unrelated to Scope B, hit while using the
tool for it).

## Threshold: measured, not guessed

Ran BOTH frozen weld-lane samples with the guard OFF (today's behavior) to
get the real d0 (undilated) contour-vs-neighbor overlap-fraction
distribution: `specs/results/rim_neighbor_guard_off_cng_feature.csv/json`
(weld_ab, n=200) and `rim_neighbor_guard_off_leu.csv/json` (leu_ab, n=200).

All nonzero contour_frac values across both samples/both kernel arms:

| value | lot | arm |
|---:|---|---|
| 0.00014 | 713 (leu) | control |
| 0.00033 | 713 (leu) | auto |
| 0.00439 | 582 (leu) | control |
| 0.00477 | 582 (leu) | auto |
| 0.00537 | 995 (leu) | auto |
| 0.01106 | 215298 (cng_feature) | auto |
| 0.01689 | 3661 (leu) | auto |
| 0.03290 | 3661 (leu) | control |
| 0.10582 | 3736 (leu) | control |
| 0.17825 | 3717 (leu) | control |

This reproduces the original "Measured evidence" table's cited lots
exactly (3717 17.8%, 3736 10.6%, 3661 3.3%/1.7%, 995 0.54%, 215298 1.1% —
all match to the last digit) and additionally surfaces lot 713 (leu,
0.014%/0.033%) and confirms 582 (0.44%/0.48%), which the original 7-lot
table cited only approximately.

The smallest real value is 0.00014 (34px, lot 713 control) — d0 is this
tool's own "definitive contamination" form (not a proximity measure like
d3), so there is no principled noise floor above zero to tolerate; 34
contiguous pixels of one coin's alpha landing inside its neighbor's is real
contamination regardless of how small the fraction reads. `src/
layer1_geometry.py`'s `RIM_NEIGHBOR_OVERLAP_MAX_DEFAULT` is set to
**0.0001** (below the smallest measured value, so every real sliver in the
sample reverts) — see the comment there for the full citation.

## Guard-ON confirmation

**cng_feature (weld_ab), full n=200, both arms:** `PASS_d0_contour: true`
(`specs/results/rim_neighbor_guard_on_cng_feature.json`) — the one known
sliver (lot 215298, auto, 1.1%) reverts to exactly 0. Zero `n_detections`
changes vs the guard-off run across all 200 lots × both arms (diffed
directly, not just read off the aggregate).

**leu (leu_ab): targeted re-confirmation, not a full 200-lot re-run.**
A full guard-on sweep across all 200 leu lots was attempted and killed
after running >2h with no completion — NOT a bug in the guard (a guard-OFF
run against the same 40-lot prefix, on the same cached raws, also failed to
finish inside 100s), but the same Hough-cost-tail phenomenon
`ks17_mask_stall_diagnosis_2026-07-22.md` documented for KS-17: some
individual leu lots' Hough recovery calls are genuinely expensive, and a
full re-run at guard-on rate is not worth paying for twice when the
guard's logic is a pure per-lot, per-candidate post-pass with no cross-lot
state. Instead, re-ran guard-on targeted at exactly the 6 lots the
guard-off sweep flagged as having real (nonzero) d0 overlap
(`tools/two_coin_weld_mask_gate.py --lots 713,582,995,3661,3736,3717`,
`specs/results/rim_neighbor_guard_on_leu_targeted.json/csv`):

| lot | arm | sliver_contour_frac OFF | sliver_contour_frac ON | ndets OFF→ON |
|---|---|---:|---:|---|
| 713 | control | 0.00014 | **0.0** | 2 → 2 |
| 713 | auto | 0.00033 | **0.0** | 2 → 2 |
| 582 | control | 0.00439 | **0.0** | 2 → 2 |
| 582 | auto | 0.00477 | **0.0** | 2 → 2 |
| 995 | auto | 0.00537 | **0.0** | 2 → 2 |
| 3661 | control | 0.03290 | **0.0** | 3 → 3 |
| 3661 | auto | 0.01689 | **0.0** | 3 → 3 |
| 3736 | control | 0.10582 | **0.0** | 3 → 3 |
| 3717 | control | 0.17825 | **0.0** | 5 → 5 |

Every real sliver in both samples reverts to exactly zero, in every arm it
appeared in, with **zero `n_detections` change** on any of the 6 lots (the
guard reverts, never drops, a candidate). `PASS_d0_contour: true` on this
targeted set too.

## Synthetic fixture (deterministic, CI)

`tests/test_rim_neighbor_aware.py` — two coins whose true circles overlap
by construction, with the shared boundary erased so Otsu sees them as
separate bitten blobs pre-recovery (mirrors the real mechanism: toning
eating the facing edge). Guard off reproduces 38%/68% overlap (both coins
recovered); guard on reverts both to exactly 0; guard stays off by default
with no env override (bit-identical to today); a threshold set above the
fixture's actual overlap correctly declines to revert. 6 tests, all green.
`tests/test_rim_recovery_cap.py` covers Scope A2 (5 tests) the same way.
Full suite: 230/230 passing (219 pre-existing + 11 new).

## Verdict

**Scope B PASSES its PRECOMMIT bar.** Undilated d0 overlap goes to exactly
zero in every real case found across both frozen samples and both kernel
arms, with zero detection-count regressions anywhere it was checked
(full cng_feature, targeted leu). Ships default-off
(`TRIVALAYA_RIM_NEIGHBOR_GUARD` unset = today's behavior, bit-identical —
verified by the full 230-test suite and by the guard-off numbers above
matching the pre-existing "Measured evidence" table exactly).

**Open item, not blocking:** a true full-200-lot leu guard-on sweep was not
completed (cost, not correctness) — if the owner wants that exact
data point before sign-off, it can be run as an unattended job (likely
1-3h+ given the demonstrated cost variance) rather than folded into this
session's interactive loop.
