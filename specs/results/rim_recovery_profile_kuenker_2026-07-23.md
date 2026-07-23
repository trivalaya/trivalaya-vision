# Scope C — kuenker rim tail confirmation (2026-07-23)

Graded against `specs/rim_recovery_neighbor_aware.md`'s Scope C PRECOMMIT
bar. Tool: `tools/rim_recovery_profile_sample.py` (new this session).
Fixture: frozen kuenker sample (`specs/two_coin_weld_sample_ids.csv`,
`purpose=kuenker_wallclock`, sale 428, n=200) — sampled at `--limit 50`
(not the full 200; see "Sample size" below).

## Result: CONFIRMED

cProfile on the 3 slowest lots by CPU-seconds (of the 50 timed):

| lot | cpu_s (pass 1) | HoughCircles self-time | % of total |
|---|---:|---:|---:|
| 1289 | 205.6 | 199.42 | 99.8% |
| 1070 | 139.1 | 138.72 | 99.8% |
| 1030 | 36.5 | 36.13 | 99.6% |

All three clear the ≥90% CONFIRMED bar (KS-17's own diagnosis measured
~98%; kuenker measures slightly higher). The call stack is identical to
KS-17's: `layer_1_structural_salience` → `_segment_and_extract_candidates`
→ `rim_logic.recover_rim` → `hough_rim_recovery` → `{HoughCircles}`, one
raw OpenCV call, no Python in the loop. Lot 1070's 139.1s CPU matches the
task brief's cited "kuenker rim tail... p99 139s" almost exactly, and lot
1289 (205.6s) lands in the same range as KS-17's own worst observed case
(166.8s) — this is the same cost class, on a differently-segmented corpus
(kuenker is `CLOSE_KERNEL_BY_HOUSE` k=7, vs cng_feature's k=3), not a
coincidence.

**New wrinkle vs KS-17: multiple Hough calls per lot are the norm here,
not the exception.** `ncalls` for `hough_rim_recovery` is 3 (lots 1289,
1070) and 2 (lot 1030) — every one of the 3 slowest kuenker lots triggers
recovery on 2-3 separate contours, whereas KS-17's CNG single-coin-per-side
photos mostly triggered it on 0-1. This means Scope A2
(`TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE`, cap invocations to the largest
qualifying contour) is plausibly a BETTER cost lever on kuenker than it
measured on KS-17 — worth a dedicated A1/A2 cost+accuracy re-measurement
on this sample before any Scope A recommendation is finalized, per the
PRECOMMIT bar's "must be separately measured" requirement (kuenker is not
assumed to inherit cng_feature's numbers).

## Distribution (n=50, timing pass only, no cProfile overhead)

p50 cpu_s 0.155 / p90 23.66 / p99 205.56 / max 205.56 — sharply bimodal
again (most lots are fast; a small tail is extremely expensive), the same
shape as KS-17.

## Sample size

50 of the frozen sample's 200 lots, not the full set — this session's Scope
B work independently demonstrated (leu) that individual lots in this cost
class can make even a `--limit 40` structural-comparison run exceed 100s,
so a full 200-lot cProfile pass was not attempted in this interactive
session. The p99/max figures above should be read as a lower bound on the
true tail; a full-200 run (unattended, likely 30-60+ min given this
sample's own p99) would sharpen it but is very unlikely to change the
CONFIRMED verdict, since the mechanism (not just the statistic) matches
KS-17's exactly.

## Verdict

**CONFIRMED.** Scope C's kuenker rim tail is the same `HoughCircles`-
inside-`hough_rim_recovery` cost class as KS-17, not a different leaf.
Any Scope A fix that is eventually adopted (see `rim_recovery_cost_ab_
ks17_2026-07-23.md` — currently neither cap800 nor cap1024 clears the
structural bar) must be separately re-measured on this sample before being
assumed to transfer, per the PRECOMMIT bar and per this file's own
ncalls-per-lot finding that kuenker's multi-contour-per-lot pattern differs
materially from KS-17's.
