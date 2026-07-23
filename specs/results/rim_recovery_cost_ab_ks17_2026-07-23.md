# Rim-recovery Scope A cost measurement — KS-17, 2026-07-23

Graded against the PRECOMMIT bars in `specs/rim_recovery_neighbor_aware.md`
("PRECOMMIT ACCEPTANCE BARS / Scope A"), committed 2026-07-22 before any row
below existed. Raw data: `specs/results/rim_recovery_cost_ab_ks17.csv/json`
(cap800) and `specs/results/rim_recovery_cost_ab_ks17_cap1024.csv/json`
(cap1024). Tool: `tools/rim_recovery_cost_ab.py`.

Sample: stride-7 draw from the KS-17 fixture (41 images / 82 sides, both
`ingest` (house=cng_feature) and `query` (house=None) lanes), not the full
287 — see the tool's own docstring for why (baseline alone costs hours at
full population on this 4-vCPU box).

## cap800 (TRIVALAYA_RIM_HOUGH_CAP=800, env-value only, zero code change)

**Cost: a clear, large win.**

| | baseline | cap800 |
|---|---:|---:|
| n_stall (wall>20s) / 82 | 42 (51%) | 0 (ingest), 1 (query) |
| p50 cpu_s | 70.8 | 12.8 |
| p90 cpu_s | 123.7 | 23.8 |
| p99 cpu_s | 208.3 | 41.2 |
| max cpu_s | 208.3 | 41.2 |
| sum cpu_s (82 sides) | 4696.6 | 888.8 |

p99 CPU-seconds drops ~80% (208.3s → 41.2s), comfortably clearing the
precommitted "≥50% reduction" bar. So does p50/p90/sum — this is not a
cherry-picked percentile.

**Structural bar: FAILS as precommitted.** Of 82 sides, only 73 have
UNCHANGED `n_detections`/`rim_recovered` between baseline and cap800 (9
sides changed outcome, 11% — the precommitted allowance was ≤2%,
individually reviewed). Of the 73 "unchanged-outcome" sides, 3 more still
show alpha-mask IoU < 0.995 (worst: 0.873), so 12/82 (14.6%) show a real
structural difference of some kind. This is not IoU noise from anti-
aliasing — inspecting the actual bboxes/circularity shows real content:

| side | baseline | cap800 | read |
|---|---|---|---|
| 755414 obv | 1 det, main coin recovered clean (circ=0.998) | 4 dets: main coin recovery **FAILS** (circ=0.063, kept as skewed 974×1071 seed) + 3 tiny noise specks (69px, 39px, 34px bbox) newly surviving as separate "detections" | **regression** — losing a correct big-coin recovery, plus spurious noise detections that the correct big circle used to contain/suppress via NMS |
| 755526 obv | 2 dets, main coin recovered clean (circ=0.997) | 2 dets, main coin recovery **FAILS** (circ=0.231) | **regression** |
| 755621 obv | 3 dets, main coin recovery **fails** at baseline too (circ=0.086) + 2 tiny noise specks | 5 dets, main coin recovery **succeeds** (circ=0.998) + 3 tiny noise specks | **improvement** — cap800 fixes a case baseline itself got wrong |
| 755377 rev, 755473 rev | recovery fails (flag 0) | recovery succeeds (flag 1) | improvement (same direction as 755621) |
| 755517 rev, 755607 obv, 755707 obv/rev | 1↔2 or 4↔5 or 2↔1 detection-count changes | | mixed — likely the same noise-blob-suppression mechanism as 755414, not independently reviewed line-by-line here |

**Mechanism** (not guessed — read from the code): `TRIVALAYA_RIM_HOUGH_CAP`
changes the resolution `hough_rim_recovery`'s accumulator searches at
(`src/rim_logic.py`). This isn't a monotonic quality knob — Hough's success
at finding a valid circle (passing the 0.12 edge-support gate) is sensitive
to accumulator resolution in *both directions*: coarser search sometimes
finds a clean circle a noisier full-res search missed (755621, 755377,
755473 — arguably a genuine improvement), and sometimes *fails* to find the
circle a full-res search found cleanly (755414, 755526 — a real
regression, since the final crop then bakes in the bitten/irregular seed
contour instead of the correct rim). Downstream, `n_detections` swings
because `non_maximum_suppression`'s containment logic depends on the
recovered circle's size: a correct big recovered circle suppresses small
noise-blob candidates as "contained"; a failed recovery (small/skewed seed)
does not, so those noise blobs survive as extra "detections."

**Verdict on cap800 as tested: does not clear the precommitted structural
bar.** The cost win is real and large, but ~11-15% of sides show a genuine
behavior change, split between regressions and improvements, not simply
"faster, otherwise identical." This is exactly the kind of thing the
mask-IoU gate exists to catch — it is not a false positive from the gate
being too strict on cosmetic drift.

## cap1024 (gentler cap, same mechanism)

| | baseline | cap1024 |
|---|---:|---:|
| n_stall (wall>20s) / 82 | 42 (ingest), 41 (query) | 6 (ingest), 8 (query) |
| p50 cpu_s | 65.7 | 29.1 |
| p90 cpu_s | 122.1 | 55.9 |
| p99 cpu_s | 206.0 | 93.1 |
| sum cpu_s | 4610.8 | 2196.7 |
| outcome_unchanged / 82 | 82 | 77 (5 changed, 6.1%) |
| alpha_iou below 0.995 (of unchanged) | 0 | 5, worst 0.804 |

p99 CPU drops ~55% (206.0s → 93.1s) — still clears the ≥50% cost bar, but
far less dramatically than cap800's ~80%. **The structural bar is not
better at 1024 than at 800**: fewer full outcome-flips (5 vs 9) but *more*
IoU-drift-only sides (5 vs 3, and a worse minimum IoU: 0.804 vs 0.873).
Combined structural impact ≈ 10/82 (12.2%) vs cap800's 12/82 (14.6%) — in
the same range, not meaningfully cleaner, for a substantially smaller cost
win.

**Reading this against the code** (`src/rim_logic.py:108-141`): the cap
only bounds where the Hough *accumulator* searches for candidate circles;
acceptance is already judged against full-resolution edges either way
("Score candidates on the FULL-RES ROI edges... acceptance is judged at
native resolution, same as the uncapped path" — existing 2026-07-11
comment). That means *any* accumulator-resolution change can make the
search propose a different (or no) candidate circle for a borderline coin,
independent of exactly how aggressive the cap is — consistent with
outcome-flips appearing at both 800 and 1024, in comparable proportion,
for different absolute cost wins. This is a property of "coarsen the
accumulator" as a strategy, not a tuning artifact fixable by picking a
gentler number.

## Verdict

**Neither cap800 nor cap1024 clears the Scope A PRECOMMIT structural bar.**
Both deliver a real, large cost win (≥50% p99 CPU-second reduction, cap800
far more so). Both also change real detection outcomes on 6-15% of sides —
some flips are improvements (Hough succeeds at low-res where it failed at
full res: 755621, 755377, 755473 obv/rev), some are regressions (a
correctly-recovered, clean big-coin circle at full res fails to recover at
all at the lower cap: 755414, 755526 obv) — and this project's own doctrine
(image-comparison / embedding-drift gates) treats "some sides get better,
some get worse" as a FAIL, not a wash, absent an explicit owner call that
the net tradeoff is acceptable.

**Recommendation: do not flip `TRIVALAYA_RIM_HOUGH_CAP`'s default as
Scope A's fix.** The mechanism precludes a values-only fix from being
"free" — lowering the cap is not simply "same result, faster" the way the
weld lane's kernel change was for its unchanged-outcome lots. Two paths
forward, neither measured yet:

1. **Owner accepts the tradeoff explicitly**, choosing a cap value by cost
   priority (cap800) or by outcome-conservatism (cap1024, though it is not
   cleanly better on that axis either) — this is a risk-acceptance
   decision, not a code decision, and belongs to the owner, not this
   ticket.
2. **A different Scope A mechanism** that doesn't touch the primary coin's
   own Hough resolution at all. A2 (`TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE`,
   implemented, tested, default off) caps *how many* contours attempt
   recovery per image but does not change the resolution the *allowed*
   (largest-area) contour's own Hough search runs at — so it cannot
   reproduce this specific regression class. It also, per an unscored
   `/tmp` smoke sample, delivers far less cost reduction alone on this
   corpus, because the expensive Hough call is almost always on the
   primary (largest) contour, which a cap≥1 always still allows. A2 is
   shipped (default off) regardless, since it is a real, independent, safe
   improvement for the *documented* "stacks 2-5x" multi-contour case even
   if it does not solve KS-17's specific single-coin-per-side cost
   profile alone.

Both cap800 and cap1024 stay implemented as named configs in
`tools/rim_recovery_cost_ab.py` (env-value only, zero code change,
`TRIVALAYA_RIM_HOUGH_CAP` already existed) so a future re-measurement or an
explicit owner risk-acceptance can be re-run without new tooling.
