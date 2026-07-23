# Rim-recovery shape guard — measured run against the RATIFIED §7 bars

2026-07-24. Measures `TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD` (mechanism #1 of
`specs/results/rim_stall_taxonomy_2026-07-23.md`) against the §7 bars **as
ratified by the owner 2026-07-23** — no silent revision. Guard ships
**default-off**; the default flip is a separate owner-gated step after Bar 4.

**Build.** Branch `rim-trigger-shape-guard` off main (`d2a4464`). New module
`src/rim_shape_guard.py`; `recover_rim` gains `skip_hough` (default False =
bit-identical); pass 2 of `_segment_and_extract_candidates` consults the
env-gated guard. When on, it suppresses ONLY the Hough branch (geometric fit
always runs), only for seed blobs passing the disc test
(`cv_r < 0.06 AND area_ratio >= 0.55`), with an optional
`largest_hole_frac < X` conjunct (`…_MAX_HOLE_FRAC`). Tests:
`tests/test_rim_trigger_shape_guard.py` (14), full suite **244 passed**.

**Kernel-env footnote (ratification condition).** Both arms run with
`TRIVALAYA_CLOSE_KERNEL_FRAC` **unset**. Under that setting the close kernel is
`k=7` fixed and the `house` argument is *never read* in
`_segment_and_extract_candidates` (`if not _frac_env: k = 7`) — so tagging
KS-17 as `cng_feature` vs its real `cng` (vs the query lane's `None`) is
provably geometry-inert. Ingest uses `house=cng_feature`, query uses
`house=None`; L1 is identical between them on identical pixels.

**Lanes.** Both route through `analyze_image` → `recover_rim`.
- **ingest**: raw (lossless PNG) per-side half, `house=cng_feature`, ALL
  detections scored.
- **query**: temp-JPEG(q95) per-side half — exactly `_mask_query_image_meta`'s
  own round-trip — `house=None`, LARGEST-contour scored (what DINOv2 sees).
  The q95 round-trip is NOT pixel-inert (measured), so the query lane is run
  independently, not derived from ingest.

**Efficiency / population accounting.** The guard can only ever REMOVE Hough
calls, so on any side where it fires on nothing, guard-on ≡ guard-off *by
construction* (same code path). Those sides are unchanged without measurement.
The A/B therefore runs both arms only on the **guard-firing** sides (identified
by a cheap no-Hough fire-scan, `tools/rim_guard_fire_scan.py`, validated to
reproduce the committed taxonomy scan's firing set exactly, 17/17 on a 40-image
check). Firing sides: **ingest 138 / 574**, **query 140 / 574**. The Bar-1
denominator is the full 574; non-firing sides count as unchanged by
construction.

Tools (all in `tools/`, guard worktree): `rim_guard_fire_scan.py`,
`rim_guard_ab.py`, `rim_guard_score.py`, `rim_guard_disposition.py`.

---

## Bit-identity of the OFF arm (foundation)

The whole comparison rests on "guard OFF == today". This holds **by
construction**: `rim_shape_guard.should_skip_hough` reads the env flag first and
returns `False` before computing anything when it is unset; `recover_rim` then
runs with `skip_hough=False`, which is the pre-guard code path verbatim (the
only textual change is an `if skip_hough:` branch that is not taken). `git diff
main -- src/rim_logic.py src/layer1_geometry.py` is limited to that branch plus
the guard call. Every A/B "OFF" arm below runs the guard module with the flag
unset, so it *is* today's behaviour. Spot-check: `tests/…::test_env_off_never_skips`
and `::test_pass2_env_off_runs_hough` assert the OFF path runs Hough exactly as
before, and the full suite (244) is green.

**Empirical confirmation.** `analyze_image` from **main** (`d2a4464`, no guard
code on the path at all) vs from this worktree with the flag unset, on 3 firing
sides (755369 obv, 755387 rev, 755366 obv): **byte-identical detections** —
same detection count, same contours, same `rim_recovered` flags on every one.
The OFF arm is not "close to" today's behaviour; it is today's behaviour.

---

## Bar 1 — outcome-identity on the affected sides (BLOCKING)

### 1a. Base disc test (cv_r < 0.06 ∧ area_ratio ≥ 0.55), no hole-frac conjunct

**INGEST lane — FAIL.** 138 firing sides run both arms (raw PNG half,
`house=cng_feature`); the other 436 are unchanged by construction.

| metric | value | bar | verdict |
|---|---|---|---|
| unchanged / population | 557 / 574 = **97.04%** | ≥ 98% | **FAIL** |
| changed-rate | 17 / 574 = **2.96%** | ≤ 2% | **FAIL** |
| worst alpha IoU (min over measured) | **0.642** | ≥ 0.995 | **FAIL** |

The changed-rate exceeding 2% is the ratified trigger for the hole-frac
conjunct re-run (§1c). But the disposition (§1b) is the decisive result: **at
least 10 of the 17 changed sides are control-right / guard-worse regressions**,
not benign drift — so the run fails the bar's qualitative clause ("acceptable
ONLY where control was visibly wrong") independent of the percentages.

*Root cause (measured, not assumed).* On high-relief coins — Greek
tetradrachms, Roman provincial bronzes, medallions — the Otsu seed blob traces
the *relief* (portrait/legend), whose OUTER envelope is roughly circular, so it
passes the disc test (`cv_r < 0.06`) even though it is not the rim.
`geometric_fit_recovery` fits a circle to those relief points and fails or
under-sizes; **Hough recovers the true rim from the edge gradient and is
load-bearing.** Skipping it reverts the crop to the relief contour (rr
`True→False`) or to an undersized circle that clips the coin. The 6-side §6.2
sample happened to draw only sides where the geometric fit already succeeded,
so it read 6/6 unchanged; the full 138 reveal that the disc test cannot
distinguish "geometric fit will recover the rim" from "only Hough can". That
distinction is exactly what the guard would need, and `cv_r`/`area_ratio` do
not carry it.

**QUERY lane — FAIL (worse than ingest).** 140 firing sides (temp-JPEG q95
half, `house=None`, largest-contour scored — the crop DINOv2 actually sees).

| metric | value | bar | verdict |
|---|---|---|---|
| unchanged / population | 553 / 574 = **96.34%** | ≥ 98% | **FAIL** |
| changed-rate | 21 / 574 = **3.66%** | ≤ 2% | **FAIL** |
| worst alpha IoU | **0.592** | ≥ 0.995 | **FAIL** |

The query lane is *more* sensitive because the single largest contour IS the
masked crop: when the guard reverts it to the relief contour, the DINOv2 input
changes wholesale. **9 of 21 are clean rr `True→False` reverts** (e.g. 755394
obv — a Parthian tetradrachm whose OFF rim circle becomes an ON relief crop,
alpha 0.592; overlay `rim_guard_dispo_query/755394_obv.jpg`). 4 sides flip
`False→True` (the largest-contour selection reshuffles when the disc's crop
shrinks) — still changes, not improvements. This is the primary screening/search
path, and it fails hardest.

### 1b. Disposition of every changed side (ingest, base)

Overlays: `specs/results/rim_guard_dispo_ingest/<id>_<side>.jpg` — GREEN = OFF
detections, RED = ON detections, thin blue = minEnclosingCircle of the largest
OFF detection. Sorted worst-alpha first. "hole-frac" = whether the §1c conjunct
would still fire on this side.

| side | ndet | bbox IoU | alpha IoU | rim_recovered | hole-frac | disposition |
|---|---|---|---|---|---|---|
| 755369 obv | 1→1 | 0.890 | 0.642 | [T]→[F] | keeps | **REGRESSION** — OFF's clean rim circle → ON relief-contaminated seed |
| 755362 rev | 1→2 | 0.763 | 0.665 | [T]→[F,T] | keeps | **REGRESSION** — rim lost + noise speck survives |
| 755402 obv | 1→1 | 0.861 | 0.682 | [T]→[F] | keeps | **REGRESSION** — clean rim → relief (two-portrait bronze) |
| 755673 obv | 1→1 | 0.687 | 0.689 | [T]→[T] | keeps | **REGRESSION** — geo circle undersized, clips coin (109704-style) |
| 755366 obv | 2→2 | 0.898 | 0.718 | [F,T]→[F,F] | keeps | **REGRESSION** — Athenian owl rim lost |
| 755605 obv | 3→3 | 0.865 | 0.729 | [F,T,T]→[F,F,T] | keeps | **REGRESSION** — a recovery lost |
| 755610 rev | 1→1 | 0.733 | 0.732 | [T]→[T] | EXCLUDES | **REGRESSION** — undersized (conjunct fixes) |
| 755439 obv | 1→3 | 0.872 | 0.755 | [T]→[F,T,T] | keeps | **REGRESSION** — jaggier main + 2 noise specks |
| 755395 obv | 1→1 | 0.868 | 0.806 | [T]→[F] | EXCLUDES | **REGRESSION** — backdrop_vignette (the §6.4 case; conjunct fixes) |
| 755673 rev | 1→1 | 0.901 | 0.816 | [T]→[F] | keeps | **REGRESSION** — rim lost |
| 755485 rev | 5→5 | 0.0 | 0.818 | [F,T,T,T,T]→[T,T,T,T,T] | keeps | ambiguous — multi-contour provincial; ON gains a recovery; greedy-match bbox artifact |
| 755684 obv | 1→3 | 0.833 | 0.820 | [T]→[T,T,T] | EXCLUDES | NMS churn — coin split + specks (conjunct fixes) |
| 755684 rev | 2→2 | 0.822 | 0.821 | [T,T]→[T,T] | keeps | shift — recovery kept, bbox moved |
| 755446 rev | 1→1 | 0.826 | 0.828 | [T]→[T] | keeps | **REGRESSION** — undersized |
| 755493 obv | 5→4 | 0.0 | 0.866 | [F,T,T,T,T]→[T,T,T,T] | keeps | ambiguous — multi-contour provincial |
| 755544 rev | 3→4 | 0.911 | 0.903 | [F,T,T]→[T,T,T,T] | keeps | NMS churn — ON gains a recovery + a detection |
| 755712 obv | 1→1 | 0.906 | 0.905 | [T]→[T] | EXCLUDES | shift — bbox moved (conjunct fixes) |

**≥ 10 REGRESSIONS** (control-right/guard-worse), 2 ambiguous multi-contour, 5
benign shift/NMS. No changed side is a clean guard-improvement. The regressions
are geometrically real (worst alpha 0.642, several rr `True→False`), not
anti-alias noise.

### 1c. Hole-frac conjunct re-run (pre-approved path if base changed-rate > 2%)

Base changed-rate 2.96% > 2% → the ratified re-run with
`…_MAX_HOLE_FRAC = 0.10`. The OFF arm is reused (guard-off is
config-independent); only ON is recomputed.

The conjunct's firing set on ingest is 121 sides (17 fewer than base). Of the
17 base-changed sides it **EXCLUDES 4** (755395 obv — the backdrop case — plus
755610 rev, 755684 obv, 755712 obv) and **KEEPS 13** — including every clean
coastline regression (755369 obv, 755402 obv, 755366 obv, 755673 obv/rev, …).

Exact re-score (reuse-off ON recomputed with the conjunct; not just the
firing-set bound):

| lane | base changed | hole-frac changed | rate | bar | verdict |
|---|---|---|---|---|---|
| ingest | 17 | **13** | 13/574 = **2.26%** | ≤ 2% | **FAIL** |
| query | 21 | **17** | 17/574 = **2.96%** | ≤ 2% | **FAIL** |

The conjunct removes exactly the class §6.4 said it would (backdrop:
755395 obv etc.), and nothing else. It does **not** touch the dominant failure
mode — high-relief coins where Hough is load-bearing — because those have low
`largest_hole_frac` (they are the coin's relief, not a background hole). Every
clean coastline regression (755369 obv, 755402 obv, 755366 obv, 755673 obv/rev,
…) survives the conjunct. **Hole-frac conjunct: FAIL on both lanes.**

---

## Bar 2 — the guard must not disarm real rim recovery (BLOCKING)

**FAIL** — and the direct evidence is Bar 1 itself: the ≥10 ingest / ≥9 query
regressions ARE correct rim recoveries being skipped. On 755369 obv, 755402
obv, 755366 obv, 755394 obv (query) and others, the OFF arm's Hough recovered
the true rim (a correct, load-bearing recovery) and the guard dropped it. That
is precisely "a correct recovery skipped", so the bar's "zero" is not met.

Two corroborating checks on the specific cases the bar names:

- **Eaten-arc (the 109704 class).** A genuinely bitten coin has a large inward
  bite, so rays into the bite reach the inner edge and `cv_r` is large
  (measured 0.17–0.31 on the synthetic bite fixtures) → `is_disc = False` → the
  guard never fires and full two-stage recovery runs. Proven in
  `tests/test_rim_trigger_shape_guard.py::test_bitten_arc_is_not_disc` and
  `::test_pass2_guard_on_declines_on_nondisc`. So the guard does NOT disarm the
  eaten-arc case — its failure is specific to relief-contaminated *discs*, not
  bitten coins.

- **Frozen weld samples.** On the `cng_feature` weld sample (240 sides from 120
  two-coin plates, `specs/two_coin_weld_sample_ids.csv`) the guard fires on
  **0 sides**: those raws are ~500 px wide, so each coin's blob is too small for
  the radial profile (enclosing radius under the 8-step floor) and `is_disc`
  returns False. The weld fixtures are therefore a *null* test for this guard —
  it never touches them — so they neither confirm nor refute Bar 2. The
  operative evidence is Bar 1 on full-resolution CNG images, where the guard
  demonstrably skips correct recoveries.

**Bar 2: FAIL** (correct recoveries are skipped on full-res CNG high-relief
coins). The one thing that passes is the narrow guarantee the guard was designed
around — it does not touch genuinely bitten coins (eaten-arc) — but that is not
the bar.

---

## Bar 3 — cost (REPORT ONLY)

On the guard-firing sides (where the guard does something), CPU-seconds:

| lane | p99 OFF → ON | total OFF → ON |
|---|---|---|
| ingest (138 sides) | 199.0 → 98.6 s (−50.4%) | 15,337 → 940 s |
| query (140 sides) | 203.7 → 110.7 s (−45.6%) | 16,012 → 1,048 s |

Two caveats keep this from being a headline number, even setting aside the
correctness FAIL:

1. The **population** p99 is barely moved. The guard is inert on the non-disc
   expensive classes — klippen (`non_circular_flan`, 15% of the Hough bill),
   `unclassified_ragged` (15%), `multi_object_weld` — so those stalls remain in
   both arms. The firing-side p99 ON is still ~99–111 s precisely because some
   firing sides *also* carry a non-disc expensive contour the guard leaves
   alone.
2. Per the owner ruling, **cost is the reward, never the justification.** A
   −50% p99 that comes with ≥10 correctness regressions is not a trade this
   project makes (it is the Scope-A failure mode again). Reported for
   completeness only.

---

## Bar 4 — cross-house transfer: kuenker + leu (BLOCKING before any default flip)

Both arms, both lanes, on the frozen kuenker (3, `purpose=kuenker_wallclock`)
and leu (6) fixtures. `TRIVALAYA_CLOSE_KERNEL_FRAC` unset here too.

| house / lane | firing sides | changed | worst alpha | notes |
|---|---|---|---|---|
| kuenker ingest | 0/3 | 0 | 1.000 | klippen are non-disc (`cv_r` high) → guard never fires |
| kuenker query | 0/3 | 0 | 1.000 | same |
| leu ingest | 1/6 (leu713) | **0** | 1.000 | leu713 fires and the skip is a correct no-op (geo recovers the same; 10.3→0.15 s) |
| leu query | 1/6 (leu713) | **0** | 1.000 | same |

**Bar 4: PASS on kuenker + leu — 0 regressions.** The guard is inert on
klippen (correctly declines, taxonomy §6.3) and, on the one leu disc it fires
on (the Celtic bronze leu713, taxonomy overlay), the geometric fit already
recovers the rim so skipping Hough is a genuine no-op — the §6 thesis holding
where it applies. Overlay: `rim_guard_dispo_bar4/leu713_single.jpg`.

**This does NOT rescue the verdict.** Bar 4 is the *transfer* check "before any
default flip"; it shows the guard is safe on kuenker/leu but says nothing about
CNG, where Bar 1 already blocks. In fact it sharpens the diagnosis: the
regressions are **CNG-house-specific** — CNG's dark, high-relief tetradrachms
and provincial bronzes are the ones whose Otsu seed is relief-contaminated yet
disc-shaped. leu's flatter, higher-contrast photos and kuenker's klippen do not
trigger the failure. Since CNG is a 42k-coin house, a mechanism that breaks it
cannot ship regardless of good behaviour elsewhere.

---

## Bar 5 — served-corpus consistency

N/A this session — no re-crop / backfill proposed. The guard stays default-off;
no served vector changes.

---

## Verdict

**Mechanism #1 (the disc-test skip-Hough-only guard) FAILS the ratified bars.
It does not ship. The guard stays default-off (already is); no default flip.**

| bar | result |
|---|---|
| Bar 1 base, ingest | **FAIL** — 97.04% unchanged, 2.96% changed, worst alpha 0.642, ≥10 regressions |
| Bar 1 base, query | **FAIL** — 96.34% unchanged, 3.66% changed, worst alpha 0.592, ≥9 regressions |
| Bar 1 hole-frac, ingest | **FAIL** — 2.26% changed; regressions survive |
| Bar 1 hole-frac, query | **FAIL** — 2.96% changed; regressions survive |
| Bar 2 (don't disarm recovery) | **FAIL** — the Bar-1 regressions are correct recoveries skipped; eaten-arc correctly declined |
| Bar 3 (cost, report-only) | −50%/−46% p99 on firing sides; moot given the FAIL |
| Bar 4 (kuenker + leu) | PASS (0 regressions) — but CNG-specific failure means it can't rescue the flip |
| Bar 5 | N/A (no backfill) |

**Why it fails — one sentence.** The disc test (`cv_r < 0.06 ∧ area_ratio ≥
0.55`) fires on any blob whose *outer envelope* is roughly circular, but on
high-relief coins the Otsu seed is the relief (portrait/legend), whose envelope
is circular yet whose geometric-circle fit fails — so on those sides Hough is
load-bearing, and skipping it reverts or undersizes a correct crop.

**Why §6 said otherwise.** The §6.2 probe drew 6 coastline sides that happened
to be geometric-fit-recoverable, reading 6/6 unchanged. The disc test does not
carry the one bit that matters — "will the geometric fit recover this rim, or
is Hough the only thing that can?" — and `cv_r`/`area_ratio`/`largest_hole_frac`
cannot supply it. This is the ratified bars doing exactly their job: a
plausible-on-a-sample mechanism blocked by the full-population measurement.

**Disposition of the code.** The guard is left in the branch
`rim-trigger-shape-guard`, **default-off and inert** (244 tests green, OFF arm
bit-identical), as the measured record. It is NOT merged to main and the
default is NOT flipped. If cost work on the coastline class resumes, the
open problem is a *positive* rim-present test (does the ROI's annulus carry a
strong circular edge that the geometric fit is missing?) that fires only when
Hough would be discarded — which is a new mechanism needing its own design and
its own ratified bars, explicitly **out of scope** here (no silent revision of
the ratified mechanism). The klippe corner-clipping correctness bug (taxonomy
§4.2) is likewise untouched and still open.
