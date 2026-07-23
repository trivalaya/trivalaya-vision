# Rim-recovery lane: Hough cost + neighbor-aware validation

> **Status: IN PROGRESS 2026-07-22** (branch
> `rim-recovery-cost-and-neighbor-aware`). Successor to the two-coin weld
> rollout ([[two-coin-weld-kernel-lane]]); scope grew 2026-07-22 to also own
> the KS-17 mask-stall cost class once `specs/results/
> ks17_mask_stall_diagnosis_2026-07-22.md` pinned its root cause. Three
> items, one lane — see "Scope" below. PRECOMMIT bars in this file are fixed
> **before** any measurement in "Results" is read against them, per this
> project's standing precommit doctrine (`analysis/prehammer_estimate/
> ea614_verdict_precommit_2026-07-21.md`).

## Scope (three items, one lane)

- **A. Hough cost.** `cv2.HoughCircles` inside `rim_logic.hough_rim_recovery`
  (Layer 1.5) is the confirmed hot leaf (~98% of stall time, one raw OpenCV
  call) behind the KS-17 mask/embed stall — sharply bimodal, ~55% of sides,
  up to 166.78 CPU-s on a single call. See "Diagnosis" below. Candidate
  fixes: cap the working resolution harder (existing
  `TRIVALAYA_RIM_HOUGH_CAP` knob, toward `two_coin_resolver.py`'s proven
  `WORKING_MAX_DIM=800`), and/or cap how many contours per image are allowed
  to invoke recovery at all.
- **B. Neighbor-aware validation** (the lane's original ticket). Layer 1.5
  can produce a coin mask that swallows part of its neighbour, and nothing
  in the pipeline can currently reject it. See "The defect" below — this
  section is unchanged from the original ticket.
- **C. Kuenker rim tail.** A separate hough=0 cost class (p99 139s) observed
  on kuenker. Working hypothesis: same `HoughCircles`-in-`recover_rim` cost
  mechanism as (A), on a differently-shaped corpus (kuenker is a
  `CLOSE_KERNEL_BY_HOUSE` k=7 house, unlike cng_feature's k=3). Must be
  **confirmed by profiling a kuenker sample**, not assumed — if a different
  leaf dominates there, (C) is its own investigation, not folded into (A)'s
  fix.

## Diagnosis (folded in from `specs/results/ks17_mask_stall_diagnosis_2026-07-22.md`)

Do not re-litigate — full detail in that file. Load-bearing facts for this
spec:

- Exact hot path: `analyze_image` → `layer_1_structural_salience` →
  `_segment_and_extract_candidates` → (per Otsu contour with
  `circularity < 0.65` AND `area/enclosing_circle_area < 0.85`) →
  `rim_logic.recover_rim` → `hough_rim_recovery` → **`cv2.HoughCircles`**.
- **The "exception → raw fallback" framing that originally motivated part of
  this ticket is RETRACTED.** Zero exceptions / zero mask fallbacks across
  62 offline reproductions and the full KS-17 production run (0/287). The
  mid-run `mask_fallback_reason: "exception"` reading was a probe's own
  ~45s timeout racing a slow-but-correct `HoughCircles` call into
  `appv2.py`'s broad `except Exception` (L848-850) — not a bug in
  `appv2.py` or the vision pipeline. **This spec does not touch that catch
  site.** The KS-17 screen sheet is full-metric; no re-runs needed on its
  account.
- 1500×1440 CNG halves sit under `MAX_DIMENSION=3200` (never downscaled) and
  barely clear `HOUGH_ROI_CAP=1280` (scale factor ≈0.85) — the accumulator
  search still runs near-full-res against a high-edge-density ROI.
  `_segment_and_extract_candidates` calls `recover_rim` once per qualifying
  contour, so a cluttered image (dirt, holder marks, toning) stacks 2-5×.
  Sharply bimodal: every fast case <1.5s wall, every stall case >20s wall,
  nothing in between — a binary code-path branch, not continuous scaling.
- Real CPU, not sleep-wait: `process_time()` runs 1.7-1.9× wall on stalls
  (OpenCV's internal thread pool).
- Nothing in file metadata (dimensions, file size, background darkness,
  final `n_det`) predicts stall vs fast — only the pre-suppression
  circularity/area-ratio gate does, and that isn't observable without
  running the pipeline.

## Item B: The defect

Layer 1.5 rim recovery can produce a coin mask that swallows part of its
neighbour, and nothing in the pipeline can currently reject it.

When a candidate's circularity is below `CIRCULARITY_RELAXED` (0.65) and its
area fills less than 85% of its `minEnclosingCircle`, `recover_rim` fits a
circle and **replaces the true contour with it**
(`src/layer1_geometry.py:271`, `final_c = new_c`). That contour is what
`crop_with_alpha` bakes into the alpha channel, so it is what reaches the
embedding.

The guard is `math_utils.validate_rim_recovery(recovered_contour,
seed_contour, image_shape)`. Its four checks are:

1. basic contour validity (`min_area=100`)
2. bounding box within 1.1× the image
3. recovered centroid within 30% of the seed's bbox size
4. recovered area ≥ 90% of the seed area

**Every one is self-referential.** The signature carries no information
about any other candidate, so the function cannot express "this rim now
overlaps the coin next to it". A rim that expands sideways into a
neighbouring flan passes all four checks cleanly.

### Measured evidence

From §4.7 of `two_coin_weld_morph_close.md`, the first run of §9.3c option
2b. Overlap is undilated — filled contour against filled contour, i.e. real
alpha contamination, measured as a fraction of the neighbour's area.

| lot | house | arm | overlap | rim_recovered |
|---|---|---|---:|---|
| 3717 | leu | control (k=7) | **17.8%** | 3 of 5 detections |
| 3736 | leu | control (k=7) | **10.6%** | 3 of 3 detections |
| 3661 | leu | control (k=7) | 3.3% | 1 of 3 |
| 3661 | leu | auto (k=5) | 1.7% | 1 of 3 |
| 995 | leu | auto (k=5) | 0.54% | 2 of 2 |
| 582 | leu | both | ~0.45% | 1 of 2 |
| 215298 | cng_feature | auto (k=3) | 1.1% | 2 of 2 |

Every lot with real overlap has at least one `rim_recovered=True`
detection, in whichever arm carries it. Lots that show overlap only under a
3px dilation have `rim_recovered=False` and are clean undilated.

**This is a defect in production today**, not one introduced by the kernel
change: the two worst cases are both in the `control` arm, which is the
fixed 7×7 that production runs right now.

### Why the kernel change is not the fix

Changing the MORPH_CLOSE kernel only alters how many blobs Layer 1.5 is
handed and how well separated they are. On lots 3717 and 3736 that happens
to take the overlap to zero, but incidentally — the k=5 segmentation gives
rim recovery better-separated seeds, so its fitted circles land in a
kinder configuration. Nothing prevents the same failure at k=5 on a
differently-shaped lot, and lot 995 shows the `auto` arm producing 0.54%
overlap where control produced none.

A durable fix helps **both** arms and is independent of the kernel.

### Sketch of the fix

Add neighbour awareness to the accept/reject decision. Two design notes
that matter:

- **It needs a second pass.** The candidate loop in
  `_segment_and_extract_candidates` appends as it goes, so at the moment
  `validate_rim_recovery` is called, later candidates do not exist yet. A
  neighbour-aware check belongs *after* the loop, before NMS: for each
  recovered candidate, compare its filled contour against the filled
  contours of all other candidates and fall back to the original seed
  contour if the overlap exceeds a threshold.
- **Falling back must be possible.** That means keeping the pre-recovery
  seed contour on the candidate (it is currently discarded when `final_c`
  is reassigned), so rejection is a revert rather than a re-run.

Threshold should be measured, not guessed — the §4.7 data suggests real
contamination starts being visible around 0.5% of the neighbour's area, but
that is 7 lots, and the rule this project keeps relearning is that
constants come from sweeps.

## Item A: Hough cost — candidate fixes

1. **Lower `TRIVALAYA_RIM_HOUGH_CAP`** (already an env-gated runtime knob,
   `src/rim_logic.py:108-118`, default 1280 from `RimRecoveryConfig
   .HOUGH_ROI_CAP`) toward `two_coin_resolver.py`'s already-proven
   `WORKING_MAX_DIM=800`. **Zero code change** — this is a config-value
   decision, tested by setting the env var, matching the weld-lane pattern
   of staging an inert env override before any restart/flip.
2. **Cap total rim-recovery invocations per image** to the single largest
   qualifying (`need_recovery=True`) contour, instead of every one. New
   env-gated behavior (default unset = unlimited = today's per-contour
   invocation, bit-identical) since this changes code, not just a constant.
   Requires splitting `_segment_and_extract_candidates`'s single contour
   loop into a cheap geometry pass (area/circularity/edge_support/
   need_recovery — no Hough) followed by a recovery-invocation pass that
   respects the cap. This restructuring is also the natural place to keep
   the seed contour for item B's revert path, so B and A2 land together.
3. **Do not** touch the `appv2.py:848-850` exception catch — the Diagnosis
   section retracts the premise that anything throws there on this code
   path.

## Item C: Kuenker rim tail — confirmation, not assumption

Profile the p99-slowest sides in the frozen kuenker sample
(`specs/two_coin_weld_sample_ids.csv`, `purpose=kuenker_wallclock`, sale
428, 200 lots, raws cached locally under `trivalaya_data/01_raw/auctions/
kuenker/428/`). If the hot leaf matches KS-17's (`cv2.HoughCircles` inside
`rim_logic.hough_rim_recovery`, ≥90% of self time), item A's fix candidates
apply here too — but must be **separately measured** on this sample, since
kuenker's k=7 segmentation hands rim recovery different seeds than
cng_feature's k=3. If a different leaf dominates, C splits into its own
ticket.

## PRECOMMIT ACCEPTANCE BARS

Fixed 2026-07-22, before any row in "Results" (to be added below) exists.
Numbers are this session's proposal, following the ea614-precommit
convention (`analysis/prehammer_estimate/ea614_verdict_precommit_2026-07-21
.md`) of committing thresholds before the graded run — edit history only,
no silent revision once results exist to game against.

### Scope A — Hough cost

Fixture: KS-17 raws (`~/trivalaya-pipeline/analysis/incoming_screen/KS-17/
incoming_images/`, 287 images / 574 sides — report on the WHOLE set, not
just the stalling subset, so a candidate cannot hide a regression on the
fast majority inside an aggregate win). Baseline = today's production
default (`HOUGH_ROI_CAP=1280`, unlimited per-image recovery attempts).
Candidates measured independently: (A1) cap=800 env-only; (A2)
largest-qualifying-contour-only; (A1+A2) combined.

A candidate PASSES iff ALL of:

- **Cost.** p99 CPU-seconds per side on the stalling subset (baseline >20s
  wall / >1.5s cpu, per the Diagnosis's bimodal split) drops by **≥50%**
  vs baseline. (Baseline worst observed: 166.78 CPU-s.)
- **Structural (mask-IoU).** For every side where `n_detections` and each
  candidate's `debug_data.rim_recovered` flag are UNCHANGED vs baseline,
  alpha-mask IoU (filled contour union, matching
  `tools/two_coin_weld_mask_gate.py`'s `alpha_iou`) ≥ **0.995** — the same
  `IOU_BAR` the weld lane used for "moving crops it wasn't supposed to
  touch."
- **Embedding drift, BOTH lanes, masked-transparent-grey128 only** (image-
  comparison doctrine, CLAUDE.md): per-side cosine similarity between
  baseline-config and candidate-config embeddings, computed through
  `appv2._mask_query_image_meta` / `cm.embed_query` — never a raw-photo
  embed. Median ≥ **0.995**, worst-case ≥ **0.98**. Measure via (i)
  `analyze_image()` direct (ingest/detection lane, house=cng_feature) AND
  (ii) `appv2._mask_query_image_meta()` (query lane, exactly as
  `embed_query` calls it) on the same images — **a candidate that clears
  one lane and not the other is a FAIL**, per the "ingest AND query
  geometric consistency is load-bearing" doctrine
  ([[two-coin-weld-kernel-lane]]). Check `masked`/`mask_fallback_reason` on
  every embed call; a silent mask no-op invalidates that row, not a pass.
- **Detection count.** `n_detections` changes on ≤2% of sides, and every
  changed side is individually reviewed and noted in the results doc (no
  silent bucket).

Ranking if more than one candidate passes: prefer the smaller code
footprint. A1 (env-value only, zero code change) beats A2; A2 alone beats
A1+A2 combined unless A1 alone misses the cost bar.

### Scope B — neighbor-aware validation

Fixture: both frozen weld-lane samples (`weld_ab`=cng_feature n=200,
`leu_ab`=leu n=200), run through BOTH `control` (fixed k=7) and `auto`
(scale-relative + per-house table) MORPH_CLOSE arms via
`tools/two_coin_weld_mask_gate.py`, extended with the neighbor-aware guard
toggled on/off. Plus a new §9.2-tier synthetic fixture (two adjacent
low-circularity fragments, deterministic).

PASSES iff ALL of:

- Undilated (`d0`) `gate_sliver_9_3c_2b` contour overlap goes to **exactly
  zero** across BOTH arms on BOTH frozen samples (verbatim from the
  original ticket's Acceptance section).
- No regression in `n_detections` or the weld-lane's GREEN rate on either
  sample — a per-lot diff, not an aggregate mean.
- Threshold is chosen from a sweep over the guard-off overlap-fraction
  distribution (report it), not a single guessed value — pick the smallest
  threshold that reverts every real sliver in the sample while reverting
  **zero** legitimately-recovered clean rims (a false-revert on a genuinely
  correct Hough recovery is a FAIL unless individually reviewed and judged
  correct).
- The synthetic fixture is deterministic: guard-off shows nonzero overlap,
  guard-on shows zero, on a fixed seed, no flakiness across reruns.

### Scope C — kuenker rim tail

Fixture: frozen kuenker sample (`kuenker_wallclock`, sale 428, n=200).

- **CONFIRMED** iff cProfile on the p99-slowest sides shows the same
  `cv2.HoughCircles`-inside-`hough_rim_recovery` leaf at ≥90% of self time
  (matching KS-17's ~98%). Otherwise **NOT CONFIRMED** and C is split into
  its own ticket, not folded into A's fix.
- If CONFIRMED, whichever A-candidate(s) passed Scope A must be
  **separately re-measured** against the SAME cost/mask-IoU/embedding-drift
  bars on this kuenker sample — passing on cng_feature does not transfer
  for free to a k=7 house.

### Cross-cutting (all three scopes)

- Every new/changed code path ships **env-gated, default = today's
  behavior** (unset env == old code path, bit-identical) — the weld-lane
  pattern. No default flip lands in this work; any production enable is a
  separate, explicit owner-gated step.
- `rim_logic.py` / `layer1_geometry.py` / `math_utils.py` are imported by
  `decode_crop.py`'s `analyze_image` call from `appv2.py` — in scope for
  CLAUDE.md's visual_search regression bar. After merge (still gate-off),
  restart the search service and run `topk_probe.py` per-slice fixtures,
  `routing_bar.py`, and `stage2_bar.py` — behavior must be byte-identical
  to pre-restart with the gate unset, since nothing was enabled.
- Everything in this section is a PROPOSAL until committed to git; edit
  freely before that commit lands, not after "Results" below has entries to
  grade against it.

## Results (2026-07-23)

- **Scope A (cost) — does not clear its bar as tested.** cap800 and
  cap1024 (`TRIVALAYA_RIM_HOUGH_CAP`, env-value only) both cut p99
  CPU-seconds ≥50% on the KS-17 fixture (cap800: -80%, cap1024: -55%), but
  both change real detection outcomes on 6-15% of sides — some improve
  (Hough succeeds at lower res where full-res failed), some regress (a
  clean full-res recovery fails at the lower cap, and the crop falls back
  to the bitten seed contour) — which is a FAIL against the precommitted
  ≤2%-individually-reviewed bar, not a rounding difference. Full detail
  and the case-by-case bbox/circularity readout: `specs/results/
  rim_recovery_cost_ab_ks17_2026-07-23.md`. **No default flip
  recommended**; this needs an explicit owner risk-acceptance call or a
  different mechanism (see that file's "Verdict").
- **Scope B (neighbor-aware validation) — PASSES.** Real-data sweep on
  both frozen weld-lane samples (cng_feature n=200 full, leu targeted at
  the 6 lots the guard-off sweep flagged) reproduces every previously-cited
  sliver (plus one new one, lot 713) and reverts every one of them to
  exactly zero d0 overlap, zero `n_detections` regressions anywhere
  checked. Threshold (`RIM_NEIGHBOR_OVERLAP_MAX_DEFAULT = 0.0001`) is set
  from the measured distribution, not guessed. Plus a deterministic
  synthetic fixture (`tests/test_rim_neighbor_aware.py`, 6 tests) proving
  the mechanism in isolation. Detail: `specs/results/
  rim_neighbor_guard_sweep_2026-07-23.md`. Ships default-off
  (`TRIVALAYA_RIM_NEIGHBOR_GUARD` unset).
  **Production-enabled 2026-07-23** (owner approval, bundled with the
  weld-lane §6.8 closure batch): `TRIVALAYA_RIM_NEIGHBOR_GUARD=1` added to
  `trivalaya-pipeline/.env`, `trivalaya-runner.service` restarted, confirmed
  live in the runner's environ and absent from `trivalaya-search`'s. The
  forced leu/75 200-lot batch gave the full-sample leu confirmation the
  offline sweep couldn't get (its own guard-on 200-lot attempt ran >2h and
  was killed): all six known sliver lots (713, 582, 995, 3661, 3717, 3736)
  are sliver-free (d0=0.0) in the real production output, zero ndets
  regressions attributable to the guard. Detail: `specs/results/
  two_coin_weld_leu_batch_20260723.md`.
- **Scope C (kuenker tail) — CONFIRMED.** Same `cv2.HoughCircles`-inside-
  `hough_rim_recovery` leaf, 99.6-99.8% of self time on the 3 slowest of a
  50-lot kuenker sample (not the full 200 — see that file's "Sample size").
  kuenker triggers 2-3 recovery calls per slow lot (vs KS-17's 0-1),
  meaning Scope A2 may transfer better here than it did on KS-17 — not yet
  separately measured. Detail: `specs/results/
  rim_recovery_profile_kuenker_2026-07-23.md`.
- **Cross-cutting: all shipped default-off, 230/230 tests passing**
  (219 pre-existing + 11 new: `tests/test_rim_neighbor_aware.py`,
  `tests/test_rim_recovery_cap.py`). Scope A2
  (`TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE`) is implemented and tested
  alongside Scope B in `src/layer1_geometry.py`'s restructured
  `_segment_and_extract_candidates` (a two-pass design was needed for both
  A2 and B simultaneously — see the function's own comments) even though
  its KS-17 cost measurement showed near-zero effect alone; it is a real,
  independent, default-off improvement for the documented "stacks 2-5x"
  multi-contour case and may matter more on kuenker (see Scope C).
- Incidental fix, unrelated to any scope: `tools/two_coin_weld_mask_gate.py`
  had a pre-existing crash (`TypeError`) in `PASS_d0_contour` when a lot
  has <2 detections; fixed (filters `""` before `max()`, matching the
  pattern `sliver_stats` already used) since it was hit while running
  Scope B's own sweep.

## Ruling

Owner ruling 2026-07-23: Scope A caps REJECTED — no speed-for-accuracy
trade. Cost work continues only via mechanisms whose outcome changes are
confined to the currently-pathological tail.

Owner ruling 2026-07-23 (mechanism #1): the disc-test skip-Hough-only guard's
PRECOMMIT bars in `specs/results/rim_stall_taxonomy_2026-07-23.md` §7 are
RATIFIED as drafted. The guard ships default-off (`TRIVALAYA_RIM_TRIGGER_
SHAPE_GUARD` unset = bit-identical); the default flip is a separate
owner-gated step after Bar 4. Measured run:
`specs/results/rim_trigger_shape_guard_bars_2026-07-24.md`.

Owner ruling 2026-07-23 (Scope B production enable, step 2 of the 4-step
sequence): approved, bundled with the weld-lane §6.8 forced leu batch.
Both validated in production the same session — see the Scope B bullet
above and `specs/results/two_coin_weld_leu_batch_20260723.md`.

## Acceptance (superseded by the PRECOMMIT bars above)

Kept for the original ticket's exact wording, now formalized into Scope B's
bars:

- Re-run `tools/two_coin_weld_mask_gate.py` on both frozen samples. The
  undilated `contour` overlap should go to zero in **both** arms — including
  control, since the fix is kernel-independent.
- No regression in detection count or GREEN rate on the frozen samples;
  rim recovery exists because it genuinely rescues fragmented coins, and a
  fix that simply stops recovering rims would trade one defect for another.
- A synthetic fixture in the §9.2 tier: two adjacent low-circularity
  fragments where the naive fit overlaps and the neighbour-aware one does
  not.

## Related

- `specs/results/ks17_mask_stall_diagnosis_2026-07-22.md` — the profiling
  that established Scope A and retracted the exception framing.
- `specs/two_coin_weld_morph_close.md` §4.7 (the measurement), §7.1 (the
  original sliver fear, now answered), §6.5 (backfill precondition, which
  this revises — Hough crops are *not* clean on leu today).
- `specs/two_coin_weld_reprocess_proposal.md` — the historical population
  that would be affected if Scope B is fixed and a backfill follows.
- `analysis/prehammer_estimate/ea614_verdict_precommit_2026-07-21.md` — the
  precommit-bar convention this file follows.
