# Vision quality & cost work — consolidated summary (July 2026)

Written 2026-07-26 (brain session) as the single entry point to the July
vision arc: the two-coin weld kernel lane (CLOSED) and the rim-recovery
lane (open, well-mapped). Every number below is measured and links to the
doc that measured it. Repos: `~/trivalaya-vision` (Layer 1/1.5 code, most
results docs), `~/trivalaya-pipeline` (house plumbing, one diagnosis doc).

> **UPDATE 2026-07-26, later the same day — owner review of this document.**
> **The rim-recovery lane is now CLOSED as a cost lane** (owner ruling; full
> reasoning and the item-by-item disposition are in
> `specs/rim_recovery_neighbor_aware.md` → "Ruling"). Decisive new
> measurement: the production job queue is **empty and has been since
> 07-23 23:18** — runner idle at ~0% CPU, zero failed/canceled/blocked jobs
> in 14 days, the only pending job deferred to `run_after=2026-07-30`.
> Nothing is waiting on Hough, so cost work has no remaining claim.
>
> Of the six open items below, **five are dropped or downgraded and one is
> promoted**: background modeling (was #4) is the shared upstream cause of
> four of the seven taxonomy classes and moves to its own ticket,
> `specs/background_estimator_repair.md`. Read the "Open items" section
> below as historical.

## Production state today (2026-07-26)

| Mechanism | Env flag | State |
|---|---|---|
| Scale-relative close kernel, per-house table (leu→k=5, cng_feature→k=3, all others k=7) | `TRIVALAYA_CLOSE_KERNEL_FRAC=auto` | **LIVE** since 07-21 15:49Z, §6.8-validated in production |
| Rim neighbor-aware guard (sliver reverter) | `TRIVALAYA_RIM_NEIGHBOR_GUARD=1` | **LIVE**, production-validated on 200-lot leu batch |
| Rim-recovery attempts cap (largest contour only) | `TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE` | shipped, **default-off** (near-zero effect on CNG; unmeasured on kuenker) |
| Hough resolution caps (800/1024) | `TRIVALAYA_RIM_HOUGH_CAP` | measured, **owner-REJECTED** (outcome changes 6–15% of sides) |
| Disc-test skip-Hough guard (mechanism #1) | `TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD` | built + measured, **FAILED bars, does not ship** (branch `rim-trigger-shape-guard`, unmerged) |
| Hough ROI cap (the 07-11 dim⁴ fix) | `HOUGH_ROI_CAP=1280` | LIVE (pre-existing) |

Query lane (`trivalaya-search`) carries **none** of the new flags — no
EnvironmentFile; verified in `/proc/<pid>/environ`. Query masking stays
pinned to k=7 geometry, consistent with the served corpus (the "both
lanes" doctrine, below).

## Lane 1 — Two-coin weld kernel (CLOSED 2026-07-23)

**Problem.** The fixed 7×7 MORPH_CLOSE kernel has ~12px bridging reach and
welds adjacent coins in two-coin lots → weld crops + forced Hough splits.
Leu's 2-blob gap median is 12.00px — exactly on the reach; leu alone was
47% of all corpus Hough-split welds. cng_feature gaps ≈7px (90.5% welds);
kuenker ≈25px (safe).

**Fix.** Scale-relative kernel (vision `e737031`), gated off, then
membership-gated auto: only measured houses move (leu→5, cng_feature→3),
everything else stays k=7. House plumbing through the pipeline
(`5e7fd95`) so `auction_house` actually reaches Layer 1 — before that fix
the table was unreachable in production (found during rollout, would have
sent leu to k=3, its worse setting).

**Measurements (chronological).**
- `results/two_coin_weld_ab_leu_20260720.*` — leu A/B, 200 lots: k=3
  kills Hough 60→0% and welds 48.5→0% but triples fragmentation → rejected.
- k=5 sweep (07-21, deterministic, byte-identical structural cols): **k=5
  is leu's operating point** — Hough 60→8.5%, weld 48.5→7.0%,
  fragmentation identical to k=7.
- §5.5 idle-box wall-clock: 0.036× vs the 1.5× bar → enable green-lit.
- `results/two_coin_weld_section68_nextday_20260722.md` — §6.8 attempt 1:
  NO DATA (empty queue since enable).
- `results/two_coin_weld_ks17_20260723.md` — §6.8 attempt 2: FALSE
  TRIGGER. KS-17's job was house-mistagged (cng_feature vs stored 'cng')
  and processed nothing; the job's "1494 coins" was a house-blind global
  pair sweep of old kuenker detections. Two operational defects found.
- `results/two_coin_weld_leu_batch_20260723.md` — **§6.8 CLOSED** by a
  forced 200-lot leu/75 production reprocess (real `vision --source` path,
  full preimage): **Hough 41.7%→8.27%** (sweep predicted ~8.5%), leak
  check clean on every other house, fragmentation 4/200 = the accepted k=5
  cost, pairing 199/200 (the miss is a pre-existing documented case).

**Instrument lessons kept:** fragment_rate was invalid on
non-2-coin-uniform houses AND was a rollout gate — validate the
instrument before trusting the gate. The bridge-reach formula is 1–2px
optimistic (ellipse diagonal connectivity) — no longer treated as exact.

## Lane 2 — Rim recovery: Hough cost + neighbor-aware validation (OPEN)

Ticket: `specs/rim_recovery_neighbor_aware.md` (bars + owner rulings
recorded in-file).

**Diagnosis** (`~/trivalaya-pipeline/specs/results/ks17_mask_stall_diagnosis_2026-07-22.md`):
~98% of all pathological vision cost is ONE leaf — `cv2.HoughCircles`
inside `rim_logic.hough_rim_recovery`, 40–166 CPU-s per firing, ~55% of
KS-17 sides, bimodal (<1.5s vs >20s), content-driven. Zero exceptions,
zero mask fallbacks (an earlier "exception→raw fallback" claim was a
probe-timeout artifact, retracted). Same mechanism confirmed on kuenker
(`results/rim_recovery_profile_kuenker_2026-07-23.md`, 99.6–99.8% of self
time) and leu.

**Scope A — resolution caps: FAIL, owner-rejected**
(`results/rim_recovery_cost_ab_ks17_2026-07-23.md`): cap800 −80% /
cap1024 −55% p99 CPU, but 6–15% of sides change detection outcome in
both directions. Owner ruling 2026-07-23: **no speed-for-accuracy trade,
ever** — cost work may only change outcomes in the already-pathological
tail.

**Scope B — neighbor-aware guard: SHIPPED, LIVE, VALIDATED**
(`results/rim_neighbor_guard_sweep_2026-07-23.md`): reverts a recovered
rim that overlaps a neighboring coin (threshold 0.0001 set from the
measured overlap distribution; smallest real sliver 0.00014). Every known
sliver in both frozen samples → exactly 0; zero detection-count
regressions; deterministic synthetic fixture. Production-validated on the
§6.8 leu batch (all 6 known sliver lots clean in real output). This fixed
a live production defect — k=7 was actively creating sliver crops on leu.

**Taxonomy — why images trip recovery at all**
(`results/rim_stall_taxonomy_2026-07-23.md`, 574 sides / 4,701 contours,
montages + full CSVs): by Hough CPU — low_contrast_coastline 40.4%
(correctly-segmented coins failing a resolution-naive circularity metric:
hull 0.965 vs raw 0.096; 4× downscale moves raw to 0.573),
backdrop_vignette_blob 22.3% (CNG's byte-identical composited backdrop
template defeats `detect_background_histogram` on 574/574 sides),
unclassified_ragged 15.1% (unresolved), non_circular_flan 15.0%
(klippen — correctly not disc-like), the rest ≤5% each. The folklore
classes (attached shadow, specular glare) measured ≈0. Where Hough is
expensive its answer is discarded 17/21 in favor of the geometric fit.

**Mechanism #1 — disc-test skip-Hough guard: FAIL, does not ship**
(`results/rim_trigger_shape_guard_bars_2026-07-24.md`): perfect on its
6-side design sample; the full-population run failed Bar 1 in both lanes
(ingest 2.96% changed / worst mask IoU 0.642; query 3.66% / 0.592) and
Bar 2 (≥10 confirmed control-right regressions ARE correct recoveries
skipped). Root cause: on dark high-relief coins the Otsu seed traces the
RELIEF — its envelope passes the disc test, but only Hough finds the true
rim. Bar 4 (kuenker+leu) passed clean → the failure is CNG-specific, but
CNG is 42k coins. Side-fact banked: the query lane's JPEG round-trip is
NOT geometry-inert vs ingest (firing sets 138 vs 140).

**Open items, ranked** — *superseded 2026-07-26; kept as the historical
ranking that the closure review overturned.* (1) trigger-metric fix —
hull/downscaled circularity so the 40% correctly-segmented class stops
triggering recovery at all (unfunded, next recommended attempt); (2) klippen
corner-clipping — rim recovery actively damages square coins (kuenker
1070: 58.6 CPU-s making the crop worse) — a CORRECTNESS bug at any
speed, independent of cost; (3) time-budget escalation; (4) background
modeling (kills the backdrop class + the estimator misfire); (5) cheap
A2-on-kuenker measurement; (6) unclassified_ragged root-cause.

**Disposition 2026-07-26:** (1) DROPPED — same feature family as the failed
mechanism #1, on the same class; a relief-tracing seed has high hull
circularity too. (2) DOWNGRADED to logged known-limitation — sized at
~400–700 coins = 0.1–0.2% of corpus, no detector exists; a
circularity/solidity DB triage is available first and needs no vision run.
(3) DROPPED — `cv2.HoughCircles` has no abort hook. (4) **PROMOTED** to
`specs/background_estimator_repair.md`. (5) DROPPED — moot without a cost
mandate. (6) DROPPED — no known correctness harm; folds into (4) if it is a
background-regime artifact.

**The ranking error worth remembering.** Items 1, 3, 5, 6 and part of 2 were
all downstream consequences of item 4. Four of the taxonomy's seven classes
(`backdrop_vignette_blob`, `relief_self_segmentation`, `sub_coin_noise_blob`,
`low_contrast_coastline` — 62.7%+ of Hough CPU) share one upstream cause:
`detect_background_histogram` returns a background level ~48 grey levels
wrong on 100% of the CNG corpus, because its corner-trust test fires 0/574
against a vignetted composited backdrop and its fallback returns
`mean(pixels<50)` = 31.2 where the truth is 79.0. Three cost mechanisms were
built and rejected attacking those classes individually; none touched the
cause. **When a taxonomy's classes correlate with a single upstream regime,
rank the regime, not the classes.**

## Test inventory

Full suite: **244 passing** (both venvs where applicable). July additions:
- `tests/test_rim_neighbor_aware.py` — 6 (synthetic sliver fixture, guard mechanism in isolation)
- `tests/test_rim_recovery_cap.py` — A2 cap behavior
- shape-guard tests — 14 (predicate + `recover_rim(skip_hough)` plumbing with spies; on the unmerged branch)
- weld-lane kernel/membership tests — incl. the assertion that the per-house clamp deliberately neuters future sweeps
- OFF-arm bit-identity is verified per mechanism (env unset ⇒ byte-identical detections vs main — checked against stored runs, not assumed)

## Method doctrine this work established (standing)

1. **Precommit bars before results exist** (ea614 convention). They
   blocked three plausible-on-a-sample mechanisms (k=3, the caps, the
   shape guard) that full-population measurement disproved.
2. **Both lanes always**: any L1/L1.5 geometry change is judged on ingest
   detection AND query masking; query/corpus geometric consistency is
   load-bearing for same-coin matching.
3. **Default-off env gates** + bit-identity proof, enable as a separate
   owner-gated step.
4. **Validate the instrument before trusting the gate** (fragment_rate;
   the probe-timeout "exception" artifact).
5. **Cost is the reward, never the justification** (owner ruling
   2026-07-23) — outcome changes confined to the already-pathological tail.

## Results-doc index

| Doc | One-line finding |
|---|---|
| `two_coin_weld_ab_leu_20260720` | k=3 kills leu welds but triples fragmentation → rejected |
| k=5 sweep (spec §, 07-21) | k=5 = leu operating point; frag identical to k=7 |
| `two_coin_weld_section68_nextday_20260722` | §6.8: no post-enable volume existed |
| `two_coin_weld_ks17_20260723` | KS-17 job house-mistag; "1494" = global pair sweep |
| `two_coin_weld_leu_batch_20260723` | §6.8 CLOSED: Hough 41.7→8.27%, all bars pass |
| `ks17_mask_stall_diagnosis_2026-07-22` (pipeline repo) | All stall cost = one HoughCircles leaf; no correctness bug |
| `rim_recovery_cost_ab_ks17_2026-07-23` | Caps: big savings, 6–15% outcome churn → rejected |
| `rim_neighbor_guard_sweep_2026-07-23` | Neighbor guard: every sliver → 0, zero regressions |
| `rim_recovery_profile_kuenker_2026-07-23` | Kuenker tail = same Hough leaf; A2 may transfer |
| `rim_stall_taxonomy_2026-07-23` | Why images trip recovery — measured class table, folklore overturned |
| `rim_trigger_shape_guard_bars_2026-07-24` | Mechanism #1 fails bars; Hough is load-bearing on CNG relief |
