# M1 — background estimator repair: build + measurement

> **▶ MEASUREMENT COMPLETE 2026-07-28. ALL BARS GREEN (0–6).** M1 is merged
> to `main` default-OFF and the serving regression is clean. What remains is
> **not measurement** — it is the owner's call on §6 (sign the class table) and
> §7 (whether to enable). The mid-measurement handoff
> `specs/results/m1_handoff_2026-07-28.md` is retained for its ten traps, which
> are still live for anyone re-running this work; its "running job" and "queued
> order" sections are now historical.

**Ticket:** `specs/background_estimator_repair.md` (Bar 0 closed by owner ruling
2026-07-28, "Proceed to M1"). **Branch:** `bg-estimator-m1`.

**Status of this document:** every bar below is MEASURED. Nothing here is a
placeholder that could be mistaken for a result.

| bar | verdict | where |
|---|---|---|
| 0 | CLOSED (owner ruling) | ticket §Results |
| 1 | **PASS** — 0.0 % → 99.7 % within ±8, both lanes | §3 |
| 2 | **PASS** — 28 tests vs frozen `_golden_pre_m1`; suite 230 → 258 | §4 |
| 3 | **PASS** — no-ops 19 → 0, **0 new**, 19/19 fixtures resolved | §5.1 |
| 4a | **PASS** — 12/12 sides adjudicated, **0 REGRESSED** | §5.2 |
| 4b | **PASS** — 80/80 masks byte-identical | §5.3 |
| 5 | **PASS** — both lanes measured separately; class table signed-ready | §6 |
| 6 | **PASS** — merged, restarted, gate unset; A/B vs frozen pre-M1 golden shows 0 of 230 fixtures differ | §5.4 |

**Production is untouched.** M1 ships behind `TRIVALAYA_BG_CORNER_LOCAL_TRUST`,
default OFF; env-unset is bit-identical to pre-M1 on every input. No default
was flipped, no re-embed or backfill was executed, no DB row was written.

---

## 1. What M1 is

`src/math_utils.py::detect_background_histogram`. The shipped pooled corner test
conflates two different things: *"the corners are noisy"* (which should reject
them) and *"the corners disagree with each other"* (which a smooth composited
backdrop vignette guarantees, even though each corner is locally clean
background). M1 judges each 5×5 patch by its **own** std and, when all four are
locally clean, returns the **median of the four corner medians**.

The new branch sits **after** the pooled test. That ordering is load-bearing: it
can only ever ADD cases where the corners are trusted, so every image whose
pooled `corner_std < 15` today returns from the old branch untouched, gate or no
gate. Bar 2 therefore holds **by construction, not by sampling**.

---

## 2. THE STRUCTURAL FINDING — the value is inert except at one threshold

**This is the headline result of the session and it reshapes the ticket.**

`detect_background_histogram` has **exactly one** production consumer:

```
src/layer1_geometry.py:592   avg_bg, _ = detect_background_histogram(gray)
src/layer1_geometry.py:600   thresh_type = INV if avg_bg > 110 else BINARY
```

`bg_type` is **discarded at the call site**. The returned value is read by a
**single binary comparison** against `BRIGHT_BACKGROUND_THRESHOLD = 110`.
Nothing else in the pipeline reads it. So the estimator's accuracy is invisible
to Layer 1 except where a change **crosses 110**.

Measured on the full KS-17 population, both geometries, separately (never
derived from one another):

| geometry | n | value changed | **crosses 110** | inert | Bar 1 (\|err\|≤8) |
|---|---|---|---|---|---|
| per-side (`--layout half`) | 574 | 572 (99.7%) | **12 (2.1%)** | 560 (97.6%) | 0.0% → **99.7%** |
| full photo (`--layout full`, true ingest) | 287 | 286 (99.7%) | **5 (1.7%)** | 281 (97.9%) | 0.0% → **99.7%** |

Data: `specs/results/m1_ab/threshold_crossing.csv`,
`threshold_crossing_fullphoto.csv`.

All crossings run `INV → BINARY`. **The per-side crossing set is exactly equal
to the Bar 0 no-op set** — set equality, zero difference in either direction:

```
crossing sides: 12    Bar 0 no-op sides: 12    crossing == no-op set: True
in crossing not no-op: []      in no-op not crossing: []
```

Every one of the 12 moves `~215 (light) → 75.0 (dark)` against an outer-ring
truth of `78.95`; error `+135` → `−4.0`.

### 2.1 What follows — recorded so no future reader inherits the broken chain

- **The dark stratum moves 31 → ~78. Both are below 110. Zero behavioral
  change.** The ticket's root-cause chain — *"`avg_bg=31 < 110` ⇒ treat as dark
  ⇒ sets up a bad Otsu split"* — does not hold, because the honest 79 selects
  the **same** polarity as the wrong 31. On dark sides the polarity decision was
  **already correct**.
- Therefore the **84% rim-recovery trip rate, the over-detection, and the
  `MAX_DETECTIONS=5` cap saturation are NOT downstream of the estimator value.**
  Their cause lies downstream of the polarity decision and is currently
  **unowned**.
- **M2 cannot deliver them either**, by the same argument: its outer-ring median
  returns ~79 on dark sides — still below 110, still `BINARY`, still identical.
- The dark-branch error is **real but inert**. Bar 0's "predicate REFUTED" was
  righter than its ruling credited: the dark branch is not merely no-op-free, it
  is *unobservable* to the only consumer.

Stamped as an addendum on `specs/results/rim_stall_taxonomy_2026-07-23.md`,
whose §4.3 "shared upstream cause of 4 of 7 classes" is falsified for the
dark-side classes and survives only for the light one.

### 2.2 Funding, restated

1. **Leg #1 — live serving doctrine violation — IS delivered by M1**, tightly
   bounded, and worth shipping.
2. **Leg #2 — CNG ingest detection quality — is NOT delivered by this ticket**,
   and this finding does not widen its scope. It needs its own root-cause
   ticket aimed downstream of polarity. **KS-17 re-ingest is not unblocked by
   M1** — see §7.

---

## 3. Bar 1 — the estimate itself is correct — **PASS**

Threshold as ratified: within **±8** grey levels of the outer-ring truth on
**≥95%** of sides.

| geometry | OFF (today) | ON (M1) | verdict |
|---|---|---|---|
| per-side, n=574 | **0 / 574 (0.0%)** | **572 / 574 (99.7%)** | **PASS** |
| full photo, n=287 | **0 / 287 (0.0%)** | **286 / 287 (99.7%)** | **PASS** |

Error distribution on the crossing sides: `+134.4 … +146.8` before,
`−3.94 … −3.95` after.

**The two residual sides are named, not rounded away.** `755710 obv` and
`755710 rev` are genuinely light-background photos (outer-ring truth ≈ 201). M1
declines there (a corner is not locally clean) and the light fallback returns
≈ 250, error ≈ +48. They do **not** cross 110 in either arm — both values are
above it, so `thresh_type` is `INV` either way and behavior is unchanged. This
is the residue M2 was scoped for.

---

## 4. Bar 2 — bit-identity where the good path already fires — **PASS**

Asserted, not sampled, per the bar.

- `tests/test_bg_corner_local_trust.py`, **28 tests, all green**. Bar 2 is
  asserted against `_golden_pre_m1` — a **frozen verbatim copy** of the shipped
  function at `a00f502`. Comparing the gate against itself would prove nothing;
  comparing against a copy of the real prior behavior is what makes
  "bit-identical when unset" a measurement rather than a restatement.
- The randomized battery (seed 20260728, 300 images) asserts, for every image:
  gate-OFF ≡ golden; and wherever pooled `corner_std < 15`, gate-ON ≡ golden
  too. The test fails if the battery does not exercise at least 30 pooled-path
  cases, so it cannot pass vacuously.
- Full vision suite: **230 green on `main` → 258 green on the branch** (+28).
  No pre-existing failures in this repo.

**Within KS-17 this bar is vacuous** — pooled corner-trust fires 0/574 there —
and the report says so rather than implying coverage. The corpus-wide bound
comes from the census (§6).

---

## 5. Bars 3, 4 — MEASURED

The A/B completed 2026-07-28: both arms 1,148 tasks (574 sides × 2 lanes),
3 workers, `cv2.setNumThreads(1)`. OFF 1,147 ok + 1 timeout; ON 1,146 ok + 2
timeouts. Data: `specs/results/m1_ab/{ab_off,ab_on}.jsonl`, report
`bar_report.txt`.

### 5.1 Bar 3 — **PASS**

| lane | n | no-ops OFF | no-ops ON | fixed | **NEW** |
|---|---|---|---|---|---|
| fullres | 572 | 12 | **0** | 12 | **0** |
| query518 | 574 | 7 | **0** | 7 | **0** |

All **19/19** Bar-0 fixture no-op sides resolved (still-no-op 0, not-covered 0).
`mask_fallback_reason` transitions: **none** — no side changed reason.

**A positive control the bar did not require.** The OFF arm's no-op set,
computed independently over all 574 sides by a different harness, is *exactly*
the 19-row Bar-0 fixture set — same 12 fullres + 7 query518 side IDs, no
additions, no misses. Since production has no `mask_noop` telemetry (owner
amendment 2026-07-28) and the bar gates on the no-op **count**, this
cross-harness agreement is the evidence that the self-computed
`mask_area_fraction` proxy measures the real thing.

**Changed set: 12 distinct sides / 19 (side,lane) tasks**, direction uniform —
`est ≈ 215 (light fallback) → 75.0` against outer-ring ≈ 79, `no-op True →
False`, detections `1 → 1`. These are the **light-fallback** class named in the
Bar 0 ruling. Five sides changed at fullres only (`755397/rev`, `755619/obv`,
`755619/rev`, `755654/obv`, `755654/rev`); at 518 px they were never no-ops.
Lanes measured separately, never derived (Bar 5's rule).

**The three timeouts are accounted, not waved.** `755630/obv/fullres` ran
173.3 s under OFF against a 180 s wall and timed out under ON. Re-run at 900 s,
one worker: it and `755632/obv` both complete at ~170 s under **both** gates
with **bit-identical** mask outcomes (area 0.309356 / 0.293081 to the digit,
dets 5→5) despite `avg_bg` moving 31.6 → 75.0. Contention against the wall, not
an M1-induced stall — and incidental confirmation of the inertness claim on two
of the slowest inert sides. Neither crosses 110. Data:
`timeout_probe_{off,on}.jsonl`.

Bar 4 stratification is **unavailable, not empty**:
`rim_stall_taxonomy_ks17_classified.csv` contains none of the 12 changed side
IDs. Moot for the gate, since amended Bar 4a is a census.

### 5.2 Bar 4a — **PASS** (zero REGRESSED)

All 12 changed-behavior sides, 24 panels (12 × 2 lanes), four independent
Sonnet readers at six panels each, neutrally framed and explicitly pointed at
what a regression looks like.

**19 IMPROVED · 5 NEUTRAL · 0 REGRESSED · 0 AMBIGUOUS.**

The dispositions corroborate the instrument rather than restate it: all five
NEUTRALs land exactly on the query518 panels of the five fullres-only sides,
and the A/B records those five as **bit-identical** area fractions
(0.344525→0.344525, 0.385924→0.385924, 0.374642→0.374642, 0.336921→0.336921,
0.33219→0.33219). The readers saw inertness where the harness measured it,
blind to it. 24 panels − 5 unchanged = 19 = the changed-task count.

**One reader miscall, overridden and recorded.** Reader C returned NEUTRAL for
`755619/obv/fullres`; direct inspection shows OFF is the source frame with the
slate backdrop intact while ON is the coin on flat grey128, and the instrument
concurs (area 0.998639 → 0.387455). Corrected to IMPROVED. The override moves
*toward* improvement and so cannot affect the zero-REGRESSED gate in either
direction. Reader agreement with the instrument: 23/24.

Two cosmetic sub-findings, neither a regression: `755617/obv` fullres has a
slightly jagged lower-right edge (background/shadow over-inclusion, no coin
metal lost); `755654/obv` fullres has a small curl artifact outside the coin
body (no clipping into the coin).

Worksheet `bar4a_adjudication.csv`, panels `overlays/`.

> **Panel caveat for future readers.** The overlay header burns
> `off_value`/`on_value` from `threshold_crossing.csv`, measured at fullres
> half geometry — so on query518 panels those numbers are the wrong geometry.
> The images are rendered live under each gate and are correct. Judge pixels.

### 5.3 Bar 4b — **PASS** (80/80 identical)

Seeded sample of 40 (seed 20260728) from the 560 value-changed /
behavior-identical sides, both lanes, mask RGBA buffers compared by sha256.

**80/80 IDENTICAL** — query518 40/40, fullres 40/40, 0 DIFFERS.

**The test is not vacuous, which is the point.** The estimator value changed on
**80 of 80** rows and the produced mask buffer was byte-identical anyway. The
inertness claim is therefore no longer an argument from the call site (`avg_bg
> 110` being the only read) — it is measured at the output. Data
`bar4b_inert.csv`.

### 5.4 Bar 6 — downstream serving regression — **PASS**

Merged to `main` (fast-forward, `a00f502` → `867d6e1`) and
`trivalaya-search.service` restarted with the gate **unset**. Gate absence
verified in the systemd unit and `.env` — `TRIVALAYA_BG_CORNER_LOCAL_TRUST` is
set nowhere.

**Serving-path diff is gate-only.** `a00f502..HEAD` touches exactly one file
under `src/`: `math_utils.py`. Its only non-gated edit refactors four
`corners.extend(...)` calls into a `patches` tuple iterated in the same order —
same values, same order, same `np.median`/`np.std`. That is precisely the
property Bar 2's frozen-golden test asserts over a 300-image randomized battery.

| check | pre | post | verdict |
|---|---|---|---|
| `/stats` scalars | 5,209 clusters / 126,475 coins / **4,139 cards** / 1,556 parent / 2,583 child | identical | **IDENTICAL** |
| `/stats` materials | 107 | 107, same list | **IDENTICAL** |
| `routing_bar.py` | PASS 241 (top-1 227, top2-3 14), RED_FLAG **0**, OUT_OF_SCOPE 4 | same | **BYTE-IDENTICAL** (`diff` clean) |
| `stage2_bar.py` | PASSED, every provider fires/abstains as specified | same | **BYTE-IDENTICAL** (`diff` clean) |
| per-slice `expected.yaml` sweep | 193 clean / 37 mismatch of 230 graded | 193 / 37 | **IDENTICAL, 0 fixtures differ** |

**The fixture sweep was run as a true A/B, not as an argument from the gate.**
`routing_bar` and `stage2_bar` had pre-restart baselines captured before the
merge, so those are literal before/after diffs. The 248-fixture `expected.yaml`
sweep did not, so the pre-M1 arm was reconstructed *without touching
production*: the frozen `_golden_pre_m1` from
`tests/test_bg_corner_local_trust.py` was monkeypatched over
`src.math_utils.detect_background_histogram` **before** `appv2`/`decode_crop`
import it, and the identical sweep re-run. Result: **0 of 230 fixtures differ**
in top-1, failing-field set, or rank-of-expected. Tools `topk_sweep.py` /
`topk_sweep_preM1.py`, data `bar6_topk_sweep*.json`.

**All 37 mismatches are therefore PRE-EXISTING and none is attributable to M1.**
Composition, recorded because it is a standing harness-maintenance finding and
not an M1 result:

- **15 size-only** — card identity (material + `stable_key`) correct, member
  count drifted. The `expected.yaml` files were captured 2026-06-06…06-27; the
  cards watermark is 2026-07-07. Size drifts on every recluster by design.
- **22 non-size** — Greek-civic / Hellenistic-royal routing residuals
  (`133_euboia_histiaia`, `149_aeolis_aigai`, `207–211_alexander*`,
  `71/72_corinth*`, …). Most carry the expected key at **rank 2–3**, which is
  why `routing_bar`'s top-3 accept-spec passes them.

> **Two harness gaps found while running this bar — flagged, NOT fixed here
> (out of scope for M1).**
> 1. **16 of 248 `expected.yaml` files do not parse** (`yaml.safe_load`
>    raises): `176/177/178_tetrarchic_*`, `245/246_sasanian_*`,
>    `49_rhodos_plinthophoric`, `53/54_antonine_bronze_*`,
>    `59/60_flavian_bronze_*`, `61/62_mg_bronze_*`, `63/64_nerva_bronze_*`,
>    `65/66_julio_claudian_bronze_*`. Any comparator that catches exceptions
>    skips them **silently** — they are invisible to the bar today.
> 2. **No batch `expected.yaml` comparator existed.** `topk_probe.py` is
>    per-fixture, `topk_probe_batch.py` prints but does not compare, and
>    `routing_bar.py` reads its own `routing_bar.yaml`, not `expected.yaml`.
>    CLAUDE.md's step 3 therefore had no automated implementation.
>    `topk_sweep.py` is the one written for this bar and is a candidate to
>    promote into `visual_search/tests/appv2_regression/`.

---

## 6. Bar 5 — the re-embed decision, as a CLASS table

Per owner scope ruling 2026-07-28: rows are **consumer artifact classes**, the
owner signs per class, and **RE-EMBED and RE-VISION are kept separate** — they
are different decisions:

- **RE-EMBED** — the crop is the same, but the *mask* changes under M1. Rides
  the next recluster cheaply.
- **RE-VISION** — the *detection geometry itself* was wrong. This is the
  modern-CNG over-detection cohort, and **M1 does not address it** (§2.1).

Census: `tools/bg_corner_trust_census.py`, 19 houses × 200 photos (spink: all
152), n=3,752, seed 20260728, **full-photo ingest geometry**, `crosses_110`
basis. **0 load_failed, 0 errors.** Data `census.csv` / `census.json`.

| # | class | population | affected bound (point / 95 % upper) | consumer impact | recommended action | rough cost |
|---|---|---|---|---|---|---|
| 1 | Served modern-corpus embeddings (per-material features / cluster vectors) | 371,747 photos ≈ 369,481 coins / 4,787 cards | **4,964 / 13,689** photos (1.3 % / 3.7 %) — **91 % is house `cng` alone** | live search ranking | **RE-EMBED** at next recluster, scoped to `cng` (+ `mashops`, `stacksbowers`) | low — rides the next recluster; ~5 k crops |
| 2 | Catalog annex vectors (`catalog_ingestion`, old-catalog plates) | Cahn 993 + Hirsch 710 + Helbing 1,177 = 2,880 lots | **261 / 375** lots — Cahn 204 (20.5 %), Hirsch 57 (8.0 %), **Helbing 0** | annex match quality | **RE-EMBED** at next annex refresh; re-run `append_search_annex.py --execute` after any recluster (standing rule) | low — hundreds of plates |
| 3 | Archived screen sheets (KS-17, eLive-93, EA-613) | KS-17 287 photos / 574 sides; EA-613 422 photos | KS-17 **measured** (not extrapolated): **12 sides** per-side / **5 photos** ingest | archived review artifacts, not live | **RE-SCORE only if re-opened** — evidence **pending audit join**, required before any re-score executes | low |
| 4 | Known-pairs scorecard baselines | 22 pairs (20 `cng`, 1 `kuenker`, 1 `gorny`) | ~**2** of 20 `cng` modern-side photos at the 10.0 % `cng` rate | held-out validation baseline | **RE-SCORE** — baseline recomputed, **never tuned** (standing prohibition) | trivial |
| 5 | Query lane | no stored artifacts | n/a | live | **auto-heals at enable, zero re-embed** | none |

**No row of this table is RE-VISION.** M1 changes which pixels get masked, not
the detection geometry. The modern-CNG over-detection cohort is RE-VISION work
and M1 does not address it (§2.1) — that separation is the point of keeping the
two columns apart, and it stays empty here on purpose.

### 6.1 Three census findings the owner should see before signing

**(a) The `crosses_110` basis is vindicated by a single house.** `Otto Helbing
Nachf.` shows **63/200 photos whose estimator VALUE changes (31.5 %) and ZERO
threshold crossings.** On an `m1_fires` basis Helbing would enter this table at
371 lots in scope; the true answer is **0**. Corpus-wide the two bases differ
3.5× (278 m1_fires vs 79 crossings); on KS-17 they differed 48×. Do not revert
the basis.

**(b) The point estimates are not tight, and the zero rows are not proven
zero.** Each house is a 200-photo sample. `leu` (125,929 photos) observed 0/200
— but 0/200 carries a 95 % upper bound of 1.88 %, i.e. **up to ~2,373 photos**.
`mashops` point-estimates 417 with a 95 % upper of 2,314. That is why the table
carries both columns: the served-corpus bound is **4,964 point / 13,689 upper**,
a ~2.8× spread. If the owner wants a tighter number for a specific house before
committing spend, raise `--per-house` for that house; the census is 103 s.

**(c) M1 does not touch the modern CNG archive.** `cng_feature` (18,709 photos,
the post-~2020 `Coin.aspx` lane) shows **0/200 m1_fires** — 189/200 already take
the pooled corner-trust path, so M1's branch is never reached. The affected CNG
mass is entirely the **older `cng` / `Lots.aspx` archive** (45,396 photos,
147/200 m1_fires, 20/200 crossings), which is the same house KS-17 belongs to
(`docs/ks17_vision_requeue_2026-07-26.md`: "the sale lives on `Lots.aspx` =
`cng`"). Consistent with the two-archive split: modern CNG studio shots have
clean corners; the older archive carries the composited backdrop vignette that
is exactly M1's target.

**Row 3 evidence gap, flagged not filled** (owner ruling): the historical no-op
audit — *which archived sheets contain raw-pixel embeds* — **does not exist**.
This table is complete for sign-off without it; **execution of row 3 is not.**

**Not done here, by instruction:** nothing executed, no per-coin lists. Those
are generated at execution time from the same queries.

---

## 7. ENABLE PROPOSAL — scoped to leg 1 only; every step separately owner-gated

**This is a proposal. Enable itself is an owner decision and is not taken
here.** Scope is **leg #1 — the live serving doctrine violation — and nothing
else.** Leg #2 (CNG ingest detection quality) is measured NOT to be delivered
by M1 (§2.1) and is excluded from this sequence by design, not by omission.

1. **Merge M1 default-OFF** and run the standing serving regression with the
   gate unset — Bar 6. *No behavior change; unset is bit-identical.*
2. **Owner signs the §6 class table**, per class. Note that rows 1 and 2 are
   the only ones proposing work, and both are RE-EMBED, not RE-VISION.
3. **Flip `TRIVALAYA_BG_CORNER_LOCAL_TRUST=1` in the serving service only**,
   and re-run the regression with the gate ON. Blast radius at 518 px serving
   geometry is bounded by the crossing set; the expected delta is the healed
   no-op class and nothing else. Serving is where the doctrine violation lives,
   so this is the step that actually pays leg 1.
4. **Re-embed per the signed table**, scoped to `cng` (91 % of the mass) plus
   `mashops` / `stacksbowers`, and to Cahn / Hirsch on the annex side. Skip
   Helbing — 0 crossings measured. Budget against the **95 % upper** column,
   not the point estimate (§6.1b).
5. ~~**KS-17 re-ingest.**~~ **STRUCK.** The re-ingest was held on the premise
   that a repaired estimator would improve CNG detection quality; §2.1 measures
   that it does not — on the dark stratum M1 moves 31 → ~78, both below 110,
   selecting the same polarity. **The M1 dependency is void and should be
   struck from the KS-17 runbook.** Re-ingest on its own merits or hold it for
   the ticket in step 6 — but do not sequence it behind M1.
6. **Open the dark-side root-cause ticket** (downstream of polarity: Otsu on a
   non-bimodal histogram). This is where leg #2's value actually lives, and it
   is unowned today.
7. **Open the standing-telemetry ticket** — `mask_area_fraction` / `mask_noop`
   as first-class fields. Every historical "mask fallback 0/N" bar checked
   `mask_fallback_reason` only and is **blind to this entire class**; this
   session had to compute the instrument by hand and validate it by
   cross-harness agreement (§5.1) precisely because the telemetry is absent.

---

## 8. Reproduction

```bash
cd /home/claudeuser/vision-wt-m1
PY=/home/claudeuser/trivalaya-pipeline/.venv/bin/python

# Bar 2 (fast, no data needed)
$PY -m pytest tests/ -q

# Structural finding, both geometries
$PY -u tools/bg_estimator_threshold_crossing.py --out .../threshold_crossing.csv
$PY -u tools/bg_estimator_threshold_crossing.py --layout full --out .../threshold_crossing_fullphoto.csv

# Bars 1/3/5 A/B  (3 workers; ~2h per arm on this box)
$PY -u tools/bg_estimator_m1_ab.py --gate off --all --out .../ab_off.jsonl --workers 3
$PY -u tools/bg_estimator_m1_ab.py --gate on  --all --out .../ab_on.jsonl  --workers 3
$PY -u tools/bg_estimator_m1_report.py --off .../ab_off.jsonl --on .../ab_on.jsonl \
     --fixtures specs/results/bg_estimator_bar0_clean_2026-07-27.jsonl \
     --classify specs/results/rim_stall_taxonomy_ks17_classified.csv

# Bar 4a / 4b
$PY -u tools/bg_estimator_m1_overlays.py --outdir .../overlays --worksheet .../bar4a.csv
$PY -u tools/bg_estimator_m1_inert_check.py --out .../bar4b.csv

# Bar 5 census (needs .env sourced for DB + Spaces)
set -a; source ~/trivalaya-pipeline/.env; set +a
$PY -u tools/bg_corner_trust_census.py --per-house 200 --out .../census.csv --summary .../census.json
```

**A note on timing numbers from this run.** Workers run with
`cv2.setNumThreads(1)` (3 workers on a 4-vCPU box that also hosts the live
search service; unpinned, OpenCV fanned out to a measured load of 9.2). Elapsed
times here are therefore **not** comparable to the Bar 0 run's, and are reported
for A/B contrast only. The latency leg is retired regardless.
