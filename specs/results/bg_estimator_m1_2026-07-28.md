# M1 — background estimator repair: build + measurement

**Ticket:** `specs/background_estimator_repair.md` (Bar 0 closed by owner ruling
2026-07-28, "Proceed to M1"). **Branch:** `bg-estimator-m1`.
**Status of this document:** Bars 1, 2 and the structural finding are MEASURED
and final. Bars 3, 4, 5, 6 are marked below with their live state — nothing in
this file is a placeholder that could be mistaken for a result.

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

## 5. Bars 3, 4, 5 — measurement state

> **Bar 3 (no-op class eliminated, none created) — RUN IN FLIGHT.**
> Full-population A/B (574 sides × 2 lanes × 2 gate arms = 2,296 mask calls)
> launched 2026-07-27 23:07 UTC, 3 workers, `cv2.setNumThreads(1)`.
> Instrument: `tools/bg_estimator_m1_ab.py`, descended from the Bar 0 probe
> (validated: reproduced its predecessor to max \|area delta\| 0.0037, no-op
> count 12 vs 12 MATCH). Per owner amendment 2026-07-28 the bar gates on the
> no-op **count**, not on which instrument counts it — production `mask_noop`
> telemetry does not exist and was **not** built in this session.

> **Bar 4a (adjudicate ALL 12 changed sides) — QUEUED**, tooling built
> (`tools/bg_estimator_m1_overlays.py`). Runs after the A/B; one heavy process
> at a time.

> **Bar 4b (byte-compare masks on 40 inert sides) — QUEUED**, tooling built
> (`tools/bg_estimator_m1_inert_check.py`). Compares sha256 of the produced
> RGBA buffers M1-on vs M1-off. Any non-identical mask is an automatic FAIL of
> the inertness claim.

> **Bar 5 (both lanes + re-embed class table) — PARTIAL.** Both lanes are
> instrumented and measured separately at their own geometry (never derived).
> The class table is §6; its per-class bound needs the census, which runs after
> the A/B per owner sequencing.

> **Bar 6 (serving regression) — QUEUED**, runs after merge with the gate OFF.

---

## 6. Bar 5 — the re-embed decision, as a CLASS table

Per owner scope ruling 2026-07-28: rows are **consumer artifact classes**, the
owner signs per class, and **RE-EMBED and RE-VISION are kept separate** — they
are different decisions:

- **RE-EMBED** — the crop is the same, but the *mask* changes under M1. Rides
  the next recluster cheaply.
- **RE-VISION** — the *detection geometry itself* was wrong. This is the
  modern-CNG over-detection cohort, and **M1 does not address it** (§2.1).

| # | class | population | affected bound | consumer impact | recommended action | rough cost |
|---|---|---|---|---|---|---|
| 1 | Served modern-corpus embeddings (per-material features / cluster vectors) | 369,481 coins / 4,787 cards | *pending census* | live search ranking | RE-EMBED at next recluster, scoped to affected houses | *pending* |
| 2 | Catalog annex vectors (`catalog_ingestion`, old-catalog plates) | Cahn 993 + Hirsch 710 + Helbing 1,177 = 2,880 lots | *pending census* | annex match quality | RE-EMBED at next annex refresh; re-run `append_search_annex.py --execute` after any recluster (standing rule) | *pending* |
| 3 | Archived screen sheets (KS-17, eLive-93, EA-613) | KS-17 287 photos / 574 sides; EA-613 422 photos | KS-17 measured: **12 sides** (per-side) / **5 photos** (ingest) | archived review artifacts, not live | RE-SCORE only if re-opened — **evidence pending audit join, required before any re-score executes** | low |
| 4 | Known-pairs scorecard baselines | `analysis/corpus_match/known_pairs/` | *pending census* | held-out validation baseline | RE-SCORE (baseline must be recomputed, never tuned) | low |
| 5 | Query lane | no stored artifacts | n/a | live | **auto-heals at enable, zero re-embed** | none |

**Row 3 evidence gap, flagged not filled** (owner ruling): the historical no-op
audit — *which archived sheets contain raw-pixel embeds* — **does not exist**.
This table is complete for sign-off without it; **execution of row 3 is not.**

**Not done here, by instruction:** nothing executed, no per-coin lists. Those
are generated at execution time from the same queries.

---

## 7. Recommended enable sequence — a PROPOSAL, every step separately owner-gated

1. **Merge M1 default-OFF** (this branch) and run the standing serving
   regression with the gate unset — Bar 6. *No behavior change.*
2. **Owner signs the Bar 5 class table**, choosing per class between re-embed
   at next recluster and no action.
3. **Flip `TRIVALAYA_BG_CORNER_LOCAL_TRUST=1`** in the serving service only,
   and re-run the regression with the gate ON. Blast radius at 518 px serving
   geometry is bounded by the crossing set; expected delta is the healed no-op
   class.
4. **Re-embed the affected corpus rows** per the signed table, scoped to the
   houses the census puts in scope.
5. **KS-17 re-ingest — DO NOT sequence this behind M1.** The re-ingest was held
   on the understanding that the repaired estimator would improve CNG detection
   quality. §2.1 measures that it will not: on the dark stratum M1 changes
   nothing. Either re-ingest now on its existing merits, or hold it for the new
   dark-side root-cause ticket — but the M1 dependency is void and should be
   struck from the KS-17 runbook.
6. **Open the dark-side root-cause ticket** (downstream of polarity: Otsu on a
   non-bimodal histogram). This is where legs #2's value actually lives.
7. **Open the standing-telemetry ticket** — `mask_area_fraction` / `mask_noop`
   as first-class fields, so every future "mask fallback 0/N" bar means
   something. Every historical one checked `mask_fallback_reason` only and is
   blind to this entire class.

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
