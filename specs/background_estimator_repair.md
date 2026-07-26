# Background estimator repair — `detect_background_histogram`

> **Status: OPEN, opened 2026-07-26** (owner-approved). Promoted out of
> `specs/rim_recovery_neighbor_aware.md`, which closed the same day as a cost
> lane. **This is not cost work.** It is the repair of a component with a
> measured 0% success rate on the largest house's corpus, and it is therefore
> NOT governed by the 2026-07-23 tail-confinement ruling ("cost work may only
> change outcomes in the already-pathological tail"). It will change outcomes
> broadly and on purpose; the bars below are built for that, not against it.
>
> PRECOMMIT bars are fixed **before** any row exists in "Results", per the
> ea614 convention (`~/trivalaya-pipeline/analysis/prehammer_estimate/
> ea614_verdict_precommit_2026-07-21.md`). Edit freely until the opening
> commit lands; after that, edit history only, no silent revision.

## The defect

`src/math_utils.py:172`, `detect_background_histogram(gray_image)`. Two paths:

```
1. corner-trust : sample four 5x5 corner patches; if pooled corner_std < 15,
                  return (corner_median, light|dark).      <- the good path
2. fallback     : histogram; if dark_peak > 2*light_peak,
                  return (mean(pixels < 50), "dark").      <- not a background
                                                              estimate
```

**Measured** (`specs/results/rim_stall_taxonomy_2026-07-23.md` §(a)–(d), full
KS-17 population, 574 sides):

- CNG's composited backdrop carries a **46-level vignette ramp** across the
  frame — corners read e.g. TL 96.0 / TR 99.4 / BL 54.6 / BR 53.4, giving
  `corner_std = 21.9`.
- So the corner-trust test fails on **574 of 574 sides**. It fires zero times
  on this corpus.
- So the fallback runs on 100% of it and returns `mean(pixels < 50)` —
  population median **31.2**, where the honest outer-ring background is
  **79.0**.
- That wrong number chooses polarity (`avg_bg=31 < 110` ⇒ treat as dark
  background) and sets up an Otsu split of a histogram that is not bimodal
  coin-vs-backdrop. **84% of sides (482/574) then trip rim recovery.**

**Root-cause reading.** The corner test conflates two different things: *"the
corners are noisy"* (which should reject them) and *"the corners disagree with
each other"* (which a smooth vignette guarantees, even though each corner is
locally clean background). A vignette is the second case. The code only
measures the first, using a pooled std across all four patches.

## Why this outranked everything else in the rim lane

From the taxonomy's own class-correlation table (§4.3), **four of the seven
failure classes** — `backdrop_vignette_blob`, `relief_self_segmentation`,
`sub_coin_noise_blob`, `low_contrast_coastline` — are all downstream of the
dark composited backdrop regime, i.e. of this function. Together they are
**62.7%+ of measured Hough CPU**. Three separate cost mechanisms were built
and rejected attacking those classes individually; none touched the shared
cause.

The regime is a **per-house constant** (the CNG backdrop template is
byte-identical corpus-wide), so any house shipping a dark vignetted composited
backdrop inherits all four classes.

## The two symptoms, previously tracked as separate tickets

**Ingest —** `backdrop_vignette_blob` (8.8% of expensive triggers, 22.3% of
Hough CPU). The threshold lands inside the vignette's own ramp, so the
brighter middle of the *backdrop* becomes foreground with the coin as a hole
punched through it. `755455 rev`: `area_frac 0.674`, ROI = the entire frame,
33.7 s. `755481 rev`: `area_frac 0.456`, 47.8 s. In both, the "recovered rim"
is **a circle drawn around the background** — and recovery was *accepted* 3/4
in this class.

**Serving —** the **L229 mask no-op**, recorded at
`~/trivalaya-pipeline/specs/old_catalog_corpus_match_process.md:592` as a
separate ticket: "`_mask_query_image_meta` silently no-ops on dark-background
photos (full-frame contour, telemetry says `masked: true`). One case in 22
pairs (L229 modern side; symmetric remask 0.797→0.830)."

**Hypothesis: these are the same bug.** `visual_search/appv2.py:801` calls the
same `analyze_image` → the same `detect_background_histogram`. Background
classified as foreground ⇒ the largest contour is the whole frame ⇒ the mask
covers everything ⇒ `masked: true` on an effectively unmasked embed.

**Third benefit, unmeasured: search-by-image latency.** The query lane runs
Layer 1 per request, so the same trigger explosion that costs batch CPU costs
*user-facing response time* on search-by-image — 140/574 query-side firings
measured during the mechanism #1 run. The production latency has never been
measured (the 40–166 s figure circulating in the notes is transferred from the
ingest profile), so treat it as motivation, not a claim. If it is ever
measured, this repair is the mechanism that addresses it without the
speed-for-accuracy trade the owner rejected on 2026-07-23: it collapses the
trigger rate at the source rather than skipping correct recoveries.

This matters more than the CPU. CLAUDE.md's image-comparison doctrine is
explicit that `masked:true` can silently no-op and that "a mask no-op is a bug
to surface, never to ignore" — an unmasked embed is a **lesser metric shipped
under a truthful-looking telemetry flag**, which is exactly the failure the
doctrine exists to prevent.

**Step 0 of this ticket is to confirm or refute that link** before any repair
is designed. Measurement launched 2026-07-26; see "Results".

## Candidate mechanisms

**M1 — widen the corner-trust test to per-corner local consistency.**
Preferred. Judge each corner patch by *its own* std and take the median of the
four corner medians, instead of rejecting whenever the four disagree. On the
CNG numbers above this yields ≈75 against an honest 79.0, versus the 31.2
shipped today.

*Blast radius bounded by construction:* this only ever ADDS cases where
corner-trust fires. Every image where `corner_std < 15` today still takes the
identical path and returns the identical value, so all non-vignetted houses
are **bit-identical by construction, not by measurement** — the same property
the weld lane and the neighbor guard were held to.

**M2 — replace the fallback's `mean(pixels < 50)` with an outer-ring median.**
The catch-all for scenes that still fail M1. Flagged in the taxonomy (§ near
line 415) as "a small, local change — but it flips polarity decisions, so it
is *not* tail-confined." Strictly worse-bounded than M1; take it only for the
residue M1 does not reach, and measure it separately.

**Do not** replace Otsu or restructure Layer 1 under this ticket. If M1+M2
leave a material residue, that is a separate design with its own bars.

## PRECOMMIT ACCEPTANCE BARS

Fixed 2026-07-26, before any measurement in "Results" is read against them.

**Note on bar shape.** The usual "≤2% of sides change" bar from the cost lane
is **deliberately not used here** and must not be imported. This repair is
*supposed* to change outcomes — a mechanism that changed nothing would have
failed to fix anything. The bars below therefore gate on *direction* and
*adjudication* of change, not on its volume, with one exception (Bar 5) that
gates volume only where volume is a cost the owner must consciously accept.

### Bar 0 — the link (gating, run first)

The step-0 measurement either CONFIRMS or REFUTES that the L229 serving no-op
and `backdrop_vignette_blob` share this cause. Report the contingency table of
{corner-trust vs fallback path} × {mask area fraction > 0.9 with
`masked:true`}, plus the full distribution of mask area fraction — not a
single thresholded count.

- **CONFIRMED** ⇒ proceed; the ticket carries both symptoms and the L229
  ticket is closed into this one.
- **REFUTED** ⇒ the serving no-op is a separate defect. Proceed on the ingest
  symptom alone and re-open L229 on its own. Do not quietly widen scope.

### Bar 1 — the estimate itself is correct

On the KS-17 population (574 sides), the repaired estimator's returned
background value must land within **±8 grey levels** of the honest outer-ring
median for that side (the taxonomy's 79.0 reference statistic), on **≥95%** of
sides. Today's shipped value misses by ~48 levels on essentially all of them.
Report the error distribution, not just the pass rate.

### Bar 2 — bit-identity where the good path already fires

On any image where `corner_std < 15` under today's code, the repaired function
must return a **byte-identical** value and `bg_type`. Asserted by unit test,
not by sampling. A violation is an automatic FAIL regardless of every other
number — this is the property that bounds the blast radius.

### Bar 3 — the no-op class is eliminated, and none is created

Measured through `appv2._mask_query_image_meta` on the KS-17 set:

- Sides exhibiting a silent no-op (`masked:true` with mask area fraction
  > 0.9) drop to **zero**.
- **Zero** sides that were correctly masked before become a no-op after.
- Every `mask_fallback_reason` transition is enumerated in the results doc.

### Bar 4 — changed sides are improvements, adjudicated

Volume of change is expected and is not itself a failure. Instead:

- Draw a **stratified random sample of 40 changed sides** (stratified by
  taxonomy class), render before/after overlays, and adjudicate each as
  IMPROVED / NEUTRAL / REGRESSED.
- PASS requires **zero REGRESSED** and **≥30 of 40 IMPROVED**.
- Any REGRESSED side is individually reported with its overlay. One confirmed
  regression blocks the default flip and sends the mechanism back to design —
  it does not get traded against the improvements.
- The sample and its seed are recorded before adjudication begins.

### Bar 5 — both lanes, and the re-embed decision is explicit

Per the standing both-lanes doctrine, run ingest (`analyze_image`,
`house=cng_feature`) AND query (`appv2._mask_query_image_meta`, `house=None`,
its own JPEG round-trip — measured non-inert, do not derive one from the
other). A mechanism that clears one lane and not the other is a FAIL.

Then report, as a number the owner signs off on before any default flip: **how
many corpus coins would need re-embedding** for corpus/query geometric
consistency, and the estimated wall-clock. Crops changing on one side of the
corpus/query boundary and not the other is the failure mode this bar exists to
prevent. Shipping the fix without the re-embed is a decision, not an
oversight — but it must be a recorded one.

### Bar 6 — downstream serving bar

After merge and with the gate OFF, run the full CLAUDE.md visual_search
regression set (`topk_probe.py` per-slice fixtures, `routing_bar.py`,
`stage2_bar.py`). With the gate unset, behavior must be byte-identical to
pre-restart. With the gate ON, any delta must be individually explainable by
this change; unexplained drift blocks.

### Cross-cutting

- Ships **env-gated, default = today's behavior**, unset ⇒ bit-identical,
  proven by test and by a stored-run diff. No default flip lands in this
  work; production enable is a separate, explicit, owner-gated step.
- `math_utils.py` is imported by `decode_crop.py`'s `analyze_image` call from
  `appv2.py`, so this is in scope for the serving regression bar (Bar 6).
- Measurement respects the image-comparison doctrine: mask telemetry and
  geometry only under Bars 0–4. Any similarity number computed anywhere in
  this ticket goes through masked transparent grey128 on both sides.

## Results

*(empty — nothing measured yet. Step-0 / Bar-0 measurement launched
2026-07-26.)*

## Related

- `specs/rim_recovery_neighbor_aware.md` — parent lane, closed 2026-07-26; its
  "Ruling" section carries the disposition of all six former open items.
- `specs/results/rim_stall_taxonomy_2026-07-23.md` — §(a)–(d) measured this
  defect; §4.2 the `backdrop_vignette_blob` cases; §4.3 the class-correlation
  table that makes this the shared cause.
- `~/trivalaya-pipeline/specs/old_catalog_corpus_match_process.md:592` — the
  L229 serving no-op, previously a separate ticket, folded in pending Bar 0.
- CLAUDE.md, "Image-comparison doctrine" — why the serving symptom outranks
  the CPU.
