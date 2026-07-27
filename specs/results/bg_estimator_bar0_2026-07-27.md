# Bar 0 — does the background estimator cause the L229 serving mask no-op?

**Run 2026-07-27, clean and serialized.** Ticket:
`specs/background_estimator_repair.md`, "Bar 0 — the link (gating, run first)".
Read-only against production: no DB writes, no shipped module changed, no fix
applied.

**Verdict in one line: Bar 0's stated predicate is REFUTED — the dark strata is
no-op-free, 0/60, in both lanes — but the conclusion that predicate was written
to license ("the L229 link dies; the serving no-op is a separate defect") is
contradicted by the same run. The no-op is caused by this function, on the other
fallback branch.**

> **OWNER RULING 2026-07-27 — recorded outcome: predicate refuted (dark-branch
> mechanism never fires); ticket-level shared-cause claim CONFIRMED via the
> light branch; scope unchanged.** Explicitly **not** the precommitted
> "refuted → CPU-only → park" outcome: that inference was built on the proxy,
> and the proxy is what failed, not the claim. Honest bookkeeping follows the
> evidence, not the wording. Funding rests on two live legs — (a) a live
> doctrine violation in production serving, (b) CNG ingest detection quality
> with KS-17 re-ingest blocked on this fix — with the latency leg **retired**
> (§6c). See `specs/background_estimator_repair.md`, "Bar 0 — OWNER RULING".

---

## 1. What was run, and under what conditions

| | |
|---|---|
| population | KS-17, `analysis/incoming_screen/KS-17/incoming_images`, 287 images / **574 sides** (`--layout half`, left=obv right=rev), house `cng_feature` |
| sample | the **exact 117 sides** of the 2026-07-26 selection: 60 dark (`avg_bg < 45`) + 57 light (`avg_bg > 85`) |
| lanes | `query518` (the real serving geometry) and `fullres` (ingest geometry) — measured separately, never derived from one another, per Bar 5 |
| tasks | 234 (117 sides × 2 lanes), **234/234 `ok`, 0 timeouts, 0 stalls, worker respawns = 1** |
| wall clock | 1,599 s |
| contention | box otherwise idle; this was the only heavy process. The 2026-07-27 contended run's **latency** numbers are discarded and not reused anywhere below. |

**On the seed.** The prior run recorded `RNG_SEED = 42`, but its dark pool was
ordered by `ProcessPoolExecutor.as_completed()` completion order, so seed 42 is
**not reproducible**. The authoritative record of which 117 sides were selected
is the preserved `l229_strata_selection.csv`; this run consumed that file
directly (`--side-list`) rather than re-sampling. Re-deriving the sample from
the seed would have silently drawn a different 60.

**Instrument validation (positive control).** A no-op-free dark result is only
meaningful if the instrument can see a no-op at all. Re-measuring the 57 light
sides reproduces the prior run's finding exactly:

- shared sides (n=94, `fullres`): median |area delta| **0.0012**, max **0.0037**,
  **0 sides** differing by >0.01
- no-op count: prior **12**, clean **12** — **MATCH**

The instrument is sensitive. The dark zero is a real zero.

---

## 2. The headline table

`mask_area_fraction` = contour area / frame area of the image handed to
`_mask_query_image_meta`. A genuine full-frame *circular* coin caps at
π/4 ≈ 0.785; ≈1.0 means the "largest contour" is the image rectangle — the
embed is effectively unmasked while telemetry still reports `masked: true`.

| lane | stratum | n | corner-trust fired | estimator err vs outer-ring truth | **silent no-ops** | max area |
|---|---|---|---|---|---|---|
| `query518` | dark | 60 | 0/60 | median **−47.8** | **0** (0.0%) | 0.8133 |
| `query518` | light | 57 | 0/57 | median +17.1 | **7** (12.3%) | 0.9961 |
| `fullres` | dark | 60 | 0/60 | median **−48.1** | **0** (0.0%) | 0.7843 |
| `fullres` | light | 57 | 0/57 | median +17.1 | **12** (21.1%) | 0.9986 |

Bar 0's requested contingency table, `fullres`:

| path | no-op | healthy |
|---|---|---|
| corner_trust | 0 | 0 |
| fallback | 12 | 105 |

The path variable is degenerate exactly as the 2026-07-26 amendment predicted —
corner-trust fires **0 / 117**, so every side is on the fallback arm and the
table has no variance. The amendment anticipated this; it is recorded, not
worked around.

Full `mask_area_fraction` distributions (the bar asks for the distribution, not
a thresholded count) are in the run JSONL and reproduced by
`tools/bg_estimator_bar0_report.py`. The shape is strongly bimodal: a healthy
mode at 0.20–0.50 (n=105) and a no-op spike at 0.95–1.00 (n=12), with **nothing
in between** — no gradual degradation, no borderline cases to adjudicate.

---

## 3. Why the predicate fails: the no-op is on the *other* branch

Split by the estimator's own branch (`fullres`, all 117 sides):

| branch | what it returns | median est | truth | n | **no-ops** |
|---|---|---|---|---|---|
| `dark_fallback` | `mean(pixels < 50)` | 30.9 | ~79 | 60 | **0** (0%) |
| `mixed_fallback` | `argmax(hist)` | 96.0 | ~79 | 43 | **0** (0%) |
| `light_fallback` | `mean(pixels > 200)` | 215.4 | ~79 | 14 | **12** (86%) |

The ticket's hypothesised chain was: *dark_fallback returns 31 where truth is 79
⇒ background classified as foreground ⇒ largest contour is the whole frame ⇒
no-op.* Measured across 60 dark sides in both lanes, that chain **never fires
once**, despite the estimator being wrong by ~48 levels on every one of them.

The no-op instead concentrates on `light_fallback` — 12 of 14 sides (86%) — and
there the error is **larger and opposite in sign**: the estimator returns ~215
where the outer ring is still the same dark ~79 CNG backdrop, an error of
**+136**. These are lots photographed with a bright coin against the standard
dark template; the histogram's light peak wins, `bg_type` is called `light`, and
polarity inverts.

So estimator error *does* track the no-op — but monotonically in the wrong
variable. Sorting by |error|:

| | n | median &#124;est_err&#124; | median est |
|---|---|---|---|
| no-op | 12 | **135.9** | 214.9 |
| healthy | 105 | 46.4 | 33.0 |

The bar named the dark strata as "the hypothesis's predicted domain." That was
the mis-specification: the predicted domain was chosen from the *ingest* symptom
(`backdrop_vignette_blob`, which is genuinely a dark-backdrop pathology), and the
serving no-op does not live there.

---

## 4. The link survives the predicate — via the shared root cause

The ticket's root-cause reading is that the corner test conflates *"corners are
noisy"* with *"corners disagree with each other."* That failure is upstream of
**all three** fallback branches, not just the dark one.

Counterfactual for **M1**, the ticket's preferred mechanism (judge each corner by
its own std; take the median of the four corner medians), computed on the 12
no-op sides:

| | shipped fallback | M1 | outer-ring truth |
|---|---|---|---|
| returned value | ~215 | **75.0** | ~79.0 |
| error | **+136** | **−4.0** | — |
| within Bar 1's ±8 | 0/12 | **12/12** | — |
| polarity | `light` | **`dark`** | — |

M1 lands inside Bar 1's ±8 tolerance on **12 of 12** no-op sides and **flips
polarity on every one**. The no-op class is therefore downstream of the same
defect the ticket exists to repair — reached by `light_fallback` rather than
`dark_fallback`.

This is a counterfactual on the estimator's return value and polarity, not a
measured re-run of the mask with M1 applied. It is strong evidence, not proof
that the masks come out healthy; that is Bar 3's job after the mechanism is
built.

---

## 5. Verdict, stated against the precommit framing

The framing fixed before measurement: *no-ops present in dark strata + tracking
estimator error ⇒ CONFIRMED; no-op-free ⇒ REFUTED (L229 link dies).*

- **Predicate: REFUTED.** Dark strata is no-op-free — 0/60 in both lanes, with a
  validated instrument and zero timeouts. Stated plainly, without hedging.
- **The conclusion the REFUTED branch prescribes is contradicted by the same
  data.** "The serving no-op is a separate defect" is not what the run shows:
  86% of the `light_fallback` branch no-ops, and the ticket's own M1 corrects
  every one of those sides to within ±8 with a polarity flip.

These two do not point the same way. Bar 0 tested a proxy (*which strata*) for
the claim it cared about (*same cause*), and the proxy and the claim came apart.
Per the ticket's no-revision rule the bar itself is not edited after the fact.

**Resolved by owner ruling, 2026-07-27:** predicate refuted (the dark-branch
mechanism never fires); **the ticket-level shared-cause claim is CONFIRMED** via
the light branch; **scope unchanged** — the ticket keeps both symptoms and L229
stays folded in. This is deliberately **not** the precommitted "refuted →
CPU-only → park" outcome, because that inference was built on the proxy rather
than on the claim. Where a precommitted bar's *wording* and its *evidence*
diverge, the bookkeeping follows the evidence and says so out loud.

What is **not** in dispute either way: the estimator is wrong on 117/117 sides
measured (median |err| 46.4 healthy / 135.9 no-op), corner-trust fires 0/117, and
21% of light-backdrop sides ship an unmasked embed under `masked: true`.

---

## 6. Three findings that change other records

**(a) The dark strata was not unmeasured.** The interim note recorded "dark
strata = 0/60 completed." The preserved per-side CSV shows the 2026-07-26 run
completed **38 of 60** dark sides (37 ok + 1 timeout), and those 38 were already
no-op-free. The re-run was still worth doing — it was contended, partial, and
single-lane — but the record should say 38, not 0. **This run's 60/60 supersedes
it.**

**(b) The serving lane is affected, and the 518 downsize halves but does not
eliminate the class.** `query518` — what production actually does, since
`_normalize_upload_bytes` thumbnails uploads to `UPLOAD_MAX_DIM=518` before
masking — shows **7** no-ops where `fullres` shows 12. The prior run measured
full-res sides only and therefore never characterised the real serving geometry.
The doctrine violation is live in production, at roughly half the full-res rate.

**(c) The "search-by-image latency" motivation is now measured, and it
collapses.** The ticket flagged 40–166 s as transferred from the ingest profile
and explicitly "motivation, not a claim." Measured clean on an idle box, the L1
mask step costs:

| lane | median | p90 | max |
|---|---|---|---|
| `query518` (serving) | **0.022 s** | 0.35 s | 1.26 s |
| `fullres` dark (ingest) | **21.45 s** | 37.74 s | 53.79 s |

The serving lane is ~1000× cheaper because it masks a 518 px image. The ingest
cost is real and large; **the user-facing latency argument should be retired**,
not carried forward. (This times the L1 mask step only — not the full endpoint,
which adds the DINOv2 forward and matching.)

---

## 7. Standing-telemetry corollary, reaffirmed

Every historical "mask fallback 0/N" bar (KS-17 screen 0/287, annex 0/5170,
diagnosis 0/62) checked `mask_fallback_reason` and never `mask_area_fraction`.
All 12 no-op sides here report `masked: true`, `mask_fallback_reason: None`,
`n_detections: 1` — **indistinguishable from a healthy mask on the telemetry
those bars looked at.** Those "0/N" claims say nothing about no-ops. Standing
telemetry needs `mask_area_fraction` added before any of them can be read as
clean. Unchanged from the 2026-07-26 corollary; this run supplies 12 more
instances of it.

---

## 8. Funding case after this run — two live legs, one retired

**Owner ruling 2026-07-27.** The case is *sharper* after Bar 0, not weaker.

**(a) A live doctrine violation in production serving.** At real 518 px serving
geometry **7 sides no-op** — raw-pixel embeds are being served under
`masked: true` **today**, indistinguishable from healthy on the telemetry any
standing bar inspects (§7). CLAUDE.md's image-comparison doctrine exists to
prevent exactly this. Full-res geometry shows 12; the downsize halves the rate
and does not remove it.

**(b) CNG ingest detection quality**, below — with the added weight that
**KS-17's re-ingest is explicitly blocked on this fix**: the 2026-07-27
three-table unwind was taken on the understanding that re-ingest waits for the
repaired estimator.

**(c) Search-by-image latency — RETIRED.** Measured at 0.022 s median on the
serving lane (§6c). The 40–166 s figure was transferred from ingest and never
described serving. The ticket carries a stamped retirement note so the dead
argument is not restated; it must not reappear in future justifications.

Independent of the link,
the estimator bug is a measured **detection-quality** defect on live CNG data:
modern `cng` over-detects at 2.76–3.16 dets/photo against ~2.0 on leu and legacy
`cng_feature`, and on KS-17 — the ticket's own worst-case slice — **34% of photos
(127) saturate the `MAX_DETECTIONS=5` per-image cap with spurious blobs**, with
13.1% RED and 49% GREEN against a 76–78% cohort norm. That evidence is untouched
by this run and stands on its own — and it is now load-bearing in a way it was
not on 2026-07-26, because KS-17's re-ingest is queued behind this repair.

---

## 9. Reproduction

```bash
V=~/trivalaya-pipeline/.venv/bin/python        # needs appv2 + vision on one path
$V ~/trivalaya-vision/tools/bg_estimator_bar0_probe.py \
    --side-list specs/results/bar0_prior_run_2026-07-27_contended/l229_strata_selection.csv \
    --strata dark,light --modes query518,fullres \
    --out specs/results/bg_estimator_bar0_clean_2026-07-27.jsonl
$V ~/trivalaya-vision/tools/bg_estimator_bar0_report.py \
    specs/results/bg_estimator_bar0_clean_2026-07-27.jsonl \
    --compare specs/results/bar0_prior_run_2026-07-27_contended/l229_mask_measure_stratified.csv
```

**Artifacts**

- `specs/results/bg_estimator_bar0_clean_2026-07-27.jsonl` — 234 per-side rows,
  this run (the measurement of record)
- `tools/bg_estimator_bar0_probe.py` — harness; persistent worker with
  parent-enforced 180 s SIGKILL, because the stalls being measured sit inside
  OpenCV C code where a Python-level alarm would not land
- `tools/bg_estimator_bar0_report.py` — contingency table + distributions
- `specs/results/bar0_prior_run_2026-07-27_contended/` — the 2026-07-26/27
  contended run, **rescued from an expired session scratchpad under `/tmp`** and
  committed here: its script, the 574-side phase-1 diagnostic, the authoritative
  117-side selection, and its 95-row partial. Its geometry is valid and
  reproduces; **its latency is contaminated and must not be reused.**
