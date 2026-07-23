# Why curated auction photos defeat Layer-1 coin segmentation

Empirical taxonomy + architecture assessment. 2026-07-23.

Companion to `specs/rim_recovery_neighbor_aware.md` (whose §Ruling now records
the owner's 2026-07-23 rejection of the Scope A caps), and successor to
`specs/results/ks17_mask_stall_diagnosis_2026-07-22.md`,
`rim_recovery_cost_ab_ks17_2026-07-23.md`,
`rim_recovery_profile_kuenker_2026-07-23.md`.

**Owner ruling this work is written against (2026-07-23):** *Scope A caps
REJECTED — no speed-for-accuracy trade. Cost work continues only via
mechanisms whose outcome changes are confined to the currently-pathological
tail.*

Tools (all new, all read-only, all in `tools/`):

| tool | what it does |
|---|---|
| `rim_stall_taxonomy.py` | replicates the primary segmentation with per-contour instrumentation; `scan` / `classify` / `overlay` / `montage` |
| `verify_taxonomy_replication.py` | proves the replication is faithful against real `layer_1_structural_salience` |
| `rim_scale_sensitivity_probe.py` | is `circularity<0.65` measuring shape, or boundary roughness at this resolution? |
| `rim_hough_yield_probe.py` | what does the expensive Hough call actually BUY? |
| `rim_trigger_shape_probe.py` | measured A/B of the #1 ranked mechanism |

No production file was modified. No DB write, no service restart, no env
change. Work done in the `hough-taxonomy` worktree; `~/trivalaya-vision` main
untouched.

---

## 0. Executive summary — what the measurements overturn

Four things in the standing framing turn out to be false on the actual data.

1. **"Professionally-photographed, near-uniform-background images."**
   The KS-17 backdrop is a *composited template* with a 46-grey-level
   corner-to-corner luminance ramp. It is byte-identical across the corpus
   (per-pixel std **0.00** across sampled images) — constant, but nowhere near
   uniform.

2. **"Background mean is 78.9–79.0 on every single image tested."**
   True of the outer-ring mean, but that is not the statistic the code uses.
   `detect_background_histogram` returns a median of **31.2** here, and it
   reaches that number through its *fallback* path — which returns the mean of
   every pixel below grey 50, i.e. a statistic of the scene's dark tail, not a
   background level. The corner path that would give a real background estimate
   fires on **0 of 574 sides**, because the template's own ramp always breaks
   its `corner_std < 15` trust test.

3. **"Detection nearly never fails outright, ~55% of sides trip the recovery
   branch."** The trip rate is **84%** of sides (482/574), not 55%. The 55%
   figure was tracking cost, and cost is a different variable: **59.6%** of
   sides carry at least one *expensive* trigger. Cost per Hough call spans
   300× and is bimodal with an empty gap.

4. **The folklore mechanism list** — attached shadow, rim-toning contrast loss,
   specular holes, gradient background, holder/label artifact, multi-object —
   presupposes that the ambiguous blob is *approximately the coin*. On this
   corpus that presupposition is usually false. The blob is frequently the
   **backdrop**, or a piece of the coin's **engraved relief**, or a **correct
   contour that the metric misreads**. "Attached shadow" and "specular holes",
   the two most-cited classes, account for **2.9%** of expensive triggers
   between them and near-zero cost.

**The single most important finding.** The contours that pay ~40% of the
entire Hough bill are *round coins that were segmented correctly*. Their
median convex-hull circularity is **0.965**; their median raw circularity is
**0.096**. The gap is pure perimeter: `4πA/P²` is quadratic in a statistic
that grows with resolution, and a low-contrast rim dithers the boundary.
Downscaling the same photo 4× raises median circularity from 0.096 to 0.573
and stops half the triggers — same coin, same contour, different number.
**The segmentation is right; the trigger metric is what fails.**

**Top-ranked mechanism** (§5), measured in §6: **skip only the Hough branch —
not rim recovery — on blobs that are provably discs.** On the expensive tier
`recover_rim` already returns the *geometric* fit 17 times out of 21; Hough
burns 40–200 CPU-seconds and is then discarded in favour of an answer computed
~20 ms earlier. On the worst class this is **6/6 outcome-identical (alpha IoU
1.0) at 99.3% less CPU**. The obvious cruder version — suppress rim recovery
altogether — **fails**, 0/8, and is reported in §6.1 because it is the version
one would reach for first.

---

## 1. Method

`layer_1_structural_salience` returns only finished, NMS-suppressed
candidates, and the contour that pays the Hough cost is very often suppressed
before it reaches the caller — which is exactly why the prior diagnosis found
`n_det` to be a non-predictor. Answering "why" therefore requires per-contour
interception.

`tools/rim_stall_taxonomy.py` re-executes the same sequence the production
function runs — same helpers (`detect_background_histogram`,
`is_contour_valid`, `compute_circularity_safe`, `_close_kernel_size`,
`geometric_fit_recovery`), same `Layer1Config` constants, same order — and
stops before pass 2's `recover_rim`, recording per contour the state that
decides the branch.

**The trigger is a conjunction of three conditions, not the two usually
cited.** Beyond `circularity < 0.65` and `area_ratio < 0.85`
(`layer1_geometry.py:277-282`), `rim_logic.recover_rim` runs
`geometric_fit_recovery` first and **short-circuits before Hough when its
combined confidence exceeds 0.65**. That third gate is cheap (~20 ms), so the
scanner evaluates all three and reports `will_hough` exactly.

**Replication fidelity** (`tools/verify_taxonomy_replication.py`), joined
against the frozen baseline timings in `rim_recovery_cost_ab_ks17.csv`:

```
predict Hough & IS  stall : 47      predict none & IS  stall : 0   <- no misses
predict Hough & NOT stall : 20      predict none & NOT stall : 15
```

**Zero false negatives**: the replication never misses a real stall. The 20
"false positives" are sides where Hough genuinely runs but on tiny ROIs at
<2 s — which is the §3 finding, not a replication defect.

**Fixtures.** KS-17 (`analysis/incoming_screen/KS-17/incoming_images`, 287
images / **574 sides**, house `cng_feature`) — full population, not a sample.
Plus the 6 known-slow leu lots (713, 582, 995, 3661, 3736, 3717) and the 3
profiled kuenker lots (1289, 1070, 1030). **The leu and kuenker sets are
pre-selected slow lots, not random samples** — they are read below for
*presence* of a mechanism, never for frequency.

---

## 2. The causal chain

Measured end to end on KS-17:

**(a) The backdrop is a composited template with a luminance ramp.**
Per-pixel std across images is `0.00` in every background region — corner
patches, edge strips, top strip — while the coin region varies with std 47.9.
The backdrop is not photographed per lot; it is the same rendered gradient
every time. Its radial profile runs ~128 grey at frame centre to ~77 at the
corners.

**(b) So the corner-consistency test never passes.** Corner 5×5 patches
measure e.g. TL 96.0 / TR 99.4 / BL 54.6 / BR 53.4 — a 46-level ramp, giving
`corner_std = 21.9`. `detect_background_histogram` trusts corners only when
`corner_std < 15`. Across the full population that test fails on **574 of 574
sides**.

**(c) So the histogram fallback runs, and it does not estimate the
background.** With `dark_peak > 2 × light_peak` it returns
`mean(pixels < 50)`. On a dark bronze coin on this backdrop that evaluates to
~32 — a statistic of the darkest fifth of the *scene*, mixing the vignette's
corner falloff with the coin's own shadowed relief. Population median: **31.2**
where the honest outer-ring background level is **79.0**.

**(d) So polarity is chosen on a wrong number, and Otsu is asked to split a
histogram that is not bimodal coin-vs-backdrop.** `avg_bg=31 < 110`
⇒ `THRESH_BINARY` (bright = foreground) on 560 of 574 sides. Otsu then lands
wherever between-class variance happens to peak — which on these scenes is
*inside the backdrop's ramp*, or *inside the coin's own tonal range*, not on
the coin's edge.

**(e) So the blobs are not coins**, and being non-circular they trip the
recovery trigger. Median valid contours per side is **1** on sides that never
trigger, and **10** on sides carrying an expensive trigger (max 49).

---

## 3. Cost is bimodal, with an empty gap, and it is entirely ROI area

Joining the scan against the frozen baseline timings:

| max Hough ROI on the side | n sides | median CPU-s | max CPU-s |
|---|---:|---:|---:|
| < 100,000 px | 33 | **0.31** | 1.5 |
| 100,000 – 600,000 px | **0** | — | — |
| 600,000 – 1,000,000 px | 2 | 21.08 | 23.4 |
| > 1,000,000 px | 47 | **98.14** | 208.3 |

Spearman(ROI px, CPU-s) = **0.839**, and *nothing at all* lands between 100k
and 600k px. Any threshold inside that gap splits stall from non-stall at
97.6% agreement with **zero false negatives**. `EXPENSIVE_ROI_PX = 600_000` in
the tool is set from this gap, not chosen a priori.

The concentration is extreme. Over 282 triggering contours carrying 1,681
CPU-seconds:

| slice | share of all Hough CPU |
|---|---:|
| top 5 contours (1.8%) | 37.8% |
| top 10 contours (3.5%) | 68.1% |
| top 28 contours (9.9%) | **99.9%** |
| cheapest 141 (50%) | 0.0% |

---

## 4. The taxonomy

Classes are named for **what the blob actually is**, assigned by ordered rules
in `rim_stall_taxonomy.py::classify_contour`, each threshold read off the
measured distributions. Two statistics do most of the separating:

- **`cv_r`** — coefficient of variation of the blob radius about its centroid.
  Answers "is this a disc?" without touching perimeter, and is
  resolution-invariant. ~0.02 for a round coin with a dithered boundary;
  ≥0.08 for a klippe or an irregular flan.
- **`largest_hole_frac`** — fraction of the blob's filled area that is *not*
  foreground. `RETR_EXTERNAL` counts holes as filled, so a blob that is really
  the backdrop (coin punched out of it) scores high; a piece of relief scores ~0.

### 4.1 Frequency — KS-17, full population (574 sides, 4,701 triggering contours)

**By count** (what drives how *often* Hough is invoked):

| mechanism | n | % |
|---|---:|---:|
| relief_self_segmentation | 3,053 | 64.9% |
| sub_coin_noise_blob | 1,264 | 26.9% |
| low_contrast_coastline | 91 | 1.9% |
| non_circular_flan | 87 | 1.9% |
| attached_artifact | 75 | 1.6% |
| multi_object_weld | 50 | 1.1% |
| unclassified_ragged | 48 | 1.0% |
| backdrop_vignette_blob | 33 | 0.7% |

**By expensive-tier count** (ROI ≥ 600k px — the sides that stall):

| mechanism | n | % of expensive |
|---|---:|---:|
| low_contrast_coastline | 91 | 24.3% |
| non_circular_flan | 87 | 23.3% |
| relief_self_segmentation | 57 | 15.2% |
| multi_object_weld | 48 | 12.8% |
| unclassified_ragged | 47 | 12.6% |
| backdrop_vignette_blob | 33 | 8.8% |
| attached_artifact | 11 | 2.9% |

**By measured Hough CPU-seconds** — the table that matters, from
`rim_hough_yield_probe.py` (282 contours, 1,681 CPU-s):

| mechanism | n | Hough CPU-s | % CPU | recovery accepted | **Hough's answer actually used** |
|---|---:|---:|---:|---:|---:|
| low_contrast_coastline | 7 | 679.4 | **40.4%** | 7/7 | **1/7** |
| backdrop_vignette_blob | 4 | 374.5 | 22.3% | 3/4 | 2/4 |
| unclassified_ragged | 3 | 253.5 | 15.1% | 3/3 | **0/3** |
| non_circular_flan | 3 | 252.4 | 15.0% | 3/3 | **0/3** |
| multi_object_weld | 3 | 84.4 | 5.0% | 1/3 | 1/3 |
| relief_self_segmentation | 179 | 31.9 | 1.9% | 147/179 | 2/179 |
| attached_artifact | 4 | 4.1 | 0.2% | 3/4 | 0/4 |
| sub_coin_noise_blob | 79 | 1.2 | 0.1% | 72/79 | 0/79 |

**Twenty contours across five classes hold 97.8% of the Hough CPU, and the
Hough result is used in only 4 of them.** In the other 16, `recover_rim`
returns the geometric fit — which had already been computed, for free, ~20 ms
before Hough started. Across the whole expensive tier the split is **17 geo /
4 hough**: 81% of the most expensive calls in the system are discarded.

### 4.2 The classes

**`low_contrast_coastline` — 24.3% of expensive triggers, 40.4% of CPU. The
worst class.** The contour *is* the coin and the coin *is* round: median
`cv_r` 0.02, largest radial dip 1°, largest spike 4°, solidity 0.86,
area_ratio 0.73. Circularity is nonetheless 0.088, because a dark bronze coin
against a dark backdrop produces a dithered, fractal-length boundary — median
perimeter **3.37×** that of an equal-area circle (p90 6.61). Nothing is wrong
with the segmentation. Example: `755609 obv` (`montage_ks17_low_contrast_coastline.jpg`)
— area_ratio 0.813, dip 0°, spike 2°, and **39.0 s** of Hough that returns
essentially the same circle.

**`non_circular_flan` — 23.3% of expensive.** The contour is the coin, and the
coin genuinely is not a circle: klippen, square strikes, irregular hand-struck
ancient flans. `cv_r ≥ 0.06` with ≥2 separate radial excursions. The
canonical case is kuenker lot 1070, a lozenge-shaped 1617 Strasbourg
klippe (`montage_kuenker_non_circular_flan.jpg`): the Otsu contour traces all
four corners perfectly, circularity reads 0.610 / 0.649, and **58.6 CPU-s**
of rim recovery is spent "recovering" a circle that **clips off all four
corners of the coin**. Here recovery is not merely wasted, it is actively
destructive. leu 713 is the ancient-flan version — a lumpy Celtic bronze at
circ 0.647 / area_ratio 0.844, both a hair under threshold.

**`relief_self_segmentation` — 64.9% of all triggers, 15.2% of expensive,
1.9% of CPU.** Otsu lands *inside the coin*, and the resulting blobs are the
lit facets of the design. `755362 obv` is the type specimen: 15 valid
contours, red boundaries tracing Alexander's hair curls, cheek and drapery.
Individually cheap, but this class alone is why the trigger fires ~3,000
times on 287 images.

**`sub_coin_noise_blob` — 26.9% of all triggers, 0.1% of CPU.** The backdrop
template's speckle and JPEG mosquito noise, each speck its own contour, each
invoking Hough at sub-millisecond cost. Visible as dozens of tiny red rings in
`755481_rev` / `755609_obv`.

**`backdrop_vignette_blob` — 8.8% of expensive, 22.3% of CPU.** The blob *is*
the backdrop. The threshold landed inside the vignette's own ramp, so the
brighter middle of the backdrop became foreground with the coin as a hole
punched through it. `755455 rev`: `area_frac 0.674`, ROI 1500×1437 (the entire
frame), **33.7 s**; `755481 rev`: `area_frac 0.456`, **47.8 s**. In both the
"recovered rim" is a circle drawn around the background.

**`multi_object_weld` — 12.8% of expensive.** Several coins fused into one
blob by `MORPH_CLOSE`. Rare in KS-17 (single coin per side) and dominant in
the leu/kuenker slow lots: leu 3717 is a 24-denarius group plate where whole
rows weld together; kuenker 1030 is four coins in a row with three fused
(`area_frac 0.51`, ROI 1648×547, 18.1 s, recovery **rejected**).
*Detection note:* the obvious test `roi_w > k·enc_r` does **not** work — roi_w
is ~2.2× enc_r for every compact blob. The tool uses bbox aspect ≥1.5 or
`area_ratio < 0.55` (equivalently `enc_r / √(area/π) > 1.35`, which is ~1.0 for
one round coin, ~1.25 for a klippe, ~1.41 for two coins, ~1.73 for three).

**`attached_artifact` — 2.9% of expensive, 0.2% of CPU.** A compact lobe on an
otherwise coin-sized blob. This is where the folklore's "attached shadow /
holder / label" actually lives, and it is nearly free. The clearest case is
strictly house-correlated: the **Künker "K" watermark** composited at top
centre appears as a near-identical ~130×100 px contour in all three kuenker
lots (circ 0.388/0.395/0.400, `area_frac` 0.0066/0.0035/0.0027).

**`unclassified_ragged` — 12.6% of expensive, 15.1% of CPU.** Honestly
unresolved: expensive blobs matching none of the above cleanly. Worth
attention before any mechanism is shipped, since it holds real cost.

### 4.3 Which classes are house-correlated

| class | correlation |
|---|---|
| `backdrop_vignette_blob`, `relief_self_segmentation`, `sub_coin_noise_blob`, `low_contrast_coastline` | **Photography-regime correlated, and the regime is a per-house constant.** All four are downstream of the CNG composited template backdrop, which is byte-identical corpus-wide. Any house shipping a dark, vignetted, composited backdrop inherits all four. |
| `multi_object_weld` | **Sale-content correlated, not house-correlated.** Driven by group lots (leu 3717's 24-coin plate, kuenker 1030's four-coin row). Any house running multi-coin lots produces it. |
| `attached_artifact` | **Strictly house-correlated** where it is a watermark (Künker "K"). |
| `non_circular_flan` | **Not house-correlated at all** — a property of the coins. Klippen, Celtic bronzes, irregular ancients. Will appear in any corpus containing them. |

Within KS-17, the regime split is measurable: sides with `avg_bg < 45`
(n=517) carry an expensive trigger **64.2%** of the time; the light-backdrop
subset `avg_bg > 85` (n=57) only **17.5%**.

---

## 5. Architecture assessment

Every candidate is judged against three constraints: **no GPU** (4 vCPU /
8 GB); **both-lanes geometric consistency** (ingest and query masking must
stay consistent with the served corpus — any geometry change needs the mask-IoU
≥ 0.995 and embedding-drift ≥ 0.995-median gates); and the **owner ruling**
that outcome changes be confined to the currently-pathological tail.

That third constraint is the sharp one, and it is what separates the
candidates. It rules out anything that alters segmentation corpus-wide, no
matter how much better the new segmentation is in principle — the 4,139 served
cards were built on today's geometry.

### Ranked

**#1 — Skip the *Hough branch only* on disc-shaped blobs, keeping the
geometric fit.** *(mine, not in the supplied list)*

This is not the formulation I started with, and the difference is the whole
finding. §6 measures both.

The naive version — "stop rim recovery firing on blobs that are provably
discs" — **FAILS, measured** (§6.1). It removes not just Hough but the
`geometric_fit_recovery` that runs before it, and that fit is what produces
the clean circle (circularity 0.997–0.998) the current crops are built from.
Killing both drops the primary detection back to the ragged seed contour
(circularity 0.14–0.28), moves the primary bbox by 8–14% IoU, and lands alpha
IoU at 0.857–0.926 — far outside the 0.995 gate. That is the Scope A failure
mode again, and it must be reported as such.

The correct intervention is narrower, and the yield data points straight at
it: **on the expensive tier `recover_rim` already returns the geometric fit 17
times out of 21**. Hough runs for 40–200 CPU-seconds and is then discarded, in
favour of an answer that was computed ~20 ms earlier. So:

```
today : recover_rim = geometric_fit ; if geo_conf <= 0.65 -> ALSO run Hough,
                                      take Hough only if hou_r > geo_r*1.05
probe : same, but skip the Hough call when the seed blob is a disc
        (cv_r < 0.06 AND area_ratio >= 0.55)
```

- *Evidence it hits the real failure*: hull circularity 0.965 vs raw 0.096;
  the 4× downscale sweep; the worst cost class (40.4% of CPU) is exactly this
  shape; and Hough's answer is used in 1 of those 7 contours.
- *Outcome preserved where it matters*: the geometric fit still runs, so the
  clean recovered circle the served corpus was built on is still produced.
  Wherever geo would have won anyway, the result is identical **by
  construction**, not by measurement.
- *Blast radius bounded by construction*: this can only ever remove Hough
  calls, never add one. Sides where nothing is removed are bit-identical.
- *Does not break the case recovery exists for*: the `area_ratio ≥ 0.55` guard
  keeps the full two-stage path on genuinely bitten/fragmented coins (the
  109704-style eaten-arc failure), which have low area_ratio by definition.
- *CPU/GPU*: `cv_r` is one 360-ray sample of an existing mask, sub-millisecond.
- *Residual risk, to be measured by Bar 1*: the 4-in-21 cases where Hough
  legitimately wins. The disc gate is designed to exclude them (a blob needing
  a genuinely different rim is not a clean disc), but that is a claim for
  measurement, not assertion.
- *Weakness*: a mitigation, not a root-cause fix. It leaves the bad
  segmentation in place and stops the system paying 40–200 s to react to it.
  It does nothing for `backdrop_vignette_blob` (22.3% of CPU), which needs #2.

**#2 — (a) border-sampled background modelling / distance-to-background
segmentation.** The root-cause fix, and this corpus makes the strongest
possible case for it: the backdrop is a **fixed template with per-pixel std
0.00**, so a background model here is not an estimate, it is exact. It would
eliminate `backdrop_vignette_blob` and `relief_self_segmentation` outright —
24% of CPU and 65% of trigger count — because "distance from the known
backdrop" is large across the whole coin and ~0 across the backdrop,
regardless of where the coin's internal tonal range sits.

*But as a wholesale replacement for global Otsu it violates the ruling*: it
changes segmentation on every image, including the 92 sides that are healthy
today, and would require re-clustering the served corpus. **It is only
shippable gated** — keep Otsu bit-identical, and take the background-model
path only when a pathology detector fires. A good detector already exists in
the data: `n_valid` is median **1** on healthy sides and median **10** on
expensive ones. Ranked #2 rather than #1 only because of that gating
requirement; on merit it is the better engineering.

A cheaper first step in the same direction, worth measuring before the full
model: **repair `detect_background_histogram` instead of replacing Otsu.** Its
fallback path returns `mean(pixels < 50)`, which is not a background estimate
by any reading, and it runs on 100% of this corpus. Replacing the fallback
with an outer-ring median (79.0 here, versus the 31.2 it currently returns)
is a small, local change — but it flips polarity decisions, so it is *not*
tail-confined and needs the same gating discussion.

**#3 — (e) per-house photography profiles.** Sound, and cheaply justified by
the std-0.00 template finding — but it is a delivery mechanism, not a
mechanism. It is how #2 would be gated (per-house backdrop plate), not a fix
in itself. Note the existing precedent and its lesson: `CLOSE_KERNEL_BY_HOUSE`
is membership-gated precisely so untabled houses stay bit-identical. Any
background profile should follow that pattern exactly.

**#4 — (c) time-budget escalation.** Rejected on the measurements. The cost
distribution is bimodal with an *empty gap*: a call is either ~0.3 s or
~100 s, so a CPU budget cannot discriminate — it can only truncate calls
mid-flight, and truncation on the 4-of-20 cases where Hough's answer is
actually used is exactly the speed-for-accuracy trade the owner rejected. It
also reintroduces the cap800/cap1024 failure mode: accumulator-resolution
changes flip outcomes in *both* directions.

**#5 — (b) Canny/edge contour as a cheaper first fallback before Hough.**
Weakly motivated here. `hough_rim_recovery` already computes full-resolution
Canny edges internally for its acceptance gate, so the edge map is not the
expensive part — the accumulator vote is. And the branch that already runs
first and costs ~20 ms (`geometric_fit_recovery`) *already wins 81% of the
expensive cases*. Adding a third cheap fallback ahead of Hough addresses a
bottleneck that the measurements say is not there.

**#6 — (d) learned salient-object segmentation on CPU (U2-Net / FastSAM
class).** Not advocated, and deliberately not benchmarked. The constraint that
kills it is not latency, it is the geometric-consistency gate: a learned
segmenter produces a *different mask everywhere*, so the entire 4,139-card
served corpus would need re-embedding and re-clustering — the opposite of
tail-confined. It should be reconsidered only if the corpus is being rebuilt
for another reason. (For reference, the honest budget on a 4-vCPU box with no
GPU is order 1–3 s per 1500×1440 side for U2-Net-class models — cheaper than
today's 98 s pathological path, but far more expensive than the 0.31 s healthy
path, which is 84% of the work.)

**Not recommended: chasing the coin/backdrop contrast itself.** The
`low_contrast_coastline` class is dark bronze on a dark backdrop. That is a
photographic fact of the source images; CNG serves a single lossy JPEG per lot
with no alpha and no background-removed variant (verified 2026-07-23 against
`auctions.cngcoins.com` — one `…/5_1.jpg`, no PNG, no zoom alternate), so
there is no upstream escape hatch.

---

## 6. Measured probes — mechanism #1

`tools/rim_trigger_shape_probe.py`. No production file modified: the probe
monkeypatches `src.layer1_geometry.recover_rim` in its own process with a
wrapper that returns `(None, 0)` for blobs failing the new conjunct and
delegates otherwise — exactly equivalent to the trigger not firing, since
pass 2 keeps the seed contour and sets `rim_recovered=False` whenever recovery
yields None.

Both arms run the **real, unmodified `layer_1_structural_salience`** on the
same side, and are compared on detection count, greedily-matched bbox IoU,
`rim_recovered` flags, the alpha mask handed downstream, and CPU seconds.

Two arms, both gated on the same scale-invariant disc test
(`cv_r < 0.06 AND area_ratio ≥ 0.55`):

- **`recover_skip`** — suppress rim recovery entirely (patches
  `layer1_geometry.recover_rim`).
- **`hough_skip`** — keep `geometric_fit_recovery`, suppress only the Hough
  branch (patches `rim_logic.hough_rim_recovery`, so `recover_rim` falls
  through to `return geo_c, geo_conf`).

### 6.1 `recover_skip` — FAILS. Reported because it is the obvious version.

Top 8 `low_contrast_coastline` sides (the worst class by CPU, §4.1):

| side | n dets | alpha IoU | primary bbox IoU | primary circularity | CPU |
|---|---|---:|---:|---|---|
| 755369 rev | 2→1 | 0.885 | 0.886 | 0.998→0.175 | 100.0→0.7 s |
| 755387 obv | 2→1 | 0.915 | 0.902 | 0.998→0.275 | 108.4→0.5 s |
| 755387 rev | 1→2 | 0.926 | 0.896 | 0.998→0.220 | 146.4→0.6 s |
| 755401 obv | 2→1 | 0.909 | 0.916 | 0.998→0.159 | 110.1→1.0 s |
| 755408 rev | 1→2 | 0.884 | 0.861 | 0.997→0.151 | 95.3→0.4 s |
| 755409 rev | 1→1 | 0.913 | 0.866 | 0.997→0.149 | 91.9→0.4 s |
| 755411 obv | 4→4 | 0.857 | 0.905 | 0.997→0.274 | 53.3→0.8 s |
| 755414 rev | 2→5 | 0.874 | 0.855 | 0.997→0.144 | 68.9→1.1 s |

**0/8 outcome-unchanged. Worst alpha IoU 0.857** — far outside the 0.995 gate.
CPU 774.3 s → 5.6 s (99.3% saved), which is exactly the seduction to resist.

*Why it fails, and it is not the reason I expected.* The recovery on this class
is **not** a no-op. The control arm's primary detection has circularity
0.997–0.998 — recovery replaced the ragged low-contrast outline with a clean
circle, and that circle is what the current crops (and therefore the served
corpus) are built from. Suppressing recovery drops the primary back to the
raw seed at circularity 0.14–0.28. This is a real geometry regression, and it
is the Scope A failure mode wearing a different hat.

*Note on detection-count churn.* Counts swing in both directions (2→1, 1→2,
2→5). That is the NMS-containment mechanism the cap800 A/B already documented:
a large recovered circle suppresses small noise-blob candidates as "contained",
and without it those specks survive as extra detections. It is a symptom of
the census, not of the coin's crop — which is why `primary_bbox_iou` is
reported separately.

### 6.2 `hough_skip` — PASSES on the worst class.

Same class, same gate, but the geometric fit is preserved and only the Hough
call is suppressed:

| side | n dets | alpha IoU | primary bbox IoU | primary circularity | Hough skipped | CPU |
|---|---|---:|---:|---|---:|---|
| 755369 rev | 2→2 | **1.0** | 1.0 | 0.998→0.998 | 1/4 | 96.5→0.7 s |
| 755387 obv | 2→2 | **1.0** | 1.0 | 0.998→0.998 | 1/5 | 110.1→0.7 s |
| 755387 rev | 1→1 | **1.0** | 1.0 | 0.998→0.998 | 1/7 | 124.2→0.7 s |
| 755401 obv | 2→2 | **1.0** | 1.0 | 0.998→0.998 | 1/12 | 106.8→1.1 s |
| 755408 rev | 1→1 | **1.0** | 1.0 | 0.997→0.997 | 1/5 | 92.2→0.5 s |
| 755409 rev | 1→1 | **1.0** | 1.0 | 0.997→0.997 | 1/4 | 71.5→0.4 s |

**6/6 outcome-unchanged. Worst alpha IoU 1.0** — not "within tolerance",
*identical*. Detection counts, primary bboxes and primary circularities all
match exactly. **CPU 601.2 s → 4.0 s (99.3% saved.)**

This is the result the owner ruling asks for: the entire cost of the worst
class is removed and nothing downstream moves, because the answer that
survives was never Hough's to begin with.

### 6.3 `hough_skip` on the classes it is *not* claiming

The gate is deliberately narrow, and this is the check that it stays narrow.
On `non_circular_flan` — klippen and irregular flans, where the blob is a coin
but genuinely not a disc — the gate correctly **never fires**:

| side | n dets | alpha IoU | Hough skipped | CPU |
|---|---|---:|---:|---|
| 755408 obv | 1→1 | 1.0 | **0/10** | 76.4→76.4 s |
| 755411 rev | 5→5 | 1.0 | **0/19** | 19.9→18.9 s |
| 755412 obv | 5→5 | 1.0 | **0/15** | 75.1→77.0 s |
| 755412 rev | 5→5 | 1.0 | **0/11** | 50.2→49.8 s |
| 755413 rev | 5→5 | 1.0 | **0/16** | 59.2→58.7 s |

**6/6 unchanged, worst alpha IoU 1.0, 0 of 71 Hough calls skipped, CPU
352.9 s → 358.2 s (−1.5%, i.e. no saving).** That is the right answer, not a
disappointing one: `cv_r` correctly reports that a klippe is not a disc, so
the mechanism declines to touch it. Mechanism #1 buys the coastline class and
**explicitly does not claim** `non_circular_flan` (15.0% of CPU) — which,
per §4.2, is the class where recovery is actively destructive and therefore
needs a different intervention entirely.

### 6.4 `hough_skip` on `backdrop_vignette_blob` — 5/6, and the one failure matters

| side | n dets | alpha IoU | primary bbox IoU | primary circularity | Hough skipped | CPU |
|---|---|---:|---:|---|---:|---|
| 755381 rev | 1→1 | 1.0 | 1.0 | 0.997→0.997 | 1/1 | 130.5→0.4 s |
| **755395 obv** | 1→1 | **0.806** | **0.868** | **0.998→0.171** | 1/3 | 107.5→0.3 s |
| 755402 rev | 1→1 | 1.0 | 1.0 | 0.998→0.998 | 0/3 | 99.7→92.1 s |
| 755426 obv | 2→2 | 1.0 | 1.0 | 0.998→0.998 | 1/3 | 131.7→0.3 s |
| 755513 obv | 5→5 | 1.0 | 1.0 | 0.998→0.998 | 1/8 | 88.5→0.7 s |

**5/6 outcome-unchanged, worst alpha IoU 0.806, CPU 644.3 s → 188.5 s (70.7%
saved).**

`755395 obv` is the residual risk made concrete: this is one of the 4-in-21
cases where **Hough's circle is the one `recover_rim` actually returns**, so
suppressing it drops the primary from a clean circle (0.998) back to the
ragged seed (0.171). The disc gate does not currently exclude it, because a
frame-spanning backdrop blob with a coin-shaped hole can still read as
low-`cv_r` on its outer boundary.

**Obvious tightening, NOT yet measured:** add `largest_hole_frac < 0.10` to the
skip gate. That is precisely the statistic §4 uses to identify
`backdrop_vignette_blob`, so it would exclude this class from the skip
entirely — narrowing mechanism #1 to the coastline class it passes cleanly on,
at the cost of the 22.3%-of-CPU backdrop class (which is #2's job anyway). This
is a hypothesis for the Bar-1 run to settle, not a result.

### 6.5 Summary of the four probe runs

| arm | class | sides unchanged | worst alpha IoU | CPU |
|---|---|---:|---:|---|
| `hough_skip` | low_contrast_coastline (worst class) | **6/6** | **1.0** | 601.2 → 4.0 s (−99.3%) |
| `hough_skip` | non_circular_flan | 6/6 | 1.0 | 352.9 → 358.2 s (−1.5%, gate declines) |
| `hough_skip` | backdrop_vignette_blob | 5/6 | 0.806 | 644.3 → 188.5 s (−70.7%) |
| `recover_skip` | low_contrast_coastline | **0/8** | 0.857 | 774.3 → 5.6 s (−99.3%) |
| `recover_skip` | HEALTHY (no trigger at all) | 12/12 | 1.0 | 1.5 → 1.5 s (inert) |

The HEALTHY row is the inertness control: on sides that never trigger, both
arms are bit-identical and neither costs anything — the guard does not perturb
the 84%-of-work fast path.

---

## 7. Measurement plan for mechanism #1 (precommit-bar style)

Following the convention of `analysis/prehammer_estimate/
ea614_verdict_precommit_2026-07-21.md` and `specs/rim_recovery_neighbor_aware.md`
§PRECOMMIT ACCEPTANCE BARS. **These bars are to be ratified before any
measurement is run against them.**

**Ships default-off** behind `TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD`, unset =
today's behaviour bit-identical, following the Scope B precedent.

**Scope of the proposal, fixed by §6:** the guard suppresses **only the Hough
branch** (`recover_rim` keeps calling `geometric_fit_recovery`), and only for
blobs passing the disc test. §6.1 already measured the wider version and it
fails; it is not on the table. The gate should additionally carry
`largest_hole_frac < 0.10` per §6.4, restricting it to the
`low_contrast_coastline` class — the run below is what decides whether that
extra conjunct is required or merely prudent.

### Bar 1 — outcome-identity on the affected sides (BLOCKING)

Full KS-17 population, both `ingest` (house=`cng_feature`) and `query`
(house=None) lanes, guard on vs off:

- **≥ 98% of sides outcome-unchanged**, where unchanged means identical
  detection count AND every matched bbox IoU ≥ 0.99 AND identical
  `rim_recovered` flags.
- **Every** changed side individually reviewed with an overlay panel and
  dispositioned in the results doc. A change is acceptable only if the probe
  arm's contour is the *seed* contour and the control arm's "recovery" was
  visibly wrong (the klippe-clipping class). Any side where control was right
  and probe is worse is a **FAIL**, not a trade.
- **Worst alpha mask IoU ≥ 0.995** across all sides — the project's standing
  geometry gate. Not a median; the minimum.

*Known live risk this bar exists to catch:* §6.4 measured **1 changed side in
6** on `backdrop_vignette_blob` (755395 obv, alpha IoU 0.806) — a case where
Hough's circle really was the returned answer. On the full population that rate
must come in under 2%, or the `largest_hole_frac` conjunct must be added and
the bar re-run.

### Bar 2 — the guard must not disarm real rim recovery (BLOCKING)

Rim recovery exists to rescue genuinely fragmented coins. On the historical
cases it was built for (the 109704-style eaten-arc failure, plus the leu/
cng_feature frozen weld samples in `specs/two_coin_weld_sample_ids.csv`):

- **Zero** sides where a control-arm recovery that was *correct* is skipped by
  the guard. Verified by construction (`area_ratio ≥ 0.55` guard) **and**
  measured on the frozen samples.

### Bar 3 — cost (REPORTED, not blocking)

- p99 CPU-seconds on the KS-17 fixture, guard on vs off. §6 measured −99.3% on
  the worst class and −70.7% on the backdrop class; the bar records the
  population number rather than gating on it, because per the owner ruling
  cost is the *reward*, never the justification.

### Bar 4 — cross-house transfer (BLOCKING before any default flip)

Per the Scope C precedent, kuenker is **not** assumed to inherit cng_feature's
numbers. Bars 1–3 re-run independently on the frozen kuenker sample
(`purpose=kuenker_wallclock`) and on leu before the default is considered.
Note kuenker's expensive tier is dominated by `non_circular_flan` (klippen),
where §4.2 shows control-arm recovery is actively destructive — so kuenker is
where Bar 1's "changed sides" are most likely to be *improvements*, and each
must still be individually dispositioned rather than waved through.

### Bar 5 — served-corpus consistency (BLOCKING before any backfill)

If and only if Bars 1–4 pass and a re-crop/backfill is proposed:
embedding-drift ≥ 0.995-median against the current served vectors, plus the
standing `visual_search` regression (`routing_bar.py`, `stage2_bar.py`,
per-slice fixtures) green.

### Explicitly out of scope for this mechanism

`non_circular_flan` (15.0% of CPU) and `unclassified_ragged` (15.1%) are **not**
addressed by mechanism #1 and will still stall — §6.3 measured the gate
declining to fire on the first of these, by design. `backdrop_vignette_blob`
(22.3%) is *partially* reached (§6.4, 70.7% CPU saved) but at a 1-in-6 outcome
change, so it is not claimed either and may be excluded outright by the
`largest_hole_frac` conjunct. All three need #2, and #2 needs its own gating
design first.

The klippe case deserves its own follow-up independent of cost: §4.2 shows
recovery *clipping the corners off* square coins. That is a correctness bug
today, at any speed.

---

## 8. Reproduction

```bash
cd ~/trivalaya-vision            # (this work: worktree on branch hough-taxonomy)
V=~/trivalaya-vision/.venv/bin/python
KS=~/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images

$V tools/rim_stall_taxonomy.py scan --images $KS --house cng_feature \
    --layout half --out specs/results/rim_stall_taxonomy_ks17_scan.json
$V tools/verify_taxonomy_replication.py \
    --scan specs/results/rim_stall_taxonomy_ks17_scan.json \
    --cost-csv specs/results/rim_recovery_cost_ab_ks17.csv --lane ingest
$V tools/rim_stall_taxonomy.py classify \
    --scan specs/results/rim_stall_taxonomy_ks17_scan.json \
    --out specs/results/rim_stall_taxonomy_ks17_classified.csv
$V tools/rim_scale_sensitivity_probe.py \
    --scan specs/results/rim_stall_taxonomy_ks17_scan.json --images $KS \
    --house cng_feature --stride 6 --out specs/results/rim_scale_sensitivity_ks17.csv
$V tools/rim_hough_yield_probe.py \
    --scan specs/results/rim_stall_taxonomy_ks17_scan.json --images $KS \
    --house cng_feature --stride 16 --out specs/results/rim_hough_yield_ks17.csv
# §6.2 — the arm that passes (and §6.1's failing arm via --arm recover_skip)
for M in low_contrast_coastline non_circular_flan backdrop_vignette_blob; do
  $V tools/rim_trigger_shape_probe.py \
      --scan specs/results/rim_stall_taxonomy_ks17_scan.json \
      --classified specs/results/rim_stall_taxonomy_ks17_classified.csv \
      --images $KS --house cng_feature --arm hough_skip --mechanism $M \
      --limit 6 --out specs/results/rim_trigger_shape_probe_houghskip_$M.csv
done
# inert-where-it-does-not-fire control
$V tools/rim_trigger_shape_probe.py ... --arm recover_skip --mechanism HEALTHY
```

`src/` carries **zero diff against main** in this worktree, and the suite is
green at **230 passed** — the same count as the pre-existing baseline. Every
arm above is applied by in-process monkeypatch inside the probe.

Overlays: `specs/results/rim_stall_taxonomy_overlays/{ks17,leu,kuenker}/`.
Montages: `specs/results/rim_stall_taxonomy_montages/montage_<fixture>_<class>.jpg`.
Panel legend — **red** = Otsu seed contour (thick = triggers Hough),
**blue** = its minEnclosingCircle, **green** = what recovery returned and
`validate_rim_recovery` accepted. Non-blob area is dimmed 55%.

### Caveats

- **leu (6 lots) and kuenker (3 lots) are pre-selected slow lots.** Their class
  mixes show *presence*, never frequency. KS-17 is the only full population
  here (574/574 sides).
- The CPU-attribution table (§4.1) is a stride-16 sample of stall sides
  (282 contours), not the full 4,701; the count tables are full-population.
- Wall/CPU numbers were taken on a contended 4-vCPU box;
  `time.process_time()` is used throughout so figures reflect compute, not
  scheduler wait. Cost *ratios* and the bimodality are contention-insensitive;
  absolute seconds are not.
- `unclassified_ragged` (12.6% of expensive, 15.1% of CPU) is unresolved.
- The `multi_object_weld` / `non_circular_flan` boundary is genuinely fuzzy at
  this feature resolution for two-coin leu blobs near `area_ratio ≈ 0.55`
  (e.g. leu 3661 c18 at 0.560). Rules are documented in the classifier.
