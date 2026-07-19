# Two-Coin Weld: scale-relative MORPH_CLOSE

L1 segmentation closes its Otsu binary with a **fixed** 7×7 ellipse at 2
iterations. On low-resolution auction images this bridges the gap between
two correctly-separated coins, producing a merged blob that then costs a
full Hough two-coin split to undo. Make the kernel scale-relative so its
bridging distance is proportional to image size rather than absolute.

Owner: vision / L1 geometry
Status: **code landed behind `TRIVALAYA_CLOSE_KERNEL_FRAC` (§6.1); default
still fixed-7. §4 corpus validation NOT started — do not flip the default.**
Version: 3
Date: 2026-07-19 (v1, v2, v3 — see §10)

### Implementation status

| item | state |
|---|---|
| §1 kernel helper + env gate + config constants | **done** |
| §1 dead-constant cleanup (`CLOSE_KERNEL_SIZE_*` deleted, `CLOSE_ITERATIONS` wired) | **done** |
| Per-house override mechanism (**deviation from §3** — see below) | **done, table empty** |
| §9 harness prerequisite (venv rebuild, `tests/` + pytest) | **done** |
| §9.1 tier 1 — kernel sizing | **done** (47 tests) |
| §9.2 tier 2 — synthetic weld fixtures | **done** (95 tests) |
| §4 / §5 / §9.3 — corpus sweep, exit criteria, real lots | **not started** — needs Spaces + DB |

Verified at landing: with the env var unset, L1 output is byte-identical to
the pre-change code on real lots (bboxes + areas, `data/test_images`), and
the §9.1 default-unchanged guard pins the kernel reaching OpenCV at (7,7).

The bridging model `gap < 2 · iterations · (k // 2)` was re-confirmed against
OpenCV 4.12 during implementation, exactly, at S ∈ {500, 1200, 3000} and
g ∈ {1..25} — including both boundaries (k=7 welds at 11, survives at 12;
k=3 welds at 3, survives at 4). The Gaussian blur contributed zero bridging
on synthetics, as v3 recorded.

### Deviation from §3: per-house overrides now exist

§3 states "There is no per-house branch," and the scale-relative design leans
on that. Owner has authorised per-house configuration on the grounds that
input images differ by house. Implemented as `Layer1Config.CLOSE_KERNEL_BY_HOUSE`,
with `house` threaded through `analyze_image` → `layer_1_structural_salience`
→ `_segment_and_extract_candidates`.

**The table ships empty, and that is deliberate.** The formula already absorbs
the axis houses most obviously differ on — dimensions. An override is only
justified where a house differs on something *else* (gap distribution, toning,
glare) *and* §4.2 measured it. The current evidence cannot separate those:
cng_feature is 500px with 5–8px gaps and kuenker is 800–3000px with ~25px
gaps, so scale and gap are fully confounded at n=14. Disentangling them is
exactly what §4.1 is for. Populating the table before that sweep would repeat
v1's error — a constant reasoned about rather than computed.

Mechanism notes:

- Precedence: `TRIVALAYA_CLOSE_KERNEL_FRAC` > per-house > global. A sweep arm
  must mean one thing corpus-wide, so the env frac outranks per-house; the
  per-house `min`/`max` still clamp, as bounds rather than the quantity under test.
- The env var gained an `auto` form meaning "scale-relative, use configured
  values." A bare numeric value pins one frac everywhere, which would make
  per-house overrides untestable. `auto` is what the flipped default becomes.
- `validate_close_kernel_overrides()` runs at import and rejects even `max`,
  `min > max`, non-positive `frac`, non-lowercase keys, and **unknown keys** —
  a typo'd `{"fraq": ...}` would otherwise validate silently and leave the
  house on the global default, i.e. a tuning that never applied.
- **Per-house is dormant until §6.3 flips the default.** With the env var
  unset, production is fixed 7×7 for every house regardless of the table.

**Batches can be house-mixed**, so this had to be a per-image argument, not
process-level config: `trivalaya_pipeline/pipeline.py:314`'s
`process_vision(auction_house=None)` defaults to all houses, and
`record.auction_house` is per-record. Setting the env var per worker would
have silently applied one house's kernel to another's images.

**Cross-repo change still required** (not made — different repo): for `house`
to reach L1 in production, `trivalaya-pipeline` must pass it at
`trivalaya_pipeline/pipeline.py:606`:

```python
result = self.vision.process_image(local_path, source_type="auction",
                                   house=record.auction_house)
```

plus forwarding through `VisionAdapter.process_image` /
`_run_vision_pipeline` (`vision_adapter.py:171,194,198`). Until then `house`
is `None` everywhere and every caller gets the global — which, with an empty
table, is identical behavior either way.

Note for §7.3: on real 1200px lots (`data/test_images/Lot_0000*.jpg`),
enabling the scale-relative path left `ndets` unchanged at 2 but **shifted
bboxes** on 4 of 5. leu is 1200px and 125k coins — this is the silent
mid-size drift §7.3 predicts, observed rather than hypothesised, and it
confirms leu needs the distributional bar (§9.3e) rather than §4.3's
identity test.

---

## Why

`layer1_geometry._segment_and_extract_candidates` runs:

```python
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
```

A 7×7 ellipse dilated twice reaches ~6px from each side — roughly **12px
of bridging power** (empirically exact: on synthetic pairs the full
chain welds gaps ≤11px and survives at 12px — §9.2), in absolute
pixels, regardless of input dimensions.

That is 1.4% of a 500px-wide image and 0.23% of a 3000px one. The same
kernel therefore has opposite effects on the two dominant auction formats:
on large images it repairs fragmented coin masks (its intent); on small
images it welds two distinct coins into one blob (unintended).

The welded blob has `aspect_ratio ≈ 2.1, circularity ≈ 0.45`, which trips
`TwoCoinResolver.should_trigger`. The pipeline then spends a HoughCircles
pass plus N×N pair scoring to recover a split that **thresholding had
already produced correctly two lines earlier**.

This is the dominant per-coin cost difference between auction houses.

---

## Measured evidence

Sampled source images pulled from Spaces, re-run through L1's exact
gray → CLAHE → Otsu chain, counting contours (≥ `MIN_AREA_PX`) before and
after the close. Read-only; no writes.

| house | raw dims | gap between coins | blobs pre-close → post-close |
|---|---|---|---|
| cng_feature | 500 × ~240 | **5–8 px** (median 7) | 2 → **1** in 14/14 lots |
| kuenker | 800–3000 wide | **~25 px** | 2 → **2** (0/10 welded) |

Measurements replicate L1's exact chain — `cvtColor` → `detect_background_histogram`
→ CLAHE(2.0, 8×8) → `GaussianBlur (7,7)` → Otsu with polarity by background
→ `MORPH_CLOSE`. Contours filtered at `MIN_AREA_PX`.

7px < 12px bridging → welded. 25px > 12px → survives. In every
cng_feature lot sampled, Otsu alone had already found both coins; 6 of 14
had *both* blobs at circularity > 0.70 (unambiguously clean circles)
before the close destroyed the separation.

The mechanism predicts observed Hough rates, which track image width
across the whole corpus:

| house | raw width | hough-split coins | rate |
|---|---|---|---|
| cng_feature | 500 | 5,196 / 6,085 | **85.4%** |
| davissons | 370 | 185 / 365 | 50.7% |
| naumann | — | 11,293 / 24,076 | 46.9% |
| leu | 1200 | 52,337 / 125,039 | 41.9% |
| nomos | — | 4,553 / 12,544 | 36.3% |
| mashops | 1700 | 18,703 / 82,821 | 22.6% |
| gorny | 1200 | 2,416 / 11,997 | 20.1% |
| obolos | ~1989 | 4,576 / 38,631 | 11.8% |
| cng | 3000 | 3,545 / 42,080 | 8.4% |
| kuenker | 800–3000 | 16 / 1,911 | **0.8%** |

Note `cng` and `cng_feature` share a name and nothing else — 3000×1440
vs 500×234, 8.4% vs 85.4%. Do not reason about "CNG" as one thing.

**Sample size caveat:** the gap measurements are 14 lots per house. That
is enough to establish the mechanism and rule out coincidence, not enough
to tune a threshold. §4 widens it before any code change lands.

---

## Observed cost

Measured from production logs, same 4-vCPU box:

| | per lot | per coin | hough rate |
|---|---|---|---|
| kuenker nightly (`vision --batch 1000`) | 1.57 s | 0.52 s | 0.8% |
| cng_feature (runner job 299) | ~3.4 s | ~1.75 s | 85% |

Hough is not the whole 3.3× per-coin gap — CPU contention and
`validate_split` contribute — so treat elimination of the weld as
*necessary but not sufficient* for closing it. §5 sets the acceptance bar
on measured wall-clock, not on the hough-rate drop alone.

---

## 1. The change

Size the structuring element relative to the image, with an absolute
floor and ceiling. Extract the sizing into a named helper so it is
unit-testable in isolation (§9.1 depends on this — do not inline it):

```python
# layer1_geometry
def _close_kernel_size(h: int, w: int, frac: float | None = None) -> int:
    """Scale-relative MORPH_CLOSE kernel. Odd, clamped, deterministic.
    Pure function of its arguments — no env reads — so §9.1 tests it
    directly."""
    f = Layer1Config.CLOSE_KERNEL_FRAC if frac is None else frac
    k = int(max(h, w) * f)   # floor, NOT round
    k = max(Layer1Config.CLOSE_KERNEL_MIN, min(k, Layer1Config.CLOSE_KERNEL_MAX))
    return k | 1  # ellipse kernels must be odd

# _segment_and_extract_candidates — the env gate lives HERE, at the
# call site, not inside the helper. Until §6.3 flips the default, an
# unset env var means literally today's fixed 7×7 (§9.1's
# default-unchanged guard pins this). Setting it enables the
# scale-relative path, with the value overriding CLOSE_KERNEL_FRAC so
# §9.3d can A/B without a deploy.
_frac_env = _os.environ.get("TRIVALAYA_CLOSE_KERNEL_FRAC")
k = _close_kernel_size(h, w, frac=float(_frac_env)) if _frac_env else 7
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    iterations=Layer1Config.CLOSE_ITERATIONS)
```

v2's snippet called `_close_kernel_size` unconditionally — landing that
as-is would have changed production behavior on merge, contradicting
§6.1's default-preserving promise and §9.1's guard. The gate above
reconciles them. When §6.3 flips the default, the else-branch becomes
`_close_kernel_size(h, w)` and the §9.1 guard is updated **in the same
commit** — it inverts by design; do not "fix" the flip by reverting it.

Proposed constants (to be confirmed by the §4 sweep, not adopted blind):

| constant | value | rationale |
|---|---|---|
| `CLOSE_KERNEL_FRAC` | 1/400 | 7px across 2400–3199px — preserves today's behavior on large images |
| `CLOSE_KERNEL_MIN` | 3 | below 3 the close is a no-op; keeps speckle suppression |
| `CLOSE_KERNEL_MAX` | 21 | bounds cost and prevents over-welding on 4000px+ plates |

Resulting kernel by input width:

| width | house | k | vs today |
|---|---|---|---|
| 370 | davissons | 3 | change |
| 500 | cng_feature | 3 | change (the fix) |
| 800–1199 | leu, gorny | 3 | change — see §7.3 |
| 1200–1599 | leu, gorny | 3 | change |
| **1600**–2399 | mashops (1700), obolos | 5 | change |
| 2400–3199 | cng (3000×1440) | **7** | **unchanged** |
| 3200 | `MAX_DIMENSION` ceiling | 9 | change |

**The 3→5 band edge is 1600, not 1700 as v2's table said.**
`int(1600/400) = 4` and `4 | 1 = 5`: the odd-bump rounds even
quotients *up*, so k reaches 5 a full 400px band earlier than a
naive reading suggests. Verified by exhaustive sweep — k transitions
at exactly 1600 → 5, 2400 → 7, 3200 → 9. No sampled house lives in
1600–1699, which is why nothing caught it; §9.1's width table now
pins 1599/1600 explicitly. Same lesson as v1's `round()` bug: every
band edge in this table must be computed, not eyeballed.

kuenker spans 800–3000px, so it straddles k=3 through k=7 depending
on the lot — it is not a single-row house, and its 0.8% hough rate
means it has the least to gain and the most to lose. Treat its §4
result per-bucket, not as one number.

At 500px this yields k=3 — bridging `2·iterations·(k//2)` = 4px,
welding only sub-4px gaps, safely under the measured 5–8px
(empirically verified: welds ≤3px, survives ≥4px — §9.2). At 1200px,
k=3 — a real behavior change for leu/gorny and exactly what §4 must
measure rather than assume.

### Floor, not round — this bit is load-bearing

**v1 of this spec specified `int(round(...))` and claimed k=7 at
3000px. That is arithmetically false:** `round(3000/400) = round(7.5)
= 8`, and `8 | 1 = 9`. The `cng` house is exactly 3000×1440 — 42,080
coins, and precisely the population §2 promises to leave alone
("the close is doing real mask repair there"). Rounding would have
given it *more* bridging, silently, on the largest-format house in
the corpus.

Truncation gives k=7 for the whole 2400–3199 band, which covers cng's
3000px natively. Two consequences:

- `int()`, never `round()`. §9.1 pins this with an explicit test.
- `k |= 1` runs **after** the clamp, so it can only ever round *up*.
  With `CLOSE_KERNEL_MAX = 21` (odd) that is safe; if MAX is ever
  changed to an even value, k can exceed it. Keep MAX odd, and assert
  it with a bare **module-level** assert in `src/config.py`. v2 said
  "next to the existing import-time asserts (`config.py:181`)" — but
  those asserts are not import-time: they sit inside
  `validate_config()`, which nothing calls except the file's own
  `__main__` block (`config.py:185`). An assert placed there never
  runs in production. The §9.1 clamp test also covers this; the
  module-level assert is the one that fires without a test suite.

### Other scale-absolute steps in the same chain

`MORPH_CLOSE` is not the only fixed-size operation before the contour
count. `layer1_geometry.py:157` runs `GaussianBlur(gray_enhanced,
(7,7), 0)` immediately before the Otsu threshold — same fixed size,
same merging effect, same 1.4%-of-a-500px-frame problem this spec
identifies for the close.

This matters for the §"Measured evidence" numbers: those gaps were
measured with the blur already applied, so an unknown share of the
bridging may be happening there rather than in the close. If so, k=3
alone will not fully recover the separation.

§4.2 therefore sweeps blur size alongside kernel size, and §4.1 adds
a pre-blur/post-blur gap measurement to attribute the bridging. Do
not change the blur in this spec — measure first, and if the blur
turns out to be the dominant term, that is a separate change with its
own blast radius.

(CLAHE `tileGridSize=(8,8)` at `:393` is already scale-relative —
8×8 tiles regardless of dimensions — so it is not implicated.)

### Clean up the dead kernel constants while here

`Layer1Config` already defines three constants that are **never
read** — the call site hardcodes all of them:

| constant | `src/config.py` | status |
|---|---|---|
| `CLOSE_KERNEL_SIZE_STANDARD = 7` | :25 | dead |
| `CLOSE_KERNEL_SIZE_HIGH = 9` | :26 | dead |
| `CLOSE_ITERATIONS = 2` | :27 | dead |

Adding `CLOSE_KERNEL_FRAC/MIN/MAX` without touching these leaves six
kernel knobs of which three are lies. This change wires
`CLOSE_ITERATIONS` (used in the snippet above) and **deletes** the two
`CLOSE_KERNEL_SIZE_*` constants.

Root cause worth recording: those two were presumably meant to hang
off the `sensitivity` parameter of `layer_1_structural_salience`
(`layer1_geometry.py:365`) — which is itself dead. `sensitivity` is
a plain `str`, is never read in the function body, and only
`Layer1Config.Standard` is ever referenced (`:405`). So
`pipeline_manager.py:61`'s `sensitivity="high"` retry re-runs
identically. Out of scope here, but do not build the new constants on
the assumption that a sensitivity switch exists.

### Why not just lower `iterations`?

`iterations=2` on a 7×7 is what produces ~12px. Dropping to 1 halves the
bridging but keeps it absolute — the scale-dependence remains, just at a
different threshold. Fixes cng_feature, leaves the class of bug intact.

### Rejected alternative: pre-close short-circuit

"If the pre-close contours already yield 2 clean circular blobs, use them
and skip close + Hough." Tempting — it's a pure fast path — but it adds a
second segmentation code path with its own acceptance criteria, and it
leaves the underlying kernel wrong so every future small-format house
re-hits it. Prefer fixing the kernel. Keep this in reserve as a fallback
if the §4 sweep shows the kernel change regresses fragmented houses.

---

## 2. Non-goals

- **Not** removing the close. On 3000×1440 `cng` images it collapses
  44–88 threshold fragments to 2–20 blobs; it is doing real mask repair
  there.
- **Not** touching `TwoCoinResolver` or the Hough parameters. If the weld
  stops happening, the resolver simply stops being triggered. Its
  behavior when genuinely needed must not change.
- **Not** re-cropping already-processed coins. Backfill is a separate
  decision (§6).
- **Not** changing `MAX_DIMENSION` (3200) or any upstream resize.

---

## 3. Blast radius

`_segment_and_extract_candidates` is on the path of **every** image the
vision pipeline processes, for every house, in both the runner and the
nightly batch. There is no per-house branch. A regression here corrupts
crops corpus-wide, and crops are overwritten in place at the same Spaces
keys — so a bad run is not trivially reversible.

This is the reason for the pilot gate in §4 rather than a direct change.

---

## 4. Validation before any code change

### 4.1 Widen the gap measurement

Sample **200 lots per house** across cng_feature, cng, kuenker, leu,
gorny, mashops, naumann, obolos, davissons (the last two are the
small-image controls). For each: image dims, contour count pre-close,
contour count post-close, inter-blob gap, per-blob circularity, **and
inter-blob gap measured pre-blur as well as post-blur** (per §1, to
attribute bridging between the blur and the close).

**Freeze the sample.** Commit the selected lot IDs to
`specs/two_coin_weld_sample_ids.csv` (columns: `house, lot_id,
raw_width, raw_height`) so re-runs after a kernel tweak are
comparable. Same discipline as the PCO pilot slate. A resampled
population between iterations makes the sweep in §4.2 uninterpretable.

Deliverable: gap distribution per house, and the fraction of lots where
`pre == 2 and post == 1` (the weld signature).

Prediction to falsify: weld rate should correlate with
`gap_px < 12` and inversely with image width. If leu (1200px, 42% hough)
shows a low weld rate, the mechanism does **not** generalize beyond small
formats and the constants in §1 need rethinking.

### 4.2 Kernel sweep

For the same sample, re-segment at `k ∈ {3, 5, 7, 9}` × `iterations ∈
{1, 2}` × `blur ∈ {(3,3), (5,5), (7,7)}` and record, per house:

- weld rate (`pre==2 → post==1`)
- fragment rate (`post > 2` where coins should be 2)
- median contour count post-close

Blur is swept as a **diagnostic only** — to quantify how much of the
bridging it owns (§1). The chosen `CLOSE_KERNEL_FRAC` must work at the
current `(7,7)` blur; a blur change is out of scope for this spec.

Pick the `CLOSE_KERNEL_FRAC` that minimizes weld rate on small formats
**without** raising fragment rate on cng/mashops/obolos.

### 4.3 Crop-equivalence check

The change alters segmentation for every house, so verify output
stability where behavior should *not* change. Take 100 lots in the
2400–3199px band (where k stays 7), run L1 before and after the patch,
and assert bboxes are byte-identical. Any diff at large scale means the
kernel math is wrong, not the policy.

**Sampling requirement — do not sample naively.** At least **20 of the
100 lots must be at exactly the cng 3000×1440 format**. 3000px is the
dimension where the v1 `round()` formula silently diverged (§1); a
sample drawn from "≥2400px" without that constraint would likely land
on 2400–2800px lots, where round and floor agree, and the whole
equivalence gate would pass green on a broken kernel. This is the
single highest-value assertion in §4 — it is the test that would have
caught v1's bug.

Record the chosen 100 in the same frozen CSV as §4.1.

---

## 5. Exit criteria

All required before bulk rollout:

0. **§9.1 and §9.2 unit tests pass.** These are cheap, deterministic,
   and gate everything below — no corpus work until the kernel math
   and the bridging model are pinned.
1. **§4.1 confirms the mechanism** at n=200/house — weld signature
   present in >70% of cng_feature lots, <5% of kuenker lots.
2. **Kernel chosen from §4.2 data**, not from the §1 guess.
3. **§4.3 passes**: bbox-identical output in the 2400–3199px band,
   including the mandatory ≥20 lots at 3000×1440.
4. **Hough rate drops** on a 200-lot cng_feature re-run: 85% → <20%.
5. **Wall-clock improves**: per-coin time on cng_feature within 1.5× of
   kuenker's 0.52 s/coin on an otherwise-idle box.
6. **Crop quality holds**: zero lots where a crop gains a sliver of the
   neighboring coin (detected mechanically per §9.3c — this is the
   important one, see §7.1), and GREEN rate does not regress.

**On the GREEN-rate bar — size it or drop it.** v1 said "on 100
re-processed lots, GREEN rate ≥ the current 83.5%." That is not
measurable at n=100: at a base rate of 83.5% the 95% CI is roughly
±7pp, so a real 3pp regression is indistinguishable from noise and a
3pp *improvement* would also read as noise. Two honest options:

- **Size to the effect.** Detecting a 5pp drop (83.5% → 78.5%) at 80%
  power needs ~n=750 per arm one-sided (~960 two-sided). Affordable
  given the queue depth (§6.4), just not at n=100.
- **Or restate as a coarse guard**: n=100 with a one-sided bound of
  "GREEN ≥ 70%", which n=100 *can* resolve, and treat the sliver check
  (§9.3c) as the real quality gate.

Prefer the second unless the queue makes 750 lots cheap. Also state
where 83.5% comes from — it is currently an unsourced number. If it
is corpus-wide, it is not the right comparison for a cng_feature-only
sample; recompute the baseline on the frozen §4.1 cng_feature lots.

---

## 6. Rollout

1. Land the kernel change behind `TRIVALAYA_CLOSE_KERNEL_FRAC` (env
   override, defaulting to current fixed-7 behavior) so it can be
   exercised without changing production defaults.
2. Run the §4 sweep using that override.
3. Flip the default once §5 passes.
4. **New lots only** at first — let the runner and nightly cron pick it
   up naturally. Do not backfill.
5. Backfill decision after ~2 weeks of new-lot data: the 5,196 existing
   cng_feature hough-split coins are candidates for reprocessing via
   `tools/reprocess_hough.py`, but only if §7 shows their current crops
   actually carry slivers. If Hough is producing clean crops today,
   backfill buys nothing but risk.

---

## 7. Failure modes

### 7.1 Slivers return (the important one)

The two-coin resolver exists *because* the naive path
(`_midpoint_binary_split`) cut merged blobs at the bbox midpoint, giving
each side a sliver of its neighbor's rim. If the kernel change stops the
weld but segmentation then yields 2 blobs whose bboxes still overlap, we
regress to slivers with none of Hough's correction.

Guard: §5.6 explicitly checks for neighbor-rim pixels in output crops,
detected mechanically per §9.3c — not by eyeballing 100 grids. Do not
accept on hough-rate and wall-clock alone.

### 7.2 Under-closing fragments small images

Coins with heavy glare or toning can threshold into pieces. On a 500px
image, k=3 may leave them fragmented where k=7 previously healed them —
turning a 2-coin lot into a 4-blob lot.

**This fails loudly only if you are watching the right signal.** v1 of
this spec claimed a fragmented lot "fails `should_trigger` (needs
exactly 1 candidate) and produces *no* detections rather than a bad
one." That is wrong, and the error is optimistic in a way that would
have hidden the regression:

- The 4 fragments each clear `MIN_AREA_PX = 300` (`config.py:17`) —
  on a 500×234 frame that is 0.26% of the image, trivially cleared.
  So they survive as 4 real candidates.
- `_suppress_contained` will not drop them: fragments of a coin are
  not contained *inside* a sibling blob.
- NMS will not merge them: non-overlapping fragments have IoU 0.
- The polarity-flip fallback (`layer1_geometry.py:424`) only fires
  when `candidates` is **empty** — 4 candidates is not empty, so it
  never triggers.
- The one filter that *can* eat fragments is
  `_suppress_background_noise` (`layer1_geometry.py:449`): it needs a
  dominant sibling (circularity > 0.85, edge support ≥ 0.70) and only
  drops candidates below 20% of its area that are also weak
  (circularity < 0.80, edge < 0.50). Half-coin fragments are too big
  to qualify; a small sliver next to one clean coin does qualify —
  and is silently dropped.

Net: the lot emits 3–4 detections when the fragments are comparable
in size, or **exactly 2** — one of them a fragment-sized crop — when
noise suppression eats the small pieces. Either way it is silent bad
output, not a loud skip.

Guard: §4.2 tracks fragment rate per kernel, not just weld rate. The
log signal is **`ndets > 2` plus the bbox-area distribution** (a
fragment crop is far smaller than its sibling coin), not
`status=skip` — a skip-count-only dashboard shows this failure as
all-clear, and an ndets-only dashboard misses the two-detection
variant too.

### 7.3 Silent behavior change on mid-size houses

leu (1200px, 125k coins) sits where k moves 7 → 3. It is by far the
largest house; a subtle crop shift there is a corpus-wide event that
would surface much later as embedding drift.

Guard: treat leu as a first-class pilot house in §4, not a control. If
its weld rate is low, consider raising `CLOSE_KERNEL_MIN` so 1200px
images retain k=5 or 7.

**leu needs a different acceptance bar than §4.3.** Its kernel moves
7 → 3 *by design*, so the bbox-identity test cannot apply — it would
fail by construction and tell us nothing. leu's bar is distributional
(§9.3e): ndets distribution unshifted, GREEN rate held. Stating this
explicitly because leu otherwise falls between §4.3's identity test
(which excludes it) and §4.1's weld-rate measurement (which does not
gate on quality), and would end up ungated despite being the largest
house in the corpus.

### 7.4 Downstream embedding invalidation

Any coin whose crop changes needs its embedding recomputed. New-lot-only
rollout (§6.4) avoids this; a backfill does not.

---

## 8. Code references

- `src/layer1_geometry.py:158-164` — the Otsu threshold + `MORPH_CLOSE`
  under change.
- `src/layer1_geometry.py:157` — `GaussianBlur((7,7))`, the *other*
  scale-absolute step in the same chain (§1).
- `src/layer1_geometry.py:424` — polarity-flip fallback; fires only on
  an empty candidate list, which is why §7.2's fragment case is silent.
- `src/layer1_geometry.py:449` — `_suppress_background_noise`; the one
  candidate filter that can silently eat small fragments, which is why
  `ndets > 2` alone is not a sufficient signal (§7.2).
- `src/layer1_geometry.py:492` — `_midpoint_binary_split`, the naive
  splitter whose slivers motivated the resolver (§7.1).
- `src/two_coin_resolver.py:61` — `should_trigger`; fires on exactly 1
  candidate with `1.50 ≤ ar ≤ 2.40` under `for_auction` (which also
  disables the circularity gate via `TRIGGER_CIRCULARITY_MAX=999.0`).
- `src/two_coin_resolver.py:183` — `_vectorized_hough`, the cost being
  avoided.
- `src/two_coin_resolver.py:304` — `_extract_crop`, radius ×1.15
  padding; one of two reasons output-bbox IoU == 0 is the wrong sliver
  test (§9.3c).
- `tools/extract_coins.py:93-98` — the 5% crop margin around the tight
  bounding rect; the other reason (§9.3c).
- `src/pipeline_manager.py:10` — `MAX_DIMENSION = 3200`; note 3000×1440
  `cng` raws are **not** downscaled, so L1 sees them at full size.
- `src/config.py:17` — `Standard.MIN_AREA_PX = 300`; the floor
  fragments must clear to survive as candidates (§7.2).
- `src/config.py:25-27` — `CLOSE_KERNEL_SIZE_STANDARD` /
  `CLOSE_KERNEL_SIZE_HIGH` / `CLOSE_ITERATIONS`, all currently dead;
  new `CLOSE_KERNEL_*` constants land here and the dead two go (§1).
- `src/config.py:175-185` — `validate_config()`. **Not** import-time:
  only the file's own `__main__` block calls it, and nothing imports
  it. This is why the `CLOSE_KERNEL_MAX`-is-odd assert goes at module
  level instead (§1).
- `src/layer1_helper.py:38` — a second `MORPH_CLOSE`. **Not on the live
  path** — only `experiments/test_helper1.py` imports it. Confirmed
  dead; listed so the next reader does not re-investigate it.
- `tools/reprocess_hough.py` — backfill path if §6.5 goes ahead.

---

## 9. Test plan

Three tiers, cheapest first. Tiers 1 and 2 gate tier 3 (§5.0) — there
is no reason to spend corpus time on a kernel whose arithmetic is
unverified.

**Infrastructure prerequisite:** there is currently **no test harness
in this repo** — no `tests/` tree, no pytest config, no fixtures
directory. What exists is ad-hoc scripts under `experiments/`
(`test_layer1.py` is four lines that print a result). Tiers 1 and 2
require standing up `tests/` + pytest first. Budget that; it is a
prerequisite, not a footnote. And it is slightly worse than "no
tests/": the repo venv is currently **broken** — its packages are
cp312 builds, but every `venv/bin/python*` symlink resolves to
`/usr/bin/python3`, which is now 3.13, so `import cv2` fails inside
the venv. Rebuild it against `/usr/bin/python3.12` (or reinstall
3.13-compatible wheels per `requirements-lock.txt`) before any pytest
work. Tier 3 can run as scripts under `tools/` in the existing style.

### 9.1 Tier 1 — kernel sizing (pure unit, no images, milliseconds)

Tests `_close_kernel_size(h, w)` directly. This tier exists because
v1's bug was pure arithmetic and would have been caught in under a
second.

| test | assertion |
|---|---|
| **width table** | table-driven over `[370, 500, 800, 1200, 1599, 1600, 1700, 1989, 2400, 2800, 3000, 3200]` with explicit expected k per §1's table — `1599 → 3` and `1600 → 5` pin the 3→5 band edge that v2's table misplaced at 1700. **Assert k == 7 across the entire 2400–3199 band** — this is the v1-bug regression test |
| **cng format pinned** | `_close_kernel_size(1440, 3000) == 7`, called out separately from the table so it cannot be silently edited away |
| **oddness** | k is odd for all w in 1..10000 |
| **clamp** | `MIN ≤ k ≤ MAX` for all w in 1..10000; and `CLOSE_KERNEL_MAX` is odd (guards the `k |= 1`-after-clamp ordering, §1) |
| **monotonic** | k is non-decreasing in `max(h, w)` |
| **orientation symmetry** | `k(1440, 3000) == k(3000, 1440)` — keys off `max`, so portrait must equal landscape |
| **floor not round** | `k(3000, 3000) == 7`, explicitly. `round()` yields 9 here; this test is the tripwire |

**Default-unchanged guard (most important test in this tier):** with
`TRIVALAYA_CLOSE_KERNEL_FRAC` unset, assert the effective kernel is
literally `(7, 7)` at every width in the table. §6.1 promises the
env-var default preserves production behavior; this makes that
promise *tested* rather than asserted, and it is what lets the change
land safely ahead of the sweep. It protects the landing window, not
the end state: when §6.3 flips the default it inverts **by design**,
and the flip commit updates it to assert the new scale-relative
default. Do not treat its post-flip failure as a regression to
revert.

### 9.2 Tier 2 — synthetic weld fixtures (deterministic, no network)

Generate images programmatically — two filled circles of radius r
separated by gap g, on a flat background, at scale S. No Spaces
access, no DB, runs in CI.

Sweep `g ∈ {3, 5, 7, 9, 12, 15, 25}` × `S ∈ {500, 1200, 3000}`:

- post-close contour count == 2 exactly where the bridging model
  predicts survival, == 1 where it predicts a weld
- the weld signature (`pre == 2 → post == 1`) appears **only** below
  the predicted threshold

This pins the px-bridging model that the entire spec rests on ("7×7
at 2 iterations ≈ 12px"). If that model is wrong, this fails in
seconds rather than after a 1800-lot sweep across nine houses.

**The model is already empirically pinned** (v3, OpenCV 4.12): on
synthetic circle pairs run through the exact blur → Otsu → close
chain, welding occurs iff `gap < 2 · iterations · (k // 2)` — at
k=7, iter=2 gaps ≤11px weld and 12px survives; at k=3, iter=2 gaps
≤3px weld and ≥4px survive. Two consequences for this tier:

- **Derive expected outcomes from that closed form**, not from a
  hardcoded results table. g=12 at k=7 sits exactly on the survive
  side of the boundary; a change to fixture radius or rendering must
  not silently flip it. A fixture disagreeing with the formula is the
  test failing loudly — which is the point.
- **Tier 2 cannot answer §1's blur-attribution question.** On clean
  synthetics the blur contributes *zero* bridging: it is symmetric,
  and Otsu re-thresholds the ramp back to the original boundary. Real
  images have shadows and gradients between coins; only §4.1's
  pre-blur/post-blur measurement on real lots can attribute bridging.
  Do not read a green tier 2 as evidence the close owns all of it.

**Fragment case (§7.2):** add glare/toning variants — punch a bright
band or a hole through one circle — in two sizes. Comparable-size
fragments: assert `ndets > 2` at k=3 on a 500px frame. Small-sliver
variant (one small fragment next to an intact circle): assert
`_suppress_background_noise` eats it and the lot emits `ndets == 2`
with one fragment-sized bbox — the assertion is on the **bbox-area
ratio** between the two detections, not the count. Both confirm
§7.2's corrected failure mode (silent bad output) rather than v1's
assumed clean skip, and the second pins the case an `ndets`-only
check misses.

**Run the §4.3 equivalence check here too**, on synthetic 3000px
inputs, so kernel-math regressions are caught in CI permanently
instead of only at the one-time manual gate.

### 9.3 Tier 3 — real lots

All sets **frozen to committed ID lists** (§4.1), so iterations are
comparable.

**a. Mechanism set** — the §4.1 n=200/house sample.

**b. Equivalence set** — §4.3's 100 lots, with the mandatory ≥20 at
3000×1440.

**c. Sliver detection — mechanical, not visual.** §7.1 calls this the
most important failure mode but v1 specified no way to detect it.
Two options, in preference order:

1. **Cross-check against Hough.** For lots where the *old* path
   (Hough split) and the *new* path (threshold-only) both yield 2
   coins, assert centers agree within a few px and radii within ~5%.
   Disagreement is the sliver signal. This is automatic over 100+
   lots and uses Hough — already trusted for this exact job — as the
   reference.
2. **Geometric fallback**, where Hough did not run. **Not**
   output-bbox IoU == 0 — v2 specified that, and it fails on
   *correct* output: extraction pads the tight bounding rect by 5%
   (`tools/extract_coins.py:93-98`) and resolver crops pad radius
   ×1.15 (`two_coin_resolver.py:304`), so on cng_feature (~11px of
   margin against a 5–8px gap) adjacent crops **always** overlap.
   Instead assert (a) the tight pre-margin `boundingRect`s of the two
   contours are near-disjoint (IoU < 0.02 — tight rects of disjoint
   contours are not guaranteed fully disjoint), and (b) the
   contour-level check, which is the real one: crops carry their coin
   contour in the alpha channel (`crop_with_alpha`), so each coin's
   fitted circle, dilated a few px, must not intersect the
   *neighbor's* alpha mask. Rim pixels inside the neighbor's mask are
   literally the sliver.

Option 1 is the real test; option 2 catches what it cannot cover.

**d. Dual-run A/B off the queue — the highest-value test available.**
Take the next **200 cng_feature lots from the queue** and run them
through both env-var states on identical inputs before flipping the
default. Compare hough rate, ndets distribution, wall-clock, and bbox
deltas.

This is the only way to get *paired* comparisons: historical crops
were overwritten in place at the same Spaces keys (§3), so the
pre-change output for already-processed lots no longer exists. The
queue backlog is the one source of unprocessed real data where both
arms can see the same input. It also front-loads §6.4's new-lots-only
rollout — the data arrives before the default flips, not after.

**e. leu distributional check** (per §7.3) — leu cannot use the
identity test. Bar: ndets distribution unshifted before/after, GREEN
rate held.

**f. Resolver-unchanged regression** (per §2's non-goal) — for lots
that *still* yield exactly 1 candidate after the change (genuinely
touching or overlapping coins), assert resolver output is unchanged —
but the bar depends on the band. In 2400–3199, where k stays 7, the
resolver's inputs (binary mask, candidate bbox) are bit-identical, so
its output must be **byte-identical** (this largely folds into §4.3).
On small formats byte-identity is unattainable *by construction*: at
k=3 the mask and bbox shift by a few px, so the resolver's crop
window and Hough accumulator shift numerically even though its code
is untouched. The bar there is tolerance-based — same split/failed
status, centers within ~2px at working scale, radii within ~2%. v2's
flat byte-identical bar would have flagged every small-format lot as
a regression and drowned the signal. Either way this separates "the
resolver fires less often" (the goal) from "the resolver behaves
differently" (a regression), which the hough-rate metric alone
conflates.

---

## 10. Revision history

### v3 — 2026-07-19

Verification pass: §9.1's assertions executed against the proposed
arithmetic (exhaustive sweep to w=10000), §9.2's bridging model run
against OpenCV 4.12 on synthetic pairs through the exact
blur → Otsu → close chain, and every code reference checked against
source. The model and all tier-1 expectations held. Six defects found
in the spec itself:

1. **§1 — band-edge error.** The 3→5 transition is at width **1600**
   (`int(1600/400) = 4`, `4 | 1 = 5`), not 1700. Table fixed; §9.1's
   width table now pins 1599/1600.
2. **§1 — the snippet was missing the §6.1 env gate**, so landing it
   as written would have changed production behavior on merge —
   contradicting the default-preserving rollout and §9.1's guard.
   Gate added at the call site; helper stays pure; guard documented
   as inverting by design at §6.3.
3. **§1/§8 — the "import-time asserts" don't exist.**
   `validate_config()` runs only under `__main__`; an assert there
   never fires in production. MAX-odd assert moved to module level.
4. **§7.2/§9.2 — `ndets > 2` is not a sufficient signal.**
   `_suppress_background_noise` (`layer1_geometry.py:449`) can eat a
   small fragment next to a dominant sibling, presenting as
   `ndets == 2` with a fragment-sized crop. Bbox-area distribution
   added to the guard and the tier-2 fragment tests.
5. **§9.3c — output-bbox IoU == 0 fails on correct output** (5% crop
   margin + ×1.15 resolver padding ⇒ adjacent crops always overlap
   on cng_feature). Replaced with tight-rect near-disjointness plus
   an alpha-mask/circle intersection check.
6. **§9.3f — flat byte-identity is unattainable on small formats**
   (mask/bbox shift at k=3 ⇒ numeric drift in the resolver). Byte
   bar restricted to the k-unchanged band; tolerance bar elsewhere.

Also recorded: welding empirically follows
`gap < 2 · iterations · (k // 2)` exactly on synthetics (k=7, i=2
welds ≤11px, survives at 12px; k=3, i=2 welds ≤3px, survives ≥4px);
the Gaussian blur contributes **zero** bridging on synthetic pairs
(Otsu restores the symmetric edge), so blur attribution (§4.1) is
answerable only on real images; and the repo venv is broken (cp312
packages under a 3.13 interpreter) and must be rebuilt before the §9
pytest prerequisite.

### v2 — 2026-07-19

Review pass against the source. Changes:

1. **§1 — corrected the kernel arithmetic (blocking).** v1's
   `int(round(...))` yields k=9 at 3000px, not the claimed k=7,
   silently changing behavior on the 42,080-coin `cng` house that §2
   promises to leave alone. Switched to floor. Added the per-width
   table and the `k |= 1`-after-clamp caveat.
2. **§1 — flagged `GaussianBlur((7,7))`** as a second scale-absolute
   step in the same chain, whose contribution is baked into the
   measured gaps. Added to the §4.2 sweep as a diagnostic.
3. **§1 — dead constants.** `CLOSE_KERNEL_SIZE_STANDARD/HIGH` and
   `CLOSE_ITERATIONS` exist in config and are never read; the
   `sensitivity` parameter they hang off is also dead. Wire one,
   delete two.
4. **§4.1 — froze the sample** to a committed ID CSV; added pre-blur
   gap measurement.
5. **§4.3 — mandated ≥20 lots at 3000×1440.** A naive "≥2400px"
   sample would have passed green on v1's broken kernel.
6. **§5 — GREEN-rate bar was unmeasurable** at n=100 (±7pp CI vs a
   3pp effect). Offered a sized alternative or a coarse bound; asked
   for the 83.5% baseline's provenance.
7. **§7.2 — corrected the failure mode.** v1 claimed fragmenting
   produces *no* detections; it actually produces 4 surviving
   candidates and silent bad crops. Log signal is `ndets > 2`, not
   `status=skip`.
8. **§7.1 / §9.3c — gave sliver detection an actual mechanism**
   (Hough cross-check, geometric fallback) instead of visual review.
9. **§7.3 — leu needs a distributional bar**, not §4.3's identity
   test, which excludes it by construction.
10. **§9 — new test plan**, three tiers, with the repo's missing test
    harness called out as a prerequisite.

Open questions carried into implementation: the 83.5% GREEN baseline's
source (§5.6); whether the blur or the close owns the bridging (§4.1);
and whether a rollback path is needed given §3's in-place-overwrite
warning — v1 has no rollback section, and the new-lots-only rollout
(§6.4) mitigates but does not cover "a week of new lots came out bad."
Recommend logging processed coin_ids during the flip window so they
can be re-run via `tools/reprocess_hough.py`.

### v1 — 2026-07-19

Initial spec: mechanism, measured evidence across 9 houses, proposed
scale-relative kernel, validation plan, rollout, failure modes.
