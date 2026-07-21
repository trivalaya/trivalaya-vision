# Two-Coin Weld: scale-relative MORPH_CLOSE

L1 segmentation closes its Otsu binary with a **fixed** 7×7 ellipse at 2
iterations. On low-resolution auction images this bridges the gap between
two correctly-separated coins, producing a merged blob that then costs a
full Hough two-coin split to undo. Make the kernel scale-relative so its
bridging distance is proportional to image size rather than absolute.

Owner: vision / L1 geometry
Status: **code landed behind `TRIVALAYA_CLOSE_KERNEL_FRAC` (§6.1); default
still fixed-7. §4 corpus validation NOT started — do not flip the default.**
Version: 4
Date: 2026-07-20 (v1–v4 — see §10)

### Implementation status

| item | state |
|---|---|
| §1 kernel helper + env gate + config constants | **done** |
| §1 dead-constant cleanup (`CLOSE_KERNEL_SIZE_*` deleted, `CLOSE_ITERATIONS` wired) | **done** |
| Per-house override mechanism (**deviation from §3** — see below) | **done, table empty** |
| §9 harness prerequisite (venv rebuild, `tests/` + pytest) | **done** |
| §9.1 tier 1 — kernel sizing | **done** (47 tests) |
| §9.2 tier 2 — synthetic weld fixtures | **done** (95 tests) |
| §4.1 frozen sample (cng_feature/Auction 91, n=200) | **done** 2026-07-20 — `specs/two_coin_weld_sample_ids.csv` |
| §4.3 frozen sample (2400–3199px band) | **not started** — draw from Spaces, not time-pressured |
| §9.3d A/B, cng_feature n=200 | **done** 2026-07-20 — see §4.5, results in `specs/results/` |
| §9.3e leu distributional check | **not started** — the house that matters most |
| §9.3c option 2b contour-level sliver check | **not started** — the definitive quality gate |
| Re-baseline of hough rates + GREEN against production (v4) | **done** 2026-07-20 |
| Re-baseline of wall-clock per coin | **done** 2026-07-20 — recovered from `created_at` spans |

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

Re-measured against production 2026-07-20 (corpus roughly doubled since
v1; rates held within ~2pp everywhere, confirming the mechanism is stable
and not an artifact of the earlier snapshot):

| house | raw width | hough-split coins | rate | v1 rate |
|---|---|---|---|---|
| cng_feature | 500 | 26,632 / 31,113 | **85.6%** | 85.4% |
| davissons | 370 | 370 / 729 | 50.8% | 50.7% |
| naumann | — | 22,586 / 48,476 | 46.6% | 46.9% |
| leu | 1200 | 106,850 / 256,170 | 41.7% | 41.9% |
| nomos | — | 10,024 / 27,592 | 36.3% | 36.3% |
| mashops | 1700 | 37,432 / 169,471 | 22.1% | 22.6% |
| gorny | 1200 | 4,832 / 23,969 | 20.2% | 20.1% |
| obolos | ~1989 | 9,674 / 83,714 | 11.6% | 11.8% |
| cng | 3000 | 7,092 / 111,572 | 6.4% | 8.4% |
| kuenker | 800–3000 | 32 / 4,781 | **0.7%** | 0.8% |

Note `cng` and `cng_feature` share a name and nothing else — 3000×1440
vs 500×234, 6.4% vs 85.6%. Do not reason about "CNG" as one thing.

### Rate is not cost: leu owns the Hough bill, not cng_feature

v1 ranked houses by hough *rate*, which put cng_feature at the top and
framed the whole spec around it. Ranked by *absolute* Hough splits — the
thing that actually costs CPU — the picture inverts:

| house | hough splits | share of all splits |
|---|---|---|
| **leu** | **106,850** | **47%** |
| mashops | 37,432 | 17% |
| cng_feature | 26,632 | 12% |
| naumann | 22,586 | 10% |
| everything else | ~32,000 | 14% |

leu does **4× more Hough splits than cng_feature** despite half its rate,
because it is 8× the corpus. cng_feature is only ~3.6% of all coins.

This matters for prioritisation and for risk. The largest available win
is on leu, which is exactly the house §7.3 flags as the riskiest: 1200px,
so k moves 7 → 3, and 256k coins means a subtle crop shift is a
corpus-wide embedding event. Confirmed empirically during implementation
— enabling the scale-relative path on real 1200px lots left `ndets`
unchanged but shifted bboxes on 4 of 5 sampled lots.

So the honest framing is not "fix cng_feature, leu is a bystander." It is
that the prize and the hazard are the *same house*, and §4.1 must treat
leu as the primary subject, not a control.

**Sample size caveat:** the gap measurements are 14 lots per house. That
is enough to establish the mechanism and rule out coincidence, not enough
to tune a threshold. §4 widens it before any code change lands.

---

## Observed cost

> **RE-MEASURED 2026-07-20 AND CONFIRMED CURRENT.** An earlier v4 draft
> marked these stale on the theory that `05ed5f7` (rim Hough ROI cap)
> would have lowered kuenker's per-coin time, since much of kuenker's
> 800–3000px range sits above the 1280px cap. **That theory was wrong and
> is retracted.** kuenker sale 428 ran at **0.522 s/coin** on 2026-07-19
> post-deploy, against 0.503–0.552 across its February pre-deploy sales.
> No movement. The reason is that kuenker's hough rate is 0.7% and rim
> recovery rarely fires on it, so the commit has almost nothing to bite
> on there. The 0.52 s/coin reference stands and §5.5 needs no
> re-derivation.

| | per lot | per coin | hough rate |
|---|---|---|---|
| kuenker nightly (`vision --batch 1000`) | 1.57 s | 0.52 s | 0.8% |
| cng_feature (runner job 299) | ~3.4 s | ~1.75 s | 85% |

Hough is not the whole 3.3× per-coin gap — CPU contention and
`validate_split` contribute — so treat elimination of the weld as
*necessary but not sufficient* for closing it. §5 sets the acceptance bar
on measured wall-clock, not on the hough-rate drop alone.

### Wall-clock, measured 2026-07-20

Recovered from `coin_detections.created_at` spans per sale — first coin to
last coin, divided by coins-1. No duration column exists, but a sale's
detection timestamps bracket its vision work closely enough to be useful:

| house | sale | coins | s/coin |
|---|---|---|---|
| kuenker | 428 (nightly, dedicated 05:00 window) | 3,003 | **0.522** |
| cng_feature | EA 455 | 1,454 | 1.231 |
| cng_feature | EA 342 | 2,106 | 1.271 |
| cng_feature | EA 145 | 868 | 1.364 |
| cng_feature | Triton XX | 3,391 | 1.358 |
| cng_feature | EA 286 | 1,128 | 1.657 |
| cng_feature | EA 453 | 1,743 | 1.757 |
| cng | EA-482 | 1,898 | **3.927** |
| cng | EA-484 | 2,477 | **4.099** |
| cng | EA-512 | 767 | **16.664** |

**cng_feature sits at ~1.23–1.76 s/coin (median ~1.35) against kuenker's
0.522 — a 2.6× gap.** §5.5's bar of "within 1.5×" means ~0.78 s/coin, so
the gap the spec set out to close is real, intact, and roughly as
described (v1 estimated ~1.75; the truth is a little better).

**But cng, not cng_feature, is the slowest house in the corpus** — 3.9 to
16.7 s/coin, 3–12× cng_feature — on a 6.4% hough rate. Whatever is
expensive there, it is *not* the weld. That is worth its own
investigation and is out of scope here; flagged so nobody reads
"cng_feature is the expensive house" out of this spec. It is the expensive
house *for Hough*, not in absolute per-coin cost.

**Caveat — these are not isolated measurements.** The runner interleaves
sales: `Adolph E. Cahn` ran 19:46–02:52 concurrently with the entire
cng_feature drain, and cng sales overlapped others. Spans therefore
include contention and overstate isolated cost, unevenly. kuenker's 0.522
is the cleanest number here (dedicated 05:00 cron window, `flock`-guarded,
scoped to one house). A true §5.5 comparison still wants both arms on an
otherwise-idle box — at time of writing the box was running a 280%-CPU
`append_search_annex` job alongside the runner.

Throughput from `created_at` is *not* a substitute: daily volume is
dominated by batch scheduling, not per-coin speed. cng_feature processed
2,449 coins on Jul 8, 2,281 on Jul 11, then 18,807 on Jul 19 — an 8×
swing across days whose hough rate barely moved (89.2% → 92.3% → 86.7%).
Reading that spike as "cng_feature got faster" would be a scheduling
artifact, not a performance change.

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

**Frozen 2026-07-20 — first 200 lots committed.** `cng_feature` /
`Auction 91`, `purpose=weld_ab`: 200 of the 1,233 lots that were pending
vision at freeze time, selected at evenly spaced indices over
lot_number-sorted order. **Deterministic with no seed** — re-running the
selection reproduces the file byte-identically. Real dimensions read from
the raw JPEGs: all 500 wide, heights 203–290 (median ~241), confirming
the "500 × ~240" figure in §"Measured evidence" now on n=200 rather
than n=14.

Columns are a superset of the schema above — `sale_id`, `lot_number` and
`purpose` were added because `lot_id` alone (the `auction_data` PK) is
opaque, and the file is intended to carry the §4.3 set too.

**Why these lots specifically.** They were *unprocessed*, and §9.3d's
paired A/B is only possible on unprocessed lots: production overwrites
crops in place at the same Spaces keys (§3), so for any already-processed
lot the pre-change output no longer exists. The runner was actively
draining the cng_feature backlog at freeze time (a sale every 15–145
minutes), so this list was captured before those lots were consumed.
Once the runner processes them the *raws* remain, but the paired-arm
opportunity is gone — re-freezing later gives a §4.1 mechanism sample,
not a §9.3d A/B sample.

**§4.3's equivalence set is not yet frozen, and is not urgent.** It needs
lots in the 2400–3199px band with ≥20 at exactly 3000×1440, and unlike
the above it does *not* require unprocessed lots — the test runs L1 twice
locally on the same input, so any cng raw works. `cng`/`EA-614` (1,007
lots pending) has not been downloaded yet, but cng raws for ~30 prior
sales are in Spaces under `raw/auctions/cng/<sale>/`. Draw it from there
whenever §4.3 runs.

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

**Partially run — leu only (§4.6, 2026-07-21):** k ∈ {3,5,7} at
`iterations=2` and blur `(7,7)`, n=200. The blur and iterations axes are
unswept, and no other house has been swept. Two corrections that any
remaining sweep must apply:

- **"fragment rate (`post > 2` where coins should be 2)" carries an
  unstated proviso** — it is only valid where every lot *is* 2-coin.
  cng_feature satisfies that; leu does not (53/200 lots have ≥3 blobs
  legitimately), and there the metric overstates fragmentation by 10×.
  Use `post > pre`, or condition on `blobs_pre_close == 2`.
- **Do not pick one global frac from a single house.** leu's optimum
  (k=5) differs from what the global formula assigns it (k=3), which is
  what `CLOSE_KERNEL_BY_HOUSE` is for.

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

## 4.5 First A/B results — 2026-07-20, n=200 cng_feature

Run with `tools/two_coin_weld_ab.py` over the frozen `Auction 91` sample,
both arms against identical decoded images in one process.
Arms: `control` (env unset ⇒ fixed 7×7) and `auto` (scale-relative ⇒ k=3
at 500px).

| metric | control (k=7) | auto (k=3) |
|---|---|---|
| hough rate | **97.5%** | **1.0%** |
| weld signature (`pre==2 → post==1`) | 90.5% | 1.0% |
| fragment rate (`post > 2`) | 0.0% | 0.5% (1 lot) |
| ndets | **2 on all 200** | **2 on all 200** |
| median s/lot | 6.71 | 0.013 |

**The mechanism is confirmed, decisively.** §5.1 wanted the weld signature
in >70% of cng_feature lots (90.5% ✓); §5.4 wanted hough 85% → <20%
(97.5% → 1.0% ✓).

**Otsu had already separated the coins in 200 of 200 lots.** Blob counts
before the close: 187 lots at exactly 2, 13 at 3–8, and **zero at 1**. The
close is not repairing anything here — it is destroying a correct
segmentation in every single lot. v1 claimed this from n=14; it holds at
n=200.

Gap distribution (min edge-to-edge, all contour points): min 5.0, p10 6.0,
**median 7.0**, p90 11.0, max 65. 183/200 fall under the 12px k=7 bridging
distance; **0/200** fall under the 4px k=3 distance. v1's "5–8 px (median
7)" from 14 lots was exactly right.

The 7pp gap between hough rate and weld rate is 14 lots where the close
merged *3–8* pre-existing blobs into 1 rather than exactly 2 — still the
close destroying a valid segmentation, just not matching the narrow
signature. Counting those, the real rate is ~97%.

### §4.1's blur-attribution question: answered — the blur owns nothing

`gap_pre_blur − gap_pre_close` across 200 real lots: **median 0.00 px**,
mean −0.28. The Gaussian blur contributes no measurable bridging on real
images, confirming on real data what §9.2 could only establish on
synthetics. **The close owns all of it**, so k=3 alone recovers the
separation and no blur change is needed. §1's "if the blur turns out to be
the dominant term" branch is closed.

(An interim smoke test on n=8 suggested ~2px of blur bridging. That was an
artifact of a subsampled gap metric, since fixed — see
`tools/two_coin_weld_ab.py::_min_gap`. Recorded because the wrong number
briefly looked like a real finding.)

### §9.3c sliver check: the new path is *cleaner* than Hough

Tight-rect IoU between the two detections in a lot — §9.3c's bar is
< 0.02:

| arm | median IoU | exactly 0 | under 0.02 |
|---|---|---|---|
| control (Hough) | 0.0079 | 43.5% | 67.0% |
| **auto (threshold-only)** | **0.0000** | **98.5%** | **99.0%** |

The threshold path produces near-disjoint detections in 99% of lots; the
Hough path manages 67%. §7.1 feared that removing the weld would regress
to slivers *with none of Hough's correction* — the geometry says the
opposite. Hough is the arm placing coins with overlap.

§9.3c option 1 (Hough-vs-threshold agreement) reads amber in isolation:
390 coin pairs, centre displacement median 5.70px / p90 12.4px, radii
agreeing to 2.2% median and within 5% on 88.7%. Radii match well; centres
do not. But the IoU table above locates the disagreement in the *Hough*
arm, so this should not be read as the new path being imprecise. Two
caveats on that comparison: it uses bbox centres as a proxy for fitted
circle centres, and the arms are structurally different quantities —
resolver crops pad radius ×1.15 (`two_coin_resolver.py:304`) while
threshold output is a tight contour rect.

**Still outstanding: §9.3c option 2b**, the contour-level check (each
coin's fitted circle, dilated, must not intersect the neighbour's alpha
mask). Tight-rect disjointness is necessary, not sufficient — extraction
adds a 5% margin on top. That check is the definitive one and has not been
run.

---

## 4.6 leu A/B and kernel sweep — 2026-07-21, n=200 leu

Run with `tools/two_coin_weld_ab.py` over 200 lots frozen from leu
`sale_id 75` ("Web Auction 42"), raws pulled from Spaces. Three arms
against identical decoded images in one process: `control` (7×7), `auto`
(scale-relative ⇒ k=3 at 1047–1200px), and `0.0042` (⇒ k=5 across the
whole observed width range, with margin on both sides).

| metric | control (k=7) | **0.0042 (k=5)** | auto (k=3) |
|---|---|---|---|
| hough rate | **60.0%** | **8.5%** | **0.0%** |
| weld signature (`pre==2 → post==1`) | 48.5% | 7.0% | 0.0% |
| fragmentation, true (`post > pre`) | 0.5% | **0.5%** | 1.5% |
| fragmentation, 2-blob cell | 0.0% | **0.0%** | 2.0% |
| ndets changed vs control | — | 4/200 | 4/200 |
| tight-rect IoU exactly 0 | 53.1% | 92.2% | 99.0% |

**The mechanism generalizes.** §4.1's falsifiable prediction offered two
branches — gaps like cng_feature (5–11px) or like kuenker (~25px). leu
came in at neither, and the middle turns out to be explanatory rather
than ambiguous.

### leu straddles its own welding threshold

Gap distribution, 2-blob cell (n=147): p10 9.00, **median 12.00**, p90
16.00. k=7's bridging reach is `2 × iterations × (k//2)` = **exactly
12px**. leu's gap distribution is centred on the threshold that decides
whether it welds.

That single fact explains the whole house:

- **Why the weld rate is 48.5%, not cng_feature's 90.5%** — only ~43% of
  leu's 2-blob lots have gaps under the k=7 reach. cng_feature's median
  of 7.0 sits far below its threshold, so nearly everything welds.
- **Why k=3 kills it completely** — a 4px reach clears all but one lot in
  the sample.
- **Why leu's Hough rate has a second cause.** Hough is 60.0% but the
  weld signature is 48.5%. Unlike cng_feature, where the 7pp gap was
  3–8-blob lots merging to 1 (still the close destroying a valid
  segmentation), leu's residual is not fully accounted for. Removing the
  weld does drive Hough to 0.0% at k=3, so the close is upstream of all
  of it, but the mechanism is not exclusively the narrow 2→1 signature.

Control Hough here is **60.0%**, against the 41.7% measured corpus-wide
in v4. Same direction, materially higher; this is one sale, and per §5's
kuenker worked example a single-sale rate is not a corpus comparison.

### The fragment-rate metric is misleading on any multi-coin house

`fragment_rate` as reported by the harness is `blobs_post_close > 2`.
On cng_feature that was unambiguous — every lot was 2-coin. **On leu it
is not**: 53/200 lots have ≥3 blobs *legitimately*, so the metric counts
real multi-coin lots as fragmentation. It reports 15.0% at k=3.

True fragmentation — `post > pre`, i.e. the close *splitting* something
it should have healed — is **1.5%**, and 2.0% restricted to the
2-blob cell. The headline figure overstates it by 10×. Any house sampled
without a coin-count filter needs the corrected metric; §4.2's "fragment
rate (`post > 2` where coins should be 2)" only holds under that proviso,
which leu does not satisfy.

The same trap applies to gap statistics: `_min_gap` measures the two
*largest* blobs, so on a 3+ blob lot it is not measuring a coin pair.
Segment by `blobs_pre_close` or the medians are not comparable across
houses. Re-scoring cng_feature this way gives a 2-blob cell of median
7.00 / p90 10.00 / 93.6% under 12px — marginally tighter than the
published 91.5% headline, and in the same direction.

### leu's operating point is k=5, not k=3

k=5 removes **86% of leu's Hough rate at zero fragmentation cost** — 0.5%
true-split and 0.0% in the 2-blob cell, identical to k=7 on both. k=3
buys the remaining 14% by tripling true fragmentation and introducing
2.0% splitting in the two-coin cell where k=5 and k=7 both have none.

That trade favours k=5. Fragmentation is a **correctness** risk — §7.2
establishes that fragments survive as real candidates and emit silent bad
crops, which across leu's 256k coins is embedding drift (§7.4). Residual
Hough is a **cost** issue. Paying correctness for cost is the wrong
direction, and the global formula assigning leu k=3 is precisely the case
§"per-house override" anticipated.

**This is the measured sweep data `CLOSE_KERNEL_BY_HOUSE` requires.** The
table remains empty pending an explicit decision to populate it; it is
recorded here, not shipped. Note also that the k=3 and k=5 arms shift
ndets on the *same* 4 lots (+1:3, +2:1), so §9.3e's primary bar passes
identically for both — the choice between them rests on fragmentation, not
on detection count.

### The bridging formula is ~1–2px optimistic

`2 × iterations × (k//2)` under-predicts welding in both arms:

| arm | predicted reach | predicted weld | observed weld |
|---|---|---|---|
| control (k=7) | 12px | 41.0% | 48.5% |
| 0.0042 (k=5) | 8px | — | 7.0% |

The 14 lots still welding at k=5 have gaps of **7.0–9.8px** against a
predicted 8px reach. The bias is consistent and one-directional, most
likely diagonal connectivity in the ellipse kernel — the formula assumes
axis-aligned bridging. No conclusion in this spec changes, but the
formula should be treated as a lower bound on reach, not an equality.
§9.2's synthetic fixtures pin the model as exact; they pass because
synthetic gaps are axis-aligned.

### Determinism verified

The k=5 run re-ran `control` over the same frozen sample. It reproduces
the first leu run **byte-identically across all 21 structural columns on
all 200 lots**. The harness is deterministic over a frozen sample, which
was previously assumed for the A/B (the freezer's determinism was
established; the A/B's was not). Both leu results are therefore
like-for-like, and a future re-run after a kernel tweak is a valid
comparison.

### Still open after this

- **§4.3 equivalence** — unrun, no cng lots sampled.
- **§9.3c option 2b**, the contour-level sliver check — unrun, and still
  the definitive quality gate.
- **Wall-clock** — both leu runs were taken at load 1.6–1.9 with the
  production runner active. Structural columns are unaffected; the timing
  columns in both JSONs are unusable. §5.5 still wants an idle box.

### What this does and does not license

Does **not** license flipping the default. Three gaps remain:

1. ~~**leu is untested**~~ — **closed by §4.6.** leu is measured at
   n=200 and the mechanism generalizes: Hough 60.0% → 0.0% at k=3. But it
   closed with a condition attached rather than cleanly. leu's gaps
   straddle the k=7 threshold (median 12.00px against a 12px reach), and
   the sweep shows **k=5, not the k=3 the global formula assigns**, is its
   operating point — k=3 raises true fragmentation 0.5% → 1.5% for a
   Hough gain k=5 mostly already captures. Flipping the default sends leu
   to k=3. So this gap is now a reason to populate
   `CLOSE_KERNEL_BY_HOUSE` before the flip, not a reason the flip is
   blocked outright.
2. **§4.3 equivalence is unrun** — no cng lots sampled yet.
3. **Wall-clock is unmeasured** in the §5.5 sense. The 6.71 → 0.013 s/lot
   figures were taken at load average 7.6 with the production runner and
   another job active; the ratio is too large to be explained by
   contention, but the absolute numbers are not usable.
---

## 4.7 Mask-IoU gate and §9.3c option 2b — 2026-07-21

Run with `tools/two_coin_weld_mask_gate.py`, which runs both arms on one
decoded image like the A/B harness but measures the **masks** rather than
the summary columns. Arms are `control` (fixed 7×7) and `auto` (now
resolving through the populated `CLOSE_KERNEL_BY_HOUSE`, so k=3 on
cng_feature and k=5 on leu). Every column is structural, so box load does
not invalidate it.

### The gated quantity is the alpha mask, not the segmentation mask

The tool measures two masks per arm, and keeping them apart is what makes
the result readable:

- **binary** — the post-close segmentation mask. Differs between arms *by
  construction*, since the kernel differs.
- **alpha** — the union of filled detection contours. This is literally
  what `pipeline_manager.crop_with_alpha` writes into the alpha channel and
  therefore what reaches the embedding (§7.4).

On cng_feature's unchanged-outcome lots the binary IoU is **0.864/0.878**
while the alpha IoU is **exactly 1.000**. The segmentation mask moves
substantially; the crop alpha does not move at all. Gating on the binary
mask would have read as serious drift and been wrong.

### cng_feature, n=200 — the gate passes but has almost no population

Only **2 of 200** lots have an unchanged detection outcome, because the
weld fires on the rest. Both show alpha IoU 1.000, so the gate passes, but
on a sample too small to mean much. The gate needs leu.

### §9.3c option 2b — run for the first time

§4.5 called this "the definitive one and has not been run". Result on
cng_feature, **undilated** — the contamination form, i.e. do the filled
contours actually overlap:

| arm | lots with contour overlap | worst |
|---|---|---|
| control (Hough) | **0 / 200** | — |
| auto (k=3) | **1 / 200** | 580px, 1.1% of the neighbour |

Strictly this fails §5's "zero lots where a crop gains a sliver" bar, at
0.5% incidence. But the cause is not the kernel.

### The one sliver is rim recovery, not the close

Lot 215298's two detections both carry `rim_recovered=True`. Layer 1.5
replaces the true contour with a HoughCircles-fitted circle
(`layer1_geometry.py:271`), and on a near-touching pair the two synthesised
circles overlap. The other two lots that show overlap only under dilation
have `rim_recovered=False` and are clean undilated.

So this is a **pre-existing Layer 1.5 behaviour that the change exposes**,
not a defect the change introduces — and it is the same failure class as
Hough's: a fitted circle does not respect its neighbour. It answers §7.1
("slivers return") directly: they do, at 0.5%, from rim recovery. A fix
belongs in `validate_rim_recovery`, not in the kernel.

### Dilation measures proximity, not contamination — do not compare it across arms

§9.3c words the check as "dilated a few px". At d=3 the result inverts:
auto shows 4 lots with contour overlap and control 0. That is an artifact.
A 3px dilation bridges any sub-3px gap, so an arm whose masks track the
true flan edge scores *worse* than one whose masks sit inside it — and
that is exactly the control/auto difference. Hough fits a circle that
under-covers an irregular ancient flan, leaving slack the dilation does not
cross; the threshold contour hugs the real outline, so it does. The
montages show this plainly on chipped and oval coins.

The tool therefore records d=0 and d=3 separately and gates on d=0. Read
d=3 as a proximity diagnostic only.

---

## 5. Acceptance criteria

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
   kuenker's per-coin time on an otherwise-idle box. **Both terms must be
   re-measured first** — the 0.52 s/coin reference predates `05ed5f7` and
   is stale (see §"Observed cost"). Measure kuenker and cng_feature in the
   same session, on the same box, or this criterion is meaningless.
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

Prefer the second unless the queue makes 750 lots cheap.

**Measured 2026-07-20 — the corpus-wide GREEN number is worse than
unsourced, it is unusable.** Daily GREEN rate across all houses swings
between **55.9% and 87.9%** over the last twelve days, driven entirely by
which house and sale the batch happened to be working:

| day | coins | GREEN |
|---|---|---|
| 2026-07-20 | 4,092 | 87.9% |
| 2026-07-19 | 27,325 | 85.2% |
| 2026-07-16 | 5,386 | 74.3% |
| 2026-07-12 | 6,830 | 63.7% |
| 2026-07-10 | 5,913 | **55.9%** |

A 32pp day-to-day range makes any corpus-wide GREEN baseline — 83.5% or
otherwise — meaningless as an acceptance reference. Per-house it is
stable enough to use, but only *within* a house and ideally within a sale:

| house | GREEN |
|---|---|
| leu | 90.8% |
| cng_feature | 84–88% (84.4 / 87.1 / 85.6 / 87.9 across four batch days) |
| cng | 80–82% |

So: **drop the corpus-wide GREEN bar entirely.** Use per-house GREEN on
the frozen §4.1 lots, compare like-for-like, and let §9.3c's sliver check
be the real quality gate as this section already recommends.

A worked example of the trap: kuenker looks like it fell from 97.0% GREEN
to 78.0% in the last 7 days — a 19pp collapse that reads as a serious
regression. It is not. kuenker has been processed in exactly two epochs:
sales 72/89/232 in February (1,778 coins, 95.8–98.3%) and sale **428** on
2026-07-19 (3,003 coins, 78.0%). It is a different sale five months
later, not a code change. Any before/after GREEN comparison that does not
hold the sale fixed will manufacture regressions like this one.

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

### 6.6 BLOCKER — `house` never reaches Layer 1 in production

**Populating `CLOSE_KERNEL_BY_HOUSE` does nothing in production today, and
enabling `auto` without fixing this regresses leu to k=3.** Found
2026-07-21 while executing the rollout.

The vision side is fully plumbed. `pipeline_manager.analyze_image` takes
`house` and forwards it to `layer_1_structural_salience`, which forwards it
to `_segment_and_extract_candidates` and `_close_kernel_size`. The
**pipeline side never supplies it**:

| where | what it does |
|---|---|
| `trivalaya_pipeline/pipeline.py:606` | `self.vision.process_image(local_path, source_type="auction")` — no house |
| `vision_adapter.py::process_image` | no `house` parameter at all |
| `vision_adapter.py::_run_vision_pipeline:198` | `self._analyze_image(str(image_path), source_type=source_type)` — no house |

`auction_house` appears nowhere in `vision_adapter.py`. So `house=None`
reaches `_close_kernel_size`, the override lookup is skipped, and every
house gets the global formula.

**Why that is worse than a no-op.** With the table unreachable, setting
`TRIVALAYA_CLOSE_KERNEL_FRAC=auto` gives every house k by width alone.
Sampling leu's welded lots from Spaces gives 1160–1200px ⇒ **k=3 on 32/32**
— precisely the regression §4.6 rejected, at 0.5% → 1.5% true
fragmentation across 256k coins. The per-house table cannot prevent it
because it is never consulted.

**The fix** is three edits in trivalaya-pipeline, and the value is already
in scope — `record.auction_house` is used twenty lines below the call site,
at `pipeline.py:625`:

1. `pipeline.py:606` — pass `house=record.auction_house`
2. `vision_adapter.py::process_image` — accept `house: Optional[str] = None`, forward it
3. `vision_adapter.py::_run_vision_pipeline` — accept it, pass to `self._analyze_image(..., house=house)`

Not applied here: it is a cross-repo production change outside the rollout
brief, and that repo had a live `append_search_annex` job writing to its
working tree at the time.

**Where "enable in production config" would land, once unblocked.** Both
vision paths source the same file, so one line covers them:
`/home/claudeuser/trivalaya-pipeline/.env`, injected into
`trivalaya-runner.service` via `EnvironmentFile=` and into the 05:00 UTC
`vision_nightly_batch.sh` via `set -a; source`. Restarting
`trivalaya-runner.service` at a job boundary picks it up; the nightly batch
picks it up on its next fire. kuenker — the nightly batch's current scope —
is pinned to k=7, so that path is a no-op by construction.

### 6.7 The table covers 60% of the welded population; `auto` decides the rest

A second finding from the same session, and the larger risk. The historical
census (`specs/two_coin_weld_reprocess_proposal.md`) counts **228,450
Hough-split detections of 779,165 (29.3%)**. cng_feature, leu and kuenker
together hold 136,062 of them — **59.6%**. The remaining 40% sits in houses
with no A/B, no sweep and no entry, which `auto` would move by width alone:

| house | Hough-split | `auto` would give |
|---|---:|---|
| mashops | 37,432 | k=3 (84%), k=5 (16%) |
| naumann | 22,586 | k=3 (100%) |
| nomos | 10,024 | k=3 (100%) |
| obolos | 9,674 | k=3 (97%) |
| cng | 7,092 | k=3 (53%), k=5 (9%), k=7 (38%) |
| gorny | 4,832 | k=3 (88%) |
| stacksbowers | 208 | **k=11 (75%)** |

**~85,000 welded detections would land on k=3** — the setting §4.6 examined
most closely and *rejected* for leu on fragmentation grounds. And
stacksbowers (4736px) would get k=11, kuenker's largest plates k=9: 20px and
16px of bridging reach against today's 12px. That is the scale-relative
formula working as designed on 4000px+ input, not a bug, but it is an
unmeasured change in the *welding* direction on houses that were never the
problem.

Enabling `auto` corpus-wide is therefore a materially bigger change than
"populate three measured houses" implies. Two ways to bound it, both cheap
relative to the blast radius: gate the scale-relative path on
table-membership so unlisted houses keep k=7 until measured, or A/B the
three largest unmeasured houses (mashops, naumann, nomos — 70,042 welded
detections between them) before the flip.

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

**The guard metric itself needs conditioning (§4.6).** `ndets > 2` and
`post > 2` both read a legitimately-multi-coin lot as a fragmented one.
On leu that inflates the measured fragment rate from 1.5% to 15.0% —
large enough to fail a rollout gate on houses that are behaving
correctly. Compare `post` against `pre` per lot, or restrict to lots
where `blobs_pre_close == 2`. This failure mode is real on leu (true
fragmentation does rise 0.5% → 1.5% at k=3, and 0.0% → 2.0% in the
2-coin cell) — it is the *magnitude* the naive metric gets wrong, and
it is why leu's operating point is k=5.

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

### v5 — 2026-07-21

leu measured at n=200 (sale_id 75) plus a k ∈ {3,5,7} sweep. Read-only;
default not flipped, `CLOSE_KERNEL_BY_HOUSE` still empty. See §4.6.

1. **The mechanism generalizes to leu** — Hough 60.0% → 0.0%, weld
   signature 48.5% → 0.0% at k=3, ndets essentially unshifted (196/200),
   tight-rect IoU 53.1% → 99.0% exactly-disjoint. v4's item 2 identified
   leu as the house the decision turns on; it turns the right way.
2. **§4.1's binary prediction was the wrong shape.** leu's gaps came in
   between the two offered branches — 2-blob median **12.00px** against
   cng_feature's 7.00 and kuenker's ~25 — because leu sits *on* its own
   welding threshold (k=7 reach is exactly 12px). This explains its 48.5%
   weld rate against cng_feature's 90.5% rather than leaving it ambiguous.
3. **leu wants k=5, not k=3.** The sweep shows k=5 removing 86% of the
   Hough rate at zero fragmentation cost, where k=3 triples true
   fragmentation for the last 14%. This is the first measured data
   qualifying under §"per-house override"; recorded, not shipped.
4. **`fragment_rate` is invalid on multi-coin houses.** `post > 2` counts
   leu's 53/200 legitimately-multi-coin lots as fragmentation, reporting
   15.0% where true splitting (`post > pre`) is 1.5%. The same segmenting
   error inflates gap medians. Both metrics need conditioning on
   `blobs_pre_close`; §4.2's definition carries an unstated 2-coin proviso.
5. **The bridging formula `2 × iters × (k//2)` is ~1–2px optimistic** and
   should be read as a lower bound. Lots weld at gaps of 7.0–9.8px against
   a predicted 8px reach at k=5, and k=7 predicts 41.0% weld against 48.5%
   observed. §9.2's synthetics miss this because they are axis-aligned.
6. **The A/B harness is deterministic over a frozen sample** — verified,
   not assumed, by re-running `control` and diffing all 21 structural
   columns across 200 lots. Previously only the freezer's determinism was
   established.
7. `tools/two_coin_weld_ab.py` gained a Spaces source (`--source`,
   `--cache-dir`), the tooling gap the handoff identified. leu raws are
   not on local disk.

### v4 — 2026-07-20

Re-baseline against production after `05ed5f7` (rim-recovery Hough ROI
cap) reached origin. Read-only DB measurement; no writes.

1. **Hough rates re-measured** on a corpus that has roughly doubled since
   v1. Every house held within ~2pp — the mechanism is stable, not a
   snapshot artifact. cng_feature is **85.6%**, essentially identical to
   v1's 85.4%: **the weld problem is entirely intact and nothing shipped
   so far has touched it.**
2. **Rate ≠ cost — the spec was aimed at the wrong house.** Ranked by
   absolute Hough splits, **leu owns 47%** (106,850) against
   cng_feature's 12% (26,632). leu does 4× the Hough work at half the
   rate, because it is 8× the corpus; cng_feature is only ~3.6% of all
   coins. The biggest win and the biggest risk (§7.3, k 7→3 on 256k
   coins) are the same house.
3. **Wall-clock recovered from `created_at` spans — and an intermediate
   v4 claim retracted.** This section first marked §"Observed cost" STALE,
   predicting `05ed5f7` had lowered kuenker's per-coin time. It has not:
   kuenker ran **0.522 s/coin** post-deploy against 0.503–0.552
   pre-deploy. The commit has little to bite on there (0.7% hough rate,
   rim recovery rarely fires). The 0.52 reference stands. cng_feature
   measures ~1.23–1.76 s/coin, a **2.6× gap** — the problem is real and
   close to v1's estimate.
4. **cng is the slowest house in the corpus, and it is not the weld.**
   3.9–16.7 s/coin at a 6.4% hough rate, 3–12× cng_feature. Recorded so
   this spec is not misread as identifying cng_feature as the expensive
   house in absolute terms. Separate investigation.
5. **Throughput is not a proxy for speed.** cng_feature ran 2,449 /
   2,281 / 18,807 coins on Jul 8 / 11 / 19 with hough rate flat at
   89.2 / 92.3 / 86.7%. The 8× volume swing is batch scheduling. Recorded
   because that spike is exactly the kind of thing that gets misread as a
   performance win.
6. **Corpus-wide GREEN bar dropped as unusable.** Daily GREEN ranges
   55.9–87.9% across twelve days depending on which house is running — a
   32pp swing that swamps any effect being tested. Replaced with
   per-house comparison on frozen lots.
7. **Documented the sale-confounding trap.** kuenker appears to fall
   97.0% → 78.0% GREEN in 7 days; it is actually February sales
   72/89/232 versus July sale 428. Not a regression. Any before/after
   that does not hold the sale fixed will invent one.

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
