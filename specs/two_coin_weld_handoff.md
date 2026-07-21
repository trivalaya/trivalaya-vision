# Handoff: run the leu A/B for the two-coin weld change

> **DONE — 2026-07-21. Results in §4.6 of the spec; read that, not this.**
>
> The mechanism generalizes: leu Hough 60.0% → 0.0% at k=3, n=200 on
> sale_id 75. The §4.1 prediction below offered 5–11px or ~25px; leu came
> in at a median of **12.00px**, sitting exactly on k=7's 12px bridging
> threshold, which explains its 48.5% weld rate rather than leaving the
> result ambiguous. A follow-on k ∈ {3,5,7} sweep shows **k=5 is leu's
> operating point, not k=3** — same Hough win, none of the fragmentation.
>
> Two things below are now known to be wrong or incomplete:
> - "leu's 41.7% Hough rate" — this sample measures **60.0%**. Single
>   sale; not a corpus comparison (gotcha #4 applies to it too).
> - Step 4's fragment-rate bar uses `post > 2`, which is invalid on a
>   multi-coin house and overstates leu's fragmentation by 10×.
>
> Retained as the record of what was asked and the environment notes,
> which are still accurate.

Companion to `specs/two_coin_weld_morph_close.md` (v4). Read this first,
then §4.5 and §7.3 of the spec. Written 2026-07-20.

---

## The job, in one paragraph

Freeze ~200 leu lots and run `tools/two_coin_weld_ab.py` against them, the
same way it was run for cng_feature. leu is 1200px, so the scale-relative
kernel moves it from k=7 to k=3. Everything measured so far is a 500px
house, and **leu is the house the whole decision turns on** — it owns 47%
of all Hough splits in the corpus (106,850, against cng_feature's 26,632),
it is 256k coins, and it is where the change is simultaneously worth the
most and riskiest.

---

## Why this run decides the project

§4.1 states the falsifiable prediction. Two outcomes, pointing opposite
ways:

- **leu's inter-coin gaps come in around 5–11px**, like cng_feature → the
  same weld mechanism is at work → k=3 collapses leu's 41.7% hough rate →
  this is the largest available win in the corpus, and the change is worth
  finishing.
- **leu's gaps come in around 25px**, like kuenker → the close is *not*
  welding leu → its 41.7% hough rate has some other cause → the change
  buys leu nothing, and the entire value case shrinks to cng_feature,
  which is **3.6% of all coins**. At that point seriously consider
  stopping: cng_feature's crops are already fine (GREEN 84–88%), so the
  change would be a cost optimisation on a small slice.

There is currently no evidence either way. Do not assume the first.

---

## State of play

Everything below is committed and pushed to `main`.

| | where |
|---|---|
| Kernel change (env-gated, **inert** in production) | `src/layer1_geometry.py::_close_kernel_size` |
| Per-house override table (**empty** by design) | `src/config.py::Layer1Config.CLOSE_KERNEL_BY_HOUSE` |
| Test suite, 201 tests | `tests/` |
| A/B harness | `tools/two_coin_weld_ab.py` |
| Sample freezer | `tools/freeze_weld_sample.py` |
| Frozen cng_feature sample (n=200) | `specs/two_coin_weld_sample_ids.csv` |
| cng_feature A/B results | `specs/results/two_coin_weld_ab_20260720.*` |

**Production is untouched.** `TRIVALAYA_CLOSE_KERNEL_FRAC` is set nowhere,
so L1 runs a literal 7×7 exactly as before. This was verified
byte-identical against pre-change code on real lots, and a test pins the
kernel reaching OpenCV by spying on `cv2.morphologyEx`.

### cng_feature result, for comparison (n=200)

| metric | control (k=7) | auto (k=3) |
|---|---|---|
| hough rate | 97.5% | 1.0% |
| weld signature | 90.5% | 1.0% |
| fragment rate | 0.0% | 0.5% |
| ndets | 2 on all 200 | 2 on all 200 |
| gap median | 7.0px | — |
| tight-rect IoU = 0 | 43.5% | 98.5% |

This is the shape of a "mechanism confirmed" result. leu's numbers will
either look like this or nothing like it.

---

## Environment — the non-obvious parts

**Use the pipeline venv for this work:**

```
~/trivalaya-pipeline/.venv/bin/python     # py3.13 — cv2 4.13, boto3, PIL, mysql.connector, pytest
~/trivalaya-vision/venv/bin/python        # py3.12 — cv2 4.12, numpy, pytest ONLY
```

The vision venv has **no boto3**, and leu raws are only in Spaces, so the
pipeline venv is the one that can do this job. Verified safe to use:

- all 201 tests pass under both venvs
- the A/B harness returns identical hough/weld/fragment/ndets/kernel
  results under cv2 4.12 and 4.13 on the same lots

(The vision venv was broken — cp312 packages under a 3.13 interpreter —
and was fixed by repointing `venv/bin/python3` at `/usr/bin/python3.12`.
`venv/` is gitignored, so that fix is local to this box and any fresh
clone will hit it again.)

**Data access** (read-only; never write):

- DB is **MySQL**, not Postgres (`ORDER BY RAND()`, `mysql.connector`).
  Credentials in `~/trivalaya-pipeline/.env`. Note `tools/reprocess_hough.py::get_db`
  hardcodes `/root/trivalaya-pipeline/.env`, which is wrong for this box —
  use `~/trivalaya-pipeline/.env`.
- Spaces credentials in `~/spaces.env`; bucket `trivalaya-data`; raws at
  `raw/auctions/<house>/<sale_id>/Lot_<5-digit>.jpg`.
- Tables: `auction_data` (one row per lot; `auction_house`, `sale_id`,
  `vision_processed`, `image_path`) joined to `coin_detections` via
  `coin_detections.auction_record_id = auction_data.id`.

---

## Steps

### 1. Freeze the sample

leu raws are **not on local disk** — `--source spaces` is required.

```bash
cd ~/trivalaya-vision
~/trivalaya-pipeline/.venv/bin/python tools/freeze_weld_sample.py \
    --house leu --sale 75 --purpose leu_ab --n 200 --source spaces
```

`sale_id 75` is leu's **"Web Auction 42"**; `sale_id 74` is **"Web Auction
43"**. The display name and the internal id are unrelated numbers — that
is the auction house's own storage convention, not something to fix. Key
every query off `sale_id`. Both sales are fully ingested and
vision-processed (3,754 and 1,960 lots). Sampling from both is reasonable;
run the freezer twice with the same `--purpose` to accumulate.

Verified working: a 5-lot Spaces smoke returned widths 1193–1200, heights
545–600. Note 1193 and 1200 both yield k=3, so there is no band-edge
hazard in this sale.

Selection is deterministic with no seed (evenly spaced over lot_number).
Re-running reproduces the file byte-identically — confirmed by
regenerating the committed cng_feature rows exactly. Commit the CSV.

### 2. Teach the A/B harness to read from Spaces

**This is the one real gap.** `tools/two_coin_weld_ab.py::_load` is
local-disk only:

```python
p = raw_root / row["house"] / row["sale_id"] / f"Lot_{row['lot_number']}.jpg"
```

It needs a Spaces fallback with a local cache — roughly 20 lines, mirroring
`freeze_weld_sample.py::_s3` and the key format above. Cache to a temp dir
so a re-run does not re-download. Note the freezer zero-pads to 5 digits
(`Lot_00001.jpg`) while `_load` does not pad; **leu keys are padded**, so
copy the freezer's format.

Add `--source {local,spaces}` to match the freezer rather than guessing.

### 3. Run it

```bash
~/trivalaya-pipeline/.venv/bin/python tools/two_coin_weld_ab.py \
    --purpose leu_ab --source spaces --out specs/results/two_coin_weld_ab_leu_<date>
```

Expect 30–45 min for 200 lots: 1200px images are ~6× the pixels of
cng_feature, offset by a lower hough rate (41.7% vs 97.5%). It prints
progress every 25 lots.

**Check the load average first.** The harness warns when it is high. The
cng_feature run was taken at load 7.6 with the production runner plus an
`append_search_annex` job active, which made the timing columns unusable —
structural columns are fine under load, timings are not. §5.5 wants an
otherwise-idle box.

### 4. Interpret against leu's own bar

leu **cannot** use §4.3's bbox-identity test — k moves 7→3 there by design,
so identity fails by construction and tells you nothing. §9.3e's bar is
distributional:

- **ndets distribution unshifted** between arms (this is the primary bar)
- gap distribution — the falsifiable prediction above
- fragment rate at k=3 must not climb (§7.2; cng_feature was 0.5%)
- tight-rect IoU should stay near-disjoint as it did on cng_feature

Already observed and worth confirming at scale: on 5 real 1200px lots from
`data/test_images`, enabling the scale-relative path left ndets unchanged
at 2 but **shifted bboxes on 4 of 5**. Small crop shifts across 256k coins
mean embedding drift, which surfaces much later (§7.4).

---

## Gotchas that already cost time

1. **`cng` and `cng_feature` are unrelated houses** — 3000×1440 vs
   500×234, 6.4% vs 85.6% hough. Never aggregate "CNG".
2. **Do not subsample contours when measuring gaps.** An early
   `_min_gap` sampled ~200 points and reported 12.0px for a lot whose true
   minimum is 11.0px — which reads as the bridging model failing (12
   survives at k=7) when it is the metric missing the closest point. The
   model is right; measure every point. Already fixed in the harness.
3. **Throughput ≠ speed.** cng_feature ran 2,449 / 2,281 / 18,807 coins on
   Jul 8 / 11 / 19 with hough rate flat. That 8× swing is batch
   scheduling. Do not read a queue drain as a performance win.
4. **Never compare GREEN rates across sales.** kuenker appears to fall
   97.0% → 78.0% in 7 days; that is February sales 72/89/232 versus July
   sale 428, not a regression. Corpus-wide daily GREEN swings 55.9–87.9%
   depending on which house is running. Compare within a house, ideally
   within a sale.
5. **`pgrep -f <script>` matches your own shell** and will kill the
   launching process. Killed a run this way.
6. **Do not name a scratch file `queue.py`** in a directory you then run
   from — `mysql.connector` imports stdlib `queue` and gets yours instead.

---

## Hard constraints

- **Read-only.** No DB writes, no Spaces uploads. Crops are overwritten in
  place at the same Spaces keys (§3), so a bad write is not trivially
  reversible.
- **Do not flip the default.** §4.3 equivalence is unrun, §9.3c option 2b
  (the contour-level alpha-mask sliver check — the definitive quality
  gate) is unrun, and leu is unmeasured. That is the whole point of this
  task.
- **Do not populate `CLOSE_KERNEL_BY_HOUSE`** from anything but measured
  sweep data. It ships empty deliberately; guessing a constant there is
  exactly the failure mode that produced the original `round()` bug.
- **Do not backfill.** Its precondition was crops carrying slivers, and
  that is now measured false — Hough crops are fine.

---

## If leu confirms the mechanism

Next in priority order, none of it started:

1. §9.3c option 2b — contour-level sliver check (definitive quality gate)
2. §4.3 equivalence — 100 lots in the 2400–3199px band, **≥20 at exactly
   3000×1440**; raws are in Spaces under `raw/auctions/cng/<sale>/`, and
   unlike the A/B this does not need unprocessed lots
3. §5.5 wall-clock on an idle box, both arms in one session
4. The cross-repo change so `house` reaches L1 —
   `trivalaya_pipeline/pipeline.py:606` passing `record.auction_house`,
   forwarded through `VisionAdapter.process_image` /
   `_run_vision_pipeline` (`vision_adapter.py:171,194,198`)

Separately and out of scope: **`cng` is the slowest house in the corpus**
at 3.9–16.7 s/coin on a 6.4% hough rate, 3–12× cng_feature. Whatever costs
that much there is not the weld. Worth its own investigation.
