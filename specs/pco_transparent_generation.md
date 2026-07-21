# PCO Transparent Generation

Add alpha-masked PNGs for the `collection_id='pco'` (Personal Coin Owner)
subset of `collection_data` so they embed via the same masked path as
auction crops instead of falling through to the rectangular fallback.

Owner: vision / pipeline plumbing
Status: spec — not yet implemented
Date: 2026-05-27

---

## Why

Today's embedding pipeline (`trivalaya-pipeline/cluster_coins.py:open_image_masked`,
line 697) loads `transparent_path` from `coin_detections`, composites the
RGBA onto solid grey, and feeds DINOv2. PCO rows never enter
`coin_detections` (no vision run, no auction context), so they have no
`transparent_path` and fall through `open_image_masked`'s `fallback_path`
branch — embedding the rectangular `*_obv.jpg` / `*_rev.jpg` directly.

That means PCO embeddings carry auction-house-style backgrounds (often
near-black velvet or museum cards) baked into the DINOv2 input. Auction
embeddings have the background masked out. The two populations are not
embedded under the same regime, which leaks a confounder into any
cluster / matcher that crosses the boundary.

Goal: produce one `*_transparent.png` per side per PCO coin, store the
key on the `collection_data` row, and wire the embedding loader to find
it. Net: PCO becomes consistent with the auction set's masked-grey
input.

---

## 1. Schema change

```sql
ALTER TABLE collection_data
  ADD COLUMN obv_transparent_path VARCHAR(500) DEFAULT NULL,
  ADD COLUMN rev_transparent_path VARCHAR(500) DEFAULT NULL;
```

**Why on `collection_data` and not via synthesized `coin_detections` rows:**
PCO rows have no vision run, no L1 candidate, no `auction_record_id`. A
synthesized `coin_detections` row would carry FK lies (the
`auction_record_id` FK in particular is NOT NULL in the existing
schema). Keep the path on the row that owns the source image.

NULL default makes the bulk job resumable by `WHERE … IS NULL`.

**Migration mechanics:** run via the project's standard migration path
(same tool used for prior `collection_data` changes — confirm with
pipeline before merging). Don't run raw `ALTER` against prod.

---

## 2. Pilot (20 specimens)

**Why before bulk:** L1 (`src/layer1_geometry.py`) is tuned for auction
plates — grey/white card backgrounds, dual-coin AR≈2 layouts, auto-Otsu
polarity check based on corner means. PCO sources are single-coin on
varied backgrounds (velvet, museum card, hand-held). Two pieces of L1
that may misbehave on those:

1. **Otsu polarity** — driven by corner-mean intensity. Near-black
   velvet inverts cleanly, but textured velvet or labels in the corners
   can fool the test.
2. **Rim recovery Hough fallback** (`src/rim_logic.py:hough_rim_recovery`,
   `param2=25`) — may pick up scale bars, label edges, or label-frame
   straight lines as false circle votes.

Anchor on measurement, not assumption.

### Pilot selection (frozen, reproducible)

Pick the 20 IDs **once** and commit them to
`specs/pco_pilot_ids.csv` (columns: `id,record_id,side,background_bucket`).
Re-pilots run against the same 20 so iterations are comparable.

Stratify, don't randomize — random 20 will concentrate on the easy
case. Target distribution:

- 7 × velvet / dark cloth background
- 7 × museum card / printed-label background
- 6 × hand-held / out-in-the-world background

Drop the previous "other / hard-to-classify" bucket — 5/20 in an
unnamed bucket validates nothing.

Selection query (for hand-picking the slate, not for execution):

```sql
SELECT id, record_id, obv_spaces_path, rev_spaces_path
FROM collection_data
WHERE collection_id='pco' AND obv_spaces_path IS NOT NULL
ORDER BY id;
-- then page through and pick per bucket
```

**Budget reality:** assembling the slate requires eyeballing
collection_data images to bucket them — 30-60 minutes of hands-on
image work, not SQL. Reflected in §9.

### Pilot procedure

For each side of each pilot coin:
1. Download source image
2. Call `layer_1_structural_salience(img, sensitivity="standard", source_type="unknown")`
   - L1 takes two independent parameters: `source_type` (string,
     "auction"/"unknown") gates the auction-specific resolvers and
     Hough config; `sensitivity` (string mapped to a `Sensitivity`
     enum internally) controls thresholding aggressiveness.
   - **`source_type="unknown"` is critical** — prevents the two-coin
     resolver from triggering and prevents the auction-tuned Hough
     config (`CoinPairConfig.for_auction()`) from loading. Confirm in
     `src/layer1_geometry.py` that the `source_type` branch actually
     does what we expect; quote the relevant lines in the pilot PR.
   - **Plan B if it doesn't:** add a new `Sensitivity.PCO` value (with
     PCO-appropriate thresholds — e.g. tighter Hough `param2`,
     polarity-detection tweaks for dark backgrounds), land that, and
     re-run the pilot with `sensitivity="pco"`. Don't bulk-process
     behind a misleading flag.
3. Build the RGBA via `src.pipeline_manager.crop_with_alpha`
4. Composite on grey (matching auction-side `open_image_masked`)
5. Save a side-by-side preview: source | transparent | composited

### Pre-pilot calibration (one-shot, ~30 min)

Before running the pilot, measure the auction-side alpha coverage so
the exit-criteria thresholds are anchored, not asserted:

```python
# Sample N=500 random auction transparents from coin_detections,
# load the RGBA, compute alpha_coverage = (alpha>0).mean()
# Report: mean, p5, p50, p95
```

Persist the result as `specs/pco_auction_alpha_calibration.json`
(global, not per-coin — kept separate from the pilot ID list). The
exit criterion below uses **[p5, p95]** from this measurement.

**Implication acknowledged:** "within auction's [p5, p95]" means 10%
of auction coins themselves would fail this check. That's the
intent — we want PCO's bar to match the auction distribution's
middle, not to be tighter than auction is on itself.

### Pixel-equivalence control test (catches "the worker is doing something different")

Pick 5 auction sides at random. Run the new PCO worker against the
auction source `*_obv.jpg` / `*_rev.jpg`. Compare each output PNG to
the existing auction `*_transparent.png` by:

- alpha-mask IoU
- RGB pixel diff (mean abs diff over the intersection of both alpha
  masks)

Don't pick thresholds in advance. First measure: run the comparison
on the 5 sides, report the actual IoU and mean-diff distribution,
then set thresholds at "2σ outside the observed noise floor" and
record them in `specs/pco_auction_alpha_calibration.json`. Starting
guesses for sanity (revise after measurement): IoU ≥ 0.97, mean abs
RGB diff ≤ 2/255.

This proves the new worker is the same code path as auction, with no
unintended divergence. If it fails, fix the worker before the L1
pilot — the L1 differences from PCO backgrounds are a separate
question.

### Exit criteria (all required to proceed to bulk)

| check | rule | fail action |
|---|---|---|
| pixel-equivalence (control) | IoU and mean RGB diff within thresholds recorded in `specs/pco_auction_alpha_calibration.json` (2σ outside measured noise floor; starting guesses IoU ≥ 0.97, RGB diff ≤ 2/255) | fix worker, do not proceed |
| ndets per side | == 1, OR largest:second area ratio ≥ 2 (take largest) | log + investigate; if systematic, tune L1 before bulk |
| contour hugs flan | visual — no clipped flan edge, no included ruler / label / shadow | same |
| alpha coverage | within [p5, p95] from auction calibration above | low → eaten; high → grabbed extra |
| composited output | visually consistent with auction transparents on grey | same |

Pass threshold: **≥18 / 20 pass all checks** → proceed. Otherwise
land a PCO-specific L1 config (a new `Sensitivity.PCO` value) before
bulk.

**Reviewer rotation:** the pilot grid (40 preview images) is reviewed
by the implementer + one teammate via PR. Two eyes on subjective
checks beats one.

---

## 3. Bulk worker — `trivalaya_pipeline/pco_make_transparent.py`

New file. Behavior contract:

**Startup:**
- Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1` **before** importing cv2/numpy.
  `cv2.setNumThreads(0)` after import. Same defensive posture as
  `tools/reprocess_hough.py` (commit `9917537`) — under a worker pool
  these prevent segfaults from thread oversubscription.
- ACL discovery: `s3.get_object_acl(Bucket=BUCKET, Key=<known auction
  transparent>)` once at startup, save the `Grants` list, pass to
  every `put_object`. Do not re-discover per side; do not hardcode
  `public-read`.

**Row selection** (NULL filter = resumability):

```sql
SELECT id, record_id, obv_spaces_path, rev_spaces_path
FROM collection_data
WHERE collection_id='pco'
  AND ((obv_spaces_path IS NOT NULL AND obv_transparent_path IS NULL)
    OR (rev_spaces_path IS NOT NULL AND rev_transparent_path IS NULL));
```

**Per side:**
1. HEAD the dst key first. If it exists, skip upload, record the key
   to commit in the DB anyway (covers the "uploaded but crashed
   before commit" case). Cheap enough to do unconditionally.
2. Download src; `cv2.imdecode → BGR`.
3. `layer_1_structural_salience(img, sensitivity="standard", source_type="unknown")`.
4. Resolve detections:
   - 0 dets → log skip, leave NULL
   - 1 det → use it
   - \>1, area ratio (largest:second) ≥ 2 → take largest, log `multi_ok`
   - \>1, area ratio < 2 → log `multi_ambiguous`, leave NULL
5. `crop_with_alpha(img, contour, bbox_with_5%_margin)` → RGBA.
5b. **Size-cap:** if `max(rgba.shape[:2]) > 518`, resize per the policy
    in `specs/transparent_png_resize.md` §2 (longest side → 518,
    `cv2.INTER_AREA`, alpha resized with the same interpolator).
    PCO worker emits 518-cap PNGs from day one — no full-res PCO
    transparents ever land in Spaces.
6. `cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])`.
7. `s3.put_object(..., Grants=<from startup>)`.

**Per row, after both sides have been attempted (not necessarily both
succeeded):**

```sql
UPDATE collection_data
SET obv_transparent_path = COALESCE(:obv_or_null, obv_transparent_path),
    rev_transparent_path = COALESCE(:rev_or_null, rev_transparent_path)
WHERE id = :id;
```

Commit whatever fields succeeded. NULL sides (skip cases) leave the
column NULL; the NULL filter re-picks them up on resume only if the
underlying source becomes processable.

**Observability during bulk:**
- Every 100 rows (or 60 s, whichever first), emit a counter line:
  `processed=N ok=N skip_ndets0=N multi_ok=N multi_ambiguous=N upload_fail=N rate=R/min`.
  An operator running `tail -f pco_transparent_log.jsonl | jq` can
  see if it's wedged. The reprocess_hough log schema covers this.

**Notes:**

- Reuse `src.pipeline_manager.crop_with_alpha` (the helper added in
  commit `b5a36cd`). Don't duplicate the alpha-merge inline.
- Spaces key pattern: sibling of source.
  `raw/pco/<col>/<rec>_obv.jpg` → `raw/pco/<col>/<rec>_obv_transparent.png`
  (parallels the auction pattern `*_obv_crop.jpg` + `*_obv_transparent.png`
  living in the same prefix).
- `--workers 16` default; CPU-bound on L1, network-bound on Spaces.
  Same scaling envelope as `tools/reprocess_hough.py`.
- Log skips/multi/fails to `pco_transparent_log.jsonl` with the same
  schema as `tools/reprocess_hough.py` for cross-tool consistency.
- **Test fixtures:** ship one per stratification bucket (3 images +
  expected masks) in `tests/fixtures/pco_*.{jpg,mask.png}`. A single
  fixture is a smoke test, not coverage.
- **Already-cropped sources:** some PCO ingest paths may have
  pre-cropped images (mostly fills frame, near-zero background). L1
  on those can over-tighten (rim recovery chases the inner field).
  If the pilot surfaces this, add an early-exit gated on source
  fill-ratio before rolling out to bulk.

---

## 4. Wire into `cluster_coins.enrich_transparent_paths`

Today (`trivalaya-pipeline/cluster_coins.py:448-499`): only looks up
`coin_detections.normalized_path` → `transparent_path`. For PCO rows
(where `id < 0`, per the negative-ID convention from
`cluster_coins.py:377`), this returns empty and the embedding pipeline
falls through to rectangular.

Add a branch:

```python
# In enrich_transparent_paths, before the existing loop:
pco_ids = [(-int(r["id"])) for r in records if int(r.get("id", 0)) < 0]
if pco_ids:
    fmt = ",".join(["%s"] * len(pco_ids))
    cursor.execute(
        f"SELECT id, obv_transparent_path, rev_transparent_path "
        f"FROM collection_data WHERE id IN ({fmt})",
        pco_ids,
    )
    pco_lookup = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    for r in records:
        rid = int(r.get("id", 0))
        if rid < 0:
            obv_tp, rev_tp = pco_lookup.get(-rid, (None, None))
            r["obv_transparent"] = obv_tp or ""
            r["rev_transparent"] = rev_tp or ""
```

Then the existing auction loop runs unchanged for the positive-ID
records.

**Half-sided rows are explicit:** a PCO row with only `obv_spaces_path`
populated gets `obv_transparent` set and `rev_transparent` empty.
Downstream then masks the obv side and falls back to rectangular on
the rev side. This is intentional, not a bug — flag it only if a
non-trivial PCO fraction is one-sided. Surface the count during
pilot review:

```sql
SELECT
  SUM(obv_spaces_path IS NOT NULL AND rev_spaces_path IS NULL) AS obv_only,
  SUM(rev_spaces_path IS NOT NULL AND obv_spaces_path IS NULL) AS rev_only,
  SUM(obv_spaces_path IS NOT NULL AND rev_spaces_path IS NOT NULL) AS both
FROM collection_data WHERE collection_id='pco';
```

**Strict mode (recommended):** add a `--require-transparent` flag to
`cluster_coins.py` that raises `MissingTransparentPath` instead of
silently falling through. It reads the documented skip list (PCO rows
logged as `ndets=0` or `multi_ambiguous`) and exempts those —
otherwise legitimate skips would block every canonical re-embed.
Default off for backwards compat; turn on for the canonical re-embed.

---

## 5. Failure modes

| condition | action |
|---|---|
| L1 returns 0 detections | log row to `pco_transparent_skipped.csv` with reason `ndets=0`; downstream `cluster_coins` exempts via skip list in strict mode |
| L1 returns >1 detections, area ratio (largest:second) ≥ 2 | take largest contour; log as `multi_ok` for review (slab edges, scale bars, shadows are the usual culprits) |
| L1 returns >1 detections, area ratio < 2 | leave row NULL; log as `multi_ambiguous` for manual triage; downstream `cluster_coins` exempts via skip list in strict mode — guards against picking a label that happens to be larger than the coin |
| Spaces 404 on source | log as `source_404`; doesn't update DB row, picked up again on resume only if source becomes available |
| Spaces upload failure | log as `upload_fail`; doesn't update DB row, picked up on resume |
| `imdecode` returns None | log as `decode_failed`; same as upload_fail |

**Never delete or overwrite the source `*_obv.jpg` / `*_rev.jpg`** — the
transparent is additive. The source remains the authoritative pixels;
the transparent is a derived artifact.

---

## 6. Acceptance & rebuild gate

### Rollout order

1. **Merge wire-up code** (§4) and `--require-transparent` flag with
   skip-list support. Behavior change is zero because
   `obv_transparent_path` / `rev_transparent_path` are still NULL on
   every PCO row — `enrich_transparent_paths` returns empty,
   downstream falls back to rectangular (the pre-change state).
2. **Capture effect-gate before-snapshot** (§6b step 1) using the
   still-rectangular PCO embeddings.
3. **Bulk-run the worker** (§3) to populate the new columns and
   upload transparents.
4. **§6a coverage gate** must pass (`pending = 0` outside the skip list).
5. **Rebuild canonical features.npy.** The rebuild driver enforces
   §6a's gate at startup.
6. **Capture effect-gate after-snapshot** (§6b step 3); compare; if
   regression, invoke §11 rollback.
7. **Enable strict mode** (`--require-transparent`) for the next
   canonical re-embed once the effect gate has passed.

### 6a. Coverage gate (cheap, structural)

```sql
SELECT COUNT(*) AS pending
FROM collection_data
WHERE collection_id='pco'
  AND ((obv_spaces_path IS NOT NULL AND obv_transparent_path IS NULL)
    OR (rev_spaces_path IS NOT NULL AND rev_transparent_path IS NULL));
```

Must return 0 (modulo the documented skip list — PCO coins genuinely
failed L1, logged as `ndets=0` or `multi_ambiguous`). Same posture as
the auction side's `--mask-background` gate for Run 5.

**Enforce the gate in code, not policy:** the rebuild driver (the
script that builds the canonical features.npy) refuses to start when
pending > 0 outside the published skip list. A `--allow-pending`
escape hatch exists for emergency rebuilds but is logged loudly.

### 6b. Effect gate (does this actually help retrieval?)

The whole point is closing the PCO↔auction embedding gap. Verify
empirically.

**Precondition: confirm the overlap set exists.** Don't trust "these
coins exist"; measure.

The exact JOIN keys are TBD during implementation — confirm with
pipeline which signal actually links a PCO row to its auction
counterpart (candidates: `record_id` correspondence,
`matcher_seed_clusters` membership, or a manually-maintained mapping
table). The implementation PR fills this in and reports the count
before proceeding.

- If overlap N ≥ 20 → run the quantitative effect gate below.
- If 5 ≤ N < 20 → run it anyway, but treat the result as directional;
  add qualitative T-SNE of PCO vs. auction populations as a second
  read.
- If N < 5 → fall back to T-SNE alone; the cosine gate is statistically
  hollow at that size.

**Ordering matters — "before" embeddings must be captured first.**
Once the wire-up merges and features.npy is rebuilt with masked PCO
embeddings, the rectangular "before" embeddings are gone. Sequence:

1. **Before the wire-up merges:** run a one-off script that, for each
   overlap coin, loads the current rectangular PCO embedding and its
   auction embedding, computes cosine distance, and writes
   `specs/pco_effect_gate_before.csv` (columns: `pco_id, auction_id, cosine_before`).
2. Merge wire-up; bulk-generate transparents; rebuild features.npy.
3. **After:** compute the same distances against the new masked PCO
   embeddings, write `pco_effect_gate_after.csv`.
4. Compare row-by-row and report: mean Δ, median Δ, distribution.

Expected direction: post-change distances drop materially (≥ 0.05
mean cosine, depending on baseline). If they don't, either the
masking isn't doing what we think or the auction set has a
confounder we haven't isolated — investigate before promoting the
re-embed.

---

## 7. Out of scope

- **Backfilling existing rectangular embeddings** — features.npy rebuild
  handles this naturally when it picks up `obv_transparent` /
  `rev_transparent` from the wired-up `enrich_transparent_paths`. No
  separate migration.
- **Non-pco collections** (other `collection_id` values in
  `collection_data`) — same approach would apply but each collection's
  source-image conventions may differ. Tackle per collection.
- **Side-correction step** — PCO has no pairing step because the human
  ingest assigned obv/rev semantics at insert time. The
  basename-vs-DB-side mismatch documented for auction crops
  (`KNOWN_ISSUES.md`) does NOT apply to PCO.

---

## 8. Code references

- `src/layer1_geometry.py::layer_1_structural_salience` — L1 entry.
  Two independent knobs: `source_type` (string, "auction" / "unknown",
  gates auction resolvers) and `sensitivity` (string → `Sensitivity`
  enum, controls thresholding)
- `src/pipeline_manager.py::crop_with_alpha` — shared raw→RGBA helper
  (commit `b5a36cd`)
- `src/rim_logic.py::recover_rim` — geometric + Hough rim recovery
  (commits `f62fa1b`, `d284286`, `c87b17f`, `f52e963`)
- `trivalaya-pipeline/cluster_coins.py:448-499` — `enrich_transparent_paths`
  (the wire-up site)
- `trivalaya-pipeline/cluster_coins.py:697` — `open_image_masked` (the
  consumer — composites RGBA on grey, hands to DINOv2)
- `trivalaya-pipeline/cluster_coins.py:377` — negative-ID convention for
  collection rows
- `trivalaya-pipeline/trivalaya_pipeline/vision_adapter.py` — auction
  pipeline's `crop_with_alpha` usage, for reference style

---

## 9. Estimated effort

| step | effort |
|---|---|
| Schema migration + sanity | 15 min |
| Auction alpha-coverage calibration (one-shot script) | 30 min |
| Overlap-set precondition query + size triage | 15 min |
| Effect gate **before** snapshot (`pco_effect_gate_before.csv`) | 30 min |
| Pilot ID slate (hands-on image bucketing) + commit `specs/pco_pilot_ids.csv` | 30-60 min |
| Pilot driver + pixel-equivalence control + 20-coin run + 2-reviewer grid review | 2-3 h |
| Bulk worker script + test fixtures (3) + thread-cap startup + ACL discovery | 3-4 h |
| Wire-up in `enrich_transparent_paths` + strict-mode flag w/ skip-list | 45 min |
| Bulk run (resumable) | depends on PCO count, ~5-10 coins/min/worker × 16 workers |
| Effect gate **after** snapshot + delta report | 1 h |
| Canonical re-embed | trivial |
| **Total active dev** | **~1-1.5 dev-days plus bulk wall time** |

## 10. Cost & storage

Back-of-envelope before merging (PCO worker emits 518-cap per §3
step 5b — see `specs/transparent_png_resize.md`):

- Let `N_pco` = `SELECT COUNT(*) FROM collection_data WHERE collection_id='pco' AND obv_spaces_path IS NOT NULL`.
- Each 518-cap transparent PNG ≈ 160-320 KB (vs. 2-4 MB at full res).
- Storage delta ≈ `N_pco × 2 sides × ~250 KB`.
- Worked example: at N_pco = 100k → ~50 GB total. Roughly 10× smaller
  than a full-res emit would be (~500 GB).

Fill in actual numbers in the implementation PR.

The 5% crop margin is a secondary storage lever once the 518 cap is
in place — tightening it reduces PNG dimensions proportionally but
the cap dominates the savings.

---

## 11. Rollback

If post-merge inspection reveals bad masks (or §6b's effect gate
regresses instead of improving):

1. **Clear the DB pointers:**
   ```sql
   UPDATE collection_data
   SET obv_transparent_path = NULL,
       rev_transparent_path = NULL
   WHERE collection_id = 'pco';
   ```
2. **Leave the S3 objects in place.** They're additive; deleting them
   gains nothing and risks racing an in-flight rebuild. A
   garbage-collection sweep can clean orphans later if needed.
3. **Rebuild features.npy** — with paths NULLed, `enrich_transparent_paths`
   falls back to rectangular PCO embeddings (the pre-change state),
   identical to what `pco_effect_gate_before.csv` captured.
4. **Turn off strict mode** in `cluster_coins.py` if it was enabled.
5. Diagnose the failure (likely L1 priors on PCO backgrounds), then
   re-pilot before retrying.

This is fully reversible because the source `*_obv.jpg` / `*_rev.jpg`
are never modified — the transparent is a derived artifact.
