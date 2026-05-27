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

### Pilot procedure

```bash
# select 20 PCO coins covering background variety
SELECT id, record_id, obv_spaces_path, rev_spaces_path
FROM collection_data
WHERE collection_id='pco'
ORDER BY RAND()
LIMIT 20;
```

For each side of each coin:
1. Download source image
2. Call `layer_1_structural_salience(img, sensitivity="standard", source_type="unknown")`
   - **`source_type="unknown"` is critical** — prevents the two-coin
     resolver from triggering and prevents the auction-tuned Hough
     config (`CoinPairConfig.for_auction()`) from loading.
3. Build the RGBA via `src.pipeline_manager.crop_with_alpha`
4. Composite on grey (matching auction-side `open_image_masked`)
5. Save a side-by-side preview: source | transparent | composited

### Exit criteria (all four required to proceed to bulk)

| check | rule | fail action |
|---|---|---|
| ndets per side | == 1 | log + investigate; if systematic, tune L1 before bulk |
| contour hugs flan | visual — no clipped flan edge, no included ruler / label / shadow | same |
| alpha coverage | between 0.55 and 0.80 (matches auction baseline) | low → eaten; high → grabbed extra |
| composited output | visually indistinguishable from an auction `*_transparent.png` composited on grey | same |

Pass threshold: **≥18 / 20 pass all four checks** → proceed. Otherwise
land a PCO-specific L1 config (a new `Sensitivity.PCO` enum entry) before
bulk.

---

## 3. Bulk worker — `trivalaya_pipeline/pco_make_transparent.py`

New file. Sketch:

```python
"""
Generate alpha-masked transparents for PCO coins and write the keys
back to collection_data.{obv,rev}_transparent_path.

Resumable by virtue of the NULL filter — restart picks up only rows
that aren't already populated.
"""
import argparse, json, logging, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import boto3, cv2, mysql.connector, numpy as np
from botocore.client import Config

# Vision repo on sys.path
sys.path.insert(0, "/path/to/trivalaya-vision")
from src.layer1_geometry import layer_1_structural_salience
from src.pipeline_manager import crop_with_alpha, _load_and_resize


def select_pco_work(cur, limit=None):
    sql = """
        SELECT id, record_id, obv_spaces_path, rev_spaces_path
        FROM collection_data
        WHERE collection_id='pco'
          AND ((obv_spaces_path IS NOT NULL AND obv_transparent_path IS NULL)
            OR (rev_spaces_path IS NOT NULL AND rev_transparent_path IS NULL))
    """
    if limit: sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    return cur.fetchall()


def process_side(s3, src_key, dst_key):
    # 1. Download
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, src_key, buf)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "decode_failed"

    # 2. L1 — single coin source, no auction priors
    res = layer_1_structural_salience(img, sensitivity="standard", source_type="unknown")
    objects = res.get("objects", [])
    if not objects:
        return None, "ndets=0"
    if len(objects) > 1:
        # Single-coin source — take largest, log for review
        objects = sorted(objects, key=lambda o: cv2.contourArea(np.asarray(o["contour"]).astype(np.int32)), reverse=True)
        log_multi(src_key, len(objects))

    contour = np.asarray(objects[0]["contour"]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    margin = int(max(w, h) * 0.05)
    H, W = img.shape[:2]
    x1, y1 = max(0, x-margin), max(0, y-margin)
    x2, y2 = min(W, x+w+margin), min(H, y+h+margin)

    # 3. Bake alpha
    rgba = crop_with_alpha(img, contour, (x1, y1, x2, y2))

    # 4. Upload
    ok, png_buf = cv2.imencode(".png", rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if not ok:
        return None, "encode_failed"
    s3.put_object(Bucket=BUCKET, Key=dst_key, Body=png_buf.tobytes(),
                  ContentType="image/png", ACL="public-read")
    return dst_key, "ok"


def derive_transparent_key(src_spaces_key, side):
    # raw/pco/<col>/<rec>_obv.jpg → raw/pco/<col>/<rec>_obv_transparent.png
    return src_spaces_key.rsplit(".", 1)[0] + "_transparent.png"


def main():
    # ... arg parsing, DB connect, worker pool, per-row commit ...
```

**Notes:**

- Reuse `src.pipeline_manager.crop_with_alpha` (the helper added in
  commit `b5a36cd`). Don't duplicate the alpha-merge inline.
- Spaces key pattern: sibling of source.
  `raw/pco/<col>/<rec>_obv.jpg` → `raw/pco/<col>/<rec>_obv_transparent.png`
  (parallels the auction pattern `*_obv_crop.jpg` + `*_obv_transparent.png`
  living in the same prefix).
- Per-row DB commit after both sides upload — partial rows are fine
  because the NULL filter re-picks them up.
- `--workers 16` default; this is CPU-bound on L1, network-bound on
  Spaces. Same scaling envelope as the auction reprocess (`tools/reprocess_hough.py`).
- Log skips/multi/fails to `pco_transparent_log.jsonl` with the same
  schema as `tools/reprocess_hough.py` for cross-tool consistency.

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

**Strict mode (recommended):** add a `--require-transparent` flag to
`cluster_coins.py` that raises `MissingTransparentPath` instead of
silently falling through. Default off for backwards compat; turn on for
the canonical re-embed.

---

## 5. Failure modes

| condition | action |
|---|---|
| L1 returns 0 detections | log row to `pco_transparent_skipped.csv` with reason `ndets=0`; downstream `cluster_coins` refuses in strict mode |
| L1 returns >1 detections | take largest contour by area; log as `multi` for review (slab edges, scale bars, shadows are the usual culprits) |
| Spaces 404 on source | log as `source_404`; doesn't update DB row, picked up again on resume only if source becomes available |
| Spaces upload failure | log as `upload_fail`; doesn't update DB row, picked up on resume |
| `imdecode` returns None | log as `decode_failed`; same as upload_fail |

**Never delete or overwrite the source `*_obv.jpg` / `*_rev.jpg`** — the
transparent is additive. The source remains the authoritative pixels;
the transparent is a derived artifact.

---

## 6. Acceptance & rebuild gate

Before the next canonical features.npy rebuild:

```sql
SELECT COUNT(*) AS pending
FROM collection_data
WHERE collection_id='pco'
  AND ((obv_spaces_path IS NOT NULL AND obv_transparent_path IS NULL)
    OR (rev_spaces_path IS NOT NULL AND rev_transparent_path IS NULL));
```

Must return 0 (modulo the documented skip list — PCO coins that
genuinely failed L1 and were logged). Same posture as the auction
side's `--mask-background` gate for Run 5.

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

- `src/layer1_geometry.py::layer_1_structural_salience` — L1 entry,
  takes `source_type` ("auction" / "unknown")
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
| Schema ALTER + sanity | 15 min |
| Pilot driver + 20-coin run + visual review | 1-2 h |
| Bulk worker script | 2-3 h |
| Wire-up in `enrich_transparent_paths` + strict-mode flag | 30 min |
| Bulk run (resumable) | depends on PCO count, ~5-10 coins/min/worker × 16 workers |
| Acceptance gate query + canonical re-embed | trivial |
| **Total active dev** | **~half day plus bulk wall time** |
