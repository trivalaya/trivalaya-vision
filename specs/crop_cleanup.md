# Trivalaya Spec: Background Crop Audit and Repair Task

## 1. Purpose

Design a quiet, selective background task that identifies poor coin crops, attempts safer recrops, and replaces the crop only when the replacement is clearly better.

This task is intended to catch uncommon but important failures such as:

* clipped or over-tight crops
* two-coin split leakage, where part of the neighboring coin remains in the crop
* badly off-center crops
* geometrically suspicious crops that are technically valid but poor for downstream ML

This is **not** a whole-corpus continuous reprocessor. It is an exception handler for suspicious crops.

---

## 2. Why this should exist

Poor crops have downstream cost:

* weaker embeddings
* worse clustering and retrieval
* misleading visual search results
* degraded confidence in the pipeline

The current vision pipeline already provides useful hooks for this design:

* `crop_quality.py` computes `GREEN / YELLOW / RED`
* `layer1_geometry.py` attaches quality information to detections
* the pipeline already stores crop and geometry metadata in `coin_detections`
* two-coin handling already has side-aware and midpoint-aware logic

Because the system already knows which detections are suspicious, automated selective repair is better than relying only on manual discovery.

---

## 3. Operating model

Use a hybrid model:

* **automatic background repair** for suspicious crops
* **manual enqueue** for bad crops discovered later
* one shared repair engine for both

This keeps the correction logic in one place while allowing quiet automation.

---

## 4. Non-goals

The task should not:

* constantly rescan every crop in the corpus
* replace crops without an explicit improvement test
* re-embed or reprocess the whole dataset unless a crop was actually changed
* act as a generic image enhancement system

---

## 5. Pipeline placement

Place the crop audit/repair task **after crop generation** and **before downstream embedding work becomes final**.

Recommended flow:

1. detect candidate
2. generate crop
3. compute quality metrics and quality flag
4. persist row in `coin_detections`
5. if suspicious, enqueue crop audit job
6. background worker attempts repair
7. if repaired crop is accepted, mark crop as repaired
8. downstream embedding/indexing uses the accepted crop version

For existing historical crops, allow a backfill mode that queues only rows matching targeted suspicion rules.

---

## 6. Existing signals to leverage

The current system already exposes useful audit signals and should reuse them rather than inventing a parallel quality framework.

### 6.1 Existing code hooks

From the current implementation:

* `crop_quality.py`

  * `get_detection_quality_flag(...)`
  * emits `GREEN`, `YELLOW`, or `RED`
  * uses reasons such as `neighbor_clamp`, `edge_clamp_*`, `low_solidity_*`, `off_center_*`, `tiny_area`, and `degenerate_aspect`
* `layer1_geometry.py`

  * already integrates the quality flagging step into L1 detections
* `two_coin_resolver.py`

  * already supports side-aware split output and remapping from local crop coordinates back to original-image coordinates

### 6.2 Existing DB fields visible in `coin_detections`

Current visible fields already useful for a repair worker include:

* identity and linkage:

  * `id`
  * `auction_record_id`
  * `coin_id`
  * `side`
  * `detection_index`
* crop asset paths:

  * `crop_path`
  * `transparent_path`
  * `normalized_path`
  * `highres_path`
* confidence / geometry-ish fields:

  * `side_confidence`
  * `circularity`
  * `solidity`
  * `coin_likelihood`
  * `edge_support`
  * `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
* metadata containers:

  * `final_classification`
  * `layer2_container`
  * `vision_meta`

These are enough to support a DB-backed audit queue with minimal schema additions.

---

## 7. Core recommendation

Implement a **selective background crop-repair worker** that only examines:

* all new `RED` crops
* selected `YELLOW` crops
* crops manually reported later
* optional targeted historical backfills

Do **not** make it a global always-on reprocessor of all crops.

---

## 8. Failure modes to target in v1

### 8.1 Over-tight crop / clipping

The crop is too tight and removes part of the coin.

### 8.2 Two-coin contamination

The crop contains meaningful area from the neighboring coin, especially near the split boundary.

### 8.3 Off-center crop

The crop includes the correct coin but is poorly centered and gives unhelpful framing.

### 8.4 Severe clamp / edge interaction

The crop geometry strongly suggests clipping against image edges or split walls.

### 8.5 Small but technically valid crop

The crop is geometrically valid but too small or poor for ML readiness.

### 8.6 ROI-boundary-limited crop

The crop is poor because the local split ROI is already pinned to one or more boundaries, so simple recropping from the existing child crop is unlikely to help.

### 8.7 Directional split-child failure

A two-coin split produces one materially degraded child while the sibling looks substantially healthier. This suggests the split partition favored one side and starved the other.

---

## 9. Missing but strongly recommended metadata

To make repair robust and cheap, persist the following fields explicitly if they are not already stored inside `vision_meta`.

### 9.1 Detection quality fields

Add these columns, or store them as stable structured JSON keys:

* `quality_flag`
* `quality_reason`
* `quality_reasons_json`
* `quality_metrics_json`
* `quality_scored_at`
* `quality_version`

### 9.2 Repair lifecycle fields

* `repair_status` (`not_needed | queued | processing | repaired | unrepaired | manual_override | ignored`)
* `repair_attempt_count`
* `repair_last_attempt_at`
* `repair_method`
* `repair_version`
* `repair_score_before`
* `repair_score_after`
* `repair_notes`
* `needs_manual_review` (boolean)
* `replaced_active_asset` (boolean)

### 9.3 Alternate asset fields

* `original_crop_path`
* `active_crop_path`
* `repaired_crop_path`
* `original_normalized_path`
* `active_normalized_path`
* `repaired_normalized_path`

The key rule is: **never lose the original crop**.

### 9.4 Two-coin split metadata

For crops that came from a two-coin split, persist:

* `is_two_coin_split`
* `source_candidate_bbox_x`
* `source_candidate_bbox_y`
* `source_candidate_bbox_w`
* `source_candidate_bbox_h`
* `split_method` (`hough | watershed | other`)
* `split_side` (`left | right`)
* `neighbor_midpoint_x`
* `expected_center_x`
* `expected_center_y`
* `expected_radius`
* `split_debug_json`

This is especially important because the bad example looks like a split-quality problem, not just a generic bbox problem.

---

## 10. Queueing rules

A detection should enter the crop audit queue if any rule below is true.

### Rule A: hard failures

Queue all rows where:

* `quality_flag = RED`

### Rule B: selected yellow warnings

Queue rows where:

* `quality_flag = YELLOW`
* and `quality_reason` or `quality_reasons_json` includes any of:

  * `neighbor_clamp`
  * `edge_clamp_*`
  * `off_center_x_*`
  * `off_center_y_*`
  * `severe_clamp*`

### Rule C: metric thresholds

Queue rows whose persisted metrics look suspicious even if quality flagging is missing or stale, for example:

* very low `coin_likelihood`
* weak `edge_support`
* unusually low `solidity`
* suspicious aspect ratio from `bbox_w / bbox_h`
* abnormal crop-vs-image scale once image size is available

### Rule D: image-content audit heuristics

Optionally queue crops after a lightweight direct crop inspection if any of the following are observed:

* foreground touches crop border too heavily
* multiple disconnected foreground masses
* coin silhouette strongly off-center
* likely foreign mass present near split wall

### Rule E: structural split-risk rules

Auto-enqueue split children when they are likely structural failures rather than harmless yellows.

Queue when all of the following hold:

* `is_two_coin_split = 1`
* `quality_flag = YELLOW`
* `quality_reason like 'edge_clamp%'` or `quality_reason like 'neighbor_clamp%'`
* and any of:

  * `bbox_x = 0`
  * `bbox_y = 0`
  * child bbox touches the local ROI max edge
  * `aspect < 0.70`
  * `aspect > 1.45`
  * `abs(off_center_x) > threshold`
  * `abs(off_center_y) > threshold`

This rule is meant to catch cases like:

* narrow, boundary-pinned children from a hough split
* very wide or very tall crops that indicate the split grabbed too much territory
* child crops whose coin center is nearly at a crop corner

### Rule F: sibling asymmetry rules

For detections produced by the same two-coin split event, queue when one child looks materially worse than its sibling.

Examples of strong asymmetry signals:

* one child is boundary-pinned and the sibling is not
* one child has far worse aspect ratio than the sibling
* one child has much larger off-center magnitude than the sibling
* one child has far smaller usable crop area than the sibling, after accounting for actual coin size

This is especially useful for directional split failures where the split favored one side and starved the other.

### Rule G: manual enqueue

Allow manual enqueue by:

* `coin_detections.id`
* `coin_id`
* `auction_record_id`
* crop path
* source image path

Manual enqueue should use the same worker and acceptance logic as automatic queueing.

---

## 11. New heuristics to add

Add two explicit audit signals that should be treated as first-class reasons.

### 11.1 Foreign-mass-near-split-boundary detector

For two-coin crops, detect whether meaningful foreground exists in the forbidden band near the split wall.

Examples:

* left crop contains substantial foreground close to its right-side midpoint wall
* right crop contains substantial foreground close to its left-side midpoint wall

This should become a first-class reason such as:

* `split_boundary_foreign_mass`
* `split_leakage_left`
* `split_leakage_right`

This directly targets cases where one crop contains part of the neighboring coin.

### 11.2 Sibling asymmetry detector

For child crops produced by the same split event, compare the two outputs directly.

This detector should emit explicit reasons such as:

* `split_child_asymmetry_aspect`
* `split_child_asymmetry_offcenter`
* `split_child_asymmetry_area`
* `split_child_asymmetry_boundary_pin`

This directly targets cases where one child is clearly damaged while the sibling is substantially healthier.

---

## 12. Repair worker responsibilities

The background worker must:

1. claim a queued detection idempotently
2. load the source image and active crop metadata
3. build one or more candidate recrops
4. score each candidate
5. accept replacement only when improvement is clear
6. persist both outcome and lineage
7. optionally enqueue downstream re-normalization / re-embedding only if the crop changed

---

## 13. Repair strategies

The worker should try a small ladder of strategies in order, from safest to most invasive. However, strategy selection should be routed by failure class rather than always starting with simple recrop.

### 13.1 Routing rule before repair

Before trying any repair, classify the case into a coarse structural class:

* `simple_recrop`
* `split_boundary_adjustment`
* `parent_roi_reresolve`
* `manual_review`

Examples:

* single-coin mild off-center crop -> `simple_recrop`
* two-coin crop with split leakage near midpoint -> `split_boundary_adjustment`
* two-coin crop with bbox pinned to ROI boundary plus `edge_clamp` -> `parent_roi_reresolve`
* ambiguous or inconsistent metadata -> `manual_review`

### 13.2 Strategy 1: bbox rebuild with better margin

For single-coin detections:

* rebuild crop from contour or expected center/radius
* test a small margin ladder, for example `8%`, `12%`, `16%`, `20%`
* reject any crop that increases border contact too much

Use when the issue is likely over-tightness or mild off-centering and the crop is not structurally ROI-limited.

### 13.3 Strategy 2: split-aware boundary adjustment

For two-coin detections:

* keep the expected side (`left` or `right`)
* enforce a soft exclusion band near `neighbor_midpoint_x`
* shift the crop boundary away from the split wall if foreign foreground is detected
* preserve enough padding on the outer side of the coin

This is the main repair path for neighbor bleed when the parent ROI itself still appears usable.

### 13.4 Strategy 3: local two-coin re-resolve

For suspicious split detections:

* reload the source image
* reconstruct the parent ROI using the stored source candidate bbox
* rerun only the local two-coin resolver on that ROI
* compare the newly produced split crop(s) to the stored crop

This is stronger than simply trimming the existing crop and should be the preferred recovery path when split metadata is present.

Important routing rule:

* if `is_two_coin_split = true`
* and the child bbox touches one or more ROI boundaries
* and `quality_reason` contains `edge_clamp`
* and off-center is materially nonzero

then bypass simple margin expansion and go straight to `local two-coin re-resolve`.

This directly covers ROI-boundary-limited split children.

### 13.5 Strategy 4: mask-based dominant-component recrop

As a last resort:

* segment foreground inside the crop
* keep the connected component containing the expected center
* suppress foreign components
* rebuild crop around the retained component with padding

This is useful when the split metadata is imperfect but the crop still contains enough evidence to isolate the intended coin.

### 13.6 Pair-aware generation with child-specific replacement

For two-coin split cases, the worker may need to regenerate both child candidates from the same parent ROI. However, acceptance should remain child-specific.

That means:

* the worker may re-resolve the pair together
* score both regenerated children against their current active versions
* replace only the degraded child if the healthier sibling does not clearly improve

This avoids needlessly replacing a decent crop just because its sibling was bad.

---

## 14. Candidate scoring

The worker must never replace a crop just because it produced a new one.

Each candidate crop should receive a repair score using weighted penalties and bonuses.

### 14.1 Suggested penalty terms

* border contact penalty
* split-wall contamination penalty
* multi-component penalty
* off-center penalty
* aspect degeneracy penalty
* low fill / tiny usable area penalty
* weak silhouette penalty
* sibling asymmetry penalty for pair-generated candidates that still produce one clearly damaged child

### 14.2 Suggested bonus terms

* improved centering
* retained coin area
* lower foreign-mass score
* better silhouette continuity
* higher expected circularity/solidity consistency

### 14.3 Acceptance rules

Recommended conservative v1 policy:

Accept replacement only if one of these is true:

* score improves by at least 20%
* old crop is `RED` and new crop reaches at least `YELLOW`
* old crop is `YELLOW` and new crop reaches `GREEN`

And all of these must also hold:

* no new clipping is introduced
* crop still matches expected side semantics
* usable coin area is not materially reduced
* crop dimensions remain valid for downstream normalization

For pair-generated candidates, replacement remains per child.

Do not replace `GREEN` crops automatically in v1.

## 15. Data integrity and lineage

Every repair attempt must preserve traceability.

Persist:

* original asset paths
* repaired asset paths
* pre/post scores
* method used
* timestamps
* version of the repair logic

This allows later audit, rollback, and evaluation.

---

## 16. Job model

### 16.1 Queue table

Recommended new table: `crop_repair_jobs`

Suggested fields:

* `id`
* `coin_detection_id`
* `job_state` (`queued | processing | succeeded | no_change | failed | abandoned`)
* `enqueue_reason`
* `priority`
* `attempt_count`
* `worker_id`
* `lease_expires_at`
* `created_at`
* `started_at`
* `finished_at`
* `error_text`
* `repair_version`

### 16.2 Idempotency

Use one active repair job per `(coin_detection_id, repair_version)`.

A rerun under a new version should be allowed.

### 16.3 Leasing

If you use multiple workers, all updates must verify current ownership, for example:

* `WHERE id = ? AND worker_id = ? AND lease_expires_at > NOW()`

This avoids stale-worker overwrite behavior.

---

## 17. Worker flow

### 17.1 High-level flow

1. claim queued job
2. load `coin_detections` row
3. load source image and active crop
4. collect persisted geometry and split metadata
5. recompute current crop audit score for baseline
6. try repair strategies in order
7. score candidates
8. choose best candidate if it beats acceptance threshold
9. write repaired assets and metadata
10. mark job outcome
11. if crop changed, enqueue dependent asset refresh

### 17.2 Dependent asset refresh

If crop changes, downstream updates may include:

* transparent asset refresh
* normalized asset refresh
* embedding refresh
* search index refresh

Only enqueue these when the active crop actually changes.

---

## 18. Performance posture

This task should be cheap because it is selective.

Recommended posture:

* low-priority worker
* bounded concurrency
* process only queued rows
* no corpus-wide rescans except explicit backfills
* retry limits on failed jobs

### Suggested defaults

* poll every few minutes
* process a small fixed batch each run
* max 2–3 attempts per job version
* backoff for repeated failures

---

## 19. Historical backfill mode

Allow a one-time or occasional backfill job that queues likely-bad existing detections using SQL filters such as:

* `quality_flag IN ('RED', 'YELLOW')`
* `quality_reason LIKE '%neighbor_clamp%'`
* `quality_reason LIKE '%edge_clamp%'`
* `coin_likelihood < threshold`
* `edge_support < threshold`
* `is_two_coin_split = 1`

This should still enqueue into `crop_repair_jobs` rather than creating a separate repair path.

---

## 20. Manual review mode

Some crops should not be auto-replaced even if suspicious.

Mark `needs_manual_review = 1` when:

* all repair strategies fail
* the best candidate is ambiguous
* split metadata is inconsistent
* source image quality is too poor to recrop safely

Manual review should show:

* original crop
* candidate repaired crop
* source image ROI
* repair reasons and scores

---

## 21. Success metrics

Track whether the worker helps rather than just doing work.

Recommended metrics:

* fraction of detections queued for audit
* repair success rate
* no-change rate
* manual-review rate
* false-improvement rate from spot checks
* downstream embedding/search improvement on repaired examples
* percentage of repaired crops from two-coin split cases

Also keep a small gold set of known bad crops for regression testing.

---

## 22. Rollout plan

### Phase 1: metadata and queue plumbing

* persist quality flag and reasons in stable form
* add `crop_repair_jobs`
* add repair lifecycle fields to `coin_detections`
* support manual enqueue

### Phase 2: conservative worker

Implement only:

* queue all `RED`
* queue selected `YELLOW`
* strategy 1 for simple recrops
* strategy 2 for split-boundary adjustment
* strict replacement thresholds

### Phase 3: local re-resolve

Add:

* strategy 3 local rerun of two-coin resolver
* explicit split-boundary foreign-mass detector

### Phase 4: evaluation and tuning

* review accepted repairs
* tune thresholds
* decide whether to widen the queue

---

## 23. Recommended default policy for Trivalaya v1

### Auto-enqueue

* all `RED`
* `YELLOW` containing `neighbor_clamp`
* `YELLOW` containing `edge_clamp`
* `YELLOW` containing `off_center_x` or `off_center_y`
* two-coin split children with `bbox_x = 0` or `bbox_y = 0`
* two-coin split children with strongly abnormal aspect ratio
* two-coin split children with strong sibling asymmetry
* manually reported crops

### Do not auto-enqueue in v1

* plain `solidity_missing` alone
* already-accepted `GREEN` crops
* large edge-to-edge intended studio shots unless other rules fire

### Auto-replace only when

* the replacement clearly improves score
* no new clipping appears
* side semantics remain consistent

### Green policy

Treat `GREEN` as a strong keep signal in v1.

* `GREEN` crops should be left alone automatically
* even if they originate from a two-coin split
* even if one metadata field such as `side` vs `inferred_side` disagrees

Metadata mismatches like side-label swaps should be handled by a separate metadata reconciliation path, not by the crop repair worker.

## 24. Final recommendation

For Trivalaya, this should be implemented as a **selective DB-backed background crop audit and repair worker**, not a purely ad hoc cleanup process.

Ad hoc review should still exist, but only as an additional way to enqueue edge cases into the same repair engine.

That gives you:

* quiet correction of uncommon failures
* protection for downstream embeddings and search
* one shared correction path for both automatic and manual cases
* low operational cost because only suspicious rows are examined

---

## 25. Implementation notes for the current codebase

The current code already points strongly toward this architecture:

* quality flagging already exists
* split-aware crop logic already exists
* crop and geometry metadata already live in the detection layer and DB

The biggest functional upgrade still needed is explicit detection of **foreign mass near the split boundary**, because that is the most direct signal for the kind of bad crop shown in the example.

Here’s how I’d answer each one.

**1. Source image access for Strategy 3**

Strategy 3 should use the **full original auction image** or at least the **parent pre-split ROI source**, not the child crop.

Best design:

* store a reliable `source_image_path` on the detection row or through a stable join to the parent image record
* do not make the worker infer this indirectly from crop paths

My preference:

* canonical full-image path should live on the auction image / auction record side
* `coin_detections` should also carry either:

  * `source_image_id`, or
  * a denormalized `source_image_path`

That gives you:

* normalized ownership of the image record
* cheap worker access without extra guesswork

For Strategy 3, the minimum required inputs are:

* `source_image_path`
* `source_candidate_bbox_*` for the parent ROI before the child split
* split metadata

If you do not currently persist the full source image path in a directly reachable way, I would make that a required prerequisite for the repair worker.

**2. Columns vs JSON for split metadata**

Use a **hybrid**:

Keep rich debug detail in JSON, but promote the small set of fields needed for queueing, filtering, and routing into real columns.

Promote to columns:

* `is_two_coin_split`
* `split_method`
* `split_side`
* `source_candidate_bbox_x`
* `source_candidate_bbox_y`
* `source_candidate_bbox_w`
* `source_candidate_bbox_h`
* `neighbor_midpoint_x`
* maybe `split_event_id` if you add it

Keep in JSON:

* verbose debug traces
* intermediate candidate lists
* hough diagnostics
* experimental fields that may change often

Why:

* Rule E/F and backfill queries will be much cheaper and cleaner with columns
* worker routing logic becomes simpler
* JSON remains good for non-indexed debug detail

So yes: I would promote the operationally important fields.

**3. Where this should live**

I would split responsibilities:

* **core image/crop logic** lives in `trivalaya-vision`
* **DB-backed queue worker** lives in `trivalaya-pipeline`

That is the cleanest separation.

Specifically:

* `trivalaya-vision`

  * crop scoring
  * foreign-mass detector
  * sibling asymmetry scoring
  * local re-resolve logic
  * candidate generation / acceptance functions
* `trivalaya-pipeline`

  * `crop_repair_jobs`
  * job claiming / leasing / retries
  * DB reads and writes
  * downstream refresh enqueueing

So this should be a **pipeline worker that calls vision repair code**, not a pure vision module and not a pure pipeline module.

**4. Foreign mass detector: transparent mask vs raw pixels**

Use the **transparent/mask representation first** if it is trustworthy and aligned to the active crop.

That is the right v1 choice.

Why the mask is better:

* simpler
* faster
* less sensitive to background texture or lighting
* directly answers the question “is there foreground inside the forbidden band?”

Recommended order:

* primary detector: use transparent/mask foreground occupancy in the exclusion band near split wall
* fallback or validation: raw-pixel analysis only when mask quality is suspect or missing

So:

* v1 should be **mask-driven**
* raw pixels are optional backup, not the default

**5. What is actually missing from 9.1**

If `quality_flag`, `quality_reason`, and `quality_metrics` already exist, then yes, the missing pieces are mostly:

* `quality_reasons_json`
* `quality_scored_at`
* `quality_version`

I would also consider one more:

* `quality_primary_reason_rank` or equivalent is optional, but not necessary if `quality_reason` is already the primary

Why these matter:

* `quality_reasons_json` preserves all triggered reasons instead of only the top one
* `quality_scored_at` tells you when the evaluation was made
* `quality_version` matters once heuristics evolve, so you can backfill or compare old vs new scoring cleanly

So the answer is basically yes: the main gap is not the existence of quality fields, but **full multi-reason persistence and versioning**.

**6. Sibling lookup for Rule F**

Do **not** rely on `(coin_id, detection_index)` alone, and do **not** rely on `(auction_record_id)` alone.

Best answer: introduce a dedicated **split event key**.

Add something like:

* `split_event_id`

Each pair of child detections produced from one two-coin split should share the same `split_event_id`.

That becomes the reliable sibling join key.

Why the other options are weaker:

* `(coin_id, detection_index)` is not a sibling key; it is too tied to output ordering and may not be stable across reruns
* `(auction_record_id)` is too broad; one record may have multiple images, reruns, or multiple detection generations
* `created_at` is useful for debugging but should not be the primary linkage key

Recommended sibling lookup order:

1. `split_event_id` if present
2. fallback to a stricter synthetic key, such as:

   * `auction_record_id`
   * `source_image_id`
   * `source_candidate_bbox_*`
   * `created_at` window
   * `is_two_coin_split = 1`

But really, I would make `split_event_id` part of the schema. It will simplify both Rule F and pair-aware re-resolve enormously.

My overall recommendation:

* require direct full-source image access
* promote operational split metadata to columns
* put the worker in `trivalaya-pipeline`, with repair/scoring logic in `trivalaya-vision`
* use the transparent mask for v1 foreign-mass detection
* add `quality_reasons_json`, `quality_scored_at`, and `quality_version`
* add `split_event_id` as the canonical sibling join key

The only one I’d call non-negotiable is `split_event_id`. Without it, sibling-aware logic becomes much more brittle.

