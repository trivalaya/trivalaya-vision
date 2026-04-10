# Review: Legend Ribbon Unwrap Evaluation Harness v1

## Questions

### 1. Eval subset origin (Section 5)
The existing `confusion_subset.csv` has 6,000+ coins across 10 authorities. Should `eval_subset.py` subsample from that CSV, or go back to the database? If subsampling, does the `difficulty_bucket` assignment happen automatically or require manual curation?

### 2. Center method definitions (Section 6.2)
Only `bbox` (bounding rect) is currently implemented in `unwrap.py`. For `moments` and `min_enclosing_circle` — are these `cv2.moments()` centroid and `cv2.minEnclosingCircle()` respectively, both operating on the largest contour? Worth stating explicitly since radius derivation differs between the three.

### 3. Polar dimension naming (Section 6.2)
The spec lists `polar_width` / `polar_height`, but the current code uses `WARP_ANGLE_BINS` (1024) and `WARP_RADIUS_BINS` (256) and transposes after `warpPolar`. Which axis is which? The mapping from spec names to the actual polar transform axes should be pinned down, or you'll get silent geometry bugs.

### 4. Normalization procedure for quality score (Section 8.4)
The composite formula uses `normalized_gradient_energy` and `normalized_row_variance_mean`, but no normalization method is defined. Options: fixed reference values, per-run min/max, percentile-based. This matters because per-run normalization makes scores non-comparable across runs, which defeats the purpose of the harness.

### 5. `nonblack_fraction` vs `nonzero_fraction` (Section 8.2)
The existing code computes a single `content_fraction` (pixels > 0). The spec adds both `nonblack_fraction` and `nonzero_fraction`. What's the intended distinction? Is `nonblack` thresholded (e.g., pixel > 10) to exclude near-black noise, while `nonzero` is strictly > 0?

### 6. `mask_area_fraction` definition (Section 8.2)
Listed but never defined. Fraction of what? The source image area covered by the alpha mask? The annulus band area that's masked? These give very different numbers.

### 7. `strict_alpha_mask` parameter (Section 6.2)
Currently the code always does strict binary thresholding. What would `strict_alpha_mask = false` do? If there's no non-strict mode planned for v1, this is a dead parameter.

### 8. `failed` vs `bad_*` boundary (Section 8.5 / 9)
A coin can have `status: success` but `quality_bucket: bad_blank`. Where exactly is the line? If `unwrap_coin` returns a ribbon array but it's 95% black, is that `status: success, bucket: bad_blank` or `status: failed`? The current code rejects ribbons below `MIN_CONTENT_FRACTION` — does that map to `failed` or `bad_blank`?

### 9. Baseline versioning (Section 14)
The spec says "do not overwrite baseline artifacts" but also that the eval subset can be "intentionally versioned." If the subset changes, the baseline is invalidated. Should there be explicit baseline versioning (e.g., `baseline_v1/`, `baseline_v2/`) or a rule that subset changes always trigger a new baseline run?

---

## Feedback

### 10. Missing: image caching
9 sweep runs x 300 coins = 2,700 S3 downloads of the same images. The spec should mandate a local cache (download once on first run, reuse across all runs in a sweep). This is the single biggest practical bottleneck and straightforward to solve.

### 11. Quality score penalty functions are undefined
The formula references `fallback_penalty` and `center_offset_penalty` but never defines them. Binary (0/1)? Linear ramp? Clipped sigmoid? These need concrete definitions or the score is unimplementable. Suggest adding a subsection with the penalty function signatures.

### 12. Sample size vs promotion thresholds
Section 16.3 requires `good_rate` to improve by 10% relative. With 300 coins and a ~60% baseline good rate, that's ~18 coins swinging. At that margin, noise from borderline cases could dominate. Consider either: (a) noting this is a heuristic, not a statistical test, or (b) adding a simple significance check (e.g., McNemar's test on paired per-coin outcomes).

### 13. `vs_baseline.png` selection criteria missing
The spec says show the same coin under baseline vs candidate, but not which coins or how many. Suggest: show the top N largest positive score deltas and top N largest negative deltas (regressions). That surfaces both wins and risks in one artifact.

### 14. Module placement ambiguity
Section 18 lists `unwrap_core.py`, `score_ribbons.py`, etc. but doesn't say where they live. Under `legend_cnn/`? A new subpackage? Also, the relationship between the proposed `unwrap_core.py` and the existing `unwrap.py` needs clarification — is it a refactor/rename, or do both coexist?

### 15. `difficulty_bucket` / `known_hard_case` are listed but unused
These are optional columns in the eval subset (Section 5.7) but never referenced in scoring, gallery selection, or reporting. If they should influence review (e.g., always include hard cases in galleries), say so. Otherwise they'll rot.

### 16. Missing: per-run timing
No mention of recording wall-clock time per run. Cheap to add, and useful for planning larger sweeps and catching performance regressions from expensive center methods.

### 17. Typo
Filename is `legend_unrwap_evaluation_harness.md` — should be `unwrap`.

This is a very good review. Most of these are real spec gaps, not nitpicks.

My overall take: the spec is directionally right, but this review correctly identifies several places where it is not yet implementation-tight enough. I would accept almost all of these comments.

Here is how I would resolve them.

## Main answers to the questions

### 1. Eval subset origin

For v1, `eval_subset.py` should subsample from the existing `confusion_subset.csv`, not go back to the DB.

Reason:

* cheaper
* deterministic
* already aligned to the Roman Imperial authority confusion use case
* avoids mixing subset-construction work with unwrap-evaluation work

I would state:

* `eval_subset.csv` is derived from `confusion_subset.csv`
* initial selection is stratified random by authority
* `known_hard_case` can be injected manually afterward
* `difficulty_bucket` is optional and should be auto-assigned later, not required for v1

So yes: start from the CSV, then allow light manual curation of hard cases.

### 2. Center method definitions

Yes, these should be explicit.

I would define:

* `bbox`: center = center of bounding rect of largest contour; radius = `max(w, h) / 2`
* `moments`: center = centroid from `cv2.moments(largest_contour)`; radius = equivalent radius or bounding-based radius, but pick one and state it
* `min_enclosing_circle`: center and radius from `cv2.minEnclosingCircle(largest_contour)`

And yes, all three should operate on the largest contour unless otherwise specified.

### 3. Polar dimension naming

Agreed, this needs to be pinned down. The current spec names are too easy to misunderstand.

I would rename them to:

* `warp_radius_bins`
* `warp_angle_bins`

That maps directly to intent and avoids width/height confusion after transpose.

Then explicitly state:

* raw `warpPolar` output is indexed as radius × angle or angle × radius depending on OpenCV behavior in practice
* our implementation canonicalizes to ribbon coordinates after transpose
* final ribbon width corresponds to angle axis
* final ribbon height corresponds to radial band thickness

That is much safer than `polar_width` and `polar_height`.

### 4. Normalization procedure

This is a real gap. Per-run normalization would indeed break cross-run comparability.

For v1, I would specify fixed normalization constants derived from the baseline run or from predetermined clipping bounds.

Best practical answer:

* use fixed clipped ranges, not per-run min/max
* example: normalize each metric with `clip((x - lo) / (hi - lo), 0, 1)`

Where `lo` and `hi` are defined once in the spec or in a baseline calibration file.

That keeps scores comparable across runs.

### 5. `nonblack_fraction` vs `nonzero_fraction`

The intended distinction should be:

* `nonzero_fraction`: pixels strictly greater than 0
* `nonblack_fraction`: pixels above a small threshold, for example `> 10/255`

That lets you separate pure mask emptiness from near-black junk/noise.

If you do not want both, then keep only `nonblack_fraction`. But if both remain, they need explicit thresholds.

### 6. `mask_area_fraction`

Agreed, undefined.

I would define it as:

* fraction of source image pixels included by the binary alpha mask

That is the simplest and most stable definition. Do not define it relative to the annulus in v1.

### 7. `strict_alpha_mask`

Also agreed. As written, it is premature.

For v1, I would remove it from the spec unless you actually plan a non-strict mode now. Otherwise it is just a dead config knob.

### 8. `failed` vs `bad_*`

This boundary should be explicit.

I would define:

* `status = failed` when the pipeline cannot produce a valid ribbon artifact at all, or when geometry fails hard enough that the output is considered invalid for evaluation
* `status = success` when a ribbon array is produced and passes minimum structural validity checks
* `quality_bucket = bad_*` applies only to successful outputs that are low quality

So:

* 95% black but technically produced ribbon:

  * either `success + bad_blank`
  * or `failed`
* choose one rule and hold it fixed

My preference for the harness:

* if ribbon file exists and passes basic dimensional/format checks, mark `success`
* then classify as `bad_blank` if content is too low

That preserves more diagnostic information.

### 9. Baseline versioning

Yes, explicit versioning is needed.

I would add:

* every eval subset version gets its own baseline
* changing the eval subset invalidates the old baseline for formal comparison
* naming convention:

  * `eval_subset_v1.csv`
  * `baseline_eval_v1/`

So yes, subset changes should trigger a new baseline run.

---

## Feedback items I strongly agree with

### 10. Missing image caching

Strong agree. This should absolutely be in the spec.

Without caching, the sweep is wasting bandwidth and time. I would make local caching mandatory for v1.

### 11. Penalty functions undefined

Strong agree. These need concrete definitions.

At minimum:

* `fallback_penalty = 1.0 if used_fallback else 0.0`
* `center_offset_penalty = clip(center_offset / max_center_offset, 0, 1)`

Simple, boring, implementable.

### 12. Promotion thresholds vs sample size

Agree. The thresholds should be described as heuristics, not statistical proof.

McNemar’s test is a nice idea, but I would probably not require it in v1. I would add a note that acceptance thresholds are heuristic and should be validated by paired visual review.

### 13. `vs_baseline.png` selection criteria

Agree. I would specify:

* top N biggest positive score deltas
* top N biggest negative score deltas
* optionally a random middle sample

That makes the comparison gallery much more informative.

### 14. Module placement ambiguity

Agree. The spec should say these live under `legend_cnn/`, and that `unwrap.py` becomes a thin CLI wrapper over `unwrap_core.py`.

### 15. Optional columns unused

Agree. Either connect them to outputs or remove them.

Best fix:

* `known_hard_case` should force inclusion in comparison galleries and summary breakout stats
* `difficulty_bucket` can remain optional but should only stay if used in reporting

### 16. Missing per-run timing

Agree. Easy win. Add:

* wall-clock runtime
* mean sec/coin
* download/cache hit rate

### 17. Typo

Yes, fix it.

---

## What I would change in the spec

I would revise the spec with these concrete changes:

* use `confusion_subset.csv` as the v1 source for `eval_subset.csv`
* rename `polar_width` / `polar_height` to `warp_radius_bins` / `warp_angle_bins`
* explicitly define all center methods
* remove `strict_alpha_mask` from v1 unless needed now
* define score normalization as fixed clipped ranges, not per-run scaling
* define `nonzero_fraction`, `nonblack_fraction`, and `mask_area_fraction`
* define `status` vs `quality_bucket` boundary
* require local image caching
* define penalty functions explicitly
* add baseline versioning tied to subset version
* specify comparison gallery coin-selection rules
* specify module placement under `legend_cnn/`
* add run timing metrics

## Bottom line

This review improves the spec materially. I would not treat it as “comments to consider later”; I would fold most of it directly into the next revision.

The only thing I would keep lightweight is the statistical-testing suggestion. Useful, but probably not required for v1.

If you want, I can now produce a revised **v1.1 markdown spec** that incorporates all of these decisions cleanly.

