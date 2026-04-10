# Legend Ribbon Unwrap Evaluation Harness v1.1

## 1. Purpose

This spec defines a repeatable evaluation harness for the legend ribbon unwrap pipeline so parameter tuning can be done through controlled experiments rather than one-off manual iteration.

The harness must allow us to:

* freeze a benchmark subset
* run multiple unwrap configurations against the same data
* score outputs automatically using proxy quality metrics
* generate ranked review artifacts
* compare candidate runs against a baseline
* choose parameter changes based on measurable improvement

The goal is not OCR-grade text extraction yet. The goal is to maximize production of visually usable, legend-bearing ribbons with low blank-space rate and low geometric failure rate.

---

## 2. Problem Statement

The current unwrap process can produce some usable legend strips, but also many ribbons dominated by:

* black wedges
* masked blank regions
* coin interior instead of legend band
* edge-only geometry
* miscentered or fallback-derived annuli

At present, improvement requires changing code, regenerating a gallery, and manually judging results. This is slow, subjective, and hard to compare across iterations.

We need an evaluation harness that turns unwrap tuning into a controlled experiment loop.

---

## 3. Scope

This v1 harness covers:

* subset selection for evaluation
* parameterized unwrap execution
* per-ribbon statistics and quality scoring
* per-run summaries
* ranked galleries
* baseline-vs-candidate comparison
* small parameter sweeps

This v1 does **not** include:

* OCR
* CNN training
* adaptive annulus learning
* automatic mask correction
* production full-dataset execution orchestration

Those may be layered on later.

---

## 4. High-Level Design

The harness will consist of five stages:

1. **Frozen evaluation subset**
2. **Parameterized unwrap runner**
3. **Per-ribbon quality scoring**
4. **Review artifact generation**
5. **Run comparison and selection**

Each run must produce a self-contained output directory with config, metrics, artifacts, and summary files.

---

## 5. Frozen Evaluation Subset

### 5.1 Objective

Create a stable benchmark dataset so all candidate unwrap configurations are evaluated on the exact same coins.

### 5.2 Source

The evaluation subset is derived from the existing `legend_cnn/data/confusion_subset.csv`. It does not query the database directly.

Rationale: cheaper, deterministic, already aligned to the Roman Imperial authority confusion use case, avoids mixing subset-construction work with unwrap-evaluation work.

### 5.3 Selection Method

Initial selection is stratified random by authority. After automated selection, `known_hard_case` entries may be manually injected to ensure coverage of difficult examples from previous galleries.

### 5.4 Requirements

The evaluation subset must:

* contain enough coins to expose common failure modes
* include multiple authorities
* include both easy and hard examples
* remain fixed across runs unless intentionally versioned

### 5.5 Initial Size

For v1, target:

* **300 coins** preferred
* acceptable range: **200–500 coins**

### 5.6 Composition

The subset should include:

* different authorities
* different flan sizes
* different portrait scales
* centered and off-center examples
* known hard cases from previous galleries
* examples with both strong and weak masks

### 5.7 File

`data/eval_subset_v1.csv`

Subset versions are explicitly numbered. Changing the subset produces a new file (e.g., `eval_subset_v2.csv`) and invalidates the previous baseline.

### 5.8 Required Columns

* `coin_id`
* `authority`
* `highres_url`
* `transparent_url`

### 5.9 Optional Columns

* `known_hard_case` — boolean; if true, forces inclusion in comparison galleries and summary breakout stats
* `source_type`
* `notes`

---

## 6. Parameterized Unwrap Runner

### 6.1 Objective

Refactor unwrap execution so each run is controlled by a parameter object rather than hardcoded values.

### 6.2 Required Parameters

Each unwrap run must support the following config fields:

#### Geometry / annulus

* `inner_r_ratio`
* `outer_r_ratio`

#### Center/radius estimation

* `center_method` — one of the following:

  * `bbox` — center = center of bounding rect of largest contour; radius = `max(w, h) / 2`
  * `moments` — center = centroid from `cv2.moments()` on largest contour; radius = equivalent radius from contour area (`sqrt(area / pi)`)
  * `min_enclosing_circle` — center and radius from `cv2.minEnclosingCircle()` on largest contour

  All three methods operate on the largest contour from the binary alpha mask. Note that radius semantics differ across methods (bounding-rect half-extent vs area-equivalent vs enclosing circle), so radius values are not directly interchangeable as diagnostics. Cross-method comparisons of overall quality scores and quality buckets are valid; direct comparison of raw radius values across methods is not.

* `max_center_offset`
* `min_radius`

#### Fallback behavior

* `fallback_enabled`
* `fallback_cx`
* `fallback_cy`
* `fallback_radius`

#### Mask handling

* `mask_threshold` — threshold applied to the alpha channel for binary mask creation (0–255)

#### Polar unwrap dimensions

* `warp_radius_bins` — number of radial bins in the polar transform (maps to radius axis)
* `warp_angle_bins` — number of angular bins in the polar transform (maps to angle axis)

After `cv2.warpPolar`, the implementation transposes to ribbon coordinates where:
* ribbon width corresponds to the angle axis
* ribbon height corresponds to the radial band thickness

#### Final ribbon dimensions

* `ribbon_width`
* `ribbon_height`

### 6.3 Config File Format

Each run must be driven by a JSON config file.

Example:

```json
{
  "config_id": "v1_ir072_or090_bbox",
  "inner_r_ratio": 0.72,
  "outer_r_ratio": 0.90,
  "center_method": "bbox",
  "max_center_offset": 150,
  "min_radius": 100,
  "fallback_enabled": true,
  "fallback_cx": 256,
  "fallback_cy": 256,
  "fallback_radius": 220,
  "mask_threshold": 127,
  "warp_radius_bins": 256,
  "warp_angle_bins": 1024,
  "ribbon_width": 512,
  "ribbon_height": 64
}
```

### 6.4 Run Entry Point

CLI example:

```bash
python -m legend_cnn.run_experiment \
  --config configs/unwrap_v1_ir072_or090_bbox.json \
  --subset data/eval_subset_v1.csv
```

---

## 7. Image Caching

### 7.1 Objective

Avoid redundant S3 downloads when running multiple experiments against the same evaluation subset.

### 7.2 Requirements

* On first access, download each coin's highres and transparent images from DO Spaces and save to a local cache directory.
* On subsequent runs, load from cache without network access.
* Cache directory: `data/image_cache/`
* Cache key: the S3 key derived from the URL, stored under asset-type subdirectories (`highres/` and `transparent/`) to prevent collisions between different asset types sharing the same base key.
* Cache must be shared across all runs in a sweep.

### 7.3 Rationale

A 9-run sweep over 300 coins without caching would require 5,400 S3 downloads (300 coins x 2 images x 9 runs). With caching, only 600 downloads are needed (once per image).

---

## 8. Per-Ribbon Output Requirements

Each processed coin must generate:

* ribbon `.npy`
* optional ribbon preview `.png`
* per-ribbon stats row
* explicit status
* failure or fallback metadata where applicable

A ribbon should never silently pass through the pipeline if critical geometry fell back or failed.

---

## 9. Per-Ribbon Quality Scoring

### 9.1 Objective

Create cheap proxy metrics that correlate with whether a ribbon is likely to be usable for legend analysis.

### 9.2 Required Per-Ribbon Metrics

#### Content metrics

* `nonblack_fraction` — fraction of ribbon pixels with intensity > 10/255. Excludes near-black noise.
* `nonzero_fraction` — fraction of ribbon pixels with intensity strictly > 0. Measures pure mask emptiness.
* `mean_intensity` — mean pixel value across the ribbon (0–1 scale)
* `std_intensity` — standard deviation of pixel values across the ribbon

#### Structure metrics

* `gradient_energy` — mean of squared Sobel gradient magnitudes across the ribbon
* `edge_density` — fraction of ribbon pixels classified as edges (e.g., via Canny)
* `row_variance_mean` — mean of per-row intensity variances; captures horizontal structure in the legend band

#### Geometry/debug metrics

* `cx` — estimated center x
* `cy` — estimated center y
* `radius` — estimated radius
* `center_offset` — Euclidean distance from estimated center to image center. v1 assumes 512x512 source images, so image center is `(256, 256)`. All source images in the current pipeline are 512x512.
* `used_fallback` — boolean
* `fallback_reason` — string or null
* `mask_area_fraction` — fraction of source image pixels included by the binary alpha mask (mask pixel count / total image pixel count)

### 9.3 Recommended Interpretations

#### Likely bad / blank

* very low `nonblack_fraction`
* very low `gradient_energy`

#### Likely bad / geometry

* large `center_offset`
* fallback used
* suspiciously small radius

#### Likely usable

* moderate to high content fraction
* meaningful edge / gradient structure
* acceptable center/radius geometry

### 9.4 Composite Quality Score

Each ribbon must receive a `quality_score` derived from the proxy metrics.

#### Normalization

Metrics are normalized using fixed clipped ranges, not per-run scaling. This ensures scores are comparable across runs.

Each metric is normalized as:

```text
normalized(x) = clip((x - lo) / (hi - lo), 0, 1)
```

Where `lo` and `hi` are fixed constants defined once and calibrated from the baseline run.

#### Initial normalization ranges

These values should be calibrated after the baseline run and then frozen:

| Metric | `lo` | `hi` | Notes |
|--------|------|------|-------|
| `gradient_energy` | TBD | TBD | Calibrate from baseline p5/p95 |
| `row_variance_mean` | TBD | TBD | Calibrate from baseline p5/p95 |

After baseline calibration, these values must be recorded in the config or in a shared normalization reference file and remain fixed for the duration of the sweep. If the baseline distribution is pathological or too narrow (e.g., nearly all ribbons have near-zero gradient energy), the ranges may be manually widened once after baseline inspection, then frozen for the sweep.

#### Penalty functions

* `fallback_penalty` = `1.0` if `used_fallback` is true, else `0.0`
* `center_offset_penalty` = `clip(center_offset / max_center_offset, 0, 1)`

#### Composite formula

```text
quality_score =
  0.35 * nonblack_fraction
+ 0.25 * normalized(gradient_energy)
+ 0.20 * normalized(row_variance_mean)
+ 0.10 * (1 - fallback_penalty)
+ 0.10 * (1 - center_offset_penalty)
```

Exact weights may be adjusted later, but the formula and normalization constants must remain fixed within a comparison batch.

### 9.5 Rule-Based Quality Buckets

Each ribbon with `status = success` must also receive one quality label:

* `good`
* `borderline`
* `bad_blank`
* `bad_geometry`
* `bad_low_signal`

These labels are for triage and reporting, not final truth. Bucket assignment logic must live in one central function in `score_ribbons.py` and bucket thresholds must remain fixed within a sweep, just like the score formula and normalization constants.

Ribbons with `status = failed` do not receive a quality bucket.

---

## 10. Status vs Quality Bucket Boundary

### 10.1 Definitions

* `status = failed` — the pipeline could not produce a valid ribbon artifact at all (e.g., download error, contour detection produced no output, warpPolar exception, output array has wrong dimensions).
* `status = success` — a ribbon array was produced and passes basic dimensional/format checks. The ribbon exists as a valid `.npy` file.

### 10.2 Rule

If a ribbon file exists and passes dimensional checks, mark `status = success` regardless of content quality. Then classify quality via the bucket system.

This preserves maximum diagnostic information. A 95% black ribbon is `status: success, quality_bucket: bad_blank`, not `status: failed`.

### 10.3 Failure Reasons

When `status = failed`, `failure_reason` must be set to one of:

* `download_error`
* `no_contour`
* `invalid_geometry` — reserved for future use if certain geometric states (e.g., radius below an absolute floor, degenerate contour) should hard-fail before unwrap rather than falling back
* `unwrap_exception`
* `invalid_dimensions`
* other descriptive string

---

## 11. Failure and Fallback Handling

### 11.1 Objective

Make bad geometry explicit instead of silently producing misleading outputs.

### 11.2 Required Behavior

For each coin, the runner must record:

* whether contour detection failed
* whether center/radius exceeded sanity limits
* whether fallback center/radius was used
* whether ribbon generation failed entirely

### 11.3 Required Fields

* `status`
* `failure_reason`
* `used_fallback`
* `fallback_reason`

### 11.4 Failure Philosophy

A run should prefer:

* explicit low-quality output with flags
* or explicit failure

over silently producing a ribbon that looks valid at the file level but is semantically wrong.

---

## 12. Run Artifact Layout

Each experiment run must write to an isolated directory.

### 12.1 Directory Structure

```text
runs/
  run_YYYYMMDD_HHMMSS_<config_id>/
    config.json
    ribbon_stats.csv
    summary.json
    ribbons/
    previews/
    galleries/
      best.png
      worst.png
      random.png
      by_authority.png
      fallbacks.png
      hard_cases.png
      vs_baseline.png
    debug/
      suspicious/
      fallback_samples/
      failures/
```

### 12.2 Required Files

#### `config.json`

Exact config used for the run.

#### `ribbon_stats.csv`

One row per coin with all stats and quality metadata.

#### `summary.json`

Aggregated run-level metrics.

---

## 13. `ribbon_stats.csv` Schema

Required columns:

* `coin_id`
* `authority`
* `ribbon_path`
* `preview_path`
* `status`
* `failure_reason`
* `quality_bucket`
* `quality_score`
* `nonblack_fraction`
* `nonzero_fraction`
* `mean_intensity`
* `std_intensity`
* `gradient_energy`
* `edge_density`
* `row_variance_mean`
* `cx`
* `cy`
* `radius`
* `center_offset`
* `mask_area_fraction`
* `used_fallback`
* `fallback_reason`
* `config_id`
* `known_hard_case`

---

## 14. `summary.json` Schema

Required fields:

* `run_id`
* `config_id`
* `n_total`
* `n_success`
* `n_failed`
* `n_good`
* `n_borderline`
* `n_bad_blank`
* `n_bad_geometry`
* `n_bad_low_signal`
* `fallback_rate`
* `mean_quality_score`
* `median_quality_score`
* `wall_clock_seconds`
* `mean_seconds_per_coin`
* `cache_hit_rate`
* `per_authority_stats`
* `hard_case_stats`

### 14.1 Per-Authority Stats

For each authority:

* `n`
* `mean_quality_score`
* `good_rate`
* `fallback_rate`
* `failure_rate`

### 14.2 Hard Case Stats

Breakout stats for coins where `known_hard_case = true`:

* `n`
* `mean_quality_score`
* `good_rate`
* `fallback_rate`

---

## 15. Review Artifact Generation

### 15.1 Objective

Replace giant undifferentiated galleries with targeted review artifacts.

### 15.2 Required Galleries Per Run

* `best.png` — top N by quality score (default N=20)
* `worst.png` — bottom N by quality score (default N=20)
* `random.png` — random sample of N ribbons (default N=20)
* `by_authority.png` — sampled per authority (default 5 per authority)
* `fallbacks.png` — all or sampled fallback cases
* `hard_cases.png` — all coins marked `known_hard_case = true`

### 15.3 Baseline Comparison Gallery

For candidate runs, generate:

* `vs_baseline.png`

This gallery must show the same coin under:

* baseline config
* candidate config

#### Coin Selection for Comparison

Show:

* top N coins with the largest positive quality score delta (improvements)
* top N coins with the largest negative quality score delta (regressions)
* optionally, a random middle sample

Default N=10 per category.

This artifact is critical for visually validating that the score improvement corresponds to genuinely better ribbons, and for catching regressions that the aggregate numbers might hide.

---

## 16. Baseline Establishment

### 16.1 Objective

Define one fixed baseline run so future candidates can be measured against it.

### 16.2 Procedure

1. Freeze `eval_subset_v1.csv`
2. Run current unwrap method unchanged
3. Save outputs as baseline
4. Calibrate normalization ranges for quality score from baseline metrics (p5/p95)
5. Record normalization constants
6. Do not overwrite baseline artifacts

### 16.3 Baseline Versioning

Each eval subset version gets its own baseline. Changing the eval subset invalidates the old baseline for formal comparison.

Naming convention:

* `eval_subset_v1.csv` → `baseline_eval_v1/`
* `eval_subset_v2.csv` → `baseline_eval_v2/`

A subset change always triggers a new baseline run and new normalization calibration.

### 16.4 Baseline Role

The baseline serves as the fixed comparison point for:

* score deltas
* fallback deltas
* visual deltas
* per-authority regressions

---

## 17. Parameter Sweep Procedure

### 17.1 Objective

Evaluate small groups of parameter combinations in a controlled way.

### 17.2 v1 Initial Sweep

First sweep should focus only on radial band placement with one center method.

Recommended sweep:

* `inner_r_ratio`: `[0.68, 0.72, 0.76]`
* `outer_r_ratio`: `[0.88, 0.90, 0.92]`
* `center_method`: `bbox`

Total: **9 runs**

### 17.3 Optional Second Sweep

If needed after the first sweep:

* keep best radial combination
* compare center methods:

  * `bbox`
  * `moments`
  * `min_enclosing_circle`

### 17.4 Sweep Constraints

Do not vary too many dimensions at once in v1. The goal is interpretable improvement, not broad hyperparameter search.

---

## 18. Run Ranking and Selection

### 18.1 Required Ranking Metrics

Runs must be ranked by:

* `good_rate`
* `mean_quality_score`
* `median_quality_score`
* `fallback_rate`
* `bad_blank_rate`
* worst-authority performance

### 18.2 Winner Selection Rule

A candidate is considered better than baseline if it:

* improves `good_rate`
* reduces `bad_blank_rate`
* does not materially worsen fallback rate
* does not cause severe authority-level regressions

### 18.3 Suggested Initial Acceptance Threshold

A candidate may be promoted if it satisfies all of:

* `good_rate` improves by at least 10 percent relative
* `bad_blank_rate` drops by at least 20 percent relative
* `fallback_rate` is equal or lower, or only slightly higher with strong visual improvement
* no authority collapses into consistently poor ribbons

These thresholds are heuristics, not statistical tests. At 300 coins, marginal improvements may be within noise. Acceptance decisions must be validated by paired visual review via the `vs_baseline.png` gallery.

---

## 19. Human Review Procedure

### 19.1 Objective

Keep human review focused and efficient.

### 19.2 Required Review Set

After automatic ranking, only visually inspect:

* top 3 candidate runs
* baseline
* baseline-vs-candidate galleries
* fallback galleries for the top 3 candidates
* hard case galleries for the top 3 candidates

### 19.3 Review Questions

For each top candidate, judge:

* are readable legends more common?
* are black wedge failures reduced?
* is the band more consistently on the legend ring?
* are authority-specific failures still present?
* are scores aligned with visual quality?
* do known hard cases improve, hold, or regress?

---

## 20. Module Structure

All harness modules live under `legend_cnn/`.

### 20.1 `legend_cnn/eval_subset.py`

Builds or refreshes `eval_subset_v<N>.csv` by stratified random sampling from `confusion_subset.csv`. Supports manual injection of `known_hard_case` entries.

### 20.2 `legend_cnn/unwrap_core.py`

Pure functions for:

* image loading (from cache or S3)
* mask extraction
* center/radius estimation (all three methods)
* polar unwrap
* ribbon generation

This is a refactored extraction from the existing `legend_cnn/unwrap.py`. The existing `unwrap.py` becomes a thin CLI wrapper that calls `unwrap_core` with hardcoded defaults for backward compatibility.

### 20.3 `legend_cnn/score_ribbons.py`

Computes per-ribbon metrics and quality buckets.

### 20.4 `legend_cnn/run_experiment.py`

Runs one config against one subset and writes all outputs.

### 20.5 `legend_cnn/sweep_experiments.py`

Runs a controlled grid of configs and aggregates summaries.

### 20.6 `legend_cnn/make_galleries.py`

Builds ranked galleries and comparison galleries.

### 20.7 `legend_cnn/compare_runs.py`

Compares candidate run summaries to baseline and emits a ranked report.

---

## 21. Logging and Reproducibility

### 21.1 Requirements

Each run must record:

* exact config
* subset file used (including version)
* timestamp
* code version if available (git hash)
* count of fallback cases
* count of failures
* count per quality bucket
* wall-clock runtime
* mean seconds per coin
* cache hit rate

### 21.2 Reproducibility Principle

Any run result should be reproducible from:

* config file
* subset file
* code at known state

---

## 22. Out-of-Scope Future Extensions

Potential future upgrades:

* adaptive annulus placement
* OCR-style textness scoring
* learned quality model
* authority-aware unwrap tuning
* active learning on flagged bad cases
* full-dataset orchestration after benchmark promotion

These are explicitly outside v1.

---

## 23. Immediate v1 Implementation Plan

### Phase 1 — Freeze Benchmark

* create `eval_subset_v1.csv` from `confusion_subset.csv`
* stratified random sample by authority
* manually inject known hard cases
* implement image caching layer

### Phase 2 — Refactor Unwrap

* extract `unwrap_core.py` from `unwrap.py`
* parameterize all geometry/mask/polar config
* implement all three center methods
* make `unwrap.py` a thin CLI wrapper over `unwrap_core`
* emit explicit fallback/failure metadata

### Phase 3 — Add Scoring

* compute per-ribbon metrics (content, structure, geometry)
* run baseline to calibrate normalization ranges
* freeze normalization constants
* assign quality score and bucket

### Phase 4 — Add Artifacts

* generate ranked galleries (best/worst/random/by_authority/fallbacks/hard_cases)
* generate baseline comparison gallery with delta-based coin selection

### Phase 5 — Run Initial Sweep

* execute 9-run radial band sweep
* rank runs automatically
* review top 3 visually

### Phase 6 — Promote Winner

* choose best candidate
* validate on larger sample before full subset use

---

## 24. Success Criteria for v1

v1 is successful if it produces a workflow where:

* unwrap changes can be compared apples-to-apples
* bad outputs are quantified rather than only noticed manually
* top candidate runs can be selected from metrics plus focused review
* the team can iterate on unwrap tuning quickly without rethinking the evaluation process each time

---

## 25. Summary

This harness converts legend ribbon tuning from an ad hoc manual loop into a repeatable experiment system.

The central principles are:

* fixed, versioned benchmark subset
* parameterized runs
* automatic proxy scoring with fixed normalization
* targeted review artifacts with delta-based comparison
* versioned baseline comparison
* explicit (heuristic) promotion criteria

That should let unwrap quality improve systematically before any downstream CNN training depends on it.
