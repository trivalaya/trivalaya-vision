# Promotion Note: promoted_v1

**Date:** 2026-04-10
**Promoted config:** `configs/promoted_v1.json`

## Parameters

| Parameter | Baseline | Promoted | Why |
|---|---|---|---|
| inner_r_ratio | 0.72 | 0.68 | Wider inner margin captures more legend band. Monotonic improvement across sweep. |
| outer_r_ratio | 0.90 | 0.92 | Slight gain. Outer ratio is second-order. |
| center_method | bbox | moments | Negligible at offset=150, but outperforms bbox at offset=200. |
| max_center_offset | 150 | 200 | Reduces fallback rate without degrading mean quality. |

## Evidence

### Radial sweep (9 configs)
- inner_r_ratio is the dominant lever. 0.68 > 0.72 > 0.76 monotonically.
- outer_r_ratio is weak and consistent: 0.92 >= 0.90 >= 0.88.
- Best radial: inner=0.68, outer=0.92 (+0.012 mean score vs baseline).

### Center-method sweep (3 configs, offset=150)
- All three methods produce ~46% fallback rate at offset=150.
- moments marginally better (+0.006), min_enclosing_circle slightly worse.
- Conclusion: center method is masked by the strict offset gate.

### Offset sweep (5 configs)
- offset=225: fallback drops 13pp but mean score drops. Bad tradeoff.
- offset=200 bbox: fallback drops 9pp, score gain only +0.004. Mediocre.
- **offset=200 moments: fallback drops 8.4pp AND score +0.012. Best of both.**
- offset=175: score matches but only recovers 2.7pp fallback.

### Why moments wins at offset=200

41 coins where moments beats bbox (>0.02 delta) vs only 3 losses.
The wins come from two mechanisms:
1. Moments centroid is more accurate for irregular contours (non-fallback wins).
2. Moments correctly identifies unreliable centers and falls back, while
   bbox lets marginal centers through (fallback-mediated wins).

### Known regressions

22 coins regress >0.05 vs baseline. All 22 follow one pattern:
baseline used fallback center (256,256) and got a decent result;
promoted config uses the actual moments center (avg offset 183px),
producing a worse ribbon. These are coins where the real contour
center is genuinely far off but the image-center fallback happened
to work better.

This is an acceptable tradeoff: 22 regressions vs 41 per-coin
wins over bbox alone, and the mean score improves overall.

## Metrics

| Metric | Baseline | Promoted |
|---|---|---|
| Mean quality score | 0.566 | 0.578 |
| Good rate | 54.7% | 64.0% |
| Fallback rate | 45.7% | 37.3% |
| Geometry flags | 52.3% | 52.0% |
| Bad blank | 0 | 0 |

## Decision

Promoted. This is the first config that improves both visual quality
and process robustness simultaneously. The regressions are understood,
localized, and outnumbered by improvements.
