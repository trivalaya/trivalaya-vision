# Baseline Review — run_20260410_002707_baseline

**Date:** 2026-04-10
**Subset:** eval_subset_v1.csv (300 coins, 30/authority, seed=42)
**Config:** baseline (bbox, ir=0.72, or=0.90)

## Summary Numbers

- 300/300 success (0 failed)
- 84 good (28%), 59 borderline (20%), 157 bad_geometry (52%)
- 0 bad_blank, 0 bad_low_signal
- Fallback rate: 45.7% (all center_out_of_bounds)
- Mean quality score: 0.566, median: 0.561

## Are the buckets visually believable?

**Partially.** The `good` vs `bad_geometry` split tracks geometry
reliability, not visual quality. This creates a known paradox:

- The top-scored `bad_geometry` ribbons (q=0.78) show **clear, legible
  legend text** — better than many `good` ribbons. They're flagged solely
  because the fallback center was used.
- The boundary `good` ribbons (q=0.55) have significant black wedge
  artifacts and partial text. They're `good` only because their
  non-fallback center happened to pass the offset threshold.
- 47% of `bad_geometry` ribbons score within the `good` score range.

**Bottom line:** the quality_score correctly ranks ribbons by visual
usability. The bucket system correctly flags geometry reliability.
These are not the same thing — a ribbon can be visually usable despite
unreliable geometry. For sweep evaluation, `mean_quality_score` is a
better measure of actual ribbon quality than `good_rate`.

## Fallback gallery

Fallback cases are diverse: some show excellent text, some show
coin portrait or massive artifacts. The fallback center (256,256,r=220)
works well when coins happen to be centered but fails badly when
they're not. This is the core problem the sweep should address.

## Authority variation

Caracalla leads at 57% good rate; Hadrian worst at 13%.
Likely driven by coin size/centering variation across authorities.

## Normalization ranges

- gradient_energy: [0.095, 0.702] — healthy 7x spread
- row_variance: [0.010, 0.158] — healthy 16x spread
- ~10% clipping at each tail (expected for p5/p95)
- Not pathologically narrow. Acceptable for sweep use.

## Implications for sweep

1. The 9-run radial sweep will mostly shift ribbons between
   borderline and good — the geometry problem dominates.
2. The second sweep (center methods) is likely more impactful
   because reducing the 46% fallback rate will move large numbers
   of ribbons from bad_geometry to good/borderline.
3. When interpreting sweep results, watch `mean_quality_score`
   and `fallback_rate` together, not just `good_rate` alone.

## Verdict

**Baseline accepted.** Norms frozen. Artifacts immutable.
