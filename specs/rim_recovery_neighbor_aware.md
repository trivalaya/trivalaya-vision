# Neighbor-aware rim recovery

> **Status: QUEUED — successor to the two-coin weld rollout.** Owner
> decision 2026-07-21: same vision-quality lane, picked up when the kernel
> rollout closes. Not a parallel front. Nothing here is implemented.

## The defect

Layer 1.5 rim recovery can produce a coin mask that swallows part of its
neighbour, and nothing in the pipeline can currently reject it.

When a candidate's circularity is below `CIRCULARITY_RELAXED` (0.65) and its
area fills less than 85% of its `minEnclosingCircle`, `recover_rim` fits a
circle and **replaces the true contour with it**
(`src/layer1_geometry.py:271`, `final_c = new_c`). That contour is what
`crop_with_alpha` bakes into the alpha channel, so it is what reaches the
embedding.

The guard is `math_utils.validate_rim_recovery(recovered_contour,
seed_contour, image_shape)`. Its four checks are:

1. basic contour validity (`min_area=100`)
2. bounding box within 1.1× the image
3. recovered centroid within 30% of the seed's bbox size
4. recovered area ≥ 90% of the seed area

**Every one is self-referential.** The signature carries no information
about any other candidate, so the function cannot express "this rim now
overlaps the coin next to it". A rim that expands sideways into a
neighbouring flan passes all four checks cleanly.

## Measured evidence

From §4.7 of `two_coin_weld_morph_close.md`, the first run of §9.3c option
2b. Overlap is undilated — filled contour against filled contour, i.e. real
alpha contamination, measured as a fraction of the neighbour's area.

| lot | house | arm | overlap | rim_recovered |
|---|---|---|---:|---|
| 3717 | leu | control (k=7) | **17.8%** | 3 of 5 detections |
| 3736 | leu | control (k=7) | **10.6%** | 3 of 3 detections |
| 3661 | leu | control (k=7) | 3.3% | 1 of 3 |
| 3661 | leu | auto (k=5) | 1.7% | 1 of 3 |
| 995 | leu | auto (k=5) | 0.54% | 2 of 2 |
| 582 | leu | both | ~0.45% | 1 of 2 |
| 215298 | cng_feature | auto (k=3) | 1.1% | 2 of 2 |

Every lot with real overlap has at least one `rim_recovered=True`
detection, in whichever arm carries it. Lots that show overlap only under a
3px dilation have `rim_recovered=False` and are clean undilated.

**This is a defect in production today**, not one introduced by the kernel
change: the two worst cases are both in the `control` arm, which is the
fixed 7×7 that production runs right now.

## Why the kernel change is not the fix

Changing the MORPH_CLOSE kernel only alters how many blobs Layer 1.5 is
handed and how well separated they are. On lots 3717 and 3736 that happens
to take the overlap to zero, but incidentally — the k=5 segmentation gives
rim recovery better-separated seeds, so its fitted circles land in a
kinder configuration. Nothing prevents the same failure at k=5 on a
differently-shaped lot, and lot 995 shows the `auto` arm producing 0.54%
overlap where control produced none.

A durable fix helps **both** arms and is independent of the kernel.

## Sketch of the fix

Add neighbour awareness to the accept/reject decision. Two design notes
that matter:

- **It needs a second pass.** The candidate loop in
  `_segment_and_extract_candidates` appends as it goes, so at the moment
  `validate_rim_recovery` is called, later candidates do not exist yet. A
  neighbour-aware check belongs *after* the loop, before NMS: for each
  recovered candidate, compare its filled contour against the filled
  contours of all other candidates and fall back to the original seed
  contour if the overlap exceeds a threshold.
- **Falling back must be possible.** That means keeping the pre-recovery
  seed contour on the candidate (it is currently discarded when `final_c`
  is reassigned), so rejection is a revert rather than a re-run.

Threshold should be measured, not guessed — the §4.7 data suggests real
contamination starts being visible around 0.5% of the neighbour's area, but
that is 7 lots, and the rule this project keeps relearning is that
constants come from sweeps.

## Acceptance

- Re-run `tools/two_coin_weld_mask_gate.py` on both frozen samples. The
  undilated `contour` overlap should go to zero in **both** arms — including
  control, since the fix is kernel-independent.
- No regression in detection count or GREEN rate on the frozen samples;
  rim recovery exists because it genuinely rescues fragmented coins, and a
  fix that simply stops recovering rims would trade one defect for another.
- A synthetic fixture in the §9.2 tier: two adjacent low-circularity
  fragments where the naive fit overlaps and the neighbour-aware one does
  not.

## Related

- `specs/two_coin_weld_morph_close.md` §4.7 (the measurement), §7.1 (the
  original sliver fear, now answered), §6.5 (backfill precondition, which
  this revises — Hough crops are *not* clean on leu today)
- `specs/two_coin_weld_reprocess_proposal.md` — the historical population
  that would be affected if this is fixed and a backfill follows
