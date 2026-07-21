# Two-coin weld: historical reprocess census and ticket proposal

> **Status: REPORT ONLY.** Nothing here has been reprocessed and nothing
> should be without a separate owner decision. Produced 2026-07-21 as step 5
> of the rollout brief. Companion to `specs/two_coin_weld_morph_close.md`.

The rollout brief asked for a count of previously-welded historical
detections per house, as a reprocess ticket proposal. This is that count,
plus the two things that turn a count into a decision: how much of each
house's welded population the chosen kernel would actually clear, and
whether the existing crops are bad enough to be worth reprocessing at all.

---

## 1. The census

Weld-caused detections are counted by their observable signature in the
database: `coin_detections.vision_metadata` carrying
`"split_method": "hough"`. Query is the one `tools/reprocess_hough.py`
uses to select its work set.

**779,165 detections across 114,346 lots; 228,450 of them (29.3%) are
Hough splits.**

| house | detections | Hough-split | rate | lots |
|---|---:|---:|---:|---:|
| leu | 256,170 | **106,850** | 41.7% | 52,337 |
| mashops | 169,471 | **37,432** | 22.1% | 18,703 |
| cng_feature | 34,116 | **29,136** | 85.4% | 14,568 |
| naumann | 48,476 | **22,586** | 46.6% | 11,293 |
| nomos | 27,592 | 10,024 | 36.3% | 4,501 |
| obolos | 83,714 | 9,674 | 11.6% | 4,571 |
| cng | 111,572 | 7,092 | 6.4% | 3,545 |
| gorny | 23,969 | 4,832 | 20.2% | 2,416 |
| davissons | 729 | 370 | 50.8% | 185 |
| stacksbowers | 2,720 | 208 | 7.6% | 104 |
| numisbids | 1,613 | 170 | 10.5% | 85 |
| kuenker | 7,149 | 76 | 1.1% | 38 |
| heritage, noonans, spink, artemide, Hirsch, Helbing, Cahn | 11,874 | 0 | 0.0% | 0 |

leu's 41.7% reproduces the handoff's corpus figure exactly. cng_feature's
85.4% matches §"Measured evidence"'s ~85%.

### Hough-split is a fair proxy for weld-caused, per the A/B

The census counts Hough splits, not the weld signature directly. The A/B
data says the two are near-identical in cause: driving the kernel to k=3
takes Hough to **0.0% on leu** and **1.0% on cng_feature**, so the close is
upstream of essentially the entire Hough population on both measured
houses. Treating the Hough-split count as the weld-caused count is
therefore sound to first order, and is what the numbers below assume.

---

## 2. What the chosen kernel would actually clear

Only two houses have measured kernels, and the clearance rate differs
because their operating points differ.

| house | chosen k | Hough before | Hough after | cleared | est. detections cleared |
|---|---|---|---|---|---:|
| cng_feature | 3 | 97.5% | 1.0% | 99% | **~28,800** |
| leu | 5 | 60.0% | 8.5% | 86% | **~91,900** |

**~120,700 detections across the two measured houses.** Note leu is
deliberately *not* at its maximum clearance: k=3 would take it to 0.0%, but
§4.6 rejected that because it triples true fragmentation. The remaining
8.5% is a cost the correctness argument buys.

### The other 40% of the welded population is unmeasured

The three houses in `CLOSE_KERNEL_BY_HOUSE` cover 136,062 of 228,450 Hough
splits — **59.6%**. The rest sit in houses with no A/B, no sweep, and no
override entry, so they would take whatever the global formula gives them.
Sampling their *welded* lots from Spaces across sales (n≈32/house, header
reads only):

| house | Hough-split | observed widths | `auto` would give |
|---|---:|---|---|
| mashops | 37,432 | 460–1700 (med 1232) | k=3 (84%), k=5 (16%) |
| naumann | 22,586 | 799–800 | k=3 (100%) |
| nomos | 10,024 | 775–1200 | k=3 (100%) |
| obolos | 9,674 | 1200–1762 | k=3 (97%), k=5 (3%) |
| cng | 7,092 | 500–3000 (med 1499) | k=3 (53%), k=5 (9%), k=7 (38%) |
| gorny | 4,832 | 1200–1600 | k=3 (88%), k=5 (12%) |
| davissons | 370 | 370 | k=3 (100%) |
| stacksbowers | 208 | 988–4796 (med 4736) | k=3 (12%), k=7 (12%), **k=11 (75%)** |
| numisbids | 170 | 803–1228 | k=3 (100%) |

Two things to flag:

1. **k=3 is the default outcome for most unmeasured houses.** k=3 is the
   setting §4.6 examined most closely and *rejected* for leu, on
   fragmentation grounds. Applying it untested to ~85,000 welded detections
   across mashops/naumann/nomos/obolos/gorny is the largest unquantified
   risk in this rollout.
2. **stacksbowers would get k=11, and kuenker's largest plates k=9** —
   *more* bridging than they have today (20px and 16px reach against
   today's 12px). That is the scale-relative formula working as designed on
   4000px+ input, not a bug, but it is an unmeasured behaviour change in
   the welding direction on houses that were never the problem.

---

## 3. Is reprocessing justified? The evidence points yes, but weakly

§6.5 sets the precondition: backfill is worth it **only if the current
crops actually carry slivers**. If Hough is producing clean crops, backfill
buys nothing but risk. Three measurements bear on this, and they do not all
agree.

- **Against reprocessing.** §4.5 measured Hough-arm crop quality directly
  and found it acceptable; per-house GREEN rates are healthy (leu 90.8%,
  cng_feature 84–88%).
- **For reprocessing.** §4.5's tight-rect IoU shows the *Hough* arm is the
  one placing coins with overlap: 67.0% of Hough lots are under the 0.02
  disjointness bar against 99.0% for the threshold path, median IoU 0.0079
  vs 0.0000. §7.1 feared removing the weld would regress to slivers; the
  geometry says the opposite.
- **For reprocessing, visually.** The montages in
  `two_coin_weld_maskgate_cng_feature_20260721_montage/` show the
  mechanism: Hough fits a *circle* to an irregular ancient flan, clipping
  rim on chipped or oval coins, while the threshold contour traces the
  actual outline. The circle is a worse mask for exactly the coins
  numismatic embeddings care most about.

**Recommendation: do not open a bulk reprocess ticket yet.** The case is
real but it is a *quality improvement* argument, not a *defect* argument,
and it is not what §6.5's precondition asked for. The cheaper decision
first is whether crop quality improves enough to justify 120,700 rewrites
that also invalidate the corresponding embeddings (§7.4). That is
answerable on a few hundred lots — compare embedding-relevant crop metrics
between arms on the frozen samples — and should precede any bulk run.

If a ticket is opened anyway, scope it to **cng_feature first** (~28,800
detections, highest weld rate, smallest blast radius, and the house whose
kernel is most firmly measured), and only then consider leu's ~91,900.

---

## 4. Hard constraints carried forward

- Crops are overwritten **in place at the same Spaces keys** (§3), so a
  reprocess is not trivially reversible and the pre-change output is gone.
- Reprocessing changes crops, which changes embeddings (§7.4). Any ticket
  must include the embedding re-index, or it ships a corpus where crops and
  embeddings disagree.
- None of this is actionable until the per-house table actually reaches
  Layer 1 in production — see the blocker in §6.6 of the main spec.
