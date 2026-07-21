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

## 2.5 Accepted: the mask-IoU drift bar — owner decision 2026-07-21

The mask-IoU gate (§4.7) is **accepted**, and the acceptance is recorded
here because it is the same ledger the reprocess decision draws on.

What was measured, on leu's 52 byte-identical-outcome lots:

| | |
|---|---|
| worst genuine alpha drift | **1.1%** |
| median | 0.14% |
| lots above the 0.995 bar | 8 of 52 |
| the 0.889 minimum | **excluded** — a segmentation failure, not a crop shift |

Accepted on the grounds that 1.1% is an order of magnitude inside the drift
band that parked dp=2.0, and — unlike dp=2.0 — the other side of the ledger
carries a *measured* benefit: 10–18% contamination removed on leu's worst
lots (§4.7's undilated sliver table). Drift of this size against a
correctness gain of that size is a trade worth making.

Two instrument-discipline notes attach to this acceptance, both cases where
the measurement corrected its own author:

- **Lot 3679's "11% drift" was not drift.** It is a 723,941px blob at
  circularity 0.337 covering 88% of the frame — L1 failing to segment, not a
  crop moving. Caught only by checking detection geometry; alpha IoU alone
  cannot distinguish the two. Screen for degenerate detections before
  quoting a tail.
- **The kuenker width range was sampled wrong.** Ordering lots by
  `sale_id, lot_number` and taking every Nth concentrates in
  alphabetically-early sales, which produced a false 417–2000px range and a
  false claim that the spec was wrong. Real range is 408–3381px.

Both are retracted in place rather than silently fixed, and both are the
reason this acceptance is worth trusting.

## 3. Is reprocessing justified? The evidence changed — it is now yes

§6.5 sets the precondition: backfill is worth it **only if the current
crops actually carry slivers**. If Hough is producing clean crops, backfill
buys nothing but risk.

**That precondition is now measured, and it is satisfied.** §9.3c option 2b
was run for the first time on 2026-07-21. Undilated, on leu:

| lot | control (production today, k=7) | auto (k=5) |
|---|---:|---:|
| 3717 | **17.8%** of the neighbouring coin | **0** |
| 3736 | **10.6%** | **0** |
| 3661 | 3.3% | 1.7% |

Production is carrying real alpha contamination on leu right now — up to
17.8% of a coin's area filled with its neighbour — and the measured kernel
removes the two worst cases entirely. This is no longer a "quality
improvement" argument. It is a defect argument, which is exactly what §6.5
asked for.

Supporting evidence points the same way:

- §4.5's tight-rect IoU already showed the *Hough* arm placing coins with
  overlap: 67.0% of Hough lots under the 0.02 disjointness bar against
  99.0% for the threshold path.
- The montages in `two_coin_weld_maskgate_*_montage/` show the mechanism —
  Hough fits a *circle* to an irregular ancient flan, clipping rim on
  chipped and oval coins, while the threshold contour traces the outline.
- Per-house GREEN rates (leu 90.8%, cng_feature 84–88%) do **not**
  contradict this. GREEN does not measure neighbour contamination, which is
  why the sliver check had to be built.

### But sequence it behind the rim-recovery fix

The contamination mechanism is **Layer 1.5 rim recovery**, not the kernel
(`specs/rim_recovery_neighbor_aware.md`). The kernel change removes the two
worst lots incidentally, by handing rim recovery better-separated seeds — it
does not make the failure impossible, and lot 995 shows `auto` producing
0.54% overlap where control produced none.

Reprocessing 120,700 detections *before* that fix would bake a
still-defective mask into a new generation of crops and embeddings, and the
crops are overwritten in place, so there is no second chance at the same
keys.

**Recommendation, revised: open the ticket, but sequence it third.**

1. Enable the per-house kernel (in progress) — new lots only, no backfill.
2. Land neighbour-aware rim validation, and re-run the sliver gate until
   undilated overlap is zero in **both** arms.
3. *Then* backfill, scoped to **cng_feature first** (~28,800 detections,
   highest weld rate, smallest blast radius, firmest kernel evidence),
   and only afterwards leu's ~91,900.

Reprocessing at step 3 buys clean crops. Reprocessing now buys a different
set of dirty ones.

---

## 4. Hard constraints carried forward

- Crops are overwritten **in place at the same Spaces keys** (§3), so a
  reprocess is not trivially reversible and the pre-change output is gone.
- Reprocessing changes crops, which changes embeddings (§7.4). Any ticket
  must include the embedding re-index, or it ships a corpus where crops and
  embeddings disagree.
- None of this is actionable until the per-house table actually reaches
  Layer 1 in production — see the blocker in §6.6 of the main spec.
