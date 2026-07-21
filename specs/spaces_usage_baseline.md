# Spaces Usage Baseline (trivalaya-data)

Measurement snapshot of the production Spaces bucket. Captured to
ground storage-related specs (notably `transparent_png_resize.md`)
in real numbers instead of back-of-envelope guesses.

Owner: vision / pipeline plumbing
Status: measurement — re-run periodically
Date: 2026-05-27 (initial scan)
Scan duration: 806 s (full bucket, 2.96M keys)

---

## Headline

- **Total: 880 GB across 2.96M keys**
- **Transparents alone: 502 GB (57% of the bucket) across 589k objects**
- HTML pages (`raw/pages/`) are only 2.5 GB — they are NOT the bulk
- Tarballs + backups: 26 GB total

---

## 1. By top-level prefix

| prefix | size | objects | what it is |
|---|---|---|---|
| `processed/` | 695 GB | 2.44M | masked PNGs + crops + thumbnails + vision artifacts |
| `raw/` | 158 GB | 518k | raw auction images + collection sources + scraped HTML |
| `tarballs/` | 15 GB | 2 | training tarballs (single big object dominates) |
| `backups/` | 11 GB | 12 | DB / config backups |
| `exports/` | 0.7 GB | 8 | data exports (JSON, mostly) |
| `results/` | 0.4 GB | 3 | npy artifacts |
| `ml/` | <0.1 GB | 4 | ML model / dataset pointers |
| everything else | <0.1 GB | — | — |

---

## 2. `processed/` breakdown (the 695 GB)

| type | count | size | avg | notes |
|---|---|---|---|---|
| `*_transparent.png` | 589k | **502 GB** | 833 KB | alpha-masked PNGs — DINOv2 input |
| `*_crop.jpg` | 584k | 110 GB | 188 KB | rectangular JPG crops, same coin count |
| other `.jpg` | 1.27M | 83 GB | 64 KB | mostly thumbnails |

Layout:
- Current pipeline writes under `processed/vision/v1/crops/<house>/<sale>/<lot>/...`
- A very small legacy population (1.8k transparents, 0.4 GB) sits under
  `processed/auctions/artemide/40/...` at tiny dimensions (~300px,
  ~190 KB each). Old; can be migrated or left for GC.

---

## 3. `raw/` breakdown (the 158 GB)

| subprefix | size | objects | obv/rev pairs | notes |
|---|---|---|---|---|
| `raw/auctions/` | 105 GB | 290k | — | 2-coin lot photos (single image per lot) |
| `raw/ocre/` | 32 GB | 24k | 12k + 12k | OCRE collection, obv/rev split at source |
| `raw/pella/` | 10 GB | 5.7k | 2.9k + 2.9k | Pella collection |
| `raw/sco/` | 4.8 GB | 4.4k | 2.2k + 2.2k | SCO collection |
| `raw/crro/` | 2.7 GB | 2.6k | 1.3k + 1.3k | CRRO collection |
| `raw/pages/` | 2.5 GB | 188k | — | scraped HTML auction pages |
| `raw/pco/` | 0.4 GB | 3.2k | 1.6k + 1.6k | **PCO** — small today; target of `pco_transparent_generation.md` |
| `raw/uploads/` | trace | 1 | — | — |

---

## 4. Transparent PNG size histogram

Pulled across all 589k transparents (`processed/.../_transparent.png`):

| bucket | count | % of count | size | % of bytes | implied dims |
|---|---|---|---|---|---|
| 0–100 KB | 24,641 | 4.2% | 1.0 GB | 0.2% | tiny (~200px) |
| 100–200 KB | 17,066 | 2.9% | 2.6 GB | 0.5% | ~300px |
| 200–300 KB | 13,941 | 2.4% | 3.6 GB | 0.7% | ~370px |
| 300–400 KB | 17,687 | 3.0% | 6.4 GB | 1.3% | ~430px |
| 400–500 KB | 36,567 | 6.2% | 17.2 GB | 3.4% | ~500px (borderline) |
| 500–750 KB | 169,353 | **28.8%** | 108.5 GB | **21.6%** | ~600px |
| 750–1000 KB | 214,730 | **36.5%** | 184.9 GB | **36.8%** | ~700px |
| 1000–1500 KB | 54,508 | 9.3% | 66.6 GB | 13.3% | ~825px |
| 1500–2000 KB | 16,244 | 2.8% | 28.4 GB | 5.7% | ~1000px |
| 2000–3000 KB | 14,975 | 2.5% | 37.4 GB | 7.4% | ~1180px |
| 3000–5000 KB | 6,232 | 1.1% | 23.4 GB | 4.7% | ~1450px |
| 5000–10000 KB | 2,525 | 0.4% | 16.9 GB | 3.4% | ~1940px |
| 10000+ KB | 322 | <0.1% | 5.3 GB | 1.1% | ~3000px+ |

**Generation split:**
- New pipeline (`processed/vision/v1/`): 587k objects, 502 GB
- Old (`processed/auctions/artemide/40/`): 1,823 objects, 0.4 GB

**Empirical compression ratio for these alpha-masked PNGs:**
~**1.75 KB per 1000 pixels**. Coin engravings + alpha channel
produce high-entropy content; PNG of typical photographic content
runs 1-3 KB/k_pixel, so this is in-range. A 518×518 RGBA PNG of a
coin lands at **~470 KB**, not the 150-300 KB the resize spec
initially assumed.

---

## 4b. Raw and crop size distributions

Full percentile dump for raws and crops (used by the
outlier-resize policy in `transparent_png_resize.md`'s Related section):

| population | n | total | mean | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| `raw/auctions/` | 289,470 | 105 GB | 353 KB | 274 KB | 357 KB | 681 KB | 813 KB | 1.58 MB | **47.7 MB** |
| `processed/.../_crop.jpg` | 584,130 | 110 GB | 183 KB | 166 KB | 204 KB | 272 KB | 373 KB | 708 KB | 4.7 MB |

**Dimension samples at percentiles** (raws):

| pct | size | dims | key |
|---|---|---|---|
| p50 | 274 KB | 1200 × 588 | `raw/auctions/gorny/1254/Lot_00258.jpg` |
| p75 | 357 KB | 1200 × 598 | `raw/auctions/leu/46/Lot_06302.jpg` |
| p90 | 681 KB | 1200 × 592 | `raw/auctions/leu/13/Lot_01100.jpg` |
| p95 | 813 KB | 1413 × 703 | `raw/auctions/obolos/21/Lot_00794.jpg` |
| p99 | 1.58 MB | 1200 × 574 | `raw/auctions/nomos/25/Lot_00057.jpg` |

(Note: p99 raw is at 1200×574 — same dims as p50 but 5.8× the bytes.
The byte-tail is content-entropy-driven, not dimension-driven, for
some fraction of the upper tail.)

**Dimension samples at percentiles** (crops):

| pct | size | dims | key |
|---|---|---|---|
| p50 | 166 KB | 666 × 580 | `…/leu/29/Lot_00090/leu_29_00090_rev_crop.jpg` |
| p75 | 204 KB | 660 × 578 | `…/mashops/cgb/Lot_387480/…_obv_crop.jpg` |
| p90 | 272 KB | 894 × 853 | `…/mashops/noel/Lot_86694/…_obv_crop.jpg` |
| p95 | 373 KB | 988 × 988 | `…/cng/EA-608/Lot_00029/cng_EA-608_00029_01_crop.jpg` |
| p99 | 708 KB | 1520 × 1415 | `…/noonans/710/Lot_01108/noonans_710_01108_obv_crop.jpg` |

**Empirical compression for JPGs:**
- Raws: ~0.3-0.5 KB per 1000 pixels (with significant variance —
  some sources use higher JPG quality)
- Crops: ~0.3-0.5 KB per 1000 pixels (same ballpark)

---

## 4c. Heavy-tail outliers (raws)

The raw byte distribution has a long tail. The single largest raw is
**47.7 MB** — two orders of magnitude above the p50 (274 KB). The
top 15 by size:

```
 48.90 MB  raw/auctions/obolos/17/Lot_00960.jpg
 37.21 MB  raw/auctions/obolos/17/Lot_00923.jpg
 37.01 MB  raw/auctions/obolos/19/Lot_01178.jpg
 36.21 MB  raw/auctions/obolos/19/Lot_01180.jpg
 34.85 MB  raw/auctions/obolos/27/Lot_00931.jpg
 34.07 MB  raw/auctions/obolos/27/Lot_00938.jpg
 33.68 MB  raw/auctions/kuenker/72/Lot_00407.jpg
 33.65 MB  raw/auctions/obolos/17/Lot_00953.jpg
 31.87 MB  raw/auctions/obolos/19/Lot_01205.jpg
 31.73 MB  raw/auctions/obolos/19/Lot_01175.jpg
 31.37 MB  raw/auctions/obolos/19/Lot_01185.jpg
 30.81 MB  raw/auctions/obolos/27/Lot_00936.jpg
 30.66 MB  raw/auctions/obolos/27/Lot_00937.jpg
 30.00 MB  raw/auctions/obolos/17/Lot_00938.jpg
 29.82 MB  raw/auctions/obolos/27/Lot_00941.jpg
```

**14 of the top 15 are from obolos**, concentrated in sales 17, 19,
and 27. One kuenker entry breaks the pattern.

**Inspection of the largest (`obolos/17/Lot_00960.jpg`):**

| field | value |
|---|---|
| dimensions | 10,549 × 3,506 px (37 MP) |
| bytes | 48.9 MB (1.3 MB/MP — high JPG quality) |
| camera | Nikon D5600 + 60mm f/2.8G macro |
| capture date | 2020-10-09, UTC+9 (Japan) |
| pipeline | `Camera Control Pro 2.25.1` → `Affinity Photo` (Nov 2020) |
| color | Adobe RGB, ICC `Nikon Adobe RGB 4.0.0.3001` |
| original camera output | 6000 × 4000 (Nikon "L" size) |
| DPI tag | 7194 (nonsense — Affinity metadata quirk, ignore) |

The on-disk width (10,549) exceeds the camera's native width (6000),
so this is a **stitched composite** — multiple captures joined into
one wide plate. Likely an obv+rev pair stitched side-by-side, or a
multi-coin reference plate.

**Implications:**

- **Not corruption.** These are genuine high-resolution coin
  photographs, intentionally produced.
- **Concentrated population.** A targeted "resize all raws in
  `raw/auctions/obolos/{17,19,27}/`" pass would knock out most of
  the heavy tail without scanning the rest of the bucket.
- **Recoverable.** The auction house presumably has the originals;
  resizing is not irreversibly destructive to the workflow.
- **L1 re-validation needed.** These are multi-coin/stitched plates,
  not standard 1200×~600 2-coin lot photos. The L1 two-coin
  resolver may behave differently on stitched-multi inputs. Any
  source-image resize spec must include 5+ obolos plates in its
  pilot to confirm L1 still produces sensible crops at the resized
  resolution.
- **Outlier-cap math** (resize longest-side from ~10,500 to ~2,400 —
  4× the mean — would still preserve more pixels than the median
  raw while reducing this one image from 48.9 MB to ~3 MB). Even
  conservative resize policies have huge yield on this tail.

When the source-image resize spec is opened, treat this section as
the starting brief on the heavy tail.

---

## 5. Dimension samples by auction house (raw vs new-pipeline crop)

Per-house sampling, one object each (illustrative — not statistical):

| house | raw dims | raw size | crop dims | crop size |
|---|---|---|---|---|
| artemide | 600 × 287 | 70 KB | (no crops in new pipeline) | — |
| cng | 500 × 247 | 34 KB | 217 × 217 | 26 KB |
| cng_feature | 500 × 234 | 31 KB | 272 × 234 | 28 KB |
| davissons | 370 × 185 | 36 KB | 188 × 185 | 16 KB |
| gorny | 1200 × 587 | 209 KB | 627 × 587 | 154 KB |
| kuenker | 800 × 397 | 123 KB | 439 × 397 | 83 KB |
| leu | 1200 × 613 | 223 KB | 655 × 613 | 163 KB |
| mashops | 1700 × 806 | 491 KB | 939 × 806 | 432 KB |

Cross-listing spread samples (noonans, obolos) produced raws up to
1989 × 905 (1126 KB) — these are the outliers feeding the 5–10 MB
transparent bucket.

**Crops are half-width of source raw at the same height** —
single-coin partition of a 2-coin lot photo, at source resolution
(no downscaling). This is structurally important; see §6.

---

## 6. "Why are crops the same total size as raws?"

Aggregate:
- 290k raw lots × 362 KB avg = **105 GB**
- 290k lots × 2 crops × 188 KB avg = **109 GB**

The 4% excess looks anomalous at first glance — crops are half the
pixels of raws but the byte totals match. Two compounding effects:

**(a) The discarded half is mostly free in JPG terms.** A raw lot
photo is two coins on a near-uniform background. JPG compresses
uniform regions to almost nothing; the bulk of the raw's bytes live
in the coin pixels (engravings, fields, edges = high entropy).
Cropping discards half the pixels but only a small fraction of the
bytes.

Concrete (mashops sample):
- Raw 1700 × 806, 491 KB → 0.36 KB per 1000 pixels
- One crop 939 × 806, 432 KB → 0.57 KB per 1000 pixels
- Per pixel, the crop is 1.6× more byte-dense than the raw —
  confirming the discarded pixels were the cheap ones.

**(b) JPG re-encoding inflates.** Decoding a JPG and re-encoding at
any quality produces slightly more bytes because the new encoder has
to represent the previous compression's artifacts as part of the
image content.

(a) and (b) together explain the 4% excess. Not a bug — JPG entropy
economics.

**Storage lever this exposes:** crops at source resolution are
themselves a separately-attackable ~110 GB. They serve only as the
rectangular fallback in `cluster_coins.open_image_masked` when no
transparent exists. A 518-cap on crops would save another ~75 GB,
but it's out of scope for the current transparent-resize spec.

---

## 7. Realistic savings from `transparent_png_resize.md`

For each histogram bucket, output PNGs after 518-cap converge to
**~470 KB** (518² × 1.75 KB/k_pixel). Bucket-by-bucket projection:

| bucket | objects | current GB | post-resize GB | savings GB |
|---|---|---|---|---|
| 0–500 KB (≤~500px) | 110k | 30.8 | 30.8 (no-op) | 0 |
| 500–1000 KB (~600-700px) | 384k | 293.4 | ~180 | ~113 |
| 1000–2000 KB (~825-1000px) | 71k | 95.0 | ~33 | ~62 |
| 2000+ KB (~1200px+) | 24k | 82.9 | ~11 | ~72 |
| **Total** | **589k** | **502** | **~255** | **~247** |

**Net: 502 GB → 255 GB, savings ~247 GB (~49% of the prefix, ~28%
of the total bucket).**

This is the corrected projection — earlier "10.5 TB" and "~380 GB"
estimates in the resize spec were wrong because:
1. Object count was over-estimated (4M assumed; actual 589k).
2. Per-object size was over-estimated (3 MB assumed; actual avg 833 KB).
3. Post-resize size was under-estimated (150-300 KB assumed; actual ~470 KB).

The corrected resize spec uses these numbers.

---

## 8. Re-running this measurement

```bash
# Full bucket scan, type histogram (806s, no GET requests):
python3 /tmp/spaces_usage_scan.py

# Transparent-only size histogram:
python3 /tmp/transparent_histogram.py

# Raw + crop size stats with dimension samples at p50/p75/p95/p99:
python3 /tmp/raw_crop_histogram.py
```

Outputs land in `/tmp/spaces_usage_report.json`,
`/tmp/transparent_histogram.json`,
`/tmp/raw_crop_histogram.json`. Rebuild this doc from those JSONs
when the bucket grows materially (e.g., +10% by either count or
bytes) or after a major migration (e.g., the resize bulk run lands).

---

## 9. Related specs

- `specs/transparent_png_resize.md` — uses §4 and §7 numbers; targets
  the 502 GB transparent prefix
- `specs/pco_transparent_generation.md` — references PCO source
  volumes from §3 (3.2k objects, 0.4 GB)
