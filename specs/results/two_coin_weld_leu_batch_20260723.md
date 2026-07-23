# Forced leu/75 200-lot batch — §6.8 closure + Scope B production validation (2026-07-23)

Step 2 of the owner-approved 07-23 bundle (brain memory `two-coin-weld-kernel-lane` /
`rim-recovery-lane`): one bounded production write, closing two open items —
the weld-kernel lane's §6.8 next-day spot check (open since 2026-07-21,
NO DATA on 07-22 and 07-23) and Scope B's (`TRIVALAYA_RIM_NEIGHBOR_GUARD`)
production enable validation.

## What ran, and how

**Mechanism (revised from the original prompt after a conflict was found and
the owner chose the fix):** a genuine `pipeline_jobs` `sale_ingestion` job
for leu/75 would re-scrape all 3,754 lots of that sale against the live
leunumismatik.com (confirmed: `trivalaya-data/scraper/scraper.py` has no
skip-if-already-scraped check) — unscoped, wasteful, and a direct violation
of "scoped to those 200 leu lots and nothing else." Instead: the 200 frozen
sample lots (`trivalaya-vision/specs/two_coin_weld_sample_ids.csv`,
`purpose=leu_ab`) were reset to `vision_processed=0` (their `coin_detections`/
`coins`/`ml_coin_dataset` rows deleted after a full preimage dump), then run
through `python -m trivalaya_pipeline vision --source leu/75` — the exact
same `Pipeline.process_vision()` production code the runner and the nightly
cron both call, with the real `record.auction_house` → `_close_kernel_size`
plumbing (§6.6), NOT `analyze_image` called directly. No scrape, no forced
full-catalog export.

1. **Enable (runner only):** `.env` backed up (`.env.bak.20260723`),
   `TRIVALAYA_RIM_NEIGHBOR_GUARD=1` added alongside the existing
   `TRIVALAYA_CLOSE_KERNEL_FRAC=auto`, `trivalaya-runner.service` restarted
   at a clean job boundary (only one nonterminal job existed, `EA-614`,
   `pending`, not running). Confirmed both vars in the new runner PID's
   `/proc/<pid>/environ`; confirmed `trivalaya-search.service` has neither
   (no `EnvironmentFile=`, explicit `Environment=` lines only — query lane
   stays pinned to k=7 as designed).
2. **Preimage:** `analysis/two_coin_weld_leu_batch_2026-07-23/preimage.json`
   in trivalaya-pipeline (406 `coin_detections`, 200 `coins`, 198
   `ml_coin_dataset` rows across the 200 `auction_data` ids), plus the
   global baseline needed for the leak/attribution check:
   `MAX(coin_detections.created_at)=2026-07-21 05:26:48` (matches the
   `two-coin-weld-kernel-lane` memory's prior finding — zero detections
   DB-wide since the 07-21 enable, until this batch), `unpaired=9431`,
   `total_detections=780266`. `rollback.sql` documents the restore procedure
   (full rows live in `preimage.json`; raw images were never touched).
3. **Reset:** `DELETE FROM coin_detections/coins/ml_coin_dataset` for the 200
   ids; `vision_processed=0` on those 200 `auction_data` rows only. Sanity
   check confirmed all 200 CSV rows are genuinely `leu/75` with matching
   `lot_number`, and none were already unprocessed before the reset.
4. **Vision reprocess:** `python -m trivalaya_pipeline vision --source
   leu/75 --batch 250`, env sourced first (`set -a; source .env`) so both
   vars were live in the process (verified via its own `/proc/<pid>/environ`
   mid-run). Result: **200 images, 411 detections** (`get_unprocessed_records`
   correctly excluded the sample's 2 unrelated pre-existing
   `vision_processed=0`/`image_path=NULL` lots, landing exactly on 200).
5. **Pair:** `python -m trivalaya_pipeline pair` — 199 records processed,
   199 coins created, 406 detections linked, zero leakage into the
   pre-existing 9,431-strong global unpaired backlog (confirmed: every new
   `coins` row's `auction_record_id` is one of the 200 sample ids). The one
   sample lot that did **not** get paired is lot 3679 — both its detections
   score below the 0.5 likelihood threshold (0.451, 0.345), which is the
   **same pre-existing segmentation-failure lot** flagged in
   `two_coin_weld_morph_close.md` §2.5 ("a 723,941px blob at circularity
   0.337 covering 88% of the frame — L1 failing to segment, not a crop
   moving"). Not a new regression.

## §6.8 measurement

Query from §6.8, restricted to `created_at > '2026-07-21 15:49:54'`:

| house | dets | hough | hough_pct |
|---|---:|---:|---:|
| leu | 411 | 34 | **8.27%** |

**Every other house: zero rows.** This is both the weld-signature result and
the leak check in one query — nothing besides this session's own leu batch
has been vision-processed since the kernel enable, on any house.

Against the §1 census baseline and the k=5 sweep's own prediction
(`two_coin_weld_morph_close.md` §4.6): leu 41.7% → ~8.5% expected, **8.27%
measured**. Matches to within the sweep's own noise band.

### Kernel evidence (not just the aggregate rate)

Per-lot comparison of this batch's detections against the `preimage.json`
baseline (same production pipeline, same 200 lots, only the kernel/guard
env differs) shows the guard's specific signature repeatedly: a detection
that was a suspiciously perfect Hough-fitted circle at k=7 (circularity
0.97–0.99) reverts to a rougher, lower-circularity seed contour at
k=5+guard, exactly on the lots the offline sweep had already flagged:

| lot | side | OLD (k=7) bbox / circ | NEW (k=5+guard) bbox / circ |
|---|---|---|---|
| 582 | reverse | (459,0,622,600) / 0.992 | (477,0,604,600) / **0.427** |
| 713 | obverse | (0,0,632,600) / 0.993 | (0,0,615,600) / **0.323** |
| 995 | reverse | (450,0,597,600) / 0.991 | (451,0,596,600) / **0.416** |
| 3661 | obverse | (734,101,346,346) / 0.974 | (744,26,325,512) / **0.237** |

This is the guard visibly doing its job in real production output, not just
an aggregate rate moving.

## Guard validation (Scope B) — measured on the ACTUAL production crops

`analysis/two_coin_weld_leu_batch_2026-07-23/measure_sliver_production.py`
downloads each detection's real transparent crop from Spaces (this batch's
actual output, not a parallel offline recompute), places it at
`(bbox_x, bbox_y)` in a full-frame canvas, and computes the same undilated
(d0, px=0) worst-pair overlap fraction as
`tools/two_coin_weld_mask_gate.py::_slivers` — for the six lots the 07-23
offline sweep (`rim_neighbor_guard_sweep_2026-07-23.md`) found with real,
nonzero contamination.

| lot | detections | worst d0 contour_frac |
|---|---:|---:|
| 713 | 2 | **0.0** |
| 582 | 2 | **0.0** |
| 995 | 2 | **0.0** |
| 3661 | 3 | **0.0** |
| 3717 | 5 | **0.0** |
| 3736 | 5 | **0.0** |

**Every known sliver lot is sliver-free in real production output.** This
is the full-sample confirmation the offline sweep couldn't get (its own
guard-on 200-lot leu attempt ran >2h and was killed; it fell back to a
6-lot targeted re-run). This session's batch reprocessed all 200 lots
through the real runner-equivalent code path and got the complete result as
a side effect of closing §6.8.

### ndets regression check, and why lot 3736's count change is not the guard's

Per-lot detection-count diff, old (preimage, k=7) vs new (k=5+guard),
across all 200 lots:

| lot | old ndets | new ndets |
|---|---:|---:|
| 19 | 2 | 3 |
| 563 | 1 | **2** (improvement — a previously merged/missed coin now splits correctly) |
| 732 | 2 | 3 |
| 3736 | 3 | 5 |

**4/200 lots (2.0%) changed count; every other lot is unchanged.** This
matches — almost exactly — the *already-measured and already-accepted*
k=5 fragmentation cost from `two_coin_weld_morph_close.md` §4.6 ("true
fragmentation 1.5% (2.0% in 2-blob cell vs 0.0% control)"), not a new
finding. Lot 3736 is the one lot among the six known-sliver lots whose
count changed (3→5). This is **not** the guard's doing: the 07-23 offline
sweep measured the guard in isolation at fixed k=7 (guard on vs off,
kernel unchanged) and found 3736's ndets unchanged (3→3) either way
(`rim_neighbor_guard_sweep_2026-07-23.md`'s targeted table). The count
change here is attributable to the kernel change (k=7→k=5) happening
simultaneously in this production batch, not to
`TRIVALAYA_RIM_NEIGHBOR_GUARD`. **Zero ndets regressions from the guard
itself; the guard only ever changes a contour's shape/circularity when it
overlaps a neighbor, never adds or drops a detection** — consistent with
every prior measurement of Scope B.

## Leak check

Global counters, before → after this session's writes:

| | before | after |
|---|---:|---:|
| `MAX(coin_detections.created_at)` | 2026-07-21 05:26:48 | 2026-07-23 23:18:27 (this batch) |
| unpaired detections | 9,431 (baseline) / 9,840 (just before pair) | 9,434 |
| detections by house, `created_at` > enable | — | **leu only** (411 dets, 34 hough) |

No other house shows any post-enable volume. Nothing besides this
session's own scoped write touched the corpus in this window.

## Propagation

199 coins (of 200 lots; lot 3679 excluded, see above) now have fresh crops
and no embedding — not "stale," genuinely new (`coins`/`ml_coin_dataset`
rows for these 200 `auction_record_id`s were deleted and recreated with new
ids as part of the reset). Standard propagation cadence applies: the next
recluster re-embeds them. **No recluster forced this session**, per the
prompt's constraint. Any prior work keying on the OLD `coin_id` values for
these 200 lots (there is no known such work — this is a narrow, low-traffic
weld-lane sample, not a card/pedigree-annotated set) would need re-keying;
flagged here for completeness, not because a conflict was found.

## Verdict

**All §6.8 checks pass: WELD KERNEL LANE CLOSED.** Hough/weld-signature
collapse matches the k=5 sweep's own prediction (8.27% vs ~8.5% expected);
kernel evidence is visible per-lot, not just in the aggregate; the leak
check is clean; the one non-paired lot and the one count-changed known-sliver
lot are both pre-existing, already-documented, non-regression findings.

**Scope B enable VALIDATED in production.** All six known sliver lots are
sliver-free in the actual production output — the complete 200-lot
confirmation the offline sweep could not obtain — with zero ndets
regressions attributable to the guard.

`TRIVALAYA_RIM_NEIGHBOR_GUARD=1` stays enabled in `trivalaya-runner.service`
going forward (already live since the Step 1 restart). No revert needed.

Related: `two_coin_weld_morph_close.md` §6.8 (updated alongside this doc),
`rim_recovery_neighbor_aware.md` (Scope B), brain memory
`two-coin-weld-kernel-lane` / `rim-recovery-lane`.
