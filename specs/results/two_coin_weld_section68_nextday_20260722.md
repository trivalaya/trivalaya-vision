# §6.8 next-day production spot check — 2026-07-22

Lane: `specs/two_coin_weld_morph_close.md` §6.8. Purpose: confirm the
2026-07-21 15:49:54 UTC production enable of `TRIVALAYA_CLOSE_KERNEL_FRAC=auto`
(membership-gated, leu→k=5, cng_feature→k=3, every other house→k=7) collapsed
the weld signature on leu/cng_feature while every other house stayed flat.

**Verdict: NO DATA — cannot be scored PASS or FAIL.** The lane stays OPEN.
This is a data-availability gap, not a regression signal; nothing about the
code, config, or membership gate is implicated (see confirmations below).

Read-only measurement. No code or config changed anywhere in this session.

## What was measured

Ran §6.8's own query verbatim (window: enable timestamp through
2026-07-22 21:34 UTC, ~29h45m):

```sql
SELECT ad.auction_house, COUNT(*) AS dets,
       SUM(cd.vision_metadata LIKE '%"split_method": "hough"%') AS hough,
       100.0 * SUM(...) / COUNT(*) AS hough_pct
FROM coin_detections cd
JOIN auction_data ad ON cd.auction_record_id = ad.id
WHERE cd.created_at > '2026-07-21 15:49:54'
GROUP BY ad.auction_house;
```

Result: **0 rows, every house.** `SELECT COUNT(*) FROM coin_detections WHERE
created_at > '2026-07-21 15:49:54'` also returns 0. No lot — leu, cng_feature,
kuenker, or otherwise — has been vision-processed since the enable landed.

## Why: traced to source, not a runner fault

1. **The job queue has been empty of due work since before the enable.** The
   last completed `sale_ingestion` job (any house) is job 310 (cng_feature,
   "Triton X"), finished 2026-07-20 04:54:24 — over 11h before the 15:49:54
   restart. Only two jobs remain non-terminal, and both are scheduled in the
   future relative to their sale's close date:
   - job 312, cng_feature "KS-17", `run_after = 2026-07-23 00:00:00`
   - job 314, cng "EA-614", `run_after = 2026-07-30 00:00:00` (EA-614 closes
     2026-07-29 per the pre-hammer estimator track)

   The runner (PID 1962, alive since the box's 2026-07-22 13:32:45 reboot,
   idle in its poll loop, 0 `running`/`claimed` rows in `pipeline_jobs`) is
   correctly finding nothing claimable — this is expected scheduling, not a
   stall.

2. **The kuenker-scoped nightly cron ran on time both mornings, and correctly
   did nothing on the second.** From
   `logs/vision_nightly/cron.log`:
   - 2026-07-21 05:00:02–05:26:48 UTC (**before** the 15:49:54 enable) —
     496 images, 1,101 coins, under the old fixed 7×7 kernel. This is the
     run that set `coin_detections.MAX(created_at) = 2026-07-21 05:26:48`.
   - 2026-07-22 05:00:02–05:00:09 UTC (**after** the enable) — "Found 0
     unprocessed images." kuenker's backlog was already fully drained by the
     prior morning's run, so this firing had nothing to contribute — not a
     fault, just no work available. (Confirmed separately: kuenker has 0 rows
     with `vision_processed=0`, versus non-zero for every other house below.)

3. **Config confirmed still live.** `TRIVALAYA_CLOSE_KERNEL_FRAC=auto` is
   present in `/proc/1962/environ` (the current runner process, post-reboot).
   `git log` on this repo is clean at `0b350ab` (the §6.8 commit). So the
   fix is correctly armed and would apply the instant a job runs — there is
   simply no post-enable job to apply it to yet.

## Context noted, out of scope for this lane

A large pre-existing `vision_processed=0` backlog exists — leu 3,965,
obolos 2,024, cng 1,381, cng_feature 1,233, nomos 933, heritage 315,
stacksbowers 220 — but every row dates from April–July 2026 with no active
`pipeline_jobs` entry claiming it (untouched by both the main runner's queue
and the kuenker-scoped cron). It predates and is unmoved by yesterday's
enable; flagged for visibility only, not investigated further here.

## What would close the lane

Re-run the §6.8 query (unchanged) once real post-enable volume exists:
either EA-614 / KS-17 close and their jobs run (2026-07-29 / 2026-07-23
earliest), or the owner chooses to force a scoped batch sooner for an
earlier same-day signal — that's a production decision for the owner, not
made here under a read-only brief.

**Lane status: OPEN.** Not closed; successor
(`specs/rim_recovery_neighbor_aware.md`) unaffected either way.
