# KS-17 (cng, Keystone 17) production spot check — 2026-07-23

Lane: `specs/two_coin_weld_morph_close.md` §6.8. Trigger: the runner Discord
line "cng_feature KS-17 ingested. 0 lots, 1494 coins" (job 312, claimed
2026-07-23 00:00:06 UTC — the first `pipeline_jobs` job to run since the
2026-07-21 15:49:54 UTC production enable of `TRIVALAYA_CLOSE_KERNEL_FRAC=auto`).

**Verdict: STILL NO DATA for §6.8 — the lane stays OPEN, 34h+ after the last
check.** Job 312 did not run vision on KS-17 at all. The apparent "first
production batch under the new kernel" does not exist; see below. This is a
data-availability + data-quality finding, not a regression signal — nothing
about the kernel code or the membership gate is implicated.

Read-only measurement throughout. No code, config, DB or service changes.

## Headline finding: KS-17 is not `cng_feature` — it's `cng`

`pipeline_jobs.job_id=312` is tagged `auction_house='cng_feature'`, and that
tag is what made this look like the tabled (k=3) house the trigger described.
It is wrong. Every one of the 374 KS-17 rows in `auction_data` is stored
under `auction_house='cng'`:

```sql
SELECT COUNT(*) FROM auction_data WHERE auction_house='cng_feature' AND sale_id='KS-17';  -- 0
SELECT COUNT(*) FROM auction_data WHERE auction_house='cng'          AND sale_id='KS-17';  -- 374
```

`cng` is the *other* CNG lot system (`Lots.aspx`, per the standing
CNG-two-archive-structure distinction), confirmed independently by the job's
own `sale_url`: `https://www.cngcoins.com/Lots.aspx?AUCTION_ID=230`. The
images are the large `cng`-format plates (3000×1440 — checked directly on 10
of the 374 raws), not `cng_feature`'s small 500×~240 format the kernel fix
was built and measured against.

**Root cause, traced to source.** `discovered_sales.id=392` (the discovery
LLM run that seeded this job, 2026-07-09, `claude-sonnet-4-6`) tagged
`auction_house: "cng_feature"` on a sale whose own `sale_url` is
unambiguously the `Lots.aspx` (`cng`) system — a discovery-agent mistag, not
a vision-pipeline bug. It propagated verbatim into `pipeline_jobs.auction_house`.
Corroborated a second way in `journalctl -u trivalaya-runner`, job 312's own
scrape stage: it ran the `cng_feature`-specific `Coin.aspx?CoinID=`
verification method against a sale that isn't on that archive at all —
`[verify] CoinID 1..20: could not extract sale name. Skipping` /
`20 consecutive empty responses — stopping` — i.e. the scraper's own
house-specific probe independently returned 0/20, for the same reason.

**Consequence: the vision stage of job 312 processed zero KS-17 images.**
`get_unprocessed_records()` filters `auction_house = %s` exactly
(`catalog.py:1142`) on the job's (wrong) tag. Scoped to `cng_feature`, it
found nothing — the real rows are `cng`:

```
Starting stage: vision (cng_feature/KS-17)
Found 0 unprocessed images
Vision complete: 0 images, 0 coins
Stage vision complete: {'processed': 0, 'detections': 0, 'errors': 0, 'skipped': 0}
```

Confirmed at the data layer: **0/374** KS-17 rows have `vision_processed=1`;
**0** `coin_detections` rows join to `auction_house='cng' AND sale_id='KS-17'`.
Job 312 is terminal (`status='done'`) — KS-17 will not be retried by the
normal queue. Absent a manually-corrected re-queue (`auction_house='cng'`),
its 374 lots will not get vision-processed by this job again. **Flagged
loudly, fixed nowhere** — re-queuing is an operator action, out of scope for
a read-only brief, and belongs with discovery-agent-ops, not this spec.

## Task 1 — kernel confirmation

Job 312 claimed **2026-07-23 00:00:06 UTC**, well after the 2026-07-21
15:49:54 UTC enable. `code_sha=9c4359db56a7020634fbfca4bc209e1e2350c123-dirty`
matches this repo's current HEAD (the `-dirty` suffix is the *pipeline*
repo's own uncommitted analysis clutter — `select_dirty()` in
`deploy_staleness.py` confirms this is informational only, never gates
correctness). `worker_id=...:1962:2702` — runner PID 1962, alive since the
box's 2026-07-22 13:32:45 reboot, the same PID confirmed carrying
`TRIVALAYA_CLOSE_KERNEL_FRAC=auto` in `/proc/1962/environ` (re-checked live,
still present). `deploy_staleness.py` reports
`vision_pending_for_runner: 0` — the vision code this runner has loaded is at
`trivalaya-vision` HEAD (`6d8ad0d`), which includes the §6.8 enable commit.

**Which kernel actually would apply, and did:** `cng` is **not** a key in
`Layer1Config.CLOSE_KERNEL_BY_HOUSE` (only `cng_feature`, `leu`, `kuenker`
are tabled). Per the membership gate
(`layer1_geometry.py:227-235`, `_tabled = house in CLOSE_KERNEL_BY_HOUSE`),
an untabled house stays on the fixed **k=7** regardless of
`TRIVALAYA_CLOSE_KERNEL_FRAC=auto`. Independently — a second, unrelated
guarantee — the raw scale-relative formula (`CLOSE_KERNEL_FRAC=1/400`)
itself floors to **k=7** at `cng`'s 3000px width (`int(3000/400)=7`), exactly
as the code's own docstring states ("Truncation yields k=7 across the whole
2400-3199 band") and as §6.7 of the spec already measured directly for `cng`
(`k=7 | 72.2% share | 12px | unchanged — §4.3's band`). Two independent
mechanisms both land on k=7 for this photo class; there was no live-fire test
of a materially different kernel here, tabled-house mislabeling
notwithstanding — moot, since vision never ran on it anyway (above).

So: **not** "a tabled house (k=3)" as the trigger implied. It's an untabled
house that gets k=7 under any reading of `auto`, and it has not actually been
vision-processed at all yet under any kernel.

## Task 2 — the 1494, fully explained (and it is not KS-17)

`pipeline_jobs`'s `pair` stage calls `Pipeline.pair_detections()` →
`CatalogDB.pair_unlinked_detections()`, which is **globally unscoped** — no
`auction_house` / `sale_id` filter anywhere in the query
(`catalog.py:1021-1032`). Job 312's pair stage therefore paired *whatever*
backlog of already-vision-processed-but-unpaired detections existed
anywhere in the DB at that moment, house-blind. Verified directly — `coins`
rows created inside the job's 00:00:06–00:05:22 window:

| house | sale | coins created | window |
|---|---|---:|---|
| kuenker | eLive-93 | 651 | 00:00:21–00:01:42 |
| kuenker | eLive-92 | 682 | 00:01:42–00:02:49 |
| kuenker | 428      | 161 | 00:02:49–00:03:10 |
| **total** | | **1494** | matches job's reported `coins_created` exactly |

These are **old** detections — `coin_detections.created_at` for all three
sales runs 2026-07-19 through **2026-07-21 05:26:48**, i.e. from the
kuenker-scoped nightly cron runs that finished *before* the 15:49:54 enable
(per the 07-22 no-data report). They had simply never been paired
(a 2–4 day pairing-lag backlog) until job 312's unscoped sweep swept them up.
**Zero relationship to KS-17's photos, and zero relationship to the new
kernel** — kuenker's override is `min=max=7`, bit-identical to the always-fixed
default, a verified no-op per §4.8/§6.6, so it would not matter even if these
were new.

Neither of the trigger's two hypotheses (benign multi-image-per-lot vs.
fragmentation) applies, because **there is no KS-17 detection data to apply
them to** — coin_detections has 0 rows for this sale. For the record, once
vision does eventually run KS-17 correctly: `cng` stores **one combined
image per lot** (`image_path = raw/auctions/cng/KS-17/Lot_NNNNN.jpg`, one row
per lot, checked for lots 1–5), both sides in a single 3000×1440 photo — the
clean expectation is 2 detections/lot × 374 = 748, not the "2 images × 2"
convention the trigger assumed (that's a `cng_feature`/`leu` habit, not
`cng`'s).

The Discord line "cng_feature KS-17 ingested … 1494 coins" is therefore
misleading on both halves: wrong house, and an unrelated backlog-clearing
side effect of running the pair stage at all, attributed to the job that
happened to trigger it.

## Task 3 — weld-signature / Hough-rate structural check

**Not run as a fresh L1 sweep on the KS-17 raws, by deliberate choice.**
Two independent reasons, stated plainly rather than silently skipped:

1. It's moot for a kernel-selection verdict — Task 1 above already shows,
   from code + config alone (membership-gate miss **and** formula
   convergence, two independent routes), that this photo class gets k=7
   under `auto`, identically to `control`. There is no live-fire batch to
   measure in the first place (Task 2).
2. **The box is currently contended by a separate, directly-related, live
   investigation into the same photo class's cost pathology.** A concurrent
   session was found running `tools/rim_recovery_cost_ab.py` and
   `tools/two_coin_weld_mask_gate.py` against these exact KS-17 images
   (`~/trivalaya-vision-worktrees/rim-recovery`, the tracked
   `rim_recovery_neighbor_aware.md` successor lane — "kuenker cost tail p99
   139s, NOT weld, rim-recovery suspect"). A first attempt at an independent
   structural probe here (`tools/two_coin_weld_ab.py`'s own
   `_diagnostics`/`_run_arm` functions, reused verbatim, read-only) confirmed
   this the hard way: even a single KS-17 image took **2.5+ minutes** with
   zero rows completed before being killed — consistent with the same
   rim-recovery cost tail already under active study elsewhere. Running a
   competing heavy CV sweep on a 4 vCPU / 8 GB box against a resource that
   another investigation is actively characterizing, for a question already
   settled without it, is not a good trade. Aborted; no partial numbers to
   report from this attempt (the killed run wrote 0 completed rows to its
   output file).

**What *is* available, and was used instead:** the incoming-screen
pre-hammer lane (`analysis/incoming_screen/KS-17/`) already ran Layer-1 on
these same 287 (of 374) raws, pinned at k=7 (its `_mask_query_image_meta` →
`analyze_image` call passes no `house`, so `house=None` ⇒ untabled ⇒ k=7 —
confirmed by reading `visual_search/appv2.py`). Its final run log
(`/tmp/screen_KS17_r3.log`) reports:

```
incoming embeddings: 0 cached, 287 new, mask fallback on 0
```

**0/287 lots (0/574 sides) failed to find a maskable contour at k=7** — no
gross detection failures on this photo class at the kernel it will actually
receive. This is a weaker signal than the spec's own weld-signature/Hough-rate
metric (it only checks "found ≥1 usable contour," not the full pre/post-close
blob count), consistent with the trigger's framing of this comparison as "a
sanity signal, not a formal bar." It corroborates, it does not replace, the
code-level guarantee in Task 1.

(Separately: an earlier diagnostic on this same corpus, `/tmp/probe_ks17.log`,
recorded several lots at 40–66s/side with `mask_fb=exception` — the mask-stall
investigation referenced above, already tracked in its own lane, resolved by
the time of the final `r3` run above. Unrelated to the kernel question; noted
for completeness only.)

**No weld/Hough numbers vs. the cng_feature k=7 baseline (~90.5%/97.5%) are
reported here**, because that baseline is the wrong comparator for this
house (§1 of the spec: "cng and cng_feature share a name and nothing else —
3000×1440 vs 500×234, 6.4% vs 85.6%"). `cng`'s own corpus-wide Hough baseline
is **6.4%** (7,092/111,572, §1's re-baseline table) — that is the relevant
prior for this photo class, and nothing in this check contradicts it.

## Task 4 — leak tripwire (other houses flat)

```sql
SELECT MAX(created_at), SUM(created_at > '2026-07-21 15:49:54')
FROM coin_detections;
-- 2026-07-21 05:26:48 | 0
```

**Zero new `coin_detections` rows, any house, since before the enable** —
re-checked live at 2026-07-23 02:33 UTC, ~34h45m after the enable. Only one
`pipeline_jobs` row was claimed/started in the entire window (job 312), and
its own vision stage processed 0 records (above). The pre-existing
vision-pending backlog (`image_path` present, `vision_processed=0`) is
**byte-identical** to the 2026-07-22 snapshot — leu 3,965, obolos 2,024,
cng 1,381, cng_feature 1,233, nomos 933, heritage 315, stacksbowers 220 — no
silent activity anywhere.

**The 2026-07-23 05:00 UTC kuenker-scoped nightly cron has not yet fired** as
of this measurement (current time 02:33 UTC, before 05:00). Per the brief:
this is **explicitly PENDING, not passed**. There is no volume anywhere to
leak from yet, so "flat" is vacuously true for every house — that is not the
same as a positive pass, and is reported as such.

## Verdict, per metric

| check | verdict |
|---|---|
| Kernel config live in the runner that claimed job 312 (`auto`, vision code at HEAD) | **PASS** — confirmed via `/proc/1962/environ`, `deploy_staleness.py`, git log |
| Job 312 is the tabled house (`cng_feature`, k=3) the trigger described | **FAIL** — it is `cng`, untabled, k=7 by two independent mechanisms |
| KS-17 vision actually ran under any kernel | **FAIL** — 0/374 vision_processed, job's own log: "Found 0 unprocessed images" |
| "1494 coins" = KS-17 detections (benign or fragmentation) | **N/A, neither** — fully traced to an unrelated global-pair sweep of a pre-existing, pre-enable kuenker backlog (682+651+161=1494 exact) |
| Weld-signature / Hough-rate vs. cng_feature k=7 baseline | **N/A, wrong comparator** — no KS-17 production data exists; cng's own baseline (6.4% Hough) is undisturbed; a from-scratch structural probe was attempted and aborted (see Task 3) rather than compete for CPU with a live, related, separate investigation for a question already settled by code |
| Screen-lane k=7 cross-check (sanity signal only) | **PASS** — 0/287 mask fallbacks |
| Other-house-flat leak check | **PENDING** — no volume anywhere yet to check against; 07-23 05:00 UTC kuenker nightly has not fired as of this measurement |
| §6.8 lane status | **STILL OPEN.** Not closed. Both houses this spec actually cares about — cng_feature (k=3) and leu (k=5) — have **zero** production volume under the new kernel, 34h45m post-enable. leu remains the biggest-win house with zero production volume, as of the 07-22 report; **that has not changed.** cng_feature is now *also* confirmed at zero, for a different and unrelated reason (a discovery-agent mistag misrouting an untabled-house sale under its name) that this check surfaced but does not fix. |

## What would close the lane

Unchanged in kind from the 07-22 report: re-run §6.8's query once real
post-enable volume exists for `cng_feature` or `leu` specifically (not `cng`,
which this check shows is a false positive for that purpose). Separately,
and out of scope here: KS-17's 374 `cng` lots need a corrected re-queue
(`auction_house='cng'`) to ever get vision-processed at all under the
current job-tagging bug — an operator/discovery-agent-ops action, not a
kernel-lane action.
