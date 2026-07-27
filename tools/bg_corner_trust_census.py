#!/usr/bin/env python
"""Corner-trust fire-rate census per auction house -- Bar 5 evidence (b).

Ticket: specs/background_estimator_repair.md, Bar 5 (owner scope ruling
2026-07-28: the re-embed deliverable is a CLASS table, and its per-class bound
comes from (a) the KS-17 A/B changed-rate and (b) this census).

Runs ONLY the corner test -- four 5x5 patch reads, no segmentation, no Otsu, no
Hough -- at production ingest geometry (`MAX_DIMENSION=3200`, full photo, which
is what `analyze_image` sees; sides are NOT split, because the ingest lane masks
from the whole frame).

Each sampled photo lands in exactly one bucket:

  pooled_fires   pooled corner_std < 15 -> the good path fires TODAY.  M1's
                 branch sits after it and is never reached, so these are
                 BIT-IDENTICAL under M1 by construction (Bar 2).  Zero rows.
  m1_fires       pooled fails, but all four patches are locally clean -> M1
                 changes the returned VALUE.
  neither        pooled fails AND at least one corner is locally noisy -> M1
                 declines and the histogram fallback runs exactly as today.
                 Unchanged.

**A changed value is not a changed outcome.**  Per the structural finding
(ticket, 2026-07-28), the value has exactly one consumer and it is read by one
comparison, `avg_bg > 110`.  So the re-embed bound is NOT `m1_fires` -- it is
`crosses_110`, the subset of `m1_fires` where the old and new values fall on
opposite sides of that threshold.  On KS-17 those differ by more than an order
of magnitude (572 values changed, 12 crossed), so reporting `m1_fires` as the
blast radius would overstate it ~48x.  Both are reported; `crosses_110` is the
one the class table consumes.

Read-only: DB SELECTs only, images fetched to the storage reader's temp cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

PIPELINE_ROOT = "/home/claudeuser/trivalaya-pipeline"
VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_DIMENSION = 3200
MARGIN = 5
POOLED_STD_MAX = 15.0
LOCAL_STD_MAX = 15.0
BRIGHT_BACKGROUND_THRESHOLD = 110   # src/config.py Layer1Config; the only read


def _load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        s = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def classify(gray, mu):
    """Return a dict describing the bucket AND whether the 110 line is crossed.

    `mu` is the imported src.math_utils, used so the OFF/ON values come from the
    SHIPPED function rather than a reimplementation of it here.
    """
    h, w = gray.shape
    if h <= 20 or w <= 20:
        return {"bucket": "too_small"}
    m = MARGIN
    patches = (gray[0:m, 0:m], gray[0:m, w-m:w], gray[h-m:h, 0:m], gray[h-m:h, w-m:w])
    pooled = np.concatenate([p.flatten() for p in patches])
    pooled_std = float(np.std(pooled))
    local_stds = [float(np.std(p)) for p in patches]

    os.environ.pop(mu.BG_CORNER_LOCAL_TRUST_ENV, None)
    off_v, _ = mu.detect_background_histogram(gray)
    os.environ[mu.BG_CORNER_LOCAL_TRUST_ENV] = "1"
    on_v, _ = mu.detect_background_histogram(gray)
    os.environ.pop(mu.BG_CORNER_LOCAL_TRUST_ENV, None)

    if pooled_std < POOLED_STD_MAX:
        bucket = "pooled_fires"
    elif all(s < LOCAL_STD_MAX for s in local_stds):
        bucket = "m1_fires"
    else:
        bucket = "neither"

    off_hi = float(off_v) > BRIGHT_BACKGROUND_THRESHOLD
    on_hi = float(on_v) > BRIGHT_BACKGROUND_THRESHOLD
    return {
        "bucket": bucket,
        "pooled_std": round(pooled_std, 3),
        "max_local_std": round(float(max(local_stds)), 3),
        "off_value": round(float(off_v), 3),
        "on_value": round(float(on_v), 3),
        "value_changed": float(off_v) != float(on_v),
        "crosses_110": off_hi != on_hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-house", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--threads", type=int, default=4, help="fetch concurrency (I/O)")
    a = ap.parse_args()

    cv2.setNumThreads(1)
    sys.path.insert(0, VISION_ROOT)
    sys.path.insert(1, PIPELINE_ROOT)
    import src.math_utils as mu  # noqa: E402
    assert os.path.realpath(mu.__file__).startswith(
        os.path.realpath(VISION_ROOT) + os.sep), mu.__file__
    from trivalaya_pipeline.storage import create_reader  # noqa: E402
    from trivalaya_pipeline.storage.adapters import add_spaces_prefix  # noqa: E402
    import mysql.connector  # noqa: E402

    # The old-catalog houses (Cahn / Hirsch / Helbing) store plates under a
    # `catalog/` key that is NOT in the default SPACES_KEY_PREFIXES set, so the
    # reader would resolve it locally and every plate would read as
    # load_failed -- silently zeroing three houses out of the census.
    add_spaces_prefix("catalog")

    conn = mysql.connector.connect(
        host=os.environ.get("TRIVALAYA_DB_HOST", "localhost"),
        user=os.environ["TRIVALAYA_DB_USER"],
        password=os.environ["TRIVALAYA_DB_PASSWORD"],
        database=os.environ.get("TRIVALAYA_DB_NAME", "auction_data"),
        charset="utf8mb4",
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT auction_house, COUNT(*)
        FROM auction_data
        WHERE image_path IS NOT NULL AND image_path <> ''
        GROUP BY auction_house
    """)
    populations = {h: n for h, n in cur.fetchall()}

    rng = random.Random(a.seed)
    sample = {}
    for house in sorted(populations):
        cur.execute("""
            SELECT id, image_path FROM auction_data
            WHERE auction_house = %s AND image_path IS NOT NULL AND image_path <> ''
        """, (house,))
        rows = cur.fetchall()
        rows.sort()                      # deterministic order before sampling
        k = min(a.per_house, len(rows))
        sample[house] = rng.sample(rows, k)
    cur.close(); conn.close()

    tasks = [(h, rid, p) for h, rows in sample.items() for rid, p in rows]
    print(f"census: {len(populations)} houses, {len(tasks)} photos "
          f"(seed={a.seed}, per_house={a.per_house})", flush=True)

    reader_cache = {}

    def measure(task):
        house, rid, path = task
        try:
            if os.path.isabs(path) and os.path.exists(path):
                local = path
            elif path.split("/", 1)[0] in ("raw", "processed", "ml", "exports",
                                           "catalog"):
                # Spaces-prefixed key -> storage reader (downloads to its cache).
                r = reader_cache.get("r")
                if r is None:
                    r = reader_cache["r"] = create_reader(path)
                local = str(r.get_local(path))
            else:
                # Relative local path (e.g. catalog/<sale>/pNNN/source.jpg) is
                # relative to the PIPELINE root, not to this script's cwd.
                cand = os.path.join(PIPELINE_ROOT, path)
                if not os.path.exists(cand):
                    cand = os.path.join(PIPELINE_ROOT, "trivalaya_data", path)
                local = cand
            img = _load_and_resize(local)
            if img is None:
                return {"house": house, "id": rid, "bucket": "load_failed"}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            row = classify(gray, mu)
            row.update({"house": house, "id": rid,
                        "w": img.shape[1], "h": img.shape[0]})
            return row
        except Exception as exc:
            return {"house": house, "id": rid, "bucket": "error",
                    "error": f"{type(exc).__name__}: {exc}"}

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for i, row in enumerate(ex.map(measure, tasks), 1):
            results.append(row)
            if i % 200 == 0:
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    fields = ["house", "id", "bucket", "pooled_std", "max_local_std",
              "off_value", "on_value", "value_changed", "crosses_110",
              "w", "h", "error"]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(results)

    agg = defaultdict(lambda: defaultdict(int))
    for r in results:
        agg[r["house"]][r["bucket"]] += 1
        if r.get("crosses_110"):
            agg[r["house"]]["_crosses"] += 1
        if r.get("value_changed"):
            agg[r["house"]]["_valchg"] += 1

    summary = {"seed": a.seed, "per_house": a.per_house,
               "threshold": BRIGHT_BACKGROUND_THRESHOLD,
               "note": ("in_scope_bound is derived from crosses_110, NOT from "
                        "m1_fires -- a changed value with no threshold crossing "
                        "is behaviorally inert (ticket, structural finding "
                        "2026-07-28)"),
               "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "houses": {}}
    for house in sorted(agg):
        c = agg[house]
        n = sum(v for k, v in c.items() if not k.startswith("_"))
        usable = n - c.get("load_failed", 0) - c.get("error", 0)
        cross = c.get("_crosses", 0)
        rate = (cross / usable) if usable else None
        pop = populations.get(house, 0)
        summary["houses"][house] = {
            "population": pop, "sampled": n, "usable": usable,
            "pooled_fires": c.get("pooled_fires", 0),
            "m1_fires": c.get("m1_fires", 0), "neither": c.get("neither", 0),
            "value_changed": c.get("_valchg", 0),
            "crosses_110": cross,
            "load_failed": c.get("load_failed", 0), "error": c.get("error", 0),
            "cross_rate": None if rate is None else round(rate, 4),
            "in_scope_bound": None if rate is None else int(round(pop * rate)),
        }
    with open(a.summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'house':22s} {'pop':>8s} {'n':>5s} {'pooled':>7s} {'m1':>5s} "
          f"{'valchg':>7s} {'cross':>6s} {'rate':>7s} {'bound':>8s}")
    tot_bound = 0
    for house, s in summary["houses"].items():
        tot_bound += s["in_scope_bound"] or 0
        print(f"{house[:22]:22s} {s['population']:8d} {s['usable']:5d} "
              f"{s['pooled_fires']:7d} {s['m1_fires']:5d} {s['value_changed']:7d} "
              f"{s['crosses_110']:6d} "
              f"{(s['cross_rate'] if s['cross_rate'] is not None else -1):7.4f} "
              f"{(s['in_scope_bound'] if s['in_scope_bound'] is not None else -1):8d}")
    print(f"{'TOTAL in-scope bound':22s} {sum(populations.values()):8d} "
          f"{'':5s} {'':7s} {'':5s} {'':7s} {'':6s} {'':7s} {tot_bound:8d}")
    print(f"\n-> {a.out}\n-> {a.summary}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
