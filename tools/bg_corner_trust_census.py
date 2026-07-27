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
                 changes the returned value.  IN SCOPE for re-embed.
  neither        pooled fails AND at least one corner is locally noisy -> M1
                 declines and the histogram fallback runs exactly as today.
                 Unchanged.

So a house's in-scope share is `m1_fires / n`, and the corpus bound for that
house is `population x m1_fires_rate`.  Houses that are all `pooled_fires` are
excluded by construction, not by measurement -- that is the point of Bar 2.

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


def _load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        s = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def classify(gray):
    """Return (bucket, pooled_std, max_local_std, pooled_median, m1_value)."""
    h, w = gray.shape
    if h <= 20 or w <= 20:
        return "too_small", None, None, None, None
    m = MARGIN
    patches = (gray[0:m, 0:m], gray[0:m, w-m:w], gray[h-m:h, 0:m], gray[h-m:h, w-m:w])
    pooled = np.concatenate([p.flatten() for p in patches])
    pooled_std = float(np.std(pooled))
    pooled_med = float(np.median(pooled))
    local_stds = [float(np.std(p)) for p in patches]
    max_local = float(max(local_stds))

    if pooled_std < POOLED_STD_MAX:
        return "pooled_fires", pooled_std, max_local, pooled_med, None
    if all(s < LOCAL_STD_MAX for s in local_stds):
        m1_val = float(np.median([float(np.median(p)) for p in patches]))
        return "m1_fires", pooled_std, max_local, pooled_med, m1_val
    return "neither", pooled_std, max_local, pooled_med, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-house", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--threads", type=int, default=4, help="fetch concurrency (I/O)")
    a = ap.parse_args()

    sys.path.insert(0, PIPELINE_ROOT)
    from trivalaya_pipeline.storage import create_reader  # noqa: E402
    import pymysql  # noqa: E402

    conn = pymysql.connect(
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
            key = path
            if os.path.isabs(path) and os.path.exists(path):
                local = path
            else:
                r = reader_cache.get("r")
                if r is None:
                    r = reader_cache["r"] = create_reader(key)
                local = str(r.get_local(key))
            img = _load_and_resize(local)
            if img is None:
                return {"house": house, "id": rid, "bucket": "load_failed"}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            b, ps, ml, pm, m1v = classify(gray)
            return {"house": house, "id": rid, "bucket": b,
                    "pooled_std": None if ps is None else round(ps, 3),
                    "max_local_std": None if ml is None else round(ml, 3),
                    "pooled_median": None if pm is None else round(pm, 2),
                    "m1_value": None if m1v is None else round(m1v, 2),
                    "w": img.shape[1], "h": img.shape[0]}
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
              "pooled_median", "m1_value", "w", "h", "error"]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(results)

    agg = defaultdict(lambda: defaultdict(int))
    for r in results:
        agg[r["house"]][r["bucket"]] += 1

    summary = {"seed": a.seed, "per_house": a.per_house,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "houses": {}}
    for house in sorted(agg):
        c = agg[house]
        n = sum(c.values())
        usable = n - c.get("load_failed", 0) - c.get("error", 0)
        m1 = c.get("m1_fires", 0)
        rate = (m1 / usable) if usable else None
        pop = populations.get(house, 0)
        summary["houses"][house] = {
            "population": pop, "sampled": n, "usable": usable,
            "pooled_fires": c.get("pooled_fires", 0),
            "m1_fires": m1, "neither": c.get("neither", 0),
            "load_failed": c.get("load_failed", 0), "error": c.get("error", 0),
            "m1_fire_rate": None if rate is None else round(rate, 4),
            "in_scope_bound": None if rate is None else int(round(pop * rate)),
        }
    with open(a.summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'house':22s} {'pop':>8s} {'n':>5s} {'pooled':>7s} {'m1':>5s} "
          f"{'neither':>8s} {'m1_rate':>8s} {'bound':>8s}")
    for house, s in summary["houses"].items():
        print(f"{house[:22]:22s} {s['population']:8d} {s['usable']:5d} "
              f"{s['pooled_fires']:7d} {s['m1_fires']:5d} {s['neither']:8d} "
              f"{(s['m1_fire_rate'] if s['m1_fire_rate'] is not None else -1):8.4f} "
              f"{(s['in_scope_bound'] if s['in_scope_bound'] is not None else -1):8d}")
    print(f"\n-> {a.out}\n-> {a.summary}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
