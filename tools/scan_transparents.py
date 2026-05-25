"""
Coverage scan: download existing transparent_path PNGs from Spaces, measure
alpha coverage, and write a per-row log + eaten coin list.

No pipeline runs. Pure read from Spaces.

Input: JSON file with [{coin_id, side, detection_index, transparent_path}, ...]
   (produced by tools/build_manifests.py on the DB host)

Usage:
    python tools/scan_transparents.py \\
        --targets /path/to/scan_targets.json \\
        --log /path/to/scan_log.jsonl \\
        --eaten-coins /path/to/eaten_coins.txt \\
        --threshold 60 \\
        --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from reprocess_hough import get_s3, BUCKET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="JSON list of scan rows")
    ap.add_argument("--log", required=True, help="JSONL output path")
    ap.add_argument("--eaten-coins", required=True, help="Text file: comma-separated coin_ids below threshold")
    ap.add_argument("--threshold", type=float, default=60.0, help="alpha %% under which a side is considered eaten")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    rows = json.load(open(args.targets))
    print(f"Loaded {len(rows):,} scan rows from {args.targets}")

    s3_per_thread: dict = {}
    def _scan(row):
        tid = threading.get_ident()
        if tid not in s3_per_thread:
            s3_per_thread[tid] = get_s3()
        cid = row["coin_id"]
        side = row["side"]
        idx = row["detection_index"]
        tpath = row.get("transparent_path")
        if not tpath:
            return {"coin_id": cid, "side": side, "det_idx": idx, "status": "skip", "reason": "no_transparent_path"}
        try:
            import io
            buf = io.BytesIO()
            s3_per_thread[tid].download_fileobj(BUCKET, tpath, buf)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            alpha = np.asarray(img.split()[3])
            pct = float((alpha > 200).mean() * 100)
            return {"coin_id": cid, "side": side, "det_idx": idx,
                    "alpha_pct": round(pct, 2), "status": "ok"}
        except Exception as e:
            return {"coin_id": cid, "side": side, "det_idx": idx,
                    "status": "fail", "error": str(e)}

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(args.log, "w")
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(_scan, rows):
            log_fp.write(json.dumps(rec) + "\n")
            n += 1
            if n % 2000 == 0:
                log_fp.flush()
                rate = n / max(time.time() - t0, 0.01)
                eta = (len(rows) - n) / max(rate, 0.01) / 60
                print(f"  {n:,}/{len(rows):,}  rate={rate:.0f}/s  eta={eta:.0f} min")
    log_fp.close()
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    # Summarize + write eaten list
    recs = [json.loads(l) for l in open(args.log)]
    ok = [r for r in recs if r["status"] == "ok"]
    eaten = [r for r in ok if r["alpha_pct"] < args.threshold]
    fail = [r for r in recs if r["status"] != "ok"]
    coins_eaten = sorted({r["coin_id"] for r in eaten})

    print()
    print(f"rows: total={len(recs):,} ok={len(ok):,} fail={len(fail):,}")
    pct_eaten = 100 * len(eaten) / max(len(ok), 1)
    print(f"  sides < {args.threshold:.0f}%: {len(eaten):,} ({pct_eaten:.1f}%)")
    print(f"coins with ≥1 eaten side: {len(coins_eaten):,}")

    with open(args.eaten_coins, "w") as f:
        f.write(",".join(str(c) for c in coins_eaten))
    print(f"eaten coin list → {args.eaten_coins}")


if __name__ == "__main__":
    main()
