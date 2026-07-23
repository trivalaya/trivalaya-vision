"""
Scope C precommit measurement: confirm (or refute) that a given sample's
slow sides share KS-17's hot leaf -- cv2.HoughCircles inside
rim_logic.hough_rim_recovery, Layer 1.5. See specs/rim_recovery_neighbor_
aware.md, "PRECOMMIT ACCEPTANCE BARS / Scope C".

Two-pass design so cProfile overhead never contaminates the timing numbers
used to pick which sides to profile:

  1. Cheap timing pass (time.process_time(), no cProfile) over every lot in
     the sample -- CPU-seconds per lot, matching the KS-17 diagnosis's own
     "process_time() reflects real compute" method note.
  2. cProfile ONLY the p99-slowest N lots from pass 1, reporting cumulative
     time per function so the hot leaf and its % of total self time are
     directly comparable to specs/results/
     ks17_mask_stall_diagnosis_2026-07-22.md's own cProfile table.

Reuses tools/two_coin_weld_ab.py's `_load` (raw fetch, local or Spaces) so
this measures the exact same frozen-sample raws the weld lane already
validated. Read-only: no DB, no Spaces writes, no service restart.

Usage (kuenker Scope C fixture):
    .venv/bin/python tools/rim_recovery_profile_sample.py \
        --purpose kuenker_wallclock --house kuenker \
        --top-n 5 --out specs/results/rim_recovery_profile_kuenker
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer1_geometry import layer_1_structural_salience  # noqa: E402
from tools.two_coin_weld_ab import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_RAW_ROOT,
    DEFAULT_SAMPLE,
    _load,
)


def _time_one(img, house: Optional[str]) -> float:
    t0 = time.process_time()
    layer_1_structural_salience(img, source_type="auction", house=house)
    return time.process_time() - t0


def _profile_one(img, house: Optional[str]) -> str:
    pr = cProfile.Profile()
    pr.enable()
    layer_1_structural_salience(img, source_type="auction", house=house)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(15)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--source", choices=["local", "spaces"], default="local")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--house", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=5,
                     help="how many of the slowest lots to cProfile")
    ap.add_argument("--out", type=Path, default=Path("rim_recovery_profile"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.sample))
            if r["purpose"] == args.purpose
            and (args.house is None or r["house"] == args.house)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit(f"no rows for purpose={args.purpose} house={args.house}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)} lots, purpose={args.purpose} house={args.house} -- timing pass")
    timed: List[Dict] = []
    for n, r in enumerate(rows, 1):
        img = _load(r, args.raw_root, args.source, args.cache_dir)
        if img is None:
            print(f"  [{n}] MISSING {r['house']}/{r['sale_id']}/{r['lot_number']}")
            continue
        cpu_s = _time_one(img, r["house"])
        timed.append({"lot_number": r["lot_number"], "lot_id": r["lot_id"],
                       "cpu_s": round(cpu_s, 3)})
        if n % 25 == 0:
            print(f"  [{n}/{len(rows)}]")

    timed.sort(key=lambda t: t["cpu_s"], reverse=True)
    slowest = timed[:args.top_n]
    print(f"\nslowest {len(slowest)}: {slowest}")

    profiles = {}
    row_by_lot = {r["lot_number"]: r for r in rows}
    for t in slowest:
        r = row_by_lot[t["lot_number"]]
        img = _load(r, args.raw_root, args.source, args.cache_dir)
        profiles[t["lot_number"]] = {
            "cpu_s_pass1": t["cpu_s"],
            "cprofile_top15": _profile_one(img, r["house"]),
        }

    def pct(vals, q):
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(q * len(s)))], 3) if s else None

    cpus = [t["cpu_s"] for t in timed]
    summary = {
        "n": len(timed),
        "cpu_s": {"p50": pct(cpus, 0.5), "p90": pct(cpus, 0.9),
                  "p99": pct(cpus, 0.99), "max": max(cpus) if cpus else None},
        "slowest": slowest,
    }

    (args.out.with_suffix(".csv")).write_text(
        "lot_number,lot_id,cpu_s\n" +
        "\n".join(f'{t["lot_number"]},{t["lot_id"]},{t["cpu_s"]}' for t in timed)
    )
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    profile_txt = args.out.parent / f"{args.out.name}_profiles.txt"
    with open(profile_txt, "w") as fh:
        for lot, p in profiles.items():
            fh.write(f"=== lot {lot} (pass1 cpu_s={p['cpu_s_pass1']}) ===\n")
            fh.write(p["cprofile_top15"])
            fh.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out.with_suffix('.csv')}\nwrote {args.out.with_suffix('.json')}"
          f"\nwrote {profile_txt}")


if __name__ == "__main__":
    main()
