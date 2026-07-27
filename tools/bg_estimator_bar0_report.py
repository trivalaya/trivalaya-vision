#!/usr/bin/env python
"""Render the Bar 0 contingency table + mask-area distribution from probe JSONL.

Bar 0 asks for "the contingency table of {corner-trust vs fallback path} x
{mask area fraction > 0.9 with masked:true}, plus the full distribution of mask
area fraction — not a single thresholded count."

Usage: bg_estimator_bar0_report.py RUN.jsonl [--compare PRIOR.csv]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter

NOOP_T = 0.9
CIRCLE_CAP = 0.7854  # pi/4 — a genuine full-frame circular coin


def load(path):
    meta, rows = None, []
    for line in open(path):
        d = json.loads(line)
        if d.get("_meta"):
            meta = d
        else:
            rows.append(d)
    return meta, rows


def histo(vals, lo=0.0, hi=1.0, nbins=20):
    if not vals:
        return
    w = (hi - lo) / nbins
    c = Counter(min(nbins - 1, int((v - lo) / w)) for v in vals)
    mx = max(c.values())
    for b in range(nbins):
        n = c.get(b, 0)
        if n == 0 and not (lo + b * w <= CIRCLE_CAP < lo + (b + 1) * w):
            continue
        bar = "#" * max(1, int(28 * n / mx)) if n else ""
        mark = "  <- pi/4 cap" if lo + b * w <= CIRCLE_CAP < lo + (b + 1) * w else ""
        print(f"    [{lo+b*w:.2f},{lo+(b+1)*w:.2f})  {n:4d} {bar}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--compare", help="prior run CSV (id,side,strata,...,mask_area_fraction)")
    a = ap.parse_args()

    meta, rows = load(a.jsonl)
    print("=" * 74)
    print("BAR 0 — background-estimator <-> serving mask no-op link")
    print("=" * 74)
    print(f"sample_source : {meta.get('sample_source')}")
    print(f"tasks         : {len(rows)} / {meta.get('n_tasks')} planned")
    print(f"timeout       : {meta.get('timeout_s')}s   modes: {meta.get('modes')}")
    print(f"started       : {meta.get('started')}")

    print("\nSTATUS")
    for (m, s), n in sorted(Counter((r["mode"], r["status"]) for r in rows).items()):
        print(f"  {m:9s} {s:16s} {n}")

    for mode in [m.strip() for m in meta["modes"].split(",")]:
        print("\n" + "-" * 74)
        print(f"MODE = {mode}")
        print("-" * 74)
        for stratum in ("dark", "light"):
            r = [x for x in rows if x["mode"] == mode and x["stratum"] == stratum
                 and x["status"] == "ok"]
            if not r:
                continue
            fr = [x["mask_area_fraction"] for x in r if x.get("mask_area_fraction") is not None]
            noop = [x for x in r if x.get("masked") and (x.get("mask_area_fraction") or 0) > NOOP_T]
            above = [v for v in fr if v > CIRCLE_CAP]

            print(f"\n  stratum={stratum}  n_ok={len(r)}  masked=True:{sum(1 for x in r if x.get('masked'))}")
            print(f"    corner-trust fired : {sum(1 for x in r if x.get('corner_path_trusted'))} / {len(r)}")
            errs = [x["est_err"] for x in r if x.get("est_err") is not None]
            if errs:
                print(f"    estimator err vs outer-ring truth: median {st.median(errs):+.1f} "
                      f"(min {min(errs):+.1f}, max {max(errs):+.1f}) grey levels")
            print(f"    SILENT NO-OPS (masked & area>{NOOP_T}) : {len(noop)}  "
                  f"({100*len(noop)/len(r):.1f}%)")
            print(f"    area > pi/4 ({CIRCLE_CAP})             : {len(above)}")
            if fr:
                print(f"    area_frac: min {min(fr):.4f}  med {st.median(fr):.4f}  max {max(fr):.4f}")
                print("    distribution:")
                histo(fr)
            if noop:
                print("    no-op sides:")
                for x in sorted(noop, key=lambda y: -y["mask_area_fraction"]):
                    print(f"      {x['id']}/{x['side']:3s} area={x['mask_area_fraction']:.6f} "
                          f"est={x['avg_bg']} truth={x['ring_truth']} err={x['est_err']:+.1f} "
                          f"n_det={x['n_detections']}")

        # Bar 0's contingency table. On this corpus the path variable is
        # degenerate (corner-trust fires 0/574) — report it anyway, explicitly,
        # since "no variance" is itself the finding the amendment predicted.
        r = [x for x in rows if x["mode"] == mode and x["status"] == "ok"]
        print(f"\n  CONTINGENCY  {{corner-trust|fallback}} x {{no-op|healthy}}   (mode={mode})")
        print(f"    {'path':14s} {'no-op':>7s} {'healthy':>8s}")
        for path, sel in (("corner_trust", True), ("fallback", False)):
            sub = [x for x in r if bool(x.get("corner_path_trusted")) is sel]
            n = sum(1 for x in sub if x.get("masked") and (x.get("mask_area_fraction") or 0) > NOOP_T)
            print(f"    {path:14s} {n:7d} {len(sub)-n:8d}")

        # The causal question, on the fallback arm only: does a WRONGER estimate
        # predict a no-op? If the link is real, no-op sides should carry the
        # larger |est_err|.
        r_ok = [x for x in r if x.get("est_err") is not None and x.get("mask_area_fraction") is not None]
        noop = [x for x in r_ok if x["mask_area_fraction"] > NOOP_T]
        heal = [x for x in r_ok if x["mask_area_fraction"] <= NOOP_T]
        if noop and heal:
            print(f"\n  DOES ESTIMATOR ERROR TRACK THE NO-OP?  (mode={mode})")
            print(f"    no-op   n={len(noop):3d}  median |est_err| = {st.median([abs(x['est_err']) for x in noop]):6.1f}"
                  f"   median est {st.median([x['avg_bg'] for x in noop]):6.1f}")
            print(f"    healthy n={len(heal):3d}  median |est_err| = {st.median([abs(x['est_err']) for x in heal]):6.1f}"
                  f"   median est {st.median([x['avg_bg'] for x in heal]):6.1f}")

    if a.compare:
        import csv
        prior = {(x["id"], x["side"]): x for x in csv.DictReader(open(a.compare))
                 if x["status"] == "ok"}
        cur = {(x["id"], x["side"]): x for x in rows
               if x["mode"] == "fullres" and x["status"] == "ok"}
        both = sorted(set(prior) & set(cur))
        print("\n" + "=" * 74)
        print(f"REPRODUCTION vs PRIOR CONTENDED RUN (fullres, n={len(both)} shared sides)")
        print("=" * 74)
        d = [abs(float(prior[k]["mask_area_fraction"]) - cur[k]["mask_area_fraction"])
             for k in both if prior[k]["mask_area_fraction"] and cur[k]["mask_area_fraction"] is not None]
        if d:
            print(f"  |area_frac delta|: median {st.median(d):.6f}  max {max(d):.6f}")
            print(f"  sides differing >0.01: {sum(1 for v in d if v > 0.01)}")
        pn = sum(1 for k in both if float(prior[k]["mask_area_fraction"] or 0) > NOOP_T)
        cn = sum(1 for k in both if (cur[k]["mask_area_fraction"] or 0) > NOOP_T)
        print(f"  no-op count: prior {pn}  clean {cn}   {'MATCH' if pn == cn else 'MISMATCH'}")


if __name__ == "__main__":
    main()
