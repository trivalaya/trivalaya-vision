#!/usr/bin/env python
"""M1 bar report -- turns the A/B jsonl pair into per-bar PASS/FAIL verdicts.

Ticket: specs/background_estimator_repair.md.  Thresholds are the RATIFIED ones
and are hardcoded here so a report run cannot quietly soften them:

  Bar 1  |estimator - outer_ring_truth| <= 8 on >= 95% of sides
  Bar 3  silent no-ops (masked AND area > 0.9) -> 0 on fixture sides;
         0 NEW no-ops across the full population, both lanes
  Bar 4  40 seeded stratified changed sides for adjudication (this script
         draws and records the sample; adjudication is a separate pass)
  Bar 5  both lanes reported at their own geometry, never derived

Bar 2 is asserted by tests/test_bg_corner_local_trust.py and bounded across the
corpus by tools/bg_corner_trust_census.py; within KS-17 it is vacuous (pooled
corner-trust fires 0/574) and this script says so rather than implying coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import Counter, defaultdict

NOOP_AREA_T = 0.9          # contour area / frame area; pi/4 ~ 0.785 caps a real coin
BAR1_TOL = 8.0
BAR1_MIN_RATE = 0.95
BAR4_SAMPLE = 40
BAR4_SEED = 20260728
CHANGE_AREA_EPS = 0.01     # below this an area delta is numerical, not behavioral


def load(path):
    rows, meta = [], None
    for line in open(path):
        r = json.loads(line)
        if r.get("_meta"):
            meta = r
        else:
            rows.append(r)
    return meta, rows


def key(r):
    return (r["id"], r["side"], r["mode"])


def is_noop(r):
    return bool(r.get("masked")) and (r.get("mask_area_fraction") or 0) > NOOP_AREA_T


def pct(n, d):
    return f"{(100.0*n/d):.1f}%" if d else "n/a"


def dist(vals):
    if not vals:
        return "n/a"
    s = sorted(vals)
    return (f"median {st.median(s):+.2f}  p05 {s[int(.05*len(s))]:+.2f}  "
            f"p95 {s[min(int(.95*len(s)), len(s)-1)]:+.2f}  "
            f"min {s[0]:+.2f}  max {s[-1]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--fixtures", default=None,
                    help="bg_estimator_bar0_clean_2026-07-27.jsonl (Bar 3 fixture sides)")
    ap.add_argument("--classify", default=None,
                    help="rim_stall_taxonomy_ks17_classified.csv (Bar 4 strata)")
    ap.add_argument("--bar4-out", default=None)
    ap.add_argument("--changed-out", default=None)
    a = ap.parse_args()

    meta_off, off_rows = load(a.off)
    meta_on, on_rows = load(a.on)
    off = {key(r): r for r in off_rows}
    on = {key(r): r for r in on_rows}

    print("=" * 78)
    print("M1 BAR REPORT -- background estimator repair")
    print("=" * 78)
    for nm, m, rows in (("OFF", meta_off, off_rows), ("ON", meta_on, on_rows)):
        bad = [r for r in rows if r.get("status") != "ok"]
        print(f"{nm:3s} gate={m['gate']:3s} sides={m['n_sides']} tasks={m['n_tasks']} "
              f"rows={len(rows)} non-ok={len(bad)} vision_root={m['vision_root']}")
        if bad:
            print(f"     non-ok statuses: {Counter(r.get('status') for r in bad)}")
    shared = sorted(set(off) & set(on))
    print(f"shared (id,side,mode) tasks: {len(shared)}")
    modes = sorted({k[2] for k in shared})

    ok = [k for k in shared
          if off[k].get("status") == "ok" and on[k].get("status") == "ok"]
    print(f"both-ok tasks: {len(ok)}\n")

    verdicts = {}

    # ── Bar 1 ────────────────────────────────────────────────────────────
    print("-" * 78)
    print("BAR 1 -- estimator within +/-8 of outer-ring truth on >=95% of sides")
    print("-" * 78)
    bar1_pass = True
    for mode in modes:
        ks = [k for k in ok if k[2] == mode]
        for label, src in (("OFF (today)", off), ("ON  (M1)", on)):
            errs = [src[k]["est_err"] for k in ks if src[k].get("est_err") is not None]
            good = sum(1 for e in errs if abs(e) <= BAR1_TOL)
            rate = good / len(errs) if errs else 0
            flag = ""
            if label.startswith("ON"):
                ok_bar = rate >= BAR1_MIN_RATE
                bar1_pass &= ok_bar
                flag = "  <= PASS" if ok_bar else "  <= FAIL"
            print(f"  {mode:9s} {label:12s} n={len(errs):4d} within+/-8 "
                  f"{good:4d} ({pct(good,len(errs)):>6s}){flag}")
            print(f"            err dist: {dist(errs)}")
    verdicts["Bar 1"] = bar1_pass
    print(f"  => Bar 1 {'PASS' if bar1_pass else 'FAIL'}\n")

    # ── Bar 3 ────────────────────────────────────────────────────────────
    print("-" * 78)
    print("BAR 3 -- silent no-op class eliminated, none created")
    print("-" * 78)
    bar3_pass = True
    new_noops_all = []
    for mode in modes:
        ks = [k for k in ok if k[2] == mode]
        n_off = [k for k in ks if is_noop(off[k])]
        n_on = [k for k in ks if is_noop(on[k])]
        new = [k for k in ks if not is_noop(off[k]) and is_noop(on[k])]
        fixed = [k for k in ks if is_noop(off[k]) and not is_noop(on[k])]
        new_noops_all += new
        bar3_pass &= (len(n_on) == 0 and len(new) == 0)
        print(f"  {mode:9s} n={len(ks):4d}  no-ops OFF {len(n_off):3d} -> ON {len(n_on):3d}"
              f"   fixed {len(fixed):3d}   NEW {len(new):3d}"
              f"{'  <= FAIL' if (n_on or new) else '  <= PASS'}")
        for k in n_on:
            print(f"      REMAINING no-op {k[0]}/{k[1]} area={on[k]['mask_area_fraction']} "
                  f"est={on[k]['avg_bg']} ({on[k]['bg_type']})")
        for k in new:
            print(f"      NEW no-op {k[0]}/{k[1]} "
                  f"area {off[k]['mask_area_fraction']} -> {on[k]['mask_area_fraction']}")

    if a.fixtures:
        _, frows = load(a.fixtures)
        fix = [r for r in frows if r.get("status") == "ok"
               and bool(r.get("masked")) and (r.get("mask_area_fraction") or 0) > NOOP_AREA_T]
        print(f"\n  Bar 0 fixture no-op sides: {len(fix)} "
              f"({Counter(r['mode'] for r in fix)})")
        miss, still = 0, 0
        for r in fix:
            k = (r["id"], r["side"], r["mode"])
            if k not in on:
                miss += 1
                continue
            if is_noop(on[k]):
                still += 1
                print(f"      STILL a no-op under M1: {k}")
        print(f"  fixture sides resolved under M1: {len(fix)-still-miss}/{len(fix)}"
              f"  (still no-op {still}, not covered {miss})")
        bar3_pass &= (still == 0)

    # mask_fallback_reason transitions, enumerated as the bar requires
    trans = Counter()
    for k in ok:
        t = (off[k].get("mask_fallback_reason"), on[k].get("mask_fallback_reason"))
        if t[0] != t[1]:
            trans[t] += 1
    print(f"\n  mask_fallback_reason transitions: "
          f"{dict(trans) if trans else 'none (no side changed reason)'}")
    verdicts["Bar 3"] = bar3_pass
    print(f"  => Bar 3 {'PASS' if bar3_pass else 'FAIL'}\n")

    # ── changed set (feeds Bars 4 and 5) ────────────────────────────────
    def changed(k):
        o, n = off[k], on[k]
        if bool(o.get("masked")) != bool(n.get("masked")):
            return True
        if o.get("n_dets_raw") != n.get("n_dets_raw"):
            return True
        ao, an = o.get("mask_area_fraction"), n.get("mask_area_fraction")
        if (ao is None) != (an is None):
            return True
        if ao is not None and abs(ao - an) > CHANGE_AREA_EPS:
            return True
        return False

    ch = [k for k in ok if changed(k)]
    est_ch = [k for k in ok if off[k]["avg_bg"] != on[k]["avg_bg"]]
    print("-" * 78)
    print("CHANGED SIDES (input to Bars 4 and 5)")
    print("-" * 78)
    for mode in modes:
        km = [k for k in ok if k[2] == mode]
        cm = [k for k in ch if k[2] == mode]
        em = [k for k in est_ch if k[2] == mode]
        print(f"  {mode:9s} n={len(km):4d}  estimator changed {len(em):4d} "
              f"({pct(len(em),len(km)):>6s})   MASK OUTCOME changed {len(cm):4d} "
              f"({pct(len(cm),len(km)):>6s})")

    if a.changed_out:
        with open(a.changed_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "side", "mode", "est_off", "est_on", "type_off", "type_on",
                        "truth", "area_off", "area_on", "dets_off", "dets_on",
                        "noop_off", "noop_on"])
            for k in ch:
                o, n = off[k], on[k]
                w.writerow([k[0], k[1], k[2], o["avg_bg"], n["avg_bg"],
                            o["bg_type"], n["bg_type"], o["ring_truth"],
                            o["mask_area_fraction"], n["mask_area_fraction"],
                            o.get("n_dets_raw"), n.get("n_dets_raw"),
                            is_noop(o), is_noop(n)])
        print(f"  changed-sides CSV -> {a.changed_out}")

    # ── Bar 4 sample ────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f"BAR 4 -- seeded stratified sample of {BAR4_SAMPLE} changed sides "
          f"(seed {BAR4_SEED})")
    print("-" * 78)
    strata = {}
    if a.classify:
        per_side = defaultdict(list)
        for r in csv.DictReader(open(a.classify)):
            per_side[(r["id"], r["side"])].append(r["mechanism"])
        for sk, mechs in per_side.items():
            strata[sk] = Counter(mechs).most_common(1)[0][0]

    def stratum_of(k):
        return strata.get((k[0], k[1]), "unclassified")

    buckets = defaultdict(list)
    for k in ch:
        buckets[stratum_of(k)].append(k)
    rng = random.Random(BAR4_SEED)
    picked = []
    if ch:
        names = sorted(buckets)
        # proportional allocation, at least one per non-empty stratum
        for nm in names:
            buckets[nm].sort()
        alloc = {nm: max(1, round(BAR4_SAMPLE * len(buckets[nm]) / len(ch)))
                 for nm in names}
        for nm in names:
            picked += rng.sample(buckets[nm], min(alloc[nm], len(buckets[nm])))
        rng.shuffle(picked)
        picked = sorted(picked[:BAR4_SAMPLE])
    for nm in sorted(buckets):
        print(f"  {nm:28s} changed {len(buckets[nm]):4d}   sampled "
              f"{sum(1 for k in picked if stratum_of(k)==nm):3d}")
    print(f"  total sampled: {len(picked)}")
    if a.bar4_out:
        with open(a.bar4_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "side", "mode", "stratum", "est_off", "est_on",
                        "type_off", "type_on", "truth", "area_off", "area_on",
                        "dets_off", "dets_on", "adjudication", "note"])
            for k in picked:
                o, n = off[k], on[k]
                w.writerow([k[0], k[1], k[2], stratum_of(k), o["avg_bg"], n["avg_bg"],
                            o["bg_type"], n["bg_type"], o["ring_truth"],
                            o["mask_area_fraction"], n["mask_area_fraction"],
                            o.get("n_dets_raw"), n.get("n_dets_raw"), "", ""])
        print(f"  Bar 4 worksheet -> {a.bar4_out}")

    # ── Bar 5 lanes + preview ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print("BAR 5 -- both lanes measured at their own geometry")
    print("-" * 78)
    for mode in modes:
        km = [k for k in ok if k[2] == mode]
        w = [off[k]["w"] for k in km]
        print(f"  {mode:9s} n={len(km):4d}  width median {st.median(w):.0f}px "
              f"(range {min(w)}-{max(w)})")
    print(f"  lanes present: {modes}  -> {'OK' if len(modes) >= 2 else 'MISSING A LANE'}")

    print("\n" + "-" * 78)
    print("REPORTED (not gated) -- detection-count and cost preview")
    print("-" * 78)
    for mode in modes:
        km = [k for k in ok if k[2] == mode]
        for label, src in (("OFF", off), ("ON ", on)):
            dets = [src[k].get("n_dets_raw") for k in km
                    if isinstance(src[k].get("n_dets_raw"), int)]
            el = [src[k]["elapsed_s"] for k in km if src[k].get("elapsed_s") is not None]
            sat = sum(1 for d in dets if d >= 5)
            print(f"  {mode:9s} {label} dets/side mean "
                  f"{(sum(dets)/len(dets) if dets else 0):.2f}  "
                  f">=5 dets {sat:4d} ({pct(sat,len(dets)):>6s})  "
                  f"mask secs median {(st.median(el) if el else 0):.3f} "
                  f"total {sum(el)/60:.1f}m")

    print("\n" + "=" * 78)
    for b, v in verdicts.items():
        print(f"  {b}: {'PASS' if v else 'FAIL'}")
    print("  Bar 2: see tests/test_bg_corner_local_trust.py (asserted) + census "
          "(corpus bound); vacuous within KS-17 (pooled corner-trust fires 0/574)")
    print("  Bar 4: sample drawn above; adjudication is a separate pass")
    print("=" * 78)


if __name__ == "__main__":
    main()
