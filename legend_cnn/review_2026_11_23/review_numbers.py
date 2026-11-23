"""Review of the April 2026 legend-ribbon experiment (2026-11-23): every number in REPORT.md, recomputed from the stored files.

Reads only. Run from the repository root:  python legend_cnn/review_2026_11_23/review_numbers.py
With --fetch-mask-sizes it also reads the first 33 bytes of 240 mask files from Spaces (a random sample, seed 11)
and rewrites mask_dims_sample.json; without it the stored sample is used.
"""
import collections
import csv
import json
import math
import random
import struct
import sys
import urllib.request
from pathlib import Path

L = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
TRAIN_RUN = "run_20260410_013529_promoted_v1"


def side_of(url):
    return "named reverse" if "_rev_" in url else "named obverse" if "_obv_" in url else "other (numbered crop)"


def house(r):
    return r["highres_url"].split("/crops/")[1].split("/")[0]


def sale(r):
    return "/".join(r["highres_url"].split("/crops/")[1].split("/")[:2])


def main():
    rows = list(csv.DictReader(open(L / "data/confusion_subset.csv")))
    by = {int(r["coin_id"]): r for r in rows}
    splits = json.load(open(L / "data/splits_v1.json"))
    tr, va = [by[c] for c in splits["train"]], [by[c] for c in splits["val"]]
    print(f"coins {len(rows)}; train {len(tr)}, val {len(va)}, test {len(splits['test'])}")

    print("\n-- 1. which side of the coin")
    print(dict(collections.Counter(side_of(r["highres_url"]) for r in rows)))

    print("\n-- 2. baselines that never see an image (validation set)")
    prior = collections.Counter(r["authority"] for r in tr)
    n = sum(prior.values())
    maj = prior.most_common(1)[0][0]
    print(f"largest class {maj}: {sum(r['authority'] == maj for r in va) / len(va):.3f}")
    print(f"entropy of the class shares (nats): {-sum(v / n * math.log(v / n) for v in prior.values()):.3f}; ln(10) = {math.log(10):.3f}")
    for name, key in (("dealer", house), ("sale", sale)):
        m = collections.defaultdict(collections.Counter)
        for r in tr:
            m[key(r)][r["authority"]] += 1
        acc = sum((m[key(r)].most_common(1)[0][0] if key(r) in m else maj) == r["authority"] for r in va) / len(va)
        print(f"most common authority of the coin's {name} ({len(m)} in train): {acc:.3f}")
    print(f"sampling error at 19.6% on {len(va)} coins: {100 * math.sqrt(0.196 * 0.804 / len(va)):.2f} points (one standard error)")

    print("\n-- 3. training logs (validation only; no test output exists)")
    for p in sorted((L / "outputs").glob("train_log_run*.csv")):
        r = list(csv.DictReader(open(p)))
        best = min(r, key=lambda x: float(x["val_loss"]))
        print(f"{p.name}: epochs {len(r)}; at lowest val loss (epoch {best['epoch']}): val acc {float(best['val_acc']):.4f}; "
              f"best val acc any epoch {max(float(x['val_acc']) for x in r):.4f}; train loss first {float(r[0]['train_loss']):.3f} last {float(r[-1]['train_loss']):.3f}")

    print("\n-- 4. geometry of the training ribbons, from the run's own ribbon_stats.csv (the images are 512 x 512)")
    st = list(csv.DictReader(open(L / "runs" / TRAIN_RUN / "ribbon_stats.csv")))
    fb = [r for r in st if r["used_fallback"] == "True"]
    nf = [r for r in st if r["used_fallback"] != "True"]
    big = [r for r in nf if float(r["radius"]) > 256]
    rest = [r for r in nf if float(r["radius"]) <= 256]
    off = [r for r in rest if float(r["center_offset"]) > 40]
    near = [r for r in rest if float(r["center_offset"]) <= 40]
    for name, g in (("fixed fallback centre (256, 256), radius 220", fb), ("own centre, radius above 256 px (larger than the image allows)", big),
                    ("own centre, radius within 256 px, centre over 40 px from the image centre", off), ("own centre, radius within 256 px, centre within 40 px", near)):
        print(f"{len(g):5d}  {len(g) / len(st):6.1%}  {name}")
    print("fallback reasons:", dict(collections.Counter(r["fallback_reason"] for r in fb)))
    print(f"'good' bucket: {sum(r['quality_bucket'] == 'good' for r in st)} of {len(st)}; among radius above 256 px: {sum(r['quality_bucket'] == 'good' for r in big)} of {len(big)}")
    print(f"ribbons less than half non-black: {sum(float(r['nonblack_fraction']) < 0.5 for r in st)} of {len(st)}")
    for run in (TRAIN_RUN, "run_20260410_161829_promoted_v1_h128"):
        c = json.load(open(L / "runs" / run / "config.json"))
        rows_ = int(c["warp_radius_bins"] * c["outer_r_ratio"]) - int(c["warp_radius_bins"] * c["inner_r_ratio"])
        print(f"{run}: radial samples in the band {rows_}, stored ribbon height {c['ribbon_height']}")

    print("\n-- 5. size of the mask files (the images are 512 x 512)")
    p = HERE / "mask_dims_sample.json"
    if "--fetch-mask-sizes" in sys.argv:
        random.seed(11)
        got = []
        for r in random.sample(rows, 240):
            try:
                req = urllib.request.Request(r["transparent_url"], headers={"Range": "bytes=0-32"})
                b = urllib.request.urlopen(req, timeout=15).read(33)
                w, h = struct.unpack(">II", b[16:24])
                got.append([r["coin_id"], w, h])
            except Exception:
                pass
        json.dump(got, open(p, "w"))
    d = json.load(open(p))
    ws = sorted(w for _, w, h in d)
    print(f"masks read {len(d)} (sample of 240, seed 11); exactly 512 x 512: {sum(w == 512 and h == 512 for _, w, h in d)}; "
          f"width 5th / median / 95th percentile: {ws[int(.05 * (len(ws) - 1))]} / {ws[int(.5 * (len(ws) - 1))]} / {ws[int(.95 * (len(ws) - 1))]}; "
          f"sides differing by over 5%: {sum(abs(w - h) / max(w, h) > 0.05 for _, w, h in d)}")


if __name__ == "__main__":
    main()
