#!/usr/bin/env python3
"""
Bar-1 / Bar-4 A/B harness for the rim-recovery shape guard.

For each side, runs the REAL `analyze_image` (guard-worktree src) twice -- guard
OFF (env unset) and guard ON (env set) -- and records the full detection set so
the bars can be scored offline. Both lanes route through `analyze_image`; the
lane differs only in the input form + house tag + downstream metric:

  ingest  : raw (lossless PNG) half, house=cng_feature, ALL detections scored
  query   : temp-JPEG half (q95, exactly _mask_query_image_meta's own round
            trip), house=None, LARGEST-contour scored (what DINOv2 sees)

The guard is toggled by env in-process (rim_shape_guard reads env at call time),
so a single worker measures both arms on identical pixels. Hough calls are
counted via a spy so "skipped" is observable. Contours are stored so alpha-mask
IoU can be computed offline. Writes JSONL incrementally (resumable): a re-run
skips (id,side,lane) rows already present in --out.

Read-only w.r.t. production: imports the guard worktree's src, no DB, no
service, nothing written outside --out. One heavy process; --workers bounds the
pool (default 3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WT = "/home/claudeuser/wt-guard"
KS17 = "/home/claudeuser/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images"

# Per-run config injected via env into pool workers (ProcessPoolExecutor on
# Linux forks, so module globals set in the parent before pool creation are
# inherited; we also read env for robustness).
_CFG = {"images_dir": KS17, "layout": "half", "house_ingest": "cng_feature"}
_HOUGH_CALLS = {"n": 0}


def _init_worker():
    """Per-process: put the guard worktree first, import, spy on Hough."""
    if WT not in sys.path:
        sys.path.insert(0, WT)
    import cv2  # noqa
    import src.rim_logic as rl
    real = rl.hough_rim_recovery

    def spy(*a, **k):
        _HOUGH_CALLS["n"] += 1
        return real(*a, **k)

    rl.hough_rim_recovery = spy


def _half(img, side):
    w = img.shape[1]
    mid = w // 2
    return img[:, :mid] if side == "obv" else img[:, mid:]


def _dets_record(res):
    import cv2
    import numpy as np
    out = []
    for d in res.get("detections", []):
        l1 = d["layer_1"]
        c = np.asarray(l1["contour"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(c)
        out.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(cv2.contourArea(c)),
            "circularity": float(l1["geometry"].get("circularity", 0.0)),
            "rim_recovered": bool((l1.get("debug_data") or {}).get("rim_recovered")),
            "contour": c.reshape(-1, 2).tolist(),
        })
    return out


def _run_arm(path, house, guard_on):
    import src.pipeline_manager as pm
    if guard_on:
        os.environ["TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD"] = "1"
        hf = os.environ.get("_BAR_HOLE_FRAC")
        if hf:
            os.environ["TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_MAX_HOLE_FRAC"] = hf
    else:
        os.environ.pop("TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD", None)
        os.environ.pop("TRIVALAYA_RIM_TRIGGER_SHAPE_GUARD_MAX_HOLE_FRAC", None)
    _HOUGH_CALLS["n"] = 0
    t = time.process_time()
    res = pm.analyze_image(path, source_type="auction", house=house)
    cpu = time.process_time() - t
    return {
        "status": res.get("status"), "scale": res.get("scale", 1.0),
        "dets": _dets_record(res), "cpu_s": round(cpu, 2),
        "hough_calls": _HOUGH_CALLS["n"],
    }


_REUSE_OFF = {}  # (id,side)->off record, when --reuse-off is given


def _process(task):
    import cv2
    ident, side, lane, house, input_form = task
    images_dir = os.environ.get("_BAR_IMAGES_DIR", _CFG["images_dir"])
    layout = os.environ.get("_BAR_LAYOUT", _CFG["layout"])
    img = cv2.imread(str(Path(images_dir) / f"{ident}.jpg"))
    if img is None:
        return {"id": ident, "side": side, "lane": lane, "error": "load_failed"}
    half = img if layout == "single" else _half(img, side)
    suffix = ".png" if input_form == "raw" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        p = tf.name
        if input_form == "raw":
            cv2.imwrite(p, half)  # lossless PNG
        else:
            cv2.imwrite(p, half, [cv2.IMWRITE_JPEG_QUALITY, 95])  # matches appv2
    try:
        reuse = _REUSE_OFF.get((ident, side))
        off = reuse if reuse is not None else _run_arm(p, house, guard_on=False)
        on = _run_arm(p, house, guard_on=True)
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass
    return {"id": ident, "side": side, "lane": lane, "house": house,
            "input_form": input_form, "off": off, "on": on,
            "off_reused": reuse is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", required=True, help="JSON [[id,side],...]")
    ap.add_argument("--lane", required=True, choices=["ingest", "query"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hole-frac", default=None, help="set guard hole-frac conjunct")
    ap.add_argument("--images-dir", default=KS17)
    ap.add_argument("--layout", choices=["half", "single"], default="half")
    ap.add_argument("--house-ingest", default="cng_feature",
                    help="house tag for the ingest lane (query lane always None)")
    ap.add_argument("--reuse-off", default=None,
                    help="base AB JSONL: reuse its OFF arm (guard-off is "
                         "config-independent), recompute only ON. For the "
                         "hole-frac re-run without a second expensive OFF pass.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.hole_frac:
        os.environ["_BAR_HOLE_FRAC"] = a.hole_frac  # inherited by workers
    os.environ["_BAR_IMAGES_DIR"] = a.images_dir
    os.environ["_BAR_LAYOUT"] = a.layout

    if a.reuse_off:
        for line in open(a.reuse_off):
            r = json.loads(line)
            if "error" not in r:
                _REUSE_OFF[(r["id"], r["side"])] = r["off"]
        print(f"[ab] reusing OFF for {len(_REUSE_OFF)} sides from {a.reuse_off}")

    house = a.house_ingest if a.lane == "ingest" else None
    input_form = "raw" if a.lane == "ingest" else "jpeg"
    sides = json.load(open(a.sides))
    if a.limit:
        sides = sides[: a.limit]

    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["id"], r["side"], r.get("lane")))
            except Exception:
                pass
    tasks = [(i, s, a.lane, house, input_form) for i, s in sides
             if (i, s, a.lane) not in done]
    print(f"[ab] lane={a.lane} house={house} form={input_form} "
          f"hole_frac={a.hole_frac} sides={len(tasks)} (skip {len(done)}) "
          f"workers={a.workers}", flush=True)

    t0 = time.perf_counter()
    n = 0
    with open(a.out, "a") as f, ProcessPoolExecutor(
            max_workers=a.workers, initializer=_init_worker) as ex:
        for rec in ex.map(_process, tasks):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if "error" not in rec:
                d = rec["off"]["hough_calls"] - rec["on"]["hough_calls"]
                print(f"  [{n}/{len(tasks)}] {rec['id']} {rec['side']} "
                      f"ndet {len(rec['off']['dets'])}->{len(rec['on']['dets'])} "
                      f"hough {rec['off']['hough_calls']}->{rec['on']['hough_calls']} "
                      f"(skip {d}) cpu {rec['off']['cpu_s']}->{rec['on']['cpu_s']}s "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)
            else:
                print(f"  [{n}/{len(tasks)}] {rec['id']} {rec['side']} ERROR", flush=True)
    print(f"[ab] done {n} sides in {time.perf_counter()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
