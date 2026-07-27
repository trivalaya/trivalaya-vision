#!/usr/bin/env python
"""Bar 4b -- byte-compare masks on the behaviorally-INERT sides.

Ticket: specs/background_estimator_repair.md, "Bar 4 — AMENDMENT
(owner-ratified 2026-07-28)", clause 4b.

The structural finding claims that on the 560 sides where M1 changes the
estimator's VALUE but the value stays on the same side of the 110 threshold,
downstream behavior is identical.  That claim is an argument from the call site
(`avg_bg > 110` is the only read).  This gives it teeth: take a seeded sample
of 40 such sides, produce the actual mask under M1-off and M1-on, and compare
the resulting pixel buffers BYTE FOR BYTE.

Expected identical on 40/40.  **Any non-identical mask is an automatic FAIL**
of the inertness claim and reopens the blast-radius question -- it does not get
explained away as numerical noise, because there is no float path between the
two runs that could legitimately differ.

Both gates run in ONE process, alternating per side, because
`detect_background_histogram` reads the env var at call time (no caching).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time

import cv2
import numpy as np

PIPELINE_ROOT = "/home/claudeuser/trivalaya-pipeline"
VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR4B_SEED = 20260728
BAR4B_N = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossing", default=f"{VISION_ROOT}/specs/results/m1_ab/threshold_crossing.csv")
    ap.add_argument("--images", default=f"{PIPELINE_ROOT}/analysis/incoming_screen/KS-17/incoming_images")
    ap.add_argument("--modes", default="query518,fullres")
    ap.add_argument("--n", type=int, default=BAR4B_N)
    ap.add_argument("--seed", type=int, default=BAR4B_SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cv2.setNumThreads(1)
    sys.path.insert(0, VISION_ROOT)
    sys.path.insert(1, PIPELINE_ROOT)
    import src.pipeline_manager as pm            # noqa: F401  (pre-import, see A/B probe)
    import src.math_utils as mu
    import visual_search.appv2 as appv2
    sys.path.insert(0, os.path.join(VISION_ROOT, "tools"))
    from rim_stall_taxonomy import _load_and_resize, split_sides
    from PIL import Image
    from io import BytesIO

    resolved = os.path.realpath(mu.__file__)
    assert resolved.startswith(os.path.realpath(VISION_ROOT) + os.sep), resolved
    GATE = mu.BG_CORNER_LOCAL_TRUST_ENV

    # Population: value changed, but NOT across the 110 threshold.
    rows = list(csv.DictReader(open(a.crossing)))
    inert = [r for r in rows
             if r["value_changed"] == "True" and r["crosses_110"] == "False"]
    inert.sort(key=lambda r: (r["id"], r["side"]))
    rng = random.Random(a.seed)
    sample = sorted(rng.sample(inert, min(a.n, len(inert))),
                    key=lambda r: (r["id"], r["side"]))
    print(f"inert population: {len(inert)}   sample: {len(sample)} "
          f"(seed={a.seed})", flush=True)

    UPLOAD_MAX_DIM = appv2.UPLOAD_MAX_DIM

    def to_query518_pil(bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        raw = buf.tobytes()
        img = Image.open(BytesIO(raw))
        w, h = img.size
        if max(w, h) <= UPLOAD_MAX_DIM:
            return Image.open(BytesIO(raw)).convert("RGB")
        if img.format == "JPEG":
            scale = max(w, h) / UPLOAD_MAX_DIM
            for s in (8, 4, 2):
                if scale >= s:
                    try:
                        img.draft("RGB", (w // s, h // s))
                    except Exception:
                        pass
                    break
        img = img.convert("RGB")
        img.thumbnail((UPLOAD_MAX_DIM, UPLOAD_MAX_DIM), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=90, optimize=True)
        return Image.open(BytesIO(out.getvalue())).convert("RGB")

    def mask_digest(pil, gate_on):
        if gate_on:
            os.environ[GATE] = "1"
        else:
            os.environ.pop(GATE, None)
        try:
            out_img, meta = appv2._mask_query_image_meta(pil)
        finally:
            os.environ.pop(GATE, None)
        arr = np.asarray(out_img.convert("RGBA"))
        return (hashlib.sha256(arr.tobytes()).hexdigest(), arr.shape,
                bool(meta.get("masked")), meta.get("mask_fallback_reason"))

    results = []
    t0 = time.time()
    modes = [m.strip() for m in a.modes.split(",")]
    for i, r in enumerate(sample, 1):
        img = _load_and_resize(os.path.join(a.images, f"{r['id']}.jpg"))
        if img is None:
            results.append({"id": r["id"], "side": r["side"], "mode": "-",
                            "verdict": "LOAD_FAILED"})
            continue
        side_bgr = split_sides(img)[r["side"]]
        for mode in modes:
            pil = (to_query518_pil(side_bgr) if mode == "query518"
                   else Image.fromarray(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB)))
            d_off = mask_digest(pil, False)
            d_on = mask_digest(pil, True)
            same = (d_off[0] == d_on[0] and d_off[1] == d_on[1]
                    and d_off[2] == d_on[2] and d_off[3] == d_on[3])
            results.append({
                "id": r["id"], "side": r["side"], "mode": mode,
                "off_value": r["off_value"], "on_value": r["on_value"],
                "sha_off": d_off[0][:16], "sha_on": d_on[0][:16],
                "shape_off": str(d_off[1]), "shape_on": str(d_on[1]),
                "masked_off": d_off[2], "masked_on": d_on[2],
                "verdict": "IDENTICAL" if same else "DIFFERS",
            })
            if not same:
                print(f"  !! DIFFERS {r['id']}/{r['side']}/{mode} "
                      f"{d_off[0][:16]} vs {d_on[0][:16]}", flush=True)
        print(f"  [{i}/{len(sample)}] {r['id']}/{r['side']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    graded = [r for r in results if r["verdict"] in ("IDENTICAL", "DIFFERS")]
    ident = sum(1 for r in graded if r["verdict"] == "IDENTICAL")
    diff = sum(1 for r in graded if r["verdict"] == "DIFFERS")
    print(f"\nBAR 4b: {ident}/{len(graded)} mask comparisons IDENTICAL, "
          f"{diff} DIFFER")
    for mode in modes:
        g = [r for r in graded if r["mode"] == mode]
        print(f"  {mode:9s} {sum(1 for r in g if r['verdict']=='IDENTICAL')}"
              f"/{len(g)} identical")
    print(f"  => Bar 4b {'PASS' if diff == 0 else 'FAIL (inertness claim refuted)'}")
    print(f"-> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
