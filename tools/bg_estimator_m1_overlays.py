#!/usr/bin/env python
"""Bar 4a -- before/after overlay panels for the changed-behavior sides.

Ticket: specs/background_estimator_repair.md, "Bar 4 — AMENDMENT
(owner-ratified 2026-07-28)", clause 4a: adjudicate ALL changed-behavior sides
(a census of 12, not a sample), zero REGRESSED required.

Renders one panel per side per lane:

    [ source side ] [ M1-OFF mask ] [ M1-ON mask ]

with the estimator numbers burned into the header, so an adjudicator can see at
a glance whether the mask went from "the whole frame" to "the coin".  The
masked output is composited on the same mid-grey the query lane uses, which is
the surface the embedding actually sees -- per the image-comparison doctrine,
this is the transparent-grey128 lane, not a raw photo.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

PIPELINE_ROOT = "/home/claudeuser/trivalaya-pipeline"
VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE = 420


def label(img, text, h=28):
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img, size=TILE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    out = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 60, np.uint8)
    y, x = (size - out.shape[0]) // 2, (size - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossing", default=f"{VISION_ROOT}/specs/results/m1_ab/threshold_crossing.csv")
    ap.add_argument("--images", default=f"{PIPELINE_ROOT}/analysis/incoming_screen/KS-17/incoming_images")
    ap.add_argument("--modes", default="query518,fullres")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--worksheet", default=None)
    a = ap.parse_args()

    cv2.setNumThreads(1)
    os.makedirs(a.outdir, exist_ok=True)
    sys.path.insert(0, VISION_ROOT)
    sys.path.insert(1, PIPELINE_ROOT)
    import src.pipeline_manager as pm            # noqa: F401
    import src.math_utils as mu
    import visual_search.appv2 as appv2
    sys.path.insert(0, os.path.join(VISION_ROOT, "tools"))
    from rim_stall_taxonomy import _load_and_resize, split_sides
    from PIL import Image
    from io import BytesIO

    # Trap 1: appv2._mask_query_image_meta hard-inserts the MAIN checkout into
    # sys.path.  The pre-imports above put the worktree's src.* in sys.modules
    # first; assert that actually happened, or the panels would show main's
    # estimator under both arms and look reassuringly identical.
    resolved = os.path.realpath(mu.__file__)
    assert resolved.startswith(os.path.realpath(VISION_ROOT) + os.sep), resolved

    GATE = mu.BG_CORNER_LOCAL_TRUST_ENV
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

    def masked_bgr(pil, gate_on):
        if gate_on:
            os.environ[GATE] = "1"
        else:
            os.environ.pop(GATE, None)
        try:
            out_img, meta = appv2._mask_query_image_meta(pil)
        finally:
            os.environ.pop(GATE, None)
        arr = np.array(out_img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), meta

    rows = [r for r in csv.DictReader(open(a.crossing)) if r["crosses_110"] == "True"]
    rows.sort(key=lambda r: (r["id"], r["side"]))
    print(f"changed-behavior sides (crossing 110): {len(rows)}")

    sheet = []
    for r in rows:
        img = _load_and_resize(os.path.join(a.images, f"{r['id']}.jpg"))
        if img is None:
            continue
        side_bgr = split_sides(img)[r["side"]]
        for mode in [m.strip() for m in a.modes.split(",")]:
            pil = (to_query518_pil(side_bgr) if mode == "query518"
                   else Image.fromarray(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB)))
            src = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            off_img, off_meta = masked_bgr(pil, False)
            on_img, on_meta = masked_bgr(pil, True)

            panel = np.hstack([
                label(fit(src), "source"),
                label(fit(off_img),
                      f"OFF est={r['off_value']} ({r['off_bg_type']}) "
                      f"{r['off_thresh']}"),
                label(fit(on_img),
                      f"ON  est={r['on_value']} ({r['on_bg_type']}) "
                      f"{r['on_thresh']}"),
            ])
            head = np.full((30, panel.shape[1], 3), 12, np.uint8)
            cv2.putText(head, f"{r['id']} {r['side']} [{mode}]  "
                              f"outer-ring truth {r['ring_truth']}  "
                              f"err {r['off_err']} -> {r['on_err']}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.vstack([head, panel])

            name = f"{r['id']}_{r['side']}_{mode}.png"
            cv2.imwrite(os.path.join(a.outdir, name), panel)
            sheet.append({
                "id": r["id"], "side": r["side"], "mode": mode, "panel": name,
                "off_value": r["off_value"], "on_value": r["on_value"],
                "ring_truth": r["ring_truth"],
                "off_thresh": r["off_thresh"], "on_thresh": r["on_thresh"],
                "masked_off": off_meta.get("masked"),
                "masked_on": on_meta.get("masked"),
                "adjudication": "", "note": "",
            })
            print(f"  wrote {name}", flush=True)

    if a.worksheet:
        with open(a.worksheet, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sheet[0].keys()))
            w.writeheader()
            w.writerows(sheet)
        print(f"-> worksheet {a.worksheet}")
    print(f"-> {len(sheet)} panels in {a.outdir}")


if __name__ == "__main__":
    main()
