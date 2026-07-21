"""
Mask-IoU gate + contour-level sliver check for the scale-relative MORPH_CLOSE
rollout.

Implements the two gates that must pass before the per-house table is enabled
in production:

  1. MASK-IoU GATE.  For every lot in a frozen sample, run `control` (fixed
     7x7) and `auto` (scale-relative + per-house override) on one decoded
     image and compare the masks.  Lots whose *detection outcome* is unchanged
     must show IoU ~= 1.0.  Drift there means the change is moving crops on
     lots it was not supposed to touch, which is embedding drift (§7.4) --
     the F6/dp=2.0 lesson that the mask is the embedding input.

  2. §9.3c OPTION 2b -- the contour-level sliver check, which §4.5 left as the
     definitive quality gate and which has never been run.  Crops bake the
     detection contour into alpha (`pipeline_manager.crop_with_alpha`), so a
     sliver is rim pixels of coin A landing inside coin B's alpha mask.  For
     each ordered pair this measures both the contour and its minEnclosing-
     circle, dilated a few px, against the neighbour's filled mask.

Two masks are measured per arm, and the distinction matters:

    binary  -- the post-close segmentation mask, straight from the real
               `_segment_and_extract_candidates`.  Differs between arms by
               construction (different kernel), so it is diagnostic, not a bar.
    alpha   -- the union of the filled detection contours: literally what
               `crop_with_alpha` writes into the alpha channel, and therefore
               what reaches the embedding.  **This is the gated quantity.**

TWO DILATIONS, and the difference decides how to read the result:

    d=0  -- the definitive form.  Do the filled contours actually overlap?
            Overlap here is contamination: opaque pixels of coin B inside
            coin A's alpha, which is literally the sliver.
    d=3  -- "dilated a few px" as §9.3c words it.  This measures *proximity*,
            not contamination, and it is NOT comparable across arms.  A 3px
            dilation bridges any gap under 3px, so an arm whose masks track
            the true flan edge scores worse than one whose masks sit inside
            it -- which is exactly the control/auto difference.  Hough fits a
            circle that under-covers an irregular ancient flan, leaving slack
            that the dilation does not cross; the threshold contour hugs the
            real outline, so it does.  Reading d=3 as "auto has more slivers"
            inverts the actual quality ordering.  Report d=0 for the gate and
            d=3 as a proximity diagnostic.

Note also that `contour` and `circle` mean different things per arm.  On the
`auto` path the crop alpha IS the contour, so `contour_*` is the production
quantity and `circle_*` is hypothetical (what a circular re-crop would catch).
On `control` the Hough split synthesises a *circle* as the contour, so the two
coincide.

STRICTLY READ-ONLY.  Reads the frozen CSV and raw JPEGs from disk or Spaces.
Touches no database, uploads nothing, writes only the report files you name.

Usage:
    # cng_feature -- raws on local disk
    python tools/two_coin_weld_mask_gate.py --house cng_feature \
        --purpose weld_ab --out specs/results/two_coin_weld_maskgate_cng_feature

    # leu -- raws only in Spaces
    python tools/two_coin_weld_mask_gate.py --house leu --purpose leu_ab \
        --source spaces --out specs/results/two_coin_weld_maskgate_leu

Timing is not measured here, so box load does not invalidate the output --
every column is structural.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer1_geometry import (                                    # noqa: E402
    _segment_and_extract_candidates,
    layer_1_structural_salience,
)
from tools.two_coin_weld_ab import (                                 # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_RAW_ROOT,
    DEFAULT_SAMPLE,
    ENV_FRAC,
    _blobs,
    _load,
    _preprocess,
)

DILATE_PX = 3          # "dilated a few px" (§9.3c option 2b)
IOU_BAR = 0.995        # what "~= 1.0" means for an unchanged-outcome lot


# --------------------------------------------------------------------------

def _set_arm(arm: str):
    if arm == "control":
        os.environ.pop(ENV_FRAC, None)
    else:
        os.environ[ENV_FRAC] = arm


def _fill(contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(m, contours, -1, 255, -1)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return inter / union


def _as_contour(obj: Dict) -> Optional[np.ndarray]:
    c = obj.get("contour")
    if c is None:
        return None
    a = np.asarray(c, dtype=np.int32)
    if a.ndim == 2:            # (N,2) -> (N,1,2), the form drawContours wants
        a = a.reshape(-1, 1, 2)
    return a if a.shape[0] >= 3 else None


def _dilate(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def _slivers(objs: List[Dict], shape: Tuple[int, int], px: int) -> Dict:
    """
    §9.3c option 2b.  For each ordered pair (i, j), how much of coin i --
    as its own contour, and as its minEnclosingCircle -- lands inside coin j's
    alpha mask once dilated by `px`.

    Returns the worst pair in the lot.  Fractions are of the *neighbour's*
    area, so "0.01" reads as "a sliver covering 1% of the neighbouring coin".
    """
    cs = [c for c in (_as_contour(o) for o in objs) if c is not None]
    if len(cs) < 2:
        return {"pairs": 0, "contour_px": "", "contour_frac": "",
                "circle_px": "", "circle_frac": ""}

    masks = [_fill([c], shape) for c in cs]
    circles = []
    for c in cs:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        m = np.zeros(shape, dtype=np.uint8)
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
        circles.append(m)

    worst_c_px = worst_c_frac = worst_k_px = worst_k_frac = 0.0
    pairs = 0
    for i in range(len(cs)):
        for j in range(len(cs)):
            if i == j:
                continue
            pairs += 1
            area_j = int(np.count_nonzero(masks[j])) or 1
            c_px = int(np.count_nonzero(_dilate(masks[i], px) & masks[j]))
            k_px = int(np.count_nonzero(_dilate(circles[i], px) & masks[j]))
            worst_c_px = max(worst_c_px, c_px)
            worst_c_frac = max(worst_c_frac, c_px / area_j)
            worst_k_px = max(worst_k_px, k_px)
            worst_k_frac = max(worst_k_frac, k_px / area_j)

    return {
        "pairs": pairs,
        "contour_px": int(worst_c_px),
        "contour_frac": round(worst_c_frac, 5),
        "circle_px": int(worst_k_px),
        "circle_frac": round(worst_k_frac, 5),
    }


def _run_arm(img: np.ndarray, arm: str, house: Optional[str]) -> Dict:
    _set_arm(arm)
    res = layer_1_structural_salience(img, source_type="auction", house=house)
    objs = res.get("objects", []) or []
    tcr = res.get("two_coin_resolution") or {}

    ge, ez, tt, hh, ww = _preprocess(img)
    _, binary = _segment_and_extract_candidates(img, ge, ez, tt, hh, ww,
                                                hh * ww, house=house)
    shape = img.shape[:2]
    contours = [c for c in (_as_contour(o) for o in objs) if c is not None]

    return {
        "objs": objs,
        "binary": (binary > 0),
        "alpha": (_fill(contours, shape) > 0),
        "ndets": len(objs),
        "bboxes": ";".join(",".join(str(v) for v in o["bbox"])
                           for o in objs if "bbox" in o),
        "hough": 1 if tcr.get("method") == "hough" else 0,
        "blobs_post_close": len(_blobs(binary)),
        # Both dilations, because they answer different questions and the
        # dilated one alone is misleading (see the module docstring's note on
        # proximity vs contamination).  d0 is the definitive §9.3c 2b form.
        "sliver": _slivers(objs, shape, 0),
        "sliver_d": _slivers(objs, shape, DILATE_PX),
    }


# --------------------------------------------------------------------------

def _montage(img: np.ndarray, control: Dict, auto: Dict, out: Path, label: str):
    """Side-by-side, contours drawn, for eyeball review of the weld splits."""
    panels = []
    for arm_name, arm in (("control k=7", control), ("auto", auto)):
        p = img.copy()
        for idx, o in enumerate(arm["objs"]):
            c = _as_contour(o)
            colour = [(0, 220, 0), (0, 140, 255), (255, 80, 80),
                      (255, 0, 255), (0, 255, 255)][idx % 5]
            if c is not None:
                cv2.drawContours(p, [c], -1, colour, 2)
            if "bbox" in o:
                x1, y1, x2, y2 = o["bbox"]
                cv2.rectangle(p, (x1, y1), (x2, y2), colour, 1)
        tag = f"{arm_name}  ndets={arm['ndets']}  hough={arm['hough']}"
        cv2.putText(p, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)

    gap = np.full((panels[0].shape[0], 6, 3), 40, dtype=np.uint8)
    strip = np.hstack([panels[0], gap, panels[1]])
    banner = np.full((22, strip.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(banner, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([banner, strip]))


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * len(s)))], 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--source", choices=["local", "spaces"], default="local")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--purpose", default="weld_ab")
    ap.add_argument("--house", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage", type=int, default=12,
                    help="how many changed-outcome lots to montage (0 = none)")
    ap.add_argument("--out", type=Path, default=Path("two_coin_weld_maskgate"))
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
    mont_dir = args.out.parent / f"{args.out.name}_montage"
    if args.montage:
        mont_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict] = []
    montaged = 0
    print(f"{len(rows)} lots, purpose={args.purpose} house={args.house}")

    for n, r in enumerate(rows, 1):
        img = _load(r, args.raw_root, args.source, args.cache_dir)
        if img is None:
            print(f"  [{n}] MISSING {r['house']}/{r['sale_id']}/{r['lot_number']}")
            continue

        control = _run_arm(img, "control", r["house"])
        auto = _run_arm(img, "auto", r["house"])

        changed = (control["ndets"] != auto["ndets"]
                   or control["bboxes"] != auto["bboxes"])

        rec = {
            "house": r["house"], "sale_id": r["sale_id"],
            "lot_id": r["lot_id"], "lot_number": r["lot_number"],
            "w": r["raw_width"], "h": r["raw_height"],
            "outcome_changed": int(changed),
            "binary_iou": round(_iou(control["binary"], auto["binary"]) or 0.0, 5),
            "alpha_iou": round(_iou(control["alpha"], auto["alpha"]) or 0.0, 5),
        }
        for arm_name, arm in (("control", control), ("auto", auto)):
            rec[f"{arm_name}.ndets"] = arm["ndets"]
            rec[f"{arm_name}.hough"] = arm["hough"]
            rec[f"{arm_name}.blobs_post_close"] = arm["blobs_post_close"]
            rec[f"{arm_name}.bboxes"] = arm["bboxes"]
            for k, v in arm["sliver"].items():
                rec[f"{arm_name}.sliver_{k}"] = v
            for k, v in arm["sliver_d"].items():
                rec[f"{arm_name}.sliverd{DILATE_PX}_{k}"] = v
        out_rows.append(rec)

        if changed and montaged < args.montage:
            _montage(img, control, auto,
                     mont_dir / f"{r['house']}_{r['lot_number']}.png",
                     f"{r['house']} sale {r['sale_id']} lot {r['lot_number']}")
            montaged += 1

        if n % 25 == 0:
            print(f"  [{n}/{len(rows)}]")

    # ---- summary ---------------------------------------------------------
    unchanged = [r for r in out_rows if not r["outcome_changed"]]
    changed_rows = [r for r in out_rows if r["outcome_changed"]]
    ua = [r["alpha_iou"] for r in unchanged]
    ub = [r["binary_iou"] for r in unchanged]
    below = [r for r in unchanged if r["alpha_iou"] < IOU_BAR]

    def sliver_stats(rows_, arm, pref="sliver"):
        f = [r[f"{arm}.{pref}_contour_frac"] for r in rows_
             if r[f"{arm}.{pref}_contour_frac"] != ""]
        c = [r[f"{arm}.{pref}_circle_frac"] for r in rows_
             if r[f"{arm}.{pref}_circle_frac"] != ""]
        return {
            "lots_with_pair": len(f),
            "contour_frac_median": _pct(f, 0.5), "contour_frac_max": max(f or [0]),
            "contour_lots_nonzero": sum(1 for x in f if x > 0),
            "circle_frac_median": _pct(c, 0.5), "circle_frac_max": max(c or [0]),
            "circle_lots_nonzero": sum(1 for x in c if x > 0),
        }

    summary = {
        "house": args.house, "purpose": args.purpose, "n": len(out_rows),
        "dilate_px": DILATE_PX, "iou_bar": IOU_BAR,
        "outcome_unchanged": len(unchanged),
        "outcome_changed": len(changed_rows),
        "gate_mask_iou": {
            "unchanged_alpha_iou": {
                "min": min(ua) if ua else None, "p10": _pct(ua, 0.10),
                "median": _pct(ua, 0.50), "mean": round(float(np.mean(ua)), 5) if ua else None,
                "exactly_1": sum(1 for x in ua if x >= 1.0),
                "below_bar": len(below),
            },
            "unchanged_binary_iou": {
                "min": min(ub) if ub else None, "p10": _pct(ub, 0.10),
                "median": _pct(ub, 0.50),
            },
            "PASS": len(below) == 0,
            "offenders": [
                {"lot_number": r["lot_number"], "alpha_iou": r["alpha_iou"],
                 "binary_iou": r["binary_iou"]}
                for r in sorted(below, key=lambda x: x["alpha_iou"])[:20]
            ],
        },
        "gate_sliver_9_3c_2b": {
            "_note": "d0 is the gate; d3 measures proximity and is NOT "
                     "comparable across arms -- see module docstring",
            "d0_changed_outcome_lots": {
                "control": sliver_stats(changed_rows, "control"),
                "auto": sliver_stats(changed_rows, "auto"),
            },
            "d0_all_lots": {
                "control": sliver_stats(out_rows, "control"),
                "auto": sliver_stats(out_rows, "auto"),
            },
            f"d{DILATE_PX}_all_lots": {
                "control": sliver_stats(out_rows, "control", f"sliverd{DILATE_PX}"),
                "auto": sliver_stats(out_rows, "auto", f"sliverd{DILATE_PX}"),
            },
            "PASS_d0_contour": (
                max([r["control.sliver_contour_frac"] for r in out_rows]
                    + [r["auto.sliver_contour_frac"] for r in out_rows]
                    or [0]) == 0
            ),
        },
        "montaged": montaged,
    }

    csv_p, json_p = args.out.with_suffix(".csv"), args.out.with_suffix(".json")
    with open(csv_p, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
    json_p.write_text(json.dumps({"summary": summary, "rows": out_rows},
                                 indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {csv_p}\nwrote {json_p}")
    if montaged:
        print(f"wrote {montaged} montages to {mont_dir}")


if __name__ == "__main__":
    main()
