"""
Scope A precommit measurement: rim-recovery Hough cost, baseline vs
candidate configs. See specs/rim_recovery_neighbor_aware.md, "PRECOMMIT
ACCEPTANCE BARS / Scope A".

Fixture: KS-17 raws (~/trivalaya-pipeline/analysis/incoming_screen/KS-17/
incoming_images/), 287 combined obv+rev JPGs (3000x1440) -> 574 sides via
the same mid-split `load_sides()` uses in corpus_match_report.py. A stride
sample is used (not the full 287) because the baseline config alone can
cost 20-166 CPU-s per stalling side (specs/results/
ks17_mask_stall_diagnosis_2026-07-22.md) -- the full set would cost hours
per config x lane on this 4-vCPU box.

For each side x lane x config: wall-clock (perf_counter), CPU-seconds
(process_time), status, n_detections, per-detection rim_recovered flags,
and an alpha mask -- union of filled detection contours for the `ingest`
lane, largest-contour-only for the `query` lane (matching
appv2._mask_query_image_meta's own selection exactly). IoU against the
baseline config is computed immediately per side so full contours never
have to be serialized to disk.

Lanes (house is the only difference -- both call the real, unmodified
analyze_image):
  ingest -- analyze_image(path, source_type="auction", house="cng_feature")
            (matches trivalaya_pipeline/pipeline.py:612's real ingest call
            for this house)
  query  -- analyze_image(path, source_type="auction")           (no house;
            matches appv2._mask_query_image_meta / decode_crop.py exactly)

TRIVALAYA_CLOSE_KERNEL_FRAC is deliberately left UNSET for every arm here
(today's live production default -- the two-coin-weld lane's flip is a
separate, already-decided front) so this isolates the rim-recovery knobs
under test.

Configs: `baseline`, `cap800` (Scope A1, env-value only), `largest1`
(Scope A2, TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE=1), and `cap800_largest1`
(both together).

STRICTLY READ-ONLY except the CSV/JSON this writes. No DB, no Spaces, no
service restart.

Usage:
    .venv/bin/python tools/rim_recovery_cost_ab.py \
        --images-dir ~/trivalaya-pipeline/analysis/incoming_screen/KS-17/incoming_images \
        --stride 7 --out specs/results/rim_recovery_cost_ab_ks17
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_manager import analyze_image  # noqa: E402

ENV_CAP = "TRIVALAYA_RIM_HOUGH_CAP"
ENV_MAX_RECOVERY = "TRIVALAYA_RIM_RECOVERY_MAX_PER_IMAGE"

CONFIGS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "cap800": {ENV_CAP: "800"},
    "cap1024": {ENV_CAP: "1024"},
    "largest1": {ENV_MAX_RECOVERY: "1"},
    "cap800_largest1": {ENV_CAP: "800", ENV_MAX_RECOVERY: "1"},
}

LANES = {
    "ingest": {"house": "cng_feature"},
    "query": {"house": None},
}


def _set_env(cfg: Dict[str, str]):
    for k in (ENV_CAP, ENV_MAX_RECOVERY):
        os.environ.pop(k, None)
    for k, v in cfg.items():
        os.environ[k] = v


def _split_sides(path: Path, tmpdir: Path) -> Tuple[Path, Path]:
    """Mid-split a combined CNG JPG into (obv, rev) temp files, once per image."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    mid = w // 2
    obv, rev = img[:, :mid], img[:, mid:]
    obv_p = tmpdir / f"{path.stem}_obv.jpg"
    rev_p = tmpdir / f"{path.stem}_rev.jpg"
    cv2.imwrite(str(obv_p), obv)
    cv2.imwrite(str(rev_p), rev)
    return obv_p, rev_p


def _fill(contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(m, contours, -1, 255, -1)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return None if union == 0 else inter / union


def _contours(dets: List[Dict]) -> List[np.ndarray]:
    out = []
    for d in dets:
        c = d.get("layer_1", {}).get("contour")
        if c is None:
            continue
        a = np.asarray(c, dtype=np.int32)
        if a.ndim == 2:
            a = a.reshape(-1, 1, 2)
        if a.shape[0] >= 3:
            out.append(a)
    return out


def _run(path: Path, house: Optional[str], shape: Tuple[int, int], lane: str) -> Dict:
    t0_wall, t0_cpu = time.perf_counter(), time.process_time()
    res = analyze_image(str(path), source_type="auction", house=house)
    wall, cpu = time.perf_counter() - t0_wall, time.process_time() - t0_cpu

    status = res.get("status")
    dets = res.get("detections", []) if status == "success" else []
    contours = _contours(dets)

    if lane == "query" and len(contours) > 1:
        # Mirrors _mask_query_image_meta: largest contour wins.
        contours = [max(contours, key=cv2.contourArea)]

    recovered_flags = [
        bool(d.get("layer_1", {}).get("debug_data", {}).get("rim_recovered"))
        for d in dets
    ]
    return {
        "status": status,
        "wall_s": round(wall, 3),
        "cpu_s": round(cpu, 3),
        "n_detections": len(dets),
        "n_rim_recovered": sum(recovered_flags),
        "rim_recovered": ";".join(str(int(x)) for x in recovered_flags),
        "alpha": (_fill(contours, shape) > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=7,
                     help="take every Nth image (sorted by filename) to bound cost")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="baseline,cap800")
    ap.add_argument("--lanes", default="ingest,query")
    ap.add_argument("--out", type=Path, default=Path("rim_recovery_cost_ab"))
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            sys.exit(f"unknown config {c!r}; known: {list(CONFIGS)}")
    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]

    files = sorted(args.images_dir.glob("*.jpg"))[::args.stride]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no images found under {args.images_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} images ({len(files) * 2} sides), "
          f"configs={configs}, lanes={lanes}")

    rows: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="rim_cost_ab_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        for n, path in enumerate(files, 1):
            obv_p, rev_p = _split_sides(path, tmpdir)
            for side, side_path in (("obv", obv_p), ("rev", rev_p)):
                img = cv2.imread(str(side_path))
                shape = img.shape[:2]
                del img
                for lane in lanes:
                    house = LANES[lane]["house"]
                    baseline_alpha = None
                    for cfg in configs:
                        _set_env(CONFIGS[cfg])
                        r = _run(side_path, house, shape, lane)
                        alpha = r.pop("alpha")
                        if cfg == "baseline":
                            baseline_alpha = alpha
                            iou = 1.0
                        else:
                            iou = _iou(baseline_alpha, alpha)
                        rows.append({
                            "id": path.stem, "side": side, "lane": lane,
                            "config": cfg, "alpha_iou_vs_baseline": iou,
                            **r,
                        })
            if n % 5 == 0 or n == len(files):
                print(f"  [{n}/{len(files)}]")
        _set_env({})  # restore clean env

    csv_p, json_p = args.out.with_suffix(".csv"), args.out.with_suffix(".json")
    with open(csv_p, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # ---- summary ----------------------------------------------------------
    def pct(vals, q):
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(q * len(s)))], 3) if s else None

    summary = {}
    for lane in lanes:
        for cfg in configs:
            sub = [r for r in rows if r["lane"] == lane and r["config"] == cfg]
            cpu = [r["cpu_s"] for r in sub]
            wall = [r["wall_s"] for r in sub]
            stall = [r for r in sub if r["wall_s"] > 20]
            unchanged = [r for r in sub if r["config"] == "baseline" or
                         (r["n_detections"] ==
                          next(b["n_detections"] for b in rows
                               if b["lane"] == lane and b["config"] == "baseline"
                               and b["id"] == r["id"] and b["side"] == r["side"])
                          and r["rim_recovered"] ==
                          next(b["rim_recovered"] for b in rows
                               if b["lane"] == lane and b["config"] == "baseline"
                               and b["id"] == r["id"] and b["side"] == r["side"]))]
            ious = [r["alpha_iou_vs_baseline"] for r in unchanged
                    if r["alpha_iou_vs_baseline"] is not None]
            summary[f"{lane}/{cfg}"] = {
                "n": len(sub),
                "n_stall_gt20s_wall": len(stall),
                "cpu_s": {"p50": pct(cpu, 0.5), "p90": pct(cpu, 0.9),
                          "p99": pct(cpu, 0.99), "max": max(cpu) if cpu else None,
                          "sum": round(sum(cpu), 1)},
                "wall_s": {"p50": pct(wall, 0.5), "p90": pct(wall, 0.9),
                           "p99": pct(wall, 0.99), "max": max(wall) if wall else None,
                           "sum": round(sum(wall), 1)},
                "outcome_unchanged": len(unchanged),
                "alpha_iou_vs_baseline": {
                    "min": min(ious) if ious else None,
                    "median": pct(ious, 0.5),
                    "below_0995": sum(1 for x in ious if x < 0.995),
                },
            }
    json_p.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {csv_p}\nwrote {json_p}")


if __name__ == "__main__":
    main()
