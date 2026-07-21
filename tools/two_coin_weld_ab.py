"""
Dual-run A/B harness for the scale-relative MORPH_CLOSE change.

Implements specs/two_coin_weld_morph_close.md §9.3d (paired A/B) and the
per-lot measurements §4.1 asks for.

Runs each lot from the frozen sample through two or more kernel arms on
*identical* in-memory input, and reports hough rate, ndets distribution,
wall-clock, bbox deltas and the weld signature.

Why paired runs matter: production overwrites crops in place at the same
Spaces keys (§3), so for an already-processed lot the pre-change output no
longer exists. Running both arms here, in one process, against one decoded
image is the only way to get a true paired comparison.

STRICTLY READ-ONLY. Reads the frozen CSV and raw JPEGs from disk or Spaces.
Touches no database, uploads nothing, and writes only the report files you
name plus a raw cache under --cache-dir.

Usage:
    python tools/two_coin_weld_ab.py --limit 5            # smoke test
    python tools/two_coin_weld_ab.py                      # full frozen sample
    python tools/two_coin_weld_ab.py --arms control,auto,0.0025
    python tools/two_coin_weld_ab.py --out /tmp/ab        # -> /tmp/ab.csv + .json

    # leu -- raws exist only in Spaces (specs/two_coin_weld_handoff.md)
    python tools/two_coin_weld_ab.py --purpose leu_ab --source spaces \
        --out specs/results/two_coin_weld_ab_leu_20260720

Arms are TRIVALAYA_CLOSE_KERNEL_FRAC values:
    control  -- var unset: today's fixed 7x7 (the baseline arm)
    auto     -- scale-relative from config, incl. any per-house overrides
    <float>  -- forces that frac for every house

NOTE ON TIMING: wall-clock is only meaningful on an otherwise-idle box
(§5.5). If the runner or another job is active, treat the per-arm times as
unusable and read only the structural columns. The tool prints a warning
when it detects competing load.
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
from typing import Dict, List, Optional

import cv2
import numpy as np

# Match tools/reprocess_hough.py: cv2 + OpenMP/BLAS threading has segfaulted
# under this workload, and single-threaded is also the only way per-arm
# timings mean anything.
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Layer1Config                                  # noqa: E402
from src.layer1_geometry import (                                    # noqa: E402
    _close_kernel_size,
    _segment_and_extract_candidates,
    layer_1_structural_salience,
)
from src.math_utils import detect_background_histogram               # noqa: E402
from src.pipeline_manager import _load_and_resize                    # noqa: E402

ENV_FRAC = "TRIVALAYA_CLOSE_KERNEL_FRAC"
DEFAULT_SAMPLE = Path(__file__).resolve().parent.parent / "specs/two_coin_weld_sample_ids.csv"
DEFAULT_RAW_ROOT = Path.home() / "trivalaya-pipeline/trivalaya_data/01_raw/auctions"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "two_coin_weld_raw_cache"


# --------------------------------------------------------------------------
# preprocessing replica -- only for the pre-close/pre-blur diagnostics §4.1
# wants. There is no hook inside the production function for these, so they
# are measured on a replica of its chain. The post-close mask, by contrast,
# comes from the real function.
# --------------------------------------------------------------------------

def _preprocess(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bg, _ = detect_background_histogram(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    v = np.median(gray_enhanced)
    sigma = Layer1Config.Standard.CANNY_SIGMA
    edges = cv2.Canny(gray_enhanced,
                      int(max(0, (1.0 - sigma) * v)),
                      int(min(255, (1.0 + sigma) * v)))
    edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return gray_enhanced, edge_zone, thresh_type, h, w


def _blobs(binary: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floor = Layer1Config.Standard.MIN_AREA_PX
    return [c for c in contours if cv2.contourArea(c) >= floor]


def _min_gap(cs: List[np.ndarray]) -> Optional[float]:
    """
    Minimum edge-to-edge distance between the two largest blobs, in px.
    This is the quantity the whole spec turns on: welding occurs iff
    gap < 2 * iterations * (k // 2).

    Every point of the smaller contour is tested -- do NOT subsample. An
    earlier version sampled ~200 points and reported 12.0px for a lot whose
    true minimum is 11.0px, which looks like the bridging model failing
    (12 should survive at k=7) when it is really the measurement missing the
    closest point. The model was right; the metric was wrong. Contours here
    run a few hundred points, so exact is cheap.
    """
    if len(cs) < 2:
        return None
    a, b = sorted(cs, key=cv2.contourArea, reverse=True)[:2]
    if len(a) > len(b):                    # probe the cheaper side
        a, b = b, a
    pa = a.reshape(-1, 2)
    if len(pa) > 4000:                     # pathological contour backstop only
        pa = pa[:: len(pa) // 4000 + 1]
    best = None
    for pt in pa:
        d = abs(cv2.pointPolygonTest(b, (float(pt[0]), float(pt[1])), True))
        best = d if best is None else min(best, d)
    return round(best, 2) if best is not None else None


def _circ(c) -> float:
    p = cv2.arcLength(c, True)
    return round(4 * np.pi * cv2.contourArea(c) / (p * p), 3) if p > 0 else 0.0


def _diagnostics(img: np.ndarray) -> Dict:
    """§4.1's arm-independent measurements, incl. the blur-attribution pair."""
    gray_enhanced, _, thresh_type, _, _ = _preprocess(img)

    _, pre_blur = cv2.threshold(gray_enhanced, 0, 255, thresh_type)
    blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)
    _, post_blur = cv2.threshold(blurred, 0, 255, thresh_type)

    cb, ca = _blobs(pre_blur), _blobs(post_blur)
    return {
        "blobs_pre_blur": len(cb),
        "blobs_pre_close": len(ca),          # post-blur, pre-close
        "gap_pre_blur": _min_gap(cb),
        "gap_pre_close": _min_gap(ca),
        "circ_pre_close": "|".join(
            str(_circ(c)) for c in sorted(ca, key=cv2.contourArea, reverse=True)[:2]
        ),
    }


# --------------------------------------------------------------------------

def _set_arm(arm: str):
    if arm == "control":
        os.environ.pop(ENV_FRAC, None)
    else:
        os.environ[ENV_FRAC] = arm


def _kernel_for(arm: str, h: int, w: int, house: Optional[str]) -> int:
    if arm == "control":
        return 7
    if arm.strip().lower() == "auto":
        return _close_kernel_size(h, w, house=house)
    return _close_kernel_size(h, w, frac=float(arm), house=house)


def _run_arm(img, arm, house) -> Dict:
    _set_arm(arm)
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    res = layer_1_structural_salience(img, source_type="auction", house=house)
    elapsed = time.perf_counter() - t0

    objs = res.get("objects", []) or []
    tcr = res.get("two_coin_resolution") or {}

    # post-close mask straight from the real segmentation function
    ge, ez, tt, hh, ww = _preprocess(img)
    _, binary = _segment_and_extract_candidates(img, ge, ez, tt, hh, ww, hh * ww,
                                                house=house)
    post = _blobs(binary)

    return {
        "kernel": _kernel_for(arm, h, w, house),
        "status": res.get("status"),
        "ndets": len(objs),
        "bboxes": ";".join(",".join(str(v) for v in o["bbox"]) for o in objs if "bbox" in o),
        "hough": 1 if tcr.get("method") == "hough" else 0,
        "tcr_triggered": 1 if tcr.get("triggered") else 0,
        "tcr_status": tcr.get("status") or "",
        "blobs_post_close": len(post),
        "gap_post_close": _min_gap(post),
        "seconds": round(elapsed, 4),
    }


_S3 = None


def _s3_client():
    """Lazy Spaces client, reusing the freezer's so there is one auth path."""
    global _S3
    if _S3 is None:
        from tools.freeze_weld_sample import _s3
        _S3 = _s3()          # also loads ~/spaces.env into os.environ
    return _S3


def _fetch_spaces(row, cache_dir: Path) -> Optional[Path]:
    """
    Pull a raw from Spaces into a local cache; return its path, or None.

    Keys ZERO-PAD the lot number to 5 digits (Lot_00001.jpg), matching
    freeze_weld_sample::dims_spaces. The local branch in _load does not pad --
    that asymmetry is real, not an oversight, so the two are kept separate.

    Cached by (house, sale, lot), so a re-run of the A/B re-downloads nothing.
    Writes via a .part temp then renames, so an interrupted download can never
    be picked up as a valid cache hit on the next run.
    """
    lot = int(row["lot_number"])
    key = f"raw/auctions/{row['house']}/{row['sale_id']}/Lot_{lot:05d}.jpg"
    dest = cache_dir / f"{row['house']}_{row['sale_id']}_{lot:05d}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        body = _s3_client().get_object(
            Bucket=os.environ["SPACES_BUCKET"], Key=key)["Body"].read()
    except Exception:
        return None
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return dest


def _load(row, raw_root: Path, source: str = "local",
          cache_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    if source == "spaces":
        p = _fetch_spaces(row, cache_dir)
        if p is None:
            return None
    else:
        # Local raws are named inconsistently across houses: cng_feature uses
        # the bare lot number (Lot_215268.jpg) while kuenker zero-pads to 5
        # (Lot_01001.jpg for lot 1001).  Try the bare form first, then the
        # padded one -- %05d is a no-op on numbers already longer than 5
        # digits, so one fallback covers both conventions.  Previously this
        # branch was bare-only, which silently loaded 0/200 kuenker lots.
        base = raw_root / row["house"] / row["sale_id"]
        lot = row["lot_number"]
        for name in (f"Lot_{lot}.jpg", f"Lot_{int(lot):05d}.jpg"):
            p = base / name
            if p.exists():
                break
        else:
            return None
    img, _ = _load_and_resize(str(p))   # same resize production applies
    return img


def _busy() -> List[str]:
    """Cheap check for load that would invalidate the timing columns."""
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        return []
    warn = []
    if load1 > 1.5:
        warn.append(f"1-min load average is {load1:.1f}")
    return warn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--source", choices=["local", "spaces"], default="local",
                    help="where raws come from; matches freeze_weld_sample "
                         "(leu raws exist only in Spaces)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help="local cache for --source spaces; a re-run reuses it")
    ap.add_argument("--arms", default="control,auto",
                    help="comma-separated: control | auto | <float>")
    ap.add_argument("--purpose", default="weld_ab", help="filter CSV by purpose column")
    ap.add_argument("--house", default=None, help="filter CSV by house")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("two_coin_weld_ab"),
                    help="output prefix; writes <out>.csv and <out>.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    base = arms[0]

    rows = list(csv.DictReader(args.sample.open()))
    if args.purpose:
        rows = [r for r in rows if r.get("purpose") == args.purpose]
    if args.house:
        rows = [r for r in rows if r["house"] == args.house]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no rows selected from the frozen sample")

    if args.source == "spaces":
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    for w in _busy():
        print(f"  !! {w} -- per-arm timings are contaminated, read structural "
              f"columns only (§5.5 wants an idle box)")

    print(f"lots={len(rows)} arms={arms} sample={args.sample.name}")

    out_rows, missing = [], 0
    for i, r in enumerate(rows, 1):
        img = _load(r, args.raw_root, args.source, args.cache_dir)
        if img is None:
            missing += 1
            continue
        house = r["house"]
        rec = {
            "house": house, "sale_id": r["sale_id"], "lot_id": r["lot_id"],
            "lot_number": r["lot_number"],
            "w": img.shape[1], "h": img.shape[0],
        }
        rec.update(_diagnostics(img))
        for arm in arms:
            for k, v in _run_arm(img, arm, house).items():
                rec[f"{arm}.{k}"] = v
        # paired deltas against the baseline arm
        for arm in arms[1:]:
            rec[f"{arm}.bbox_identical"] = int(
                rec[f"{arm}.bboxes"] == rec[f"{base}.bboxes"])
            rec[f"{arm}.ndets_delta"] = rec[f"{arm}.ndets"] - rec[f"{base}.ndets"]
        # §4.1 weld signature, per arm
        for arm in arms:
            rec[f"{arm}.weld"] = int(rec["blobs_pre_close"] == 2
                                     and rec[f"{arm}.blobs_post_close"] == 1)
        out_rows.append(rec)
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    _set_arm("control")   # never leave the env dirty

    if not out_rows:
        sys.exit(f"no lots could be loaded (missing={missing}); "
                 f"check --raw-root / --source")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wri.writeheader()
        wri.writerows(out_rows)

    n = len(out_rows)
    summary = {"lots": n, "missing_raws": missing, "arms": arms,
               "sample": str(args.sample), "by_arm": {}}
    for arm in arms:
        secs = [r[f"{arm}.seconds"] for r in out_rows]
        dets = [r[f"{arm}.ndets"] for r in out_rows]
        summary["by_arm"][arm] = {
            "hough_rate": round(sum(r[f"{arm}.hough"] for r in out_rows) / n, 4),
            "weld_rate": round(sum(r[f"{arm}.weld"] for r in out_rows) / n, 4),
            "fragment_rate": round(
                sum(1 for r in out_rows if r[f"{arm}.blobs_post_close"] > 2) / n, 4),
            "mean_ndets": round(sum(dets) / n, 3),
            "ndets_hist": {str(k): dets.count(k) for k in sorted(set(dets))},
            "median_seconds": round(sorted(secs)[n // 2], 4),
            "total_seconds": round(sum(secs), 2),
            "kernels": sorted({r[f"{arm}.kernel"] for r in out_rows}),
        }
        if arm != base:
            summary["by_arm"][arm]["bbox_identical_vs_" + base] = round(
                sum(r[f"{arm}.bbox_identical"] for r in out_rows) / n, 4)

    gaps = [r["gap_pre_close"] for r in out_rows if r["gap_pre_close"] is not None]
    pre_blur = [r["gap_pre_blur"] for r in out_rows if r["gap_pre_blur"] is not None]
    if gaps:
        gaps.sort()
        summary["gap_pre_close"] = {
            "n": len(gaps), "median": gaps[len(gaps) // 2],
            "p10": gaps[len(gaps) // 10], "p90": gaps[(9 * len(gaps)) // 10],
        }
    if pre_blur and gaps:
        # §4.1's blur-attribution question: how much bridging does the blur own?
        pb = sorted(pre_blur)
        summary["gap_pre_blur"] = {"n": len(pb), "median": pb[len(pb) // 2]}

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {csv_path}  ({n} lots, {missing} raws missing)")
    print(f"wrote {json_path}\n")
    print(json.dumps(summary["by_arm"], indent=2))
    if "gap_pre_close" in summary:
        print("\ngap_pre_close:", json.dumps(summary["gap_pre_close"]))
    if "gap_pre_blur" in summary:
        print("gap_pre_blur :", json.dumps(summary["gap_pre_blur"]))


if __name__ == "__main__":
    main()
