#!/usr/bin/env python3
"""
Throwaway measurement script for the hypothesis:

    "L229 mask no-op" (serving lane, appv2._mask_query_image_meta silently
    returning masked:true over a full-frame contour) IS the same bug as the
    `backdrop_vignette_blob` ingest failure class -- both downstream of
    detect_background_histogram (~/trivalaya-vision/src/math_utils.py:172)
    returning a wrong background level on CNG's composited vignetted backdrop.

READ-ONLY. Does not modify any production file, DB row, or service. Calls
the real, unmodified production functions:
  - trivalaya-vision src.math_utils.detect_background_histogram
  - trivalaya-vision src.pipeline_manager._load_and_resize
  - trivalaya-pipeline visual_search.appv2._mask_query_image_meta

Per-side preprocessing (splitting a combined obv+rev photo at the midline,
then feeding each half through appv2._mask_query_image_meta) exactly mirrors
corpus_match_report.py::load_sides / embed_query, which is what
screen_incoming_sale.py (the KS-17 screening tool) itself calls.

Two phases, per the revised plan (KS-17's corner-trust path fires on 0/574
sides -- a plain sweep has no path-variance to build a contingency table
from):

  PHASE 1 (cheap, ALL 574 sides): detect_background_histogram only (corner
  patches + one histogram calc, no Hough / no recovery). Gives the full
  avg_bg / bg_type / corner_std / bg_path distribution and the outer-ring-
  median counterfactual for every side.

  PHASE 2 (expensive, STRATIFIED SAMPLE): appv2._mask_query_image_meta on
  (a) every side with avg_bg > 85 (the light-backdrop subset -- the closest
  thing to a corner-trust arm in this corpus) and (b) a seeded random 60
  sides from the avg_bg < 45 subset. Each call runs in its own forked
  subprocess with a hard 180s wall-clock timeout (killed on timeout, logged
  as status=timeout, never blocks the sweep). Per-side wall-clock is
  recorded as an incidental latency by-product (offline-measured, NOT a
  production observation).

No image similarity / embeddings are computed anywhere in this script.
"""
import csv
import glob
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

REPO = "/home/claudeuser/trivalaya-pipeline"
VISION = "/home/claudeuser/trivalaya-vision"
sys.path.insert(0, REPO)
sys.path.insert(0, VISION)

# Imported ONCE, in the parent, before any pool forks -- children inherit the
# already-imported module via COW fork, no repeated import cost.
from visual_search import appv2  # noqa: E402
from src.pipeline_manager import _load_and_resize  # noqa: E402
from src.math_utils import detect_background_histogram  # noqa: E402

IMG_DIR = os.path.join(REPO, "analysis/incoming_screen/KS-17/incoming_images")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
BG_CSV = os.path.join(SCRATCH, "l229_bg_diagnostic_all574.csv")
MASK_CSV = os.path.join(SCRATCH, "l229_mask_measure_stratified.csv")
STRATA_CSV = os.path.join(SCRATCH, "l229_strata_selection.csv")

CORNER_STD_TRUST = 15.0  # mirrors detect_background_histogram's own gate, read-only replica
GREY = 128
GREY_TOL = 2
MASK_TIMEOUT_S = 180
RNG_SEED = 42
DARK_SAMPLE_N = 60
LIGHT_THRESH = 85.0   # avg_bg > 85 -- light-backdrop subset
DARK_THRESH = 45.0    # avg_bg < 45 -- dark-fallback subset
N_WORKERS = 3         # leave 1 vCPU free on this 4-vCPU box for the serving stack

CTX = mp.get_context("fork")


# ── shared helpers ──────────────────────────────────────────────────────
def corner_std_of(gray):
    h, w = gray.shape
    margin = 5
    if h <= 20 or w <= 20:
        return None
    corners = np.concatenate([
        gray[0:margin, 0:margin].flatten(),
        gray[0:margin, w - margin:w].flatten(),
        gray[h - margin:h, 0:margin].flatten(),
        gray[h - margin:h, w - margin:w].flatten(),
    ]).astype(np.float64)
    return float(np.std(corners))


def outer_ring_pixels(gray, frac=0.03):
    h, w = gray.shape
    t = max(4, int(frac * min(h, w)))
    return np.concatenate([
        gray[:t].ravel(), gray[-t:].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel()])


def bg_diagnostic(gray):
    """Call the REAL detect_background_histogram (authoritative value +
    label), plus classify which internal branch produced it, plus compute
    the counterfactual dark-fallback value using an outer-ring median."""
    cstd = corner_std_of(gray)
    avg_bg, bg_type = detect_background_histogram(gray)  # real production fn, unmodified

    if cstd is not None and cstd < CORNER_STD_TRUST:
        path = "corner_trust"
    else:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        dark_peak = float(hist[0:50].sum())
        light_peak = float(hist[205:256].sum())
        if light_peak > dark_peak * 2:
            path = "light_fallback"
        elif dark_peak > light_peak * 2:
            path = "dark_fallback"
        else:
            path = "mixed_fallback"

    counterfactual_bg = None
    counterfactual_bg_type = None
    if path == "dark_fallback":
        ring = outer_ring_pixels(gray)
        counterfactual_bg = float(np.median(ring))
        counterfactual_bg_type = "light" if counterfactual_bg > 127 else "dark"

    return avg_bg, bg_type, cstd, path, counterfactual_bg, counterfactual_bg_type


def load_sides(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    mid = w // 2
    return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))


def side_gray_via_pipeline_roundtrip(side_img):
    """Reproduce the EXACT preprocessing analyze_image sees: temp-JPEG round
    trip (same as _mask_query_image_meta's own internal one) + _load_and_resize
    + BGR2GRAY. Cheap (no Hough) -- used for phase 1."""
    rgb_arr = np.array(side_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tf.name
    tf.close()
    try:
        cv2.imwrite(tmp_path, bgr)
        img2, scale2 = _load_and_resize(tmp_path)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return gray
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def mask_area_fraction(side_img, masked_img, masked_flag):
    """Fraction of the ORIGINAL frame covered by the resulting alpha mask.

    _mask_query_image_meta composites onto exact mid-grey (128,128,128)
    wherever alpha==0 and leaves the original pixel wherever alpha==255 (hard
    binary mask, no antialiasing in cv2.drawContours(thickness=-1)). Count
    non-grey128 pixels in the returned (possibly bbox-cropped) image, divide
    by the ORIGINAL full-frame pixel count (alpha is 0 everywhere outside the
    crop bbox by construction, so this is exact).
    """
    if not masked_flag:
        return None
    w, h = side_img.size
    frame_px = w * h
    arr = np.asarray(masked_img)
    grey_px = np.all(np.abs(arr.astype(np.int16) - GREY) <= GREY_TOL, axis=-1)
    fg_px = int((~grey_px).sum())
    return fg_px / frame_px


# ── PHASE 1: cheap bg diagnostic, ALL 574 sides ─────────────────────────
BG_FIELDNAMES = [
    "id", "side", "avg_bg", "bg_type", "corner_std", "bg_path",
    "counterfactual_ring_median", "counterfactual_bg_type", "error",
]


def phase1_worker(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    rows = []
    try:
        obv, rev = load_sides(path)
        for side_name, side_img in (("obv", obv), ("rev", rev)):
            try:
                gray = side_gray_via_pipeline_roundtrip(side_img)
                avg_bg, bg_type, cstd, path_, cf_bg, cf_type = bg_diagnostic(gray)
                rows.append({
                    "id": stem, "side": side_name, "avg_bg": avg_bg, "bg_type": bg_type,
                    "corner_std": cstd, "bg_path": path_,
                    "counterfactual_ring_median": cf_bg,
                    "counterfactual_bg_type": cf_type, "error": "",
                })
            except Exception as exc:
                rows.append({"id": stem, "side": side_name, "error": repr(exc)})
    except Exception as exc:
        rows.append({"id": stem, "side": "ERROR", "error": repr(exc)})
    return rows


def run_phase1():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"[phase1] {len(paths)} images -> {len(paths) * 2} sides (cheap, no Hough)", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = 0
    t0 = time.time()
    with open(BG_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BG_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(phase1_worker, p): p for p in paths}
            for fut in as_completed(futs):
                for r in fut.result():
                    w.writerow(r)
                fh.flush()
                done += 1
                if done % 50 == 0 or done == len(paths):
                    print(f"[phase1] {done}/{len(paths)} images, {time.time() - t0:.0f}s elapsed", flush=True)
    print(f"[phase1] DONE {time.time() - t0:.0f}s total", flush=True)


# ── PHASE 2: expensive mask-path measurement, stratified sample ────────
def _mask_call_target(path, side_name, out_q):
    """Runs in a freshly forked child (fork-COW from an already-imported
    parent, so no import cost). Computes the full mask + telemetry + area
    fraction and puts a small dict back -- never the image itself."""
    try:
        obv, rev = load_sides(path)
        side_img = obv if side_name == "obv" else rev
        masked_img, meta = appv2._mask_query_image_meta(side_img)
        area_frac = mask_area_fraction(side_img, masked_img, meta["masked"])
        out_q.put({
            "status": "ok",
            "masked": meta["masked"],
            "mask_fallback_reason": meta["mask_fallback_reason"],
            "n_detections": meta["n_detections"],
            "mask_area_fraction": area_frac,
        })
    except Exception as exc:
        out_q.put({"status": "error", "error": repr(exc)})


def run_mask_with_hard_timeout(path, side_name, timeout=MASK_TIMEOUT_S):
    """Hard-kill timeout wrapper: forks a NEW process per call (cheap fork,
    no re-import) so a stalled call can be terminated without losing the
    warm parent/worker state. Returns (result_dict, elapsed_s)."""
    q = CTX.Queue()
    p = CTX.Process(target=_mask_call_target, args=(path, side_name, q))
    t0 = time.time()
    p.start()
    p.join(timeout)
    elapsed = time.time() - t0
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)
        return ({"status": "timeout", "masked": None, "mask_fallback_reason": None,
                  "n_detections": None, "mask_area_fraction": None}, elapsed)
    try:
        res = q.get_nowait()
    except Exception:
        res = {"status": "error", "error": "child exited with no result"}
        res.setdefault("masked", None)
        res.setdefault("mask_fallback_reason", None)
        res.setdefault("n_detections", None)
        res.setdefault("mask_area_fraction", None)
    return res, elapsed


MASK_FIELDNAMES = [
    "id", "side", "strata", "status", "masked", "mask_fallback_reason",
    "n_detections", "mask_area_fraction", "wall_s", "error",
]


def phase2_worker(item):
    """Runs inside a warm ProcessPoolExecutor worker (persistent, already has
    appv2 imported via inherited fork from the parent at module load time).
    Each side's actual mask call is done in its OWN nested forked child with
    a hard timeout, so a stall never blocks this worker permanently."""
    path, side_name, stem, strata = item
    res, elapsed = run_mask_with_hard_timeout(path, side_name)
    row = {
        "id": stem, "side": side_name, "strata": strata,
        "status": res.get("status"),
        "masked": res.get("masked"),
        "mask_fallback_reason": res.get("mask_fallback_reason"),
        "n_detections": res.get("n_detections"),
        "mask_area_fraction": res.get("mask_area_fraction"),
        "wall_s": round(elapsed, 3),
        "error": res.get("error", ""),
    }
    return row


def select_strata():
    """Read phase-1 output, build the stratified sample: ALL avg_bg>85 sides
    + a seeded random 60 of the avg_bg<45 sides."""
    light, dark = [], []
    with open(BG_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            try:
                avg_bg = float(row["avg_bg"])
            except (TypeError, ValueError):
                continue
            key = (row["id"], row["side"])
            if avg_bg > LIGHT_THRESH:
                light.append(key)
            elif avg_bg < DARK_THRESH:
                dark.append(key)
    rng = random.Random(RNG_SEED)
    dark_sample = rng.sample(dark, min(DARK_SAMPLE_N, len(dark)))
    print(f"[strata] light(avg_bg>{LIGHT_THRESH}) n={len(light)}; "
          f"dark(avg_bg<{DARK_THRESH}) pool n={len(dark)}, sampled {len(dark_sample)} "
          f"(seed={RNG_SEED})", flush=True)
    with open(STRATA_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "side", "strata"])
        for (i, s) in light:
            w.writerow([i, s, "light"])
        for (i, s) in dark_sample:
            w.writerow([i, s, "dark_sample"])
    return light, dark_sample


def run_phase2():
    light, dark_sample = select_strata()
    id_to_path = {}
    for p in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
        id_to_path[os.path.splitext(os.path.basename(p))[0]] = p

    items = []
    for (i, s) in light:
        items.append((id_to_path[i], s, i, "light"))
    for (i, s) in dark_sample:
        items.append((id_to_path[i], s, i, "dark_sample"))

    print(f"[phase2] {len(items)} sides to run through _mask_query_image_meta "
          f"(hard timeout {MASK_TIMEOUT_S}s/side)", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = 0
    t0 = time.time()
    with open(MASK_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MASK_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(phase2_worker, it): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"id": it[2], "side": it[1], "strata": it[3],
                           "status": "error", "error": repr(exc)}
                w.writerow(row)
                fh.flush()
                done += 1
                if done % 5 == 0 or done == len(items):
                    print(f"[phase2] {done}/{len(items)} sides, {time.time() - t0:.0f}s elapsed "
                          f"(last: {row.get('id')}/{row.get('side')} status={row.get('status')} "
                          f"wall_s={row.get('wall_s')})", flush=True)
    print(f"[phase2] DONE {time.time() - t0:.0f}s total", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "phase1"):
        run_phase1()
    if mode in ("all", "phase2"):
        run_phase2()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
