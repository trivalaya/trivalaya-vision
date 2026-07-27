#!/usr/bin/env python
"""Bar 0 probe — does the background-estimator defect cause the serving mask no-op?

Ticket: specs/background_estimator_repair.md, "Bar 0 — the link (gating)".

Measures, per KS-17 side, the SERVING mask path (`appv2._mask_query_image_meta`)
alongside the estimator's own numbers, so the contingency table

    {corner-trust vs fallback path} x {mask_area_fraction > 0.9 with masked:true}

can be built with the full mask-area distribution behind it (the bar asks for the
distribution, not a thresholded count).

`mask_area_fraction` is contour_area / frame_area of the image handed to the mask
function.  A genuine full-frame circular coin caps at pi/4 ~ 0.785; EXACTLY 1.0
means the "largest contour" is the image rectangle, i.e. the background was
classified as foreground and the embed is effectively unmasked while telemetry
still reports `masked: true`.  That is the silent no-op class (L229).

Two input modes, measured separately and never derived from one another (the
ticket's both-lanes rule, Bar 5 — the query lane has its own JPEG round-trip and
it is measured non-inert):

  query518 : the real serving lane — side downsized to UPLOAD_MAX_DIM=518 and
             re-encoded JPEG q90 exactly as `_normalize_upload_bytes` does,
             then decoded and masked.
  fullres  : the side at ingest geometry (MAX_DIMENSION=3200 load, then split),
             which is what the taxonomy scan measured its estimator numbers on.

Architecture: a persistent worker (7s import cost paid once) fed one task per
line on stdin; the PARENT enforces the per-side hard timeout and SIGKILLs +
respawns the worker on a stall, because the stalls being measured sit inside
OpenCV C code where a Python-level alarm would not land.  A timeout is a data
row, not a run-ending failure.  Every row is flushed as it lands, so an
interrupted sweep is still readable.

Read-only with respect to production: no DB writes, no file mutation outside the
output path, no change to any shipped module (analyze_image is wrapped by a
capturing proxy for the duration of a call, then restored).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import select
import subprocess
import sys
import time
from pathlib import Path

VISION_ROOT = "/home/claudeuser/trivalaya-vision"
PIPELINE_ROOT = "/home/claudeuser/trivalaya-pipeline"


# ─────────────────────────────── worker ───────────────────────────────────

def _worker_main() -> None:
    """Read task JSON per line on stdin, emit result JSON per line on stdout."""
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image

    sys.path.insert(0, PIPELINE_ROOT)
    sys.path.insert(0, VISION_ROOT)

    import visual_search.appv2 as appv2
    import src.pipeline_manager as pm
    from src.math_utils import detect_background_histogram

    sys.path.insert(0, os.path.join(VISION_ROOT, "tools"))
    from rim_stall_taxonomy import (  # noqa: E402
        _load_and_resize, split_sides, backdrop_ring_mean, corner_ramp,
    )

    UPLOAD_MAX_DIM = appv2.UPLOAD_MAX_DIM

    def to_query518_pil(bgr):
        """Mirror _normalize_upload_bytes + _bytes_to_pil for an already-decoded side.

        The serving lane receives JPEG bytes; we re-encode the side to JPEG first
        so the round-trip (which the ticket flags as non-inert) is present.
        """
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("jpeg encode failed")
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

    def measure(task):
        img = _load_and_resize(task["path"])
        if img is None:
            return {"status": "load_failed"}
        side_bgr = split_sides(img)[task["side"]]

        if task["mode"] == "query518":
            pil = to_query518_pil(side_bgr)
            meas_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            meas_bgr = side_bgr
            pil = Image.fromarray(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB))

        # Estimator + truth at the SAME resolution the mask call will see, so
        # "estimator wrong" and "mask no-op" are statements about one image.
        gray = cv2.cvtColor(meas_bgr, cv2.COLOR_BGR2GRAY)
        avg_bg, bg_type = detect_background_histogram(gray)
        ring = backdrop_ring_mean(gray)
        ramp, cstd = corner_ramp(gray)
        H, W = meas_bgr.shape[:2]

        # Capture analyze_image's result without altering shipped behavior.
        captured = {}
        real = pm.analyze_image

        def proxy(*a, **kw):
            res = real(*a, **kw)
            captured["res"] = res
            return res

        pm.analyze_image = proxy
        t0 = time.perf_counter()
        try:
            _out_img, meta = appv2._mask_query_image_meta(pil)
        finally:
            pm.analyze_image = real
        elapsed = time.perf_counter() - t0

        area_frac = None
        n_raw = None
        res = captured.get("res")
        if meta.get("masked") and res:
            dets = res.get("detections") or []
            n_raw = len(dets)
            scale = res.get("scale", 1.0) or 1.0
            try:
                det = max(dets, key=lambda d: cv2.contourArea(
                    np.asarray(d.get("layer_1", {}).get("contour", []), dtype=np.int32)))
                c = np.asarray(det["layer_1"]["contour"], dtype=np.int32)
                if scale != 1.0:
                    c = (c / scale).astype(np.int32)
                area_frac = float(cv2.contourArea(c)) / float(W * H)
            except Exception as exc:  # pragma: no cover - defensive
                area_frac = None
                n_raw = f"contour_error:{exc}"

        return {
            "status": "ok",
            "w": W, "h": H,
            "avg_bg": round(float(avg_bg), 3),
            "bg_type": bg_type,
            "ring_truth": round(ring, 3),
            "est_err": round(float(avg_bg) - ring, 3),
            "corner_std": round(float(cstd), 3),
            "corner_ramp": round(float(ramp), 3),
            "corner_path_trusted": bool(cstd < 15),
            "masked": bool(meta.get("masked")),
            "mask_fallback_reason": meta.get("mask_fallback_reason"),
            "n_detections": meta.get("n_detections"),
            "n_dets_raw": n_raw,
            "mask_area_fraction": None if area_frac is None else round(area_frac, 6),
            "elapsed_s": round(elapsed, 3),
        }

    sys.stderr.write("WORKER_READY\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        try:
            row = measure(task)
        except Exception as exc:
            row = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        row.update({k: task[k] for k in ("id", "side", "mode", "stratum")})
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()


# ─────────────────────────────── parent ───────────────────────────────────

class Worker:
    """Persistent worker with parent-side hard timeout (SIGKILL + respawn)."""

    def __init__(self, python: str, script: str):
        self.python, self.script = python, script
        self.proc = None
        self.spawns = 0
        self._spawn()

    def _spawn(self):
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        self.proc = subprocess.Popen(
            [self.python, self.script, "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self.spawns += 1
        # Wait for readiness so import time is not charged to the first side.
        t0 = time.time()
        while time.time() - t0 < 180:
            if select.select([self.proc.stderr], [], [], 1.0)[0]:
                if "WORKER_READY" in (self.proc.stderr.readline() or ""):
                    return
            if self.proc.poll() is not None:
                raise RuntimeError("worker died during startup")
        raise RuntimeError("worker startup timed out")

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                pass

    def run(self, task, timeout_s):
        """Return a result dict; on stall kill + respawn and return status=timeout."""
        try:
            self.proc.stdin.write(json.dumps(task) + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.kill(); self._spawn()
            return {"status": "worker_write_failed"}

        deadline = time.time() + timeout_s
        buf = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._spawn()
                return {"status": "worker_died"}
            if select.select([self.proc.stdout], [], [], 0.5)[0]:
                chunk = self.proc.stdout.readline()
                if chunk:
                    buf += chunk
                    if buf.endswith("\n"):
                        try:
                            return json.loads(buf)
                        except json.JSONDecodeError:
                            return {"status": "bad_json", "raw": buf[:400]}
        self.kill()
        self._spawn()
        return {"status": "timeout", "elapsed_s": round(timeout_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scan", default=f"{VISION_ROOT}/specs/results/rim_stall_taxonomy_ks17_scan.json")
    ap.add_argument("--images", default=f"{PIPELINE_ROOT}/analysis/incoming_screen/KS-17/incoming_images")
    ap.add_argument("--out", required=False)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-dark", type=int, default=60)
    ap.add_argument("--side-list", default=None,
                    help="CSV id,side,strata — the authoritative sample. Overrides "
                         "seeded resampling (the 2026-07-26 run's dark pool order came "
                         "from ProcessPoolExecutor completion order, so its seed 42 is "
                         "NOT reproducible; the selection CSV is).")
    ap.add_argument("--strata", default="dark,light", help="comma list, run in this order")
    ap.add_argument("--modes", default="query518,fullres")
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()

    if a.worker:
        _worker_main()
        return

    if not a.out:
        ap.error("--out required")

    scan = json.load(open(a.scan))
    light = sorted([r for r in scan if r["avg_bg"] > 85], key=lambda r: (r["id"], r["side"]))
    dark = sorted([r for r in scan if r["avg_bg"] < 45], key=lambda r: (r["id"], r["side"]))
    by_key = {(r["id"], r["side"]): r for r in scan}

    if a.side_list:
        import csv as _csv
        want = {"dark": [], "light": []}
        for row in _csv.DictReader(open(a.side_list)):
            st = "dark" if row["strata"].startswith("dark") else "light"
            rec = by_key.get((row["id"], row["side"]))
            if rec is None:
                raise SystemExit(f"side {row['id']}/{row['side']} not in scan")
            want[st].append(rec)
        dark_sample = sorted(want["dark"], key=lambda r: (r["id"], r["side"]))
        light = sorted(want["light"], key=lambda r: (r["id"], r["side"]))
        sample_source = f"side-list:{a.side_list}"
    else:
        rng = random.Random(a.seed)
        dark_sample = sorted(rng.sample(dark, min(a.n_dark, len(dark))),
                             key=lambda r: (r["id"], r["side"]))
        sample_source = f"seed:{a.seed}"

    pools = {"dark": dark_sample, "light": light}
    tasks = []
    for stratum in a.strata.split(","):
        for r in pools[stratum.strip()]:
            for mode in a.modes.split(","):
                tasks.append({
                    "id": r["id"], "side": r["side"], "stratum": stratum.strip(),
                    "mode": mode.strip(),
                    "path": str(Path(a.images) / f"{r['id']}.jpg"),
                    "scan_avg_bg": r["avg_bg"],
                })

    meta = {
        "_meta": True, "sample_source": sample_source,
        "seed": a.seed, "n_dark_pool": len(dark), "n_dark_sample": len(dark_sample),
        "n_light": len(light), "timeout_s": a.timeout, "modes": a.modes, "strata": a.strata,
        "scan": a.scan, "images": a.images, "n_tasks": len(tasks),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(meta), flush=True)

    w = Worker(sys.executable, os.path.abspath(__file__))
    t_start = time.time()
    with open(a.out, "w") as fh:
        fh.write(json.dumps(meta) + "\n"); fh.flush()
        for i, task in enumerate(tasks, 1):
            row = w.run(task, a.timeout)
            for k in ("id", "side", "mode", "stratum"):
                row.setdefault(k, task[k])
            row["scan_avg_bg"] = task["scan_avg_bg"]
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[{i}/{len(tasks)}] {task['stratum']:5s} {task['id']}/{task['side']:3s} "
                  f"{task['mode']:8s} {row.get('status')} "
                  f"masked={row.get('masked')} area={row.get('mask_area_fraction')} "
                  f"est={row.get('avg_bg')} truth={row.get('ring_truth')} "
                  f"{row.get('elapsed_s')}s  [{time.time()-t_start:.0f}s elapsed]",
                  flush=True)
    w.kill()
    print(f"done -> {a.out}  ({time.time()-t_start:.0f}s, worker spawns={w.spawns})", flush=True)


if __name__ == "__main__":
    main()
