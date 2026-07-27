#!/usr/bin/env python
"""M1 A/B probe -- background-estimator repair, Bars 1/3/4/5.

Ticket: specs/background_estimator_repair.md, mechanism M1.

Descends from `tools/bg_estimator_bar0_probe.py` (the Bar 0 instrument, which
reproduced its predecessor to max |area delta| 0.0037 and matched its no-op
count 12 vs 12).  Same worker architecture and the same self-computed
`mask_area_fraction`, because production has no first-class no-op telemetry --
that gap is real and still open (Bar 0 write-up S7).  Per owner ruling
2026-07-28: the bar gates on the no-op COUNT, not on which instrument counts it.

What is added over the Bar 0 probe:

  --gate on|off   sets TRIVALAYA_BG_CORNER_LOCAL_TRUST in the worker env, so
                  both arms run from ONE checkout.  Gate OFF is bit-identical
                  to pre-M1 (asserted in tests/test_bg_corner_local_trust.py),
                  which is what makes a one-checkout A/B legitimate.
  --all           the full 574-side KS-17 population, not the Bar 0 sample.
  --workers N     shard the task list; each worker keeps the parent-side hard
                  timeout + SIGKILL-and-respawn behavior.

IMPORT-PATH HAZARD, defused here.  `appv2._mask_query_image_meta` contains a
hardcoded `sys.path.insert(0, "/home/claudeuser/trivalaya-vision")` -- i.e. it
would jump the queue and import MAIN's `src.*` even when this script runs from
a worktree, silently splitting the measurement (estimator numbers from the
worktree, mask numbers from main).  The worker therefore imports
`src.pipeline_manager` FIRST with the worktree at sys.path[0], so the module is
already in sys.modules by the time appv2 asks for it.  The resolved
`src.math_utils.__file__` is then ASSERTED to live under --vision-root and is
echoed into the run meta; a mismatch aborts rather than producing numbers.

Read-only with respect to production: no DB writes, no mutation outside --out,
no shipped module changed (analyze_image is wrapped by a capturing proxy for
the duration of a call, then restored).
"""
from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

PIPELINE_ROOT = "/home/claudeuser/trivalaya-pipeline"
GATE_ENV = "TRIVALAYA_BG_CORNER_LOCAL_TRUST"


# ─────────────────────────────── worker ───────────────────────────────────

def _worker_main() -> None:
    """Read task JSON per line on stdin, emit result JSON per line on stdout."""
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image

    # OpenCV keeps its OWN thread pool -- OMP_NUM_THREADS does not bound it.
    # Left alone, N workers each fan out across every core, oversubscribe a
    # 4-vCPU box (measured load 9.2 at N=3) and steal CPU from the live
    # visual_search service. Pin to 1 so worker count IS core count.
    cv2.setNumThreads(1)

    vision_root = os.environ["M1_VISION_ROOT"]

    # Worktree FIRST, and pre-import src.* so appv2's hardcoded path insert
    # cannot swap in main's copy behind our back (see module docstring).
    sys.path.insert(0, vision_root)
    sys.path.insert(1, PIPELINE_ROOT)
    import src.pipeline_manager as pm          # noqa: E402
    import src.math_utils as mu                # noqa: E402
    from src.math_utils import detect_background_histogram  # noqa: E402

    resolved = os.path.realpath(mu.__file__)
    if not resolved.startswith(os.path.realpath(vision_root) + os.sep):
        sys.stderr.write(f"WORKER_FATAL src.math_utils resolved to {resolved}, "
                         f"expected under {vision_root}\n")
        sys.stderr.flush()
        raise SystemExit(2)

    import visual_search.appv2 as appv2        # noqa: E402
    # Re-assert after appv2's import-time sys.path surgery.
    if os.path.realpath(sys.modules["src.math_utils"].__file__) != resolved:
        sys.stderr.write("WORKER_FATAL src.math_utils swapped by appv2 import\n")
        sys.stderr.flush()
        raise SystemExit(2)

    sys.path.insert(0, os.path.join(vision_root, "tools"))
    from rim_stall_taxonomy import (  # noqa: E402
        _load_and_resize, split_sides, backdrop_ring_mean, corner_ramp,
    )

    UPLOAD_MAX_DIM = appv2.UPLOAD_MAX_DIM

    def to_query518_pil(bgr):
        """Mirror _normalize_upload_bytes + _bytes_to_pil for a decoded side."""
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

        gray = cv2.cvtColor(meas_bgr, cv2.COLOR_BGR2GRAY)
        avg_bg, bg_type = detect_background_histogram(gray)
        ring = backdrop_ring_mean(gray)
        ramp, cstd = corner_ramp(gray)
        H, W = meas_bgr.shape[:2]

        # Per-corner local stats -- lets the report say WHY M1 did or did not
        # fire on a side without re-reading the image.
        m = 5
        patches = (gray[0:m, 0:m], gray[0:m, W-m:W],
                   gray[H-m:H, 0:m], gray[H-m:H, W-m:W])
        corner_local_std_max = float(max(float(np.std(p)) for p in patches))
        corner_medians = [float(np.median(p)) for p in patches]

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
            "corner_local_std_max": round(corner_local_std_max, 3),
            "corner_medians": [round(x, 1) for x in corner_medians],
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

    def __init__(self, python: str, script: str, env_extra: dict):
        self.python, self.script = python, script
        self.env_extra = env_extra
        self.proc = None
        self.spawns = 0
        self._spawn()

    def _spawn(self):
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env.update(self.env_extra)
        self.proc = subprocess.Popen(
            [self.python, self.script, "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self.spawns += 1
        t0 = time.time()
        while time.time() - t0 < 300:
            if select.select([self.proc.stderr], [], [], 1.0)[0]:
                line = self.proc.stderr.readline() or ""
                if "WORKER_FATAL" in line:
                    raise RuntimeError(f"worker refused to start: {line.strip()}")
                if "WORKER_READY" in line:
                    return
            if self.proc.poll() is not None:
                err = ""
                try:
                    err = self.proc.stderr.read()[-800:]
                except Exception:
                    pass
                raise RuntimeError(f"worker died during startup: {err}")
        raise RuntimeError("worker startup timed out")

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                pass

    def run(self, task, timeout_s):
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
    ap.add_argument("--vision-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--gate", choices=["on", "off"], required=False)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--images", default=f"{PIPELINE_ROOT}/analysis/incoming_screen/KS-17/incoming_images")
    ap.add_argument("--out", required=False)
    ap.add_argument("--side-list", default=None, help="CSV id,side[,strata]")
    ap.add_argument("--all", action="store_true", help="full population from --scan")
    ap.add_argument("--modes", default="query518,fullres")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    if a.worker:
        _worker_main()
        return
    if not a.out or not a.gate:
        ap.error("--out and --gate required")

    vision_root = os.path.realpath(a.vision_root)
    scan_path = a.scan or f"{vision_root}/specs/results/rim_stall_taxonomy_ks17_scan.json"
    scan = json.load(open(scan_path))
    by_key = {(r["id"], r["side"]): r for r in scan}

    def stratum_of(r):
        if r["avg_bg"] < 45:
            return "dark"
        if r["avg_bg"] > 85:
            return "light"
        return "mid"

    if a.side_list:
        import csv as _csv
        rows = [by_key[(r["id"], r["side"])] for r in _csv.DictReader(open(a.side_list))]
        sample_source = f"side-list:{a.side_list}"
    elif a.all:
        rows = list(scan)
        sample_source = f"all:{scan_path}"
    else:
        ap.error("one of --all / --side-list required")
    rows = sorted(rows, key=lambda r: (r["id"], r["side"]))

    tasks = []
    for r in rows:
        for mode in a.modes.split(","):
            tasks.append({
                "id": r["id"], "side": r["side"], "stratum": stratum_of(r),
                "mode": mode.strip(),
                "path": str(Path(a.images) / f"{r['id']}.jpg"),
                "scan_avg_bg": r["avg_bg"],
            })

    meta = {
        "_meta": True, "gate": a.gate, "gate_env": GATE_ENV,
        "vision_root": vision_root, "sample_source": sample_source,
        "n_sides": len(rows), "n_tasks": len(tasks), "modes": a.modes,
        "timeout_s": a.timeout, "workers": a.workers, "scan": scan_path,
        "images": a.images,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(meta), flush=True)

    env_extra = {"M1_VISION_ROOT": vision_root}
    if a.gate == "on":
        env_extra[GATE_ENV] = "1"
    else:
        env_extra.pop(GATE_ENV, None)
        os.environ.pop(GATE_ENV, None)

    lock = threading.Lock()
    fh = open(a.out, "w")
    fh.write(json.dumps(meta) + "\n"); fh.flush()
    t_start = time.time()
    done = [0]
    spawn_counts = []

    def run_shard(shard_id):
        w = Worker(sys.executable, os.path.abspath(__file__), env_extra)
        try:
            for task in tasks[shard_id::a.workers]:
                row = w.run(task, a.timeout)
                for k in ("id", "side", "mode", "stratum"):
                    row.setdefault(k, task[k])
                row["scan_avg_bg"] = task["scan_avg_bg"]
                row["gate"] = a.gate
                with lock:
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    done[0] += 1
                    n = done[0]
                    if n % 20 == 0 or n == len(tasks):
                        el = time.time() - t_start
                        rate = n / max(el, 1e-6)
                        eta = (len(tasks) - n) / max(rate, 1e-9)
                        print(f"[{n}/{len(tasks)}] {el:.0f}s elapsed, "
                              f"ETA {eta/60:.1f}m", flush=True)
        finally:
            spawn_counts.append(w.spawns)
            w.kill()

    threads = [threading.Thread(target=run_shard, args=(i,), daemon=True)
               for i in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    fh.close()
    print(f"done -> {a.out}  ({time.time()-t_start:.0f}s, gate={a.gate}, "
          f"worker spawns={spawn_counts})", flush=True)


if __name__ == "__main__":
    main()
