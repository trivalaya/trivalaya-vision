"""
Reprocess coins whose v1 crops came from the buggy two-coin Hough path.

Reads coin_detections rows where vision_metadata.split_method=hough,
downloads the auction source image from Spaces, re-runs the (now-fixed)
pipeline, and overwrites the existing v1 crop + _224 jpegs in Spaces.

DB rows are not modified — only the JPEG files at the stored paths are.

Usage:
    python tools/reprocess_hough.py --limit 50 --workers 4
    python tools/reprocess_hough.py --all --workers 8
    python tools/reprocess_hough.py --coin-ids 126427,185807
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
import mysql.connector
import numpy as np

# Critical for high-worker batches: each Python worker thread should drive
# cv2 single-threaded so internal openmp doesn't oversubscribe and race.
# A 32-worker batch on a 32-vCPU box segfaulted in cv2.abi3.so without this.
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from botocore.client import Config

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.pipeline_manager import analyze_image, crop_with_alpha  # noqa: E402

log = logging.getLogger("reprocess")


# ---------------------------------------------------------------------------
# Config / clients
# ---------------------------------------------------------------------------

def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip("'\""))


def get_db():
    _load_env(Path("/root/trivalaya-pipeline/.env"))
    return mysql.connector.connect(
        host=os.environ["TRIVALAYA_DB_HOST"],
        user=os.environ["TRIVALAYA_DB_USER"],
        password=os.environ["TRIVALAYA_DB_PASSWORD"],
        database=os.environ["TRIVALAYA_DB_NAME"],
    )


def get_s3():
    return boto3.client(
        "s3",
        region_name=os.environ["SPACES_REGION"],
        endpoint_url=os.environ["SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


BUCKET = os.environ.get("SPACES_BUCKET", "trivalaya-data")


# ---------------------------------------------------------------------------
# Per-coin reprocess
# ---------------------------------------------------------------------------

@dataclass
class CoinJob:
    coin_id: int
    auction_record_id: int
    auction_house: str
    sale_id: str
    lot_number: str
    rows: List[Dict]  # coin_detections rows (each has side, normalized_path, crop_path)


def load_jobs_from_file(path: str) -> List[CoinJob]:
    """Load a pre-baked job list (built on the DB host). Skips fetch_jobs entirely
    so a burst worker needs no MySQL connection — only Spaces creds + this file.

    Expected JSON: list of {coin_id, auction_house, sale_id, lot_number, rows:[{side,
    detection_index, crop_path, normalized_path, transparent_path}, ...]}.
    """
    with open(path) as f:
        data = json.load(f)
    jobs: List[CoinJob] = []
    for d in data:
        jobs.append(CoinJob(
            coin_id=int(d["coin_id"]),
            auction_record_id=int(d.get("auction_record_id", 0)),
            auction_house=str(d["auction_house"]),
            sale_id=str(d["sale_id"]),
            lot_number=str(d["lot_number"]),
            rows=d["rows"],
        ))
    return jobs


_LOT_RE = re.compile(r"/(Lot_\d+)/[^/]+$")


def _lot_dir_from_path(path: str) -> Optional[str]:
    m = _LOT_RE.search(path)
    return m.group(1) if m else None


def fetch_jobs(coin_ids: Optional[List[int]] = None,
               limit: Optional[int] = None,
               sample: bool = False,
               non_green_only: bool = False,
               require_hough: bool = True) -> List[CoinJob]:
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    params: tuple = ()
    # When filtering by non-GREEN, restrict to coin_ids whose ANY hough-split
    # detection row is non-GREEN, then pull BOTH sides for those coins so each
    # job has the obv+rev pair the pipeline produces.
    if non_green_only:
        coin_filter = ""
        if coin_ids:
            place = ",".join(["%s"] * len(coin_ids))
            coin_filter = f" AND coin_id IN ({place})"
            params = tuple(coin_ids)
        cur.execute(f"""
            SELECT DISTINCT coin_id
            FROM coin_detections
            WHERE vision_metadata LIKE '%"split_method": "hough"%'
              AND quality_flag <> 'GREEN'
              {coin_filter}
        """, params)
        hit_ids = [r["coin_id"] for r in cur.fetchall()]
        if not hit_ids:
            cur.close(); conn.close(); return []
        coin_ids = hit_ids
        params = ()  # rebuilt below

    where_parts = []
    if require_hough:
        where_parts.append("cd.vision_metadata LIKE '%\"split_method\": \"hough\"%'")
    if coin_ids:
        place = ",".join(["%s"] * len(coin_ids))
        where_parts.append(f"cd.coin_id IN ({place})")
        params = tuple(coin_ids)
    where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    order = " ORDER BY RAND() " if sample else " ORDER BY cd.coin_id "
    sql = f"""
        SELECT cd.id, cd.coin_id, cd.auction_record_id, cd.side, cd.detection_index,
               cd.crop_path, cd.normalized_path, cd.transparent_path,
               a.auction_house, a.sale_id, a.lot_number
        FROM coin_detections cd
        JOIN auction_data a ON a.id = cd.auction_record_id
        {where}
        {order}
    """
    cur.execute(sql, params)
    by_coin: Dict[int, List[Dict]] = {}
    for r in cur.fetchall():
        by_coin.setdefault(r["coin_id"], []).append(r)
        if limit and not sample and len(by_coin) >= limit and r["coin_id"] in by_coin:
            # keep collecting rows for already-added coins, but stop adding new ones
            pass
    cur.close()
    conn.close()

    jobs: List[CoinJob] = []
    for cid, rows in by_coin.items():
        first = rows[0]
        jobs.append(CoinJob(
            coin_id=cid,
            auction_record_id=first["auction_record_id"],
            auction_house=first["auction_house"],
            sale_id=str(first["sale_id"]),
            lot_number=str(first["lot_number"]),
            rows=rows,
        ))
    if limit:
        if sample:
            random.shuffle(jobs)
        jobs = jobs[:limit]
    return jobs


def source_key_for(job: CoinJob) -> Optional[str]:
    """Derive raw/auctions/<...>/Lot_<n>.jpg from a normalized_path."""
    norm = job.rows[0].get("normalized_path") or ""
    lot_dir = _lot_dir_from_path(norm)
    if not lot_dir:
        return None
    # processed/vision/v1/crops/<house>/<sale>/Lot_xxx/<file>.jpg
    parts = norm.split("/")
    try:
        idx = parts.index("crops")
        house, sale, _lot = parts[idx + 1], parts[idx + 2], parts[idx + 3]
    except (ValueError, IndexError):
        return None
    return f"raw/auctions/{house}/{sale}/{lot_dir}.jpg"


def encode_jpg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def reprocess_one(job: CoinJob, s3, dry_run: bool = False) -> Dict:
    out: Dict = {"coin_id": job.coin_id, "status": "?"}
    src_key = source_key_for(job)
    if not src_key:
        out["status"] = "skip"; out["reason"] = "no_source_key"; return out

    # 1. Download source from Spaces
    try:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, src_key, buf)
        data = buf.getvalue()
    except Exception as e:
        out["status"] = "fail"; out["stage"] = "download"; out["error"] = str(e); return out

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        out["status"] = "fail"; out["stage"] = "decode"; return out

    H, W = img.shape[:2]
    MAX = 3200
    scale = MAX / max(H, W) if max(H, W) > MAX else 1.0

    # Write source to a temp file for analyze_image (it expects a path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data); tmp_path = f.name
    try:
        res = analyze_image(tmp_path, source_type="auction")
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    if res.get("status") != "success":
        out["status"] = "fail"; out["stage"] = "pipeline"; out["error"] = res.get("last_error","")
        return out

    dets = res["detections"]
    if len(dets) != 2:
        out["status"] = "skip"; out["reason"] = f"ndets={len(dets)}"; return out

    # Sort detections left-to-right (obv then rev)
    def cx(d):
        c = np.asarray(d["layer_1"]["contour"]).astype(np.int32)
        M = cv2.moments(c)
        return int(M["m10"] / M["m00"]) if M["m00"] else 0
    dets.sort(key=cx)

    # Build extracted crops + transparents
    crops_by_side: Dict[str, Tuple[bytes, bytes, bytes]] = {}
    for i, d in enumerate(dets):
        side = "obverse" if i == 0 else "reverse"
        c_small = np.asarray(d["layer_1"]["contour"]).astype(np.int32)
        c_orig = (c_small / scale).astype(np.int32) if scale != 1.0 else c_small
        x, y, w, h = cv2.boundingRect(c_orig)
        margin = int(max(w, h) * 0.05)
        x1 = max(0, x - margin); y1 = max(0, y - margin)
        x2 = min(W, x + w + margin); y2 = min(H, y + h + margin)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            out["status"] = "fail"; out["stage"] = f"crop_{side}"; out["error"] = "empty"; return out
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        # Transparent PNG (RGBA) — alpha from contour mask
        rgba = crop_with_alpha(img, c_orig, (x1, y1, x2, y2))
        ok, png_buf = cv2.imencode(".png", rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
        if not ok:
            out["status"] = "fail"; out["stage"] = f"png_{side}"; return out
        crops_by_side[side] = (encode_jpg(crop), encode_jpg(resized), png_buf.tobytes())

    # Upload to existing keys
    uploaded: List[str] = []
    for row in job.rows:
        side = row["side"]
        if side not in crops_by_side:
            out["status"] = "skip"; out["reason"] = f"no_side_{side}"; return out
        crop_bytes, sq_bytes, png_bytes = crops_by_side[side]
        if dry_run:
            uploaded.append(f"DRY {row['crop_path']}")
            uploaded.append(f"DRY {row['normalized_path']}")
            uploaded.append(f"DRY {row.get('transparent_path','')}")
            continue
        if row.get("crop_path"):
            s3.put_object(Bucket=BUCKET, Key=row["crop_path"], Body=crop_bytes,
                          ContentType="image/jpeg", ACL="public-read")
            uploaded.append(row["crop_path"])
        if row.get("normalized_path"):
            s3.put_object(Bucket=BUCKET, Key=row["normalized_path"], Body=sq_bytes,
                          ContentType="image/jpeg", ACL="public-read")
            uploaded.append(row["normalized_path"])
        if row.get("transparent_path"):
            s3.put_object(Bucket=BUCKET, Key=row["transparent_path"], Body=png_bytes,
                          ContentType="image/png", ACL="public-read")
            uploaded.append(row["transparent_path"])

    out["status"] = "ok"; out["uploaded"] = uploaded
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--coin-ids", type=str, default=None, help="comma-separated coin_ids")
    ap.add_argument("--coin-ids-file", type=str, default=None, help="path to file with coin_ids (comma- or newline-separated)")
    ap.add_argument("--resume", action="store_true", help="skip coin_ids already 'ok' in --log")
    ap.add_argument("--sample", action="store_true", help="random sample (uses ORDER BY RAND)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", type=str, default="/tmp/v1v2/reprocess_log.jsonl")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs-file", type=str, default=None,
                    help="Pre-baked JSON job list (skip DB; for burst workers without MySQL access)")
    ap.add_argument("--non-green", action="store_true",
                    help="Restrict to coins with at least one non-GREEN hough-split detection row")
    ap.add_argument("--no-hough-filter", action="store_true",
                    help="Don't require split_method=hough — useful for explicit coin_ids that need the rim-recovery fix even though they weren't Hough-split")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    coin_ids = None
    if args.coin_ids:
        coin_ids = [int(x) for x in args.coin_ids.split(",") if x.strip()]
    if args.coin_ids_file:
        raw = Path(args.coin_ids_file).read_text().replace("\n", ",")
        file_ids = [int(x) for x in raw.split(",") if x.strip()]
        coin_ids = (coin_ids or []) + file_ids

    limit = args.limit
    if args.all:
        limit = None

    t0 = time.time()
    if args.jobs_file:
        log.info("Loading jobs from %s (no DB)", args.jobs_file)
        jobs = load_jobs_from_file(args.jobs_file)
    else:
        log.info("Fetching job list (limit=%s sample=%s non_green=%s no_hough=%s)...",
                 limit, args.sample, args.non_green, args.no_hough_filter)
        jobs = fetch_jobs(coin_ids=coin_ids, limit=limit, sample=args.sample,
                          non_green_only=args.non_green,
                          require_hough=not args.no_hough_filter)

    # --resume filtering applies to both paths
    if args.resume and Path(args.log).exists():
        done = set()
        for line in open(args.log):
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["coin_id"])
            except json.JSONDecodeError:
                continue
        if done:
            before = len(jobs)
            jobs = [j for j in jobs if j.coin_id not in done]
            log.info("--resume: %d already-ok jobs skipped, %d remaining",
                     before - len(jobs), len(jobs))

    log.info("Got %d coin jobs in %.1fs", len(jobs), time.time() - t0)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(args.log, "a")

    counters = {"ok": 0, "skip": 0, "fail": 0}
    t_start = time.time()

    s3_per_thread: Dict[int, object] = {}

    def worker(job: CoinJob):
        import threading
        tid = threading.get_ident()
        if tid not in s3_per_thread:
            s3_per_thread[tid] = get_s3()
        return reprocess_one(job, s3_per_thread[tid], dry_run=args.dry_run)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_cid = {ex.submit(worker, j): j.coin_id for j in jobs}
        for i, fut in enumerate(as_completed(fut_to_cid), 1):
            cid = fut_to_cid[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"coin_id": cid, "status": "fail", "stage": "exception", "error": str(e)}
            log_fp.write(json.dumps(rec) + "\n"); log_fp.flush()
            counters[rec["status"]] = counters.get(rec["status"], 0) + 1
            if i % 25 == 0 or i == len(jobs):
                rate = i / max(time.time() - t_start, 0.01)
                log.info("%d/%d ok=%d skip=%d fail=%d  %.1f coins/s",
                         i, len(jobs), counters["ok"], counters["skip"], counters["fail"], rate)

    log_fp.close()
    log.info("Done. ok=%d skip=%d fail=%d total=%d", counters["ok"], counters["skip"], counters["fail"], len(jobs))


if __name__ == "__main__":
    main()
