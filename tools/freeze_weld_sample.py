"""
Freeze a lot sample into specs/two_coin_weld_sample_ids.csv.

Implements specs/two_coin_weld_morph_close.md §4.1's "freeze the sample"
requirement: the population must be written down and committed so a re-run
after a kernel tweak compares like-for-like. A resampled population between
iterations makes the §4.2 sweep uninterpretable.

Selection is DETERMINISTIC WITH NO SEED -- lots are sorted by lot_number and
taken at evenly spaced indices, so re-running reproduces the file
byte-identically and the sample spans the whole sale rather than clustering
at one end. Verify with: run twice, diff.

Dimensions are read from the actual raw JPEGs, never assumed. Raws may be
local or in Spaces; --source picks. Appends to the CSV by default so several
houses can accumulate in one file, keyed by the `purpose` column.

STRICTLY READ-ONLY against the database and Spaces.

Usage:
    # cng_feature (raws local) -- reproduces the committed rows
    python tools/freeze_weld_sample.py --house cng_feature --sale "Auction 91" \
        --purpose weld_ab --n 200

    # leu (raws in Spaces) -- see specs/two_coin_weld_handoff.md
    python tools/freeze_weld_sample.py --house leu --sale 75 \
        --purpose leu_ab --n 200 --source spaces

    --replace-purpose   drop existing rows with this purpose before appending
    --pending-only      restrict to lots not yet vision_processed
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "specs/two_coin_weld_sample_ids.csv"
LOCAL_RAW_ROOT = Path.home() / "trivalaya-pipeline/trivalaya_data/01_raw/auctions"
PIPELINE_ENV = Path.home() / "trivalaya-pipeline/.env"
SPACES_ENV = Path.home() / "spaces.env"

FIELDS = ["house", "sale_id", "lot_id", "lot_number", "raw_width", "raw_height", "purpose"]


def load_env(p: Path):
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip("'\""))


def db_rows(house: str, sale: str, pending_only: bool) -> List[Dict]:
    import mysql.connector
    load_env(PIPELINE_ENV)
    conn = mysql.connector.connect(
        host=os.environ["TRIVALAYA_DB_HOST"], user=os.environ["TRIVALAYA_DB_USER"],
        password=os.environ["TRIVALAYA_DB_PASSWORD"], database=os.environ["TRIVALAYA_DB_NAME"])
    cur = conn.cursor(dictionary=True)
    sql = ("SELECT id, lot_number, sale_id, image_path FROM auction_data "
           "WHERE auction_house=%s AND sale_id=%s AND image_path IS NOT NULL")
    if pending_only:
        sql += " AND (vision_processed=0 OR vision_processed IS NULL)"
    sql += " ORDER BY lot_number"
    cur.execute(sql, (house, sale))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def spaced(rows: List[Dict], n: int) -> List[Dict]:
    """Evenly spaced indices -- deterministic, spans the sale, needs no seed."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def _s3():
    import boto3
    load_env(SPACES_ENV)
    return boto3.client(
        "s3", endpoint_url=os.environ["SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["SPACES_REGION"])


def dims_local(row) -> Optional[tuple]:
    from PIL import Image
    p = Path(row["image_path"])
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def dims_spaces(row, house, s3, bucket, probe=65536) -> Optional[tuple]:
    """
    Range-GET just the header. A JPEG's SOF marker is near the front, so
    64KB is plenty and this avoids pulling ~126KB x N just for dimensions.
    """
    from PIL import Image
    key = f"raw/auctions/{house}/{row['sale_id']}/Lot_{int(row['lot_number']):05d}.jpg"
    try:
        body = s3.get_object(Bucket=bucket, Key=key,
                             Range=f"bytes=0-{probe - 1}")["Body"].read()
        with Image.open(io.BytesIO(body)) as im:
            return im.size
    except Exception:
        try:  # header split across the range boundary -> take the whole object
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            with Image.open(io.BytesIO(body)) as im:
                return im.size
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house", required=True)
    ap.add_argument("--sale", required=True, help="auction_data.sale_id (NOT the "
                    "display name -- leu sale_id 75 is 'Web Auction 42')")
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--source", choices=["local", "spaces"], default="local")
    ap.add_argument("--pending-only", action="store_true")
    ap.add_argument("--replace-purpose", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    avail = db_rows(args.house, args.sale, args.pending_only)
    if not avail:
        sys.exit(f"no lots for {args.house}/{args.sale} "
                 f"(check sale_id -- it is the house's internal id, not the display name)")
    picked = spaced(avail, args.n)
    print(f"{args.house}/{args.sale}: {len(avail)} candidates -> selecting {len(picked)}")

    s3 = bucket = None
    if args.source == "spaces":
        s3 = _s3()
        bucket = os.environ["SPACES_BUCKET"]

    new_rows, failed = [], 0
    for i, r in enumerate(picked, 1):
        wh = dims_local(r) if args.source == "local" else dims_spaces(r, args.house, s3, bucket)
        if wh is None:
            failed += 1
            continue
        new_rows.append({
            "house": args.house, "sale_id": r["sale_id"], "lot_id": r["id"],
            "lot_number": r["lot_number"], "raw_width": wh[0], "raw_height": wh[1],
            "purpose": args.purpose,
        })
        if i % 50 == 0:
            print(f"  dims {i}/{len(picked)}")

    if not new_rows:
        sys.exit(f"no dimensions could be read (failed={failed}); wrong --source?")

    existing = []
    if args.out.exists():
        existing = list(csv.DictReader(args.out.open()))
        if args.replace_purpose:
            existing = [r for r in existing if r.get("purpose") != args.purpose]

    with args.out.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=FIELDS)
        wri.writeheader()
        wri.writerows(existing)
        wri.writerows(new_rows)

    widths = sorted({r["raw_width"] for r in new_rows})
    heights = [r["raw_height"] for r in new_rows]
    print(f"\nappended {len(new_rows)} rows ({failed} dim reads failed) -> {args.out}")
    print(f"  widths : {widths}")
    print(f"  heights: min={min(heights)} max={max(heights)}")
    print(f"  total rows in file: {len(existing) + len(new_rows)}")


if __name__ == "__main__":
    main()
