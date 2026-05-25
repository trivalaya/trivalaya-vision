"""
Build self-contained JSON manifests for the burst droplet.

Three artifacts:
  /tmp/v1v2/manifest_19k_remainder.json  — remaining 19K coins (~3K)
  /tmp/v1v2/manifest_priority_73k.json   — Hough-bug + nonGREEN, not yet OK
  /tmp/v1v2/scan_targets_186k.json       — every GREEN-non-hough coin for
                                            the coverage scan (read-only)

The burst droplet runs:
  python tools/reprocess_hough.py --jobs-file manifest_19k_remainder.json --workers 32 --log L1.jsonl
  python tools/reprocess_hough.py --jobs-file manifest_priority_73k.json  --workers 32 --log L2.jsonl
  python scan_186k.py            (coverage scan; produces eaten list)
  python tools/reprocess_hough.py --jobs-file manifest_eaten_xxk.json     --workers 32 --log L3.jsonl
"""
import os, json, glob, sys
for line in open("/root/trivalaya-pipeline/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip("'\""))
import mysql.connector
conn = mysql.connector.connect(
    host=os.environ["TRIVALAYA_DB_HOST"], user=os.environ["TRIVALAYA_DB_USER"],
    password=os.environ["TRIVALAYA_DB_PASSWORD"], database=os.environ["TRIVALAYA_DB_NAME"])
cur = conn.cursor(dictionary=True)


def rows_for(coin_ids):
    if not coin_ids:
        return []
    fmt = ",".join(["%s"] * len(coin_ids))
    cur.execute(f"""
        SELECT cd.coin_id, cd.auction_record_id, cd.side, cd.detection_index,
               cd.crop_path, cd.normalized_path, cd.transparent_path,
               a.auction_house, a.sale_id, a.lot_number
        FROM coin_detections cd
        JOIN auction_data a ON a.id = cd.auction_record_id
        WHERE cd.coin_id IN ({fmt})
        ORDER BY cd.coin_id, cd.detection_index
    """, tuple(coin_ids))
    return cur.fetchall()


def build_manifest(coin_ids, out_path):
    rows = rows_for(coin_ids)
    by_coin = {}
    for r in rows:
        cid = r["coin_id"]
        if cid not in by_coin:
            by_coin[cid] = {
                "coin_id": cid,
                "auction_record_id": r["auction_record_id"],
                "auction_house": r["auction_house"],
                "sale_id": str(r["sale_id"]),
                "lot_number": str(r["lot_number"]),
                "rows": [],
            }
        by_coin[cid]["rows"].append({
            "side": r["side"],
            "detection_index": r["detection_index"],
            "crop_path": r["crop_path"],
            "normalized_path": r["normalized_path"],
            "transparent_path": r["transparent_path"],
        })
    out = list(by_coin.values())
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path}  ({len(out):,} coins, {sum(len(c['rows']) for c in out):,} sides)")


# ---- already-reprocessed set ----
# Only count logs from batches that wrote the transparent_path (post commit 8a87271).
# Pre-fix batches uploaded crop_path + normalized_path but skipped the transparent PNG,
# so their coin_ids still need redoing for the canonical embedding artifact.
POST_FIX_LOGS = [
    "/tmp/v1v2/transparent_redo_log.jsonl",   # explicit transparent flush (≥8a87271)
    "/tmp/v1v2/inspect30_v2_log.jsonl",       # 30-coin re-run with transparent
    "/tmp/v1v2/flavian_322_log.jsonl",
    "/tmp/v1v2/eaten116_log.jsonl",
    "/tmp/v1v2/severan_920_log.jsonl",
    "/tmp/v1v2/severan_eaten_235_log.jsonl",
]
done = set()
for p in POST_FIX_LOGS:
    if not os.path.exists(p):
        continue
    for line in open(p):
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add(r["coin_id"])
        except Exception:
            continue
print(f"done (post-transparent-fix): {len(done):,}")


# ---- (1) 19K REMAINDER ----
with open("/tmp/v1v2/transparent_redo.txt") as f:
    redo_ids = [int(x) for x in f.read().strip().split(",") if x.strip()]
remainder_19k = [c for c in redo_ids if c not in done]
print(f"\n=== 19K remainder ===")
build_manifest(remainder_19k, "/tmp/v1v2/manifest_19k_remainder.json")


# ---- (2) PRIORITY 73K ----
print(f"\n=== Priority 73K (Hough OR nonGREEN, not yet OK) ===")
cur.execute("""
    SELECT coin_id,
           MAX(CASE WHEN quality_flag <> 'GREEN' THEN 1 ELSE 0 END) AS has_nongreen,
           MAX(CASE WHEN vision_metadata LIKE '%"split_method": "hough"%' THEN 1 ELSE 0 END) AS has_hough
    FROM coin_detections GROUP BY coin_id
""")
all_coins = cur.fetchall()
priority_ids = sorted({
    r["coin_id"] for r in all_coins
    if r["coin_id"] is not None
    and (r["has_hough"] or r["has_nongreen"])
    and r["coin_id"] not in done
})
print(f"  priority count: {len(priority_ids):,}")
build_manifest(priority_ids, "/tmp/v1v2/manifest_priority_73k.json")


# ---- (3) SCAN TARGETS 186K (GREEN-non-hough untouched) ----
print(f"\n=== 186K GREEN-non-hough scan targets ===")
scan_ids = sorted({
    r["coin_id"] for r in all_coins
    if r["coin_id"] is not None
    and (not r["has_hough"]) and (not r["has_nongreen"])
    and r["coin_id"] not in done
})
print(f"  scan count: {len(scan_ids):,}")
# For the scan we only need coin_id + transparent_path (no source download needed)
cur.execute(f"""
    SELECT coin_id, side, detection_index, transparent_path
    FROM coin_detections
    WHERE coin_id IN ({','.join(['%s'] * len(scan_ids))})
      AND transparent_path IS NOT NULL
    ORDER BY coin_id, detection_index
""", tuple(scan_ids))
scan_rows = cur.fetchall()
with open("/tmp/v1v2/scan_targets_186k.json", "w") as f:
    json.dump(scan_rows, f, default=str)
print(f"wrote /tmp/v1v2/scan_targets_186k.json  ({len(scan_ids):,} coins, {len(scan_rows):,} rows)")

print("\nAll manifests built.")
