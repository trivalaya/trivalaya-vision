#!/usr/bin/env python3
"""Step 1: Select confusion subset for legend stylometry PoC.

Mines the clustering DB for Roman Imperial authorities that co-occur in the
same leaf clusters (high authority entropy = still confused after drill-down).
Outputs a CSV of coins with image URLs for the unwrap pipeline.

    python -m legend_cnn.select_subset
"""

import sys
from collections import Counter, defaultdict
from math import log2

import mysql.connector
import pandas as pd

from legend_cnn.config import (
    CONFUSION_SUBSET_CSV,
    DATA_DIR,
    DB_CONFIG,
    MIN_AUTHORITY_IMAGES,
    SPACES_BUCKET,
    SPACES_ENDPOINT,
)


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_latest_run_id(conn):
    """Get the most recent clustering run_id from cluster_run_xref."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT run_id FROM cluster_run_xref "
        "ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        print("ERROR: No runs found in cluster_run_xref")
        sys.exit(1)
    return row["run_id"]


def load_leaf_clusters(conn, run_id):
    """Load leaf clusters for the run.

    A leaf cluster is one whose cluster_id does not appear as another row's
    parent_cluster_id.  This gives us the deepest drill-down level: if a
    top-level cluster was split, we get the sub-clusters; if not, we get the
    top-level cluster itself.
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT cluster_id, coin_count, purity, authority_purity
        FROM cluster_run_xref
        WHERE run_id = %s
          AND cluster_id NOT IN (
              SELECT DISTINCT parent_cluster_id
              FROM cluster_run_xref
              WHERE run_id = %s AND parent_cluster_id IS NOT NULL
          )
    """, (run_id, run_id))
    rows = cur.fetchall()
    cur.close()
    return rows


def load_cluster_coins_with_authority(conn, run_id, cluster_ids):
    """Load coin_id + authority for all coins in the given clusters.

    Filters to Roman Imperial only and coins with a non-empty authority.
    """
    if not cluster_ids:
        return []

    cur = conn.cursor(dictionary=True)
    # Process in chunks to avoid overly large IN clauses
    all_rows = []
    chunk_size = 500
    for i in range(0, len(cluster_ids), chunk_size):
        chunk = cluster_ids[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(f"""
            SELECT ccx.coin_id, ccx.cluster_id, m.authority
            FROM coin_cluster_xref ccx
            JOIN ml_coin_dataset m ON m.coin_id = ccx.coin_id
            WHERE ccx.run_id = %s
              AND ccx.cluster_id IN ({placeholders})
              AND m.period = 'Roman Imperial'
              AND m.authority IS NOT NULL
              AND m.authority != ''
        """, [run_id] + chunk)
        all_rows.extend(cur.fetchall())
    cur.close()
    return all_rows


def shannon_entropy(counts):
    """Shannon entropy in bits from a dict/Counter of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * log2(p)
    return ent


def rank_confused_authorities(cluster_coins):
    """Rank authority pairs by co-occurrence in high-entropy clusters.

    Returns:
        pair_scores: dict of (auth_a, auth_b) -> co-occurrence count
        cluster_entropies: dict of cluster_id -> entropy
    """
    # Group by cluster
    cluster_to_coins = defaultdict(list)
    for row in cluster_coins:
        cluster_to_coins[row["cluster_id"]].append(row["authority"])

    # Compute per-cluster authority entropy
    cluster_entropies = {}
    for cid, authorities in cluster_to_coins.items():
        counts = Counter(authorities)
        cluster_entropies[cid] = shannon_entropy(counts)

    # From high-entropy clusters, count co-occurring authority pairs
    pair_scores = Counter()
    for cid, authorities in cluster_to_coins.items():
        if cluster_entropies[cid] < 1.0:  # skip near-pure clusters
            continue
        counts = Counter(authorities)
        auths = sorted(counts.keys())
        for i, a in enumerate(auths):
            for b in auths[i + 1:]:
                pair_scores[(a, b)] += min(counts[a], counts[b])

    return pair_scores, cluster_entropies


def select_target_authorities(pair_scores, authority_counts):
    """Select 5-10 authorities from the most confused pairs.

    Greedily picks authorities that appear in the highest-scoring pairs,
    filtering by minimum image count.
    """
    # Filter to authorities meeting minimum
    eligible = {a for a, c in authority_counts.items() if c >= MIN_AUTHORITY_IMAGES}

    selected = set()
    # Sort pairs by co-occurrence score descending
    for (a, b), _score in pair_scores.most_common():
        if len(selected) >= 10:
            break
        if a in eligible and a not in selected:
            selected.add(a)
        if b in eligible and b not in selected:
            selected.add(b)

    if len(selected) < 5:
        # Pad with highest-count eligible authorities not yet selected
        remaining = sorted(
            [(a, c) for a, c in authority_counts.items()
             if a in eligible and a not in selected],
            key=lambda x: -x[1],
        )
        for a, _ in remaining:
            if len(selected) >= 5:
                break
            selected.add(a)

    return selected


def fetch_all_coins_for_authorities(conn, authorities):
    """Fetch ALL coins in ml_coin_dataset for the given authorities.

    Not limited to any cluster — pulls every Roman Imperial coin that
    matches one of the selected authority names.
    """
    cur = conn.cursor(dictionary=True)
    placeholders = ",".join(["%s"] * len(authorities))
    cur.execute(f"""
        SELECT m.coin_id, m.authority
        FROM ml_coin_dataset m
        WHERE m.period = 'Roman Imperial'
          AND m.authority IN ({placeholders})
          AND m.coin_id IS NOT NULL
    """, list(authorities))
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_image_paths(conn, coin_ids):
    """Fetch highres_path and transparent_path from coin_detections.

    If multiple detections exist per coin_id, takes the latest (MAX id).
    Returns dict: coin_id -> {highres_path, transparent_path}
    """
    if not coin_ids:
        return {}

    cur = conn.cursor(dictionary=True)
    result = {}
    chunk_size = 1000
    for i in range(0, len(coin_ids), chunk_size):
        chunk = coin_ids[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(f"""
            SELECT cd.coin_id, cd.highres_path, cd.transparent_path
            FROM coin_detections cd
            INNER JOIN (
                SELECT coin_id, MAX(id) AS max_id
                FROM coin_detections
                WHERE coin_id IN ({placeholders})
                  AND highres_path IS NOT NULL AND highres_path != ''
                GROUP BY coin_id
            ) latest ON cd.id = latest.max_id
        """, chunk)
        for row in cur.fetchall():
            result[row["coin_id"]] = {
                "highres_path": row["highres_path"],
                "transparent_path": row["transparent_path"] or "",
            }
    cur.close()
    return result


def spaces_url(path):
    """Build full DO Spaces URL from a relative path."""
    if not path:
        return ""
    return f"{SPACES_ENDPOINT}/{SPACES_BUCKET}/{path}"


def main():
    conn = get_connection()

    # 1. Find latest run
    run_id = get_latest_run_id(conn)
    print(f"Latest run_id: {run_id}")

    # 2. Load leaf clusters
    leaves = load_leaf_clusters(conn, run_id)
    print(f"Leaf clusters: {len(leaves)}")

    leaf_ids = [r["cluster_id"] for r in leaves]

    # 3. Load coins with authority for those clusters (Roman Imperial only)
    cluster_coins = load_cluster_coins_with_authority(conn, run_id, leaf_ids)
    print(f"Roman Imperial coins in leaf clusters: {len(cluster_coins)}")

    if not cluster_coins:
        print("ERROR: No Roman Imperial coins found. Check period values in ml_coin_dataset.")
        sys.exit(1)

    # 4. Authority distribution
    authority_counts = Counter(r["authority"] for r in cluster_coins)
    print(f"\nAll Roman Imperial authorities: {len(authority_counts)}")
    print("Top 20 by count:")
    for auth, cnt in authority_counts.most_common(20):
        print(f"  {auth:30s} {cnt:>6,d}")

    # 5. Rank by entropy / co-occurrence
    pair_scores, cluster_entropies = rank_confused_authorities(cluster_coins)
    high_ent = sum(1 for e in cluster_entropies.values() if e >= 1.0)
    print(f"\nHigh-entropy clusters (>= 1 bit): {high_ent} / {len(cluster_entropies)}")

    print("\nTop confused authority pairs:")
    for (a, b), score in pair_scores.most_common(15):
        print(f"  {a:25s} <-> {b:25s}  score={score}")

    # 6. Select target authorities
    selected = select_target_authorities(pair_scores, authority_counts)
    print(f"\nSelected authorities ({len(selected)}):")
    for auth in sorted(selected):
        print(f"  {auth:30s} {authority_counts[auth]:>6,d} images")

    # 7. Fetch ALL coins for selected authorities (not just cluster-bound)
    print("\nFetching all coins for selected authorities...")
    all_coins = fetch_all_coins_for_authorities(conn, sorted(selected))
    # Deduplicate by coin_id
    seen = set()
    unique_coins = []
    for r in all_coins:
        if r["coin_id"] not in seen:
            seen.add(r["coin_id"])
            unique_coins.append(r)
    print(f"Total coins for {len(selected)} authorities: {len(unique_coins)}")

    # 8. Fetch image paths from coin_detections
    coin_ids = [r["coin_id"] for r in unique_coins]
    image_paths = fetch_image_paths(conn, coin_ids)
    conn.close()

    # 9. Build output dataframe
    auth_to_int = {a: i for i, a in enumerate(sorted(selected))}

    rows = []
    for r in unique_coins:
        cid = r["coin_id"]
        if cid not in image_paths:
            continue
        imgs = image_paths[cid]
        if not imgs["highres_path"]:
            continue
        rows.append({
            "coin_id": cid,
            "authority": r["authority"],
            "authority_label_int": auth_to_int[r["authority"]],
            "highres_url": spaces_url(imgs["highres_path"]),
            "transparent_url": spaces_url(imgs["transparent_path"]),
        })

    df = pd.DataFrame(rows)
    print(f"\nFinal subset: {len(df)} coins with valid images")

    # 10. Summary
    print("\n=== SUMMARY ===")
    print(f"Authorities: {df['authority'].nunique()}")
    print(f"Total images: {len(df)}")
    print("\nPer-authority counts:")
    for auth, group in df.groupby("authority"):
        print(f"  [{auth_to_int[auth]}] {auth:30s} {len(group):>6,d}")

    # 11. Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONFUSION_SUBSET_CSV, index=False)
    print(f"\nSaved to {CONFUSION_SUBSET_CSV}")


if __name__ == "__main__":
    main()
