#!/usr/bin/env python3
"""M1 serving-flip regression sweep — full-coverage A/B instrument.

Step 3 of the enable sequence needs to enforce, literally:
  (b) no top-1 card/material change on the six healing fixtures
  (d) NO drift on any fixture outside the six

`topk_sweep.py` (Bar 6) cannot enforce (d): it `continue`s past the 16
fixtures whose expected.yaml does not parse and past any dir with no
expected_top1 block, so those fixtures are invisible to it. This sweep grades
the same way where it can, but it RECORDS EVERY fixture dir that has an
obv/rev pair — graded or not — and adds two things Bar 6's tool did not have:

  * sha256 of the MASKED image actually fed to DINOv2, per side. This is the
    strongest possible drift detector and the one that matches §7.1's claim
    directly: a side whose estimator does not cross 110 produces a
    bit-identical mask, hence a bit-identical sha. Top-1 equality is a weaker
    consequence of that.
  * the mask no-op tripwire fields (pipeline cf7e6dd) per side, so the healing
    itself is witnessed rather than inferred from cosines.

Arm is whatever TRIVALAYA_BG_CORNER_LOCAL_TRUST is in this process env
(src/math_utils reads it per call, so no import-order concern).

Usage: m1_flip_sweep.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path("/home/claudeuser/trivalaya-pipeline")
HERE = REPO / "visual_search/tests/appv2_regression"
sys.path.insert(0, str(REPO))

from visual_search.appv2 import (  # noqa: E402
    STATE, UPLOAD_MAX_DIM, _build_material_centroids, _discover_materials,
    _dino_forward, _load_cards_from_db, _load_dino, _mask_query_image_meta,
)

GATE = "TRIVALAYA_BG_CORNER_LOCAL_TRUST"


def cap(img):
    """Same pre-mask resize as topk_probe.py / topk_sweep.py."""
    w, h = img.size
    if max(w, h) <= UPLOAD_MAX_DIM:
        return img.convert("RGB")
    img = img.convert("RGB").copy()
    img.thumbnail((UPLOAD_MAX_DIM, UPLOAD_MAX_DIM), Image.LANCZOS)
    return img


def find_pair(d: Path):
    obv = rev = None
    for p in d.iterdir():
        if p.stem == "obv":
            obv = p
        elif p.stem == "rev":
            rev = p
    return obv, rev


def embed_side(path: Path):
    """Replicates _build_query_vector's per-half step, keeping the telemetry.

    _build_query_vector -> _embed_image -> _mask_query_image (which is a
    back-compat wrapper over _mask_query_image_meta) -> _dino_forward. Calling
    the _meta form directly is the same pixels, same vector; it just does not
    throw the mask record away.
    """
    img = cap(Image.open(path))
    masked, meta = _mask_query_image_meta(img)
    rec = dict(meta)
    rec["sha_masked"] = hashlib.sha256(masked.tobytes()).hexdigest()
    rec["size_wh"] = list(masked.size)
    return _dino_forward(masked), rec


def main():
    out_path = Path(sys.argv[1])
    arm = "on" if os.environ.get(GATE, "").strip().lower() in ("1", "true") else "off"
    print(f"arm={arm}  {GATE}={os.environ.get(GATE)!r}", flush=True)
    # L1-lane provenance, RECORDED not enforced. appv2.py:43-49 setdefaults every
    # key of .env into os.environ at import time, so CLOSE_KERNEL_FRAC /
    # RIM_NEIGHBOR_GUARD are present here no matter what the invoking shell did —
    # and they are equally present inside the uvicorn service, which imports the
    # same module. Bar lane == service lane == the lane §7.1 measured in (its
    # tools/bg_estimator_m1_ab.py imports appv2 too). Held constant across arms.
    lane = {k: os.environ.get(k) for k in
            ("TRIVALAYA_CLOSE_KERNEL_FRAC", "TRIVALAYA_RIM_NEIGHBOR_GUARD")}
    print(f"L1 lane (post-import): {lane}", flush=True)

    STATE.model, STATE.device = _load_dino()
    materials = _discover_materials()
    blocks, keys = [], []
    for mat in materials:
        block, cids, _ = _build_material_centroids(
            mat["material"], mat["features_path"], mat["clusters_path"])
        blocks.append(block)
        keys.extend((mat["material"], c) for c in cids)
    centroids = np.vstack(blocks)
    cards_by_id, cluster_to_card_id, _ = _load_cards_from_db()
    print(f"centroids {centroids.shape}  materials {len(materials)}", flush=True)

    dirs = sorted(p for p in HERE.iterdir() if p.is_dir())
    rows, skipped = [], []
    t0 = time.time()
    for i, d in enumerate(dirs, 1):
        obv, rev = find_pair(d)
        if not obv or not rev:
            skipped.append({"fixture": d.name, "why": "no_obv_rev_pair"})
            continue

        q_obv, m_obv = embed_side(obv)
        q_rev, m_rev = embed_side(rev)
        q = np.concatenate([q_obv, q_rev]).astype(np.float32)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n

        sims = centroids @ q
        top = np.argsort(-sims)[:3]
        got = []
        for idx in top:
            mat, cid = keys[idx]
            card = cards_by_id.get(cluster_to_card_id.get((mat, cid))) or {}
            got.append({"material": mat, "cluster_id": int(cid),
                        "cos": round(float(sims[idx]), 6),
                        "stable_key": card.get("stable_key"),
                        "headline": card.get("headline"),
                        "ext_sig": card.get("extended_signature"),
                        "size": card.get("size") or card.get("n_coins")})
        t1 = got[0]

        # --- grading vs expected.yaml, same logic as Bar 6's topk_sweep.py ---
        grade = {"state": None, "checks": [], "fails": [], "rank_of_expected": None,
                 "exp_material": None, "exp_stable_key": None}
        exp_path = d / "expected.yaml"
        if not exp_path.exists():
            grade["state"] = "no_expected_yaml"
        else:
            try:
                spec = yaml.safe_load(exp_path.read_text()) or {}
            except Exception as exc:
                spec = None
                grade["state"] = f"unparseable:{type(exc).__name__}"
            if spec is not None:
                exp = spec.get("expected_top1") or {}
                if not exp:
                    grade["state"] = "no_expected_top1"
                else:
                    grade["state"] = "graded"
                    grade["exp_material"] = exp.get("material")
                    grade["exp_stable_key"] = exp.get("stable_key")
                    if exp.get("material"):
                        grade["checks"].append("material")
                        if t1["material"] != exp["material"]:
                            grade["fails"].append("material")
                    if exp.get("stable_key"):
                        grade["checks"].append("stable_key")
                        if t1["stable_key"] != exp["stable_key"]:
                            grade["fails"].append("stable_key")
                    want_sig = exp.get("extended_signature") or exp.get("ext_sig")
                    if want_sig:
                        grade["checks"].append("ext_sig")
                        if t1["ext_sig"] != want_sig:
                            grade["fails"].append("ext_sig")
                    if exp.get("size") is not None and t1["size"] is not None:
                        grade["checks"].append("size")
                        if int(t1["size"]) != int(exp["size"]):
                            grade["fails"].append("size")
                    if exp.get("stable_key"):
                        for r, g in enumerate(got, 1):
                            if g["stable_key"] == exp["stable_key"]:
                                grade["rank_of_expected"] = r
                                break

        rows.append({"fixture": d.name, "sides": {"obv": m_obv, "rev": m_rev},
                     "top3": got, "grade": grade})
        if i % 25 == 0:
            print(f"  [{i}/{len(dirs)}] {time.time()-t0:.0f}s", flush=True)

    payload = {"arm": arm, "gate_env": os.environ.get(GATE), "l1_lane": lane,
               "n_fixture_dirs": len(dirs), "rows": rows, "skipped": skipped,
               "elapsed_s": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(payload, indent=1))

    noops = [(r["fixture"], s) for r in rows for s in ("obv", "rev")
             if r["sides"][s].get("mask_noop")]
    fallbacks = [(r["fixture"], s) for r in rows for s in ("obv", "rev")
                 if not r["sides"][s].get("masked")]
    unmeasured = [(r["fixture"], s) for r in rows for s in ("obv", "rev")
                  if r["sides"][s].get("masked")
                  and r["sides"][s].get("mask_area_fraction") is None]
    graded = [r for r in rows if r["grade"]["state"] == "graded" and r["grade"]["checks"]]
    print("\n" + "=" * 78)
    print(f"arm                       : {arm}")
    print(f"fixture dirs              : {len(dirs)}")
    print(f"  swept (obv+rev present) : {len(rows)}")
    print(f"  skipped (no pair)       : {len(skipped)}")
    print(f"  GRADED vs expected.yaml : {len(graded)}"
          f"  clean={sum(1 for r in graded if not r['grade']['fails'])}"
          f"  mismatch={sum(1 for r in graded if r['grade']['fails'])}")
    print(f"mask telemetry (sides)    : no-op={len(noops)}  fallback={len(fallbacks)}"
          f"  UNMEASURED={len(unmeasured)}")
    for f, s in noops:
        print(f"    NO-OP  {f} [{s}]")
    for f, s in fallbacks:
        print(f"    FALLBK {f} [{s}]")
    print(f"elapsed {payload['elapsed_s']}s  -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
