#!/usr/bin/env python3
"""Bar 6 -- per-slice topk fixture sweep vs each expected.yaml.

CLAUDE.md standard regression, step 3: run every fixture under
visual_search/tests/appv2_regression/NN_*/ and compare TOP-1 against its
expected.yaml, keyed on NATURAL SURROGATES (material + stable_key /
extended_signature / size) -- never card_id, which is non-durable.

One process, model + centroids loaded once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path("/home/claudeuser/trivalaya-pipeline/visual_search/tests/appv2_regression")
REPO = Path("/home/claudeuser/trivalaya-pipeline")
sys.path.insert(0, str(REPO))


# --- PRE-M1 ARM -------------------------------------------------------------
# Monkeypatch src.math_utils.detect_background_histogram with the frozen
# verbatim pre-M1 copy kept in tests/test_bg_corner_local_trust.py, BEFORE
# appv2/decode_crop import it.  No production file is touched.
import importlib.util as _ilu
sys.path.insert(0, "/home/claudeuser/trivalaya-vision")
import src.math_utils as _mu
_spec = _ilu.spec_from_file_location(
    "_goldenmod", "/home/claudeuser/trivalaya-vision/tests/test_bg_corner_local_trust.py")
_gm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_gm)
_mu.detect_background_histogram = _gm._golden_pre_m1
print("PRE-M1 ARM: detect_background_histogram <- _golden_pre_m1", flush=True)
# ----------------------------------------------------------------------------

from visual_search.appv2 import (  # noqa: E402
    STATE, UPLOAD_MAX_DIM, _build_material_centroids, _build_query_vector,
    _discover_materials, _load_cards_from_db, _load_dino,
)


def cap(img):
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


def main():
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

    dirs = sorted(p for p in HERE.iterdir() if p.is_dir() and (p / "expected.yaml").exists())
    rows, unparseable, nopair, noexp = [], [], [], []
    for i, d in enumerate(dirs, 1):
        try:
            spec = yaml.safe_load((d / "expected.yaml").read_text()) or {}
        except Exception as exc:
            unparseable.append((d.name, f"{type(exc).__name__}"))
            continue
        exp = spec.get("expected_top1") or {}
        if not exp:
            noexp.append(d.name)
            continue
        obv, rev = find_pair(d)
        if not obv or not rev:
            nopair.append(d.name)
            continue

        q = _build_query_vector(cap(Image.open(obv)), cap(Image.open(rev)))
        sims = centroids @ q
        top = np.argsort(-sims)[:3]
        got = []
        for idx in top:
            mat, cid = keys[idx]
            card = cards_by_id.get(cluster_to_card_id.get((mat, cid))) or {}
            got.append({"material": mat, "cos": round(float(sims[idx]), 4),
                        "stable_key": card.get("stable_key"),
                        "headline": card.get("headline"),
                        "ext_sig": card.get("extended_signature"),
                        "size": card.get("size") or card.get("n_coins")})
        t1 = got[0]

        checks, fails = [], []
        if exp.get("material"):
            ok = t1["material"] == exp["material"]
            checks.append("material"); fails += [] if ok else ["material"]
        if exp.get("stable_key"):
            ok = t1["stable_key"] == exp["stable_key"]
            checks.append("stable_key"); fails += [] if ok else ["stable_key"]
        if exp.get("extended_signature") or exp.get("ext_sig"):
            want = exp.get("extended_signature") or exp.get("ext_sig")
            ok = t1["ext_sig"] == want
            checks.append("ext_sig"); fails += [] if ok else ["ext_sig"]
        if exp.get("size") is not None and t1["size"] is not None:
            ok = int(t1["size"]) == int(exp["size"])
            checks.append("size"); fails += [] if ok else ["size"]

        # top-3 recovery: did the expected key appear at rank 2/3 instead?
        rank = None
        if exp.get("stable_key"):
            for r, g in enumerate(got, 1):
                if g["stable_key"] == exp["stable_key"]:
                    rank = r
                    break
        rows.append({"fixture": d.name, "checks": checks, "fails": fails,
                     "rank_of_expected": rank, "top1": t1,
                     "exp_stable_key": exp.get("stable_key"),
                     "exp_material": exp.get("material")})
        if i % 25 == 0:
            print(f"  [{i}/{len(dirs)}]", flush=True)

    out = Path("/home/claudeuser/vision-wt-m1/specs/results/m1_ab/bar6_topk_sweep_preM1.json")
    out.write_text(json.dumps(
        {"rows": rows, "unparseable": unparseable, "no_expected_top1": noexp,
         "no_image_pair": nopair}, indent=1))

    graded = [r for r in rows if r["checks"]]
    clean = [r for r in graded if not r["fails"]]
    dirty = [r for r in graded if r["fails"]]
    print("\n" + "=" * 78)
    print(f"fixtures with expected.yaml : {len(dirs)}")
    print(f"  UNPARSEABLE yaml          : {len(unparseable)}  (pre-existing harness gap)")
    print(f"  no expected_top1 block    : {len(noexp)}")
    print(f"  no obv/rev image pair     : {len(nopair)}")
    print(f"  GRADED                    : {len(graded)}")
    print(f"    top-1 matches expected  : {len(clean)}")
    print(f"    top-1 MISMATCH          : {len(dirty)}")
    print("=" * 78)
    for r in dirty:
        print(f"\n  {r['fixture']}  fails={r['fails']}  expected_at_rank={r['rank_of_expected']}")
        print(f"     want mat={r['exp_material']} key={r['exp_stable_key']}")
        print(f"     got  mat={r['top1']['material']} key={r['top1']['stable_key']} "
              f"cos={r['top1']['cos']} headline={r['top1']['headline']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
