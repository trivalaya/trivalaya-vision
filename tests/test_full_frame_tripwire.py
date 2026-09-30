"""#231 C1 -- the full-frame tripwire in crop_quality.get_detection_quality_flag.

bbox crop-box area / frame area >= T AND coin_likelihood < 0.5  =>  RED 'full_frame_low_likelihood'.
Fixtures: lot 593 (cng FA-133, auction_data 772708; stored detection 1174986): crop box 0,0,3000x1440 (the whole
frame), solidity 0.954, coin_likelihood 0.352, aspect 2.083 -- GREEN/'clean' before this change.
"""
import importlib.util
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import crop_quality as cq  # noqa: E402
from src.crop_quality import get_detection_quality_flag as new_gate  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "crop_quality_pre231", os.path.join(os.path.dirname(__file__), "fixtures", "crop_quality_pre231.py"))
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
old_gate = _old.get_detection_quality_flag

W593, H593 = 3000, 1440
CAND593 = {"solidity": 0.954, "cx": 1500.0, "cy": 720.0 + 0.076 * H593 * 0, "coin_likelihood": 0.352}


def test_constants_are_the_stated_ones():
    assert cq.FULL_FRAME_MIN_REL_AREA == 0.85
    assert cq.FULL_FRAME_MAX_LIKELIHOOD == 0.5
    assert cq.FULL_FRAME_REASON == "full_frame_low_likelihood"


def test_lot_593_before_and_after():
    box = (0, 0, W593, H593)
    kw = dict(img_w=W593, img_h=H593, crop_margin_px=360)
    # BEFORE: GREEN / clean  (the defect)
    assert old_gate(candidate={k: v for k, v in CAND593.items() if k != "coin_likelihood"}, crop_box=box, **kw)[:2] == ("GREEN", "clean")
    # AFTER: RED / full_frame_low_likelihood, with the evidence in the metrics
    flag, reason, det = new_gate(candidate=CAND593, crop_box=box, **kw)
    assert (flag, reason) == ("RED", "full_frame_low_likelihood")
    assert det["reasons"][0] == "full_frame_low_likelihood"
    assert det["metrics"]["rel_area"] == 1.0 and det["metrics"]["coin_likelihood"] == 0.352
    assert det["metrics"]["aspect"] == 2.083


def test_negative_likelihood_at_or_above_pairing_gate_does_not_fire():
    box = (0, 0, W593, H593)
    for lik in (0.5, 0.501, 0.64, 0.9):
        assert new_gate({**CAND593, "coin_likelihood": lik}, box, W593, H593, crop_margin_px=360)[:2] == ("GREEN", "clean"), lik
    assert new_gate({**CAND593, "coin_likelihood": 0.499}, box, W593, H593, crop_margin_px=360)[:2] == ("RED", "full_frame_low_likelihood")


def test_absent_likelihood_never_fires():
    c = {k: v for k, v in CAND593.items() if k != "coin_likelihood"}
    assert new_gate(c, (0, 0, W593, H593), W593, H593, crop_margin_px=360)[:2] == ("GREEN", "clean")
    assert new_gate({**c, "coin_likelihood": None}, (0, 0, W593, H593), W593, H593, crop_margin_px=360)[:2] == ("GREEN", "clean")


def test_area_boundary_T():
    # 1000x1000 frame; crop (0,0,1000,h): rel_area = h/1000, aspect 1000/h inside [0.45, 2.2]
    def at(h): return new_gate({"solidity": 0.95, "cx": 500, "cy": h / 2, "coin_likelihood": 0.3}, (0, 0, 1000, h), 1000, 1000)
    assert at(850)[:2] == ("RED", "full_frame_low_likelihood")          # rel_area == T fires (>=)
    assert at(849)[0] != "RED" or at(849)[1] != "full_frame_low_likelihood"   # just below T does not


def test_existing_red_reasons_keep_precedence():
    # aspect 2.97 frame (naumann-style strip): stays RED degenerate_aspect, NOT relabelled
    flag, reason, _ = new_gate({"solidity": 0.95, "cx": 1500, "cy": 250, "coin_likelihood": 0.3}, (0, 0, 3000, 1010), 3000, 1010)
    assert (flag, reason) == ("RED", "degenerate_aspect_2.97")
    assert new_gate({"solidity": 1, "coin_likelihood": 0.3}, (5, 5, 5, 9), 100, 100)[1] == "empty_crop"
    assert new_gate({"solidity": 1, "coin_likelihood": 0.3}, (0, 0, 40, 40), 100000, 100000)[1] == "tiny_area"
    # the case the seeded grid found: RED 'severe_clamp' (0.85 <= rel_area < 0.90, 2 edge hits, ragged) is NOT relabelled
    cand = {"cx": 2034.0, "cy": 1762.5, "solidity": 0.017, "coin_likelihood": 0.456}
    assert new_gate(cand, (0, 445, 4000, 3000), 4000, 3000)[:2] == old_gate({k: v for k, v in cand.items() if k != "coin_likelihood"}, (0, 445, 4000, 3000), 4000, 3000)[:2] == ("RED", "severe_clamp")


def test_byte_identical_outside_the_conjunction_seeded_grid():
    """20,000 seeded inputs: wherever the conjunction does NOT hold the new gate's output equals the frozen
    pre-#231 gate's exactly; where it holds the result is RED/full_frame_low_likelihood UNLESS the old gate already
    returned RED (hard failure or severe_clamp), in which case it equals the old result -- the tripwire upgrades,
    it never relabels."""
    rng = random.Random(231)
    fired = unchanged = 0
    for _ in range(20000):
        W, H = rng.choice([(3000, 1440), (600, 300), (1000, 1000), (2400, 2400), (788, 394), (4000, 3000)])
        x1 = rng.choice([0, 0, rng.randint(0, W // 3)]); y1 = rng.choice([0, 0, rng.randint(0, H // 3)])
        x2 = rng.choice([W, W, rng.randint(x1 + 1, W)]); y2 = rng.choice([H, H, rng.randint(y1 + 1, H)])
        cand = {"solidity": rng.choice([None, rng.random()]), "cx": rng.choice([None, (x1 + x2) / 2 + rng.randint(-50, 50)]),
                "cy": rng.choice([None, (y1 + y2) / 2 + rng.randint(-50, 50)])}
        lik = rng.choice([None, rng.random(), 0.35, 0.5, 0.9])
        kw = dict(img_w=W, img_h=H, crop_margin_px=rng.choice([0, 12, int(0.12 * max(x2 - x1, y2 - y1))]),
                  neighbor_midpoint=rng.choice([None, W // 2]), side=rng.choice([None, "left", "right"]))
        o = old_gate(cand, (x1, y1, x2, y2), **kw)
        n = new_gate({**cand, "coin_likelihood": lik}, (x1, y1, x2, y2), **kw)
        rel = (x2 - x1) * (y2 - y1) / (W * H)
        holds = lik is not None and rel >= 0.85 and lik < 0.5
        if holds and o[0] != "RED":
            assert n[:2] == ("RED", "full_frame_low_likelihood"); fired += 1
        else:
            assert n == o, (cand, lik, (x1, y1, x2, y2), W, H, o, n); unchanged += 1
    assert fired > 500 and unchanged > 5000     # both arms actually exercised


def test_layer1_single_candidate_wiring():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from src import layer1_geometry as l1
    h, w = 1440, 3000
    contour = np.array([[[2, 2]], [[w - 3, 2]], [[w - 3, h - 3]], [[2, h - 3]]], dtype=np.int32)

    def cand(lik):
        return {"geometry": {"solidity": 0.954, "coin_likelihood": lik, "area": 1, "circularity": 0.3},
                "contour": contour, "bbox": cv2.boundingRect(contour), "debug_data": {}}
    for lik, expect in ((0.352, ("RED", "full_frame_low_likelihood")), (0.9, ("GREEN", "clean"))):
        c = [cand(lik)]
        l1._flag_single_coin_candidates(c, (h, w))
        assert (c[0]["debug_data"]["quality_flag"], c[0]["debug_data"]["quality_reason"]) == expect, lik
