import cv2
import numpy as np
import os
from typing import Literal
from src.layer1_geometry import layer_1_structural_salience
from src.layer2_context import layer_2_context_probe

SourceType = Literal["auction", "unknown"]

MAX_DIMENSION = 3200 

def _load_and_resize(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 1.0
    if image_path.lower().endswith(".png"):
        img_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_alpha is not None and img_alpha.shape[2] == 4:
            alpha = img_alpha[:,:,3]
            img[alpha == 0] = [255, 255, 255]
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale

def analyze_image(image_path, source_type: SourceType = "unknown"):
    img, scale = _load_and_resize(image_path)
    if img is None: return {"status": "error", "error": "Load failed"}

    # Pass ndarray directly — no temp JPEG re-encoding.
    # This eliminates pixel-level differences between direct L1 calls
    # and the full pipeline path.

    # 1. Run Layer 1
    l1_result = layer_1_structural_salience(img, sensitivity="standard", source_type=source_type)

    if "objects" not in l1_result:
         l1_result = layer_1_structural_salience(img, sensitivity="high", source_type=source_type)

    if "objects" not in l1_result:
         return {"status": "failed", "last_error": "No objects found"}

    # 2. Run Layer 2 on EACH object
    detected_objects = l1_result["objects"]
    final_results = []

    for i, obj_data in enumerate(detected_objects):
        single_l1 = {
            "layer": 1,
            "classification": obj_data["classification"],
            "geometry": obj_data["geometry"],
            "contour": obj_data["contour"],
            "debug_data": obj_data["debug_data"]
        }

        l2_result = layer_2_context_probe(img, single_l1)

        container = l2_result['context_classification']['inferred_container']

        # Standard L2 container → classification mapping
        # (Semantic salvage rule removed in PR-5 — L5 salvage gates handle this)
        if container == "Round_Artifact":
            final_decision = "Round Object (Coin/Medallion)"
        elif container == "Coin_Planchet":
            final_decision = "Round Object (Coin)"
        elif container == "Fragment":
            final_decision = "Fragment"
        elif obj_data['classification']['label'] in ["Circle", "Polygon"]:
            final_decision = "Geometric Form"
        else:
            final_decision = "Artifact (Isolated)"

        final_results.append({
            "id": i+1,
            "final_classification": final_decision,
            "layer_1": single_l1,
            "layer_2": l2_result
        })

    out = {
        "status": "success",
        "detections": final_results,
    }
    # Propagate two-coin resolver metadata for upstream logging
    if "two_coin_resolution" in l1_result:
        out["two_coin_resolution"] = l1_result["two_coin_resolution"]
    # Propagate enclosure A/B trace (PR-6) into each detection for SideRecord
    if "l3_l4_ab" in l1_result:
        for det in final_results:
            det["l3_l4_ab"] = l1_result["l3_l4_ab"]
    return out
