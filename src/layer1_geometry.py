import cv2
import numpy as np
from typing import Dict, List
from src.config import Layer1Config, CoinConfig
from src.math_utils import (
    detect_background_histogram,
    is_contour_valid,
    compute_circularity_safe,
    non_maximum_suppression,
    validate_rim_recovery
)
from src.rim_logic import recover_rim


def layer_1_structural_salience(image_path: str, sensitivity: str = "standard") -> Dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"layer": 1, "status": "error", "error": "image_load_failed"}

    h, w = img.shape[:2]
    total_area = h * w

    # --- Grayscale + background handling ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bg, _ = detect_background_histogram(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Threshold polarity ---
    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Edge detection ---
    v = np.median(gray)
    sigma = Layer1Config.Standard.CANNY_SIGMA
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    edge_zone = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    # --- Binary segmentation ---
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: List[Dict] = []

    for c in contours:
        valid, _ = is_contour_valid(c, min_area=Layer1Config.Standard.MIN_AREA_PX)
        if not valid:
            continue

        area = cv2.contourArea(c)
        if area > Layer1Config.MAX_AREA_RATIO * total_area:
            continue

        # --- Edge support ---
        perimeter_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(perimeter_mask, [c], -1, 255, 1)
        perimeter_px = cv2.countNonZero(perimeter_mask)
        if perimeter_px == 0:
            continue

        overlap = cv2.bitwise_and(perimeter_mask, edge_zone)
        edge_support = cv2.countNonZero(overlap) / perimeter_px

        # Allow circular shapes through even with weaker edges
        circularity = compute_circularity_safe(c)
        if edge_support < Layer1Config.Standard.EDGE_SUPPORT_MIN and circularity < 0.70:
            continue

        # --- Optional rim recovery ---
        final_c = c
        recovered = False

        if CoinConfig.ENABLE_RIM_RECOVERY and circularity < Layer1Config.CIRCULARITY_RELAXED:
            new_c, conf = recover_rim(img, c)
            if new_c is not None and validate_rim_recovery(new_c, c, (h, w)):
                final_c = new_c
                recovered = True
                area = cv2.contourArea(final_c)
                circularity = compute_circularity_safe(final_c)

        if final_c is None or len(final_c) < 3:
            continue

        hull = cv2.convexHull(final_c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        bbox = cv2.boundingRect(final_c)

        coin_likelihood = (
            0.45 * circularity +
            0.35 * edge_support +
            0.20 * solidity
        )

        label = "Circle" if circularity > Layer1Config.CIRCULARITY_THRESHOLD else "Geometric"

        candidates.append({
            "score": area * edge_support,
            "classification": {
                "label": label,
                "confidence": round(edge_support, 2)
            },
            "geometry": {
                "area": int(area),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3),
                "coin_likelihood": round(coin_likelihood, 3)
            },
            "contour": final_c,
            "bbox": bbox,
            "debug_data": {
                "rim_recovered": recovered
            }
        })

    if not candidates:
        return {"layer": 1, "status": "no_structurally_valid_object"}

    # --- NMS ---
    candidates = non_maximum_suppression(
        candidates,
        iou_threshold=Layer1Config.NMS_IOU_THRESHOLD,
        containment_threshold=0.40
    )

    return {
        "layer": 1,
        "status": "success",
        "objects": candidates[:Layer1Config.MAX_DETECTIONS]
    }
