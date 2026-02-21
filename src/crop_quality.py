"""
Crop Quality Flag — evaluates crop quality for ML readiness.

Returns (flag, primary_reason, details_dict) where flag is RED/YELLOW/GREEN.
Used by layer1_geometry.py for all L1 detections (two-coin splits and single-coin candidates).
"""

from typing import Dict, List, Optional, Tuple, Any

def get_detection_quality_flag(
    candidate: Dict,
    crop_box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    neighbor_midpoint: Optional[int] = None,
    side: Optional[str] = None,  # 'left' or 'right'
    *,
    crop_margin_px: int = 0,
    min_abs_area: int = 2500,        # ~50x50px
    min_rel_area: float = 0.005,     # 0.5% of image
    min_aspect: float = 0.45,
    max_aspect: float = 2.2,
    low_solidity_thr: float = 0.90,
    offcenter_thr: float = 0.15,     # >15% deviation
    clamp_tol_px: int = 1,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Evaluates crop quality for ML readiness.

    Args:
        side: 'left' (checks x2 vs midpoint) or 'right' (checks x1 vs midpoint).
              If None, checks both edges against midpoint.
        crop_margin_px: Padding (in px) added around the coin bbox to form the
              crop box.  Used for margin-aware edge clamping — only flags an
              edge when the *coin itself* (not just the margin) is near the
              image boundary.

    Returns:
        (flag, primary_reason, details_dict)
    """
    x1, y1, x2, y2 = crop_box
    w = x2 - x1
    h = y2 - y1
    reasons: List[str] = []

    # metrics snapshot for logging
    metrics = {
        "w": w, "h": h,
        "aspect": 0.0, "solidity": None,
        "off_center_x": None, "off_center_y": None
    }

    # ---- 1. RED: Hard Failures (Geometric Invalidity) ----
    if w <= 0 or h <= 0:
        return "RED", "empty_crop", {"reasons": ["empty_crop"], "metrics": metrics}

    aspect = w / h if h > 0 else 0.0
    metrics["aspect"] = round(aspect, 3)

    if aspect < min_aspect or aspect > max_aspect:
        return "RED", f"degenerate_aspect_{aspect:.2f}", {"reasons": [f"degenerate_aspect"], "metrics": metrics}

    # Area Check
    crop_area = w * h
    img_area = max(1, img_w * img_h)
    rel_area = crop_area / img_area

    if crop_area < min_abs_area and rel_area < min_rel_area:
        return "RED", "tiny_area", {"reasons": [f"tiny_area"], "metrics": metrics}

    # ---- 2. YELLOW: Warnings (Contextual Issues) ----

    # Extract coin center early — needed by edge clamp and off-center checks
    cx = candidate.get("cx")
    cy = candidate.get("cy")

    # A. Neighbor Clamp (Directional)
    if neighbor_midpoint is not None:
        hit_clamp = False
        if side == 'left':
            # Left coin: Check if Right Edge (x2) hit the wall
            if abs(x2 - neighbor_midpoint) <= clamp_tol_px: hit_clamp = True
        elif side == 'right':
            # Right coin: Check if Left Edge (x1) hit the wall
            if abs(x1 - neighbor_midpoint) <= clamp_tol_px: hit_clamp = True
        else:
            # Side unknown: Check both
            if abs(x2 - neighbor_midpoint) <= clamp_tol_px or abs(x1 - neighbor_midpoint) <= clamp_tol_px:
                hit_clamp = True

        if hit_clamp:
            reasons.append("neighbor_clamp")

    # B. Edge Clamp (Image boundary) — margin-aware
    #
    # Size exemption: if crop covers ≥90% of the image, edge-to-edge is the
    # intended photography style and edge clamping is expected, not a defect.
    #
    # Margin-aware: when crop_margin_px > 0, only count an edge as hit when
    # the coin's own extent (crop minus margin) reaches the image boundary,
    # not when only the padding is consumed.
    edge_hits = 0
    if rel_area < 0.90:
        coin_half_w = max(1.0, w / 2.0 - crop_margin_px)
        coin_half_h = max(1.0, h / 2.0 - crop_margin_px)
        edge_tol = 2  # px

        # Approximate coin edges using center + half-extent
        if cx is not None:
            coin_left = cx - coin_half_w
            coin_right = cx + coin_half_w
        else:
            coin_left = x1 + crop_margin_px
            coin_right = x2 - crop_margin_px

        if cy is not None:
            coin_top = cy - coin_half_h
            coin_bottom = cy + coin_half_h
        else:
            coin_top = y1 + crop_margin_px
            coin_bottom = y2 - crop_margin_px

        if coin_left <= edge_tol: edge_hits += 1
        if coin_top <= edge_tol: edge_hits += 1
        if coin_right >= img_w - edge_tol: edge_hits += 1
        if coin_bottom >= img_h - edge_tol: edge_hits += 1

    if edge_hits > 0:
        reasons.append(f"edge_clamp_{edge_hits}")

    # C. Solidity
    solidity = candidate.get("solidity")
    metrics["solidity"] = solidity
    if solidity is None:
        reasons.append("solidity_missing")
    elif solidity < low_solidity_thr:
        reasons.append(f"low_solidity_{solidity:.2f}")

    # D. 2D Off-Center Check
    # Horizontal Offset
    if cx is None:
        reasons.append("cx_missing")
    else:
        crop_cx = x1 + (w / 2.0)
        off_x = abs(crop_cx - cx) / max(1.0, w)
        metrics["off_center_x"] = round(off_x, 3)
        if off_x > offcenter_thr:
            reasons.append(f"off_center_x_{off_x:.2f}")

    # Vertical Offset (New)
    if cy is not None:
        crop_cy = y1 + (h / 2.0)
        off_y = abs(crop_cy - cy) / max(1.0, h)
        metrics["off_center_y"] = round(off_y, 3)
        if off_y > offcenter_thr:
            reasons.append(f"off_center_y_{off_y:.2f}")

    # ---- 3. RED ESCALATION ----
    is_weak_geometry = any("low_solidity" in r for r in reasons)
    is_smallish = rel_area < (min_rel_area * 1.5)

    # Escalation: 2+ edges hit AND (ragged OR small)
    if edge_hits >= 2 and (is_weak_geometry or is_smallish):
        reasons.insert(0, "severe_clamp_escalation")
        return "RED", "severe_clamp", {"reasons": reasons, "metrics": metrics}

    # ---- 4. FINAL VERDICT ----
    if reasons:
        return "YELLOW", reasons[0], {"reasons": reasons, "metrics": metrics}

    return "GREEN", "clean", {"reasons": ["clean"], "metrics": metrics}
