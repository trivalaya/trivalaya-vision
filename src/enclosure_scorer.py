"""
Enclosure-first candidate ranking (PR-6).

Replaces the legacy `area * edge_support` scoring with a bounded linear
score that explicitly rewards the largest enclosed coin-like region.

The module is observe-only by default — it computes a shadow ranking
alongside the control ranking and emits an A/B trace for analysis.
Promotion to authoritative requires explicit CP-3 approval.

v0 weights favour enclosure scale (area_norm = 0.42) over contrast
(edge_support = 0.18), with penalties for border clipping and
off-center placement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnclosureScorerConfig:
    """Weight configuration for enclosure-first scoring v0.

    Sum of positive weights = 0.82; penalties = 0.18.
    internal_texture_dominance is stubbed at 0.00 in v0.
    """
    w_a: float = 0.42   # area_norm — enclosure scale (dominant)
    w_c: float = 0.22   # circularity — coin-like shape
    w_e: float = 0.18   # edge_support — rim evidence
    w_b: float = 0.10   # border_clip_penalty
    w_o: float = 0.08   # offcenter_penalty
    w_t: float = 0.00   # internal_texture_dominance (inactive v0)
    circ_cap: float = 0.85  # circularity saturation cap — PR-6 golden
    relative_area: bool = True  # use area/max_candidate_area — PR-6 golden


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_border_clip_penalty(contour: np.ndarray, image_shape: tuple) -> float:
    """Fraction of contour points within the border band of image edges.

    Border band width = max(3, int(0.01 * max(h, w))).

    Returns:
        Float in [0, 1].  0 = no clipping, 1 = fully on border.
    """
    if contour is None or len(contour) == 0:
        return 0.0

    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return 0.0

    band = max(3, int(0.01 * max(h, w)))
    pts = contour.reshape(-1, 2)
    if len(pts) == 0:
        return 0.0

    xs, ys = pts[:, 0], pts[:, 1]
    near_border = (
        (xs < band) | (xs >= w - band) |
        (ys < band) | (ys >= h - band)
    )
    return float(np.sum(near_border)) / len(pts)


def compute_offcenter_penalty(contour: np.ndarray, image_shape: tuple) -> float:
    """Distance of contour centroid from image center, normalised to [0, 1].

    Uses cv2.moments for centroid.  max_possible_dist = 0.5 * sqrt(h² + w²).

    Returns:
        Float in [0, 1].  0 = perfectly centred, 1 = at corner.
    """
    if contour is None or len(contour) == 0:
        return 0.0

    h, w = image_shape[:2]
    if h == 0 or w == 0:
        return 0.0

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    center_x, center_y = w / 2.0, h / 2.0
    dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    max_dist = 0.5 * np.sqrt(float(h) ** 2 + float(w) ** 2)

    if max_dist == 0:
        return 0.0
    return min(1.0, dist / max_dist)


def compute_enclosure_metrics(candidate: dict, image_shape: tuple) -> dict:
    """Compute all enclosure metrics for a single L1 candidate.

    Args:
        candidate: L1 candidate dict with keys 'contour', 'geometry',
                   'classification', 'bbox'.
        image_shape: (h, w) or (h, w, c).

    Returns:
        Dict with metric values and ``metric_status`` map.
    """
    h, w = image_shape[:2]
    image_area = h * w

    contour = candidate.get("contour")
    area = candidate["geometry"]["area"]
    edge_support = candidate["classification"]["confidence"]

    # Hull circularity: 4π·area / hull_perimeter² — ignores edge clips,
    # test cuts, and rim irregularities that penalise raw contour circularity.
    circularity = candidate["geometry"]["circularity"]  # fallback
    if contour is not None:
        cnt = np.array(contour, dtype=np.int32) if isinstance(contour, list) else contour
        if len(cnt) >= 5:
            hull = cv2.convexHull(cnt)
            hull_perim = cv2.arcLength(hull, True)
            if hull_perim > 0:
                circularity = min(1.0, (4 * np.pi * area) / (hull_perim ** 2))

    # area_norm ∈ [0, 1]
    area_norm = area / image_area if image_area > 0 else 0.0
    area_norm = min(1.0, max(0.0, area_norm))

    border_clip = compute_border_clip_penalty(contour, image_shape)
    offcenter = compute_offcenter_penalty(contour, image_shape)

    return {
        "area_norm": round(area_norm, 4),
        "circularity": round(float(circularity), 4),
        "edge_support": round(float(edge_support), 4),
        "border_clip_penalty": round(border_clip, 4),
        "offcenter_penalty": round(offcenter, 4),
        "internal_texture_dominance": None,
        "metric_status": {
            "area_norm": "measured",
            "circularity": "measured",
            "edge_support": "measured",
            "border_clip_penalty": "measured",
            "offcenter_penalty": "measured",
            "internal_texture_dominance": "not_measured",
        },
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_enclosure_v0(
    metrics: dict,
    config: Optional[EnclosureScorerConfig] = None,
) -> float:
    """Bounded linear enclosure score (v0).

    S = w_a·area_norm + w_c·circularity + w_e·edge_support
      − w_b·border_clip − w_o·offcenter − w_t·texture

    Returns:
        Float score (can be negative if penalties dominate).
    """
    if config is None:
        config = EnclosureScorerConfig()

    area_norm = metrics.get("area_norm", 0.0) or 0.0
    circularity = min(metrics.get("circularity", 0.0) or 0.0, config.circ_cap)
    edge_support = metrics.get("edge_support", 0.0) or 0.0
    border_clip = metrics.get("border_clip_penalty", 0.0) or 0.0
    offcenter = metrics.get("offcenter_penalty", 0.0) or 0.0
    # v0: texture always 0.0, weight 0.00
    texture = 0.0

    score = (
        config.w_a * area_norm
        + config.w_c * circularity
        + config.w_e * edge_support
        - config.w_b * border_clip
        - config.w_o * offcenter
        - config.w_t * texture
    )
    return round(score, 6)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _shadow_sort_key(score_b: float, metrics: dict, idx: int) -> tuple:
    """Deterministic sort key for shadow ranking (Section 4.3).

    Tie-break order (when scores are within 1e-6):
      1. Higher area_norm
      2. Higher edge_support
      3. Higher circularity
      4. Lower border_clip_penalty
      5. Stable candidate index
    """
    quantized = round(score_b, 6)
    return (
        -quantized,
        -(metrics.get("area_norm", 0.0) or 0.0),
        -(metrics.get("edge_support", 0.0) or 0.0),
        -(metrics.get("circularity", 0.0) or 0.0),
        (metrics.get("border_clip_penalty", 0.0) or 0.0),
        idx,
    )


def rank_candidates_ab(
    candidates: list,
    image_shape: tuple,
    config: Optional[EnclosureScorerConfig] = None,
) -> dict:
    """Compute A/B ranking for a list of L1 candidates.

    Args:
        candidates: L1 candidate dicts (post-NMS/suppression).
        image_shape: (h, w) or (h, w, c).
        config: Scoring weights.  Defaults to v0.

    Returns:
        Full ``l3_l4_ab`` trace dict per PR-6 Section 5.
    """
    if config is None:
        config = EnclosureScorerConfig()

    if not candidates:
        return {
            "ab_experiment_id": "pr6_enclosure_v0",
            "selector_control": {
                "winner_id": -1,
                "winner_score": 0.0,
                "rank_list": [],
            },
            "selector_shadow": {
                "winner_id": -1,
                "winner_score": 0.0,
                "rank_list": [],
            },
            "candidate_scores": [],
            "agreement": True,
            "winner_delta": 0.0,
        }

    # --- Compute metrics for every candidate ---
    all_metrics: List[dict] = []
    for cand in candidates:
        all_metrics.append(compute_enclosure_metrics(cand, image_shape))

    # --- Relative area: replace area_norm with area/max_candidate_area ---
    if config.relative_area and len(all_metrics) > 1:
        max_area = max(m["area_norm"] for m in all_metrics)
        if max_area > 0:
            for m in all_metrics:
                m["area_norm"] = round(m["area_norm"] / max_area, 4)

    # --- Score A (control): area * edge_support (current behaviour) ---
    control_scores: List[float] = []
    for cand in candidates:
        score_a = cand["geometry"]["area"] * cand["classification"]["confidence"]
        control_scores.append(score_a)

    # --- Score B (shadow): enclosure v0 ---
    shadow_scores: List[float] = []
    for m in all_metrics:
        shadow_scores.append(score_enclosure_v0(m, config))

    # --- Rank A: descending by control score, stable by index ---
    control_ranked = sorted(
        range(len(control_scores)),
        key=lambda i: (-control_scores[i], i),
    )

    # --- Rank B: descending by shadow score, deterministic tie-break ---
    shadow_ranked = sorted(
        range(len(shadow_scores)),
        key=lambda i: _shadow_sort_key(shadow_scores[i], all_metrics[i], i),
    )

    # --- Build candidate_scores ---
    candidate_scores = []
    for i, m in enumerate(all_metrics):
        candidate_scores.append({
            "candidate_id": i,
            "area_norm": m["area_norm"],
            "circularity": m["circularity"],
            "edge_support": m["edge_support"],
            "border_clip_penalty": m["border_clip_penalty"],
            "offcenter_penalty": m["offcenter_penalty"],
            "internal_texture_dominance": m["internal_texture_dominance"],
            "score_control": round(control_scores[i], 2),
            "score_shadow": shadow_scores[i],
            "metric_status": m["metric_status"],
        })

    # --- Agreement + delta ---
    control_winner = control_ranked[0]
    shadow_winner = shadow_ranked[0]
    agreement = control_winner == shadow_winner

    if len(shadow_ranked) > 1:
        winner_delta = round(
            shadow_scores[shadow_ranked[0]] - shadow_scores[shadow_ranked[1]],
            6,
        )
    else:
        winner_delta = 0.0

    return {
        "ab_experiment_id": "pr6_enclosure_v0",
        "selector_control": {
            "winner_id": control_winner,
            "winner_score": round(control_scores[control_winner], 2),
            "rank_list": control_ranked,
        },
        "selector_shadow": {
            "winner_id": shadow_winner,
            "winner_score": shadow_scores[shadow_winner],
            "rank_list": shadow_ranked,
        },
        "candidate_scores": candidate_scores,
        "agreement": agreement,
        "winner_delta": winner_delta,
    }
