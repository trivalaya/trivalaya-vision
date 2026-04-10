"""Per-ribbon quality scoring: metrics, composite score, and quality buckets.

All scoring logic lives here. Bucket thresholds and the composite formula
must remain fixed within a sweep.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


# ── Normalization Ranges ─────────────────────────────────────────────

@dataclass
class NormRanges:
    """Fixed clipping bounds for score normalization.

    Calibrated from baseline p5/p95. May be manually widened once after
    baseline inspection, then frozen for the sweep.
    """
    gradient_energy_lo: float = 0.0
    gradient_energy_hi: float = 0.01    # placeholder — calibrate from baseline
    row_variance_lo: float = 0.0
    row_variance_hi: float = 0.01       # placeholder — calibrate from baseline

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _normalize(x, lo, hi):
    """Clip-normalize x to [0, 1] using fixed bounds."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ── Per-Ribbon Metrics ───────────────────────────────────────────────


def compute_ribbon_metrics(ribbon):
    """Compute all per-ribbon quality metrics.

    Args:
        ribbon: float32 ndarray (H x W), 0-1 scale

    Returns:
        dict of metric_name -> value
    """
    # Content metrics
    nonzero_fraction = float((ribbon > 0).mean())
    nonblack_fraction = float((ribbon > 10.0 / 255.0).mean())
    mean_intensity = float(ribbon.mean())
    std_intensity = float(ribbon.std())

    # Structure metrics — work on uint8 for Sobel/Canny
    ribbon_u8 = (ribbon * 255).clip(0, 255).astype(np.uint8)

    # Gradient energy: mean of squared Sobel magnitudes
    sx = cv2.Sobel(ribbon_u8, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(ribbon_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = sx ** 2 + sy ** 2
    gradient_energy = float(grad_mag_sq.mean()) / (255.0 ** 2)  # normalize to ~0-1 range

    # Edge density: fraction of Canny edge pixels
    edges = cv2.Canny(ribbon_u8, 50, 150)
    edge_density = float((edges > 0).mean())

    # Row variance mean: mean of per-row intensity variances
    row_vars = np.var(ribbon, axis=1)
    row_variance_mean = float(row_vars.mean())

    return {
        "nonblack_fraction": round(nonblack_fraction, 4),
        "nonzero_fraction": round(nonzero_fraction, 4),
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "gradient_energy": round(gradient_energy, 6),
        "edge_density": round(edge_density, 4),
        "row_variance_mean": round(row_variance_mean, 6),
    }


# ── Composite Quality Score ──────────────────────────────────────────


def compute_quality_score(metrics, geometry_meta, norm_ranges, max_center_offset=150.0):
    """Compute the composite quality score.

    Args:
        metrics: dict from compute_ribbon_metrics
        geometry_meta: dict with used_fallback, center_offset keys
        norm_ranges: NormRanges instance
        max_center_offset: from UnwrapConfig, for penalty scaling

    Returns:
        float in [0, 1]
    """
    nonblack = metrics["nonblack_fraction"]
    grad_norm = _normalize(
        metrics["gradient_energy"],
        norm_ranges.gradient_energy_lo,
        norm_ranges.gradient_energy_hi,
    )
    rowvar_norm = _normalize(
        metrics["row_variance_mean"],
        norm_ranges.row_variance_lo,
        norm_ranges.row_variance_hi,
    )

    fallback_penalty = 1.0 if geometry_meta.get("used_fallback") else 0.0
    center_offset = geometry_meta.get("center_offset", 0.0)
    offset_penalty = float(np.clip(center_offset / max_center_offset, 0.0, 1.0))

    score = (
        0.35 * nonblack
        + 0.25 * grad_norm
        + 0.20 * rowvar_norm
        + 0.10 * (1.0 - fallback_penalty)
        + 0.10 * (1.0 - offset_penalty)
    )
    return round(float(score), 4)


# ── Quality Buckets ──────────────────────────────────────────────────


def assign_quality_bucket(metrics, geometry_meta, quality_score):
    """Assign a visual/content quality bucket to a successful ribbon.

    Buckets are strictly about ribbon content quality, not process
    reliability. Geometry concerns (fallback, offset) live on a
    separate axis via assign_geometry_flags().

    Bucket thresholds must remain fixed within a sweep.

    Returns one of: good, borderline, bad_blank, bad_low_signal
    """
    # Bad blank: very low content
    if metrics["nonblack_fraction"] < 0.15:
        return "bad_blank"

    # Bad low signal: content exists but no structure
    if metrics["gradient_energy"] < 0.0005 and metrics["edge_density"] < 0.02:
        return "bad_low_signal"

    # Good vs borderline by composite score
    if quality_score >= 0.55:
        return "good"

    return "borderline"


def assign_geometry_flags(geometry_meta):
    """Assign process/geometry reliability flags.

    These are independent of quality_bucket. A ribbon can be
    good (visually usable) AND have geometry flags (unreliable process).

    Returns list of flag strings (empty if geometry is clean).
    """
    flags = []
    if geometry_meta.get("used_fallback"):
        reason = geometry_meta.get("fallback_reason", "unknown")
        flags.append(f"fallback:{reason}")
    if geometry_meta.get("center_offset", 0) > 120:
        flags.append("high_center_offset")
    r = geometry_meta.get("radius", 999)
    if r < 120:
        flags.append("small_radius")
    return flags


# ── Calibration Helper ───────────────────────────────────────────────


def calibrate_norm_ranges(all_metrics, percentiles=(5, 95)):
    """Calibrate normalization ranges from a list of metric dicts (baseline run).

    Args:
        all_metrics: list of dicts from compute_ribbon_metrics
        percentiles: (lo_pct, hi_pct) tuple

    Returns:
        NormRanges
    """
    ge = [m["gradient_energy"] for m in all_metrics if m["gradient_energy"] is not None]
    rv = [m["row_variance_mean"] for m in all_metrics if m["row_variance_mean"] is not None]

    lo_p, hi_p = percentiles

    return NormRanges(
        gradient_energy_lo=float(np.percentile(ge, lo_p)) if ge else 0.0,
        gradient_energy_hi=float(np.percentile(ge, hi_p)) if ge else 0.01,
        row_variance_lo=float(np.percentile(rv, lo_p)) if rv else 0.0,
        row_variance_hi=float(np.percentile(rv, hi_p)) if rv else 0.01,
    )
