"""Core unwrap functions: parameterized center/radius, polar unwrap, caching.

All geometry and image-processing logic lives here. Higher-level scripts
(unwrap.py, run_experiment.py) call these functions.
"""

import hashlib
import io
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import boto3
import cv2
import numpy as np

from legend_cnn.config import (
    IMAGE_CACHE_DIR,
    SPACES_ACCESS_KEY,
    SPACES_BUCKET,
    SPACES_ENDPOINT,
    SPACES_REGION,
    SPACES_SECRET_KEY,
)


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class UnwrapConfig:
    """All parameters that control a single unwrap run."""

    config_id: str = "default"

    # Geometry / annulus
    inner_r_ratio: float = 0.68
    outer_r_ratio: float = 0.92

    # Center/radius estimation
    center_method: str = "moments"       # bbox | moments | min_enclosing_circle
    max_center_offset: float = 200.0
    min_radius: float = 100.0

    # Fallback behavior
    fallback_enabled: bool = True
    fallback_cx: float = 256.0
    fallback_cy: float = 256.0
    fallback_radius: float = 220.0

    # Mask handling
    mask_threshold: int = 127

    # Polar unwrap dimensions
    warp_radius_bins: int = 256
    warp_angle_bins: int = 1024

    # Final ribbon dimensions
    ribbon_width: int = 512
    ribbon_height: int = 64

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Only pass known fields to avoid errors from extra keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── S3 / Image Cache ─────────────────────────────────────────────────


def get_s3_client():
    if not SPACES_ACCESS_KEY or not SPACES_SECRET_KEY:
        print("ERROR: Spaces credentials not set (SPACES_ACCESS_KEY / AWS_ACCESS_KEY_ID)")
        sys.exit(1)
    return boto3.session.Session().client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=SPACES_ACCESS_KEY,
        aws_secret_access_key=SPACES_SECRET_KEY,
    )


def s3_key_from_url(url):
    """Extract the S3 key from a full Spaces URL."""
    prefix = f"{SPACES_ENDPOINT}/{SPACES_BUCKET}/"
    if url.startswith(prefix):
        return url[len(prefix):]
    return url


def _cache_path(s3_key, asset_type):
    """Build local cache path: image_cache/{asset_type}/{s3_key}."""
    return IMAGE_CACHE_DIR / asset_type / s3_key


def download_image_cached(s3, url, asset_type, flags=cv2.IMREAD_COLOR):
    """Download an image, using local cache to avoid redundant S3 fetches.

    Returns (ndarray, cache_hit: bool).
    """
    s3_key = s3_key_from_url(url)
    cached = _cache_path(s3_key, asset_type)

    if cached.exists():
        img = cv2.imread(str(cached), flags)
        if img is not None:
            return img, True

    # Download from S3
    resp = s3.get_object(Bucket=SPACES_BUCKET, Key=s3_key)
    data = resp["Body"].read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)

    if img is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename to avoid partial/corrupt cache files
        tmp = cached.with_suffix(cached.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            tmp.rename(cached)
        except OSError:
            tmp.unlink(missing_ok=True)

    return img, False


def download_image_raw(s3, key, flags=cv2.IMREAD_COLOR):
    """Download an image from S3 without caching (backward compat)."""
    resp = s3.get_object(Bucket=SPACES_BUCKET, Key=key)
    data = resp["Body"].read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


# ── Center / Radius Estimation ───────────────────────────────────────


def _find_largest_contour(binary_mask):
    """Find the largest contour from a binary mask. Returns contour or None."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def estimate_center_radius(binary_mask, method="bbox"):
    """Estimate coin center and radius from binary mask.

    Methods:
        bbox:                center of bounding rect; radius = max(w, h) / 2
        moments:             centroid from cv2.moments; radius = sqrt(area / pi)
        min_enclosing_circle: cv2.minEnclosingCircle on largest contour

    Returns (cx, cy, radius) or None if no contour found.
    """
    contour = _find_largest_contour(binary_mask)
    if contour is None:
        return None

    if method == "bbox":
        x, y, w, h = cv2.boundingRect(contour)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        r = int(max(w, h) / 2)
        return cx, cy, r

    if method == "moments":
        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        area = cv2.contourArea(contour)
        r = int(np.sqrt(area / np.pi))
        return cx, cy, r

    if method == "min_enclosing_circle":
        (cx, cy), r = cv2.minEnclosingCircle(contour)
        return int(cx), int(cy), int(r)

    raise ValueError(f"Unknown center_method: {method!r}")


# ── Core Unwrap ──────────────────────────────────────────────────────


def unwrap_coin(img, mask, cfg: UnwrapConfig):
    """Full unwrap pipeline: mask, find center, polar unwrap, crop annulus.

    Returns:
        ribbon: float32 ndarray (ribbon_height x ribbon_width), 0-1 scale
        metadata: dict with all geometry/debug/fallback fields
    """
    alpha = mask[:, :, 3]
    img_h, img_w = img.shape[:2]
    img_cx, img_cy = img_w // 2, img_h // 2

    # 1. Binary mask
    _, binary_mask = cv2.threshold(alpha, cfg.mask_threshold, 255, cv2.THRESH_BINARY)
    binary_norm = (binary_mask / 255).astype(np.uint8)
    if binary_norm.shape[:2] != img.shape[:2]:
        binary_norm = cv2.resize(binary_norm, (img_w, img_h),
                                 interpolation=cv2.INTER_NEAREST)
    clean_gray = img * binary_norm

    # Mask area fraction
    mask_area_fraction = float(binary_norm.sum()) / (img_h * img_w)

    # 2. Center / radius estimation
    used_fallback = False
    fallback_reason = None

    result = estimate_center_radius(binary_mask, cfg.center_method)
    if result is None:
        if cfg.fallback_enabled:
            cx, cy, r = int(cfg.fallback_cx), int(cfg.fallback_cy), int(cfg.fallback_radius)
            used_fallback = True
            fallback_reason = "no_contour"
        else:
            return None, {
                "status": "failed",
                "failure_reason": "no_contour",
                "used_fallback": False,
                "fallback_reason": None,
            }
    else:
        cx, cy, r = result

        # Sanity checks
        center_dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        if center_dist > cfg.max_center_offset:
            if cfg.fallback_enabled:
                cx, cy, r = int(cfg.fallback_cx), int(cfg.fallback_cy), int(cfg.fallback_radius)
                used_fallback = True
                fallback_reason = "center_out_of_bounds"
            else:
                return None, {
                    "status": "failed",
                    "failure_reason": "invalid_geometry",
                    "used_fallback": False,
                    "fallback_reason": None,
                }
        elif r < cfg.min_radius:
            if cfg.fallback_enabled:
                cx, cy, r = int(cfg.fallback_cx), int(cfg.fallback_cy), int(cfg.fallback_radius)
                used_fallback = True
                fallback_reason = "radius_too_small"
            else:
                return None, {
                    "status": "failed",
                    "failure_reason": "invalid_geometry",
                    "used_fallback": False,
                    "fallback_reason": None,
                }

    center_offset = float(np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2))

    # 3. Polar unwrap
    raw = cv2.warpPolar(
        clean_gray,
        dsize=(cfg.warp_radius_bins, cfg.warp_angle_bins),
        center=(cx, cy),
        maxRadius=r,
        flags=cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )

    # 4. Transpose, crop annulus, resize
    ribbon = raw.T
    inner_row = int(cfg.warp_radius_bins * cfg.inner_r_ratio)
    outer_row = int(cfg.warp_radius_bins * cfg.outer_r_ratio)
    cropped = ribbon[inner_row:outer_row, :]

    if cropped.size == 0:
        return None, {
            "status": "failed",
            "failure_reason": "invalid_dimensions",
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
        }

    final = cv2.resize(cropped, (cfg.ribbon_width, cfg.ribbon_height))
    final = final.astype(np.float32) / 255.0

    metadata = {
        "status": "success",
        "failure_reason": None,
        "cx": cx,
        "cy": cy,
        "radius": r,
        "center_offset": round(center_offset, 2),
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "mask_area_fraction": round(mask_area_fraction, 4),
    }

    return final, metadata


def process_coin(s3, row, cfg: UnwrapConfig, ribbon_dir: Path, save_preview=False):
    """Download, unwrap, and save one coin.

    Returns (coin_id, ribbon_path_or_None, metadata_dict).
    """
    coin_id = row["coin_id"]
    highres_url = row.get("highres_url", "")
    transparent_url = row.get("transparent_url", "")

    if not highres_url or not transparent_url:
        return coin_id, None, {
            "status": "failed",
            "failure_reason": "download_error",
            "used_fallback": False,
            "fallback_reason": None,
            "cache_hit_highres": False,
            "cache_hit_transparent": False,
        }

    try:
        img, hit_h = download_image_cached(s3, highres_url, "highres", cv2.IMREAD_GRAYSCALE)
        if img is None:
            return coin_id, None, {
                "status": "failed",
                "failure_reason": "download_error",
                "used_fallback": False,
                "fallback_reason": None,
                "cache_hit_highres": hit_h,
                "cache_hit_transparent": False,
            }

        mask, hit_t = download_image_cached(s3, transparent_url, "transparent", cv2.IMREAD_UNCHANGED)
        if mask is None or len(mask.shape) < 3 or mask.shape[2] < 4:
            return coin_id, None, {
                "status": "failed",
                "failure_reason": "download_error",
                "used_fallback": False,
                "fallback_reason": None,
                "cache_hit_highres": hit_h,
                "cache_hit_transparent": hit_t,
            }

        ribbon, meta = unwrap_coin(img, mask, cfg)
        meta["cache_hit_highres"] = hit_h
        meta["cache_hit_transparent"] = hit_t

        if ribbon is None:
            return coin_id, None, meta

        # Save ribbon
        ribbon_dir.mkdir(parents=True, exist_ok=True)
        ribbon_path = ribbon_dir / f"{coin_id}.npy"
        np.save(ribbon_path, ribbon)

        # Optional preview
        preview_path = None
        if save_preview:
            preview_dir = ribbon_dir.parent / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"{coin_id}.png"
            preview_img = (ribbon * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(preview_path), preview_img)
            preview_path = str(preview_path)

        return coin_id, str(ribbon_path), meta

    except Exception as e:
        return coin_id, None, {
            "status": "failed",
            "failure_reason": f"unwrap_exception: {e}",
            "used_fallback": False,
            "fallback_reason": None,
            "cache_hit_highres": False,
            "cache_hit_transparent": False,
        }
