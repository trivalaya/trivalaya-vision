"""Shared constants for the legend stylometry PoC."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
LEGEND_CNN_DIR = Path(__file__).resolve().parent
DATA_DIR = LEGEND_CNN_DIR / "data"
RIBBON_DIR = DATA_DIR / "ribbons"
CHECKPOINT_DIR = LEGEND_CNN_DIR / "checkpoints"
OUTPUT_DIR = LEGEND_CNN_DIR / "outputs"

CONFUSION_SUBSET_CSV = DATA_DIR / "confusion_subset.csv"

# ── Eval harness ──────────────────────────────────────────────────────
EVAL_SUBSET_CSV = DATA_DIR / "eval_subset_v1.csv"
IMAGE_CACHE_DIR = DATA_DIR / "image_cache"
RUNS_DIR = LEGEND_CNN_DIR / "runs"
CONFIGS_DIR = LEGEND_CNN_DIR / "configs"
NORM_RANGES_FILE = DATA_DIR / "norm_ranges.json"

# ── MySQL (matches trivalaya-pipeline pattern) ─────────────────────────
DB_CONFIG = {
    "host": os.environ.get("TRIVALAYA_DB_HOST", "127.0.0.1"),
    "user": os.environ.get("TRIVALAYA_DB_USER", "auction_user"),
    "password": os.environ.get("TRIVALAYA_DB_PASSWORD", os.environ.get("MYSQL_PASSWORD", "")),
    "database": os.environ.get("TRIVALAYA_DB_NAME", "auction_data"),
}

# ── DO Spaces (boto3 / S3-compatible) ──────────────────────────────────
SPACES_REGION = os.environ.get("SPACES_REGION", "sfo3")
SPACES_BUCKET = os.environ.get("SPACES_BUCKET", "trivalaya-data")
SPACES_ENDPOINT = f"https://{SPACES_REGION}.digitaloceanspaces.com"
SPACES_ACCESS_KEY = os.environ.get("SPACES_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", ""))
SPACES_SECRET_KEY = os.environ.get("SPACES_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))

# ── Subset selection ───────────────────────────────────────────────────
MIN_AUTHORITY_IMAGES = 200
TARGET_AUTHORITY_COUNT = (5, 10)       # aim for 5-10 authorities
TARGET_TOTAL_IMAGES = 10_000

# ── Unwrap (promoted_v1: 2026-04-10) ──────────────────────────────────
INNER_R_RATIO = 0.68
OUTER_R_RATIO = 0.92
RIBBON_HEIGHT = 64
RIBBON_WIDTH = 512
WARP_RADIUS_BINS = 256
WARP_ANGLE_BINS = 1024
MIN_CONTENT_FRACTION = 0.15     # reject ribbons with < 15% non-black pixels
