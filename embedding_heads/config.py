"""Shared config for embedding heads experiment."""

from pathlib import Path

EMBED_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = EMBED_DIR / "checkpoints"
OUTPUT_DIR = EMBED_DIR / "outputs"
DATA_DIR = EMBED_DIR / "data"

# Source data (pipeline prod)
PROD_DIR = Path("/root/trivalaya-pipeline/cluster_output_prod")
OBV_FEATURES_PATH = PROD_DIR / "obv_features_768.npy"
PROD_META_PATH = PROD_DIR / "cluster_results.enriched.csv"

# Legend CNN confusion subset (our 10 authorities)
CONFUSION_SUBSET_PATH = Path("/root/trivalaya-vision/legend_cnn/data/confusion_subset.csv")
SPLITS_PATH = Path("/root/trivalaya-vision/legend_cnn/data/splits_v1.json")

EMBED_DIM = 768
