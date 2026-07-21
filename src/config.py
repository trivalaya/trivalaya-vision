"""
Configuration Module for Coin Detection Pipeline
All magic numbers and thresholds consolidated here for tuning and maintainability.
"""

# === IMAGE PREPROCESSING ===
MAX_DIMENSION = 3200  # Resize limit to prevent memory issues

# === LAYER 1: GEOMETRY DETECTION ===
class Layer1Config:
    """Structural salience detection parameters"""
    
    # Sensitivity Presets
    class Standard:
        EDGE_SUPPORT_MIN = 0.05  # Minimum edge overlap to consider contour valid
        CANNY_SIGMA = 0.33       # Canny threshold calculation multiplier
        MIN_AREA_PX = 300        # Minimum contour area in pixels
        
    class High:
        EDGE_SUPPORT_MIN = 0.02
        CANNY_SIGMA = 0.60
        MIN_AREA_PX = 200
    
    # Morphological Operations
    # MORPH_CLOSE kernel is scale-relative: k = int(max(h, w) * FRAC), clamped
    # to [MIN, MAX], then bumped odd.  A fixed 7x7 has ~12px of bridging power
    # regardless of input size, which welds correctly-separated coins on small
    # auction formats (500px) while doing real mask repair on 3000px plates.
    # See specs/two_coin_weld_morph_close.md.
    CLOSE_KERNEL_FRAC = 1 / 400  # 7px across the 2400-3199px band (cng: 3000)
    CLOSE_KERNEL_MIN = 3         # below 3 the close is a no-op
    CLOSE_KERNEL_MAX = 21        # bounds cost / over-welding on 4000px+ plates
    CLOSE_ITERATIONS = 2         # Number of closing operations

    # Per-house kernel overrides.  Populated 2026-07-21 -- see below.
    #
    # Keys are auction_house as stored in auction_data.auction_house
    # (matched case-insensitively).  Values may override any of
    # frac / min / max; unspecified keys fall back to the globals above.
    # Precedence: TRIVALAYA_CLOSE_KERNEL_FRAC > per-house > global.
    #
    # Entries are only legitimate with measurement behind them.  Guessing a
    # constant here would repeat the exact failure this spec spent three
    # revisions correcting: v1's round() gave the 42,080-coin cng house more
    # bridging than it had today, silently, because a constant was reasoned
    # about rather than computed.  Each entry below cites its measurement.
    #
    # Every entry pins min == max, which fixes k outright and makes it
    # independent of both image width and CLOSE_KERNEL_FRAC.  That is
    # deliberate: what §4.5/§4.6 measured is a *kernel*, not a fraction, and
    # encoding the measured quantity directly is what keeps a later change to
    # the global frac from silently moving a house off its measured operating
    # point.  Widths cited are from the raws on disk on 2026-07-21.
    #
    #   cng_feature -- k=3.  §4.5, n=200 (Auction 91), the `auto` arm:
    #     Hough 97.5% -> 1.0%, weld signature 90.5% -> 1.0%, true
    #     fragmentation 0.5%, ndets 2 on all 200 in both arms.  Otsu had
    #     already separated the coins in 200/200 lots; the 7x7 close was
    #     destroying a correct segmentation.  Corpus is 1231/1233 lots at
    #     500px (outliers 714 and 1278), and the global formula yields k=3
    #     across all of them -- so this entry is identical to `auto` on 100%
    #     of the observed corpus and exists to say so explicitly.
    #
    #   leu -- k=5.  §4.6, n=200 (sale_id 75), the `0.0042` sweep arm, which
    #     yielded k=5 on all 200 lots (widths 1047-1200).  k=5 removes 86% of
    #     leu's Hough rate at zero fragmentation cost (0.5% true-split, 0.0%
    #     in the 2-blob cell -- identical to k=7 on both).  **The global
    #     formula assigns leu k=3**, which buys the remaining 14% by tripling
    #     true fragmentation and introducing 2.0% splitting in the two-coin
    #     cell.  Fragmentation is a correctness risk (§7.2 -> §7.4 embedding
    #     drift across 256k coins); residual Hough is a cost issue.  This
    #     entry is the whole reason the table exists.
    #
    #   kuenker -- k=7.  **STATUS-QUO PIN, NOT A SWEEP RESULT.**  No A/B or
    #     kernel sweep has been run on kuenker; its evidence is §"Measured
    #     evidence"'s n=10 gap sample (~25px gaps, 0/10 welded at k=7) plus a
    #     0.7% corpus Hough rate -- enough to say the 7x7 close is not
    #     hurting kuenker, not enough to say k=7 is optimal.  It is pinned
    #     because enabling `auto` would otherwise move kuenker off today's
    #     behaviour unmeasured: the raws are 417-2000px (median 2000), so the
    #     global formula yields k=5 on 87% of lots and k=3 on the 330-lot
    #     small tail -- it never yields 7.  (§1's "kuenker spans 800-3000px,
    #     so it straddles k=3 through k=7" does not match the raws on disk;
    #     max observed is 2000.)  Remove this entry when kuenker is measured.
    CLOSE_KERNEL_BY_HOUSE: dict = {
        "cng_feature": {"min": 3, "max": 3},
        "leu": {"min": 5, "max": 5},
        "kuenker": {"min": 7, "max": 7},
    }
    # HAZARD, now that the table is non-empty: per-house min/max clamp an
    # explicit TRIVALAYA_CLOSE_KERNEL_FRAC as well as the global one.  That is
    # deliberate and tested (test_per_house_bounds_still_clamp_an_explicit_frac)
    # -- bounds are bounds, not the quantity under test -- but with min == max
    # pinned on all three houses it means **a future sweep arm is a no-op on
    # every house in this table**.  The k=3/5/7 sweep that produced leu's entry
    # would, re-run today, return k=5 for every arm and look like a null result.
    # A sweep must clear the table (or pass house=None) for the arm to mean
    # anything.  tools/two_coin_weld_ab.py carries the same warning.

    # Background Detection
    BRIGHT_BACKGROUND_THRESHOLD = 110   # Gray value to classify as light background
    VERY_BRIGHT_THRESHOLD = 200         # For aggressive foreground filtering
    
    # Contour Filtering
    MAX_AREA_RATIO = 0.98       # Maximum contour area as fraction of image
    BORDER_TOLERANCE_PX = 2     # Pixels from edge to consider "touching border"
    BORDER_TOUCH_MAX_AREA = 0.30  # Max area ratio for border-touching contours
    
    # Geometry Classification
    CIRCULARITY_THRESHOLD = 0.82  # Minimum circularity to label as "Circle"
    SOLIDITY_THRESHOLD = 0.88     # Maximum solidity for "Complex" shapes
    POLYGON_APPROXIMATION = 0.03  # Epsilon for polygon vertex reduction
    
    # === RIM RECOVERY TRIGGERS (Layer 1.5) ===
    CIRCULARITY_STRICT = 0.90     # OLD threshold - too strict for ancient coins
    CIRCULARITY_RELAXED = 0.65    # Triggers recovery; paired with area_ratio gate
                                  # (mask area / minEnclosingCircle area) < 0.85
                                  # to skip clean coins whose contour is slightly
                                  # bumpy but still fills its circle.
    MIN_AREA_RATIO = 0.20         # Trigger if object is small (was 0.40)
    RIM_GAP_THRESHOLD = 0.85      # Expected area completeness ratio
    MIN_SOLIDITY_FOR_RECOVERY = 0.85
    
    # NMS (Non-Maximum Suppression)
    NMS_IOU_THRESHOLD = 0.50      # IoU overlap threshold for duplicate detection
    NMS_CENTER_DISTANCE_RATIO = 0.10  # Max center distance as fraction of bbox
    MAX_DETECTIONS = 5            # Maximum objects to return per image


# Module-level (import-time) guards: _close_kernel_size applies `k |= 1` AFTER
# clamping, so an even MAX would let k exceed it by one.  Deliberately not in
# validate_config() -- that function is only called from this file's __main__
# block, so an assert there never runs in production.
assert Layer1Config.CLOSE_KERNEL_MAX % 2 == 1, \
    "CLOSE_KERNEL_MAX must be odd (k |= 1 runs after the clamp)"

def validate_close_kernel_overrides():
    """
    The MAX-is-odd invariant has to hold for every per-house override too, and
    a bad entry there is far easier to introduce than a bad global.  Called at
    import (below), so a malformed table fails fast rather than silently
    producing an out-of-range kernel on one house's images.

    A no-op while the table is empty -- which is exactly when it is cheapest
    to add, and it fires the moment someone populates it wrongly.
    """
    allowed = {"frac", "min", "max"}
    for house, ovr in Layer1Config.CLOSE_KERNEL_BY_HOUSE.items():
        assert house == house.strip().lower(), \
            f"CLOSE_KERNEL_BY_HOUSE keys must be lowercase/stripped: {house!r}"
        assert set(ovr) <= allowed, \
            f"CLOSE_KERNEL_BY_HOUSE[{house!r}]: unknown key(s) " \
            f"{sorted(set(ovr) - allowed)}"
        lo = ovr.get("min", Layer1Config.CLOSE_KERNEL_MIN)
        hi = ovr.get("max", Layer1Config.CLOSE_KERNEL_MAX)
        assert hi % 2 == 1, \
            f"CLOSE_KERNEL_BY_HOUSE[{house!r}]['max'] must be odd"
        assert lo >= 1, f"CLOSE_KERNEL_BY_HOUSE[{house!r}]['min'] must be >= 1"
        assert lo <= hi, f"CLOSE_KERNEL_BY_HOUSE[{house!r}]: min ({lo}) > max ({hi})"
        assert ovr.get("frac", Layer1Config.CLOSE_KERNEL_FRAC) > 0, \
            f"CLOSE_KERNEL_BY_HOUSE[{house!r}]['frac'] must be > 0"


validate_close_kernel_overrides()


# === LAYER 1.5: RIM RECOVERY ===
class RimRecoveryConfig:
    """Circle hypothesis and rim reconstruction parameters"""
    
    SEARCH_PAD_RATIO = 1.0        # Search area around seed (multiplier of bbox size)
    SEED_RADIUS_TOLERANCE = 0.9   # Min ratio of recovered rim to seed size
    
    # Hough Circle Parameters
    HOUGH_DP = 1.5                # Inverse accumulator resolution
    HOUGH_MIN_DIST = 1000         # Minimum distance between circle centers
    HOUGH_PARAM1 = 100            # Canny edge threshold for Hough
    HOUGH_PARAM2 = 30             # Accumulator threshold (lower = more circles)

    # Perf: rim-recovery HoughCircles runs on a copy of the ROI downscaled to
    # this longest side; the winning circle is scaled back to image coords.
    # Cost grows ~dim^4 x edge density, so the cap is the difference between
    # seconds and CPU-minutes on large inputs. 0 = uncapped (legacy full-res).
    # Runtime override: TRIVALAYA_RIM_HOUGH_CAP.
    HOUGH_ROI_CAP = 1280

    # Validation
    CENTER_ALIGNMENT_TOLERANCE = 0.8  # Max distance from seed center (as ratio of radius)
    EDGE_SUPPORT_MIN = 0.15       # Minimum edge confirmation (was 0.08 - too low!)
    EDGE_SUPPORT_FALLBACK = 0.12  # Lower threshold for poor quality images
    EDGE_RING_THICKNESS = 3       # Pixels for edge sampling ring
    
    # Fit Quality
    FIT_ERROR_EXCELLENT = 0.20    # Median error/radius for "Clean" classification
    FIT_ERROR_ACCEPTABLE = 0.50   # Maximum error for "Rough" classification


# === LAYER 2: CONTEXT ANALYSIS ===
class Layer2Config:
    """Scene understanding and container detection parameters"""
    
    # Self-Contained Object Detection (Shortcut)
    AREA_RATIO_LARGE = 0.15       # Objects >15% of image are "big"
    SOLIDITY_DISTINCT = 0.85      # Solidity threshold for "distinct" objects
    MIN_INTERNAL_CONTRAST = 20    # Sobel magnitude for "has content"
    
    # Ray Casting
    RAY_ANGLE_STEP_DEG = 5        # Angular resolution (72 rays at 5°)
    RAY_STEP_SIZE_PX = 4          # Distance between samples along ray
    RAY_DISTANCE_RATIO = 1.1      # Max ray length as fraction of image size
    
    # Edge Detection
    CANNY_LOW = 30
    CANNY_HIGH = 100
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Container Detection
    ENCLOSURE_CONFIDENCE_MIN = 0.35  # Minimum ray hit ratio to infer container
    MIN_WALL_POINTS = 10            # Minimum edge points for fitting
    MASK_SMALL_OBJECTS_RATIO = 0.10  # Only mask objects <10% of image
    
    # Classification Thresholds
    CIRCULARITY_ROUND = 0.60      # Minimum circularity for "Round_Artifact"


# === EXTRACTION PIPELINE ===
class ExtractionConfig:
    """File output and naming conventions"""
    
    CROP_MARGIN_RATIO = 0.05      # Margin around bounding box (5% of max dimension)
    
    # Domain-agnostic classification mapping
    GENERIC_CLASS_MAP = {
        "Circle": "circular_objects",
        "Polygon": "polygonal_objects",
        "Complex": "irregular_objects",
        "Fragment": "fragments",
        "Unknown": "review"
    }


# === COIN-SPECIFIC CONFIGURATION ===
# NOTE: These settings are for numismatic analysis
# When adding other domains (pottery, PCB, etc.), create separate config classes
class CoinConfig:
    """Numismatic-specific parameters and behaviors"""
    
    # Rim Recovery (should this be attempted?)
    ENABLE_RIM_RECOVERY = True
    
    # Scene Understanding
    ASSUME_TWO_OBJECTS_ARE_PAIR = True  # Infer obv/rev from 2-object scenes
    
    # Classification Mapping (user-friendly terms)
    CLASS_MAP = {
        "Round Object (Coin/Medallion)": "coins",
        "Round Object (Coin)": "coins",
        "Geometric Form": "fragments",
        "Fragment": "fragments",
        "Artifact (Isolated)": "artifacts",
        "Unknown": "review"
    }
    
    # Quality thresholds for "coin-like" objects
    MIN_CIRCULARITY = 0.65  # Minimum to consider as potential coin
    MIN_EDGE_QUALITY = 0.12  # Minimum edge support for coin detection


# === DIAGNOSTIC / DEBUG ===
class DiagnosticConfig:
    """Parameters for visualization and debugging tools"""
    
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    CANNY_SIGMA_MULTIPLIER = 0.33
    
    CORNER_SAMPLES = 4  # Number of corner pixels to sample for BG detection
    LIGHT_BG_THRESHOLD = 127  # Gray value threshold for inverting


# === VALIDATION ===
def validate_config():
    """Sanity checks for configuration consistency"""
    assert Layer1Config.CIRCULARITY_RELAXED < Layer1Config.CIRCULARITY_STRICT, \
        "Relaxed circularity must be lower than strict"
    assert RimRecoveryConfig.EDGE_SUPPORT_MIN > RimRecoveryConfig.EDGE_SUPPORT_FALLBACK, \
        "Primary edge support must exceed fallback"
    assert Layer1Config.NMS_IOU_THRESHOLD <= 1.0, "IoU must be <= 1.0"
    print("✓ Configuration validated")

if __name__ == "__main__":
    validate_config()