# Two-Coin Resolver Integration Guide

## Overview

The two-coin resolver addresses the problem of auction-style images where obverse and reverse sides are shown side-by-side. Standard contour-based detection merges these into a single blob with low circularity; this module splits them into separate detections.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      IMAGE INPUT                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Initial Contour Detection                             │
│  - Otsu threshold → binary mask                                  │
│  - findContours → candidates                                     │
│  - Edge support + circularity filtering                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TRIGGER CHECK: Is this a merged two-coin blob?                 │
│  Uses TwoCoinResolver.should_trigger() for consistent behavior  │
│  Conditions (all must be true):                                  │
│  - Exactly 1 candidate                                           │
│  - Circularity < 0.60 (merged blob signature)                   │
│  - Area ratio > 0.25 (substantial size)                         │
│  - Aspect ratio > 1.4 (wide = side-by-side)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ Triggered                      │ Not triggered
              ▼                                ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  TWO-COIN RESOLVER          │    │  Continue with single       │
│                             │    │  candidate                  │
│  1. Hough circle detection  │    └─────────────────────────────┘
│     + pair validation       │
│     ↓                       │
│  2. Watershed fallback      │
│     (if Hough fails)        │
│     ↓                       │
│  3. needs_review            │
│     (if both fail)          │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: List of coin candidates with centers, radii, crops     │
│  - Each coin flagged as 'obverse' or 'reverse'                  │
│  - Synthetic circular contours (clamped to image bounds)        │
└─────────────────────────────────────────────────────────────────┘
```

## Files

1. **`two_coin_resolver.py`** - Core module with:
   - `TwoCoinResolver` class
   - `CoinPairConfig` dataclass for tuning (single source of truth for thresholds)
   - Hough detection + pair validation
   - Watershed fallback with proper marker logic
   - Edge density validation to prevent false positives

2. **`layer1_geometry_updated.py`** - Updated Layer 1 with integration point
   - Uses `resolver.should_trigger()` for consistent threshold behavior
   - Synthetic contours clamped to image bounds

## Integration Steps

### Option A: Replace layer1_geometry.py

Simply replace your existing `src/layer1_geometry.py` with `layer1_geometry_updated.py`.

### Option B: Add as separate module

1. Copy `two_coin_resolver.py` to `src/`
2. Add trigger check after NMS in your existing `layer1_geometry.py`:

```python
from src.two_coin_resolver import TwoCoinResolver

# After NMS, check for two-coin trigger
if len(candidates) == 1:
    resolver = TwoCoinResolver()
    should_trigger, _ = resolver.should_trigger(candidates, (h, w))
    
    if should_trigger:
        result = resolver.resolve(img, binary, gray)
        
        if result['status'] == 'split':
            # Convert to your candidate format
            candidates = convert_split_to_candidates(result['coins'])
```

## Tuning Parameters

All trigger thresholds are in `CoinPairConfig` (single source of truth):

### Trigger Conditions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRIGGER_CIRCULARITY_MAX` | 0.60 | Max circularity to trigger |
| `TRIGGER_AREA_RATIO_MIN` | 0.25 | Min blob area as fraction of image |
| `TRIGGER_ASPECT_RATIO_MIN` | 1.4 | Min width/height ratio |

### Hough Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HOUGH_DP` | 1.2 | Accumulator resolution |
| `HOUGH_PARAM1` | 100 | Canny high threshold |
| `HOUGH_PARAM2` | 30 | Accumulator threshold |
| `MIN_RADIUS_RATIO` | 0.10 | Min coin radius as fraction of image width |
| `MAX_RADIUS_RATIO` | 0.45 | Max coin radius |

### Pair Validation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y_ALIGNMENT_MAX` | 0.15 | Max y-difference as fraction of height |
| `RADIUS_RATIO_MIN` | 0.70 | Min ratio of smaller/larger radius |
| `CENTER_SEPARATION_MIN` | 1.0 | Min center distance as multiple of radius |
| `CENTER_SEPARATION_MAX` | 3.5 | Max center distance |
| `RIM_EDGE_DENSITY_MIN` | 0.08 | Min edge pixels in rim annulus |

## Watershed Implementation Notes

The watershed fallback uses proper marker logic:
- Tries multiple distance transform thresholds (0.25 to 0.60) to find one yielding exactly 2 foreground regions
- Falls back to local maxima detection if threshold sweep fails
- Marker preparation: `markers = labels + 1` first, then `markers[unknown == 255] = 0`
- This ensures unknown regions remain 0 for watershed to fill

Polarity convention: `binary_mask` has foreground=255 (coins), background=0

## Expected Failure Cases

Hough may fail on:
- Worn/weak rims with poor edge contrast
- Lighting gradients creating false arcs
- Non-coin circular shapes (holders, rings)
- Heavily clipped coins at image borders

Watershed may fail on:
- Touching coins with no gap in binary mask (distance transform has single peak)
- Uneven coin sizes
- Complex backgrounds

When both fail, the image is flagged `needs_review` with debug info.

## Test Results

On the 10 test images provided:
- **10/10 successfully split via Hough**
- Pair scores ranged from 0.86 to 0.98
- Watershed successfully handles 3/4 test cases when Hough is disabled

## Output Format

```python
{
    "layer": 1,
    "status": "success",
    "objects": [
        {
            "classification": {"label": "Circle", "confidence": 0.85},
            "geometry": {"area": ..., "circularity": 0.95, ...},
            "contour": np.array(...),  # Synthetic circle contour (clamped to bounds)
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "radius": r,
            "side": "obverse",  # or "reverse"
            "debug_data": {"from_two_coin_split": True, "split_method": "hough"}
        },
        {...}  # Second coin
    ],
    "two_coin_resolution": {
        "triggered": True,
        "status": "split",
        "method": "hough",
        "pair_score": 0.95
    }
}
```

## Status Values

- `TwoCoinResolver.resolve()` returns `status: 'split' | 'failed'`
- Top-level integration returns `status: 'split' | 'needs_review'` (more semantic for downstream)
