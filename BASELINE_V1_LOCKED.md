# Trivalaya Vision - Baseline v1 (Production)

**Date:** December 31, 2025
**Status:** Production-Ready ✅

## Performance Metrics
- **Detection Rate:** ~90% (up from ~40%)
- **Coins Extracted:** 404
- **Fragments:** 1
- **Artifacts:** 6
- **Obv/Rev Splitting:** ~380/404 successful (94%)
- **Known Merged Pairs:** ~20-30 (aspect ratio ~2.0)

## Architecture
### Core Pipeline (Domain-Agnostic)
- Layer 1: Geometric detection + validation
- Layer 2: Context analysis + ray casting
- NMS: IoU (0.50) + Containment (0.40) + Proximity

### Coin-Specific Enhancements
- Geometric circle fitting (Kasa's method)
- Arc coverage confidence weighting
- Annulus edge support (adaptive band width)
- Semantic salvage (coin_likelihood > 0.65 override)

### Configuration
- Morphological kernel: (7, 7) ellipse, 2 iterations
- Edge support threshold: 0.05 (relaxed from 0.10)
- Circularity relaxed trigger: 0.75 (from 0.90)
- NMS containment: 0.40 (from 0.50)

## Known Limitations (Acceptable)
1. **Merged obv/rev pairs (~20-30 images)**
   - Aspect ratio ~2.0 indicates side-by-side coins
   - Morphological closing bridges gap
   - **Status:** Acceptable - clean extractions, manual split trivial
   
2. **Some touching coins merge**
   - Expected behavior when coins physically overlap
   - Conservative choice prevents fragment noise

3. **Very low contrast rims may be missed**
   - Edge cases with extreme lighting/corrosion

## Technical Debt Acknowledged
- Rim recovery in Layer 1 (should be plugin)
- Clustering code removed (wasn't stable)
- Morphological kernel fixed (not adaptive)

## Next Refinement (Planned - Not Urgent)
### Aspect Ratio Split Hint (Coin Plugin)
**Trigger:** `domain=coin AND detections=1 AND 1.7 < aspect < 2.3`

**Algorithm:**
1. Attempt secondary split (vertical projection / two-circle fit)
2. Validate both halves (coin_likelihood, size, symmetry)
3. Accept only if validation passes
4. Otherwise keep merged detection

**Safety:** Late-stage, reversible, opt-in, validated

**Location:** `src/plugins/coin_analyzer.py` (NOT in Layer 1)

## Success Criteria Met ✅
- [x] 90%+ detection rate
- [x] Fragment explosion eliminated
- [x] Explainable decisions
- [x] Clean metadata output
- [x] Production-ready stability
- [x] Acceptable false positive/negative rates

---
**Signed off for production use - December 31, 2025**
**Further optimization deferred pending real-world feedback.**
