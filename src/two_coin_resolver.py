"""
Two-Coin Resolver Module

Detects and splits auction-style two-coin images (obverse/reverse side-by-side).
Uses a fallback chain: Hough circles → Watershed → needs_review

This runs as "Layer 0.5" - before full structural analysis, to restore 
single-coin assumptions for downstream processing.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoinPairConfig:
    """Configuration for two-coin detection - tune these for your data"""
    
    # Hough circle parameters
    HOUGH_DP: float = 1.2
    HOUGH_PARAM1: int = 100  # Canny high threshold
    HOUGH_PARAM2: int = 30   # Accumulator threshold (lower = more circles)
    
    # Pair validation thresholds
    Y_ALIGNMENT_MAX: float = 0.15      # Max y-difference as fraction of image height
    RADIUS_RATIO_MIN: float = 0.70     # Min ratio of smaller/larger radius
    RADIUS_RATIO_MAX: float = 1.30     # Max ratio (inverse of min)
    CENTER_SEPARATION_MIN: float = 1.0  # Min center distance as multiple of avg radius
    CENTER_SEPARATION_MAX: float = 3.5  # Max center distance as multiple of avg radius
    
    # Size bounds (as fraction of image dimension)
    MIN_RADIUS_RATIO: float = 0.10     # Coin radius at least 10% of image width
    MAX_RADIUS_RATIO: float = 0.45     # Coin radius at most 45% of image width
    
    # Edge density validation
    RIM_EDGE_DENSITY_MIN: float = 0.08  # Min edge pixels in rim annulus
    
    # Trigger conditions (when to attempt two-coin resolution)
    TRIGGER_CIRCULARITY_MAX: float = 0.60  # Single blob with circ below this
    TRIGGER_AREA_RATIO_MIN: float = 0.25   # Blob must be at least 25% of image
    TRIGGER_ASPECT_RATIO_MIN: float = 1.4  # Blob width/height suggests two coins
    
    # Watershed parameters
    WATERSHED_DIST_THRESHOLD: float = 0.4  # Distance transform threshold for markers


class TwoCoinResolver:
    """
    Detects and splits two-coin auction images.
    
    Usage:
        resolver = TwoCoinResolver()
        result = resolver.resolve(image, binary_mask, trigger_contour)
        
        if result['status'] == 'split':
            coin1_crop = result['coins'][0]['crop']
            coin2_crop = result['coins'][1]['crop']
    """
    
    def __init__(self, config: Optional[CoinPairConfig] = None):
        self.config = config or CoinPairConfig()
    
    def should_trigger(self, candidates: List[Dict], image_shape: Tuple[int, int]) -> Tuple[bool, Optional[Dict]]:
        """
        Check if two-coin resolution should be attempted.
        
        Trigger conditions (ALL must be true):
        1. Exactly one candidate found
        2. Low circularity (merged blob signature)
        3. Large area (not a fragment)
        4. Wide aspect ratio (suggests side-by-side coins)
        
        Returns:
            (should_trigger, trigger_candidate)
        """
        h, w = image_shape
        cfg = self.config
        
        if len(candidates) != 1:
            return False, None
        
        cand = candidates[0]
        geom = cand['geometry']
        bbox = cand['bbox']
        
        circularity = geom['circularity']
        area_ratio = geom['area'] / (h * w)
        bbox_aspect = bbox[2] / bbox[3] if bbox[3] > 0 else 1.0  # width/height
        
        # Must be substantial size (not a fragment)
        if area_ratio < cfg.TRIGGER_AREA_RATIO_MIN:
            logger.debug(f"Two-coin trigger rejected: area_ratio={area_ratio:.2f} < {cfg.TRIGGER_AREA_RATIO_MIN}")
            return False, None
        
        # Require BOTH low circularity AND wide aspect ratio
        # This matches the "merged blob + side-by-side" pattern
        is_low_circularity = circularity < cfg.TRIGGER_CIRCULARITY_MAX
        is_wide_aspect = bbox_aspect > cfg.TRIGGER_ASPECT_RATIO_MIN
        
        if is_low_circularity and is_wide_aspect:
            logger.info(f"Two-coin resolver triggered: circ={circularity:.2f}, aspect={bbox_aspect:.2f}")
            return True, cand
        
        return False, None
    
    def resolve(self, 
                image: np.ndarray, 
                binary_mask: np.ndarray,
                gray: np.ndarray) -> Dict:
        """
        Attempt to split a two-coin image.
        
        Fallback chain:
        1. Hough circle detection with pair validation
        2. Watershed segmentation on binary mask
        3. Return needs_review if both fail
        
        Returns:
            {
                'status': 'split' | 'single' | 'needs_review',
                'method': 'hough' | 'watershed' | None,
                'coins': [{'crop': ndarray, 'bbox': tuple, 'center': tuple, 'radius': float}, ...],
                'debug': {...}
            }
        """
        h, w = image.shape[:2]
        
        # === ATTEMPT 1: Hough Circle Detection ===
        hough_result = self._try_hough_detection(image, gray, (h, w))
        
        if hough_result['status'] == 'split':
            logger.info(f"Two-coin split successful via Hough (score={hough_result['debug'].get('pair_score', 0):.2f})")
            return hough_result
        
        # === ATTEMPT 2: Watershed Segmentation ===
        watershed_result = self._try_watershed_split(image, binary_mask, (h, w))
        
        if watershed_result['status'] == 'split':
            logger.info("Two-coin split successful via Watershed")
            return watershed_result
        
        # === FALLBACK: Mark for review ===
        logger.warning("Two-coin resolution failed - marking needs_review")
        return {
            'status': 'needs_review',
            'method': None,
            'coins': [],
            'debug': {
                'hough_reason': hough_result['debug'].get('failure_reason', 'unknown'),
                'watershed_reason': watershed_result['debug'].get('failure_reason', 'unknown')
            }
        }
    
    def _try_hough_detection(self, 
                              image: np.ndarray, 
                              gray: np.ndarray,
                              shape: Tuple[int, int]) -> Dict:
        """
        Detect coin pair using Hough circles with strict validation.
        """
        h, w = shape
        cfg = self.config
        
        # Prepare image for Hough
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate radius bounds
        min_radius = int(w * cfg.MIN_RADIUS_RATIO)
        max_radius = int(w * cfg.MAX_RADIUS_RATIO)
        min_dist = int(w * 0.25)  # Coins should be at least 1/4 image apart
        
        # Run Hough circle detection
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=cfg.HOUGH_DP,
            minDist=min_dist,
            param1=cfg.HOUGH_PARAM1,
            param2=cfg.HOUGH_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is None or len(circles[0]) < 2:
            return {
                'status': 'failed',
                'method': 'hough',
                'coins': [],
                'debug': {'failure_reason': 'insufficient_circles', 'circles_found': 0 if circles is None else len(circles[0])}
            }
        
        circles = circles[0]
        
        # Find best valid pair
        best_pair = None
        best_score = 0
        rejection_reasons = []
        
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                c1, c2 = circles[i], circles[j]
                is_valid, score, reason = self._validate_coin_pair(c1, c2, image, gray, (h, w))
                
                if is_valid and score > best_score:
                    best_score = score
                    best_pair = (c1, c2)
                elif not is_valid:
                    rejection_reasons.append(reason)
        
        if best_pair is None:
            return {
                'status': 'failed',
                'method': 'hough',
                'coins': [],
                'debug': {
                    'failure_reason': 'no_valid_pair',
                    'circles_found': len(circles),
                    'rejection_reasons': list(set(rejection_reasons))[:5]
                }
            }
        
        # Sort left-to-right (obverse typically on left)
        c1, c2 = best_pair
        if c1[0] > c2[0]:
            c1, c2 = c2, c1
        
        # Extract crops
        coins = []
        for idx, (cx, cy, r) in enumerate([c1, c2]):
            crop, bbox = self._extract_coin_crop(image, cx, cy, r, (h, w))
            coins.append({
                'crop': crop,
                'bbox': bbox,
                'center': (float(cx), float(cy)),
                'radius': float(r),
                'side': 'obverse' if idx == 0 else 'reverse'
            })
        
        return {
            'status': 'split',
            'method': 'hough',
            'coins': coins,
            'debug': {
                'pair_score': best_score,
                'circles_evaluated': len(circles)
            }
        }
    
    def _validate_coin_pair(self,
                            c1: np.ndarray,
                            c2: np.ndarray,
                            image: np.ndarray,
                            gray: np.ndarray,
                            shape: Tuple[int, int]) -> Tuple[bool, float, str]:
        """
        Validate that two circles form a legitimate coin pair.
        
        Returns:
            (is_valid, score, rejection_reason)
        """
        h, w = shape
        cfg = self.config
        
        x1, y1, r1 = float(c1[0]), float(c1[1]), float(c1[2])
        x2, y2, r2 = float(c2[0]), float(c2[1]), float(c2[2])
        
        # 1. Horizontal arrangement: one in left half, one in right half
        if not ((x1 < w/2 and x2 > w/2) or (x2 < w/2 and x1 > w/2)):
            return False, 0, "not_horizontal_arrangement"
        
        # 2. Y-alignment: centers should be roughly at same height
        y_diff = abs(y1 - y2) / h
        if y_diff > cfg.Y_ALIGNMENT_MAX:
            return False, 0, f"y_misaligned_{y_diff:.2f}"
        
        # 3. Similar radii (same coin, different sides)
        radius_ratio = min(r1, r2) / max(r1, r2)
        if radius_ratio < cfg.RADIUS_RATIO_MIN:
            return False, 0, f"radius_mismatch_{radius_ratio:.2f}"
        
        # 4. Appropriate separation (not overlapping, not too far apart)
        center_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        avg_radius = (r1 + r2) / 2
        separation_ratio = center_dist / avg_radius
        
        if separation_ratio < cfg.CENTER_SEPARATION_MIN:
            return False, 0, f"overlapping_{separation_ratio:.2f}"
        if separation_ratio > cfg.CENTER_SEPARATION_MAX:
            return False, 0, f"too_far_apart_{separation_ratio:.2f}"
        
        # 5. Circles mostly within image bounds
        for x, y, r in [(x1, y1, r1), (x2, y2, r2)]:
            margin = r * 0.15  # Allow 15% clipping
            if x - r < -margin or x + r > w + margin:
                return False, 0, "x_out_of_bounds"
            if y - r < -margin or y + r > h + margin:
                return False, 0, "y_out_of_bounds"
        
        # 6. Edge density validation (optional but recommended)
        # Check that each circle has actual edge evidence at its rim
        edge_density_ok = True
        for x, y, r in [(x1, y1, r1), (x2, y2, r2)]:
            density = self._compute_rim_edge_density(gray, x, y, r, (h, w))
            if density < cfg.RIM_EDGE_DENSITY_MIN:
                edge_density_ok = False
                break
        
        if not edge_density_ok:
            return False, 0, "weak_rim_edges"
        
        # === SCORING ===
        # Higher score = more "ideal" pair
        
        # Prefer similar radii
        radius_score = radius_ratio
        
        # Prefer vertically centered
        avg_y = (y1 + y2) / 2
        y_centering = 1 - abs(avg_y - h/2) / (h/2)
        
        # Prefer horizontally symmetric
        avg_x = (x1 + x2) / 2
        x_symmetry = 1 - abs(avg_x - w/2) / (w/2)
        
        # Prefer good y-alignment
        y_alignment_score = 1 - (y_diff / cfg.Y_ALIGNMENT_MAX)
        
        score = (
            radius_score * 0.30 +
            y_centering * 0.25 +
            x_symmetry * 0.20 +
            y_alignment_score * 0.25
        )
        
        return True, score, "valid"
    
    def _compute_rim_edge_density(self,
                                   gray: np.ndarray,
                                   cx: float, cy: float, r: float,
                                   shape: Tuple[int, int],
                                   band_width: int = 5) -> float:
        """
        Compute edge density in an annulus around the circle rim.
        This validates that Hough isn't hallucinating circles from noise.
        """
        h, w = shape
        
        # Create annulus mask
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        
        cv2.circle(outer_mask, (int(cx), int(cy)), int(r + band_width), 255, -1)
        cv2.circle(inner_mask, (int(cx), int(cy)), max(1, int(r - band_width)), 255, -1)
        
        annulus = cv2.subtract(outer_mask, inner_mask)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute density
        annulus_pixels = cv2.countNonZero(annulus)
        if annulus_pixels == 0:
            return 0.0
        
        overlap = cv2.bitwise_and(annulus, edges)
        edge_pixels = cv2.countNonZero(overlap)
        
        return edge_pixels / annulus_pixels
    
    def _try_watershed_split(self,
                              image: np.ndarray,
                              binary_mask: np.ndarray,
                              shape: Tuple[int, int]) -> Dict:
        """
        Attempt to split merged blob using distance transform + watershed.
        
        Strategy: Try multiple distance thresholds to find one that yields
        exactly 2 foreground regions (the two coins).
        
        Polarity convention: binary_mask has foreground=255 (coins), background=0
        """
        h, w = shape
        
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        
        if max_dist < 10:  # Too small to be meaningful
            return {
                'status': 'failed',
                'method': 'watershed',
                'coins': [],
                'debug': {'failure_reason': 'dist_transform_too_small'}
            }
        
        # Try multiple thresholds to find one that gives exactly 3 components
        # (background + 2 coins)
        threshold_candidates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
        best_labels = None
        best_sure_fg = None
        best_threshold = None
        
        for thresh_ratio in threshold_candidates:
            _, sure_fg = cv2.threshold(dist_transform, 
                                        thresh_ratio * max_dist,
                                        255, cv2.THRESH_BINARY)
            sure_fg = np.uint8(sure_fg)
            
            num_labels, labels = cv2.connectedComponents(sure_fg)
            
            if num_labels == 3:  # background + 2 coins - perfect!
                best_labels = labels
                best_sure_fg = sure_fg
                best_threshold = thresh_ratio
                break
        
        if best_labels is None:
            # Try local maxima approach as last resort
            try:
                from scipy import ndimage
                
                # Find local maxima with a smaller window
                local_max = ndimage.maximum_filter(dist_transform, size=max(15, int(w * 0.1)))
                local_max_mask = (dist_transform == local_max) & (dist_transform > 0.25 * max_dist)
                local_max_mask = np.uint8(local_max_mask) * 255
                
                # Clean up with morphology
                kernel = np.ones((5, 5), np.uint8)
                local_max_mask = cv2.dilate(local_max_mask, kernel, iterations=2)
                
                num_labels, labels = cv2.connectedComponents(local_max_mask)
                
                if num_labels == 3:
                    best_labels = labels
                    best_sure_fg = local_max_mask  # Use the cleaned local maxima mask
                    best_threshold = "local_maxima"
            except ImportError:
                pass  # scipy not available, skip local maxima approach
        
        if best_labels is None or best_sure_fg is None:
            return {
                'status': 'failed',
                'method': 'watershed',
                'coins': [],
                'debug': {
                    'failure_reason': 'no_threshold_yields_2_regions',
                    'thresholds_tried': len(threshold_candidates)
                }
            }
        
        # Use the consistent sure_fg from whichever method succeeded
        labels = best_labels
        sure_fg = best_sure_fg
        
        # Compute expanded foreground and unknown region
        # expanded_fg = dilated coin region (foreground expanded outward)
        # unknown = expanded_fg - sure_fg (the uncertain border zone watershed will fill)
        kernel = np.ones((3, 3), np.uint8)
        expanded_fg = cv2.dilate(binary_mask, kernel, iterations=3)
        unknown = cv2.subtract(expanded_fg, sure_fg)
        
        # Prepare markers for watershed
        # Key: markers must have 0 for unknown regions (watershed will fill these)
        # Background = 1, coin regions = 2, 3, etc.
        markers = labels + 1  # Shift so background becomes 1, coins become 2 and 3
        markers[unknown == 255] = 0  # Mark unknown regions AFTER incrementing
        # Note: we do NOT set binary_mask==0 to 1 here, as that would overwrite unknown=0 pixels
        
        # Apply watershed
        img_color = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers_ws = cv2.watershed(img_color, markers.astype(np.int32))
        
        # Extract the two coin regions
        coins = []
        for label_id in [2, 3]:  # The two coin markers
            mask = np.uint8(markers_ws == label_id) * 255
            
            if cv2.countNonZero(mask) < 100:
                continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # Get largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(c)
            
            # Compute approximate center and radius
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + cw/2, y + ch/2
            
            radius = (cw + ch) / 4  # Approximate
            
            # Extract crop
            margin = int(max(cw, ch) * 0.05)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + cw + margin)
            y2 = min(h, y + ch + margin)
            
            crop = image[y1:y2, x1:x2].copy()
            
            coins.append({
                'crop': crop,
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'center': (cx, cy),
                'radius': radius,
                'side': None  # Will be assigned after sorting
            })
        
        if len(coins) != 2:
            return {
                'status': 'failed',
                'method': 'watershed',
                'coins': [],
                'debug': {'failure_reason': f'extracted_{len(coins)}_regions'}
            }
        
        # Sort left-to-right and assign sides
        coins.sort(key=lambda c: c['center'][0])
        coins[0]['side'] = 'obverse'
        coins[1]['side'] = 'reverse'
        
        # Compute watershed result stats for debug
        ws_unique_labels = len(np.unique(markers_ws))
        
        return {
            'status': 'split',
            'method': 'watershed',
            'coins': coins,
            'debug': {
                'threshold_used': best_threshold,
                'cc_labels': 3,  # We only succeed if connected components found exactly 3
                'ws_unique_labels': ws_unique_labels
            }
        }
    
    def _extract_coin_crop(self,
                           image: np.ndarray,
                           cx: float, cy: float, r: float,
                           shape: Tuple[int, int],
                           margin_ratio: float = 0.08) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a square crop centered on the detected coin.
        """
        h, w = shape
        
        margin = int(r * margin_ratio)
        half_size = int(r + margin)
        
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size))
        y2 = min(h, int(cy + half_size))
        
        crop = image[y1:y2, x1:x2].copy()
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        return crop, bbox


# === INTEGRATION HELPER ===

def check_and_resolve_two_coins(image: np.ndarray,
                                 binary_mask: np.ndarray,
                                 gray: np.ndarray,
                                 candidates: List[Dict],
                                 config: Optional[CoinPairConfig] = None) -> Dict:
    """
    Convenience function to check for and resolve two-coin images.
    
    Call this after initial contour detection in layer1_geometry.
    
    Args:
        image: Original BGR image
        binary_mask: Thresholded binary mask from Otsu
        gray: Grayscale image (after CLAHE if applied)
        candidates: List of candidate detections from contour analysis
        config: Optional configuration overrides
    
    Returns:
        {
            'triggered': bool,
            'result': resolver result dict if triggered, else None
        }
    """
    resolver = TwoCoinResolver(config)
    h, w = image.shape[:2]
    
    should_trigger, trigger_cand = resolver.should_trigger(candidates, (h, w))
    
    if not should_trigger:
        return {'triggered': False, 'result': None}
    
    result = resolver.resolve(image, binary_mask, gray)
    
    return {'triggered': True, 'result': result}
