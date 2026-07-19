"""
Shared fixtures for the MORPH_CLOSE weld tests.

Everything here is generated programmatically -- no Spaces access, no DB, no
network -- so tiers 1 and 2 run in CI.  See specs/two_coin_weld_morph_close.md.
"""

import cv2
import numpy as np
import pytest

from src.config import Layer1Config
from src.layer1_geometry import _segment_and_extract_candidates
from src.math_utils import detect_background_histogram

ENV_FRAC = "TRIVALAYA_CLOSE_KERNEL_FRAC"

BG_LEVEL = 40   # flat dark background -> THRESH_BINARY polarity
FG_LEVEL = 210  # coin fill


def bridging_px(k: int, iterations: int = None) -> int:
    """
    Closed-form bridging distance of a MORPH_CLOSE ellipse.

    Empirically pinned against OpenCV 4.12 on synthetic pairs run through the
    exact blur -> Otsu -> close chain: welding occurs iff
    ``gap < 2 * iterations * (k // 2)``.  Tier-2 expectations are derived from
    this, never from a hardcoded results table, so a change to fixture radius
    or rendering cannot silently flip a boundary case.
    """
    if iterations is None:
        iterations = Layer1Config.CLOSE_ITERATIONS
    return 2 * iterations * (k // 2)


def render_pair(scale: int, gap: int, radius: int = None,
                height: int = None) -> np.ndarray:
    """
    Two filled circles separated by exactly `gap` px of background, centred in
    a `scale`-wide frame.  Returns BGR uint8.
    """
    r = radius if radius is not None else max(20, int(scale * 0.18))
    h = height if height is not None else max(2 * r + 40, int(scale * 0.45))

    img = np.full((h, scale, 3), BG_LEVEL, np.uint8)
    cy = h // 2
    span = 4 * r + gap
    x0 = (scale - span) // 2
    cv2.circle(img, (x0 + r, cy), r, (FG_LEVEL,) * 3, -1)
    cv2.circle(img, (x0 + 3 * r + gap, cy), r, (FG_LEVEL,) * 3, -1)
    return img


def _preprocess(img: np.ndarray):
    """
    Replicate layer_1_structural_salience's preprocessing up to the point where
    it hands off to _segment_and_extract_candidates, so tests exercise the real
    segmentation function rather than a copy of it.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bg, _ = detect_background_histogram(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    v = np.median(gray_enhanced)
    sigma = Layer1Config.Standard.CANNY_SIGMA
    edges = cv2.Canny(gray_enhanced,
                      int(max(0, (1.0 - sigma) * v)),
                      int(min(255, (1.0 + sigma) * v)))
    edge_zone = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    return gray_enhanced, edge_zone, thresh_type, h, w


def segment(img: np.ndarray):
    """Run the real _segment_and_extract_candidates. Returns (candidates, binary)."""
    gray_enhanced, edge_zone, thresh_type, h, w = _preprocess(img)
    return _segment_and_extract_candidates(
        img, gray_enhanced, edge_zone, thresh_type, h, w, h * w
    )


def blob_count_post_close(img: np.ndarray) -> int:
    """Contours (>= MIN_AREA_PX) in the binary mask after the close."""
    _, binary = segment(img)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    floor = Layer1Config.Standard.MIN_AREA_PX
    return sum(1 for c in contours if cv2.contourArea(c) >= floor)


def blob_count_pre_close(img: np.ndarray) -> int:
    """
    Same chain, stopping before MORPH_CLOSE.  This is a measurement replica
    (there is no pre-close hook in the production function), used only to
    establish that Otsu alone already separated the coins.
    """
    gray_enhanced, _, thresh_type, _, _ = _preprocess(img)
    blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    floor = Layer1Config.Standard.MIN_AREA_PX
    return sum(1 for c in contours if cv2.contourArea(c) >= floor)


@pytest.fixture
def close_kernels(monkeypatch):
    """
    Records the kernel shape of every MORPH_CLOSE issued during the test.

    This observes the *call site* rather than the sizing helper, which is what
    makes the default-unchanged guard (SS9.1) meaningful: it proves the kernel
    actually reaching OpenCV is 7x7, not merely that the helper would return 7.
    """
    seen = []
    real = cv2.morphologyEx

    def spy(src, op, kernel, *args, **kwargs):
        if op == cv2.MORPH_CLOSE:
            seen.append(kernel.shape)
        return real(src, op, kernel, *args, **kwargs)

    monkeypatch.setattr(cv2, "morphologyEx", spy)
    return seen


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No test may inherit the operator's env override."""
    monkeypatch.delenv(ENV_FRAC, raising=False)
