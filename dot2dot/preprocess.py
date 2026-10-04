"""Stage 1: load an image and prepare it for contour detection."""

from pathlib import Path

import cv2
import numpy as np

# Working resolution. Photos straight off a phone are ~4000px wide, which is
# slow to process and full of noise we don't want anyway.
MAX_SIDE = 800


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR array (OpenCV's default channel order)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Shrink the image so its longest side is at most `max_side` pixels."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to a single-channel image and blur away fine texture."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A small blur removes pixel-level noise (grain, JPEG artifacts) so the
    # threshold step produces smooth outlines instead of jagged ones.
    return cv2.GaussianBlur(gray, (5, 5), 0)


def preprocess(path: str | Path) -> np.ndarray:
    """Load, resize and grayscale an image in one call."""
    return to_grayscale(resize(load_image(path)))
