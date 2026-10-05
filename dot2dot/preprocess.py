"""Stage 1: load an image and prepare it for contour detection."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Teach Pillow to open iPhone .HEIC photos.
register_heif_opener()

# Working resolution. Photos straight off a phone are ~4000px wide, which is
# slow to process and full of noise we don't want anyway.
MAX_SIDE = 800


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR array (OpenCV's default channel order)."""
    try:
        with Image.open(path) as image:
            return _to_bgr(image)
    except (OSError, ValueError) as error:  # missing, unreadable, or not an image
        raise FileNotFoundError(f"Could not read image: {path}") from error


def decode_image(data: bytes) -> np.ndarray:
    """Decode an image from raw file bytes (e.g. an upload) into a BGR array."""
    try:
        with Image.open(BytesIO(data)) as image:
            return _to_bgr(image)
    except (OSError, ValueError) as error:
        raise ValueError("Could not read image: unsupported or corrupt file") from error


def _to_bgr(image: Image.Image) -> np.ndarray:
    """Normalize any Pillow image to an upright, 8-bit, 3-channel BGR array.

    - Phone photos store their rotation in EXIF metadata; apply it so the
      picture isn't sideways.
    - Transparent areas become white. (Left alone they'd turn black, and a
      cut-out photo would look like a subject surrounded by a sea of ink.)
    """
    image = ImageOps.exif_transpose(image)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255
    rgb = rgba[:, :, :3] * alpha + 255 * (1 - alpha)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


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
