"""Stage 2: separate the subject from the background and trace its outline.

Phase 1 uses simple thresholding, which works well for a subject on a plain,
contrasting background. Phase 2 replaces this with real background removal.
"""

import cv2
import numpy as np

# Ignore blobs smaller than this fraction of the image (specks, noise).
MIN_AREA_FRACTION = 0.01


def make_mask(gray: np.ndarray) -> np.ndarray:
    """Return a black/white mask where the subject is white (255)."""
    # Otsu's method picks the threshold automatically by finding the value
    # that best splits the histogram into two groups (subject vs background).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # We don't know if the subject is darker or lighter than the background.
    # Assume the image border is mostly background: if the border came out
    # white, flip the mask.
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # "Closing" fills small holes and gaps so the outline is one clean shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_outlines(mask: np.ndarray, max_outlines: int = 1) -> list[np.ndarray]:
    """Find the largest outer outlines in the mask.

    Each outline is an (N, 2) float array of (x, y) points tracing the shape.
    """
    # RETR_EXTERNAL: only outer boundaries, not holes inside shapes.
    # CHAIN_APPROX_NONE: keep every boundary pixel; we simplify later ourselves.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area = MIN_AREA_FRACTION * mask.shape[0] * mask.shape[1]
    large = [c for c in contours if cv2.contourArea(c) >= min_area]
    large.sort(key=cv2.contourArea, reverse=True)

    # OpenCV returns shape (N, 1, 2) int arrays; flatten to (N, 2) floats.
    return [c.reshape(-1, 2).astype(np.float64) for c in large[:max_outlines]]
