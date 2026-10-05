import cv2
import numpy as np

from dot2dot.lineart import clip_paths, ink_mask, silhouette
from dot2dot.models import Path


def blob_drawing(frame: bool = False) -> np.ndarray:
    """A red blob with a black outline that has a gap in it, on light gray."""
    img = np.full((400, 400, 3), (235, 235, 235), np.uint8)
    cv2.circle(img, (200, 200), 120, (40, 40, 220), -1)
    cv2.circle(img, (200, 200), 120, (0, 0, 0), 5)
    cv2.line(img, (320, 190), (320, 210), (40, 40, 220), 9)  # gap in the outline
    if frame:
        cv2.rectangle(img, (0, 0), (399, 399), (200, 150, 100), 2)
    return img


def test_silhouette_survives_gap_in_outline():
    img = blob_drawing()
    outer = silhouette(img, ink_mask(img))
    assert outer is not None and outer.essential and outer.closed
    # The fill color blocks the "paint", so we get the whole disc, not a sliver.
    assert abs(cv2.contourArea(outer.points.astype(np.float32)) - np.pi * 122**2) < 0.1 * np.pi * 122**2


def test_silhouette_ignores_frame_around_image():
    img = blob_drawing(frame=True)
    outer = silhouette(img, ink_mask(img))
    assert outer is not None
    assert outer.points[:, 0].max() < 340  # the disc, not the frame


def test_clip_paths_splits_loop_where_it_enters_band():
    square = np.array(
        [(x, 0) for x in range(100)] + [(100, y) for y in range(100)]
        + [(x, 100) for x in range(100, 0, -1)] + [(0, y) for y in range(100, 0, -1)],
        dtype=np.float64,
    )
    band = np.zeros((120, 120), np.uint8)
    band[:, 90:] = 255  # covers the right side of the square
    pieces = clip_paths([Path(square, closed=True)], band)
    assert len(pieces) == 1
    assert not pieces[0].closed
    assert pieces[0].points[:, 0].max() < 90
