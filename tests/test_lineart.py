import cv2
import numpy as np

from dot2dot.lineart import clip_paths, extract_paths, ink_mask, remove_frame, silhouette_paths, subject_region
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


def silhouette(img):
    subject = subject_region(img, ink_mask(img))
    return silhouette_paths(subject)[0] if subject is not None else None


def test_silhouette_survives_gap_in_outline():
    img = blob_drawing()
    outer = silhouette(img)
    assert outer is not None and outer.essential and outer.closed
    # The fill color blocks the "paint", so we get the whole disc, not a sliver.
    assert abs(cv2.contourArea(outer.points.astype(np.float32)) - np.pi * 122**2) < 0.1 * np.pi * 122**2


def test_silhouette_ignores_frame_around_image():
    img = blob_drawing(frame=True)
    outer = silhouette(img)
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


def coloring_page(frame: bool = True, text: bool = True, feet: bool = True) -> np.ndarray:
    """Black outlines on white: a body, two separate feet, a page frame and a caption."""
    img = np.full((500, 400, 3), 255, np.uint8)
    cv2.ellipse(img, (200, 230), (100, 150), 0, 0, 360, (0, 0, 0), 4)  # body
    cv2.circle(img, (200, 180), 25, (0, 0, 0), 4)  # an interior feature
    if feet:
        for x in (150, 250):
            cv2.ellipse(img, (x, 420), (35, 15), 0, 0, 360, (0, 0, 0), 4)
    if frame:
        cv2.rectangle(img, (10, 10), (389, 489), (0, 0, 0), 3)
    if text:
        cv2.putText(img, "PENGUIN", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def test_page_frame_is_removed():
    img = coloring_page(text=False, feet=False)
    cleaned = remove_frame(ink_mask(img))
    assert cleaned[:20, :].max() == 0 and cleaned[:, :20].max() == 0  # frame gone
    assert cleaned[230, 96:106].max() > 0  # body outline kept


def test_coloring_page_traces_subject_not_frame():
    paths = extract_paths(coloring_page())
    outlines = [p for p in paths if p.essential]
    # Body plus two separate feet; the frame is not an outline.
    assert len(outlines) == 3
    assert all(p.points[:, 0].min() > 50 for p in outlines)


def test_text_outside_subject_is_ignored():
    paths = extract_paths(coloring_page())
    # Nothing traced in the caption's corner of the page.
    assert not any(((p.points[:, 0] < 140) & (p.points[:, 1] < 50)).any() for p in paths)


def test_interior_feature_is_traced():
    paths = extract_paths(coloring_page())
    interior = [p for p in paths if not p.essential]
    assert any(np.hypot(*(p.points.mean(axis=0) - (200, 180))) < 15 for p in interior)
