"""Sketch style: turn any picture into a pencil sketch, then trace the sketch.

One pipeline for cartoons, coloring pages and photos:

1. Pencil sketch of the grayscale image ("color dodge": divide the image by
   a blurred, inverted copy of itself). Flat areas, even solid black ones,
   turn white; edges turn into dark lines. Bold outlines come out thick,
   fine details (web patterns, eyelashes) come out thin.
2. Color edges: lines between colors of similar brightness (an orange beak
   on a white face, a yellow face on a white page) vanish in grayscale, so
   an edge detector on the color channels adds them back where the sketch
   has no line.
3. Clean up: keep only the subject (background removal / paint-bucket),
   drop page frames and captions, and turn big solid blobs into outlines.
4. Trace: thin the lines to 1px centerlines and trace them into paths. Each
   path remembers how thick its line was; thick lines (outlines) get more
   weight when dots are shared out, so the shape comes first and thin
   details are added as the dot budget allows.
"""

from dataclasses import replace

import cv2
import numpy as np
from skimage.morphology import skeletonize

from dot2dot import lineart
from dot2dot.models import Path
from dot2dot.skeleton import trace_skeleton

# Pencil-sketch blur (pixels at the 1000px working size): bigger = bolder lines.
PENCIL_SIGMA = 6
# Sketch pixels darker than this (0-255) are lines.
LINE_THRESHOLD = 200
# Color-edge detector thresholds (on normalized color-gradient strength).
CHROMA_LOW, CHROMA_HIGH = 40, 100
# Lines at least this wide (pixels) count as bold outlines.
BOLD_WIDTH = 6
# Bold lines count this much more than thin ones when sharing out dots.
BOLD_WEIGHT = 1.6
# Paths shorter than this (pixels) are noise (texture specks, hair strands).
MIN_PATH_LENGTH = 25


def pencil_sketch(image: np.ndarray, sigma: float = PENCIL_SIGMA) -> np.ndarray:
    """Grayscale pencil sketch: white paper, dark lines (uint8)."""
    # Lift pure black (0) to 1: otherwise a solid black area divides 0 by 0
    # and stays black instead of turning into an outline.
    gray = np.maximum(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1)
    blurred_inverse = cv2.GaussianBlur(255 - gray, (0, 0), sigma)
    # "Color dodge": where the image matches its blurred surroundings the
    # ratio is ~1 (white); where it's darker than its surroundings, a line.
    return cv2.divide(gray, 255 - blurred_inverse, scale=256)


def color_edges(image: np.ndarray) -> np.ndarray:
    """Edges between colors, regardless of brightness (white = edge)."""
    smoothed = cv2.bilateralFilter(image, 9, 50, 7)  # flatten texture, keep edges
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    strength = np.zeros(image.shape[:2], np.float32)
    for channel in (1, 2):  # a (green-red) and b (blue-yellow)
        gx = cv2.Sobel(lab[..., channel], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(lab[..., channel], cv2.CV_32F, 0, 1)
        strength = np.maximum(strength, np.hypot(gx, gy))
    strength = cv2.convertScaleAbs(strength, alpha=255 / max(float(strength.max()), 1.0))
    return cv2.Canny(strength, CHROMA_LOW, CHROMA_HIGH)


def line_drawing(image: np.ndarray) -> np.ndarray:
    """The combined line drawing (white = line): pencil sketch + missing color edges."""
    lines = np.where(pencil_sketch(image) < LINE_THRESHOLD, 255, 0).astype(np.uint8)
    # Only add color edges where the sketch has no line nearby.
    extra = cv2.dilate(color_edges(image), np.ones((3, 3), np.uint8))
    extra = cv2.subtract(extra, cv2.dilate(lines, np.ones((9, 9), np.uint8)))
    return np.maximum(lines, extra)


def subject_mask(image: np.ndarray) -> np.ndarray:
    """Where the subject is: paint-bucket on plain backgrounds, else background removal."""
    _, painted = lineart.find_subject(image)
    if painted is not None:
        return painted
    from dot2dot import photo

    return photo.subject_mask(image)


def cleaned_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The line drawing with page frames and background removed, plus the subject mask."""
    lines = line_drawing(image)
    # Drop page frames (long straight border lines), then everything outside the subject.
    lines = cv2.subtract(lines, lineart.find_frame(lines))
    subject = subject_mask(image)
    lines[cv2.dilate(subject, np.ones((15, 15), np.uint8)) == 0] = 0
    return lines, subject


def preview(image: np.ndarray) -> np.ndarray:
    """The cleaned line drawing as a picture (black lines on white), for display."""
    lines, _ = cleaned_lines(image)
    return cv2.cvtColor(255 - lines, cv2.COLOR_GRAY2BGR)


def extract_paths(image: np.ndarray) -> list[Path]:
    lines, subject = cleaned_lines(image)

    # Line width at every line pixel: twice the distance to the nearest background pixel.
    width = cv2.distanceTransform(lines, cv2.DIST_L2, 5) * 2

    # Big solid areas (a black helmet, dark hair) become outlines, not scribbles.
    thick, thin = lineart.split_thick_and_thin(lines)
    paths = trace_skeleton(skeletonize(thin > 0)) + lineart.fill_outlines(thick)
    paths = [p for p in paths if p.length() >= MIN_PATH_LENGTH]
    paths = [_weigh_by_width(p, width) for p in paths]
    # Line thickness means importance in drawings, not in photos: facial
    # features are thin, faint lines. Faces get extra weight either way.
    from dot2dot import photo

    faces = photo.detect_faces(image)
    paths = [_boost_faces(p, faces) for p in paths]

    # Silhouette loops for each subject part, interior lines inside the subject.
    return lineart.assemble(subject, paths, edge_width=4)


def _boost_faces(path: Path, faces) -> Path:
    from dot2dot import photo

    boosted = photo._weight_by_faces(path, faces)
    return replace(path, weight=max(path.weight, boosted.weight)) if boosted is not path else path


def _weigh_by_width(path: Path, width: np.ndarray) -> Path:
    """Bold lines (outlines) get more weight than thin ones (details)."""
    height, w = width.shape
    xs = np.clip(path.points[:, 0].astype(int), 0, w - 1)
    ys = np.clip(path.points[:, 1].astype(int), 0, height - 1)
    # Sample the widest point near the centerline (the skeleton sits mid-line).
    local = cv2.dilate(width, np.ones((5, 5), np.uint8))[ys, xs]
    bold = float(np.median(local)) >= BOLD_WIDTH
    return replace(path, weight=BOLD_WEIGHT if bold else 1.0)
