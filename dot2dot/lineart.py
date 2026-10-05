"""Extract strokes from inked drawings (comics, cartoons, coloring-book art).

Instead of tracing only the subject's border, this follows the drawing's own
dark ink lines, including everything inside: eyes, emblems, muscle lines, seams.

1. Ink mask: keep only near-black pixels. Colored fills and thin colored
   detail lines (like Spider-Man's dark-red webbing) drop out.
2. Silhouette: everything enclosed by the outermost ink line, traced as one
   unbroken loop. It is marked essential so it is never dropped.
3. Split the remaining ink into thin lines and thick filled areas.
4. Thin lines: skeletonize (shrink to 1px-wide centerlines) and trace.
5. Thick areas: trace their border as a loop.
6. Clip away interior lines that run along the silhouette, so the outer
   edge isn't traced twice.
"""

import cv2
import numpy as np
from dataclasses import replace

from skimage.morphology import skeletonize

from dot2dot.models import Path
from dot2dot.skeleton import trace_skeleton

# Pixels darker than this (0-255, brightest color channel) count as ink.
INK_THRESHOLD = 60
# Ink wider than this many pixels is treated as a filled area, not a line.
THICK_WIDTH = 15
# Ignore filled areas smaller than this fraction of the image.
MIN_FILL_AREA_FRACTION = 0.0005
# The silhouette must cover at least this fraction of the image to be trusted.
MIN_SILHOUETTE_FRACTION = 0.05
# How different (Lab color distance) a pixel must be from the background
# color to count as part of the subject.
BACKGROUND_TOLERANCE = 20
# Width of the strip along the image edges used to find the background.
EDGE_BAND_FRACTION = 0.02
# Interior lines within this many pixels of the silhouette are clipped away.
SILHOUETTE_BAND = 8
# Pieces left after clipping that are shorter than this (pixels) are dropped.
MIN_PIECE_LENGTH = 10


def ink_mask(image: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """White where the drawing has dark ink, black elsewhere."""
    # HSV "value" is the brightest of the three color channels. Unlike plain
    # grayscale, it keeps dark-but-saturated colors (deep red, navy) out of
    # the ink mask; only truly dark pixels have all three channels low.
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = np.where(value < threshold, 255, 0).astype(np.uint8)
    # "Opening" erases specks and hairlines thinner than the 3x3 kernel.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def silhouette(image: np.ndarray, ink: np.ndarray) -> Path | None:
    """The outer border of the subject.

    Works like a paint-bucket fill: pour paint into the background near the
    image edges. Paint only flows through background-colored pixels and
    stops at ink, so it can't leak through a small gap in the outline when
    the subject's own colors block the way. Whatever stays dry is the subject.
    """
    edge = _edge_band(ink.shape)
    barrier = np.maximum(ink, _differs_from_background(image, edge))
    # Seal tiny gaps so the paint can't leak inside.
    sealed = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Split the open (non-barrier) area into connected regions. Every region
    # that reaches the edge band is background. Seeding from a band instead
    # of the very edge copes with images that have a thin frame drawn around them.
    _, regions = cv2.connectedComponents(np.where(sealed == 0, 1, 0).astype(np.uint8), connectivity=4)
    background_ids = np.unique(regions[edge & (sealed == 0)])
    background_ids = background_ids[background_ids != 0]  # 0 = barrier pixels
    subject = np.where(np.isin(regions, background_ids), 0, 255).astype(np.uint8)
    subject[~_inner_area(ink.shape)] = 0  # Ignore any frame drawn along the edge.

    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Too small: probably noise. Nearly the whole image: the fill failed.
    if not MIN_SILHOUETTE_FRACTION * ink.size <= area <= 0.9 * ink.size:
        return None
    return Path(largest.reshape(-1, 2).astype(np.float64), closed=True, essential=True)


def _edge_margin(shape: tuple[int, ...]) -> int:
    return max(4, round(EDGE_BAND_FRACTION * max(shape[:2])))


def _inner_area(shape: tuple[int, ...]) -> np.ndarray:
    """True everywhere except a thin strip along the image edges."""
    inner = np.zeros(shape[:2], dtype=bool)
    m = _edge_margin(shape) // 2
    inner[m:-m, m:-m] = True
    return inner


def _edge_band(shape: tuple[int, ...]) -> np.ndarray:
    """True in a strip just inside the image edges, where background is expected."""
    band = np.zeros(shape[:2], dtype=bool)
    m = _edge_margin(shape)
    band[:m, :] = band[-m:, :] = band[:, :m] = band[:, -m:] = True
    return band & _inner_area(shape)


def _differs_from_background(image: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """White where a pixel's color is clearly not the background color."""
    # Lab color space is built so that distance matches perceived difference.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = np.median(lab[edge], axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    return np.where(distance > BACKGROUND_TOLERANCE, 255, 0).astype(np.uint8)


def split_thick_and_thin(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate filled ink areas from ink lines."""
    # Opening with a big disk keeps only regions the disk fits inside,
    # i.e. ink wider than THICK_WIDTH. Lines vanish; fills survive.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THICK_WIDTH, THICK_WIDTH))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk)
    # Remove the fills (slightly grown) from the ink so lines stop at their edge.
    thin = cv2.subtract(ink, cv2.dilate(thick, np.ones((5, 5), np.uint8)))
    return thick, thin


def fill_outlines(thick: np.ndarray) -> list[Path]:
    min_area = MIN_FILL_AREA_FRACTION * thick.shape[0] * thick.shape[1]
    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [
        Path(c.reshape(-1, 2).astype(np.float64), closed=True)
        for c in contours
        if cv2.contourArea(c) >= min_area
    ]


def clip_paths(paths: list[Path], band: np.ndarray) -> list[Path]:
    """Cut out the parts of each path that fall inside `band` (a mask).

    A loop that dips into the band becomes one or more open pieces.
    """
    height, width = band.shape
    result = []
    for path in paths:
        xs = np.clip(path.points[:, 0].astype(int), 0, width - 1)
        ys = np.clip(path.points[:, 1].astype(int), 0, height - 1)
        inside = band[ys, xs] > 0
        if not inside.any():
            result.append(path)
            continue
        points = path.points
        if path.closed:
            # Rotate the loop to start inside the band, so no outside run
            # is split across the end/start seam.
            shift = int(np.argmax(inside))
            points, inside = np.roll(points, -shift, axis=0), np.roll(inside, -shift)
        # Find runs of consecutive outside points.
        edges = np.diff(np.concatenate([[0], (~inside).astype(int), [0]]))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            piece = replace(path, points=points[start:end], closed=False)
            if end - start >= 2 and piece.length() >= MIN_PIECE_LENGTH:
                result.append(piece)
    return result


def silhouette_band(outer: Path, shape: tuple[int, ...]) -> np.ndarray:
    """A mask covering a strip along the silhouette, used to clip duplicates."""
    band = np.zeros(shape[:2], np.uint8)
    cv2.polylines(band, [outer.points.astype(np.int32)], outer.closed, 255, 2 * SILHOUETTE_BAND)
    return band


def trace_ink(ink: np.ndarray, band: np.ndarray | None = None, include_fills: bool = True) -> list[Path]:
    """Trace an ink mask into paths: centerlines of lines (+ borders of fills).

    Anything inside `band` (the strip along the silhouette) is clipped away.
    """
    thick, thin = split_thick_and_thin(ink)
    paths = trace_skeleton(skeletonize(thin > 0))
    if include_fills:
        paths += fill_outlines(thick)
    return clip_paths(paths, band) if band is not None else paths


def extract_paths(image: np.ndarray) -> list[Path]:
    """Silhouette loop + centerlines of interior ink lines + borders of fills."""
    ink = ink_mask(image)
    outer = silhouette(image, ink)
    if outer is None:
        return trace_ink(ink)
    return [outer] + trace_ink(ink, silhouette_band(outer, ink.shape))
