"""Extract strokes from inked drawings (comics, cartoons, coloring-book art).

Instead of tracing only the subject's border, this follows the drawing's own
dark ink lines, including everything inside: eyes, emblems, muscle lines, seams.

1. Ink mask: keep only near-black pixels. Colored fills and thin colored
   detail lines (like Spider-Man's dark-red webbing) drop out. Page frames
   (long straight border lines on coloring pages) are erased.
2. Subject: everything enclosed by the outermost ink, found with a
   paint-bucket fill from the image edges. Each sizeable part (body, feet)
   becomes an essential silhouette loop that is never dropped.
3. Split the remaining ink into thin lines and thick filled areas.
4. Thin lines: skeletonize (shrink to 1px-wide centerlines) and trace.
5. Thick areas: trace their borders, including holes (eye whites in a
   black head).
6. Keep only lines inside the subject (drops text, watermarks, scenery),
   and clip lines that run along the silhouette so edges aren't traced twice.
"""

import cv2
import numpy as np
from dataclasses import replace

from skimage.morphology import skeletonize

from dot2dot.models import Path
from dot2dot.skeleton import trace_skeleton

# Pixels darker than this (0-255, brightest color channel) count as ink.
INK_THRESHOLD = 60
# Ink wider than this many pixels (at the 1000px working size) is treated as
# a filled area, not a line. Bold cartoon outlines reach ~15px; shadows and
# solid shapes are 30px+.
THICK_WIDTH = 24
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
# The silhouette counts this much more than interior lines when sharing out dots.
SILHOUETTE_WEIGHT = 1.3
# Pieces left after clipping that are shorter than this (pixels) are dropped.
MIN_PIECE_LENGTH = 10
# Separate subject parts (feet, a detached hand) are kept if at least this
# fraction of the largest part's area.
MIN_PART_FRACTION = 0.03
# A straight line at least this fraction of the image width/height, lying in
# the outer FRAME_BAND of the image, is a page frame.
FRAME_LINE_FRACTION = 0.5
FRAME_BAND = 0.15
# The paint-bucket fill is only trusted when at least this share of the
# pixels along the image edges match the background color (plain backdrop).
MIN_PLAIN_EDGE_SHARE = 0.6
# An interior line is kept if at least this share of it lies inside the subject.
MIN_INSIDE_SHARE = 0.6


def ink_mask(image: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """White where the drawing has dark ink, black elsewhere."""
    # HSV "value" is the brightest of the three color channels. Unlike plain
    # grayscale, it keeps dark-but-saturated colors (deep red, navy) out of
    # the ink mask; only truly dark pixels have all three channels low.
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = np.where(value < threshold, 255, 0).astype(np.uint8)
    # "Opening" erases specks and hairlines thinner than the 3x3 kernel.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_frame(marks: np.ndarray) -> np.ndarray:
    """Mask of long, thin straight lines near the image border (page frames)."""
    height, width = marks.shape
    # Only thin strokes can be frame lines. Removing everything a 9x9 square
    # fits inside leaves just the thin parts, so the edge of a big solid
    # shape (a yellow emoji face) can't pass for a frame.
    ink = cv2.subtract(marks, cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)))
    # Bridge small gaps along each direction (faint anti-aliased pixels), then
    # opening with a long thin kernel keeps only straight runs at least that long.
    across = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    down = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    horizontal = cv2.morphologyEx(across, cv2.MORPH_OPEN, np.ones((1, int(width * FRAME_LINE_FRACTION)), np.uint8))
    vertical = cv2.morphologyEx(down, cv2.MORPH_OPEN, np.ones((int(height * FRAME_LINE_FRACTION), 1), np.uint8))
    band_y, band_x = int(height * FRAME_BAND), int(width * FRAME_BAND)
    frame = np.zeros_like(ink)
    frame[:band_y] = horizontal[:band_y]
    frame[-band_y:] = horizontal[-band_y:]
    frame[:, :band_x] |= vertical[:, :band_x]
    frame[:, -band_x:] |= vertical[:, -band_x:]
    # Once a row/column is confirmed as part of a frame, take the whole line,
    # including stretches where the drawing touches it (which break the run).
    rows = np.flatnonzero(frame.any(axis=1) & ((np.arange(height) < band_y) | (np.arange(height) >= height - band_y)))
    cols = np.flatnonzero(frame.any(axis=0) & ((np.arange(width) < band_x) | (np.arange(width) >= width - band_x)))
    frame[rows, :] |= horizontal[rows, :] | marks[rows, :] & (horizontal[rows, :].any(axis=1, keepdims=True) * 255).astype(np.uint8)
    frame[:, cols] |= vertical[:, cols] | marks[:, cols] & (vertical[:, cols].any(axis=0, keepdims=True) * 255).astype(np.uint8)
    # Grow slightly to also cover the anti-aliased fringe of the frame line.
    return cv2.dilate(frame, np.ones((5, 5), np.uint8)) if frame.any() else frame


def remove_frame(ink: np.ndarray) -> np.ndarray:
    return cv2.subtract(ink, find_frame(ink))


def subject_region(image: np.ndarray, ink: np.ndarray, ignore: np.ndarray | None = None) -> np.ndarray | None:
    """A mask of the subject (white), or None if it can't be separated.

    Works like a paint-bucket fill: pour paint into the background near the
    image edges. Paint only flows through background-colored pixels and
    stops at ink, so it can't leak through a small gap in the outline when
    the subject's own colors block the way. Whatever stays dry is the subject.
    Pixels in `ignore` (e.g. a page frame) never block the paint.
    """
    edge = _edge_band(ink.shape)
    differs = _differs_from_background(image, edge)
    if np.mean(differs[edge] == 0) < MIN_PLAIN_EDGE_SHARE:
        return None  # Busy or blurred background: the fill would be meaningless.
    barrier = np.maximum(ink, differs)
    if ignore is not None:
        barrier[ignore > 0] = 0
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

    area = np.count_nonzero(subject)
    # Too small: probably noise. Nearly the whole image: the fill failed.
    if not MIN_SILHOUETTE_FRACTION * ink.size <= area <= 0.9 * ink.size:
        return None
    return subject


def silhouette_paths(subject: np.ndarray) -> list[Path]:
    """One essential loop per sizeable part of the subject mask (body, feet...)."""
    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(cv2.contourArea(c) for c in contours)
    return [
        Path(c.reshape(-1, 2).astype(np.float64), closed=True, essential=True, weight=SILHOUETTE_WEIGHT)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(c) >= MIN_PART_FRACTION * largest
    ]


def keep_inside(paths: list[Path], subject: np.ndarray) -> list[Path]:
    """Drop paths that lie mostly outside the subject (text, frames, scenery)."""
    grown = cv2.dilate(subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * SILHOUETTE_BAND + 1,) * 2))
    height, width = subject.shape
    kept = []
    for path in paths:
        xs = np.clip(path.points[:, 0].astype(int), 0, width - 1)
        ys = np.clip(path.points[:, 1].astype(int), 0, height - 1)
        if (grown[ys, xs] > 0).mean() >= MIN_INSIDE_SHARE:
            kept.append(path)
    return kept


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
    """Borders of solid ink areas, including the edges of holes inside them."""
    min_area = MIN_FILL_AREA_FRACTION * thick.shape[0] * thick.shape[1]
    # RETR_CCOMP returns outer borders *and* hole borders (e.g. a white eye
    # inside a black head), where RETR_EXTERNAL would only give the outside.
    contours, _ = cv2.findContours(thick, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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


def silhouette_band(outers: list[Path], shape: tuple[int, ...]) -> np.ndarray:
    """A mask covering a strip along the silhouette(s), used to clip duplicates."""
    band = np.zeros(shape[:2], np.uint8)
    for outer in outers:
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


def drop_frame_edges(path: Path, shape: tuple[int, ...], width: int) -> list[Path]:
    """Remove stretches of a path that just run along the image border.

    When the subject is cut off by the picture's edge (shoulders in a
    portrait, an emoji cropped tight), the outline follows the frame there.
    That's the edge of the photo, not part of the drawing.
    """
    border = np.zeros(shape[:2], np.uint8)
    border[:width, :] = border[-width:, :] = 255
    border[:, :width] = border[:, -width:] = 255
    return clip_paths([path], border)


def assemble(subject: np.ndarray, interior: list[Path], edge_width: int) -> list[Path]:
    """Silhouette loops for every subject part + interior lines inside the subject."""
    outers = silhouette_paths(subject)
    # Rebuild the mask from the parts we kept, so specks (like the holes in
    # letters of a caption) don't count as subject.
    kept = np.zeros_like(subject)
    cv2.drawContours(kept, [o.points.astype(np.int32).reshape(-1, 1, 2) for o in outers], -1, 255, cv2.FILLED)
    interior = keep_inside(clip_paths(interior, silhouette_band(outers, subject.shape)), kept)
    edges = [piece for outer in outers for piece in drop_frame_edges(outer, subject.shape, edge_width)]
    return edges + interior


def find_subject(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Ink mask (with any page frame removed) and the subject mask (or None)."""
    ink = ink_mask(image)
    # Frames can be gray or colored, so look for them among all marks that
    # stand out from the background, not just black ink.
    frame = find_frame(np.maximum(ink, _differs_from_background(image, _edge_band(ink.shape))))
    ink = cv2.subtract(ink, frame)
    return ink, subject_region(image, ink, ignore=frame)


def extract_paths(image: np.ndarray) -> list[Path]:
    """Silhouette loop(s) + centerlines of interior ink lines + borders of fills."""
    ink, subject = find_subject(image)
    if subject is None:
        return trace_ink(ink)
    # The subject mask was cut off half an edge margin in from the border.
    return assemble(subject, trace_ink(ink), _edge_margin(ink.shape) // 2 + 3)
