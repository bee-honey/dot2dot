"""Wire the stages together: image path in, Puzzle out."""

from dataclasses import dataclass, replace
from pathlib import Path as FilePath

import numpy as np

from dot2dot import lineart, photo
from dot2dot.contours import find_outlines, make_mask
from dot2dot.labels import choose_font_size, outward_normals, place_labels
from dot2dot.models import Dot, Path, Puzzle, Stroke
from dot2dot.order import order_paths
from dot2dot.preprocess import load_image, resize, to_grayscale
from dot2dot.simplify import allocate_dots, select_paths, simplify

STYLES = ("outline", "lineart", "photo")
# Line art and photos keep more detail, so they're processed at a higher resolution.
WORKING_SIZE = {"outline": 800, "lineart": 1000, "photo": 1000}


@dataclass(frozen=True)
class Result:
    """A finished puzzle plus what it was built from, for quality checks."""

    puzzle: Puzzle
    # Every path found in the image, before any were dropped or simplified:
    # the "reference drawing" the puzzle is trying to reproduce.
    reference: list[Path]


def extract_paths(image: np.ndarray, style: str) -> list[Path]:
    if style == "outline":
        outlines = find_outlines(make_mask(to_grayscale(image)), None)
        # The largest outline (find_outlines sorts by area) is the main subject.
        return [Path(points, closed=True, essential=(i == 0)) for i, points in enumerate(outlines)]
    if style == "lineart":
        return lineart.extract_paths(image)
    if style == "photo":
        return photo.extract_paths(image)
    raise ValueError(f"Unknown style {style!r}; choose from {', '.join(STYLES)}")


def generate(
    image: str | FilePath | np.ndarray,
    num_dots: int = 60,
    max_paths: int | None = 1,
    style: str = "outline",
) -> Puzzle:
    """Turn an image into a connect-the-dots puzzle with roughly `num_dots` dots.

    `image` is a file path or an already-loaded BGR array.
    max_paths limits how many separate lines/shapes are traced (None = no limit).
    """
    return build(image, num_dots, max_paths, style).puzzle


def build(
    image: str | FilePath | np.ndarray,
    num_dots: int = 60,
    max_paths: int | None = 1,
    style: str = "outline",
) -> Result:
    """Like `generate`, but also returns the reference paths for quality checks."""
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; choose from {', '.join(STYLES)}")
    if not isinstance(image, np.ndarray):
        image = load_image(image)
    image = resize(image, WORKING_SIZE[style])
    height, width = image.shape[:2]

    reference = extract_paths(image, style)
    if not reference:
        raise ValueError("No clear subject found in the image")
    paths = select_paths(reference, num_dots, max_paths)

    counts = allocate_dots(paths, num_dots)
    simplified = [replace(p, points=simplify(p, n)) for p, n in zip(paths, counts)]
    ordered = order_paths(simplified)

    # Flatten the ordered paths into one numbered list of dots.
    points = np.concatenate([p.points for p in ordered])
    strokes, preferred = [], []
    for path in ordered:
        start = sum(s.end - s.start for s in strokes)
        strokes.append(Stroke(start, start + len(path.points), path.closed))
        preferred.append(outward_normals(path.points) if path.closed else np.zeros_like(path.points))

    numbers = list(range(1, len(points) + 1))
    rings = np.zeros(len(points), dtype=bool)
    rings[[s.start for s in strokes[1:]]] = True
    font_size = choose_font_size(points, width, height)
    draft = Puzzle(width, height, [Dot(n, x, y) for n, (x, y) in zip(numbers, points)], strokes, font_size)
    segments = np.array([[[a.x, a.y], [b.x, b.y]] for a, b in draft.segments()]).reshape(-1, 2, 2)
    directions = place_labels(points, numbers, segments, np.concatenate(preferred), font_size, rings)

    dots = [
        Dot(n, float(x), float(y), float(dx), float(dy), bool(ring))
        for n, (x, y), (dx, dy), ring in zip(numbers, points, directions, rings)
    ]
    return Result(Puzzle(width, height, dots, strokes, font_size), reference)
