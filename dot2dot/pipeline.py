"""Wire the stages together: image path in, Puzzle out."""

from pathlib import Path

from dot2dot.contours import find_outlines, make_mask
from dot2dot.labels import outward_normals
from dot2dot.models import Dot, Puzzle
from dot2dot.order import order_outlines
from dot2dot.preprocess import preprocess
from dot2dot.simplify import allocate_dots, simplify


def generate(image_path: str | Path, num_dots: int = 60, max_outlines: int = 1) -> Puzzle:
    """Turn an image into a connect-the-dots puzzle with roughly `num_dots` dots."""
    gray = preprocess(image_path)
    mask = make_mask(gray)

    outlines = find_outlines(mask, max_outlines)
    if not outlines:
        raise ValueError(f"No clear subject found in {image_path}")

    counts = allocate_dots(outlines, num_dots)
    simplified = [simplify(outline, count) for outline, count in zip(outlines, counts)]
    ordered = order_outlines(simplified)

    dots: list[Dot] = []
    stroke_starts: list[int] = []
    for points in ordered:
        stroke_starts.append(len(dots))
        for (x, y), (dx, dy) in zip(points, outward_normals(points)):
            dots.append(Dot(len(dots) + 1, float(x), float(y), float(dx), float(dy)))

    height, width = gray.shape
    return Puzzle(width=width, height=height, dots=dots, stroke_starts=stroke_starts)
