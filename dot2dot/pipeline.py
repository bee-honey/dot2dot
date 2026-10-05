"""Wire the stages together: image path in, Puzzle out."""

from dataclasses import dataclass, replace
from pathlib import Path as FilePath

import numpy as np

from dot2dot import lineart, mystery, photo
from dot2dot.contours import find_outlines, make_mask
from dot2dot.guidance import apply_plan
from dot2dot.labels import choose_font_size, outward_normals, place_labels
from dot2dot.models import Dot, Path, Puzzle, Stroke
from dot2dot.order import order_paths
from dot2dot.planner import Plan
from dot2dot.preprocess import load_image, resize, to_grayscale
from dot2dot.simplify import allocate_dots, declutter, max_readable_dots, select_paths, simplify

STYLES = ("auto", "outline", "lineart", "photo")
# Line art and photos keep more detail, so they're processed at a higher resolution.
WORKING_SIZE = {"outline": 800, "lineart": 1000, "photo": 1000}


@dataclass(frozen=True)
class Result:
    """A finished puzzle plus what it was built from, for quality checks."""

    puzzle: Puzzle
    # Every path found in the image, before any were dropped or simplified:
    # the "reference drawing" the puzzle is trying to reproduce.
    reference: list[Path]
    style: str  # the style actually used (resolves "auto")
    plan: Plan | None = None


def choose_style(image: np.ndarray, plan: Plan | None = None) -> str:
    """Pick a style when the user asked for "auto".

    A plain background means the subject can be cut out by color, so line art
    works; anything busier (a scene, a blurred photo backdrop) needs the photo
    pipeline's background removal, whatever the plan says.
    """
    _, subject = lineart.find_subject(resize(image, WORKING_SIZE["lineart"]))
    if subject is None:
        return "photo"
    # The plan may switch a plain-background picture to the photo pipeline,
    # but never to the bare outline style (that would drop interior details).
    return "photo" if plan is not None and plan.style == "photo" else "lineart"


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
    plan: Plan | None = None,
    mystery_level: int = 0,
) -> Result:
    """Like `generate`, but also returns the reference paths for quality checks.

    With a `plan` (from the AI planner), ignored regions are dropped,
    must-include parts get extra dots, and missing parts are repaired.
    With `mystery_level` 1-2, filler strokes hide the picture until solved.
    """
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; choose from {', '.join(STYLES)}")
    if not isinstance(image, np.ndarray):
        image = load_image(image)
    seed = image.tobytes()[:: max(1, image.size // 4096)]  # cheap image fingerprint
    if style == "auto":
        style = choose_style(image, plan)
    image = resize(image, WORKING_SIZE[style])
    height, width = image.shape[:2]

    reference = extract_paths(image, style)
    if plan is not None:
        reference = apply_plan(reference, plan, image)
    if not reference:
        raise ValueError("No clear subject found in the image")
    paths = select_paths(reference, num_dots, max_paths)
    # Asking for more dots than the lines can legibly hold just piles up numbers.
    readable = max_readable_dots(paths, max(width, height))
    if num_dots > readable:
        num_dots = readable
        paths = select_paths(reference, num_dots, max_paths)

    if mystery_level:
        if mystery_level >= 2:
            spacing = sum(p.length() for p in paths) / num_dots
            paths = [replace(p, weight=p.weight * mystery.FEATURE_WEIGHT) if mystery.is_feature(p, spacing) else p
                     for p in paths]
        paths = paths + mystery.add_fillers(paths, image.shape, num_dots, mystery_level, seed)

    counts = allocate_dots(paths, num_dots)
    simplified = [replace(p, points=simplify(p, n)) for p, n in zip(paths, counts)]
    if mystery_level:
        # Start in the page corner, and save small give-away loops (eyes) for last.
        spacing = sum(p.length() for p in paths) / num_dots
        features = [p for p in simplified if mystery.is_feature(p, spacing)]
        rest = [p for p in simplified if not mystery.is_feature(p, spacing)]
        ordered = order_paths(rest, start_at=np.array([0.0, 0.0]))
        pen = ordered[-1].points[0 if ordered[-1].closed else -1] if ordered else np.zeros(2)
        ordered += order_paths(features, start_at=pen)
    else:
        ordered = order_paths(simplified)
    ordered = declutter(ordered, max(width, height))

    # Flatten the ordered paths into one numbered list of dots.
    points = np.concatenate([p.points for p in ordered])
    strokes, preferred = [], []
    for path in ordered:
        start = sum(s.end - s.start for s in strokes)
        strokes.append(Stroke(start, start + len(path.points), path.closed, path.filler))
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
    return Result(Puzzle(width, height, dots, strokes, font_size), reference, style, plan)
