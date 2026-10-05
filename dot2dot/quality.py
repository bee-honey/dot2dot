"""Score how well a puzzle reproduces the drawing it came from.

Three questions, each answered with a number from 0 to 1:

- outline coverage: how much of the main silhouette is traced by solution lines?
- detail coverage:  how much of the interior line work is traced?
- accuracy:         how much of the solution's line length lies on a real line
                    (rather than cutting a shortcut across empty space)?

Plus a count of crowded labels (numbers that overlap a dot or another number).

"Traced" means within a small tolerance (about 1% of the image size), because
dots are joined by straight lines and can't follow every curve exactly.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from dot2dot.labels import DIGIT_HEIGHT, DIGIT_WIDTH, DOT_RADIUS
from dot2dot.models import Path, Puzzle
from dot2dot.render import label_center

TOLERANCE_FRACTION = 0.012
# Reference lines shorter than this fraction of the image size are ignored:
# tiny specks no reasonable puzzle should be expected to include.
MIN_REFERENCE_FRACTION = 0.02

MISSING_COLOR = (255, 0, 200)  # BGR magenta
COVERED_COLOR = (190, 190, 190)
SOLUTION_COLOR = (60, 60, 212)  # BGR red


@dataclass(frozen=True)
class Quality:
    outline_coverage: float | None  # None when there's no silhouette to compare
    detail_coverage: float | None  # None when there's no interior detail
    accuracy: float
    crowded_labels: int
    total_dots: int
    # (part name, share traced) for each must-include part from an AI plan.
    parts: tuple[tuple[str, float], ...] = ()

    @property
    def parts_coverage(self) -> float | None:
        return sum(c for _, c in self.parts) / len(self.parts) if self.parts else None

    @property
    def overall(self) -> float:
        """One weighted score, 0-100. The silhouette and named parts matter most."""
        parts = [
            (0.45, self.outline_coverage),
            (0.25, self.detail_coverage),
            (0.20, self.accuracy),
            (0.10, 1 - self.crowded_labels / max(self.total_dots, 1)),
            # With a plan, missing parts (a beak, feet) weigh heavily.
            (0.50, self.parts_coverage),
        ]
        weighted = [(w, v) for w, v in parts if v is not None]
        return 100 * sum(w * v for w, v in weighted) / sum(w for w, _ in weighted)

    def summary(self) -> str:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.0%}"

        text = (
            f"Quality {self.overall:.0f}/100 | outline {pct(self.outline_coverage)} | "
            f"detail {pct(self.detail_coverage)} | accuracy {pct(self.accuracy)} | "
            f"crowded labels {self.crowded_labels}"
        )
        if self.parts:
            text += "\nParts: " + ", ".join(f"{name} {pct(c)}" for name, c in self.parts)
        return text


def evaluate(puzzle: Puzzle, reference: list[Path], plan=None) -> Quality:
    """Score the puzzle against the reference lines (and the plan's must-include parts)."""
    tolerance = TOLERANCE_FRACTION * max(puzzle.width, puzzle.height)
    outline_mask, detail_mask = _reference_masks(puzzle, reference)
    solution_mask = _solution_mask(puzzle)

    # Distance from every pixel to the nearest solution line / reference line.
    to_solution = _distance_to(solution_mask)
    to_reference = _distance_to(np.maximum(outline_mask, detail_mask))

    def coverage(mask: np.ndarray) -> float | None:
        on = mask > 0
        return float((to_solution[on] <= tolerance).mean()) if on.any() else None

    samples = _sample_segments(puzzle)
    accuracy = float((to_reference[samples[:, 1], samples[:, 0]] <= tolerance).mean()) if len(samples) else 0.0

    parts = ()
    if plan is not None and plan.must_include:
        from dot2dot.guidance import part_coverage

        parts = tuple(part_coverage(plan, solution_mask, reference, tolerance))

    return Quality(
        outline_coverage=coverage(outline_mask),
        detail_coverage=coverage(detail_mask),
        accuracy=accuracy,
        crowded_labels=count_crowded_labels(puzzle),
        total_dots=len(puzzle.dots),
        parts=parts,
    )


def coverage_image(puzzle: Puzzle, reference: list[Path]) -> np.ndarray:
    """A picture of the check: covered lines gray, missed lines magenta, solution red."""
    tolerance = TOLERANCE_FRACTION * max(puzzle.width, puzzle.height)
    outline_mask, detail_mask = _reference_masks(puzzle, reference, thickness=3)
    reference_mask = np.maximum(outline_mask, detail_mask) > 0
    covered = _distance_to(_solution_mask(puzzle)) <= tolerance

    image = np.full((puzzle.height, puzzle.width, 3), 255, np.uint8)
    image[reference_mask & covered] = COVERED_COLOR
    image[reference_mask & ~covered] = MISSING_COLOR
    for a, b in puzzle.segments():
        cv2.line(image, (round(a.x), round(a.y)), (round(b.x), round(b.y)), SOLUTION_COLOR, 1, cv2.LINE_AA)
    return image


def count_crowded_labels(puzzle: Puzzle) -> int:
    """Numbers whose box overlaps another number's box or another dot."""
    font = puzzle.font_size
    centers = np.array([label_center(d, font) for d in puzzle.dots])
    half_w = np.array([DIGIT_WIDTH * font * len(str(d.number)) / 2 for d in puzzle.dots])
    half_h = DIGIT_HEIGHT * font / 2
    dots = np.array([(d.x, d.y) for d in puzzle.dots])

    # Label vs label: boxes overlap when they overlap on both axes.
    dx = np.abs(centers[:, None, 0] - centers[None, :, 0]) < (half_w[:, None] + half_w[None, :])
    dy = np.abs(centers[:, None, 1] - centers[None, :, 1]) < 2 * half_h
    overlaps = dx & dy
    np.fill_diagonal(overlaps, False)

    # Label vs dot (any dot other than its own).
    radius = DOT_RADIUS * font
    near_x = np.abs(centers[:, None, 0] - dots[None, :, 0]) < half_w[:, None] + radius
    near_y = np.abs(centers[:, None, 1] - dots[None, :, 1]) < half_h + radius
    covers_dot = near_x & near_y
    np.fill_diagonal(covers_dot, False)

    return int((overlaps.any(axis=1) | covers_dot.any(axis=1)).sum())


def _reference_masks(puzzle: Puzzle, reference: list[Path], thickness: int = 1):
    min_length = MIN_REFERENCE_FRACTION * max(puzzle.width, puzzle.height)
    outline = np.zeros((puzzle.height, puzzle.width), np.uint8)
    detail = np.zeros_like(outline)
    for path in reference:
        if path.length() < min_length:
            continue
        target = outline if path.essential else detail
        cv2.polylines(target, [path.points.astype(np.int32)], path.closed, 255, thickness)
    return outline, detail


def _solution_mask(puzzle: Puzzle) -> np.ndarray:
    mask = np.zeros((puzzle.height, puzzle.width), np.uint8)
    for a, b in puzzle.segments():
        cv2.line(mask, (round(a.x), round(a.y)), (round(b.x), round(b.y)), 255, 1)
    return mask


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """For every pixel, the distance to the nearest white pixel in `mask`."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    # distanceTransform measures distance to the nearest *zero* pixel, so invert.
    return cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)


def _sample_segments(puzzle: Puzzle, step: float = 2.0) -> np.ndarray:
    """Points every `step` pixels along all solution segments, as integer (x, y)."""
    samples = []
    for a, b in puzzle.segments():
        count = max(2, int(np.hypot(b.x - a.x, b.y - a.y) / step))
        t = np.linspace(0, 1, count)[:, None]
        samples.append(np.array([a.x, a.y]) * (1 - t) + np.array([b.x, b.y]) * t)
    if not samples:
        return np.zeros((0, 2), int)
    points = np.round(np.concatenate(samples)).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, puzzle.width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, puzzle.height - 1)
    return points
