"""Data types passed between pipeline stages."""

from dataclasses import dataclass

import numpy as np


# eq=False: compare paths by identity. The generated field-by-field __eq__
# would compare NumPy arrays, which don't give a single True/False.
@dataclass(frozen=True, eq=False)
class Path:
    """A traced line in image pixel coordinates, before it becomes dots.

    `points` is an (N, 2) array of (x, y). A closed path is a loop (an outline);
    an open path has two free ends (a line inside the drawing).
    """

    points: np.ndarray
    closed: bool
    # Essential paths (like the subject's silhouette) are never dropped.
    essential: bool = False
    # Importance multiplier: a path with weight 3 gets dots as if it were
    # three times as long (used to give faces more detail).
    weight: float = 1.0

    def length(self) -> float:
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum()
        if self.closed:
            steps += np.linalg.norm(self.points[-1] - self.points[0])
        return float(steps)

    def weighted_length(self) -> float:
        return self.length() * self.weight


@dataclass(frozen=True)
class Dot:
    """A single numbered dot, in image pixel coordinates."""

    number: int
    x: float
    y: float
    # Unit vector from the dot toward where its number is printed.
    label_dx: float = 0.0
    label_dy: float = -1.0
    # True for the first dot of every line after the first (drawn with a ring).
    starts_stroke: bool = False


@dataclass(frozen=True)
class Stroke:
    """A run of consecutive dots drawn without lifting the pencil.

    Covers dots[start:end]. A closed stroke ends by joining back to its first dot.
    """

    start: int
    end: int
    closed: bool


@dataclass(frozen=True)
class Puzzle:
    """A finished puzzle: dots in drawing order plus the canvas size."""

    width: int
    height: int
    dots: list[Dot]
    strokes: list[Stroke]
    # Label font size in image pixels; renderers scale it with everything else.
    font_size: float

    def segments(self) -> list[tuple[Dot, Dot]]:
        """Pairs of dots that should be connected in the solution."""
        pairs = []
        for stroke in self.strokes:
            run = self.dots[stroke.start : stroke.end]
            pairs.extend(zip(run, run[1:]))
            if stroke.closed and len(run) > 2:
                pairs.append((run[-1], run[0]))
        return pairs

    def stroke_start_dots(self) -> list[Dot]:
        """First dot of every stroke after the first: where to lift the pencil."""
        return [d for d in self.dots if d.starts_stroke]
