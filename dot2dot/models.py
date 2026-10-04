"""Data types passed between pipeline stages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dot:
    """A single numbered dot, in image pixel coordinates."""

    number: int
    x: float
    y: float
    # Unit vector pointing away from the shape; used to place the label
    # outside the outline instead of on top of the line.
    label_dx: float = 0.0
    label_dy: float = -1.0


@dataclass(frozen=True)
class Puzzle:
    """A finished puzzle: dots in drawing order plus the canvas size."""

    width: int
    height: int
    dots: list[Dot]
    # Index into `dots` where each new stroke begins. The solution draws
    # dots[i-1] -> dots[i] unless i is a stroke start, in which case the
    # pen "lifts" (no line is drawn).
    stroke_starts: list[int]

    def segments(self) -> list[tuple[Dot, Dot]]:
        """Pairs of dots that should be connected in the solution."""
        starts = set(self.stroke_starts)
        pairs = []
        for i in range(1, len(self.dots)):
            if i not in starts:
                pairs.append((self.dots[i - 1], self.dots[i]))
        # Close each stroke back to its first dot (outlines are loops).
        boundaries = self.stroke_starts + [len(self.dots)]
        for start, end in zip(boundaries, boundaries[1:]):
            if end - start > 2:
                pairs.append((self.dots[end - 1], self.dots[start]))
        return pairs
