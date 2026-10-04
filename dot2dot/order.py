"""Stage 4: decide which dot is #1 and which way the numbering runs.

Rules for Phase 1:
- Each outline is numbered clockwise, so the drawing feels natural.
- The first outline starts at its topmost point.
- When there are several outlines, after finishing one we jump to the
  nearest point on the closest remaining outline (fewest long jumps).
"""

import numpy as np


def signed_area(points: np.ndarray) -> float:
    """Shoelace formula. In image coordinates (y points down), a positive
    result means the points run clockwise on screen."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def make_clockwise(points: np.ndarray) -> np.ndarray:
    return points if signed_area(points) >= 0 else points[::-1]


def rotate_to_start(points: np.ndarray, start: int) -> np.ndarray:
    """Re-index a closed loop so `points[start]` becomes the first point."""
    return np.roll(points, -start, axis=0)


def topmost_index(points: np.ndarray) -> int:
    # lexsort sorts by the last key first: y, then x as a tie-breaker.
    return int(np.lexsort((points[:, 0], points[:, 1]))[0])


def order_outlines(outlines: list[np.ndarray]) -> list[np.ndarray]:
    """Return the outlines in drawing order, each rotated to its start dot."""
    remaining = [make_clockwise(o) for o in outlines]

    first = remaining.pop(0)
    ordered = [rotate_to_start(first, topmost_index(first))]

    while remaining:
        # Closed loops end where they start, so the pen is back at point 0.
        pen = ordered[-1][0]
        best_outline, best_index, best_distance = 0, 0, float("inf")
        for i, outline in enumerate(remaining):
            distances = np.linalg.norm(outline - pen, axis=1)
            j = int(np.argmin(distances))
            if distances[j] < best_distance:
                best_outline, best_index, best_distance = i, j, distances[j]
        nxt = remaining.pop(best_outline)
        ordered.append(rotate_to_start(nxt, best_index))

    return ordered
