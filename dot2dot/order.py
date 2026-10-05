"""Stage 4: decide which dot is #1 and the order everything is drawn in.

Rules:
- Loops are numbered clockwise, so the drawing feels natural.
- The puzzle starts on the main path (the silhouette if there is one,
  otherwise the longest path), at
  its topmost dot for a loop or its higher end for an open line.
- After finishing a stroke, jump to the nearest unfinished stroke. A loop can
  be entered at any of its dots; an open line from either end.
"""

from dataclasses import replace

import numpy as np

from dot2dot.models import Path


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


def _entry(path: Path, pen: np.ndarray) -> tuple[float, np.ndarray]:
    """Best way to start drawing `path` from pen position `pen`.

    Returns (distance to travel, points re-ordered to start there).
    """
    points = path.points
    if path.closed:
        distances = np.linalg.norm(points - pen, axis=1)
        i = int(np.argmin(distances))
        return float(distances[i]), rotate_to_start(points, i)
    to_first = float(np.linalg.norm(points[0] - pen))
    to_last = float(np.linalg.norm(points[-1] - pen))
    return (to_first, points) if to_first <= to_last else (to_last, points[::-1])


def order_paths(paths: list[Path]) -> list[Path]:
    """Return paths in drawing order, each re-ordered to start at its first dot."""
    remaining = [replace(p, points=make_clockwise(p.points)) if p.closed else p for p in paths]

    first = remaining.pop(max(range(len(remaining)), key=lambda i: (remaining[i].essential, remaining[i].length())))
    points = first.points
    if first.closed:
        points = rotate_to_start(points, topmost_index(points))
    elif points[-1][1] < points[0][1]:
        points = points[::-1]
    ordered = [replace(first, points=points)]
    pen = points[0] if first.closed else points[-1]

    while remaining:
        entries = [_entry(p, pen) for p in remaining]
        best = min(range(len(entries)), key=lambda i: entries[i][0])
        path = remaining.pop(best)
        points = entries[best][1]
        ordered.append(replace(path, points=points))
        # Loops end back where they started; open lines end at their far end.
        pen = points[0] if path.closed else points[-1]
    return ordered
