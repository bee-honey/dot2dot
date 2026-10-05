"""Stage 3: reduce a dense path (thousands of pixels) to exactly N dots.

Strategy:
1. Douglas-Peucker (cv2.approxPolyDP) keeps the "important" points: corners
   and sharp bends. We binary-search its tolerance so it returns <= N points.
2. Pixel outlines are staircases, so a single corner often comes back as a
   clump of 2-3 near-identical points. Merge each clump into one dot.
3. Fill in the longest gaps along the path until there are N dots, so
   they stay evenly spread on long edges and smooth curves.

Every dot we return is an actual pixel from the original path, so dots
always sit exactly on the shape. Open paths always keep both end points.
"""

import cv2
import numpy as np

from dot2dot.models import Path

MIN_DOTS_CLOSED = 3
MIN_DOTS_OPEN = 2
# A path is only kept if its length would earn at least this many dots.
# Lines worth just 1-2 dots add clutter (and a "lift your pencil" ring) for little gain.
MIN_EARNED_DOTS = 2.5
# Dots closer than this fraction of the average spacing get merged.
MIN_SPACING_FRACTION = 0.35


def min_dots(path: Path) -> int:
    return MIN_DOTS_CLOSED if path.closed else MIN_DOTS_OPEN


def allocate_dots(paths: list[Path], total: int) -> list[int]:
    """Split `total` dots across paths in proportion to their (weighted) length."""
    lengths = np.array([p.weighted_length() for p in paths])
    shares = lengths / lengths.sum() * total
    counts = np.floor(shares).astype(int)
    # Hand the leftover dots to the paths that lost the most to rounding.
    leftover = total - counts.sum()
    for i in np.argsort(shares - counts)[::-1][:leftover]:
        counts[i] += 1
    return [max(int(c), min_dots(p)) for c, p in zip(counts, paths)]


def select_paths(paths: list[Path], total: int, max_paths: int | None = None) -> list[Path]:
    """Keep the longest paths; drop ones too short to earn a few dots.

    With 60 dots spread over 10,000 px of lines, each dot "covers" ~170 px,
    so a 100 px squiggle isn't worth including.
    """
    # Essential paths first, then the rest from longest to shortest.
    paths = sorted(paths, key=lambda p: (p.essential, p.weighted_length()), reverse=True)
    if max_paths is not None:
        paths = paths[:max_paths]
    while len(paths) > 1 and not paths[-1].essential:
        spacing = sum(p.weighted_length() for p in paths) / total
        if paths[-1].weighted_length() >= spacing * max(MIN_EARNED_DOTS, min_dots(paths[-1]) - 0.5):
            break
        paths.pop()
    return paths


def simplify(path: Path, n: int) -> np.ndarray:
    """Pick exactly `n` points from `path`, preserving corners. Returns (n, 2)."""
    points = path.points
    if len(points) <= n:
        return points.copy()

    arc, length = _arc_lengths(points, path.closed)
    epsilon = _find_epsilon(points, n, path.closed)
    indices = _douglas_peucker_indices(points, epsilon, path.closed)
    min_spacing = length / n * MIN_SPACING_FRACTION
    indices = _merge_clumps(indices, arc, length, path.closed, min_spacing)
    indices = _fill_gaps(points, indices, n, arc, length, path.closed)
    return points[indices]


def _arc_lengths(points: np.ndarray, closed: bool) -> tuple[np.ndarray, float]:
    """arc[k] = distance travelled along the path from point 0 to point k.

    Also returns the total length (including the closing edge for loops).
    """
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    length = arc[-1] + (np.linalg.norm(points[-1] - points[0]) if closed else 0.0)
    return arc, float(length)


def _approx(points: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
    return cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2)


def _douglas_peucker_indices(points: np.ndarray, epsilon: float, closed: bool) -> list[int]:
    """Run Douglas-Peucker and return the indices of the points it kept."""
    approx = _approx(points, epsilon, closed)
    # approxPolyDP returns coordinates, not indices. Each returned point is a
    # copy of an original point, so find its position by nearest match.
    distances = np.linalg.norm(points[None, :, :] - approx[:, None, :], axis=2)
    indices = set(np.argmin(distances, axis=1).tolist())
    if not closed:
        indices |= {0, len(points) - 1}
    return sorted(indices)


def _find_epsilon(points: np.ndarray, n: int, closed: bool) -> float:
    """Binary-search the smallest tolerance that yields at most `n` points.

    Larger epsilon = coarser approximation = fewer points.
    """
    low, high = 0.0, _arc_lengths(points, closed)[1] / 4
    for _ in range(30):
        mid = (low + high) / 2
        if len(_approx(points, mid, closed)) <= n:
            high = mid
        else:
            low = mid
    return high


def _merge_clumps(
    indices: list[int], arc: np.ndarray, length: float, closed: bool, min_spacing: float
) -> list[int]:
    """Replace each run of closely spaced points with one representative."""
    last = len(arc) - 1
    clumps: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if arc[index] - arc[clumps[-1][-1]] < min_spacing:
            clumps[-1].append(index)
        else:
            clumps.append([index])

    def representative(clump: list[int]) -> int:
        # Open paths must keep their end points; otherwise use the middle.
        if not closed and 0 in clump:
            return 0
        if not closed and last in clump:
            return last
        return clump[len(clump) // 2]

    merged = [representative(clump) for clump in clumps]
    if closed:
        # The path is a loop: the last dot may be crowding the first one.
        if len(merged) > 1 and length - arc[merged[-1]] + arc[merged[0]] < min_spacing:
            merged.pop()
    elif merged[-1] != last:
        merged.append(last)
    return merged


def _fill_gaps(
    points: np.ndarray, indices: list[int], n: int, arc: np.ndarray, length: float, closed: bool
) -> list[int]:
    """Add points at the midpoint of the longest gaps until there are `n`."""
    total_points = len(points)
    indices = sorted(indices)
    while len(indices) < n:
        best_gap, best_mid = -1.0, None
        # Loops also have a gap from the last dot back around to the first.
        gap_count = len(indices) if closed else len(indices) - 1
        for i in range(gap_count):
            start, end = indices[i], indices[(i + 1) % len(indices)]
            span = (end - start) % total_points or total_points
            if span < 2:
                continue  # Adjacent pixels; nothing to insert between them.
            gap = (arc[end] - arc[start]) % length or length
            if gap > best_gap:
                target = (arc[start] + gap / 2) % length
                mid = int(np.searchsorted(arc, target)) % total_points
                if mid not in (start, end):
                    best_gap, best_mid = gap, mid
        if best_mid is None:
            break
        indices = sorted(indices + [best_mid])
    return indices
