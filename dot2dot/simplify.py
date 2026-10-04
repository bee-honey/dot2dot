"""Stage 3: reduce a dense outline (thousands of pixels) to exactly N dots.

Strategy:
1. Douglas-Peucker (cv2.approxPolyDP) keeps the "important" points: corners
   and sharp bends. We binary-search its tolerance so it returns <= N points.
2. Pixel outlines are staircases, so a single corner often comes back as a
   clump of 2-3 near-identical points. Merge each clump into one dot.
3. Fill in the longest gaps along the outline until there are N dots, so
   they stay evenly spread on long edges and smooth curves.

Every dot we return is an actual pixel from the original outline, so dots
always sit exactly on the shape.
"""

import cv2
import numpy as np

MIN_DOTS_PER_OUTLINE = 3
# Dots closer than this fraction of the average spacing get merged.
MIN_SPACING_FRACTION = 0.35


def perimeter(outline: np.ndarray) -> float:
    return cv2.arcLength(outline.astype(np.float32), closed=True)


def allocate_dots(outlines: list[np.ndarray], total: int) -> list[int]:
    """Split `total` dots across outlines in proportion to their perimeter."""
    lengths = np.array([perimeter(o) for o in outlines])
    shares = lengths / lengths.sum() * total
    counts = np.floor(shares).astype(int)
    # Hand the leftover dots to the outlines that lost the most to rounding.
    leftover = total - counts.sum()
    for i in np.argsort(shares - counts)[::-1][:leftover]:
        counts[i] += 1
    return [max(int(c), MIN_DOTS_PER_OUTLINE) for c in counts]


def simplify(outline: np.ndarray, n: int) -> np.ndarray:
    """Pick exactly `n` points from `outline`, preserving corners. Returns (n, 2)."""
    if len(outline) <= n:
        return outline.copy()

    arc, loop_length = _arc_lengths(outline)
    epsilon = _find_epsilon(outline, n)
    indices = _douglas_peucker_indices(outline, epsilon)
    indices = _merge_clumps(indices, arc, loop_length, min_spacing=loop_length / n * MIN_SPACING_FRACTION)
    indices = _fill_gaps(outline, indices, n, arc, loop_length)
    return outline[indices]


def _arc_lengths(outline: np.ndarray) -> tuple[np.ndarray, float]:
    """arc[k] = distance travelled along the outline from point 0 to point k.

    Also returns the total length of the closed loop.
    """
    steps = np.linalg.norm(np.diff(outline, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    loop_length = arc[-1] + np.linalg.norm(outline[-1] - outline[0])
    return arc, float(loop_length)


def _merge_clumps(
    indices: list[int], arc: np.ndarray, loop_length: float, min_spacing: float
) -> list[int]:
    """Replace each run of closely spaced points with its middle point."""
    clumps: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if arc[index] - arc[clumps[-1][-1]] < min_spacing:
            clumps[-1].append(index)
        else:
            clumps.append([index])
    merged = [clump[len(clump) // 2] for clump in clumps]
    # The outline is a loop: the last dot may be crowding the first one.
    if len(merged) > 1 and loop_length - arc[merged[-1]] + arc[merged[0]] < min_spacing:
        merged.pop()
    return merged


def _douglas_peucker_indices(outline: np.ndarray, epsilon: float) -> list[int]:
    """Run Douglas-Peucker and return the indices of the points it kept."""
    approx = cv2.approxPolyDP(outline.astype(np.float32).reshape(-1, 1, 2), epsilon, closed=True)
    approx = approx.reshape(-1, 2)
    # approxPolyDP returns coordinates, not indices. Each returned point is a
    # copy of an original point, so find its position by nearest match.
    distances = np.linalg.norm(outline[None, :, :] - approx[:, None, :], axis=2)
    return sorted(set(np.argmin(distances, axis=1).tolist()))


def _find_epsilon(outline: np.ndarray, n: int) -> float:
    """Binary-search the smallest tolerance that yields at most `n` points.

    Larger epsilon = coarser approximation = fewer points.
    """
    low, high = 0.0, perimeter(outline) / 4
    for _ in range(30):
        mid = (low + high) / 2
        count = len(cv2.approxPolyDP(outline.astype(np.float32).reshape(-1, 1, 2), mid, closed=True))
        if count <= n:
            high = mid
        else:
            low = mid
    return high


def _fill_gaps(
    outline: np.ndarray, indices: list[int], n: int, arc: np.ndarray, loop_length: float
) -> list[int]:
    """Add points at the midpoint of the longest gaps until there are `n`."""
    total_points = len(outline)
    indices = sorted(indices)
    while len(indices) < n:
        best_gap, best_mid = -1.0, None
        for i, start in enumerate(indices):
            end = indices[(i + 1) % len(indices)]
            span = (end - start) % total_points or total_points
            if span < 2:
                continue  # Adjacent pixels; nothing to insert between them.
            gap = (arc[end] - arc[start]) % loop_length or loop_length
            if gap > best_gap:
                target = (arc[start] + gap / 2) % loop_length
                mid = int(np.searchsorted(arc, target)) % total_points
                if mid not in (start, end):
                    best_gap, best_mid = gap, mid
        if best_mid is None:
            break
        indices = sorted(indices + [best_mid])
    return indices
