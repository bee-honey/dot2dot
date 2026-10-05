"""Stage 3: reduce a dense path (thousands of pixels) to N well-placed dots.

Strategy:
1. Find real corners: points where the line turns sharply over a short
   distance. Every corner gets a dot, exactly on it.
2. Between corners, space dots evenly along the line, with extra density
   where it curves. We measure position along the path in "effort":
       effort = distance travelled + SPACING_PER_RADIAN * amount of turning
   and place dots at equal steps of effort. A straight run gets dots by
   length alone; a tight curve (an eye, a fingertip) gets extra dots so it
   stays round; a gentle curve gets evenly spaced dots, never clumps.

Every dot is an actual pixel from the original path, so dots always sit
exactly on the shape. Open paths always keep both end points.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from dot2dot.models import Path

MIN_DOTS_CLOSED = 3
MIN_DOTS_OPEN = 2
# A path is only kept if it would earn at least this many dots.
# Lines worth just 1-2 dots add clutter (and a "lift your pencil" ring) for little gain.
MIN_EARNED_DOTS = 2.5
# Each radian of turning costs this many average dot spacings of "effort",
# so a small loop (2π) earns about 3 extra dots on top of its length share.
# Mostly matters for small loops like eyes; kept modest so busy drawings
# don't take dots away from the silhouette.
SPACING_PER_RADIAN = 0.5
# A turn sharper than this (measured over a short window) is a corner.
CORNER_ANGLE = np.radians(45)
# Pixel outlines are staircases; smooth this much (in pixels) before
# measuring direction, so pixel steps and wobbles don't count as turning.
SMOOTHING = 6
# Dots closer than this fraction of the image size are hard to read, so we
# never place more dots than the lines can hold at this spacing.
MIN_SPACING_FRACTION = 1 / 70


def min_dots(path: Path) -> int:
    return MIN_DOTS_CLOSED if path.closed else MIN_DOTS_OPEN


def turning(path: Path) -> float:
    """Total amount the path turns, in radians (a circle turns 2π)."""
    return float(_step_turns(path).sum())


def _cost(path: Path, spacing: float) -> float:
    """How many dots' worth of effort a path needs, before weighting."""
    return path.length() + SPACING_PER_RADIAN * spacing * turning(path)


def allocate_dots(paths: list[Path], total: int) -> list[int]:
    """Split `total` dots across paths by length, curviness and importance."""
    spacing = sum(p.length() for p in paths) / max(total, 1)
    costs = np.array([_cost(p, spacing) * p.weight for p in paths])
    shares = costs / costs.sum() * total
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


def declutter(paths: list[Path], image_size: int) -> list[Path]:
    """Drop dots that sit almost on top of a dot from an earlier path.

    Where lines meet (an eye corner, a junction), every line ends with its
    own dot at the same spot, and the stacked numbers become unreadable.
    Paths are processed in drawing order; the first one to claim a spot keeps it.
    """
    radius = 0.6 * MIN_SPACING_FRACTION * image_size
    claimed = np.zeros((0, 2))
    result = []
    for path in paths:
        points = path.points
        if len(claimed):
            nearest = np.linalg.norm(points[:, None, :] - claimed[None, :, :], axis=2).min(axis=1)
            keep = nearest >= radius
            if keep.sum() >= min_dots(path):
                points = points[keep]
            elif keep.sum() == 0:
                continue  # entirely on top of other lines; nothing new to draw
        result.append(replace(path, points=points))
        claimed = np.concatenate([claimed, points])
    return result


def max_readable_dots(paths: list[Path], image_size: int) -> int:
    """The most dots these paths can hold while staying legible."""
    return max(int(sum(p.length() for p in paths) / (MIN_SPACING_FRACTION * image_size)), 3)


def simplify(path: Path, n: int) -> np.ndarray:
    """Pick about `n` points along `path`: corners first, the rest evenly. Returns (k, 2)."""
    points = path.points
    if len(points) <= n:
        return points.copy()

    count = len(points)
    step_lengths = _step_lengths(path)
    spacing = step_lengths.sum() / n
    effort = np.concatenate([[0.0], np.cumsum(step_lengths + SPACING_PER_RADIAN * spacing * _step_turns(path))])
    total_effort = effort[-1]

    anchors = _anchors(path, n, spacing)
    # Distribute the remaining dots across the gaps between anchors, in
    # proportion to how much effort each gap takes.
    gaps = []
    for i, start in enumerate(anchors):
        if i + 1 < len(anchors):
            end_effort = effort[anchors[i + 1]]
        elif path.closed:
            end_effort = total_effort + effort[anchors[0]]  # wrap around to the first anchor
        else:
            break
        gaps.append((effort[start], end_effort))
    sizes = np.array([b - a for a, b in gaps])
    extra = _split(n - len(anchors), sizes)

    indices = []
    for anchor_index, (a, b), k in zip(anchors, gaps, extra):
        indices.append(anchor_index)
        for target in a + (b - a) * np.arange(1, k + 1) / (k + 1):
            indices.append(_index_at(effort, target % total_effort if path.closed else target, count))
    if not path.closed:
        indices.append(anchors[-1])
    return points[_dedupe(indices)]


def _step_lengths(path: Path) -> np.ndarray:
    """Distance from each point to the next (including last→first for loops)."""
    points = path.points
    following = np.roll(points, -1, axis=0) if path.closed else points[1:]
    return np.linalg.norm(following - points[: len(following)], axis=1)


def _step_turns(path: Path) -> np.ndarray:
    """How much the direction changes at each step, in radians (always >= 0)."""
    points = path.points
    if len(points) < 3:
        return np.zeros(len(points) if path.closed else max(len(points) - 1, 0))
    mode = "wrap" if path.closed else "nearest"
    smooth = gaussian_filter1d(points, SMOOTHING, axis=0, mode=mode)
    following = np.roll(smooth, -1, axis=0) if path.closed else smooth[1:]
    deltas = following - smooth[: len(following)]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    if path.closed:
        change = np.roll(angles, -1) - angles
    else:
        change = np.append(np.diff(angles), 0.0)
    # Wrap into [-π, π] so going from 179° to -179° counts as 2°, not 358°.
    return np.abs((change + np.pi) % (2 * np.pi) - np.pi)


def _anchors(path: Path, n: int, spacing: float) -> list[int]:
    """Indices that must get a dot: corners, plus both ends of an open path."""
    points = path.points
    count = len(points)
    arc = np.concatenate([[0.0], np.cumsum(_step_lengths(path))])
    length = arc[-1]
    window = float(np.clip(spacing * 0.5, 4, 15))

    # Turning angle at each point, measured `window` pixels back and forward.
    if path.closed:
        extended = np.concatenate([arc[:-1] - length, arc[:-1], arc[:-1] + length])
        positions = arc[:-1]
        back = np.searchsorted(extended, positions - window) % count
        ahead = np.searchsorted(extended, positions + window) % count
    else:
        positions = arc
        back = np.clip(np.searchsorted(arc, positions - window), 0, count - 1)
        ahead = np.clip(np.searchsorted(arc, positions + window), 0, count - 1)
    v_in = points - points[back]
    v_out = points[ahead] - points
    norms = np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1)
    cosine = np.einsum("ij,ij->i", v_in, v_out) / np.where(norms == 0, 1, norms)
    angle = np.where(norms == 0, 0.0, np.arccos(np.clip(cosine, -1, 1)))

    # Keep the sharpest corners, at least one window apart.
    corners: list[int] = []
    for i in np.argsort(angle)[::-1]:
        if angle[i] < CORNER_ANGLE:
            break
        if corners:
            gap = np.abs(positions[i] - positions[corners])
            if path.closed:
                gap = np.minimum(gap, length - gap)  # distance the short way round the loop
            if gap.min() < window:
                continue
        corners.append(int(i))

    if path.closed:
        anchors = sorted(corners[:n]) or [0]
    else:
        ends = {0, count - 1}
        inner = [c for c in corners if window <= positions[c] <= length - window]
        anchors = sorted(ends | set(inner[: max(n - 2, 0)]))
    return anchors


def _split(total: int, sizes: np.ndarray) -> list[int]:
    """Divide `total` items across buckets in proportion to `sizes`."""
    if total <= 0 or sizes.sum() <= 0:
        return [0] * len(sizes)
    shares = sizes / sizes.sum() * total
    counts = np.floor(shares).astype(int)
    for i in np.argsort(shares - counts)[::-1][: total - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def _index_at(effort: np.ndarray, target: float, count: int) -> int:
    return int(min(np.searchsorted(effort, target), len(effort) - 1)) % count


def _dedupe(indices: list[int]) -> list[int]:
    seen, result = set(), []
    for i in indices:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result
