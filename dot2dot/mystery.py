"""Mystery mode: keep the picture hidden until the dots are connected.

A normal puzzle gives itself away: dots crowd along the subject's outline
and the rest of the page is empty, so the dot cloud already has the
subject's shape. Mystery mode:

1. Filler strokes: gentle curved lines in the empty background, numbered
   like everything else, so dots cover the whole page evenly. When solved
   they read as background texture (drawn light blue in the answer key).
2. Reveal order: numbering starts in a corner of the page instead of on the
   silhouette, and small feature loops (eyes, buttons) come last, so the
   picture emerges late in the drawing.

Levels: 0 = off; 1 = fillers in the background; 2 = more fillers, also in
empty areas inside the subject, and small give-away loops get fewer dots.
"""

import hashlib

import cv2
import numpy as np

from dot2dot.models import Path

# Share of all dots that go to filler strokes, per level.
FILLER_SHARE = {0: 0.0, 1: 0.35, 2: 0.5}
# Fillers keep this many dot-spacings away from the picture's lines.
CLEARANCE = 0.5
# At level 2, feature loops (eyes) get this fraction of their usual dots.
FEATURE_WEIGHT = 0.5
# Filler stroke length, in dot spacings.
MIN_LENGTH, MAX_LENGTH = 2, 6
# How sharply fillers may bend (radians per pixel) and how quickly that changes.
MAX_BEND, BEND_JITTER = 0.02, 0.004
# A closed path this many dot spacings around or less counts as a feature
# (an eye, a button) and is numbered last.
FEATURE_SIZE = 12


def add_fillers(paths: list[Path], shape: tuple[int, ...], num_dots: int, level: int, seed: bytes = b"") -> list[Path]:
    """Filler strokes to add so roughly FILLER_SHARE[level] of the dots are filler."""
    share = FILLER_SHARE.get(level, 0.0)
    if share <= 0 or not paths:
        return []
    height, width = shape[:2]
    picture_length = sum(p.length() for p in paths)
    spacing = picture_length / max(num_dots * (1 - share), 1)
    target_length = picture_length * share / (1 - share)

    # Where fillers may go: away from the picture's lines and its interior,
    # and inside the page margins.
    blocked = np.zeros((height, width), np.uint8)
    for path in paths:
        cv2.polylines(blocked, [path.points.astype(np.int32)], path.closed, 255, max(1, int(2 * CLEARANCE * spacing)))
        # Level 1 keeps fillers out of the subject; level 2 may use its empty areas.
        if path.essential and path.closed and level < 2:
            cv2.fillPoly(blocked, [path.points.astype(np.int32)], 255)
    margin = int(spacing)
    blocked[:margin], blocked[-margin:], blocked[:, :margin], blocked[:, -margin:] = 255, 255, 255, 255

    # Seeded by the image, so the same picture always gets the same puzzle.
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "little"))
    fillers: list[Path] = []
    total = 0.0
    misses = 0
    while total < target_length and misses < 3000:
        # Pick random spots until one is free (cheaper than listing free pixels).
        x, y = int(rng.integers(width)), int(rng.integers(height))
        if blocked[y, x]:
            misses += 1
            continue
        stroke = _wander(np.array([x, y], float), blocked, spacing, rng)
        if stroke is None:
            misses += 1
            continue
        fillers.append(Path(stroke, closed=False, filler=True))
        total += fillers[-1].length()
        # Reserve the space around this stroke so the next one keeps its distance.
        cv2.polylines(blocked, [stroke.astype(np.int32)], False, 255, max(1, int(2 * CLEARANCE * spacing)))
    return fillers


def _wander(start: np.ndarray, blocked: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray | None:
    """Trace a gently curving line from `start` until it runs out of room."""
    height, width = blocked.shape
    target = rng.uniform(MIN_LENGTH, MAX_LENGTH) * spacing
    heading = rng.uniform(0, 2 * np.pi)
    bend = 0.0
    points = [start]
    position = start.copy()
    while len(points) < target:
        bend = float(np.clip(bend + rng.normal(0, BEND_JITTER), -MAX_BEND, MAX_BEND))
        heading += bend
        position = position + (np.cos(heading), np.sin(heading))
        x, y = int(position[0]), int(position[1])
        if not (0 <= x < width and 0 <= y < height) or blocked[y, x]:
            break
        points.append(position)
    if len(points) < MIN_LENGTH * spacing:
        return None
    return np.array(points)


def is_feature(path: Path, spacing: float) -> bool:
    """Small closed loops inside the picture (eyes, buttons): numbered last."""
    return path.closed and not path.essential and not path.filler and path.length() <= FEATURE_SIZE * spacing


def dot_spread(points: np.ndarray, width: int, height: int, cells: int = 6) -> float:
    """How evenly dots cover the page: 1 = perfectly even, toward 0 = clumped.

    Counts dots in a 6x6 grid and compares the counts' spread to their mean
    (1 / (1 + coefficient of variation)).
    """
    if len(points) == 0:
        return 0.0
    xs = np.clip((points[:, 0] / width * cells).astype(int), 0, cells - 1)
    ys = np.clip((points[:, 1] / height * cells).astype(int), 0, cells - 1)
    counts = np.bincount(ys * cells + xs, minlength=cells * cells)
    return float(1 / (1 + counts.std() / counts.mean()))
