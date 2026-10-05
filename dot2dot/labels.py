"""Stage 5: choose where each dot's number is printed.

For every dot we try 16 directions around it and keep the one where the
number's box has the most clearance from other dots, from the solution
lines, and from numbers already placed. On outlines we nudge the choice
toward the outside of the shape, which is where readers expect numbers.
"""

import cv2
import numpy as np

CANDIDATE_COUNT = 16
# Approximate glyph size relative to font size (Helvetica digits).
DIGIT_WIDTH = 0.6
DIGIT_HEIGHT = 0.75
DOT_RADIUS = 0.22  # relative to font size
RING_RADIUS = DOT_RADIUS * 2.2  # ring around a dot that starts a new line
GAP = 0.15  # space between dot and number, relative to font size


def choose_font_size(points: np.ndarray, width: int, height: int) -> float:
    """Pick a font size (image pixels) that suits how tightly packed the dots are."""
    size = max(width, height)
    if len(points) < 2:
        return size / 60
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    typical_gap = float(np.median(distances.min(axis=1)))
    return float(np.clip(typical_gap * 0.6, size / 110, size / 55))


def outward_normals(points: np.ndarray) -> np.ndarray:
    """Return an (N, 2) array of unit vectors pointing out of the closed loop."""
    # Tangent at each point: direction from the previous dot to the next one.
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.where(lengths == 0, 1, lengths)

    # Rotate the tangent 90 degrees to get a normal.
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])

    # The rotation might point inward; test a point a few pixels along each
    # normal and flip the ones that land inside the shape.
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    for i, (point, normal) in enumerate(zip(points, normals)):
        probe = point + normal * 3
        if cv2.pointPolygonTest(contour, (float(probe[0]), float(probe[1])), False) > 0:
            normals[i] = -normal
    return normals


def place_labels(
    points: np.ndarray,
    numbers: list[int],
    segments: np.ndarray,
    preferred: np.ndarray,
    font_size: float,
    rings: np.ndarray,
) -> np.ndarray:
    """Return an (N, 2) array of unit directions from each dot to its label.

    points:    (N, 2) dot positions
    segments:  (M, 2, 2) solution line segments [[x1, y1], [x2, y2]]
    preferred: (N, 2) preferred direction per dot, or zeros for "no preference"
    rings:     (N,) True for dots drawn with a ring (they take up more room)
    """
    angles = np.linspace(0, 2 * np.pi, CANDIDATE_COUNT, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    radii = np.where(rings, RING_RADIUS, DOT_RADIUS) * font_size
    half_height = DIGIT_HEIGHT * font_size / 2
    cap = 1.5 * font_size  # Beyond this much clearance, extra space doesn't matter.

    placed_centers: list[np.ndarray] = []
    placed_half_widths: list[float] = []
    result = np.zeros_like(points)

    for i, (point, number) in enumerate(zip(points, numbers)):
        half_width = DIGIT_WIDTH * font_size * len(str(number)) / 2
        half_size = np.array([half_width, half_height])

        # Push each candidate box just far enough that its edge clears the dot.
        reach = np.abs(directions) @ half_size  # box extent along each direction
        centers = point + directions * (radii[i] + GAP * font_size + reach)[:, None]

        clearance = np.full(CANDIDATE_COUNT, cap)
        others = np.delete(points, i, axis=0)
        if len(others):
            gaps = _box_to_points(centers, half_size, others) - np.delete(radii, i)
            clearance = np.minimum(clearance, gaps.min(axis=1))
        if len(segments):
            clearance = np.minimum(clearance, _box_to_segments(centers, half_size, segments).min(axis=1))
        if placed_centers:
            gaps = _box_to_boxes(centers, half_size, np.array(placed_centers), np.array(placed_half_widths), half_height)
            clearance = np.minimum(clearance, gaps.min(axis=1))

        score = clearance + 0.25 * font_size * (directions @ preferred[i])
        best = int(np.argmax(score))
        result[i] = directions[best]
        placed_centers.append(centers[best])
        placed_half_widths.append(half_width)
    return result


def _box_to_points(centers: np.ndarray, half_size: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each candidate box (K) to each point (P). Returns (K, P)."""
    offset = np.abs(points[None, :, :] - centers[:, None, :]) - half_size
    return np.linalg.norm(np.maximum(offset, 0), axis=2)


def _box_to_segments(centers: np.ndarray, half_size: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Approximate distance from each candidate box (K) to each segment (M)."""
    a, b = segments[:, 0], segments[:, 1]  # (M, 2)
    ab = b - a
    ab_len2 = np.maximum((ab**2).sum(axis=1), 1e-9)
    # Closest point on each segment to each box center (projection, clamped).
    t = np.clip(((centers[:, None, :] - a) * ab).sum(axis=2) / ab_len2, 0, 1)
    closest = a + t[:, :, None] * ab
    delta = centers[:, None, :] - closest
    distance = np.linalg.norm(delta, axis=2)
    unit = delta / np.maximum(distance, 1e-9)[:, :, None]
    # Subtract how far the box extends toward the segment.
    return distance - np.abs(unit) @ half_size


def _box_to_boxes(centers, half_size, other_centers, other_half_widths, half_height) -> np.ndarray:
    """Separation between each candidate box (K) and each placed label (L); negative = overlap."""
    dx = np.abs(centers[:, None, 0] - other_centers[None, :, 0]) - (half_size[0] + other_half_widths)
    dy = np.abs(centers[:, None, 1] - other_centers[None, :, 1]) - (half_size[1] + half_height)
    return np.maximum(dx, dy)
