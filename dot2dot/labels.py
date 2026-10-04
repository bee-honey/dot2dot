"""Stage 5: choose which side of each dot its number goes on.

Phase 1 rule: put the number just outside the shape, along the outward
normal (the direction perpendicular to the outline, pointing away from it).
Collision-free placement comes in Phase 2.
"""

import cv2
import numpy as np


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
