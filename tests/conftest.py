"""Shared pytest fixtures (like a JUnit base class or @BeforeEach helpers)."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def square_image(tmp_path):
    """A black square on a white background, saved to a temp file."""
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)
    path = tmp_path / "square.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def two_shapes_image(tmp_path):
    """Two separate black circles on white."""
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.circle(img, (150, 200), 100, (0, 0, 0), -1)
    cv2.circle(img, (450, 200), 60, (0, 0, 0), -1)
    path = tmp_path / "circles.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def square_outline():
    """Every pixel along the edge of a 100x100 square, clockwise from (0, 0)."""
    top = [(x, 0) for x in range(100)]
    right = [(100, y) for y in range(100)]
    bottom = [(x, 100) for x in range(100, 0, -1)]
    left = [(0, y) for y in range(100, 0, -1)]
    return np.array(top + right + bottom + left, dtype=np.float64)
