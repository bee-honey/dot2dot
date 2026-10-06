import cv2
import numpy as np

from dot2dot import sketch
from dot2dot.pipeline import build


def test_pencil_sketch_turns_solid_black_into_an_outline():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (80, 80), (220, 220), (0, 0, 0), -1)  # a solid black square
    sk = sketch.pencil_sketch(img)
    assert sk[150, 150] > 200  # the inside comes out white (paper)
    assert sk[150, 78:84].min() < 120  # the edge is a dark line


def test_color_edges_find_yellow_on_white():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(img, (150, 150), 100, (60, 210, 250), -1)  # yellow disc: almost white in grayscale
    lines = sketch.line_drawing(img)
    ring = np.zeros_like(lines)
    cv2.circle(ring, (150, 150), 100, 255, 7)
    assert (lines[ring > 0] > 0).mean() > 0.5  # the disc's edge is found


def test_sketch_style_builds_a_puzzle():
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.circle(img, (200, 200), 150, (0, 0, 0), 6)
    cv2.circle(img, (150, 160), 20, (0, 0, 0), 2)  # thin interior details
    cv2.circle(img, (250, 160), 20, (0, 0, 0), 2)
    result = build(img, num_dots=60, max_paths=None, style="sketch")
    assert result.style == "sketch"
    assert 40 <= len(result.puzzle.dots) <= 60
    assert len(result.puzzle.strokes) >= 2  # outline plus interior details
