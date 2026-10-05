import cv2
import numpy as np
from skimage.morphology import skeletonize

from dot2dot.skeleton import trace_skeleton


def skeleton_of(draw) -> np.ndarray:
    img = np.zeros((200, 200), np.uint8)
    draw(img)
    return skeletonize(img > 0)


def test_ring_becomes_one_closed_path():
    paths = trace_skeleton(skeleton_of(lambda img: cv2.circle(img, (100, 100), 60, 255, 3)))
    assert len(paths) == 1
    assert paths[0].closed
    assert abs(paths[0].length() - 2 * np.pi * 60) < 40


def test_straight_line_is_one_open_path():
    paths = trace_skeleton(skeleton_of(lambda img: cv2.line(img, (20, 100), (180, 100), 255, 3)))
    assert len(paths) == 1
    assert not paths[0].closed


def test_crossing_lines_stay_whole():
    def draw(img):
        cv2.line(img, (20, 100), (180, 100), 255, 3)
        cv2.line(img, (100, 20), (100, 180), 255, 3)

    paths = trace_skeleton(skeleton_of(draw))
    # Each line passes straight through the junction, so we get two long
    # strokes rather than four short arms.
    assert len(paths) == 2
    assert all(p.length() > 140 for p in paths)
