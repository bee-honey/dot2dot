"""Render a Puzzle as a bitmap (NumPy image), e.g. to show an AI judge.

The SVG and PDF renderers make files for people; this makes in-memory images
for code: the guess test, the evaluation grid, thumbnails.
"""

import cv2
import numpy as np

from dot2dot.labels import DIGIT_HEIGHT, DOT_RADIUS, RING_RADIUS
from dot2dot.models import Puzzle
from dot2dot.render import label_center

LINE_COLOR = (60, 60, 212)  # BGR red
FILLER_COLOR = (225, 180, 140)  # BGR light blue: background filler lines
# Hershey font glyphs are about this tall (pixels) at fontScale 1.
HERSHEY_CAP_HEIGHT = 22


def render(puzzle: Puzzle, size: int = 1024, solution: bool = False, numbers: bool = True, margin: int = 24) -> np.ndarray:
    """The puzzle page (or answer key) as a BGR image whose longest side is `size`."""
    scale = (size - 2 * margin) / max(puzzle.width, puzzle.height)
    width = round(puzzle.width * scale) + 2 * margin
    height = round(puzzle.height * scale) + 2 * margin
    canvas = np.full((height, width, 3), 255, np.uint8)

    def at(x: float, y: float) -> tuple[int, int]:
        return round(margin + x * scale), round(margin + y * scale)

    font = puzzle.font_size * scale
    radius = max(1, round(DOT_RADIUS * font))

    if solution:
        line_width = max(1, round(radius * 0.8))
        for filler, color in ((True, FILLER_COLOR), (False, LINE_COLOR)):
            for a, b in puzzle.segments(filler=filler):
                cv2.line(canvas, at(a.x, a.y), at(b.x, b.y), color, line_width, cv2.LINE_AA)

    for dot in puzzle.stroke_start_dots():
        cv2.circle(canvas, at(dot.x, dot.y), max(2, round(RING_RADIUS * font)), (0, 0, 0), 1, cv2.LINE_AA)
    for dot in puzzle.dots:
        cv2.circle(canvas, at(dot.x, dot.y), radius, (0, 0, 0), -1, cv2.LINE_AA)

    if numbers:
        font_scale = DIGIT_HEIGHT * font / HERSHEY_CAP_HEIGHT
        thickness = 1 if font < 14 else 2
        for dot in puzzle.dots:
            text = str(dot.number)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx, cy = at(*label_center(dot, puzzle.font_size))
            cv2.putText(canvas, text, (cx - w // 2, cy + h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas
