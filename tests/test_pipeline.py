import cv2
import numpy as np
import pytest

from dot2dot.cli import main
from dot2dot.pipeline import generate
from dot2dot.render import label_center, to_svg


@pytest.fixture
def face_drawing(tmp_path):
    """A simple inked cartoon face: head outline, two eyes, a smile."""
    img = np.full((500, 500, 3), (80, 80, 220), np.uint8)  # colored background
    cv2.circle(img, (250, 250), 180, (0, 0, 0), 6)
    cv2.circle(img, (185, 200), 30, (0, 0, 0), 6)
    cv2.circle(img, (315, 200), 30, (0, 0, 0), 6)
    cv2.ellipse(img, (250, 300), (90, 50), 0, 20, 160, (0, 0, 0), 6)
    path = tmp_path / "face.png"
    cv2.imwrite(str(path), img)
    return path


def test_dots_are_numbered_in_order(square_image):
    puzzle = generate(square_image, num_dots=20)
    assert [d.number for d in puzzle.dots] == list(range(1, 21))


def test_labels_sit_outside_a_simple_outline(square_image):
    puzzle = generate(square_image, num_dots=20)
    center_x, center_y = puzzle.width / 2, puzzle.height / 2
    for dot in puzzle.dots:
        lx, ly = label_center(dot, puzzle.font_size)
        assert np.hypot(lx - center_x, ly - center_y) > np.hypot(dot.x - center_x, dot.y - center_y)


def test_multiple_outlines_become_separate_strokes(two_shapes_image):
    puzzle = generate(two_shapes_image, num_dots=30, max_paths=2)
    assert len(puzzle.strokes) == 2
    assert puzzle.dots[puzzle.strokes[1].start].starts_stroke
    # No solution line should bridge the two circles.
    for a, b in puzzle.segments():
        assert abs(a.x - b.x) < 250


def test_lineart_traces_interior_features(face_drawing):
    puzzle = generate(face_drawing, num_dots=60, max_paths=None, style="lineart")
    closed = [s for s in puzzle.strokes if s.closed]
    open_ = [s for s in puzzle.strokes if not s.closed]
    assert len(closed) == 3  # head + two eyes
    assert len(open_) == 1  # smile
    assert sum(s.end - s.start for s in puzzle.strokes) == len(puzzle.dots)


def test_open_strokes_are_not_closed_in_solution(face_drawing):
    puzzle = generate(face_drawing, num_dots=60, max_paths=None, style="lineart")
    (smile,) = [s for s in puzzle.strokes if not s.closed]
    first, last = puzzle.dots[smile.start], puzzle.dots[smile.end - 1]
    assert (last, first) not in puzzle.segments()


def test_svg_has_one_label_per_dot(square_image):
    puzzle = generate(square_image, num_dots=15)
    assert to_svg(puzzle).count("<text") == 15
    assert "<line" in to_svg(puzzle, solution=True)


def test_cli_writes_pdf(square_image, tmp_path):
    out = tmp_path / "puzzle.pdf"
    assert main([str(square_image), "--dots", "12", "--out", str(out), "--svg"]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "puzzle_solution.svg").exists()


def test_cli_lineart_style(face_drawing, tmp_path):
    out = tmp_path / "face.pdf"
    assert main([str(face_drawing), "--style", "lineart", "--dots", "40", "--out", str(out)]) == 0
    assert out.exists()


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.png")]) == 1
    assert "Could not read image" in capsys.readouterr().err


def test_transparent_background_is_treated_as_white(tmp_path):
    img = np.zeros((300, 300, 4), np.uint8)  # fully transparent
    cv2.circle(img, (150, 150), 90, (40, 40, 200, 255), -1)  # opaque red disc
    path = tmp_path / "cutout.png"
    cv2.imwrite(str(path), img)
    puzzle = generate(path, num_dots=20, style="lineart", max_paths=None)
    xs = [d.x / puzzle.width for d in puzzle.dots]
    # The disc (x from 0.2 to 0.8 of the width), not the whole canvas, was traced.
    assert min(xs) > 0.13 and max(xs) < 0.87
