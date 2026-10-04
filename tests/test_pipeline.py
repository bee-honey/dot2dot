from dot2dot.cli import main
from dot2dot.pipeline import generate
from dot2dot.render import to_svg


def test_dots_are_numbered_in_order(square_image):
    puzzle = generate(square_image, num_dots=20)
    assert [d.number for d in puzzle.dots] == list(range(1, 21))


def test_labels_point_away_from_shape(square_image):
    puzzle = generate(square_image, num_dots=20)
    center_x, center_y = puzzle.width / 2, puzzle.height / 2
    for dot in puzzle.dots:
        toward_label = (dot.x - center_x) * dot.label_dx + (dot.y - center_y) * dot.label_dy
        assert toward_label > 0


def test_multiple_outlines_become_separate_strokes(two_shapes_image):
    puzzle = generate(two_shapes_image, num_dots=30, max_outlines=2)
    assert puzzle.stroke_starts[0] == 0
    assert len(puzzle.stroke_starts) == 2
    # No solution line should bridge the two circles.
    for a, b in puzzle.segments():
        assert abs(a.x - b.x) < 250


def test_svg_has_one_label_per_dot(square_image):
    puzzle = generate(square_image, num_dots=15)
    assert to_svg(puzzle).count("<text") == 15
    assert "<line" in to_svg(puzzle, solution=True)


def test_cli_writes_pdf(square_image, tmp_path):
    out = tmp_path / "puzzle.pdf"
    assert main([str(square_image), "--dots", "12", "--out", str(out), "--svg"]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "puzzle_solution.svg").exists()


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.png")]) == 1
    assert "Could not read image" in capsys.readouterr().err
