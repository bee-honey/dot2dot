import numpy as np

from dot2dot.models import Path
from dot2dot.pipeline import build
from dot2dot.quality import coverage_image, evaluate


def test_simple_outline_scores_high(square_image):
    result = build(square_image, num_dots=20)
    quality = evaluate(result.puzzle, result.reference)
    assert quality.outline_coverage == 1.0
    assert quality.accuracy > 0.95
    assert quality.overall > 90


def interior_line(puzzle) -> Path:
    """A horizontal line across the middle of the image (inside the square)."""
    w, h = puzzle.width, puzzle.height
    return Path(np.array([[0.375 * w, 0.5 * h], [0.625 * w, 0.5 * h]]), closed=False)


def test_missing_detail_lowers_detail_coverage(square_image):
    result = build(square_image, num_dots=20)
    # Pretend the image also had a long interior line the puzzle ignored.
    extra = interior_line(result.puzzle)
    quality = evaluate(result.puzzle, result.reference + [extra])
    assert quality.detail_coverage == 0.0
    assert quality.overall < evaluate(result.puzzle, result.reference).overall


def test_coverage_image_marks_missed_lines_magenta(square_image):
    result = build(square_image, num_dots=20)
    image = coverage_image(result.puzzle, result.reference + [interior_line(result.puzzle)])
    assert (image[result.puzzle.height // 2, result.puzzle.width // 2] == (255, 0, 200)).all()
