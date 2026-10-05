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


def test_missing_detail_lowers_detail_coverage(square_image):
    result = build(square_image, num_dots=20)
    # Pretend the image also had a long interior line the puzzle ignored.
    extra = Path(np.array([[150.0, 200.0], [250.0, 200.0]]), closed=False)
    quality = evaluate(result.puzzle, result.reference + [extra])
    assert quality.detail_coverage == 0.0
    assert quality.overall < evaluate(result.puzzle, result.reference).overall


def test_coverage_image_marks_missed_lines_magenta(square_image):
    result = build(square_image, num_dots=20)
    extra = Path(np.array([[150.0, 200.0], [250.0, 200.0]]), closed=False)
    image = coverage_image(result.puzzle, result.reference + [extra])
    assert (image[200, 200] == (255, 0, 200)).all()
