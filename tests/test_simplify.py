import numpy as np

from dot2dot.models import Path
from dot2dot.simplify import allocate_dots, select_paths, simplify


def test_returns_exactly_n_points(square_outline):
    for n in (4, 10, 37):
        assert len(simplify(Path(square_outline, closed=True), n)) == n


def test_keeps_corners(square_outline):
    points = {tuple(p) for p in simplify(Path(square_outline, closed=True), 8)}
    assert {(0, 0), (100, 0), (100, 100), (0, 100)} <= points


def test_points_lie_on_original_outline(square_outline):
    original = {tuple(p) for p in square_outline}
    assert all(tuple(p) in original for p in simplify(Path(square_outline, closed=True), 20))


def test_open_path_keeps_both_ends():
    line = np.array([(x, 0) for x in range(200)], dtype=np.float64)
    points = simplify(Path(line, closed=False), 5)
    assert len(points) == 5
    assert tuple(points[0]) == (0, 0) and tuple(points[-1]) == (199, 0)


def test_short_path_is_returned_unchanged():
    tiny = np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float64)
    assert np.array_equal(simplify(Path(tiny, closed=True), 10), tiny)


def test_allocate_dots_is_proportional_to_length(square_outline):
    big = Path(square_outline * 3, closed=True)
    assert allocate_dots([big, Path(square_outline, closed=True)], 40) == [30, 10]


def test_select_paths_drops_tiny_paths(square_outline):
    big = Path(square_outline * 10, closed=True)  # ~4000 px around
    speck = Path(np.array([[0, 0], [10, 0]], dtype=np.float64), closed=False)
    assert select_paths([speck, big], total=40) == [big]
