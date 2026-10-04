import numpy as np

from dot2dot.simplify import allocate_dots, simplify


def test_returns_exactly_n_points(square_outline):
    for n in (4, 10, 37):
        assert len(simplify(square_outline, n)) == n


def test_keeps_corners(square_outline):
    points = {tuple(p) for p in simplify(square_outline, 8)}
    assert {(0, 0), (100, 0), (100, 100), (0, 100)} <= points


def test_points_lie_on_original_outline(square_outline):
    original = {tuple(p) for p in square_outline}
    assert all(tuple(p) in original for p in simplify(square_outline, 20))


def test_short_outline_is_returned_unchanged():
    tiny = np.array([[0, 0], [5, 0], [5, 5]], dtype=np.float64)
    assert np.array_equal(simplify(tiny, 10), tiny)


def test_allocate_dots_is_proportional_to_perimeter(square_outline):
    big = square_outline * 3
    assert allocate_dots([big, square_outline], 40) == [30, 10]
