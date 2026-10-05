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


def test_allocate_dots_favors_longer_paths(square_outline):
    big = Path(square_outline * 3, closed=True)
    big_count, small_count = allocate_dots([big, Path(square_outline, closed=True)], 40)
    assert big_count + small_count == 40
    assert big_count > 2 * small_count


def test_allocate_dots_gives_curvy_paths_extra():
    straight = Path(np.array([(x, 0.0) for x in range(300)]), closed=False)
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    loop = Path(np.column_stack([150 + 48 * np.cos(angles), 150 + 48 * np.sin(angles)]), closed=True)
    straight_count, loop_count = allocate_dots([straight, loop], 30)
    # Same length (~300 px), but the loop turns a full circle, so it earns more.
    assert loop_count > straight_count


def test_dots_on_a_circle_are_evenly_spaced():
    angles = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    # Rounded to whole pixels, like a traced outline: a staircase circle.
    circle = np.round(np.column_stack([300 + 200 * np.cos(angles), 300 + 200 * np.sin(angles)]))
    points = simplify(Path(circle, closed=True), 40)
    gaps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    assert gaps.min() > 0.7 * gaps.mean()  # no clumps


def test_select_paths_drops_tiny_paths(square_outline):
    big = Path(square_outline * 10, closed=True)  # ~4000 px around
    speck = Path(np.array([[0, 0], [10, 0]], dtype=np.float64), closed=False)
    assert select_paths([speck, big], total=40) == [big]
