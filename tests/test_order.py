import numpy as np

from dot2dot.models import Path
from dot2dot.order import order_paths, signed_area


def square_at(x, size=10):
    return np.array([[x, 0], [x + size, 0], [x + size, size], [x, size]], dtype=np.float64)


def test_reverses_counter_clockwise_loop():
    counter_clockwise = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64)
    (ordered,) = order_paths([Path(counter_clockwise, closed=True)])
    assert signed_area(ordered.points) > 0


def test_starts_at_topmost_point_of_longest_path():
    square = np.array([[10, 10], [0, 10], [0, 0], [10, 0]], dtype=np.float64)
    (ordered,) = order_paths([Path(square, closed=True)])
    assert tuple(ordered.points[0]) == (0, 0)


def test_jumps_to_nearest_path_next():
    first, far, near = square_at(0, size=50), square_at(500), square_at(60)
    ordered = order_paths([Path(p, closed=True) for p in (far, first, near)])
    assert ordered[0].points[:, 0].max() == 50
    assert ordered[1].points[0][0] == 60


def test_open_path_entered_from_nearer_end():
    loop = Path(square_at(0, size=50), closed=True)
    line = Path(np.array([[300, 0], [200, 0], [100, 0]], dtype=np.float64), closed=False)
    ordered = order_paths([loop, line])
    assert tuple(ordered[1].points[0]) == (100, 0)
