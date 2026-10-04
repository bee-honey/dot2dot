import numpy as np

from dot2dot.order import order_outlines, signed_area


def test_reverses_counter_clockwise_outline():
    counter_clockwise = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64)
    (ordered,) = order_outlines([counter_clockwise])
    assert signed_area(ordered) > 0


def test_starts_at_topmost_point():
    square = np.array([[10, 10], [0, 10], [0, 0], [10, 0]], dtype=np.float64)
    (ordered,) = order_outlines([square])
    assert tuple(ordered[0]) == (0, 0)


def test_jumps_to_nearest_outline_next():
    def square_at(x):
        return np.array([[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]], dtype=np.float64)

    near, far = square_at(20), square_at(500)
    ordered = order_outlines([square_at(0), far, near])
    assert ordered[1][0][0] == 20
