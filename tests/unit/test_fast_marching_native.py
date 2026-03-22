import numpy as np
import pytest

from recovar.heterogeneity import fast_marching

pytestmark = pytest.mark.unit


def test_native_availability_flag_is_boolean():
    assert isinstance(fast_marching.native_available(), bool)


@pytest.mark.skipif(not fast_marching.native_available(), reason="native fast marching extension unavailable")
def test_native_backend_matches_python_fallback_on_random_point_sources():
    rng = np.random.default_rng(123)

    for order in (1, 2):
        for ndim in (2, 3, 4):
            shape = (5,) * ndim
            speed = 0.25 + rng.random(shape)
            dx = 0.3 + rng.random(ndim)
            start = tuple(axis // 2 for axis in shape)
            phi = np.ones(shape, dtype=np.float64)
            phi[start] = -1.0

            expected = fast_marching._python_travel_time(phi, speed, dx, order)
            got = fast_marching._native_travel_time(phi, speed, np.asarray(dx, dtype=np.float64), order)
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not fast_marching.native_available(), reason="native fast marching extension unavailable")
def test_native_backend_matches_python_fallback_zero_contour_error():
    phi = np.ones((4, 4), dtype=np.float64)
    speed = np.ones((4, 4), dtype=np.float64)
    dx = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="no zero contour"):
        fast_marching._python_travel_time(phi, speed, dx, order=2)
    with pytest.raises(ValueError, match="no zero contour"):
        fast_marching._native_travel_time(phi, speed, dx, order=2)


@pytest.mark.skipif(not fast_marching.native_available(), reason="native fast marching extension unavailable")
@pytest.mark.parametrize(
    ("shape", "start_index", "dx"),
    [
        ((7,), (0,), 1.0),
        ((7,), (6,), 0.75),
        ((1, 7), (0, 0), [0.8, 1.2]),
        ((6, 6), (0, 5), [1.0, 0.8]),
        ((5, 4, 3), (4, 2, 0), [0.8, 1.2, 0.6]),
    ],
)
def test_native_backend_matches_python_fallback_on_boundary_sources(shape, start_index, dx):
    rng = np.random.default_rng(500 + int(np.prod(shape)) + sum(start_index))
    speed = 0.25 + rng.random(shape)
    phi = np.ones(shape, dtype=np.float64)
    phi[start_index] = -1.0
    dx = np.asarray(dx if np.ndim(dx) else [dx] * len(shape), dtype=np.float64)

    expected = fast_marching._python_travel_time(phi, speed, dx, order=2)
    got = fast_marching._native_travel_time(phi, speed, dx, order=2)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not fast_marching.native_available(), reason="native fast marching extension unavailable")
@pytest.mark.parametrize(
    ("phi", "speed", "dx"),
    [
        (
            np.broadcast_to(np.arange(5, dtype=np.float64)[:, None], (5, 4)),
            np.array(
                [
                    [1.0, 1.1, 0.9, 1.2],
                    [1.3, 0.8, 1.4, 1.0],
                    [1.2, 1.5, 0.7, 1.1],
                    [0.9, 1.0, 1.2, 1.3],
                    [1.1, 1.4, 1.0, 0.8],
                ],
                dtype=np.float64,
            ),
            [0.5, 1.25],
        ),
        (
            np.indices((4, 5, 3), dtype=np.float64)[0] + np.indices((4, 5, 3), dtype=np.float64)[2] - 0.75,
            0.8 + np.linspace(0.0, 1.0, num=60, dtype=np.float64).reshape(4, 5, 3),
            [0.5, 1.0, 1.5],
        ),
    ],
)
def test_native_backend_matches_python_fallback_on_boundary_contours(phi, speed, dx):
    dx = np.asarray(dx, dtype=np.float64)

    expected = fast_marching._python_travel_time(phi, speed, dx, order=2)
    got = fast_marching._native_travel_time(phi, speed, dx, order=2)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
