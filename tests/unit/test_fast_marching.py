import numpy as np
import pytest

from recovar.heterogeneity import fast_marching

pytestmark = pytest.mark.unit


def _assert_matches_skfmm(phi, speed, dx, order):
    skfmm = pytest.importorskip("skfmm")
    expected = np.asarray(skfmm.travel_time(phi, speed=speed, dx=dx, order=order), dtype=np.float64)
    got = fast_marching.travel_time(phi, speed=speed, dx=dx, order=order)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_travel_time_matches_known_order_one_reference():
    speed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    phi = np.ones_like(speed)
    phi[3] = -1.0

    got = fast_marching.travel_time(phi, speed, dx=1.0, order=1)
    expected = np.array(
        [1.666666666667, 0.666666666667, 0.166666666667, 0.125, 0.1, 0.266666666667, 0.409523809524],
        dtype=np.float64,
    )

    np.testing.assert_allclose(got, expected, rtol=3e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("phi", "speed", "dx", "expected"),
    [
        (
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            1.0,
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            1.0,
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
        ),
        (
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            2.0,
            np.array([2.0, 0.0, 2.0], dtype=np.float64),
        ),
        (
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            [2.0],
            np.array([2.0, 0.0, 2.0], dtype=np.float64),
        ),
        (
            np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64),
            1.0,
            np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64),
        ),
        (
            np.array([1.0, 0.0, -1.0], dtype=np.float64),
            np.array([2.0, 2.0, 2.0], dtype=np.float64),
            1.0,
            np.array([0.5, 0.0, 0.5], dtype=np.float64),
        ),
        (
            np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float64),
            np.ones(6, dtype=np.float64),
            1.0,
            np.array([2.5, 1.5, 0.5, 0.5, 1.5, 2.5], dtype=np.float64),
        ),
        (
            np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.ones(6, dtype=np.float64),
            1.0,
            np.array([2.5, 1.5, 0.5, 0.5, 1.5, 2.5], dtype=np.float64),
        ),
        (
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            1.0,
            np.array([0.0, 0.0], dtype=np.float64),
        ),
    ],
)
def test_travel_time_matches_skfmm_doctest_edge_cases(phi, speed, dx, expected):
    got = fast_marching.travel_time(phi, speed, dx=dx, order=2)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_travel_time_matches_known_order_two_reference():
    speed = np.array(
        [
            [1.2, 1.1, 0.9, 0.8],
            [1.3, 1.5, 1.0, 0.7],
            [1.4, 1.2, 0.6, 0.9],
            [1.1, 1.0, 0.8, 1.2],
        ],
        dtype=np.float64,
    )
    phi = np.ones_like(speed)
    phi[1, 1] = -1.0

    got = fast_marching.travel_time(phi, speed, dx=[0.7, 1.4], order=2)
    expected = np.array(
        [
            [1.008660898117, 0.318181818182, 1.302329959782, 2.694667968787],
            [0.538461538462, 0.2086996779, 0.7, 2.336233225967],
            [0.925284597892, 0.291666666667, 1.648969598517, 2.863622061434],
            [1.44191903272, 0.925122114856, 2.318303130951, 3.320403240876],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)


def test_point_source_travel_time_matches_known_three_dimensional_reference():
    speed = np.array(
        [
            [
                [1.443056105572, 1.011327552814, 1.476243705707, 0.580836023896],
                [1.107355831995, 0.876486584378, 1.301901206986, 0.674527816145],
                [1.371635274187, 1.043941400763, 1.402215079716, 0.97715352384],
                [0.93049627773, 1.288946717545, 1.484152999932, 0.869725792652],
            ],
            [
                [1.468932869317, 1.429026387766, 0.677692585762, 1.108851616844],
                [1.204864745565, 1.442803679129, 1.165657416966, 0.633395755452],
                [0.997867598785, 0.993619833957, 1.000226189348, 1.45858228611],
                [0.849937399811, 0.723771146778, 1.022087002211, 1.141170923965],
            ],
            [
                [1.439107053425, 1.082015800647, 0.767833390291, 1.429774697204],
                [0.991725284613, 1.175800864934, 0.976038910476, 0.716980026422],
                [1.19255204471, 1.270630483091, 0.690788505259, 0.959916012945],
                [0.861815058901, 0.670729817286, 0.721351659352, 1.462507411326],
            ],
            [
                [1.384223051631, 0.882291826195, 1.239473243136, 0.541968037182],
                [1.415057201246, 1.037170851214, 1.49322397881, 0.818518361696],
                [1.419987354684, 0.804982534001, 0.975528723609, 0.964517873378],
                [1.029638948933, 0.69423984995, 1.29891721514, 0.655969122886],
            ],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [
                [2.004519353746, 1.629554644205, 2.145485887583, 3.434744602164],
                [1.701425387762, 1.231167478535, 1.755463291939, 2.871040845749],
                [1.340140091705, 0.845361198145, 1.465144882315, 2.240416678918],
                [2.063096658045, 1.508246804652, 1.944387714609, 2.736124247907],
            ],
            [
                [1.760782544032, 1.239080781458, 2.252877100466, 3.12905869898],
                [1.352812099961, 0.701857804501, 1.4926634608, 2.707104930789],
                [0.94827493751, 0.301926344869, 1.166751605982, 1.95126623559],
                [1.732492657603, 1.225785781973, 1.851189472654, 2.496424375795],
            ],
            [
                [1.654644515025, 1.124440518317, 2.068262845054, 2.864583286363],
                [1.131046558838, 0.382717868774, 1.31467977739, 2.511653667253],
                [0.461195795877, 0.178888427754, 0.79619159361, 1.885173990484],
                [1.346977258714, 0.670910979233, 1.709635108132, 2.367389788881],
            ],
            [
                [1.819736111266, 1.552291042139, 1.977103253877, 3.203120120342],
                [1.287324936075, 0.860945073738, 1.466274669692, 2.51705889591],
                [0.809972919782, 0.372678892078, 1.208834685488, 2.16598837895],
                [1.576852569811, 1.284954080695, 1.769750074548, 2.786625241484],
            ],
        ],
        dtype=np.float64,
    )

    got = fast_marching.point_source_travel_time(speed, start_index=(2, 2, 1), dx=[0.6, 0.9, 1.1], order=2)
    np.testing.assert_allclose(got, expected, rtol=3e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("shape", "start_index", "dx"),
    [
        ((7,), (0,), 1.0),
        ((7,), (6,), 0.75),
        ((1, 7), (0, 0), [0.8, 1.2]),
        ((6, 6), (0, 0), [0.5, 1.5]),
        ((6, 6), (0, 5), [1.0, 0.8]),
        ((5, 4, 3), (0, 0, 0), [0.5, 1.1, 1.7]),
        ((5, 4, 3), (4, 2, 0), [0.8, 1.2, 0.6]),
    ],
)
def test_point_source_travel_time_supports_boundary_sources(shape, start_index, dx):
    rng = np.random.default_rng(17 + int(np.prod(shape)))
    speed = 0.4 + rng.random(shape)
    phi = np.ones(shape, dtype=np.float64)
    phi[start_index] = -1.0

    got = fast_marching.point_source_travel_time(speed, start_index=start_index, dx=dx, order=2)
    expected = fast_marching.travel_time(phi, speed, dx=dx, order=2)

    assert got.shape == speed.shape
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("start_index", "error_type", "message"),
    [
        ((1,), ValueError, "dimensionality"),
        ((-1, 0), IndexError, "out of bounds"),
        ((0, 3), IndexError, "out of bounds"),
        ((0.0, 1.0, 2.0), ValueError, "dimensionality"),
    ],
)
def test_point_source_travel_time_rejects_invalid_start_index(start_index, error_type, message):
    speed = np.ones((3, 3), dtype=np.float64)

    with pytest.raises(error_type, match=message):
        fast_marching.point_source_travel_time(speed, start_index=start_index, dx=1.0, order=2)


def test_travel_time_scalar_dx_matches_vector_dx():
    speed = np.array(
        [
            [1.0, 1.1, 0.9, 1.2],
            [0.8, 1.3, 1.4, 0.7],
            [1.5, 1.2, 1.0, 1.1],
        ],
        dtype=np.float64,
    )
    phi = np.ones_like(speed)
    phi[0, 0] = -1.0

    got_scalar = fast_marching.travel_time(phi, speed, dx=0.75, order=2)
    got_vector = fast_marching.travel_time(phi, speed, dx=[0.75, 0.75], order=2)

    np.testing.assert_allclose(got_scalar, got_vector, rtol=0.0, atol=0.0)


def test_travel_time_rejects_invalid_dx():
    phi = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
    speed = np.ones_like(phi)

    with pytest.raises(ValueError, match="dx must be a scalar or have length"):
        fast_marching.travel_time(phi, speed, dx=[1.0, 2.0], order=2)
    with pytest.raises(ValueError, match="dx must be finite and strictly positive"):
        fast_marching.travel_time(phi, speed, dx=np.nan, order=2)
    with pytest.raises(ValueError, match="dx must be finite and strictly positive"):
        fast_marching.travel_time(phi, speed, dx=0.0, order=2)


def test_travel_time_rejects_speed_shape_mismatch_and_scalar_speed():
    phi = np.array([-1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        fast_marching.travel_time(phi, np.array([2.0], dtype=np.float64), dx=1.0, order=2)
    with pytest.raises(ValueError, match="same shape"):
        fast_marching.travel_time(phi, 2.0, dx=1.0, order=2)


@pytest.mark.parametrize(
    ("phi", "speed", "order", "message"),
    [
        (np.array([np.nan, 1.0]), np.ones(2), 2, "phi must be finite"),
        (np.array([-1.0, 1.0]), np.array([1.0, np.inf]), 2, "speed must be finite"),
        (np.array([-1.0, 1.0]), np.array([1.0, 0.0]), 2, "speed must be strictly positive"),
        (np.array([-1.0, 1.0]), np.ones(2), 3, "order must be 1 or 2"),
    ],
)
def test_travel_time_rejects_invalid_inputs(phi, speed, order, message):
    with pytest.raises(ValueError, match=message):
        fast_marching.travel_time(phi, speed, dx=1.0, order=order)


def test_travel_time_raises_without_zero_contour():
    phi = np.ones((4, 4), dtype=np.float64)
    speed = np.ones((4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="no zero contour"):
        fast_marching.travel_time(phi, speed, dx=1.0, order=2)


def test_travel_time_with_boundary_zero_contour_keeps_boundary_at_zero():
    phi = np.broadcast_to(np.arange(5, dtype=np.float64)[:, None], (5, 4))
    speed = np.ones_like(phi)

    got = fast_marching.travel_time(phi, speed, dx=[0.5, 1.0], order=2)

    np.testing.assert_allclose(got[0], 0.0, rtol=0.0, atol=0.0)
    assert np.all(got[1:] > 0)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    ("shape", "start_index", "dx"),
    [
        ((7,), (0,), 1.0),
        ((7,), (6,), 0.75),
        ((1, 7), (0, 0), [0.8, 1.2]),
        ((6, 6), (0, 0), [0.5, 1.5]),
        ((6, 6), (0, 5), [1.0, 0.8]),
        ((5, 4, 3), (0, 0, 0), [0.5, 1.1, 1.7]),
        ((5, 4, 3), (4, 2, 0), [0.8, 1.2, 0.6]),
    ],
)
def test_optional_skfmm_parity_on_boundary_point_sources(shape, start_index, dx, order):
    seed = 1000 + order * 100 + int(np.prod(shape)) + sum(start_index)
    rng = np.random.default_rng(seed)
    speed = 0.35 + rng.random(shape)
    phi = np.ones(shape, dtype=np.float64)
    phi[start_index] = -1.0

    _assert_matches_skfmm(phi, speed, dx=dx, order=order)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    ("phi", "speed", "dx"),
    [
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([1.1, 0.9, 1.2, 1.0, 1.3], dtype=np.float64),
            0.75,
        ),
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
            np.indices((4, 3, 5), dtype=np.float64)[0],
            0.6 + np.linspace(0.0, 1.0, num=60, dtype=np.float64).reshape(4, 3, 5),
            [0.4, 0.9, 1.3],
        ),
    ],
)
def test_optional_skfmm_parity_on_boundary_zero_contours(phi, speed, dx, order):
    _assert_matches_skfmm(phi, speed, dx=dx, order=order)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    ("phi", "speed", "dx"),
    [
        (
            np.array([-1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([1.0, 1.2, 0.8, 1.1, 1.3], dtype=np.float64),
            0.5,
        ),
        (
            np.indices((5, 6), dtype=np.float64)[0] - 0.5,
            0.7 + np.linspace(0.0, 1.0, num=30, dtype=np.float64).reshape(5, 6),
            [0.9, 1.4],
        ),
        (
            np.indices((4, 5, 3), dtype=np.float64)[0] + np.indices((4, 5, 3), dtype=np.float64)[2] - 0.75,
            0.8 + np.linspace(0.0, 1.0, num=60, dtype=np.float64).reshape(4, 5, 3),
            [0.5, 1.0, 1.5],
        ),
    ],
)
def test_optional_skfmm_parity_on_boundary_adjacent_sign_changes(phi, speed, dx, order):
    _assert_matches_skfmm(phi, speed, dx=dx, order=order)


def test_optional_skfmm_parity_on_random_point_sources():
    rng = np.random.default_rng(0)

    for order in (1, 2):
        for ndim in (2, 3):
            for _ in range(5):
                shape = (6,) * ndim
                speed = 0.5 + rng.random(shape)
                dx = 0.3 + rng.random(ndim) * 1.7
                start = tuple(axis // 2 for axis in shape)
                phi = np.ones(shape, dtype=np.float64)
                phi[start] = -1.0

                _assert_matches_skfmm(phi, speed, dx=dx, order=order)
