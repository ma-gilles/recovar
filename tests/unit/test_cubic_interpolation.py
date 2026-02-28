import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lineax")

from recovar import cubic_interpolation as ci

pytestmark = pytest.mark.unit


def test_calculate_spline_coefficients_shape_preserved():
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    coeff = ci.calculate_spline_coefficients(x)
    coeff = np.asarray(coeff)
    # Current solver constructs two extrapolated boundary coefficients in 1D.
    assert coeff.shape == (x.shape[0] + 2,)
    assert np.all(np.isfinite(coeff))


def test_map_coordinates_rejects_non_cubic_order():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    coords = np.array([1.0, 2.0], dtype=np.float32), np.array([1.5, 2.5], dtype=np.float32)
    with pytest.raises(NotImplementedError, match="order=3"):
        ci.map_coordinates(arr, coords, order=1)


def test_map_coordinates_with_cubic_spline_returns_expected_shape():
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    coeff = ci.calculate_spline_coefficients(arr)
    xs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ys = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    out = ci.map_coordinates_with_cubic_spline(coeff, (xs, ys), mode="fill", cval=0.0)
    out = np.asarray(out)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
