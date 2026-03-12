import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.core import cubic_interpolation as ci

pytestmark = pytest.mark.unit


def test_calculate_spline_coefficients_shape_preserved():
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    coeff = ci.calculate_spline_coefficients(x)
    coeff = np.asarray(coeff)
    # Periodic solver: output shape == input shape (no boundary padding).
    assert coeff.shape == x.shape
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


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_calculate_spline_coefficients_gpu(gpu_device):
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)

    cpu_coeff = np.asarray(ci.calculate_spline_coefficients(x))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_coeff = np.asarray(ci.calculate_spline_coefficients(x_g))

    np.testing.assert_allclose(cpu_coeff, gpu_coeff, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_map_coordinates_with_cubic_spline_gpu(gpu_device):
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    coeff = ci.calculate_spline_coefficients(arr)
    xs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ys = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    cpu_out = np.asarray(ci.map_coordinates_with_cubic_spline(coeff, (xs, ys), mode="fill", cval=0.0))

    with jax.default_device(gpu_device):
        coeff_g = jax.device_put(jnp.array(np.asarray(coeff)), gpu_device)
        xs_g = jax.device_put(jnp.array(xs), gpu_device)
        ys_g = jax.device_put(jnp.array(ys), gpu_device)
        gpu_out = np.asarray(ci.map_coordinates_with_cubic_spline(coeff_g, (xs_g, ys_g), mode="fill", cval=0.0))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4, rtol=1e-4)
