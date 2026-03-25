import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.heterogeneity import latent_density as ld

pytestmark = pytest.mark.unit


def test_grid_mapping_roundtrip():
    bounds = np.array([[-2.0, 2.0], [0.0, 10.0]], dtype=np.float32)
    num_points = 11
    x = np.array([0.4, 6.0], dtype=np.float32)
    g = ld.pca_coord_to_grid(x, bounds, num_points, to_int=False)
    x_rt = ld.grid_to_pca_coord(g, bounds, num_points)
    assert np.allclose(x_rt, x, atol=1e-6)

    gi = ld.pca_coord_to_grid(x, bounds, num_points, to_int=True)
    assert gi.dtype.kind in ("i", "u")
    assert gi.shape == (2,)


def test_get_grid_z_mappings_callable():
    bounds = np.array([[-1.0, 1.0]], dtype=np.float32)
    grid_to_z, z_to_grid = ld.get_grid_z_mappings(bounds, num_points=5)
    z = np.array([0.25], dtype=np.float32)
    g = z_to_grid(z, to_int=False)
    z_back = grid_to_z(g)
    assert np.allclose(z_back, z, atol=1e-6)


def test_make_latent_space_grid_from_bounds_shape():
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    grid = ld.make_latent_space_grid_from_bounds(bounds, num_points=4)
    assert grid.shape == (4**3, 3)
    assert np.all(np.isfinite(np.asarray(grid)))


def test_compute_latent_space_bounds_percentile():
    zs = np.array(
        [
            [-10.0, 1.0],
            [0.0, 2.0],
            [10.0, 3.0],
            [100.0, 4.0],
        ],
        dtype=np.float32,
    )
    b = ld.compute_latent_space_bounds(zs, percentile=25)
    assert b.shape == (2, 2)
    assert b[0, 0] >= -10.0
    assert b[0, 1] <= 100.0
    assert b[1, 0] >= 1.0
    assert b[1, 1] <= 4.0


def test_compute_log_det_cov_identity():
    """log det of identity matrix should be 0."""
    cov = np.eye(3, dtype=np.float32)[None]
    result = np.asarray(ld.compute_log_det_cov(cov))
    assert result.shape == (1,)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_compute_log_det_cov_scaled_identity():
    """log det of 2*I should be dim*log(2)."""
    cov = 2.0 * np.eye(3, dtype=np.float32)[None]
    result = np.asarray(ld.compute_log_det_cov(cov))
    expected = 3 * np.log(2.0)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_compute_log_det_cov_singular_safe():
    """Singular matrix should not produce -inf due to our safeguard."""
    cov = np.zeros((1, 2, 2), dtype=np.float32)
    result = np.asarray(ld.compute_log_det_cov(cov))
    assert np.all(np.isfinite(result))


def test_compute_latent_quadratic_forms_at_mean_is_zero():
    """Quadratic form at the mean should be zero."""
    xs = np.array([[1.0, 2.0]], dtype=np.float32)  # 1 image, 2D
    cov = np.eye(2, dtype=np.float32)[None]  # identity covariance
    test_pts = xs.copy()  # test at the mean
    result = np.asarray(ld.compute_latent_quadratic_forms(test_pts, xs, cov))
    assert result.shape == (1, 1)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_compute_latent_quadratic_forms_away_from_mean():
    """Quadratic form should increase with distance from mean."""
    xs = np.array([[0.0, 0.0]], dtype=np.float32)
    cov = np.eye(2, dtype=np.float32)[None]
    test_pts = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    result = np.asarray(ld.compute_latent_quadratic_forms(test_pts, xs, cov))
    assert result.shape == (1, 2)
    # Distance 2 should give 4x the quadratic form of distance 1
    np.testing.assert_allclose(result[0, 1] / result[0, 0], 4.0, atol=1e-4)


def test_compute_residuals_single_matches_manual():
    """Verify compute_residuals_single = 0.5 * (x-mu)^T Sigma (x-mu)."""
    import jax.numpy as jnp

    mu = np.array([1.0, 2.0], dtype=np.float32)
    cov = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float32)
    test_pt = np.array([3.0, 1.0], dtype=np.float32)
    diff = test_pt - mu
    expected = 0.5 * diff @ cov @ diff
    result = float(ld.compute_residuals_single(jnp.array(test_pt), jnp.array(mu), jnp.array(cov)))
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_compute_det_cov_xs_normalized_max_is_one():
    cov = np.stack(
        [
            np.eye(2, dtype=np.float32),
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        ],
        axis=0,
    )
    d = np.asarray(ld.compute_det_cov_xs(cov))
    assert d.shape == (2,)
    assert np.isclose(np.max(d), 1.0)
    assert np.all(d > 0)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_grid_mapping_roundtrip_gpu(gpu_device):
    bounds = np.array([[-2.0, 2.0], [0.0, 10.0]], dtype=np.float32)
    num_points = 11
    x = np.array([0.4, 6.0], dtype=np.float32)

    cpu_g = np.asarray(ld.pca_coord_to_grid(x, bounds, num_points, to_int=False))
    cpu_rt = np.asarray(ld.grid_to_pca_coord(cpu_g, bounds, num_points))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        bounds_g = jax.device_put(jnp.array(bounds), gpu_device)
        gpu_g = np.asarray(ld.pca_coord_to_grid(x_g, bounds_g, num_points, to_int=False))
        gpu_rt = np.asarray(ld.grid_to_pca_coord(gpu_g, bounds_g, num_points))

    np.testing.assert_allclose(cpu_g, gpu_g, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_rt, gpu_rt, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_compute_det_cov_xs_gpu(gpu_device):
    cov = np.stack(
        [
            np.eye(2, dtype=np.float32),
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        ],
        axis=0,
    )

    cpu_d = np.asarray(ld.compute_det_cov_xs(cov))

    with jax.default_device(gpu_device):
        cov_g = jax.device_put(jnp.array(cov), gpu_device)
        gpu_d = np.asarray(ld.compute_det_cov_xs(cov_g))

    np.testing.assert_allclose(cpu_d, gpu_d, atol=1e-5, rtol=1e-5)
