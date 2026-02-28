import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxopt")

from recovar.heterogeneity import deconvolve_density as dd

pytestmark = pytest.mark.unit


def test_estimate_kernel_by_sampling_shape_and_normalization():
    grid = np.zeros((5, 5, 2), dtype=np.float32)
    cov_zs = np.stack(
        [
            np.array([[1.2, 0.1], [0.1, 1.0]], dtype=np.float32),
            np.array([[0.9, 0.0], [0.0, 1.1]], dtype=np.float32),
            np.array([[1.1, -0.05], [-0.05, 0.95]], dtype=np.float32),
        ],
        axis=0,
    )
    gauss_cov = np.array([[0.2, 0.0], [0.0, 0.25]], dtype=np.float32)

    kernel = dd.estimate_kernel_by_sampling(grid, cov_zs, gauss_cov, num_samples=40)
    assert kernel.shape == (5, 5)
    assert np.all(np.isfinite(kernel))
    assert np.isclose(float(np.sum(kernel)), 1.0, atol=1e-5)
    assert float(np.max(kernel)) > 0.0


def test_plot_density_runs_for_small_1d_input():
    raw = np.linspace(0.1, 1.0, 16).astype(np.float32)
    sols = [raw * 0.9, raw * 1.1]
    alphas = np.array([1e-2, 1e-1], dtype=np.float32)
    dd.plot_density(sols, raw, alphas)


def test_compute_deconvolved_density_rejects_unknown_kernel_option():
    density = np.ones((5, 5), dtype=np.float32)
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    total_covar = np.eye(2, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    grids = np.stack([xx, yy], axis=-1)

    with pytest.raises(NotImplementedError, match="Unknown kernel_option"):
        dd.compute_deconvolved_density(
            density=density,
            kernel=kernel,
            total_covar=total_covar,
            grids=grids,
            kernel_option="bad_option",
            alphas=np.array([1.0], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_estimate_kernel_by_sampling_gpu(gpu_device):
    grid = np.zeros((5, 5, 2), dtype=np.float32)
    cov_zs = np.stack(
        [
            np.array([[1.2, 0.1], [0.1, 1.0]], dtype=np.float32),
            np.array([[0.9, 0.0], [0.0, 1.1]], dtype=np.float32),
            np.array([[1.1, -0.05], [-0.05, 0.95]], dtype=np.float32),
        ],
        axis=0,
    )
    gauss_cov = np.array([[0.2, 0.0], [0.0, 0.25]], dtype=np.float32)

    cpu_kernel = np.asarray(dd.estimate_kernel_by_sampling(grid, cov_zs, gauss_cov, num_samples=40))

    with jax.default_device(gpu_device):
        grid_g = jax.device_put(jnp.array(grid), gpu_device)
        cov_g = jax.device_put(jnp.array(cov_zs), gpu_device)
        gauss_g = jax.device_put(jnp.array(gauss_cov), gpu_device)
        gpu_kernel = np.asarray(dd.estimate_kernel_by_sampling(grid_g, cov_g, gauss_g, num_samples=40))

    np.testing.assert_allclose(cpu_kernel, gpu_kernel, atol=1e-4, rtol=1e-4)
