import sys
import types

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxopt")

import jax
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize

from recovar.heterogeneity import deconvolve_density as dd

pytestmark = pytest.mark.unit


@pytest.fixture
def cpu_device():
    return jax.devices("cpu")[0]


def _legacy_estimate_kernel_by_sampling(grids_inp, cov_zs, gauss_kde_covariance, num_samples=5000):
    grid_size = jnp.max(grids_inp, axis=np.arange(grids_inp.ndim - 1)) - jnp.min(
        grids_inp, axis=np.arange(grids_inp.ndim - 1)
    )
    coord_pca_1d = []
    num_points = grids_inp.shape[0]

    pca_dim_max = grids_inp.shape[-1]
    for pca_dim in range(pca_dim_max):
        coord_pca = jnp.flip(jnp.linspace(-grid_size[pca_dim] / 2, grid_size[pca_dim] / 2, num_points, endpoint=False))
        coord_pca_1d.append(coord_pca)
    grids = jnp.meshgrid(*coord_pca_1d, indexing="ij")
    grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32)

    kernel_on_grid = jnp.zeros((grids_flat.shape[0],), dtype=jnp.float32)
    sampled_indices = np.random.randint(0, cov_zs.shape[0], size=num_samples)
    for idx in sampled_indices:
        covar_data = jnp.linalg.pinv(cov_zs[idx])
        total_covar = covar_data + gauss_kde_covariance
        kernel_on_grid += jax.scipy.stats.multivariate_normal.pdf(
            grids_flat, np.zeros(total_covar.shape[0]), total_covar
        )

    kernel_on_grid = kernel_on_grid / jnp.sum(kernel_on_grid)
    return kernel_on_grid.reshape(grids_inp.shape[:-1])


def _legacy_compute_deconvolved_density(
    density,
    kernel,
    total_covar,
    grids,
    kernel_option="sampling",
    alphas=None,
    maxiter=500,
):
    alphas = np.flip(np.logspace(-3, 2, 5)) if alphas is None else alphas

    def compute_kernel_on_grid_nd(grids_inp):
        grid_size = jnp.max(grids_inp, axis=np.arange(grids_inp.ndim - 1)) - jnp.min(
            grids_inp, axis=np.arange(grids_inp.ndim - 1)
        )
        coord_pca_1d = []
        num_points = grids_inp.shape[0]

        pca_dim_max = grids_inp.shape[-1]
        for pca_dim in range(pca_dim_max):
            coord_pca = jnp.flip(
                jnp.linspace(
                    -grid_size[pca_dim] / 2,
                    grid_size[pca_dim] / 2,
                    num_points,
                    endpoint=False,
                )
            )
            coord_pca_1d.append(coord_pca)
        grids = jnp.meshgrid(*coord_pca_1d, indexing="ij")
        grids_flat = jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(np.float32)
        kernel_on_grid = jax.scipy.stats.multivariate_normal.pdf(
            grids_flat, np.zeros(total_covar.shape[0]), total_covar
        )
        kernel_on_grid = kernel_on_grid / jnp.sum(kernel_on_grid)
        return kernel_on_grid.reshape(grids_inp.shape[:-1])

    if kernel_option == "sampling":
        kernel_on_grid = kernel
    elif kernel_option == "avg_cov":
        kernel_on_grid = compute_kernel_on_grid_nd(grids).astype(np.float32)
    else:
        raise NotImplementedError(f"Unknown kernel_option={kernel_option}")

    density = density.astype(np.float32) / np.mean(density)

    density = jnp.array(density)
    kernel_on_grid = jnp.array(kernel_on_grid)

    def forward_model_grid(fun_on_grid):
        return dd.convolve_with_pad_nd(fun_on_grid, kernel_on_grid)

    @jax.jit
    def ridge_reg_objective_grid(fun_on_grid, alpha=0.0):
        residuals = forward_model_grid(fun_on_grid) - density
        if fun_on_grid.ndim == 1:
            dx = grids[1, :] - grids[0, :]
        elif fun_on_grid.ndim == 2:
            dx = grids[1, 1, :] - grids[0, 0, :]
        elif fun_on_grid.ndim == 3:
            dx = grids[1, 1, 1, :] - grids[0, 0, 0, :]
        elif fun_on_grid.ndim == 4:
            dx = grids[1, 1, 1, 1, :] - grids[0, 0, 0, 0, :]
        elif fun_on_grid.ndim == 5:
            dx = grids[1, 1, 1, 1, 1, :] - grids[0, 0, 0, 0, 0, :]
        else:
            raise ValueError(f"Unsupported grid dimensionality: {fun_on_grid.ndim}")

        dx /= jnp.mean(dx)
        return 1e8 * (
            jnp.mean((residuals * 1e0) ** 2) + alpha * jnp.mean(jnp.array(jnp.gradient(fun_on_grid, *dx)) ** 2)
        )

    cost = np.zeros_like(alphas)
    reg_cost = np.zeros_like(alphas)
    lbfgsb_sols = []

    for alpha_idx, alpha in enumerate(alphas):
        w_init = jnp.array(density)
        lbfgsb = ScipyBoundedMinimize(fun=ridge_reg_objective_grid, method="l-bfgs-b", maxiter=maxiter)
        lower_bounds = jnp.zeros_like(w_init)
        upper_bounds = jnp.ones_like(w_init) * jnp.inf
        bounds = (lower_bounds, upper_bounds)
        lbfgsb_sol_p = lbfgsb.run(init_params=w_init, bounds=bounds, alpha=alpha)
        lbfgsb_sol = lbfgsb_sol_p.params
        baseline = ridge_reg_objective_grid(lbfgsb_sol * 0, alpha=0)
        cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha=0) / baseline
        reg_cost[alpha_idx] = ridge_reg_objective_grid(lbfgsb_sol, alpha=alpha) / baseline
        lbfgsb_sols.append(np.array(lbfgsb_sol))

    return lbfgsb_sols, cost, reg_cost, alphas


def test_get_raw_density_rejects_outliers_in_requested_pca_dim(monkeypatch):
    zdim = 4
    pca_dim = 2
    num_points = 3
    zs = np.arange(16, dtype=np.float32).reshape(4, 4)
    cov_zs = np.zeros((4, 4, 4), dtype=np.float32)
    diag_values = np.array(
        [
            [10.0, 10.0, 1.0, 1.0],
            [1.0, 1.0, 100.0, 100.0],
            [8.0, 8.0, 1.0, 1.0],
            [0.5, 0.5, 90.0, 90.0],
        ],
        dtype=np.float32,
    )
    for idx, diag in enumerate(diag_values):
        cov_zs[idx] = np.diag(diag)

    class FakePipelineOutput:
        def get_embedding_component(self, entry, key):
            assert key == zdim
            if entry == "latent_coords_noreg":
                return zs
            if entry == "latent_precision_noreg":
                return cov_zs
            raise KeyError(entry)

        def get(self, key):
            raise AssertionError(f"unexpected fallback get({key!r})")

    captured = {}

    class FakeGaussianKDE:
        def __init__(self, dataset, bandwidth):
            captured["kde_dataset"] = np.asarray(dataset)
            captured["bandwidth"] = bandwidth
            self.covariance = np.eye(pca_dim, dtype=np.float32) * 0.25

    def fake_compute_latent_space_density_kde(zs_arg, pca_dim_max, num_points, percentile):
        captured["density_zs"] = np.asarray(zs_arg)
        captured["density_args"] = (pca_dim_max, num_points, percentile)
        return np.ones((num_points, num_points), dtype=np.float32), np.array([[-1.0, 1.0], [-2.0, 2.0]])

    def fake_make_latent_space_grid_from_bounds(bounds, num_points):
        axes = [np.linspace(lo, hi, num_points, dtype=np.float32) for lo, hi in bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([axis.reshape(-1) for axis in mesh], axis=-1)

    def fake_estimate_kernel_by_sampling(grids, cov_arg, gauss_covariance, num_samples):
        captured["kernel_cov_zs"] = np.asarray(cov_arg)
        captured["kernel_gauss_covariance"] = np.asarray(gauss_covariance)
        captured["kernel_num_samples"] = num_samples
        return np.ones(grids.shape[:-1], dtype=np.float32)

    monkeypatch.setattr(dd.jax.scipy.stats, "gaussian_kde", FakeGaussianKDE)
    monkeypatch.setattr(dd.latent_density, "compute_latent_space_density_kde", fake_compute_latent_space_density_kde)
    monkeypatch.setattr(
        dd.latent_density, "make_latent_space_grid_from_bounds", fake_make_latent_space_grid_from_bounds
    )
    monkeypatch.setattr(dd, "estimate_kernel_by_sampling", fake_estimate_kernel_by_sampling)

    density, kernel, total_covar, grids, bounds = dd.get_raw_density(
        FakePipelineOutput(),
        zdim=zdim,
        noreg=True,
        pca_dim_max=pca_dim,
        percentile_reject=50,
        num_points=num_points,
        percentile_bound=7,
    )

    expected_good = [0, 2]
    expected_zs = zs[expected_good, :pca_dim]
    expected_cov_zs = cov_zs[expected_good, :pca_dim, :pca_dim]
    np.testing.assert_array_equal(captured["density_zs"], expected_zs)
    np.testing.assert_array_equal(captured["kde_dataset"], expected_zs.T)
    np.testing.assert_array_equal(captured["kernel_cov_zs"], expected_cov_zs)
    assert captured["density_args"] == (pca_dim, num_points, 7)
    assert captured["bandwidth"] == "silverman"
    assert captured["kernel_num_samples"] == 5000
    assert density.shape == (num_points, num_points)
    assert kernel.shape == (num_points, num_points)
    assert grids.shape == (num_points, num_points, pca_dim)
    np.testing.assert_array_equal(bounds, np.array([[-1.0, 1.0], [-2.0, 2.0]]))
    expected_total_covar = np.mean(np.linalg.inv(expected_cov_zs), axis=0) + np.eye(pca_dim) * 0.25
    np.testing.assert_allclose(total_covar, expected_total_covar)


def test_estimate_kernel_by_sampling_shape_and_normalization(cpu_device):
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

    with jax.default_device(cpu_device):
        kernel = dd.estimate_kernel_by_sampling(grid, cov_zs, gauss_cov, num_samples=40)
    assert kernel.shape == (5, 5)
    assert np.all(np.isfinite(kernel))
    assert np.isclose(float(np.sum(kernel)), 1.0, atol=1e-5)
    assert float(np.max(kernel)) > 0.0


def test_estimate_kernel_by_sampling_matches_legacy_result(cpu_device):
    grid = np.zeros((7, 7, 2), dtype=np.float32)
    cov_zs = np.stack(
        [
            np.array([[1.1, 0.08], [0.08, 0.95]], dtype=np.float32),
            np.array([[0.85, -0.03], [-0.03, 1.2]], dtype=np.float32),
            np.array([[1.3, 0.02], [0.02, 0.9]], dtype=np.float32),
            np.array([[0.95, 0.01], [0.01, 1.05]], dtype=np.float32),
        ],
        axis=0,
    )
    gauss_cov = np.array([[0.25, 0.01], [0.01, 0.2]], dtype=np.float32)

    with jax.default_device(cpu_device):
        np.random.seed(7)
        legacy = _legacy_estimate_kernel_by_sampling(grid, cov_zs, gauss_cov, num_samples=120)
        np.random.seed(7)
        current = dd.estimate_kernel_by_sampling(grid, cov_zs, gauss_cov, num_samples=120, batch_size=8)

    np.testing.assert_allclose(np.asarray(current), np.asarray(legacy), rtol=3e-6, atol=3e-6)


def test_compute_deconvolved_density_matches_legacy_small_case(cpu_device):
    density = np.array([0.2, 0.5, 1.0, 0.6, 0.25], dtype=np.float32)
    kernel = np.array([0.05, 0.2, 0.5, 0.2, 0.05], dtype=np.float32)
    kernel = kernel / kernel.sum()
    total_covar = np.array([[0.6]], dtype=np.float32)
    grids = np.linspace(-1.0, 1.0, density.size, dtype=np.float32)[:, None]
    alphas = np.array([1e-1], dtype=np.float32)

    with jax.default_device(cpu_device):
        legacy_sols, legacy_cost, legacy_reg_cost, legacy_alphas = _legacy_compute_deconvolved_density(
            density=density,
            kernel=kernel,
            total_covar=total_covar,
            grids=grids,
            kernel_option="sampling",
            alphas=alphas,
            maxiter=25,
        )
        sols, cost, reg_cost, got_alphas = dd.compute_deconvolved_density(
            density=density,
            kernel=kernel,
            total_covar=total_covar,
            grids=grids,
            kernel_option="sampling",
            alphas=alphas,
            maxiter=25,
        )

    np.testing.assert_allclose(got_alphas, legacy_alphas)
    np.testing.assert_allclose(sols[0], legacy_sols[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cost, legacy_cost, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reg_cost, legacy_reg_cost, rtol=1e-6, atol=1e-6)


def test_plot_density_runs_for_small_1d_input(monkeypatch):
    stub_output_module = types.ModuleType("recovar.output.output")
    stub_output_module.sum_over_other = lambda density, axes: np.sum(
        density, axis=tuple(i for i in range(density.ndim) if i not in axes)
    )
    monkeypatch.setitem(sys.modules, "recovar.output.output", stub_output_module)

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
