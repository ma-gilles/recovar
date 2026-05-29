import math

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from helpers.tiny_synthetic import make_tiny_cryo_dataset_with_images
from recovar.heterogeneity import local_polynomial_regression as lpr
from recovar.output import output as output_mod
from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit


def test_gaussian_window_polynomial_moments_match_monte_carlo():
    rng = np.random.default_rng(123)
    latent_diff = np.array([-0.7, 0.1, 1.2], dtype=np.float32)
    latent_precision = np.array([3.0, 5.0, 2.0], dtype=np.float32)
    h = 1.3
    degree = 3

    m, M = lpr.gaussian_window_polynomial_moments_1d(latent_diff, latent_precision, h, degree)

    n_base = 120_000
    eps = rng.standard_normal((n_base, latent_diff.size))
    eps = np.concatenate([eps, -eps], axis=0)
    x_minus_target = latent_diff[None, :] + eps / np.sqrt(latent_precision[None, :])
    weights = np.exp(-0.5 * (x_minus_target / h) ** 2)
    t = x_minus_target / h
    features = np.stack([t**r / math.factorial(r) for r in range(degree + 1)], axis=-1)

    m_mc = np.mean(weights[..., None] * features, axis=0)
    M_mc = np.mean(weights[..., None, None] * features[..., :, None] * features[..., None, :], axis=0)

    np.testing.assert_allclose(m, m_mc, atol=6e-3, rtol=6e-3)
    np.testing.assert_allclose(M, M_mc, atol=8e-3, rtol=8e-3)


def test_gaussian_window_polynomial_moments_zero_noise_limit():
    latent_diff = np.array([-0.5, 0.25, 1.0], dtype=np.float32)
    latent_precision = np.full(latent_diff.shape, 1.0e12, dtype=np.float32)
    h = 0.75
    degree = 4

    m, M = lpr.gaussian_window_polynomial_moments_1d(latent_diff, latent_precision, h, degree)

    alpha = np.exp(-0.5 * (latent_diff / h) ** 2)
    t = latent_diff / h
    for r in range(degree + 1):
        np.testing.assert_allclose(m[:, r], alpha * t**r / math.factorial(r), atol=2e-5, rtol=2e-5)
        for s in range(degree + 1):
            expected = alpha * t ** (r + s) / (math.factorial(r) * math.factorial(s))
            np.testing.assert_allclose(M[:, r, s], expected, atol=2e-5, rtol=2e-5)


def test_degree_zero_is_noise_aware_gaussian_kernel_weight():
    latent_diff = np.array([-1.0, 0.0, 0.75], dtype=np.float32)
    latent_precision = np.array([4.0, 9.0, 16.0], dtype=np.float32)
    h = 1.25

    m, M = lpr.gaussian_window_polynomial_moments_1d(latent_diff, latent_precision, h, degree=0)

    var = 1.0 / latent_precision
    alpha = h / np.sqrt(h * h + var) * np.exp(-0.5 * latent_diff**2 / (h * h + var))
    np.testing.assert_allclose(m[:, 0], alpha, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(M[:, 0, 0], alpha, atol=1e-6, rtol=1e-6)


def test_nonmonomial_target_eval_combines_coefficients():
    theta = np.array(
        [
            [
                [1.0 + 1.0j, 2.0 + 2.0j],
                [3.0 + 0.5j, 4.0 + 1.5j],
                [5.0 - 1.0j, 6.0 - 2.0j],
            ]
        ],
        dtype=np.complex64,
    )
    target_eval = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)

    evaluated = lpr.evaluate_local_polynomial_target_coefficients(theta, target_eval)

    expected = 2.0 * theta[:, 0] - theta[:, 1] + 0.5 * theta[:, 2]
    np.testing.assert_allclose(evaluated, expected)
    np.testing.assert_allclose(lpr.evaluate_local_polynomial_target_coefficients(theta), theta[:, 0])


def test_local_poly_singular_voxel_solve_uses_min_norm_fallback():
    class _Dataset:
        volume_shape = (4, 4, 4)

    cryo = _Dataset()
    half_volume_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(cryo.volume_shape)))
    lhs = np.zeros((1, 2, 2, half_volume_size), dtype=np.float32)
    rhs = np.zeros((1, 2, half_volume_size), dtype=np.complex64)

    coeffs, diagnostics = lpr.solve_local_polynomial_fourier_coefficients(
        lhs,
        rhs,
        cryo,
        pol_reg_matrices=np.zeros((1, 2, 2), dtype=np.float32),
        upsampling_factor=1,
        return_diagnostics=True,
    )

    assert coeffs.shape == rhs.shape
    np.testing.assert_allclose(coeffs, 0.0)
    assert diagnostics[0]["pinv_fallback_voxel_count"] > 0


def test_weighted_cholesky_basis_whitens_local_quadrature_cloud():
    latent_diff = np.linspace(-1.0, 1.0, 25).astype(np.float32)
    latent_precision = np.full(latent_diff.shape, 4.0, dtype=np.float32)

    spec = lpr.local_polynomial_basis_spec_1d(
        latent_diff,
        latent_precision,
        h=0.8,
        degree=3,
        n_quadrature=7,
        basis="weighted_cholesky",
        cholesky_jitter=1e-10,
    )

    np.testing.assert_allclose(spec["target_eval"], np.asarray(spec["target_eval"]))
    np.testing.assert_allclose(spec["basis_gram"], np.eye(4), atol=3e-5, rtol=3e-5)
    assert spec["basis_info"]["basis_gram_condition"] < 1.01


def test_polynomial_regularization_matrices():
    latent_diff = np.linspace(-0.5, 0.5, 9).astype(np.float32)
    latent_precision = np.full(latent_diff.shape, 5.0, dtype=np.float32)
    spec = lpr.local_polynomial_basis_spec_1d(
        latent_diff,
        latent_precision,
        h=1.0,
        degree=3,
        n_quadrature=5,
        basis="monomial",
    )

    np.testing.assert_allclose(
        lpr.local_polynomial_regularization_matrix(spec, pol_reg_type="none", pol_reg_eta=10.0),
        np.eye(4),
    )
    np.testing.assert_allclose(
        lpr.local_polynomial_regularization_matrix(spec, pol_reg_type="coeff", pol_reg_eta=0.0),
        np.eye(4),
    )
    coeff = lpr.local_polynomial_regularization_matrix(
        spec,
        pol_reg_type="coeff",
        pol_reg_eta=0.5,
        pol_reg_power=2,
    )
    np.testing.assert_allclose(np.diag(coeff), np.array([1.0, 1.5, 3.0, 5.5], dtype=np.float32))
    deriv2 = lpr.local_polynomial_regularization_matrix(spec, pol_reg_type="deriv2", pol_reg_eta=0.5)
    assert deriv2.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(deriv2) > 0)


def test_local_poly_rejects_non_1d_and_invalid_precision():
    with pytest.raises(NotImplementedError):
        lpr.coerce_1d_latent_differences(np.zeros((5, 2), dtype=np.float32))
    with pytest.raises(NotImplementedError):
        lpr.coerce_1d_latent_precision(np.ones((5, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        lpr.coerce_1d_latent_precision(np.array([1.0, 0.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        lpr.gaussian_window_polynomial_moments_1d(
            np.array([0.0, 0.1], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
            h=1.0,
            degree=1,
        )


def test_local_poly_bandwidth_grid_uses_uncertainty_and_min_particles():
    latent_diff = np.array([0.05, 0.2, 0.4, 1.5], dtype=np.float32)
    latent_precision = np.full(latent_diff.shape, 4.0, dtype=np.float32)
    multipliers = np.array([1.0, 2.0], dtype=np.float32)

    h_grid = lpr.local_poly_bandwidth_grid_1d(
        latent_diff,
        latent_precision,
        n_min_particles=3,
        multipliers=multipliers,
    )

    # sigma_ref = 0.5, so 1.25*sigma_ref = 0.625, larger than the 3rd-nearest
    # target distance 0.4.
    np.testing.assert_allclose(h_grid, np.array([0.625, 1.25], dtype=np.float32), atol=1e-6)


def test_post_process_predivided_fourier_volume_matches_filter_path():
    rng = np.random.default_rng(456)
    volume_shape = (6, 6, 6)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))
    ft_ctf_half = rng.random(half_size).astype(np.float32) + 0.5
    spatial_rhs = jnp.asarray(rng.standard_normal(volume_shape).astype(np.float32))
    f_ty_half = np.asarray(ftu.get_dft3_real(spatial_rhs)).reshape(-1).astype(np.complex64)

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        reg_half = np.asarray(
            relion_functions.adjust_regularization_relion_style(
                jnp.asarray(ft_ctf_half),
                volume_shape,
                tau=None,
                padding_factor=1,
                half_volume=True,
            )
        ).reshape(-1)
        valid_full = relion_functions.mask.get_radial_mask(volume_shape, radius=volume_shape[0] // 2 - 1).reshape(-1)
        valid_half = np.asarray(ftu.full_volume_to_half_volume(jnp.asarray(valid_full), volume_shape)).reshape(-1).real
        divided_half = f_ty_half * valid_half / reg_half
        from_filter = np.asarray(
            relion_functions.post_process_from_filter_v2(
                jnp.asarray(ft_ctf_half),
                jnp.asarray(f_ty_half),
                volume_shape,
                1,
                tau=None,
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=True,
                return_real_space=True,
            )
        )
        predivided = np.asarray(
            relion_functions.post_process_predivided_fourier_volume(
                jnp.asarray(divided_half),
                volume_shape,
                1,
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=True,
                return_real_space=True,
            )
        )

    np.testing.assert_allclose(predivided, from_filter, atol=1e-5, rtol=1e-5)


def test_local_poly_estimator_returns_candidates_lhs_rhs_and_no_debug_files(tmp_path, monkeypatch):
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8, seed=23)
    latent_differences = np.linspace(-0.8, 0.8, cryo.n_images).astype(np.float32)
    latent_precision = np.full(cryo.n_images, 4.0, dtype=np.float32)
    h_grid = np.array([0.75, 1.5], dtype=np.float32)
    monkeypatch.chdir(tmp_path)

    estimates, lhs, rhs = lpr.estimate_local_polynomial_volumes(
        cryo,
        latent_differences,
        latent_precision,
        h_grid,
        degree=1,
        batch_size=4,
        tau=None,
        grid_correct=False,
        use_spherical_mask=False,
        return_lhs_rhs=True,
        upsampling_factor=1,
        return_real_space=True,
        bandwidth_batch_size=1,
    )

    half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(cryo.volume_shape)))
    assert estimates.shape[0] == h_grid.size
    assert lhs.shape == (h_grid.size, 2, 2, half_vol_size)
    assert rhs.shape == (h_grid.size, 2, half_vol_size)
    assert np.isfinite(estimates).all()
    assert np.isfinite(lhs).all()
    assert np.isfinite(rhs).all()
    assert list(tmp_path.iterdir()) == []


def test_local_poly_monomial_no_reg_is_default_behavior(monkeypatch, tmp_path):
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8, seed=31)
    latent_differences = np.linspace(-0.8, 0.8, cryo.n_images).astype(np.float32)
    latent_precision = np.full(cryo.n_images, 4.0, dtype=np.float32)
    h_grid = np.array([1.0], dtype=np.float32)
    monkeypatch.chdir(tmp_path)

    default = lpr.estimate_local_polynomial_volumes(
        cryo,
        latent_differences,
        latent_precision,
        h_grid,
        degree=2,
        batch_size=4,
        tau=None,
        grid_correct=False,
        use_spherical_mask=False,
        upsampling_factor=1,
        return_real_space=True,
        bandwidth_batch_size=1,
    )
    explicit = lpr.estimate_local_polynomial_volumes(
        cryo,
        latent_differences,
        latent_precision,
        h_grid,
        degree=2,
        batch_size=4,
        tau=None,
        grid_correct=False,
        use_spherical_mask=False,
        upsampling_factor=1,
        return_real_space=True,
        bandwidth_batch_size=1,
        basis="monomial",
        pol_reg_type="none",
        pol_reg_eta=0.0,
    )

    np.testing.assert_allclose(explicit, default, atol=0, rtol=0)


def test_compute_and_save_reweighted_passes_local_poly_latent_inputs(monkeypatch, tmp_path):
    class _Dataset:
        tilt_series_flag = False
        grid_size = 4
        voxel_size = 1.0

        def split_halfset_array(self, arr, per_particle=False):
            arr = np.asarray(arr)
            return [arr[:2], arr[2:]]

    captured = {}

    def _capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    from recovar.heterogeneity import heterogeneity_volume

    monkeypatch.setattr(heterogeneity_volume, "make_volumes_kernel_estimate_local", _capture)

    zs = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    precision = np.ones((4, 1, 1), dtype=np.float32) * 4.0
    target = np.array([[0.5]], dtype=np.float32)
    multipliers = np.array([1.0, 2.0], dtype=np.float32)

    output_mod.compute_and_save_reweighted(
        _Dataset(),
        target,
        zs,
        precision,
        str(tmp_path),
        B_factor=0.0,
        n_bins=3,
        n_min_particles=1,
        kernel_regression_mode="local_poly",
        local_poly_degree=2,
        local_poly_bandwidth_multipliers=multipliers,
    )

    kwargs = captured["kwargs"]
    assert kwargs["kernel_regression_mode"] == "local_poly"
    assert kwargs["local_poly_degree"] == 2
    np.testing.assert_array_equal(kwargs["local_poly_bandwidth_multipliers"], multipliers)
    np.testing.assert_allclose(kwargs["local_poly_latent_differences"][0], np.array([-1.5, -0.5], dtype=np.float32))
    np.testing.assert_allclose(kwargs["local_poly_latent_differences"][1], np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(kwargs["local_poly_latent_precision"][0], np.array([4.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(kwargs["local_poly_latent_precision"][1], np.array([4.0, 4.0], dtype=np.float32))
