import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.reconstruction import regularization
import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def test_jax_scipy_nd_image_mean_inner_handles_zero_count_indices():
    inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    labels = np.array([0, 0, 2], dtype=np.int32)
    index = np.array([0, 1, 2, 3], dtype=np.int32)
    out = regularization.jax_scipy_nd_image_mean_inner(inp, labels=labels, index=index)
    out = np.asarray(out)
    assert np.all(np.isfinite(out))
    assert np.isclose(out[0], 1.5)
    assert np.isclose(out[1], 0.0)  # missing label
    assert np.isclose(out[2], 3.0)
    assert np.isclose(out[3], 0.0)  # missing label


def test_jax_scipy_nd_image_mean_complex_path_returns_complex64():
    inp = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    labels = np.array([0, 0], dtype=np.int32)
    index = np.array([0], dtype=np.int32)
    out = regularization.jax_scipy_nd_image_mean(inp, labels=labels, index=index)
    out = np.asarray(out)
    assert out.dtype == np.complex64
    assert np.isclose(out[0], 2 + 3j)


def test_get_fsc_gpu_returns_finite_values():
    shape = (4, 4, 4)
    rng = np.random.default_rng(0)
    v1 = (rng.normal(size=np.prod(shape)) + 1j * rng.normal(size=np.prod(shape))).astype(np.complex64)
    v2 = (rng.normal(size=np.prod(shape)) + 1j * rng.normal(size=np.prod(shape))).astype(np.complex64)
    fsc = regularization.get_fsc_gpu(v1, v2, shape, substract_shell_mean=False, frequency_shift=0)
    fsc = np.asarray(fsc)
    assert fsc.ndim == 1
    assert np.all(np.isfinite(fsc))


def test_compute_fsc_prior_gpu_v2_and_prior_iteration_are_finite():
    shape = (4, 4, 4)
    n = int(np.prod(shape))
    rng = np.random.default_rng(0)

    image0 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    image1 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    lhs = np.abs(rng.normal(size=n)).astype(np.float32) + 0.5
    prior0 = np.ones(n, dtype=np.float32)

    prior, fsc_raw, prior_avg = regularization.compute_fsc_prior_gpu_v2(
        shape, image0, image1, lhs, prior0, frequency_shift=0, substract_shell_mean=False, upsampling_factor=1
    )
    prior = np.asarray(prior)
    fsc_raw = np.asarray(fsc_raw)
    prior_avg = np.asarray(prior_avg)

    assert prior.shape == (n,)
    assert fsc_raw.ndim == 1
    assert prior_avg.ndim == 1
    assert np.all(np.isfinite(prior))
    assert np.all(np.isfinite(fsc_raw))
    assert np.all(np.isfinite(prior_avg))


def test_prior_iteration_relion_style_and_downsample_from_fsc():
    shape = (4, 4, 4)
    n = int(np.prod(shape))
    rng = np.random.default_rng(1)

    h0 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    h1 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    b0 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    b1 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    prior0 = np.ones(n, dtype=np.float32)

    cov_col, prior, fsc = regularization.prior_iteration_relion_style(
        H0=h0,
        H1=h1,
        B0=b0,
        B1=b1,
        frequency_shift=np.array([0, 0, 0], dtype=np.int32),
        init_regularization=prior0,
        substract_shell_mean=False,
        volume_shape=shape,
        kernel="triangular",
        use_spherical_mask=False,
        grid_correct=False,
        volume_mask=None,
        prior_iterations=2,
        downsample_from_fsc_flag=False,
    )
    cov_col = np.asarray(cov_col)
    prior = np.asarray(prior)
    fsc = np.asarray(fsc)
    assert cov_col.shape == (n,)
    assert prior.shape == (n,)
    assert fsc.ndim == 1
    assert np.all(np.isfinite(cov_col))
    assert np.all(np.isfinite(prior))
    assert np.all(np.isfinite(fsc))

    # Explicitly exercise downsample masking behavior.
    arr = np.ones(n, dtype=np.float32)
    # Keep low-freq bins, drop high-freq bins.
    fsc_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    out = regularization.downsample_from_fsc(arr, fsc_mask, shape)
    out = np.asarray(out)
    assert out.shape == (n,)
    assert np.any(out == 0.0)


def test_compute_relion_fsc_from_backprojector_uses_relion_rounding_and_half_layout():
    shape = (4, 4, 4)
    padded_shape = (8, 8, 8)

    data0 = np.zeros(padded_shape, dtype=np.complex128)
    data1 = np.zeros(padded_shape, dtype=np.complex128)
    weight0 = np.zeros(padded_shape, dtype=np.float64)
    weight1 = np.zeros(padded_shape, dtype=np.float64)

    # Valid RELION half-layout sample: logical RELION (z=0, y=2, x=0)
    # downsamples to shell 1 for padding_factor=2. In RECOVAR's centered
    # full-array layout, RELION x is axis 0 for this helper.
    valid_idx = (4, 4, 6)
    data0[valid_idx] = 1.0
    data1[valid_idx] = 1.0
    weight0[valid_idx] = 1.0
    weight1[valid_idx] = 1.0

    # Logical RELION x=-1 would banker-round to x=0, but RELION's ROUND(-0.5)
    # returns -1 and the compact half layout excludes it.
    invalid_negative_x_idx = (3, 4, 6)
    data0[invalid_negative_x_idx] = 100.0
    data1[invalid_negative_x_idx] = -100.0
    weight0[invalid_negative_x_idx] = 1.0
    weight1[invalid_negative_x_idx] = 1.0

    fsc = np.asarray(
        regularization.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            shape,
            padding_factor=2,
            r_max=1,
        )
    )

    assert fsc[0] == 1.0
    np.testing.assert_allclose(fsc[1], 1.0, atol=1e-7, rtol=1e-7)


def test_compute_relion_fsc_from_backprojector_accepts_packed_half_accumulators():
    shape = (4, 4, 4)
    padding_factor = 2
    padded_shape = tuple(s * padding_factor for s in shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    rng = np.random.default_rng(0)

    data0_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    data1_half = (
        rng.normal(size=half_shape).astype(np.float32)
        + 1j * rng.normal(size=half_shape).astype(np.float32)
    )
    weight0_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)
    weight1_half = (0.25 + rng.random(size=half_shape)).astype(np.float32)

    data0_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(data0_half, padded_shape))
    data1_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(data1_half, padded_shape))
    weight0_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(weight0_half, padded_shape))
    weight1_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(weight1_half, padded_shape))

    fsc_from_full = np.asarray(
        regularization.compute_relion_fsc_from_backprojector(
            data0_full.reshape(-1),
            data1_full.reshape(-1),
            weight0_full.reshape(-1),
            weight1_full.reshape(-1),
            shape,
            padding_factor=padding_factor,
        )
    )
    fsc_from_half = np.asarray(
        regularization.compute_relion_fsc_from_backprojector(
            data0_half.reshape(-1),
            data1_half.reshape(-1),
            weight0_half.reshape(-1),
            weight1_half.reshape(-1),
            shape,
            padding_factor=padding_factor,
        )
    )

    np.testing.assert_allclose(fsc_from_half, fsc_from_full, atol=1e-6, rtol=1e-6)


def test_compute_relion_fsc_from_backprojector_applies_exact_rmax_before_shell_binning():
    shape = (4, 4, 4)
    data0 = np.zeros(shape, dtype=np.complex128)
    data1 = np.zeros(shape, dtype=np.complex128)
    weight0 = np.ones(shape, dtype=np.float64)
    weight1 = np.ones(shape, dtype=np.float64)

    # Shifted-grid coordinate (z=1, y=0, x=0): exact R=1, shell 1.
    inside = (3, 2, 2)
    data0[inside] = 1.0
    data1[inside] = 1.0

    # Shifted-grid coordinate (z=1, y=1, x=0): exact R=sqrt(2) > r_max,
    # but ROUND(R) == 1. RELION excludes it before shell binning.
    rounded_shell_but_outside_rmax = (3, 3, 2)
    data0[rounded_shell_but_outside_rmax] = 1.0
    data1[rounded_shell_but_outside_rmax] = -1.0

    fsc = np.asarray(
        regularization.compute_relion_fsc_from_backprojector(
            data0.reshape(-1),
            data1.reshape(-1),
            weight0.reshape(-1),
            weight1.reshape(-1),
            shape,
            padding_factor=1,
            r_max=1,
        )
    )

    np.testing.assert_allclose(fsc[1], 1.0, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    ("padding_factor", "weight_value", "expected_sigma2"),
    [
        (1, 2.0, 1.0 / 2.0),
        (2, 2.0, 1.0 / (2.0 * 8.0)),
    ],
)
def test_compute_relion_tau2_from_weights_constant_weight_details(padding_factor, weight_value, expected_sigma2):
    shape = (8, 8, 8)
    n_shells = shape[0] // 2 + 1
    padded_shape = tuple(s * padding_factor for s in shape)
    weight = np.full(np.prod(padded_shape), weight_value, dtype=np.float32)
    fsc = np.full(n_shells, 0.5, dtype=np.float32)  # SSNR = 1

    prior, fsc_out, details = regularization.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        shape,
        padding_factor=padding_factor,
        return_details=True,
    )

    np.testing.assert_allclose(np.asarray(fsc_out), 0.5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(details["avg_weight_shells"]), weight_value, atol=1e-6)
    np.testing.assert_allclose(np.asarray(details["sigma2_shells"]), expected_sigma2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(details["prior_shells"]), expected_sigma2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(prior), expected_sigma2, atol=1e-6)
    assert np.asarray(details["shell_count"]).shape == (n_shells,)
    assert np.all(np.asarray(details["shell_count"]) > 0)


def test_compute_relion_tau2_from_weights_respects_relion_rmax_support():
    shape = (8, 8, 8)
    padding_factor = 2
    n_shells = shape[0] // 2 + 1
    padded_shape = tuple(s * padding_factor for s in shape)
    weight = np.ones(np.prod(padded_shape), dtype=np.float32)
    fsc = np.full(n_shells, 0.5, dtype=np.float32)

    prior, _, details = regularization.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )

    shell_count = np.asarray(details["shell_count"])
    prior_shells = np.asarray(details["prior_shells"])
    assert shell_count[4] == 0.0
    assert prior_shells[4] <= 1e-12

    radii = np.asarray(
        regularization.fourier_transform_utils.get_grid_of_radial_distances(
            shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1)
    )
    assert np.all(np.asarray(prior)[radii == 4] <= 1e-12)


def test_compute_relion_tau2_from_weights_rejects_wrong_grid_size():
    shape = (8, 8, 8)
    fsc = np.full(shape[0] // 2 + 1, 0.5, dtype=np.float32)
    bad_weight = np.ones(np.prod(shape) - 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected full or half Fourier weight"):
        regularization.compute_relion_tau2_from_weights(
            bad_weight,
            bad_weight,
            fsc,
            shape,
            padding_factor=1,
        )


def test_relion_weight_shell_stats_floor_bins_reconstruct_support():
    shape = (8, 8, 8)
    padding_factor = 2
    current_size = 6
    r_max = current_size // 2
    full_shape = tuple(s * padding_factor for s in shape)
    weight = np.ones(np.prod(full_shape), dtype=np.float32)

    stats_floor = regularization._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding="floor",
    )
    stats_round = regularization._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=r_max,
        shell_rounding="round",
    )

    coords = np.arange(-(full_shape[0] // 2), full_shape[0] // 2)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    mask = (xx >= 0) & (radius < padding_factor * r_max)
    shell = np.floor(radius / padding_factor).astype(int)
    expected = np.bincount(shell[mask].ravel(), minlength=shape[0] // 2 + 1)

    np.testing.assert_array_equal(np.asarray(stats_floor["shell_count"])[: expected.shape[0]], expected)
    assert not np.array_equal(
        np.asarray(stats_floor["shell_count"])[: expected.shape[0]],
        np.asarray(stats_round["shell_count"])[: expected.shape[0]],
    )


def test_relion_weight_shell_stats_large_grid_cpu_path_matches_device_path(monkeypatch):
    shape = (8, 8, 8)
    padding_factor = 2
    full_shape = tuple(s * padding_factor for s in shape)
    rng = np.random.default_rng(10)
    weight = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)

    device_stats = regularization._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        shell_rounding="round",
    )
    monkeypatch.setattr(regularization, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    cpu_stats = regularization._compute_relion_weight_shell_stats(
        weight,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        shell_rounding="round",
    )

    for key in ("shell_sum", "shell_count", "avg_weight_shells"):
        np.testing.assert_allclose(np.asarray(cpu_stats[key]), np.asarray(device_stats[key]), rtol=1e-6, atol=1e-6)


def test_compute_relion_tau2_from_weights_large_grid_cpu_path_matches_device_path(monkeypatch):
    shape = (8, 8, 8)
    padding_factor = 2
    full_shape = tuple(s * padding_factor for s in shape)
    n_shells = shape[0] // 2 + 1
    rng = np.random.default_rng(11)
    weight0 = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)
    weight1 = (0.25 + rng.random(np.prod(full_shape))).astype(np.float32)
    fsc = np.linspace(0.95, 0.25, n_shells, dtype=np.float32)

    prior_device, _, details_device = regularization.compute_relion_tau2_from_weights(
        weight0,
        weight1,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )
    monkeypatch.setattr(regularization, "_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS", 1)
    prior_cpu, _, details_cpu = regularization.compute_relion_tau2_from_weights(
        weight0,
        weight1,
        fsc,
        shape,
        padding_factor=padding_factor,
        r_max=3,
        return_details=True,
    )

    np.testing.assert_allclose(np.asarray(prior_cpu), np.asarray(prior_device), rtol=1e-6, atol=1e-6)
    for key in ("shell_sum", "shell_count", "avg_weight_shells", "prior_shells"):
        np.testing.assert_allclose(
            np.asarray(details_cpu[key]),
            np.asarray(details_device[key]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_compute_relion_tau2_from_iref_power_spectrum_matches_relion_binding_scaling():
    from pathlib import Path

    from recovar.utils.helpers import load_relion_volume

    relion_dir = Path(
        "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128/relion_pdb_k4_os0_ref"
    )
    volume_path = relion_dir / "run_it000_class001.mrc"
    model_path = relion_dir / "run_it001_model.star"

    vol_recovar = np.asarray(load_relion_volume(str(volume_path)), dtype=np.float64)
    ft_recovar = np.asarray(fourier_transform_utils.get_dft3(jnp.asarray(vol_recovar)).reshape(-1))

    tau2, details = regularization.compute_relion_tau2_from_iref_power_spectrum(
        ft_recovar,
        vol_recovar.shape,
        padding_factor=2,
        current_size=56,
        return_details=True,
    )

    expected_tau2 = []
    in_class = False
    in_loop = False
    for line in model_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "data_model_class_1":
            in_class = True
            continue
        if in_class and stripped == "loop_":
            in_loop = True
            continue
        if in_loop and stripped.startswith("data_model_"):
            break
        if in_loop and stripped and stripped[0].isdigit():
            expected_tau2.append(float(stripped.split()[7]))
            if len(expected_tau2) == 2:
                break

    assert tau2.shape == (np.prod(vol_recovar.shape),)
    assert details["tau2_shells"].shape[0] >= len(expected_tau2)
    np.testing.assert_allclose(np.asarray(details["tau2_shells"][: len(expected_tau2)]), expected_tau2, rtol=2e-2, atol=2e-6)


# ---------------------------------------------------------------------------
# covariance_update_col tests
# ---------------------------------------------------------------------------


def test_covariance_update_col_basic_formula():
    """Verify Wiener filter: cov = B / (H + 1/prior)."""
    import jax.numpy as jnp

    H = np.array([2.0, 4.0], dtype=np.float32)
    B = np.array([6.0, 8.0], dtype=np.float32)
    prior = np.array([1.0, 2.0], dtype=np.float32)
    out = np.asarray(regularization.covariance_update_col(jnp.array(H), jnp.array(B), jnp.array(prior)))
    expected = B / (H + 1.0 / prior)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_covariance_update_col_zeros_below_epsilon():
    """Voxels with H ~ 0 must produce zero output."""
    import jax.numpy as jnp

    H = np.array([0.0, 3.0], dtype=np.float32)
    B = np.array([99.0, 6.0], dtype=np.float32)
    prior = np.array([1.0, 1.0], dtype=np.float32)
    out = np.asarray(regularization.covariance_update_col(jnp.array(H), jnp.array(B), jnp.array(prior)))
    assert float(out[0]) == 0.0, "Uncovered voxel should produce zero"
    assert np.isfinite(out[1])


def test_covariance_update_col_complex_b():
    """Works with complex B (common case)."""
    import jax.numpy as jnp

    H = np.array([2.0], dtype=np.float32)
    B = np.array([2.0 + 4.0j], dtype=np.complex64)
    prior = np.array([1.0], dtype=np.float32)
    out = np.asarray(regularization.covariance_update_col(jnp.array(H), jnp.array(B), jnp.array(prior)))
    expected = B / (H + 1.0 / prior)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# average_over_shells / sum_over_shells tests
# ---------------------------------------------------------------------------


def test_average_over_shells_uniform_input():
    """Uniform input -> every populated shell average is the uniform value."""
    shape = (8, 8, 8)
    n = int(np.prod(shape))
    ones = np.ones(n, dtype=np.float32)
    out = np.asarray(regularization.average_over_shells(ones, shape))
    assert out.shape == (shape[0] // 2 - 1,)
    populated = out > 0
    np.testing.assert_allclose(out[populated], 1.0, rtol=1e-5)


def test_average_over_shells_output_shape():
    shape = (6, 6, 6)
    n = int(np.prod(shape))
    x = np.random.default_rng(0).random(n).astype(np.float32)
    out = np.asarray(regularization.average_over_shells(x, shape))
    assert out.shape == (shape[0] // 2 - 1,)


def test_sum_over_shells_output_shape():
    shape = (8, 8, 8)
    n = int(np.prod(shape))
    x = np.ones(n, dtype=np.float32)
    out = np.asarray(regularization.sum_over_shells(x, shape))
    assert out.shape == (shape[0] // 2 - 1,)


def test_sum_vs_average_uniform():
    """For uniform input, sum >= average for every populated shell."""
    shape = (8, 8, 8)
    n = int(np.prod(shape))
    x = np.ones(n, dtype=np.float32)
    sums = np.asarray(regularization.sum_over_shells(x, shape))
    avgs = np.asarray(regularization.average_over_shells(x, shape))
    populated = avgs > 0
    assert np.all(sums[populated] >= avgs[populated] - 1e-5)


def test_sum_over_shells_nonneg_input():
    """Non-negative input produces non-negative sums."""
    shape = (8, 8, 8)
    n = int(np.prod(shape))
    x = np.abs(np.random.default_rng(42).random(n)).astype(np.float32)
    out = np.asarray(regularization.sum_over_shells(x, shape))
    assert np.all(out >= -1e-7)


# ---------------------------------------------------------------------------
# downsample_lhs tests
# ---------------------------------------------------------------------------


def test_downsample_lhs_shape_factor1():
    """Factor=1 preserves shape."""
    import jax.numpy as jnp

    shape = (4, 4, 4)
    lhs = jnp.ones(shape, dtype=jnp.float32)
    out = regularization.downsample_lhs(lhs, shape, upsampling_factor=1)
    assert out.shape == shape


def test_downsample_lhs_shape_factor2():
    """Factor=2 halves each dimension."""
    import jax.numpy as jnp

    shape = (8, 8, 8)
    lhs = jnp.ones(shape, dtype=jnp.float32)
    out = regularization.downsample_lhs(lhs, shape, upsampling_factor=2)
    assert out.shape == (4, 4, 4)


def test_downsample_lhs_nonnegative():
    """Output is clipped to >= 0."""
    import jax.numpy as jnp

    shape = (4, 4, 4)
    rng = np.random.default_rng(7)
    lhs = jnp.array(rng.normal(size=shape).astype(np.float32))
    out = np.asarray(regularization.downsample_lhs(lhs, shape, upsampling_factor=1))
    assert np.all(out >= -1e-7)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_jax_scipy_nd_image_mean_inner_gpu(gpu_device):
    inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    labels = np.array([0, 0, 2], dtype=np.int32)
    index = np.array([0, 1, 2, 3], dtype=np.int32)

    cpu_out = np.asarray(regularization.jax_scipy_nd_image_mean_inner(inp, labels=labels, index=index))

    with jax.default_device(gpu_device):
        inp_g = jax.device_put(jnp.array(inp), gpu_device)
        labels_g = jax.device_put(jnp.array(labels), gpu_device)
        index_g = jax.device_put(jnp.array(index), gpu_device)
        gpu_out = np.asarray(regularization.jax_scipy_nd_image_mean_inner(inp_g, labels=labels_g, index=index_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_get_fsc_gpu_on_gpu(gpu_device):
    shape = (4, 4, 4)
    rng = np.random.default_rng(0)
    v1 = (rng.normal(size=np.prod(shape)) + 1j * rng.normal(size=np.prod(shape))).astype(np.complex64)
    v2 = (rng.normal(size=np.prod(shape)) + 1j * rng.normal(size=np.prod(shape))).astype(np.complex64)

    cpu_fsc = np.asarray(regularization.get_fsc_gpu(v1, v2, shape, substract_shell_mean=False, frequency_shift=0))

    with jax.default_device(gpu_device):
        v1_g = jax.device_put(jnp.array(v1), gpu_device)
        v2_g = jax.device_put(jnp.array(v2), gpu_device)
        gpu_fsc = np.asarray(
            regularization.get_fsc_gpu(v1_g, v2_g, shape, substract_shell_mean=False, frequency_shift=0)
        )

    np.testing.assert_allclose(cpu_fsc, gpu_fsc, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_compute_fsc_prior_gpu_v2_on_gpu(gpu_device):
    shape = (4, 4, 4)
    n = int(np.prod(shape))
    rng = np.random.default_rng(0)

    image0 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    image1 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    lhs = np.abs(rng.normal(size=n)).astype(np.float32) + 0.5
    prior0 = np.ones(n, dtype=np.float32)

    cpu_prior, cpu_fsc, cpu_avg = regularization.compute_fsc_prior_gpu_v2(
        shape, image0, image1, lhs, prior0, frequency_shift=0, substract_shell_mean=False, upsampling_factor=1
    )
    cpu_prior, cpu_fsc, cpu_avg = np.asarray(cpu_prior), np.asarray(cpu_fsc), np.asarray(cpu_avg)

    with jax.default_device(gpu_device):
        i0_g = jax.device_put(jnp.array(image0), gpu_device)
        i1_g = jax.device_put(jnp.array(image1), gpu_device)
        lhs_g = jax.device_put(jnp.array(lhs), gpu_device)
        p0_g = jax.device_put(jnp.array(prior0), gpu_device)
        gpu_prior, gpu_fsc, gpu_avg = regularization.compute_fsc_prior_gpu_v2(
            shape, i0_g, i1_g, lhs_g, p0_g, frequency_shift=0, substract_shell_mean=False, upsampling_factor=1
        )
        gpu_prior, gpu_fsc, gpu_avg = np.asarray(gpu_prior), np.asarray(gpu_fsc), np.asarray(gpu_avg)

    np.testing.assert_allclose(cpu_prior, gpu_prior, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpu_fsc, gpu_fsc, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(cpu_avg, gpu_avg, atol=1e-4, rtol=1e-4)
