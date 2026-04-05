import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.reconstruction import regularization

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

    h0 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    h1 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    b0 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    b1 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)

    p, f = regularization.prior_iteration(
        H0=h0,
        H1=h1,
        B0=b0,
        B1=b1,
        frequency_shift=np.array([0, 0, 0], dtype=np.int32),
        init_regularization=prior0,
        substract_shell_mean=False,
        volume_shape=shape,
        prior_iterations=3,
    )
    p = np.asarray(p)
    f = np.asarray(f)
    assert p.shape == (n,)
    assert f.ndim == 1
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(f))


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


def test_compute_relion_prior_from_reconstruction_stats_matches_direct_weight_formula():
    shape = (4, 4, 4)
    n = int(np.prod(shape))
    rng = np.random.default_rng(2)

    h0 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    h1 = np.abs(rng.normal(size=n)).astype(np.float32) + 1.0
    b0 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    b1 = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    prior0 = np.ones(n, dtype=np.float32)

    prior, fsc = regularization.compute_relion_prior_from_reconstruction_stats(
        h0,
        h1,
        b0,
        b1,
        shape,
        prior0,
        padding_factor=2,
        prior_iterations=2,
    )
    prior = np.asarray(prior)
    fsc = np.asarray(fsc)

    h0_pad = regularization.relion_functions.zero_pad_fourier_volume(h0, shape, 2)
    h1_pad = regularization.relion_functions.zero_pad_fourier_volume(h1, shape, 2)
    b0_pad = regularization.relion_functions.zero_pad_fourier_volume(b0, shape, 2)
    b1_pad = regularization.relion_functions.zero_pad_fourier_volume(b1, shape, 2)
    unreg0 = regularization.relion_functions.post_process_from_filter_v2(
        h0_pad,
        b0_pad,
        shape,
        volume_upsampling_factor=2,
        tau=None,
        kernel="triangular",
        use_spherical_mask=True,
        grid_correct=True,
        gridding_correct="square",
    )
    unreg1 = regularization.relion_functions.post_process_from_filter_v2(
        h1_pad,
        b1_pad,
        shape,
        volume_upsampling_factor=2,
        tau=None,
        kernel="triangular",
        use_spherical_mask=True,
        grid_correct=True,
        gridding_correct="square",
    )
    expected_prior, expected_fsc, _ = regularization.compute_fsc_prior_gpu(
        shape,
        unreg0,
        unreg1,
        (h0 + h1) / 2.0,
    )

    assert prior.shape == (n,)
    assert fsc.ndim == 1
    assert np.all(np.isfinite(prior))
    assert np.all(np.isfinite(fsc))
    np.testing.assert_allclose(prior, np.asarray(expected_prior), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fsc, np.asarray(expected_fsc), rtol=1e-5, atol=1e-5)


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
