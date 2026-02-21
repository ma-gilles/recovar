import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import regularization

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
