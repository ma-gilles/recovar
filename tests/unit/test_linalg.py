import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.fourier_transform_utils as fourier_transform_utils
import recovar.linalg as linalg

pytestmark = pytest.mark.unit


def test_blockwise_xtx_matches_direct():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(12, 5)) + 1j * rng.normal(size=(12, 5))).astype(np.complex64)
    out = linalg.blockwise_X_T_X(x, batch_size=4)
    ref = np.conj(x).T @ x
    np.testing.assert_allclose(out, ref, atol=5e-3, rtol=5e-3)


def test_broadcast_dot_and_outer():
    x = np.array([[1 + 1j, 2 + 0j], [0 + 1j, 3 + 2j]], dtype=np.complex64)
    y = np.array([[2 + 0j, 1 + 1j], [1 - 1j, 0 + 2j]], dtype=np.complex64)

    dot = np.asarray(linalg.broadcast_dot(x, y))
    ref_dot = np.sum(np.conj(x) * y, axis=-1)
    np.testing.assert_allclose(dot, ref_dot, atol=1e-6, rtol=1e-6)

    outer = np.asarray(linalg.broadcast_outer(x, y))
    ref_outer = np.stack([np.outer(xi, np.conj(yi)) for xi, yi in zip(x, y)], axis=0)
    np.testing.assert_allclose(outer, ref_outer, atol=1e-6, rtol=1e-6)


def test_inner_product_and_batch_inner_product():
    rng = np.random.default_rng(17)
    x = (rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))).astype(np.complex64)
    y = (rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))).astype(np.complex64)

    nonbatch = np.asarray(linalg.inner_product(x[0], y[0]))
    ref_nonbatch = np.vdot(x[0], y[0])
    np.testing.assert_allclose(nonbatch, ref_nonbatch, atol=1e-6, rtol=1e-6)

    batch = np.asarray(linalg.batch_inner_product(x, y))
    ref_batch = np.array([np.vdot(xi, yi) for xi, yi in zip(x, y)], dtype=np.complex64)
    np.testing.assert_allclose(batch, ref_batch, atol=1e-6, rtol=1e-6)


def test_inner_product_shape_validation():
    with pytest.raises(ValueError, match="same shape"):
        linalg.inner_product(np.zeros((2,), dtype=np.complex64), np.zeros((3,), dtype=np.complex64))
    with pytest.raises(ValueError, match="same shape"):
        linalg.batch_inner_product(np.zeros((2, 3), dtype=np.complex64), np.zeros((2, 4), dtype=np.complex64))
    with pytest.raises(ValueError, match="ndim>=2"):
        linalg.batch_inner_product(np.zeros((3,), dtype=np.complex64), np.zeros((3,), dtype=np.complex64))


def test_half_spectrum_last_axis_weights_even_and_odd():
    np.testing.assert_array_equal(
        np.asarray(linalg.half_spectrum_last_axis_weights(8)),
        np.array([1, 2, 2, 2, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(linalg.half_spectrum_last_axis_weights(7)),
        np.array([1, 2, 2, 2], dtype=np.float32),
    )


def test_half_spectrum_inner_product_corrects_naive_dot_and_matches_full_fft():
    rng = np.random.default_rng(29)
    shape = (6, 8)
    x = rng.normal(size=shape).astype(np.float32)
    y = rng.normal(size=shape).astype(np.float32)

    fx = np.asarray(fourier_transform_utils.get_dft2(x))
    fy = np.asarray(fourier_transform_utils.get_dft2(y))
    sfx = np.asarray(fourier_transform_utils.get_dft2_real(x))
    sfy = np.asarray(fourier_transform_utils.get_dft2_real(y))

    full_ip = np.asarray(linalg.inner_product(fx, fy))
    naive_half_ip = np.vdot(sfx, sfy)
    corrected_half_ip = np.asarray(linalg.half_spectrum_inner_product(sfx, sfy, shape))

    # Naive packed-spectrum dot product misses Hermitian-pair contributions.
    assert np.abs(corrected_half_ip - naive_half_ip) > 1e-2
    np.testing.assert_allclose(corrected_half_ip, full_ip, atol=1e-5, rtol=1e-5)

    # Parseval under "backward" FFT normalization.
    spatial_ip = np.vdot(x, y)
    np.testing.assert_allclose(corrected_half_ip / np.prod(shape), spatial_ip, atol=1e-5, rtol=1e-5)


def test_batch_half_spectrum_inner_product_matches_full_fft_and_spatial_scaling():
    rng = np.random.default_rng(31)
    shape = (6, 8)
    batch = 4
    x = rng.normal(size=(batch,) + shape).astype(np.float32)
    y = rng.normal(size=(batch,) + shape).astype(np.float32)

    fx = np.asarray(fourier_transform_utils.get_dft2(x))
    fy = np.asarray(fourier_transform_utils.get_dft2(y))
    sfx = np.asarray(fourier_transform_utils.get_dft2_real(x))
    sfy = np.asarray(fourier_transform_utils.get_dft2_real(y))

    full_ip = np.asarray(linalg.batch_inner_product(fx, fy))
    corrected_half_ip = np.asarray(linalg.batch_half_spectrum_inner_product(sfx, sfy, shape))
    np.testing.assert_allclose(corrected_half_ip, full_ip, atol=1e-5, rtol=1e-5)

    spatial_ip = np.sum(np.conj(x) * y, axis=(-2, -1))
    np.testing.assert_allclose(corrected_half_ip / np.prod(shape), spatial_ip, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("full_shape", [(4, 8), (4, 4, 8)])
def test_half_spectrum_inner_products_match_full_case(full_shape):
    rng = np.random.default_rng(19)
    if len(full_shape) == 2:
        x_real = rng.normal(size=full_shape).astype(np.float32)
        y_real = rng.normal(size=full_shape).astype(np.float32)
        x_full = np.asarray(fourier_transform_utils.get_dft2(x_real))
        y_full = np.asarray(fourier_transform_utils.get_dft2(y_real))
        x_half = np.asarray(fourier_transform_utils.get_dft2_real(x_real))
        y_half = np.asarray(fourier_transform_utils.get_dft2_real(y_real))
    else:
        x_real = rng.normal(size=full_shape).astype(np.float32)
        y_real = rng.normal(size=full_shape).astype(np.float32)
        x_full = np.asarray(fourier_transform_utils.get_dft3(x_real))
        y_full = np.asarray(fourier_transform_utils.get_dft3(y_real))
        x_half = np.asarray(fourier_transform_utils.get_dft3_real(x_real))
        y_half = np.asarray(fourier_transform_utils.get_dft3_real(y_real))

    full_ip = np.asarray(linalg.inner_product(x_full, y_full))
    half_ip = np.asarray(linalg.half_spectrum_inner_product(x_half, y_half, full_shape))
    np.testing.assert_allclose(half_ip, full_ip, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("full_shape", [(4, 8), (4, 4, 8)])
def test_batch_half_spectrum_inner_products_match_full_case(full_shape):
    rng = np.random.default_rng(23)
    batch = 3
    if len(full_shape) == 2:
        x_real = rng.normal(size=(batch,) + full_shape).astype(np.float32)
        y_real = rng.normal(size=(batch,) + full_shape).astype(np.float32)
        x_full = np.asarray(fourier_transform_utils.get_dft2(x_real))
        y_full = np.asarray(fourier_transform_utils.get_dft2(y_real))
        x_half = np.asarray(fourier_transform_utils.get_dft2_real(x_real))
        y_half = np.asarray(fourier_transform_utils.get_dft2_real(y_real))
    else:
        x_real = rng.normal(size=(batch,) + full_shape).astype(np.float32)
        y_real = rng.normal(size=(batch,) + full_shape).astype(np.float32)
        x_full = np.asarray(fourier_transform_utils.get_dft3(x_real))
        y_full = np.asarray(fourier_transform_utils.get_dft3(y_real))
        x_half = np.asarray(fourier_transform_utils.get_dft3_real(x_real))
        y_half = np.asarray(fourier_transform_utils.get_dft3_real(y_real))

    full_ip = np.asarray(linalg.batch_inner_product(x_full, y_full))
    half_ip = np.asarray(linalg.batch_half_spectrum_inner_product(x_half, y_half, full_shape))
    np.testing.assert_allclose(half_ip, full_ip, atol=1e-3, rtol=1e-4)


def test_half_spectrum_inner_product_rejects_bad_shape():
    with pytest.raises(ValueError, match="2 or 3 dims"):
        linalg.half_spectrum_inner_product(
            np.zeros((2, 2), dtype=np.complex64),
            np.zeros((2, 2), dtype=np.complex64),
            full_shape=(2,),
        )


def test_multiply_along_axis_and_l2_distance():
    a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.asarray(linalg.multiply_along_axis(a, b, axis=1))
    np.testing.assert_allclose(out[:, 0, :], a[:, 0, :] * 1.0)
    np.testing.assert_allclose(out[:, 1, :], a[:, 1, :] * 2.0)
    np.testing.assert_allclose(out[:, 2, :], a[:, 2, :] * 3.0)

    x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    d = np.asarray(linalg.l2_distance(x, y))
    ref = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(d, ref, atol=1e-6, rtol=1e-6)
