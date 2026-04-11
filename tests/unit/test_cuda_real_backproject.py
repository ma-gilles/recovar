"""Tests for real-valued CUDA backproject (float32/float64 inputs).

Real-valued backprojection is used for Fourier quantities that are inherently
real (CTF^2, noise variance, CTF^4).  The CUDA kernel uses 1 float per voxel
(not 2) and 1 atomicAdd per scatter (not 2), giving 2x memory and scatter
efficiency compared to promoting to complex.

Reference: complex backproject with the same data (as complex with zero
imaginary part), then take .real of the result.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _use_custom_cuda_lib(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)


def _skip_if_no_cuda():
    from recovar.cuda_backproject import cuda_available

    if not cuda_available():
        pytest.skip("CUDA backproject not available")


def _random_rotations(n, rng):
    """Generate n proper rotation matrices."""
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q.astype(np.float32)


# ── Parametrize over order and half_vol/half_img combos ──

_COMBOS = [
    (0, False, False),
    (0, False, True),
    (0, True, False),
    (0, True, True),
    (1, False, False),
    (1, False, True),
    (1, True, False),
    (1, True, True),
]


@pytest.mark.parametrize("order,half_vol,half_img", _COMBOS)
def test_real_backproject_matches_complex(order, half_vol, half_img):
    """Real backproject should match .real of complex backproject."""
    _skip_if_no_cuda()
    import recovar.core.fourier_transform_utils as ftu
    from recovar.cuda_backproject import backproject

    N = 16
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_images = 5
    rng = np.random.default_rng(42)

    rots = jnp.array(_random_rotations(n_images, rng))

    # Real-valued "images" (e.g. CTF^2)
    iw = N // 2 + 1 if half_img else N
    n_pix = N * iw
    real_imgs = jnp.array(rng.standard_normal((n_images, n_pix)).astype(np.float32))

    # Complex version: same data with zero imaginary
    complex_imgs = real_imgs.astype(jnp.complex64)

    # Volume sizes
    if half_vol:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(half_shape))
    else:
        vol_size = int(np.prod(volume_shape))

    # Real backproject
    real_vol = jnp.zeros(vol_size, dtype=jnp.float32)
    result_real = backproject(
        real_vol, real_imgs, rots, image_shape, volume_shape, order=order, half_volume=half_vol, half_image=half_img
    )

    # Complex backproject
    complex_vol = jnp.zeros(vol_size, dtype=jnp.complex64)
    result_complex = backproject(
        complex_vol,
        complex_imgs,
        rots,
        image_shape,
        volume_shape,
        order=order,
        half_volume=half_vol,
        half_image=half_img,
    )

    # Real result should match .real of complex result
    np.testing.assert_allclose(
        np.array(result_real),
        np.array(result_complex.real),
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"order={order}, half_vol={half_vol}, half_img={half_img}",
    )
    # Imaginary part of complex result should be zero (real input)
    assert np.max(np.abs(np.array(result_complex.imag))) < 1e-5


def test_real_backproject_output_dtype():
    """Real backproject should return float32, not complex64."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject

    N = 8
    rng = np.random.default_rng(123)
    rots = jnp.array(_random_rotations(2, rng))
    imgs = jnp.array(rng.standard_normal((2, N * N)).astype(np.float32))
    vol = jnp.zeros(N**3, dtype=jnp.float32)

    result = backproject(vol, imgs, rots, (N, N), (N, N, N), order=1)
    assert result.dtype == jnp.float32


def test_real_backproject_accumulator():
    """Real backproject with non-zero initial volume should accumulate."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject

    N = 8
    rng = np.random.default_rng(99)
    rots = jnp.array(_random_rotations(3, rng))
    imgs = jnp.array(rng.standard_normal((3, N * N)).astype(np.float32))

    # Start from zero
    vol0 = jnp.zeros(N**3, dtype=jnp.float32)
    r1 = backproject(vol0, imgs, rots, (N, N), (N, N, N), order=1)

    # Start from ones — result should be r1 + 1
    vol1 = jnp.ones(N**3, dtype=jnp.float32)
    r2 = backproject(vol1, imgs, rots, (N, N), (N, N, N), order=1)

    np.testing.assert_allclose(np.array(r2), np.array(r1) + 1.0, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("order,half_vol,half_img", _COMBOS)
def test_real_batch_backproject_matches_complex(order, half_vol, half_img):
    """Real batch_backproject should match .real of complex batch_backproject."""
    _skip_if_no_cuda()
    import recovar.core.fourier_transform_utils as ftu
    from recovar.cuda_backproject import batch_backproject

    N = 12
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_images = 3
    batch_size = 4
    rng = np.random.default_rng(77)

    rots = jnp.array(_random_rotations(n_images, rng))

    iw = N // 2 + 1 if half_img else N
    n_pix = N * iw
    real_imgs = jnp.array(rng.standard_normal((batch_size, n_images, n_pix)).astype(np.float32))
    complex_imgs = real_imgs.astype(jnp.complex64)

    if half_vol:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(half_shape))
    else:
        vol_size = int(np.prod(volume_shape))

    real_vols = jnp.zeros((batch_size, vol_size), dtype=jnp.float32)
    result_real = batch_backproject(
        real_vols, real_imgs, rots, image_shape, volume_shape, order=order, half_volume=half_vol, half_image=half_img
    )

    complex_vols = jnp.zeros((batch_size, vol_size), dtype=jnp.complex64)
    result_complex = batch_backproject(
        complex_vols,
        complex_imgs,
        rots,
        image_shape,
        volume_shape,
        order=order,
        half_volume=half_vol,
        half_image=half_img,
    )

    np.testing.assert_allclose(
        np.array(result_real),
        np.array(result_complex.real),
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"batch: order={order}, half_vol={half_vol}, half_img={half_img}",
    )


def test_real_adjoint_slice_volume():
    """adjoint_slice_volume with real input should return real output."""
    _skip_if_no_cuda()
    import recovar.core.slicing as slicing

    N = 12
    n_images = 4
    rng = np.random.default_rng(55)
    rots = jnp.array(_random_rotations(n_images, rng))

    # Real slices (e.g. CTF^2)
    real_slices = jnp.array(rng.standard_normal((n_images, N * N)).astype(np.float32))

    result = slicing.adjoint_slice_volume(real_slices, rots, (N, N), (N, N, N), "linear_interp")
    assert result.dtype == jnp.float32

    # Compare to complex path
    complex_slices = real_slices.astype(jnp.complex64)
    result_complex = slicing.adjoint_slice_volume(complex_slices, rots, (N, N), (N, N, N), "linear_interp")
    np.testing.assert_allclose(
        np.array(result),
        np.array(result_complex.real),
        atol=1e-5,
        rtol=1e-5,
    )
