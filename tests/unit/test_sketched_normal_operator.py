"""Tests for recovar.ppca.sketched_normal — all native half-image/half-volume."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig
from recovar.ppca.sketched_normal import _sketched_normal_batch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ones_ctf(ctf_params, image_shape, voxel_size, **kw):
    half_image = kw.get("half_image", False)
    shape = ftu.image_shape_to_half_image_shape(image_shape) if half_image else image_shape
    return np.ones((ctf_params.shape[0], int(np.prod(shape))), dtype=np.float32)


def _make_config(grid_size=8, disc_type="linear_interp"):
    """Default to linear_interp to match per_image_backproject CUDA kernel."""
    return ForwardModelConfig(
        image_shape=(grid_size, grid_size),
        volume_shape=(grid_size,) * 3,
        grid_size=grid_size,
        voxel_size=1.0,
        padding=0,
        disc_type=disc_type,
        ctf=core.as_ctf_evaluator(_ones_ctf),
    )


def _hermitian_half_vol(rng, volume_shape):
    """Random Hermitian volume in half-volume layout."""
    real_space = rng.normal(size=volume_shape).astype(np.float32)
    fourier = np.fft.fftn(np.fft.fftshift(real_space))
    half = ftu.full_volume_to_half_volume(
        fourier.astype(np.complex64).reshape(-1), volume_shape
    )
    return half


def _hermitian_half_images(rng, n, image_shape):
    """Random Hermitian images in half-image layout."""
    real_space = rng.normal(size=(n, *image_shape)).astype(np.float32)
    fourier = np.fft.fft2(np.fft.fftshift(real_space, axes=(-2, -1)))
    full_flat = fourier.reshape(n, -1).astype(np.complex64)
    return np.asarray(ftu.full_image_to_half_image(full_flat, image_shape))


def _random_rotations(rng, n):
    mats = rng.normal(size=(n, 3, 3)).astype(np.float32)
    out = []
    for i in range(n):
        q, r = np.linalg.qr(mats[i])
        q *= np.sign(np.linalg.det(q))
        out.append(q)
    return np.stack(out)


def _setup(rng, grid_size=8, n_images=4, rank=2, sketch_rank=3, qrank=2):
    config = _make_config(grid_size=grid_size)
    vs = config.volume_shape
    half_vol_shape = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vol_shape))
    half_im_shape = ftu.image_shape_to_half_image_shape(config.image_shape)
    half_im_size = int(np.prod(half_im_shape))

    U_X_half = jnp.array(np.stack([_hermitian_half_vol(rng, vs) for _ in range(rank)], axis=-1))
    sigma_X = jnp.array(rng.uniform(0.5, 2.0, size=(rank,)).astype(np.float32))
    V_X = jnp.array(rng.normal(size=(n_images, rank)).astype(np.float32))
    images_half = jnp.array(_hermitian_half_images(rng, n_images, config.image_shape))
    mean_half = jnp.array(_hermitian_half_vol(rng, vs))
    rotations = jnp.array(_random_rotations(rng, n_images))
    translations = jnp.zeros((n_images, 2), dtype=np.float32)
    ctf_params = jnp.zeros((n_images, 9), dtype=np.float32)
    noise_half = jnp.ones((n_images, half_im_size), dtype=np.float32)

    S_left_half = jnp.array(np.stack([_hermitian_half_vol(rng, vs) for _ in range(sketch_rank)]))
    Q_right = jnp.array(rng.normal(size=(n_images, qrank)).astype(np.float32))

    rfft_w = jnp.tile(
        linalg.half_spectrum_last_axis_weights(config.image_shape[1]),
        (config.image_shape[0], 1),
    ).reshape(-1)

    return dict(
        config=config, U_X_half=U_X_half, sigma_X=sigma_X, V_X=V_X,
        images_half=images_half, mean_half=mean_half,
        rotations=rotations, translations=translations,
        ctf_params=ctf_params, noise_half=noise_half,
        S_left_half=S_left_half, Q_right=Q_right,
        rfft_w=rfft_w,
        half_vol_size=half_vol_size, half_im_size=half_im_size,
    )


def _call_batch(s, S_left_half=None, Q_batch=None):
    """Helper to call _sketched_normal_batch with setup dict."""
    c = s["config"]
    return _sketched_normal_batch(
        s["images_half"], s["mean_half"], s["U_X_half"], s["sigma_X"],
        s["V_X"], s["ctf_params"], s["rotations"], s["translations"],
        s["noise_half"], c.voxel_size,
        c.image_shape, c.volume_shape, c.ctf, c.disc_type, c.disc_type,
        S_left_half=S_left_half, Q_batch=Q_batch,
    )


# ---------------------------------------------------------------------------
# Test 1: Right sketch vs per-column backprojection
# ---------------------------------------------------------------------------

def test_right_sketch_vs_per_column():
    """Right sketch = per_image_backproject(CTF_w * r) @ Q."""
    rng = np.random.default_rng(42)
    s = _setup(rng)
    c = s["config"]

    residual, right_contrib, _ = _call_batch(s, Q_batch=s["Q_right"])

    # Reference: backproject each image, then matmul by Q
    CTF_w = jnp.ones_like(s["noise_half"])
    adjoint_input_half = CTF_w * residual
    adjoint_input_full = ftu.half_image_to_full_image(adjoint_input_half, c.image_shape)

    n_images = residual.shape[0]
    half_vol_size = s["half_vol_size"]
    bp = np.zeros((half_vol_size, n_images), dtype=np.float32)
    for i in range(n_images):
        g_i = core.adjoint_slice_volume(
            adjoint_input_full[i:i+1], s["rotations"][i:i+1],
            c.image_shape, c.volume_shape, c.disc_type,
            half_volume=True,
        )
        bp[:, i] = np.asarray(g_i).real

    ref = bp @ np.asarray(s["Q_right"])  # (half_vol, qrank)

    # Tolerance for float32 differences between per_image_backproject CUDA
    # kernel and per-column adjoint_slice_volume.
    np.testing.assert_allclose(
        np.asarray(right_contrib), ref,
        atol=2.0, rtol=5e-3,
    )


# ---------------------------------------------------------------------------
# Test 2: Left sketch adjoint consistency
# ---------------------------------------------------------------------------

def test_left_sketch_adjoint_consistency():
    """Left sketch = S_L @ per_image_backproject(CTF_w * r).

    Reference: backproject each image separately, then S_L @ bp.
    """
    rng = np.random.default_rng(7)
    s = _setup(rng)
    c = s["config"]

    residual, _, left_cols = _call_batch(s, S_left_half=s["S_left_half"])

    CTF_w = jnp.ones_like(s["noise_half"])
    adjoint_input_half = CTF_w * residual
    adjoint_input_full = ftu.half_image_to_full_image(adjoint_input_half, c.image_shape)

    n_images = residual.shape[0]
    half_vol_size = s["half_vol_size"]
    bp = np.zeros((half_vol_size, n_images), dtype=np.float32)
    for i in range(n_images):
        g_i = core.adjoint_slice_volume(
            adjoint_input_full[i:i+1], s["rotations"][i:i+1],
            c.image_shape, c.volume_shape, c.disc_type,
            half_volume=True,
        )
        bp[:, i] = np.asarray(g_i).real

    ref = (np.asarray(s["S_left_half"]) @ bp).real

    np.testing.assert_allclose(
        np.asarray(left_cols), ref,
        atol=2.0, rtol=5e-3,
    )


# ---------------------------------------------------------------------------
# Test 3: Zero residual gives zero output
# ---------------------------------------------------------------------------

def test_zero_residual():
    """If images = forward(X) + projected_mean, residual and sketches ≈ 0."""
    rng = np.random.default_rng(99)
    s = _setup(rng, n_images=3, rank=2, sketch_rank=2, qrank=2)
    c = s["config"]
    from recovar.ppca.ppca import batch_over_vol_slice_volume_half

    # Construct images = CTF * P(mean) + CTF * P(U_X) @ diag(s) @ V^T
    # With CTF=1, noise=1: images = P(mean) + P(U_X) @ C^T
    C = s["V_X"] * s["sigma_X"][None, :]
    PU = batch_over_vol_slice_volume_half(
        s["U_X_half"], s["rotations"], c.image_shape, c.volume_shape, c.disc_type,
    )  # (batch, rank, half_image)
    predicted = jnp.einsum("bri,br->bi", PU, C)

    P_mean = core.slice_volume(
        s["mean_half"], s["rotations"], c.image_shape, c.volume_shape,
        c.disc_type, half_image=True, half_volume=True,
    )
    exact_images = predicted + P_mean

    residual, right, left = _sketched_normal_batch(
        exact_images, s["mean_half"], s["U_X_half"], s["sigma_X"],
        s["V_X"], s["ctf_params"], s["rotations"], s["translations"],
        s["noise_half"], c.voxel_size,
        c.image_shape, c.volume_shape, c.ctf, c.disc_type, c.disc_type,
        S_left_half=s["S_left_half"], Q_batch=s["Q_right"],
    )

    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(right), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(left), 0.0, atol=5e-4)


# ---------------------------------------------------------------------------
# Test 4: Linearity in residual
# ---------------------------------------------------------------------------

def test_sketch_linearity_in_data():
    """G(X) is affine in b (the data), so scaling data scales the sketch.

    With mean=0 and X=0, G(X) = A*(-b), so sketch ∝ data.
    """
    rng = np.random.default_rng(123)
    s = _setup(rng)
    c = s["config"]
    alpha = 2.5

    # X = 0: zero basis
    U_zero = jnp.zeros_like(s["U_X_half"])
    sigma_zero = jnp.zeros_like(s["sigma_X"])
    V_zero = jnp.zeros_like(s["V_X"])
    mean_zero = jnp.zeros_like(s["mean_half"])

    def run(images):
        _, right, left = _sketched_normal_batch(
            images, mean_zero, U_zero, sigma_zero, V_zero,
            s["ctf_params"], s["rotations"], s["translations"],
            s["noise_half"], c.voxel_size,
            c.image_shape, c.volume_shape, c.ctf, c.disc_type, c.disc_type,
            S_left_half=s["S_left_half"], Q_batch=s["Q_right"],
        )
        return right, left

    r1, l1 = run(s["images_half"])
    r_scaled, l_scaled = run(alpha * s["images_half"])

    np.testing.assert_allclose(
        np.asarray(r_scaled), alpha * np.asarray(r1), atol=0.1, rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(l_scaled), alpha * np.asarray(l1), atol=0.1, rtol=1e-2,
    )
