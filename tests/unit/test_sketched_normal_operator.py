"""Tests for recovar.ppca.sketched_normal — real-space API."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.ppca.ppca import batch_over_vol_slice_volume_half
from recovar.ppca.sketched_normal import (
    _half_fourier_to_real_vols,
    _real_vols_to_half_fourier,
    _sketched_normal_batch,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ones_ctf(ctf_params, image_shape, voxel_size, **kw):
    half_image = kw.get("half_image", False)
    shape = ftu.image_shape_to_half_image_shape(image_shape) if half_image else image_shape
    return np.ones((ctf_params.shape[0], int(np.prod(shape))), dtype=np.float32)


def _setup(rng, grid_size=8, n_images=4, rank=2, sketch_rank=3, qrank=2):
    vs = (grid_size,) * 3
    vol_size = grid_size**3
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    half_vol_size = int(np.prod(half_vs))
    image_shape = (grid_size, grid_size)
    half_im_size = int(np.prod(ftu.image_shape_to_half_image_shape(image_shape)))

    # Real-space volumes
    U_real = rng.normal(size=(vol_size, rank)).astype(np.float32)
    sigma = rng.uniform(0.5, 2.0, size=(rank,)).astype(np.float32)
    V = rng.normal(size=(n_images, rank)).astype(np.float32)
    mean_real = rng.normal(size=(vol_size,)).astype(np.float32)
    S_real = rng.normal(size=(sketch_rank, vol_size)).astype(np.float32)
    Q = rng.normal(size=(n_images, qrank)).astype(np.float32)

    # Convert to half-Fourier for internal kernel
    U_half = _real_vols_to_half_fourier(U_real.T, vs).T
    mean_half = _real_vols_to_half_fourier(mean_real.reshape(1, -1), vs)[0]
    S_half = _real_vols_to_half_fourier(S_real, vs)

    # Half-image data
    images_real = rng.normal(size=(n_images, *image_shape)).astype(np.float32)
    images_ft = np.fft.fft2(np.fft.fftshift(images_real, axes=(-2, -1)))
    images_half = np.asarray(
        ftu.full_image_to_half_image(images_ft.reshape(n_images, -1).astype(np.complex64), image_shape)
    )

    rotations = []
    mats = rng.normal(size=(n_images, 3, 3)).astype(np.float32)
    for i in range(n_images):
        q, r = np.linalg.qr(mats[i])
        q *= np.sign(np.linalg.det(q))
        rotations.append(q)
    rotations = np.stack(rotations)

    return dict(
        vs=vs,
        vol_size=vol_size,
        half_vol_size=half_vol_size,
        image_shape=image_shape,
        half_im_size=half_im_size,
        U_real=U_real,
        sigma=sigma,
        V=V,
        mean_real=mean_real,
        S_real=S_real,
        Q=Q,
        U_half=U_half,
        mean_half=mean_half,
        S_half=S_half,
        images_half=jnp.array(images_half),
        rotations=jnp.array(rotations),
        translations=jnp.zeros((n_images, 2), dtype=np.float32),
        ctf_params=jnp.zeros((n_images, 9), dtype=np.float32),
        noise_half=jnp.ones((n_images, half_im_size), dtype=np.float32),
        ctf=core.as_ctf_evaluator(_ones_ctf),
    )


def _call_kernel(s, S_half=None, Q=None):
    return _sketched_normal_batch(
        s["images_half"],
        s["mean_half"],
        jnp.array(s["U_half"]),
        jnp.array(s["sigma"]),
        s["V"],
        s["ctf_params"],
        s["rotations"],
        s["translations"],
        s["noise_half"],
        1.0,
        s["image_shape"],
        s["vs"],
        s["ctf"],
        "linear_interp",
        "linear_interp",
        S_left_half=jnp.array(S_half) if S_half is not None else None,
        Q_batch=jnp.array(Q) if Q is not None else None,
    )


# ---------------------------------------------------------------------------
# Test: real ↔ half-Fourier roundtrip
# ---------------------------------------------------------------------------


def test_real_half_fourier_roundtrip():
    rng = np.random.default_rng(0)
    vs = (8, 8, 8)
    vols = rng.normal(size=(3, 512)).astype(np.float32)
    half = _real_vols_to_half_fourier(vols, vs)
    back = _half_fourier_to_real_vols(half, vs)
    np.testing.assert_allclose(back, vols, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: right sketch vs per-image reference
# ---------------------------------------------------------------------------


def test_right_sketch_vs_per_column():
    rng = np.random.default_rng(42)
    s = _setup(rng)

    right, _ = _call_kernel(s, Q=s["Q"])

    # Reference: per-image backproject, then matmul
    _, left_ref = _call_kernel(s)  # just to get residual shape
    # Re-run to get bp directly — use a trivial Q=[1] per image
    n = s["V"].shape[0]
    ref = np.zeros((s["half_vol_size"], s["Q"].shape[1]), dtype=np.complex64)
    for j in range(s["Q"].shape[1]):
        Q_col = s["Q"][:, j : j + 1]
        r_j, _ = _call_kernel(s, Q=Q_col)
        ref[:, j] = np.asarray(r_j).reshape(-1)

    np.testing.assert_allclose(np.asarray(right), ref, atol=2.0, rtol=5e-3)


# ---------------------------------------------------------------------------
# Test: left sketch vs per-image reference
# ---------------------------------------------------------------------------


def test_left_sketch_vs_per_row():
    rng = np.random.default_rng(7)
    s = _setup(rng)

    _, left = _call_kernel(s, S_half=s["S_half"])

    # Reference: one row at a time
    ref = np.zeros_like(np.asarray(left))
    for si in range(s["S_half"].shape[0]):
        _, left_row = _call_kernel(s, S_half=s["S_half"][si : si + 1])
        ref[si] = np.asarray(left_row).reshape(-1)

    np.testing.assert_allclose(np.asarray(left), ref, atol=2.0, rtol=5e-3)


# ---------------------------------------------------------------------------
# Test: zero residual → zero output
# ---------------------------------------------------------------------------


def test_zero_residual():
    rng = np.random.default_rng(99)
    s = _setup(rng, n_images=3, rank=2, sketch_rank=2, qrank=2)

    # Construct images = forward(X) + forward(mean) so residual = 0
    U_half = jnp.array(s["U_half"])
    mean_half = jnp.array(s["mean_half"])
    C = s["V"] * s["sigma"][None, :]

    PU = batch_over_vol_slice_volume_half(
        U_half,
        s["rotations"],
        s["image_shape"],
        s["vs"],
        "linear_interp",
    )
    predicted = jnp.einsum("bri,br->bi", PU, jnp.array(C))
    P_mean = core.slice_volume(
        mean_half,
        s["rotations"],
        s["image_shape"],
        s["vs"],
        "linear_interp",
        half_image=True,
        half_volume=True,
    )
    exact_images = predicted + P_mean

    right, left = _sketched_normal_batch(
        exact_images,
        mean_half,
        U_half,
        jnp.array(s["sigma"]),
        jnp.array(s["V"]),
        s["ctf_params"],
        s["rotations"],
        s["translations"],
        s["noise_half"],
        1.0,
        s["image_shape"],
        s["vs"],
        s["ctf"],
        "linear_interp",
        "linear_interp",
        S_left_half=jnp.array(s["S_half"]),
        Q_batch=jnp.array(s["Q"]),
    )

    np.testing.assert_allclose(np.asarray(right), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(left), 0.0, atol=5e-4)


# ---------------------------------------------------------------------------
# Test: linearity (scale data → scale output)
# ---------------------------------------------------------------------------


def test_linearity():
    rng = np.random.default_rng(123)
    s = _setup(rng)
    alpha = 2.5

    # X=0, mean=0 → G(X) = -A*(b), so scaling images scales output
    U_zero = jnp.zeros_like(jnp.array(s["U_half"]))
    s_zero = jnp.zeros_like(jnp.array(s["sigma"]))
    V_zero = jnp.zeros_like(jnp.array(s["V"]))
    mean_zero = jnp.zeros_like(jnp.array(s["mean_half"]))

    def run(imgs):
        return _sketched_normal_batch(
            imgs,
            mean_zero,
            U_zero,
            s_zero,
            V_zero,
            s["ctf_params"],
            s["rotations"],
            s["translations"],
            s["noise_half"],
            1.0,
            s["image_shape"],
            s["vs"],
            s["ctf"],
            "linear_interp",
            "linear_interp",
            S_left_half=jnp.array(s["S_half"]),
            Q_batch=jnp.array(s["Q"]),
        )

    r1, l1 = run(s["images_half"])
    r_s, l_s = run(alpha * s["images_half"])

    np.testing.assert_allclose(np.asarray(r_s), alpha * np.asarray(r1), atol=0.1, rtol=1e-2)
    np.testing.assert_allclose(np.asarray(l_s), alpha * np.asarray(l1), atol=0.1, rtol=1e-2)
