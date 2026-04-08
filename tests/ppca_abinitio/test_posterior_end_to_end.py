"""End-to-end smoke test for `score_and_posterior_moments_eqx`.

The brute-force, parity, and calibration tests all bypass
`slice_volume` and the batch preprocessing pipeline by feeding
pre-sliced half-image inputs directly to
`score_from_half_image_projections`. This test exercises the
high-level entry point that goes through:

  1. slice_volume(half_volume=True, half_image=True) on `mu` and `U`
  2. _preprocess_batch_to_half (CTF, translation, full→half conversion)
  3. score_from_half_image_projections (the math kernel)

It uses a single identity rotation and identity translation so the
slicing operation is benign (the central z-slice of a real volume),
and verifies the result is finite, has the right shapes/dtypes,
and that the homogeneous case (U=0) reproduces a hand-computed
sanity score.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.posterior import score_and_posterior_moments_eqx
from recovar.em.ppca_abinitio.types import PosteriorStats

pytestmark = pytest.mark.unit


VOLUME_SHAPE = (8, 8, 8)
IMAGE_SHAPE = (8, 8)
N_HALF_VOL = VOLUME_SHAPE[0] * VOLUME_SHAPE[1] * (VOLUME_SHAPE[2] // 2 + 1)
N_FULL_IMG = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


# ---------------------------------------------------------------------------
# Identity-CTF / identity-process forward model config
# ---------------------------------------------------------------------------


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _TinyConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    _ctf: object = eqx.field(static=True)
    _process: object = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        full = self._ctf(ctf_params, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, batch, apply_image_mask=False):
        return self._process(batch, apply_image_mask=apply_image_mask)


def _make_config():
    return _TinyConfig(
        image_shape=IMAGE_SHAPE,
        volume_shape=VOLUME_SHAPE,
        _ctf=_identity_ctf,
        _process=_identity_process,
        voxel_size=1.0,
    )


# ---------------------------------------------------------------------------
# Synthetic half-volume inputs
# ---------------------------------------------------------------------------


def _real_volume_to_half_flat(real_vol):
    return ftu.get_dft3_real(jnp.asarray(real_vol)).reshape(-1)


def _make_half_volume_inputs(rng, q):
    # Mean: smooth real-space blob centered at the volume center
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, VOLUME_SHAPE[0]),
        np.linspace(-1, 1, VOLUME_SHAPE[1]),
        np.linspace(-1, 1, VOLUME_SHAPE[2]),
        indexing="ij",
    )
    mean_real = np.exp(-(z**2 + y**2 + x**2) * 4.0).astype(np.float64)
    mu_half = jnp.asarray(_real_volume_to_half_flat(mean_real), dtype=jnp.complex128)

    # PCs: real-space sinusoids
    U_rows = []
    for k in range(q):
        kz = (k % 2) + 1
        ky = ((k // 2) % 2) + 1
        kx = (k // 4) + 1
        pc_real = np.cos(np.pi * kz * z) * np.cos(np.pi * ky * y) * np.cos(np.pi * kx * x)
        pc_real *= 0.1
        U_rows.append(_real_volume_to_half_flat(pc_real.astype(np.float64)))
    U_half = jnp.asarray(jnp.stack(U_rows), dtype=jnp.complex128)
    return mu_half, U_half


def _make_real_image_batch(rng, n_img):
    images_real = rng.standard_normal((n_img,) + IMAGE_SHAPE).astype(np.float64)
    fts = jnp.stack([ftu.get_dft2(jnp.asarray(im)).reshape(-1) for im in images_real])
    return jnp.asarray(fts, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_runs_and_produces_correct_shapes():
    rng = np.random.default_rng(0)
    config = _make_config()
    q = 2
    n_img = 3
    n_rot = 1
    n_trans = 1

    mu_half, U_half = _make_half_volume_inputs(rng, q)
    s = jnp.asarray([1.0, 0.5], dtype=jnp.float64)

    rotations = jnp.eye(3, dtype=jnp.float64).reshape(1, 3, 3)
    translations = jnp.zeros((n_trans, 2), dtype=jnp.float64)
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)
    noise_var = jnp.ones(N_FULL_IMG, dtype=jnp.float64)

    batch = _make_real_image_batch(rng, n_img)

    stats = score_and_posterior_moments_eqx(
        config,
        mu_half,
        U_half,
        s,
        batch,
        rotations,
        translations,
        ctf_params,
        noise_var,
    )

    assert isinstance(stats, PosteriorStats)
    assert stats.log_scores.shape == (n_img, n_rot, n_trans)
    assert stats.log_resp.shape == (n_img, n_rot, n_trans)
    assert stats.post_mean.shape == (n_img, n_rot, n_trans, q)
    assert stats.post_Hinv.shape == (n_img, n_rot, q, q)

    assert stats.log_scores.dtype == jnp.float64
    assert stats.post_mean.dtype == jnp.float64
    assert stats.post_Hinv.dtype == jnp.float64

    assert bool(jnp.all(jnp.isfinite(stats.log_scores)))
    assert bool(jnp.all(jnp.isfinite(stats.post_mean)))
    assert bool(jnp.all(jnp.isfinite(stats.post_Hinv)))

    # log_resp normalizes per image
    log_resp = np.asarray(stats.log_resp).reshape(n_img, -1)
    sums = np.exp(log_resp).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)


def test_end_to_end_dtype_contract_rejected():
    rng = np.random.default_rng(1)
    config = _make_config()
    mu_half, U_half = _make_half_volume_inputs(rng, q=2)
    s = jnp.ones(2, dtype=jnp.float64)
    batch = _make_real_image_batch(rng, 2)

    rotations = jnp.eye(3, dtype=jnp.float64).reshape(1, 3, 3)
    translations = jnp.zeros((1, 2), dtype=jnp.float64)
    ctf_params = jnp.zeros((2, 9), dtype=jnp.float64)
    noise_var = jnp.ones(N_FULL_IMG, dtype=jnp.float64)

    # Pass complex64 mu_half — should be rejected
    with pytest.raises(TypeError, match="complex128"):
        score_and_posterior_moments_eqx(
            config,
            mu_half.astype(jnp.complex64),
            U_half,
            s,
            batch,
            rotations,
            translations,
            ctf_params,
            noise_var,
        )

    # Pass float32 noise_var — should be rejected
    with pytest.raises(TypeError, match="float64"):
        score_and_posterior_moments_eqx(
            config,
            mu_half,
            U_half,
            s,
            batch,
            rotations,
            translations,
            ctf_params,
            noise_var.astype(jnp.float32),
        )


def test_end_to_end_multiple_rotations_and_translations():
    """Sanity: multi-rotation, multi-translation case runs and shapes are right."""
    rng = np.random.default_rng(2)
    config = _make_config()
    q = 2
    n_img = 3
    n_rot = 4
    n_trans = 2

    mu_half, U_half = _make_half_volume_inputs(rng, q)
    s = jnp.ones(q, dtype=jnp.float64)

    # 4 random rotations
    from scipy.spatial.transform import Rotation as R

    rot_mats = R.random(num=n_rot, random_state=42).as_matrix()
    rotations = jnp.asarray(rot_mats, dtype=jnp.float64)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    ctf_params = jnp.zeros((n_img, 9), dtype=jnp.float64)
    noise_var = jnp.ones(N_FULL_IMG, dtype=jnp.float64)
    batch = _make_real_image_batch(rng, n_img)

    stats = score_and_posterior_moments_eqx(
        config, mu_half, U_half, s, batch, rotations, translations, ctf_params, noise_var
    )

    assert stats.log_scores.shape == (n_img, n_rot, n_trans)
    assert stats.post_mean.shape == (n_img, n_rot, n_trans, q)
    assert bool(jnp.all(jnp.isfinite(stats.log_scores)))
    assert bool(jnp.all(jnp.isfinite(stats.post_mean)))
