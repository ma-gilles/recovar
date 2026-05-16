"""Strategy `w_only_mask` should mask W but leave mu untouched."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_refinement.postprocess import (
    PostprocessConfig,
    postprocess_ppca_half_volumes,
)


def _half_size(volume_shape):
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _random_mu_W(volume_shape, q, *, seed):
    rng = np.random.default_rng(seed)
    half = _half_size(volume_shape)
    mu = (rng.standard_normal(half) + 1j * rng.standard_normal(half)).astype(np.complex64)
    W = (rng.standard_normal((half, q)) + 1j * rng.standard_normal((half, q))).astype(np.complex64)
    return jnp.asarray(mu), jnp.asarray(W)


def test_w_only_mask_leaves_mu_untouched_and_masks_W():
    volume_shape = (16, 16, 16)
    q = 3
    mu, W = _random_mu_W(volume_shape, q, seed=11)
    mask = np.zeros(volume_shape, dtype=np.float32)
    # central cubic region
    mask[4:12, 4:12, 4:12] = 1.0

    result = postprocess_ppca_half_volumes(
        mu,
        W,
        volume_shape,
        config=PostprocessConfig(
            strategy="w_only_mask",
            external_mask_volume=mask,
            grid_correct=False,
        ),
    )
    # mu must pass through identically — strategy 'w_only_mask' should never
    # touch mu, mask or no mask.
    np.testing.assert_array_equal(np.asarray(result.mu_half), np.asarray(mu))

    # W must be zero outside the mask (in real space). Reconstruct one PC's
    # real-space volume to verify.
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    W_real_stack = []
    for k in range(q):
        full_F = np.asarray(
            ftu.half_volume_to_full_volume(jnp.asarray(result.W_half[:, k]), volume_shape)
        )
        real = np.asarray(ftu.get_idft3(full_F.reshape(volume_shape)).real)
        W_real_stack.append(real)
    W_real = np.stack(W_real_stack, axis=0)
    outside = (mask == 0.0)
    assert np.allclose(W_real[:, outside], 0.0, atol=1e-4), (
        "w_only_mask should zero W outside the external mask"
    )


def test_w_only_mask_with_zero_q_returns_inputs_unchanged():
    volume_shape = (8, 8, 8)
    mu = jnp.asarray(np.arange(_half_size(volume_shape), dtype=np.complex64))
    W = jnp.zeros((_half_size(volume_shape), 0), dtype=jnp.complex64)
    mask = np.ones(volume_shape, dtype=np.float32)

    result = postprocess_ppca_half_volumes(
        mu,
        W,
        volume_shape,
        config=PostprocessConfig(
            strategy="w_only_mask",
            external_mask_volume=mask,
            grid_correct=False,
        ),
    )
    np.testing.assert_array_equal(np.asarray(result.mu_half), np.asarray(mu))
    assert result.W_half.shape == W.shape


def test_mean_and_w_mask_still_masks_both():
    volume_shape = (16, 16, 16)
    q = 2
    mu, W = _random_mu_W(volume_shape, q, seed=23)
    mask = np.zeros(volume_shape, dtype=np.float32)
    mask[4:12, 4:12, 4:12] = 1.0

    result = postprocess_ppca_half_volumes(
        mu,
        W,
        volume_shape,
        config=PostprocessConfig(
            strategy="mean_and_w_mask",
            external_mask_volume=mask,
            grid_correct=False,
        ),
    )
    # mu has been background-filled — it's NOT equal to input mu.
    assert not np.allclose(np.asarray(result.mu_half), np.asarray(mu)), (
        "mean_and_w_mask should modify mu (regression vs w_only_mask)"
    )
