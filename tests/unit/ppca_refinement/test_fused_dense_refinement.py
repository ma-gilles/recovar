import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.em.dense_single_volume.helpers.adjoint import batch_adjoint_slice_volume_half
from recovar.em.ppca_refinement.engine import (
    DensePPCAFusedBlock,
    _score_gamma_and_moments,
    fused_dense_pose_ppca_block,
    run_dense_ppca_fused_refinement_blocks,
)
from recovar.ppca.triangular import _tri_size


pytestmark = pytest.mark.unit


def _tiny_block(q=1):
    image_shape = (4, 4)
    volume_shape = (4, 4, 4)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    B, T, R, P = 2, 2, 2, q + 1
    base = np.arange(B * T * F, dtype=np.float32).reshape(B, T, F) / 17.0
    Y1 = (base + 1j * (base[..., ::-1] / 11.0)).astype(np.complex64)
    proj_base = np.arange(R * P * F, dtype=np.float32).reshape(R, P, F) / 19.0
    proj_aug = (proj_base - 0.2 + 1j * (proj_base / 13.0)).astype(np.complex64)
    if q:
        proj_aug[:, 1:, :] *= 0.1
    ctf2_over_noise = np.array(
        [[1.0, 0.5, 0.25, 1.5, 0.75, 1.25, 1.0, 0.8, 0.6, 1.2, 1.4, 0.9],
         [0.7, 1.1, 1.3, 0.4, 0.9, 1.6, 0.5, 1.0, 1.2, 0.8, 0.6, 1.5]],
        dtype=np.float32,
    )
    y_norm = np.array([2.0, 1.5], dtype=np.float32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (R, 3, 3)).copy()
    return image_shape, volume_shape, Y1, proj_aug, ctf2_over_noise, y_norm, rotations


def test_fused_dense_block_matches_block_presummed_backprojection():
    image_shape, volume_shape, Y1, proj_aug, ctf2_over_noise, y_norm, rotations = _tiny_block(q=1)
    P = proj_aug.shape[1]
    tri = _tri_size(P)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((tri, half_vol), dtype=jnp.float32)

    rhs_fused, lhs_fused, diag_fused = fused_dense_pose_ppca_block(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        rotations,
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )

    gamma, alpha, G_tri, diag = _score_gamma_and_moments(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        None,
        1e-3,
    )
    rhs_images = jnp.einsum(
        "btr,btrp,btf->prf",
        gamma.astype(jnp.complex64),
        jnp.conj(alpha).astype(jnp.complex64),
        jnp.asarray(Y1).astype(jnp.complex64),
    ).astype(jnp.complex64)
    rhs_manual = batch_adjoint_slice_volume_half(
        rhs_images,
        jnp.asarray(rotations),
        rhs0,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
        True,
    )

    lhs_images = jnp.einsum(
        "btr,btrk,bf->krf",
        gamma.astype(jnp.float32),
        G_tri,
        jnp.asarray(ctf2_over_noise).astype(jnp.float32),
    ).real.astype(jnp.float32)
    lhs_manual = batch_adjoint_slice_volume_half(
        lhs_images,
        jnp.asarray(rotations),
        lhs0,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
        True,
    )

    np.testing.assert_allclose(np.asarray(rhs_fused), np.asarray(rhs_manual), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(lhs_fused), np.asarray(lhs_manual), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(diag_fused.logZ), np.asarray(diag.logZ), rtol=1e-6, atol=1e-6)


def test_fused_dense_block_matches_slow_pose_loop_with_reconstruction_inputs():
    image_shape, volume_shape, Y1, proj_aug, ctf2_over_noise, y_norm, rotations = _tiny_block(q=2)
    P = proj_aug.shape[1]
    tri = _tri_size(P)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((tri, half_vol), dtype=jnp.float32)
    Y1_recon = (Y1 * np.asarray(1.25 - 0.1j, dtype=np.complex64)).astype(np.complex64)
    ctf2_recon = (ctf2_over_noise * np.asarray([1.1, 0.7], dtype=np.float32)[:, None]).astype(np.float32)

    rhs_fused, lhs_fused, _diag_fused = fused_dense_pose_ppca_block(
        Y1,
        proj_aug,
        ctf2_over_noise,
        y_norm,
        rotations,
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
        Y1_recon=Y1_recon,
        ctf2_over_noise_recon=ctf2_recon,
    )

    gamma, alpha, G_tri, _diag = _score_gamma_and_moments(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        None,
        1e-3,
    )
    rhs_images = jnp.zeros((P, rotations.shape[0], Y1.shape[-1]), dtype=jnp.complex64)
    lhs_images = jnp.zeros((tri, rotations.shape[0], Y1.shape[-1]), dtype=jnp.float32)
    for r in range(rotations.shape[0]):
        for p in range(P):
            rhs_img = jnp.sum(
                gamma[:, :, r].astype(jnp.complex64)[:, :, None]
                * jnp.conj(alpha[:, :, r, p]).astype(jnp.complex64)[:, :, None]
                * jnp.asarray(Y1_recon),
                axis=(0, 1),
            )
            rhs_images = rhs_images.at[p, r].set(rhs_img)
        for k in range(tri):
            lhs_img = jnp.sum(
                gamma[:, :, r].astype(jnp.float32)[:, :, None]
                * G_tri[:, :, r, k].real.astype(jnp.float32)[:, :, None]
                * jnp.asarray(ctf2_recon)[:, None, :],
                axis=(0, 1),
            )
            lhs_images = lhs_images.at[k, r].set(lhs_img)

    rhs_manual = batch_adjoint_slice_volume_half(
        rhs_images,
        jnp.asarray(rotations),
        rhs0,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
        True,
    )
    lhs_manual = batch_adjoint_slice_volume_half(
        lhs_images,
        jnp.asarray(rotations),
        lhs0,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
        True,
    )

    np.testing.assert_allclose(np.asarray(rhs_fused), np.asarray(rhs_manual), rtol=3e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(lhs_fused), np.asarray(lhs_manual), rtol=3e-3, atol=2e-3)


def test_run_dense_ppca_fused_refinement_blocks_returns_augmented_update():
    image_shape, volume_shape, Y1, proj_aug, ctf2_over_noise, y_norm, rotations = _tiny_block(q=1)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    block = DensePPCAFusedBlock(
        Y1=jnp.asarray(Y1),
        proj_aug=jnp.asarray(proj_aug),
        ctf2_over_noise=jnp.asarray(ctf2_over_noise),
        y_norm=jnp.asarray(y_norm),
        rotations=jnp.asarray(rotations),
    )
    result = run_dense_ppca_fused_refinement_blocks(
        [block],
        q=1,
        image_shape=image_shape,
        volume_shape=volume_shape,
        mean_prior=jnp.ones((half_vol,), dtype=jnp.float32) * 10.0,
        W_prior=jnp.ones((half_vol, 1), dtype=jnp.float32) * 5.0,
        enforce_x0=False,
    )

    assert result.stats.rhs.shape == (half_vol, 2)
    assert result.stats.lhs_tri.shape == (half_vol, 3)
    assert result.mu_half.shape == (half_vol,)
    assert result.W_half.shape == (half_vol, 1)
    assert result.stats.n_images == Y1.shape[0]
    assert np.isfinite(result.stats.log_likelihood)
    assert np.isfinite(result.diagnostics["pmax_mean"])


def test_run_dense_ppca_fused_refinement_blocks_q_zero_shape():
    image_shape, volume_shape, Y1, proj_aug, ctf2_over_noise, y_norm, rotations = _tiny_block(q=0)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    block = DensePPCAFusedBlock(
        Y1=jnp.asarray(Y1),
        proj_aug=jnp.asarray(proj_aug),
        ctf2_over_noise=jnp.asarray(ctf2_over_noise),
        y_norm=jnp.asarray(y_norm),
        rotations=jnp.asarray(rotations),
    )
    result = run_dense_ppca_fused_refinement_blocks(
        [block],
        q=0,
        image_shape=image_shape,
        volume_shape=volume_shape,
        mean_prior=jnp.ones((half_vol,), dtype=jnp.float32),
        W_prior=jnp.zeros((half_vol, 0), dtype=jnp.float32),
        enforce_x0=False,
    )

    assert result.stats.rhs.shape == (half_vol, 1)
    assert result.stats.lhs_tri.shape == (half_vol, 1)
    assert result.mu_half.shape == (half_vol,)
    assert result.W_half.shape == (half_vol, 0)
