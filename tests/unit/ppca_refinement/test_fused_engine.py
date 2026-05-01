"""Phase A.1 (M10 follow-up) tests: fused dense engine.

The fused engine ``fused_dense_pose_ppca_block`` interleaves per-rotation
backprojection with the E-step pass-2 score normalization. We verify it
against the unfused ``dense_pose_ppca_E_step_blocked`` plus a hand-rolled
NumPy backprojection: the two paths must produce identical rhs/lhs_tri
half-volumes (within float32 + CUDA-kernel tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.em.dense_single_volume.helpers.backprojection import (  # noqa: E402
    batch_adjoint_slice_volume_half,
)
from recovar.em.ppca_refinement.dense_engine import (  # noqa: E402
    dense_pose_ppca_E_step_blocked,
    fused_dense_pose_ppca_block,
)
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = pytest.mark.unit


def _make_test_block(rng, *, B, T, R, F, q, image_shape):
    P = q + 1
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64) * 1e-2
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64) * 1e-2
    ctf2 = rng.uniform(0.1, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    # Rotations: random + identity to exercise the kernel path.
    rotations_block = np.zeros((R, 3, 3), dtype=np.float32)
    for r in range(R):
        # Use small rotations near identity to keep numerics stable.
        theta = 0.1 * rng.standard_normal(3)
        cx, cy, cz = np.cos(theta)
        sx, sy, sz = np.sin(theta)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rotations_block[r] = (Rz @ Ry @ Rx).astype(np.float32)
    return Y1, proj_aug, ctf2, y_norm, rotations_block


def _hand_backproject_from_unfused(
    Y1,
    proj_aug,
    ctf2,
    y_norm,
    rotations_block,
    image_shape,
    volume_shape,
    half_vol,
):
    """Reference: run the unfused engine to get image-level aggregates,
    then re-derive the per-(b, t, r) γ + α + G_tri arrays and apply
    per-rotation backprojection by hand (calling the same primitive
    the fused engine uses)."""
    from recovar.em.ppca_refinement.dense_engine import _per_pose_stats_block
    from recovar.ppca.pose_marginal import (
        compute_ppca_pose_scores_and_moments_no_contrast,
    )

    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    q = P - 1

    yn, tm, num, g, hz, Hz = _per_pose_stats_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )
    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=False,
    )
    score = np.asarray(score)
    score_flat = score.reshape(B, T * R)
    logZ = np.asarray(jax.scipy.special.logsumexp(jnp.asarray(score_flat), axis=-1))

    score2, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=True,
    )
    score2 = np.asarray(score2)
    alpha = np.asarray(alpha)
    G_tri = np.asarray(G_tri)
    gamma = np.exp(score2 - logZ[:, None, None])  # [B, T, R]

    rhs = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs_tri_vol = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    disc_type = "linear_interp"

    for r_idx in range(R):
        gamma_r = gamma[:, :, r_idx]  # [B, T]
        alpha_r = alpha[:, :, r_idx, :]  # [B, T, P]
        G_tri_r = G_tri[:, :, r_idx, :]  # [B, T, tri(P)]

        Z_rp = np.einsum("bt, btp, btf -> bpf", gamma_r, alpha_r, Y1)  # [B, P, F]
        Z_pbf = np.transpose(Z_rp, (1, 0, 2)).astype(np.complex64)
        rotation = rotations_block[r_idx]
        rotations_per_image = jnp.broadcast_to(
            jnp.asarray(rotation)[None, :, :],
            (B, 3, 3),
        )
        rhs = batch_adjoint_slice_volume_half(
            jnp.asarray(Z_pbf),
            rotations_per_image,
            rhs,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            half_volume=True,
        )

        w_rs = np.einsum("bt, btk -> bk", gamma_r, G_tri_r).real.astype(np.float32)
        weighted = w_rs[:, :, None] * ctf2[:, None, :]  # [B, tri(P), F]
        weighted_sbf = np.transpose(weighted, (1, 0, 2))  # [tri(P), B, F]
        lhs_tri_vol = batch_adjoint_slice_volume_half(
            jnp.asarray(weighted_sbf),
            rotations_per_image,
            lhs_tri_vol,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            half_volume=True,
        )
    return np.asarray(rhs), np.asarray(lhs_tri_vol)


def test_fused_engine_matches_hand_rolled_reference():
    """Fused engine produces same rhs / lhs_tri as the hand-rolled
    reference path (same primitives, just expressed differently)."""
    rng = np.random.default_rng(42)
    B, T, R, q = 2, 2, 3, 2
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    import recovar.core.fourier_transform_utils as ftu

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_test_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )

    # Reference path.
    rhs_ref, lhs_tri_ref = _hand_backproject_from_unfused(
        Y1,
        proj_aug,
        ctf2,
        y_norm,
        rotations_block,
        image_shape,
        volume_shape,
        half_vol,
    )

    # Fused path.
    P = q + 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_new, lhs_new, diag = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )

    np.testing.assert_allclose(np.asarray(rhs_new), rhs_ref, rtol=1e-3, atol=5e-3)
    np.testing.assert_allclose(np.asarray(lhs_new), lhs_tri_ref, rtol=1e-3, atol=5e-3)
    # Diagnostics shape sanity.
    assert diag.logZ.shape == (B,)
    assert diag.pmax.shape == (B,)


def test_fused_engine_diagnostics_match_unfused():
    """Pass-1 diagnostics (logZ, pmax, best pose) must match the unfused
    engine, since they're computed from the same M1 score function with
    no backprojection involved."""
    rng = np.random.default_rng(7)
    B, T, R, q = 3, 2, 4, 1
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    import recovar.core.fourier_transform_utils as ftu

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_test_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )

    _, unfused_diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )

    P = q + 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    _, _, fused_diag = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )

    np.testing.assert_allclose(
        np.asarray(fused_diag.logZ),
        np.asarray(unfused_diag.logZ),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_diag.pmax),
        np.asarray(unfused_diag.pmax),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(fused_diag.best_rotation_idx),
        np.asarray(unfused_diag.best_rotation_idx),
    )
    np.testing.assert_array_equal(
        np.asarray(fused_diag.best_translation_idx),
        np.asarray(unfused_diag.best_translation_idx),
    )
    # D12 fields must match too.
    np.testing.assert_allclose(
        np.asarray(fused_diag.best_log_score_per_image),
        np.asarray(unfused_diag.best_log_score_per_image),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_diag.rotation_posterior_sums),
        np.asarray(unfused_diag.rotation_posterior_sums),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fused_diag.max_posterior_per_image),
        np.asarray(unfused_diag.max_posterior_per_image),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fused_engine_q_zero_runs():
    """q=0 means only the mean component; α_aug ≡ [1] regardless of pose.
    The fused engine should still run cleanly and accumulate Y1 (γ-weighted)
    into rhs[0, :] and ctf2_over_noise into lhs_tri[0, :]."""
    rng = np.random.default_rng(11)
    B, T, R, q = 2, 2, 3, 0
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    import recovar.core.fourier_transform_utils as ftu

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_test_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )

    P = q + 1  # = 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_new, lhs_new, diag = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )
    assert rhs_new.shape == (P, half_vol)
    assert lhs_new.shape == (_tri_size(P), half_vol)
    assert np.all(np.isfinite(np.asarray(diag.logZ)))


def test_fused_engine_rejects_shape_mismatch():
    rng = np.random.default_rng(0)
    Y1, proj_aug, ctf2, y_norm, rotations = _make_test_block(
        rng,
        B=2,
        T=2,
        R=3,
        F=8 * 5,
        q=1,
        image_shape=(8, 8),
    )
    rhs0 = jnp.zeros((2, 100), dtype=jnp.complex64)
    lhs0 = jnp.zeros((3, 100), dtype=jnp.float32)
    with pytest.raises(ValueError, match="rotations_block"):
        fused_dense_pose_ppca_block(
            jnp.asarray(Y1),
            jnp.asarray(proj_aug),
            jnp.asarray(ctf2),
            jnp.asarray(y_norm),
            jnp.zeros((2, 3, 3), dtype=jnp.float32),  # wrong R
            (8, 8),
            (8, 8, 8),
            rhs0,
            lhs0,
        )


def test_fused_engine_with_pose_log_prior():
    rng = np.random.default_rng(13)
    B, T, R, q = 2, 2, 3, 1
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    import recovar.core.fourier_transform_utils as ftu

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_test_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )
    pose_log_prior = rng.standard_normal((B, R, T)).astype(np.float32)

    P = q + 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_a, lhs_a, _ = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )
    rhs_b, lhs_b, _ = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
        pose_log_prior=jnp.asarray(pose_log_prior),
    )
    # Different pose priors → different accumulators.
    assert not np.allclose(np.asarray(rhs_a), np.asarray(rhs_b), rtol=1e-3, atol=1e-3)
