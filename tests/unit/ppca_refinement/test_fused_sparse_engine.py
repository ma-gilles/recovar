"""Phase A.3 (M10 follow-up) tests: fused sparse engine.

Critical gate: when the sparse layout enumerates every (b, r, t) tuple
from a dense block, the fused sparse engine must produce the same
rhs / lhs_tri half-volume accumulators as the fused dense engine
(within float32 + CUDA-kernel tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.dense_engine import (  # noqa: E402
    fused_dense_pose_ppca_block,
)
from recovar.em.ppca_refinement.sparse_engine import (  # noqa: E402
    SparseHypothesisLayout,
    fused_sparse_pose_ppca_block,
)
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = pytest.mark.unit


def _make_block(rng, *, B, T, R, F, q, image_shape):
    P = q + 1
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64) * 1e-2
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64) * 1e-2
    ctf2 = rng.uniform(0.1, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    rotations_block = np.zeros((R, 3, 3), dtype=np.float32)
    for r in range(R):
        theta = 0.1 * rng.standard_normal(3)
        cx, cy, cz = np.cos(theta)
        sx, sy, sz = np.sin(theta)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rotations_block[r] = (Rz @ Ry @ Rx).astype(np.float32)
    return Y1, proj_aug, ctf2, y_norm, rotations_block


def _flatten_to_sparse(Y1, proj_aug, ctf2, y_norm, rotations_block):
    """Convert dense (B, T, R, F) block into a flat layout enumerating
    every (b, r, t) hypothesis with its own rotation."""
    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    Nh = B * T * R

    Y1_flat = np.empty((Nh, F), dtype=Y1.dtype)
    proj_flat = np.empty((Nh, P, F), dtype=proj_aug.dtype)
    ctf2_flat = np.empty((Nh, F), dtype=ctf2.dtype)
    y_norm_flat = np.empty((Nh,), dtype=y_norm.dtype)
    image_id = np.empty((Nh,), dtype=np.int32)
    rotations_per_hyp = np.empty((Nh, 3, 3), dtype=np.float32)

    h = 0
    for b in range(B):
        for t in range(T):
            for r in range(R):
                Y1_flat[h] = Y1[b, t]
                proj_flat[h] = proj_aug[r]
                ctf2_flat[h] = ctf2[b]
                y_norm_flat[h] = y_norm[b]
                image_id[h] = b
                rotations_per_hyp[h] = rotations_block[r]
                h += 1

    layout = SparseHypothesisLayout(
        Y1=jnp.asarray(Y1_flat),
        proj_aug=jnp.asarray(proj_flat),
        ctf2_over_noise=jnp.asarray(ctf2_flat),
        y_norm=jnp.asarray(y_norm_flat),
        pose_log_prior=None,
        image_id=jnp.asarray(image_id),
        n_images=B,
    )
    return layout, jnp.asarray(rotations_per_hyp)


def test_fused_sparse_unpruned_equals_fused_dense():
    """Critical gate: unpruned-sparse with every (b,r,t) flat hypothesis
    produces identical rhs/lhs_tri to the fused dense engine."""
    rng = np.random.default_rng(42)
    B, T, R, q = 2, 2, 3, 2
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )

    P = q + 1
    rhs_dense = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs_dense = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_dense, lhs_dense, _ = fused_dense_pose_ppca_block(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        rhs_dense,
        lhs_dense,
    )

    layout, rotations_per_hyp = _flatten_to_sparse(
        Y1,
        proj_aug,
        ctf2,
        y_norm,
        rotations_block,
    )
    rhs_sparse = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs_sparse = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_sparse, lhs_sparse, diag = fused_sparse_pose_ppca_block(
        layout,
        rotations_per_hyp,
        image_shape,
        volume_shape,
        rhs_sparse,
        lhs_sparse,
    )

    np.testing.assert_allclose(
        np.asarray(rhs_sparse),
        np.asarray(rhs_dense),
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(lhs_sparse),
        np.asarray(lhs_dense),
        rtol=2e-3,
        atol=5e-3,
    )
    assert diag.logZ.shape == (B,)


def test_fused_sparse_q_zero_runs():
    rng = np.random.default_rng(11)
    B, T, R, q = 2, 2, 2, 0
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )
    layout, rotations_per_hyp = _flatten_to_sparse(
        Y1,
        proj_aug,
        ctf2,
        y_norm,
        rotations_block,
    )

    P = q + 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_new, lhs_new, diag = fused_sparse_pose_ppca_block(
        layout,
        rotations_per_hyp,
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )
    assert rhs_new.shape == (P, half_vol)
    assert lhs_new.shape == (_tri_size(P), half_vol)
    assert np.all(np.isfinite(np.asarray(diag.logZ)))


def test_fused_sparse_rejects_shape_mismatch():
    rng = np.random.default_rng(0)
    Y1, proj_aug, ctf2, y_norm, rotations = _make_block(
        rng,
        B=2,
        T=2,
        R=3,
        F=8 * 5,
        q=1,
        image_shape=(8, 8),
    )
    layout, _ = _flatten_to_sparse(Y1, proj_aug, ctf2, y_norm, rotations)
    rhs0 = jnp.zeros((2, 100), dtype=jnp.complex64)
    lhs0 = jnp.zeros((3, 100), dtype=jnp.float32)
    bad_rotations = jnp.zeros((5, 3, 3), dtype=jnp.float32)  # wrong Nh
    with pytest.raises(ValueError, match="rotations_per_hyp"):
        fused_sparse_pose_ppca_block(
            layout,
            bad_rotations,
            (8, 8),
            (8, 8, 8),
            rhs0,
            lhs0,
        )


def test_fused_sparse_with_pose_log_prior():
    rng = np.random.default_rng(7)
    B, T, R, q = 2, 2, 3, 1
    image_shape = (8, 8)
    F = image_shape[0] * (image_shape[1] // 2 + 1)
    volume_shape = (8, 8, 8)
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol = int(np.prod(half_vs))

    Y1, proj_aug, ctf2, y_norm, rotations_block = _make_block(
        rng,
        B=B,
        T=T,
        R=R,
        F=F,
        q=q,
        image_shape=image_shape,
    )
    layout, rotations_per_hyp = _flatten_to_sparse(
        Y1,
        proj_aug,
        ctf2,
        y_norm,
        rotations_block,
    )
    Nh = layout.image_id.shape[0]
    pose_log_prior = jnp.asarray(rng.standard_normal(Nh).astype(np.float32))
    layout_with_prior = SparseHypothesisLayout(
        Y1=layout.Y1,
        proj_aug=layout.proj_aug,
        ctf2_over_noise=layout.ctf2_over_noise,
        y_norm=layout.y_norm,
        pose_log_prior=pose_log_prior,
        image_id=layout.image_id,
        n_images=layout.n_images,
    )

    P = q + 1
    rhs0 = jnp.zeros((P, half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros((_tri_size(P), half_vol), dtype=jnp.float32)
    rhs_a, _, _ = fused_sparse_pose_ppca_block(
        layout,
        rotations_per_hyp,
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )
    rhs_b, _, _ = fused_sparse_pose_ppca_block(
        layout_with_prior,
        rotations_per_hyp,
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )
    assert not np.allclose(np.asarray(rhs_a), np.asarray(rhs_b), rtol=1e-3, atol=1e-3)
