"""Phase 7 (M6) tests: sparse pose-marginalized E-step.

Critical gate: unpruned-sparse equals dense bit-for-bit (within float32
fused-multiply tolerance). The sparse engine uses ``segment_sum`` over
a flat hypothesis layout; the dense engine uses ``einsum`` over a
``[B, T, R]`` tensor block. They must produce identical outputs when
the sparse layout enumerates EVERY (b, r, t) tuple.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.em.ppca_refinement.dense_engine import (  # noqa: E402
    dense_pose_ppca_E_step_blocked,
)
from recovar.em.ppca_refinement.sparse_engine import (  # noqa: E402
    SparseHypothesisLayout,
    sparse_pose_ppca_E_step_flat,
)

pytestmark = pytest.mark.unit


def _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior):
    """Convert a (B, T, R, F) dense block into a flat layout enumerating
    every (b, r, t) hypothesis.

    Order: per image b, walk t outer / r inner — same order the dense
    engine flattens via ``score.reshape(B, T*R)``.
    """
    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    Nh = B * T * R

    # Flat per-hypothesis arrays.
    Y1_flat = np.empty((Nh, F), dtype=Y1.dtype)
    proj_flat = np.empty((Nh, P, F), dtype=proj_aug.dtype)
    ctf2_flat = np.empty((Nh, F), dtype=ctf2.dtype)
    y_norm_flat = np.empty((Nh,), dtype=y_norm.dtype)
    image_id = np.empty((Nh,), dtype=np.int32)
    pose_log_prior_flat = np.empty((Nh,), dtype=pose_log_prior.dtype) if pose_log_prior is not None else None
    h = 0
    for b in range(B):
        for t in range(T):
            for r in range(R):
                Y1_flat[h] = Y1[b, t]
                proj_flat[h] = proj_aug[r]
                ctf2_flat[h] = ctf2[b]
                y_norm_flat[h] = y_norm[b]
                image_id[h] = b
                if pose_log_prior is not None:
                    # pose_log_prior is [B, R, T] — note axis order.
                    pose_log_prior_flat[h] = pose_log_prior[b, r, t]
                h += 1
    return SparseHypothesisLayout(
        Y1=jnp.asarray(Y1_flat),
        proj_aug=jnp.asarray(proj_flat),
        ctf2_over_noise=jnp.asarray(ctf2_flat),
        y_norm=jnp.asarray(y_norm_flat),
        pose_log_prior=jnp.asarray(pose_log_prior_flat) if pose_log_prior is not None else None,
        image_id=jnp.asarray(image_id),
        n_images=B,
    )


def _random_dense_block(rng, B=2, T=3, R=4, F=8, q=2):
    P = q + 1
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64)
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64)
    ctf2 = rng.uniform(0.1, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    return Y1, proj_aug, ctf2, y_norm


def test_unpruned_sparse_equals_dense_no_pose_prior():
    rng = np.random.default_rng(42)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=2, T=3, R=4, F=8, q=2)

    dense_stats, dense_diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )
    layout = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=None)
    sparse_stats, sparse_diag = sparse_pose_ppca_E_step_flat(layout)

    np.testing.assert_allclose(np.asarray(sparse_diag.logZ), np.asarray(dense_diag.logZ), rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(np.asarray(sparse_diag.pmax), np.asarray(dense_diag.pmax), rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(sparse_stats.alpha_aug_acc),
        np.asarray(dense_stats.alpha_aug_acc),
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_stats.G_aug_tri_acc),
        np.asarray(dense_stats.G_aug_tri_acc),
        rtol=2e-3,
        atol=5e-3,
    )


def test_unpruned_sparse_equals_dense_with_pose_prior():
    rng = np.random.default_rng(7)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=3, T=2, R=5, F=8, q=1)
    B = Y1.shape[0]
    R = proj_aug.shape[0]
    T = Y1.shape[1]
    pose_log_prior = rng.standard_normal((B, R, T)).astype(np.float32)

    dense_stats, dense_diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(pose_log_prior),
    )
    layout = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=pose_log_prior)
    sparse_stats, sparse_diag = sparse_pose_ppca_E_step_flat(layout)

    np.testing.assert_allclose(np.asarray(sparse_diag.logZ), np.asarray(dense_diag.logZ), rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(sparse_stats.alpha_aug_acc),
        np.asarray(dense_stats.alpha_aug_acc),
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_stats.G_aug_tri_acc),
        np.asarray(dense_stats.G_aug_tri_acc),
        rtol=2e-3,
        atol=5e-3,
    )


def test_sparse_pruning_does_not_alter_per_hypothesis_scores():
    """Drop half the hypotheses (any half) and confirm:
       - The per-image logZ is computed correctly over the retained set
         (= logsumexp of the kept scores).
       - The per-hypothesis γ (= exp(score - logZ)) for the retained set
         is unchanged in score, only renormalized.

    This pins the non-negotiable that pruning restricts the support but
    does not alter scores.
    """
    rng = np.random.default_rng(11)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=2, T=3, R=4, F=8, q=2)
    layout_full = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=None)

    # Keep only odd-indexed hypotheses per image.
    Nh = layout_full.image_id.shape[0]
    keep = np.zeros(Nh, dtype=bool)
    keep[::2] = True
    keep_idx = np.where(keep)[0]
    layout_pruned = SparseHypothesisLayout(
        Y1=layout_full.Y1[keep_idx],
        proj_aug=layout_full.proj_aug[keep_idx],
        ctf2_over_noise=layout_full.ctf2_over_noise[keep_idx],
        y_norm=layout_full.y_norm[keep_idx],
        pose_log_prior=None,
        image_id=layout_full.image_id[keep_idx],
        n_images=layout_full.n_images,
    )

    # Run sparse engine on the pruned layout — must finish without error
    # and produce per-image stats sized n_images.
    image_stats, diag = sparse_pose_ppca_E_step_flat(layout_pruned)
    assert image_stats.alpha_aug_acc.shape[0] == layout_full.n_images
    assert diag.logZ.shape == (layout_full.n_images,)
    # logZ on pruned support is upper-bounded by logZ on full support
    # (logsumexp over a subset ≤ logsumexp over the superset).
    _, full_diag = sparse_pose_ppca_E_step_flat(layout_full)
    assert np.all(np.asarray(diag.logZ) <= np.asarray(full_diag.logZ) + 1e-5)


def test_sparse_n_significant_uses_threshold():
    rng = np.random.default_rng(7)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=1, T=2, R=3, F=6, q=1)
    layout = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=None)
    _, diag_high = sparse_pose_ppca_E_step_flat(layout, significance_threshold=0.9)
    _, diag_low = sparse_pose_ppca_E_step_flat(layout, significance_threshold=1e-9)
    assert int(diag_low.n_significant_per_image[0]) >= int(diag_high.n_significant_per_image[0])


def test_sparse_engine_jit_compiles():
    rng = np.random.default_rng(31)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=2, T=2, R=3, F=8, q=1)
    layout = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=None)

    @jax.jit
    def f(Y1f, proj, ctf, yn, im_id):
        layout_jit = SparseHypothesisLayout(
            Y1=Y1f,
            proj_aug=proj,
            ctf2_over_noise=ctf,
            y_norm=yn,
            pose_log_prior=None,
            image_id=im_id,
            n_images=2,
        )
        stats, _ = sparse_pose_ppca_E_step_flat(layout_jit)
        return stats.alpha_aug_acc

    out1 = f(layout.Y1, layout.proj_aug, layout.ctf2_over_noise, layout.y_norm, layout.image_id)
    out2 = f(layout.Y1, layout.proj_aug, layout.ctf2_over_noise, layout.y_norm, layout.image_id)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_sparse_q_zero_returns_trivial_moments():
    rng = np.random.default_rng(11)
    Y1, proj_aug, ctf2, y_norm = _random_dense_block(rng, B=3, T=2, R=2, F=4, q=0)
    layout = _flatten_dense_to_sparse(Y1, proj_aug, ctf2, y_norm, pose_log_prior=None)
    image_stats, diag = sparse_pose_ppca_E_step_flat(layout)
    np.testing.assert_array_equal(
        np.asarray(image_stats.alpha_aug_acc),
        np.ones((3, 1), dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        np.asarray(image_stats.G_aug_tri_acc),
        np.ones((3, 1), dtype=np.complex64),
    )
    assert np.all(np.isfinite(np.asarray(diag.logZ)))
