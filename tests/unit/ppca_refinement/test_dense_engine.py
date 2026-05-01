"""Phase 5 (M4) tests: dense pose-marginalized E-step.

The dense engine is tested in isolation against a brute-force enumeration
that calls the M1 per-pose function directly for every (b, r, t)
hypothesis and accumulates by hand. This verifies the engine produces
identical logZ / γ / α_aug / G_aug_tri to the analytic ground truth.

Tests run on CPU (small problems, no GPU needed) and are fast.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.em.ppca_refinement.dense_engine import (  # noqa: E402
    dense_pose_ppca_E_step_blocked,
)
from recovar.ppca.pose_marginal import (  # noqa: E402
    compute_ppca_pose_scores_and_moments_no_contrast,
)
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = pytest.mark.unit


def _random_problem(rng, B=2, T=3, R=4, F=8, q=2):
    P = q + 1
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64)
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64)
    ctf2_over_noise = rng.uniform(0.1, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    return Y1, proj_aug, ctf2_over_noise, y_norm


def _brute_force(Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior=None):
    """Per-(b, r, t) brute-force enumeration. Returns the same fields the
    blocked engine returns. Vectorizes over (b, r, t) but uses the M1
    function pose-by-pose so this is the analytic ground truth."""
    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    q = P - 1
    tri = _tri_size(P)

    # Build per-pose stats with the same conventions as the engine.
    K_aug = np.einsum("bf, rpf, rqf -> brpq", ctf2_over_noise.astype(np.complex64), np.conj(proj_aug), proj_aug)
    nu_mm = K_aug[..., 0, 0].real
    h_zm = K_aug[..., 1:, 0]
    Hzz = K_aug[..., 1:, 1:]
    D = np.einsum("btf, rpf -> btrp", np.conj(Y1), proj_aug)
    t_mx = D[..., 0].real
    g_zx = D[..., 1:]
    yn = np.broadcast_to(y_norm[:, None, None], (B, T, R)).astype(np.float32)
    num = np.broadcast_to(nu_mm[:, None, :], (B, T, R)).astype(np.float32)
    hz = np.broadcast_to(h_zm[:, None, :, :], (B, T, R, q)).astype(np.complex64)
    Hz = np.broadcast_to(Hzz[:, None, :, :, :], (B, T, R, q, q)).astype(np.complex64)

    score, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(yn),
        jnp.asarray(t_mx),
        jnp.asarray(num),
        jnp.asarray(g_zx),
        jnp.asarray(hz),
        jnp.asarray(Hz),
        return_moments=True,
    )
    score = np.asarray(score)
    alpha = np.asarray(alpha)
    G_tri = np.asarray(G_tri)
    if pose_log_prior is not None:
        # pose_log_prior has shape [B, R, T] → swap to [B, T, R].
        score = score + np.swapaxes(pose_log_prior, -1, -2)

    score_flat = score.reshape(B, T * R)
    logZ = jax.scipy.special.logsumexp(jnp.asarray(score_flat), axis=-1)
    logZ = np.asarray(logZ)
    gamma = np.exp(score - logZ[:, None, None])
    alpha_acc = np.einsum("btr, btrp -> bp", gamma.astype(alpha.dtype), alpha)
    G_acc = np.einsum("btr, btrk -> bk", gamma.astype(G_tri.dtype), G_tri)
    pmax = np.max(gamma.reshape(B, T * R), axis=-1)

    return logZ, pmax, alpha_acc, G_acc, gamma


def test_dense_engine_matches_brute_force_random_block():
    rng = np.random.default_rng(42)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=2, T=3, R=4, F=8, q=2)

    image_stats, diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )
    logZ_bf, pmax_bf, alpha_bf, G_bf, _ = _brute_force(Y1, proj_aug, ctf2, y_norm)

    np.testing.assert_allclose(np.asarray(diag.logZ), logZ_bf, rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(np.asarray(diag.pmax), pmax_bf, rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(image_stats.alpha_aug_acc),
        alpha_bf,
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(image_stats.G_aug_tri_acc),
        G_bf,
        rtol=2e-3,
        atol=5e-3,
    )


def test_dense_engine_matches_brute_force_with_pose_prior():
    """Adding a non-uniform pose log prior shifts the score additively.
    The engine must apply it correctly to both pass 1 (logZ) and pass 2
    (γ accumulation)."""
    rng = np.random.default_rng(7)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=3, T=2, R=5, F=8, q=1)
    B, T, F = Y1.shape
    R = proj_aug.shape[0]
    pose_log_prior = rng.standard_normal((B, R, T)).astype(np.float32)

    image_stats, diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        jnp.asarray(pose_log_prior),
    )
    logZ_bf, _, alpha_bf, G_bf, _ = _brute_force(
        Y1,
        proj_aug,
        ctf2,
        y_norm,
        pose_log_prior=pose_log_prior,
    )
    np.testing.assert_allclose(np.asarray(diag.logZ), logZ_bf, rtol=2e-3, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(image_stats.alpha_aug_acc),
        alpha_bf,
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(image_stats.G_aug_tri_acc),
        G_bf,
        rtol=2e-3,
        atol=5e-3,
    )


def test_dense_engine_q_zero_returns_trivial_moments():
    """With q=0 (only the mean component), augmented moments collapse to
    [1] and tri(1)=[1] regardless of pose; logZ still depends on the
    homogeneous ρ = ||x||² - 2 Re<x, m> + ||m||²."""
    rng = np.random.default_rng(11)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=2, T=2, R=3, F=6, q=0)
    image_stats, diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )
    np.testing.assert_array_equal(
        np.asarray(image_stats.alpha_aug_acc),
        np.ones((2, 1), dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        np.asarray(image_stats.G_aug_tri_acc),
        np.ones((2, 1), dtype=np.complex64),
    )
    # logZ should be finite and not NaN/Inf.
    assert np.all(np.isfinite(np.asarray(diag.logZ)))


def test_dense_engine_argmax_pose_matches_brute_force():
    rng = np.random.default_rng(2024)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=4, T=3, R=5, F=8, q=2)

    image_stats, diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )

    # Brute force argmax via M1.
    K_aug = np.einsum("bf, rpf, rqf -> brpq", ctf2.astype(np.complex64), np.conj(proj_aug), proj_aug)
    nu_mm = K_aug[..., 0, 0].real
    h_zm = K_aug[..., 1:, 0]
    Hzz = K_aug[..., 1:, 1:]
    D = np.einsum("btf, rpf -> btrp", np.conj(Y1), proj_aug)
    t_mx = D[..., 0].real
    g_zx = D[..., 1:]
    B, T = Y1.shape[:2]
    R = proj_aug.shape[0]
    q = proj_aug.shape[1] - 1
    yn = np.broadcast_to(y_norm[:, None, None], (B, T, R))
    num = np.broadcast_to(nu_mm[:, None, :], (B, T, R))
    hz = np.broadcast_to(h_zm[:, None, :, :], (B, T, R, q))
    Hz = np.broadcast_to(Hzz[:, None, :, :, :], (B, T, R, q, q))
    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        jnp.asarray(yn),
        jnp.asarray(t_mx),
        jnp.asarray(num),
        jnp.asarray(g_zx),
        jnp.asarray(hz),
        jnp.asarray(Hz),
        return_moments=False,
    )
    score = np.asarray(score)
    flat = score.reshape(B, T * R)
    bf_best_flat = np.argmax(flat, axis=-1)
    bf_best_t = bf_best_flat // R
    bf_best_r = bf_best_flat % R
    np.testing.assert_array_equal(np.asarray(diag.best_rotation_idx), bf_best_r)
    np.testing.assert_array_equal(np.asarray(diag.best_translation_idx), bf_best_t)


def test_dense_engine_jit_compiles():
    rng = np.random.default_rng(31)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=2, T=2, R=3, F=8, q=1)
    f_jit = jax.jit(
        lambda *a: dense_pose_ppca_E_step_blocked(*a)[0].alpha_aug_acc,
    )
    out1 = f_jit(jnp.asarray(Y1), jnp.asarray(proj_aug), jnp.asarray(ctf2), jnp.asarray(y_norm))
    out2 = f_jit(jnp.asarray(Y1), jnp.asarray(proj_aug), jnp.asarray(ctf2), jnp.asarray(y_norm))
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_dense_engine_rejects_shape_mismatch():
    rng = np.random.default_rng(99)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=2, T=2, R=3, F=8, q=1)
    bad_ctf2 = ctf2[:, :4]  # wrong F
    with pytest.raises(ValueError, match="ctf2_over_noise shape"):
        dense_pose_ppca_E_step_blocked(
            jnp.asarray(Y1),
            jnp.asarray(proj_aug),
            jnp.asarray(bad_ctf2),
            jnp.asarray(y_norm),
        )


def test_dense_engine_n_significant_uses_threshold():
    """``n_significant_per_image`` counts hypotheses with γ above
    ``significance_threshold``."""
    rng = np.random.default_rng(7)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=1, T=2, R=3, F=6, q=1)
    image_stats, diag_high = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        significance_threshold=0.9,
    )
    image_stats, diag_low = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
        significance_threshold=1e-9,
    )
    # Lower threshold ⇒ at least as many significant.
    assert int(diag_low.n_significant_per_image[0]) >= int(diag_high.n_significant_per_image[0])
    # Total hypotheses = 2 * 3 = 6.
    assert int(diag_low.n_significant_per_image[0]) <= 6


def test_dense_engine_omitted_log_mass_is_zero_for_dense():
    rng = np.random.default_rng(1)
    Y1, proj_aug, ctf2, y_norm = _random_problem(rng, B=3, T=2, R=2, F=4, q=1)
    _, diag = dense_pose_ppca_E_step_blocked(
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2),
        jnp.asarray(y_norm),
    )
    np.testing.assert_array_equal(np.asarray(diag.omitted_log_mass), np.zeros(3, dtype=np.float32))
