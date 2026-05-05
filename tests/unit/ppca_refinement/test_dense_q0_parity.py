import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement.dense_engine import dense_pose_ppca_E_step_blocked


pytestmark = pytest.mark.unit


def _fixture(q):
    Y1 = jnp.asarray(
        [
            [[1.0 + 0.1j, 0.2 - 0.3j, -0.4 + 0.5j], [0.7 - 0.2j, -0.1 + 0.4j, 0.3 + 0.2j]],
            [[-0.2 + 0.6j, 0.5 + 0.1j, 0.1 - 0.7j], [0.4 + 0.3j, -0.6 + 0.2j, 0.2 - 0.1j]],
        ],
        dtype=jnp.complex64,
    )
    mean_proj = jnp.asarray(
        [[0.5 - 0.1j, -0.2 + 0.3j, 0.1 + 0.4j], [-0.3 + 0.2j, 0.6 + 0.1j, -0.2 - 0.5j]],
        dtype=jnp.complex64,
    )
    if q:
        zeros = jnp.zeros((2, q, 3), dtype=jnp.complex64)
        proj_aug = jnp.concatenate([mean_proj[:, None, :], zeros], axis=1)
    else:
        proj_aug = mean_proj[:, None, :]
    ctf2_over_noise = jnp.asarray([[1.0, 0.5, 2.0], [0.75, 1.25, 0.5]], dtype=jnp.float32)
    y_norm = jnp.asarray([2.5, 1.75], dtype=jnp.float32)
    return Y1, proj_aug, ctf2_over_noise, y_norm


def _manual_homogeneous_logz(Y1, mean_proj, ctf2_over_noise, y_norm):
    K = jnp.einsum("bf,rf,rf->br", ctf2_over_noise.astype(mean_proj.dtype), jnp.conj(mean_proj), mean_proj).real
    D = jnp.einsum("btf,rf->btr", jnp.conj(Y1), mean_proj).real
    score = -0.5 * (y_norm[:, None, None] - 2.0 * D + K[:, None, :])
    flat = np.asarray(score).reshape(score.shape[0], -1)
    max_score = np.max(flat, axis=1, keepdims=True)
    return np.squeeze(max_score, axis=1) + np.log(np.sum(np.exp(flat - max_score), axis=1))


def test_dense_q0_matches_homogeneous_score_expression():
    Y1, proj_aug, ctf2_over_noise, y_norm = _fixture(q=0)
    stats, diagnostics = dense_pose_ppca_E_step_blocked(Y1, proj_aug, ctf2_over_noise, y_norm)

    expected_logz = _manual_homogeneous_logz(Y1, proj_aug[:, 0, :], ctf2_over_noise, y_norm)
    np.testing.assert_allclose(np.asarray(diagnostics.logZ), expected_logz, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.alpha_aug_acc), np.ones((2, 1)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(stats.G_aug_tri_acc), np.ones((2, 1)), rtol=1e-6, atol=1e-6)


def test_dense_w_zero_matches_q0_log_evidence_and_best_pose():
    q0 = _fixture(q=0)
    w0 = _fixture(q=2)
    _stats_q0, diag_q0 = dense_pose_ppca_E_step_blocked(*q0)
    _stats_w0, diag_w0 = dense_pose_ppca_E_step_blocked(*w0)

    np.testing.assert_allclose(np.asarray(diag_w0.logZ), np.asarray(diag_q0.logZ), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(diag_w0.best_rotation_idx), np.asarray(diag_q0.best_rotation_idx))
    np.testing.assert_array_equal(np.asarray(diag_w0.best_translation_idx), np.asarray(diag_q0.best_translation_idx))
