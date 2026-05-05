import numpy as np
import pytest

from recovar.ppca import AugmentedPPCAStats, pack_upper_tri, solve_augmented_ppca_mstep


pytestmark = pytest.mark.unit


def test_augmented_mstep_solves_joint_cross_terms():
    lhs = np.array(
        [
            [[2.0, 0.25], [0.25, 4.0]],
            [[3.0, -0.5], [-0.5, 5.0]],
        ],
        dtype=np.float32,
    )
    mean_prior = np.array([10.0, 20.0], dtype=np.float32)
    W_prior = np.array([[5.0], [8.0]], dtype=np.float32)
    reg = np.stack([1.0 / mean_prior, 1.0 / W_prior[:, 0]], axis=1)
    theta_true = np.array([[1.5, -0.75], [-2.0, 0.5]], dtype=np.complex64)
    rhs = np.einsum("fij,fj->fi", lhs + np.eye(2, dtype=np.float32)[None] * reg[:, None, :], theta_true)

    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=pack_upper_tri(lhs))
    mu, W = solve_augmented_ppca_mstep(stats, mean_prior=mean_prior, W_prior=W_prior)

    np.testing.assert_allclose(np.asarray(mu), theta_true[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(W[:, 0]), theta_true[:, 1], rtol=1e-6, atol=1e-6)


def test_augmented_mstep_q_zero_reduces_to_homogeneous_scalar_solve():
    lhs = np.array([[[2.0]], [[4.0]], [[8.0]]], dtype=np.float32)
    mean_prior = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    theta_true = np.array([3.0, -1.0, 0.25], dtype=np.complex64)
    rhs = (lhs[:, 0, 0] + 1.0 / mean_prior) * theta_true

    stats = AugmentedPPCAStats(rhs=rhs[:, None], lhs_tri=pack_upper_tri(lhs))
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=np.zeros((3, 0), dtype=np.float32),
    )

    np.testing.assert_allclose(np.asarray(mu), theta_true, rtol=1e-6, atol=1e-6)
    assert np.asarray(W).shape == (3, 0)


def test_augmented_mstep_chunked_matches_full_solve():
    lhs = np.array(
        [
            [[2.0, 0.1], [0.1, 1.5]],
            [[3.0, -0.2], [-0.2, 2.5]],
            [[1.5, 0.3], [0.3, 1.2]],
            [[2.2, 0.0], [0.0, 1.8]],
        ],
        dtype=np.float32,
    )
    rhs = np.array(
        [
            [1.0 + 0.0j, 0.2 - 0.1j],
            [0.5 + 0.3j, -0.2 + 0.4j],
            [0.1 - 0.2j, 0.7 + 0.1j],
            [-0.3 + 0.5j, 0.4 - 0.6j],
        ],
        dtype=np.complex64,
    )
    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=pack_upper_tri(lhs))
    mean_prior = np.ones((4,), dtype=np.float32) * 10.0
    W_prior = np.ones((4, 1), dtype=np.float32) * 5.0

    mu_full, W_full = solve_augmented_ppca_mstep(stats, mean_prior=mean_prior, W_prior=W_prior)
    mu_chunk, W_chunk = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=W_prior,
        chunk_size=2,
    )

    np.testing.assert_allclose(np.asarray(mu_chunk), np.asarray(mu_full), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(W_chunk), np.asarray(W_full), rtol=1e-6, atol=1e-6)
