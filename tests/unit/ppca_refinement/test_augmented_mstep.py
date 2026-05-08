import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.mean_regularization import relion_style_mean_precision_from_filter
from recovar.ppca import (
    AugmentedPPCAStats,
    augmented_ppca_mstep_objective,
    pack_upper_tri,
    solve_augmented_ppca_mstep,
)
from recovar.reconstruction import relion_functions


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


def test_augmented_mstep_objective_improves_at_solved_model():
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
    lhs_reg = lhs + np.eye(2, dtype=np.float32)[None] * reg[:, None, :]
    rhs = np.einsum("fij,fj->fi", lhs_reg, theta_true)

    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=pack_upper_tri(lhs))
    mu, W = solve_augmented_ppca_mstep(stats, mean_prior=mean_prior, W_prior=W_prior)
    solved = augmented_ppca_mstep_objective(stats, mu, W, mean_prior=mean_prior, W_prior=W_prior)
    zero = augmented_ppca_mstep_objective(
        stats,
        np.zeros((2,), dtype=np.complex64),
        np.zeros((2, 1), dtype=np.complex64),
        mean_prior=mean_prior,
        W_prior=W_prior,
    )

    expected_total = 0.5 * np.real(np.sum(np.conj(theta_true) * rhs))
    np.testing.assert_allclose(np.asarray(solved.total), expected_total, rtol=1e-6, atol=1e-6)
    assert float(solved.total) > float(zero.total)


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


def test_augmented_mstep_q_zero_accepts_explicit_mean_precision():
    lhs = np.array([[[2.0]], [[4.0]], [[8.0]]], dtype=np.float32)
    mean_precision = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    theta_true = np.array([3.0, -1.0, 0.25], dtype=np.complex64)
    rhs = (lhs[:, 0, 0] + mean_precision) * theta_true

    stats = AugmentedPPCAStats(rhs=rhs[:, None], lhs_tri=pack_upper_tri(lhs))
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=np.ones((3,), dtype=np.float32),
        W_prior=np.zeros((3, 0), dtype=np.float32),
        mean_precision=mean_precision,
    )

    np.testing.assert_allclose(np.asarray(mu), theta_true, rtol=1e-6, atol=1e-6)
    assert np.asarray(W).shape == (3, 0)


def test_augmented_mstep_fixed_mean_solves_conditional_w_equation():
    lhs = np.array(
        [
            [[2.0, 0.25, -0.10], [0.25, 4.0, 0.30], [-0.10, 0.30, 3.0]],
            [[3.0, -0.50, 0.20], [-0.50, 5.0, -0.40], [0.20, -0.40, 2.5]],
        ],
        dtype=np.float32,
    )
    fixed_mu = np.array([1.5 + 0.25j, -2.0 + 0.1j], dtype=np.complex64)
    W_true = np.array([[0.75 - 0.5j, -0.25 + 0.1j], [0.5 + 0.2j, -0.4 - 0.3j]], dtype=np.complex64)
    W_prior = np.array([[5.0, 8.0], [6.0, 7.0]], dtype=np.float32)
    W_reg = 1.0 / W_prior

    rhs_w = np.einsum("fqp,fp->fq", lhs[:, 1:, 1:] + np.eye(2)[None] * W_reg[:, None, :], W_true)
    rhs_w = rhs_w + lhs[:, 1:, 0] * fixed_mu[:, None]
    rhs_mu = np.array([10.0 - 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    rhs = np.concatenate([rhs_mu[:, None], rhs_w], axis=1).astype(np.complex64)

    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=pack_upper_tri(lhs))
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=np.ones((2,), dtype=np.float32),
        W_prior=W_prior,
        fixed_mean=fixed_mu,
    )

    np.testing.assert_allclose(np.asarray(mu), fixed_mu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(W), W_true, rtol=1e-6, atol=1e-6)


def test_augmented_mstep_fixed_mean_objective_improves_conditional_w():
    lhs = np.array(
        [
            [[2.0, 0.25, -0.10], [0.25, 4.0, 0.30], [-0.10, 0.30, 3.0]],
            [[3.0, -0.50, 0.20], [-0.50, 5.0, -0.40], [0.20, -0.40, 2.5]],
        ],
        dtype=np.float32,
    )
    fixed_mu = np.array([1.5 + 0.25j, -2.0 + 0.1j], dtype=np.complex64)
    W_true = np.array([[0.75 - 0.5j, -0.25 + 0.1j], [0.5 + 0.2j, -0.4 - 0.3j]], dtype=np.complex64)
    W_prior = np.array([[5.0, 8.0], [6.0, 7.0]], dtype=np.float32)
    W_reg = 1.0 / W_prior
    rhs_w = np.einsum("fqp,fp->fq", lhs[:, 1:, 1:] + np.eye(2)[None] * W_reg[:, None, :], W_true)
    rhs_w = rhs_w + lhs[:, 1:, 0] * fixed_mu[:, None]
    rhs_mu = np.array([10.0 - 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    rhs = np.concatenate([rhs_mu[:, None], rhs_w], axis=1).astype(np.complex64)
    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=pack_upper_tri(lhs))

    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=np.ones((2,), dtype=np.float32),
        W_prior=W_prior,
        fixed_mean=fixed_mu,
    )
    solved = augmented_ppca_mstep_objective(
        stats,
        mu,
        W,
        mean_prior=np.ones((2,), dtype=np.float32),
        W_prior=W_prior,
    )
    zero_w = augmented_ppca_mstep_objective(
        stats,
        fixed_mu,
        np.zeros_like(W_true),
        mean_prior=np.ones((2,), dtype=np.float32),
        W_prior=W_prior,
    )

    assert float(solved.total) > float(zero_w.total)


def test_relion_style_mean_precision_matches_kclass_denominator_adjustment():
    volume_shape = (8, 8, 8)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))
    mean_filter = np.linspace(0.5, 4.0, half_size, dtype=np.float32)
    tau_half = np.linspace(0.2, 2.0, half_size, dtype=np.float32)

    precision = relion_style_mean_precision_from_filter(
        mean_filter,
        tau_half,
        volume_shape,
        tau2_fudge=2.0,
        minres_map=2,
    )
    tau_full = ftu.half_volume_to_full_volume(tau_half.reshape(half_shape), volume_shape).reshape(-1).real
    regularized = relion_functions.adjust_regularization_relion_style(
        mean_filter,
        volume_shape,
        tau=tau_full,
        half_volume=True,
        tau2_fudge=2.0,
        minres_map=2,
    )

    np.testing.assert_allclose(
        np.asarray(precision),
        np.asarray(regularized - mean_filter),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(np.asarray(precision) >= 0.0)


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
