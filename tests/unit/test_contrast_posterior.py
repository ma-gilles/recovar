"""Tests for recovar.heterogeneity.contrast_posterior.

Covers:
- Quadrature construction
- Fixed-contrast (no-contrast) regression
- Spectral vs direct-Cholesky equivalence
- Profile-MAP regression against original solver
- Posterior moment sanity (PSD, sum-to-1, consistency)
- Prior limits (flat, concentrated)
- Near-singular cases
- Quadrature convergence
- Single-node marginalization matches no-contrast
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from recovar.heterogeneity.contrast_posterior import (
    LatentPosteriorResult,
    LegacyEmbeddingResult,
    make_contrast_quadrature,
    solve_latent_posterior,
    solve_marginalized_contrast,
    solve_no_contrast,
    solve_profile_contrast,
    _spectral_decomposition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_stats(rng, B, K):
    """Generate random sufficient statistics for B images, K latent dims."""
    # Make a valid PSD H from random factors
    F = rng.standard_normal((B, K, K + 2)).astype(np.float32)
    H = np.einsum("bij,bkj->bik", F, F)  # (B, K, K)
    H = 0.5 * (H + np.swapaxes(H, -1, -2))  # ensure exact symmetry
    g = rng.standard_normal((B, K)).astype(np.float32)
    h = rng.standard_normal((B, K)).astype(np.float32)
    t = rng.standard_normal((B,)).astype(np.float32)
    nu = np.abs(rng.standard_normal((B,))).astype(np.float32) + 1.0
    y_norm_sq = np.abs(rng.standard_normal((B,))).astype(np.float32) + 10.0
    lambdas = np.abs(rng.standard_normal((K,))).astype(np.float32) + 0.1
    return (
        jnp.array(H),
        jnp.array(g),
        jnp.array(h),
        jnp.array(t),
        jnp.array(nu),
        jnp.array(y_norm_sq),
        jnp.array(lambdas),
    )


def _direct_cholesky_solve(H, g, h, t, nu, y_norm_sq, lambdas, c):
    """Reference solver using direct Cholesky at a single contrast value.

    Returns (mu, Sigma, logdet_A, neg_log_score) for one image.
    """
    K = lambdas.shape[0]
    A = np.diag(1.0 / lambdas) + c ** 2 * H
    q = c * (g - c * h)
    mu = np.linalg.solve(A, q)
    Sigma = np.linalg.inv(A)
    logdet = np.linalg.slogdet(A)[1]
    rho = float(y_norm_sq) - 2.0 * c * float(t) + c ** 2 * float(nu)
    quad = float(q @ Sigma @ q)
    return mu, Sigma, logdet, rho - quad + logdet


# ---------------------------------------------------------------------------
# Quadrature tests
# ---------------------------------------------------------------------------


class TestMakeContrastQuadrature:

    def test_gauss_legendre_default(self):
        nodes, weights = make_contrast_quadrature()
        assert nodes.shape == (16,)
        assert weights.shape == (16,)
        # Nodes should be in [0, 3]
        assert float(jnp.min(nodes)) >= 0.0
        assert float(jnp.max(nodes)) <= 3.0
        # Weights should be positive and integrate to interval length
        assert jnp.all(weights > 0)
        np.testing.assert_allclose(float(jnp.sum(weights)), 3.0, atol=1e-5)

    def test_gauss_legendre_custom_interval(self):
        nodes, weights = make_contrast_quadrature(interval=(0.5, 2.5), n_nodes=8)
        assert nodes.shape == (8,)
        assert float(jnp.min(nodes)) >= 0.5
        assert float(jnp.max(nodes)) <= 2.5
        np.testing.assert_allclose(float(jnp.sum(weights)), 2.0, atol=1e-5)

    def test_trapezoid_uniform(self):
        nodes, weights = make_contrast_quadrature(rule="trapezoid", n_nodes=5, interval=(0, 4))
        np.testing.assert_allclose(np.array(nodes), [0, 1, 2, 3, 4], atol=1e-5)
        np.testing.assert_allclose(float(jnp.sum(weights)), 4.0, atol=1e-5)

    def test_trapezoid_explicit_nodes(self):
        custom_nodes = np.array([0.0, 0.5, 1.5, 3.0])
        nodes, weights = make_contrast_quadrature(rule="trapezoid", nodes=custom_nodes)
        assert nodes.shape == (4,)
        np.testing.assert_allclose(float(jnp.sum(weights)), 3.0, atol=1e-5)

    def test_custom_rule(self):
        n = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        nodes, weights = make_contrast_quadrature(rule="custom", nodes=n, weights=w)
        np.testing.assert_allclose(np.array(nodes), n, atol=1e-6)
        np.testing.assert_allclose(np.array(weights), w, atol=1e-6)

    def test_custom_requires_both(self):
        with pytest.raises(ValueError, match="custom rule requires"):
            make_contrast_quadrature(rule="custom", nodes=np.array([1.0]))

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="Unknown quadrature rule"):
            make_contrast_quadrature(rule="simpson")

    def test_gauss_legendre_integrates_polynomial_exactly(self):
        """n-point Gauss-Legendre integrates degree 2n-1 polynomials exactly."""
        n = 8
        nodes, weights = make_contrast_quadrature(
            rule="gauss_legendre", interval=(0, 3), n_nodes=n
        )
        # Integrate x^(2n-1) = x^15 on [0, 3]: exact = 3^16 / 16
        exact = 3.0 ** 16 / 16.0
        approx = float(jnp.sum(weights * nodes ** 15))
        np.testing.assert_allclose(approx, exact, rtol=1e-4)


# ---------------------------------------------------------------------------
# No-contrast mode
# ---------------------------------------------------------------------------


class TestNoContrast:

    def test_basic_solve(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=4, K=3)
        result = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)

        assert result.mean_z.shape == (4, 3)
        assert result.cov_z.shape == (4, 3, 3)
        assert result.best_contrast.shape == (4,)
        np.testing.assert_allclose(np.array(result.best_contrast), 1.0)
        np.testing.assert_allclose(np.array(result.mean_c), 1.0)

    def test_matches_direct_solve(self):
        """No-contrast result matches a direct A^{-1} q computation."""
        rng = np.random.default_rng(123)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=4)
        result = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)

        for b in range(2):
            H_np = np.array(H[b])
            g_np = np.array(g[b])
            h_np = np.array(h[b])
            lam_np = np.array(lambdas)
            A = np.diag(1.0 / lam_np) + H_np
            q = g_np - h_np
            mu_ref = np.linalg.solve(A, q)
            Sigma_ref = np.linalg.inv(A)
            np.testing.assert_allclose(np.array(result.mean_z[b]), mu_ref, atol=1e-5)
            np.testing.assert_allclose(np.array(result.cov_z[b]), Sigma_ref, atol=1e-5)

    def test_legacy_output(self):
        rng = np.random.default_rng(99)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        result, legacy = solve_no_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas, return_legacy=True
        )
        assert legacy.xs.shape == (2, 3)
        assert legacy.precision.shape == (2, 3, 3)
        np.testing.assert_allclose(np.array(legacy.xs), np.array(result.mean_z), atol=1e-6)

    def test_second_moment_consistency(self):
        """E[zz^T] - E[z]E[z]^T == Cov[z]."""
        rng = np.random.default_rng(77)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=5)
        result = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)
        recon_cov = np.array(result.second_moment_z) - np.einsum(
            "bi,bj->bij", np.array(result.mean_z), np.array(result.mean_z)
        )
        np.testing.assert_allclose(np.array(result.cov_z), recon_cov, atol=1e-5)


# ---------------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------------


class TestSpectralDecomposition:

    def test_eigenvalues_nonneg(self):
        rng = np.random.default_rng(42)
        H, g, h, _, _, _, lambdas = _random_stats(rng, B=3, K=4)
        d, T, alpha, beta = _spectral_decomposition(H, lambdas, g, h)
        assert jnp.all(d >= 0)

    def test_reconstruction(self):
        """Verify T diag(1/(1+c^2 d)) T^T == A(c)^{-1} in original basis."""
        rng = np.random.default_rng(55)
        H, g, h, _, _, _, lambdas = _random_stats(rng, B=2, K=3)
        d, T, alpha, beta = _spectral_decomposition(H, lambdas, g, h)

        c = 1.5
        c2 = c ** 2
        for b in range(2):
            s = 1.0 / (1.0 + np.array(d[b]) * c2)
            T_np = np.array(T[b])
            Sigma_spectral = T_np @ np.diag(s) @ T_np.T

            H_np = np.array(H[b])
            lam_np = np.array(lambdas)
            A_direct = np.diag(1.0 / lam_np) + c2 * H_np
            Sigma_direct = np.linalg.inv(A_direct)

            np.testing.assert_allclose(Sigma_spectral, Sigma_direct, atol=1e-4)


# ---------------------------------------------------------------------------
# Spectral vs direct Cholesky equivalence
# ---------------------------------------------------------------------------


class TestSpectralVsCholesky:
    """Compare the spectral solver outputs against a direct Cholesky reference."""

    def _reference_marginalized(self, H_np, g_np, h_np, t_np, nu_np, ynorm_np, lam_np, nodes, weights, c_mean, c_var):
        """Naive per-node Cholesky reference for one image."""
        K = lam_np.shape[0]
        C = len(nodes)
        mus = np.zeros((C, K))
        Sigmas = np.zeros((C, K, K))
        log_unnorm = np.zeros(C)

        for j in range(C):
            c = nodes[j]
            mu_j, Sigma_j, logdet_j, _ = _direct_cholesky_solve(
                H_np, g_np, h_np, t_np, nu_np, ynorm_np, lam_np, c
            )
            mus[j] = mu_j
            Sigmas[j] = Sigma_j

            rho = ynorm_np - 2.0 * c * t_np + c ** 2 * nu_np
            quad = mu_j @ np.diag(1.0 / lam_np + c ** 2 * np.diag(H_np)) @ mu_j
            # Actually recompute quad correctly
            q = c * (g_np - c * h_np)
            quad = q @ Sigma_j @ q

            is_finite = np.isfinite(c_var)
            if is_finite:
                log_prior = -0.5 * (c - c_mean) ** 2 / c_var
            else:
                log_prior = 0.0

            log_unnorm[j] = np.log(max(weights[j], 1e-30)) + log_prior - 0.5 * (rho - quad + logdet_j)

        # Softmax
        log_unnorm -= np.max(log_unnorm)
        omega = np.exp(log_unnorm)
        omega /= omega.sum()

        mean_z = omega @ mus
        second_moment_z = sum(omega[j] * (Sigmas[j] + np.outer(mus[j], mus[j])) for j in range(C))
        cov_z = second_moment_z - np.outer(mean_z, mean_z)
        mean_c = omega @ nodes
        second_moment_c = omega @ (nodes ** 2)
        mean_cz = sum(omega[j] * nodes[j] * mus[j] for j in range(C))
        mean_c2z = sum(omega[j] * nodes[j] ** 2 * mus[j] for j in range(C))
        second_moment_czz = sum(omega[j] * nodes[j] ** 2 * (Sigmas[j] + np.outer(mus[j], mus[j])) for j in range(C))

        return {
            "mean_z": mean_z,
            "cov_z": cov_z,
            "second_moment_z": second_moment_z,
            "mean_c": mean_c,
            "second_moment_c": second_moment_c,
            "mean_cz": mean_cz,
            "mean_c2z": mean_c2z,
            "second_moment_czz": second_moment_czz,
            "omega": omega,
        }

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_moments_match_cholesky(self, K):
        """Spectral marginalized solver matches direct Cholesky on random data."""
        rng = np.random.default_rng(42 + K)
        B = 3
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=B, K=K)
        nodes, weights = make_contrast_quadrature(
            rule="gauss_legendre", interval=(0.5, 2.5), n_nodes=8
        )

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )

        # Float32 tolerance scales with K (more operations -> more rounding)
        atol = 2e-4 if K <= 4 else 5e-3

        for b in range(B):
            ref = self._reference_marginalized(
                np.array(H[b]), np.array(g[b]), np.array(h[b]),
                float(t[b]), float(nu[b]), float(y_norm_sq[b]),
                np.array(lambdas), np.array(nodes), np.array(weights),
                1.0, np.inf,
            )
            np.testing.assert_allclose(np.array(result.mean_z[b]), ref["mean_z"], atol=atol)
            np.testing.assert_allclose(np.array(result.cov_z[b]), ref["cov_z"], atol=atol)
            np.testing.assert_allclose(np.array(result.second_moment_z[b]), ref["second_moment_z"], atol=atol)
            np.testing.assert_allclose(float(result.mean_c[b]), ref["mean_c"], atol=atol)
            np.testing.assert_allclose(float(result.second_moment_c[b]), ref["second_moment_c"], atol=atol)
            np.testing.assert_allclose(np.array(result.mean_cz[b]), ref["mean_cz"], atol=atol)
            np.testing.assert_allclose(np.array(result.mean_c2z[b]), ref["mean_c2z"], atol=atol)
            np.testing.assert_allclose(np.array(result.second_moment_czz[b]), ref["second_moment_czz"], atol=atol)
            np.testing.assert_allclose(np.array(result.contrast_weights_posterior[b]), ref["omega"], atol=atol)


# ---------------------------------------------------------------------------
# Profile mode regression
# ---------------------------------------------------------------------------


class TestProfileContrast:

    def test_matches_original_solver(self):
        """Profile mode should select the same best contrast and z as the original _solve_batch_from_stats."""
        from recovar.heterogeneity.embedding import _solve_batch_from_stats

        rng = np.random.default_rng(42)
        B, K = 5, 4
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=B, K=K)
        contrast_grid = jnp.linspace(0.04, 2.0, 50, dtype=jnp.float32)
        c_mean = jnp.float32(1.0)
        c_var = jnp.float32(np.inf)

        # Original solver
        xs_orig, c_orig, cov_orig = _solve_batch_from_stats(
            g, h, H, y_norm_sq, t, nu, lambdas, contrast_grid, c_mean, c_var
        )

        # New profile solver
        result, legacy = solve_profile_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_grid, c_mean, c_var,
            return_legacy=True,
        )

        np.testing.assert_allclose(np.array(legacy.contrasts), np.array(c_orig), atol=1e-5)
        np.testing.assert_allclose(np.array(legacy.xs), np.array(xs_orig), atol=1e-4)

    def test_profile_with_gaussian_prior(self):
        """Profile mode with finite prior variance should shift contrast toward prior mean."""
        rng = np.random.default_rng(77)
        B, K = 3, 3
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=B, K=K)
        nodes = jnp.linspace(0.1, 2.0, 30, dtype=jnp.float32)

        # Flat prior
        result_flat = solve_profile_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, jnp.float32(1.0), jnp.float32(np.inf),
        )
        # Tight prior at c=1
        result_tight = solve_profile_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, jnp.float32(1.0), jnp.float32(0.01),
        )
        # With tight prior, best_contrast should be closer to 1.0
        dist_flat = np.abs(np.array(result_flat.best_contrast) - 1.0)
        dist_tight = np.abs(np.array(result_tight.best_contrast) - 1.0)
        assert np.all(dist_tight <= dist_flat + 1e-6)


# ---------------------------------------------------------------------------
# Marginalized contrast
# ---------------------------------------------------------------------------


class TestMarginalizedContrast:

    def test_single_node_matches_no_contrast(self):
        """Marginalize with one node at c=1 should give same result as no-contrast."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=4)

        result_none = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)
        result_marg = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            jnp.array([1.0]),  # single node at c=1
            jnp.array([1.0]),  # weight=1
            jnp.float32(1.0),
            jnp.float32(np.inf),
        )

        np.testing.assert_allclose(
            np.array(result_marg.mean_z),
            np.array(result_none.mean_z),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(result_marg.cov_z),
            np.array(result_none.cov_z),
            atol=1e-5,
        )

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=5, K=3)
        nodes, weights = make_contrast_quadrature(n_nodes=16)

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        weight_sums = np.array(jnp.sum(result.contrast_weights_posterior, axis=-1))
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)

    def test_cov_z_symmetric_psd(self):
        """Posterior covariance should be symmetric PSD."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=4, K=5)
        nodes, weights = make_contrast_quadrature(n_nodes=16)

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        cov_np = np.array(result.cov_z)
        for b in range(4):
            # Symmetry
            np.testing.assert_allclose(cov_np[b], cov_np[b].T, atol=1e-5)
            # PSD: all eigenvalues >= 0
            eigvals = np.linalg.eigvalsh(cov_np[b])
            assert np.all(eigvals >= -1e-5), f"Negative eigenvalue: {eigvals.min()}"

    def test_second_moment_consistency(self):
        """E[zz^T] - E[z]E[z]^T == Cov[z]."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=4)
        nodes, weights = make_contrast_quadrature(n_nodes=16)

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        recon_cov = np.array(result.second_moment_z) - np.einsum(
            "bi,bj->bij", np.array(result.mean_z), np.array(result.mean_z)
        )
        np.testing.assert_allclose(np.array(result.cov_z), recon_cov, atol=1e-4)

    def test_concentrated_prior_selects_single_node(self):
        """Very tight prior near c=1 should concentrate posterior there."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        nodes, weights = make_contrast_quadrature(
            rule="gauss_legendre", interval=(0.5, 1.5), n_nodes=16
        )

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(0.001),  # very tight prior
        )
        # Posterior mean contrast should be very close to 1.0
        np.testing.assert_allclose(np.array(result.mean_c), 1.0, atol=0.05)

    def test_contrast_weighted_moments_consistency(self):
        """E[c^2 z z^T] should be >= E[c z] E[c z]^T in PSD sense (by convexity)."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        nodes, weights = make_contrast_quadrature(n_nodes=16)

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        # E[c^2 zz^T] - E[cz]E[cz]^T should be PSD
        # This is Cov[cz | y] which must be PSD
        cz_mean = np.array(result.mean_cz)
        czz_second = np.array(result.second_moment_czz)
        for b in range(2):
            cov_cz = czz_second[b] - np.outer(cz_mean[b], cz_mean[b])
            eigvals = np.linalg.eigvalsh(cov_cz)
            assert np.all(eigvals >= -1e-4), f"Negative eigenvalue in Cov[cz]: {eigvals.min()}"


# ---------------------------------------------------------------------------
# Near-singular cases
# ---------------------------------------------------------------------------


class TestNearSingular:

    def test_small_eigenvalues(self):
        """Solver should handle very small prior eigenvalues (strong regularization)."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, _ = _random_stats(rng, B=2, K=3)
        lambdas = jnp.array([1e-6, 1e-6, 1e-6], dtype=jnp.float32)

        result = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)
        # With tiny eigenvalues, z should be close to zero (strong prior)
        assert np.all(np.abs(np.array(result.mean_z)) < 1.0)
        assert np.isfinite(np.array(result.mean_z)).all()

    def test_rank_deficient_H(self):
        """Solver should handle rank-deficient H (some eigenvalues of G are zero)."""
        rng = np.random.default_rng(42)
        K = 4
        B = 2
        # Create rank-2 H from rank-2 factors
        F = rng.standard_normal((B, K, 2)).astype(np.float32)
        H = jnp.array(np.einsum("bij,bkj->bik", F, F))
        _, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=B, K=K)

        nodes, weights = make_contrast_quadrature(n_nodes=8)
        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        assert np.isfinite(np.array(result.mean_z)).all()
        assert np.isfinite(np.array(result.cov_z)).all()

    def test_large_contrast_values(self):
        """Solver should be stable with large contrast values in the interval."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        nodes, weights = make_contrast_quadrature(
            rule="gauss_legendre", interval=(0.0, 10.0), n_nodes=16
        )

        result = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes, weights,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        assert np.isfinite(np.array(result.mean_z)).all()
        assert np.isfinite(np.array(result.cov_z)).all()

    def test_very_large_eigenvalues(self):
        """Lambda -> inf means flat prior (unregularized)."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, _ = _random_stats(rng, B=2, K=3)
        lambdas = jnp.array([1e8, 1e8, 1e8], dtype=jnp.float32)

        result = solve_no_contrast(H, g, h, t, nu, y_norm_sq, lambdas)
        # With huge eigenvalues, Lambda^{-1} ~ 0, so A ~ H and mu ~ H^{-1} (g-h)
        assert np.isfinite(np.array(result.mean_z)).all()


# ---------------------------------------------------------------------------
# Quadrature convergence
# ---------------------------------------------------------------------------


class TestQuadratureConvergence:

    def test_gauss_legendre_converges(self):
        """16-point GL should be close to 64-point GL reference."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=4)

        nodes_16, weights_16 = make_contrast_quadrature(n_nodes=16)
        nodes_64, weights_64 = make_contrast_quadrature(n_nodes=64)

        result_16 = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes_16, weights_16,
            jnp.float32(1.0), jnp.float32(np.inf),
        )
        result_64 = solve_marginalized_contrast(
            H, g, h, t, nu, y_norm_sq, lambdas,
            nodes_64, weights_64,
            jnp.float32(1.0), jnp.float32(np.inf),
        )

        np.testing.assert_allclose(
            np.array(result_16.mean_z),
            np.array(result_64.mean_z),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(result_16.cov_z),
            np.array(result_64.cov_z),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(result_16.mean_c),
            np.array(result_64.mean_c),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------


class TestSolveLatentPosterior:

    def test_dispatch_none(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        result = solve_latent_posterior(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_mode="none",
        )
        np.testing.assert_allclose(np.array(result.best_contrast), 1.0)

    def test_dispatch_profile(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        result = solve_latent_posterior(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_mode="profile",
            n_contrast_nodes=16,
            contrast_interval=(0.5, 2.0),
        )
        assert result.mean_z.shape == (2, 3)
        assert result.profile_scores is not None

    def test_dispatch_marginalize(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        result = solve_latent_posterior(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_mode="marginalize",
            n_contrast_nodes=16,
        )
        assert result.mean_z.shape == (2, 3)
        weight_sums = np.array(jnp.sum(result.contrast_weights_posterior, axis=-1))
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-5)

    def test_dispatch_with_legacy(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        result, legacy = solve_latent_posterior(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_mode="marginalize",
            return_legacy=True,
        )
        assert isinstance(result, LatentPosteriorResult)
        assert isinstance(legacy, LegacyEmbeddingResult)
        assert legacy.xs.shape == (2, 3)

    def test_dispatch_unknown_mode(self):
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        with pytest.raises(ValueError, match="Unknown contrast_mode"):
            solve_latent_posterior(
                H, g, h, t, nu, y_norm_sq, lambdas,
                contrast_mode="bogus",
            )

    def test_explicit_nodes_trapezoid_fallback(self):
        """User-supplied nodes without weights should get trapezoid weights."""
        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=2, K=3)
        custom_nodes = jnp.linspace(0.5, 2.0, 10)

        result = solve_latent_posterior(
            H, g, h, t, nu, y_norm_sq, lambdas,
            contrast_mode="marginalize",
            contrast_nodes=custom_nodes,
        )
        assert result.contrast_weights_posterior.shape == (2, 10)


# ---------------------------------------------------------------------------
# Integration with embedding._solve_batch_from_stats_v2
# ---------------------------------------------------------------------------


class TestEmbeddingIntegration:

    def test_v2_profile_matches_v1(self):
        """_solve_batch_from_stats_v2 in profile mode returns same as original."""
        from recovar.heterogeneity.embedding import (
            _solve_batch_from_stats,
            _solve_batch_from_stats_v2,
        )

        rng = np.random.default_rng(42)
        B, K = 5, 4
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=B, K=K)
        cg = jnp.linspace(0.04, 2.0, 50, dtype=jnp.float32)
        c_mean = jnp.float32(1.0)
        c_var = jnp.float32(np.inf)

        xs_v1, c_v1, cov_v1 = _solve_batch_from_stats(
            g, h, H, y_norm_sq, t, nu, lambdas, cg, c_mean, c_var
        )
        xs_v2, c_v2, cov_v2 = _solve_batch_from_stats_v2(
            g, h, H, y_norm_sq, t, nu, lambdas, cg, c_mean, c_var,
            contrast_mode="profile",
        )

        np.testing.assert_allclose(np.array(xs_v2), np.array(xs_v1), atol=1e-5)
        np.testing.assert_allclose(np.array(c_v2), np.array(c_v1), atol=1e-5)
        np.testing.assert_allclose(np.array(cov_v2), np.array(cov_v1), atol=1e-4)

    def test_v2_none_mode(self):
        """_solve_batch_from_stats_v2 none mode returns c=1."""
        from recovar.heterogeneity.embedding import _solve_batch_from_stats_v2

        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=3)
        cg = jnp.ones(1, dtype=jnp.float32)
        c_mean = jnp.float32(1.0)
        c_var = jnp.float32(np.inf)

        xs, c, cov = _solve_batch_from_stats_v2(
            g, h, H, y_norm_sq, t, nu, lambdas, cg, c_mean, c_var,
            contrast_mode="none",
        )
        np.testing.assert_allclose(np.array(c), 1.0)

    def test_v2_marginalize_mode(self):
        """_solve_batch_from_stats_v2 marginalize mode returns finite results."""
        from recovar.heterogeneity.embedding import _solve_batch_from_stats_v2

        rng = np.random.default_rng(42)
        H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B=3, K=3)
        nodes, weights = make_contrast_quadrature(n_nodes=16)
        c_mean = jnp.float32(1.0)
        c_var = jnp.float32(np.inf)

        xs, c, cov = _solve_batch_from_stats_v2(
            g, h, H, y_norm_sq, t, nu, lambdas, nodes, c_mean, c_var,
            contrast_mode="marginalize",
            contrast_weights=weights,
        )
        assert np.isfinite(np.array(xs)).all()
        assert np.isfinite(np.array(c)).all()
        assert np.isfinite(np.array(cov)).all()
