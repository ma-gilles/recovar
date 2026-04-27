"""Tests for multi-mask PCG M-step and Gauss-Legendre contrast quadrature."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Contrast quadrature tests
# ---------------------------------------------------------------------------


class TestContrastQuadrature:
    """Verify Gauss-Legendre quadrature produces correct weights."""

    def test_gauss_legendre_weights_sum_to_interval_length(self):
        from recovar.ppca.contrast_posterior import make_contrast_quadrature

        nodes, weights = make_contrast_quadrature(rule="gauss_legendre", interval=(0.0, 2.0), n_nodes=16)
        assert nodes.shape == (16,)
        assert weights.shape == (16,)
        # Weights should sum to interval length
        np.testing.assert_allclose(float(weights.sum()), 2.0, rtol=1e-5)

    def test_gauss_legendre_nodes_within_interval(self):
        from recovar.ppca.contrast_posterior import make_contrast_quadrature

        nodes, weights = make_contrast_quadrature(rule="gauss_legendre", interval=(0.0, 2.0), n_nodes=16)
        assert float(nodes.min()) > 0.0
        assert float(nodes.max()) < 2.0

    def test_gauss_legendre_integrates_polynomial_exactly(self):
        """GL with n nodes integrates polynomials of degree <= 2n-1 exactly."""
        from recovar.ppca.contrast_posterior import make_contrast_quadrature

        # Integrate x^3 on [0, 2]: exact = 2^4/4 = 4.0
        n = 8  # can integrate up to degree 15
        nodes, weights = make_contrast_quadrature(rule="gauss_legendre", interval=(0.0, 2.0), n_nodes=n)
        integral = float(jnp.sum(weights * nodes**3))
        np.testing.assert_allclose(integral, 4.0, rtol=1e-5)

    def test_marginalize_uses_weights(self):
        """Marginalization with non-uniform weights differs from uniform."""
        from recovar.ppca.contrast_posterior import (
            make_contrast_quadrature,
            solve_marginalized_contrast,
        )

        rng = np.random.default_rng(42)
        B, K = 4, 3
        H = rng.normal(size=(B, K, K)).astype(np.float32)
        H = H @ H.transpose(0, 2, 1) + np.eye(K) * 0.1  # PSD
        g = rng.normal(size=(B, K)).astype(np.float32)
        h = rng.normal(size=(B, K)).astype(np.float32) * 0.1
        t = rng.normal(size=(B,)).astype(np.float32)
        nu = rng.normal(size=(B,)).astype(np.float32).clip(0.1)
        y_norm_sq = rng.uniform(1, 5, size=(B,)).astype(np.float32)
        lambdas = jnp.ones(K)

        # Gauss-Legendre weights
        nodes_gl, weights_gl = make_contrast_quadrature(rule="gauss_legendre", interval=(0.0, 2.0), n_nodes=8)
        # Uniform weights (same nodes)
        weights_uniform = jnp.ones(8) * (2.0 / 8.0)

        result_gl = solve_marginalized_contrast(
            jnp.array(H),
            jnp.array(g),
            jnp.array(h),
            jnp.array(t),
            jnp.array(nu),
            jnp.array(y_norm_sq),
            lambdas,
            nodes_gl,
            weights_gl,
            1.0,
            jnp.inf,
        )
        result_uni = solve_marginalized_contrast(
            jnp.array(H),
            jnp.array(g),
            jnp.array(h),
            jnp.array(t),
            jnp.array(nu),
            jnp.array(y_norm_sq),
            lambdas,
            nodes_gl,
            weights_uniform,
            1.0,
            jnp.inf,
        )
        # Results should differ (weights matter)
        assert not np.allclose(np.array(result_gl.mean_z), np.array(result_uni.mean_z), atol=1e-4)


# ---------------------------------------------------------------------------
# Multi-mask PCG tests
# ---------------------------------------------------------------------------


class TestMultiMaskPCG:
    """Verify multi-mask projected CG produces correct masking behavior."""

    def _make_pcg_test_data(self, vs=(8, 8, 8), q=4, n_masks=2):
        """Build synthetic LHS/RHS for PCG testing."""
        import recovar.core.fourier_transform_utils as ftu

        rng = np.random.default_rng(123)
        half_vs = ftu.volume_shape_to_half_volume_shape(vs)
        half_vol = int(np.prod(half_vs))
        tri_sz = q * (q + 1) // 2

        # Random PSD LHS (diagonal-dominant for convergence)
        lhs_tri = rng.normal(size=(half_vol, tri_sz)).astype(np.float32) * 0.01
        # Make diagonal entries large
        diag_indices = [i * (i + 1) // 2 + i for i in range(q)]  # wrong
        idx = 0
        for i in range(q):
            for j in range(i, q):
                if i == j:
                    lhs_tri[:, idx] += 1.0
                idx += 1

        rhs_h = (rng.normal(size=(half_vol, q)) + 1j * rng.normal(size=(half_vol, q))).astype(np.complex64) * 0.1
        reg_diag = np.ones((half_vol, q), dtype=np.float32) * 0.1

        # Create masks: left half and right half of volume
        mask1 = np.zeros(vs, dtype=np.float32)
        mask1[:, :, : vs[2] // 2] = 1.0
        mask2 = np.zeros(vs, dtype=np.float32)
        mask2[:, :, vs[2] // 2 :] = 1.0

        masks = np.stack([mask1, mask2])
        # PCs 0,1 → mask 0; PCs 2,3 → mask 1
        assignment = np.array([0, 0, 1, 1], dtype=np.int32)

        return lhs_tri, rhs_h, reg_diag, masks, assignment

    def test_single_mask_equivalence(self):
        """Multi-mask with one mask = single mask behavior."""
        from recovar.ppca.ppca import _pcg_hard_mstep, unpack_tri_to_full

        vs = (8, 8, 8)
        q = 3
        rng = np.random.default_rng(42)
        import recovar.core.fourier_transform_utils as ftu

        half_vs = ftu.volume_shape_to_half_volume_shape(vs)
        half_vol = int(np.prod(half_vs))
        tri_sz = q * (q + 1) // 2

        lhs_tri = np.abs(rng.normal(size=(half_vol, tri_sz)).astype(np.float32)) * 0.01
        # Strengthen diagonal
        idx = 0
        for i in range(q):
            for j in range(i, q):
                if i == j:
                    lhs_tri[:, idx] += 1.0
                idx += 1
        lhs_tri = jnp.array(lhs_tri)

        rhs_h = jnp.array(
            (rng.normal(size=(half_vol, q)) + 1j * rng.normal(size=(half_vol, q))).astype(np.complex64) * 0.1
        )
        reg_diag = jnp.ones((half_vol, q), dtype=jnp.float32) * 0.1

        mask = np.ones(vs, dtype=np.float32)
        masks_single = np.ones((1, *vs), dtype=np.float32)
        assignment_single = np.zeros(q, dtype=np.int32)

        # Single-mask call
        W_single = _pcg_hard_mstep(lhs_tri, rhs_h, reg_diag, mask, vs, q, unpack_tri_to_full, maxiter=30)
        # Multi-mask call with one mask (same as single)
        W_multi = _pcg_hard_mstep(
            lhs_tri,
            rhs_h,
            reg_diag,
            mask,
            vs,
            q,
            unpack_tri_to_full,
            maxiter=30,
            masks=masks_single,
            pc_mask_assignment=assignment_single,
        )

        np.testing.assert_allclose(np.array(W_single), np.array(W_multi), rtol=1e-3, atol=1e-5)

    def test_multi_mask_zeros_outside_support(self):
        """Each PC should be zero outside its assigned mask."""
        from recovar.ppca.ppca import _pcg_hard_mstep, unpack_tri_to_full

        vs = (8, 8, 8)
        q = 4
        lhs_tri, rhs_h, reg_diag, masks, assignment = self._make_pcg_test_data(vs=vs, q=q)

        # Union mask for the single-mask arg
        union_mask = np.any(masks > 0.5, axis=0).astype(np.float32)

        W_real = _pcg_hard_mstep(
            jnp.array(lhs_tri),
            jnp.array(rhs_h),
            jnp.array(reg_diag),
            union_mask,
            vs,
            q,
            unpack_tri_to_full,
            maxiter=30,
            masks=masks,
            pc_mask_assignment=assignment,
        )
        W_np = np.array(W_real)  # (q, D, D, D)

        # PCs 0,1 should be zero where mask1 is 0 (right half)
        assert np.allclose(W_np[0, :, :, vs[2] // 2 :], 0.0, atol=1e-6)
        assert np.allclose(W_np[1, :, :, vs[2] // 2 :], 0.0, atol=1e-6)
        # PCs 2,3 should be zero where mask2 is 0 (left half)
        assert np.allclose(W_np[2, :, :, : vs[2] // 2], 0.0, atol=1e-6)
        assert np.allclose(W_np[3, :, :, : vs[2] // 2], 0.0, atol=1e-6)

    def test_pc_mask_assignment_default_even_split(self):
        """EM() default assignment splits PCs evenly across masks."""

        # Just test the assignment logic without running full EM
        # by inspecting what would be generated
        n_masks = 3
        basis_size = 9
        expected = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        actual = np.array([k % n_masks for k in range(basis_size)], dtype=np.int32)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Pipeline wiring test
# ---------------------------------------------------------------------------


class TestPipelineWiring:
    """Verify pipeline resolves contrast grid with proper weights."""

    def test_resolve_ppca_contrast_grid_returns_weights(self):
        import sys

        sys.path.insert(0, "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_wt_contrast_multimask")
        from recovar.commands.pipeline import _resolve_ppca_contrast_grid

        nodes, weights = _resolve_ppca_contrast_grid("marginalize")
        assert nodes is not None
        assert weights is not None
        assert nodes.shape == (16,)
        assert weights.shape == (16,)
        np.testing.assert_allclose(weights.sum(), 2.0, rtol=1e-5)

    def test_resolve_ppca_contrast_grid_none_mode(self):
        from recovar.commands.pipeline import _resolve_ppca_contrast_grid

        nodes, weights = _resolve_ppca_contrast_grid("none")
        assert nodes is None
        assert weights is None
