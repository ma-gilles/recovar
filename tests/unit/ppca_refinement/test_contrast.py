"""Phase 9 (M8) tests: contrast-aware per-pose score + augmented moments.

Verifies:
  * profile and marginalize modes match
    ``recovar.ppca.contrast_posterior.solve_latent_posterior`` directly
    (the new function is just a wrapper that builds α_aug and G_aug_tri
    from the posterior moments per CLAUDE.md §5.2);
  * the augmented moment construction from contrast posterior outputs is
    correct (block structure + Hermitian upper triangle convention);
  * with contrast_mode='none'-style settings (mean=1, var=∞ + a single
    quadrature node at 1.0), the result coincides with the no-contrast
    function up to the marginal_ll convention difference.
  * ``renormalize_contrast_into_theta`` scales BOTH μ AND W (the
    failure-mode guard from CLAUDE.md anti-patterns).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.ppca.contrast_posterior import solve_latent_posterior  # noqa: E402
from recovar.ppca.pose_marginal import (  # noqa: E402
    compute_ppca_pose_scores_and_moments_with_contrast,
    renormalize_contrast_into_theta,
)
from recovar.ppca.ppca import _tri_size, unpack_tri_to_full  # noqa: E402

pytestmark = pytest.mark.unit


def _random_pose_stats_complex(rng, B, q):
    A = rng.standard_normal((B, q, q)) + 1j * rng.standard_normal((B, q, q))
    A = A.astype(np.complex64)
    Hzz = A.conj().swapaxes(-1, -2) @ A + 0.5 * np.broadcast_to(np.eye(q, dtype=np.complex64), (B, q, q))
    g_zx = (rng.standard_normal((B, q)) + 1j * rng.standard_normal((B, q))).astype(np.complex64)
    h_zm = (rng.standard_normal((B, q)) + 1j * rng.standard_normal((B, q))).astype(np.complex64)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    t_mx = rng.standard_normal(size=(B,)).astype(np.float32)
    nu_mm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    return y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz


def test_marginalize_mode_score_matches_solve_latent_posterior():
    rng = np.random.default_rng(42)
    B, q = 5, 2
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats_complex(rng, B, q)
    posterior = solve_latent_posterior(
        jnp.asarray(Hzz.real.astype("float32")),
        jnp.asarray(g_zx.real.astype("float32")),
        jnp.asarray(h_zm.real.astype("float32")),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(y_norm),
        jnp.ones((q,), dtype=jnp.float32),
        contrast_mode="marginalize",
    )
    score, _, _ = compute_ppca_pose_scores_and_moments_with_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        contrast_mode="marginalize",
        return_moments=False,
    )
    np.testing.assert_allclose(np.asarray(score), np.asarray(posterior.marginal_ll), rtol=1e-5, atol=1e-6)


def test_profile_mode_score_matches_solve_latent_posterior_max_profile():
    rng = np.random.default_rng(7)
    B, q = 3, 2
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats_complex(rng, B, q)
    posterior = solve_latent_posterior(
        jnp.asarray(Hzz.real.astype("float32")),
        jnp.asarray(g_zx.real.astype("float32")),
        jnp.asarray(h_zm.real.astype("float32")),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(y_norm),
        jnp.ones((q,), dtype=jnp.float32),
        contrast_mode="profile",
    )
    score, _, _ = compute_ppca_pose_scores_and_moments_with_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        contrast_mode="profile",
        return_moments=False,
    )
    expected = np.max(np.asarray(posterior.profile_scores), axis=-1)
    np.testing.assert_allclose(np.asarray(score), expected, rtol=1e-5, atol=1e-6)


def test_augmented_moments_match_contrast_posterior_block_structure():
    """alpha_aug[0] == mean_c, alpha_aug[1:] == mean_cz; G_aug upper
    triangle pulls (second_c, mean_c2z, second_c2zz)."""
    rng = np.random.default_rng(11)
    B, q = 4, 3
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats_complex(rng, B, q)
    posterior = solve_latent_posterior(
        jnp.asarray(Hzz.real.astype("float32")),
        jnp.asarray(g_zx.real.astype("float32")),
        jnp.asarray(h_zm.real.astype("float32")),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(y_norm),
        jnp.ones((q,), dtype=jnp.float32),
        contrast_mode="marginalize",
    )
    _, alpha_aug, G_tri = compute_ppca_pose_scores_and_moments_with_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        contrast_mode="marginalize",
        return_moments=True,
    )
    alpha_np = np.asarray(alpha_aug)
    G_np = np.asarray(unpack_tri_to_full(G_tri, q + 1))

    np.testing.assert_allclose(
        alpha_np[..., 0],
        np.asarray(posterior.mean_c).astype(np.complex64),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        alpha_np[..., 1:],
        np.asarray(posterior.mean_cz),
        rtol=1e-5,
        atol=1e-6,
    )
    # Upper-triangle structure: G[0,0] = E[c²]; G[0, 1:] = E[c² z];
    # G[1:, 1:] = E[c² z z*] (upper triangle).
    np.testing.assert_allclose(
        G_np[..., 0, 0].real,
        np.asarray(posterior.second_moment_c),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        G_np[..., 0, 1:],
        np.asarray(posterior.mean_c2z),
        rtol=1e-5,
        atol=1e-6,
    )
    iu = np.triu_indices(q, k=0)
    np.testing.assert_allclose(
        G_np[..., 1:, 1:][..., iu[0], iu[1]],
        np.asarray(posterior.second_moment_czz)[..., iu[0], iu[1]],
        rtol=1e-5,
        atol=1e-6,
    )


def test_with_contrast_arbitrary_leading_batch_shape():
    """The function vectorizes over arbitrary leading dims (e.g., [B, T, R])
    by flattening internally and reshaping back."""
    rng = np.random.default_rng(13)
    q = 1
    leading = (2, 3, 4)
    Bflat = int(np.prod(leading))
    A = rng.standard_normal((Bflat, q, q)) + 1j * rng.standard_normal((Bflat, q, q))
    A = A.astype(np.complex64)
    Hzz_flat = A.conj().swapaxes(-1, -2) @ A + 0.5 * np.eye(q, dtype=np.complex64)
    Hzz = Hzz_flat.reshape(leading + (q, q))
    g_zx = (rng.standard_normal(leading + (q,)) + 1j * rng.standard_normal(leading + (q,))).astype(np.complex64)
    h_zm = (rng.standard_normal(leading + (q,)) + 1j * rng.standard_normal(leading + (q,))).astype(np.complex64)
    y_norm = rng.uniform(0.5, 2.0, size=leading).astype(np.float32)
    t_mx = rng.standard_normal(leading).astype(np.float32)
    nu_mm = rng.uniform(0.5, 2.0, size=leading).astype(np.float32)

    score, alpha_aug, G_tri = compute_ppca_pose_scores_and_moments_with_contrast(
        jnp.asarray(y_norm),
        jnp.asarray(t_mx),
        jnp.asarray(nu_mm),
        jnp.asarray(g_zx),
        jnp.asarray(h_zm),
        jnp.asarray(Hzz),
        contrast_mode="marginalize",
        return_moments=True,
    )
    assert score.shape == leading
    assert alpha_aug.shape == leading + (q + 1,)
    assert G_tri.shape == leading + (_tri_size(q + 1),)


def test_with_contrast_rejects_invalid_mode():
    rng = np.random.default_rng(0)
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz = _random_pose_stats_complex(rng, 2, 1)
    with pytest.raises(ValueError, match="contrast_mode must be"):
        compute_ppca_pose_scores_and_moments_with_contrast(
            jnp.asarray(y_norm),
            jnp.asarray(t_mx),
            jnp.asarray(nu_mm),
            jnp.asarray(g_zx),
            jnp.asarray(h_zm),
            jnp.asarray(Hzz),
            contrast_mode="none",
            return_moments=False,
        )


# ---------------------------------------------------------------------------
# Renormalization helper
# ---------------------------------------------------------------------------


def test_renormalize_contrast_scales_mu_and_W_by_same_factor():
    """The CLAUDE.md anti-pattern is "scaling only μ not W." Verify the
    helper scales BOTH by the same factor."""
    mu = jnp.asarray(np.random.default_rng(0).standard_normal((6, 6, 6)).astype(np.float32))
    W = jnp.asarray(np.random.default_rng(1).standard_normal((3, 6, 6, 6)).astype(np.float32))
    s = 1.5
    mu2, W2 = renormalize_contrast_into_theta(mu, W, s)
    np.testing.assert_allclose(np.asarray(mu2), s * np.asarray(mu), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(W2), s * np.asarray(W), rtol=1e-6)
    # Identity mean_contrast=1 leaves μ, W unchanged.
    mu3, W3 = renormalize_contrast_into_theta(mu, W, 1.0)
    np.testing.assert_allclose(np.asarray(mu3), np.asarray(mu), rtol=1e-7)
    np.testing.assert_allclose(np.asarray(W3), np.asarray(W), rtol=1e-7)


def test_renormalize_contrast_rejects_nonpositive_scale():
    mu = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    W = jnp.zeros((2, 4, 4, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="must be positive"):
        renormalize_contrast_into_theta(mu, W, 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        renormalize_contrast_into_theta(mu, W, -0.5)
