"""The batched algebraic Gaussian scorers of sparse pass 2 share one score-terms owner."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import sparse_pass2_scoring as sp
def test_score_terms_match_the_documented_algebra():
    rng = np.random.default_rng(0); B, T, R, N = 2, 2, 3, 5
    shifted = jnp.asarray((rng.standard_normal((B, T, N)) + 1j * rng.standard_normal((B, T, N))).astype(np.complex64))
    corr = jnp.asarray(np.abs(rng.standard_normal((B, N))).astype(np.float32) + 0.1)
    proj = jnp.asarray((rng.standard_normal((B, R, N)) + 1j * rng.standard_normal((B, R, N))).astype(np.complex64))
    hw = jnp.asarray(np.abs(rng.standard_normal(N)).astype(np.float32) + 0.1)
    rprior = jnp.asarray(rng.standard_normal((B, R)).astype(np.float32)); tprior = jnp.asarray(rng.standard_normal((B, T)).astype(np.float32))
    preprior, scores = sp._gaussian_algebraic_score_terms(shifted, corr, proj, hw, rprior, tprior)
    weights = corr * hw[None, :]
    cross = jnp.einsum("btn,bn,brn->brt", jnp.conj(shifted), weights, proj, precision=jax.lax.Precision.HIGHEST).real
    proj_norm = 0.5 * jnp.einsum("bn,brn->br", weights, proj.real * proj.real + proj.imag * proj.imag, precision=jax.lax.Precision.HIGHEST)
    assert preprior.shape == scores.shape == (B, R, T)
    assert np.array_equal(np.asarray(preprior), np.asarray(cross - proj_norm[:, :, None]))
    assert np.array_equal(np.asarray(scores), np.asarray(preprior + rprior[:, :, None] + tprior[:, None, :]))
    mask = jnp.asarray(rng.random((B, R, T)) > 0.5)
    masked = sp._score_pass2_bucket_gaussian_algebraic(shifted, corr, proj, hw, rprior, tprior, mask)
    # Compare jitted against jitted: the production scorer is exactly this graph, so the
    # masking rule is pinned bitwise. (An eager recomputation would differ in the last ULP
    # because XLA fuses the einsums differently; old-versus-new equality is proven in the
    # package receipt by identical jaxprs and bit-identical outputs.)
    reference = jax.jit(
        lambda s, c, p, h, r, t, m: jnp.where(sp._gaussian_algebraic_score_terms(s, c, p, h, r, t)[1] > -jnp.inf, jnp.where(m, sp._gaussian_algebraic_score_terms(s, c, p, h, r, t)[1], -jnp.inf), -jnp.inf)
    )(shifted, corr, proj, hw, rprior, tprior, mask)
    assert np.array_equal(np.asarray(masked), np.asarray(reference))
    comp_scores, comp_preprior = sp._score_pass2_bucket_gaussian_algebraic_components(shifted, corr, proj, hw, rprior, tprior, mask)
    # Separate jitted kernels: equal to float32 rounding, not necessarily bitwise.
    np.testing.assert_allclose(np.asarray(comp_scores), np.asarray(masked), rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(comp_preprior)[np.asarray(mask)], np.asarray(jnp.where(mask, preprior, -jnp.inf))[np.asarray(mask)], rtol=1e-6, atol=0.0
    )
    assert np.all(np.isneginf(np.asarray(comp_preprior)[~np.asarray(mask)]))


def test_batched_scorers_use_the_owner():
    for fn in (sp._score_pass2_bucket_gaussian_algebraic_components, sp._score_pass2_bucket_gaussian_algebraic):
        source = inspect.getsource(fn)
        assert source.count("_gaussian_algebraic_score_terms(") == 1
        assert "jnp.einsum(" not in source
    assert inspect.getsource(sp._score_pass2_bucket_gaussian_algebraic_single_cached).count("jnp.einsum(") == 2
    assert inspect.getsource(sp).count('"btn,bn,brn->brt"') == 1
