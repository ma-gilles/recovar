"""The dense scorers derive their scores from one cross/model-energy GEMM owner."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.scoring import scoring


def test_components_match_the_documented_gemms():
    rng = np.random.default_rng(0)
    n_images, n_trans, n_half, n_rot = 2, 3, 6, 4
    shifted = jnp.asarray((rng.standard_normal((n_images * n_trans, n_half)) + 1j * rng.standard_normal((n_images * n_trans, n_half))).astype(np.complex64))
    ctf2 = jnp.asarray(np.abs(rng.standard_normal((n_images, n_half))).astype(np.float32))
    proj_w = jnp.asarray((rng.standard_normal((n_rot, n_half)) + 1j * rng.standard_normal((n_rot, n_half))).astype(np.complex64))
    proj_abs2 = jnp.asarray(np.abs(rng.standard_normal((n_rot, n_half))).astype(np.float32))
    cross, norms = scoring._e_step_block_score_components(shifted, ctf2, proj_w, proj_abs2, n_images, n_trans)
    assert cross.shape == (n_images, n_rot, n_trans) and norms.shape == (n_images, n_rot)
    expected_cross = (-2.0 * jnp.matmul(jnp.conj(shifted), proj_w.T, precision=jax.lax.Precision.HIGHEST).real).reshape(n_images, n_trans, n_rot).swapaxes(1, 2)
    expected_norms = jnp.matmul(ctf2, proj_abs2.T, precision=jax.lax.Precision.HIGHEST)
    assert np.array_equal(np.asarray(cross), np.asarray(expected_cross))
    assert np.array_equal(np.asarray(norms), np.asarray(expected_norms))
    residual = scoring._e_step_block_scores(shifted, jnp.zeros(n_images), ctf2, proj_w, proj_abs2, jnp.ones(n_half), n_images, n_trans, (4, 4), (4, 4, 4))
    assert np.array_equal(np.asarray(residual), np.asarray(-0.5 * (cross + norms[..., None])))
    cc = scoring._e_step_block_scores_normalized_cc(shifted, jnp.zeros(n_images), ctf2, proj_w, proj_abs2, n_images, n_trans, (4, 4), (4, 4, 4))
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    assert np.array_equal(np.asarray(cc), np.asarray((-0.5 * cross) / denom[..., None]))


def test_dense_scorers_use_the_owner():
    for fn in (
        scoring._e_step_block_scores,
        scoring._e_step_block_scores_windowed,
        scoring._e_step_block_scores_normalized_cc,
        scoring._e_step_block_scores_windowed_normalized_cc,
    ):
        source = inspect.getsource(fn)
        assert source.count("cross, norms = _e_step_block_score_components(") == 1
        assert "jnp.matmul(" not in source
    assert not hasattr(scoring, "_e_step_block_score_components_windowed")
    module_source = inspect.getsource(scoring)
    assert module_source.count("precision=jax.lax.Precision.HIGHEST") == 4
