"""Multi-translation parity for the half-image kernel.

Existing kernel tests use `n_trans=1` (or all-zero translations).
This file exercises the kernel with multiple non-trivial translations
and verifies it agrees with the production score path. This catches:

  - Broadcast/reshape bugs in the (n_img, n_trans, half_image) handling
  - Incorrect translation application order (full vs half image)
  - Wrong sign convention or wrong axis for trans expansion
  - Broken `n_trans` axis in the post_mean / log_scores tensors

We use **non-astigmatic CTF** to avoid the known full-vs-half production
discrepancy at the y-Nyquist row (see test_kernel_real_ctf_parity.py for
the documentation of that pre-existing inconsistency).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import equinox as eqx
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu
import recovar.em.core as em_core_em
import recovar.em.heterogeneity as hetero
from recovar.core.ctf import CTFEvaluator, CTFMode

pytestmark = [pytest.mark.unit, pytest.mark.slow]


IMAGE_SHAPE = (8, 8)
N_FULL = 64
VOLUME_SHAPE = (8, 8, 8)


def _real_ctf(p, sh, vs, *, half_image=False):
    return CTFEvaluator(mode=CTFMode.SPA)(p, sh, vs, half_image=half_image)


def _identity_process(b, apply_image_mask=False):
    return b


class _Cfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        return _real_ctf(p, self.image_shape, self.voxel_size, half_image=half_image)

    def process_fn(self, b, apply_image_mask=False):
        return b


def _real_to_full(real):
    return jnp.asarray(ftu.get_dft2(jnp.asarray(real)).reshape(-1), dtype=jnp.complex128)


def _make_inputs(rng, n_rot, n_pc, n_img):
    """Build half-volume mu, U_rows, scalar s, real-derived batch, non-astig CTF."""
    # mu and U live in volume space — but for the score parity test we
    # bypass slicing and pass projections directly. So we build mean
    # projections as random Hermitian images and U_rows similarly.
    mean_proj_full = jnp.stack([_real_to_full(rng.standard_normal(IMAGE_SHAPE)) for _ in range(n_rot)])
    u_proj_full = jnp.stack(
        [jnp.stack([_real_to_full(0.1 * rng.standard_normal(IMAGE_SHAPE)) for _ in range(n_pc)]) for _ in range(n_rot)]
    )
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=jnp.float64)
    batch_full = jnp.stack([_real_to_full(rng.standard_normal(IMAGE_SHAPE)) for _ in range(n_img)])

    DFU = rng.uniform(10000, 30000, size=n_img).astype(np.float64)
    DFV = DFU.copy()  # non-astigmatic
    DFANG = np.zeros(n_img, dtype=np.float64)
    ctf_params = jnp.asarray(
        np.stack(
            [
                DFU,
                DFV,
                DFANG,
                np.full(n_img, 300.0),
                np.full(n_img, 2.7),
                np.full(n_img, 0.07),
                np.zeros(n_img),
                np.zeros(n_img),
                np.ones(n_img),
            ],
            axis=-1,
        ),
        dtype=jnp.float64,
    )
    return mean_proj_full, u_proj_full, s, batch_full, ctf_params


def _production_residual(mean_proj_full, u_proj_full, s, batch_full, ctf_params, translations):
    """Compute the same residual `(n_img, n_rot, n_trans)` that the kernel would
    return as `-2 log_scores`, via the full-image production path."""
    config = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=2.0)
    noise_var = jnp.ones(N_FULL, dtype=jnp.float64)

    residuals = em_core_em.compute_dot_products_eqx(
        config, mean_proj_full, batch_full, translations, ctf_params, noise_var
    )
    bHb = hetero.compute_bHb_terms(
        mean_proj_full,
        u_proj_full,
        s,
        batch_full,
        translations,
        ctf_params,
        _real_ctf,
        noise_var,
        2.0,
        IMAGE_SHAPE,
        _identity_process,
    )
    residuals = residuals - bHb
    proj_norms = em_core_em.compute_CTFed_proj_norms_eqx(config, jnp.abs(mean_proj_full) ** 2, ctf_params, noise_var)
    return np.asarray(residuals + proj_norms[..., None])


def _kernel_log_scores_via_pre_sliced(mean_proj_full, u_proj_full, s, batch_full, ctf_params, translations):
    """Call the kernel with pre-sliced (already-projection) inputs and
    multiple translations. Mirrors how the kernel is called in
    `score_and_posterior_moments_eqx` but bypasses slice_volume by
    treating the random Hermitian inputs as already-sliced projections.
    """
    from recovar.em.ppca_abinitio.posterior import (
        _preprocess_batch_to_half,
        make_half_image_weights,
        score_from_half_image_projections,
    )

    config = _Cfg(image_shape=IMAGE_SHAPE, volume_shape=VOLUME_SHAPE, voxel_size=2.0)
    noise_var = jnp.ones(N_FULL, dtype=jnp.float64)

    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    weights_half = make_half_image_weights(IMAGE_SHAPE)
    mean_proj_half = ftu.full_image_to_half_image(mean_proj_full, IMAGE_SHAPE)
    u_proj_half = ftu.full_image_to_half_image(u_proj_full.reshape(n_rot * q, N_FULL), IMAGE_SHAPE).reshape(
        n_rot, q, n_half
    )
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_var
    )
    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_translation_log_scores_match_production():
    """Multi-translation parity at non-astigmatic CTF: kernel must
    match production for every (rotation, translation) pair, after
    centering by the per-image mean."""
    rng = np.random.default_rng(0xC0FFEE)
    n_rot, n_pc, n_img = 4, 2, 5
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img)

    # Multiple non-trivial translations: a small grid of 5 shifts
    translations = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [2.0, -1.0]],
        dtype=jnp.float64,
    )
    n_trans = translations.shape[0]

    prod_residual = _production_residual(mp, up, s, bf, cp, translations)
    prod_log = -0.5 * prod_residual  # (n_img, n_rot, n_trans)

    stats = _kernel_log_scores_via_pre_sliced(mp, up, s, bf, cp, translations)
    half_log = np.asarray(stats.log_scores)
    assert half_log.shape == (n_img, n_rot, n_trans), f"unexpected shape {half_log.shape}"

    # Center over (rot, trans) per image — removes the per-image constant
    prod_centered = prod_log - prod_log.reshape(n_img, -1).mean(axis=-1)[:, None, None]
    half_centered = half_log - half_log.reshape(n_img, -1).mean(axis=-1)[:, None, None]
    np.testing.assert_allclose(half_centered, prod_centered, rtol=1e-9, atol=1e-11)


def test_multi_translation_post_mean_shape_and_translation_dependence():
    """At multiple non-zero translations, the post_mean tensor must
    have the right shape AND the values must depend on the translation
    (i.e. moving the image actually changes the posterior mean).
    """
    rng = np.random.default_rng(0xBEEF)
    n_rot, n_pc, n_img = 3, 2, 4
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img)
    translations = jnp.asarray([[0.0, 0.0], [1.5, 0.5], [-0.5, 2.0]], dtype=jnp.float64)
    n_trans = 3

    stats = _kernel_log_scores_via_pre_sliced(mp, up, s, bf, cp, translations)
    pm = np.asarray(stats.post_mean)
    assert pm.shape == (n_img, n_rot, n_trans, n_pc), f"unexpected shape {pm.shape}"

    # post_mean must vary across translation axis (otherwise the
    # translation is being silently dropped)
    spread_across_trans = np.max(np.std(pm, axis=2))
    assert spread_across_trans > 1e-6, (
        f"post_mean is invariant across translations (std={spread_across_trans:.2e}); "
        f"the n_trans axis is not being threaded through correctly."
    )


def test_multi_translation_log_resp_normalizes_per_image():
    """The log_resp tensor must normalize over (n_rot * n_trans) per image."""
    rng = np.random.default_rng(0xDEAD)
    n_rot, n_pc, n_img = 4, 2, 3
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0], [0.5, -1.5]], dtype=jnp.float64)
    n_trans = 4

    stats = _kernel_log_scores_via_pre_sliced(mp, up, s, bf, cp, translations)
    lr = np.asarray(stats.log_resp).reshape(n_img, -1)
    sums = np.exp(lr).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)


def test_multi_translation_zero_trans_subset_matches_single_trans():
    """If translations = [[0,0], anything], then the t=0 column must
    equal the result obtained with translations=[[0,0]] alone. This
    catches inter-translation contamination bugs."""
    rng = np.random.default_rng(0xF00D)
    n_rot, n_pc, n_img = 3, 2, 4
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img)

    trans_single = jnp.zeros((1, 2), dtype=jnp.float64)
    trans_multi = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.5]], dtype=jnp.float64)

    stats_single = _kernel_log_scores_via_pre_sliced(mp, up, s, bf, cp, trans_single)
    stats_multi = _kernel_log_scores_via_pre_sliced(mp, up, s, bf, cp, trans_multi)

    # log_scores at trans 0 should match exactly (not centered — these
    # are the same calculation, no normalization in between)
    ls_single = np.asarray(stats_single.log_scores)[..., 0]
    ls_multi_at_0 = np.asarray(stats_multi.log_scores)[..., 0]
    np.testing.assert_allclose(ls_multi_at_0, ls_single, rtol=1e-12, atol=1e-13)

    pm_single = np.asarray(stats_single.post_mean)[..., 0, :]
    pm_multi_at_0 = np.asarray(stats_multi.post_mean)[..., 0, :]
    np.testing.assert_allclose(pm_multi_at_0, pm_single, rtol=1e-12, atol=1e-13)
