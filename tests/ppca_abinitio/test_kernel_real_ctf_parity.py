"""Production-path parity for the half-image kernel with non-trivial CTF.

`test_score_matches_e_step_residual_ref.py` only verifies parity with
identity CTF. This file exercises the full assembly with a real
`recovar.core.ctf.CTFEvaluator` populated with realistic defocus values
to localize where the half-image kernel agrees (or disagrees) with the
production score path.

Findings (debug session, 2026-04-08, this branch)
-------------------------------------------------

The kernel mirrors `recovar.em.dense_single_volume.engine_v2`'s
half-image weighted-inner-product convention exactly. The half-image
identity ``Re<a, b>_full = sum_k w(k) Re[conj(a_half) b_half]`` holds
when the operands are Hermitian-symmetric.

For real-image FT inputs, ``batch * CTF`` is Hermitian iff the CTF is
centrosymmetric (``CTF[k] = CTF[-k]``). The astigmatic SPA CTF formula
satisfies this for all interior frequencies, BUT in the **fftshifted
even-N layout** the y-Nyquist row (row 0, freq y = -N/2) has no partner
row in the array — its conjugate partner +N/2 is aliased. The literal
formula then gives ``CTF[0, j] != CTF[0, N-j]`` because
``atan2(-N/2, fx) != atan2(-N/2, -fx)`` once astigmatism is present.

Consequence: with even-N images and astigmatic CTF, the **full-image**
production path (`compute_dot_products_eqx` + `compute_bHb_terms`) and
the **half-image** production path (`engine_v2`, which the kernel
mirrors) disagree on the y-Nyquist row's contribution. This is a
pre-existing inconsistency between two production code paths, not a
bug in the new kernel.

The discrepancy disappears in any of these cases:
  - **Identity CTF** (already tested in test_score_matches_e_step_residual_ref.py)
  - **Non-astigmatic CTF** (DFU == DFV, DFANG = 0): the formula has no
    angle dependence, so ``CTF[0, j] = CTF[0, N-j]`` trivially.
  - **Odd-N image shape**: there is no Nyquist row. The half-image
    layout has ``W = (N+1)/2`` columns and the y-axis goes from
    ``-(N-1)/2`` to ``+(N-1)/2``, both representable.
  - **Larger N**: the divergence does NOT shrink with N — the Nyquist
    row contributes a constant fraction (1 row out of N) and the
    asymmetric contribution per cell does not shrink either.

These tests pin the parity facts. They do not assert parity at
even-N + astigmatic CTF; that is a known production discrepancy.
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
from recovar.em.ppca_abinitio.posterior import (
    make_half_image_weights,
    score_from_half_image_projections,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Real CTF stand-in
# ---------------------------------------------------------------------------


def _real_ctf(CTF_params, image_shape, voxel_size, *, half_image=False):
    return CTFEvaluator(mode=CTFMode.SPA)(CTF_params, image_shape, voxel_size, half_image=half_image)


def _identity_process(batch, apply_image_mask=False):
    return batch


class _RealCTFConfig(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, ctf_params, *, half_image=False):
        return _real_ctf(ctf_params, self.image_shape, self.voxel_size, half_image=half_image)

    def process_fn(self, batch, apply_image_mask=False):
        return _identity_process(batch, apply_image_mask=apply_image_mask)


def _real_to_full(real, image_shape):
    return jnp.asarray(ftu.get_dft2(jnp.asarray(real)).reshape(-1), dtype=jnp.complex128)


def _make_inputs(rng, n_rot, n_pc, n_img, image_shape, *, astigmatic):
    n_full = int(np.prod(image_shape))

    mean_proj = jnp.stack(
        [_real_to_full(rng.standard_normal(image_shape).astype(np.float64), image_shape) for _ in range(n_rot)]
    )
    u_proj = jnp.stack(
        [
            jnp.stack(
                [
                    _real_to_full(0.1 * rng.standard_normal(image_shape).astype(np.float64), image_shape)
                    for _ in range(n_pc)
                ]
            )
            for _ in range(n_rot)
        ]
    )
    s = jnp.asarray(0.5 + rng.uniform(size=n_pc), dtype=jnp.float64)
    batch = jnp.stack(
        [_real_to_full(rng.standard_normal(image_shape).astype(np.float64), image_shape) for _ in range(n_img)]
    )

    DFU = rng.uniform(10000.0, 30000.0, size=n_img).astype(np.float64)
    if astigmatic:
        DFV = DFU + rng.uniform(-500.0, 500.0, size=n_img)
        DFANG = rng.uniform(0.0, 360.0, size=n_img)
    else:
        DFV = DFU.copy()
        DFANG = np.zeros(n_img, dtype=np.float64)
    VOLT = np.full(n_img, 300.0, dtype=np.float64)
    CS = np.full(n_img, 2.7, dtype=np.float64)
    W = np.full(n_img, 0.07, dtype=np.float64)
    PHASE = np.zeros(n_img, dtype=np.float64)
    BFACTOR = np.zeros(n_img, dtype=np.float64)
    CONTRAST = np.ones(n_img, dtype=np.float64)
    ctf_params = jnp.asarray(
        np.stack([DFU, DFV, DFANG, VOLT, CS, W, PHASE, BFACTOR, CONTRAST], axis=-1),
        dtype=jnp.float64,
    )
    return mean_proj, u_proj, s, batch, ctf_params


def _assemble_production(mean_proj_full, u_proj_full, s, batch_full, ctf_params, image_shape):
    config = _RealCTFConfig(image_shape=image_shape, voxel_size=2.0)
    n_full = int(np.prod(image_shape))
    trans = jnp.zeros((1, 2), dtype=jnp.float64)
    noise_var = jnp.ones(n_full, dtype=jnp.float64)

    residuals = em_core_em.compute_dot_products_eqx(config, mean_proj_full, batch_full, trans, ctf_params, noise_var)
    bHb = hetero.compute_bHb_terms(
        mean_proj_full,
        u_proj_full,
        s,
        batch_full,
        trans,
        ctf_params,
        _real_ctf,
        noise_var,
        2.0,
        image_shape,
        _identity_process,
    )
    residuals = residuals - bHb
    proj_norms = em_core_em.compute_CTFed_proj_norms_eqx(config, jnp.abs(mean_proj_full) ** 2, ctf_params, noise_var)
    return np.asarray(residuals + proj_norms[..., None])


def _call_kernel(mean_proj_full, u_proj_full, s, batch_full, ctf_params, image_shape):
    H, W = image_shape
    n_full = H * W
    n_half = H * (W // 2 + 1)
    n_img = batch_full.shape[0]
    n_rot = mean_proj_full.shape[0]
    q = u_proj_full.shape[1]

    weights_half = make_half_image_weights(image_shape)
    mean_proj_half = ftu.full_image_to_half_image(jnp.asarray(mean_proj_full), image_shape)
    u_proj_half = ftu.full_image_to_half_image(
        jnp.asarray(u_proj_full).reshape(n_rot * q, n_full), image_shape
    ).reshape(n_rot, q, n_half)
    batch_half = ftu.full_image_to_half_image(jnp.asarray(batch_full), image_shape)
    ctf_half = _real_ctf(ctf_params, image_shape, 2.0, half_image=True)
    noise_var_half = jnp.ones(n_half, dtype=jnp.float64)
    ctf2_over_nv_half = (ctf_half**2) / noise_var_half[None, :]
    shifted_half = (batch_half * ctf_half / noise_var_half[None, :])[:, None, :]

    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


def _check_parity(image_shape, *, astigmatic, atol, rtol):
    rng = np.random.default_rng(0xCAFE5)
    n_rot, n_pc, n_img = 4, 2, 5
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img, image_shape, astigmatic=astigmatic)
    prod_residual = _assemble_production(mp, up, s, bf, cp, image_shape)
    prod_log_scores = -0.5 * prod_residual[..., 0]
    stats = _call_kernel(mp, up, s, bf, cp, image_shape)
    half_log_scores = np.asarray(stats.log_scores)[..., 0]
    prod_centered = prod_log_scores - prod_log_scores.mean(axis=1, keepdims=True)
    half_centered = half_log_scores - half_log_scores.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(half_centered, prod_centered, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# PASSING parity tests
# ---------------------------------------------------------------------------


def test_kernel_matches_production_non_astigmatic_even_N():
    """Non-astigmatic CTF (DFU == DFV, DFANG = 0): no angle dependence,
    so the y-Nyquist row stays Hermitian and the kernel matches
    production at machine precision."""
    _check_parity((8, 8), astigmatic=False, atol=1e-10, rtol=1e-10)


def test_kernel_matches_production_astigmatic_odd_N():
    """Odd image shape: no Nyquist row. The kernel matches the
    full-image production path even with astigmatic CTF (only
    accumulated GEMM noise)."""
    _check_parity((7, 7), astigmatic=True, atol=1e-3, rtol=1e-5)


def test_kernel_matches_production_astigmatic_odd_N_larger():
    _check_parity((9, 9), astigmatic=True, atol=1e-3, rtol=1e-5)


# ---------------------------------------------------------------------------
# DOCUMENTING the known even-N + astigmatic discrepancy
# ---------------------------------------------------------------------------


def test_known_discrepancy_even_N_astigmatic():
    """Pin the known production inconsistency at even N + astigmatic
    CTF. The full-image and half-image production paths disagree on
    the y-Nyquist row contribution because the literal SPA CTF formula
    is not centrosymmetric on that row.

    This is a pre-existing inconsistency between
    `compute_dot_products_eqx` + `compute_bHb_terms` (full-image) and
    `engine_v2` (half-image, which the kernel mirrors). It is NOT a
    bug introduced by the kernel.

    If a future change unifies the two production paths to handle the
    Nyquist row consistently (e.g. by symmetrizing the CTF on row 0
    before scoring), this test will start failing — at which point the
    kernel test should be tightened to assert parity at even-N +
    astigmatic CTF instead.
    """
    rng = np.random.default_rng(0xCAFE5)
    image_shape = (8, 8)
    n_rot, n_pc, n_img = 4, 2, 5
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img, image_shape, astigmatic=True)
    prod_residual = _assemble_production(mp, up, s, bf, cp, image_shape)
    prod_log_scores = -0.5 * prod_residual[..., 0]
    stats = _call_kernel(mp, up, s, bf, cp, image_shape)
    half_log_scores = np.asarray(stats.log_scores)[..., 0]
    prod_centered = prod_log_scores - prod_log_scores.mean(axis=1, keepdims=True)
    half_centered = half_log_scores - half_log_scores.mean(axis=1, keepdims=True)
    diff = float(np.max(np.abs(prod_centered - half_centered)))
    # Discrepancy is large (O(100) at N=8 with random data). Pin a wide band.
    assert diff > 1.0, (
        f"Expected the known full-vs-half production discrepancy at even N + astigmatic "
        f"CTF to be O(>1) but got {diff:.3e}. If this is now <= 1.0, the production paths "
        f"may have been unified — see test docstring for what to update."
    )


def test_kernel_at_real_ctf_differs_from_identity_ctf():
    """Sanity: real CTF must produce DIFFERENT scores from identity
    CTF on the same inputs (verifies the CTF is actually applied)."""
    rng = np.random.default_rng(0xC0DE5)
    image_shape = (8, 8)
    n_rot, n_pc, n_img = 3, 2, 4
    mp, up, s, bf, cp = _make_inputs(rng, n_rot, n_pc, n_img, image_shape, astigmatic=True)
    stats_real = _call_kernel(mp, up, s, bf, cp, image_shape)

    # Identity CTF kernel call
    H, W = image_shape
    n_full = H * W
    n_half = H * (W // 2 + 1)
    weights_half = make_half_image_weights(image_shape)
    mean_proj_half = ftu.full_image_to_half_image(mp, image_shape)
    u_proj_half = ftu.full_image_to_half_image(up.reshape(n_rot * n_pc, n_full), image_shape).reshape(
        n_rot, n_pc, n_half
    )
    batch_half = ftu.full_image_to_half_image(bf, image_shape)
    ctf2_over_nv_half = jnp.ones((n_img, n_half), dtype=jnp.float64)
    shifted_half = batch_half[:, None, :]

    stats_id = score_from_half_image_projections(
        mean_proj_half=mean_proj_half.astype(jnp.complex128),
        u_proj_half=u_proj_half.astype(jnp.complex128),
        s=jnp.asarray(s, dtype=jnp.float64),
        shifted_half=shifted_half.astype(jnp.complex128),
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )

    diff = float(jnp.max(jnp.abs(stats_real.log_scores - stats_id.log_scores)))
    assert diff > 1e-3, (
        f"real CTF and identity CTF give nearly identical scores ({diff:.2e}); "
        "the CTF is not actually being applied through the kernel."
    )
