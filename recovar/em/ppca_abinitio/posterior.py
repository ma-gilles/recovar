"""Score and posterior moments for the half-volume PPCA helper.

Computes, in float64, for every (image, rotation, translation) triple
on the fixed grid:

  - `log_scores[i, r, t]` — marginal log-likelihood `log p(y_i | g)`
    up to a g-independent additive constant;
  - `log_resp[i, r, t]` — image-normalized log responsibilities;
  - `post_mean[i, r, t, k]` — posterior mean of the latent
    coefficient `α_i` under the (i, g) Gaussian model;
  - `post_Hinv[i, r, k, l]` — inverse of the posterior precision
    matrix `H_{i,r}`. Translation-independent (Section 4.6 of the
    spec), so the `n_trans` axis is dropped.

Half-volume layout
------------------

Per spec Section 0.3.1, `mu` and the rows of `U` live in the
**rfft-packed half-volume layout** `(N0, N1, N2//2+1)` flattened
to `half_volume_size`. Slicing uses
`recovar.core.slicing.slice_volume(..., half_volume=True,
half_image=True)` and produces half-image projections of shape
`(n_rot, half_image_size)` per volume.

Inner products on the half-spectrum are computed with the
**rfft Hermitian weights** (`recovar.em.ppca_abinitio.half_volume.
make_half_volume_weights` for 3D, `engine_v2.make_half_image_weights`
for 2D). With weights, `Σ_k w(k) Re[conj(a_half) b_half] = Re<a, b>_full`
exactly. The weights are pre-absorbed into the projection arrays
before each GEMM, mirroring `recovar/em/dense_single_volume/engine_v2.py:241-282`.

Translation handling matches the engine_v2 pattern: phase shifts are
applied in the **full-image** layout via
`core.batch_trans_translate_images`, then converted to half-image
via `full_image_to_half_image` (`engine_v2.py:202-208`). This is
because the phase factors live on the full grid; doing the shift
directly in the half-spectrum requires an explicit half-image phase
operator we have not written yet.

Float64 contract
----------------

This module runs strictly in `complex128` / `float64`. Per Audit 2,
production runs at float32 because `CryoEMDataset.dtype` defaults to
`np.complex64`; the float64 mode is achievable end-to-end and is what
makes the parity tests achievable at `rtol=1e-10`.
"""

from __future__ import annotations

from typing import Iterator

import jax
import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.configs import ForwardModelConfig
from recovar.core.slicing import slice_volume

from .types import PosteriorBlock, PosteriorStats

# ---------------------------------------------------------------------------
# 2D rfft Hermitian weights (mirror of engine_v2.make_half_image_weights)
# ---------------------------------------------------------------------------


def make_half_image_weights(image_shape) -> jnp.ndarray:
    """Hermitian weights for 2D rfft inner products. 3D analog is in
    `recovar.em.ppca_abinitio.half_volume.make_half_volume_weights`.

    Mirrors `recovar/em/dense_single_volume/engine_v2.py:56` exactly,
    but in float64 (engine_v2 uses float32).
    """
    H, W = (int(s) for s in image_shape)
    w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float64)
    w = w.at[:, 0].set(1.0)
    if W % 2 == 0:
        w = w.at[:, -1].set(1.0)
    return w.reshape(-1)


# ---------------------------------------------------------------------------
# Slicing helpers (half-volume → half-image, q rows of U)
# ---------------------------------------------------------------------------


# NOTE: v0 ab-initio uses NEAREST discretization throughout. The
# rationale (per the design memo of 2026-04-09):
#
#   1. Forward model parity. The synthetic simulator and the inversion
#      code use the same slice operator A_g, so there is no
#      forward/inverse mismatch contributing to the FRE floor.
#
#   2. Diagonal M-step operator. With nearest, A_g[pixel, voxel] is
#      binary (0 or 1), so the per-pose Gram operator
#      `A_g^T diag(CTF^2/sigma^2) A_g` is exactly diagonal in the
#      voxel basis. This makes the PPCA M-step a per-voxel q×q solve
#      with no PCG, no preconditioner, and no nullspace handling.
#
#   3. No mask, no gridding correction. Both of those are deferred
#      post-v0; nearest sidesteps them entirely.
#
# If we ever switch to linear-interp slicing, the per-voxel diagonal
# approximation has ~80% relative error per voxel and the M-step
# requires either a kernel-regression reinterpretation or a CG solve.
# See `docs/math/ppca_closed_form_mstep.md` for the discussion.


def _slice_mu_half(mu_half, rotations, image_shape, volume_shape) -> jnp.ndarray:
    """Slice the half-volume `mu` through `rotations`. Returns
    `(n_rot, half_image_size)` complex128.

    Uses nearest discretization (see module note above).
    """
    return slice_volume(
        mu_half,
        rotations,
        image_shape,
        volume_shape,
        "nearest",
        half_volume=True,
        half_image=True,
    )


def _slice_U_half(U_half, rotations, image_shape, volume_shape) -> jnp.ndarray:
    """Slice each row of `U_half` through `rotations`. Returns
    `(n_rot, q, half_image_size)` complex128.

    Uses nearest discretization (see module note above).
    """

    def slice_one(u_row):
        return slice_volume(
            u_row,
            rotations,
            image_shape,
            volume_shape,
            "nearest",
            half_volume=True,
            half_image=True,
        )

    # vmap over q axis. Output: (q, n_rot, N_half_image). Swap to (n_rot, q, ...).
    u_proj_q_first = jax.vmap(slice_one)(U_half)
    return jnp.swapaxes(u_proj_q_first, 0, 1)


# ---------------------------------------------------------------------------
# Per-batch preprocessing: full-image translate + convert to half-image
# ---------------------------------------------------------------------------


def _preprocess_batch_to_half(
    config: ForwardModelConfig,
    batch_full,
    translations,
    ctf_params,
    noise_variance_full,
):
    """Apply CTF/noise weighting, full-image phase shifts, and
    convert to half-image layout. Mirrors engine_v2 lines 196-214.

    Parameters
    ----------
    batch_full : (n_img, full_image_size) complex128
    translations : (n_trans, 2) float64
    ctf_params : per-image CTF params
    noise_variance_full : (full_image_size,) float64

    Returns
    -------
    shifted_half : (n_img, n_trans, half_image_size) complex128
        `S_t (CTF * y / σ²)` in half-image layout.
    ctf2_over_nv_half : (n_img, half_image_size) float64
        `(CTF² / σ²)` in half-image layout. Used for the H matrix
        and the projection-norm score term.
    ctf_half : (n_img, half_image_size) float64
        `CTF` in half-image layout (used for the residualized mean
        update later).
    """
    n_img = batch_full.shape[0]
    image_shape = config.image_shape

    CTF_full = config.compute_ctf(ctf_params, half_image=False)  # (n_img, full)
    processed = config.process_fn(batch_full, apply_image_mask=False)
    ctf_weighted = processed * CTF_full / noise_variance_full

    shifted = core.batch_trans_translate_images(
        ctf_weighted,
        jnp.repeat(translations[None], n_img, axis=0),
        image_shape,
    )  # (n_img, n_trans, full)
    shifted_half_flat = ftu.full_image_to_half_image(shifted.reshape(n_img * shifted.shape[1], -1), image_shape)
    shifted_half = shifted_half_flat.reshape(n_img, shifted.shape[1], -1)

    ctf2_over_nv = (CTF_full**2) / noise_variance_full
    ctf2_over_nv_half = ftu.full_image_to_half_image(ctf2_over_nv, image_shape)

    ctf_half = ftu.full_image_to_half_image(CTF_full, image_shape)

    return shifted_half, ctf2_over_nv_half, ctf_half


# ---------------------------------------------------------------------------
# Core math: H, b, posterior moments, log_scores
# ---------------------------------------------------------------------------


def _build_H_and_log_det(u_proj_half, s, ctf2_over_nv_half, weights_half):
    """Build `H[r, i, k, l] = diag(1/s) + Σ_k w(k) (CTF²/σ²)[i,k] u[r,k] conj(u[r,l])`
    and its log-determinant.

    Returns
    -------
    H : (n_rot, n_img, q, q) float64
    L : (n_rot, n_img, q, q) float64 — Cholesky factor (for downstream solve)
    log_det_H : (n_rot, n_img) float64
    """
    n_rot, q, _ = u_proj_half.shape
    # Outer products: (n_rot, q, q, N_half)
    u_outer = u_proj_half[:, :, None, :] * jnp.conj(u_proj_half)[:, None, :, :]
    # Pre-weight CTF²/σ² by w; transpose to put pixel axis last for the contraction.
    ctf2_w = ctf2_over_nv_half * weights_half[None, :]  # (n_img, N_half)
    # Contract pixel axis: (n_rot, q, q, N_half) @ (N_half, n_img) → (n_rot, q, q, n_img)
    H_no_diag = (u_outer @ ctf2_w.T).real
    H_no_diag = jnp.transpose(H_no_diag, (0, 3, 1, 2))  # (n_rot, n_img, q, q)
    diag_inv_s = jnp.diag(1.0 / s)  # (q, q)
    H = H_no_diag + diag_inv_s
    L = jnp.linalg.cholesky(H)
    log_det_H = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(L, axis1=-2, axis2=-1))), axis=-1)
    return H, L, log_det_H


def _build_b(
    mean_proj_half,
    u_proj_half,
    shifted_half,
    ctf2_over_nv_half,
    weights_half,
):
    """Build `b[r, i, k, t]` per the production convention so that the
    posterior mean is `m = H^{-1} b`.

    Mirrors `recovar/em/heterogeneity.py:127:compute_bLambdainvPU_terms`,
    but in half-image layout with rfft weights.

    Production formula (full layout):

        b1[r, i, k, t] = Re[ Σ_k conj(S_t (CTF y / σ²)[i,k]) · U[r,k,k] ]
        b2[r, i, k]    = Re[ Σ_k (CTF² / σ²)[i,k] · conj(μ[r,k]) · U[r,k,k] ]
        b              = -0.5 · 2 · (-b1 + b2[..., None])  =  b1 - b2

    With half-image weights, both contractions get a `w(k)` factor.

    Returns
    -------
    b : (n_rot, n_img, q, n_trans) float64
    """
    n_rot, q, _ = u_proj_half.shape
    n_img, n_trans, n_half = shifted_half.shape
    u_proj_T = jnp.swapaxes(u_proj_half, 1, 2)  # (n_rot, N_half, q)

    # b1: contract conj(shifted_half * w) with u_proj over the half-image pixel axis.
    shifted_flat = shifted_half.reshape(n_img * n_trans, n_half)
    shifted_conj_w = jnp.conj(shifted_flat) * weights_half[None, :]
    # (n_img*n_trans, N_half) @ (n_rot, N_half, q) → (n_rot, n_img*n_trans, q)
    b1 = (shifted_conj_w @ u_proj_T).real
    b1 = b1.reshape(n_rot, n_img, n_trans, q)
    b1 = jnp.transpose(b1, (0, 1, 3, 2))  # (n_rot, n_img, q, n_trans)

    # b2: (CTF²/σ² · w) @ (u · conj(μ)).
    ctf2_w = ctf2_over_nv_half * weights_half[None, :]  # (n_img, N_half)
    u_times_mu_conj = u_proj_T * jnp.conj(mean_proj_half)[..., None]  # (n_rot, N_half, q)
    b2 = (ctf2_w @ u_times_mu_conj).real  # (n_rot, n_img, q)

    return b1 - b2[..., None]


def _solve_posterior_moments(L, b):
    """Given Cholesky factor `L` of H and the b tensor, return
    `m = H^{-1} b` (= post_mean) and `bHb = b^T m`.

    L : (n_rot, n_img, q, q)
    b : (n_rot, n_img, q, n_trans)

    Returns
    -------
    m : (n_rot, n_img, q, n_trans)
    bHb : (n_rot, n_img, n_trans)
    """
    z = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    m = jax.scipy.linalg.solve_triangular(jnp.swapaxes(L, -1, -2), z, lower=False)
    bHb = jnp.sum(b * m, axis=-2)  # contract q axis; b real, m real → result real
    return m, bHb


def _homogeneous_residual_half(mean_proj_half, shifted_half, ctf2_over_nv_half, weights_half):
    """Compute the homogeneous score residual

        residual[i, r, t] = -2 Re < S_t (CTF y / σ²), proj_r >  +  ‖CTF · proj_r‖² / σ²

    in half-image layout with rfft weights. Per spec the constant
    `‖y‖² / σ²` is omitted (it cancels in image-wise normalization).

    Mirrors `recovar/em/dense_single_volume/engine_v2.py:241-282`.
    """
    n_img, n_trans, n_half = shifted_half.shape
    n_rot = mean_proj_half.shape[0]
    proj_w = mean_proj_half * weights_half[None, :]
    # Cross term
    shifted_flat = shifted_half.reshape(n_img * n_trans, n_half)
    cross = -2.0 * (jnp.conj(shifted_flat) @ proj_w.T).real  # (n_img*n_trans, n_rot)
    cross = cross.reshape(n_img, n_trans, n_rot)
    cross = jnp.swapaxes(cross, 1, 2)  # (n_img, n_rot, n_trans)
    # Norm term
    proj_abs2_w = (jnp.abs(mean_proj_half) ** 2) * weights_half[None, :]
    norms = ctf2_over_nv_half @ proj_abs2_w.T  # (n_img, n_rot)
    return cross + norms[..., None]


# ---------------------------------------------------------------------------
# Public kernel: posterior from pre-sliced half-image projections
# ---------------------------------------------------------------------------


def score_from_half_image_projections(
    mean_proj_half: jnp.ndarray,
    u_proj_half: jnp.ndarray,
    s: jnp.ndarray,
    shifted_half: jnp.ndarray,
    ctf2_over_nv_half: jnp.ndarray,
    weights_half: jnp.ndarray,
) -> PosteriorStats:
    """Posterior helper kernel that takes pre-sliced half-image inputs.

    Separating this from `score_and_posterior_moments_eqx` lets the
    brute-force and production-parity tests bypass `slice_volume`
    and the batch preprocessing pipeline, isolating the math from
    the slicing convention. The high-level entry point below
    delegates to this kernel.

    Parameters
    ----------
    mean_proj_half : (n_rot, half_image_size) complex128
        Mean projections in half-image (rfft) layout.
    u_proj_half : (n_rot, q, half_image_size) complex128
        PC projections in half-image layout.
    s : (q,) float64
        Latent prior variances. Strictly positive.
    shifted_half : (n_img, n_trans, half_image_size) complex128
        `S_t (CTF · y / σ²)` in half-image layout — translation
        already applied (in full image, then converted).
    ctf2_over_nv_half : (n_img, half_image_size) float64
        `CTF² / σ²` in half-image layout.
    weights_half : (half_image_size,) float64
        rfft Hermitian weights from `make_half_image_weights`.

    Returns
    -------
    PosteriorStats with shapes:
        log_scores : (n_img, n_rot, n_trans)
        log_resp   : (n_img, n_rot, n_trans)
        post_mean  : (n_img, n_rot, n_trans, q)
        post_Hinv  : (n_img, n_rot, q, q)
    """
    H, L, log_det_H = _build_H_and_log_det(u_proj_half, s, ctf2_over_nv_half, weights_half)
    b = _build_b(mean_proj_half, u_proj_half, shifted_half, ctf2_over_nv_half, weights_half)
    m, bHb = _solve_posterior_moments(L, b)
    homog = _homogeneous_residual_half(mean_proj_half, shifted_half, ctf2_over_nv_half, weights_half)

    # Heterogeneous correction = bHb - log_det_H, transposed to (n_img, n_rot, *)
    bHb_per = jnp.transpose(bHb, (1, 0, 2))
    log_det_H_per = jnp.transpose(log_det_H, (1, 0))
    het_correction = bHb_per - log_det_H_per[..., None]

    residual = homog - het_correction
    log_scores = -0.5 * residual

    n_img = log_scores.shape[0]
    log_scores_flat = log_scores.reshape(n_img, -1)
    log_norm = jax.scipy.special.logsumexp(log_scores_flat, axis=-1, keepdims=True)
    log_resp = (log_scores_flat - log_norm).reshape(log_scores.shape)

    post_mean = jnp.transpose(m, (1, 0, 3, 2))  # (n_img, n_rot, n_trans, q)
    Hinv = jnp.linalg.inv(H)
    post_Hinv = jnp.transpose(Hinv, (1, 0, 2, 3))  # (n_img, n_rot, q, q)

    return PosteriorStats(
        log_scores=log_scores,
        log_resp=log_resp,
        post_mean=post_mean,
        post_Hinv=post_Hinv,
    )


# ---------------------------------------------------------------------------
# Public entry point: fully-materialized posterior (test-only for real workloads)
# ---------------------------------------------------------------------------


def score_and_posterior_moments_eqx(
    config: ForwardModelConfig,
    mu_half: jnp.ndarray,
    U_half: jnp.ndarray,
    s: jnp.ndarray,
    batch_full: jnp.ndarray,
    rotations: jnp.ndarray,
    translations: jnp.ndarray,
    ctf_params: jnp.ndarray,
    noise_variance_full: jnp.ndarray,
) -> PosteriorStats:
    """Compute the fully-materialized posterior tensors for one batch.

    Use only for tiny CPU tests / Stage 0A correctness checks. Real
    workloads must use `iter_posterior_blocks`.
    """
    if mu_half.dtype != jnp.complex128:
        raise TypeError(f"mu_half must be complex128, got {mu_half.dtype}")
    if U_half.dtype != jnp.complex128:
        raise TypeError(f"U_half must be complex128, got {U_half.dtype}")
    if s.dtype != jnp.float64:
        raise TypeError(f"s must be float64, got {s.dtype}")
    if batch_full.dtype != jnp.complex128:
        raise TypeError(f"batch_full must be complex128, got {batch_full.dtype}")
    if noise_variance_full.dtype != jnp.float64:
        raise TypeError(f"noise_variance_full must be float64, got {noise_variance_full.dtype}")

    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half = make_half_image_weights(image_shape)

    # 1. Slice mu and U into half-image projections
    mean_proj_half = _slice_mu_half(mu_half, rotations, image_shape, volume_shape)
    u_proj_half = _slice_U_half(U_half, rotations, image_shape, volume_shape)

    # Force float64 dtype on projections (slice_volume may downcast)
    mean_proj_half = mean_proj_half.astype(jnp.complex128)
    u_proj_half = u_proj_half.astype(jnp.complex128)

    # 2. Preprocess batch
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_variance_full
    )
    shifted_half = shifted_half.astype(jnp.complex128)
    ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float64)

    # 3. Delegate to the kernel
    return score_from_half_image_projections(
        mean_proj_half=mean_proj_half,
        u_proj_half=u_proj_half,
        s=s,
        shifted_half=shifted_half,
        ctf2_over_nv_half=ctf2_over_nv_half,
        weights_half=weights_half,
    )


# ---------------------------------------------------------------------------
# Streaming block iterator (for real workloads)
# ---------------------------------------------------------------------------


def iter_posterior_blocks(
    config: ForwardModelConfig,
    mu_half: jnp.ndarray,
    U_half: jnp.ndarray,
    s: jnp.ndarray,
    batch_full: jnp.ndarray,
    rotations: jnp.ndarray,
    translations: jnp.ndarray,
    ctf_params: jnp.ndarray,
    noise_variance_full: jnp.ndarray,
    *,
    rot_block_size: int,
    trans_block_size: int,
) -> Iterator[PosteriorBlock]:
    """Yield posterior blocks of size `(rot_block, trans_block)`.

    For real workloads. The block iterator never materializes the
    full `(n_img, n_rot, n_trans, q)` `post_mean` tensor; the M-step
    accumulators consume blocks and add their contributions into
    running `(volume_size, q)` arrays.

    For v0 we implement this as a thin Python loop around the same
    `score_and_posterior_moments_eqx` math, sliced over rotation
    and translation blocks. A future optimization can JIT-compile
    a single-block kernel and avoid the per-block recompilation.
    """
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    for r0 in range(0, n_rot, rot_block_size):
        r1 = min(r0 + rot_block_size, n_rot)
        rot_block = rotations[r0:r1]
        for t0 in range(0, n_trans, trans_block_size):
            t1 = min(t0 + trans_block_size, n_trans)
            trans_block = translations[t0:t1]
            stats = score_and_posterior_moments_eqx(
                config,
                mu_half,
                U_half,
                s,
                batch_full,
                rot_block,
                trans_block,
                ctf_params,
                noise_variance_full,
            )
            yield PosteriorBlock(
                rot_slice=slice(r0, r1),
                trans_slice=slice(t0, t1),
                log_scores=stats.log_scores,
                post_mean=stats.post_mean,
                post_Hinv=stats.post_Hinv,
            )
