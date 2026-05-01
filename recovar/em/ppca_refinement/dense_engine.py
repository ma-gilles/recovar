"""Dense pose-marginalized PPCA E-step (Milestone 4).

Two-pass blockwise engine over (image_batch B × rotation R × translation T).
Pass 1 computes the per-image evidence ``logZ``, the best (rotation, shift)
pose, and the maximum posterior probability ``pmax``. Pass 2 recomputes
the per-(b, r, t) score, builds posteriors ``γ = exp(score - logZ)``, and
accumulates the augmented moments ``α_aug`` and ``G_aug_tri`` weighted by
``γ`` so the M-step driver can backproject them into ``rhs`` and
``lhs_tri`` half-volumes.

This module is **engine-only** — it does not load datasets, build
samplers, or do half-volume backprojection. Those concerns live in
``recovar.em.ppca_refinement.iterations`` (M5+). Inputs are pre-built
JAX tensors; outputs are per-image augmented stats. This decoupling is
deliberate: the engine becomes brute-force testable in isolation against
the M1 per-pose function (see
``tests/unit/ppca_refinement/test_dense_engine.py``).

The score function used is exactly
``recovar.ppca.pose_marginal.compute_ppca_pose_scores_and_moments_no_contrast``
(M1) — dense and sparse engines must call the same score function (CLAUDE.md
non-negotiable #8). M8 will add a contrast-aware sibling.

Memory invariants (CLAUDE.md §8.6):

  Allowed inside-block tensors:
    score       [B, T, R]
    alpha_aug   [B, T, R, P]
    G_aug_tri   [B, T, R, tri(P)]
    K_aug       [B, R, P, P]

  Forbidden:
    [N_images, N_rot, N_trans, *]    — never materialize global posterior

The engine is JIT-compiled per (B, R, T, F, P) shape tuple; static-shape
JIT means callers should bucket image batches and rotation blocks to a
small set of (B, R) sizes, otherwise XLA recompiles. M5's driver handles
the bucketing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from recovar.ppca.pose_marginal import (
    compute_ppca_pose_scores_and_moments_no_contrast,
)

__all__ = [
    "DenseImageStats",
    "PosteriorDiagnostics",
    "dense_pose_ppca_E_step_blocked",
    "fused_dense_pose_ppca_block",
]


class DenseImageStats(NamedTuple):
    """Per-image augmented moments accumulated over (R, T) in pass 2.

    These are *image-level* — the M5 driver subsequently backprojects them
    to half-volume ``rhs`` and ``lhs_tri`` via the existing
    half-spectrum-adjoint helpers in
    ``recovar.em.dense_single_volume.helpers.backprojection``.
    """

    alpha_aug_acc: jax.Array  # [B, P] complex64
    G_aug_tri_acc: jax.Array  # [B, tri(P)] complex64
    log_evidence: jax.Array  # [B] real32 — Σ_b logZ_b is .sum()


class PosteriorDiagnostics(NamedTuple):
    logZ: jax.Array  # [B] real32
    pmax: jax.Array  # [B] real32
    best_rotation_idx: jax.Array  # [B] int32
    best_translation_idx: jax.Array  # [B] int32
    n_significant_per_image: jax.Array  # [B] int32 — count of (r, t) with γ > τ_sig
    omitted_log_mass: jax.Array  # [B] real32 — log(1 - Σ γ inside support); 0 here (dense)


def _per_pose_stats_block(Y1, proj_aug, ctf2_over_noise, y_norm):
    """Compute per-(B, T, R) sufficient stats and per-(B, R) second-order
    Gram. JIT-friendly. No leading-batch broadcasting outside the engine.

    Y1:               [B, T, F] complex64 — pre-shifted CTF-weighted whitened image (= C·y·e^{−2πik·t}/σ²)
    proj_aug:         [R, P, F] complex64 — augmented templates (without per-image weights)
    ctf2_over_noise:  [B, F] real32       — C² / σ² per-image, per-pixel
    y_norm:           [B] real32          — sum_f |y_b|² / σ²_b   (constant in r, t)
    """
    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape

    # K_aug[b, r, p, q] = sum_f (C²/σ²)[b, f] · conj(proj_aug)[r, p, f] · proj_aug[r, q, f]
    # Translation-independent. Hermitian in (p, q).
    K_aug = jnp.einsum(
        "bf, rpf, rqf -> brpq",
        ctf2_over_noise.astype(proj_aug.dtype),
        jnp.conj(proj_aug),
        proj_aug,
    )  # [B, R, P, P] complex64
    nu_mm = K_aug[..., 0, 0].real  # [B, R]
    h_zm = K_aug[..., 1:, 0]  # [B, R, q]
    Hzz = K_aug[..., 1:, 1:]  # [B, R, q, q]

    # D[b, t, r, p] = sum_f conj(Y1)[b, t, f] · proj_aug[r, p, f]
    # Yields t_mx (p=0) and g_zx (p>=1) at every (B, T, R) hypothesis.
    D = jnp.einsum(
        "btf, rpf -> btrp",
        jnp.conj(Y1),
        proj_aug,
    )  # [B, T, R, P] complex
    t_mx = D[..., 0].real  # [B, T, R]
    g_zx = D[..., 1:]  # [B, T, R, q]

    # Broadcast translation-independent stats over T.
    h_zm_btr = jnp.broadcast_to(h_zm[:, None, :, :], (B, T, R, h_zm.shape[-1]))  # [B, T, R, q]
    Hzz_btr = jnp.broadcast_to(Hzz[:, None, :, :, :], (B, T, R, Hzz.shape[-2], Hzz.shape[-1]))  # [B, T, R, q, q]
    nu_mm_btr = jnp.broadcast_to(nu_mm[:, None, :], (B, T, R))  # [B, T, R]
    y_norm_btr = jnp.broadcast_to(y_norm[:, None, None], (B, T, R))  # [B, T, R]

    return y_norm_btr, t_mx, nu_mm_btr, g_zx, h_zm_btr, Hzz_btr


def dense_pose_ppca_E_step_blocked(
    Y1: jax.Array,
    proj_aug: jax.Array,
    ctf2_over_noise: jax.Array,
    y_norm: jax.Array,
    pose_log_prior: jax.Array | None = None,
    *,
    significance_threshold: float = 1e-3,
):
    """Two-pass dense pose-marginalized E-step on a (B, R, T, F) block.

    Parameters
    ----------
    Y1:
        ``[B, T, F]`` complex64 — pre-shifted CTF-weighted whitened images,
        i.e. ``Y1[b, t, f] = (C_b · y_b · phase_t)[f] / σ²_b[f]``. This
        convention matches CLAUDE.md §8.3 and is consistent with the
        legacy ``recovar.ppca.ppca._e_step_half_inner`` weighting.
    proj_aug:
        ``[R, P, F]`` complex64 with ``P = q + 1`` augmented components
        ``[μ, W₁, …, W_q]``. Pure projections of the augmented templates;
        no per-image CTF or noise weighting baked in (those come via
        ``ctf2_over_noise`` and the ``Y1`` weighting).
    ctf2_over_noise:
        ``[B, F]`` real32 — ``C² / σ²`` per pixel.
    y_norm:
        ``[B]`` real32 — ``Σ_f |y_b[f]|² / σ²_b[f]``, pose- and
        translation-independent per image.
    pose_log_prior:
        Optional ``[B, R, T]`` real32 — ``log π_irt``. None ⇒ uniform 0.
    significance_threshold:
        ``γ_irt > τ_sig`` counts as significant for the per-image
        ``n_significant`` diagnostic. Same threshold convention as the
        k-class engine.

    Returns
    -------
    image_stats:
        :class:`DenseImageStats` with ``α_aug_acc`` and ``G_aug_tri_acc``
        per image. The M5 driver uses these to drive backprojection into
        half-volume ``rhs`` and ``lhs_tri``.
    diagnostics:
        :class:`PosteriorDiagnostics` with ``logZ``, ``pmax``,
        argmax (rotation, translation), ``n_significant`` per image, and
        an ``omitted_log_mass`` field that is identically zero for the
        dense engine (full support).
    """
    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    q = P - 1
    if proj_aug.shape[-1] != F:
        raise ValueError(f"proj_aug last dim {proj_aug.shape[-1]} != F={F}")
    if ctf2_over_noise.shape != (B, F):
        raise ValueError(f"ctf2_over_noise shape {ctf2_over_noise.shape} != (B={B}, F={F})")
    if y_norm.shape != (B,):
        raise ValueError(f"y_norm shape {y_norm.shape} != (B={B},)")

    # ---- Pass 1: per-(b, t, r) score, logZ, best pose ----------------------
    yn, tm, num, g, hz, Hz = _per_pose_stats_block(Y1, proj_aug, ctf2_over_noise, y_norm)
    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=False,
    )  # [B, T, R]
    if pose_log_prior is not None:
        if pose_log_prior.shape != (B, R, T):
            raise ValueError(f"pose_log_prior shape {pose_log_prior.shape} != (B={B}, R={R}, T={T})")
        # Reshape from [B, R, T] to [B, T, R] to match score axis order.
        score = score + jnp.swapaxes(pose_log_prior, -1, -2)

    score_flat = score.reshape(B, T * R)  # [B, T·R]
    logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)  # [B]

    best_flat = jnp.argmax(score_flat, axis=-1)  # [B]
    best_t = (best_flat // R).astype(jnp.int32)
    best_r = (best_flat % R).astype(jnp.int32)

    # γ for diagnostics-only at this stage.
    gamma_pass1 = jnp.exp(score - logZ[:, None, None])  # [B, T, R]
    pmax = jnp.max(gamma_pass1.reshape(B, T * R), axis=-1)  # [B]
    n_sig = jnp.sum(gamma_pass1 > significance_threshold, axis=(-1, -2)).astype(jnp.int32)
    omitted_log_mass = jnp.zeros((B,), dtype=jnp.float32)  # dense engine: full support

    # ---- Pass 2: recompute and accumulate moments --------------------------
    if q == 0:
        # No latent — α_aug ≡ [1] and G_aug_tri ≡ [1] regardless of (r, t).
        # The accumulator collapses to "sum of γ" = 1 per image times [1, 1].
        ones_complex = jnp.ones((B, 1), dtype=jnp.complex64)
        image_stats = DenseImageStats(
            alpha_aug_acc=ones_complex,
            G_aug_tri_acc=ones_complex,
            log_evidence=logZ,
        )
        return image_stats, PosteriorDiagnostics(
            logZ=logZ,
            pmax=pmax,
            best_rotation_idx=best_r,
            best_translation_idx=best_t,
            n_significant_per_image=n_sig,
            omitted_log_mass=omitted_log_mass,
        )

    score2, alpha_brt, G_tri_brt = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=True,
    )  # alpha [B,T,R,P], G [B,T,R,tri(P)]
    if pose_log_prior is not None:
        score2 = score2 + jnp.swapaxes(pose_log_prior, -1, -2)
    gamma = jnp.exp(score2 - logZ[:, None, None])  # [B, T, R]

    # Accumulate γ·α and γ·G_tri across (T, R) per image.
    # Broadcast γ over the trailing P / tri(P) axis.
    alpha_aug_acc = jnp.einsum(
        "btr, btrp -> bp",
        gamma.astype(alpha_brt.dtype),
        alpha_brt,
    )  # [B, P] complex
    G_aug_tri_acc = jnp.einsum(
        "btr, btrk -> bk",
        gamma.astype(G_tri_brt.dtype),
        G_tri_brt,
    )  # [B, tri(P)] complex

    image_stats = DenseImageStats(
        alpha_aug_acc=alpha_aug_acc,
        G_aug_tri_acc=G_aug_tri_acc,
        log_evidence=logZ,
    )
    return image_stats, PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=best_r,
        best_translation_idx=best_t,
        n_significant_per_image=n_sig,
        omitted_log_mass=omitted_log_mass,
    )


# ===========================================================================
# Fused production engine (Phase A.1 — M10 follow-up)
# ===========================================================================
#
# The non-fused ``dense_pose_ppca_E_step_blocked`` returns image-level
# aggregates that are too aggregated for proper per-rotation backprojection
# (it sums γα across both R and T inside a block). The fused production
# engine interleaves pass-2 score normalization with per-rotation
# backprojection so γα at each rotation r is consumed immediately and never
# materialized across all R rotations.
#
# Memory invariant: per-rotation tensors stay in [B, P, F] / [B, tri(P), F]
# scope; no [B, R, P, F] or [B, R, tri(P), F] global accumulator.


def fused_dense_pose_ppca_block(
    Y1,
    proj_aug,
    ctf2_over_noise,
    y_norm,
    rotations_block,
    image_shape,
    volume_shape,
    rhs_volume,
    lhs_tri_volume,
    pose_log_prior=None,
    *,
    significance_threshold: float = 1e-3,
    disc_type_backproject: str = "linear_interp",
):
    """One pass of the fused dense engine: pass-1 (logZ + best pose) +
    pass-2 (γ + per-rotation backprojection).

    Parameters
    ----------
    Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior:
        Same as :func:`dense_pose_ppca_E_step_blocked`.
    rotations_block:
        ``[R, 3, 3]`` real32. The per-rotation matrices that produced
        ``proj_aug``. Required so backprojection happens at the matching
        rotation.
    image_shape, volume_shape:
        Forwarded to ``batch_adjoint_slice_volume_half``.
    rhs_volume:
        ``[P, half_vol]`` complex64. Mutable RHS accumulator (this
        function returns a new array; caller assigns).
    lhs_tri_volume:
        ``[tri(P), half_vol]`` real32. Mutable LHS-tri accumulator.
    disc_type_backproject:
        Discretization for backprojection. Must be one of ``"linear_interp"``
        or ``"nearest"`` — the slicing kernel forbids cubic backprojection.

    Returns
    -------
    rhs_volume_new, lhs_tri_volume_new : updated accumulators.
    diagnostics : :class:`PosteriorDiagnostics`.

    Note on the LHS real-projection
    -------------------------------
    The augmented Gram ``G_aug`` is complex-Hermitian. The legacy
    ``recovar.ppca.ppca._pcg_hard_mstep`` accepts a REAL ``lhs_tri``;
    the upper-triangle pack via ``unpack_tri_to_full`` produces a
    *symmetric* (not Hermitian) operator. To match the legacy contract
    we project ``γ · G_aug_tri`` to its real part before backprojecting.
    For half-spectrum images with full-spec Parseval weights the
    imaginary cross-component parts approximately cancel — same
    approximation the legacy E-step makes.
    """
    # Lazy import to avoid pulling the dense_single_volume backprojection
    # helpers when only the test-only ``dense_pose_ppca_E_step_blocked``
    # is used.
    from recovar.em.dense_single_volume.helpers.backprojection import (
        batch_adjoint_slice_volume_half,
    )

    B, T, F = Y1.shape
    R, P, _ = proj_aug.shape
    q = P - 1
    if rotations_block.shape != (R, 3, 3):
        raise ValueError(f"rotations_block shape {rotations_block.shape} != ({R}, 3, 3)")
    if rhs_volume.shape[0] != P:
        raise ValueError(f"rhs_volume shape {rhs_volume.shape} not compatible with P={P}")
    tri_size = P * (P + 1) // 2
    if lhs_tri_volume.shape[0] != tri_size:
        raise ValueError(f"lhs_tri_volume shape {lhs_tri_volume.shape} not compatible with tri({P})={tri_size}")

    # ---- Pass 1 -------------------------------------------------------------
    yn, tm, num, g, hz, Hz = _per_pose_stats_block(Y1, proj_aug, ctf2_over_noise, y_norm)
    score, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=False,
    )
    if pose_log_prior is not None:
        if pose_log_prior.shape != (B, R, T):
            raise ValueError(f"pose_log_prior shape {pose_log_prior.shape} != (B={B}, R={R}, T={T})")
        score = score + jnp.swapaxes(pose_log_prior, -1, -2)

    score_flat = score.reshape(B, T * R)
    logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)
    best_flat = jnp.argmax(score_flat, axis=-1)
    best_t = (best_flat // R).astype(jnp.int32)
    best_r = (best_flat % R).astype(jnp.int32)

    gamma_for_diag = jnp.exp(score - logZ[:, None, None])
    pmax = jnp.max(gamma_for_diag.reshape(B, T * R), axis=-1)
    n_sig = jnp.sum(gamma_for_diag > significance_threshold, axis=(-1, -2)).astype(jnp.int32)
    omitted_log_mass = jnp.zeros((B,), dtype=jnp.float32)

    # ---- Pass 2 + per-rotation backprojection -------------------------------
    score2, alpha, G_tri = compute_ppca_pose_scores_and_moments_no_contrast(
        yn,
        tm,
        num,
        g,
        hz,
        Hz,
        return_moments=True,
    )
    if pose_log_prior is not None:
        score2 = score2 + jnp.swapaxes(pose_log_prior, -1, -2)
    gamma = jnp.exp(score2 - logZ[:, None, None])  # [B, T, R]

    rhs_dtype = rhs_volume.dtype
    lhs_dtype = lhs_tri_volume.dtype
    ctf2_c = ctf2_over_noise.astype(rhs_dtype)

    # Loop over rotations in the block, applying backprojection immediately.
    # JAX-static loop via Python (ok for production block sizes ~ 10² rotations
    # per block; the legacy K-class engine uses the same pattern).
    for r_idx in range(R):
        gamma_r = gamma[:, :, r_idx]  # [B, T]
        alpha_r = alpha[:, :, r_idx, :]  # [B, T, P]
        G_tri_r = G_tri[:, :, r_idx, :]  # [B, T, tri(P)]
        rotation = rotations_block[r_idx]  # [3, 3]
        rotations_per_image = jnp.broadcast_to(rotation[None, :, :], (B, 3, 3))

        # RHS: Z_rp[B, P, F] = sum_t γ_brt α_brt,p · Y1[b, t]
        Z_rp = jnp.einsum(
            "bt, btp, btf -> bpf",
            gamma_r.astype(rhs_dtype),
            alpha_r,
            Y1,
        )  # [B, P, F]
        # batch_adjoint_slice_volume_half wants [n_volumes, n_images, half_F]
        # with one accumulator per volume. n_volumes = P, n_images = B.
        Z_pbf = jnp.transpose(Z_rp, (1, 0, 2)).astype(rhs_dtype)  # [P, B, F]
        rhs_volume = batch_adjoint_slice_volume_half(
            Z_pbf,
            rotations_per_image,
            rhs_volume,
            image_shape,
            volume_shape,
            disc_type_backproject,
            half_image=True,
            half_volume=True,
        )

        # LHS: w_rs[B, tri(P)] = sum_t γ_brt G_aug_tri_brt,rs
        # Project to real per the legacy lhs_tri convention.
        w_rs = jnp.einsum(
            "bt, btk -> bk",
            gamma_r,
            G_tri_r,
        ).real.astype(lhs_dtype)  # [B, tri(P)]
        # weighted_ctf2[B, tri(P), F] = w_rs[..., None] * ctf2_over_noise[:, None, :]
        weighted_ctf2 = w_rs[:, :, None] * ctf2_over_noise[:, None, :]  # [B, tri(P), F]
        weighted_ctf2_sbf = jnp.transpose(weighted_ctf2, (1, 0, 2))  # [tri(P), B, F]
        lhs_tri_volume = batch_adjoint_slice_volume_half(
            weighted_ctf2_sbf.astype(lhs_dtype),
            rotations_per_image,
            lhs_tri_volume,
            image_shape,
            volume_shape,
            disc_type_backproject,
            half_image=True,
            half_volume=True,
        )

    diagnostics = PosteriorDiagnostics(
        logZ=logZ,
        pmax=pmax,
        best_rotation_idx=best_r,
        best_translation_idx=best_t,
        n_significant_per_image=n_sig,
        omitted_log_mass=omitted_log_mass,
    )
    return rhs_volume, lhs_tri_volume, diagnostics
