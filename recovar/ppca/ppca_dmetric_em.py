"""D-metric U/B generalized EM for PPCA refit.

Model
-----
    x_i = mu + U alpha_i,   alpha_i ~ N(0, B),   U^T D U = I_q

where D is a fixed positive-definite metric, here a per-voxel diagonal in
Fourier space obtained from the shell prior (without the /npc division used
by the standard PPCA prior).

Two B-update variants:
    * "noreg":  B = (1/n) sum_i T_i + eps * I
    * "reg":    B = (sum_i T_i + kappa B_0) / (n + kappa)
              with B_0 fixed at initialization from U_0^T Gamma U_0 (Gamma = D^{-1}).

U-update: Euclidean gradient descent on Q(U) = sum_i [tr(U^T M_i U T_i)
- 2 b_i^T U m_i], with backtracking line search and a D-metric QR retraction
after every accepted step.  Gauge fix at the end of each outer iteration:
diagonalize B and rotate U, m into its eigenbasis.

Reuses heavily from ``recovar.ppca.ppca_refit`` and
``recovar.ppca.ppca_refit_iterative``.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.ppca import prior_estimation as pe
from recovar.ppca.ppca_refit import (
    _forward_model_from_map,
    _compute_Gi_hi_batch,
    batch_over_vol_slice_volume,
    compute_per_image_Gi_hi,
    compute_embeddings_from_UB,
    create_postprocessed_result_dir,
)
from recovar.ppca.ppca import (
    batch_over_vol_adjoint_slice_volume,
    batch_over_vol_adjoint_slice_volume_half,
    batch_over_vol_slice_volume_half,
    _half_slice_volume,
    _tri_size,
)
from recovar.reconstruction.pcg_mean import pcg_mstep

logger = logging.getLogger(__name__)


# =============================================================================
# Shell prior -> per-voxel metric D (Fourier diagonal)
# =============================================================================

def compute_D_fourier_diag(dataset, mean_fourier, npc, volume_shape, batch_size):
    """Compute per-voxel inverse-variance metric D from the shell prior.

    The standard PPCA shell prior divides the shell variance by ``npc``
    (see ``prior_estimation.make_radial_prior_from_shell_total``).  For D we
    want the *un-divided* variance, so we undo that division here.

    Returns
    -------
    D_diag : (vol_size,) float64 array
        Per-voxel Fourier-space precision = 1 / shell_variance.
    Gamma_diag : (vol_size,) float64 array
        Per-voxel Fourier-space variance  = D_diag^{-1}  (= shell_variance).
    info : dict
        Bookkeeping (shell stats, etc.).
    """
    prior_info = pe.estimate_hybrid_shell_prior_from_data(
        dataset, mean_fourier, npc, volume_shape, batch_size,
    )
    W_prior = np.asarray(prior_info["W_prior"])  # (vol_size, npc) — all columns identical
    # Undo the /npc that make_radial_prior_from_shell_total performed.
    shell_var_per_voxel = np.asarray(W_prior[:, 0], dtype=np.float64) * float(npc)
    shell_var_per_voxel = np.maximum(shell_var_per_voxel, 1e-12)
    D_diag = 1.0 / shell_var_per_voxel
    Gamma_diag = shell_var_per_voxel
    logger.info(
        "D metric: shell variance min=%.3e, max=%.3e, median=%.3e",
        shell_var_per_voxel.min(), shell_var_per_voxel.max(),
        np.median(shell_var_per_voxel),
    )
    return D_diag, Gamma_diag, prior_info


# =============================================================================
# Initialization of (U, B) from PPCA W
# =============================================================================

def init_UB_from_ppca(po, zdim, D_diag, Gamma_diag):
    """Initialize U and B from a PPCA result.

    Starting point:
        W_0[v, j] = U_ppca_fourier[v, j] * sqrt(s_ppca[j])  (Fourier space)
        B_0 = W_0^T diag(Gamma) W_0
        U_0 = W_0 (B_0)^{-1/2}

    Then U_0^T D U_0 = I_q by construction (assuming B_0 PSD).

    Returns
    -------
    U_fourier : (vol_size, q) complex128
    B0 : (q, q) float64
    U_real_init : (q, *volume_shape) float32 (with volume mask + gridding)
    """
    volume_shape = tuple(po.params["volume_shape"])
    vol_size = int(np.prod(volume_shape))

    # Load real-space U and apply baseline conventions (focus mask + gridding correction).
    # The baseline PPCA EM uses focus_mask (= mask.mrc, soft cosine), NOT the dilated mask.
    u_real = po.get_u_real(zdim).astype(np.float32)  # (q, *vol_shape)
    try:
        focus_mask = utils.load_mrc(po.paths.mask_volume)
        for k in range(zdim):
            u_real[k] = u_real[k] * focus_mask
        logger.info("init_UB_from_ppca: applied focus mask to U")
    except Exception as e:
        logger.warning("init_UB_from_ppca: no focus mask applied (%s)", e)
    try:
        from recovar.reconstruction.relion_functions import griddingCorrect_square
        for k in range(zdim):
            u_real[k] = np.asarray(
                griddingCorrect_square(jnp.array(u_real[k]), volume_shape[0], 1, order=1)[0]
            )
        logger.info("init_UB_from_ppca: applied gridding correction to U")
    except Exception as e:
        logger.warning("init_UB_from_ppca: no gridding correction (%s)", e)

    # Convert to Fourier
    U_fourier = np.zeros((vol_size, zdim), dtype=np.complex128)
    for j in range(zdim):
        U_fourier[:, j] = np.asarray(ftu.get_dft3(u_real[j])).reshape(-1)

    # W = U * sqrt(s) in Fourier space
    s_ppca = np.asarray(po.params["s"][:zdim], dtype=np.float64)
    s_ppca = np.maximum(s_ppca, 0.0)
    W_fourier = U_fourier * np.sqrt(s_ppca)[None, :]

    # B_0 = W^T Gamma W  — real symmetric since U is Hermitian-paired in Fourier.
    # Compute in float64 via W^H diag(Gamma) W (taking the real part).
    weighted = W_fourier * Gamma_diag[:, None]
    B0 = (np.conj(W_fourier).T @ weighted).real.astype(np.float64)
    B0 = 0.5 * (B0 + B0.T)

    # Sanity check: B0 should be PSD.
    eigvals_B0 = np.linalg.eigvalsh(B0)
    logger.info(
        "B0 eigenvalues (W^T Gamma W): min=%.3e max=%.3e",
        eigvals_B0.min(), eigvals_B0.max(),
    )
    B0 = B0 + max(0.0, -eigvals_B0.min() + 1e-8) * np.eye(zdim)

    # U_0 = W B_0^{-1/2}
    w_B0, V_B0 = np.linalg.eigh(B0)
    w_B0 = np.maximum(w_B0, 1e-12)
    B0_inv_sqrt = V_B0 @ np.diag(1.0 / np.sqrt(w_B0)) @ V_B0.T
    U_fourier_new = W_fourier @ B0_inv_sqrt

    return U_fourier_new.astype(np.complex128), B0, u_real


# =============================================================================
# E-step pass: compute G, h AND (optionally) b_i m_i^T backprojection
# =============================================================================

@jax.jit
def _gradient_terms_batch(
    PU, CTF, centered_images, T_batch, m_batch,
):
    """Compute the two per-batch backprojection tensors for the U-gradient.

    Returns pre-backprojection arrays shaped (batch, image_size, q):
        bp_term1 = CTF * (T_batch @ PU)              (for sum_i M_i U T_i)
        bp_term2 = CTF * centered * m_batch^T        (for sum_i b_i m_i^T)
    """
    T_PU = jnp.matmul(T_batch, PU)                       # (batch, q, img)
    bp_term1 = CTF[:, None, :] * T_PU                    # (batch, q, img)
    bp_term1 = bp_term1.transpose(0, 2, 1)               # (batch, img, q)

    bp_term2 = CTF[:, :, None] * centered_images[:, :, None] * m_batch[:, None, :]
    return bp_term1, bp_term2


def _compute_Gh_and_maybe_grad(
    dataset,
    mean_jax,
    U_jax,
    volume_shape,
    image_shape,
    voxel_size,
    batch_size,
    apply_image_mask,
    disc_type,
    disc_type_mean,
    need_grad,
    B_inv=None,
    eps_B=1e-12,
    return_ll=False,
):
    """Single dataset pass.

    Always computes G_all (n, q, q) and h_all (n, q).
    If ``need_grad=True``, also does an on-the-fly E-step with ``B_inv`` to
    produce (m_i, T_i) for each image, and accumulates

        grad_U = sum_i [ M_i U T_i  -  b_i m_i^T ]

    in Fourier space (vol_size, q).  Avoids a second dataset pass.

    Also optionally tracks the observed data log-likelihood (for line search).

    Returns
    -------
    G_all : (n, q, q) float64
    h_all : (n, q) float64
    m_all : (n, q) float64 or None
    T_all : (n, q, q) float64 or None
    grad_U : (vol_size, q) complex128 or None
    ll : float (only meaningful if ``return_ll``)
    """
    q = U_jax.shape[1]
    vol_size = int(np.prod(volume_shape))
    n_images = dataset.n_images

    G_all = np.zeros((n_images, q, q), dtype=np.float64)
    h_all = np.zeros((n_images, q), dtype=np.float64)
    m_all = np.zeros((n_images, q), dtype=np.float64) if need_grad else None
    T_all = np.zeros((n_images, q, q), dtype=np.float64) if need_grad else None

    grad_U = jnp.zeros((vol_size, q), dtype=jnp.complex128) if need_grad else None

    if need_grad:
        assert B_inv is not None
        B_inv_jax = jnp.array(B_inv, dtype=jnp.float64)
        eye_q = jnp.eye(q, dtype=jnp.float64)

    ll_sum = 0.0
    n_processed = 0
    t0 = time.time()

    for (
        batch, rotation_matrices, translations, ctf_params,
        _noise_variance, _particle_indices, image_indices,
    ) in dataset.iter_batches(
        batch_size,
        by_image=not getattr(dataset, "tilt_series_flag", False),
    ):
        images = dataset.process_images(batch, apply_image_mask=apply_image_mask)
        noise_variance = dataset.noise.get(image_indices)

        images = core.translate_images(
            images, translations, image_shape,
        ) / jnp.sqrt(noise_variance)

        CTF = dataset.ctf_evaluator(
            ctf_params, image_shape, voxel_size,
        ) / jnp.sqrt(noise_variance)

        projected_mean = _forward_model_from_map(
            mean_jax, ctf_params, rotation_matrices, image_shape, volume_shape,
            voxel_size, dataset.ctf_evaluator, disc_type_mean,
        ) / jnp.sqrt(noise_variance)

        centered_images = images - projected_mean

        PU = batch_over_vol_slice_volume(
            U_jax, rotation_matrices, image_shape, volume_shape, disc_type,
        )
        PU = PU * CTF[:, None, :]

        G_batch, h_batch = _compute_Gi_hi_batch(PU, centered_images)  # (b,q,q), (b,q)

        indices = np.asarray(image_indices)
        G_all[indices] = np.asarray(G_batch, dtype=np.float64)
        h_all[indices] = np.asarray(h_batch, dtype=np.float64)

        if need_grad:
            # E-step (batch) for current B
            M = B_inv_jax[None] + G_batch.astype(jnp.float64)          # (b,q,q)
            L = jnp.linalg.cholesky(M)
            P = jax.scipy.linalg.cho_solve((L, True), jnp.broadcast_to(eye_q, M.shape))
            h64 = h_batch.astype(jnp.float64)
            m = jax.scipy.linalg.cho_solve((L, True), h64[:, :, None]).squeeze(-1)  # (b,q)
            T = P + m[:, :, None] * m[:, None, :]                     # (b,q,q)

            m_all[indices] = np.asarray(m, dtype=np.float64)
            T_all[indices] = np.asarray(T, dtype=np.float64)

            if return_ll:
                logdet_M = 2.0 * jnp.sum(
                    jnp.log(jnp.abs(jnp.diagonal(L, axis1=1, axis2=2))), axis=-1,
                )
                quad = jnp.sum(h64 * m, axis=-1)
                r2 = jnp.sum(jnp.abs(centered_images) ** 2, axis=-1).real
                ll_batch = -0.5 * (logdet_M + r2 - quad)
                ll_sum += float(jnp.sum(ll_batch))

            # Backprojection for gradient.  Convert T, m to float64 then
            # complex for the vmapped adjoint (which is complex-typed).
            T_cplx = T.astype(U_jax.dtype)
            m_cplx = m.astype(U_jax.dtype)
            PU_cplx = PU.astype(U_jax.dtype)
            centered_cplx = centered_images.astype(U_jax.dtype)
            CTF_cplx = CTF.astype(U_jax.dtype)

            bp_term1, bp_term2 = _gradient_terms_batch(
                PU_cplx, CTF_cplx, centered_cplx, T_cplx, m_cplx,
            )

            gt1 = batch_over_vol_adjoint_slice_volume(
                bp_term1, rotation_matrices, image_shape, volume_shape, disc_type,
            )
            gt2 = batch_over_vol_adjoint_slice_volume(
                bp_term2, rotation_matrices, image_shape, volume_shape, disc_type,
            )
            grad_U = grad_U + (gt1 - gt2)

        n_processed += len(indices)

    elapsed = time.time() - t0
    logger.info(
        "  dataset pass: n_images=%d, need_grad=%s, elapsed=%.1f s",
        n_processed, need_grad, elapsed,
    )

    if need_grad:
        grad_U = np.asarray(grad_U)
    return G_all, h_all, m_all, T_all, grad_U, ll_sum


# =============================================================================
# NLL (marginal) for line search
# =============================================================================

def _nll_from_Gh(G_all, h_all, B):
    """Marginal NLL (up to image-independent constants) given G, h, B.

        nll = n * log|B| + sum_i log|B^{-1}+G_i| - sum_i h_i^T (B^{-1}+G_i)^{-1} h_i
    """
    n, q, _ = G_all.shape
    B_inv = np.linalg.inv(B)
    sign_B, logdet_B = np.linalg.slogdet(B)
    M = B_inv[None] + G_all
    sign_M, logdet_M = np.linalg.slogdet(M)
    P = np.linalg.inv(M)
    m = np.einsum("nij,nj->ni", P, h_all)
    quad = np.einsum("ni,ni->n", h_all, m)
    return float(n * logdet_B + np.sum(logdet_M) - np.sum(quad))


# =============================================================================
# D-metric retraction
# =============================================================================

def d_metric_qr_retract(U_fourier, sqrt_D):
    """Retract a Fourier-space (vol_size, q) matrix onto {U : U^T D U = I_q}.

    Uses the standard trick: let V = diag(sqrt_D) @ U; QR factor V = Q R with Q
    orthonormal in the unweighted Euclidean metric; then U_new = diag(1/sqrt_D) Q.

    Since D is real Fourier-diagonal, all arithmetic is elementwise in v.
    """
    V = U_fourier * sqrt_D[:, None]
    Q, R = np.linalg.qr(V)
    signs = np.sign(np.diag(R).real)
    signs = np.where(signs == 0, 1.0, signs)
    Q = Q * signs[None, :]
    return Q / sqrt_D[:, None]


def apply_mask_gridding_then_retract(U_fourier, focus_mask, volume_shape, sqrt_D,
                                      use_gridding_correction=True):
    """Match baseline PPCA EM convention: each iter does fourier→real→mask→gridding→fourier.

    Then re-retract to the D-Stiefel manifold so the constraint U^T D U = I still holds.
    """
    q = U_fourier.shape[1]
    vol_size = U_fourier.shape[0]
    # Fourier → real
    U_real = np.zeros((q, *volume_shape), dtype=np.float32)
    for j in range(q):
        U_real[j] = np.asarray(ftu.get_idft3(U_fourier[:, j].reshape(volume_shape))).real.astype(np.float32)
    # Apply soft focus mask
    if focus_mask is not None:
        for j in range(q):
            U_real[j] = U_real[j] * focus_mask
    # Gridding correction (divide by sinc²)
    if use_gridding_correction:
        from recovar.reconstruction.relion_functions import griddingCorrect_square
        for j in range(q):
            U_real[j] = np.asarray(
                griddingCorrect_square(jnp.array(U_real[j]), volume_shape[0], 1, order=1)[0]
            )
    # Real → Fourier
    U_fourier_new = np.zeros_like(U_fourier)
    for j in range(q):
        U_fourier_new[:, j] = np.asarray(ftu.get_dft3(U_real[j])).reshape(-1)
    # Re-retract to D-Stiefel manifold
    return d_metric_qr_retract(U_fourier_new, sqrt_D)


# =============================================================================
# U-step via PCG (replaces gradient descent + line search)
# =============================================================================

def _accumulate_lhs_rhs_for_ustep(
    dataset, mean_jax, m_all, T_all,
    volume_shape, image_shape, voxel_size,
    batch_size, apply_image_mask, disc_type, disc_type_mean,
):
    """Accumulate per-(half-)voxel LHS, RHS for the EM-optimal U-step.

    Solves for U_hat in:
        sum_i M_i U_hat T_α_i = sum_i b_i m_α_i^T

    where M_i = A_i^T Σ_i^{-1} A_i and b_i = A_i^T Σ_i^{-1} r_i.
    Per-voxel form (half-volume Fourier):
        lhs_acc[v, q, q] = sum_i (CTF/σ)²[v] T_α_i  via half-volume backprojection
        rhs_acc[v, q]    = sum_i (CTF/σ)[v] r_i[v] m_α_i^T  via half-volume backprojection

    These are exactly what `pcg_mstep` consumes.
    """
    n_images = m_all.shape[0]
    q = m_all.shape[1]
    vol_size = int(np.prod(volume_shape))
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vs))

    # Accumulate in FULL-volume Fourier (matches batch_over_vol_adjoint_slice_volume),
    # then convert to half-volume at the end for pcg_mstep.
    lhs_acc = jnp.zeros((vol_size, q * q), dtype=jnp.float32)
    rhs_acc = jnp.zeros((vol_size, q), dtype=jnp.complex64)

    n_processed = 0
    t0 = time.time()
    for (
        batch, rotation_matrices, translations, ctf_params,
        _noise_variance, _particle_indices, image_indices,
    ) in dataset.iter_batches(
        batch_size,
        by_image=not getattr(dataset, "tilt_series_flag", False),
    ):
        images = dataset.process_images(batch, apply_image_mask=apply_image_mask)
        noise_variance = dataset.noise.get(image_indices)
        # Half-image whitening: for half-volume backprojection we need half-image too
        # But noise.get returns half_image-sized noise variance per image. Use that.

        images = core.translate_images(
            images, translations, image_shape,
        ) / jnp.sqrt(noise_variance)

        CTF = dataset.ctf_evaluator(
            ctf_params, image_shape, voxel_size,
        ) / jnp.sqrt(noise_variance)

        # Project mean
        projected_mean = _forward_model_from_map(
            mean_jax, ctf_params, rotation_matrices, image_shape, volume_shape,
            voxel_size, dataset.ctf_evaluator, disc_type_mean,
        ) / jnp.sqrt(noise_variance)
        centered_images = images - projected_mean

        ctf_sq = (CTF * jnp.conj(CTF)).real  # |CTF/σ|² = (CTF/σ)² since real

        # Posteriors for this batch
        idx = np.asarray(image_indices)
        m_b = jnp.array(m_all[idx], dtype=jnp.float32)  # (b, q)
        T_b = jnp.array(T_all[idx], dtype=jnp.float32)  # (b, q, q)

        # before_backproj for LHS: (b, image_size, q*q) — broadcast (CTF/σ)² with T_α
        # broadcast: ctf_sq (b, img) → (b, img, 1, 1); T_b (b, q, q) → (b, 1, q, q)
        before_lhs = ctf_sq[:, :, None, None] * T_b[:, None, :, :]   # (b, img, q, q)
        before_lhs = before_lhs.reshape(before_lhs.shape[0], before_lhs.shape[1], q * q)

        # before_backproj for RHS: (b, image_size, q) — CTF/σ * centered * m_α
        # CTF (b, img), centered (b, img), m_b (b, q)
        before_rhs = (CTF[:, :, None] * centered_images[:, :, None]) * m_b[:, None, :]   # (b, img, q)

        # Full-volume backprojection
        lhs_acc = lhs_acc + batch_over_vol_adjoint_slice_volume(
            before_lhs, rotation_matrices, image_shape, volume_shape, disc_type,
        )
        rhs_acc = rhs_acc + batch_over_vol_adjoint_slice_volume(
            before_rhs, rotation_matrices, image_shape, volume_shape, disc_type,
        )

        n_processed += len(idx)

    elapsed = time.time() - t0
    logger.info(
        "  U-step accumulation pass: n_images=%d, elapsed=%.1f s",
        n_processed, elapsed,
    )

    # Convert from full Fourier (vol_size,...) to half-volume Fourier for pcg_mstep
    lhs_full = np.asarray(lhs_acc).reshape(vol_size, q, q)
    rhs_full = np.asarray(rhs_acc)
    # full → half conversion via column-wise full_volume_to_half_volume
    lhs_half = np.zeros((half_vol_size, q, q), dtype=np.float32)
    for i in range(q):
        for j in range(q):
            lhs_half[:, i, j] = ftu.full_volume_to_half_volume(
                lhs_full[:, i, j].reshape(volume_shape), volume_shape,
            ).reshape(-1)
    rhs_half = np.zeros((half_vol_size, q), dtype=np.complex64)
    for i in range(q):
        rhs_half[:, i] = ftu.full_volume_to_half_volume(
            rhs_full[:, i].reshape(volume_shape), volume_shape,
        ).reshape(-1)

    return jnp.asarray(lhs_half), jnp.asarray(rhs_half)


def u_step_pcg(
    dataset, mean_jax, U_real_warmstart, m_all, T_all,
    volume_shape, image_shape, voxel_size, batch_size,
    apply_image_mask, disc_type, disc_type_mean,
    focus_mask, reg_diag_half, sqrt_D_c,
    pcg_maxiter=20, pcg_tol=1e-4,
):
    """EM-optimal U-step via masked PCG, then D-Stiefel retraction.

    Steps:
      1. Accumulate per-half-voxel LHS, RHS using α-space moments (m_α, T_α).
      2. Solve the masked linear system via pcg_mstep:
            (LHS_v + diag(reg)) U_hat[v] = RHS_v   subject to U_hat supported in mask.
      3. D-Stiefel retract: U_hat → DFT → diag(√D)·U_hat → QR → diag(1/√D)·Q.

    Returns:
      U_fourier_new : (vol_size, q) complex128 — on the D-Stiefel manifold
      pcg_residuals : list of float
      U_real_pcg : (q, *vol_shape) — pre-retraction PCG solution (for diagnostics)
    """
    q = m_all.shape[1]

    # 1. Accumulate
    lhs_acc, rhs_acc = _accumulate_lhs_rhs_for_ustep(
        dataset, mean_jax, m_all, T_all,
        volume_shape, image_shape, voxel_size,
        batch_size, apply_image_mask, disc_type, disc_type_mean,
    )

    # 2. PCG solve on the masked space
    focus_mask_jax = jnp.asarray(focus_mask, dtype=jnp.float32) if focus_mask is not None \
        else jnp.ones(volume_shape, dtype=jnp.float32)

    U_real_pcg, residuals = pcg_mstep(
        lhs_fourier=lhs_acc,           # (half_vol, q, q)
        rhs_fourier=rhs_acc,            # (half_vol, q)
        reg_diag=reg_diag_half,         # (half_vol, q)
        mask=focus_mask_jax,
        volume_shape=volume_shape,
        W0_real=jnp.asarray(U_real_warmstart, dtype=jnp.float32) if U_real_warmstart is not None else None,
        maxiter=pcg_maxiter,
        tol=pcg_tol,
        precondition=True,
    )
    U_real_pcg = np.asarray(U_real_pcg)  # (q, *vol_shape)

    # 3. D-Stiefel retract
    vol_size = int(np.prod(volume_shape))
    U_fourier_new = np.zeros((vol_size, q), dtype=np.complex128)
    for j in range(q):
        U_fourier_new[:, j] = np.asarray(ftu.get_dft3(U_real_pcg[j])).reshape(-1)
    U_fourier_new = d_metric_qr_retract(U_fourier_new, sqrt_D_c)

    return U_fourier_new, residuals, U_real_pcg


# =============================================================================
# Main algorithm
# =============================================================================

def dmetric_em(
    pipeline_output,
    output_dir,
    *,
    zdim: int = 10,
    batch_size: int = 128,
    n_iters: int = 20,
    regularize_B: bool = False,
    kappa: float = 1.0,
    eps_B: float = 1e-8,
    n_line_search: int = 10,   # unused with PCG U-step (kept for back-compat)
    apply_image_mask: bool = True,
    save_per_iter: bool = True,
    disc_type: str = "linear_interp",
    disc_type_mean: str = "cubic",
    pcg_maxiter: int = 20,
    pcg_tol: float = 1e-4,
):
    """Run the D-metric generalized EM (Algorithm).

    Parameters
    ----------
    pipeline_output : PipelineOutput
        Source PPCA result.
    output_dir : str
        Directory to write the postprocessed result.
    zdim : int
        Number of PCs.
    n_iters : int
        Outer EM iterations.
    regularize_B : bool
        If True, use B = (sum_i T_i + kappa B_0) / (n + kappa).
    kappa : float
        Prior strength for the regularized B update.
    eps_B : float
        Ridge added to B in the unregularized path.
    n_line_search : int
        Max backtracking steps per U update.
    """
    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])
    image_shape = tuple(volume_shape[:2])
    voxel_size = float(po.params["voxel_size"])
    vol_size = int(np.prod(volume_shape))

    if zdim is None:
        zdim = len(po._list_saved_eigenvector_indices())
    zdim = int(zdim)
    q = zdim

    logger.info("=" * 72)
    logger.info(
        "D-metric generalized EM (regularize_B=%s, kappa=%.2f, zdim=%d, n_iters=%d)",
        regularize_B, kappa, zdim, n_iters,
    )
    logger.info("=" * 72)

    dataset = po.get("dataset")

    # --- Mean (Fourier) + cubic coefficients for slicing ---
    mean_real = utils.load_mrc(po.paths.mean_volume)
    mean_fourier = np.asarray(ftu.get_dft3(mean_real)).reshape(-1)
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)
    mean_jax = jnp.array(mean_for_slicing)

    # --- D metric from shell prior (without /npc) ---
    D_diag, Gamma_diag, prior_info = compute_D_fourier_diag(
        dataset, mean_fourier, zdim, volume_shape, batch_size,
    )

    # Rescale D so that U^T D U ≈ I_q for the PPCA initial U.
    # Without this rescaling D is typically ~500x too small (PPCA u_real has
    # norm 1/√vol_size and the shell prior is in different units), and the
    # initial D-Stiefel retraction has to inflate U by sqrt(scale) which puts
    # the algorithm in a totally different basin from the PPCA starting point.
    u_real_init = po.get_u_real(zdim).astype(np.float32)
    U_fourier_init = np.zeros((vol_size, zdim), dtype=np.complex128)
    for j in range(zdim):
        U_fourier_init[:, j] = np.asarray(ftu.get_dft3(u_real_init[j])).reshape(-1)
    UtDU_init = (np.conj(U_fourier_init).T @ (U_fourier_init * D_diag[:, None])).real
    diag_mean = float(np.mean(np.diag(UtDU_init)))
    if diag_mean > 0:
        scale = 1.0 / diag_mean
        D_diag = D_diag * scale
        Gamma_diag = Gamma_diag / scale
        logger.info("D rescaled by %.3e so U^T D U ≈ I for PPCA initial U", scale)

    sqrt_D = np.sqrt(D_diag).astype(np.float64)
    sqrt_D_c = sqrt_D.astype(np.complex128)

    # --- W_prior ridge for the U-step (matching baseline PPCA M-step) ---
    # The standard PPCA M-step adds diag(1/W_prior) to the per-voxel LHS,
    # which corresponds to a Gaussian prior on each voxel of W.
    # In gradient form, this adds (1/W_prior)[v] * U[v, k] to grad_U[v, k].
    W_prior_full = np.asarray(prior_info["W_prior"])  # (vol_size, q), all columns identical
    reg_diag_full = (1.0 / np.maximum(W_prior_full[:, 0], 1e-16)).astype(np.float64)  # (vol_size,)
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    W_prior_half = ftu.full_volume_to_half_volume(W_prior_full.T, volume_shape).T
    reg_diag_half = (1.0 / np.maximum(W_prior_half, 1e-16)).astype(np.float32)
    reg_diag_half = jnp.asarray(reg_diag_half)

    # --- Initialize U and B from PPCA ---
    U_fourier, B0, _u_real_masked = init_UB_from_ppca(po, zdim, D_diag, Gamma_diag)
    B = B0.copy()

    # Load focus mask (= mask.mrc, soft cosine) for per-iter mask + gridding step.
    # Matches what baseline PPCA EM does: each iter applies this to U after the M-step.
    try:
        focus_mask_arr = np.asarray(utils.load_mrc(po.paths.mask_volume), dtype=np.float32)
        logger.info("Loaded focus_mask for per-iter mask+gridding (mean=%.4f)", focus_mask_arr.mean())
    except Exception as e:
        focus_mask_arr = None
        logger.warning("No focus mask loaded for per-iter step (%s)", e)

    # Retract the initial U (may not be exactly on the manifold due to
    # masking/gridding changing its Fourier content away from the PPCA basis).
    U_fourier = d_metric_qr_retract(U_fourier, sqrt_D_c)
    UtDU = (np.conj(U_fourier).T @ (U_fourier * sqrt_D_c[:, None] ** 2)).real
    dev = np.linalg.norm(UtDU - np.eye(q))
    logger.info("  initial ||U^T D U - I|| = %.3e", dev)

    # Storage for per-iteration diagnostics
    history = []
    iters_dir = os.path.join(output_dir, "iterations") if save_per_iter else None
    if iters_dir is not None:
        os.makedirs(iters_dir, exist_ok=True)

    # Euclidean gradient step size (backtracking starts here each iter).
    eta = 1.0
    nll_prev = None

    for it in range(n_iters):
        t_it = time.time()
        logger.info("--- D-metric EM iter %d/%d ---", it + 1, n_iters)

        U_jax = jnp.asarray(U_fourier, dtype=jnp.complex128)

        # --- Compute G, h, m, T, grad_U in one pass ---
        B_inv = np.linalg.inv(B)
        G_all, h_all, m_all, T_all, grad_U_fourier, _ = _compute_Gh_and_maybe_grad(
            dataset, mean_jax, U_jax, volume_shape, image_shape, voxel_size,
            batch_size, apply_image_mask, disc_type, disc_type_mean,
            need_grad=True, B_inv=B_inv, return_ll=False,
        )
        n = G_all.shape[0]

        # --- B-step ---
        S = T_all.sum(axis=0)
        S = 0.5 * (S + S.T)
        if regularize_B:
            B_new = (S + kappa * B0) / (n + kappa)
        else:
            B_new = S / n + eps_B * np.eye(q)
        B_new = 0.5 * (B_new + B_new.T)
        # Ensure PSD
        w_B, V_B = np.linalg.eigh(B_new)
        w_B = np.maximum(w_B, eps_B)
        B_new = (V_B * w_B[None, :]) @ V_B.T
        eigvals_B_new_sorted = np.sort(w_B)[::-1]
        logger.info(
            "  B-step: eigenvalues (desc)=%s",
            np.array2string(eigvals_B_new_sorted, precision=3),
        )

        # --- E-step REDO with new B ---
        # The grad_U we computed earlier used (m, T) from the OLD B.  After the
        # B-step changed B, the (m, T) are stale.  Recompute fresh moments and
        # gradient using B_new so the U-step is consistent with EM theory.
        B_inv_new = np.linalg.inv(B_new)
        G_all, h_all, m_all, T_all, grad_U_fourier, _ = _compute_Gh_and_maybe_grad(
            dataset, mean_jax, U_jax, volume_shape, image_shape, voxel_size,
            batch_size, apply_image_mask, disc_type, disc_type_mean,
            need_grad=True, B_inv=B_inv_new, return_ll=False,
        )

        # Add per-voxel ridge gradient: (1/W_prior)[v] * U[v, k]
        # (matches baseline PPCA M-step regularization)
        grad_U_fourier = grad_U_fourier + reg_diag_full[:, None] * np.asarray(U_fourier)

        # Regularization contribution to NLL: sum_{v,k} (1/W_prior)[v] |U[v,k]|^2
        def _reg_term(U_arr):
            return float(np.sum(reg_diag_full[:, None] * (U_arr.real ** 2 + U_arr.imag ** 2)))

        # --- NLL before U-step (for line search comparisons) ---
        nll_before = _nll_from_Gh(G_all, h_all, B_new) + _reg_term(np.asarray(U_fourier))
        logger.info("  NLL after B-step (re-Estep, +reg): %.6e", nll_before)

        # --- U-step: gradient descent on D-Stiefel manifold + line search ---
        grad_norm = float(np.linalg.norm(grad_U_fourier))
        logger.info("  ||grad_U||=%.3e", grad_norm)

        U_accepted = U_fourier
        nll_after = nll_before
        if grad_norm > 1e-14:
            direction = grad_U_fourier / max(grad_norm, 1e-30)
            cur_eta = eta
            accepted = False
            for bt in range(n_line_search):
                U_prop = U_fourier - cur_eta * direction
                U_prop = d_metric_qr_retract(U_prop, sqrt_D_c)
                U_jax_prop = jnp.asarray(U_prop, dtype=jnp.complex128)
                G_prop, h_prop, _, _, _, _ = _compute_Gh_and_maybe_grad(
                    dataset, mean_jax, U_jax_prop, volume_shape, image_shape,
                    voxel_size, batch_size, apply_image_mask, disc_type,
                    disc_type_mean, need_grad=False, return_ll=False,
                )
                nll_prop = _nll_from_Gh(G_prop, h_prop, B_new) + _reg_term(np.asarray(U_prop))
                logger.info(
                    "    line search bt=%d eta=%.3e nll=%.6e (delta=%.3e)",
                    bt, cur_eta, nll_prop, nll_prop - nll_before,
                )
                if nll_prop < nll_before - 1e-10 * abs(nll_before):
                    U_accepted = U_prop
                    nll_after = nll_prop
                    G_all = G_prop
                    h_all = h_prop
                    eta = min(cur_eta * 1.5, 10.0)
                    accepted = True
                    break
                cur_eta *= 0.5
            if not accepted:
                eta = max(cur_eta, 1e-12)
                logger.info("    line search did not improve NLL; keeping U unchanged")

        U_fourier = U_accepted

        B = B_new

        # --- Gauge fix: diagonalize B, rotate U and m ---
        w_B, R_B = np.linalg.eigh(B)
        order = np.argsort(w_B)[::-1]
        w_B = w_B[order]
        R_B = R_B[:, order]
        B = np.diag(w_B).astype(np.float64)
        U_fourier = U_fourier @ R_B.astype(U_fourier.dtype)
        if m_all is not None:
            m_all = m_all @ R_B

        # --- Diagnostics & per-iter save ---
        UtDU = (np.conj(U_fourier).T @ (U_fourier * sqrt_D_c[:, None] ** 2)).real
        manifold_dev = float(np.linalg.norm(UtDU - np.eye(q)))

        elapsed = time.time() - t_it
        logger.info(
            "  iter %d done: B eigvals=%s, ||U^TDU-I||=%.2e, %.1fs",
            it + 1,
            np.array2string(w_B[: min(q, 8)], precision=3),
            manifold_dev,
            elapsed,
        )

        hist_entry = {
            "iter": it + 1,
            "eigenvalues": w_B.tolist(),
            "nll_after_Bstep": nll_before,
            "nll_after_Ustep": nll_after,
            "manifold_dev": manifold_dev,
            "elapsed_s": elapsed,
        }
        history.append(hist_entry)

        if save_per_iter:
            # Save U_real for this iter (pre-gridding-inverse; real-space IFFT of U_fourier).
            U_real_iter = np.zeros((q, *volume_shape), dtype=np.float32)
            for j in range(q):
                U_real_iter[j] = np.asarray(
                    ftu.get_idft3(U_fourier[:, j].reshape(volume_shape)).real,
                    dtype=np.float32,
                )
            np.save(
                os.path.join(iters_dir, f"U_real_iter{it + 1:03d}.npy"),
                U_real_iter,
            )

        if nll_prev is not None:
            rel = abs(nll_after - nll_prev) / max(abs(nll_prev), 1.0)
            if rel < 1e-7:
                logger.info(
                    "  Converged: rel NLL change %.2e < 1e-7 at iter %d", rel, it + 1,
                )
                nll_prev = nll_after
                break
        nll_prev = nll_after

    # =========================================================================
    # Finalize: recompute embeddings, save result dir
    # =========================================================================
    logger.info("Finalizing: recomputing final G, h for embeddings.")

    U_jax = jnp.asarray(U_fourier, dtype=jnp.complex128)
    G_final, h_final, _, _, _, _ = _compute_Gh_and_maybe_grad(
        dataset, mean_jax, U_jax, volume_shape, image_shape, voxel_size,
        batch_size, apply_image_mask, disc_type, disc_type_mean,
        need_grad=False, return_ll=False,
    )

    latent_coords, precision_all = compute_embeddings_from_UB(G_final, h_final, B)

    # Convert final U to real space for saving as MRCs.
    U_real_final = np.zeros((q, *volume_shape), dtype=np.float32)
    for j in range(q):
        U_real_final[j] = np.asarray(
            ftu.get_idft3(U_fourier[:, j].reshape(volume_shape)).real,
            dtype=np.float32,
        )

    # Eigenvalues = diagonal of (gauge-fixed) B
    eigenvalues = np.diag(B).astype(np.float64)
    new_s = eigenvalues.astype(np.float32)

    n_images = latent_coords.shape[0]
    embeddings_dict = {
        "latent_coords": {q: latent_coords.astype(np.float32)},
        "latent_coords_noreg": {q: latent_coords.astype(np.float32)},
        "latent_precision": {q: precision_all.astype(np.float32)},
        "latent_precision_noreg": {q: precision_all.astype(np.float32)},
        "contrasts": {q: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {q: np.ones(n_images, dtype=np.float32)},
    }
    try:
        src_c = po.get_embedding_component("contrasts", q)
        embeddings_dict["contrasts"][q] = np.asarray(src_c, dtype=np.float32)
        src_cnr = po.get_embedding_component("contrasts_noreg", q)
        embeddings_dict["contrasts_noreg"][q] = np.asarray(src_cnr, dtype=np.float32)
    except Exception:
        pass

    method_info = {
        "method": "dmetric_em_reg" if regularize_B else "dmetric_em_noreg",
        "zdim": q,
        "n_iters": n_iters,
        "regularize_B": regularize_B,
        "kappa": float(kappa),
        "eps_B": float(eps_B),
        "eigenvalues": eigenvalues.tolist(),
        "history": history,
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        U_real_final,
        new_s,
        embeddings_dict,
        method_info=method_info,
    )

    return {
        "eigenvalues": eigenvalues,
        "B": B,
        "U_real": U_real_final,
        "embeddings": latent_coords,
        "history": history,
    }


# =============================================================================
# High-level entry points for CLI dispatch
# =============================================================================

def run_dmetric_em_noreg(pipeline_output, output_dir, *, zdim=10, batch_size=128,
                         n_iters=20, eps_B=1e-8, **kwargs):
    """Unregularized variant: B = (1/n) sum T_i + eps I."""
    return dmetric_em(
        pipeline_output, output_dir,
        zdim=zdim, batch_size=batch_size, n_iters=n_iters,
        regularize_B=False, eps_B=eps_B, **kwargs,
    )


def run_dmetric_em_reg(pipeline_output, output_dir, *, zdim=10, batch_size=128,
                       n_iters=20, kappa=1.0, **kwargs):
    """Regularized variant: B = (sum T_i + kappa B_0) / (n + kappa)."""
    return dmetric_em(
        pipeline_output, output_dir,
        zdim=zdim, batch_size=batch_size, n_iters=n_iters,
        regularize_B=True, kappa=kappa, **kwargs,
    )
