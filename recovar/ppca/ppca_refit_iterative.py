"""Algorithms 2 and 3: iterative PPCA refit with joint U/B updates.

Algorithm 2 (``alternating_UB_stiefel``): alternating EM on the Stiefel
manifold.  Each outer iteration runs an E-step (posterior moments), a
closed-form B-step, and a gradient-based U-step with Stiefel retraction.

Algorithm 3 (``whitening_manifold_UB``): keeps U on the whitening manifold
{U : (1/n) sum G_i(U) = I} and updates B with an explicit EM step.

Both algorithms accept experiment_datasets (CryoEMDataset or list thereof)
and initial U, B.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core import linalg
from recovar.ppca.ppca import (
    _normalize_experiment_datasets,
    _iter_processed_batches,
    _prepare_mean_estimate_for_slicing,
    _forward_model_from_map,
    batch_over_vol_slice_volume,
    batch_over_vol_adjoint_slice_volume,
)

logger = logging.getLogger(__name__)

# Try importing companion module (ppca_refit.py).  If it doesn't exist yet,
# the stubs below provide the interface so the rest of this module can be
# imported and tested independently.
try:
    from recovar.ppca.ppca_refit import (
        compute_per_image_Gi_hi,
        compute_embeddings_from_UB,
        create_postprocessed_result_dir,
    )
except ImportError:
    logger.debug("recovar.ppca.ppca_refit not available; stub imports used.")

    def compute_per_image_Gi_hi(*args, **kwargs):
        raise NotImplementedError("ppca_refit.compute_per_image_Gi_hi not yet available")

    def compute_embeddings_from_UB(*args, **kwargs):
        raise NotImplementedError("ppca_refit.compute_embeddings_from_UB not yet available")

    def create_postprocessed_result_dir(*args, **kwargs):
        raise NotImplementedError("ppca_refit.create_postprocessed_result_dir not yet available")


# ============================================================================
# Utility / helper functions
# ============================================================================

def _sym(A):
    """Symmetric part: (A + A^T) / 2."""
    return 0.5 * (A + A.T)


def _ensure_psd(B, eps=1e-8):
    """Clip eigenvalues of symmetric B to [eps, inf] and re-symmetrize."""
    eigvals, eigvecs = jnp.linalg.eigh(B)
    eigvals = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


def _sqrtm_psd(B):
    """Matrix square root of a symmetric PSD matrix via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(B)
    eigvals = jnp.maximum(eigvals, 0.0)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T


def _inv_sqrtm_psd(B, eps=1e-12):
    """Inverse matrix square root of a symmetric PSD matrix."""
    eigvals, eigvecs = jnp.linalg.eigh(B)
    eigvals = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T


def _compute_freq_sq_grid(volume_shape):
    """Compute |k|^2 for each voxel in the centered DFT grid.

    Returns a flat array of shape (volume_size,) with the squared frequency
    magnitude at each Fourier voxel.
    """
    coords = ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size=1, scaled=False)
    return jnp.sum(coords ** 2, axis=-1)


def stiefel_project(grad, U_flat):
    """Project Euclidean gradient to Stiefel tangent space at U.

    Parameters
    ----------
    grad : (d, q) -- Euclidean gradient in flattened volume space.
    U_flat : (d, q) -- current point on Stiefel manifold.

    Returns
    -------
    grad_tangent : (d, q) -- tangent vector satisfying U^T g + g^T U = 0.
    """
    UtG = U_flat.T @ grad
    return grad - U_flat @ _sym(UtG)


def qr_retraction(U_stepped_flat):
    """QR retraction to Stiefel manifold.

    Parameters
    ----------
    U_stepped_flat : (d, q) -- point near manifold.

    Returns
    -------
    Q : (d, q) -- orthonormal columns.
    """
    Q, R = jnp.linalg.qr(U_stepped_flat)
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    Q = Q * signs[None, :]
    return Q


def _U_real_to_fourier_flat(U_real, volume_shape):
    """Convert U from real space (q, *volume_shape) to Fourier flat (volume_size, q)."""
    U_fourier = jax.vmap(lambda u: ftu.get_dft3(u).ravel())(U_real)  # (q, volume_size)
    return U_fourier.T  # (volume_size, q)


def _U_fourier_flat_to_real(U_fourier_flat, volume_shape):
    """Convert U from Fourier flat (volume_size, q) to real space (q, *volume_shape)."""
    U_fourier = U_fourier_flat.T  # (q, volume_size)
    U_real = jax.vmap(lambda u: ftu.get_idft3(u.reshape(volume_shape)).real)(U_fourier)
    return U_real


# ============================================================================
# Core E-step: compute posterior moments for all images given (U, B)
# ============================================================================

def _e_step_full(
    experiment_datasets,
    mean_estimate,
    U_fourier,
    B,
    batch_size,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    compute_ll=True,
):
    """E-step: compute posterior moments m_i, P_i, T_i for all images.

    Model: x_i = mu + U alpha_i, alpha_i ~ N(0, B).
    With noise-whitened CTF: PU_i = CTF_wh_i * slice(U).

    G_i = PU_i^H PU_i  (q x q)
    h_i = PU_i^H r_i   (q,)
    P_i = (B^{-1} + G_i)^{-1}
    m_i = P_i h_i
    T_i = P_i + m_i m_i^T

    Returns
    -------
    m_all : (n_images, q) -- posterior means.
    T_all : (n_images, q, q) -- posterior second moments.
    ll_sum : float -- summed log-likelihood (if compute_ll).
    n_total : int -- total number of images.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    q = U_fourier.shape[1]

    B_inv = jnp.linalg.inv(B)

    mean_for_slicing = _prepare_mean_estimate_for_slicing(
        mean_estimate, None, ref.volume_shape, disc_type_mean,
    )

    m_all_list = []
    T_all_list = []
    ll_sum = jnp.array(0.0, dtype=jnp.float64)
    n_total = 0

    # Precompute log|B| for NLL (constant across all images)
    if compute_ll:
        L_B = jnp.linalg.cholesky(B)
        logdetB = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(L_B))))
    else:
        logdetB = 0.0

    for experiment_dataset in dataset_list:
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            n_batch = batch.shape[0]
            n_total += n_batch

            images = core.translate_images(batch, translations, experiment_dataset.image_shape)
            images = images / jnp.sqrt(noise_variance)
            CTF = experiment_dataset.ctf_evaluator(
                ctf_params, experiment_dataset.image_shape, experiment_dataset.voxel_size,
            ) / jnp.sqrt(noise_variance)

            projected_mean = _forward_model_from_map(
                mean_for_slicing, ctf_params, rotation_matrices,
                experiment_dataset.image_shape, experiment_dataset.volume_shape,
                experiment_dataset.voxel_size, experiment_dataset.ctf_evaluator,
                disc_type_mean, skip_ctf=False,
            ) / jnp.sqrt(noise_variance)

            centered_images = images - projected_mean

            # PU: (batch, q, image_size) -- noise-whitened projections of basis
            PU = batch_over_vol_slice_volume(
                U_fourier, rotation_matrices,
                experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type,
            )
            PU = PU * CTF[:, None, :]

            G = (jnp.conj(PU) @ PU.transpose(0, 2, 1)).real  # (batch, q, q)
            h = (jnp.conj(PU) @ centered_images[:, :, None]).real.squeeze(-1)  # (batch, q)

            M = B_inv[None, :, :] + G  # (batch, q, q)
            # Use Cholesky for numerical stability
            L_M = jnp.linalg.cholesky(M)
            # P = M^{-1} via Cholesky: solve M P = I
            eye_q = jnp.broadcast_to(jnp.eye(q), M.shape)
            P = jax.scipy.linalg.cho_solve((L_M, True), eye_q)
            m = jax.scipy.linalg.cho_solve((L_M, True), h[:, :, None]).squeeze(-1)  # (batch, q)
            T = P + m[:, :, None] * m[:, None, :]  # (batch, q, q)

            m_all_list.append(np.array(m))
            T_all_list.append(np.array(T))

            if compute_ll:
                d_n = images.shape[-1]
                r2 = jnp.sum(jnp.abs(centered_images) ** 2, axis=-1).real
                quad = jnp.sum(h * m, axis=-1)
                # log|B^{-1} + G_i| from Cholesky factor L_M already computed
                logdetM = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(L_M, axis1=1, axis2=2))), axis=-1)
                # Full NLL: log|B| + log|B^{-1}+G_i| + r2 - h^T(B^{-1}+G_i)^{-1}h
                ll_batch = -0.5 * (d_n * jnp.log(2.0 * jnp.pi) + logdetB + logdetM + r2 - quad)
                ll_sum += jnp.sum(ll_batch)

    m_all = np.concatenate(m_all_list, axis=0)
    T_all = np.concatenate(T_all_list, axis=0)
    return m_all, T_all, float(ll_sum), n_total


# ============================================================================
# U-gradient via backprojection
# ============================================================================

@functools.partial(jax.jit, static_argnums=[7, 8, 12, 13, 14])
def _u_gradient_batch(
    images,
    mean,
    U_fourier,
    T_batch,
    m_batch,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    noise_variance,
    translations,
    ctf_evaluator,
    disc_type="linear_interp",
    disc_type_mean="cubic",
):
    """Compute gradient contribution from one batch of images.

    Returns (grad_term1 - grad_term2) summed over batch, shape (volume_size, q).

    The Euclidean gradient of the EM surrogate is:
        dQ/dU = sum_i [ M_i U T_i  -  b_i m_i^T ]
    where M_i = A_i^T D_i A_i, D_i = diag(CTF_i^2/noise_i), b_i = A_i^T D_i r_i.

    With noise-whitened CTF_wh = CTF/sqrt(noise) and PU = CTF_wh * slice(U):

    Term 1 (sum_i M_i U T_i):
        [M_i U T_i]_{v,j} = A_i^T D_i * sum_k (A_i U_k) T_i[k,j]
        Since D_i (A_i U_k) = CTF_wh^2 * slice(U_k) = CTF_wh * PU_k:
          before_backproj[i,pixel,j] = CTF_wh[i,pixel] * sum_k PU[i,k,pixel] * T_i[k,j]
          = CTF_wh * (T @ PU) per image, then backproject.

    Term 2 (sum_i b_i m_i^T):
        b_i = A_i^T D_i r_i.  With whitened quantities,
        b_i[v] = backproject_i(CTF_wh * centered_wh)[v], since
        A_i^T diag(CTF_wh) centered_wh = A_i^T CTF r / noise = A_i^T D_i r_i.
        So: before_backproj[i,pixel,j] = CTF_wh * centered_wh * m_i[j].
    """
    q = U_fourier.shape[1]

    # Noise-whiten
    images_w = core.translate_images(images, translations, image_shape) / jnp.sqrt(noise_variance)
    CTF = ctf_evaluator(CTF_params, image_shape, voxel_size) / jnp.sqrt(noise_variance)

    projected_mean = _forward_model_from_map(
        mean, CTF_params, rotation_matrices, image_shape, volume_shape,
        voxel_size, ctf_evaluator, disc_type_mean, skip_ctf=False,
    ) / jnp.sqrt(noise_variance)

    centered_images = images_w - projected_mean

    # PU: (batch, q, image_size) -- noise-whitened
    PU = batch_over_vol_slice_volume(
        U_fourier, rotation_matrices, image_shape, volume_shape, disc_type,
    )
    PU = PU * CTF[:, None, :]

    # --- Term 1: backproject( CTF_wh * (T @ PU) ) ---
    T_PU = jnp.matmul(T_batch, PU)  # (batch, q, image_size)
    bp_term1 = CTF[:, None, :] * T_PU  # (batch, q, image_size)
    bp_term1 = bp_term1.transpose(0, 2, 1)  # (batch, image_size, q)

    # batch_over_vol_adjoint_slice_volume: vmapped over last axis of slices
    # Input: (batch, image_size, q), output: (volume_size, q)
    grad_term1 = batch_over_vol_adjoint_slice_volume(
        bp_term1, rotation_matrices, image_shape, volume_shape, disc_type,
    )

    # --- Term 2: backproject( CTF_wh * centered_wh * m^T ) ---
    bp_term2 = CTF[:, :, None] * centered_images[:, :, None] * m_batch[:, None, :]
    # Shape: (batch, image_size, q)

    grad_term2 = batch_over_vol_adjoint_slice_volume(
        bp_term2, rotation_matrices, image_shape, volume_shape, disc_type,
    )

    return grad_term1 - grad_term2


def compute_U_gradient_via_backprojection(
    experiment_datasets,
    mean_estimate,
    U_fourier,
    T_all,
    m_all,
    batch_size,
    volume_shape,
    disc_type_mean="cubic",
    disc_type="linear_interp",
):
    """Full dataset pass to compute the Euclidean gradient of Q(U) w.r.t. U.

    Parameters
    ----------
    experiment_datasets : dataset or list of datasets.
    mean_estimate : mean volume (Fourier, possibly spline coefficients).
    U_fourier : (volume_size, q) -- current basis in Fourier space.
    T_all : (n_images, q, q) -- posterior second moments from E-step.
    m_all : (n_images, q) -- posterior means from E-step.
    batch_size : int.
    volume_shape : tuple (D, D, D).

    Returns
    -------
    grad_U : (volume_size, q) -- Euclidean gradient in Fourier space.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    q = U_fourier.shape[1]
    volume_size = int(np.prod(volume_shape))

    mean_for_slicing = _prepare_mean_estimate_for_slicing(
        mean_estimate, None, volume_shape, disc_type_mean,
    )

    grad_U = jnp.zeros((volume_size, q), dtype=U_fourier.dtype)
    idx = 0

    for experiment_dataset in dataset_list:
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            n_batch = batch.shape[0]
            T_batch = jnp.array(T_all[idx:idx + n_batch])
            m_batch = jnp.array(m_all[idx:idx + n_batch])
            noise_variance = experiment_dataset.noise.get(batch_image_ind)

            grad_batch = _u_gradient_batch(
                batch, mean_for_slicing, U_fourier, T_batch, m_batch,
                ctf_params, rotation_matrices,
                experiment_dataset.image_shape, experiment_dataset.volume_shape,
                experiment_dataset.voxel_size, noise_variance, translations,
                experiment_dataset.ctf_evaluator,
                disc_type=disc_type, disc_type_mean=disc_type_mean,
            )
            grad_U = grad_U + grad_batch
            idx += n_batch

    return grad_U


def _apply_laplacian_sq_fourier(U_fourier, volume_shape):
    """Apply L^T L = |k|^4 in Fourier space to each column of U.

    L is the 3D discrete Laplacian (self-adjoint), so L^T L = L^2.
    In Fourier space L = -|k|^2, hence L^T L = |k|^4.

    Parameters
    ----------
    U_fourier : (volume_size, q) -- basis columns in Fourier space.
    volume_shape : (D, D, D).

    Returns
    -------
    LtL_U : (volume_size, q) -- L^T L U in Fourier space.
    """
    freq_sq = _compute_freq_sq_grid(volume_shape)
    k4 = freq_sq ** 2
    return k4[:, None] * U_fourier


# ============================================================================
# Algorithm 2: Alternating U/B on Stiefel manifold
# ============================================================================

def alternating_UB_stiefel(
    experiment_datasets,
    mean_estimate,
    U_init,
    B_init,
    n_outer=10,
    n_inner_U=3,
    lambda_U=0.0,
    batch_size=128,
    eps=1e-8,
    eta_init=1e-3,
    disc_type_mean="cubic",
    disc_type="linear_interp",
):
    """Algorithm 2: full alternating U/B on the Stiefel manifold.

    Each outer iteration:
      1. E-step: compute posterior moments {m_i, T_i} given (U, B).
      2. B-step: B = mean(T_i) + eps*I  (closed-form).
      3. U-step: Stiefel gradient descent with QR retraction and backtracking
         line search (n_inner_U steps per outer iteration).

    Parameters
    ----------
    experiment_datasets : CryoEMDataset or list thereof.
    mean_estimate : mean volume in Fourier space (or spline coefs if disc_type_mean='cubic').
    U_init : (q, *volume_shape) -- real-space orthonormal initial basis.
    B_init : (q, q) -- initial latent covariance.
    n_outer : number of outer EM iterations.
    n_inner_U : number of Stiefel retraction steps per U-update.
    lambda_U : regularization weight for smoothness penalty tr(U^T L^T L U).
    batch_size : batch size for dataset iteration.
    eps : floor for B eigenvalues.
    eta_init : initial step size for U gradient steps.
    disc_type_mean, disc_type : interpolation types.

    Returns
    -------
    U_final : (q, *volume_shape) -- real-space orthonormal basis.
    B_final : (q, q) -- latent covariance.
    m_all : (n_images, q) -- final posterior means.
    history : list of dicts with convergence diagnostics.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    volume_shape = ref.volume_shape
    volume_size = int(np.prod(volume_shape))
    q = U_init.shape[0]

    U_fourier = _U_real_to_fourier_flat(jnp.array(U_init, dtype=jnp.complex128), volume_shape)
    B = _ensure_psd(jnp.array(B_init, dtype=jnp.float64), eps)

    history = []
    eta = eta_init
    grad_norm = 0.0
    ll_prev = -np.inf
    convergence_tol = 1e-6

    for outer in range(n_outer):
        t0 = time.time()
        logger.info(f"=== Stiefel outer iteration {outer + 1}/{n_outer} ===")

        # ---- E-step ----
        m_all, T_all, ll, n_total = _e_step_full(
            experiment_datasets, mean_estimate, U_fourier, B,
            batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
            compute_ll=True,
        )
        ll_per_image = ll / max(n_total, 1)
        logger.info(f"  E-step: ll={ll:.6f}, ll/n={ll_per_image:.6f}, n_images={n_total}")

        # ---- Check convergence ----
        if ll_prev > -np.inf:
            rel_change = abs(ll - ll_prev) / max(abs(ll_prev), 1.0)
            logger.info(f"  Relative NLL change: {rel_change:.2e}")
            if rel_change < convergence_tol:
                logger.info(f"  Converged (relative change {rel_change:.2e} < {convergence_tol:.1e}).")
                history.append({
                    "outer_iter": outer + 1,
                    "ll": ll,
                    "B_eigenvalues": B_eigvals.tolist() if outer > 0 else [],
                    "grad_norm": grad_norm,
                    "eta": eta,
                    "elapsed_s": time.time() - t0,
                    "converged": True,
                })
                break
        ll_prev = ll

        # ---- B-step (closed-form) ----
        B = _ensure_psd(_sym(jnp.mean(jnp.array(T_all), axis=0) + eps * jnp.eye(q)), eps)
        B_eigvals = np.sort(np.linalg.eigvalsh(np.array(B)))[::-1]
        logger.info(f"  B-step: eigenvalues={B_eigvals}")

        # ---- U-step (Stiefel gradient descent with retraction) ----
        for inner in range(n_inner_U):
            grad_U = compute_U_gradient_via_backprojection(
                experiment_datasets, mean_estimate, U_fourier,
                T_all, m_all, batch_size, volume_shape,
                disc_type_mean=disc_type_mean, disc_type=disc_type,
            )

            if lambda_U > 0:
                grad_U = grad_U + 2.0 * lambda_U * _apply_laplacian_sq_fourier(U_fourier, volume_shape)

            # Stiefel operations in real space (orthonormality is in real space)
            U_real_flat = _U_fourier_flat_to_real(U_fourier, volume_shape).reshape(q, -1).T
            grad_real_flat = _U_fourier_flat_to_real(grad_U, volume_shape).reshape(q, -1).T

            grad_tangent = stiefel_project(grad_real_flat, U_real_flat)
            grad_norm = float(jnp.linalg.norm(grad_tangent))
            logger.info(f"  U-step {inner + 1}/{n_inner_U}: ||grad_tangent||={grad_norm:.6e}")

            if grad_norm < 1e-12:
                logger.info("  Gradient near zero, skipping U-step.")
                break

            # Normalize gradient for scale-invariant step size
            descent_dir = grad_tangent / grad_norm

            # Backtracking line search
            # Initial effective step = eta (which is normalized by grad_norm)
            current_eta = eta
            accepted = False
            for bt in range(12):
                U_real_new = qr_retraction(U_real_flat - current_eta * descent_dir)
                U_fourier_new = _U_real_to_fourier_flat(
                    jnp.array(U_real_new.T.reshape(q, *volume_shape), dtype=jnp.complex128),
                    volume_shape,
                )
                _, _, ll_new, _ = _e_step_full(
                    experiment_datasets, mean_estimate, U_fourier_new, B,
                    batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
                    compute_ll=True,
                )
                if ll_new > ll:
                    U_fourier = U_fourier_new
                    ll = ll_new
                    eta = min(current_eta * 1.5, 1.0)
                    accepted = True
                    logger.info(f"    Accepted: eta={current_eta:.2e}, ll={ll_new:.6f}, bt={bt}")
                    break
                current_eta *= 0.5

            if not accepted:
                logger.info(f"    Rejected U-step after backtracking (final eta={current_eta:.2e}).")
                eta = max(current_eta, 1e-10)

            # Re-run E-step with updated U for next inner iteration
            if inner < n_inner_U - 1:
                m_all, T_all, ll, n_total = _e_step_full(
                    experiment_datasets, mean_estimate, U_fourier, B,
                    batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
                    compute_ll=True,
                )

        elapsed = time.time() - t0
        logger.info(f"  Outer iteration {outer + 1} done in {elapsed:.1f}s, ll={ll:.6f}")

        history.append({
            "outer_iter": outer + 1,
            "ll": ll,
            "B_eigenvalues": B_eigvals.tolist(),
            "grad_norm": grad_norm,
            "eta": eta,
            "elapsed_s": elapsed,
            "converged": False,
        })

    # Convert final U to real space
    U_final = _U_fourier_flat_to_real(U_fourier, volume_shape)
    m_final, _, _, _ = _e_step_full(
        experiment_datasets, mean_estimate, U_fourier, B,
        batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
        compute_ll=False,
    )

    return np.array(U_final), np.array(B), np.array(m_final), history


# ============================================================================
# Algorithm 3: Whitening-manifold + explicit B
# ============================================================================

def _compute_Sigma_U(
    experiment_datasets,
    mean_estimate,
    U_fourier,
    batch_size,
    disc_type_mean="cubic",
    disc_type="linear_interp",
):
    """Compute Sigma(U) = (1/n) sum_i G_i(U) where G_i = PU_i^H PU_i.

    Returns
    -------
    Sigma : (q, q) -- average Gram matrix.
    n_total : int -- number of images.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    q = U_fourier.shape[1]

    Sigma_sum = jnp.zeros((q, q), dtype=jnp.float64)
    n_total = 0

    for experiment_dataset in dataset_list:
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            n_total += batch.shape[0]

            CTF = experiment_dataset.ctf_evaluator(
                ctf_params, experiment_dataset.image_shape, experiment_dataset.voxel_size,
            ) / jnp.sqrt(noise_variance)

            PU = batch_over_vol_slice_volume(
                U_fourier, rotation_matrices,
                experiment_dataset.image_shape, experiment_dataset.volume_shape, disc_type,
            )
            PU = PU * CTF[:, None, :]

            G = (jnp.conj(PU) @ PU.transpose(0, 2, 1)).real
            Sigma_sum += jnp.sum(G, axis=0)

    return Sigma_sum / float(n_total), n_total


def _whitening_retraction(U_fourier, Sigma):
    """Whitening retraction: U -> U @ Sigma(U)^{-1/2}.

    Projects U onto the whitening manifold M = {U : (1/n) sum G_i(U) = I}.
    """
    return U_fourier @ _inv_sqrtm_psd(Sigma)


def whitening_manifold_UB(
    experiment_datasets,
    mean_estimate,
    U_init,
    B_init,
    n_iters=10,
    lambda_U=0.0,
    batch_size=128,
    eps=1e-8,
    eta_init=1e-3,
    whiten_every=3,
    disc_type_mean="cubic",
    disc_type="linear_interp",
):
    """Algorithm 3: whitening-manifold update with explicit B.

    Keeps U on the whitening manifold M = {U : (1/n) sum G_i(U) = I}.
    Uses QR retraction for cheap iterations and full whitening retraction
    every ``whiten_every`` iterations.

    Parameters
    ----------
    experiment_datasets : CryoEMDataset or list thereof.
    mean_estimate : mean volume in Fourier space.
    U_init : (q, *volume_shape) -- real-space orthonormal initial basis.
    B_init : (q, q) -- initial latent covariance.
    n_iters : number of iterations.
    lambda_U : regularization weight for smoothness penalty.
    batch_size : batch size for dataset iteration.
    eps : floor for eigenvalues.
    eta_init : initial step size.
    whiten_every : full whitening retraction every K iterations.
    disc_type_mean, disc_type : interpolation types.

    Returns
    -------
    U_final : (q, *volume_shape) -- real-space basis.
    B_final : (q, q) -- latent covariance.
    m_all : (n_images, q) -- final posterior means.
    history : list of dicts with convergence diagnostics.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    volume_shape = ref.volume_shape
    q = U_init.shape[0]

    U_fourier = _U_real_to_fourier_flat(jnp.array(U_init, dtype=jnp.complex128), volume_shape)
    B = _ensure_psd(jnp.array(B_init, dtype=jnp.float64), eps)

    # Step 1: Initialize U on the whitening manifold
    logger.info("Initializing U on the whitening manifold...")
    Sigma_init, n_total = _compute_Sigma_U(
        experiment_datasets, mean_estimate, U_fourier, batch_size,
        disc_type_mean=disc_type_mean, disc_type=disc_type,
    )
    logger.info(f"  Initial Sigma eigenvalues: {np.sort(np.linalg.eigvalsh(np.array(Sigma_init)))[::-1]}")
    U_fourier = _whitening_retraction(U_fourier, Sigma_init)

    Sigma_check, _ = _compute_Sigma_U(
        experiment_datasets, mean_estimate, U_fourier, batch_size,
        disc_type_mean=disc_type_mean, disc_type=disc_type,
    )
    wh_err = float(jnp.linalg.norm(Sigma_check - jnp.eye(q)))
    logger.info(f"  After whitening retraction: ||Sigma - I|| = {wh_err:.6e}")

    history = []
    eta = eta_init
    grad_norm = 0.0
    ll_prev = -np.inf
    convergence_tol = 1e-6

    for it in range(n_iters):
        t0 = time.time()
        logger.info(f"=== Whitening manifold iteration {it + 1}/{n_iters} ===")

        # ---- E-step ----
        m_all, T_all, ll, n_total = _e_step_full(
            experiment_datasets, mean_estimate, U_fourier, B,
            batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
            compute_ll=True,
        )
        ll_per_image = ll / max(n_total, 1)
        logger.info(f"  E-step: ll={ll:.6f}, ll/n={ll_per_image:.6f}, n_images={n_total}")

        # ---- Check convergence ----
        if ll_prev > -np.inf:
            rel_change = abs(ll - ll_prev) / max(abs(ll_prev), 1.0)
            logger.info(f"  Relative NLL change: {rel_change:.2e}")
            if rel_change < convergence_tol:
                logger.info(f"  Converged (relative change {rel_change:.2e} < {convergence_tol:.1e}).")
                history.append({
                    "iter": it + 1,
                    "ll": ll,
                    "B_eigenvalues": B_eigvals.tolist() if it > 0 else [],
                    "grad_norm": grad_norm,
                    "eta": eta,
                    "whitening_error": float(jnp.linalg.norm(
                        _compute_Sigma_U(experiment_datasets, mean_estimate, U_fourier, batch_size,
                                         disc_type_mean=disc_type_mean, disc_type=disc_type)[0] - jnp.eye(q)
                    )),
                    "elapsed_s": time.time() - t0,
                    "converged": True,
                })
                break
        ll_prev = ll

        # ---- B-step (exact EM) ----
        B = _ensure_psd(_sym(jnp.mean(jnp.array(T_all), axis=0) + eps * jnp.eye(q)), eps)
        B_eigvals = np.sort(np.linalg.eigvalsh(np.array(B)))[::-1]
        logger.info(f"  B-step: eigenvalues={B_eigvals}")

        # ---- U-step ----
        grad_U = compute_U_gradient_via_backprojection(
            experiment_datasets, mean_estimate, U_fourier,
            T_all, m_all, batch_size, volume_shape,
            disc_type_mean=disc_type_mean, disc_type=disc_type,
        )

        if lambda_U > 0:
            grad_U = grad_U + 2.0 * lambda_U * _apply_laplacian_sq_fourier(U_fourier, volume_shape)

        U_real_flat = _U_fourier_flat_to_real(U_fourier, volume_shape).reshape(q, -1).T
        grad_real_flat = _U_fourier_flat_to_real(grad_U, volume_shape).reshape(q, -1).T

        grad_norm = float(jnp.linalg.norm(grad_real_flat))
        logger.info(f"  U-step: ||grad||={grad_norm:.6e}")

        if grad_norm < 1e-12:
            logger.info("  Gradient near zero, skipping U-step.")
        else:
            # Normalize direction for scale-invariant step size
            descent_dir = grad_real_flat / grad_norm

            # Backtracking line search
            current_eta = eta
            accepted = False
            for bt in range(12):
                U_tmp = U_real_flat - current_eta * descent_dir

                # Retract: QR most iterations, full whitening periodically
                do_full_whiten = ((it + 1) % whiten_every == 0) or (it == n_iters - 1)

                if do_full_whiten and bt == 0:
                    U_fourier_tmp = _U_real_to_fourier_flat(
                        jnp.array(U_tmp.T.reshape(q, *volume_shape), dtype=jnp.complex128),
                        volume_shape,
                    )
                    Sigma_tmp, _ = _compute_Sigma_U(
                        experiment_datasets, mean_estimate, U_fourier_tmp, batch_size,
                        disc_type_mean=disc_type_mean, disc_type=disc_type,
                    )
                    U_fourier_new = _whitening_retraction(U_fourier_tmp, Sigma_tmp)
                    logger.info("    Full whitening retraction applied.")
                else:
                    U_real_new = qr_retraction(U_tmp)
                    U_fourier_new = _U_real_to_fourier_flat(
                        jnp.array(U_real_new.T.reshape(q, *volume_shape), dtype=jnp.complex128),
                        volume_shape,
                    )
                    if bt == 0:
                        logger.info("    QR retraction applied.")

                _, _, ll_new, _ = _e_step_full(
                    experiment_datasets, mean_estimate, U_fourier_new, B,
                    batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
                    compute_ll=True,
                )

                if ll_new > ll:
                    U_fourier = U_fourier_new
                    ll = ll_new
                    eta = min(current_eta * 1.5, 1.0)
                    accepted = True
                    logger.info(f"    Accepted: eta={current_eta:.2e}, ll={ll_new:.6f}, bt={bt}")
                    break
                current_eta *= 0.5

            if not accepted:
                logger.info(f"    Rejected U-step after backtracking (final eta={current_eta:.2e}).")
                eta = max(current_eta, 1e-10)

        # Monitor whitening constraint violation
        Sigma_now, _ = _compute_Sigma_U(
            experiment_datasets, mean_estimate, U_fourier, batch_size,
            disc_type_mean=disc_type_mean, disc_type=disc_type,
        )
        wh_err = float(jnp.linalg.norm(Sigma_now - jnp.eye(q)))

        elapsed = time.time() - t0
        logger.info(f"  Iteration {it + 1} done in {elapsed:.1f}s, ll={ll:.6f}, ||Sigma-I||={wh_err:.6e}")

        history.append({
            "iter": it + 1,
            "ll": ll,
            "B_eigenvalues": B_eigvals.tolist(),
            "grad_norm": grad_norm,
            "eta": eta,
            "whitening_error": wh_err,
            "elapsed_s": elapsed,
            "converged": False,
        })

    # Final output
    U_final = _U_fourier_flat_to_real(U_fourier, volume_shape)
    m_final, _, _, _ = _e_step_full(
        experiment_datasets, mean_estimate, U_fourier, B,
        batch_size, disc_type_mean=disc_type_mean, disc_type=disc_type,
        compute_ll=False,
    )

    return np.array(U_final), np.array(B), np.array(m_final), history


# =============================================================================
# High-level entry points (PipelineOutput -> result dir)
# =============================================================================


def _load_iterative_inputs(po, zdim):
    """Extract inputs for iterative methods from a PipelineOutput."""
    volume_shape = tuple(po.params["volume_shape"])
    dataset = po.get("dataset")
    if zdim is None:
        zdim = len(po._list_saved_eigenvector_indices())
    zdim = min(zdim, len(po._list_saved_eigenvector_indices()))
    mean_fourier = po.get("mean")
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)
    U_init = po.get_u_real(zdim)
    s_ppca = np.asarray(po.params["s"][:zdim], dtype=np.float64)
    B_init = np.diag(s_ppca)
    return dataset, mean_for_slicing, U_init, B_init, zdim


def _build_embeddings_from_mB(m_final, B_final, actual_zdim):
    """Convert posterior means + B to PipelineOutput embeddings."""
    eigenvalues = np.sort(np.linalg.eigvalsh(B_final))[::-1]
    new_s = eigenvalues.astype(np.float32)
    eigh_vals, eigh_vecs = np.linalg.eigh(B_final)
    idx = np.argsort(eigh_vals)[::-1]
    eigh_vals, eigh_vecs = eigh_vals[idx], eigh_vecs[:, idx]
    m_rotated = m_final @ eigh_vecs
    latent_coords = (m_rotated * np.sqrt(np.maximum(eigh_vals, 0.0))[None, :]).astype(np.float32)
    n_images = m_final.shape[0]
    precision_all = np.tile(np.eye(actual_zdim, dtype=np.float32), (n_images, 1, 1))
    embeddings_dict = {
        "latent_coords": {actual_zdim: latent_coords},
        "latent_coords_noreg": {actual_zdim: latent_coords},
        "latent_precision": {actual_zdim: precision_all},
        "latent_precision_noreg": {actual_zdim: precision_all},
        "contrasts": {actual_zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {actual_zdim: np.ones(n_images, dtype=np.float32)},
    }
    return new_s, embeddings_dict


def _save_and_embed(po, output_dir, U_final, B_final, actual_zdim, batch_size, method_info):
    """Shared logic: save U/s, then recompute embeddings via compute_per_image_Gi_hi.

    This avoids the z-space vs α-space confusion by recomputing posteriors
    from scratch in the orthonormal U frame — the same path that works for refit_b.

    Convention: W = U sqrt(B). Do SVD(W) in Fourier space → orthonormal U with
    Fourier-unit-norm (= 1/√vol_size real norm) and s = singular_values².
    """
    from recovar.ppca.ppca_refit import (
        compute_per_image_Gi_hi,
        compute_embeddings_from_UB,
        create_postprocessed_result_dir,
    )

    # Build W = U sqrt(B) in Fourier space, do SVD for PPCA convention
    volume_shape = U_final.shape[1:]
    vol_size = int(np.prod(volume_shape))
    B_sqrt = _sqrtm_psd(B_final)

    # U to Fourier flat
    U_fourier = np.zeros((vol_size, actual_zdim), dtype=np.complex64)
    for j in range(actual_zdim):
        U_fourier[:, j] = ftu.get_dft3(U_final[j]).reshape(-1)

    W_fourier = U_fourier @ np.array(B_sqrt, dtype=np.complex64)
    U_svd, S_svd, Vt_svd = np.linalg.svd(W_fourier, full_matrices=False)
    new_s = (S_svd ** 2).astype(np.float32)

    # Convert SVD basis to real space with PPCA-convention norm (1/√vol_size)
    U_orth = np.zeros((actual_zdim, *volume_shape), dtype=np.float32)
    for j in range(actual_zdim):
        U_orth[j] = ftu.get_idft3(U_svd[:, j].reshape(volume_shape)).real.astype(np.float32)

    # Save the result dir with placeholder embeddings
    n_images = po.params.get("n_images", 100000)
    dummy_embeddings = {
        "latent_coords": {actual_zdim: np.zeros((n_images, actual_zdim), dtype=np.float32)},
        "latent_coords_noreg": {actual_zdim: np.zeros((n_images, actual_zdim), dtype=np.float32)},
        "latent_precision": {actual_zdim: np.tile(np.eye(actual_zdim, dtype=np.float32), (n_images, 1, 1))},
        "latent_precision_noreg": {actual_zdim: np.tile(np.eye(actual_zdim, dtype=np.float32), (n_images, 1, 1))},
        "contrasts": {actual_zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {actual_zdim: np.ones(n_images, dtype=np.float32)},
    }
    method_info["eigenvalues"] = new_s.tolist()
    create_postprocessed_result_dir(
        po.result_path.rstrip("/"), output_dir, U_orth, new_s, dummy_embeddings, method_info=method_info,
    )

    # Now reload the saved output and compute proper embeddings
    from recovar.output.output import PipelineOutput as PO
    po_new = PO(output_dir)
    G_all, h_all = compute_per_image_Gi_hi(po_new, batch_size=batch_size, zdim=actual_zdim)

    # B in the SVD eigenbasis = diag(new_s) since SVD(W) gives s = singular_values²
    B_diag = np.diag(new_s[:actual_zdim].astype(np.float64))
    latent_coords, precision_all = compute_embeddings_from_UB(G_all, h_all, B_diag)

    # Overwrite embeddings
    from recovar.output.output_paths import ResultPaths
    paths = ResultPaths(output_dir)
    import os
    zdim_dir = paths.embedding_zdim_dir(actual_zdim)
    os.makedirs(zdim_dir, exist_ok=True)
    np.save(os.path.join(zdim_dir, "latent_coords.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_coords_noreg.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision.npy"), precision_all.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision_noreg.npy"), precision_all.astype(np.float32))

    logger.info("Recomputed embeddings via compute_per_image_Gi_hi for %s", method_info.get("method"))
    return new_s


def run_stiefel_ub(po, output_dir, zdim=None, batch_size=128, n_outer=10, n_inner_u=3, lambda_U=0.0):
    """Run Algorithm 2: full alternating U/B on Stiefel manifold."""
    from recovar.ppca.ppca_refit import create_postprocessed_result_dir
    dataset, mean_for_slicing, U_init, B_init, actual_zdim = _load_iterative_inputs(po, zdim)
    logger.info("Running Stiefel U/B: q=%d, n_outer=%d, lambda_U=%.2e", actual_zdim, n_outer, lambda_U)
    U_final, B_final, m_final, history = alternating_UB_stiefel(
        dataset, mean_for_slicing, U_init, B_init,
        n_outer=n_outer, n_inner_U=n_inner_u, lambda_U=lambda_U, batch_size=batch_size,
    )
    method_info = {"method": "stiefel_ub", "n_outer": n_outer, "n_inner_u": n_inner_u,
                   "lambda_U": lambda_U, "zdim": actual_zdim}
    new_s = _save_and_embed(po, output_dir, U_final, B_final, actual_zdim, batch_size, method_info)
    return {"eigenvalues": new_s, "U": U_final, "B": B_final, "history": history}


def run_whitening_manifold_ub(po, output_dir, zdim=None, batch_size=128, n_iters=10, lambda_U=0.0):
    """Run Algorithm 3: whitening-manifold + explicit B."""
    dataset, mean_for_slicing, U_init, B_init, actual_zdim = _load_iterative_inputs(po, zdim)
    logger.info("Running Whitening Manifold U/B: q=%d, n_iters=%d, lambda_U=%.2e", actual_zdim, n_iters, lambda_U)
    U_final, B_final, m_final, history = whitening_manifold_UB(
        dataset, mean_for_slicing, U_init, B_init,
        n_iters=n_iters, lambda_U=lambda_U, batch_size=batch_size,
    )
    method_info = {"method": "whitening_manifold_ub", "n_iters": n_iters,
                   "lambda_U": lambda_U, "zdim": actual_zdim,
                   "B_whitened_eigenvalues": np.sort(np.linalg.eigvalsh(B_final))[::-1].tolist()}
    new_s = _save_and_embed(po, output_dir, U_final, B_final, actual_zdim, batch_size, method_info)
    return {"eigenvalues": new_s, "U": U_final, "B": B_final, "history": history}
