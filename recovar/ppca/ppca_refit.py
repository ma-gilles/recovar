"""Post-processing algorithms for PPCA: refit latent covariance B.

PPCA finds a good subspace (span of U) but underestimates the eigenvalue
spectrum, leading to suboptimal embeddings.  The routines here take an
existing PPCA result (basis U, eigenvalues S) and refit the latent
covariance B to improve embeddings.

Algorithms implemented:
    1. Fixed-span B refit via EM  (``refit_B_fixed_span``)
    5. Temperature diagnostic     (``fit_temperature_scalar``, ``fit_temperature_diagonal``)
"""

import copy
import logging
import os
import pickle
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.output.output_paths import ResultPaths

logger = logging.getLogger(__name__)

# Re-use the vmapped slice routine from ppca.py
batch_over_vol_slice_volume = jax.vmap(
    core.slice_volume, in_axes=(1, None, None, None, None), out_axes=1
)


# =============================================================================
# Shared infrastructure
# =============================================================================


def _load_mean_fourier(po):
    """Load mean volume from PipelineOutput in Fourier space, flat."""
    mean_real = utils.load_mrc(po.paths.mean_volume)
    return ftu.get_dft3(mean_real).reshape(-1)


def _load_u_fourier(po, zdim):
    """Load eigenvectors from PipelineOutput in Fourier space.

    Returns array of shape (vol_size, zdim) — column layout matching
    the convention used by ``batch_over_vol_slice_volume``.
    """
    u_real = po.get_u_real(zdim)  # (zdim, *volume_shape)
    n = u_real.shape[0]
    vol_size = int(np.prod(po.params["volume_shape"]))
    u_fourier = np.empty((vol_size, n), dtype=np.complex64)
    for i in range(n):
        u_fourier[:, i] = np.asarray(
            ftu.get_dft3(u_real[i]), dtype=np.complex64
        ).reshape(-1)
    return u_fourier


def _forward_model_from_map(
    volume, ctf_params, rotation_matrices, image_shape, volume_shape,
    voxel_size, ctf_evaluator, disc_type_mean, skip_ctf=False,
):
    """Project a Fourier volume through rotations and CTF."""
    slices = core.slice_volume(
        volume, rotation_matrices, image_shape, volume_shape, disc_type_mean,
    )
    if not skip_ctf:
        slices = slices * ctf_evaluator(ctf_params, image_shape, voxel_size)
    return slices


@jax.jit
def _compute_Gi_hi_batch(PU, centered_images):
    """Compute G_i and h_i for a batch of images.

    Args:
        PU: (n_batch, q, image_size) — noise-whitened projected eigenvectors
        centered_images: (n_batch, image_size) — noise-whitened centered images

    Returns:
        G_batch: (n_batch, q, q) — G_i = conj(PU) @ PU^T
        h_batch: (n_batch, q) — h_i = conj(PU) @ centered_image
    """
    # G_i = conj(PU) @ PU^T  -> (batch, q, q)
    G_batch = (jnp.conj(PU) @ PU.transpose(0, 2, 1)).real
    # h_i = conj(PU) @ y  -> (batch, q)
    h_batch = (jnp.conj(PU) @ centered_images[..., None]).real.squeeze(-1)
    return G_batch, h_batch


def compute_per_image_Gi_hi(pipeline_output, batch_size=128, zdim=None, disc_type="linear_interp",
                            apply_image_mask=True, apply_volume_mask=True, apply_gridding_correction=True):
    """Compute per-image sufficient statistics G_i and h_i.

    For each image i:
        G_i(U) = PU_i^T PU_i / sigma_i^2   (q x q)
        h_i(U) = PU_i^T r_i / sigma_i^2     (q,)

    where PU_i = CTF_i * A_i U (projected eigenvectors through rotation + CTF),
    and r_i = y_i - projected_mean, all in noise-whitened space.

    For parity with the baseline pipeline, by default:
      - apply_image_mask=True (applies dataset's image mask)
      - apply_volume_mask=True (multiplies U by the dilated volume mask in real space)
      - apply_gridding_correction=True (divides U by sinc² in real space)
    """
    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])
    image_shape = tuple(volume_shape[:2])
    voxel_size = float(po.params["voxel_size"])

    # Load dataset
    dataset = po.get("dataset")

    # Determine zdim
    if zdim is None:
        zdim = len(po._list_saved_eigenvector_indices())
    zdim = min(zdim, len(po._list_saved_eigenvector_indices()))

    # Load mean and eigenvectors
    mean_fourier = _load_mean_fourier(po)

    # Load eigenvectors as real-space, apply volume mask + gridding correction (matching baseline pipeline)
    u_real = po.get_u_real(zdim)  # (q, *vol_shape) real space
    if apply_volume_mask:
        try:
            dilated_mask = utils.load_mrc(po.paths.dilated_mask_volume)
            for k in range(zdim):
                u_real[k] = u_real[k] * dilated_mask
            logger.info("compute_per_image_Gi_hi: applied dilated volume mask to U")
        except Exception as e:
            logger.warning(f"compute_per_image_Gi_hi: could not apply volume mask: {e}")
    if apply_gridding_correction:
        try:
            from recovar.reconstruction.relion_functions import griddingCorrect_square
            for k in range(zdim):
                u_real[k] = np.asarray(griddingCorrect_square(jnp.array(u_real[k]), volume_shape[0], 1, order=1)[0])
            logger.info("compute_per_image_Gi_hi: applied gridding correction to U")
        except Exception as e:
            logger.warning(f"compute_per_image_Gi_hi: could not apply gridding correction: {e}")

    # Convert (masked, gridding-corrected) U back to Fourier
    vol_size = int(np.prod(volume_shape))
    U_fourier = np.zeros((vol_size, zdim), dtype=np.complex64)
    for j in range(zdim):
        U_fourier[:, j] = ftu.get_dft3(u_real[j]).reshape(-1)

    # Prepare mean for cubic slicing
    disc_type_mean = "cubic"
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)

    # Convert U to JAX
    U_jax = jnp.array(U_fourier)
    mean_jax = jnp.array(mean_for_slicing)

    q = zdim
    n_images = dataset.n_images
    G_all = np.zeros((n_images, q, q), dtype=np.float64)
    h_all = np.zeros((n_images, q), dtype=np.float64)

    logger.info(
        "Computing per-image G_i, h_i: n_images=%d, q=%d, batch_size=%d",
        n_images, q, batch_size,
    )
    t0 = time.time()
    n_processed = 0

    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in dataset.iter_batches(
        batch_size,
        by_image=not getattr(dataset, "tilt_series_flag", False),
    ):
        images = dataset.process_images(batch, apply_image_mask=apply_image_mask)
        noise_variance = dataset.noise.get(image_indices)

        # Translate and noise-whiten images
        images = core.translate_images(
            images, translations, image_shape,
        ) / jnp.sqrt(noise_variance)

        # CTF noise-whitened
        CTF = dataset.ctf_evaluator(
            ctf_params, image_shape, voxel_size,
        ) / jnp.sqrt(noise_variance)

        # Project mean through rotation + CTF (noise-whitened)
        projected_mean = _forward_model_from_map(
            mean_jax, ctf_params, rotation_matrices, image_shape, volume_shape,
            voxel_size, dataset.ctf_evaluator, disc_type_mean,
        ) / jnp.sqrt(noise_variance)

        centered_images = images - projected_mean

        # Project U through rotations: (n_batch, q, image_size)
        PU = batch_over_vol_slice_volume(
            U_jax, rotation_matrices, image_shape, volume_shape, disc_type,
        )
        # Apply CTF: CTF has shape (n_batch, image_size), PU is (n_batch, q, image_size)
        PU = PU * CTF[:, None, :]

        # Compute G_i and h_i
        G_batch, h_batch = _compute_Gi_hi_batch(PU, centered_images)

        # Store results — image_indices maps batch positions to dataset positions
        indices = np.asarray(image_indices)
        G_all[indices] = np.asarray(G_batch, dtype=np.float64)
        h_all[indices] = np.asarray(h_batch, dtype=np.float64)

        n_processed += len(indices)
        if n_processed % (batch_size * 10) == 0:
            logger.info("  processed %d / %d images", n_processed, n_images)

    elapsed = time.time() - t0
    logger.info("Computed G_i, h_i for %d images in %.1f s", n_processed, elapsed)

    return G_all, h_all


# =============================================================================
# Algorithm 1: Fixed-span B refit via EM
# =============================================================================


def refit_B_fixed_span(G_all, h_all, B_init=None, n_iters=20, eps=1e-8, rtol=1e-6):
    """Refit the latent covariance B by EM, keeping the subspace fixed.

    Given precomputed per-image sufficient statistics G_i and h_i,
    iterates the EM update:
        E-step: P_i = (B^{-1} + G_i)^{-1},  m_i = P_i h_i
        M-step: B = mean_i(P_i + m_i m_i^T) + eps * I

    After convergence, diagonalizes B = R Lambda R^T.

    Args:
        G_all: (n_images, q, q) per-image precision contributions.
        h_all: (n_images, q) per-image information vectors.
        B_init: Initial B matrix (q, q). Default: identity.
        n_iters: Maximum number of EM iterations.
        eps: Small ridge for numerical stability.
        rtol: Relative tolerance for early stopping on B change.

    Returns:
        eigenvalues: (q,) diagonal of Lambda (sorted descending).
        rotation: (q, q) rotation R such that B = R diag(eigenvalues) R^T.
        m_all: (n_images, q) posterior means in original U coordinates.
        B: (q, q) final covariance matrix.
        nll_history: list of negative log-likelihood values per iteration.
    """
    n, q, _ = G_all.shape
    if B_init is None:
        B = np.eye(q, dtype=np.float64)
    else:
        B = np.array(B_init, dtype=np.float64)

    nll_history = []

    for it in range(n_iters):
        t0 = time.time()
        B_inv = np.linalg.inv(B)

        # Vectorized E-step
        # P_all = inv(B_inv + G_i) for each image
        P_all = np.linalg.inv(B_inv[None] + G_all)  # (n, q, q)
        m_all = np.einsum("nij,nj->ni", P_all, h_all)  # (n, q)

        # Second moment: T_i = P_i + m_i m_i^T
        T_all = P_all + np.einsum("ni,nj->nij", m_all, m_all)  # (n, q, q)

        # M-step
        B_new = np.mean(T_all, axis=0) + eps * np.eye(q)
        # Symmetrize
        B_new = 0.5 * (B_new + B_new.T)

        # Compute NLL for monitoring:
        # nll_i = log|B| + log|B^{-1} + G_i| - h_i^T P_i h_i  (up to const)
        sign_B, logdet_B = np.linalg.slogdet(B)
        _, logdet_BinvG = np.linalg.slogdet(B_inv[None] + G_all)  # (n,)
        quad = np.einsum("ni,ni->n", h_all, m_all)  # h^T P h = h^T m
        nll = float(np.sum(logdet_B + logdet_BinvG - quad))
        nll_history.append(nll)

        # Check NLL monotonicity
        if len(nll_history) >= 2 and nll > nll_history[-2] + 1e-6 * abs(nll_history[-2]):
            logger.warning(
                "  EM iter %d: NLL increased %.6e -> %.6e (delta=%.3e)",
                it + 1, nll_history[-2], nll, nll - nll_history[-2],
            )

        B_change = np.linalg.norm(B_new - B)
        B_norm = np.linalg.norm(B)
        rel_change = B_change / max(B_norm, 1e-12)

        elapsed = time.time() - t0
        logger.info(
            "  EM iter %2d/%d: NLL=%.6e, |B_change|=%.3e (rel=%.3e)  (%.1f s)",
            it + 1, n_iters, nll, B_change, rel_change, elapsed,
        )

        B = B_new

        # Early stopping
        if rel_change < rtol:
            logger.info("  EM converged at iter %d (rel_change=%.3e < rtol=%.3e)", it + 1, rel_change, rtol)
            break

    # Final E-step with converged B
    B_inv = np.linalg.inv(B)
    P_all = np.linalg.inv(B_inv[None] + G_all)
    m_all = np.einsum("nij,nj->ni", P_all, h_all)

    # Self-consistency check: mean(T_i) should ≈ B (up to eps*I ridge)
    T_all = P_all + np.einsum("ni,nj->nij", m_all, m_all)
    mean_T = np.mean(T_all, axis=0)
    consistency_err = np.linalg.norm(mean_T - (B - eps * np.eye(q))) / max(np.linalg.norm(B), 1e-12)
    if consistency_err > 1e-4:
        logger.warning(
            "  Self-consistency check: ||mean(T_i) - (B - eps*I)|| / ||B|| = %.3e (expected ~0)",
            consistency_err,
        )
    else:
        logger.info("  Self-consistency check passed: relative error = %.3e", consistency_err)

    # Diagonalize B
    eigenvalues, rotation = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    rotation = rotation[:, idx]

    return eigenvalues, rotation, m_all, B, nll_history


def refit_B_fixed_span_with_embeddings(G_all, h_all, B_init=None, n_iters=20, eps=1e-8, rtol=1e-6):
    """Refit B and return embeddings in the covariance-PCA convention.

    Calls ``refit_B_fixed_span`` and then rotates posterior means into
    the eigenbasis of B, scaling to match the covariance-PCA convention
    where latent_coords ~ sqrt(eigenvalue) * z.

    Returns:
        new_eigenvalues: (q,) sorted descending.
        new_rotation: (q, q) rotation from old U basis to new eigenbasis.
        embeddings: (n_images, q) in covariance-PCA convention.
        B: (q, q) final covariance.
        nll_history: list of NLL per iteration.
    """
    eigenvalues, rotation, m_all, B, nll_history = refit_B_fixed_span(
        G_all, h_all, B_init=B_init, n_iters=n_iters, eps=eps, rtol=rtol,
    )

    # Rotate posterior means into B's eigenbasis
    # m_rotated = m @ R  (each row rotated)
    m_rotated = m_all @ rotation  # (n, q)

    # Scale by sqrt(eigenvalue) to match covariance-PCA convention
    embeddings = m_rotated * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]

    return eigenvalues, rotation, embeddings, B, nll_history


# =============================================================================
# Algorithm 5: Temperature diagnostic
# =============================================================================


def fit_temperature_scalar(G_all, h_all, B0, n_grid=200):
    """Find optimal scalar temperature tau such that B = tau * B0.

    Minimizes the negative marginal log-likelihood over a scalar tau:
        f(tau) = sum_i [q*log(tau) + log|B0_inv/tau + G_i| - h_i^T (B0_inv/tau + G_i)^{-1} h_i]

    Uses eigendecomposition of G_i to avoid repeated matrix inversions:
        G_i = V_i Lambda_i V_i^T  =>  (B0_inv/tau + G_i) in eigenbasis is
        V_i^T B0_inv V_i / tau + Lambda_i, which for diagonal B0 is still
        a general q x q matrix. We precompute V_i^T B0_inv V_i once.

    The grid search is adaptive: first coarse, then refined around minimum.

    Args:
        G_all: (n_images, q, q) per-image precision contributions.
        h_all: (n_images, q) per-image information vectors.
        B0: (q, q) diagonal matrix from PPCA eigenvalues.
        n_grid: Number of grid points for initial search.

    Returns:
        tau_opt: Optimal temperature scalar.
        tau_grid: Array of tau values evaluated.
        nll_grid: Array of NLL values at each tau.
    """
    n, q, _ = G_all.shape
    B0_inv = np.linalg.inv(B0)

    # Precompute eigendecomposition of each G_i for fast NLL evaluation.
    # G_i = V_i diag(lam_i) V_i^T
    # In the eigenbasis: B0_inv/tau + G_i -> V_i^T (B0_inv/tau) V_i + diag(lam_i)
    # We precompute V_i^T B0_inv V_i and V_i^T h_i.
    t_pre = time.time()
    G_eigvals, G_eigvecs = np.linalg.eigh(G_all)  # (n, q), (n, q, q)
    # Rotate B0_inv into each image's eigenbasis: (n, q, q)
    B0_inv_rotated = np.einsum("nki,kl,nlj->nij", G_eigvecs, B0_inv, G_eigvecs)
    # Rotate h into eigenbasis: (n, q)
    h_rotated = np.einsum("nki,nk->ni", G_eigvecs, h_all)
    logger.info("  Precomputed G_i eigendecompositions in %.1f s", time.time() - t_pre)

    def nll_at_tau(log_tau):
        tau = np.exp(log_tau)
        # M_rotated = B0_inv_rotated / tau + diag(G_eigvals)
        M_rotated = B0_inv_rotated / tau + np.eye(q)[None] * G_eigvals[:, None, :]  # (n, q, q)
        # For each image, solve M_rotated @ x = h_rotated and compute logdet
        # This is still q x q per image but the eigendecomposition is already done
        P_rotated = np.linalg.inv(M_rotated)  # (n, q, q)
        _, logdet_M = np.linalg.slogdet(M_rotated)  # (n,)
        m_rotated = np.einsum("nij,nj->ni", P_rotated, h_rotated)
        quad = np.einsum("ni,ni->n", h_rotated, m_rotated)
        nll = float(np.sum(q * log_tau + logdet_M - quad))
        return nll

    # Adaptive grid search:
    # Phase 1: Coarse grid over wide range
    n_coarse = min(n_grid, 50)
    log_tau_coarse = np.linspace(-4.0, 4.0, n_coarse)
    nll_coarse = np.array([nll_at_tau(lt) for lt in log_tau_coarse])
    best_coarse_idx = np.argmin(nll_coarse)
    log_tau_center = log_tau_coarse[best_coarse_idx]

    # Check if minimum is at boundary — expand if needed
    if best_coarse_idx == 0:
        logger.warning("Temperature minimum at lower boundary (tau=%.4f), expanding search to tau=%.4e",
                       np.exp(log_tau_coarse[0]), np.exp(-8.0))
        log_tau_coarse_ext = np.linspace(-8.0, log_tau_coarse[0], 20)
        nll_ext = np.array([nll_at_tau(lt) for lt in log_tau_coarse_ext])
        log_tau_coarse = np.concatenate([log_tau_coarse_ext, log_tau_coarse])
        nll_coarse = np.concatenate([nll_ext, nll_coarse])
        best_coarse_idx = np.argmin(nll_coarse)
        log_tau_center = log_tau_coarse[best_coarse_idx]
    elif best_coarse_idx == n_coarse - 1:
        logger.warning("Temperature minimum at upper boundary (tau=%.4f), expanding search to tau=%.4e",
                       np.exp(log_tau_coarse[-1]), np.exp(8.0))
        log_tau_coarse_ext = np.linspace(log_tau_coarse[-1], 8.0, 20)
        nll_ext = np.array([nll_at_tau(lt) for lt in log_tau_coarse_ext])
        log_tau_coarse = np.concatenate([log_tau_coarse, log_tau_coarse_ext])
        nll_coarse = np.concatenate([nll_coarse, nll_ext])
        best_coarse_idx = np.argmin(nll_coarse)
        log_tau_center = log_tau_coarse[best_coarse_idx]

    # Phase 2: Fine grid around coarse minimum
    n_fine = n_grid - len(log_tau_coarse)
    if n_fine > 0:
        half_width = max(0.5, (log_tau_coarse[1] - log_tau_coarse[0]) * 3) if len(log_tau_coarse) > 1 else 1.0
        log_tau_fine = np.linspace(log_tau_center - half_width, log_tau_center + half_width, n_fine)
        nll_fine = np.array([nll_at_tau(lt) for lt in log_tau_fine])
        log_tau_grid = np.concatenate([log_tau_coarse, log_tau_fine])
        nll_grid = np.concatenate([nll_coarse, nll_fine])
    else:
        log_tau_grid = log_tau_coarse
        nll_grid = nll_coarse

    # Sort by log_tau for clean output
    sort_idx = np.argsort(log_tau_grid)
    log_tau_grid = log_tau_grid[sort_idx]
    nll_grid = nll_grid[sort_idx]

    # Refine with scalar optimizer
    best_idx = np.argmin(nll_grid)
    bracket_lo = log_tau_grid[max(0, best_idx - 2)]
    bracket_hi = log_tau_grid[min(len(log_tau_grid) - 1, best_idx + 2)]
    result = scipy.optimize.minimize_scalar(
        nll_at_tau, bounds=(bracket_lo, bracket_hi), method="bounded",
    )
    log_tau_opt = result.x
    tau_opt = np.exp(log_tau_opt)

    tau_grid = np.exp(log_tau_grid)
    logger.info("Temperature scalar fit: tau_opt=%.4f (log=%.4f), NLL=%.6e", tau_opt, log_tau_opt, result.fun)

    return tau_opt, tau_grid, nll_grid


def fit_temperature_diagonal(G_all, h_all, B0, rho=0.01, maxiter=100):
    """Optimize per-component temperature D so that B = D B0 D.

    Optimizes log(d_j) where D = diag(exp(log_d)):
        nll = sum_i [log|B| + log|B^{-1} + G_i| - h_i^T (B^{-1} + G_i)^{-1} h_i]
              + rho * sum_j (log d_j)^2

    Uses L-BFGS-B with analytical gradients. For diagonal B0, B is diagonal
    with B_jj = d_j^2 * B0_jj. Writing x_j = log(d_j):

        dB_jj/dx_j     =  2 B_jj
        dB_inv_jj/dx_j = -2 B_inv_jj

    The gradient of the NLL w.r.t. x_j:
        df/dx_j = sum_i [2 - 2*B_inv_jj*P_i[j,j] - 2*B_inv_jj*m_i[j]^2]
                  + 2*rho*x_j

    where P_i = (B_inv + G_i)^{-1} and m_i = P_i h_i.

    Args:
        G_all: (n_images, q, q) per-image precision contributions.
        h_all: (n_images, q) per-image information vectors.
        B0: (q, q) diagonal matrix from PPCA eigenvalues.
        rho: Regularization strength penalizing deviation from d=1.
        maxiter: Maximum L-BFGS-B iterations.

    Returns:
        D_opt: (q,) optimal diagonal scaling factors.
        nll_opt: NLL at optimum.
        log_d_opt: (q,) optimal log(d_j).
    """
    n, q, _ = G_all.shape
    b0_diag = np.diag(B0)  # (q,) — B0 is diagonal

    def nll_and_grad(log_d):
        d = np.exp(log_d)
        b_diag = d ** 2 * b0_diag  # diagonal of B
        b_inv_diag = 1.0 / b_diag  # diagonal of B_inv

        B_inv = np.diag(b_inv_diag)
        logdet_B = np.sum(np.log(b_diag))

        M = B_inv[None] + G_all  # (n, q, q)
        P = np.linalg.inv(M)  # (n, q, q) — posterior covariance
        _, logdet_M = np.linalg.slogdet(M)  # (n,)
        m = np.einsum("nij,nj->ni", P, h_all)  # (n, q)
        quad = np.einsum("ni,ni->n", h_all, m)  # (n,)

        nll = float(np.sum(logdet_B + logdet_M - quad))
        nll += rho * float(np.sum(log_d ** 2))

        # Gradient: for diagonal B0, dB_inv[j,j]/d(log d_j) = -2 * B_inv[j,j]
        # dNLL/d(log d_j) = sum_i [ -(-2 B_inv_jj) * P_i[j,j] + (-2 B_inv_jj) * m_i[j]^2 ]
        #                   + n * (2 B_inv_jj * B_jj)  [= n * 2 from tr(B_inv dB)]
        #                   + 2 * rho * log_d_j
        #
        # Simplifying:
        # = sum_i [2 * B_inv_jj * P_i[j,j] - 2 * B_inv_jj * m_i[j]^2] + 2*n + 2*rho*log_d_j
        #
        # Wait, let's be more careful. The NLL terms involving B:
        #   f = n * log|B| + sum_i [log|B_inv + G_i| - h^T (B_inv + G_i)^{-1} h]
        #
        # df/d(log d_j) = n * tr(B_inv * dB/d(log d_j))
        #                 + sum_i tr((B_inv + G_i)^{-1} * dB_inv/d(log d_j))
        #                 - sum_i m_i^T (dB_inv/d(log d_j)) m_i
        #
        # For diagonal B0:
        #   dB/d(log d_j) has only [j,j] = 2 * B[j,j]
        #   dB_inv/d(log d_j) has only [j,j] = -2 * B_inv[j,j]
        #
        # So:
        #   n * B_inv[j,j] * 2 * B[j,j] = 2n
        #   sum_i P_i[j,j] * (-2 * B_inv[j,j])
        #   - sum_i m_i[j]^2 * (-2 * B_inv[j,j])
        #
        # = 2n - 2 * B_inv[j,j] * sum_i P_i[j,j] + 2 * B_inv[j,j] * sum_i m_i[j]^2

        P_diag_sum = np.sum(P[:, np.arange(q), np.arange(q)], axis=0)  # (q,) sum_i P_i[j,j]
        m_sq_sum = np.sum(m ** 2, axis=0)  # (q,) sum_i m_i[j]^2

        # df/dx_j = sum_i [2 - 2*B_inv_jj*P_i[j,j] - 2*B_inv_jj*m_i[j]^2] + 2*rho*x_j
        # The P_i[j,j] term comes from d(logdet M)/dx_j = tr(P_i dB_inv/dx_j)
        # The m_i[j]^2 term comes from d(-h^T P h)/dx_j = -2*B_inv_jj*m_j^2
        grad = 2.0 * n - 2.0 * b_inv_diag * (P_diag_sum + m_sq_sum)
        grad += 2.0 * rho * log_d

        return nll, grad

    log_d_init = np.zeros(q, dtype=np.float64)

    logger.info("Fitting diagonal temperature: q=%d, rho=%.4f, maxiter=%d", q, rho, maxiter)
    result = scipy.optimize.minimize(
        nll_and_grad,
        log_d_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "disp": False},
    )
    log_d_opt = result.x
    D_opt = np.exp(log_d_opt)

    logger.info(
        "Diagonal temperature fit: D_opt=%s, NLL=%.6e, converged=%s (nit=%d)",
        np.array2string(D_opt, precision=4), result.fun, result.success, result.nit,
    )
    return D_opt, result.fun, log_d_opt


# =============================================================================
# Embedding computation
# =============================================================================


def compute_embeddings_from_B(G_all, h_all, B):
    """Compute posterior mean embeddings given B and precomputed G_i, h_i.

    Args:
        G_all: (n_images, q, q)
        h_all: (n_images, q)
        B: (q, q) latent covariance matrix.

    Returns:
        m_all: (n_images, q) posterior means.
        P_all: (n_images, q, q) posterior precision (= inverse posterior cov).
    """
    B_inv = np.linalg.inv(B)
    posterior_cov = np.linalg.inv(B_inv[None] + G_all)  # (n, q, q)
    m_all = np.einsum("nij,nj->ni", posterior_cov, h_all)
    # Return precision = B_inv + G_i (not the covariance)
    precision_all = B_inv[None] + G_all
    return m_all, precision_all


def compute_embeddings_from_UB(G_all, h_all, B, scale_to_covariance_convention=True):
    """Compute embeddings in the eigenbasis of B, optionally scaling.

    This diagonalizes B = R Lambda R^T, rotates the posterior means
    into the eigenbasis, and optionally scales by sqrt(Lambda).

    Args:
        G_all: (n_images, q, q)
        h_all: (n_images, q)
        B: (q, q) latent covariance.
        scale_to_covariance_convention: If True, scale by sqrt(eigenvalues).

    Returns:
        latent_coords: (n_images, q)
        latent_precision: (n_images, q, q) in original U basis
    """
    m_all, precision_all = compute_embeddings_from_B(G_all, h_all, B)

    eigenvalues, rotation = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    rotation = rotation[:, idx]

    # Rotate to eigenbasis
    m_rotated = m_all @ rotation  # (n, q)

    if scale_to_covariance_convention:
        latent_coords = m_rotated * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    else:
        latent_coords = m_rotated

    return latent_coords, precision_all


# =============================================================================
# PipelineOutput creation
# =============================================================================


def create_postprocessed_result_dir(
    source_result_dir,
    output_dir,
    new_u_real,
    new_s,
    new_embeddings_dict,
    method_info=None,
):
    """Create a new result directory with postprocessed PPCA results.

    Copies essential files from the source PPCA result and writes new
    eigenvectors, eigenvalues, and embeddings.

    Args:
        source_result_dir: Path to original PPCA result directory.
        output_dir: Path for the new result directory.
        new_u_real: (q, *volume_shape) new eigenvectors in real space.
        new_s: (q,) new eigenvalues (rescaled).
        new_embeddings_dict: dict with keys matching _EMBEDDING_FIELDS, each
            mapping zdim -> array. E.g.:
            {
                'latent_coords': {10: array(n, 10)},
                'latent_precision': {10: array(n, 10, 10)},
                ...
            }
        method_info: Optional dict with postprocessing metadata.
    """
    src_paths = ResultPaths(source_result_dir)
    dst_paths = ResultPaths(output_dir)
    dst_paths.ensure_dirs()

    # ---- Copy/symlink shared files ----
    _copy_if_exists(src_paths.halfsets, dst_paths.halfsets)
    _copy_if_exists(src_paths.particles_halfsets, dst_paths.particles_halfsets)
    _copy_if_exists(src_paths.covariance_cols, dst_paths.covariance_cols)

    # Symlink mean and masks
    for attr in ("mean_volume", "mask_volume", "dilated_mask_volume"):
        src = getattr(src_paths, attr)
        dst = getattr(dst_paths, attr)
        if os.path.exists(src):
            _symlink_safe(src, dst)

    # ---- Update params ----
    src_params = utils.pickle_load(src_paths.params)
    new_params = copy.deepcopy(src_params)

    # Pad new_s to match original length if needed
    orig_s = np.asarray(new_params.get("s", np.zeros(0)))
    padded_s = np.zeros(max(len(orig_s), len(new_s)), dtype=np.float32)
    padded_s[: len(new_s)] = new_s
    new_params["s"] = padded_s

    if method_info is not None:
        new_params["ppca_refit_info"] = method_info

    utils.pickle_dump(new_params, dst_paths.params)

    # ---- Write eigenvector MRCs ----
    volume_shape = tuple(new_params["volume_shape"])
    voxel_size = float(new_params.get("voxel_size", 1.0))
    for i in range(new_u_real.shape[0]):
        path = dst_paths.eigenvector(i)
        utils.write_mrc(path, new_u_real[i].astype(np.float32), voxel_size=voxel_size)

    # ---- Write variance volume ----
    n_eigs = min(10, len(new_s))
    variance = np.zeros(volume_shape, dtype=np.float32)
    for i in range(n_eigs):
        variance += float(new_s[i]) * (new_u_real[i].astype(np.float32) ** 2)
    variance_path = dst_paths.variance(n_eigs)
    utils.write_mrc(variance_path, variance, voxel_size=voxel_size)

    # Also write variance_10 if n_eigs != 10
    if n_eigs != 10:
        variance10 = np.zeros(volume_shape, dtype=np.float32)
        for i in range(min(10, len(new_s), new_u_real.shape[0])):
            variance10 += float(new_s[i]) * (new_u_real[i].astype(np.float32) ** 2)
        utils.write_mrc(dst_paths.variance(10), variance10, voxel_size=voxel_size)

    # ---- Write embeddings ----
    _save_embeddings_per_zdim(dst_paths, new_embeddings_dict)

    logger.info("Created postprocessed result dir: %s", output_dir)


def _save_embeddings_per_zdim(paths, embedding_dict):
    """Save embeddings as per-zdim .npy files."""
    _EMBEDDING_FIELDS = [
        "latent_coords", "latent_coords_noreg",
        "latent_precision", "latent_precision_noreg",
        "contrasts", "contrasts_noreg",
    ]
    all_zdims = set()
    for field in _EMBEDDING_FIELDS:
        if field in embedding_dict and isinstance(embedding_dict[field], dict):
            all_zdims.update(embedding_dict[field].keys())

    for zdim in sorted(all_zdims):
        zdim_dir = paths.embedding_zdim_dir(zdim)
        os.makedirs(zdim_dir, exist_ok=True)
        for field in _EMBEDDING_FIELDS:
            if field in embedding_dict and zdim in embedding_dict[field]:
                arr = np.asarray(embedding_dict[field][zdim])
                np.save(os.path.join(zdim_dir, f"{field}.npy"), arr)

    logger.info("Saved embeddings for zdims %s", sorted(all_zdims))


def _copy_if_exists(src, dst):
    """Copy file if source exists."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _symlink_safe(src, dst):
    """Create a symlink, using absolute path for source."""
    src_abs = os.path.abspath(src)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src_abs, dst)


# =============================================================================
# High-level entry points
# =============================================================================


def run_refit_b(pipeline_output, output_dir, zdim=None, batch_size=128, n_iters=50, rtol=1e-6):
    """Run Algorithm 1: fixed-span B refit via EM.

    Args:
        pipeline_output: PipelineOutput object.
        output_dir: Path for the output result directory.
        zdim: Number of PCs to use. None = all saved.
        batch_size: Batch size for G_i/h_i computation.
        n_iters: Maximum number of EM iterations (uses early stopping).
        rtol: Relative tolerance for early stopping.

    Returns:
        dict with eigenvalues, rotation, embeddings, B, nll_history.
    """
    po = pipeline_output

    # Compute sufficient statistics
    G_all, h_all = compute_per_image_Gi_hi(po, batch_size=batch_size, zdim=zdim)
    q = G_all.shape[1]
    actual_zdim = q

    # Initialize B from PPCA eigenvalues (the scale of W = U diag(sqrt(S)))
    s_ppca = np.asarray(po.params["s"][:q], dtype=np.float64)
    B_init = np.diag(s_ppca)
    logger.info("Running B refit EM: q=%d, n_iters=%d, rtol=%.1e, B_init diag=%s",
                q, n_iters, rtol, np.array2string(s_ppca[:5], precision=2))
    eigenvalues, rotation, embeddings, B, nll_history = refit_B_fixed_span_with_embeddings(
        G_all, h_all, B_init=B_init, n_iters=n_iters, rtol=rtol,
    )

    # Compute precision for output
    _, precision_all = compute_embeddings_from_B(G_all, h_all, B)

    # Build new U_real: rotate original eigenvectors by R
    u_real = po.get_u_real(actual_zdim)  # (q, *vol_shape)
    volume_shape = u_real.shape[1:]
    # new_u_real[j] = sum_i R[i,j] * u_real[i]
    new_u_real = np.einsum("ij,i...->j...", rotation, u_real)

    new_s = eigenvalues.astype(np.float32)

    # Build embeddings dict
    n_images = embeddings.shape[0]
    embeddings_dict = {
        "latent_coords": {actual_zdim: embeddings.astype(np.float32)},
        "latent_coords_noreg": {actual_zdim: embeddings.astype(np.float32)},
        "latent_precision": {actual_zdim: precision_all.astype(np.float32)},
        "latent_precision_noreg": {actual_zdim: precision_all.astype(np.float32)},
        "contrasts": {actual_zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {actual_zdim: np.ones(n_images, dtype=np.float32)},
    }

    # Try to copy contrasts from source
    try:
        src_contrasts = po.get_embedding_component("contrasts", actual_zdim)
        embeddings_dict["contrasts"][actual_zdim] = np.asarray(src_contrasts, dtype=np.float32)
        src_contrasts_nr = po.get_embedding_component("contrasts_noreg", actual_zdim)
        embeddings_dict["contrasts_noreg"][actual_zdim] = np.asarray(src_contrasts_nr, dtype=np.float32)
    except (KeyError, Exception):
        pass

    method_info = {
        "method": "refit_b",
        "n_iters": n_iters,
        "zdim": actual_zdim,
        "eigenvalues": eigenvalues.tolist(),
        "nll_history": nll_history,
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        new_u_real,
        new_s,
        embeddings_dict,
        method_info=method_info,
    )

    return {
        "eigenvalues": eigenvalues,
        "rotation": rotation,
        "embeddings": embeddings,
        "B": B,
        "nll_history": nll_history,
    }


def run_temperature_scalar(pipeline_output, output_dir, zdim=None, batch_size=128, n_grid=200):
    """Run Algorithm 5: temperature scalar diagnostic.

    Args:
        pipeline_output: PipelineOutput object.
        output_dir: Path for the output result directory.
        zdim: Number of PCs. None = all saved.
        batch_size: Batch size for G_i/h_i computation.
        n_grid: Grid points for temperature search.

    Returns:
        dict with tau_opt, tau_grid, nll_grid, eigenvalues, embeddings.
    """
    po = pipeline_output

    G_all, h_all = compute_per_image_Gi_hi(po, batch_size=batch_size, zdim=zdim)
    q = G_all.shape[1]
    actual_zdim = q

    # Build B0 from PPCA eigenvalues
    s_ppca = np.asarray(po.params["s"][:q], dtype=np.float64)
    B0 = np.diag(s_ppca)

    logger.info("Running temperature scalar fit: q=%d", q)
    tau_opt, tau_grid, nll_grid = fit_temperature_scalar(G_all, h_all, B0, n_grid=n_grid)

    # Build final B and compute embeddings
    B = tau_opt * B0
    new_s = (tau_opt * s_ppca).astype(np.float32)

    latent_coords, precision_all = compute_embeddings_from_UB(G_all, h_all, B)

    # Eigenvectors stay the same (no rotation for scalar temperature)
    u_real = po.get_u_real(actual_zdim)

    n_images = latent_coords.shape[0]
    embeddings_dict = {
        "latent_coords": {actual_zdim: latent_coords.astype(np.float32)},
        "latent_coords_noreg": {actual_zdim: latent_coords.astype(np.float32)},
        "latent_precision": {actual_zdim: precision_all.astype(np.float32)},
        "latent_precision_noreg": {actual_zdim: precision_all.astype(np.float32)},
        "contrasts": {actual_zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {actual_zdim: np.ones(n_images, dtype=np.float32)},
    }
    try:
        src_c = po.get_embedding_component("contrasts", actual_zdim)
        embeddings_dict["contrasts"][actual_zdim] = np.asarray(src_c, dtype=np.float32)
        src_cnr = po.get_embedding_component("contrasts_noreg", actual_zdim)
        embeddings_dict["contrasts_noreg"][actual_zdim] = np.asarray(src_cnr, dtype=np.float32)
    except (KeyError, Exception):
        pass

    method_info = {
        "method": "temperature_scalar",
        "tau_opt": float(tau_opt),
        "zdim": actual_zdim,
        "eigenvalues_original": s_ppca.tolist(),
        "eigenvalues_scaled": new_s.tolist(),
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        u_real,
        new_s,
        embeddings_dict,
        method_info=method_info,
    )

    return {
        "tau_opt": tau_opt,
        "tau_grid": tau_grid,
        "nll_grid": nll_grid,
        "eigenvalues": new_s,
        "embeddings": latent_coords,
    }


def run_temperature_diagonal(pipeline_output, output_dir, zdim=None, batch_size=128, rho=0.01, maxiter=100):
    """Run Algorithm 5 variant: per-component diagonal temperature.

    Args:
        pipeline_output: PipelineOutput object.
        output_dir: Path for the output result directory.
        zdim: Number of PCs. None = all saved.
        batch_size: Batch size for G_i/h_i computation.
        rho: Regularization strength.
        maxiter: Max L-BFGS-B iterations.

    Returns:
        dict with D_opt, nll_opt, eigenvalues, embeddings.
    """
    po = pipeline_output

    G_all, h_all = compute_per_image_Gi_hi(po, batch_size=batch_size, zdim=zdim)
    q = G_all.shape[1]
    actual_zdim = q

    s_ppca = np.asarray(po.params["s"][:q], dtype=np.float64)
    B0 = np.diag(s_ppca)

    logger.info("Running diagonal temperature fit: q=%d, rho=%.4f", q, rho)
    D_opt, nll_opt, log_d_opt = fit_temperature_diagonal(G_all, h_all, B0, rho=rho, maxiter=maxiter)

    # Build final B = D B0 D
    D_mat = np.diag(D_opt)
    B = D_mat @ B0 @ D_mat
    new_s_diag = D_opt ** 2 * s_ppca  # diagonal of B = d_j^2 * s_j
    new_s = new_s_diag.astype(np.float32)

    latent_coords, precision_all = compute_embeddings_from_UB(G_all, h_all, B)

    u_real = po.get_u_real(actual_zdim)

    n_images = latent_coords.shape[0]
    embeddings_dict = {
        "latent_coords": {actual_zdim: latent_coords.astype(np.float32)},
        "latent_coords_noreg": {actual_zdim: latent_coords.astype(np.float32)},
        "latent_precision": {actual_zdim: precision_all.astype(np.float32)},
        "latent_precision_noreg": {actual_zdim: precision_all.astype(np.float32)},
        "contrasts": {actual_zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {actual_zdim: np.ones(n_images, dtype=np.float32)},
    }
    try:
        src_c = po.get_embedding_component("contrasts", actual_zdim)
        embeddings_dict["contrasts"][actual_zdim] = np.asarray(src_c, dtype=np.float32)
        src_cnr = po.get_embedding_component("contrasts_noreg", actual_zdim)
        embeddings_dict["contrasts_noreg"][actual_zdim] = np.asarray(src_cnr, dtype=np.float32)
    except (KeyError, Exception):
        pass

    method_info = {
        "method": "temperature_diagonal",
        "D_opt": D_opt.tolist(),
        "rho": float(rho),
        "zdim": actual_zdim,
        "eigenvalues_original": s_ppca.tolist(),
        "eigenvalues_scaled": new_s.tolist(),
        "nll_opt": float(nll_opt),
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        u_real,
        new_s,
        embeddings_dict,
        method_info=method_info,
    )

    return {
        "D_opt": D_opt,
        "nll_opt": nll_opt,
        "eigenvalues": new_s,
        "embeddings": latent_coords,
    }
