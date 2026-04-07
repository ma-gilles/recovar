"""Algorithm 4A/4B: Coordinate-aware regularization A/B test for PPCA.

Tests the hypothesis that PPCA eigenvalue spectrum shrinkage is caused by
regularizing in the wrong coordinate system.  The model is:

    x_i = mu + U alpha_i,    alpha_i ~ N(0, B),    U^T U = I_q

Algorithm 4 runs the same U/B alternation (Stiefel manifold EM) twice,
differing ONLY in the regularizer:

  4A  reg_mode="grid":     R(U) = lambda * ||L_g U||_F^2
      Fourier multiplier:  |k|^4

  4B  reg_mode="physical":  R(U) = lambda * ||L_v K^{-1} U||_F^2
      Fourier multiplier:  |k|^4 / K(k)^2  where K = sinc^2(kx)sinc^2(ky)sinc^2(kz)

where K is the trilinear interpolation envelope (per-axis sinc^2) and
L_g, L_v are 3D Laplacians on the grid / physical volume respectively.
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.core import linalg
from recovar.ppca.ppca import (
    E_M_step_batch_half,
    _e_step_half_inner,
    _iter_processed_batches_half,
    _normalize_experiment_datasets,
    _prepare_mean_estimate_for_slicing,
    _tri_size,
    batch_over_vol_slice_volume_half,
    unpack_tri_to_full,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fourier-space regularizer multipliers
# =============================================================================


def _compute_freq_squared_half(volume_shape):
    """Compute |k|^2 for each voxel in the half-volume Fourier representation.

    Uses unscaled integer frequency indices so that Nyquist = N/2.
    Returns a flat array of shape (half_volume_size,) in float64.
    """
    D1, D2, D3 = volume_shape
    kx = np.fft.fftshift(np.arange(D1) - D1 // 2).astype(np.float64)
    ky = np.fft.fftshift(np.arange(D2) - D2 // 2).astype(np.float64)
    kz = np.arange(D3 // 2 + 1).astype(np.float64)  # rfft axis
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    freq_sq = KX**2 + KY**2 + KZ**2
    return freq_sq.ravel()


def _compute_sinc4_trilinear_half(volume_shape):
    """Compute |sinc_trilinear(k)|^4 for each half-volume voxel.

    The trilinear interpolation kernel in Fourier space is:
        K(k) = sinc^2(kx/N) * sinc^2(ky/N) * sinc^2(kz/N)
    so  |K(k)|^2 = sinc^4(kx/N) * sinc^4(ky/N) * sinc^4(kz/N)
    and |K(k)|^4 = product of sinc^8 per axis (but we actually need
    |K|^4 = (sinc^2 * sinc^2 * sinc^2)^4... let me be careful).

    Actually K = sinc^2(kx/Nx) sinc^2(ky/Ny) sinc^2(kz/Nz)  (for order=1 trilinear).
    K^{-1} in Fourier = 1/K.
    |K^{-1}|^2 = 1/K^2.
    We need K^{-T} L^T L K^{-1} in Fourier = |k|^4 / K^2.
    Since K is real and positive, K^{-T} = K^{-1}, so the operator is |k|^4 * K^{-2}.
    K^2 = sinc^4(kx/N) * sinc^4(ky/N) * sinc^4(kz/N).

    Returns K^2 as flat array, shape (half_volume_size,).
    """
    D1, D2, D3 = volume_shape

    def _sinc_axis(n, length):
        """sinc(k/N) for shifted frequency axis of length n."""
        k = np.fft.fftshift(np.arange(n) - n // 2).astype(np.float64)
        x = k / length
        safe_x = np.where(np.abs(x) < 1e-12, 1.0, x)
        return np.where(np.abs(x) < 1e-12, 1.0, np.sin(np.pi * safe_x) / (np.pi * safe_x))

    def _sinc_rfft_axis(n, length):
        """sinc(k/N) for rfft axis [0, 1, ..., n//2]."""
        k = np.arange(n // 2 + 1).astype(np.float64)
        x = k / length
        safe_x = np.where(np.abs(x) < 1e-12, 1.0, x)
        return np.where(np.abs(x) < 1e-12, 1.0, np.sin(np.pi * safe_x) / (np.pi * safe_x))

    sx = _sinc_axis(D1, D1)
    sy = _sinc_axis(D2, D2)
    sz = _sinc_rfft_axis(D3, D3)

    SX, SY, SZ = np.meshgrid(sx, sy, sz, indexing="ij")
    # K = sinc^2 per axis (order=1 trilinear), so K = SX^2 * SY^2 * SZ^2
    # K^2 = SX^4 * SY^4 * SZ^4
    K_sq = (SX * SY * SZ) ** 4  # K^2 = (sinc_x^2 sinc_y^2 sinc_z^2)^2
    return K_sq.ravel()


def compute_reg_multiplier(volume_shape, reg_mode, clip_max=1e8):
    """Compute the Fourier-space multiplier for the regularizer gradient.

    For reg_mode="grid":     multiplier = |k|^4
    For reg_mode="physical": multiplier = |k|^4 / K^2

    where K = sinc^2 trilinear kernel.

    The gradient of the regularizer is:
        grad_U R = 2 * lambda_U * IDFT[ multiplier * DFT[U] ]

    Parameters
    ----------
    volume_shape : tuple of int
        (D1, D2, D3) volume dimensions.
    reg_mode : str
        "grid" or "physical".
    clip_max : float
        Maximum value for the multiplier (clips 1/K^2 near Nyquist).

    Returns
    -------
    multiplier : ndarray, shape (half_volume_size,), float64
        The Fourier multiplier.
    """
    freq_sq = _compute_freq_squared_half(volume_shape)
    freq_fourth = freq_sq**2

    if reg_mode == "grid":
        return np.clip(freq_fourth, 0, clip_max)
    elif reg_mode == "physical":
        K_sq = _compute_sinc4_trilinear_half(volume_shape)
        # multiplier = |k|^4 / K^2, with clipping to avoid blow-up at Nyquist
        multiplier = np.where(K_sq > 1e-16, freq_fourth / K_sq, 0.0)
        return np.clip(multiplier, 0, clip_max)
    else:
        raise ValueError(f"Unknown reg_mode '{reg_mode}', must be 'grid' or 'physical'")


# =============================================================================
# Convergence history
# =============================================================================


@dataclass
class ConvergenceRecord:
    """Single iteration's diagnostics."""
    iteration: int
    neg_ll: float
    grad_norm: float
    B_eigenvalues: np.ndarray
    reg_value: float
    wall_time: float


@dataclass
class ConvergenceHistory:
    """Collects per-iteration diagnostics."""
    records: list = field(default_factory=list)

    def append(self, rec: ConvergenceRecord):
        self.records.append(rec)

    def log_latest(self):
        r = self.records[-1]
        eigs_str = ", ".join(f"{v:.4e}" for v in r.B_eigenvalues[:5])
        logger.info(
            f"  iter {r.iteration:3d} | -LL {r.neg_ll:12.2f} | "
            f"||grad|| {r.grad_norm:.4e} | reg {r.reg_value:.4e} | "
            f"B_eigs [{eigs_str}] | {r.wall_time:.1f}s"
        )

    def log_latest_with_step(self, effective_step):
        r = self.records[-1]
        eigs_str = ", ".join(f"{v:.4e}" for v in r.B_eigenvalues[:5])
        logger.info(
            f"  iter {r.iteration:3d} | -LL {r.neg_ll:12.2f} | "
            f"||grad|| {r.grad_norm:.4e} | step {effective_step:.4e} | "
            f"reg {r.reg_value:.4e} | "
            f"B_eigs [{eigs_str}] | {r.wall_time:.1f}s"
        )


# =============================================================================
# Stiefel manifold utilities
# =============================================================================


def _project_stiefel_tangent(U, G):
    """Project Euclidean gradient G onto the tangent space of the Stiefel manifold at U.

    U: (q, vol_size) with U @ U^T = I_q  (rows are orthonormal basis vectors)
    G: (q, vol_size) Euclidean gradient

    Tangent projection:  G_tan = G - U @ sym(U^T G)
    where sym(A) = (A + A^T) / 2.

    Note: U and G here are real-space volumes flattened to (q, vol_size).
    The inner product is the standard Euclidean one.
    """
    UG = U @ G.T  # (q, q)
    sym_UG = 0.5 * (UG + UG.T)
    return G - sym_UG @ U


def _retract_qr(U, xi, step_size):
    """QR retraction on Stiefel manifold.

    U_new = qr(U + step_size * xi).Q
    Ensures U_new^T U_new = I.
    """
    Y = U + step_size * xi
    Q, R = np.linalg.qr(Y.T, mode="reduced")  # Y^T = (vol_size, q), Q: (vol_size, q)
    # Fix sign convention: make diagonal of R positive
    signs = np.sign(np.diag(R))
    signs = np.where(signs == 0, 1.0, signs)
    Q = Q * signs[None, :]
    return Q.T  # (q, vol_size)


# =============================================================================
# Core: E-step (compute posterior moments) using existing half-spectrum code
# =============================================================================


def _run_estep(
    dataset_list,
    mean_estimate,
    mean_estimate_raw,
    W_half,
    batch_size,
    volume_shape,
    disc_type_mean="cubic",
    disc_type="linear_interp",
):
    """Run E-step over all datasets, accumulating posteriors and backprojection stats.

    W_half: (half_vol, q) half-volume Fourier basis = CTF-weighted loadings.
            In our case this is DFT(U) * sqrt(B eigenvalues) reshaped.

    Returns
    -------
    expected_zs : (N, q) posterior means
    second_moment_zs : (N, q, q) posterior second moments
    lhs_summed : (half_vol, tri_sz) accumulated LHS for M-step gradient
    rhs_summed : (half_vol, q) accumulated RHS for M-step gradient
    neg_ll : float, negative log-likelihood
    """
    ref = dataset_list[0]
    basis_size = W_half.shape[1]
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    tri_sz = _tri_size(basis_size)

    mean_for_slicing = _prepare_mean_estimate_for_slicing(
        mean_estimate, mean_estimate_raw, volume_shape, disc_type_mean,
    )

    lhs_summed = jnp.zeros((half_volume_size, tri_sz), dtype=ref.dtype_real)
    rhs_summed = jnp.zeros((half_volume_size, basis_size), dtype=ref.dtype)
    ll_sum = jnp.array(0.0, dtype=ref.dtype)

    all_ez = []
    all_smz = []

    for ds in dataset_list:
        for batch_half, ctf_params, rotation_matrices, translations, batch_idx in _iter_processed_batches_half(
            ds, batch_size
        ):
            noise_variance_half = ds.noise.get_half(batch_idx)
            lhs_summed, rhs_summed, ez, smz, ll_batch, _, _mc = E_M_step_batch_half(
                batch_half,
                lhs_summed,
                rhs_summed,
                mean_for_slicing,
                W_half,
                ctf_params,
                rotation_matrices,
                translations,
                ds.image_shape,
                ds.volume_shape,
                ds.grid_size,
                ds.voxel_size,
                noise_variance_half,
                ds.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=True,
            )
            all_ez.append(np.array(ez))
            all_smz.append(np.array(smz))
            ll_sum += ll_batch

    expected_zs = np.concatenate(all_ez, axis=0)
    second_moment_zs = np.concatenate(all_smz, axis=0)
    neg_ll = float(-ll_sum.real)

    return expected_zs, second_moment_zs, lhs_summed, rhs_summed, neg_ll


# =============================================================================
# Core: Euclidean gradient of Q(U) via backprojection statistics
# =============================================================================


def _compute_euclidean_gradient_from_stats(
    lhs_summed, rhs_summed, U_half, B, volume_shape, n_images,
):
    """Compute the Euclidean gradient of Q(U) + const from the accumulated stats.

    The EM surrogate Q(U) (up to constants) can be written in terms of the
    sufficient statistics accumulated during the E-step. The key insight is that
    the existing E-step accumulates:

        LHS[v] = sum_i  CTF_i^2(v) * E[z z^T | y_i]   (per-voxel, upper-tri packed)
        RHS[v] = sum_i  CTF_i(v) * (y_i - A mu_i) * E[z | y_i]^T

    The PPCA M-step solves  LHS[v] * W[v] = RHS[v]  per voxel.

    In our Stiefel formulation W = U @ sqrt(B), so:
        grad_U Q = [LHS * (U @ B) - RHS] @ sqrt(B)^T  (in Fourier, per voxel)

    But we need the gradient in real space for the Stiefel update. We compute:
        1. grad_W Q in Fourier (per-voxel from LHS/RHS)
        2. Convert W-gradient to U-gradient: grad_U = grad_W @ sqrt(B)^T
        3. IDFT to real space

    Parameters
    ----------
    lhs_summed : (half_vol, tri_sz)
    rhs_summed : (half_vol, q)
    U_half : (half_vol, q) = DFT_real(U), the half-volume Fourier of orthonormal basis
    B : (q, q) latent covariance
    volume_shape : tuple
    n_images : int

    Returns
    -------
    grad_U_real : (q, *volume_shape) real-space Euclidean gradient of Q w.r.t. U
    """
    q = B.shape[0]
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))

    # sqrt(B) via eigendecomposition (B is symmetric PSD)
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 0)
    sqrt_B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T  # (q, q)
    sqrt_B_j = jnp.array(sqrt_B, dtype=jnp.float64)

    # W_half = U_half @ sqrt(B)^T  in Fourier
    W_half = U_half @ sqrt_B_j.T  # (half_vol, q)

    # Compute grad_W per voxel:  grad_W[v] = LHS[v] @ W[v] - RHS[v]
    # LHS is upper-tri packed, need to unpack
    _CHUNK = 200_000
    grad_W_parts = []
    for i0 in range(0, half_volume_size, _CHUNK):
        i1 = min(i0 + _CHUNK, half_volume_size)
        lhs_full = unpack_tri_to_full(lhs_summed[i0:i1], q)  # (chunk, q, q)
        w_chunk = W_half[i0:i1]  # (chunk, q)
        rhs_chunk = rhs_summed[i0:i1]  # (chunk, q)
        # grad_W = LHS @ W - RHS, shape (chunk, q)
        gw = jnp.einsum("vij,vj->vi", lhs_full, w_chunk) - rhs_chunk
        grad_W_parts.append(gw)
    grad_W = jnp.concatenate(grad_W_parts, axis=0)  # (half_vol, q)

    # grad_U = grad_W @ sqrt(B)^T  (chain rule for W = U @ sqrt(B)^T)
    grad_U_half = grad_W @ sqrt_B_j  # (half_vol, q)

    # IDFT to real space: each column is a q-component
    grad_U_real = np.zeros((q, *volume_shape), dtype=np.float64)
    for k in range(q):
        col = np.array(grad_U_half[:, k]).reshape(half_volume_shape)
        grad_U_real[k] = np.array(ftu.get_idft3_real(jnp.array(col), volume_shape)).real

    return grad_U_real


# =============================================================================
# Core: Regularizer gradient in real space
# =============================================================================


def _compute_reg_gradient(U_real, reg_multiplier, volume_shape):
    """Compute the real-space gradient of the regularizer R(U).

    grad_U R = 2 * IDFT[ multiplier * DFT[U] ]

    (The lambda_U factor is applied outside this function.)

    Parameters
    ----------
    U_real : (q, *volume_shape) real-space orthonormal basis
    reg_multiplier : (half_vol_size,) Fourier multiplier
    volume_shape : tuple

    Returns
    -------
    grad_reg_real : (q, *volume_shape) real-space gradient
    """
    q = U_real.shape[0]
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    reg_mult = jnp.array(reg_multiplier.reshape(half_volume_shape))
    grad_reg = np.zeros_like(U_real)

    for k in range(q):
        U_k_ft = ftu.get_dft3_real(jnp.array(U_real[k]))  # half-volume Fourier
        reg_ft = reg_mult * U_k_ft
        grad_reg[k] = 2.0 * np.array(ftu.get_idft3_real(reg_ft, volume_shape)).real

    return grad_reg


def _compute_reg_value(U_real, reg_multiplier, volume_shape):
    """Compute the regularizer value: R(U) = ||sqrt(multiplier) * DFT[U]||_F^2.

    This equals tr(U^T Gamma U) where Gamma is the operator with the given
    Fourier multiplier.
    """
    q = U_real.shape[0]
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    reg_mult = jnp.array(reg_multiplier.reshape(half_volume_shape))

    # For rfft, we need to account for Hermitian symmetry weights
    w_1d = linalg.half_spectrum_last_axis_weights(volume_shape[2])
    w_3d = jnp.ones(half_volume_shape)
    w_3d = w_3d.at[:, :, :].set(w_3d * w_1d[None, None, :])

    total = 0.0
    for k in range(q):
        U_k_ft = ftu.get_dft3_real(jnp.array(U_real[k]))
        # |DFT[u]|^2 * multiplier, weighted for rfft
        total += float(jnp.sum(w_3d * reg_mult * jnp.abs(U_k_ft) ** 2).real)
    return total


# =============================================================================
# Main algorithm: refit_UB_with_regularization
# =============================================================================


def refit_UB_with_regularization(
    experiment_datasets,
    mean_estimate,
    U_init,
    B_init,
    reg_mode="grid",
    lambda_U=1e-4,
    n_outer=10,
    n_inner_U=3,
    batch_size=128,
    eps=1e-8,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    step_size=1e-3,
    mean_estimate_raw=None,
    volume_mask=None,
    n_burn_in_B=5,
):
    """Run coordinate-aware regularized PPCA refit (Algorithm 4A/4B).

    Parameters
    ----------
    experiment_datasets : CryoEMDataset or list of CryoEMDataset
        The cryo-EM dataset(s).
    mean_estimate : ndarray
        Mean volume in Fourier space (or cubic spline coefficients).
    U_init : (q, *volume_shape) real-space orthonormal basis from PPCA
        Must satisfy U_init @ U_init^T approx I_q.
    B_init : (q, q) initial latent covariance
    reg_mode : str
        "grid" for Algorithm 4A or "physical" for Algorithm 4B.
    lambda_U : float
        Regularization weight.
    n_outer : int
        Number of outer EM iterations.
    n_inner_U : int
        Number of Stiefel retraction steps per U-update.
    batch_size : int
        Batch size for E-step.
    eps : float
        Small constant for numerical stability.
    disc_type_mean : str
        Interpolation type for mean projection.
    disc_type : str
        Interpolation type for basis slicing.
    step_size : float
        Target step size for Stiefel gradient descent (gradient is normalized
        so that effective step = step_size, regardless of gradient magnitude).
    mean_estimate_raw : ndarray or None
        Raw Fourier mean (needed if disc_type_mean="cubic").
    volume_mask : ndarray or None
        Real-space support mask.
    n_burn_in_B : int
        Number of B-only EM iterations before starting joint U,B updates.
        This stabilizes B so the U-gradient is on the right scale.

    Returns
    -------
    U_final : (q, *volume_shape) refined orthonormal basis
    B_final : (q, q) refined latent covariance
    posteriors : dict with 'expected_zs' (N, q) and 'second_moment_zs' (N, q, q)
    history : ConvergenceHistory
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    volume_shape = ref.volume_shape
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)

    q = U_init.shape[0]
    assert U_init.shape[1:] == tuple(volume_shape), (
        f"U_init shape {U_init.shape} does not match volume_shape {volume_shape}"
    )

    # Work in float64 for precision
    U_real = np.array(U_init, dtype=np.float64)
    B = np.array(B_init, dtype=np.float64)

    # ---- Rescale U to have unit real-space norm per row ----
    # The PPCA eigenvectors have Fourier-space unit norm, so their real-space
    # norm is 1/sqrt(N^3).  The Stiefel retraction (QR) normalizes rows to
    # unit norm.  If we don't rescale first, the retraction will change ||U||
    # by a factor of sqrt(N^3), destroying the model.
    #
    # We absorb the scale factor into B so that W = DFT(U) @ sqrt(B) is
    # unchanged:
    #   W = DFT(U_orig) @ sqrt(B_orig)
    #     = DFT(scale * U_norm) @ sqrt(B_orig)       [U_orig = scale * U_norm]
    #     = DFT(U_norm) @ (scale * sqrt(B_orig))
    #   => B_internal = scale^2 * B_orig
    row_norms = np.array([np.linalg.norm(U_real[k].ravel()) for k in range(q)])
    u_scale = np.mean(row_norms)  # uniform scale (all rows have ~same norm)
    if u_scale > 0:
        U_real = U_real / u_scale
        B = B * (u_scale ** 2)
        logger.info(
            f"Rescaled U by 1/{u_scale:.4e} (now ||U_row||~1); "
            f"B scaled by {u_scale**2:.4e}"
        )

    # Precompute regularizer multiplier
    reg_multiplier = compute_reg_multiplier(volume_shape, reg_mode)
    logger.info(
        f"Coordinate-regularized refit: mode={reg_mode}, lambda={lambda_U:.2e}, "
        f"n_outer={n_outer}, n_inner={n_inner_U}, q={q}, vol={volume_shape}"
    )
    logger.info(
        f"  Reg multiplier: min={reg_multiplier.min():.2e}, "
        f"max={reg_multiplier.max():.2e}, mean={reg_multiplier.mean():.2e}"
    )

    # If mean_estimate_raw is not provided and disc_type_mean is "cubic",
    # the caller should have already precomputed coefficients.
    if mean_estimate_raw is None:
        mean_estimate_raw = mean_estimate

    # Apply volume mask to initial U if provided
    if volume_mask is not None:
        mask = np.array(volume_mask, dtype=np.float64).reshape(volume_shape)
        U_real = U_real * mask[None]
        # Re-orthonormalize after masking
        U_flat = U_real.reshape(q, -1)
        Q, R = np.linalg.qr(U_flat.T, mode="reduced")
        signs = np.sign(np.diag(R))
        signs = np.where(signs == 0, 1.0, signs)
        U_real = (Q * signs[None, :]).T.reshape(q, *volume_shape)

    # Count total images
    n_images = sum(ds.n_images for ds in dataset_list)

    history = ConvergenceHistory()

    # ==================================================================
    # Phase 0: B burn-in — stabilize B before joint U,B updates.
    # The PPCA eigenvalues s can be far from the EM fixed-point B,
    # causing huge gradients on the first joint iteration.
    # ==================================================================
    if n_burn_in_B > 0:
        logger.info("")
        logger.info("--- B burn-in phase (%d iterations, U fixed) ---", n_burn_in_B)
    for burn_iter in range(n_burn_in_B):
        t0 = time.time()
        eigvals_B, eigvecs_B = np.linalg.eigh(B)
        eigvals_B = np.maximum(eigvals_B, eps)
        sqrt_B = eigvecs_B @ np.diag(np.sqrt(eigvals_B)) @ eigvecs_B.T

        U_half = np.zeros((int(np.prod(half_volume_shape)), q), dtype=np.complex64)
        for k in range(q):
            U_half[:, k] = np.array(
                ftu.get_dft3_real(jnp.array(U_real[k], dtype=jnp.float32))
            ).ravel()
        W_half = jnp.array(U_half, dtype=jnp.complex64) @ jnp.array(sqrt_B.T, dtype=jnp.float32)

        expected_zs, second_moment_zs, _, _, neg_ll = _run_estep(
            dataset_list, mean_estimate, mean_estimate_raw, W_half,
            batch_size, volume_shape,
            disc_type_mean=disc_type_mean, disc_type=disc_type,
        )
        mean_smz = np.mean(second_moment_zs, axis=0)
        B = sqrt_B @ mean_smz @ sqrt_B.T
        B = 0.5 * (B + B.T)

        B_eigs = np.sort(np.linalg.eigvalsh(B))[::-1]
        elapsed = time.time() - t0
        logger.info(
            "  burn-in %2d | -LL %12.2f | B_eigs [%s] | %.1fs",
            burn_iter, neg_ll,
            ", ".join(f"{v:.4e}" for v in B_eigs[:5]),
            elapsed,
        )

    # Print header for main iterations
    logger.info("")
    logger.info("=" * 100)
    logger.info(f"COORDINATE-REGULARIZED PPCA REFIT ({reg_mode.upper()})")
    logger.info("=" * 100)
    logger.info(
        f"{'Iter':>4} | {'NegLL':>12} | {'||grad||':>12} | {'step':>10} | {'Reg(U)':>12} | "
        f"{'B_eigs (top 5)':>40} | {'Time':>6}"
    )
    logger.info("-" * 110)

    for outer in range(n_outer):
        t0 = time.time()

        # ==============================================================
        # 1. E-step: compute posterior moments using W = U @ sqrt(B)
        # ==============================================================
        eigvals_B, eigvecs_B = np.linalg.eigh(B)
        eigvals_B = np.maximum(eigvals_B, eps)
        sqrt_B = eigvecs_B @ np.diag(np.sqrt(eigvals_B)) @ eigvecs_B.T

        U_half = np.zeros((int(np.prod(half_volume_shape)), q), dtype=np.complex64)
        for k in range(q):
            U_half[:, k] = np.array(
                ftu.get_dft3_real(jnp.array(U_real[k], dtype=jnp.float32))
            ).ravel()

        W_half = jnp.array(U_half, dtype=jnp.complex64) @ jnp.array(sqrt_B.T, dtype=jnp.float32)

        expected_zs, second_moment_zs, lhs_summed, rhs_summed, neg_ll = _run_estep(
            dataset_list,
            mean_estimate,
            mean_estimate_raw,
            W_half,
            batch_size,
            volume_shape,
            disc_type_mean=disc_type_mean,
            disc_type=disc_type,
        )

        # ==============================================================
        # 2. B-step: B = (1/N) * sum_i E[alpha_i alpha_i^T | y_i]
        # ==============================================================
        # E[alpha alpha^T | y] = second_moment_zs (in z-space)
        # Since W = U @ sqrt(B), z = sqrt(B)^{-1} alpha
        # => alpha = sqrt(B) z
        # => E[alpha alpha^T] = sqrt(B) E[zz^T] sqrt(B)^T
        mean_smz = np.mean(second_moment_zs, axis=0)  # (q, q)
        B_new = sqrt_B @ mean_smz @ sqrt_B.T
        B_new = 0.5 * (B_new + B_new.T)  # Symmetrize
        B = B_new

        # ==============================================================
        # 3. U-step: gradient descent on Q(U) + lambda * R(U)
        #            with Stiefel retraction
        # ==============================================================
        effective_step = 0.0
        for inner in range(n_inner_U):
            # Recompute U_half for current U_real
            U_half_cur = np.zeros((int(np.prod(half_volume_shape)), q), dtype=np.complex64)
            for k in range(q):
                U_half_cur[:, k] = np.array(
                    ftu.get_dft3_real(jnp.array(U_real[k], dtype=jnp.float32))
                ).ravel()

            # Euclidean gradient of Q(U) from backprojection stats
            grad_Q = _compute_euclidean_gradient_from_stats(
                lhs_summed, rhs_summed,
                jnp.array(U_half_cur, dtype=jnp.complex64),
                B, volume_shape, n_images,
            )

            # Regularizer gradient
            grad_R = _compute_reg_gradient(U_real, reg_multiplier, volume_shape)
            grad_R *= lambda_U

            # Total Euclidean gradient
            grad_total = grad_Q + grad_R

            # Apply mask if provided
            if volume_mask is not None:
                grad_total *= mask[None]

            # Project to Stiefel tangent space
            U_flat = U_real.reshape(q, -1)
            grad_flat = grad_total.reshape(q, -1)
            xi = _project_stiefel_tangent(U_flat, grad_flat)

            # Normalize the tangent vector so the step is controlled.
            # Without normalization, the first iteration's gradient can be
            # 1e8+ which overwhelms the QR retraction and randomizes U.
            xi_norm = float(np.linalg.norm(xi))
            if xi_norm > 1e-12:
                xi_normalized = xi / xi_norm
                effective_step = step_size  # step_size controls the actual step magnitude
            else:
                xi_normalized = xi
                effective_step = 0.0

            # QR retraction (negative normalized gradient for descent)
            U_flat_new = _retract_qr(U_flat, -xi_normalized, effective_step)
            U_real = U_flat_new.reshape(q, *volume_shape)

            # Apply mask after retraction
            if volume_mask is not None:
                U_real *= mask[None]
                # Re-orthonormalize
                U_flat = U_real.reshape(q, -1)
                Q_qr, R_qr = np.linalg.qr(U_flat.T, mode="reduced")
                signs = np.sign(np.diag(R_qr))
                signs = np.where(signs == 0, 1.0, signs)
                U_real = (Q_qr * signs[None, :]).T.reshape(q, *volume_shape)

        # ==============================================================
        # 4. Diagnostics
        # ==============================================================
        grad_norm = float(np.linalg.norm(grad_total))
        reg_val = float(lambda_U * _compute_reg_value(U_real, reg_multiplier, volume_shape))
        B_eigs = np.sort(np.linalg.eigvalsh(B))[::-1]
        wall_time = time.time() - t0

        rec = ConvergenceRecord(
            iteration=outer,
            neg_ll=neg_ll,
            grad_norm=grad_norm,
            B_eigenvalues=B_eigs,
            reg_value=reg_val,
            wall_time=wall_time,
        )
        history.append(rec)
        history.log_latest_with_step(effective_step)

    # ==============================================================
    # Final E-step for posteriors
    # ==============================================================
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_B = np.maximum(eigvals_B, eps)
    sqrt_B = eigvecs_B @ np.diag(np.sqrt(eigvals_B)) @ eigvecs_B.T

    U_half_final = np.zeros((int(np.prod(half_volume_shape)), q), dtype=np.complex64)
    for k in range(q):
        U_half_final[:, k] = np.array(
            ftu.get_dft3_real(jnp.array(U_real[k], dtype=jnp.float32))
        ).ravel()
    W_half_final = jnp.array(U_half_final) @ jnp.array(sqrt_B.T, dtype=jnp.float32)

    expected_zs, second_moment_zs, _, _, final_neg_ll = _run_estep(
        dataset_list,
        mean_estimate,
        mean_estimate_raw,
        W_half_final,
        batch_size,
        volume_shape,
        disc_type_mean=disc_type_mean,
        disc_type=disc_type,
    )

    logger.info(f"\nFinal -LL: {final_neg_ll:.2f}")
    logger.info(f"Final B eigenvalues (internal): {np.sort(np.linalg.eigvalsh(B))[::-1]}")

    # ---- Reverse the U/B rescaling ----
    # Internal: U_internal (unit real-space norm), B_internal = scale^2 * B_orig
    # Output:   U_output = scale * U_internal,  B_output = B_internal / scale^2
    if u_scale > 0:
        U_real = U_real * u_scale
        B = B / (u_scale ** 2)
        sqrt_B = sqrt_B / u_scale
    logger.info(f"Final B eigenvalues (output scale): {np.sort(np.linalg.eigvalsh(B))[::-1]}")

    # Convert z-space posteriors to alpha-space: alpha = sqrt(B) z
    posteriors = {
        "expected_zs": expected_zs @ sqrt_B.T,  # (N, q) in alpha-space
        "second_moment_zs": second_moment_zs,  # keep in z-space for reference
        "expected_zs_z": expected_zs,  # z-space posteriors
        "sqrt_B": sqrt_B,
    }

    return (
        U_real.astype(np.float32),
        B.astype(np.float64),
        posteriors,
        history,
    )


# =============================================================================
# CLI entry point
# =============================================================================


def run_coord_reg_experiment(
    ppca_result_dir,
    output_dir,
    reg_mode,
    lambda_U=1e-4,
    n_outer=10,
    n_inner_U=3,
    zdim=None,
    batch_size=128,
    step_size=1e-3,
    n_burn_in_B=5,
):
    """Run coordinate-aware regularization experiment from a saved PipelineOutput.

    Parameters
    ----------
    ppca_result_dir : str
        Path to a recovar pipeline output directory.
    output_dir : str
        Where to save the results.
    reg_mode : str
        "grid" (4A) or "physical" (4B).
    lambda_U : float
        Regularization strength.
    n_outer : int
        Number of outer EM iterations.
    n_inner_U : int
        Number of Stiefel retraction steps per U-update.
    zdim : int or None
        Number of principal components to use. If None, uses all saved.
    batch_size : int
        Batch size for E-step.
    step_size : float
        Step size for Stiefel gradient descent (gradient is normalized).
    n_burn_in_B : int
        Number of B-only EM iterations before starting joint U,B updates.
    """
    import os

    from recovar.output.output import PipelineOutput

    logger.info(f"Loading pipeline output from {ppca_result_dir}")
    po = PipelineOutput(ppca_result_dir)

    volume_shape = tuple(po.params["volume_shape"])

    # Load U and eigenvalues
    if zdim is None:
        zdim = len(po._select_saved_eigenvector_indices(None))
    U_init = po.get_u_real(zdim)  # (zdim, *vol_shape)
    s_ppca = po.get("s")[:zdim]  # eigenvalues (variances)
    B_init = np.diag(s_ppca.astype(np.float64))

    # Load mean
    mean_fourier = po.get("mean")  # Fourier-space mean, flat
    mean_estimate_raw = mean_fourier

    logger.info(f"Loaded U: {U_init.shape}, B diag: {np.diag(B_init)[:5]}")

    # Load dataset
    # Load dataset (includes noise model)
    ds = po.get("dataset")

    # Volume mask
    volume_mask = None
    mask_path = os.path.join(ppca_result_dir, "model", "volume_mask.mrc")
    if os.path.isfile(mask_path):
        volume_mask = utils.load_mrc(mask_path)
        logger.info(f"Loaded volume mask from {mask_path}")

    # Run the refit
    U_final, B_final, posteriors, history = refit_UB_with_regularization(
        ds,
        mean_estimate_raw,
        U_init,
        B_init,
        reg_mode=reg_mode,
        lambda_U=lambda_U,
        n_outer=n_outer,
        n_inner_U=n_inner_U,
        batch_size=batch_size,
        step_size=step_size,
        mean_estimate_raw=mean_estimate_raw,
        volume_mask=volume_mask,
        n_burn_in_B=n_burn_in_B,
    )

    # Use the same recompute-from-scratch approach as refit_b:
    # save U, then compute G_i/h_i + embeddings in the proper frame.
    from recovar.ppca.ppca_refit_iterative import _save_and_embed
    from recovar.output.output import PipelineOutput as PO

    actual_zdim = U_final.shape[0]
    method_info = {
        "method": f"coord_reg_{reg_mode}",
        "reg_mode": reg_mode,
        "lambda_U": lambda_U,
        "n_outer": n_outer,
        "n_inner_U": n_inner_U,
        "zdim": actual_zdim,
    }

    po_src = PO(ppca_result_dir)
    new_s = _save_and_embed(po_src, output_dir, U_final, B_final, actual_zdim, batch_size, method_info)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  reg_mode={reg_mode}, lambda_U={lambda_U}")
    logger.info(f"  Final eigenvalues: {new_s[:5]}")

    return {"eigenvalues": new_s, "U": U_final, "B": B_final, "history": history}
