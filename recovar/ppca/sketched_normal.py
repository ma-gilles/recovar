"""Sketched normal-operator products for low-rank recovery.

Given X = U diag(s) V^T and whitened forward model A, the normal residual
gradient is G(X) = A*(A(X) - b).  This module computes S @ G(X) and
G(X) @ Q without forming the dense (volume_size x n_images) matrix G(X).

Public API:

    op = SketchedNormalOperator(cryo, mean_fourier, batch_size)
    left  = op.left_matvec(U, s, V, S)         # S @ G(X)
    right = op.right_matvec(U, s, V, Q)         # G(X) @ Q
    left, right = op.both_matvecs(U, s, V, S, Q)

Inputs U, S, Q are real-space.  Mean is Fourier (complex, from pipeline).
Fourier / half-volume conversion happens internally.

Key optimization: per_image_backproject gives (half_vol, batch) in one
CUDA call, then S @ bp and bp @ Q are matmuls — cost is O(batch)
backprojections regardless of sketch rank.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.ppca.ppca import batch_over_vol_slice_volume_half

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real ↔ half-volume Fourier conversion
# ---------------------------------------------------------------------------


def _real_vols_to_half_fourier(vols_real, volume_shape):
    """(*, vol_size) real → (*, half_vol_size) complex half-volume Fourier."""
    vols = np.asarray(vols_real)
    leading = vols.shape[:-1]
    flat = vols.reshape(-1, *volume_shape)
    ft = np.asarray(ftu.get_dft3(flat))  # (batch, *vol_shape) complex
    half = np.asarray(ftu.full_volume_to_half_volume(ft.reshape(-1, int(np.prod(volume_shape))), volume_shape))
    half_vol_size = half.shape[-1]
    return half.reshape(*leading, half_vol_size).astype(np.complex64)


def _half_fourier_to_real_vols(vols_half, volume_shape):
    """(*, half_vol_size) complex half-volume Fourier → (*, vol_size) real."""
    vols = np.asarray(vols_half)
    leading = vols.shape[:-1]
    half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    vol_size = int(np.prod(volume_shape))
    flat = vols.reshape(-1, half_vol_size)
    full = np.asarray(ftu.half_volume_to_full_volume(flat, volume_shape))
    real = np.asarray(ftu.get_idft3(full.reshape(-1, *volume_shape))).real
    return real.reshape(*leading, vol_size).astype(np.float32)


# ---------------------------------------------------------------------------
# JIT-compiled batch kernel (internal, half-volume)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=[10, 11, 12, 13, 14])
def _sketched_normal_batch(
    images_half,
    mean_half,
    U_X_half,
    sigma_X,
    V_X_batch,
    CTF_params,
    rotation_matrices,
    translations,
    noise_variance_half,
    voxel_size,
    image_shape,
    volume_shape,
    ctf_evaluator,
    disc_type,
    disc_type_mean,
    S_left_half=None,
    Q_batch=None,
):
    """Residual + per-image complex Fourier adjoint + matmul sketches.

    The per-image complex adjoint keeps real AND imaginary parts of the
    half-image residual.  Using `per_image_backproject` with `.real(...)`
    (the earlier implementation) silently dropped the imaginary half of the
    Hermitian Fourier residual — verified numerically to equal exactly
    Re[A*(A(X)-b)], i.e. about half the true gradient.
    """
    half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    batch_size = images_half.shape[0]

    # Whiten
    images_w = core.translate_images(images_half, translations, image_shape, half_image=True) / jnp.sqrt(
        noise_variance_half
    )

    CTF_w = ctf_evaluator(CTF_params, image_shape, voxel_size, half_image=True) / jnp.sqrt(noise_variance_half)

    projected_mean_w = (
        core.slice_volume(
            mean_half,
            rotation_matrices,
            image_shape,
            volume_shape,
            disc_type_mean,
            half_image=True,
            half_volume=True,
        )
        * ctf_evaluator(
            CTF_params,
            image_shape,
            voxel_size,
            half_image=True,
        )
        / jnp.sqrt(noise_variance_half)
    )

    # Predicted from X = U diag(s) V^T
    PU_X = batch_over_vol_slice_volume_half(
        U_X_half,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type,
    )
    PU_X *= CTF_w[:, None, :]
    predicted_w = jnp.einsum("bri,br->bi", PU_X, V_X_batch * sigma_X[None, :])

    # Residual (complex Hermitian half-image)
    residual = predicted_w - (images_w - projected_mean_w)

    # Per-image complex Fourier adjoint → (half_vol, batch) complex.
    # We vmap adjoint_slice_volume over images to keep bp separated per
    # image.  core.adjoint_slice_volume is the proven-correct complex
    # Fourier adjoint used by the PPCA M-step RHS.
    ctf_res_half = CTF_w * residual  # (batch, half_im) complex
    max_r = image_shape[0] // 2 - 1

    def _adj_single(slice_1p, rot_1):
        return core.adjoint_slice_volume(
            slice_1p[None, :],
            rot_1[None, :, :],
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            half_volume=True,
            max_r=max_r,
        )

    bp = jax.vmap(_adj_single)(ctf_res_half, rotation_matrices)
    # bp has shape (batch, half_vol) complex.  Transpose to (half_vol, batch)
    # to match the right/left matmul conventions.
    bp = bp.reshape(batch_size, half_vol_size).T

    # Sketches via complex matmul.  `_right_to_real` later IDFTs the right
    # result back to real space.  The left sketch is a real-space inner
    # product <S[i], bp_real[:, j]>.  By Parseval (packed Hermitian, N even):
    #   <f,g>_real = (1/V) * {2·Re[Σ_half conj(F)·G] − Σ_{x=0,Nyq} conj(F)·G}
    # Caller applies the (V/2) scale; the DC+Nyquist planes are self-
    # conjugate so the leading term uses conj(S_half).
    right_contrib = bp @ Q_batch.astype(bp.dtype) if Q_batch is not None else None
    left_cols = (jnp.conj(S_left_half.astype(bp.dtype)) @ bp).real if S_left_half is not None else None

    return right_contrib, left_cols


# ---------------------------------------------------------------------------
# Internal dataset loop (half-volume)
# ---------------------------------------------------------------------------


def _compute_sketches_half(
    cryo,
    U_half,
    sigma,
    V,
    mean_half,
    batch_size,
    S_half=None,
    Q=None,
    disc_type="linear_interp",
    disc_type_mean="linear_interp",
):
    """Loop over image batches, accumulate sketches.  All half-volume."""
    from recovar.ppca.ppca import _iter_processed_batches_half

    half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(cryo.volume_shape)))
    n_images = cryo.n_images

    U_half = jnp.asarray(U_half)
    sigma = jnp.asarray(sigma)
    V = jnp.asarray(V)
    mean_half = jnp.asarray(mean_half)

    right_acc = None
    if Q is not None:
        Q = jnp.asarray(Q)
        right_acc = jnp.zeros((half_vol_size, Q.shape[1]), dtype=jnp.float32)

    left_result = None
    if S_half is not None:
        S_half = jnp.asarray(S_half)
        left_result = jnp.zeros((S_half.shape[0], n_images), dtype=jnp.float32)

    for batch_half, ctf_params, rotation_matrices, translations, image_indices in _iter_processed_batches_half(
        cryo, batch_size
    ):
        right_contrib, left_cols = _sketched_normal_batch(
            batch_half,
            mean_half,
            U_half,
            sigma,
            V[image_indices],
            ctf_params,
            rotation_matrices,
            translations,
            cryo.noise.get_half(image_indices),
            cryo.voxel_size,
            cryo.image_shape,
            cryo.volume_shape,
            cryo.ctf_evaluator,
            disc_type,
            disc_type_mean,
            S_left_half=S_half,
            Q_batch=Q[image_indices] if Q is not None else None,
        )

        if right_contrib is not None:
            right_acc = right_acc + right_contrib
        if left_cols is not None:
            left_result = left_result.at[:, image_indices].set(left_cols)

    return right_acc, left_result


# ---------------------------------------------------------------------------
# Public API — all real-space
# ---------------------------------------------------------------------------


class SketchedNormalOperator:
    """Sketched products of G(X) = A*(A(X) - b) [+ lam * D^2 X].

    U, S, Q inputs are real-space.  Mean is Fourier (complex) or real.

    Parameters
    ----------
    cryo : CryoEMDataset
        Dataset with images and noise model (.noise.get_half()).
    mean : array
        Mean volume.  Complex → Fourier domain (from pipeline).
        Real → real-space (DFT'd internally).
    batch_size : int
        Images per GPU batch.
    disc_type : str
        Interpolation type (default "linear_interp").
    D2_fourier : (vol_size,) real, optional
        Fourier-diagonal D^2 for the radial Fourier prior R(X) = (lam/2)||D X||_F^2.
        If provided together with ``prior_lambda > 0``, the analytic prior gradient
        lam * D^2 X is added to all matvecs:  (D^2 X) Q and S (D^2 X).
        D is Fourier-diagonal so D^2 U is computed as IFFT(D2_fourier * FFT(U_col))
        per column of U.
    prior_lambda : float, default 0.0
        Strength of the radial Fourier prior.  Zero disables the prior.
    """

    def __init__(self, cryo, mean, batch_size=500, disc_type="linear_interp", D2_fourier=None, prior_lambda=0.0):
        self.cryo = cryo
        self.vs = cryo.volume_shape
        self.vol_size = int(np.prod(self.vs))
        self.half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(self.vs)))
        self.n_images = cryo.n_images
        self.batch_size = batch_size
        self.disc_type = disc_type

        # Mean: accept Fourier (complex, from pipeline) or real-space (real)
        mean_flat = np.asarray(mean).reshape(-1)
        if np.iscomplexobj(mean_flat):
            # Fourier-domain mean — convert to half-volume directly
            if mean_flat.size == self.vol_size:
                self.mean_half = np.asarray(ftu.full_volume_to_half_volume(mean_flat, self.vs)).astype(np.complex64)
            else:
                self.mean_half = mean_flat.astype(np.complex64)
        else:
            # Real-space mean
            self.mean_half = _real_vols_to_half_fourier(mean_flat.reshape(1, -1), self.vs)[0]

        # Optional radial Fourier prior
        self.prior_lambda = float(prior_lambda)
        if D2_fourier is not None:
            D2 = np.asarray(D2_fourier).reshape(-1).astype(np.float32)
            if D2.size != self.vol_size:
                raise ValueError(f"D2_fourier must have size vol_size={self.vol_size}, got {D2.size}")
            self.D2_fourier = D2
        else:
            self.D2_fourier = None

    def _D2U_real(self, U):
        """Apply diagonal Fourier operator D^2 to each column of real-space U.

        Returns (vol_size, rank) real.  Computed as IFFT(D2_fourier * FFT(U_col))
        per column; the result is real because D^2 is real and Hermitian-diagonal.
        """
        if self.D2_fourier is None or U.shape[1] == 0:
            return U
        from recovar.core import linalg as _linalg

        U_j = jnp.asarray(U, dtype=jnp.float32)
        U_f = np.asarray(_linalg.batch_dft3(U_j, self.vs, U.shape[1]))  # (V, rank) complex
        U_fw = U_f * self.D2_fourier[:, None]
        U_w_vol = U_fw.T.reshape(U.shape[1], *self.vs)  # (rank, *vs) complex
        return (
            np.asarray(ftu.get_idft3(jnp.asarray(U_w_vol))).real.reshape(U.shape[1], self.vol_size).T.astype(np.float32)
        )

    def _prior_right(self, U, s, V, Q):
        """Prior contribution to right_matvec: lam * (D^2 U) diag(s) V^T Q."""
        if self.prior_lambda == 0.0 or self.D2_fourier is None or len(s) == 0:
            return 0.0
        D2U = self._D2U_real(U)
        return self.prior_lambda * ((D2U * s) @ (V.T @ np.asarray(Q, dtype=np.float32)))

    def _prior_left(self, U, s, V, S):
        """Prior contribution to left_matvec: lam * S (D^2 U) diag(s) V^T."""
        if self.prior_lambda == 0.0 or self.D2_fourier is None or len(s) == 0:
            return 0.0
        D2U = self._D2U_real(U)
        return self.prior_lambda * ((np.asarray(S, dtype=np.float32) @ (D2U * s)) @ V.T)

    def _to_U_half(self, U):
        """(vol_size, rank) real → (half_vol, rank) complex."""
        U = np.asarray(U)
        if U.ndim > 2:
            U = U.reshape(-1, U.shape[-1])
        return _real_vols_to_half_fourier(U.T, self.vs).T

    def _to_S_half(self, S):
        """(s, vol_size) real → (s, half_vol) complex."""
        return _real_vols_to_half_fourier(np.asarray(S), self.vs)

    def _right_to_real(self, right_half):
        """(half_vol, t) half-Fourier → (vol_size, t) real."""
        return _half_fourier_to_real_vols(np.asarray(right_half).T, self.vs).T

    def right_matvec_fourier(self, U_fourier, s, V, Q):
        """Like right_matvec but U_fourier is already in Fourier domain (vol_size,rank).

        Use this when you have Fourier-domain eigenvectors (e.g. from gt.get_vol_svd()).
        """
        U_half = np.asarray(ftu.full_volume_to_half_volume(np.asarray(U_fourier).T, self.vs).T).astype(np.complex64)
        right_half, _ = _compute_sketches_half(
            self.cryo,
            U_half,
            s,
            V,
            self.mean_half,
            self.batch_size,
            Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type,
            disc_type_mean=self.disc_type,
        )
        return self._right_to_real(right_half)

    def right_matvec(self, U, s, V, Q):
        """Compute G(X) @ Q.

        Parameters
        ----------
        U : (vol_size, rank) — real-space basis columns of X.
        s : (rank,) — singular values.
        V : (n_images, rank) — right factor.
        Q : (n_images, t) — right sketch matrix.

        Returns
        -------
        (vol_size, t) — real-space result.
        """
        U_half = self._to_U_half(U)
        right_half, _ = _compute_sketches_half(
            self.cryo,
            U_half,
            s,
            V,
            self.mean_half,
            self.batch_size,
            Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type,
            disc_type_mean=self.disc_type,
        )
        base = self._right_to_real(right_half)
        return base + self._prior_right(U, s, V, Q)

    def left_matvec(self, U, s, V, S):
        """Compute S @ G(X).

        Parameters
        ----------
        U : (vol_size, rank) — real-space basis columns of X.
        s : (rank,) — singular values.
        V : (n_images, rank) — right factor.
        S : (sketch_rank, vol_size) — real-space left sketch matrix.

        Returns
        -------
        (sketch_rank, n_images) — real result.
        """
        U_half = self._to_U_half(U)
        S_half = self._to_S_half(S)
        _, left = _compute_sketches_half(
            self.cryo,
            U_half,
            s,
            V,
            self.mean_half,
            self.batch_size,
            S_half=S_half,
            disc_type=self.disc_type,
            disc_type_mean=self.disc_type,
        )
        return np.asarray(left) + self._prior_left(U, s, V, S)

    def both_matvecs(self, U, s, V, S, Q):
        """Compute S @ G(X) and G(X) @ Q in one pass.

        Returns
        -------
        left : (sketch_rank, n_images)
        right : (vol_size, t)
        """
        U_half = self._to_U_half(U)
        S_half = self._to_S_half(S)
        right_half, left = _compute_sketches_half(
            self.cryo,
            U_half,
            s,
            V,
            self.mean_half,
            self.batch_size,
            S_half=S_half,
            Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type,
            disc_type_mean=self.disc_type,
        )
        if self.prior_lambda != 0.0 and self.D2_fourier is not None and len(s) > 0:
            D2U = self._D2U_real(U)
            S_np = np.asarray(S, dtype=np.float32)
            Q_np = np.asarray(Q, dtype=np.float32)
            left_prior = self.prior_lambda * ((S_np @ (D2U * s)) @ V.T)
            right_prior = self.prior_lambda * ((D2U * s) @ (V.T @ Q_np))
            return np.asarray(left) + left_prior, self._right_to_real(right_half) + right_prior
        return np.asarray(left), self._right_to_real(right_half)
