"""Sketched normal-operator products for low-rank recovery.

Public API — all inputs/outputs are real-space:

    op = SketchedNormalOperator(cryo, mean_real, batch_size)
    left  = op.left_matvec(U, s, V, S)      # S_L @ G(X)
    right = op.right_matvec(U, s, V, Q)      # G(X) @ Q_R
    left, right = op.both_matvecs(U, s, V, S, Q)

where U (vol_shape, rank), S (s, vol_size), Q (n_images, t) are all real.

Internally uses half-volume Fourier convention + per_image_backproject.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.core import linalg
import recovar.core.fourier_transform_utils as ftu
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
    half = np.asarray(ftu.full_volume_to_half_volume(
        ft.reshape(-1, int(np.prod(volume_shape))), volume_shape
    ))
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
    images_half, mean_half, U_X_half, sigma_X, V_X_batch,
    CTF_params, rotation_matrices, translations,
    noise_variance_half, voxel_size,
    image_shape, volume_shape, ctf_evaluator, disc_type, disc_type_mean,
    S_left_half=None, Q_batch=None,
):
    """Residual + per-image backproject + matmul sketches.  All half-volume."""
    from recovar.cuda_backproject import per_image_backproject

    half_vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    batch_size = images_half.shape[0]

    # Whiten
    images_w = core.translate_images(
        images_half, translations, image_shape, half_image=True
    ) / jnp.sqrt(noise_variance_half)

    CTF_w = ctf_evaluator(
        CTF_params, image_shape, voxel_size, half_image=True
    ) / jnp.sqrt(noise_variance_half)

    projected_mean_w = core.slice_volume(
        mean_half, rotation_matrices, image_shape, volume_shape,
        disc_type_mean, half_image=True, half_volume=True,
    ) * ctf_evaluator(
        CTF_params, image_shape, voxel_size, half_image=True,
    ) / jnp.sqrt(noise_variance_half)

    # Predicted from X = U diag(s) V^T
    PU_X = batch_over_vol_slice_volume_half(
        U_X_half, rotation_matrices, image_shape, volume_shape, disc_type,
    )
    PU_X *= CTF_w[:, None, :]
    predicted_w = jnp.einsum("bri,br->bi", PU_X, V_X_batch * sigma_X[None, :])

    # Residual
    residual = predicted_w - (images_w - projected_mean_w)

    # Per-image backproject → (half_vol, batch)
    adjoint_full = ftu.half_image_to_full_image(CTF_w * residual, image_shape)
    real_dtype = adjoint_full.real.dtype
    bp = per_image_backproject(
        jnp.zeros((half_vol_size, batch_size), dtype=real_dtype),
        adjoint_full.real.astype(real_dtype),
        rotation_matrices, image_shape, volume_shape,
        max_r=image_shape[0] // 2 - 1,
    )

    # Sketches via matmul
    right_contrib = bp @ Q_batch if Q_batch is not None else None
    left_cols = (S_left_half @ bp).real if S_left_half is not None else None

    return right_contrib, left_cols


# ---------------------------------------------------------------------------
# Internal dataset loop (half-volume)
# ---------------------------------------------------------------------------

def _compute_sketches_half(
    cryo, U_half, sigma, V, mean_half,
    batch_size, S_half=None, Q=None,
    disc_type="linear_interp", disc_type_mean="linear_interp",
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

    for (batch_half, ctf_params, rotation_matrices,
         translations, image_indices) in _iter_processed_batches_half(cryo, batch_size):

        right_contrib, left_cols = _sketched_normal_batch(
            batch_half, mean_half, U_half, sigma, V[image_indices],
            ctf_params, rotation_matrices, translations,
            cryo.noise.get_half(image_indices), cryo.voxel_size,
            cryo.image_shape, cryo.volume_shape, cryo.ctf_evaluator,
            disc_type, disc_type_mean,
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
    """Sketched products of G(X) = A*(A(X) - b).

    All public methods take and return real-space arrays.
    Fourier / half-volume conversion happens internally.

    Parameters
    ----------
    cryo : CryoEMDataset
        Dataset with images and noise model.
    mean_real : (vol_size,) or (N, N, N)
        Mean volume in real space.
    batch_size : int
        Images per GPU batch.
    disc_type : str
        Interpolation type (default "linear_interp").
    """

    def __init__(self, cryo, mean, batch_size=500,
                 disc_type="linear_interp"):
        self.cryo = cryo
        self.vs = cryo.volume_shape
        self.vol_size = int(np.prod(self.vs))
        self.half_vol_size = int(np.prod(
            ftu.volume_shape_to_half_volume_shape(self.vs)
        ))
        self.n_images = cryo.n_images
        self.batch_size = batch_size
        self.disc_type = disc_type

        # Mean: accept Fourier (complex, from pipeline) or real-space (real)
        mean_flat = np.asarray(mean).reshape(-1)
        if np.iscomplexobj(mean_flat):
            # Fourier-domain mean — convert to half-volume directly
            if mean_flat.size == self.vol_size:
                self.mean_half = np.asarray(
                    ftu.full_volume_to_half_volume(mean_flat, self.vs)
                ).astype(np.complex64)
            else:
                self.mean_half = mean_flat.astype(np.complex64)
        else:
            # Real-space mean
            self.mean_half = _real_vols_to_half_fourier(
                mean_flat.reshape(1, -1), self.vs
            )[0]

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
        return _half_fourier_to_real_vols(
            np.asarray(right_half).T, self.vs
        ).T

    def right_matvec_fourier(self, U_fourier, s, V, Q):
        """Like right_matvec but U_fourier is already in Fourier domain (vol_size,rank).

        Use this when you have Fourier-domain eigenvectors (e.g. from gt.get_vol_svd()).
        """
        U_half = np.asarray(ftu.full_volume_to_half_volume(
            np.asarray(U_fourier).T, self.vs
        ).T).astype(np.complex64)
        right_half, _ = _compute_sketches_half(
            self.cryo, U_half, s, V, self.mean_half,
            self.batch_size, Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type, disc_type_mean=self.disc_type,
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
            self.cryo, U_half, s, V, self.mean_half,
            self.batch_size, Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type, disc_type_mean=self.disc_type,
        )
        return self._right_to_real(right_half)

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
            self.cryo, U_half, s, V, self.mean_half,
            self.batch_size, S_half=S_half,
            disc_type=self.disc_type, disc_type_mean=self.disc_type,
        )
        return np.asarray(left)

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
            self.cryo, U_half, s, V, self.mean_half,
            self.batch_size, S_half=S_half,
            Q=np.asarray(Q, dtype=np.float32),
            disc_type=self.disc_type, disc_type_mean=self.disc_type,
        )
        return np.asarray(left), self._right_to_real(right_half)
