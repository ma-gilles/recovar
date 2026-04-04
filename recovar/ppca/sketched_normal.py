"""Sketched normal-operator products for low-rank recovery.

Computes  S_L @ G(X)  and  G(X) @ Q_R  without forming dense G(X),
where G(X) = A^*(A(X) - b) is the normal residual gradient.

Follows the same whitened half-image convention as ppca.E_M_step_batch_half.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.core import linalg
import recovar.core.fourier_transform_utils as ftu
from recovar.ppca.ppca import (
    _forward_model_from_map,
    batch_over_vol_slice_volume_half,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JIT-compiled batch kernel
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=[10, 11, 12, 13, 14])
def _sketched_normal_batch(
    images_half,          # (batch, half_image)
    mean,                 # (half_vol,)
    U_X_half,             # (half_vol, rank)
    sigma_X,              # (rank,)
    V_X_batch,            # (batch, rank)
    CTF_params,           # (batch, 9)
    rotation_matrices,    # (batch, 3, 3)
    translations,         # (batch, 2)
    noise_variance_half,  # (batch, half_image) or broadcastable
    voxel_size,           # scalar
    image_shape,          # static
    volume_shape,         # static
    ctf_evaluator,        # static
    disc_type,            # static
    disc_type_mean,       # static
    # Optional sketch matrices (pass zeros if unused)
    S_left_half=None,     # (sketch_rank, half_vol) or None
    Q_batch=None,         # (batch, qrank) or None
):
    """JIT-compiled core: residual + both sketches for one image batch.

    Everything is in half-image / half-volume convention.
    The PC loop is unrolled at trace time — XLA fuses into one kernel.
    """
    half_vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    rank = U_X_half.shape[1]

    rfft_w = jnp.tile(
        linalg.half_spectrum_last_axis_weights(image_shape[1]),
        (image_shape[0], 1),
    ).reshape(-1)

    # --- Whiten ---
    images_w = core.translate_images(
        images_half, translations, image_shape, half_image=True
    ) / jnp.sqrt(noise_variance_half)

    CTF_w = ctf_evaluator(
        CTF_params, image_shape, voxel_size, half_image=True
    ) / jnp.sqrt(noise_variance_half)

    projected_mean_w = core.slice_volume(
        mean, rotation_matrices, image_shape, volume_shape,
        disc_type_mean, half_image=True, half_volume=True,
    ) * ctf_evaluator(
        CTF_params, image_shape, voxel_size, half_image=True,
    ) / jnp.sqrt(noise_variance_half)

    centered_images_w = images_w - projected_mean_w

    # --- Predicted images from X = U diag(s) V^T ---
    # batch_over_vol_slice_volume_half vmaps over volume column axis (1)
    # input:  U_X_half (half_vol, rank)
    # output: PU_X (n_images, rank, half_image)
    PU_X = batch_over_vol_slice_volume_half(
        U_X_half, rotation_matrices, image_shape, volume_shape, disc_type,
    )
    PU_X *= CTF_w[:, None, :]  # whiten: (batch, rank, half_image)

    C_batch = V_X_batch * sigma_X[None, :]  # (batch, rank)
    predicted_w = jnp.einsum("bri,br->bi", PU_X, C_batch)

    # --- Residual ---
    residual = predicted_w - centered_images_w  # (batch, half_image)

    # --- Right sketch: G(X) @ Q_R ---
    right_contrib = None
    if Q_batch is not None:
        adjoint_input = CTF_w * residual  # (batch, half_image)
        # weighted_slices[j, i, :] = adjoint_input[i,:] * Q[i,j]
        weighted_slices = adjoint_input[None, :, :] * Q_batch.T[:, :, None]
        # (qrank, batch, half_image) → batch_adjoint gives (qrank, half_vol)
        right_contrib = core.batch_adjoint_slice_volume(
            weighted_slices, rotation_matrices, image_shape, volume_shape,
            disc_type, half_image=True, half_volume=True,
        ).T  # → (half_vol, qrank)

    # --- Left sketch: S_L @ G(X) ---
    left_cols = None
    if S_left_half is not None:
        # Project S_left rows: batch_over_vol_slice_volume_half vmaps over
        # column axis (1), so pass S_left_half.T = (half_vol, sketch_rank).
        PS = batch_over_vol_slice_volume_half(
            S_left_half.T, rotation_matrices, image_shape, volume_shape, disc_type,
        )  # (batch, sketch_rank, half_image)
        # Contract: (S_L G)_{s,i} = sum_k PS[i,s,k] * CTF_w[i,k] * r[i,k] * w[k]
        adjoint_input = CTF_w * residual  # (batch, half_image)
        left_cols = jnp.einsum(
            "bsi,bi,i->sb",
            PS, adjoint_input, rfft_w,
        ).real  # (sketch_rank, batch)  — real for Hermitian data

    return residual, right_contrib, left_cols


# ---------------------------------------------------------------------------
# Dataset-level driver
# ---------------------------------------------------------------------------


def compute_normal_residual_sketches(
    experiment_dataset,
    U_X_half,
    sigma_X,
    V_X,
    mean_half,
    batch_size,
    left_sketch_half=None,
    right_sketch=None,
    disc_type="linear_interp",
    disc_type_mean="linear_interp",
):
    """Compute S_L @ G(X) and/or G(X) @ Q_R without forming dense G(X).

    Parameters
    ----------
    experiment_dataset : CryoEMDataset
        Must have .noise with .get_half().
    U_X_half : (half_vol, rank) — basis in half-volume layout.
    sigma_X : (rank,) — singular values.
    V_X : (n_images, rank) — right factor.
    mean_half : (half_vol,) — mean volume in half-volume layout.
    batch_size : int
    left_sketch_half : (s, half_vol) or None — S_L in half-volume layout.
    right_sketch : (n_images, t) or None — Q_R.
    disc_type : str
    disc_type_mean : str

    Returns
    -------
    dict with "left" : (s, n_images) or None, "right" : (half_vol, t) or None.
    """
    if left_sketch_half is None and right_sketch is None:
        return {"left": None, "right": None}

    cryo = experiment_dataset
    half_vol_shape = ftu.volume_shape_to_half_volume_shape(cryo.volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    n_images = cryo.n_images

    U_X_half = jnp.asarray(U_X_half)
    sigma_X = jnp.asarray(sigma_X)
    V_X = jnp.asarray(V_X)
    mean_half = jnp.asarray(mean_half)

    right_acc = None
    if right_sketch is not None:
        qrank = right_sketch.shape[1]
        right_sketch = jnp.asarray(right_sketch)
        right_acc = jnp.zeros((half_vol_size, qrank), dtype=cryo.dtype)

    left_result = None
    if left_sketch_half is not None:
        sketch_rank = left_sketch_half.shape[0]
        left_sketch_half = jnp.asarray(left_sketch_half)
        left_result = jnp.zeros((sketch_rank, n_images), dtype=cryo.dtype_real)

    from recovar.ppca.ppca import _iter_processed_batches_half

    for (batch_half, ctf_params, rotation_matrices,
         translations, image_indices) in _iter_processed_batches_half(cryo, batch_size):

        noise_half = cryo.noise.get_half(image_indices)

        _, right_contrib, left_cols = _sketched_normal_batch(
            batch_half,
            mean_half,
            U_X_half,
            sigma_X,
            V_X[image_indices],
            ctf_params,
            rotation_matrices,
            translations,
            noise_half,
            cryo.voxel_size,
            cryo.image_shape,
            cryo.volume_shape,
            cryo.ctf_evaluator,
            disc_type,
            disc_type_mean,
            S_left_half=left_sketch_half,
            Q_batch=right_sketch[image_indices] if right_sketch is not None else None,
        )

        if right_contrib is not None:
            right_acc = right_acc + right_contrib
        if left_cols is not None:
            left_result = left_result.at[:, image_indices].set(left_cols)

    return {
        "left": left_result,
        "right": right_acc,
    }
