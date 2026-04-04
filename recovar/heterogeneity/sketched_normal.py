"""Sketched normal-operator products for low-rank recovery.

Given X = U_X diag(sigma_X) V_X^T and the whitened forward model
  A_i(x) = CTF_i * slice_i(x) / sqrt(noise_var_i),
  b_i    = (y_i - CTF_i * slice_i(mu)) / sqrt(noise_var_i),
the normal residual gradient is G(X) = [g_1, ..., g_n] where
  g_i = A_i^*(A_i x_i - b_i).

We compute S_L @ G(X)  and  G(X) @ Q_R  without forming dense G(X).

All operations use the half-image (rfft2) convention.  The three
batch-level primitives are JIT-compiled — the PC/sketch chunking
loops unroll at trace time so XLA sees the full graph and fuses
everything into one kernel per image batch.

See docs/math/sketched_normal_operator.md for derivations.
"""

import logging

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from recovar import core
from recovar.core import fourier_transform_utils as ftu
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig
from recovar.core.slicing import batch_adjoint_slice_volume
from recovar.heterogeneity import covariance_core
import recovar.core.forward as core_forward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_half_image(arr, image_shape):
    """Convert full-spectrum images to half-image (rfft) if needed."""
    full_size = int(np.prod(image_shape))
    half_size = int(np.prod(ftu.image_shape_to_half_image_shape(image_shape)))
    if arr.shape[-1] == full_size:
        return ftu.full_image_to_half_image(arr, image_shape)
    elif arr.shape[-1] == half_size:
        return arr
    raise ValueError(
        f"Image size {arr.shape[-1]} doesn't match "
        f"full ({full_size}) or half ({half_size})"
    )


def _ensure_half_noise(noise_variance, image_shape, batch_shape):
    """Ensure noise_variance is in half-image format and batch-broadcastable."""
    half_size = int(np.prod(ftu.image_shape_to_half_image_shape(image_shape)))
    full_size = int(np.prod(image_shape))
    nv = jnp.asarray(noise_variance)
    if nv.ndim < 2:
        return jnp.broadcast_to(nv, (*batch_shape, half_size))
    if nv.shape[-1] == full_size:
        return ftu.full_image_to_half_image(nv, image_shape)
    if nv.shape[-1] == half_size:
        return nv
    return jnp.broadcast_to(nv, (*batch_shape, half_size))


# ---------------------------------------------------------------------------
# JIT-compiled batch-level primitives
# ---------------------------------------------------------------------------


@eqx.filter_jit
def compute_residual_batch_from_factors(
    config: ForwardModelConfig,
    U_X,
    sigma_X,
    V_X_batch,
    images_batch,
    mean,
    rotation_matrices,
    translations,
    ctf_params,
    noise_variance,
    pc_batch_size: int = 10,
):
    """Compute whitened residual batch: r_i = A_i(x_i) - b_i.

    JIT-compiled.  The PC chunking loop unrolls at trace time so XLA
    fuses the full residual computation into one kernel.

    Parameters
    ----------
    config : ForwardModelConfig
    U_X : (volume_size, rank) — basis columns of X.
    sigma_X : (rank,) — singular values.
    V_X_batch : (batch, rank) — rows of V_X for this batch.
    images_batch : (batch, image_size) — raw Fourier images (full or half).
    mean : (volume_size,) — mean volume in Fourier space.
    rotation_matrices : (batch, 3, 3)
    translations : (batch, 2)
    ctf_params : (batch, 9)
    noise_variance : (batch, image_size) or scalar-broadcastable
    pc_batch_size : int — PCs to project at once (default 10).  Controls
        peak memory: pc_batch_size * batch * half_image_size * 8 bytes.

    Returns
    -------
    residual : (batch, half_image_size) — whitened residual, half-image.
    CTF_w : (batch, half_image_size) — whitened CTF, half-image.
    """
    n_images = images_batch.shape[0]
    rank = U_X.shape[1]

    # Preprocess images: convert to half, translate
    images = _to_half_image(images_batch, config.image_shape)
    images = core.translate_images(
        images, translations, config.image_shape, half_image=True
    )

    # CTF and noise in half-image format
    CTF = config.compute_ctf(ctf_params, half_image=True)
    noise_var = _ensure_half_noise(noise_variance, config.image_shape, (n_images,))
    safe_noise_std = jnp.sqrt(
        jnp.maximum(noise_var, jnp.finfo(noise_var.dtype).tiny)
    )
    CTF_w = CTF / safe_noise_std

    # Whitened projected mean
    projected_mean = core_forward.forward_model(
        config, mean, ctf_params, rotation_matrices,
        skip_ctf=False, half_image=True,
    )
    whitened_data = images / safe_noise_std - projected_mean / safe_noise_std

    # Predicted images from factored X, chunked over PCs
    C_batch = V_X_batch * sigma_X[None, :]
    half_image_size = int(
        np.prod(ftu.image_shape_to_half_image_shape(config.image_shape))
    )
    predicted_w = jnp.zeros((n_images, half_image_size), dtype=images.dtype)

    for j_start in range(0, rank, pc_batch_size):
        j_end = min(j_start + pc_batch_size, rank)
        AUs_chunk = covariance_core.batch_vol_forward_from_map(
            config, U_X[:, j_start:j_end].T, ctf_params, rotation_matrices,
            skip_ctf=False, half_image=True,
        )
        AUs_chunk = AUs_chunk / safe_noise_std[None, :, :]
        predicted_w = predicted_w + jnp.einsum(
            "jip,ij->ip", AUs_chunk, C_batch[:, j_start:j_end]
        )

    residual = predicted_w - whitened_data
    return residual, CTF_w


@eqx.filter_jit
def right_sketch_normal_residual_batch(
    residual_batch,
    Q_batch,
    CTF_w,
    rotation_matrices,
    image_shape: tuple,
    volume_shape: tuple,
    disc_type: str,
    sketch_chunk_size: int = 10,
):
    """Batch contribution to G(X) @ Q_R via multi-channel backprojection.

    JIT-compiled.  Chunked over sketch columns; the loop unrolls at
    trace time.

    Parameters
    ----------
    residual_batch : (batch, half_image_size) — whitened residuals.
    Q_batch : (batch, qrank) — rows of Q_R for this batch.
    CTF_w : (batch, half_image_size) — whitened CTF.
    rotation_matrices : (batch, 3, 3)
    image_shape, volume_shape, disc_type : forward model geometry.
    sketch_chunk_size : int — columns to backproject at once (default 10).

    Returns
    -------
    contribution : (qrank, volume_size)
    """
    qrank = Q_batch.shape[1]
    volume_size = int(np.prod(volume_shape))
    adjoint_input = CTF_w * residual_batch

    contribution = jnp.zeros((qrank, volume_size), dtype=adjoint_input.dtype)
    for j_start in range(0, qrank, sketch_chunk_size):
        j_end = min(j_start + sketch_chunk_size, qrank)
        Q_chunk = Q_batch[:, j_start:j_end]
        weighted_slices = adjoint_input[None, :, :] * Q_chunk.T[:, :, None]
        chunk_result = batch_adjoint_slice_volume(
            weighted_slices, rotation_matrices, image_shape, volume_shape,
            disc_type, half_image=True,
        )
        contribution = contribution.at[j_start:j_end].set(chunk_result)
    return contribution


@eqx.filter_jit
def left_sketch_normal_residual_batch(
    config: ForwardModelConfig,
    S_left,
    residual_batch,
    CTF_w,
    rotation_matrices,
    ctf_params,
    hermitian_weights,
    sketch_chunk_size: int = 10,
):
    """Batch columns of S_left @ G(X) without backprojection.

    JIT-compiled.  Chunked over sketch rows; the loop unrolls at
    trace time.

    Parameters
    ----------
    config : ForwardModelConfig
    S_left : (sketch_rank, volume_size) — left sketch matrix.
    residual_batch : (batch, half_image_size) — whitened residuals.
    CTF_w : (batch, half_image_size) — whitened CTF.
    rotation_matrices : (batch, 3, 3)
    ctf_params : (batch, 9)
    hermitian_weights : (half_image_size,) — sqrt(w) for half-spectrum sums.
    sketch_chunk_size : int — rows to project at once (default 10).

    Returns
    -------
    out : (sketch_rank, batch) — real-valued columns of S_L @ G(X).
    """
    sketch_rank = S_left.shape[0]
    n_images = residual_batch.shape[0]

    adjoint_input_w = (CTF_w * residual_batch) * hermitian_weights[None, :]
    out = jnp.zeros((sketch_rank, n_images), dtype=adjoint_input_w.real.dtype)

    for s_start in range(0, sketch_rank, sketch_chunk_size):
        s_end = min(s_start + sketch_chunk_size, sketch_rank)
        projected_chunk = covariance_core.batch_vol_forward_from_map(
            config, S_left[s_start:s_end], ctf_params, rotation_matrices,
            skip_ctf=True, half_image=True,
        )
        projected_chunk = projected_chunk * hermitian_weights[None, None, :]
        chunk_out = jnp.einsum(
            "sik,ik->si", projected_chunk, adjoint_input_w
        ).real
        out = out.at[s_start:s_end].set(chunk_out)

    return out


# ---------------------------------------------------------------------------
# Public dataset-level driver
# ---------------------------------------------------------------------------


def compute_normal_residual_sketches(
    experiment_dataset,
    U_X,
    sigma_X,
    V_X,
    mean,
    noise_variance,
    batch_size,
    left_sketch=None,
    right_sketch=None,
    disc_type="linear_interp",
    pc_batch_size=10,
    sketch_chunk_size=10,
):
    """Compute S_left @ G(X) and/or G(X) @ Q_R without forming dense G(X).

    Iterates over image batches; each batch calls the JIT-compiled
    primitives above.

    Parameters
    ----------
    experiment_dataset : CryoEMDataset
    U_X : (volume_size, rank) — left factor of X.
    sigma_X : (rank,) — singular values of X.
    V_X : (n_images, rank) — right factor of X.
    mean : (volume_size,) — mean volume (Fourier).
    noise_variance : per-image or global noise variance.
    batch_size : int — images per batch.
    left_sketch : (s, volume_size) or None — S_left matrix.
    right_sketch : (n_images, t) or None — Q_R matrix.
    disc_type : str — discretization type (default "linear_interp").
    pc_batch_size : int — PCs to project at once (default 10).
    sketch_chunk_size : int — sketch columns/rows at once (default 10).

    Returns
    -------
    dict with keys:
        "left" : (s, n_images) array or None — S_left @ G(X).
        "right" : (volume_size, t) array or None — G(X) @ Q_R.
    """
    if left_sketch is None and right_sketch is None:
        return {"left": None, "right": None}

    volume_size = int(np.prod(experiment_dataset.volume_shape))
    n_images = experiment_dataset.n_images

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type
    )
    hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape)

    right_acc = None
    if right_sketch is not None:
        qrank = right_sketch.shape[1]
        right_acc = jnp.zeros(
            (qrank, volume_size), dtype=experiment_dataset.dtype
        )

    left_result = None
    if left_sketch is not None:
        sketch_rank = left_sketch.shape[0]
        left_result = jnp.zeros(
            (sketch_rank, n_images), dtype=experiment_dataset.dtype_real
        )

    U_X = jnp.asarray(U_X)
    sigma_X = jnp.asarray(sigma_X)
    V_X = jnp.asarray(V_X)
    mean = jnp.asarray(mean)
    if left_sketch is not None:
        left_sketch = jnp.asarray(left_sketch)
    if right_sketch is not None:
        right_sketch = jnp.asarray(right_sketch)

    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        _batch_noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(batch_size):

        nv = noise_variance
        if hasattr(nv, "shape") and nv.ndim >= 2:
            nv = nv[image_indices]
        elif hasattr(nv, "shape") and nv.ndim == 1 and nv.shape[0] == n_images:
            nv = nv[image_indices]

        V_X_batch = V_X[image_indices]
        residual, CTF_w = compute_residual_batch_from_factors(
            config, U_X, sigma_X, V_X_batch, images, mean,
            rotation_matrices, translations, ctf_params, nv,
            pc_batch_size=pc_batch_size,
        )

        if right_sketch is not None:
            Q_batch = right_sketch[image_indices]
            right_contrib = right_sketch_normal_residual_batch(
                residual, Q_batch, CTF_w, rotation_matrices,
                config.image_shape, config.volume_shape, config.disc_type,
                sketch_chunk_size=sketch_chunk_size,
            )
            right_acc = right_acc + right_contrib

        if left_sketch is not None:
            left_cols = left_sketch_normal_residual_batch(
                config, left_sketch, residual, CTF_w,
                rotation_matrices, ctf_params, hermitian_weights,
                sketch_chunk_size=sketch_chunk_size,
            )
            left_result = left_result.at[:, image_indices].set(left_cols)

    return {
        "left": left_result,
        "right": right_acc.T if right_acc is not None else None,
    }
