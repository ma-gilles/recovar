"""Compose deferred VDAM packing using the existing host-selected rows."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class DeferredVdamHostPackedOperands(NamedTuple):
    posterior: jax.Array
    posterior_sum_t: jax.Array
    images: jax.Array
    ctf: jax.Array
    minvsigma2: jax.Array
    noise_projection: jax.Array
    ctf_probs: jax.Array


def _gather_deferred_vdam_host_plan(
    reconstruction_probs,
    reconstruction_probs_sum_t,
    source_images,
    source_ctf,
    source_minvsigma2,
    flat_noise_projection,
    take_indices,
    row_mask,
    flat_take_indices,
):
    """Gather without altering the host support decision or packed shape.

    Callers provide the same valid indices and mask produced by the mature
    host planner. No support detection, reduction, or capacity selection is
    performed here. The fringe image count follows the descriptor shape.
    """
    if take_indices.ndim != 2 or take_indices.dtype != jnp.int32:
        raise ValueError("take indices must be an int32 matrix")
    if row_mask.shape != take_indices.shape or row_mask.dtype != jnp.bool_:
        raise ValueError("row mask must be a matching boolean matrix")
    if flat_take_indices.shape != take_indices.shape or flat_take_indices.dtype != jnp.int32:
        raise ValueError("flat indices must be a matching int32 matrix")
    batch_size = take_indices.shape[0]
    if reconstruction_probs.ndim != 3 or reconstruction_probs.shape[0] < batch_size:
        raise ValueError("posterior must cover the selected image rows")
    if reconstruction_probs_sum_t.shape != reconstruction_probs.shape[:2]:
        raise ValueError("posterior sums must match dense image/rotation axes")
    if source_images.ndim != 2 or source_images.shape[0] < batch_size:
        raise ValueError("source images must cover the selected image rows")
    if source_ctf.shape != source_images.shape or source_minvsigma2.shape != source_images.shape:
        raise ValueError("source image, CTF and inverse-noise shapes must match")
    if flat_noise_projection.ndim != 2 or flat_noise_projection.shape[1] != source_images.shape[1]:
        raise ValueError("flat projection and source pixel axes must match")

    posterior = jnp.take_along_axis(reconstruction_probs[:batch_size], take_indices[:, :, None], axis=1)
    posterior = jnp.where(row_mask[:, :, None], posterior, 0.0)
    posterior_sum_t = jnp.take_along_axis(reconstruction_probs_sum_t[:batch_size], take_indices, axis=1)
    posterior_sum_t = jnp.where(row_mask, posterior_sum_t, 0.0)
    noise_projection = jnp.take(
        jnp.asarray(flat_noise_projection, dtype=jnp.complex64),
        flat_take_indices.reshape(-1),
        axis=0,
    ).reshape((*flat_take_indices.shape, flat_noise_projection.shape[-1]))
    noise_projection = jnp.where(row_mask[:, :, None], noise_projection, 0.0)
    return (
        posterior,
        posterior_sum_t,
        source_images[:batch_size],
        source_ctf[:batch_size],
        source_minvsigma2[:batch_size],
        noise_projection,
    )


@jax.jit
def pack_deferred_vdam_host_plan(
    reconstruction_probs,
    reconstruction_probs_sum_t,
    source_images,
    source_ctf,
    source_minvsigma2,
    flat_noise_projection,
    take_indices,
    row_mask,
    flat_take_indices,
):
    """Pack compact operands and run the unchanged sequential denominator.

    Host rotation gathering and all downstream noise/BPref consumers remain
    outside this boundary. The denominator retains its CUDA arithmetic and
    traversal; this wrapper only composes the existing device operations.
    """
    from recovar import cuda_backproject

    packed = _gather_deferred_vdam_host_plan(
        reconstruction_probs,
        reconstruction_probs_sum_t,
        source_images,
        source_ctf,
        source_minvsigma2,
        flat_noise_projection,
        take_indices,
        row_mask,
        flat_take_indices,
    )
    posterior, _, _, ctf, minvsigma2, _ = packed
    ctf_probs = cuda_backproject.relion_vdam_mstep_denominator_f32(ctf, minvsigma2, posterior)
    ctf_probs = jnp.where(row_mask[:, :, None], ctf_probs, 0.0)
    return DeferredVdamHostPackedOperands(*packed, ctf_probs)
