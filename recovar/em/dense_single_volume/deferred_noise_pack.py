"""Device packing for the final partial bucket's shared noise computation."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("target_batch",))
def _pad_noise_pixels(probs, projection, ctf_probs, indices, spare_index, *, target_batch):
    batch = probs.shape[0]
    if projection.shape[0] != batch or ctf_probs.shape[0] != batch or indices.shape != (batch,):
        raise ValueError("Noise pixel input leading dimensions differ")
    if target_batch < batch:
        raise ValueError("Cannot truncate noise pixel rows")

    def pad(value):
        return jnp.pad(value, ((0, target_batch - batch), *((0, 0) for _ in value.shape[1:])))

    return (
        pad(probs),
        pad(projection),
        pad(ctf_probs),
        jnp.concatenate((indices, jnp.full((target_batch - batch,), spare_index, dtype=indices.dtype))),
    )


def pack_noise_pixel_capacity(probs, projection, ctf_probs, indices, *, target_batch, n_images, norm_capacity):
    """Pad only noise pixel operands to the already allocated scalar batch.

    The caller supplies a scalar valid-image mask whose rows beyond the pixel
    batch are false. Those rows contribute zero to every statistic. Their norm
    scatter indices use the unused, in-bounds slot immediately after the logical
    image count. When the norm carry has no spare slot, retain the original
    pixel shape. Original pixel rows/indices and BPref operands are unchanged.
    """
    if n_images < 0 or norm_capacity < n_images:
        raise ValueError("Invalid logical image count or norm capacity")
    if target_batch < probs.shape[0]:
        raise ValueError("Cannot truncate noise pixel rows")
    if target_batch == probs.shape[0] or norm_capacity == n_images:
        return probs, projection, ctf_probs, indices
    return _pad_noise_pixels(
        probs,
        projection,
        ctf_probs,
        indices,
        jnp.asarray(n_images, dtype=indices.dtype),
        target_batch=target_batch,
    )
