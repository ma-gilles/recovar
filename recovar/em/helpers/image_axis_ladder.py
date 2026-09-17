"""Image-axis ladder: pad a per-bucket image count onto a few fixed rungs.

Jitted pass-2 helpers retrace on every distinct ``(images, rotations, pixels)``
operand signature. The rotation axis is already a power of two and the pixel
axis takes a handful of values, so the unpadded image axis multiplies the
number of programs (measured on the 10k EMPIAR-10097 subset: 36-39 distinct
image counts per function, jobs 14010842 / 14014342). Padding that axis to a
ladder rung and slicing results back keeps every real image's output
mathematically identical because these helpers never reduce across images.

Padded rows are made inert by construction (a neutral fill per operand) and are
discarded by the caller. Note that a different leading dimension can change the
GPU reduction order over the other axes, so padded evaluation is a float32
reordering of the same arithmetic, not a bitwise identity on GPU.
"""

from __future__ import annotations

import jax.numpy as jnp

_IMAGE_AXIS_LADDER_FINE_STEP = 4
_IMAGE_AXIS_LADDER_FINE_LIMIT = 32


def image_axis_ladder_size(n_images: int) -> int:
    """Round a per-bucket image count up to the image-axis ladder.

    Step 4 up to 32, powers of two above. Chosen from the measured distribution
    of per-bucket image counts on the 10k EMPIAR-10097 subset (job 14010842):
    39 distinct counts, 36 of them in 1..40, three large (126, 194, 220). This
    ladder cuts distinct operand signatures for the per-image pass-2 functions
    from 168 to 57 (2.9x) for 22.6% padding waste; a step-8 variant reaches 37
    signatures but costs 37.3% waste, and padding waste is paid every iteration
    while compilation is partly amortised.
    """

    n = int(n_images)
    if n <= 0:
        return 0
    if n <= _IMAGE_AXIS_LADDER_FINE_LIMIT:
        return -(-n // _IMAGE_AXIS_LADDER_FINE_STEP) * _IMAGE_AXIS_LADDER_FINE_STEP
    return 1 << (n - 1).bit_length()


def pad_image_axis(array, padded_images: int, fill=0):
    """Pad a leading per-image axis with ``fill`` rows, or return it unchanged."""

    n = int(array.shape[0])
    if padded_images <= n:
        return array
    pad = [(0, int(padded_images) - n)] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, pad, constant_values=fill)


def pad_image_tensor(array, n_images: int, padded_images: int, fill=0):
    """Pad a tensor only when it owns the image axis; leave broadcast operands alone.

    A tensor owns the image axis when its leading dimension equals the bucket's
    image count. Operands carried with a broadcast leading axis of 1 broadcast
    against the padded axis by themselves and must not be padded. Pixel-axis
    operands must never be passed here.
    """

    if array is None or not hasattr(array, "shape") or getattr(array, "ndim", 0) == 0:
        return array
    if int(array.shape[0]) != int(n_images):
        return array
    return pad_image_axis(jnp.asarray(array), padded_images, fill)


def slice_image_axis(result, n_images: int):
    """Slice the leading axis of an array or of every array in a tuple back to ``n_images``."""

    if isinstance(result, tuple):
        return tuple(slice_image_axis(item, n_images) for item in result)
    return result[: int(n_images)]
