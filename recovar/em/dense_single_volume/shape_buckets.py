"""Static-shape bucket helpers for dense RELION refinement.

JAX recompiles when array shapes change.  RELION-style pruning changes local
support sizes frequently, so hot EM loops should round those sizes to a small
set of padded shape classes and pass masks/counts for the valid rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapeBucket:
    """A compile-cache shape class for one padded EM bucket."""

    image_batch_size: int
    rotation_count: int
    translation_count: int
    half_pixel_count: int

    @property
    def pose_count(self) -> int:
        return int(self.rotation_count) * int(self.translation_count)


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to a positive multiple."""

    value = int(value)
    multiple = int(multiple)
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def power_of_two_bucket(value: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Return a power-of-two padded bucket for low-cardinality shape classes."""

    value = int(value)
    minimum = int(minimum)
    if value <= 0:
        return max(1, minimum)
    bucket = 1 << max(value - 1, minimum - 1).bit_length()
    if maximum is not None:
        bucket = min(bucket, int(maximum))
    return max(bucket, value, minimum)


def coarse_bucket(value: int, *, small_power2_max: int, large_multiple: int, minimum: int = 1) -> int:
    """Bucket small values by power-of-two and large values by coarse multiples."""

    value = int(value)
    if value <= int(small_power2_max):
        return power_of_two_bucket(value, minimum=minimum, maximum=small_power2_max)
    return round_up_to_multiple(value, large_multiple)


def dense_shape_bucket(
    *,
    image_batch_size: int,
    rotation_count: int,
    translation_count: int,
    half_pixel_count: int,
    image_batch_multiple: int = 8,
    rotation_multiple: int = 64,
    translation_multiple: int = 1,
    half_pixel_multiple: int = 256,
) -> ShapeBucket:
    """Return the padded shape class for a dense/global EM bucket."""

    return ShapeBucket(
        image_batch_size=round_up_to_multiple(image_batch_size, image_batch_multiple),
        rotation_count=round_up_to_multiple(rotation_count, rotation_multiple),
        translation_count=round_up_to_multiple(translation_count, translation_multiple),
        half_pixel_count=round_up_to_multiple(half_pixel_count, half_pixel_multiple),
    )


def local_shape_bucket(
    *,
    image_batch_size: int,
    local_rotation_count: int,
    translation_count: int,
    half_pixel_count: int,
    image_batch_multiple: int = 8,
    local_rotation_small_max: int = 4096,
    local_rotation_large_multiple: int = 512,
    translation_multiple: int = 1,
    half_pixel_multiple: int = 256,
) -> ShapeBucket:
    """Return the padded shape class for an exact-local EM bucket."""

    return ShapeBucket(
        image_batch_size=round_up_to_multiple(image_batch_size, image_batch_multiple),
        rotation_count=coarse_bucket(
            local_rotation_count,
            small_power2_max=local_rotation_small_max,
            large_multiple=local_rotation_large_multiple,
            minimum=16,
        ),
        translation_count=round_up_to_multiple(translation_count, translation_multiple),
        half_pixel_count=round_up_to_multiple(half_pixel_count, half_pixel_multiple),
    )


def pad_axis(array, axis: int, size: int, *, value=0):
    """Pad one axis to ``size`` without changing existing values."""

    arr = np.asarray(array)
    axis = int(axis)
    size = int(size)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"axis {axis} out of bounds for array with ndim={arr.ndim}")
    if arr.shape[axis] > size:
        raise ValueError(f"cannot pad axis {axis} from {arr.shape[axis]} down to {size}")
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, size - arr.shape[axis])
    return np.pad(arr, pad_width, mode="constant", constant_values=value)


def pad_batch_data_ctf_and_valid_mask(batch_data, ctf_params, target_batch_size: int):
    """Pad raw image and CTF rows together and return a valid-image mask.

    Dummy CTF rows repeat the first real CTF row, matching the dense/local
    big-JIT callers that mask padded images before using those rows.
    """

    actual_batch_size = int(np.asarray(batch_data).shape[0])
    padded_batch_size = int(max(actual_batch_size, target_batch_size))
    if actual_batch_size == padded_batch_size:
        return (
            batch_data,
            ctf_params,
            np.ones(actual_batch_size, dtype=bool),
            actual_batch_size,
            padded_batch_size,
        )

    ctf_params_np = np.asarray(ctf_params)
    padded_ctf_params = pad_axis(ctf_params_np, 0, padded_batch_size, value=0)
    if actual_batch_size > 0:
        padded_ctf_params[actual_batch_size:] = ctf_params_np[0]
    valid_image_mask = np.zeros(padded_batch_size, dtype=bool)
    valid_image_mask[:actual_batch_size] = True
    return (
        pad_axis(batch_data, 0, padded_batch_size, value=0),
        padded_ctf_params,
        valid_image_mask,
        actual_batch_size,
        padded_batch_size,
    )
