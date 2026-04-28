"""Shared half-spectrum image preprocessing helpers for dense EM engines."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.padding as padding
from recovar import core
from recovar.core import mask as core_mask
from recovar.data_io.image_backends import _apply_relion_soft_image_mask_numpy


@jax.jit
def apply_half_translation_phases(weighted_half, translation_phases_half):
    return (weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        weighted_half.shape[0] * translation_phases_half.shape[0],
        weighted_half.shape[1],
    )


def can_native_half_preprocess(experiment_dataset, batch) -> bool:
    if batch is None:
        return False
    shape = getattr(batch, "shape", None)
    if shape is None or len(shape) != 3:
        return False
    image_shape = tuple(int(v) for v in getattr(experiment_dataset, "image_shape", ()))
    if tuple(int(v) for v in shape[-2:]) != image_shape:
        return False
    dtype = getattr(batch, "dtype", None)
    return dtype is None or not np.issubdtype(np.dtype(dtype), np.complexfloating)


def process_images_half_native(experiment_dataset, batch, apply_image_mask: bool):
    """GPU-native half-spectrum preprocessing for real-space image batches."""

    images = jnp.asarray(batch)
    if apply_image_mask:
        image_mask = getattr(experiment_dataset, "image_mask", None)
        if image_mask is not None:
            backend = image_preprocess_backend(experiment_dataset)
            if getattr(backend, "image_mask_mode", None) == "relion_background_fill":
                images = core_mask.apply_relion_soft_image_mask(images, image_mask)
            else:
                images = images * jnp.asarray(image_mask)
    images = images * jnp.asarray(getattr(experiment_dataset, "data_multiplier", 1), dtype=images.dtype)
    return padding.padded_rfft(images, int(experiment_dataset.grid_size), int(experiment_dataset.padding))


def process_half_image(
    experiment_dataset,
    batch,
    config,
    apply_image_mask: bool,
    *,
    native_half_preprocess: bool = False,
):
    if native_half_preprocess:
        return process_images_half_native(experiment_dataset, batch, apply_image_mask)
    process_half_fn = getattr(experiment_dataset, "process_images_half", None)
    if process_half_fn is not None:
        return process_half_fn(batch, apply_image_mask=apply_image_mask)
    processed_full = config.process_fn(batch, apply_image_mask=apply_image_mask)
    return fourier_transform_utils.full_image_to_half_image(processed_full, config.image_shape)


def translate_full_images_to_half(weighted_full, translations, image_shape, n_images: int, n_trans: int):
    """Apply full-spectrum translation phases and return flattened half images.

    TODO(DENSE_ENGINE_BOUNDARY/E003): replace this boundary with native
    half-image preprocessing once the dense path no longer depends on
    full-spectrum translation.
    """

    shifted = core.batch_trans_translate_images(
        weighted_full,
        jnp.repeat(translations[None], n_images, axis=0),
        image_shape,
    )
    return fourier_transform_utils.full_image_to_half_image(
        shifted.reshape(n_images * n_trans, -1),
        image_shape,
    )


def half_translation_phase_table(translations, image_shape):
    lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
        image_shape,
        voxel_size=1,
        scaled=True,
    )
    phase_arg = jnp.einsum(
        "td,pd->tp",
        jnp.asarray(translations, dtype=jnp.float32),
        lattice_half,
    )
    return jnp.exp(-2j * jnp.pi * phase_arg)


def image_preprocess_backend(experiment_dataset):
    image_source = getattr(experiment_dataset, "image_source", None)
    return getattr(image_source, "backend", image_source)


def try_process_masked_and_unmasked_half_together(experiment_dataset, batch):
    """Process RELION masked score images and unmasked M-step images in one FFT call."""

    process_half_fn = getattr(experiment_dataset, "process_images_half", None)
    if process_half_fn is None:
        return None
    backend = image_preprocess_backend(experiment_dataset)
    if getattr(backend, "image_mask_mode", None) != "relion_background_fill":
        return None
    if os.environ.get("RECOVAR_RELION_NUMPY_IMAGE_FFT") != "1":
        return None

    image_mask = getattr(backend, "image_mask", None)
    if image_mask is None:
        image_mask = getattr(backend, "mask", None)
    if image_mask is None:
        image_mask = getattr(experiment_dataset, "image_mask", None)
    if image_mask is None:
        return None

    batch_np = np.asarray(batch)
    image_mask_np = np.asarray(image_mask)
    if batch_np.ndim != 3 or tuple(batch_np.shape[-2:]) != tuple(image_mask_np.shape):
        return None

    masked_batch = _apply_relion_soft_image_mask_numpy(batch_np, image_mask_np)
    combined_batch = np.concatenate((masked_batch, batch_np), axis=0)
    combined_half = process_half_fn(combined_batch, apply_image_mask=False)
    n_images = int(batch_np.shape[0])
    return combined_half[:n_images], combined_half[n_images:]
