"""Shared half-spectrum image preprocessing helpers for dense EM engines."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import mask as core_mask
import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.core.padding as padding

from .half_spectrum import make_half_image_weights

SUPPORTED_IMAGE_MASK_MODES = frozenset({"relion_background_fill", "multiply"})


@jax.jit
def apply_half_translation_phases(weighted_half, translation_phases_half):
    return (weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        weighted_half.shape[0] * translation_phases_half.shape[0],
        weighted_half.shape[1],
    )


def process_half_image(
    experiment_dataset,
    batch,
    apply_image_mask: bool,
):
    process_half_fn = getattr(experiment_dataset, "process_images_half", None)
    if process_half_fn is None:
        raise ValueError("Dense EM requires experiment_dataset.process_images_half")
    return process_half_fn(batch, apply_image_mask=apply_image_mask)


def process_half_image_pair(
    experiment_dataset,
    batch,
    *,
    apply_image_mask_a: bool,
    apply_image_mask_b: bool,
):
    """Return two half-spectrum preprocessing variants for the same batch."""

    if apply_image_mask_a == apply_image_mask_b:
        processed = process_half_image(experiment_dataset, batch, apply_image_mask_a)
        return processed, processed
    process_pair_fn = getattr(experiment_dataset, "process_images_half_pair", None)
    if process_pair_fn is None:
        return (
            process_half_image(experiment_dataset, batch, apply_image_mask_a),
            process_half_image(experiment_dataset, batch, apply_image_mask_b),
        )
    return process_pair_fn(
        batch,
        apply_image_mask_a=apply_image_mask_a,
        apply_image_mask_b=apply_image_mask_b,
    )


def preprocess_half_image_device(
    batch,
    image_mask,
    config,
    *,
    apply_image_mask: bool,
    mask_mode: str,
):
    """JAX packed-half preprocessing for raw real-space image batches."""

    images = jnp.asarray(batch)
    if apply_image_mask:
        if mask_mode == "relion_background_fill":
            images = core_mask.apply_relion_soft_image_mask(images, image_mask)
        elif mask_mode == "multiply":
            images = images * jnp.asarray(image_mask)
        elif mask_mode == "none":
            pass
        else:
            raise ValueError(f"unknown image mask mode {mask_mode!r}")
    images = images * jnp.asarray(config.data_multiplier, dtype=images.dtype)
    return padding.padded_rfft(images, int(config.grid_size), int(config.padding))


def preprocess_half_image_pair_device(
    batch,
    image_mask,
    config,
    *,
    apply_image_mask_a: bool,
    apply_image_mask_b: bool,
    mask_mode: str,
):
    """Return two device-side half-spectrum preprocessing variants."""

    processed_a = preprocess_half_image_device(
        batch,
        image_mask,
        config,
        apply_image_mask=apply_image_mask_a,
        mask_mode=mask_mode,
    )
    if apply_image_mask_a == apply_image_mask_b:
        return processed_a, processed_a
    processed_b = preprocess_half_image_device(
        batch,
        image_mask,
        config,
        apply_image_mask=apply_image_mask_b,
        mask_mode=mask_mode,
    )
    return processed_a, processed_b


def _dense_batch_half_inputs(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    apply_image_mask: bool,
    translation_phases_half=None,
):
    processed_half = process_half_image(
        experiment_dataset,
        batch,
        apply_image_mask,
    )
    ctf_half = config.compute_ctf_half(ctf_params)
    noise_variance_half = jnp.asarray(noise_variance)
    if translation_phases_half is None:
        translation_phases_half = half_translation_phase_table(translations, config.image_shape)
    return processed_half, ctf_half, noise_variance_half, translation_phases_half


def dense_batch_half_input_pair(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    apply_image_mask_a: bool,
    apply_image_mask_b: bool,
    translation_phases_half=None,
):
    """Shared inputs plus two preprocessed half-image variants."""

    processed_a, processed_b = process_half_image_pair(
        experiment_dataset,
        batch,
        apply_image_mask_a=apply_image_mask_a,
        apply_image_mask_b=apply_image_mask_b,
    )
    ctf_half = config.compute_ctf_half(ctf_params)
    noise_variance_half = jnp.asarray(noise_variance)
    if translation_phases_half is None:
        translation_phases_half = half_translation_phase_table(translations, config.image_shape)
    return processed_a, processed_b, ctf_half, noise_variance_half, translation_phases_half


def preprocess_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    score_with_masked_images=False,
    translation_phases_half=None,
):
    """Preprocess one dense image batch for E-step scoring."""

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        score_with_masked_images,
        translation_phases_half=translation_phases_half,
    )
    score_weighted_half = processed_half * ctf_half / noise_variance_half
    shifted_half = apply_half_translation_phases(score_weighted_half, translation_phases_half)
    half_weights = make_half_image_weights(config.image_shape)
    batch_norm = jnp.sum(
        (jnp.abs(processed_half) ** 2 / noise_variance_half) * half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    ctf2_over_nv_half = ctf_half**2 / noise_variance_half
    return shifted_half, batch_norm, ctf2_over_nv_half


def prepare_reconstruction_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    translation_phases_half=None,
):
    """Preprocess one dense image batch for the unmasked M-step path."""

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        False,
        translation_phases_half=translation_phases_half,
    )
    return apply_half_translation_phases(processed_half * ctf_half / noise_variance_half, translation_phases_half)


def preprocess_batch_firstiter_cc(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    score_with_masked_images=False,
    translation_phases_half=None,
):
    """Preprocess one dense image batch for RELION's iter-1 normalized CC scoring."""

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        score_with_masked_images,
        translation_phases_half=translation_phases_half,
    )
    safe_ctf_half = jnp.where(jnp.abs(ctf_half) > 1e-8, 1.0 / ctf_half, 0.0)
    shifted_half = apply_half_translation_phases(
        processed_half * safe_ctf_half,
        translation_phases_half,
    )
    half_weights = make_half_image_weights(config.image_shape)
    image_power = jnp.sum((jnp.abs(processed_half) ** 2) * half_weights[None, :], axis=-1, keepdims=True).real
    ctf2_half = ctf_half**2
    ctf2_over_nv_half = ctf2_half / noise_variance_half
    return shifted_half, image_power, ctf2_half, ctf2_over_nv_half


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


def backend_half_preprocess_uses_host_images(experiment_dataset) -> bool:
    """Return whether the dataset backend half preprocessing expects host arrays."""

    backend = image_preprocess_backend(experiment_dataset)
    return getattr(backend, "image_mask_mode", "multiply") == "relion_background_fill"


def resolve_image_mask_for_half_preprocess(
    experiment_dataset,
    image_shape,
    *,
    require_mask: bool,
):
    """Return the image mask and mode used by native packed-half preprocessing."""

    backend = image_preprocess_backend(experiment_dataset)
    mask_mode = getattr(backend, "image_mask_mode", "multiply")
    if mask_mode not in SUPPORTED_IMAGE_MASK_MODES:
        raise ValueError(
            "Unsupported image_mask_mode for native half preprocessing: "
            f"{mask_mode!r}. Expected one of {sorted(SUPPORTED_IMAGE_MASK_MODES)}.",
        )

    image_mask = getattr(backend, "image_mask", None)
    if image_mask is None:
        image_mask = getattr(backend, "mask", None)
    if image_mask is None:
        image_mask = getattr(experiment_dataset, "image_mask", None)
    if image_mask is None:
        if require_mask:
            raise ValueError(
                "score_with_masked_images=True requires an image mask for native half preprocessing",
            )
        return np.ones(tuple(image_shape), dtype=np.float32), "none"
    return image_mask, mask_mode
