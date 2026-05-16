"""Shared half-spectrum image preprocessing helpers for dense EM engines."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils

from .half_spectrum import make_half_image_weights

SUPPORTED_IMAGE_MASK_MODES = frozenset({"relion_background_fill", "multiply"})


@jax.jit
def apply_half_translation_phases(weighted_half, translation_phases_half):
    return (weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        weighted_half.shape[0] * translation_phases_half.shape[0],
        weighted_half.shape[1],
    )


def _cast_shift_inputs(
    processed_half,
    ctf_half,
    noise_variance_half=None,
    translation_phases_half=None,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
):
    if score_complex_dtype is not None:
        processed_half = processed_half.astype(score_complex_dtype)
        if translation_phases_half is not None:
            translation_phases_half = translation_phases_half.astype(score_complex_dtype)
    if score_real_dtype is not None:
        ctf_half = ctf_half.astype(score_real_dtype)
        if noise_variance_half is not None:
            noise_variance_half = noise_variance_half.astype(score_real_dtype)
    return processed_half, ctf_half, noise_variance_half, translation_phases_half


def _norm_inputs(processed_half, noise_variance_half=None, half_weights=None, *, norm_real_dtype=None):
    if norm_real_dtype is None:
        return processed_half, noise_variance_half, half_weights
    norm_complex_dtype = jnp.complex128 if norm_real_dtype == jnp.float64 else jnp.complex64
    processed_half = processed_half.astype(norm_complex_dtype)
    if noise_variance_half is not None:
        noise_variance_half = noise_variance_half.astype(norm_real_dtype)
    if half_weights is not None:
        half_weights = half_weights.astype(norm_real_dtype)
    return processed_half, noise_variance_half, half_weights


def process_half_image(
    experiment_dataset,
    batch,
    apply_image_mask: bool,
):
    process_half_fn = getattr(experiment_dataset, "process_images_half", None)
    if process_half_fn is None:
        raise ValueError("Dense EM requires experiment_dataset.process_images_half")
    return process_half_fn(batch, apply_image_mask=apply_image_mask)


def _dense_batch_half_inputs(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    apply_image_mask: bool,
):
    processed_half = process_half_image(
        experiment_dataset,
        batch,
        apply_image_mask,
    )
    ctf_half = config.compute_ctf_half(ctf_params)
    noise_variance_half = jnp.asarray(noise_variance)
    translation_phases_half = half_translation_phase_table(translations, config.image_shape)
    return processed_half, ctf_half, noise_variance_half, translation_phases_half


def preprocess_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    score_with_masked_images=False,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
    norm_real_dtype=None,
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
    )
    shift_processed_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    score_weighted_half = shift_processed_half * shift_ctf_half / shift_noise_half
    shifted_half = apply_half_translation_phases(score_weighted_half, shift_phases_half)
    half_weights = make_half_image_weights(config.image_shape)
    norm_processed_half, norm_noise_half, norm_half_weights = _norm_inputs(
        processed_half,
        noise_variance_half,
        half_weights,
        norm_real_dtype=norm_real_dtype,
    )
    batch_norm = jnp.sum(
        (jnp.abs(norm_processed_half) ** 2 / norm_noise_half) * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    weight_ctf_half = shift_ctf_half if score_real_dtype is not None else ctf_half
    weight_noise_half = shift_noise_half if score_real_dtype is not None else noise_variance_half
    ctf2_over_nv_half = weight_ctf_half**2 / weight_noise_half
    return shifted_half, batch_norm, ctf2_over_nv_half


def prepare_reconstruction_batch(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
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
    )
    shift_processed_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    return apply_half_translation_phases(
        shift_processed_half * shift_ctf_half / shift_noise_half,
        shift_phases_half,
    )


def preprocess_batch_firstiter_cc(
    experiment_dataset,
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    score_with_masked_images=False,
    window_indices=None,
    *,
    score_complex_dtype=None,
    score_real_dtype=None,
    norm_real_dtype=None,
):
    """Preprocess one dense image batch for RELION's iter-1 normalized CC scoring.

    RELION ml_optimiser.cpp:7967-7978 computes ``exp_local_sqrtXi2`` as
    ``sqrt(sum(|Fimg|^2))`` over the *windowed* half-image (post
    ``windowFourierTransform`` to ``current_size``) with NO Hermitian
    doubling. The CC denominator at ml_optimiser.cpp:8773 then divides by
    ``sqrt(suma2) * exp_local_sqrtXi2[img_id]`` where both sums use the same
    windowed pixel set. To match: when ``window_indices`` is supplied, sum
    only over those pixels; otherwise fall back to the full half-image.
    """

    processed_half, ctf_half, noise_variance_half, translation_phases_half = _dense_batch_half_inputs(
        experiment_dataset,
        batch,
        ctf_params,
        noise_variance,
        translations,
        config,
        score_with_masked_images,
    )
    # RELION ml_optimiser.cpp:8758-8774 (do_firstiter_cc CC branch) iterates
    # `Frefctf = CTF * F_proj` against `Fimg_shift = Fimg * shift_phase` directly:
    #
    #   diff2 -= Frefctf.real * Fimg_shift.real
    #   diff2 -= Frefctf.imag * Fimg_shift.imag
    #
    # Recovar's score formula has CTF on the image side via `shifted_score =
    # Fimg * CTF * shift / Xi2`. Previously this was produced via
    # divide-then-multiply (`Fimg / CTF * shift` then `* ctf^2 / Xi2`). Build
    # `Fimg * CTF * shift` directly here to match RELION's path and avoid the
    # 1/CTF inversion at low-CTF pixels (which was thresholded to 0 below
    # |CTF|<1e-8 and could lose precision near the threshold).
    shift_processed_half, shift_ctf_half, _, shift_phases_half = _cast_shift_inputs(
        processed_half,
        ctf_half,
        translation_phases_half=translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    shifted_half = apply_half_translation_phases(
        shift_processed_half * shift_ctf_half,
        shift_phases_half,
    )
    norm_processed_half, _, _ = _norm_inputs(
        processed_half,
        norm_real_dtype=norm_real_dtype,
    )
    abs2_half = jnp.abs(norm_processed_half) ** 2
    if window_indices is not None:
        windowed_abs2 = abs2_half[:, window_indices]
        image_power = jnp.sum(windowed_abs2, axis=-1, keepdims=True).real
    else:
        image_power = jnp.sum(abs2_half, axis=-1, keepdims=True).real
    weight_ctf_half = shift_ctf_half if score_real_dtype is not None else ctf_half
    weight_noise_half = (
        noise_variance_half.astype(score_real_dtype) if score_real_dtype is not None else noise_variance_half
    )
    ctf2_half = weight_ctf_half**2
    ctf2_over_nv_half = ctf2_half / weight_noise_half
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
