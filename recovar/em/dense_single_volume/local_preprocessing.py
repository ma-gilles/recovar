"""Host orchestration of score, reconstruction and noise inputs for local buckets.

The split path reuses the compiled preprocessing primitive for exact BPref
operands. Keep its cast/order policy distinct from the general image backend;
local_big_jit owns fused execution, while this module owns the split preparation.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import relion_ctf
from recovar.em.dense_single_volume.helpers.image_shifts import (
    apply_relion_integer_pre_shifts,
    integer_pre_shifts_or_none,
)
from recovar.em.dense_single_volume.helpers.jax_runtime import block_until_ready as _block_until_ready
from recovar.em.dense_single_volume.helpers.preprocessing import (
    _cast_shift_inputs,
    _norm_inputs,
    prepare_batch_preprocess_operands,
    process_half_image,
    resolve_image_mask_for_half_preprocess,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    apply_half_translation_phases as _apply_half_translation_phases,
)
from recovar.em.dense_single_volume.helpers.preprocessing import (
    half_translation_phase_table as _half_translation_phase_table,
)
from recovar.em.dense_single_volume.local_big_jit import _preprocess_half as _big_jit_preprocess_half
from recovar.em.dense_single_volume.local_caches import _LocalProcessedHalfCache


def _translate_bpref_images(images, weighted_ctf_half, translation_angles, image_shape):
    """Keep BPref image/CTF operand rounding separate from score translation."""
    if translation_angles is None:
        raise ValueError("exact RELION BPref operands require RELION translation angles")
    from recovar import cuda_backproject

    return cuda_backproject.relion_translate_bpref_f32(
        jnp.asarray(images, dtype=jnp.complex64),
        jnp.asarray(weighted_ctf_half, dtype=jnp.float32),
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.arange(images.shape[1], dtype=jnp.int32),
        image_shape,
    )


def prepare_local_bucket(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    noise_variance_half,
    translation_phases_half,
    config,
    norm_half_weights,
    score_with_masked_images: bool,
    relion_score_translation_angles=None,
    image_pre_shifts=None,
    processed_half_cache: _LocalProcessedHalfCache | None = None,
    timer: dict[str, float] | None = None,
    synchronize_profile: bool = False,
    score_complex_dtype=None,
    score_real_dtype=None,
    norm_real_dtype=None,
    relion_exact_bpref_operands: bool = False,
):
    """Prepare score, reconstruction, and noise inputs for one local bucket.

    This keeps the exact-local path separate from the dense engine and avoids
    recomputing CTF / translation tiling scaffolding across masked, unmasked,
    and noise-specific preprocessing.
    """

    integer_t0 = time.time()
    real_space_pre_shift_applied = False
    if processed_half_cache is None or not processed_half_cache.integer_pre_shifts_applied:
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, image_indices, batch=batch)
        if integer_pre_shifts is not None:
            batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)
            real_space_pre_shift_applied = True
    else:
        integer_pre_shifts = None
        real_space_pre_shift_applied = True
    if timer is not None:
        timer["integer_shift_s"] += time.time() - integer_t0

    phase_t0 = time.time()
    translation_phases_half = jnp.asarray(translation_phases_half)
    raw_translations = translation_phases_half.shape[-1] == len(config.image_shape)
    if raw_translations:
        # Backward compatibility for tests and direct callers that pass raw
        # translations instead of the precomputed phase table used by the hot path.
        translation_phases_half = _half_translation_phase_table(
            translation_phases_half,
            config.image_shape,
        )
    if raw_translations and synchronize_profile:
        _block_until_ready(translation_phases_half)
    if raw_translations and timer is not None:
        timer["translation_phase_s"] += time.time() - phase_t0

    exact_image_mask = None
    exact_image_mask_mode = None
    if relion_exact_bpref_operands:
        exact_image_mask, exact_image_mask_mode = resolve_image_mask_for_half_preprocess(
            experiment_dataset,
            config.image_shape,
            require_mask=score_with_masked_images,
        )

    def _process_half(apply_image_mask: bool):
        if relion_exact_bpref_operands:
            # Contribution capture forces the split scorer, but its operands
            # must still be those of the production big-JIT path.  In
            # particular, do not fall back through the image-source backend:
            # a relion_cuda source requires separate normalization/shift
            # operands and would introduce a different preprocessing graph.
            return _big_jit_preprocess_half(
                jnp.asarray(batch),
                jnp.asarray(exact_image_mask),
                config,
                apply_image_mask=apply_image_mask,
                mask_mode=exact_image_mask_mode,
            )
        # Integer shifts are applied to ``batch`` above and image/scale
        # corrections are applied to the Fourier result by the caller.  The
        # strict RELION CUDA backend still requires explicit typed operands at
        # its boundary, so provide the corresponding identity values here.
        # Other backends receive ``None`` and retain their existing path.
        _, _, _, _, relion_preprocess_kwargs = prepare_batch_preprocess_operands(
            experiment_dataset,
            batch,
            image_indices,
        )
        return process_half_image(
            experiment_dataset,
            batch,
            apply_image_mask,
            relion_preprocess_kwargs=relion_preprocess_kwargs,
        )

    ctf_t0 = time.time()
    if relion_exact_bpref_operands:
        ctf_rfloat = np.asarray(
            relion_ctf._relion_exact_ctf_half_from_source_star_host(
                experiment_dataset,
                image_indices,
                config.image_shape,
            ),
            dtype=np.float64,
        )
        inverse_noise_rfloat_cast = np.reciprocal(
            np.asarray(noise_variance_half, dtype=np.float64)
        ).astype(np.float32)
        ctf_half = jnp.asarray(ctf_rfloat, dtype=jnp.float64).astype(jnp.float32)
        inverse_noise_half = jnp.asarray(inverse_noise_rfloat_cast, dtype=jnp.float32)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        ctf2_over_nv_recon_half = weighted_ctf_half * ctf_half
        ctf2_over_nv_score_half = jnp.asarray(
            (
                inverse_noise_rfloat_cast[None, :].astype(np.float64)
                * ctf_rfloat
                * ctf_rfloat
            ).astype(np.float32),
            dtype=jnp.float32,
        )
    else:
        ctf_eval_params = (
            jnp.asarray(ctf_params, dtype=score_real_dtype)
            if score_real_dtype is not None
            else ctf_params
        )
        ctf_half = config.compute_ctf_half(ctf_eval_params)
        ctf2_over_nv_ctf = ctf_half.astype(score_real_dtype) if score_real_dtype is not None else ctf_half
        ctf2_over_nv_noise = (
            noise_variance_half.astype(score_real_dtype) if score_real_dtype is not None else noise_variance_half
        )
        ctf2_over_nv_recon_half = ctf2_over_nv_ctf**2 / ctf2_over_nv_noise
        ctf2_over_nv_score_half = ctf2_over_nv_recon_half
        weighted_ctf_half = None
    if synchronize_profile:
        _block_until_ready(ctf2_over_nv_score_half, ctf2_over_nv_recon_half)
    if timer is not None:
        timer["ctf_s"] += time.time() - ctf_t0

    if processed_half_cache is None:
        score_process_t0 = time.time()
        processed_score_half = _process_half(score_with_masked_images)
        if synchronize_profile:
            _block_until_ready(processed_score_half)
        if timer is not None:
            timer["score_process_s"] += time.time() - score_process_t0
    else:
        cache_fetch_t0 = time.time()
        processed_score_half = jnp.asarray(processed_half_cache.score_half[np.asarray(image_indices, dtype=np.int32)])
        if timer is not None:
            timer["cache_fetch_s"] += time.time() - cache_fetch_t0

    shift_score_t0 = time.time()
    shift_processed_score_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
        processed_score_half,
        ctf_half,
        noise_variance_half,
        translation_phases_half,
        score_complex_dtype=score_complex_dtype,
        score_real_dtype=score_real_dtype,
    )
    score_weighted_half = (
        shift_processed_score_half * weighted_ctf_half
        if relion_exact_bpref_operands
        else shift_processed_score_half * shift_ctf_half / shift_noise_half
    )
    if relion_score_translation_angles is not None:
        from recovar import cuda_backproject

        translation_dtype = jnp.asarray(relion_score_translation_angles).dtype

        def _cuda_translate(weighted_half):
            pixel_indices = jnp.arange(weighted_half.shape[1], dtype=jnp.int32)
            if translation_dtype == jnp.dtype(jnp.float64):
                return cuda_backproject.relion_translate_score_f64(
                    jnp.asarray(weighted_half, dtype=jnp.complex128),
                    jnp.asarray(relion_score_translation_angles, dtype=jnp.float64),
                    pixel_indices,
                    config.image_shape,
                )
            return cuda_backproject.relion_translate_score_f32(
                jnp.asarray(weighted_half, dtype=jnp.complex64),
                jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
                pixel_indices,
                config.image_shape,
            )

        shifted_score_half = _cuda_translate(score_weighted_half)
    else:
        shifted_score_half = _apply_half_translation_phases(score_weighted_half, shift_phases_half)
    if synchronize_profile:
        _block_until_ready(shifted_score_half)
    if timer is not None:
        timer["tile_shift_score_s"] += time.time() - shift_score_t0

    norm_t0 = time.time()
    norm_processed_score_half, norm_noise_half, norm_weights = _norm_inputs(
        processed_score_half,
        noise_variance_half,
        norm_half_weights,
        norm_real_dtype=norm_real_dtype,
    )
    norm_power_over_noise = (
        jnp.abs(norm_processed_score_half) ** 2
        * inverse_noise_half.astype(norm_processed_score_half.real.dtype)[None, :]
        if relion_exact_bpref_operands
        else jnp.abs(norm_processed_score_half) ** 2 / norm_noise_half
    )
    batch_norm = jnp.sum(
        norm_power_over_noise * norm_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real
    if synchronize_profile:
        _block_until_ready(batch_norm)
    if timer is not None:
        timer["norm_s"] += time.time() - norm_t0

    if score_with_masked_images:
        if processed_half_cache is None:
            recon_process_t0 = time.time()
            processed_recon_half = _process_half(False)
            if synchronize_profile:
                _block_until_ready(processed_recon_half)
            if timer is not None:
                timer["recon_process_s"] += time.time() - recon_process_t0
        else:
            cache_fetch_t0 = time.time()
            if processed_half_cache.recon_half is None:
                raise RuntimeError("processed half-image cache is missing unmasked reconstruction images")
            processed_recon_half = jnp.asarray(
                processed_half_cache.recon_half[np.asarray(image_indices, dtype=np.int32)]
            )
            if timer is not None:
                timer["cache_fetch_s"] += time.time() - cache_fetch_t0

        shift_recon_t0 = time.time()
        shift_processed_recon_half, shift_ctf_half, shift_noise_half, shift_phases_half = _cast_shift_inputs(
            processed_recon_half,
            ctf_half,
            noise_variance_half,
            translation_phases_half,
            score_complex_dtype=score_complex_dtype,
            score_real_dtype=score_real_dtype,
        )
        recon_weighted_half = (
            shift_processed_recon_half * weighted_ctf_half
            if relion_exact_bpref_operands
            else shift_processed_recon_half * shift_ctf_half / shift_noise_half
        )
        if relion_exact_bpref_operands:
            shifted_recon_half = _translate_bpref_images(
                processed_recon_half, weighted_ctf_half, relion_score_translation_angles, config.image_shape,
            )
        elif relion_score_translation_angles is not None and translation_dtype == jnp.dtype(jnp.float64):
            shifted_recon_half = _cuda_translate(recon_weighted_half)
        else:
            shifted_recon_half = _apply_half_translation_phases(
                recon_weighted_half,
                shift_phases_half,
            )
        if synchronize_profile:
            _block_until_ready(shifted_recon_half)
        if timer is not None:
            timer["tile_shift_recon_s"] += time.time() - shift_recon_t0
    else:
        if relion_exact_bpref_operands:
            shifted_recon_half = _translate_bpref_images(
                processed_score_half, weighted_ctf_half, relion_score_translation_angles, config.image_shape,
            )
        elif relion_score_translation_angles is None or translation_dtype == jnp.dtype(jnp.float64):
            shifted_recon_half = shifted_score_half
        else:
            shifted_recon_half = _apply_half_translation_phases(
                score_weighted_half,
                shift_phases_half,
            )
    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_score_half,
        ctf2_over_nv_recon_half,
        processed_score_half,
        real_space_pre_shift_applied,
    )

