"""Per-bucket input preparation of the sparse bucketed pass 2.

The bucket's images, CTF and noise operands, its translation phase tables and
RELION translation angles, and the small CTF/noise algebra those operands
share. ``sparse_pass2_bucketed`` prepares every bucket through this owner.
"""

from __future__ import annotations

import functools
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.em.diagnostics import finite_check
from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from recovar.em.helpers.image_shifts import apply_relion_integer_pre_shifts, half_image_phase_factors
from recovar.em.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
    relion_half_translation_lattice,
    prepare_batch_preprocess_operands,
    process_half_image,
)
from recovar.em.relion import relion_ctf
from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_corr_img_from_native_noise_variance,
    _relion_cuda_corr_img_from_rfloat_ctf,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _scaled_half_lattice_cached(image_shape):
    """Scaled packed-half frequency lattice, computed once per image shape.

    Same device arithmetic as before (byte-identical result); the eager JAX
    grid/meshgrid/index chain was being re-dispatched for every bucket of the
    sparse pass (~3% of a warm ordinary iteration on the 10k subset).
    """

    # RELION's label for the packed Nyquist row of an uncropped half image; see
    # recovar/em/helpers/preprocessing.py::relion_half_translation_lattice.
    return jnp.asarray(
        relion_half_translation_lattice(tuple(int(size) for size in image_shape))
    )


def _half_translation_phase_table_for_indices(translations, image_shape, pixel_indices):
    lattice_half = _scaled_half_lattice_cached(tuple(int(size) for size in image_shape))
    lattice_window = lattice_half[jnp.asarray(pixel_indices, dtype=jnp.int32)]
    phase_arg = jnp.einsum(
        "td,pd->tp",
        jnp.asarray(translations, dtype=jnp.float32),
        lattice_window,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.exp(-2j * jnp.pi * phase_arg)


def _translation_phase_table_for_indices(
    translations,
    image_shape,
    pixel_indices,
    translation_phases_half,
):
    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    if translation_phases_half is None:
        return _half_translation_phase_table_for_indices(translations, image_shape, pixel_indices)
    return translation_phases_half[:, pixel_indices]


def _relion_translation_angles_f32(translations, image_shape):
    """Return RELION fine-score ``(tx, ty)`` radians with host rounding."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError(
            "RELION score translations must have shape (T, 2), got "
            f"{translations_f64.shape}"
        )
    return np.asarray(
        -2.0 * np.pi * translations_f64 / float(image_size),
        dtype=np.float32,
    )


def _relion_translation_angles_f64(translations, image_shape):
    """Return RELION double-ACC ``(tx, ty)`` translation radians."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError(
            "RELION score translations must have shape (T, 2), got "
            f"{translations_f64.shape}"
        )
    return -2.0 * np.pi * translations_f64 / float(image_size)


def _relion_cuda_score_translation_angles_if_available(
    translations,
    image_shape,
    *,
    enabled,
    dtype=np.float32,
):
    """Prepare exact score-translation angles or retain the JAX fallback."""

    if not enabled or jax.default_backend() != "gpu":
        return None
    from recovar import cuda_backproject

    if not cuda_backproject.cuda_available():
        logger.warning(
            "Exact RELION fine Gaussian scoring is retaining JAX translation "
            "phase arithmetic because custom CUDA is unavailable"
        )
        return None
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"RELION CUDA translation dtype must be float32 or float64, got {dtype}")
    logger.info(
        "Exact RELION fine Gaussian scoring: using CUDA %s score/M-step translation",
        "sincosf" if dtype == np.dtype(np.float32) else "sincos",
    )
    return jnp.asarray(
        np.asarray(
            -2.0 * np.pi * np.asarray(translations, dtype=np.float64) / float(image_shape[0]),
            dtype=dtype,
        ),
        dtype=dtype,
    )


def _reorder_to_indices(image_indices_returned, requested_image_indices, *arrays):
    """Reorder per-image arrays so they match the order returned by the dataset."""
    if np.array_equal(image_indices_returned, requested_image_indices):
        return arrays
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(requested_image_indices).tolist())}
    order = np.array([position[int(idx)] for idx in np.asarray(image_indices_returned).tolist()], dtype=np.int64)
    return tuple(arr[order] for arr in arrays)


@jax.jit
def _take_columns(values, column_indices):
    """Gather Fourier-window columns in one compiled program per shape."""

    return values[:, column_indices]


@jax.jit
def _ctf2_over_noise_and_ctf2(ctf_half, noise_variance_half):
    """Return ``CTF^2 / sigma2`` and ``CTF^2`` for the generic bucket path."""

    return ctf_half**2 / noise_variance_half, ctf_half**2


@partial(jax.jit, static_argnames=("multiply_inverse_noise",))
def _gaussian_batch_norm(processed_score_half_raw, noise_operand, norm_half_weights, *, multiply_inverse_noise: bool):
    """Dense ``run_em`` image-norm term: weighted |image|^2 / sigma2 summed per image."""

    power = jnp.abs(processed_score_half_raw) ** 2
    if multiply_inverse_noise:
        score_power_over_noise = power * noise_operand[None, :]
    else:
        score_power_over_noise = power / noise_operand
    return jnp.sum(
        score_power_over_noise * norm_half_weights[None, :],
        axis=-1,
        keepdims=True,
    ).real


@jax.jit
def _ctf_over_noise_weighted_pair(processed_score_half_raw, processed_recon_half_raw, ctf_half, noise_variance_half):
    """Weight score and reconstruction images by ``CTF / sigma2`` in one program."""

    return (
        processed_score_half_raw * ctf_half / noise_variance_half,
        processed_recon_half_raw * ctf_half / noise_variance_half,
    )


@jax.jit
def _divide_by_safe_ctf(sparse_score_input_half, ctf_half):
    """Divide by the CTF where it is safely non-zero, else pass through."""

    ctf_safe = jnp.abs(ctf_half) > 1e-8
    return jnp.where(
        ctf_safe,
        sparse_score_input_half / ctf_half,
        sparse_score_input_half,
    )


class UnshiftedBucketOperands(NamedTuple):
    """Every per-image operand :func:`_prepare_bucket_io` builds before shifting.

    The bucket preparation is two halves: a per-image half that weights, corrects
    and pre-centers each image, and a translation half that tiles the result over
    the fine translations. Only the first half is per-image pure, so it is the
    half the device-resident driver hoists to once per pass over a set of images
    (:mod:`recovar.em.sparse_pass2.resident_operands`). This record is that half's
    output, named exactly as the locals it replaced; :func:`_prepare_bucket_io`
    consumes it and appends the translation half unchanged.
    """

    image_shape: object
    use_normalized_cc: object
    relion_cuda_preprocess: object
    integer_pre_shifts: object
    batch_corr_np: object
    batch_scale_np: object
    relion_preprocess_kwargs: object
    real_space_pre_shift_applied: object
    ctf_half_rfloat: object
    acc_real_dtype: object
    ctf_half: object
    batch_scale: object
    direct_pixel_correction_full: object
    inverse_noise_half: object
    weighted_ctf_half: object
    ctf2_over_nv_half: object
    ctf2_score_half: object
    ctf2_over_nv_recon_half: object
    processed_score_half_raw: object
    processed_recon_half_raw: object
    batch_norm: object
    score_weighted_half: object
    recon_weighted_half: object
    recon_bpref_input_half: object
    folded_normalized_cc_operands: object
    sparse_score_input_half: object
    processed_score_half_for_noise: object


def prepare_unshifted_bucket_operands(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    *,
    noise_variance_half,
    config,
    score_with_masked_images,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    return_direct_scoring_io=False,
    score_only=False,
    score_mode="gaussian",
    window_indices=None,
    relion_exact_normalized_cc_operands=False,
    relion_exact_bpref_operands=False,
) -> UnshiftedBucketOperands:
    """Per-image half of :func:`_prepare_bucket_io`, statement for statement.

    Extracted so the resident driver can prepare these operands once for a whole
    half instead of once per chunk. Nothing here depends on the fine translations,
    so a call covering any set of images returns exactly the rows a per-bucket
    call would; the translation-dependent work stays in :func:`_prepare_bucket_io`.
    """

    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")

    image_shape = config.image_shape
    use_normalized_cc = score_mode == "normalized_cc"
    (
        relion_cuda_preprocess,
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        relion_preprocess_kwargs,
    ) = prepare_batch_preprocess_operands(
        experiment_dataset,
        batch,
        image_indices,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        dtype=(np.float64 if use_float64_scoring else np.float32),
    )
    real_space_pre_shift_applied = integer_pre_shifts is not None
    if real_space_pre_shift_applied and not relion_cuda_preprocess:
        batch = apply_relion_integer_pre_shifts(batch, integer_pre_shifts)

    ctf_half_rfloat = (
        relion_ctf._relion_exact_ctf_half_from_source_star(
            experiment_dataset,
            image_indices,
            image_shape,
        )
        if relion_exact_bpref_operands
        else None
    )
    acc_real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    ctf_half = (
        jnp.asarray(ctf_half_rfloat, dtype=acc_real_dtype)
        if ctf_half_rfloat is not None
        else config.compute_ctf_half(jnp.asarray(ctf_params, dtype=acc_real_dtype))
    )
    batch_scale = jnp.asarray(batch_scale_np, dtype=ctf_half.dtype)
    relion_score_corr_img_half = None
    direct_pixel_correction_full = None
    if relion_exact_bpref_operands:
        if relion_preprocess_kwargs is None:
            raise ValueError(
                "exact RELION BPref operands require RELION CUDA preprocessing"
            )
        relion_preprocess_kwargs = dict(relion_preprocess_kwargs)
        relion_preprocess_kwargs["relion_fft_per_image"] = True
        # RELION computes minvsigma2 from its binary64 sigma2 spectrum, then
        # stores the reciprocal as XFLOAT. Preserve that cast boundary and
        # its scalar multiplication order instead of dividing by an already
        # rounded variance. XFLOAT is float64 under ACC_DOUBLE_PRECISION.
        inverse_noise_half = jnp.reciprocal(
            jnp.asarray(noise_variance_half, dtype=jnp.float64)
        ).astype(acc_real_dtype)
        weighted_ctf_half = ctf_half * inverse_noise_half[None, :]
        ctf2_over_nv_half = weighted_ctf_half * ctf_half
        relion_score_corr_img_half = _relion_cuda_corr_img_from_native_noise_variance(
            noise_variance_half[None, :],
            ctf_half_rfloat,
            image_shape,
            batch_scale[:, None] if scale_corrections is not None else None,
            output_dtype=acc_real_dtype,
        )
        ctf2_score_half = ctf_half**2
    else:
        inverse_noise_half = None
        weighted_ctf_half = None
        ctf2_over_nv_half, ctf2_score_half = _ctf2_over_noise_and_ctf2(ctf_half, noise_variance_half)

    # Raw processed half-spectrum images (BEFORE any per-image correction).
    # The score path uses masked images iff ``score_with_masked_images`` is True,
    # while the reconstruction path always uses the unmasked (raw) images.
    processed_score_half_raw = process_half_image(
        experiment_dataset,
        batch,
        score_with_masked_images,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
    )
    if score_with_masked_images:
        processed_recon_half_raw = process_half_image(
            experiment_dataset,
            batch,
            False,
            relion_preprocess_kwargs=relion_preprocess_kwargs,
        )
    else:
        processed_recon_half_raw = processed_score_half_raw

    if use_normalized_cc:
        # RELION firstiter_cc uses unweighted image power over the same Fourier
        # window as the score denominator, with no Hermitian doubling.
        if relion_exact_normalized_cc_operands:
            # ``exp_local_sqrtXi2`` is formed from an RFLOAT (float64) serial
            # sum before RELION casts its reciprocal to XFLOAT.  A float32 XLA
            # reduction can move that reciprocal by one ULP, which is enough
            # to erase demonstrated fine-translation score margins.
            norm_real = processed_score_half_raw.real.astype(jnp.float64)
            norm_imag = processed_score_half_raw.imag.astype(jnp.float64)
            abs2_half = norm_real * norm_real + norm_imag * norm_imag
        else:
            abs2_half = jnp.abs(processed_score_half_raw) ** 2
        if window_indices is not None:
            abs2_half = abs2_half[:, window_indices]
        batch_norm = jnp.sum(abs2_half, axis=-1, keepdims=True).real
    else:
        # batch_norm starts from raw processed-score images, then follows dense
        # run_em's image-only correction convention below.
        norm_half_weights = make_half_image_weights(image_shape)
        batch_norm = _gaussian_batch_norm(
            processed_score_half_raw,
            inverse_noise_half if relion_exact_bpref_operands else noise_variance_half,
            norm_half_weights,
            multiply_inverse_noise=bool(relion_exact_bpref_operands),
        )

    if relion_exact_bpref_operands:
        score_weighted_half = processed_score_half_raw * weighted_ctf_half
        recon_weighted_half = processed_recon_half_raw * weighted_ctf_half
        recon_bpref_input_half = processed_recon_half_raw
    else:
        score_weighted_half, recon_weighted_half = _ctf_over_noise_weighted_pair(
            processed_score_half_raw,
            processed_recon_half_raw,
            ctf_half,
            noise_variance_half,
        )
        recon_bpref_input_half = None
    folded_normalized_cc_operands = (
        use_normalized_cc and not relion_exact_normalized_cc_operands
    )
    sparse_score_input_half = (
        processed_score_half_raw * ctf_half
        if folded_normalized_cc_operands
        else processed_score_half_raw
    )
    processed_score_half_for_noise = processed_score_half_raw

    # Per-image image corrections follow dense run_em's image-only convention.
    if image_corrections is not None:
        batch_corr = jnp.asarray(batch_corr_np)
        image_only_corr = batch_corr / batch_scale
        # Note: corrections are applied to the per-translation-tiled arrays in
        # run_em, but multiplication by a per-image scalar commutes with the
        # tiling and shifting so we apply it before tiling for efficiency.
        applied_corr = batch_scale if relion_cuda_preprocess else batch_corr
        score_weighted_half = score_weighted_half * applied_corr[:, None]
        recon_weighted_half = recon_weighted_half * applied_corr[:, None]
        if relion_exact_bpref_operands:
            recon_bpref_input_half = recon_bpref_input_half * applied_corr[:, None]
        if return_direct_scoring_io:
            direct_raw_corr = batch_corr / batch_scale
            if folded_normalized_cc_operands:
                sparse_score_input_half = sparse_score_input_half * applied_corr[:, None]
            elif not relion_cuda_preprocess:
                sparse_score_input_half = sparse_score_input_half * direct_raw_corr[:, None]
        if not relion_cuda_preprocess:
            batch_norm = batch_norm * (image_only_corr**2)[:, None]
            processed_score_half_for_noise = processed_score_half_for_noise * image_only_corr[:, None]

    # Per-image scale correction on CTF^2/noise.
    if scale_corrections is not None:
        ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
        ctf2_score_half = ctf2_score_half * (batch_scale**2)[:, None]
        if return_direct_scoring_io:
            if not folded_normalized_cc_operands and not relion_exact_bpref_operands:
                sparse_score_input_half = sparse_score_input_half / batch_scale[:, None]

    # BPref operands remain in their demonstrated native XFLOAT order. Only
    # fine-score corr_img uses RELION's distinct RFLOAT-square construction.
    ctf2_over_nv_recon_half = ctf2_over_nv_half
    if relion_score_corr_img_half is not None:
        ctf2_over_nv_half = relion_score_corr_img_half

    if return_direct_scoring_io and not folded_normalized_cc_operands:
        if relion_exact_bpref_operands:
            pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
                batch_scale[:, None],
                ctf_half_rfloat,
                output_dtype=acc_real_dtype,
            )
            direct_pixel_correction_full = pixel_correction
            sparse_score_input_half = sparse_score_input_half * pixel_correction
        else:
            sparse_score_input_half = _divide_by_safe_ctf(sparse_score_input_half, ctf_half)
    if score_only and not return_direct_scoring_io:
        raise ValueError("score-only sparse pass-2 requires direct scoring I/O")

    # Per-image pre-centering: phase shift in Fourier space after scalar corrections.
    if image_pre_shifts is not None and not real_space_pre_shift_applied:
        batch_shifts = jnp.asarray(np.asarray(image_pre_shifts)[np.asarray(image_indices)])
        phase_factors = half_image_phase_factors(image_shape, batch_shifts, dtype=batch_shifts.dtype)
        if not score_only:
            score_weighted_half = score_weighted_half * phase_factors
            recon_weighted_half = recon_weighted_half * phase_factors
            if relion_exact_bpref_operands:
                recon_bpref_input_half = recon_bpref_input_half * phase_factors
        if return_direct_scoring_io:
            sparse_score_input_half = sparse_score_input_half * phase_factors

    if finite_check.finite_check_enabled():
        # P4-D. The per-image per-pixel operands, before any translation or
        # posterior weighting. The accumulator geometry of job 14178118 points
        # at exactly these: a scattered corruption here ruins the weighted
        # image sum and the CTF sum at the same pixel of the same image, which
        # is the only operand-level fault consistent with the nested bad voxel
        # sets and their flat radial profile.
        finite_check.check_bundle(
            "bucket-io-operands",
            {
                "ctf_half": ctf_half,
                "inverse_noise_half": inverse_noise_half,
                "weighted_ctf_half": weighted_ctf_half,
                "ctf2_over_nv_half": ctf2_over_nv_half,
                "ctf2_over_nv_recon_half": ctf2_over_nv_recon_half,
                "recon_weighted_half": recon_weighted_half,
                "recon_bpref_input_half": recon_bpref_input_half,
                "batch_scale": batch_scale,
            },
            dominated_by=(
                # weighted_ctf is the CTF times the inverse noise, and
                # ctf2_over_nv is that times the CTF again times the squared
                # per-image scale. Both are elementwise, so each maximum is
                # bounded by the product of its factors' maxima. These fire at
                # any magnitude, unlike a finiteness test, and they watch the
                # one operand family that enters both M-step sums with the
                # same power, which is what the accumulator nesting requires.
                ("weighted_ctf_half", ("ctf_half", "inverse_noise_half"), 1e-3),
                (
                    "ctf2_over_nv_recon_half",
                    ("weighted_ctf_half", "ctf_half", "batch_scale", "batch_scale"),
                    1e-3,
                ),
            ),
            context=finite_check.describe_context(
                images=int(np.asarray(image_indices).size),
                first_image=int(np.asarray(image_indices).reshape(-1)[0]),
            ),
            image_ids=image_indices,
        )
    return UnshiftedBucketOperands(
        image_shape=image_shape,
        use_normalized_cc=use_normalized_cc,
        relion_cuda_preprocess=relion_cuda_preprocess,
        integer_pre_shifts=integer_pre_shifts,
        batch_corr_np=batch_corr_np,
        batch_scale_np=batch_scale_np,
        relion_preprocess_kwargs=relion_preprocess_kwargs,
        real_space_pre_shift_applied=real_space_pre_shift_applied,
        ctf_half_rfloat=ctf_half_rfloat,
        acc_real_dtype=acc_real_dtype,
        ctf_half=ctf_half,
        batch_scale=batch_scale,
        direct_pixel_correction_full=direct_pixel_correction_full,
        inverse_noise_half=inverse_noise_half,
        weighted_ctf_half=weighted_ctf_half,
        ctf2_over_nv_half=ctf2_over_nv_half,
        ctf2_score_half=ctf2_score_half,
        ctf2_over_nv_recon_half=ctf2_over_nv_recon_half,
        processed_score_half_raw=processed_score_half_raw,
        processed_recon_half_raw=processed_recon_half_raw,
        batch_norm=batch_norm,
        score_weighted_half=score_weighted_half,
        recon_weighted_half=recon_weighted_half,
        recon_bpref_input_half=recon_bpref_input_half,
        folded_normalized_cc_operands=folded_normalized_cc_operands,
        sparse_score_input_half=sparse_score_input_half,
        processed_score_half_for_noise=processed_score_half_for_noise,
    )


def _prepare_bucket_io(
    experiment_dataset,
    batch,
    ctf_params,
    image_indices,
    noise_variance_half,
    fine_translations,
    config,
    n_trans,
    score_with_masked_images,
    half_spectrum_scoring,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    return_direct_scoring_io=False,
    score_only=False,
    score_mode="gaussian",
    window_indices=None,
    recon_window_indices=None,
    translation_phases_half=None,
    score_translation_phases=None,
    recon_translation_phases=None,
    relion_score_translation_angles=None,
    return_windowed_shifted=False,
    return_shifted_score=True,
    relion_exact_normalized_cc_operands=False,
    relion_exact_bpref_operands=False,
):
    """Run preprocessing for a batch of images (translations tiled, CTF/noise ratios).

    Mirrors the ``run_em``/``_preprocess_batch`` pipeline exactly so the
    bucketed sparse pass-2 path is bit-for-bit identical to calling
    ``run_em`` per image.
    """
    if return_windowed_shifted:
        if window_indices is None:
            raise ValueError("return_windowed_shifted requires window_indices")
        if recon_window_indices is None:
            recon_window_indices = window_indices

    unshifted = prepare_unshifted_bucket_operands(
        experiment_dataset,
        batch,
        ctf_params,
        image_indices,
        noise_variance_half=noise_variance_half,
        config=config,
        score_with_masked_images=score_with_masked_images,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=use_float64_scoring,
        return_direct_scoring_io=return_direct_scoring_io,
        score_only=score_only,
        score_mode=score_mode,
        window_indices=window_indices,
        relion_exact_normalized_cc_operands=relion_exact_normalized_cc_operands,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
    )
    (
        image_shape,
        use_normalized_cc,
        relion_cuda_preprocess,
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        relion_preprocess_kwargs,
        real_space_pre_shift_applied,
        ctf_half_rfloat,
        acc_real_dtype,
        ctf_half,
        batch_scale,
        direct_pixel_correction_full,
        inverse_noise_half,
        weighted_ctf_half,
        ctf2_over_nv_half,
        ctf2_score_half,
        ctf2_over_nv_recon_half,
        processed_score_half_raw,
        processed_recon_half_raw,
        batch_norm,
        score_weighted_half,
        recon_weighted_half,
        recon_bpref_input_half,
        folded_normalized_cc_operands,
        sparse_score_input_half,
        processed_score_half_for_noise,
    ) = unshifted


    score_weighted_half_for_score = score_weighted_half

    if translation_phases_half is None and not return_windowed_shifted:
        translation_phases_half = half_translation_phase_table(fine_translations, image_shape)

    def _cuda_translate_score(values, pixel_indices):
        if relion_score_translation_angles is None:
            return None
        from recovar import cuda_backproject

        values = jnp.asarray(values)
        pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
        if values.dtype == jnp.complex128:
            return cuda_backproject.relion_translate_score_f64(
                values,
                jnp.asarray(relion_score_translation_angles, dtype=jnp.float64),
                pixel_indices,
                image_shape,
            )
        return cuda_backproject.relion_translate_score_f32(
            jnp.asarray(values, dtype=jnp.complex64),
            jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            pixel_indices,
            image_shape,
        )
    if score_only:
        shifted_score_half = None
        shifted_recon_half = None
        shifted_score_half_with_dc = None
        ctf2_over_nv_half_with_dc = None
    else:
        if return_windowed_shifted:
            score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
            recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
            score_phase = (
                score_translation_phases
                if score_translation_phases is not None
                else _translation_phase_table_for_indices(
                    fine_translations,
                    image_shape,
                    score_indices,
                    translation_phases_half,
                )
            )
            recon_phase = (
                recon_translation_phases
                if recon_translation_phases is not None
                else _translation_phase_table_for_indices(
                    fine_translations,
                    image_shape,
                    recon_indices,
                    translation_phases_half,
                )
            )
            shifted_score_half = None
            if return_shifted_score:
                score_weighted_window = _take_columns(score_weighted_half_for_score, score_indices)
                shifted_score_half = _cuda_translate_score(
                    score_weighted_window,
                    score_indices,
                )
                if shifted_score_half is None:
                    shifted_score_half = apply_half_translation_phases(
                        score_weighted_window,
                        score_phase,
                    )
            if relion_exact_bpref_operands:
                if relion_score_translation_angles is None:
                    raise ValueError(
                        "exact RELION BPref operands require RELION translation angles"
                    )
                from recovar import cuda_backproject

                translate_bpref = (
                    cuda_backproject.relion_translate_bpref_f64
                    if use_float64_scoring
                    else cuda_backproject.relion_translate_bpref_f32
                )
                complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
                shifted_recon_half = translate_bpref(
                    jnp.asarray(recon_bpref_input_half[:, recon_indices], dtype=complex_dtype),
                    jnp.asarray(weighted_ctf_half[:, recon_indices], dtype=acc_real_dtype),
                    jnp.asarray(relion_score_translation_angles, dtype=acc_real_dtype),
                    recon_indices,
                    image_shape,
                )
            else:
                recon_weighted_window = _take_columns(recon_weighted_half, recon_indices)
                shifted_recon_half = _cuda_translate_score(
                    recon_weighted_window,
                    recon_indices,
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_window,
                        recon_phase,
                    )
            if score_with_masked_images:
                score_weighted_recon_window = _take_columns(score_weighted_half, recon_indices)
                shifted_score_half_with_dc = _cuda_translate_score(
                    score_weighted_recon_window,
                    recon_indices,
                )
                if shifted_score_half_with_dc is None:
                    shifted_score_half_with_dc = apply_half_translation_phases(
                        score_weighted_recon_window,
                        recon_phase,
                    )
            else:
                shifted_score_half_with_dc = shifted_recon_half
        else:
            if relion_exact_bpref_operands:
                if relion_score_translation_angles is None:
                    raise ValueError(
                        "exact RELION BPref operands require RELION translation angles"
                    )
                from recovar import cuda_backproject

                translate_bpref = (
                    cuda_backproject.relion_translate_bpref_f64
                    if use_float64_scoring
                    else cuda_backproject.relion_translate_bpref_f32
                )
                complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
                exact_shifted_recon_half = translate_bpref(
                    jnp.asarray(recon_bpref_input_half, dtype=complex_dtype),
                    jnp.asarray(weighted_ctf_half, dtype=acc_real_dtype),
                    jnp.asarray(relion_score_translation_angles, dtype=acc_real_dtype),
                    jnp.arange(recon_bpref_input_half.shape[1], dtype=jnp.int32),
                    image_shape,
                )
            else:
                exact_shifted_recon_half = None
            shifted_score_half = None
            if return_shifted_score:
                full_pixel_indices = jnp.arange(score_weighted_half_for_score.shape[1], dtype=jnp.int32)
                shifted_score_half = _cuda_translate_score(
                    score_weighted_half_for_score,
                    full_pixel_indices,
                )
                if shifted_score_half is None:
                    shifted_score_half = apply_half_translation_phases(
                        score_weighted_half_for_score,
                        translation_phases_half,
                    )
            if score_with_masked_images:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else _cuda_translate_score(
                        recon_weighted_half,
                        jnp.arange(recon_weighted_half.shape[1], dtype=jnp.int32),
                    )
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_half,
                        translation_phases_half,
                    )
                shifted_score_half_with_dc = _cuda_translate_score(
                    score_weighted_half,
                    jnp.arange(score_weighted_half.shape[1], dtype=jnp.int32),
                )
                if shifted_score_half_with_dc is None:
                    shifted_score_half_with_dc = apply_half_translation_phases(
                        score_weighted_half,
                        translation_phases_half,
                    )
            else:
                shifted_recon_half = (
                    exact_shifted_recon_half
                    if exact_shifted_recon_half is not None
                    else _cuda_translate_score(
                        recon_weighted_half,
                        jnp.arange(recon_weighted_half.shape[1], dtype=jnp.int32),
                    )
                )
                if shifted_recon_half is None:
                    shifted_recon_half = apply_half_translation_phases(
                        recon_weighted_half,
                        translation_phases_half,
                    )
                shifted_score_half_with_dc = shifted_recon_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_recon_half

    shifted_corrected_score_half = None
    direct_score_input = None
    direct_preprocessed_score_input = None
    direct_pixel_correction = None
    if return_direct_scoring_io:
        if return_windowed_shifted:
            score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
            direct_score_input = _take_columns(sparse_score_input_half, score_indices)
            direct_preprocessed_score_input = _take_columns(processed_score_half_raw, score_indices)
            if direct_pixel_correction_full is not None:
                direct_pixel_correction = _take_columns(direct_pixel_correction_full, score_indices)
            direct_score_pixel_indices = score_indices
        else:
            direct_score_input = sparse_score_input_half
            direct_preprocessed_score_input = processed_score_half_raw
            direct_pixel_correction = direct_pixel_correction_full
            direct_score_pixel_indices = jnp.arange(
                sparse_score_input_half.shape[1],
                dtype=jnp.int32,
            )
        if relion_score_translation_angles is not None:
            from recovar import cuda_backproject

            shifted_corrected_score_half = _cuda_translate_score(
                direct_score_input,
                direct_score_pixel_indices,
            )
        else:
            if return_windowed_shifted:
                score_phase = (
                    score_translation_phases
                    if score_translation_phases is not None
                    else _translation_phase_table_for_indices(
                        fine_translations,
                        image_shape,
                        direct_score_pixel_indices,
                        translation_phases_half,
                    )
                )
            else:
                score_phase = translation_phases_half
            shifted_corrected_score_half = apply_half_translation_phases(
                direct_score_input,
                score_phase,
            )

    if half_spectrum_scoring and not use_normalized_cc:
        dc_shell_idx = make_shell_indices_half(image_shape)
        dc_mask = dc_shell_idx == 0
        if not score_only and shifted_score_half is not None:
            if return_windowed_shifted:
                score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
                shifted_score_half = jnp.where(dc_mask[score_indices][None, :], 0.0, shifted_score_half)
            else:
                shifted_score_half = jnp.where(dc_mask[None, :], 0.0, shifted_score_half)
        ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    if return_direct_scoring_io and use_normalized_cc:
        inv_xi2 = (1.0 / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))).astype(
            precision_policy.score_real_dtype,
        )
        if folded_normalized_cc_operands:
            shifted_corrected_score_half = shifted_corrected_score_half * jnp.repeat(inv_xi2, n_trans, axis=0)
            ctf2_over_nv_half = ctf2_score_half * inv_xi2
        else:
            # buildCorrImage evaluates CTF * CTF in RFLOAT and casts the full
            # product to XFLOAT once.  Reusing the generic float32 CTF-square
            # path changes most corr_img pixels by one or two ULPs.  Preserve
            # the source order here; the existing RECOVAR FFT normalization
            # is already encoded in ``inv_xi2``.
            corr_ctf_rfloat = (
                ctf_half.astype(jnp.float64)
                if ctf_half_rfloat is None
                else ctf_half_rfloat
            )
            ctf2_over_nv_half = _relion_cuda_corr_img_from_rfloat_ctf(
                inv_xi2,
                corr_ctf_rfloat,
                batch_scale[:, None] if scale_corrections is not None else None,
            )
    if return_windowed_shifted:
        score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
        ctf2_over_nv_half = _take_columns(ctf2_over_nv_half, score_indices)
        if ctf2_over_nv_half_with_dc is not None:
            recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
            ctf2_over_nv_half_with_dc = _take_columns(ctf2_over_nv_half_with_dc, recon_indices)
    if score_only:
        ctf2_over_nv_half = ctf2_over_nv_half.astype(precision_policy.score_real_dtype)
    else:
        if shifted_score_half is not None:
            shifted_score_half = shifted_score_half.astype(precision_policy.score_complex_dtype)
        ctf2_over_nv_half = ctf2_over_nv_half.astype(precision_policy.score_real_dtype)
        if precision_policy.use_float64_scoring:
            shifted_recon_half = shifted_recon_half.astype(precision_policy.score_complex_dtype)
            shifted_score_half_with_dc = shifted_score_half_with_dc.astype(precision_policy.score_complex_dtype)
            ctf2_over_nv_half_with_dc = ctf2_over_nv_half_with_dc.astype(precision_policy.score_real_dtype)
    if return_direct_scoring_io:
        shifted_corrected_score_half = shifted_corrected_score_half.astype(
            precision_policy.score_complex_dtype,
        )

    return (
        shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        shifted_corrected_score_half,
        direct_score_input,
        direct_preprocessed_score_input,
        direct_pixel_correction,
        (
            None
            if relion_preprocess_kwargs is None
            else relion_preprocess_kwargs.get("relion_normalization_factors")
        ),
        integer_pre_shifts,
        batch_corr_np,
        batch_scale_np,
        inverse_noise_half,
        ctf_half_rfloat,
    )
