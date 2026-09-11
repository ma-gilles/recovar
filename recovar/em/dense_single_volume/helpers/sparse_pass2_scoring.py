"""Per-bucket and per-pair scoring kernels of the sparse bucketed pass 2.

The algebraic Gaussian score, RELION's accelerated-GPU fine diff2 and its
diff2-to-score conversion, the normalized cross-correlation score, RELION's
powerClass noise operands and the compact-pair variants of each scorer.
``sparse_pass2_bucketed`` selects a scorer per pass and calls it per bucket.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.deterministic_reduce import deterministic_reductions_enabled
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.half_spectrum import bin_shell_values_jax

_RELION_FINE_DIFF2_FUSED_FFI_ENV = "RECOVAR_RELION_FINE_DIFF2_FUSED_FFI"


_RELION_CUDA_FINE_REF3D_BLOCK_SIZE = 256


def _gaussian_algebraic_score_terms(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
):
    """Historical algebraic Gaussian scores of one bucket, before candidate masking.

    ``cross[b, r, t] = Re(conj(shifted[b, t]) . (corr_img_score[b] * half_weights) . proj[b, r])``
    and ``proj_norm[b, r] = 0.5 * (corr_img_score[b] * half_weights) . |proj[b, r]|^2`` in
    HIGHEST precision; the prior-free score is ``cross - proj_norm`` and the score adds the
    rotation and translation log priors. Shared by the production algebraic scorer and its
    components variant, which mask the results differently.
    """

    weights = corr_img_score * half_weights[None, :]
    cross = jnp.einsum(
        "btn,bn,brn->brt",
        jnp.conj(shifted_corrected),
        weights,
        proj_half,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    proj_norm = 0.5 * jnp.einsum(
        "bn,brn->br",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    preprior_scores = cross - proj_norm[:, :, None]
    scores = preprior_scores + rotation_log_prior[:, :, None] + translation_log_prior[:, None, :]
    return preprior_scores, scores


@jax.jit
def _score_pass2_bucket_gaussian_algebraic_components(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
):
    """Return historical algebraic Gaussian scores and their pre-prior terms.

    The extra output is used only by scoped BPref diagnostics. Production exact
    RELION scoring uses the direct ``diff2`` tree below.
    """

    preprior_scores, scores = _gaussian_algebraic_score_terms(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        rotation_log_prior,
        translation_log_prior,
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    preprior_scores = jnp.where(candidate_mask & jnp.isfinite(preprior_scores), preprior_scores, -jnp.inf)
    return scores, preprior_scores


@jax.jit
def _score_pass2_bucket_gaussian_algebraic(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
):
    """Historical algebraic Gaussian scorer used outside exact CUDA mode."""

    _, scores = _gaussian_algebraic_score_terms(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        rotation_log_prior,
        translation_log_prior,
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_bucket_gaussian_algebraic_single_cached(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
):
    """Single-image cached variant of the historical algebraic scorer."""

    weights = corr_img_score * half_weights
    cross = jnp.einsum(
        "tn,n,rn->rt",
        jnp.conj(shifted_corrected),
        weights,
        proj_half,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    proj_norm = 0.5 * jnp.einsum(
        "n,rn->r",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = (
        cross
        - proj_norm[:, None]
        + rotation_log_prior[:, None]
        + translation_log_prior[None, :]
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


def _relion_cuda_fine_reduce_lanes(lanes):
    """Reduce the 256 shared-memory lanes used by RELION CUDA REF3D fine diff2.

    Preserves the caller's dtype (promoted against float32 as a floor) instead
    of forcing float32, so a caller that has already promoted its operands to
    float64 (to match a ``ACC_DOUBLE_PRECISION`` RELION oracle) is not
    silently narrowed back down here.
    """

    lanes = jnp.asarray(lanes, dtype=jnp.result_type(lanes, jnp.float32))
    if lanes.shape[-1] != _RELION_CUDA_FINE_REF3D_BLOCK_SIZE:
        raise ValueError(
            "RELION CUDA fine reduction needs exactly "
            f"{_RELION_CUDA_FINE_REF3D_BLOCK_SIZE} lanes, got {lanes.shape[-1]}"
        )
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    return lanes[..., 0]


def _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, compact_indices):
    """Map RELION's full current-size packed pixel order to compact score rows."""

    image_size = int(image_shape[0])
    current_size = image_size if current_size is None else int(current_size)
    original_half_width = int(image_shape[1]) // 2 + 1
    current_half_width = current_size // 2 + 1
    compact_indices = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    centered_rows = compact_indices // original_half_width
    columns = compact_indices % original_half_width
    ky = centered_rows - image_size // 2
    fftw_rows = np.where(ky < 0, ky + current_size, ky)
    if np.any(fftw_rows < 0) or np.any(fftw_rows >= current_size):
        raise ValueError("compact score indices contain rows outside the RELION current-size crop")
    if np.any(columns < 0) or np.any(columns >= current_half_width):
        raise ValueError("compact score indices contain columns outside the RELION current-size crop")
    relion_flat_indices = fftw_rows * current_half_width + columns
    if np.unique(relion_flat_indices).size != relion_flat_indices.size:
        raise ValueError("compact score indices do not map uniquely into RELION's current-size layout")
    lookup = np.full(current_size * current_half_width, -1, dtype=np.int32)
    lookup[relion_flat_indices] = np.arange(compact_indices.size, dtype=np.int32)
    return lookup


def _relion_cuda_fine_diff2_sum(
    reference,
    shifted_image,
    pixel_weight,
    relion_full_to_compact=None,
    *,
    use_fused_ffi=False,
):
    """Accumulate direct Gaussian diff2 without materializing ``(..., N)``.

    Pinned RELION CUDA's ``REF3D=true, DATA3D=false`` path uses
    ``D2F_BLOCK_SIZE_REF3D=256`` and ``D2F_CHUNK_REF3D=7``. The chunk controls
    translation batching; thread ``tid`` still accumulates pixels
    ``tid + pass * 256`` sequentially in XFLOAT (float32), followed by the
    shared-memory 128,64,...,1 reduction tree. Keeping only the 256 lanes per
    hypothesis avoids the much larger hypothesis-by-pixel temporary.
    """

    complex_dtype = jnp.result_type(reference, shifted_image, jnp.complex64)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_image = jnp.asarray(shifted_image, dtype=complex_dtype)
    pixel_weight = jnp.asarray(pixel_weight, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if shifted_image.shape[-1] != n_values or pixel_weight.shape[-1] != n_values:
        raise ValueError(
            "RELION CUDA fine diff2 operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_image.shape[-1]}, "
            f"weight={pixel_weight.shape[-1]}"
        )
    output_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_image.shape[:-1], pixel_weight.shape[:-1]
    )
    if n_values == 0:
        return jnp.zeros(output_shape, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )
    if bool(use_fused_ffi) or parse_env_flag(
        _RELION_FINE_DIFF2_FUSED_FFI_ENV,
        default=False,
    ):
        from recovar import cuda_backproject

        if reference.ndim == 4:
            if (
                shifted_image.ndim != 4
                or pixel_weight.ndim != 4
                or reference.shape[2] != 1
                or shifted_image.shape[1] != 1
                or pixel_weight.shape[1:3] != (1, 1)
                or reference.shape[0] != shifted_image.shape[0]
                or reference.shape[0] != pixel_weight.shape[0]
            ):
                raise ValueError(
                    "fused rectangular fine diff2 received unsupported broadcast shapes: "
                    f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
                )
            fine_diff2_rectangular = (
                cuda_backproject.relion_fine_diff2_rectangular_f64
                if real_dtype == jnp.float64
                else cuda_backproject.relion_fine_diff2_rectangular_f32
            )
            return fine_diff2_rectangular(
                reference[:, :, 0, :],
                shifted_image[:, 0, :, :],
                pixel_weight[:, 0, 0, :],
                relion_full_to_compact,
            )
        if reference.ndim == 3 and shifted_image.ndim == 3 and pixel_weight.ndim == 3:
            if (
                reference.shape == shifted_image.shape
                and pixel_weight.shape[1] == 1
                and reference.shape[0] == pixel_weight.shape[0]
            ):
                fine_diff2_pairs = (
                    cuda_backproject.relion_fine_diff2_pairs_f64
                    if real_dtype == jnp.float64
                    else cuda_backproject.relion_fine_diff2_pairs_f32
                )
                return fine_diff2_pairs(
                    reference,
                    shifted_image,
                    pixel_weight[:, 0, :],
                    relion_full_to_compact,
                )
            if (
                reference.shape[1] == 1
                and shifted_image.shape[0] == 1
                and pixel_weight.shape[:2] == (1, 1)
            ):
                fine_diff2_rectangular = (
                    cuda_backproject.relion_fine_diff2_rectangular_f64
                    if real_dtype == jnp.float64
                    else cuda_backproject.relion_fine_diff2_rectangular_f32
                )
                return fine_diff2_rectangular(
                    reference[:, 0, :][None, :, :],
                    shifted_image[0, :, :][None, :, :],
                    pixel_weight[0, 0, :][None, :],
                    relion_full_to_compact,
                )[0]
        raise ValueError(
            "fused fine diff2 received unsupported operand ranks/shapes: "
            f"{reference.shape}, {shifted_image.shape}, {pixel_weight.shape}"
        )
    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    lanes = jnp.zeros(output_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        img_pass = jnp.take(shifted_image, safe_rows, axis=-1)
        weight_pass = jnp.take(pixel_weight, safe_rows, axis=-1)
        diff_real = ref_pass.real - img_pass.real
        diff_imag = ref_pass.imag - img_pass.imag
        terms = (
            (diff_real * diff_real + diff_imag * diff_imag)
            * jnp.asarray(0.5, dtype=real_dtype)
            * weight_pass
        )
        terms = jnp.where(valid_pixel, terms, jnp.asarray(0.0, dtype=real_dtype))
        return lane_values + terms

    lanes = jax.lax.fori_loop(0, n_passes, accumulate_pass, lanes)
    return _relion_cuda_fine_reduce_lanes(lanes)


def _relion_cuda_fine_normalized_cc_score(
    reference,
    shifted_score,
    score_weight,
    half_weights,
    relion_full_to_compact=None,
):
    """Reproduce RELION CUDA's 256-lane fine normalized-CC reduction.

    The pinned ``cuda_kernel_diff2_CC_fine<REF3D=true>`` accumulates numerator
    and reference norm over pixels ``tid + pass * 256`` in RELION's XFLOAT,
    then uses the same shared-memory tree as fine Gaussian ``diff2``. XFLOAT is
    float32 in RELION's default accelerated build but float64 whenever
    ``ACC_DOUBLE_PRECISION`` is set (our double-precision oracle build); this
    reduction follows the caller's operand dtype instead of hardcoding
    float32, so it matches whichever precision the RELION oracle actually
    used. RECOVAR stores the score window in centered compact order, so
    ``relion_full_to_compact`` restores RELION's packed current-size FFTW
    pixel order before accumulation.
    """

    complex_dtype = jnp.result_type(reference, shifted_score, jnp.complex64)
    real_dtype = jnp.result_type(score_weight, half_weights, jnp.float32)
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_score = jnp.asarray(shifted_score, dtype=complex_dtype)
    score_weight = jnp.asarray(score_weight, dtype=real_dtype)
    half_weights = jnp.asarray(half_weights, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if (
        shifted_score.shape[-1] != n_values
        or score_weight.shape[-1] != n_values
        or half_weights.shape != (n_values,)
    ):
        raise ValueError(
            "RELION CUDA fine normalized-CC operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_score.shape[-1]}, "
            f"score_weight={score_weight.shape[-1]}, half_weights={half_weights.shape}"
        )
    numerator_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_score.shape[:-1], score_weight.shape[:-1]
    )
    norm_shape = jnp.broadcast_shapes(reference.shape[:-1], score_weight.shape[:-1])
    if n_values == 0:
        return jnp.full(numerator_shape, -jnp.inf, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )

    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    numerator_lanes = jnp.zeros(numerator_shape + (block_size,), dtype=real_dtype)
    norm_lanes = jnp.zeros(norm_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        numerator, norm = lane_values
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        shifted_pass = jnp.take(shifted_score, safe_rows, axis=-1)
        score_weight_pass = jnp.take(score_weight, safe_rows, axis=-1)
        half_weight_pass = jnp.take(half_weights, safe_rows, axis=-1)
        numerator_terms = (
            ref_pass.real * shifted_pass.real + ref_pass.imag * shifted_pass.imag
        ) * score_weight_pass * half_weight_pass
        norm_terms = (
            ref_pass.real * ref_pass.real + ref_pass.imag * ref_pass.imag
        ) * score_weight_pass * half_weight_pass
        zero = jnp.asarray(0.0, dtype=real_dtype)
        numerator_terms = jnp.where(valid_pixel, numerator_terms, zero)
        norm_terms = jnp.where(valid_pixel, norm_terms, zero)
        return numerator + numerator_terms, norm + norm_terms

    numerator_lanes, norm_lanes = jax.lax.fori_loop(
        0,
        n_passes,
        accumulate_pass,
        (numerator_lanes, norm_lanes),
    )
    numerator = _relion_cuda_fine_reduce_lanes(numerator_lanes)
    norm = _relion_cuda_fine_reduce_lanes(norm_lanes)
    return numerator / jnp.sqrt(
        jnp.maximum(norm, jnp.asarray(1e-30, dtype=real_dtype))
    )


def _relion_cuda_fine_pixel_weights(corr_img_score, half_weights):
    """Form RELION XFLOAT pixel weights in the active ACC precision."""

    real_dtype = jnp.result_type(corr_img_score, half_weights, jnp.float32)
    return jnp.asarray(corr_img_score, dtype=real_dtype) * jnp.asarray(
        half_weights, dtype=real_dtype
    )


def _relion_cuda_corr_img_from_rfloat_ctf(
    inverse_noise,
    ctf_rfloat,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form XFLOAT ``corr_img`` after RELION's RFLOAT CTF square.

    The deployed mixed-precision build stores ``Minvsigma2`` and ``corr_img``
    as float32 (XFLOAT), but evaluates the CTF and ``CTF * CTF`` as float64
    (RFLOAT).  The compound multiplication promotes Minvsigma2 to float64 and
    casts the product back to float32 before the optional float32 scale square.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    inverse_noise_rfloat = jnp.asarray(inverse_noise, dtype=output_dtype).astype(jnp.float64)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    ctf_squared_rfloat = jax.lax.optimization_barrier(ctf_rfloat * ctf_rfloat)
    corr_img = jax.lax.optimization_barrier(
        inverse_noise_rfloat * ctf_squared_rfloat
    ).astype(output_dtype)
    if scale is not None:
        scale = jnp.asarray(scale, dtype=output_dtype)
        scale_squared = jax.lax.optimization_barrier(scale * scale)
        corr_img = corr_img * scale_squared
    return corr_img


def _relion_cuda_corr_img_from_native_noise_variance(
    noise_variance,
    ctf_rfloat,
    image_shape,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form score-unit ``corr_img`` with RELION's native-FFT cast order.

    RECOVAR stores the noise variance in its normalized-FFT units, larger than
    RELION's variance by ``N**4``.  Reciprocating that value into float32 and
    then applying the compensating Fourier scale is algebraically correct but
    changes ``Minvsigma2`` by one ULP on real parity fixtures.  RELION first
    reciprocates its native-unit binary64 variance into XFLOAT, forms the
    CTF-square product, and only then does RECOVAR need to convert the completed
    XFLOAT operand back to normalized-FFT score units. ``output_dtype`` selects
    the score dtype of that final conversion (float32 production, float64 for
    double-precision scoring); the RELION-side casts above it are unchanged.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    image_size = int(image_shape[0])
    if tuple(image_shape) != (image_size, image_size):
        raise ValueError(f"RELION corr_img requires a square image, got {image_shape}")
    native_fourier_scale_rfloat = jnp.asarray(image_size**4, dtype=jnp.float64)
    native_variance = (
        jnp.asarray(noise_variance, dtype=jnp.float64)
        / native_fourier_scale_rfloat
    )
    native_inverse_noise = jnp.reciprocal(native_variance).astype(jnp.float32)
    native_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
        native_inverse_noise,
        ctf_rfloat,
        scale,
    )
    # XLA's float32 division may lower to a reciprocal multiply and differs
    # from correctly rounded division by one ULP.  This conversion is not a
    # RELION operation, so perform it in binary64 and cast once to preserve the
    # native XFLOAT operand under RECOVAR's Fourier normalization.
    return (
        native_corr_img.astype(jnp.float64) / native_fourier_scale_rfloat
    ).astype(output_dtype)


def _relion_cuda_pixel_correction_from_rfloat_ctf(
    scale,
    ctf_rfloat,
    *,
    output_dtype=jnp.float32,
):
    """Form RELION's XFLOAT score-image correction from an RFLOAT CTF."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    scale = jnp.asarray(scale, dtype=output_dtype)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    pixel_correction = jax.lax.optimization_barrier(jnp.reciprocal(scale))
    corrected = jax.lax.optimization_barrier(
        pixel_correction.astype(jnp.float64) / ctf_rfloat
    ).astype(output_dtype)
    return jnp.where(jnp.abs(ctf_rfloat) > 1e-8, corrected, pixel_correction)


_RELION_CUDA_POWERCLASS_BLOCK_SIZE = 128


class _RelionPowerClassOperands(NamedTuple):
    """RELION ``powerClass`` operands of one flattened centred rfft image batch."""

    relion_image: jnp.ndarray
    real_dtype: object
    image_height: int
    image_width: int
    half_width: int
    resolution_limit: object
    shell: np.ndarray
    valid: np.ndarray


def _relion_powerclass_resolution_limit(current_size, runtime_current_size, *, image_width):
    """RELION's first high-resolution shell, ``current_size / 2 + 1`` (static or traced)."""

    if runtime_current_size is None:
        if current_size is None:
            current_size = image_width
        return int(current_size) // 2 + 1
    return jnp.asarray(runtime_current_size, dtype=jnp.int32) // 2 + 1


def _relion_powerclass_packed_image(processed_score_half, *, image_shape, dtype=None):
    """Repack centred rfft images into RELION's unshifted ``Faux`` layout.

    RELION's packed rows are ``0, +1, ..., +Nyquist, -Nyquist+1, ..., -1`` and
    its FFT amplitudes are smaller than RECOVAR's by the real-space pixel
    count. ``dtype`` forces the complex working precision (the native kernels
    take complex64); otherwise complex128 inputs keep double precision.
    Returns ``(relion_image, real_dtype, image_height, image_width, half_width)``.
    """

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    if dtype is None:
        processed_score_half = jnp.asarray(processed_score_half)
        complex_dtype = (
            jnp.complex128
            if processed_score_half.dtype == jnp.dtype(jnp.complex128)
            else jnp.complex64
        )
        processed_score_half = processed_score_half.astype(complex_dtype)
    else:
        complex_dtype = dtype
        processed_score_half = jnp.asarray(processed_score_half, dtype=complex_dtype)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=real_dtype)
    return relion_image, real_dtype, image_height, image_width, half_width


def _relion_powerclass_operands(processed_score_half, *, image_shape, current_size, runtime_current_size):
    """RELION ``powerClass`` operands for the JAX reproductions.

    Besides the packed image this supplies the integer shell of every packed
    pixel (CUDA ``__float2int_rn(sqrtf(...))``, or the double variant under
    ``ACC_DOUBLE_PRECISION``) and the kernel's pixel validity: shells 1 to
    ``half_width - 1`` excluding the redundant negative-row DC column.
    """

    relion_image, real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
        processed_score_half, image_shape=image_shape
    )
    resolution_limit = _relion_powerclass_resolution_limit(
        current_size, runtime_current_size, image_width=image_width
    )
    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    shell_real_dtype = np.float64 if real_dtype == jnp.float64 else np.float32
    shell = np.rint(np.sqrt(radius_squared.astype(shell_real_dtype))).astype(np.int32)
    valid = (
        (shell > 0)
        & (shell < half_width)
        & ~((columns == 0) & (signed_rows < 0))
    ).reshape(-1)
    return _RelionPowerClassOperands(
        relion_image, real_dtype, image_height, image_width, half_width, resolution_limit, shell, valid
    )


def _relion_powerclass_native_spectrum_highres(processed_score_half, *, image_shape, current_size, runtime_current_size):
    """Run the native CUDA ``powerClass`` atomics on complex64 packed images.

    Returns the per-image spectrum with the block-tree ``highres_Xi2`` scalar
    appended, together with ``(image_height, image_width, half_width)``.
    """

    from recovar import cuda_backproject

    relion_image, _real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
        processed_score_half, image_shape=image_shape, dtype=jnp.complex64
    )
    relion_image = relion_image.astype(jnp.complex64)
    if runtime_current_size is None:
        spectrum_and_highres = cuda_backproject.relion_powerclass_spectrum_highres_f32(
            relion_image,
            xdim=half_width,
            ydim=image_height,
            resolution_limit=int(current_size) // 2 + 1,
        )
    else:
        spectrum_and_highres = (
            cuda_backproject.relion_powerclass_spectrum_highres_runtime_f32(
                relion_image,
                jnp.asarray(runtime_current_size, dtype=jnp.int32) // 2 + 1,
                xdim=half_width,
                ydim=image_height,
            )
        )
    return spectrum_and_highres, image_height, image_width, half_width


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_xi2_half(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the class-power high-resolution image tail used by fine diff2.

    RELION's CUDA ``powerClass`` kernel (``cuda_kernels/helper.cuh``) bins the
    unshifted, unnormalised ``Faux`` image in XFLOAT, reduces each contiguous
    128-pixel block with a shared-memory tree, and atomically accumulates bins
    at or above ``current_size / 2 + 1``. XFLOAT is float32 normally and
    float64 under ``ACC_DOUBLE_PRECISION``. ``diff2_fine`` then adds half of
    that scalar to every fine-search hypothesis (``cuda_kernels/diff2.cuh``).

    RECOVAR stores the y axis centred and its FFT amplitudes are larger by the
    real-space pixel count. Convert both conventions before reproducing the
    matching XFLOAT power and block reduction. The final cross-block
    accumulation uses ascending block order; RELION's atomic arrival order is
    not specified, so its last bit may vary between launches while the
    per-block arithmetic is fixed.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    real_dtype = operands.real_dtype
    valid = jnp.asarray(operands.valid) & (
        jnp.asarray(operands.shell.reshape(-1), dtype=jnp.int32) >= operands.resolution_limit
    )

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    power = jnp.where(jnp.asarray(valid)[None, :], power, jnp.asarray(0.0, dtype=real_dtype))

    block_size = _RELION_CUDA_POWERCLASS_BLOCK_SIZE
    n_blocks = (power.shape[-1] + block_size - 1) // block_size
    power = jnp.pad(power, ((0, 0), (0, n_blocks * block_size - power.shape[-1])))
    block_lanes = power.reshape((power.shape[0], n_blocks, block_size))
    for width in (64, 32, 16, 8, 4, 2, 1):
        block_lanes = block_lanes[..., :width] + block_lanes[..., width : 2 * width]
        block_lanes = jax.lax.optimization_barrier(block_lanes)
    block_sums = block_lanes[..., 0]

    def add_block(block_index, total):
        total = total + block_sums[:, block_index]
        return jax.lax.optimization_barrier(total)

    highres_xi2 = jax.lax.fori_loop(
        0,
        n_blocks,
        add_block,
        jnp.zeros((relion_image.shape[0],), dtype=real_dtype),
    )
    return highres_xi2 * jnp.asarray(0.5, dtype=real_dtype)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_xi2_half_atomic(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Run the native CUDA powerClass atomics used by exact fine scoring."""

    spectrum_and_highres, _image_height, _image_width, _half_width = _relion_powerclass_native_spectrum_highres(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    return spectrum_and_highres[:, -1] * jnp.asarray(0.5, dtype=jnp.float32)


def _relion_powerclass_highres_xi2_half_to_norm_units(highres_xi2_half, image_shape):
    """Convert RELION's half-Xi2 FFT units to RECOVAR norm N^4 units."""

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    highres = jnp.asarray(highres_xi2_half)
    highres = highres * jnp.asarray(2.0, dtype=highres.dtype)
    highres = jax.lax.optimization_barrier(highres)
    return highres * jnp.asarray((image_height * image_width) ** 2, dtype=highres.dtype)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Return source-faithful powerClass high-shell power in RECOVAR N^4 units."""

    return _relion_powerclass_highres_xi2_half_to_norm_units(
        _relion_cuda_powerclass_highres_xi2_half(
            processed_score_half,
            image_shape=image_shape,
            current_size=current_size,
            runtime_current_size=runtime_current_size,
        ),
        image_shape,
    )


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the high-shell norm term from RELION's power spectrum.

    RELION's ``powerClass`` kernel produces two independently reduced values:
    a block-tree ``highres_Xi2`` scalar used by fine scoring, and an
    atomically binned shell spectrum.  Norm correction consumes the latter,
    summing its high shells sequentially in host RFLOAT.  These reductions are
    numerically distinct, so the fine-score scalar cannot be reused here.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    image_height, image_width, half_width = operands.image_height, operands.image_width, operands.half_width
    resolution_limit = operands.resolution_limit
    shell = np.where(operands.valid, operands.shell.reshape(-1), half_width).astype(np.int32)

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    spectrum = jax.vmap(
        lambda row: bin_shell_values_jax(row, jnp.asarray(shell), half_width)
    )(power)

    # RELION copies the float32 spectrum to the host and adds the selected
    # shells into an RFLOAT accumulator in increasing shell order.
    def add_shell(shell_index, total):
        return total + spectrum[:, shell_index].astype(jnp.float64)

    high_shell = jax.lax.fori_loop(
        resolution_limit,
        half_width,
        add_shell,
        jnp.zeros((relion_image.shape[0],), dtype=jnp.float64),
    )
    return high_shell * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float64)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Return RELION's atomically binned per-image power spectrum in N^4 units."""

    if deterministic_reductions_enabled():
        # The CUDA powerClass kernel bins |F|^2 with float atomicAdd per pixel,
        # so its shell sums vary between launches (verified across processes).
        # Under the opt-in, bin the identical float32 per-pixel values with the
        # fixed-order shell reduction instead; same operands, fixed order.
        relion_image, _real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
            processed_score_half, image_shape=image_shape, dtype=jnp.complex64
        )
        relion_image = relion_image.astype(jnp.complex64)
        rows = np.arange(image_height, dtype=np.int32)[:, None]
        columns = np.arange(half_width, dtype=np.int32)[None, :]
        signed_rows = np.where(rows < half_width, rows, rows - image_height)
        radius_squared = columns * columns + signed_rows * signed_rows
        shell = np.rint(np.sqrt(radius_squared.astype(np.float32))).astype(np.int32)
        valid = (
            (shell > 0)
            & (shell < half_width)
            & ~((columns == 0) & (signed_rows < 0))
        ).reshape(-1)
        shell = np.where(valid, shell.reshape(-1), half_width).astype(np.int32)
        power = relion_image.real * relion_image.real
        power = jax.lax.optimization_barrier(power)
        power = power + relion_image.imag * relion_image.imag
        spectrum = jax.vmap(
            lambda row: bin_shell_values_jax(row, jnp.asarray(shell), half_width)
        )(power)
        return spectrum * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float32)
    spectrum_and_highres, image_height, image_width, half_width = _relion_powerclass_native_spectrum_highres(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    return spectrum_and_highres[:, :half_width] * jnp.asarray(
        (image_height * image_width) ** 2,
        dtype=jnp.float32,
    )


def _relion_powerclass_noise_terms(
    processed_score_half_for_noise,
    *,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
):
    """RELION ``powerClass`` terms one sparse pass-2 batch needs.

    Exact fine Gaussian scoring adds half of the block-tree ``highres_Xi2``
    to every hypothesis, and norm correction at a current size consumes the
    high-shell power: the atomically binned spectrum in source-faithful mode,
    otherwise the same ``highres_Xi2`` converted to RECOVAR's N^4 units
    (``ml_optimiser.cpp`` ``storeWeightedSums``). Returns
    ``(highres_xi2_half, norm_high_shell)`` with ``None`` for terms the batch
    does not need.
    """

    relion_highres_xi2_half = None
    if use_exact_relion_gaussian or (accumulate_noise and current_size is not None):
        relion_highres_xi2_half = _relion_cuda_powerclass_highres_xi2_half(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
        )
    if accumulate_noise and current_size is not None and relion_highres_xi2_half is not None:
        if source_faithful_spectrum_norm:
            relion_norm_high_shell = _relion_cuda_powerclass_spectrum_highres_norm_units(
                processed_score_half_for_noise,
                image_shape=image_shape,
                current_size=current_size,
            )
        else:
            relion_norm_high_shell = _relion_powerclass_highres_xi2_half_to_norm_units(
                relion_highres_xi2_half,
                image_shape,
            )
    else:
        relion_norm_high_shell = None
    return relion_highres_xi2_half, relion_norm_high_shell


@jax.jit
def _relion_cuda_fine_diff2_min(diff2, candidate_mask):
    """Return one finite XFLOAT minimum per image over a raw diff2 tensor."""

    minimum = _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask)
    return jnp.where(
        jnp.isfinite(minimum),
        minimum,
        jnp.asarray(0.0, dtype=minimum.dtype),
    )


@jax.jit
def _relion_cuda_fine_partition_diff2_min_or_inf(diff2, candidate_mask):
    """Reduce one partition, retaining ``+inf`` for all-invalid images."""

    diff2 = jnp.asarray(diff2)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    if diff2.shape != candidate_mask.shape:
        raise ValueError(
            "RELION raw diff2 and candidate mask shapes must match: "
            f"diff2={diff2.shape}, mask={candidate_mask.shape}"
        )
    if diff2.ndim < 2:
        raise ValueError(f"RELION raw diff2 needs a leading image axis, got {diff2.shape}")
    valid = candidate_mask & jnp.isfinite(diff2)
    reduction_axes = tuple(range(1, diff2.ndim))
    return jnp.min(jnp.where(valid, diff2, jnp.inf), axis=reduction_axes)


def _relion_cuda_fine_global_diff2_min(raw_diff2_by_partition, masks_by_partition):
    """Return the common per-image minimum spanning chunks and/or classes."""

    if len(raw_diff2_by_partition) != len(masks_by_partition):
        raise ValueError("RELION raw diff2 partitions and masks must have equal lengths")
    if not raw_diff2_by_partition:
        raise ValueError("RELION common-min reduction needs at least one partition")
    partition_minima = []
    for raw_diff2, mask in zip(raw_diff2_by_partition, masks_by_partition, strict=True):
        host_staged_partition = isinstance(raw_diff2, np.ndarray)
        raw_diff2_device = jnp.asarray(raw_diff2)
        mask_device = jnp.asarray(mask, dtype=bool)
        partition_minimum = _relion_cuda_fine_partition_diff2_min_or_inf(
            raw_diff2_device,
            mask_device,
        )
        if host_staged_partition:
            # K-class staging deliberately serializes each D2H-staged class
            # partition back through the device. Synchronize the tiny reduced
            # result before releasing the raw upload so successive classes
            # cannot become simultaneously resident through async dispatch.
            partition_minimum = jax.block_until_ready(partition_minimum)
        partition_minima.append(partition_minimum)
        del raw_diff2_device
    return _relion_cuda_fine_finite_common_min(tuple(partition_minima))


@jax.jit
def _relion_cuda_fine_finite_common_min(partition_minima):
    """Reduce per-partition minima to one finite per-image common minimum."""

    common_min = jnp.min(jnp.stack(partition_minima, axis=0), axis=0)
    return jnp.where(
        jnp.isfinite(common_min),
        common_min,
        jnp.asarray(0.0, dtype=common_min.dtype),
    )


def _relion_cuda_fine_log_evidence_offset(min_diff2):
    """Undo RELION's common-min score centering for absolute log evidence."""

    return -jnp.asarray(min_diff2)


@jax.jit
def _relion_cuda_fine_diff2_to_scores(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    *,
    min_diff2=None,
):
    """Apply RELION's XFLOAT fine diff2-to-log-weight conversion order.

    RELION first finds one common minimum over the full valid fine candidate
    set for each image. Its CUDA conversion kernel then evaluates, in XFLOAT,
    ``((orientation_log_prior + translation_log_prior) + min_diff2) - diff2``.
    The common-min term cancels algebraically in normalized probabilities but
    its placement changes float32 tie-breaking at diff2 magnitudes around 1e3.

    ``min_diff2`` may be supplied by a caller that splits one image's full
    candidate set across score chunks or classes. Otherwise it is computed
    over the candidate set represented by ``diff2``. K-class callers must
    therefore supply an external minimum spanning every class; a per-class
    call is not a claim of K-class bit parity.
    """

    diff2 = jnp.asarray(diff2)
    real_dtype = diff2.dtype
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=real_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=real_dtype)
    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    valid = candidate_mask & jnp.isfinite(diff2)
    if min_diff2 is None:
        local_min = _relion_cuda_fine_diff2_min(diff2, candidate_mask)
    else:
        local_min = jnp.asarray(min_diff2, dtype=real_dtype)
    has_valid = jnp.any(valid, axis=tuple(range(1, diff2.ndim)))
    local_min = jnp.where(has_valid, local_min, jnp.asarray(0.0, dtype=real_dtype))
    min_shape = (diff2.shape[0],) + (1,) * (diff2.ndim - 1)
    # RELION's exponentiation kernel rejects candidates below the supplied
    # global minimum. This is normally impossible for a self-consistent
    # partition, but is observable at cross-partition float32 boundaries.
    valid = valid & (diff2 >= local_min.reshape(min_shape))

    scores = rotation_log_prior + translation_log_prior
    scores = jax.lax.optimization_barrier(scores)
    scores = scores + local_min.reshape(min_shape)
    scores = jax.lax.optimization_barrier(scores)
    scores = scores - diff2
    scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)
    return scores


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Return positive float32 RELION fine-pass costs without priors or centering."""

    weights = _relion_cuda_fine_pixel_weights(
        corr_img_score, jnp.asarray(half_weights)[None, :]
    )
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, :, None, :],
        shifted_corrected[:, None, :, :],
        weights[:, None, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)[:, None, None]
    return diff2


@jax.jit
def _score_pass2_bucket_relion_gpu_diff2_from_raw(
    diff2,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    min_diff2,
):
    """Convert retained raw costs with the same jitted exact score arithmetic."""

    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image operand divided by score weight factors
    corr_img_score,  # (B, N) real, Gaussian projection-norm score weight
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    rotation_log_prior,  # (B, R) real
    translation_log_prior,  # (B, T) real
    candidate_mask,  # (B, R, T) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # (B,) optional external common minimum across chunks/classes
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """RELION GPU-style direct ``diff2`` scoring for pass-2 diagnostics.

    RELION's CUDA fine-search kernel first corrects the image by the same
    scalar factors carried by the projection-norm weight, then accumulates a
    direct ``|Fref - Fimg_corrected_shift|^2 * corr_img`` form.  This is
    algebraically equivalent to the dense cross-minus-norm expression but has
    different float32 rounding. The positive diff2 values are converted with
    RELION's common-min and prior-addition order below.

    Extremely small CTF/noise combinations can still overflow the direct form
    on long 256px runs.  Treat non-finite candidates as impossible hypotheses
    rather than letting NaNs enter posterior and noise accumulators.
    """

    score_dtype = (
        jnp.float64
        if jnp.result_type(shifted_corrected, corr_img_score) == jnp.complex128
        or jnp.asarray(corr_img_score).dtype == jnp.float64
        else jnp.float32
    )
    rotation_log_prior = jnp.asarray(rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    diff2 = _score_pass2_bucket_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        rotation_log_prior[:, :, None],
        translation_log_prior[:, None, :],
        candidate_mask,
        min_diff2=min_diff2,
    )


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_bucket_relion_gpu_diff2_single_cached_raw(
    shifted_corrected,  # (T, N) complex
    corr_img_score,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # scalar float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Single-image cached positive-cost variant without priors or centering."""

    weights = _relion_cuda_fine_pixel_weights(corr_img_score, half_weights)
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_half[:, None, :],
        shifted_corrected[None, :, :],
        weights[None, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)
    return diff2


@jax.jit
def _score_pass2_bucket_relion_gpu_normalized_cc(
    shifted_score,  # (B, T, N) complex, RELION-corrected image after shift
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (B, R, T) bool
    relion_full_to_compact=None,  # packed current-size FFTW order -> compact row
):
    """RELION iter-1 normalized-CC scoring for sparse pass-2 buckets."""

    scores = _relion_cuda_fine_normalized_cc_score(
        proj_half[:, :, None, :],
        shifted_score[:, None, :, :],
        score_weight[:, None, None, :],
        half_weights,
        relion_full_to_compact,
    )
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_bucket_normalized_cc(
    shifted_score,  # (B, T, N) complex, image * CTF * shift / Xi2
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (B, R, T) bool
):
    """Historical algebraic normalized-CC scorer retained outside K=1."""

    cross_products = (
        proj_half[:, :, None, :].real * shifted_score[:, None, :, :].real
        + proj_half[:, :, None, :].imag * shifted_score[:, None, :, :].imag
    ) * jnp.asarray(half_weights, dtype=proj_half.real.dtype)[None, None, None, :]
    # Sum in the operands' own precision (natural promotion) rather than
    # forcing float32 -- this term is the CC numerator and must match the
    # denominator's precision (below, via Precision.HIGHEST) or a genuine
    # near-tie candidate ranking can flip relative to RELION's RFLOAT/XFLOAT
    # arithmetic. See docs/math/relion_parity_agent_notes.md.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    ) * half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,brn->br",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom[:, :, None]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_bucket_normalized_cc_single_cached(
    shifted_score,  # (T, N) complex
    score_weight,  # (N,) real
    proj_half,  # (R, N) complex
    half_weights,  # (N,) real
    candidate_mask,  # (R, T) bool
):
    """Historical cached normalized-CC scorer retained outside K=1."""

    cross_products = (
        proj_half[:, None, :].real * shifted_score[None, :, :].real
        + proj_half[:, None, :].imag * shifted_score[None, :, :].imag
    ) * jnp.asarray(half_weights, dtype=proj_half.real.dtype)[None, None, :]
    # See _score_pass2_bucket_normalized_cc: sum in the operands' own
    # precision instead of forcing float32, to match the einsum denominator's
    # precision below and avoid spurious near-tie ranking flips.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_half.real * proj_half.real + proj_half.imag * proj_half.imag
    ) * half_weights[None, :]
    norms = jnp.einsum(
        "n,rn->r",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom[:, None]
    scores = jnp.where(candidate_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@jax.jit
def _score_pass2_pairs_gaussian_algebraic(
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    pair_rotation_log_prior,
    translation_log_prior,
    local_rotation_row,
    translation_idx,
    pair_mask,
):
    """Compact-pair variant of the historical algebraic Gaussian scorer."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    shifted_pair = shifted_corrected[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    weights = corr_img_score * half_weights[None, :]
    cross = jnp.einsum(
        "bpn,bn,bpn->bp",
        jnp.conj(shifted_pair),
        weights,
        proj_pair,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2 = proj_pair.real * proj_pair.real + proj_pair.imag * proj_pair.imag
    proj_norm = 0.5 * jnp.einsum(
        "bn,bpn->bp",
        weights,
        proj_abs2,
        precision=jax.lax.Precision.HIGHEST,
    )
    translation_prior = translation_log_prior[row, safe_translation_idx]
    scores = cross - proj_norm + pair_rotation_log_prior + translation_prior
    scores = jnp.where(pair_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)


@partial(jax.jit, static_argnames=("dtype",))
def _gather_pair_translation_log_prior(bucket_translation_prior, translation_idx, pair_mask, *, dtype):
    """Gather each compact pair's translation log prior in one compiled program.

    Masked pairs read translation 0; callers mask their scores separately.
    """

    row = jnp.arange(translation_idx.shape[0])[:, None]
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    return jnp.asarray(bucket_translation_prior[row, safe_translation_idx], dtype=dtype)


@jax.jit
def _gather_projection_cache_rows(cache_score, cache_recon, cache_recon_abs2, rotation_indices):
    """Gather one bucket's score/recon/abs2 projection-cache rows in one program."""

    return (
        cache_score[rotation_indices],
        cache_recon[rotation_indices],
        cache_recon_abs2[rotation_indices],
    )


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_pairs_relion_gpu_diff2_raw(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """Return positive float32 RELION costs for compact candidate pairs."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)

    shifted_pair = shifted_corrected[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    weights = _relion_cuda_fine_pixel_weights(
        corr_img_score, jnp.asarray(half_weights)[None, :]
    )
    diff2 = _relion_cuda_fine_diff2_sum(
        proj_pair,
        shifted_pair,
        weights[:, None, :],
        relion_full_to_compact,
        use_fused_ffi=use_fused_ffi,
    )
    if highres_xi2_half is not None:
        diff2 = diff2 + jnp.asarray(highres_xi2_half, dtype=diff2.dtype)[:, None]
    return diff2


@partial(jax.jit, static_argnames=("use_fused_ffi",))
def _score_pass2_pairs_relion_gpu_diff2(
    shifted_corrected,  # (B, T, N) complex, image / (CTF * scale)
    corr_img_score,  # (B, N) real, Minvsigma2 * CTF^2 * scale^2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    pair_rotation_log_prior,  # (B, P) real
    translation_log_prior,  # (B, T) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
    relion_full_to_compact=None,  # (current_size * (current_size // 2 + 1),) int
    min_diff2=None,  # (B,) optional external common minimum across classes
    highres_xi2_half=None,  # (B,) float32 powerClass tail already divided by two
    *,
    use_fused_ffi=False,
):
    """RELION GPU-style Gaussian scoring for compact pass-2 pairs."""

    batch = shifted_corrected.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    score_dtype = jnp.float64 if jnp.asarray(corr_img_score).dtype == jnp.float64 else jnp.float32
    pair_rotation_log_prior = jnp.asarray(pair_rotation_log_prior, dtype=score_dtype)
    translation_log_prior = jnp.asarray(translation_log_prior, dtype=score_dtype)
    trans_prior = jnp.asarray(
        translation_log_prior[row, safe_translation_idx], dtype=score_dtype
    )
    diff2 = _score_pass2_pairs_relion_gpu_diff2_raw(
        shifted_corrected,
        corr_img_score,
        proj_half,
        half_weights,
        local_rotation_row,
        translation_idx,
        pair_mask,
        relion_full_to_compact,
        highres_xi2_half,
        use_fused_ffi=use_fused_ffi,
    )
    return _relion_cuda_fine_diff2_to_scores(
        diff2,
        pair_rotation_log_prior,
        trans_prior,
        pair_mask,
        min_diff2=min_diff2,
    )


@jax.jit
def _score_pass2_pairs_normalized_cc(
    shifted_score,  # (B, T, N) complex, image * CTF * shift / Xi2
    score_weight,  # (B, N) real, CTF^2 / Xi2
    proj_half,  # (B, R, N) complex
    half_weights,  # (N,) real
    local_rotation_row,  # (B, P) int
    translation_idx,  # (B, P) int
    pair_mask,  # (B, P) bool
):
    """Historical compact-pair normalized-CC scorer retained outside K=1."""

    batch = shifted_score.shape[0]
    row = jnp.arange(batch)[:, None]
    safe_rotation_row = jnp.where(pair_mask, local_rotation_row, 0).astype(jnp.int32)
    safe_translation_idx = jnp.where(pair_mask, translation_idx, 0).astype(jnp.int32)
    shifted_pair = shifted_score[row, safe_translation_idx, :]
    proj_pair = proj_half[row, safe_rotation_row, :]
    cross_products = (
        proj_pair.real * shifted_pair.real + proj_pair.imag * shifted_pair.imag
    ) * jnp.asarray(half_weights, dtype=proj_pair.real.dtype)[None, None, :]
    # See _score_pass2_bucket_normalized_cc: sum in the operands' own
    # precision instead of forcing float32.
    cross = -2.0 * jnp.sum(cross_products, axis=-1)
    proj_abs2_weighted = (
        proj_pair.real * proj_pair.real + proj_pair.imag * proj_pair.imag
    ) * half_weights[None, None, :]
    norms = jnp.einsum(
        "bn,bpn->bp",
        score_weight,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    scores = (-0.5 * cross) / denom
    scores = jnp.where(pair_mask, scores, -jnp.inf)
    return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
