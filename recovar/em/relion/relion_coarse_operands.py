"""RELION exact coarse operands of the K-class significance pass.

RELION's coarse Gaussian square operands (generic and sincosf), the exact
operand assembly with its half-image preprocessing and CC inverse power,
the pose tie-break keys and rescore winner slots, and the opt-ins that
select the exact path.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.env_flags import parse_env_strict_flag
from recovar.em.scoring.coarse_gaussian_gemm import (
    _COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV,
    _COARSE_GAUSSIAN_GEMM_HYBRID_ENV,
    _K1_RELION_EXACT_COARSE_OPERANDS_ENV,
    _K1_RELION_F32_COARSE_SUPPORT_ENV,
)

_RELION_ACC_DOUBLE_FLOORF_QUIRK_ENV = "RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK"


_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV = (
    "RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS"
)


_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE_ENV = (
    "RECOVAR_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE"
)


_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV = (
    "RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS"
)


_COARSE_OPERAND_PROGRAM_ENV = "RECOVAR_COARSE_OPERAND_PROGRAM"


def _repeat_pad_batch_axis(value, target_size: int):
    """Pad a non-empty image batch by repeating row zero.

    Repeating a real row keeps normalized-CC and exact CUDA preprocessing
    finite. Callers must discard the repeated rows from all science outputs.
    """

    array = np.asarray(value)
    target_size = int(target_size)
    if array.shape[0] >= target_size:
        return array
    if array.shape[0] == 0:
        raise ValueError("cannot repeat-pad an empty image batch")
    return np.concatenate(
        [array, np.repeat(array[:1], target_size - array.shape[0], axis=0)],
        axis=0,
    )


def _relion_acc_double_floorf_quirk_enabled() -> bool:
    """Match RELION's texture-free ACC projector coordinate flooring."""

    return parse_env_strict_flag(_RELION_ACC_DOUBLE_FLOORF_QUIRK_ENV)


def _k1_relion_exact_coarse_operands_enabled(*, default: bool = False) -> bool:
    """Return whether coarse Gaussian scoring uses native RFLOAT CTF operands."""

    return parse_env_strict_flag(_K1_RELION_EXACT_COARSE_OPERANDS_ENV, default=default)


def _k1_relion_exact_coarse_skip_generic_operands_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact coarse operands bypass overwritten generic operands."""

    return parse_env_strict_flag(_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV, default=default)


def _resolve_k1_relion_exact_coarse_skip_generic_operands(
    *,
    requested: bool,
    exact_coarse_operands_enabled: bool,
) -> bool:
    """Resolve the exact-source-only operand path, failing closed."""

    if requested and not exact_coarse_operands_enabled:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV}=1 requires "
            f"effective {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1 Gaussian scoring",
        )
    return bool(requested and exact_coarse_operands_enabled)


def _k1_relion_exact_coarse_assembly_profile_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact-coarse call-count diagnostics are published."""

    return parse_env_strict_flag(_K1_RELION_EXACT_COARSE_ASSEMBLY_PROFILE_ENV, default=default)


def _k1_relion_exact_compact_preprocess_enabled(
    *,
    default: bool = False,
) -> bool:
    """Return whether exact compact scoring skips unused generic preprocessing."""

    return parse_env_strict_flag(_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV, default=default)


def _resolve_k1_relion_exact_compact_preprocess(
    *,
    requested: bool,
    exact_coarse_skip_generic_operands_enabled: bool,
    exact_coarse_operands_enabled: bool,
    coarse_gaussian_gemm_hybrid_requested: bool,
    coarse_gaussian_gemm_compact_posterior_requested: bool,
    score_mode: str,
    any_diagnostic_requested: bool,
) -> bool:
    """Resolve the narrow exact+compact preprocessing specialization."""

    if not requested:
        return False
    prefix = f"{_K1_RELION_EXACT_COMPACT_PREPROCESS_ENV}=1 requires"
    if not exact_coarse_skip_generic_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS_ENV}=1",
        )
    if not exact_coarse_operands_enabled:
        raise ValueError(f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1")
    if not coarse_gaussian_gemm_hybrid_requested:
        raise ValueError(f"{prefix} {_COARSE_GAUSSIAN_GEMM_HYBRID_ENV}=1")
    if not coarse_gaussian_gemm_compact_posterior_requested:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR_ENV}=1",
        )
    if score_mode != "gaussian":
        raise ValueError(f"{prefix} score_mode='gaussian'")
    if any_diagnostic_requested:
        raise ValueError(f"{prefix} paired coarse diagnostics to be disabled")
    return True


def _coarse_operand_program_enabled(*, default: bool = False) -> bool:
    """Return whether the coarse operand assembly runs as one jitted program.

    Off by default: the eager assembly, one compiled program per primitive,
    stays the oracle. With the flag on the same Python function is traced once
    per batch shape, so a coarse image batch costs one program call instead of
    roughly fifty-five single-primitive dispatches.
    """

    return parse_env_strict_flag(_COARSE_OPERAND_PROGRAM_ENV, default=default)


def _k1_relion_f32_coarse_support_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION CUDA float32 coarse support is active."""

    return parse_env_strict_flag(_K1_RELION_F32_COARSE_SUPPORT_ENV, default=default)


def _relion_coarse_sincosf_operands(
    unshifted_score_weighted,
    score_weight_half,
    half_weights,
    score_indices,
    score_active_mask,
):
    """Assemble the coarse sincosf operands of one image batch.

    These are the statements of RELION's coarse sincosf operand assembly, less
    the translate FFI call that consumes ``unshifted_corrected`` and the
    reshape of its result, which stay with the caller in
    :func:`_relion_coarse_gaussian_square_operands_sincosf`. ``pixel_weight``
    does not depend on the FFI output, so it is produced here rather than
    after the call.

    :data:`_relion_coarse_sincosf_operand_program` is ``jax.jit`` of this exact
    function, which is what ``RECOVAR_COARSE_OPERAND_PROGRAM=1`` selects: the
    two paths share their source, so the only difference between them is that
    XLA sees the whole chain at once. Nothing here reduces along an axis and no
    multiply feeds an add, so that cannot re-associate an expression or
    contract a multiply-add; the paired unit tests hold the pair bitwise.
    """

    square_score_weight = score_weight_half[:, score_indices]
    square_score_weight = jnp.where(
        score_active_mask[None, :],
        square_score_weight,
        jnp.zeros((), dtype=square_score_weight.dtype),
    )
    square_unshifted_weighted = unshifted_score_weighted[:, score_indices]
    nonzero_weight = square_score_weight != 0.0
    safe_weight = jnp.where(nonzero_weight, square_score_weight, 1.0)
    unshifted_corrected = square_unshifted_weighted / safe_weight
    unshifted_corrected = jnp.where(
        nonzero_weight,
        unshifted_corrected,
        jnp.zeros((), dtype=unshifted_corrected.dtype),
    )
    use_float64 = unshifted_corrected.dtype == jnp.complex128
    complex_dtype = jnp.complex128 if use_float64 else jnp.complex64
    real_dtype = jnp.float64 if use_float64 else jnp.float32
    pixel_weight = square_score_weight * half_weights[score_indices][None, :]
    return (
        jnp.asarray(unshifted_corrected, dtype=complex_dtype),
        jnp.asarray(pixel_weight, dtype=real_dtype),
    )


_relion_coarse_sincosf_operand_program = jax.jit(_relion_coarse_sincosf_operands)


def _relion_exact_coarse_operands(
    ctf_half_rfloat,
    batch_scale_exact,
    processed_direct,
    score_indices,
    score_active_mask,
    noise_variance_half,
    half_weights,
    *,
    image_shape,
    use_float64_scoring,
    scale_corrections_enabled,
):
    """Assemble the exact-source coarse operands of one image batch.

    The statements of the exact assembly in
    :func:`_assemble_relion_exact_coarse_gaussian_operands`, less the host CTF
    read, the translate FFI call, the reshape of its result and ``powerClass``.
    Neither returned operand depends on the FFI output, so both are produced
    before the call instead of straddling it.

    RELION's cast order inside ``_relion_cuda_pixel_correction_from_rfloat_ctf``
    and the ``corr_img`` helpers is carried by their own ``optimization_barrier``
    calls, which survive tracing, and nothing here reduces along an axis or
    adds a product. :data:`_relion_exact_coarse_operand_program` is ``jax.jit``
    of this function.
    """

    from recovar.em.sparse_pass2.sparse_pass2_scoring import (
        _relion_cuda_corr_img_from_native_noise_variance,
        _relion_cuda_corr_img_from_rfloat_ctf,
        _relion_cuda_pixel_correction_from_rfloat_ctf,
    )

    real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
    pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
        batch_scale_exact[:, None],
        ctf_half_rfloat,
        output_dtype=real_dtype,
    )
    processed_score = jnp.asarray(processed_direct, dtype=complex_dtype)[:, score_indices]
    exact_unshifted_corrected = processed_score * pixel_correction
    exact_unshifted_corrected = jnp.where(
        score_active_mask[None, :],
        exact_unshifted_corrected,
        jnp.zeros((), dtype=exact_unshifted_corrected.dtype),
    ).astype(complex_dtype)
    score_noise_variance = noise_variance_half[score_indices]
    if use_float64_scoring:
        inverse_noise_half = jnp.reciprocal(jnp.asarray(score_noise_variance, dtype=jnp.float64))
        exact_corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
            inverse_noise_half[None, :], ctf_half_rfloat,
            batch_scale_exact[:, None] if scale_corrections_enabled else None,
            output_dtype=real_dtype,
        )
    else:
        exact_corr_img = _relion_cuda_corr_img_from_native_noise_variance(
            score_noise_variance[None, :],
            ctf_half_rfloat,
            image_shape,
            batch_scale_exact[:, None] if scale_corrections_enabled else None,
        )
    exact_square_corr_img = exact_corr_img
    exact_square_corr_img = jnp.where(
        score_active_mask[None, :],
        exact_square_corr_img,
        jnp.zeros((), dtype=exact_square_corr_img.dtype),
    )
    pixel_weight = jnp.asarray(
        exact_square_corr_img
        * jnp.asarray(half_weights[score_indices], dtype=real_dtype)[None, :],
        dtype=real_dtype,
    )
    return exact_unshifted_corrected, pixel_weight


_relion_exact_coarse_operand_program = jax.jit(
    _relion_exact_coarse_operands,
    static_argnames=(
        "image_shape",
        "use_float64_scoring",
        "scale_corrections_enabled",
    ),
)


class RelionCcCoarseOperands(NamedTuple):
    """Normalized-CC tree-rescore operands of one coarse image batch."""

    unshifted_corrected: jax.Array
    corr_img: jax.Array
    windowed_unshifted: jax.Array
    windowed_corr_img: jax.Array


def _relion_cc_coarse_operands(
    processed,
    ctf_rfloat,
    inverse_power,
    batch_scale_f32,
    phase_factors,
    window_indices,
    *,
    scale_corrections_enabled,
) -> RelionCcCoarseOperands:
    """Assemble the normalized-CC (``--firstiter_cc``) tree-rescore operands.

    ``inverse_power`` is computed by the caller and passed in. It is a shell
    reduction over ``|Fimg|**2``, and fusing a reduction into its producer is
    the one transformation this pass has measured to move a last bit, so it
    stays outside the program. :data:`_relion_cc_coarse_operand_program` is
    ``jax.jit`` of this function.
    """

    from recovar.em.sparse_pass2.sparse_pass2_scoring import (
        _relion_cuda_corr_img_from_rfloat_ctf,
        _relion_cuda_pixel_correction_from_rfloat_ctf,
    )

    pixel_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
        batch_scale_f32[:, None],
        ctf_rfloat,
    )
    unshifted_corrected = jnp.asarray(
        processed * pixel_correction,
        dtype=jnp.complex64,
    )
    if phase_factors is not None:
        unshifted_corrected = unshifted_corrected * phase_factors
    corr_img = _relion_cuda_corr_img_from_rfloat_ctf(
        inverse_power,
        ctf_rfloat,
        batch_scale_f32[:, None] if scale_corrections_enabled else None,
    )
    if window_indices is None:
        return RelionCcCoarseOperands(
            unshifted_corrected,
            corr_img,
            unshifted_corrected,
            corr_img,
        )
    return RelionCcCoarseOperands(
        unshifted_corrected,
        corr_img,
        unshifted_corrected[:, window_indices],
        corr_img[:, window_indices],
    )


_relion_cc_coarse_operand_program = jax.jit(
    _relion_cc_coarse_operands,
    static_argnames=("scale_corrections_enabled",),
)


def assemble_relion_cc_coarse_operands(
    processed,
    ctf_rfloat,
    inverse_power,
    batch_scale_f32,
    *,
    phase_factors=None,
    window_indices=None,
    scale_corrections_enabled: bool,
) -> RelionCcCoarseOperands:
    """Build one coarse batch's ``--firstiter_cc`` tree-rescore operands.

    ``RECOVAR_COARSE_OPERAND_PROGRAM=1`` traces the assembly once per batch
    shape; the default eager path is the oracle. ``ctf_rfloat`` is already
    repeat-padded by the caller when the coarse image batch is padded, so every
    operand carries the padded extent and the caller slices the repeated rows
    off with the rest.
    """

    assemble = (
        _relion_cc_coarse_operand_program
        if _coarse_operand_program_enabled()
        else _relion_cc_coarse_operands
    )
    return RelionCcCoarseOperands(
        *assemble(
            processed,
            ctf_rfloat,
            inverse_power,
            batch_scale_f32,
            phase_factors,
            window_indices,
            scale_corrections_enabled=bool(scale_corrections_enabled),
        )
    )


def _relion_coarse_gaussian_square_operands(
    shifted_half,
    score_weight_half,
    half_weights,
    score_indices,
    score_active_mask,
    *,
    batch_size: int,
    n_trans: int,
):
    """Derive RELION square-difference operands from accepted score inputs."""

    square_score_weight = score_weight_half[:, score_indices]
    square_score_weight = jnp.where(
        score_active_mask[None, :],
        square_score_weight,
        jnp.zeros((), dtype=square_score_weight.dtype),
    )
    square_shifted_weighted = shifted_half.reshape(
        batch_size,
        n_trans,
        -1,
    )[:, :, score_indices]
    nonzero_weight = square_score_weight != 0.0
    safe_weight = jnp.where(nonzero_weight, square_score_weight, 1.0)
    shifted_corrected = square_shifted_weighted / safe_weight[:, None, :]
    shifted_corrected = jnp.where(
        nonzero_weight[:, None, :],
        shifted_corrected,
        jnp.zeros((), dtype=shifted_corrected.dtype),
    )
    pixel_weight = square_score_weight * half_weights[score_indices][None, :]
    output_complex_dtype = (
        jnp.complex128 if shifted_corrected.dtype == jnp.complex128 else jnp.complex64
    )
    output_real_dtype = jnp.float64 if output_complex_dtype == jnp.complex128 else jnp.float32
    return (
        jnp.asarray(shifted_corrected, dtype=output_complex_dtype),
        jnp.asarray(pixel_weight, dtype=output_real_dtype),
    )


def _relion_coarse_gaussian_square_operands_sincosf(
    unshifted_score_weighted,
    score_weight_half,
    half_weights,
    score_indices,
    score_active_mask,
    translations,
    image_shape,
    *,
    translation_phase_source=None,
    relion_translation_angle_scale=1.0,
    return_unshifted=False,
):
    """Build corrected coarse images with RELION's CUDA sin/cos path."""

    from recovar.em.cuda import kernels as em_cuda_kernels
    from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f64
    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    if translation_phase_source is None:
        translation_phase_source = translations
    assemble = (
        _relion_coarse_sincosf_operand_program
        if _coarse_operand_program_enabled()
        else _relion_coarse_sincosf_operands
    )
    unshifted_corrected, pixel_weight = assemble(
        unshifted_score_weighted,
        score_weight_half,
        half_weights,
        score_indices,
        score_active_mask,
    )
    # ``unshifted_corrected`` arrives already cast, and that cast is to
    # complex128 exactly when the uncast quotient is complex128, so this test
    # selects the same dtypes the assembly used.
    use_float64 = unshifted_corrected.dtype == jnp.complex128
    complex_dtype = jnp.complex128 if use_float64 else jnp.complex64
    real_dtype = jnp.float64 if use_float64 else jnp.float32
    angle_dtype = np.float64 if use_float64 else np.float32
    translation_angles = np.asarray(
        _relion_translation_angles_f64(
            translation_phase_source,
            image_shape,
            angle_scale=relion_translation_angle_scale,
        ),
        dtype=angle_dtype,
    )
    translate_score = (
        em_cuda_kernels.relion_translate_score_f64
        if use_float64
        else em_cuda_kernels.relion_translate_score_f32
    )
    shifted_corrected = translate_score(
        jnp.asarray(unshifted_corrected, dtype=complex_dtype),
        jnp.asarray(translation_angles, dtype=real_dtype),
        score_indices,
        image_shape,
    )
    result = (
        shifted_corrected.reshape(
            unshifted_corrected.shape[0],
            len(translations),
            unshifted_corrected.shape[1],
        ),
        pixel_weight,
    )
    if return_unshifted:
        return (*result, unshifted_corrected)
    return result


class RelionExactCoarseGaussianOperands(NamedTuple):
    """Exact-source operands consumed by both EM and InitialModel scoring."""

    shifted_corrected: jax.Array
    pixel_weight: jax.Array
    unshifted_corrected: jax.Array
    initial_diff2: jax.Array
    translation_angles: jax.Array


def _process_relion_exact_coarse_half_image(
    experiment_dataset,
    batch_data,
    score_with_masked_images: bool,
    *,
    relion_preprocess_kwargs,
):
    """Run the one canonical per-image RELION FFT used by exact coarse scoring."""

    if relion_preprocess_kwargs is None:
        raise ValueError(
            f"{_K1_RELION_EXACT_COARSE_OPERANDS_ENV} requires RELION CUDA "
            "image preprocessing",
        )
    from recovar.em.helpers.preprocessing import process_half_image

    exact_preprocess_kwargs = dict(relion_preprocess_kwargs)
    exact_preprocess_kwargs["relion_fft_per_image"] = True
    return process_half_image(
        experiment_dataset,
        batch_data,
        score_with_masked_images,
        relion_preprocess_kwargs=exact_preprocess_kwargs,
    )


def _assemble_relion_exact_coarse_gaussian_operands(
    experiment_dataset,
    processed_direct,
    indices,
    *,
    batch_scale_np,
    actual_batch_size: int,
    batch_size: int,
    score_indices,
    score_indices_np,
    score_active_mask,
    translations_source,
    image_shape,
    noise_variance_half,
    scale_corrections_enabled: bool,
    half_weights,
    powerclass,
    current_size,
    runtime_current_size=None,
    use_float64_scoring: bool = False,
    relion_translation_angle_scale: float = 1.0,
) -> RelionExactCoarseGaussianOperands:
    """Assemble the single exact-source operand set without generic formulas."""

    from recovar.em.cuda import kernels as em_cuda_kernels
    from recovar.em.relion.relion_ctf import _relion_exact_ctf_half_from_source_star_host
    from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
        _relion_translation_angles_f32,
        _relion_translation_angles_f64,
    )

    real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
    angle_fn = _relion_translation_angles_f64 if use_float64_scoring else _relion_translation_angles_f32
    translate_fn = em_cuda_kernels.relion_translate_score_f64 if use_float64_scoring else em_cuda_kernels.relion_translate_score_f32

    ctf_half_rfloat_np = np.asarray(
        _relion_exact_ctf_half_from_source_star_host(
            experiment_dataset,
            indices,
            image_shape,
            pixel_indices=score_indices_np,
        ),
        dtype=np.float64,
    )
    if batch_size > actual_batch_size:
        ctf_half_rfloat_np = _repeat_pad_batch_axis(
            ctf_half_rfloat_np,
            batch_size,
        )
    ctf_half_rfloat = jnp.asarray(ctf_half_rfloat_np, dtype=jnp.float64)
    batch_scale_exact = jnp.asarray(batch_scale_np, dtype=real_dtype)
    assemble = (
        _relion_exact_coarse_operand_program
        if _coarse_operand_program_enabled()
        else _relion_exact_coarse_operands
    )
    exact_unshifted_corrected, pixel_weight = assemble(
        ctf_half_rfloat,
        batch_scale_exact,
        processed_direct,
        score_indices,
        score_active_mask,
        noise_variance_half,
        half_weights,
        image_shape=tuple(int(size) for size in image_shape),
        use_float64_scoring=bool(use_float64_scoring),
        scale_corrections_enabled=bool(scale_corrections_enabled),
    )
    translation_angles = jnp.asarray(
        angle_fn(translations_source, image_shape, angle_scale=relion_translation_angle_scale),
        dtype=real_dtype,
    )
    shifted_corrected = translate_fn(
        exact_unshifted_corrected,
        translation_angles,
        score_indices,
        image_shape,
    ).reshape(batch_size, int(translation_angles.shape[0]), -1)
    return RelionExactCoarseGaussianOperands(
        shifted_corrected=jnp.asarray(shifted_corrected, dtype=complex_dtype),
        pixel_weight=pixel_weight,
        unshifted_corrected=exact_unshifted_corrected,
        initial_diff2=powerclass(
            processed_direct,
            image_shape=image_shape,
            current_size=current_size,
            runtime_current_size=runtime_current_size,
        ),
        translation_angles=translation_angles,
    )


def _relion_cc_inverse_power_from_processed(processed_half, score_indices=None):
    """Return RELION firstiter-CC ``1/sum(norm(Fimg))`` in binary64.

    The strict tree rescore uses a per-image FFT to reproduce RELION's
    ``windowFourierTransform``. Its normalization must come from that same
    Fourier array; reusing the batched-FFT norm leaves a one-ULP ``corr_img``
    mismatch at marginal translation ties.
    """

    processed_half = jnp.asarray(processed_half, dtype=jnp.complex128)
    if score_indices is not None:
        processed_half = processed_half[:, jnp.asarray(score_indices, dtype=jnp.int32)]
    power_terms = (
        processed_half.real * processed_half.real
        + processed_half.imag * processed_half.imag
    )
    image_power = jnp.sum(power_terms, axis=-1, keepdims=True)
    return jnp.reciprocal(
        jnp.maximum(image_power, jnp.asarray(1e-30, dtype=jnp.float64))
    )


def _infer_relion_coarse_healpix_order(
    n_rotations: int,
    symmetry_label: str = "C1",
) -> int | None:
    """Infer a complete RELION coarse-grid order, or return ``None``."""

    from recovar.em.sampling import rotation_grid_size

    for order in range(9):
        try:
            grid_size = rotation_grid_size(order, symmetry_label)
        except ValueError:
            # Some high-order point groups have an empty ASU on the coarsest
            # HEALPix grids.  They cannot match this observed rotation count.
            continue
        if int(grid_size) == int(n_rotations):
            return order
    return None




def _relion_coarse_pose_tie_break_keys(
    candidate_pose_ids,
    *,
    n_trans: int,
    healpix_order: int,
    coarse_rotation_ids=None,
    symmetry_label: str = "C1",
):
    """Map RECOVAR pose ids to RELION's direction-major coarse order."""

    from recovar.em.sampling import rotation_grid_n_in_planes, rotation_grid_size

    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int64)
    if candidate_pose_ids.ndim != 2:
        raise ValueError(
            f"candidate_pose_ids must have shape (n_rows, n_candidates), got {candidate_pose_ids.shape}",
        )
    n_trans = int(n_trans)
    if n_trans <= 0:
        raise ValueError(f"n_trans must be positive, got {n_trans}")
    local_rotation_ids = candidate_pose_ids // n_trans
    if coarse_rotation_ids is None:
        canonical_rotation_ids = local_rotation_ids
    else:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if np.any(local_rotation_ids < 0) or np.any(local_rotation_ids >= coarse_rotation_ids.size):
            raise ValueError("candidate pose references a rotation outside coarse_rotation_ids")
        canonical_rotation_ids = coarse_rotation_ids[local_rotation_ids]

    healpix_order = int(healpix_order)
    n_psi = int(rotation_grid_n_in_planes(healpix_order))
    n_rotations = int(rotation_grid_size(healpix_order, symmetry_label))
    n_directions = n_rotations // n_psi
    if np.any(canonical_rotation_ids < 0) or np.any(canonical_rotation_ids >= n_rotations):
        raise ValueError(
            "canonical coarse rotation ids must index the complete "
            f"RELION order-{healpix_order} grid of size {n_rotations}",
        )
    psi_ids = canonical_rotation_ids // n_directions
    direction_ids = canonical_rotation_ids % n_directions
    relion_rotation_ids = direction_ids * n_psi + psi_ids
    return relion_rotation_ids * n_trans + candidate_pose_ids % n_trans




def _select_relion_coarse_rescore_winner_slots(
    scores,
    candidate_pose_ids,
    *,
    n_trans: int,
    healpix_order: int | None,
    coarse_rotation_ids=None,
    score_dtype=np.float32,
    symmetry_label: str = "C1",
):
    """Select maxima, resolving exact score ties in RELION's flat order."""

    scores = np.asarray(scores, dtype=score_dtype)
    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int64)
    if scores.shape != candidate_pose_ids.shape or scores.ndim != 2:
        raise ValueError(
            "scores and candidate_pose_ids must have the same "
            f"(n_rows, n_candidates) shape, got {scores.shape} and {candidate_pose_ids.shape}",
        )
    maxima = np.max(scores, axis=1, keepdims=True)
    tied = scores == maxima
    exact_ties = np.count_nonzero(np.sum(tied, axis=1) > 1)
    if healpix_order is None:
        # Compatibility for synthetic/non-HEALPix callers. Production
        # RELION-parity dispatch always supplies or infers the coarse order.
        tie_break_keys = candidate_pose_ids
    else:
        tie_break_keys = _relion_coarse_pose_tie_break_keys(
            candidate_pose_ids,
            n_trans=n_trans,
            healpix_order=healpix_order,
            coarse_rotation_ids=coarse_rotation_ids,
            symmetry_label=symmetry_label,
        )
    masked_keys = np.where(tied, tie_break_keys, np.iinfo(np.int64).max)
    return np.argmin(masked_keys, axis=1).astype(np.int32), int(exact_ties)
