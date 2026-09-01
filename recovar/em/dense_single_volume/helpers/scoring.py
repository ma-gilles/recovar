"""JAX scoring and M-step kernels shared by dense single-volume EM helpers."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .dtype_policy import DensePrecisionPolicy


@jax.jit
def _relion_coarse_128lane_float32_reduce(values):
    """Reduce packed pixel contributions like ``diff2_CC_coarse``.

    RELION assigns pixels ``lane + 128 * pass`` to each CUDA lane, accumulates
    the passes sequentially in float32, then applies a 64, 32, ..., 1 shared
    memory tree.  ``values`` may have arbitrary leading dimensions; its last
    axis is the packed FFTW pixel identity.
    """

    values = jnp.asarray(values, dtype=jnp.float32)
    n_pixels = int(values.shape[-1])
    n_passes = (n_pixels + 127) // 128
    padded = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, n_passes * 128 - n_pixels)])
    passes = padded.reshape(values.shape[:-1] + (n_passes, 128))
    lanes = jnp.zeros(values.shape[:-1] + (128,), dtype=jnp.float32)
    for pass_index in range(n_passes):
        lanes = lanes + passes[..., pass_index, :]
    for stride in (64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :stride] + lanes[..., stride : 2 * stride]
    return lanes[..., 0]


@jax.jit
def _relion_coarse_cc_atomic_score_from_components(numerator, norm):
    """Apply RELION's 128 identical atomic additions to coarse CC scores.

    ``cuda_kernel_diff2_CC_coarse`` deliberately has every one of its 128
    threads atomically add the same reduced score divided by 128.  The
    repeated float32 rounding is observable at near ties and is not equivalent
    to multiplying one rounded contribution by 128.
    """

    numerator = jnp.asarray(numerator, dtype=jnp.float32)
    norm = jnp.asarray(norm, dtype=jnp.float32)
    contribution = numerator / (
        jnp.asarray(128.0, dtype=jnp.float32)
        * jnp.sqrt(jnp.maximum(norm, jnp.asarray(1e-30, dtype=jnp.float32)))
    )

    def add_once(_, accumulated):
        return accumulated + contribution

    return jax.lax.fori_loop(
        0,
        128,
        add_once,
        jnp.zeros_like(contribution),
    )


@jax.jit
def _relion_coarse_normalized_cc_rescore_jax(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
):
    """Portable JAX replay of RELION's float32 coarse lane tree."""

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex64)[..., fftw_order]
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float32)[..., fftw_order]
    projection = jnp.asarray(projection_candidates, dtype=jnp.complex64)[..., fftw_order]
    weights = jnp.asarray(half_weights, dtype=jnp.float32)[fftw_order]
    numerator_pixels = jnp.real(jnp.conj(shifted) * projection) * weights
    norm_pixels = score_weight * (jnp.abs(projection) ** 2) * weights
    numerator = _relion_coarse_128lane_float32_reduce(numerator_pixels)
    norm = _relion_coarse_128lane_float32_reduce(norm_pixels)
    return _relion_coarse_cc_atomic_score_from_components(numerator, norm)


def _relion_coarse_normalized_cc_rescore(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
    *,
    projector_full=None,
    rotation_matrices=None,
    current_size=None,
    padding_factor=None,
    projector_max_r=None,
    translation_angles=None,
    numerator_weight_candidates=None,
):
    """Rescore bounded candidates with RELION's native coarse CUDA tree.

    Inputs have shape ``(..., n_pixels)`` in RECOVAR compact centered-row
    order.  ``fftw_order`` maps that last axis to RELION's packed current-size
    FFTW order. On GPU the custom kernel preserves CUDA operand contraction,
    the 128-lane tree, ``sqrtf``, and RELION's repeated atomic additions. The
    JAX implementation remains a portable CPU fallback for structural tests.
    """

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex64)
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float32)
    if shifted.shape != score_weight.shape or shifted.ndim < 2:
        raise ValueError(
            "shifted and score-weight candidates must have one common "
            f"rank-2-or-higher shape, got {shifted.shape} and "
            f"{score_weight.shape}",
        )
    native_texture_requested = projector_full is not None
    native_texture_args = (
        rotation_matrices,
        current_size,
        padding_factor,
        projector_max_r,
    )
    if native_texture_requested and any(value is None for value in native_texture_args):
        raise ValueError(
            "native texture normalized-CC replay requires rotations, current "
            "size, padding factor, and projector maximum radius",
        )
    projection = None
    if projection_candidates is not None:
        projection = jnp.asarray(projection_candidates, dtype=jnp.complex64)
        if projection.shape != shifted.shape:
            raise ValueError(
                "projection candidates must match shifted candidates, got "
                f"{projection.shape} and {shifted.shape}",
            )
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        if cuda_backproject.custom_cuda_requested():
            n_pixels = int(shifted.shape[-1])
            leading_shape = shifted.shape[:-1]
            if native_texture_requested:
                rotations = jnp.asarray(rotation_matrices, dtype=jnp.float32)
                if rotations.shape != leading_shape + (3, 3):
                    raise ValueError(
                        "candidate rotations must match candidate leading shape, "
                        f"got {rotations.shape} for {leading_shape}",
                    )
                candidate_translation_angles = None
                if translation_angles is not None:
                    candidate_translation_angles = jnp.asarray(
                        translation_angles,
                        dtype=jnp.float32,
                    )
                    if candidate_translation_angles.shape != leading_shape + (2,):
                        raise ValueError(
                            "translation angles must match candidate leading shape, "
                            f"got {candidate_translation_angles.shape} for {leading_shape}",
                        )
                    candidate_translation_angles = candidate_translation_angles.reshape(
                        -1, 2
                    )
                native_numerator_weight = None
                if numerator_weight_candidates is not None:
                    native_numerator_weight = jnp.asarray(
                        numerator_weight_candidates,
                        dtype=jnp.float32,
                    )
                    if native_numerator_weight.shape != shifted.shape:
                        raise ValueError(
                            "numerator weights must match image candidates, got "
                            f"{native_numerator_weight.shape} and {shifted.shape}",
                        )
                    native_numerator_weight = native_numerator_weight.reshape(
                        -1, n_pixels
                    )
                native_scores = (
                    cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32(
                        jnp.asarray(projector_full, dtype=jnp.complex64),
                        rotations.reshape(-1, 3, 3),
                        shifted.reshape(-1, n_pixels),
                        score_weight.reshape(-1, n_pixels),
                        jnp.asarray(half_weights, dtype=jnp.float32),
                        jnp.asarray(fftw_order, dtype=jnp.int32),
                        int(current_size),
                        int(padding_factor),
                        int(projector_max_r),
                        translation_angles=candidate_translation_angles,
                        numerator_weight=native_numerator_weight,
                    )
                )
                return native_scores.reshape(leading_shape)
            if projection is None:
                raise ValueError(
                    "preprojected normalized-CC replay requires projection candidates"
                )
            native_shape = (1, int(shifted.size // n_pixels), n_pixels)
            native_scores = cuda_backproject.relion_coarse_normalized_cc_pairs_f32(
                shifted.reshape(native_shape),
                score_weight.reshape(native_shape),
                projection.reshape(native_shape),
                jnp.asarray(half_weights, dtype=jnp.float32),
                jnp.asarray(fftw_order, dtype=jnp.int32),
            )
            return native_scores.reshape(leading_shape)
    if native_texture_requested:
        raise RuntimeError(
            "native texture normalized-CC replay requires custom CUDA on a JAX GPU"
        )
    if projection is None:
        raise ValueError("portable normalized-CC replay requires projection candidates")
    return _relion_coarse_normalized_cc_rescore_jax(
        shifted,
        score_weight,
        projection,
        jnp.asarray(half_weights, dtype=jnp.float32),
        jnp.asarray(fftw_order, dtype=jnp.int32),
    )


def _score_rotation_block(
    window_spec,
    *,
    shifted_score,
    batch_norm,
    score_weight,
    proj_half,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
    score_mode: str,
    precision_policy: DensePrecisionPolicy,
):
    """Score one rotation block against the active Fourier-window spec."""

    proj_score = window_spec.score_values(proj_half)
    proj_abs2_score = window_spec.score_values(proj_abs2_half)
    proj_score, proj_abs2_score = precision_policy.cast_projection_scores(
        proj_score,
        proj_abs2_score,
    )
    weights = window_spec.score_values(half_weights)
    proj_weighted = proj_score * weights
    proj_abs2_weighted = proj_abs2_score * weights
    n_score = window_spec.n_score
    if score_mode == "normalized_cc":
        return _e_step_block_scores_windowed_normalized_cc(
            shifted_score,
            batch_norm,
            score_weight,
            proj_weighted,
            proj_abs2_weighted,
            n_images,
            n_trans,
            n_score,
            image_shape,
            volume_shape,
        )
    return _e_step_block_scores_windowed(
        shifted_score,
        batch_norm,
        score_weight,
        proj_weighted,
        proj_abs2_weighted,
        weights,
        n_images,
        n_trans,
        n_score,
        image_shape,
        volume_shape,
    )


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _e_step_block_scores(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using half-spectrum GEMMs.

    The cross-term GEMM uses weighted projections (half_weights absorbed into
    projections, precomputed once per rotation block) to recover the full inner
    product from half-spectrum data:

        cross[i,r] = -2 Re(conj(shifted_half) @ proj_half_weighted.T)

    The norm-term similarly uses half-weighted |proj|^2.
    """
    rot_block_size = proj_half_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_half),
            proj_half_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_half,
        proj_abs2_half.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    half_weights_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using windowed half-spectrum GEMMs."""
    rot_block_size = proj_windowed_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_windowed),
            proj_windowed_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_windowed,
        proj_abs2_windowed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(
    jax.jit,
    static_argnames=("n_images", "n_trans", "image_shape", "volume_shape"),
)
def _relion_coarse_gaussian_gemm_scores_jit(
    projected_reference,
    projected_reference_abs2,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    n_images: int,
    n_trans: int,
    image_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
):
    """Score exact RELION coarse operands with the mature half-spectrum GEMMs."""

    active = jnp.arange(n_images, dtype=jnp.int32) < jnp.asarray(
        actual_image_count,
        dtype=jnp.int32,
    )
    shifted_corrected = jnp.where(
        active[:, None, None],
        shifted_corrected,
        jnp.zeros((), dtype=shifted_corrected.dtype),
    )
    pixel_weight = jnp.where(
        active[:, None],
        pixel_weight,
        jnp.zeros((), dtype=pixel_weight.dtype),
    )
    initial_diff2 = jnp.where(
        active,
        initial_diff2,
        jnp.zeros((), dtype=initial_diff2.dtype),
    )

    # RELION's exact coarse operands store the image divided by its pixel
    # correction separately from corr_img * half_weight.  Absorb that
    # image-specific weight on the image side, then reuse the same two GEMMs
    # as ordinary dense EM.  Candidate axes remain [image, rotation,
    # translation]; only the pixel reduction topology changes.
    weighted_shifted = shifted_corrected * pixel_weight[:, None, :]
    model_scores = _e_step_block_scores_windowed(
        weighted_shifted.reshape(n_images * n_trans, -1),
        jnp.zeros((n_images, 1), dtype=pixel_weight.dtype),
        pixel_weight,
        projected_reference,
        projected_reference_abs2,
        jnp.ones((projected_reference.shape[-1],), dtype=pixel_weight.dtype),
        n_images,
        n_trans,
        int(projected_reference.shape[-1]),
        image_shape,
        volume_shape,
    )

    # The mature dense score omits the pose-independent image term.  Restore
    # it here because the exact RELION coarse FFI reports absolute diff2 and
    # the public log-evidence path deliberately applies no later offset.
    image_power = (
        shifted_corrected.real * shifted_corrected.real
        + shifted_corrected.imag * shifted_corrected.imag
    )
    image_diff2 = jnp.asarray(0.5, dtype=pixel_weight.dtype) * jnp.sum(
        image_power * pixel_weight[:, None, :],
        axis=-1,
    )
    scores = model_scores - image_diff2[:, None, :] - initial_diff2[:, None, None]
    return jnp.where(
        active[:, None, None],
        scores,
        jnp.zeros((), dtype=scores.dtype),
    )


def _relion_coarse_gaussian_gemm_scores(
    projected_reference,
    projected_reference_abs2,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    image_shape,
    volume_shape,
):
    """Validate and score one projection-once coarse Gaussian macro batch.

    The arithmetic is mathematically equivalent in exact arithmetic to
    RELION's direct-square
    ``0.5 * weight * |reference - shifted_image|**2 + initial_diff2``.
    It intentionally reuses :func:`_e_step_block_scores_windowed` so both EM
    and InitialModel exercise the mature half-spectrum GEMMs instead of a
    second VDAM scoring implementation.  This expands the square into model,
    cross, and image terms and therefore changes both operation order and
    cancellation behavior; it is not merely a parallel-reduction reorder.
    Keep the path qualification-only until paired production operands show
    repeat-bounded, non-directional, non-growing drift, unchanged discrete
    choices/support and final quality, plus a material end-to-end speedup.
    """

    projected_reference = jnp.asarray(projected_reference)
    projected_reference_abs2 = jnp.asarray(projected_reference_abs2)
    shifted_corrected = jnp.asarray(shifted_corrected)
    pixel_weight = jnp.asarray(pixel_weight)
    initial_diff2 = jnp.asarray(initial_diff2)
    if projected_reference.ndim != 2 or shifted_corrected.ndim != 3:
        raise ValueError(
            "coarse GEMM macro expects projected_reference=(R,F) and "
            f"shifted_corrected=(B,T,F), got {projected_reference.shape} and "
            f"{shifted_corrected.shape}",
        )
    n_images, n_trans, n_pixels = map(int, shifted_corrected.shape)
    expected_projection_shape = (int(projected_reference.shape[0]), n_pixels)
    if tuple(projected_reference.shape) != expected_projection_shape:
        raise ValueError(
            "coarse GEMM projection and image pixels must match, got "
            f"{projected_reference.shape} and {shifted_corrected.shape}",
        )
    if tuple(projected_reference_abs2.shape) != expected_projection_shape:
        raise ValueError(
            "coarse GEMM projection abs2 must match the projection, got "
            f"{projected_reference_abs2.shape} and {projected_reference.shape}",
        )
    if tuple(pixel_weight.shape) != (n_images, n_pixels):
        raise ValueError(
            "coarse GEMM pixel_weight must have shape "
            f"({n_images}, {n_pixels}), got {pixel_weight.shape}",
        )
    if tuple(initial_diff2.shape) != (n_images,):
        raise ValueError(
            "coarse GEMM initial_diff2 must have one value per image, got "
            f"{initial_diff2.shape}",
        )
    if projected_reference.dtype != shifted_corrected.dtype:
        raise TypeError(
            "coarse GEMM projection and shifted images must share a complex "
            f"dtype, got {projected_reference.dtype} and {shifted_corrected.dtype}",
        )
    expected_real_dtype = jnp.asarray(projected_reference.real).dtype
    if (
        projected_reference_abs2.dtype != expected_real_dtype
        or pixel_weight.dtype != expected_real_dtype
        or initial_diff2.dtype != expected_real_dtype
    ):
        raise TypeError(
            "coarse GEMM abs2, pixel weights, and initial diff2 must use the "
            f"projection's real dtype {expected_real_dtype}",
        )
    if not isinstance(actual_image_count, jax.core.Tracer):
        actual_count = int(np.asarray(actual_image_count))
        if actual_count < 0 or actual_count > n_images:
            raise ValueError(
                "coarse GEMM actual_image_count must be in "
                f"[0, {n_images}], got {actual_count}",
            )
    return _relion_coarse_gaussian_gemm_scores_jit(
        projected_reference,
        projected_reference_abs2,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        n_images=n_images,
        n_trans=n_trans,
        image_shape=tuple(int(value) for value in image_shape),
        volume_shape=tuple(int(value) for value in volume_shape),
    )


def _coarse_gaussian_direct_macro_diagnostics(
    direct_scores,
    macro_scores,
    *,
    direct_support=None,
    macro_support=None,
):
    """Summarize paired direct-square and expanded-GEMM score surfaces.

    Inputs use the public coarse layout ``[image, class, rotation,
    translation]``.  This helper deliberately reports raw deltas, signed bias,
    winner margins, and exact discrete differences without defining a
    promotion tolerance.  Repeated H100 artifacts can therefore establish a
    native/repeat envelope without silently turning one observed delta into an
    acceptance threshold.
    """

    direct = np.asarray(direct_scores)
    macro = np.asarray(macro_scores)
    if direct.shape != macro.shape or direct.ndim != 4:
        raise ValueError(
            "paired coarse score diagnostics require equal "
            "[image, class, rotation, translation] arrays, got "
            f"{direct.shape} and {macro.shape}",
        )
    if direct.dtype != macro.dtype or direct.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "paired coarse score diagnostics require one shared float32 or "
            f"float64 dtype, got {direct.dtype} and {macro.dtype}",
        )
    direct_flat = direct.reshape(direct.shape[0], -1)
    macro_flat = macro.reshape(macro.shape[0], -1)
    delta = macro_flat.astype(np.float64) - direct_flat.astype(np.float64)
    absolute_delta = np.abs(delta)
    relative_delta_to_direct = np.full(delta.shape, np.nan, dtype=np.float64)
    np.divide(
        delta,
        np.abs(direct_flat.astype(np.float64)),
        out=relative_delta_to_direct,
        where=direct_flat != 0,
    )

    def ordered_float_bits(values):
        values = np.asarray(values).copy()
        values[values == 0] = 0
        if values.dtype == np.float32:
            unsigned = values.view(np.uint32)
            sign_mask = np.uint32(1 << 31)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            ).astype(np.uint64)
        else:
            unsigned = values.view(np.uint64)
            sign_mask = np.uint64(1 << 63)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            )
        return ordered

    direct_ordered = ordered_float_bits(direct_flat)
    macro_ordered = ordered_float_bits(macro_flat)
    ulp_delta = np.maximum(direct_ordered, macro_ordered) - np.minimum(
        direct_ordered,
        macro_ordered,
    )
    nonfinite = ~np.isfinite(direct_flat) | ~np.isfinite(macro_flat)
    ulp_delta = np.where(nonfinite, np.iinfo(np.uint64).max, ulp_delta)

    def winner_and_margin(values):
        winner = np.argmax(values, axis=1).astype(np.int64)
        if values.shape[1] == 1:
            margin = np.full(values.shape[0], np.inf, dtype=np.float64)
        else:
            largest_two = np.partition(values.astype(np.float64), -2, axis=1)[:, -2:]
            margin = largest_two[:, 1] - largest_two[:, 0]
        return winner, margin

    direct_winner, direct_margin = winner_and_margin(direct_flat)
    macro_winner, macro_margin = winner_and_margin(macro_flat)
    diagnostics = {
        "score_delta": delta.reshape(direct.shape),
        "absolute_score_delta": absolute_delta.reshape(direct.shape),
        "relative_score_delta_to_direct": relative_delta_to_direct.reshape(direct.shape),
        "ulp_score_delta": ulp_delta.reshape(direct.shape),
        "score_precision_bits": np.asarray(direct.dtype.itemsize * 8, dtype=np.int64),
        "direct_max_abs_score_per_image": np.max(
            np.abs(direct_flat.astype(np.float64)),
            axis=1,
        ),
        "macro_max_abs_score_per_image": np.max(
            np.abs(macro_flat.astype(np.float64)),
            axis=1,
        ),
        "signed_mean_delta_per_image": np.mean(delta, axis=1),
        "rms_delta_per_image": np.sqrt(np.mean(delta * delta, axis=1)),
        "max_abs_delta_per_image": np.max(np.abs(delta), axis=1),
        "positive_delta_count_per_image": np.count_nonzero(delta > 0.0, axis=1),
        "negative_delta_count_per_image": np.count_nonzero(delta < 0.0, axis=1),
        "zero_delta_count_per_image": np.count_nonzero(delta == 0.0, axis=1),
        "exact_zero_direct_nonzero_macro_per_image": np.all(
            direct_flat == 0.0,
            axis=1,
        )
        & np.any(macro_flat != 0.0, axis=1),
        "direct_argmax": direct_winner,
        "macro_argmax": macro_winner,
        "argmax_equal": direct_winner == macro_winner,
        "direct_winner_margin": direct_margin,
        "macro_winner_margin": macro_margin,
    }
    if (direct_support is None) != (macro_support is None):
        raise ValueError("direct_support and macro_support must be supplied together")
    if direct_support is not None:
        direct_support = np.asarray(direct_support, dtype=bool)
        macro_support = np.asarray(macro_support, dtype=bool)
        if direct_support.shape != direct_flat.shape or macro_support.shape != direct_flat.shape:
            raise ValueError(
                "coarse support diagnostics must match flattened score surfaces, got "
                f"{direct_support.shape}, {macro_support.shape}, and {direct_flat.shape}",
            )
        diagnostics.update(
            direct_support=direct_support.reshape(direct.shape),
            macro_support=macro_support.reshape(direct.shape),
            support_equal=np.all(direct_support == macro_support, axis=1),
            support_symmetric_difference_count=np.count_nonzero(
                direct_support != macro_support,
                axis=1,
            ).astype(np.int64),
        )
    return diagnostics


def _coarse_gaussian_repeat_spread_diagnostics(score_delta_repeats):
    """Report repeat-to-repeat drift without defining an acceptance epsilon.

    ``score_delta_repeats`` has layout
    ``[repeat,image,class,rotation,translation]`` and should be assembled from
    immutable same-hardware paired captures.  The caller must compare this raw
    spread with native/direct repeats, scale panels, discrete outcomes, and
    final quality; this helper cannot promote the GEMM path by itself.
    """

    deltas = np.asarray(score_delta_repeats, dtype=np.float64)
    if deltas.ndim != 5 or deltas.shape[0] < 2:
        raise ValueError(
            "coarse GEMM repeat diagnostics require at least two "
            "[repeat,image,class,rotation,translation] score-delta surfaces",
        )
    flattened = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    signed_mean_per_repeat_image = np.mean(flattened, axis=2)
    max_abs_per_repeat_image = np.max(np.abs(flattened), axis=2)
    return {
        "repeat_count": np.asarray(deltas.shape[0], dtype=np.int64),
        "signed_mean_delta_per_repeat_image": signed_mean_per_repeat_image,
        "max_abs_delta_per_repeat_image": max_abs_per_repeat_image,
        "signed_mean_repeat_spread_per_image": np.ptp(
            signed_mean_per_repeat_image,
            axis=0,
        ),
        "max_abs_repeat_spread_per_image": np.ptp(
            max_abs_per_repeat_image,
            axis=0,
        ),
        "elementwise_delta_repeat_spread": np.ptp(deltas, axis=0),
        "qualification_status": np.asarray(
            "NO_GO_raw_repeat_spread_requires_native_scale_discrete_quality_runtime_context"
        ),
    }


def _coarse_gaussian_scale_panel_diagnostics(
    operand_scales,
    score_deltas_by_scale,
    *,
    precision_bits: int,
):
    """Classify strictly growing multiscale cancellation drift as NO-GO.

    This is an ordering test over raw observations, not an epsilon threshold.
    A non-growing panel remains unqualified until same-hardware repeats,
    discrete outcomes, final quality, and clean runtime all pass.
    """

    scales = np.asarray(operand_scales, dtype=np.float64).reshape(-1)
    deltas = np.asarray(score_deltas_by_scale, dtype=np.float64)
    if scales.size < 2 or deltas.ndim != 5 or deltas.shape[0] != scales.size:
        raise ValueError(
            "coarse GEMM scale diagnostics require matching scale and "
            "[scale,image,class,rotation,translation] arrays with at least two scales",
        )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0) or np.any(np.diff(scales) <= 0.0):
        raise ValueError("coarse GEMM operand scales must be finite, positive, and increasing")
    if int(precision_bits) not in (32, 64):
        raise ValueError("coarse GEMM precision_bits must be 32 or 64")
    flattened = deltas.reshape(scales.size, -1)
    max_abs_by_scale = np.max(np.abs(flattened), axis=1)
    signed_mean_by_scale = np.mean(flattened, axis=1)
    growing_steps = max_abs_by_scale[1:] > max_abs_by_scale[:-1]
    scale_amplified = bool(np.any(growing_steps))
    return {
        "operand_scales": scales,
        "precision_bits": np.asarray(precision_bits, dtype=np.int64),
        "signed_mean_delta_by_scale": signed_mean_by_scale,
        "max_abs_delta_by_scale": max_abs_by_scale,
        "scale_growth_steps": growing_steps,
        "scale_amplified": np.asarray(scale_amplified),
        "qualification_status": np.asarray(
            (
                "NO_GO_scale-amplified_drift"
                if scale_amplified
                else "NO_GO_unqualified_non-growing_scale_panel"
            )
        ),
    }


def _coarse_gaussian_qualification_decision(
    *,
    exact_arithmetic_equivalent: bool,
    repeat_stable: bool | None,
    unbiased_non_directional: bool | None,
    bounded_non_growing: bool | None,
    discrete_choices_equal: bool | None,
    final_basin_quality_equal: bool | None,
    material_runtime_win: bool | None,
    scale_amplified: bool | None,
    negative_implied_diff2: bool,
    nonfinite_scores: bool,
    exact_zero_cancellation_drift: bool,
):
    """Apply the coarse GEMM promotion policy without a numeric tolerance.

    Mathematically equivalent reduction-order noise may pass without bitwise
    score equality only after independent evidence establishes repeat
    stability, non-directional bias, bounded/non-growing scale and iteration
    behavior, exact discrete decisions, unchanged final basin/quality, and a
    material clean-run speedup.  Cancellation evidence, non-finite scores, or
    a negative implied diff2 is an unconditional NO-GO.  ``None`` means the
    corresponding empirical gate has not run and therefore cannot promote.
    """

    failures = []
    pending = []
    if not exact_arithmetic_equivalent:
        failures.append("not_exact-arithmetic-equivalent")
    if scale_amplified is True:
        failures.append("scale-amplified_drift")
    elif scale_amplified is None:
        pending.append("multiscale_growth")
    if negative_implied_diff2:
        failures.append("negative_implied_diff2")
    if nonfinite_scores:
        failures.append("nonfinite_scores")
    if exact_zero_cancellation_drift:
        failures.append("exact-zero_cancellation_drift")
    for name, value in (
        ("repeat_stability", repeat_stable),
        ("unbiased_non-directional", unbiased_non_directional),
        ("bounded_non-growing", bounded_non_growing),
        ("discrete_choices", discrete_choices_equal),
        ("final_basin_quality", final_basin_quality_equal),
        ("material_runtime_win", material_runtime_win),
    ):
        if value is None:
            pending.append(name)
        elif not value:
            failures.append(name)
    if failures:
        status = "NO_GO"
    elif pending:
        status = "NO_GO_UNQUALIFIED"
    else:
        status = "GO_STABLE_BOUNDED_MATHEMATICALLY_EQUIVALENT_NOISE"
    return {
        "status": status,
        "failure_reasons": tuple(failures),
        "pending_gates": tuple(pending),
        "requires_bitwise_score_identity": False,
        "requires_exact_discrete_identity": True,
    }


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _e_step_block_scores_normalized_cc(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """RELION iter-1 normalized cross-correlation score."""
    del batch_norm, image_shape, volume_shape
    rot_block_size = proj_half_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_half),
            proj_half_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_half,
        proj_abs2_half.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _e_step_block_scores_windowed_normalized_cc(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Windowed RELION iter-1 normalized cross-correlation score."""
    del batch_norm, n_windowed, image_shape, volume_shape
    rot_block_size = proj_windowed_weighted.shape[0]
    cross = (
        -2.0
        * jnp.matmul(
            jnp.conj(shifted_windowed),
            proj_windowed_weighted.T,
            precision=jax.lax.Precision.HIGHEST,
        ).real
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv_windowed,
        proj_abs2_windowed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]


def _winner_take_all_probs_for_block(best_argmax, r0, actual_rot, rotation_block_size, n_trans, dtype):
    """Return one-hot pose probabilities for one rotation block."""
    best_argmax = jnp.asarray(best_argmax, dtype=jnp.int32)
    winning_rot = best_argmax // n_trans
    winning_trans = best_argmax % n_trans
    in_block = (winning_rot >= r0) & (winning_rot < (r0 + actual_rot))
    safe_actual_rot = max(int(actual_rot), 1)
    local_rot = jnp.clip(winning_rot - r0, 0, safe_actual_rot - 1)
    flat_local = local_rot * n_trans + winning_trans
    probs = jax.nn.one_hot(
        flat_local,
        rotation_block_size * n_trans,
        dtype=dtype,
    ).reshape(best_argmax.shape[0], rotation_block_size, n_trans)
    return probs * in_block[:, None, None]


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def _m_step_block_windowed(
    shifted_windowed,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_windowed,
    Ft_y,
    Ft_ctf,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Normalize scores to probabilities and compute one windowed M-step block.

    Uses an isfinite-guarded ``exp(scores - log_Z)`` so K-class adaptive 2-pass
    poses where ``scores = -inf`` and ``log_Z = -inf`` give ``probs = 0`` rather
    than NaN.
    """
    rot_block_size = rotations_block.shape[0]
    diff = scores_block - log_Z[:, None, None]
    probs = jnp.where(jnp.isfinite(diff), jnp.exp(diff), 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_windowed = P @ shifted_windowed
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_windowed = probs_sum_t.T @ ctf2_over_nv_windowed
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed


@partial(jax.jit, static_argnums=())
def _update_logsumexp(max_s, sum_exp, scores_block):
    """Streaming logsumexp update from one score block.

    Robust to all-(-inf) score blocks (K-class adaptive 2-pass with an
    empty significance mask): a finite ``safe_new_max`` is used inside the
    exp so we never form -inf - (-inf) = NaN; ``new_max`` is still returned
    as -inf so the streaming logsumexp is exactly -inf for empty inputs.
    """

    accumulator_dtype = sum_exp.dtype
    scores_flat = scores_block.reshape(scores_block.shape[0], -1)
    block_max = jnp.max(scores_flat, axis=1)
    new_max = jnp.maximum(max_s, block_max)
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, jnp.zeros_like(new_max))
    exp_terms = jnp.sum(
        jnp.exp((scores_flat - safe_new_max[:, None]).astype(accumulator_dtype)),
        axis=1,
    )
    safe_max_s = jnp.where(jnp.isfinite(max_s), max_s, jnp.zeros_like(max_s))
    old_term = jnp.where(
        jnp.isfinite(max_s),
        sum_exp * jnp.exp((safe_max_s - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(sum_exp),
    )
    sum_exp = old_term + exp_terms
    return new_max, sum_exp


@jax.jit
def _merge_block_logsumexp(max_s, sum_exp, block_max, block_sum_exp):
    """Merge one pre-reduced block logsumexp into streaming batch stats.

    See ``_update_logsumexp`` for the all-(-inf) handling. The ``safe_*``
    shifts here mirror the same pattern: when both ``max_s`` and
    ``block_max`` are -inf the merge degenerates to 0 + 0 = 0, giving a
    final ``log_Z = -inf`` that the K-class aggregator treats as "no
    contribution" rather than NaN.
    """

    accumulator_dtype = sum_exp.dtype
    new_max = jnp.maximum(max_s, block_max)
    safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, jnp.zeros_like(new_max))
    safe_max_s = jnp.where(jnp.isfinite(max_s), max_s, jnp.zeros_like(max_s))
    safe_block_max = jnp.where(jnp.isfinite(block_max), block_max, jnp.zeros_like(block_max))
    old_term = jnp.where(
        jnp.isfinite(max_s),
        sum_exp * jnp.exp((safe_max_s - safe_new_max).astype(accumulator_dtype)),
        jnp.zeros_like(sum_exp),
    )
    block_term = jnp.where(
        jnp.isfinite(block_max),
        block_sum_exp.astype(accumulator_dtype)
        * jnp.exp(
            (safe_block_max - safe_new_max).astype(accumulator_dtype),
        ),
        jnp.zeros_like(block_sum_exp, dtype=accumulator_dtype),
    )
    return new_max, old_term + block_term


@partial(jax.jit, static_argnums=(5, 6))
def _m_step_block_compute(
    shifted_half,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_half,
    n_images,
    n_trans,
):
    """Normalize scores to probabilities and compute one non-windowed M-step block.

    Uses an isfinite-guarded ``exp(scores - log_Z)`` so K-class adaptive 2-pass
    poses where ``scores = -inf`` and ``log_Z = -inf`` give ``probs = 0`` rather
    than NaN.
    """
    rot_block_size = rotations_block.shape[0]
    diff = scores_block - log_Z[:, None, None]
    probs = jnp.where(jnp.isfinite(diff), jnp.exp(diff), 0.0)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_half = P @ shifted_half
    probs_sum_t = jnp.sum(probs, axis=-1)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_half
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return probs, block_best, block_argmax, summed_half, ctf_probs_half
