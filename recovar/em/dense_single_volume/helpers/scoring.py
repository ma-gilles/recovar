"""JAX scoring and M-step kernels shared by dense single-volume EM helpers."""

from functools import partial

import jax
import jax.numpy as jnp

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
