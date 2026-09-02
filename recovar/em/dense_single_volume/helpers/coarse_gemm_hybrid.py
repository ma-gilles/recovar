"""Certified block selection building blocks for a K=1 coarse-score hybrid.

This module provides the small, default-off pieces needed to turn streamed
candidate score intervals into a fail-closed set of aligned 16-rotation
blocks.  The significance engine owns the production orchestration; exact
rescoring and posterior arithmetic remain on the existing dense E-step path.
"""

from __future__ import annotations

import hashlib
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SOURCE_ROTATION_BLOCK_SIZE = 16
DEFAULT_ROTATION_BLOCK_CAPACITY = 64
RELION_COARSE_NONZERO_SCORE_SPAN = 138.0


class CoarseGemmHybridIntervalState(NamedTuple):
    """Compact complete-coverage reductions for raw and post-prior intervals."""

    raw_block_lower_max: jax.Array
    raw_block_upper_max: jax.Array
    posterior_block_lower_max: jax.Array
    posterior_block_upper_max: jax.Array
    rotation_visit_count: jax.Array
    candidate_count: jax.Array
    invalid_candidate_count: jax.Array


class CoarseGemmHybridBlockSelection(NamedTuple):
    """Fixed-capacity block selection, or a reason to use full direct scoring."""

    eligible: bool
    fallback_reason: str | None
    block_ids: np.ndarray
    block_count: np.ndarray
    posterior_block_count: np.ndarray
    raw_max_block_count: np.ndarray


class CoarseGemmHybridDenseScores(NamedTuple):
    """Exact selected scores restored to the full RELION coarse layout.

    ``posterior_scores_flat`` has shape ``[B, R*T]`` in source-rotation-major,
    translation-minor order.  Omitted candidates and padded image rows are
    represented by ``-inf``.  Retaining the full logical candidate length is
    intentional: removing exact zero-weight candidates changes the reduction
    topology of RELION's float32 CUB scan and can change its denominator and
    normalized probabilities by one or more ULPs.
    """

    posterior_scores_flat: jax.Array
    raw_score_max: jax.Array
    min_diff2_offsets: jax.Array
    best_score: jax.Array
    best_pose: jax.Array
    selected_output_valid: jax.Array


class CoarseGemmExpandedF64GammaBounds(NamedTuple):
    """Upward-rounded coefficients for the promoted expanded-square bound."""

    cross: float
    energy: float
    initial_diff2: float
    energy_envelope: float


class CoarseGemmCertificateTopology(NamedTuple):
    """Immutable link between the GEMM terms and direct CUDA traversal."""

    full_position_count: int
    compact_pixel_count: int
    translation_count: int
    full_to_compact: np.ndarray
    full_to_compact_sha256: str
    expanded_f64_gammas: CoarseGemmExpandedF64GammaBounds
    direct_f32_gamma: float


def _upward_rounded_gamma(operation_count: int, unit_roundoff: float) -> float:
    """Return an upward-rounded ``gamma_k`` for a reviewed operation count."""

    operation_count = operator.index(operation_count)
    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    scaled_roundoff = np.float64(operation_count) * np.float64(unit_roundoff)
    if not np.isfinite(scaled_roundoff) or scaled_roundoff >= 1.0:
        raise ValueError("gamma operation count is outside its valid range")
    denominator = np.nextafter(
        np.float64(1.0) - scaled_roundoff,
        np.float64(-np.inf),
    )
    return float(
        np.nextafter(
            scaled_roundoff / denominator,
            np.float64(np.inf),
        )
    )


def _upward_add_f64(left, right):
    """Add once in FP64, then step the result toward positive infinity."""

    with jax.enable_x64(True):
        return jnp.nextafter(
            jnp.asarray(left, dtype=jnp.float64)
            + jnp.asarray(right, dtype=jnp.float64),
            jnp.float64(jnp.inf),
        )


def _upward_multiply_f64(left, right):
    """Multiply once in FP64, then step the result toward positive infinity."""

    with jax.enable_x64(True):
        return jnp.nextafter(
            jnp.asarray(left, dtype=jnp.float64)
            * jnp.asarray(right, dtype=jnp.float64),
            jnp.float64(jnp.inf),
        )


def certified_f64_expanded_score_gammas(
    traversed_full_position_bound: int,
) -> CoarseGemmExpandedF64GammaBounds:
    """Return coefficients for the explicit-component promoted score bound.

    The position bound covers both each energy reduction and the full pixel
    traversal represented by the cross term.  It must therefore describe the
    stored scorer input, not merely the number of nonzero compact weights.
    """

    try:
        n_positions = operator.index(traversed_full_position_bound)
    except TypeError as error:
        raise ValueError("traversed_full_position_bound must be an integer") from error
    if n_positions <= 0:
        raise ValueError("traversed_full_position_bound must be positive")
    unit_roundoff = 2.0**-53
    return CoarseGemmExpandedF64GammaBounds(
        cross=_upward_rounded_gamma(2 * n_positions + 4, unit_roundoff),
        energy=_upward_rounded_gamma(n_positions + 5, unit_roundoff),
        initial_diff2=_upward_rounded_gamma(3, unit_roundoff),
        energy_envelope=_upward_rounded_gamma(n_positions + 2, unit_roundoff),
    )


def _validate_full_to_compact_mapping(
    full_to_compact,
    *,
    compact_pixel_count: int,
) -> np.ndarray:
    """Return a validated view of one complete direct-kernel lookup."""

    mapping = np.asarray(full_to_compact)
    if mapping.dtype != np.dtype(np.int32) or mapping.ndim != 1 or mapping.size <= 0:
        raise ValueError("full_to_compact must be a nonempty rank-1 int32 array")
    if np.any(mapping < -1) or np.any(mapping >= compact_pixel_count):
        raise ValueError("full_to_compact contains an out-of-range compact pixel ID")
    visited = np.sort(mapping[mapping >= 0])
    if not np.array_equal(
        visited,
        np.arange(compact_pixel_count, dtype=np.int32),
    ):
        raise ValueError("full_to_compact must visit every compact pixel exactly once")
    return mapping


def _full_to_compact_sha256(mapping: np.ndarray) -> str:
    return hashlib.sha256(mapping.tobytes(order="C")).hexdigest()


def plan_coarse_gemm_certificate_topology(
    full_to_compact,
    *,
    compact_pixel_count: int,
    translation_count: int,
) -> CoarseGemmCertificateTopology:
    """Copy, seal, and bind the actual direct lookup to all coefficients.

    Every compact GEMM pixel must occur exactly once in the direct kernel's
    full traversal; ``-1`` entries are allowed and represent skipped full-grid
    positions.  The selected-source16 wrapper later consumes this owned copy,
    so the proof coefficients and exact rescore cannot receive different
    lookup arrays.
    """

    try:
        compact_pixels = operator.index(compact_pixel_count)
        translations = operator.index(translation_count)
    except TypeError as error:
        raise ValueError("certificate topology counts must be integers") from error
    if compact_pixels <= 0:
        raise ValueError("compact_pixel_count must be positive")
    mapping = _validate_full_to_compact_mapping(
        full_to_compact,
        compact_pixel_count=compact_pixels,
    )
    sealed_mapping = np.array(mapping, copy=True, order="C")
    sealed_mapping.setflags(write=False)
    full_positions = int(sealed_mapping.size)
    return CoarseGemmCertificateTopology(
        full_position_count=full_positions,
        compact_pixel_count=compact_pixels,
        translation_count=translations,
        full_to_compact=sealed_mapping,
        full_to_compact_sha256=_full_to_compact_sha256(sealed_mapping),
        expanded_f64_gammas=certified_f64_expanded_score_gammas(full_positions),
        direct_f32_gamma=certified_f32_dot_product_gamma(
            full_positions,
            translations,
        ),
    )


def validate_coarse_gemm_certificate_topology(
    topology: CoarseGemmCertificateTopology,
) -> None:
    """Recompute every lookup, count, digest, and coefficient invariant."""

    if not isinstance(topology, CoarseGemmCertificateTopology):
        raise TypeError("topology must be a CoarseGemmCertificateTopology")
    mapping = np.asarray(topology.full_to_compact)
    if not mapping.flags.c_contiguous or mapping.flags.writeable:
        raise ValueError(
            "certificate topology lookup must be a contiguous read-only snapshot",
        )
    try:
        full_positions = operator.index(topology.full_position_count)
        compact_pixels = operator.index(topology.compact_pixel_count)
        translations = operator.index(topology.translation_count)
    except TypeError as error:
        raise ValueError("certificate topology counts must be integers") from error
    if not isinstance(topology.full_to_compact_sha256, str):
        raise ValueError("certificate topology digest must be a string")
    if not isinstance(
        topology.expanded_f64_gammas,
        CoarseGemmExpandedF64GammaBounds,
    ):
        raise ValueError("certificate topology FP64 gammas have an invalid type")
    gamma_values = (*topology.expanded_f64_gammas, topology.direct_f32_gamma)
    if any(np.asarray(value).shape != () for value in gamma_values):
        raise ValueError("certificate topology gammas must be scalars")
    expected = plan_coarse_gemm_certificate_topology(
        mapping,
        compact_pixel_count=compact_pixels,
        translation_count=translations,
    )
    if (
        full_positions != expected.full_position_count
        or compact_pixels != expected.compact_pixel_count
        or translations != expected.translation_count
        or topology.full_to_compact_sha256 != expected.full_to_compact_sha256
        or topology.expanded_f64_gammas != expected.expanded_f64_gammas
        or topology.direct_f32_gamma != expected.direct_f32_gamma
    ):
        raise ValueError(
            "certificate topology count, digest, or gamma invariant changed",
        )


@jax.jit
def coarse_gemm_expanded_score_eta_f64(
    reference_energy_hat,
    image_energy_hat,
    initial_diff2,
    *,
    cross_gamma,
    energy_gamma,
    initial_diff2_gamma,
    energy_envelope_gamma,
):
    """Bound promoted expanded-score error from its existing energy dots.

    ``reference_energy_hat`` is ``[image, rotation]`` and
    ``image_energy_hat`` is ``[image, translation]``.  All inputs are the
    results of explicit component-square FP64 arithmetic.  Invalid values
    become NaN bounds so the compact selector routes the batch to direct
    scoring.
    """

    with jax.enable_x64(True):
        reference_energy = jnp.asarray(reference_energy_hat, dtype=jnp.float64)
        image_energy = jnp.asarray(image_energy_hat, dtype=jnp.float64)
        initial = jnp.asarray(initial_diff2, dtype=jnp.float64)
        if reference_energy.ndim != 2 or image_energy.ndim != 2:
            raise ValueError("energy estimates must be rank-2 arrays")
        if reference_energy.shape[0] != image_energy.shape[0]:
            raise ValueError("reference and image energies must share their image axis")
        if initial.shape != (reference_energy.shape[0],):
            raise ValueError("initial_diff2 must have one value per image")

        coefficients = tuple(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (
                cross_gamma,
                energy_gamma,
                initial_diff2_gamma,
                energy_envelope_gamma,
            )
        )
        if any(value.shape != () for value in coefficients):
            raise ValueError("all expanded-score gamma coefficients must be scalar")
        cross_coefficient, energy_coefficient, initial_coefficient, alpha = coefficients

        positive_inf = jnp.float64(jnp.inf)
        negative_inf = jnp.float64(-jnp.inf)
        denominator = jnp.nextafter(jnp.float64(1.0) - alpha, negative_inf)
        reference_upper = jnp.nextafter(reference_energy / denominator, positive_inf)
        image_upper = jnp.nextafter(image_energy / denominator, positive_inf)
        reference_candidate = reference_upper[:, :, None]
        image_candidate = image_upper[:, None, :]
        energy_sum = jnp.nextafter(reference_candidate + image_candidate, positive_inf)
        energy_product = jnp.nextafter(reference_candidate * image_candidate, positive_inf)
        cross_envelope = jnp.nextafter(jnp.sqrt(energy_product), positive_inf)
        cross_term = jnp.nextafter(cross_coefficient * cross_envelope, positive_inf)
        energy_term = jnp.nextafter(energy_coefficient * energy_sum, positive_inf)
        energy_term = jnp.nextafter(jnp.float64(0.5) * energy_term, positive_inf)
        initial_term = jnp.nextafter(
            initial_coefficient * initial[:, None, None],
            positive_inf,
        )
        eta = jnp.nextafter(
            jnp.nextafter(cross_term + energy_term, positive_inf) + initial_term,
            positive_inf,
        )
        coefficient_valid = (
            jnp.isfinite(cross_coefficient)
            & (cross_coefficient >= 0.0)
            & jnp.isfinite(energy_coefficient)
            & (energy_coefficient >= 0.0)
            & jnp.isfinite(initial_coefficient)
            & (initial_coefficient >= 0.0)
            & jnp.isfinite(alpha)
            & (alpha >= 0.0)
            & (alpha < 1.0)
            & jnp.isfinite(denominator)
            & (denominator > 0.0)
        )
        input_valid = (
            jnp.all(jnp.isfinite(reference_energy) & (reference_energy >= 0.0), axis=1)[:, None, None]
            & jnp.all(jnp.isfinite(image_energy) & (image_energy >= 0.0), axis=1)[:, None, None]
            & (jnp.isfinite(initial) & (initial >= 0.0))[:, None, None]
        )
        output_valid = (
            jnp.isfinite(reference_candidate)
            & jnp.isfinite(image_candidate)
            & jnp.isfinite(cross_envelope)
            & jnp.isfinite(eta)
            & (eta >= 0.0)
        )
        return jnp.where(coefficient_valid & input_valid & output_valid, eta, jnp.nan)


def certified_f32_dot_product_gamma(
    traversed_full_position_bound: int,
    n_translations: int,
) -> float:
    """Return an upward-rounded ``gamma_k`` for the direct CUDA topology.

    ``traversed_full_position_bound`` is the upper bound on full image
    positions traversed by the direct kernel, including positions whose
    compacted scientific weight may be zero.  It is not the number of retained
    or non-skipped compact terms.
    """

    try:
        n_positions = operator.index(traversed_full_position_bound)
        n_translations = operator.index(n_translations)
    except TypeError as error:
        raise ValueError("kernel dimensions must be integers") from error
    if n_positions <= 0 or not 1 <= n_translations <= 128:
        raise ValueError(
            "traversed_full_position_bound must be positive and n_translations must be in [1, 128]",
        )
    translations_per_lane = 128 // n_translations
    lane_term_bound = ((n_positions + 31) // 32) * ((32 + translations_per_lane - 1) // translations_per_lane)
    operation_count = lane_term_bound + translations_per_lane + 5
    unit_roundoff = 2.0**-24
    scaled_roundoff = operation_count * unit_roundoff
    denominator = 1.0 - scaled_roundoff
    if denominator <= 0.0:
        raise ValueError("dot-product error-bound denominator is non-positive")
    gamma = scaled_roundoff / denominator
    return float(np.nextafter(np.float64(gamma), np.float64(np.inf)))


@jax.jit
def coarse_gemm_direct_f32_ftz_envelope_and_range(
    reference_component_abs_max,
    image_component_abs_max,
    weight_max,
    initial_diff2,
    traversed_full_position_count,
    direct_gamma,
):
    """Return a provisional FTZ envelope and a fail-closed FP32 range gate.

    This is the executable counterpart of the provisional range model in
    ``docs/math/vdam_coarse_gemm_error_certificate.md``.  The deliberately
    generous additive envelope covers an absolute ``FLT_MIN`` perturbation at
    every direct-kernel operation and input, amplified by a power-of-two
    safety factor.  It is not a replacement for the backend/SASS audit needed
    before enabling the hybrid path by default.

    The returned arrays have shape ``[image, rotation, 1]``.  A false range
    value must invalidate every translation for that image/rotation pair.
    """

    with jax.enable_x64(True):
        reference_max = jnp.asarray(reference_component_abs_max, dtype=jnp.float64)
        image_max = jnp.asarray(image_component_abs_max, dtype=jnp.float64)
        maximum_weight = jnp.asarray(weight_max, dtype=jnp.float64)
        initial = jnp.asarray(initial_diff2, dtype=jnp.float64)
        n_positions = jnp.asarray(traversed_full_position_count, dtype=jnp.float64)
        gamma = jnp.asarray(direct_gamma, dtype=jnp.float64)
        if reference_max.ndim != 1:
            raise ValueError("reference_component_abs_max must be rank 1")
        if (
            image_max.ndim != 1
            or maximum_weight.shape != image_max.shape
            or initial.shape != image_max.shape
        ):
            raise ValueError("image range summaries must be matching rank-1 arrays")
        if n_positions.shape != () or gamma.shape != ():
            raise ValueError("direct range coefficients must be scalar")

        f32_min_normal = jnp.float64(np.finfo(np.float32).tiny)
        f32_max = jnp.float64(np.finfo(np.float32).max)
        unit_roundoff = jnp.float64(2.0**-24)
        one_plus_u = _upward_add_f64(jnp.float64(1.0), unit_roundoff)

        # Maxima are component maxima, so |real_delta| and |imag_delta| are
        # each bounded by reference_max + image_max.  Every operation below is
        # rounded upward in FP64 and mirrors a local source16 FP32 stage.
        difference = _upward_add_f64(
            reference_max[None, :],
            image_max[:, None],
        )
        direct_difference = _upward_multiply_f64(one_plus_u, difference)
        direct_difference = _upward_add_f64(
            direct_difference,
            _upward_multiply_f64(jnp.float64(3.0), f32_min_normal),
        )
        difference_square = _upward_multiply_f64(
            direct_difference,
            direct_difference,
        )
        imaginary_square = _upward_add_f64(
            _upward_multiply_f64(one_plus_u, difference_square),
            f32_min_normal,
        )
        square_sum_input = _upward_add_f64(
            difference_square,
            imaginary_square,
        )
        square_sum = _upward_add_f64(
            _upward_multiply_f64(one_plus_u, square_sum_input),
            f32_min_normal,
        )
        half_square_sum = _upward_multiply_f64(jnp.float64(0.5), square_sum)
        half_square_sum = _upward_multiply_f64(one_plus_u, half_square_sum)
        half_square_sum = _upward_add_f64(half_square_sum, f32_min_normal)
        loaded_weight = _upward_add_f64(
            maximum_weight[:, None],
            f32_min_normal,
        )
        local_term = _upward_multiply_f64(half_square_sum, loaded_weight)
        local_term = _upward_multiply_f64(one_plus_u, local_term)
        local_term = _upward_add_f64(local_term, f32_min_normal)

        # A large power-of-two factor keeps the provisional FTZ/DAZ model
        # independent of fragile compiler details while remaining negligible
        # at ordinary cryo-EM scales.  Keep this term separate from eta: eta
        # bounds the FP64 expanded center, whereas rho only covers additive
        # behavior absent from the relative-error direct theorem.
        scale_difference = jnp.maximum(jnp.float64(1.0), difference)
        magnitude_scale = _upward_multiply_f64(
            scale_difference,
            scale_difference,
        )
        magnitude_scale = _upward_multiply_f64(
            magnitude_scale,
            jnp.maximum(jnp.float64(1.0), maximum_weight[:, None]),
        )
        operation_budget = _upward_add_f64(
            n_positions,
            jnp.float64(128.0),
        )
        operation_budget = _upward_multiply_f64(
            jnp.float64(2.0**20),
            operation_budget,
        )
        rho = _upward_multiply_f64(operation_budget, f32_min_normal)
        rho = _upward_multiply_f64(rho, magnitude_scale)

        exact_term_upper = _upward_multiply_f64(
            difference,
            difference,
        )
        exact_term_upper = _upward_multiply_f64(
            exact_term_upper,
            maximum_weight[:, None],
        )
        exact_residual_upper = _upward_multiply_f64(
            n_positions,
            exact_term_upper,
        )
        exact_residual_upper = _upward_add_f64(
            initial[:, None],
            exact_residual_upper,
        )
        direct_upper = _upward_add_f64(jnp.float64(1.0), gamma)
        direct_upper = _upward_multiply_f64(direct_upper, exact_residual_upper)
        direct_upper = _upward_add_f64(direct_upper, rho)
        inputs_valid = (
            (jnp.isfinite(reference_max) & (reference_max >= 0.0))[None, :]
            & (jnp.isfinite(image_max) & (image_max >= 0.0))[:, None]
            & (jnp.isfinite(maximum_weight) & (maximum_weight >= 0.0))[:, None]
            & (jnp.isfinite(initial) & (initial >= 0.0))[:, None]
            & jnp.isfinite(n_positions)
            & (n_positions >= 1.0)
            & jnp.isfinite(gamma)
            & (gamma >= 0.0)
        )
        local_values = (
            direct_difference,
            difference_square,
            imaginary_square,
            square_sum_input,
            square_sum,
            half_square_sum,
            loaded_weight,
            local_term,
        )
        local_range_valid = jnp.ones_like(difference, dtype=jnp.bool_)
        for value in local_values:
            local_range_valid &= jnp.isfinite(value) & (value <= f32_max)
        range_valid = (
            inputs_valid
            & local_range_valid
            & jnp.isfinite(rho)
            & (rho >= 0.0)
            & jnp.isfinite(direct_upper)
            & (direct_upper <= f32_max)
        )
        return rho[..., None], range_valid[..., None]


def coarse_gemm_hybrid_interval_state_bytes(
    batch_size: int,
    n_rotations: int,
) -> int:
    """Return persistent bytes for the four interval tables and coverage state."""

    batch_size = operator.index(batch_size)
    n_rotations = operator.index(n_rotations)
    if batch_size <= 0 or n_rotations <= 0:
        raise ValueError("batch_size and n_rotations must be positive")
    n_blocks = (n_rotations + SOURCE_ROTATION_BLOCK_SIZE - 1) // SOURCE_ROTATION_BLOCK_SIZE
    return int(
        4 * batch_size * n_blocks * np.dtype(np.float64).itemsize
        + n_rotations * np.dtype(np.int32).itemsize
        + 2 * batch_size * np.dtype(np.int32).itemsize
    )


@jax.jit
def coarse_gemm_direct_score_intervals(
    expanded_score_f64,
    eta_f64,
    gamma_f64,
    ftz_absolute_error_f64=0.0,
    direct_range_valid=True,
):
    """Construct outward-rounded direct-score intervals around live ``g64``.

    ``eta_f64`` certifies ``abs(g64 - S)`` and may be scalar or broadcastable
    to ``g64``.  ``ftz_absolute_error_f64`` is a separate additive envelope
    for behavior excluded by the relative-error theorem.  Invalid/nonfinite
    inputs become NaN endpoints so the streamed state and host selector fail
    closed without inspecting a full score cube.
    """

    with jax.enable_x64(True):
        score = jnp.asarray(expanded_score_f64, dtype=jnp.float64)
        eta = jnp.broadcast_to(jnp.asarray(eta_f64, dtype=jnp.float64), score.shape)
        ftz_error = jnp.broadcast_to(
            jnp.asarray(ftz_absolute_error_f64, dtype=jnp.float64),
            score.shape,
        )
        range_valid = jnp.broadcast_to(jnp.asarray(direct_range_valid), score.shape)
        gamma = jnp.asarray(gamma_f64, dtype=jnp.float64)
        if gamma.shape != ():
            raise ValueError("gamma_f64 must be scalar")
        if range_valid.dtype != jnp.bool_:
            raise TypeError("direct_range_valid must have boolean dtype")
        positive_inf = jnp.full_like(score, jnp.inf)
        negative_inf = jnp.full_like(score, -jnp.inf)
        q_upper = jnp.nextafter(
            jnp.maximum(jnp.float64(0.0), -score + eta),
            positive_inf,
        )
        gamma_term = jnp.nextafter(gamma * q_upper, positive_inf)
        error_upper = jnp.nextafter(eta + ftz_error, positive_inf)
        error_upper = jnp.nextafter(error_upper + gamma_term, positive_inf)
        lower = jnp.nextafter(score - error_upper, negative_inf)
        upper = jnp.nextafter(score + error_upper, positive_inf)
        valid = (
            jnp.isfinite(score)
            & jnp.isfinite(eta)
            & (eta >= 0.0)
            & jnp.isfinite(ftz_error)
            & (ftz_error >= 0.0)
            & range_valid
            # A squared-difference score is non-positive.  Given |g-S|<=eta,
            # g<=eta is the subtraction-free form of the required g-eta<=0
            # certificate check and avoids another rounded endpoint operation.
            & (score <= eta)
            & jnp.isfinite(gamma)
            & (gamma >= 0.0)
            & jnp.isfinite(error_upper)
            & jnp.isfinite(lower)
            & jnp.isfinite(upper)
            & (lower <= upper)
        )
        nan = jnp.full_like(score, jnp.nan)
        return jnp.where(valid, lower, nan), jnp.where(valid, upper, nan)


def _enclosing_f32_add(lower, upper, prior):
    """Enclose one production-order FP32 addition, entirely on device."""

    prior_f32 = jnp.asarray(prior, dtype=jnp.float32)
    prior_f64 = prior_f32.astype(jnp.float64)
    negative_inf64 = jnp.full_like(lower, -jnp.inf)
    positive_inf64 = jnp.full_like(upper, jnp.inf)
    lower_sum = jnp.nextafter(lower + prior_f64, negative_inf64)
    upper_sum = jnp.nextafter(upper + prior_f64, positive_inf64)
    lower_f32 = lower_sum.astype(jnp.float32)
    upper_f32 = upper_sum.astype(jnp.float32)
    lower_f32 = jnp.where(
        lower_f32.astype(jnp.float64) > lower_sum,
        jnp.nextafter(lower_f32, jnp.full_like(lower_f32, -jnp.inf)),
        lower_f32,
    )
    upper_f32 = jnp.where(
        upper_f32.astype(jnp.float64) < upper_sum,
        jnp.nextafter(upper_f32, jnp.full_like(upper_f32, jnp.inf)),
        upper_f32,
    )
    result_lower = lower_f32.astype(jnp.float64)
    result_upper = upper_f32.astype(jnp.float64)
    valid = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower <= upper)
        & jnp.isfinite(prior_f32)
        & jnp.isfinite(result_lower)
        & jnp.isfinite(result_upper)
        & (result_lower <= result_upper)
    )
    nan = jnp.full_like(result_lower, jnp.nan)
    return jnp.where(valid, result_lower, nan), jnp.where(valid, result_upper, nan)


@jax.jit
def propagate_coarse_gemm_intervals_through_f32_priors(
    raw_lower,
    raw_upper,
    *,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
):
    """Apply class, rotation, then translation priors as dense ``_add_priors``.

    The class addition is mandatory even for K=1.  Endpoint work stays in JAX;
    malformed shapes fail while tracing and invalid values become NaN bounds.
    """

    with jax.enable_x64(True):
        lower = jnp.asarray(raw_lower, dtype=jnp.float64)
        upper = jnp.asarray(raw_upper, dtype=jnp.float64)
        if lower.ndim != 3 or lower.shape != upper.shape:
            raise ValueError("raw endpoints must be equal [image, rotation, translation] arrays")
        class_prior = jnp.asarray(class_log_prior, dtype=jnp.float32)
        if class_prior.shape != ():
            raise ValueError("class_log_prior must be scalar")
        lower, upper = _enclosing_f32_add(lower, upper, class_prior)
        if rotation_log_prior is not None:
            rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
            if rotation_prior.shape != (lower.shape[1],):
                raise ValueError("rotation_log_prior must have one value per rotation")
            lower, upper = _enclosing_f32_add(
                lower,
                upper,
                rotation_prior[None, :, None],
            )
        if translation_log_prior is not None:
            translation_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
            if translation_prior.shape == (lower.shape[2],):
                translation_prior = translation_prior[None, None, :]
            elif translation_prior.shape == (lower.shape[0], lower.shape[2]):
                translation_prior = translation_prior[:, None, :]
            else:
                raise ValueError(
                    "translation_log_prior must be [translation] or [image, translation]",
                )
            lower, upper = _enclosing_f32_add(lower, upper, translation_prior)
        return lower, upper


def initialize_coarse_gemm_hybrid_interval_state(
    batch_size: int,
    n_rotations: int,
) -> CoarseGemmHybridIntervalState:
    """Create an empty compact state; rotations need not yet be block aligned."""

    batch_size = operator.index(batch_size)
    n_rotations = operator.index(n_rotations)
    if batch_size <= 0 or n_rotations <= 0:
        raise ValueError("batch_size and n_rotations must be positive")
    n_blocks = (n_rotations + SOURCE_ROTATION_BLOCK_SIZE - 1) // SOURCE_ROTATION_BLOCK_SIZE
    with jax.enable_x64(True):
        empty = jnp.full((batch_size, n_blocks), -jnp.inf, dtype=jnp.float64)
        return CoarseGemmHybridIntervalState(
            raw_block_lower_max=empty,
            raw_block_upper_max=empty,
            posterior_block_lower_max=empty,
            posterior_block_upper_max=empty,
            rotation_visit_count=jnp.zeros((n_rotations,), dtype=jnp.int32),
            candidate_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            invalid_candidate_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        )


@jax.jit
def _update_coarse_gemm_hybrid_interval_state(
    state,
    raw_lower,
    raw_upper,
    posterior_lower,
    posterior_upper,
    rotation_ids,
    actual_image_count,
):
    batch_size, rotation_count, translation_count = raw_lower.shape
    active_rows = jnp.arange(batch_size, dtype=jnp.int32) < actual_image_count
    valid = (
        jnp.isfinite(raw_lower)
        & jnp.isfinite(raw_upper)
        & (raw_lower <= raw_upper)
        & jnp.isfinite(posterior_lower)
        & jnp.isfinite(posterior_upper)
        & (posterior_lower <= posterior_upper)
    )

    def update_block_maximum(current, values):
        rotation_maximum = jnp.max(
            jnp.where(active_rows[:, None, None], values, -jnp.inf),
            axis=2,
        )
        block_ids = rotation_ids // SOURCE_ROTATION_BLOCK_SIZE
        return current.at[:, block_ids].max(rotation_maximum)

    visits = state.rotation_visit_count.at[rotation_ids].add(
        jnp.ones((rotation_count,), dtype=jnp.int32),
    )
    candidate_increment = jnp.int32(rotation_count * translation_count)
    invalid_increment = jnp.sum(~valid, axis=(1, 2), dtype=jnp.int32)
    return CoarseGemmHybridIntervalState(
        raw_block_lower_max=update_block_maximum(state.raw_block_lower_max, raw_lower),
        raw_block_upper_max=update_block_maximum(state.raw_block_upper_max, raw_upper),
        posterior_block_lower_max=update_block_maximum(
            state.posterior_block_lower_max,
            posterior_lower,
        ),
        posterior_block_upper_max=update_block_maximum(
            state.posterior_block_upper_max,
            posterior_upper,
        ),
        rotation_visit_count=visits,
        candidate_count=state.candidate_count + jnp.where(active_rows, candidate_increment, jnp.int32(0)),
        invalid_candidate_count=state.invalid_candidate_count + jnp.where(active_rows, invalid_increment, jnp.int32(0)),
    )


def update_coarse_gemm_hybrid_interval_state(
    state: CoarseGemmHybridIntervalState,
    raw_lower,
    raw_upper,
    posterior_lower,
    posterior_upper,
    *,
    rotation_offset: int,
    valid_rotation_count: int,
    actual_image_count: int,
) -> CoarseGemmHybridIntervalState:
    """Stream one contiguous rotation chunk without inspecting its score cubes."""

    arrays = tuple(
        jnp.asarray(value)
        for value in (
            raw_lower,
            raw_upper,
            posterior_lower,
            posterior_upper,
        )
    )
    if arrays[0].ndim != 3 or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all endpoints must be equal [image, rotation, translation] arrays")
    batch_size, supplied_rotations, _ = arrays[0].shape
    rotation_offset = operator.index(rotation_offset)
    valid_rotation_count = operator.index(valid_rotation_count)
    actual_image_count = operator.index(actual_image_count)
    n_rotations = state.rotation_visit_count.shape[0]
    if (
        batch_size != state.raw_block_lower_max.shape[0]
        or rotation_offset < 0
        or valid_rotation_count <= 0
        or valid_rotation_count > supplied_rotations
        or rotation_offset + valid_rotation_count > n_rotations
        or actual_image_count <= 0
        or actual_image_count > batch_size
    ):
        raise ValueError("invalid hybrid interval update bounds")
    rotation_ids = jnp.arange(
        rotation_offset,
        rotation_offset + valid_rotation_count,
        dtype=jnp.int32,
    )
    sliced = tuple(value[:, :valid_rotation_count, :] for value in arrays)
    return _update_coarse_gemm_hybrid_interval_state(
        state,
        *sliced,
        rotation_ids,
        jnp.int32(actual_image_count),
    )


def _failed_selection(batch_size: int, capacity: int, reason: str):
    width = max(1, capacity)
    return CoarseGemmHybridBlockSelection(
        eligible=False,
        fallback_reason=reason,
        block_ids=np.full((batch_size, width), -1, dtype=np.int32),
        block_count=np.zeros((batch_size,), dtype=np.int32),
        posterior_block_count=np.zeros((batch_size,), dtype=np.int32),
        raw_max_block_count=np.zeros((batch_size,), dtype=np.int32),
    )


def select_coarse_gemm_hybrid_rotation_blocks(
    state: CoarseGemmHybridIntervalState,
    *,
    actual_image_count: int,
    n_rotations: int,
    n_translations: int,
    certificate_valid: bool | None = None,
    block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
    nonzero_score_span: float = RELION_COARSE_NONZERO_SCORE_SPAN,
) -> CoarseGemmHybridBlockSelection:
    """Transfer compact state and select a conservative source-block union."""

    batch_size = state.raw_block_lower_max.shape[0]
    try:
        actual_image_count = operator.index(actual_image_count)
        n_rotations = operator.index(n_rotations)
        n_translations = operator.index(n_translations)
        block_capacity = operator.index(block_capacity)
        nonzero_score_span = float(nonzero_score_span)
    except (TypeError, ValueError, OverflowError):
        return _failed_selection(batch_size, 1, "invalid_selection_configuration")
    if not isinstance(certificate_valid, (bool, np.bool_)) or not certificate_valid:
        return _failed_selection(
            batch_size,
            block_capacity,
            "missing_or_invalid_certificate_contract",
        )
    if n_rotations <= 0 or n_rotations % SOURCE_ROTATION_BLOCK_SIZE:
        return _failed_selection(batch_size, block_capacity, "rotation_tail_not_supported")
    if (
        actual_image_count <= 0
        or actual_image_count > batch_size
        or n_translations <= 0
        or block_capacity <= 0
        or not np.isfinite(nonzero_score_span)
        or nonzero_score_span < 0.0
    ):
        return _failed_selection(batch_size, block_capacity, "invalid_selection_configuration")
    n_blocks = n_rotations // SOURCE_ROTATION_BLOCK_SIZE
    block_fields = (
        state.raw_block_lower_max,
        state.raw_block_upper_max,
        state.posterior_block_lower_max,
        state.posterior_block_upper_max,
    )
    if (
        state.rotation_visit_count.shape != (n_rotations,)
        or state.candidate_count.shape != (batch_size,)
        or state.invalid_candidate_count.shape != (batch_size,)
        or any(field.shape != (batch_size, n_blocks) for field in block_fields)
    ):
        return _failed_selection(batch_size, block_capacity, "incomplete_block_table")
    visits = np.asarray(state.rotation_visit_count, dtype=np.int32)
    if not np.all(visits == 1):
        return _failed_selection(
            batch_size,
            block_capacity,
            "incomplete_or_duplicate_rotation_coverage",
        )
    expected_candidates = n_rotations * n_translations
    counts = np.asarray(state.candidate_count, dtype=np.int64)[:actual_image_count]
    if not np.all(counts == expected_candidates):
        return _failed_selection(batch_size, block_capacity, "incomplete_candidate_coverage")
    invalid = np.asarray(state.invalid_candidate_count, dtype=np.int64)[:actual_image_count]
    if np.any(invalid):
        return _failed_selection(
            batch_size,
            block_capacity,
            "invalid_or_nonfinite_candidate_interval",
        )
    raw_lower, raw_upper, posterior_lower, posterior_upper = (
        np.asarray(field, dtype=np.float64)[:actual_image_count] for field in block_fields
    )
    if not (
        np.all(np.isfinite(raw_lower))
        and np.all(np.isfinite(raw_upper))
        and np.all(np.isfinite(posterior_lower))
        and np.all(np.isfinite(posterior_upper))
        and np.all(raw_lower <= raw_upper)
        and np.all(posterior_lower <= posterior_upper)
    ):
        return _failed_selection(
            batch_size,
            block_capacity,
            "invalid_or_nonfinite_block_interval",
        )

    block_ids = np.full((batch_size, block_capacity), -1, dtype=np.int32)
    block_count = np.zeros((batch_size,), dtype=np.int32)
    posterior_count = np.zeros((batch_size,), dtype=np.int32)
    raw_count = np.zeros((batch_size,), dtype=np.int32)
    for row in range(actual_image_count):
        raw_ids = np.flatnonzero(raw_upper[row] >= np.max(raw_lower[row]))
        posterior_ids = np.flatnonzero(
            posterior_upper[row] >= np.max(posterior_lower[row]) - nonzero_score_span,
        )
        selected = np.union1d(raw_ids, posterior_ids).astype(np.int32, copy=False)
        if selected.size == 0:
            return _failed_selection(batch_size, block_capacity, "empty_block_selection")
        if selected.size > block_capacity:
            return _failed_selection(batch_size, block_capacity, "block_capacity_overflow")
        block_ids[row, : selected.size] = selected
        block_count[row] = selected.size
        posterior_count[row] = posterior_ids.size
        raw_count[row] = raw_ids.size
    return CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=block_ids,
        block_count=block_count,
        posterior_block_count=posterior_count,
        raw_max_block_count=raw_count,
    )


def validate_coarse_gemm_hybrid_block_selection_for_rescore(
    selection: CoarseGemmHybridBlockSelection,
    *,
    actual_image_count: int,
    n_rotations: int,
) -> None:
    """Validate the host selection before dispatching or assembling exact scores.

    The selector emits a strictly increasing active prefix of source-16 block
    IDs followed by ``-1`` padding.  Enforcing that contract here makes the
    selected FFI slot order an explicit, fail-closed boundary rather than an
    implicit assumption in posterior assembly.
    """

    if not isinstance(selection, CoarseGemmHybridBlockSelection):
        raise TypeError("selection must be a CoarseGemmHybridBlockSelection")
    try:
        actual_images = operator.index(actual_image_count)
        rotations = operator.index(n_rotations)
    except TypeError as error:
        raise ValueError("rescore selection counts must be integers") from error
    if rotations <= 0 or rotations % SOURCE_ROTATION_BLOCK_SIZE:
        raise ValueError("rescore selection requires complete source-16 rotation blocks")

    block_ids = np.asarray(selection.block_ids)
    count_fields = tuple(
        np.asarray(field)
        for field in (
            selection.block_count,
            selection.posterior_block_count,
            selection.raw_max_block_count,
        )
    )
    if block_ids.dtype != np.dtype(np.int32) or block_ids.ndim != 2:
        raise TypeError("selection block_ids must be a rank-2 int32 array")
    batch_size, capacity = block_ids.shape
    if batch_size <= 0 or capacity <= 0 or actual_images <= 0 or actual_images > batch_size:
        raise ValueError("rescore selection has invalid image or capacity counts")
    if any(field.dtype != np.dtype(np.int32) or field.shape != (batch_size,) for field in count_fields):
        raise TypeError("selection count fields must be matching rank-1 int32 arrays")
    if not isinstance(selection.eligible, (bool, np.bool_)) or not selection.eligible:
        raise ValueError("rescore assembly requires an eligible block selection")
    if selection.fallback_reason is not None:
        raise ValueError("eligible rescore selection must not carry a fallback reason")

    block_count, posterior_count, raw_count = count_fields
    n_source_blocks = rotations // SOURCE_ROTATION_BLOCK_SIZE
    for row in range(batch_size):
        count = int(block_count[row])
        posterior = int(posterior_count[row])
        raw = int(raw_count[row])
        if row >= actual_images:
            if count != 0 or posterior != 0 or raw != 0 or np.any(block_ids[row] != -1):
                raise ValueError("padded image rows must contain only zero counts and -1 IDs")
            continue
        if (
            count <= 0
            or count > capacity
            or posterior <= 0
            or raw <= 0
            or posterior > count
            or raw > count
            or count < max(posterior, raw)
            or count > posterior + raw
        ):
            raise ValueError("active image row has inconsistent selected block counts")
        active_ids = block_ids[row, :count]
        if (
            np.any(active_ids < 0)
            or np.any(active_ids >= n_source_blocks)
            or (active_ids.size > 1 and np.any(np.diff(active_ids) <= 0))
        ):
            raise ValueError(
                "active source-16 block IDs must be unique, in range, and strictly increasing",
            )
        if np.any(block_ids[row, count:] != -1):
            raise ValueError("inactive selected-block slots must use the reserved -1 ID")


@jax.jit
def _assemble_coarse_gemm_hybrid_dense_scores_f32_jit(
    selected_diff2,
    block_ids,
    block_count,
    actual_image_count,
    class_log_prior,
    rotation_log_prior,
    translation_log_prior,
    use_rotation_log_prior,
    use_translation_log_prior,
):
    """Device implementation for exact full-layout score assembly."""

    batch_size, capacity, _source_block_size, n_translations = selected_diff2.shape
    n_rotations = rotation_log_prior.shape[0]
    active_rows = jnp.arange(batch_size, dtype=jnp.int32) < actual_image_count
    active_slots = active_rows[:, None] & (
        jnp.arange(capacity, dtype=jnp.int32)[None, :] < block_count[:, None]
    )
    safe_block_ids = jnp.where(active_slots, block_ids, jnp.int32(0))
    rotation_ids = (
        safe_block_ids[:, :, None] * jnp.int32(SOURCE_ROTATION_BLOCK_SIZE)
        + jnp.arange(SOURCE_ROTATION_BLOCK_SIZE, dtype=jnp.int32)[None, None, :]
    )
    active_candidates = active_slots[:, :, None, None]

    raw_scores = -selected_diff2
    raw_scores_for_reduction = jnp.where(active_candidates, raw_scores, -jnp.inf)
    raw_score_max = jnp.max(raw_scores_for_reduction, axis=(1, 2, 3))

    posterior_scores = raw_scores + class_log_prior
    posterior_scores = jax.lax.cond(
        use_rotation_log_prior,
        lambda values: values + rotation_log_prior[rotation_ids][..., None],
        lambda values: values,
        posterior_scores,
    )
    posterior_scores = jax.lax.cond(
        use_translation_log_prior,
        lambda values: values + translation_log_prior[:, None, None, :],
        lambda values: values,
        posterior_scores,
    )

    valid_active_diff2 = jnp.all(
        jnp.where(
            active_candidates,
            jnp.isfinite(selected_diff2) & (selected_diff2 >= jnp.float32(0.0)),
            True,
        ),
        axis=(1, 2, 3),
    )
    positive_inf_padding = jnp.all(
        jnp.where(active_candidates, True, jnp.isposinf(selected_diff2)),
        axis=(1, 2, 3),
    )
    finite_active_posterior = jnp.all(
        jnp.where(active_candidates, jnp.isfinite(posterior_scores), True),
        axis=(1, 2, 3),
    )
    selected_output_valid = (
        valid_active_diff2
        & positive_inf_padding
        & finite_active_posterior
        & jnp.where(active_rows, jnp.isfinite(raw_score_max), True)
    )

    # Invalid active values are never published.  The host caller must inspect
    # ``selected_output_valid`` and use full rectangular direct scoring for the
    # whole padded image batch when any row is false.
    scatter_values = jnp.where(
        active_candidates & jnp.isfinite(posterior_scores),
        posterior_scores,
        -jnp.inf,
    )
    dense_scores = jnp.full(
        (batch_size, n_rotations, n_translations),
        -jnp.inf,
        dtype=jnp.float32,
    )
    batch_ids = jnp.broadcast_to(
        jnp.arange(batch_size, dtype=jnp.int32)[:, None, None],
        rotation_ids.shape,
    )
    # Host validation rejects duplicate active source blocks.  ``max`` makes
    # the many inactive slots targeting the safe block-zero index true no-ops.
    dense_scores = dense_scores.at[batch_ids, rotation_ids, :].max(scatter_values)
    posterior_scores_flat = dense_scores.reshape(batch_size, n_rotations * n_translations)
    best_score = jnp.max(posterior_scores_flat, axis=1)
    best_pose = jnp.argmax(posterior_scores_flat, axis=1).astype(jnp.int32)
    min_diff2_offsets = jnp.where(active_rows, -raw_score_max, jnp.float32(0.0))
    return CoarseGemmHybridDenseScores(
        posterior_scores_flat=posterior_scores_flat,
        raw_score_max=raw_score_max,
        min_diff2_offsets=min_diff2_offsets,
        best_score=best_score,
        best_pose=best_pose,
        selected_output_valid=selected_output_valid,
    )


def assemble_coarse_gemm_hybrid_dense_scores_f32(
    selected_diff2,
    selection: CoarseGemmHybridBlockSelection,
    *,
    actual_image_count: int,
    n_rotations: int,
    class_log_prior,
    rotation_log_prior=None,
    translation_log_prior=None,
) -> CoarseGemmHybridDenseScores:
    """Restore selected exact diff2 values to RELION's full flat pose table.

    The selected CUDA output is ``[B,Q,16,T]`` in selected-slot order, while
    downstream support IDs and exact ties use global ``rotation*T+translation``
    order.  This helper validates the selector's ordered source-block contract,
    scatters by the explicit source IDs, applies priors in the production
    class/rotation/translation float32 sequence, and reconstructs the raw-score
    maximum needed for RELION's ``min_diff2`` offset.
    """

    validate_coarse_gemm_hybrid_block_selection_for_rescore(
        selection,
        actual_image_count=actual_image_count,
        n_rotations=n_rotations,
    )
    diff2 = jnp.asarray(selected_diff2)
    if diff2.dtype != jnp.float32:
        raise TypeError("selected source-16 diff2 values must have float32 dtype")
    if diff2.ndim != 4:
        raise ValueError("selected source-16 diff2 values must have shape [B,Q,16,T]")
    batch_size, capacity, source_block_size, n_translations = map(int, diff2.shape)
    if (
        source_block_size != SOURCE_ROTATION_BLOCK_SIZE
        or n_translations <= 0
        or n_translations > 128
        or tuple(selection.block_ids.shape) != (batch_size, capacity)
    ):
        raise ValueError("selected diff2 shape does not match the block selection")

    class_prior = jnp.asarray(class_log_prior, dtype=jnp.float32)
    if class_prior.shape != ():
        raise ValueError("class_log_prior must be scalar")
    if rotation_log_prior is None:
        rotation_prior = jnp.zeros((n_rotations,), dtype=jnp.float32)
        use_rotation_prior = jnp.bool_(False)
    else:
        rotation_prior = jnp.asarray(rotation_log_prior, dtype=jnp.float32)
        if rotation_prior.shape != (n_rotations,):
            raise ValueError("rotation_log_prior must have one value per source rotation")
        use_rotation_prior = jnp.bool_(True)
    if translation_log_prior is None:
        translation_prior = jnp.zeros((batch_size, n_translations), dtype=jnp.float32)
        use_translation_prior = jnp.bool_(False)
    else:
        translation_prior = jnp.asarray(translation_log_prior, dtype=jnp.float32)
        if translation_prior.shape == (n_translations,):
            translation_prior = jnp.broadcast_to(
                translation_prior[None, :],
                (batch_size, n_translations),
            )
        elif translation_prior.shape != (batch_size, n_translations):
            raise ValueError(
                "translation_log_prior must have shape [T] or [B,T]",
            )
        use_translation_prior = jnp.bool_(True)

    return _assemble_coarse_gemm_hybrid_dense_scores_f32_jit(
        diff2,
        jnp.asarray(selection.block_ids, dtype=jnp.int32),
        jnp.asarray(selection.block_count, dtype=jnp.int32),
        jnp.asarray(actual_image_count, dtype=jnp.int32),
        class_prior,
        rotation_prior,
        translation_prior,
        use_rotation_prior,
        use_translation_prior,
    )
