"""Bounded streaming diagnostics for coarse direct-square/GEMM scoring.

The expanded coarse GEMM can score more than a billion GF46 candidates.  A
paired score-cube capture is therefore the wrong diagnostic boundary.  This
module keeps two small device-resident tables instead: the highest candidates
ranked by the direct scorer and the highest candidates ranked by the GEMM
scorer.  Each table carries the paired score and global candidate id.  Online
reductions cover every finite score pair, so the final host summary can report
global error bounds, winner margins, posterior-cutoff errors, exact production
GEMM support differences, and conservative exact-rescore supersets.

The path is diagnostic-only.  It does not select production candidates or
change score, posterior, or support arithmetic.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_RESCORE_BANDS = (
    0.0,
    1.0e-4,
    2.5e-4,
    5.0e-4,
    1.0e-3,
    2.0e-3,
    5.0e-3,
    1.0e-2,
    2.0e-2,
    5.0e-2,
    1.0e-1,
    2.5e-1,
    5.0e-1,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    80.0,
    96.0,
    112.0,
    128.0,
    136.0,
    138.0,
    140.0,
    144.0,
    160.0,
)

# RELION shifts the best coarse score to 50 before exponentiation and its CUDA
# exponent helper returns zero only below -88.  A direct-score candidate can
# therefore have nonzero weight down to 138 score units below the direct
# maximum.  If direct and macro scores differ by at most E on a complete
# streamed surface, macro candidates within 138 + 2E of the macro maximum are
# a conservative superset of every candidate with nonzero direct weight.
RELION_COARSE_NONZERO_SCORE_SPAN = 138.0
DEFAULT_SOURCE_ROTATION_BLOCK_SIZE = 16
DEFAULT_ROTATION_BLOCK_CAPACITY = 64


class CoarseGemmStreamingState(NamedTuple):
    """Device-resident dual-top-k and all-candidate reduction state.

    The ranked tables contain ``retained_topk + 1`` columns.  The final column
    is a boundary sentinel used to prove that cutoff ties and score bands did
    not overflow the retained table.
    """

    direct_scores: jax.Array
    direct_paired_macro_scores: jax.Array
    direct_candidate_ids: jax.Array
    macro_scores: jax.Array
    macro_paired_direct_scores: jax.Array
    macro_candidate_ids: jax.Array
    direct_logsumexp_max: jax.Array
    direct_logsumexp_sum: jax.Array
    max_abs_delta: jax.Array
    signed_delta_sum: jax.Array
    squared_delta_sum: jax.Array
    finite_pair_count: jax.Array
    nonfinite_pair_count: jax.Array


def coarse_gemm_streaming_state_bytes(
    batch_size: int,
    retained_topk: int,
    *,
    score_dtype=np.float32,
) -> int:
    """Return persistent state bytes, including one overflow sentinel slot."""

    batch_size = int(batch_size)
    retained_topk = int(retained_topk)
    if batch_size <= 0 or retained_topk <= 0:
        raise ValueError("batch_size and retained_topk must be positive")
    slots = retained_topk + 1
    score_bytes = np.dtype(score_dtype).itemsize
    # Four score tables and two int32 candidate-id tables.
    ranked_bytes = batch_size * slots * (4 * score_bytes + 2 * np.dtype(np.int32).itemsize)
    # One score-typed maximum, four float64 accumulators, and two int64 counts.
    scalar_bytes = batch_size * (score_bytes + 4 * np.dtype(np.float64).itemsize + 2 * np.dtype(np.int64).itemsize)
    return int(ranked_bytes + scalar_bytes)


def initialize_coarse_gemm_streaming_state(
    batch_size: int,
    retained_topk: int,
    *,
    score_dtype=jnp.float32,
) -> CoarseGemmStreamingState:
    """Create an empty dual-top-k state for one fixed-capacity image batch."""

    batch_size = int(batch_size)
    retained_topk = int(retained_topk)
    if batch_size <= 0 or retained_topk <= 0:
        raise ValueError("batch_size and retained_topk must be positive")
    slots = retained_topk + 1
    score_dtype = jnp.dtype(score_dtype)
    scores = jnp.full((batch_size, slots), -jnp.inf, dtype=score_dtype)
    candidate_ids = jnp.full((batch_size, slots), -1, dtype=jnp.int32)
    # RECOVAR normally leaves JAX x64 disabled.  This diagnostic deliberately
    # scopes x64 to its reductions so the stored dtype matches the advertised
    # analytic-float64 support/error semantics without changing global config.
    with jax.enable_x64(True):
        return CoarseGemmStreamingState(
            direct_scores=scores,
            direct_paired_macro_scores=scores,
            direct_candidate_ids=candidate_ids,
            macro_scores=scores,
            macro_paired_direct_scores=scores,
            macro_candidate_ids=candidate_ids,
            direct_logsumexp_max=jnp.full(
                (batch_size,),
                -jnp.inf,
                dtype=score_dtype,
            ),
            direct_logsumexp_sum=jnp.zeros((batch_size,), dtype=jnp.float64),
            max_abs_delta=jnp.zeros((batch_size,), dtype=jnp.float64),
            signed_delta_sum=jnp.zeros((batch_size,), dtype=jnp.float64),
            squared_delta_sum=jnp.zeros((batch_size,), dtype=jnp.float64),
            finite_pair_count=jnp.zeros((batch_size,), dtype=jnp.int64),
            nonfinite_pair_count=jnp.zeros((batch_size,), dtype=jnp.int64),
        )


def _merge_ranked_candidates(
    retained_scores,
    retained_paired_scores,
    retained_candidate_ids,
    block_scores,
    block_paired_scores,
    block_candidate_ids,
):
    """Merge one flat score block into a fixed-capacity ranked table."""

    capacity = int(retained_scores.shape[1])
    combined_scores = jnp.concatenate((retained_scores, block_scores), axis=1)
    combined_paired = jnp.concatenate(
        (retained_paired_scores, block_paired_scores),
        axis=1,
    )
    combined_ids = jnp.concatenate((retained_candidate_ids, block_candidate_ids), axis=1)
    ranked_scores, positions = jax.lax.top_k(combined_scores, capacity)
    return (
        ranked_scores,
        jnp.take_along_axis(combined_paired, positions, axis=1),
        jnp.take_along_axis(combined_ids, positions, axis=1),
    )


def _update_online_logsumexp(previous_max, previous_sum, scores, finite):
    """Update an exact-math float64 log-sum-exp reduction from one block."""

    block_max = jnp.max(jnp.where(finite, scores, -jnp.inf), axis=1)
    combined_max = jnp.maximum(previous_max, block_max)
    has_combined = jnp.isfinite(combined_max)
    safe_max = jnp.where(has_combined, combined_max, jnp.zeros_like(combined_max))
    previous_scale = jnp.where(
        jnp.isfinite(previous_max),
        jnp.exp(
            previous_max.astype(jnp.float64) - safe_max.astype(jnp.float64),
        ),
        jnp.float64(0.0),
    )
    block_sum = jnp.sum(
        jnp.where(
            finite,
            jnp.exp(scores.astype(jnp.float64) - safe_max[:, None].astype(jnp.float64)),
            jnp.float64(0.0),
        ),
        axis=1,
        dtype=jnp.float64,
    )
    return (
        jnp.where(has_combined, combined_max, -jnp.inf),
        previous_sum * previous_scale + block_sum,
    )


@jax.jit
def _update_coarse_gemm_streaming_state(
    state: CoarseGemmStreamingState,
    direct_scores,
    macro_scores,
    *,
    candidate_offset: int,
    actual_image_count,
) -> CoarseGemmStreamingState:
    """Consume one paired ``[image, ...candidate]`` score block on device.

    ``candidate_offset`` is the flattened global class/rotation/translation id
    of the block's first candidate.  Padded rotations should already carry
    ``-inf`` in both score arrays; padded images are excluded with
    ``actual_image_count``.
    """

    direct = jnp.asarray(direct_scores)
    macro = jnp.asarray(macro_scores)
    if direct.shape != macro.shape or direct.ndim < 2:
        raise ValueError(
            "streaming coarse GEMM diagnostics require equal [image, candidate...] "
            f"arrays, got {direct.shape} and {macro.shape}",
        )
    if direct.dtype != macro.dtype or direct.dtype != state.direct_scores.dtype:
        raise TypeError(
            "streaming coarse GEMM score dtypes must match the state, got "
            f"{direct.dtype}, {macro.dtype}, and {state.direct_scores.dtype}",
        )
    batch_size = int(direct.shape[0])
    if batch_size != int(state.direct_scores.shape[0]):
        raise ValueError(
            f"streaming coarse GEMM batch size differs from the state: {batch_size} != {state.direct_scores.shape[0]}",
        )
    direct = direct.reshape(batch_size, -1)
    macro = macro.reshape(batch_size, -1)
    active = jnp.arange(batch_size, dtype=jnp.int32) < jnp.asarray(
        actual_image_count,
        dtype=jnp.int32,
    )
    candidate_present = active[:, None] & ~(jnp.isneginf(direct) & jnp.isneginf(macro))
    finite = candidate_present & jnp.isfinite(direct) & jnp.isfinite(macro)
    ranked_direct = jnp.where(finite, direct, -jnp.inf)
    ranked_macro = jnp.where(finite, macro, -jnp.inf)
    block_ids = jnp.broadcast_to(
        (jnp.arange(int(direct.shape[1]), dtype=jnp.int32) + jnp.asarray(candidate_offset, dtype=jnp.int32))[None, :],
        direct.shape,
    )
    block_ids = jnp.where(finite, block_ids, jnp.int32(-1))

    direct_ranked = _merge_ranked_candidates(
        state.direct_scores,
        state.direct_paired_macro_scores,
        state.direct_candidate_ids,
        ranked_direct,
        ranked_macro,
        block_ids,
    )
    macro_ranked = _merge_ranked_candidates(
        state.macro_scores,
        state.macro_paired_direct_scores,
        state.macro_candidate_ids,
        ranked_macro,
        ranked_direct,
        block_ids,
    )
    direct_max, direct_sum = _update_online_logsumexp(
        state.direct_logsumexp_max,
        state.direct_logsumexp_sum,
        direct,
        finite,
    )
    delta = jnp.where(
        finite,
        macro.astype(jnp.float64) - direct.astype(jnp.float64),
        jnp.float64(0.0),
    )
    finite_count = jnp.sum(finite, axis=1, dtype=jnp.int64)
    active_candidate_count = jnp.sum(candidate_present, axis=1, dtype=jnp.int64)
    return CoarseGemmStreamingState(
        direct_scores=direct_ranked[0],
        direct_paired_macro_scores=direct_ranked[1],
        direct_candidate_ids=direct_ranked[2],
        macro_scores=macro_ranked[0],
        macro_paired_direct_scores=macro_ranked[1],
        macro_candidate_ids=macro_ranked[2],
        direct_logsumexp_max=direct_max,
        direct_logsumexp_sum=direct_sum,
        max_abs_delta=jnp.maximum(
            state.max_abs_delta,
            jnp.max(jnp.abs(delta), axis=1),
        ),
        signed_delta_sum=state.signed_delta_sum + jnp.sum(delta, axis=1, dtype=jnp.float64),
        squared_delta_sum=state.squared_delta_sum + jnp.sum(delta * delta, axis=1, dtype=jnp.float64),
        finite_pair_count=state.finite_pair_count + finite_count,
        nonfinite_pair_count=state.nonfinite_pair_count + active_candidate_count - finite_count,
    )


def update_coarse_gemm_streaming_state(
    state: CoarseGemmStreamingState,
    direct_scores,
    macro_scores,
    *,
    candidate_offset: int,
    actual_image_count,
) -> CoarseGemmStreamingState:
    """Consume a paired block with diagnostic-local float64 reductions."""

    with jax.enable_x64(True):
        return _update_coarse_gemm_streaming_state(
            state,
            direct_scores,
            macro_scores,
            candidate_offset=candidate_offset,
            actual_image_count=actual_image_count,
        )


def _analytic_support_from_ranked(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    log_z: float,
    adaptive_fraction: float,
    max_significants: int,
) -> tuple[np.ndarray, float, bool, int, float]:
    """Derive direct support and a top-k/tie coverage certificate."""

    retained_topk = scores.size - 1
    retained_scores = scores[:retained_topk]
    retained_ids = candidate_ids[:retained_topk]
    valid = np.isfinite(retained_scores) & (retained_ids >= 0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0 or not np.isfinite(log_z):
        return np.empty((0,), dtype=np.int32), math_nan(), False, 0, math_nan()

    retained_scores = retained_scores[:valid_count]
    retained_ids = retained_ids[:valid_count]
    probabilities = np.exp(retained_scores.astype(np.float64) - float(log_z))
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    crosses = np.flatnonzero(cumulative > float(adaptive_fraction))
    crossed = bool(crosses.size)
    cutoff_index = int(crosses[0]) if crossed else valid_count - 1
    if int(max_significants) > 0:
        cutoff_index = min(cutoff_index, int(max_significants) - 1)
    cutoff_score = float(retained_scores[cutoff_index])
    selected = retained_scores >= cutoff_score
    selected_ids = retained_ids[selected].astype(np.int32, copy=False)

    sentinel_score = float(scores[retained_topk])
    tie_complete = not np.isfinite(sentinel_score) or sentinel_score < cutoff_score
    cap_proves_cutoff = int(max_significants) > 0 and int(max_significants) <= retained_topk
    posterior_proves_cutoff = crossed
    all_candidates_retained = not np.isfinite(sentinel_score)
    coverage = bool(tie_complete and (cap_proves_cutoff or posterior_proves_cutoff or all_candidates_retained))
    cutoff_excess = float(cumulative[cutoff_index] - float(adaptive_fraction))
    return selected_ids, cutoff_score, coverage, cutoff_index + 1, cutoff_excess


def math_nan() -> float:
    """A named NaN keeps fail-closed summary branches easy to audit."""

    return float("nan")


def _ranked_lookup(
    candidate_ids: np.ndarray,
    values: np.ndarray,
    requested_ids: np.ndarray,
) -> tuple[np.ndarray, bool]:
    positions = {
        int(candidate_id): position for position, candidate_id in enumerate(candidate_ids) if int(candidate_id) >= 0
    }
    missing = [int(candidate_id) for candidate_id in requested_ids if int(candidate_id) not in positions]
    if missing:
        return np.full(requested_ids.shape, np.nan, dtype=np.float64), False
    return (
        np.asarray([values[positions[int(candidate_id)]] for candidate_id in requested_ids], dtype=np.float64),
        True,
    )


def _threshold_count_with_coverage(
    ranked_scores: np.ndarray,
    threshold: float,
) -> tuple[int, bool]:
    retained_topk = ranked_scores.size - 1
    retained = ranked_scores[:retained_topk]
    count = int(np.count_nonzero(retained >= float(threshold)))
    sentinel = float(ranked_scores[retained_topk])
    coverage = bool(not np.isfinite(sentinel) or sentinel < float(threshold))
    return count, coverage


def _source_rotation_block_ids(
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    n_rotations: int,
    n_translations: int,
    source_rotation_block_size: int,
) -> np.ndarray:
    """Map selected flattened candidates to deterministic class/block ids."""

    selected = (candidate_ids >= 0) & np.isfinite(scores) & (scores >= float(threshold))
    ids = candidate_ids[selected].astype(np.int64, copy=False)
    if ids.size == 0:
        return np.empty((0,), dtype=np.int32)
    candidates_per_class = int(n_rotations) * int(n_translations)
    blocks_per_class = (int(n_rotations) + int(source_rotation_block_size) - 1) // int(source_rotation_block_size)
    class_ids = ids // candidates_per_class
    rotation_ids = (ids % candidates_per_class) // int(n_translations)
    block_ids = class_ids * blocks_per_class + rotation_ids // int(source_rotation_block_size)
    return np.unique(block_ids).astype(np.int32, copy=False)


def summarize_coarse_gemm_streaming_state(
    state: CoarseGemmStreamingState,
    macro_support_mask,
    *,
    actual_image_count: int,
    adaptive_fraction: float,
    max_significants: int,
    band_widths=DEFAULT_RESCORE_BANDS,
    n_rotations: int | None = None,
    n_translations: int | None = None,
    source_rotation_block_size: int = DEFAULT_SOURCE_ROTATION_BLOCK_SIZE,
    rotation_block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
) -> dict[str, np.ndarray]:
    """Finalize one batch without materializing either paired score cube.

    ``macro_support_mask`` is the exact production support already computed by
    the GEMM arm.  Direct support is reconstructed from the streamed direct
    log-sum-exp and top-k table.  Every derived comparison carries an explicit
    coverage certificate; a truncated cutoff tie is never reported as exact.
    """

    actual_image_count = int(actual_image_count)
    state_batch_size = int(state.direct_scores.shape[0])
    if actual_image_count <= 0 or actual_image_count > state_batch_size:
        raise ValueError(
            "actual_image_count must be positive and no larger than the state "
            f"batch: {actual_image_count} vs {state_batch_size}",
        )
    direct_scores = np.asarray(state.direct_scores)[:actual_image_count]
    direct_paired_macro = np.asarray(state.direct_paired_macro_scores)[:actual_image_count]
    direct_ids = np.asarray(state.direct_candidate_ids, dtype=np.int32)[:actual_image_count]
    macro_scores = np.asarray(state.macro_scores)[:actual_image_count]
    macro_paired_direct = np.asarray(state.macro_paired_direct_scores)[:actual_image_count]
    macro_ids = np.asarray(state.macro_candidate_ids, dtype=np.int32)[:actual_image_count]
    direct_max = np.asarray(state.direct_logsumexp_max, dtype=np.float64)[:actual_image_count]
    direct_sum = np.asarray(state.direct_logsumexp_sum, dtype=np.float64)[:actual_image_count]
    with np.errstate(divide="ignore", invalid="ignore"):
        direct_log_z = direct_max + np.log(direct_sum)
    macro_support = np.asarray(macro_support_mask, dtype=bool)[:actual_image_count]
    if macro_support.ndim != 2 or macro_support.shape[0] != actual_image_count:
        raise ValueError(
            f"macro_support_mask must have shape [image, flattened candidate], got {macro_support.shape}",
        )
    bands = np.asarray(tuple(float(value) for value in band_widths), dtype=np.float64)
    if bands.ndim != 1 or bands.size == 0 or np.any(~np.isfinite(bands)) or np.any(bands < 0.0):
        raise ValueError("band_widths must be a non-empty sequence of finite non-negative values")
    if np.any(np.diff(bands) < 0.0):
        raise ValueError("band_widths must be sorted in non-decreasing order")
    rotation_layout_available = n_rotations is not None or n_translations is not None
    if rotation_layout_available:
        if n_rotations is None or n_translations is None:
            raise ValueError(
                "n_rotations and n_translations must be provided together",
            )
        n_rotations = int(n_rotations)
        n_translations = int(n_translations)
        source_rotation_block_size = int(source_rotation_block_size)
        rotation_block_capacity = int(rotation_block_capacity)
        if (
            min(
                n_rotations,
                n_translations,
                source_rotation_block_size,
                rotation_block_capacity,
            )
            <= 0
        ):
            raise ValueError("rotation-block diagnostic dimensions must be positive")
        candidates_per_class = n_rotations * n_translations
        if macro_support.shape[1] % candidates_per_class != 0:
            raise ValueError(
                "macro support candidate count is not divisible by the supplied "
                f"rotation/translation layout: {macro_support.shape[1]} vs "
                f"{n_rotations} * {n_translations}",
            )
    else:
        n_rotations = -1
        n_translations = -1
        source_rotation_block_size = int(source_rotation_block_size)
        rotation_block_capacity = int(rotation_block_capacity)
        if source_rotation_block_size <= 0 or rotation_block_capacity <= 0:
            raise ValueError("rotation-block diagnostic dimensions must be positive")

    scalar_float_names = (
        "direct_cutoff_score",
        "macro_cutoff_score",
        "direct_cutoff_excess_mass",
        "direct_winner_margin",
        "macro_winner_margin",
        "winner_observed_minimum_band",
        "winner_global_error_safe_band",
        "support_observed_minimum_band",
        "support_global_error_safe_band",
        "relion_nonzero_surface_error_safe_width_from_macro_max",
    )
    scalar_int_names = (
        "direct_cutoff_rank",
        "direct_support_count",
        "macro_support_count",
        "support_false_negative_count",
        "support_false_positive_count",
        "direct_winner_candidate_id",
        "macro_winner_candidate_id",
        "winner_observed_superset_count",
        "winner_global_error_superset_count",
        "support_observed_superset_count",
        "support_global_error_superset_count",
        "relion_nonzero_surface_error_safe_superset_count",
        "relion_nonzero_surface_error_safe_source_rotation_block_count",
        "relion_nonzero_surface_error_safe_source_rotation_block_sentinel_id",
    )
    scalar_bool_names = (
        "direct_support_coverage",
        "macro_support_ranked_coverage",
        "support_comparison_coverage",
        "direct_winner_unique",
        "macro_winner_unique",
        "winner_comparison_coverage",
        "winner_equal",
        "winner_observed_superset_coverage",
        "winner_global_error_superset_coverage",
        "support_observed_superset_coverage",
        "support_global_error_superset_coverage",
        "all_candidate_error_coverage",
        "relion_nonzero_surface_error_safe_superset_coverage",
        "relion_nonzero_surface_error_safe_source_rotation_block_overflow",
        "relion_nonzero_surface_error_safe_source_rotation_block_list_coverage",
    )
    floats = {name: np.full(actual_image_count, np.nan, dtype=np.float64) for name in scalar_float_names}
    integers = {name: np.full(actual_image_count, -1, dtype=np.int64) for name in scalar_int_names}
    booleans = {name: np.zeros(actual_image_count, dtype=bool) for name in scalar_bool_names}
    near_count = np.zeros((actual_image_count, bands.size), dtype=np.int64)
    near_max_abs = np.full((actual_image_count, bands.size), np.nan, dtype=np.float64)
    near_rms = np.full((actual_image_count, bands.size), np.nan, dtype=np.float64)
    near_coverage = np.zeros((actual_image_count, bands.size), dtype=bool)
    band_superset_count = np.full((actual_image_count, bands.size), -1, dtype=np.int64)
    band_superset_coverage = np.zeros((actual_image_count, bands.size), dtype=bool)
    macro_max_band_superset_count = np.full(
        (actual_image_count, bands.size),
        -1,
        dtype=np.int64,
    )
    macro_max_band_superset_coverage = np.zeros(
        (actual_image_count, bands.size),
        dtype=bool,
    )
    macro_max_band_source_rotation_block_count = np.full(
        (actual_image_count, bands.size),
        -1,
        dtype=np.int64,
    )
    macro_max_band_source_rotation_block_count_coverage = np.zeros(
        (actual_image_count, bands.size),
        dtype=bool,
    )
    safe_source_rotation_block_ids = np.full(
        (actual_image_count, rotation_block_capacity),
        -1,
        dtype=np.int32,
    )

    max_abs_delta = np.asarray(state.max_abs_delta, dtype=np.float64)[:actual_image_count]
    finite_count = np.asarray(state.finite_pair_count, dtype=np.int64)[:actual_image_count]
    nonfinite_count = np.asarray(state.nonfinite_pair_count, dtype=np.int64)[:actual_image_count]
    expected_candidate_count = int(macro_support.shape[1])
    for row in range(actual_image_count):
        direct_support_ids, direct_cutoff, direct_coverage, cutoff_rank, cutoff_excess = _analytic_support_from_ranked(
            direct_scores[row],
            direct_ids[row],
            log_z=float(direct_log_z[row]),
            adaptive_fraction=float(adaptive_fraction),
            max_significants=int(max_significants),
        )
        macro_support_ids = np.flatnonzero(macro_support[row]).astype(np.int32)
        macro_support_scores, macro_support_ids_retained = _ranked_lookup(
            macro_ids[row, :-1],
            macro_scores[row, :-1],
            macro_support_ids,
        )
        direct_support_macro_scores, direct_pair_coverage = _ranked_lookup(
            direct_ids[row, :-1],
            direct_paired_macro[row, :-1],
            direct_support_ids,
        )

        floats["direct_cutoff_score"][row] = direct_cutoff
        floats["direct_cutoff_excess_mass"][row] = cutoff_excess
        integers["direct_cutoff_rank"][row] = cutoff_rank
        integers["direct_support_count"][row] = direct_support_ids.size
        integers["macro_support_count"][row] = macro_support_ids.size
        error_coverage = bool(finite_count[row] == expected_candidate_count and nonfinite_count[row] == 0)
        booleans["all_candidate_error_coverage"][row] = error_coverage
        direct_coverage = bool(direct_coverage and error_coverage)
        booleans["direct_support_coverage"][row] = direct_coverage

        direct_valid = (direct_ids[row, :-1] >= 0) & np.isfinite(direct_scores[row, :-1])
        macro_valid = (macro_ids[row, :-1] >= 0) & np.isfinite(macro_scores[row, :-1])
        direct_valid_count = int(np.count_nonzero(direct_valid))
        macro_valid_count = int(np.count_nonzero(macro_valid))
        direct_winner_unique = bool(
            np.isfinite(direct_scores[row, 0]) and direct_scores[row, 1] < direct_scores[row, 0]
        )
        macro_winner_unique = bool(np.isfinite(macro_scores[row, 0]) and macro_scores[row, 1] < macro_scores[row, 0])
        booleans["direct_winner_unique"][row] = direct_winner_unique
        booleans["macro_winner_unique"][row] = macro_winner_unique
        winner_coverage = bool(error_coverage and direct_winner_unique and macro_winner_unique)
        booleans["winner_comparison_coverage"][row] = winner_coverage
        if direct_valid_count:
            integers["direct_winner_candidate_id"][row] = int(direct_ids[row, 0])
            floats["direct_winner_margin"][row] = (
                math_nan() if direct_valid_count < 2 else float(direct_scores[row, 0] - direct_scores[row, 1])
            )
        if macro_valid_count:
            integers["macro_winner_candidate_id"][row] = int(macro_ids[row, 0])
            floats["macro_winner_margin"][row] = (
                math_nan() if macro_valid_count < 2 else float(macro_scores[row, 0] - macro_scores[row, 1])
            )
        if winner_coverage:
            booleans["winner_equal"][row] = bool(direct_ids[row, 0] == macro_ids[row, 0])
            winner_observed_band = max(
                0.0,
                float(macro_scores[row, 0] - direct_paired_macro[row, 0]),
            )
            winner_global_band = 2.0 * float(max_abs_delta[row])
            floats["winner_observed_minimum_band"][row] = winner_observed_band
            floats["winner_global_error_safe_band"][row] = winner_global_band
            for prefix, width in (
                ("winner_observed", winner_observed_band),
                ("winner_global_error", winner_global_band),
            ):
                count, coverage = _threshold_count_with_coverage(
                    macro_scores[row],
                    float(macro_scores[row, 0]) - width,
                )
                integers[f"{prefix}_superset_count"][row] = count
                booleans[f"{prefix}_superset_coverage"][row] = coverage

        macro_ranked_coverage = False
        if macro_support_ids_retained and macro_support_scores.size:
            macro_cutoff = float(np.min(macro_support_scores))
            sentinel_score = float(macro_scores[row, -1])
            macro_ranked_coverage = bool(
                error_coverage and (not np.isfinite(sentinel_score) or sentinel_score < macro_cutoff)
            )
            floats["macro_cutoff_score"][row] = macro_cutoff
        booleans["macro_support_ranked_coverage"][row] = macro_ranked_coverage
        comparison_coverage = bool(
            error_coverage and direct_coverage and macro_ranked_coverage and direct_pair_coverage
        )
        booleans["support_comparison_coverage"][row] = comparison_coverage
        if comparison_coverage:
            direct_set = set(int(value) for value in direct_support_ids)
            macro_set = set(int(value) for value in macro_support_ids)
            integers["support_false_negative_count"][row] = len(direct_set - macro_set)
            integers["support_false_positive_count"][row] = len(macro_set - direct_set)
        if macro_ranked_coverage:
            if comparison_coverage and direct_support_macro_scores.size:
                observed_band = max(
                    0.0,
                    float(macro_cutoff - np.min(direct_support_macro_scores)),
                )
                global_band = max(
                    0.0,
                    float(macro_cutoff - direct_cutoff + max_abs_delta[row]),
                )
                floats["support_observed_minimum_band"][row] = observed_band
                floats["support_global_error_safe_band"][row] = global_band
                for prefix, width in (
                    ("support_observed", observed_band),
                    ("support_global_error", global_band),
                ):
                    count, coverage = _threshold_count_with_coverage(
                        macro_scores[row],
                        macro_cutoff - width,
                    )
                    integers[f"{prefix}_superset_count"][row] = count
                    booleans[f"{prefix}_superset_coverage"][row] = coverage

            paired_delta = np.zeros(macro_valid.shape, dtype=np.float64)
            paired_delta[macro_valid] = macro_scores[row, :-1][macro_valid].astype(np.float64) - macro_paired_direct[
                row, :-1
            ][macro_valid].astype(np.float64)
            for band_index, width in enumerate(bands):
                lower = macro_cutoff - float(width)
                upper = macro_cutoff + float(width)
                selected = macro_valid & (macro_scores[row, :-1] >= lower) & (macro_scores[row, :-1] <= upper)
                values = paired_delta[selected]
                near_count[row, band_index] = values.size
                if values.size:
                    near_max_abs[row, band_index] = float(np.max(np.abs(values)))
                    near_rms[row, band_index] = float(np.sqrt(np.mean(values * values)))
                _, coverage = _threshold_count_with_coverage(macro_scores[row], lower)
                near_coverage[row, band_index] = bool(coverage and error_coverage)
                count, coverage = _threshold_count_with_coverage(macro_scores[row], lower)
                band_superset_count[row, band_index] = count
                band_superset_coverage[row, band_index] = bool(
                    coverage and error_coverage,
                )

        if macro_valid_count:
            macro_max = float(macro_scores[row, 0])
            for band_index, width in enumerate(bands):
                count, coverage = _threshold_count_with_coverage(
                    macro_scores[row],
                    macro_max - float(width),
                )
                coverage = bool(coverage and error_coverage)
                macro_max_band_superset_count[row, band_index] = count
                macro_max_band_superset_coverage[row, band_index] = coverage
                if coverage and rotation_layout_available:
                    block_ids = _source_rotation_block_ids(
                        macro_ids[row, :-1],
                        macro_scores[row, :-1],
                        macro_max - float(width),
                        n_rotations=n_rotations,
                        n_translations=n_translations,
                        source_rotation_block_size=source_rotation_block_size,
                    )
                    macro_max_band_source_rotation_block_count[
                        row,
                        band_index,
                    ] = block_ids.size
                    macro_max_band_source_rotation_block_count_coverage[
                        row,
                        band_index,
                    ] = True
            if error_coverage:
                safe_width = RELION_COARSE_NONZERO_SCORE_SPAN + 2.0 * float(
                    max_abs_delta[row],
                )
                floats["relion_nonzero_surface_error_safe_width_from_macro_max"][row] = safe_width
                count, coverage = _threshold_count_with_coverage(
                    macro_scores[row],
                    macro_max - safe_width,
                )
                integers["relion_nonzero_surface_error_safe_superset_count"][row] = count
                booleans["relion_nonzero_surface_error_safe_superset_coverage"][row] = coverage
                if coverage and rotation_layout_available:
                    block_ids = _source_rotation_block_ids(
                        macro_ids[row, :-1],
                        macro_scores[row, :-1],
                        macro_max - safe_width,
                        n_rotations=n_rotations,
                        n_translations=n_translations,
                        source_rotation_block_size=source_rotation_block_size,
                    )
                    integers["relion_nonzero_surface_error_safe_source_rotation_block_count"][row] = block_ids.size
                    copy_count = min(block_ids.size, rotation_block_capacity)
                    safe_source_rotation_block_ids[row, :copy_count] = block_ids[:copy_count]
                    overflow = bool(block_ids.size > rotation_block_capacity)
                    booleans["relion_nonzero_surface_error_safe_source_rotation_block_overflow"][row] = overflow
                    booleans["relion_nonzero_surface_error_safe_source_rotation_block_list_coverage"][
                        row
                    ] = not overflow
                    if overflow:
                        integers["relion_nonzero_surface_error_safe_source_rotation_block_sentinel_id"][row] = int(
                            block_ids[rotation_block_capacity]
                        )

    signed_sum = np.asarray(state.signed_delta_sum, dtype=np.float64)[:actual_image_count]
    squared_sum = np.asarray(state.squared_delta_sum, dtype=np.float64)[:actual_image_count]
    signed_mean = np.full(actual_image_count, np.nan, dtype=np.float64)
    rms = np.full(actual_image_count, np.nan, dtype=np.float64)
    np.divide(signed_sum, finite_count, out=signed_mean, where=finite_count > 0)
    np.sqrt(
        np.divide(
            squared_sum,
            finite_count,
            out=np.full(actual_image_count, np.nan, dtype=np.float64),
            where=finite_count > 0,
        ),
        out=rms,
    )
    payload = {
        "schema": np.asarray("recovar.coarse_gemm_streaming_rescore.v1"),
        "direct_support_semantics": np.asarray("analytic_float64_streamed_logsumexp_topk_not_relion_cub"),
        "macro_support_semantics": np.asarray("exact_production_relion_f32_support_mask"),
        "qualification_status": np.asarray("DIAGNOSTIC_ONLY_NO_PRODUCTION_SELECTION"),
        "winner_equal_semantics": np.asarray(
            "defined_only_when_winner_comparison_coverage_is_true",
        ),
        "surface_error_bound_semantics": np.asarray(
            "complete_observed_streamed_candidate_surface_only_not_unseen_datasets",
        ),
        "production_fallback_requirement": np.asarray(
            "FULL_DIRECT_FALLBACK_UNLESS_ERROR_BOUND_AND_CANDIDATE_SENTINEL_ARE_CERTIFIED",
        ),
        "superset_count_semantics": np.asarray(
            "retained_lower_bound_when_corresponding_coverage_is_false",
        ),
        "source_rotation_block_semantics": np.asarray(
            "global_id_is_class_times_blocks_per_class_plus_original_rotation_div_block_size",
        ),
        "source_rotation_block_list_semantics": np.asarray(
            "first_Q_ids_invalid_minus_one_plus_separate_Q_plus_1_overflow_sentinel",
        ),
        "retained_topk": np.asarray(direct_scores.shape[1] - 1, dtype=np.int64),
        "adaptive_fraction": np.asarray(adaptive_fraction, dtype=np.float64),
        "max_significants": np.asarray(max_significants, dtype=np.int64),
        "relion_coarse_nonzero_score_span": np.asarray(
            RELION_COARSE_NONZERO_SCORE_SPAN,
            dtype=np.float64,
        ),
        "n_rotations": np.asarray(n_rotations, dtype=np.int64),
        "n_translations": np.asarray(n_translations, dtype=np.int64),
        "source_rotation_block_size": np.asarray(
            source_rotation_block_size,
            dtype=np.int64,
        ),
        "source_rotation_block_capacity": np.asarray(
            rotation_block_capacity,
            dtype=np.int64,
        ),
        "band_widths": bands,
        "finite_pair_count": finite_count,
        "nonfinite_pair_count": nonfinite_count,
        "expected_candidate_count_per_image": np.asarray(
            expected_candidate_count,
            dtype=np.int64,
        ),
        "all_candidate_max_abs_delta": max_abs_delta,
        "all_candidate_signed_mean_delta": signed_mean,
        "all_candidate_rms_delta": rms,
        "direct_log_z": direct_log_z,
        "near_macro_cutoff_count": near_count,
        "near_macro_cutoff_max_abs_delta": near_max_abs,
        "near_macro_cutoff_rms_delta": near_rms,
        "near_macro_cutoff_coverage": near_coverage,
        "band_superset_count": band_superset_count,
        "band_superset_coverage": band_superset_coverage,
        "macro_max_band_superset_count": macro_max_band_superset_count,
        "macro_max_band_superset_coverage": macro_max_band_superset_coverage,
        "macro_max_band_source_rotation_block_count": (macro_max_band_source_rotation_block_count),
        "macro_max_band_source_rotation_block_count_coverage": (macro_max_band_source_rotation_block_count_coverage),
        "relion_nonzero_surface_error_safe_source_rotation_block_ids": (safe_source_rotation_block_ids),
    }
    payload.update(floats)
    payload.update(integers)
    payload.update(booleans)
    return payload


def write_coarse_gemm_streaming_summary(
    output_path: str,
    state: CoarseGemmStreamingState,
    macro_support_mask,
    *,
    original_indices,
    local_indices,
    actual_image_count: int,
    padded_image_count: int,
    adaptive_fraction: float,
    max_significants: int,
    diagnostic_run_id: str,
    diagnostic_call_id: str,
    debug_iteration: int | None,
    current_size: int | None,
    band_widths=DEFAULT_RESCORE_BANDS,
    n_rotations: int | None = None,
    n_translations: int | None = None,
    source_rotation_block_size: int = DEFAULT_SOURCE_ROTATION_BLOCK_SIZE,
    rotation_block_capacity: int = DEFAULT_ROTATION_BLOCK_CAPACITY,
) -> None:
    """Write one immutable compact batch summary."""

    if os.path.exists(output_path):
        raise FileExistsError(
            f"refusing to overwrite a coarse GEMM streaming summary: {output_path}",
        )
    actual_image_count = int(actual_image_count)
    padded_image_count = int(padded_image_count)
    if padded_image_count != int(state.direct_scores.shape[0]):
        raise ValueError(
            "padded_image_count must equal the streaming state batch size: "
            f"{padded_image_count} vs {state.direct_scores.shape[0]}",
        )
    payload = summarize_coarse_gemm_streaming_state(
        state,
        macro_support_mask,
        actual_image_count=actual_image_count,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        band_widths=band_widths,
        n_rotations=n_rotations,
        n_translations=n_translations,
        source_rotation_block_size=source_rotation_block_size,
        rotation_block_capacity=rotation_block_capacity,
    )
    original_indices = np.asarray(original_indices, dtype=np.int64)
    local_indices = np.asarray(local_indices, dtype=np.int64)
    if original_indices.shape != (actual_image_count,) or local_indices.shape != (actual_image_count,):
        raise ValueError(
            "streaming summary particle ids must match actual_image_count, got "
            f"{original_indices.shape}, {local_indices.shape}, and {actual_image_count}",
        )
    payload.update(
        diagnostic_run_id=np.asarray(str(diagnostic_run_id)),
        diagnostic_call_id=np.asarray(str(diagnostic_call_id)),
        debug_iteration=np.asarray(-1 if debug_iteration is None else int(debug_iteration), dtype=np.int64),
        current_size=np.asarray(-1 if current_size is None else int(current_size), dtype=np.int64),
        actual_image_count=np.asarray(actual_image_count, dtype=np.int64),
        padded_image_count=np.asarray(padded_image_count, dtype=np.int64),
        original_indices=original_indices,
        local_indices=local_indices,
        paired_capture_active=np.asarray(True),
        clean_timing_eligible=np.asarray(False),
        stores_score_cube=np.asarray(False),
        production_behavior_changed=np.asarray(False),
    )
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _count_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    values = values[values >= 0]
    if values.size == 0:
        return {
            "observations": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "observations": int(values.size),
        "minimum": int(np.min(values)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": int(np.max(values)),
    }


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.max(finite))


def _finite_min(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.min(finite))


def aggregate_coarse_gemm_streaming_summaries(
    artifact_paths,
) -> dict:
    """Build one compact, JSON-safe report across streamed batch artifacts."""

    paths = [os.fspath(path) for path in artifact_paths]
    if not paths:
        raise ValueError("at least one streaming summary artifact is required")
    batches: list[dict[str, np.ndarray]] = []
    expected_bands = None
    retained_topk = None
    rotation_block_capacity = None
    for path in paths:
        with np.load(path, allow_pickle=False) as artifact:
            if artifact["schema"].item() != "recovar.coarse_gemm_streaming_rescore.v1":
                raise ValueError(f"unexpected streaming summary schema: {path}")
            bands = np.asarray(artifact["band_widths"], dtype=np.float64)
            artifact_topk = int(artifact["retained_topk"])
            artifact_block_capacity = int(artifact["source_rotation_block_capacity"])
            if expected_bands is None:
                expected_bands = bands
                retained_topk = artifact_topk
                rotation_block_capacity = artifact_block_capacity
            elif (
                not np.array_equal(bands, expected_bands)
                or artifact_topk != retained_topk
                or artifact_block_capacity != rotation_block_capacity
            ):
                raise ValueError(
                    "streaming summary artifacts disagree on bands or capacities",
                )
            batches.append({key: np.asarray(artifact[key]) for key in artifact.files})

    def joined(name: str) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(batch[name]) for batch in batches],
            axis=0,
        )

    finite_count = joined("finite_pair_count").astype(np.int64, copy=False)
    nonfinite_count = joined("nonfinite_pair_count").astype(np.int64, copy=False)
    signed_mean = joined("all_candidate_signed_mean_delta").astype(
        np.float64,
        copy=False,
    )
    rms = joined("all_candidate_rms_delta").astype(np.float64, copy=False)
    total_finite = int(np.sum(finite_count, dtype=np.int64))
    weighted_signed_mean = None if total_finite == 0 else float(np.nansum(signed_mean * finite_count) / total_finite)
    weighted_rms = (
        None
        if total_finite == 0
        else float(
            np.sqrt(np.nansum(rms * rms * finite_count) / total_finite),
        )
    )

    winner_coverage = joined("winner_comparison_coverage").astype(bool)
    winner_equal = joined("winner_equal").astype(bool)
    support_coverage = joined("support_comparison_coverage").astype(bool)
    support_fn = joined("support_false_negative_count").astype(np.int64)
    support_fp = joined("support_false_positive_count").astype(np.int64)
    safe_pair_coverage = joined(
        "relion_nonzero_surface_error_safe_superset_coverage",
    ).astype(bool)
    safe_pair_count = joined(
        "relion_nonzero_surface_error_safe_superset_count",
    ).astype(np.int64)
    safe_block_coverage = joined(
        "relion_nonzero_surface_error_safe_source_rotation_block_list_coverage",
    ).astype(bool)
    safe_block_count = joined(
        "relion_nonzero_surface_error_safe_source_rotation_block_count",
    ).astype(np.int64)
    safe_block_overflow = joined(
        "relion_nonzero_surface_error_safe_source_rotation_block_overflow",
    ).astype(bool)
    safe_width = joined(
        "relion_nonzero_surface_error_safe_width_from_macro_max",
    ).astype(np.float64)

    band_records = []
    macro_pair_counts = joined("macro_max_band_superset_count").astype(np.int64)
    macro_pair_coverage = joined("macro_max_band_superset_coverage").astype(bool)
    macro_block_counts = joined(
        "macro_max_band_source_rotation_block_count",
    ).astype(np.int64)
    macro_block_coverage = joined(
        "macro_max_band_source_rotation_block_count_coverage",
    ).astype(bool)
    near_count = joined("near_macro_cutoff_count").astype(np.int64)
    near_coverage = joined("near_macro_cutoff_coverage").astype(bool)
    near_max = joined("near_macro_cutoff_max_abs_delta").astype(np.float64)
    near_rms = joined("near_macro_cutoff_rms_delta").astype(np.float64)
    for index, width in enumerate(expected_bands):
        covered_near = near_coverage[:, index]
        covered_near_count = near_count[:, index][covered_near]
        covered_near_rms = near_rms[:, index][covered_near]
        near_total = int(np.sum(covered_near_count, dtype=np.int64))
        band_records.append(
            {
                "width": float(width),
                "macro_max_pair_coverage_count": int(
                    np.count_nonzero(macro_pair_coverage[:, index]),
                ),
                "macro_max_pair_count": _count_distribution(
                    macro_pair_counts[:, index][macro_pair_coverage[:, index]],
                ),
                "source_rotation_block_coverage_count": int(
                    np.count_nonzero(macro_block_coverage[:, index]),
                ),
                "source_rotation_block_count": _count_distribution(
                    macro_block_counts[:, index][macro_block_coverage[:, index]],
                ),
                "near_cutoff_coverage_count": int(np.count_nonzero(covered_near)),
                "near_cutoff_candidate_count": near_total,
                "near_cutoff_max_abs_delta": _finite_max(
                    near_max[:, index][covered_near],
                ),
                "near_cutoff_weighted_rms_delta": (
                    None
                    if near_total == 0
                    else float(
                        np.sqrt(
                            np.nansum(
                                covered_near_rms * covered_near_rms * covered_near_count,
                            )
                            / near_total,
                        ),
                    )
                ),
            },
        )

    particle_count = int(finite_count.size)
    return {
        "schema_version": 1,
        "qualification_status": "DIAGNOSTIC_ONLY_NO_PRODUCTION_SELECTION",
        "production_fallback_requirement": (
            "FULL_DIRECT_FALLBACK_UNLESS_ERROR_BOUND_AND_CANDIDATE_SENTINEL_ARE_CERTIFIED"
        ),
        "particle_count": particle_count,
        "retained_topk": int(retained_topk),
        "source_rotation_block_capacity": int(rotation_block_capacity),
        "all_candidate": {
            "error_coverage_count": int(
                np.count_nonzero(joined("all_candidate_error_coverage")),
            ),
            "finite_pair_count": total_finite,
            "nonfinite_pair_count": int(np.sum(nonfinite_count, dtype=np.int64)),
            "max_abs_delta": _finite_max(joined("all_candidate_max_abs_delta")),
            "weighted_signed_mean_delta": weighted_signed_mean,
            "weighted_rms_delta": weighted_rms,
        },
        "winner": {
            "comparison_coverage_count": int(np.count_nonzero(winner_coverage)),
            "mismatch_count": int(np.count_nonzero(~winner_equal[winner_coverage])),
            "minimum_direct_margin": _finite_min(
                joined("direct_winner_margin")[winner_coverage],
            ),
            "minimum_macro_margin": _finite_min(
                joined("macro_winner_margin")[winner_coverage],
            ),
        },
        "support": {
            "comparison_coverage_count": int(np.count_nonzero(support_coverage)),
            "false_negative_total": int(np.sum(support_fn[support_coverage])),
            "false_positive_total": int(np.sum(support_fp[support_coverage])),
            "mismatch_particle_count": int(
                np.count_nonzero(
                    (support_fn[support_coverage] > 0) | (support_fp[support_coverage] > 0),
                ),
            ),
        },
        "relion_nonzero_rescore": {
            "score_span": RELION_COARSE_NONZERO_SCORE_SPAN,
            "safe_pair_coverage_count": int(np.count_nonzero(safe_pair_coverage)),
            "safe_width_max": _finite_max(safe_width[safe_pair_coverage]),
            "safe_pair_count": _count_distribution(
                safe_pair_count[safe_pair_coverage],
            ),
            "safe_source_rotation_block_list_coverage_count": int(
                np.count_nonzero(safe_block_coverage),
            ),
            "safe_source_rotation_block_overflow_count": int(
                np.count_nonzero(safe_block_overflow),
            ),
            "safe_source_rotation_block_count": _count_distribution(
                safe_block_count[safe_pair_coverage],
            ),
        },
        "bands": band_records,
    }
