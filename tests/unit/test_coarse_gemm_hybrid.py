from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    CoarseGemmHybridBlockSelection,
    assemble_coarse_gemm_hybrid_compact_scores_f32,
    assemble_coarse_gemm_hybrid_dense_scores_f32,
    certified_f32_dot_product_gamma,
    coarse_gemm_direct_score_intervals,
    coarse_gemm_hybrid_interval_state_bytes,
    initialize_coarse_gemm_hybrid_interval_state,
    map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids,
    propagate_coarse_gemm_intervals_through_f32_priors,
    select_coarse_gemm_hybrid_rotation_blocks,
    update_coarse_gemm_hybrid_interval_state,
    validate_coarse_gemm_hybrid_block_selection_for_rescore,
)

pytestmark = pytest.mark.unit


def _stream_intervals(
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    posterior_lower: np.ndarray,
    posterior_upper: np.ndarray,
    *,
    actual_image_count: int,
    chunk_sizes=(11, 19, 18),
):
    state = initialize_coarse_gemm_hybrid_interval_state(
        raw_lower.shape[0],
        raw_lower.shape[1],
    )
    start = 0
    with jax.enable_x64(True):
        arrays = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in (raw_lower, raw_upper, posterior_lower, posterior_upper)
        )
        for chunk_size in chunk_sizes:
            stop = min(start + chunk_size, raw_lower.shape[1])
            if stop <= start:
                continue
            state = update_coarse_gemm_hybrid_interval_state(
                state,
                *(value[:, start:stop, :] for value in arrays),
                rotation_offset=start,
                valid_rotation_count=stop - start,
                actual_image_count=actual_image_count,
            )
            start = stop
    assert start == raw_lower.shape[1]
    return state


def _complete_candidate_specific_state():
    batch_size, n_rotations, n_translations = 3, 48, 3
    shape = (batch_size, n_rotations, n_translations)
    raw_lower = np.empty(shape, dtype=np.float64)
    raw_upper = np.empty(shape, dtype=np.float64)
    posterior_lower = np.empty(shape, dtype=np.float64)
    posterior_upper = np.empty(shape, dtype=np.float64)
    block_values = (
        # Raw row 0 selects blocks 0 and 1; posterior selects 0 and 2.
        ([10.0, 0.0, -10.0], [10.0, 11.0, -9.0], [200.0, 50.0, 0.0], [200.0, 50.0, 63.0]),
        # Both raw and posterior row 1 select only block 1.
        ([0.0, 100.0, 0.0], [10.0, 100.0, 10.0], [0.0, 200.0, 0.0], [0.0, 200.0, 0.0]),
    )
    for row, values in enumerate(block_values):
        for block_id in range(3):
            sl = slice(16 * block_id, 16 * (block_id + 1))
            raw_lower[row, sl, :] = values[0][block_id]
            raw_upper[row, sl, :] = values[1][block_id]
            posterior_lower[row, sl, :] = values[2][block_id]
            posterior_upper[row, sl, :] = values[3][block_id]
    # A padded row may be garbage; it must never enter compact coverage state.
    for value in (raw_lower, raw_upper, posterior_lower, posterior_upper):
        value[2] = np.nan
    state = _stream_intervals(
        raw_lower,
        raw_upper,
        posterior_lower,
        posterior_upper,
        actual_image_count=2,
    )
    return state, n_rotations, n_translations


def _rescore_selection() -> CoarseGemmHybridBlockSelection:
    return CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray(
            [
                [0, 2, -1],
                [1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=np.int32,
        ),
        block_count=np.asarray([2, 1, 0], dtype=np.int32),
        posterior_block_count=np.asarray([2, 1, 0], dtype=np.int32),
        raw_max_block_count=np.asarray([1, 1, 0], dtype=np.int32),
    )


def test_certified_gamma_uses_direct_kernel_traversed_position_bound() -> None:
    gamma = certified_f32_dot_product_gamma(5100, 29)
    translations_per_lane = 128 // 29
    lane_terms = ((5100 + 31) // 32) * ((32 + translations_per_lane - 1) // translations_per_lane)
    operation_count = lane_terms + translations_per_lane + 5
    expected = (operation_count * 2.0**-24) / (1.0 - operation_count * 2.0**-24)

    assert translations_per_lane == 4
    assert lane_terms == 1280
    assert operation_count == 1289
    assert gamma == np.nextafter(np.float64(expected), np.float64(np.inf))
    assert gamma == pytest.approx(7.68362904774204e-5, rel=1e-15)


@pytest.mark.parametrize(
    "n_positions,n_translations",
    [(0, 29), (5100, 0), (5100, 129), (1 << 30, 1)],
)
def test_certified_gamma_rejects_invalid_kernel_dimensions(
    n_positions: int,
    n_translations: int,
) -> None:
    with pytest.raises(ValueError):
        certified_f32_dot_product_gamma(n_positions, n_translations)


def test_compact_interval_state_has_small_explicit_byte_ceiling() -> None:
    assert coarse_gemm_hybrid_interval_state_bytes(2, 48) == 400
    # GF46 has 36,864 rotations; even a padded batch of 32 keeps the complete
    # four-table certificate under 2.5 MiB rather than materializing a cube.
    assert coarse_gemm_hybrid_interval_state_bytes(32, 36_864) < int(2.5 * 2**20)


def test_direct_intervals_are_centered_on_g64_with_directed_rounding() -> None:
    score = np.asarray([[-7.25, -3.5], [-2.0, -100.0]], dtype=np.float64)
    eta = np.asarray([[0.125], [0.25]], dtype=np.float64)
    gamma = certified_f32_dot_product_gamma(5100, 29)
    with jax.enable_x64(True):
        lower, upper = coarse_gemm_direct_score_intervals(score, eta, gamma)

    positive_inf = np.float64(np.inf)
    negative_inf = np.float64(-np.inf)
    q_upper = np.nextafter(np.maximum(0.0, -score + eta), positive_inf)
    gamma_term = np.nextafter(gamma * q_upper, positive_inf)
    error_upper = np.nextafter(eta + gamma_term, positive_inf)
    expected_lower = np.nextafter(score - error_upper, negative_inf)
    expected_upper = np.nextafter(score + error_upper, positive_inf)

    assert np.array_equal(np.asarray(lower), expected_lower)
    assert np.array_equal(np.asarray(upper), expected_upper)


def test_invalid_certificate_values_become_fail_closed_intervals() -> None:
    with jax.enable_x64(True):
        negative_eta = coarse_gemm_direct_score_intervals(
            jnp.zeros((1, 1, 1), dtype=jnp.float64),
            jnp.float64(-1.0),
            jnp.float64(0.0),
        )
        invalid_gamma = coarse_gemm_direct_score_intervals(
            jnp.zeros((1, 1, 1), dtype=jnp.float64),
            jnp.float64(0.0),
            jnp.float64(jnp.inf),
        )
        broken_score_contract = coarse_gemm_direct_score_intervals(
            jnp.ones((1, 1, 1), dtype=jnp.float64),
            jnp.float64(0.5),
            jnp.float64(0.0),
        )
    assert all(np.isnan(np.asarray(value)).all() for value in negative_eta)
    assert all(np.isnan(np.asarray(value)).all() for value in invalid_gamma)
    assert all(np.isnan(np.asarray(value)).all() for value in broken_score_contract)


def test_invalid_eta_flows_through_compact_state_to_direct_fallback() -> None:
    with jax.enable_x64(True):
        lower, upper = coarse_gemm_direct_score_intervals(
            jnp.zeros((1, 16, 2), dtype=jnp.float64),
            jnp.float64(-1.0),
            jnp.float64(0.0),
        )
    state = initialize_coarse_gemm_hybrid_interval_state(1, 16)
    state = update_coarse_gemm_hybrid_interval_state(
        state,
        lower,
        upper,
        lower,
        upper,
        rotation_offset=0,
        valid_rotation_count=16,
        actual_image_count=1,
    )
    selection = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=1,
        n_rotations=16,
        n_translations=2,
        certificate_valid=True,
    )
    assert not selection.eligible
    assert selection.fallback_reason == "invalid_or_nonfinite_candidate_interval"


def test_prior_intervals_enclose_dense_add_priors_f32_order() -> None:
    raw = np.asarray([[[1.0e8, -1.0e8]]], dtype=np.float32)
    class_prior = np.float32(-1.0e8)
    rotation_prior = np.asarray([3.0], dtype=np.float32)
    translation_prior = np.asarray([0.25, -0.5], dtype=np.float32)
    with jax.enable_x64(True):
        lower, upper = propagate_coarse_gemm_intervals_through_f32_priors(
            raw.astype(np.float64),
            raw.astype(np.float64),
            class_log_prior=class_prior,
            rotation_log_prior=rotation_prior,
            translation_log_prior=translation_prior,
        )

    # This is the exact class -> rotation -> translation sequence in the dense
    # E-step's nested _add_priors helper, including the K=1 class addition.
    production = jnp.asarray(raw)
    production = production + jnp.asarray(class_prior, dtype=production.real.dtype)
    production = production + jnp.asarray(rotation_prior)[None, :, None]
    production = production + jnp.asarray(translation_prior)[None, None, :]
    production = np.asarray(production, dtype=np.float64)
    assert np.all(np.asarray(lower) <= production)
    assert np.all(production <= np.asarray(upper))
    assert production[0, 0, 0] == np.float32(3.25)

    wrong_order = jnp.asarray(raw) + jnp.asarray(rotation_prior)[None, :, None]
    wrong_order = wrong_order + jnp.asarray(class_prior, dtype=wrong_order.real.dtype)
    wrong_order = wrong_order + jnp.asarray(translation_prior)[None, None, :]
    assert np.asarray(wrong_order)[0, 0, 0] == np.float32(0.25)


def test_candidate_intervals_select_complete_source16_block_union() -> None:
    state, n_rotations, n_translations = _complete_candidate_specific_state()
    selection = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=2,
        n_rotations=n_rotations,
        n_translations=n_translations,
        certificate_valid=True,
        block_capacity=4,
    )

    assert selection.eligible
    assert selection.fallback_reason is None
    assert selection.block_ids.tolist() == [
        [0, 1, 2, -1],
        [1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert selection.block_count.tolist() == [3, 1, 0]
    assert selection.posterior_block_count.tolist() == [2, 1, 0]
    assert selection.raw_max_block_count.tolist() == [2, 1, 0]
    assert np.asarray(state.rotation_visit_count).tolist() == [1] * n_rotations
    assert np.asarray(state.candidate_count).tolist() == [144, 144, 0]
    assert np.asarray(state.invalid_candidate_count).tolist() == [0, 0, 0]


def test_f32_endpoints_scatter_into_f64_compact_state() -> None:
    shape = (1, 16, 2)
    lower = np.full(shape, np.float32(-3.5), dtype=np.float32)
    upper = np.full(shape, np.float32(-3.25), dtype=np.float32)
    state = initialize_coarse_gemm_hybrid_interval_state(1, 16)
    state = update_coarse_gemm_hybrid_interval_state(
        state,
        lower,
        upper,
        lower,
        upper,
        rotation_offset=0,
        valid_rotation_count=16,
        actual_image_count=1,
    )
    assert state.raw_block_lower_max.dtype == jnp.float64
    assert state.posterior_block_upper_max.dtype == jnp.float64
    assert np.asarray(state.raw_block_lower_max).tolist() == [[-3.5]]
    assert np.asarray(state.posterior_block_upper_max).tolist() == [[-3.25]]


@pytest.mark.parametrize("certificate_valid", [None, False])
def test_selector_requires_explicit_valid_certificate_contract(certificate_valid) -> None:
    state, n_rotations, n_translations = _complete_candidate_specific_state()
    selection = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=2,
        n_rotations=n_rotations,
        n_translations=n_translations,
        certificate_valid=certificate_valid,
    )
    assert not selection.eligible
    assert selection.fallback_reason == "missing_or_invalid_certificate_contract"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("coverage", "incomplete_or_duplicate_rotation_coverage"),
        ("count", "incomplete_candidate_coverage"),
        ("candidate", "invalid_or_nonfinite_candidate_interval"),
        ("block", "invalid_or_nonfinite_block_interval"),
    ],
)
def test_selector_fails_closed_on_incomplete_or_invalid_compact_state(
    mutation: str,
    reason: str,
) -> None:
    state, n_rotations, n_translations = _complete_candidate_specific_state()
    if mutation == "coverage":
        state = state._replace(rotation_visit_count=state.rotation_visit_count.at[7].set(2))
    elif mutation == "count":
        state = state._replace(candidate_count=state.candidate_count.at[0].add(-1))
    elif mutation == "candidate":
        state = state._replace(invalid_candidate_count=state.invalid_candidate_count.at[0].set(1))
    else:
        state = state._replace(raw_block_upper_max=state.raw_block_upper_max.at[0, 0].set(jnp.inf))
    selection = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=2,
        n_rotations=n_rotations,
        n_translations=n_translations,
        certificate_valid=True,
    )
    assert not selection.eligible
    assert selection.fallback_reason == reason


def test_selector_fails_closed_on_rotation_tail_and_capacity_overflow() -> None:
    state, n_rotations, n_translations = _complete_candidate_specific_state()
    overflow = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=2,
        n_rotations=n_rotations,
        n_translations=n_translations,
        certificate_valid=True,
        block_capacity=2,
    )
    assert not overflow.eligible
    assert overflow.fallback_reason == "block_capacity_overflow"

    tail_shape = (1, 47, 2)
    zeros = np.zeros(tail_shape, dtype=np.float64)
    tail_state = _stream_intervals(
        zeros,
        zeros,
        zeros,
        zeros,
        actual_image_count=1,
        chunk_sizes=(47,),
    )
    tail = select_coarse_gemm_hybrid_rotation_blocks(
        tail_state,
        actual_image_count=1,
        n_rotations=47,
        n_translations=2,
        certificate_valid=True,
    )
    assert not tail.eligible
    assert tail.fallback_reason == "rotation_tail_not_supported"


def test_exact_selected_scores_restore_relion_flat_order_raw_max_and_padding() -> None:
    selection = _rescore_selection()
    batch_size, capacity = selection.block_ids.shape
    n_rotations, n_translations = 48, 2
    selected_diff2 = np.full(
        (batch_size, capacity, 16, n_translations),
        np.inf,
        dtype=np.float32,
    )
    selected_diff2[0, 0] = np.arange(16, dtype=np.float32)[:, None] * np.float32(
        4.0
    ) + np.asarray([1.0, 2.0], dtype=np.float32)[None, :]
    selected_diff2[0, 1] = np.arange(16, dtype=np.float32)[:, None] * np.float32(
        3.0
    ) + np.asarray([0.25, 1.25], dtype=np.float32)[None, :]
    selected_diff2[1, 0] = np.arange(16, dtype=np.float32)[:, None] * np.float32(
        2.0
    ) + np.asarray([5.0, 6.0], dtype=np.float32)[None, :]
    class_prior = np.float32(-1.25)
    rotation_prior = np.arange(n_rotations, dtype=np.float32) * np.float32(0.125)
    translation_prior = np.asarray([0.5, -0.75], dtype=np.float32)

    assembled = assemble_coarse_gemm_hybrid_dense_scores_f32(
        jnp.asarray(selected_diff2),
        selection,
        actual_image_count=2,
        n_rotations=n_rotations,
        class_log_prior=class_prior,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )

    expected = np.full(
        (batch_size, n_rotations, n_translations),
        -np.inf,
        dtype=np.float32,
    )
    for row in range(2):
        for slot in range(int(selection.block_count[row])):
            source_block = int(selection.block_ids[row, slot])
            for offset in range(16):
                rotation = 16 * source_block + offset
                values = -selected_diff2[row, slot, offset]
                values = values + class_prior
                values = values + rotation_prior[rotation]
                values = values + translation_prior
                expected[row, rotation] = values

    actual = np.asarray(assembled.posterior_scores_flat).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(assembled.raw_score_max),
        np.asarray([-0.25, -5.0, -np.inf], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(assembled.min_diff2_offsets),
        np.asarray([0.25, 5.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(assembled.best_score),
        np.max(expected.reshape(batch_size, -1), axis=1),
    )
    np.testing.assert_array_equal(
        np.asarray(assembled.best_pose),
        np.argmax(expected.reshape(batch_size, -1), axis=1).astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(assembled.selected_output_valid),
        np.ones(batch_size, dtype=bool),
    )
    assert np.all(np.isneginf(actual[0, 16:32]))
    assert np.all(np.isneginf(actual[1, :16]))
    assert np.all(np.isneginf(actual[2]))


def test_exact_selected_score_ties_use_global_relion_flat_order() -> None:
    selection = CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray([[0, 2]], dtype=np.int32),
        block_count=np.asarray([2], dtype=np.int32),
        posterior_block_count=np.asarray([2], dtype=np.int32),
        raw_max_block_count=np.asarray([1], dtype=np.int32),
    )
    diff2 = np.full((1, 2, 16, 3), np.float32(10.0), dtype=np.float32)
    diff2[0, 0, 7, 2] = np.float32(1.0)
    diff2[0, 1, 0, 0] = np.float32(1.0)

    assembled = assemble_coarse_gemm_hybrid_dense_scores_f32(
        diff2,
        selection,
        actual_image_count=1,
        n_rotations=48,
        class_log_prior=0.0,
    )

    assert int(np.asarray(assembled.best_pose)[0]) == 7 * 3 + 2
    assert int(np.asarray(assembled.best_pose)[0]) < 32 * 3


def test_compact_selected_scores_match_dense_oracle_at_global_pose_ids() -> None:
    selection = _rescore_selection()
    batch_size, capacity = selection.block_ids.shape
    n_rotations, n_translations = 48, 3
    selected_diff2 = np.full(
        (batch_size, capacity, 16, n_translations),
        np.inf,
        dtype=np.float32,
    )
    rng = np.random.default_rng(2_609)
    selected_diff2[0, :2] = rng.uniform(
        0.25,
        30.0,
        size=(2, 16, n_translations),
    ).astype(np.float32)
    selected_diff2[1, :1] = rng.uniform(
        0.25,
        30.0,
        size=(1, 16, n_translations),
    ).astype(np.float32)
    rotation_prior = np.linspace(-0.75, 0.5, n_rotations, dtype=np.float32)
    translation_prior = np.asarray(
        [[0.0, -0.25, 0.5], [0.125, 0.25, -0.5], [9.0, 9.0, 9.0]],
        dtype=np.float32,
    )
    kwargs = dict(
        actual_image_count=2,
        n_rotations=n_rotations,
        class_log_prior=np.float32(-1.125),
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )

    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        selected_diff2,
        selection,
        **kwargs,
    )
    dense = assemble_coarse_gemm_hybrid_dense_scores_f32(
        selected_diff2,
        selection,
        **kwargs,
    )
    compact_scores = np.asarray(compact.posterior_scores_flat)
    dense_scores = np.asarray(dense.posterior_scores_flat)
    compact_mask = np.isfinite(compact_scores)
    compact_ids = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
        compact,
        compact_mask,
    )

    for row in range(2):
        active_ids = compact_ids[row]
        assert np.all(np.diff(active_ids) > 0)
        np.testing.assert_array_equal(
            compact_scores[row, compact_mask[row]],
            dense_scores[row, active_ids],
        )
        assert np.all(np.isneginf(compact_scores[row, ~compact_mask[row]]))
    assert compact_ids[2].size == 0
    np.testing.assert_array_equal(
        np.asarray(compact.raw_score_max),
        np.asarray(dense.raw_score_max),
    )
    np.testing.assert_array_equal(
        np.asarray(compact.min_diff2_offsets),
        np.asarray(dense.min_diff2_offsets),
    )
    np.testing.assert_array_equal(
        np.asarray(compact.best_score),
        np.asarray(dense.best_score),
    )
    np.testing.assert_array_equal(
        np.asarray(compact.best_pose),
        np.asarray(dense.best_pose),
    )
    np.testing.assert_array_equal(
        np.asarray(compact.selected_output_valid),
        np.asarray(dense.selected_output_valid),
    )


def test_compact_selected_score_ties_follow_ascending_source16_pose_order() -> None:
    selection = CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray([[1, 3, -1]], dtype=np.int32),
        block_count=np.asarray([2], dtype=np.int32),
        posterior_block_count=np.asarray([2], dtype=np.int32),
        raw_max_block_count=np.asarray([1], dtype=np.int32),
    )
    diff2 = np.full((1, 3, 16, 2), np.inf, dtype=np.float32)
    diff2[0, :2] = np.float32(20.0)
    diff2[0, 0, 15, 1] = np.float32(1.0)
    diff2[0, 1, 0, 0] = np.float32(1.0)

    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        diff2,
        selection,
        actual_image_count=1,
        n_rotations=64,
        class_log_prior=0.0,
    )

    active_ids = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
        compact,
        np.isfinite(np.asarray(compact.posterior_scores_flat)),
    )[0]
    assert np.all(np.diff(active_ids) > 0)
    assert int(np.asarray(compact.best_pose)[0]) == 31 * 2 + 1
    assert int(np.asarray(compact.best_pose)[0]) < 48 * 2


def test_compact_support_mapping_is_sparse_ordered_and_fail_closed() -> None:
    selection = CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray([[1, 3, -1], [2, -1, -1]], dtype=np.int32),
        block_count=np.asarray([2, 1], dtype=np.int32),
        posterior_block_count=np.asarray([2, 1], dtype=np.int32),
        raw_max_block_count=np.asarray([1, 1], dtype=np.int32),
    )
    diff2 = np.full((2, 3, 16, 3), np.inf, dtype=np.float32)
    diff2[0, :2] = np.float32(4.0)
    diff2[1, :1] = np.float32(5.0)
    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        diff2,
        selection,
        actual_image_count=2,
        n_rotations=64,
        class_log_prior=0.0,
    )
    mask = np.zeros(compact.posterior_scores_flat.shape, dtype=bool)
    # Compact positions intentionally cross the noncontiguous source-block
    # boundary. The mapped IDs must stay in global rotation-major order.
    mask[0, [0, 17, 47, 48, 53]] = True
    mask[1, [1, 31]] = True

    mapped = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(compact, mask)

    np.testing.assert_array_equal(mapped[0], [48, 65, 95, 144, 149])
    np.testing.assert_array_equal(mapped[1], [97, 127])
    invalid_mask = mask.copy()
    invalid_mask[1, 48] = True
    with pytest.raises(ValueError, match="inactive capacity slot"):
        map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
            compact,
            invalid_mask,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "strictly increasing"),
        ("unsorted", "strictly increasing"),
        ("out_of_range", "strictly increasing"),
        ("interior_padding", "strictly increasing"),
        ("padded_row", "padded image rows"),
        ("capacity", "inconsistent selected block counts"),
        ("ineligible", "eligible block selection"),
    ],
)
def test_rescore_selection_validation_fails_closed(mutation: str, match: str) -> None:
    selection = _rescore_selection()
    block_ids = selection.block_ids.copy()
    block_count = selection.block_count.copy()
    if mutation == "duplicate":
        block_ids[0, :2] = 1
    elif mutation == "unsorted":
        block_ids[0, :2] = [2, 0]
    elif mutation == "out_of_range":
        block_ids[0, 1] = 3
    elif mutation == "interior_padding":
        block_ids[0, 0] = -1
    elif mutation == "padded_row":
        block_ids[2, 0] = 0
    elif mutation == "capacity":
        block_count[0] = 4
    else:
        selection = selection._replace(eligible=False, fallback_reason="test_fallback")
    selection = selection._replace(block_ids=block_ids, block_count=block_count)

    with pytest.raises(ValueError, match=match):
        validate_coarse_gemm_hybrid_block_selection_for_rescore(
            selection,
            actual_image_count=2,
            n_rotations=48,
        )


def test_selected_ffi_nonfinite_active_or_nonpositive_inf_padding_requests_fallback() -> None:
    selection = _rescore_selection()
    diff2 = np.full((3, 3, 16, 2), np.inf, dtype=np.float32)
    diff2[0, :2] = np.float32(1.0)
    diff2[1, 0] = np.float32(2.0)
    diff2[0, 1, 3, 0] = np.nan
    diff2[1, 1, 0, 0] = np.float32(0.0)

    assembled = assemble_coarse_gemm_hybrid_dense_scores_f32(
        diff2,
        selection,
        actual_image_count=2,
        n_rotations=48,
        class_log_prior=0.0,
    )

    np.testing.assert_array_equal(
        np.asarray(assembled.selected_output_valid),
        np.asarray([False, False, True]),
    )
    assert np.all(np.isneginf(np.asarray(assembled.posterior_scores_flat)[2]))


def test_full_logical_scatter_preserves_exact_relion_posterior_and_support(
    monkeypatch,
) -> None:
    from recovar.em.dense_single_volume.helpers import oversampling

    cpu_device = jax.devices("cpu")[0]
    monkeypatch.setattr(oversampling.jax, "default_backend", lambda: "cpu")
    relion_cuda_f32_coarse_posterior = oversampling.relion_cuda_f32_coarse_posterior
    relion_cuda_f32_coarse_posterior.clear_cache()

    selection = CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray([[0, 2]], dtype=np.int32),
        block_count=np.asarray([2], dtype=np.int32),
        posterior_block_count=np.asarray([2], dtype=np.int32),
        raw_max_block_count=np.asarray([1], dtype=np.int32),
    )
    rng = np.random.default_rng(19)
    full_diff2 = rng.uniform(1.0, 15.0, size=(1, 48, 3)).astype(np.float32)
    # Every omitted source block has exact RELION expf weight zero, but remains
    # finite in the rectangular reference table.
    full_diff2[:, 16:32] += np.float32(200.0)
    selected_diff2 = np.stack(
        [full_diff2[:, 0:16], full_diff2[:, 32:48]],
        axis=1,
    )
    rotation_prior = np.linspace(-0.5, 0.5, 48, dtype=np.float32)
    translation_prior = np.asarray([0.0, -0.25, 0.125], dtype=np.float32)
    class_prior = np.float32(-0.75)
    assembled = assemble_coarse_gemm_hybrid_dense_scores_f32(
        selected_diff2,
        selection,
        actual_image_count=1,
        n_rotations=48,
        class_log_prior=class_prior,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )
    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        selected_diff2,
        selection,
        actual_image_count=1,
        n_rotations=48,
        class_log_prior=class_prior,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )
    full_scores = -jnp.asarray(full_diff2)
    full_scores = full_scores + class_prior
    full_scores = full_scores + jnp.asarray(rotation_prior)[None, :, None]
    full_scores = full_scores + jnp.asarray(translation_prior)[None, None, :]
    full_raw_max = jnp.max(-jnp.asarray(full_diff2), axis=(1, 2))
    kwargs = dict(adaptive_fraction=0.999, max_significants=7, tie_score_ulps=0)
    with jax.default_device(cpu_device):
        expected = relion_cuda_f32_coarse_posterior(
            jax.device_put(full_scores.reshape(1, -1), cpu_device),
            min_diff2_offsets=jax.device_put(-full_raw_max, cpu_device),
            **kwargs,
        )
        actual = relion_cuda_f32_coarse_posterior(
            jax.device_put(assembled.posterior_scores_flat, cpu_device),
            min_diff2_offsets=jax.device_put(assembled.min_diff2_offsets, cpu_device),
            **kwargs,
        )
        compact_result = relion_cuda_f32_coarse_posterior(
            jax.device_put(compact.posterior_scores_flat, cpu_device),
            min_diff2_offsets=jax.device_put(compact.min_diff2_offsets, cpu_device),
            **kwargs,
        )

    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(expected_value))
    expected_support_ids = np.flatnonzero(np.asarray(expected[1])[0]).astype(np.int32)
    compact_support_ids = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
        compact,
        np.asarray(compact_result[1]),
    )[0]
    np.testing.assert_array_equal(compact_support_ids, expected_support_ids)
    np.testing.assert_array_equal(
        np.asarray(compact_result[2]),
        np.asarray(expected[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_result[3]),
        np.asarray(expected[3]),
    )


def test_compacting_exact_zero_weight_candidates_changes_float32_scan_rounding(
    monkeypatch,
) -> None:
    from recovar.em.dense_single_volume.helpers import oversampling

    cpu_device = jax.devices("cpu")[0]
    monkeypatch.setattr(oversampling.jax, "default_backend", lambda: "cpu")
    relion_cuda_f32_coarse_posterior = oversampling.relion_cuda_f32_coarse_posterior
    relion_cuda_f32_coarse_posterior.clear_cache()

    compact_scores = np.asarray(
        [-0.37921745, -5.3189178, -1.539008],
        dtype=np.float32,
    )
    global_pose_ids = np.asarray([2, 6, 7], dtype=np.int32)
    full_scores = np.full((18,), -np.inf, dtype=np.float32)
    full_scores[global_pose_ids] = compact_scores
    kwargs = dict(
        adaptive_fraction=0.999,
        max_significants=500,
        tie_score_ulps=0,
    )
    with jax.default_device(cpu_device):
        offsets = jnp.zeros((1,), dtype=jnp.float32)
        full = relion_cuda_f32_coarse_posterior(
            jnp.asarray(full_scores[None, :]),
            min_diff2_offsets=offsets,
            **kwargs,
        )
        compact = relion_cuda_f32_coarse_posterior(
            jnp.asarray(compact_scores[None, :]),
            min_diff2_offsets=offsets,
            **kwargs,
        )
    full_np = tuple(np.asarray(value) for value in full)
    compact_np = tuple(np.asarray(value) for value in compact)

    np.testing.assert_array_equal(full_np[1][0, global_pose_ids], compact_np[1][0])
    np.testing.assert_array_equal(full_np[2], compact_np[2])
    np.testing.assert_array_equal(full_np[3], compact_np[3])
    np.testing.assert_array_equal(full_np[5], compact_np[5])
    assert not np.array_equal(full_np[4], compact_np[4])
    assert not np.array_equal(full_np[0][0, global_pose_ids], compact_np[0][0])


@pytest.mark.gpu
def test_compact_hybrid_gpu_positive_oracle_and_fixed_capacity_support(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
) -> None:
    """Compact scores retain dense positive-only support on the live CUB path."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import oversampling

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    selection = CoarseGemmHybridBlockSelection(
        eligible=True,
        fallback_reason=None,
        block_ids=np.asarray(
            [[0, 2, -1], [1, 3, -1], [0, -1, -1]],
            dtype=np.int32,
        ),
        block_count=np.asarray([2, 2, 1], dtype=np.int32),
        posterior_block_count=np.asarray([2, 2, 1], dtype=np.int32),
        raw_max_block_count=np.asarray([1, 1, 1], dtype=np.int32),
    )
    rng = np.random.default_rng(7_331)
    diff2 = np.full((3, 3, 16, 3), np.inf, dtype=np.float32)
    for row, count in enumerate(selection.block_count):
        diff2[row, :count] = rng.uniform(
            0.25,
            20.0,
            size=(int(count), 16, 3),
        ).astype(np.float32)
    rotation_prior = np.linspace(-0.25, 0.5, 64, dtype=np.float32)
    translation_prior = np.asarray([0.0, -0.125, 0.25], dtype=np.float32)
    kwargs = dict(
        actual_image_count=3,
        n_rotations=64,
        class_log_prior=np.float32(-0.5),
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )
    with jax.default_device(gpu_device):
        compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
            jnp.asarray(diff2),
            selection,
            **kwargs,
        )
        dense = assemble_coarse_gemm_hybrid_dense_scores_f32(
            jnp.asarray(diff2),
            selection,
            **kwargs,
        )
        posterior_kwargs = dict(
            adaptive_fraction=0.999,
            max_significants=7,
            tie_score_ulps=0,
        )
        dense_positive = oversampling.relion_cuda_f32_coarse_posterior(
            dense.posterior_scores_flat,
            min_diff2_offsets=dense.min_diff2_offsets,
            filter_positive_before_sort=True,
            **posterior_kwargs,
        )
        compact_positive = oversampling.relion_cuda_f32_coarse_posterior(
            compact.posterior_scores_flat,
            min_diff2_offsets=compact.min_diff2_offsets,
            filter_positive_before_sort=True,
            **posterior_kwargs,
        )
        compact_fixed = oversampling.relion_cuda_f32_coarse_posterior(
            compact.posterior_scores_flat,
            min_diff2_offsets=compact.min_diff2_offsets,
            filter_positive_before_sort=False,
            **posterior_kwargs,
        )
        jax.block_until_ready((dense_positive, compact_positive, compact_fixed))

    dense_support = tuple(
        np.flatnonzero(row).astype(np.int32)
        for row in np.asarray(dense_positive[1], dtype=bool)
    )
    compact_positive_support = (
        map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
            compact,
            np.asarray(compact_positive[1], dtype=bool),
        )
    )
    compact_fixed_support = map_coarse_gemm_hybrid_compact_mask_to_global_pose_ids(
        compact,
        np.asarray(compact_fixed[1], dtype=bool),
    )
    for dense_ids, positive_ids, fixed_ids in zip(
        dense_support,
        compact_positive_support,
        compact_fixed_support,
        strict=True,
    ):
        np.testing.assert_array_equal(positive_ids, dense_ids)
        np.testing.assert_array_equal(fixed_ids, dense_ids)
    for field_index in (2, 3, 5):
        np.testing.assert_array_equal(
            np.asarray(compact_positive[field_index]).view(np.uint32),
            np.asarray(dense_positive[field_index]).view(np.uint32),
        )
    np.testing.assert_array_equal(
        np.asarray(compact_fixed[2]),
        np.asarray(dense_positive[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_fixed[3]),
        np.asarray(dense_positive[3]),
    )
