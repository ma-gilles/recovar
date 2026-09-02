from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    certified_f32_dot_product_gamma,
    coarse_gemm_direct_score_intervals,
    coarse_gemm_hybrid_interval_state_bytes,
    initialize_coarse_gemm_hybrid_interval_state,
    propagate_coarse_gemm_intervals_through_f32_priors,
    select_coarse_gemm_hybrid_rotation_blocks,
    update_coarse_gemm_hybrid_interval_state,
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
