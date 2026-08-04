from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_em_k4_allclass_native_boundary import (
    BOUNDARY_ORDER,
    candidate_key_from_flat_index,
    candidate_key_from_sequence_index,
    classify_first_unequal_boundary,
    exact_rotation_permutation,
    float32_metric,
    nonnegative_support_boundary,
)


@pytest.mark.unit
@pytest.mark.parametrize("failed_index", range(len(BOUNDARY_ORDER)))
def test_classification_stops_at_first_unequal_boundary(failed_index: int) -> None:
    stage_exact = {
        boundary: index != failed_index
        for index, boundary in enumerate(BOUNDARY_ORDER)
    }

    assert classify_first_unequal_boundary(stage_exact) == BOUNDARY_ORDER[failed_index]


@pytest.mark.unit
def test_classification_routes_closed_observed_stages_to_unobserved_bpref() -> None:
    stage_exact = {boundary: True for boundary in BOUNDARY_ORDER}

    assert classify_first_unequal_boundary(stage_exact) == "bpref_operands_unobserved"


@pytest.mark.unit
def test_classification_rejects_reordered_boundaries() -> None:
    stage_exact = {boundary: True for boundary in reversed(BOUNDARY_ORDER)}

    with pytest.raises(ValueError, match="identity/order changed"):
        classify_first_unequal_boundary(stage_exact)


@pytest.mark.unit
def test_rotation_permutation_is_a_native_to_recovar_gather() -> None:
    recovar = np.stack(
        (
            np.eye(3, dtype=np.float32),
            np.diag(np.asarray((1, -1, -1), dtype=np.float32)),
            np.diag(np.asarray((-1, 1, -1), dtype=np.float32)),
        )
    )
    native = recovar[[2, 0, 1]]

    assert exact_rotation_permutation(native, recovar).tolist() == [2, 0, 1]


@pytest.mark.unit
def test_rotation_permutation_rejects_nonbijective_recovar_table() -> None:
    recovar = np.stack((np.eye(3, dtype=np.float32),) * 2)

    with pytest.raises(ValueError, match="not unique"):
        exact_rotation_permutation(recovar, recovar)


@pytest.mark.unit
def test_float32_metric_is_bitwise_and_scale_sensitive() -> None:
    left = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)
    right = left.copy()
    exact = float32_metric(left, right)
    right[1] = np.nextafter(right[1], np.float32(np.inf))
    changed = float32_metric(left, right)

    assert exact["bitwise_exact"] is True
    assert exact["bitwise_mismatch_count"] == 0
    assert changed["bitwise_exact"] is False
    assert changed["bitwise_mismatch_count"] == 1
    assert changed["first_mismatch_flat_index"] == 1
    assert changed["max_abs"] > 0
    assert changed["finite_mismatch_max_ulp"] == 1
    assert changed["finite_mismatch_p50_ulp"] == 1
    assert changed["finite_mismatch_p95_ulp"] == 1
    assert changed["correlation_used"] is False


@pytest.mark.unit
def test_float32_metric_ulp_distance_is_ordered_across_signs() -> None:
    left = np.asarray((-1.0, 1.0), dtype=np.float32)
    right = np.asarray(
        (
            np.nextafter(left[0], np.float32(-np.inf)),
            np.nextafter(left[1], np.float32(np.inf)),
        ),
        dtype=np.float32,
    )

    metric = float32_metric(left, right)

    assert metric["finite_mismatch_max_ulp"] == 1
    assert metric["finite_mismatch_p50_ulp"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_support_boundary_preserves_dtype_bits_and_margin(dtype) -> None:
    values = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=dtype)
    active = np.asarray((True, True, True, False))
    selected = np.asarray((False, True, True, False))

    boundary = nonnegative_support_boundary(
        values,
        active,
        selected,
        threshold=dtype(0.2),
    )

    assert boundary["dtype"] == np.dtype(dtype).name
    assert boundary["selected_count"] == 2
    assert boundary["excluded_active_count"] == 1
    assert boundary["minimum_selected"] == float(dtype(0.2))
    assert boundary["maximum_excluded_active"] == float(dtype(0.1))
    assert boundary["selected_minus_excluded_margin"] > 0
    assert boundary["selected_minus_excluded_margin_ulps"] > 0
    assert boundary["recorded_threshold"]["replays_selection_exact"] is True
    assert boundary["recorded_threshold"]["minimum_selected_minus_threshold"] == 0


@pytest.mark.unit
def test_support_boundary_rejects_inverted_selection() -> None:
    values = np.asarray((0.1, 0.2), dtype=np.float32)

    with pytest.raises(ValueError, match="boundary is inverted"):
        nonnegative_support_boundary(
            values,
            np.asarray((True, True)),
            np.asarray((True, False)),
        )


@pytest.mark.unit
def test_flat_candidate_key_uses_rotation_major_recovar_coordinates() -> None:
    inverse = np.asarray((2, 0, 1), dtype=np.int64)

    assert candidate_key_from_flat_index(
        9,
        (3, 4),
        recovar_to_native_rotation=inverse,
    ) == {
        "native_rotation_local": 1,
        "recovar_rotation_local": 2,
        "translation_id": 1,
    }
    assert candidate_key_from_flat_index(
        None,
        (3, 4),
        recovar_to_native_rotation=inverse,
    ) is None


@pytest.mark.unit
def test_sequence_candidate_key_preserves_joined_native_and_recovar_rows() -> None:
    assert candidate_key_from_sequence_index(
        1,
        native_rotation=np.asarray((5, 2, 8)),
        recovar_rotation=np.asarray((1, 7, 3)),
        translation=np.asarray((4, 6, 0)),
    ) == {
        "native_rotation_local": 2,
        "recovar_rotation_local": 7,
        "translation_id": 6,
    }
