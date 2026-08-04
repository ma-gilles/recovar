from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_em_k4_allclass_native_boundary import (
    BOUNDARY_ORDER,
    classify_first_unequal_boundary,
    exact_rotation_permutation,
    float32_metric,
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
    assert changed["correlation_used"] is False
