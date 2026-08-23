from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_em_k4_allclass_recovar_repeatability import (
    GROUP_ORDER,
    array_metric,
    classify_first_unequal_group,
)


@pytest.mark.unit
@pytest.mark.parametrize("failed_index", range(len(GROUP_ORDER)))
def test_classification_stops_at_first_unequal_group(failed_index: int) -> None:
    group_exact = {
        group: index != failed_index
        for index, group in enumerate(GROUP_ORDER)
    }

    assert classify_first_unequal_group(group_exact) == GROUP_ORDER[failed_index]


@pytest.mark.unit
def test_classification_reports_exact_observed_boundary() -> None:
    group_exact = {group: True for group in GROUP_ORDER}

    assert classify_first_unequal_group(group_exact) == "all_observed_pass2_fields_exact"


@pytest.mark.unit
def test_classification_rejects_reordered_groups() -> None:
    group_exact = {group: True for group in reversed(GROUP_ORDER)}

    with pytest.raises(ValueError, match="identity/order changed"):
        classify_first_unequal_group(group_exact)


@pytest.mark.unit
def test_array_metric_compares_nan_payload_bytes() -> None:
    left = np.asarray((1.0, np.nan), dtype=np.float32)
    right = left.copy()
    exact = array_metric(left, right)
    right.view(np.uint32)[1] += np.uint32(1)
    changed = array_metric(left, right)

    assert exact["byte_exact"] is True
    assert changed["byte_exact"] is False
    assert changed["element_byte_mismatch_count"] == 1
    assert changed["same_nan_mask"] is True
    assert changed["correlation_used"] is False


@pytest.mark.unit
def test_array_metric_reports_finite_delta() -> None:
    left = np.asarray((1.0, 2.0), dtype=np.float32)
    right = left.copy()
    right[1] = np.nextafter(right[1], np.float32(np.inf))
    metric = array_metric(left, right)

    assert metric["byte_exact"] is False
    assert metric["first_mismatch_flat_index"] == 1
    assert metric["max_abs_finite_delta"] > 0
