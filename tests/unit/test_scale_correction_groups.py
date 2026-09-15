"""Scale-statistics axes survive empty groups and class-specific subsets."""

import numpy as np
import pytest

from recovar.em.classification.k_class import _full_group_count_from_kwargs
from recovar.em.helpers.scale_groups import prepare_scale_correction_groups

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "ids,explicit,expected_count",
    [(None, None, 0), (None, 5, 5), ([], None, 1), ([], 5, 5),
     ([0, 2], None, 3), ([0, 2], 1, 3), ([0, 2], 5, 5),
     ([[0], [2]], 5.0, 5)],
)
def test_full_group_axis_is_preserved(ids, explicit, expected_count):
    original = None if ids is None else np.asarray(ids)
    normalized, count = prepare_scale_correction_groups(
        ids, explicit, n_images=0 if original is None else original.size,
    )
    assert count == expected_count
    if ids is None:
        assert normalized is None
    else:
        assert normalized.dtype == np.int64
        np.testing.assert_array_equal(normalized, original.reshape(-1))
    assert _full_group_count_from_kwargs(
        dict(group_ids=ids, scale_correction_group_count=explicit),
    ) == (expected_count or None)


@pytest.mark.parametrize("explicit", [-1, 1.5, "bad", np.nan, np.inf, -np.inf])
def test_invalid_explicit_count_is_rejected_before_group_shape(explicit):
    with pytest.raises((ValueError, OverflowError)) as error:
        prepare_scale_correction_groups([0, 1], explicit, n_images=1)
    assert "group_ids must have shape" not in str(error.value)


def test_group_shape_is_checked_before_negative_ids():
    with pytest.raises(ValueError, match=r"group_ids must have shape \(1,\), got \(2,\)"):
        prepare_scale_correction_groups([-1, 0], n_images=1)


def test_negative_ids_are_rejected():
    with pytest.raises(ValueError, match="group_ids must be non-negative"):
        prepare_scale_correction_groups([-1, 0], n_images=2)


def test_existing_int64_conversion_and_input_storage_are_preserved():
    ids = np.asarray([0, 2, 0, 4], dtype=np.int64)
    normalized, count = prepare_scale_correction_groups(ids, n_images=4)
    assert np.shares_memory(ids, normalized)
    np.testing.assert_array_equal(ids, [0, 2, 0, 4])
    assert count == 5
    # The existing boundary casts IDs; this cleanup must not tighten it silently.
    normalized, count = prepare_scale_correction_groups([0.9, 2.9], n_images=2)
    np.testing.assert_array_equal(normalized, [0, 2])
    assert count == 3


def test_class_subset_keeps_absent_scale_groups():
    full_count = _full_group_count_from_kwargs(dict(group_ids=[0, 4, 2, 4]))
    subset, count = prepare_scale_correction_groups([0, 2], full_count, n_images=2)
    np.testing.assert_array_equal(subset, [0, 2])
    assert count == full_count == 5
