"""Exact host contracts for LocalBucketSpec fixed-capacity conversion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _pack_fixed_capacity_local_candidate_rows,
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    _fixed_capacity_calls_from_local_buckets,
)

pytestmark = pytest.mark.unit


def _bucket(image_indices, row_counts, *, radix, image_capacity):
    image_indices = np.asarray(image_indices, dtype=np.int32)
    row_counts = np.asarray(row_counts, dtype=np.int32)
    n_images = int(image_indices.size)
    candidate_rows = np.arange(radix, dtype=np.int32)[None, :]
    rotation_mask = candidate_rows < row_counts[:, None]
    rotation_ids = np.where(
        rotation_mask,
        image_indices[:, None] * 100 + candidate_rows,
        -1,
    ).astype(np.int32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (n_images, radix, 3, 3),
    ).copy()
    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=image_capacity,
        bucket_rotation_count=radix,
        actual_rotation_counts=row_counts,
        local_rotation_ids=rotation_ids,
        local_rotations=rotations,
        local_rotation_log_prior=np.where(rotation_mask, 0.0, -1e30).astype(np.float32),
        local_rotation_mask=rotation_mask,
        translation_log_prior=np.zeros((n_images, 1), dtype=np.float32),
    )


def _authoritative_buckets():
    return (
        _bucket([9, 3, 7], [2, 4, 3], radix=4, image_capacity=3),
        _bucket([8], [1], radix=4, image_capacity=3),
        _bucket([1, 5], [8, 5], radix=8, image_capacity=2),
    )


def _sealed_order(image_indices=(9, 3, 7, 8, 1, 5)):
    return _seal_fixed_capacity_physical_order(np.asarray(image_indices, dtype=np.int32))


def _plan_converted(calls, *, expected_order=None, palette=None):
    if expected_order is None:
        expected_order = _sealed_order()
    if palette is None:
        palette = {4: (3,), 8: (2,)}
    return _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=expected_order,
        physical_image_capacity=8,
        physical_row_capacity=32,
        physical_call_capacity=5,
        image_capacity_palette=palette,
        logical_cutoff=70,
        logical_cutoff_capacity=128,
        enabled=True,
    )


def test_local_bucket_conversion_preserves_full_tail_chronology_radix_and_rows():
    buckets = _authoritative_buckets()
    expected_order = _sealed_order()
    calls = _fixed_capacity_calls_from_local_buckets(
        buckets,
        expected_order=expected_order,
    )

    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0].image_indices, [9, 3, 7])
    np.testing.assert_array_equal(calls[1].image_indices, [8])
    np.testing.assert_array_equal(calls[2].image_indices, [1, 5])
    np.testing.assert_array_equal(calls[0].row_counts, [2, 4, 3])
    np.testing.assert_array_equal(calls[1].row_counts, [1])
    np.testing.assert_array_equal(calls[2].row_counts, [8, 5])
    assert [call.radix_bucket for call in calls] == [4, 4, 8]
    assert [call.image_capacity for call in calls] == [3, 3, 2]

    plan = _plan_converted(calls, expected_order=expected_order)
    np.testing.assert_array_equal(plan.image_indices, [9, 3, 7, 8, 1, 5, -1, -1])
    np.testing.assert_array_equal(plan.row_offsets, [0, 2, 6, 9, 10, 18, 23, 23, 23])
    np.testing.assert_array_equal(plan.call_image_offsets, [0, 3, 4, 6, 6])
    np.testing.assert_array_equal(plan.call_row_offsets, [0, 9, 10, 23, 23])
    np.testing.assert_array_equal(plan.call_valid_images, [3, 1, 2, 0, 0])
    np.testing.assert_array_equal(plan.call_valid_rows, [9, 1, 13, 0, 0])
    np.testing.assert_array_equal(plan.call_image_capacities, [3, 3, 2, 0, 0])
    np.testing.assert_array_equal(plan.call_radix_buckets, [4, 4, 8, 0, 0])
    packed_candidate_ids = _pack_fixed_capacity_local_candidate_rows(
        plan,
        [bucket.local_rotation_ids for bucket in buckets],
        fill_value=-999,
    )
    expected_candidate_ids = np.concatenate(
        [
            bucket.local_rotation_ids[row, : int(row_count)]
            for bucket in buckets
            for row, row_count in enumerate(bucket.actual_rotation_counts)
        ]
    )
    np.testing.assert_array_equal(packed_candidate_ids[:23], expected_candidate_ids)
    np.testing.assert_array_equal(packed_candidate_ids[23:], np.full(9, -999, dtype=np.int32))


def test_local_bucket_conversion_snapshots_mutable_bucket_arrays():
    bucket = _bucket([9, 3], [2, 3], radix=4, image_capacity=3)
    calls = _fixed_capacity_calls_from_local_buckets(
        (bucket,),
        expected_order=_sealed_order((9, 3)),
    )

    bucket.image_indices[:] = -1
    bucket.actual_rotation_counts[:] = 1

    np.testing.assert_array_equal(calls[0].image_indices, [9, 3])
    np.testing.assert_array_equal(calls[0].row_counts, [2, 3])


def test_converted_tail_keeps_authoritative_capacity_instead_of_smallest_palette_entry():
    tail = (_bucket([8], [1], radix=4, image_capacity=3),)
    expected_order = _sealed_order((8,))
    calls = _fixed_capacity_calls_from_local_buckets(
        tail,
        expected_order=expected_order,
    )
    plan = _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=expected_order,
        physical_image_capacity=1,
        physical_row_capacity=4,
        physical_call_capacity=1,
        image_capacity_palette={4: (1, 3)},
        logical_cutoff=16,
        logical_cutoff_capacity=32,
        enabled=True,
    )

    np.testing.assert_array_equal(plan.call_image_capacities, [3])


def test_converted_plan_rejects_chronology_or_capacity_palette_changes():
    buckets = _authoritative_buckets()
    expected_order = _sealed_order()

    for invalid_buckets in (tuple(reversed(buckets)), buckets[:-1]):
        with pytest.raises(ValueError, match="chronology does not match the sealed physical order"):
            _fixed_capacity_calls_from_local_buckets(
                invalid_buckets,
                expected_order=expected_order,
            )

    calls = _fixed_capacity_calls_from_local_buckets(
        buckets,
        expected_order=expected_order,
    )
    with pytest.raises(ValueError, match="preserved image capacity"):
        _plan_converted(
            calls,
            expected_order=expected_order,
            palette={4: (1,), 8: (2,)},
        )


def test_converted_plan_rejects_duplicate_physical_images():
    with pytest.raises(ValueError, match="must not contain duplicate image IDs"):
        _sealed_order((9, 3, 7, 7, 1, 5))


def test_physical_order_seal_is_an_independent_immutable_snapshot():
    source = np.asarray([9, 3, 7], dtype=np.int32)
    expected_order = _seal_fixed_capacity_physical_order(source)

    source[:] = -1

    np.testing.assert_array_equal(expected_order.image_indices, [9, 3, 7])
    assert expected_order.image_indices.flags.writeable is False


def test_local_bucket_conversion_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="bucket sequence cannot be empty"):
        _fixed_capacity_calls_from_local_buckets(
            (),
            expected_order=_sealed_order((9,)),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("empty_bucket", "bucket 0 cannot be empty"),
        ("row_axis", "row counts must match"),
        ("image_capacity", "image capacity is smaller"),
        ("radix", "radix must be positive"),
        ("row_count", "row count is outside"),
        ("zero_row_count", "row count is outside"),
        ("mask_shape", "rotation mask must match"),
        ("mask_membership", "candidate membership must be a dense ordered prefix"),
        ("noninteger_image_capacity", "bucket_image_count must be an integer"),
        ("noninteger_radix", "bucket_rotation_count must be an integer"),
    ),
)
def test_local_bucket_conversion_fails_closed_on_unsupported_topology(change, message):
    bucket = _bucket([9, 3], [2, 3], radix=4, image_capacity=3)
    if change == "empty_bucket":
        bucket = _bucket([], [], radix=4, image_capacity=3)
    elif change == "row_axis":
        bucket = replace(bucket, actual_rotation_counts=np.asarray([2], dtype=np.int32))
    elif change == "image_capacity":
        bucket = replace(bucket, bucket_image_count=1)
    elif change == "radix":
        bucket = replace(bucket, bucket_rotation_count=0)
    elif change == "row_count":
        bucket = replace(bucket, actual_rotation_counts=np.asarray([5, 3], dtype=np.int32))
    elif change == "zero_row_count":
        bucket = replace(bucket, actual_rotation_counts=np.asarray([0, 3], dtype=np.int32))
    elif change == "mask_shape":
        bucket = replace(bucket, local_rotation_mask=bucket.local_rotation_mask[:, :3])
    elif change == "mask_membership":
        mask = bucket.local_rotation_mask.copy()
        mask[0, 0] = False
        bucket = replace(bucket, local_rotation_mask=mask)
    elif change == "noninteger_image_capacity":
        bucket = replace(bucket, bucket_image_count=3.5)
    elif change == "noninteger_radix":
        bucket = replace(bucket, bucket_rotation_count=4.0)

    with pytest.raises(ValueError, match=message):
        _fixed_capacity_calls_from_local_buckets(
            (bucket,),
            expected_order=_sealed_order((9, 3)),
        )
