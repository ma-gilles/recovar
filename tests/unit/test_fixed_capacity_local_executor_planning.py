"""Host contracts for the shared fixed-capacity whole-local executor seam."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _fixed_capacity_plan_descriptor_fingerprint,
    _FixedCapacityLocalCall,
    _pack_fixed_capacity_local_candidate_rows,
    _pack_fixed_capacity_local_images,
    _plan_fixed_capacity_whole_local,
)

pytestmark = pytest.mark.unit


def _calls():
    return (
        _FixedCapacityLocalCall(
            image_indices=np.asarray([9, 3], dtype=np.int32),
            row_counts=np.asarray([2, 4], dtype=np.int32),
            radix_bucket=4,
        ),
        _FixedCapacityLocalCall(
            image_indices=np.asarray([7], dtype=np.int32),
            row_counts=np.asarray([1], dtype=np.int32),
            radix_bucket=4,
        ),
        _FixedCapacityLocalCall(
            image_indices=np.asarray([8, 1], dtype=np.int32),
            row_counts=np.asarray([5, 8], dtype=np.int32),
            radix_bucket=8,
        ),
    )


def _plan(*, calls=None, logical_cutoff=70, enabled=True):
    if calls is None:
        calls = _calls()
    return _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=np.concatenate(
            [np.asarray(call.image_indices) for call in calls],
        ),
        physical_image_capacity=8,
        physical_row_capacity=32,
        physical_call_capacity=6,
        image_capacity_palette={4: (2,), 8: (2, 4)},
        logical_cutoff=logical_cutoff,
        logical_cutoff_capacity=128,
        enabled=enabled,
    )


def test_fixed_capacity_whole_local_seam_is_shared_and_default_off():
    assert _plan_fixed_capacity_whole_local.__module__ == "recovar.em.dense_single_volume.batch_planning"
    assert inspect.signature(_plan_fixed_capacity_whole_local).parameters["enabled"].default is False
    assert _plan(enabled=False) is None


def test_fixed_capacity_plan_rejects_an_empty_call_program():
    with pytest.raises(ValueError, match="call program cannot be empty"):
        _plan_fixed_capacity_whole_local(
            (),
            expected_image_order=np.zeros(0, dtype=np.int32),
            physical_image_capacity=1,
            physical_row_capacity=1,
            physical_call_capacity=1,
            image_capacity_palette={4: (1,)},
            logical_cutoff=1,
            logical_cutoff_capacity=1,
            enabled=True,
        )


def test_authoritative_fixed_capacity_calls_require_an_independent_order_seal():
    call = _FixedCapacityLocalCall(
        image_indices=np.asarray([4], dtype=np.int32),
        row_counts=np.asarray([2], dtype=np.int32),
        radix_bucket=2,
        image_capacity=1,
    )

    with pytest.raises(ValueError, match="independently sealed expected physical order"):
        _plan_fixed_capacity_whole_local(
            (call,),
            expected_image_order=np.asarray([4], dtype=np.int32),
            physical_image_capacity=1,
            physical_row_capacity=2,
            physical_call_capacity=1,
            image_capacity_palette={2: (1,)},
            logical_cutoff=1,
            logical_cutoff_capacity=1,
            enabled=True,
        )


def test_fixed_capacity_plan_preserves_call_particle_and_radix_chronology():
    plan = _plan()

    assert plan.valid_call_count == 3
    assert plan.valid_image_count == 5
    assert plan.valid_row_count == 20
    np.testing.assert_array_equal(plan.image_indices, [9, 3, 7, 8, 1, -1, -1, -1])
    np.testing.assert_array_equal(plan.row_offsets, [0, 2, 6, 7, 12, 20, 20, 20, 20])
    np.testing.assert_array_equal(plan.call_valid_mask, [True, True, True, False, False, False])
    np.testing.assert_array_equal(plan.call_image_offsets, [0, 2, 3, 5, 5, 5])
    np.testing.assert_array_equal(plan.call_row_offsets, [0, 6, 7, 20, 20, 20])
    np.testing.assert_array_equal(plan.call_valid_images, [2, 1, 2, 0, 0, 0])
    np.testing.assert_array_equal(plan.call_valid_rows, [6, 1, 13, 0, 0, 0])
    np.testing.assert_array_equal(plan.call_image_capacities, [2, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(plan.call_radix_buckets, [4, 4, 8, 0, 0, 0])
    assert plan.logical_cutoff.shape == ()
    assert int(plan.logical_cutoff) == 70


def test_fixed_capacity_plan_descriptors_are_snapshotted_read_only_and_fingerprinted():
    plan = _plan()
    same_plan = _plan()
    changed_cutoff = _plan(logical_cutoff=71)

    descriptor_arrays = (
        "image_indices",
        "row_offsets",
        "call_valid_mask",
        "call_image_offsets",
        "call_row_offsets",
        "call_valid_images",
        "call_valid_rows",
        "call_image_capacities",
        "call_radix_buckets",
        "logical_cutoff",
    )
    for field_name in descriptor_arrays:
        assert getattr(plan, field_name).flags.writeable is False
    assert plan.descriptor_fingerprint == _fixed_capacity_plan_descriptor_fingerprint(plan)
    assert plan.descriptor_fingerprint == same_plan.descriptor_fingerprint
    assert plan.descriptor_fingerprint != changed_cutoff.descriptor_fingerprint

    source = plan.image_indices.copy()
    snapshotted = replace(plan, image_indices=source)
    source[:] = -1
    np.testing.assert_array_equal(snapshotted.image_indices, plan.image_indices)


@pytest.mark.parametrize(
    ("field_name", "wrong_dtype"),
    (
        ("image_indices", np.float64),
        ("image_indices", np.int32),
        ("row_offsets", np.float64),
        ("row_offsets", np.int32),
        ("call_valid_mask", np.int8),
        ("call_image_offsets", np.float32),
        ("call_image_offsets", np.int64),
        ("call_row_offsets", np.int32),
        ("call_valid_images", np.int64),
        ("call_valid_rows", np.int32),
        ("call_image_capacities", np.int64),
        ("call_radix_buckets", np.int64),
        ("logical_cutoff", np.int64),
    ),
)
def test_fixed_capacity_plan_rejects_noncanonical_descriptor_dtypes(field_name, wrong_dtype):
    plan = _plan()
    wrong = np.asarray(getattr(plan, field_name)).astype(wrong_dtype)

    with pytest.raises(ValueError, match=f"{field_name} must have canonical dtype"):
        replace(plan, **{field_name: wrong})


@pytest.mark.parametrize(
    "field_name",
    (
        "physical_image_capacity",
        "physical_row_capacity",
        "physical_call_capacity",
        "logical_cutoff_capacity",
        "valid_image_count",
        "valid_row_count",
        "valid_call_count",
    ),
)
def test_fixed_capacity_plan_rejects_noninteger_scalar_descriptors(field_name):
    plan = _plan()

    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        replace(plan, **{field_name: float(getattr(plan, field_name))})


@pytest.mark.parametrize(
    "field_name",
    (
        "image_indices",
        "row_offsets",
        "call_valid_mask",
        "call_image_offsets",
        "call_row_offsets",
        "call_valid_images",
        "call_valid_rows",
        "call_image_capacities",
        "call_radix_buckets",
        "logical_cutoff",
    ),
)
def test_fixed_capacity_plan_rejects_noncanonical_descriptor_shapes(field_name):
    plan = _plan()
    source = np.asarray(getattr(plan, field_name))
    wrong = np.asarray([source.item()], dtype=source.dtype) if source.ndim == 0 else source[:-1]

    with pytest.raises(ValueError, match=f"{field_name} must have canonical shape"):
        replace(plan, **{field_name: wrong})


def test_fixed_capacity_packers_keep_real_rows_and_poison_only_inert_tails():
    calls = _calls()
    plan = _plan(calls=calls)
    image_values = [
        np.asarray([[90, 91], [30, 31]], dtype=np.int32),
        np.asarray([[70, 71]], dtype=np.int32),
        np.asarray([[80, 81], [10, 11]], dtype=np.int32),
    ]
    candidate_values = []
    expected_candidate_rows = []
    next_value = 100
    for call in calls:
        value = np.full(
            (call.image_indices.size, call.radix_bucket, 2),
            -777,
            dtype=np.int32,
        )
        for image_index, row_count in enumerate(call.row_counts.tolist()):
            real = np.arange(
                next_value,
                next_value + int(row_count) * 2,
                dtype=np.int32,
            ).reshape(int(row_count), 2)
            next_value += int(row_count) * 2
            value[image_index, : int(row_count)] = real
            expected_candidate_rows.append(real)
        candidate_values.append(value)

    packed_images = _pack_fixed_capacity_local_images(
        plan,
        image_values,
        fill_value=-999,
    )
    packed_candidates = _pack_fixed_capacity_local_candidate_rows(
        plan,
        candidate_values,
        fill_value=-999,
    )

    np.testing.assert_array_equal(
        packed_images[: plan.valid_image_count],
        np.concatenate(image_values),
    )
    assert np.all(packed_images[plan.valid_image_count :] == -999)
    np.testing.assert_array_equal(
        packed_candidates[: plan.valid_row_count],
        np.concatenate(expected_candidate_rows),
    )
    assert np.all(packed_candidates[plan.valid_row_count :] == -999)
    assert not np.any(packed_candidates[: plan.valid_row_count] == -777)


def test_full_and_tail_programs_share_physical_shapes_with_runtime_counts_and_cutoff():
    full = _plan(logical_cutoff=70)
    tail_calls = (
        _FixedCapacityLocalCall(
            image_indices=np.asarray([9], dtype=np.int32),
            row_counts=np.asarray([2], dtype=np.int32),
            radix_bucket=4,
        ),
        _FixedCapacityLocalCall(
            image_indices=np.asarray([3, 7], dtype=np.int32),
            row_counts=np.asarray([4, 1], dtype=np.int32),
            radix_bucket=4,
        ),
        _FixedCapacityLocalCall(
            image_indices=np.asarray([8], dtype=np.int32),
            row_counts=np.asarray([5], dtype=np.int32),
            radix_bucket=8,
        ),
    )
    tail = _plan(calls=tail_calls, logical_cutoff=72)

    fixed_array_names = (
        "image_indices",
        "row_offsets",
        "call_valid_mask",
        "call_image_offsets",
        "call_row_offsets",
        "call_valid_images",
        "call_valid_rows",
        "call_image_capacities",
        "call_radix_buckets",
    )
    assert {name: getattr(full, name).shape for name in fixed_array_names} == {
        name: getattr(tail, name).shape for name in fixed_array_names
    }
    np.testing.assert_array_equal(full.call_image_capacities[:3], [2, 2, 2])
    np.testing.assert_array_equal(tail.call_image_capacities[:3], [2, 2, 2])
    np.testing.assert_array_equal(full.call_valid_images[:3], [2, 1, 2])
    np.testing.assert_array_equal(tail.call_valid_images[:3], [1, 2, 1])
    assert int(full.logical_cutoff) == 70
    assert int(tail.logical_cutoff) == 72


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("chronology", "expected physical image order"),
        ("row_count", "no larger than radix bucket"),
        ("palette", "palette has no entry"),
        ("image_capacity", "image storage overflow"),
        ("row_capacity", "candidate-row storage overflow"),
        ("call_capacity", "call program overflow"),
        ("cutoff", "logical_cutoff"),
    ),
)
def test_fixed_capacity_plan_fails_closed_on_unsupported_topology(change, message):
    calls = list(_calls())
    kwargs = {
        "calls": calls,
        "expected_image_order": np.asarray([9, 3, 7, 8, 1]),
        "physical_image_capacity": 8,
        "physical_row_capacity": 32,
        "physical_call_capacity": 6,
        "image_capacity_palette": {4: (2,), 8: (2,)},
        "logical_cutoff": 70,
        "logical_cutoff_capacity": 128,
        "enabled": True,
    }
    if change == "chronology":
        kwargs["expected_image_order"] = np.asarray([9, 7, 3, 8, 1])
    elif change == "row_count":
        calls[0] = _FixedCapacityLocalCall(
            image_indices=np.asarray([9, 3]),
            row_counts=np.asarray([5, 4]),
            radix_bucket=4,
        )
    elif change == "palette":
        kwargs["image_capacity_palette"] = {4: (2,)}
    elif change == "image_capacity":
        kwargs["physical_image_capacity"] = 4
    elif change == "row_capacity":
        kwargs["physical_row_capacity"] = 19
    elif change == "call_capacity":
        kwargs["physical_call_capacity"] = 2
    elif change == "cutoff":
        kwargs["logical_cutoff"] = 129

    with pytest.raises(ValueError, match=message):
        _plan_fixed_capacity_whole_local(**kwargs)


def test_candidate_packer_rejects_wrong_radix_shape_before_execution():
    plan = _plan()
    wrong = [
        np.zeros((2, 3), dtype=np.int32),
        np.zeros((1, 4), dtype=np.int32),
        np.zeros((2, 8), dtype=np.int32),
    ]

    with pytest.raises(ValueError, match="unexpected leading shape"):
        _pack_fixed_capacity_local_candidate_rows(plan, wrong)
