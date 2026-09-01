"""Host contracts for compact fixed-capacity LocalBucketSpec payloads."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    _fixed_capacity_calls_from_local_buckets,
    _pack_fixed_capacity_local_hypothesis_program,
)

pytestmark = pytest.mark.unit


def _bucket(
    image_indices,
    row_counts,
    *,
    radix,
    image_capacity,
    with_mstep=True,
    with_posterior=True,
    with_sample_mask=True,
):
    image_indices = np.asarray(image_indices, dtype=np.int32)
    row_counts = np.asarray(row_counts, dtype=np.int32)
    n_images = int(image_indices.size)
    n_translations = 3
    candidate_columns = np.arange(radix, dtype=np.int32)[None, :]
    rotation_mask = candidate_columns < row_counts[:, None]

    rotation_ids = np.full((n_images, radix), -1, dtype=np.int64)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (n_images, radix, 3, 3),
    ).copy()
    mstep_rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (n_images, radix, 3, 3),
    ).copy()
    rotation_log_prior = np.full((n_images, radix), -1e30, dtype=np.float32)
    posterior_ids = np.full((n_images, radix), -1, dtype=np.int16)
    sample_mask = np.zeros((n_images, radix, n_translations), dtype=bool)
    translation_log_prior = np.empty((n_images, n_translations), dtype=np.float64)

    for row, (image_id, count) in enumerate(zip(image_indices.tolist(), row_counts.tolist(), strict=True)):
        rotation_ids[row, :count] = image_id * 100 + np.arange(count, dtype=np.int64)
        rotations[row, :count] = np.arange(count * 9, dtype=np.float64).reshape(count, 3, 3) + image_id * 1000
        mstep_rotations[row, :count] = np.arange(count * 9, dtype=np.float32).reshape(count, 3, 3) + image_id * 2000
        rotation_log_prior[row, :count] = np.arange(count, dtype=np.float32) * np.float32(-0.125) + np.float32(image_id)
        posterior_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int16)
        sample_mask[row, :count] = (
            np.arange(count, dtype=np.int32)[:, None] + np.arange(n_translations, dtype=np.int32)[None, :]
        ) % 2 == 0
        translation_log_prior[row] = image_id + np.asarray([0.25, -0.5, 1.75], dtype=np.float64)

    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=image_capacity,
        bucket_rotation_count=radix,
        actual_rotation_counts=row_counts,
        local_rotation_ids=rotation_ids,
        local_rotations=rotations,
        local_mstep_rotations=mstep_rotations if with_mstep else None,
        local_rotation_log_prior=rotation_log_prior,
        local_rotation_mask=rotation_mask,
        translation_log_prior=translation_log_prior,
        local_rotation_posterior_ids=posterior_ids if with_posterior else None,
        local_sample_mask=sample_mask if with_sample_mask else None,
    )


def _buckets(*, with_mstep=True, with_posterior=True, with_sample_mask=True):
    kwargs = {
        "with_mstep": with_mstep,
        "with_posterior": with_posterior,
        "with_sample_mask": with_sample_mask,
    }
    return (
        _bucket([9, 3], [2, 4], radix=4, image_capacity=2, **kwargs),
        _bucket([7], [1], radix=4, image_capacity=2, **kwargs),
        _bucket([8, 1], [5, 8], radix=8, image_capacity=2, **kwargs),
    )


def _sealed_program(*, buckets=None):
    if buckets is None:
        buckets = _buckets()
    expected_order = _seal_fixed_capacity_physical_order(
        np.concatenate([bucket.image_indices for bucket in buckets]),
    )
    calls = _fixed_capacity_calls_from_local_buckets(
        buckets,
        expected_order=expected_order,
    )
    plan = _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=expected_order,
        physical_image_capacity=8,
        physical_row_capacity=32,
        physical_call_capacity=6,
        image_capacity_palette={4: (2,), 8: (2,)},
        logical_cutoff=70,
        logical_cutoff_capacity=128,
        enabled=True,
    )
    return buckets, expected_order, plan


def _assert_bitwise_equal(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def test_fixed_capacity_hypothesis_packer_is_shared_default_off_and_inert():
    assert _pack_fixed_capacity_local_hypothesis_program.__module__ == "recovar.em.dense_single_volume.local_layout"
    assert inspect.signature(_pack_fixed_capacity_local_hypothesis_program).parameters["enabled"].default is False
    assert _pack_fixed_capacity_local_hypothesis_program(None, None, None) is None


def test_fixed_capacity_hypothesis_packer_bitwise_round_trips_every_active_field():
    buckets, expected_order, plan = _sealed_program()
    program = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        expected_order,
        enabled=True,
    )

    assert program.valid_image_count == 5
    assert program.valid_row_count == 20
    assert program.mstep_rotations_fall_back_to_score is False
    np.testing.assert_array_equal(program.image_indices, [9, 3, 7, 8, 1, -1, -1, -1])
    np.testing.assert_array_equal(program.row_offsets, [0, 2, 6, 7, 12, 20, 20, 20, 20])
    np.testing.assert_array_equal(
        program.valid_image_mask,
        [True, True, True, True, True, False, False, False],
    )
    np.testing.assert_array_equal(
        program.valid_candidate_row_mask,
        [True] * 20 + [False] * 12,
    )

    physical_image = 0
    for bucket in buckets:
        for local_image, count in enumerate(bucket.actual_rotation_counts.tolist()):
            row_start = int(program.row_offsets[physical_image])
            row_stop = int(program.row_offsets[physical_image + 1])
            assert row_stop - row_start == count
            _assert_bitwise_equal(
                program.local_rotation_ids[row_start:row_stop],
                bucket.local_rotation_ids[local_image, :count],
            )
            _assert_bitwise_equal(
                program.local_rotations[row_start:row_stop],
                bucket.local_rotations[local_image, :count],
            )
            _assert_bitwise_equal(
                program.local_mstep_rotations[row_start:row_stop],
                bucket.local_mstep_rotations[local_image, :count],
            )
            _assert_bitwise_equal(
                program.local_rotation_log_prior[row_start:row_stop],
                bucket.local_rotation_log_prior[local_image, :count],
            )
            _assert_bitwise_equal(
                program.translation_log_prior[physical_image],
                bucket.translation_log_prior[local_image],
            )
            _assert_bitwise_equal(
                program.local_rotation_posterior_ids[row_start:row_stop],
                bucket.local_rotation_posterior_ids[local_image, :count],
            )
            _assert_bitwise_equal(
                program.local_sample_mask[row_start:row_stop],
                bucket.local_sample_mask[local_image, :count],
            )
            physical_image += 1

    assert sum(np.asarray(bucket.local_rotation_ids).size for bucket in buckets) == 28
    assert program.valid_row_count == 20
    assert program.local_rotation_ids.dtype == np.dtype(np.int64)
    assert program.local_rotations.dtype == np.dtype(np.float64)
    assert program.local_mstep_rotations.dtype == np.dtype(np.float32)
    assert program.local_rotation_log_prior.dtype == np.dtype(np.float32)
    assert program.translation_log_prior.dtype == np.dtype(np.float64)
    assert program.local_rotation_posterior_ids.dtype == np.dtype(np.int16)


def test_fixed_capacity_hypothesis_packer_snapshots_arrays_and_makes_them_read_only():
    buckets, expected_order, plan = _sealed_program()
    expected_first_rotation = buckets[0].local_rotations[0, 0].copy()
    expected_first_translation = buckets[0].translation_log_prior[0].copy()
    program = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        expected_order,
        enabled=True,
    )

    buckets[0].local_rotations[:] = 0
    buckets[0].translation_log_prior[:] = 0
    _assert_bitwise_equal(program.local_rotations[0], expected_first_rotation)
    _assert_bitwise_equal(program.translation_log_prior[0], expected_first_translation)

    array_fields = (
        "image_indices",
        "row_offsets",
        "valid_image_mask",
        "valid_candidate_row_mask",
        "local_rotation_ids",
        "local_rotations",
        "local_mstep_rotations",
        "local_rotation_log_prior",
        "translation_log_prior",
        "local_rotation_posterior_ids",
        "local_sample_mask",
    )
    for field_name in array_fields:
        assert getattr(program, field_name).flags.writeable is False


def test_fixed_capacity_hypothesis_packer_poisoned_tails_are_masked_and_inert():
    buckets, expected_order, plan = _sealed_program()
    program = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        expected_order,
        enabled=True,
    )

    candidate_tail = slice(program.valid_row_count, program.physical_row_capacity)
    image_tail = slice(program.valid_image_count, program.physical_image_capacity)
    assert not np.any(program.valid_candidate_row_mask[candidate_tail])
    assert not np.any(program.valid_image_mask[image_tail])
    assert np.all(program.local_rotation_ids[candidate_tail] == -1)
    assert np.all(np.isnan(program.local_rotations[candidate_tail]))
    assert np.all(np.isnan(program.local_mstep_rotations[candidate_tail]))
    assert np.all(np.isneginf(program.local_rotation_log_prior[candidate_tail]))
    assert np.all(np.isneginf(program.translation_log_prior[image_tail]))
    assert np.all(program.local_rotation_posterior_ids[candidate_tail] == -1)
    assert not np.any(program.local_sample_mask[candidate_tail])


def test_fixed_capacity_hypothesis_packer_resolves_uniform_mstep_fallback_and_absent_optionals():
    buckets = _buckets(
        with_mstep=False,
        with_posterior=False,
        with_sample_mask=False,
    )
    buckets, expected_order, plan = _sealed_program(buckets=buckets)
    program = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        expected_order,
        enabled=True,
    )

    assert program.mstep_rotations_fall_back_to_score is True
    assert program.local_rotation_posterior_ids is None
    assert program.local_sample_mask is None
    _assert_bitwise_equal(
        program.local_mstep_rotations[: program.valid_row_count],
        program.local_rotations[: program.valid_row_count],
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "local_mstep_rotations",
        "local_rotation_posterior_ids",
        "local_sample_mask",
    ),
)
def test_fixed_capacity_hypothesis_packer_rejects_mixed_optional_topology(field_name):
    buckets, expected_order, plan = _sealed_program()
    changed = list(buckets)
    changed[1] = replace(changed[1], **{field_name: None})

    with pytest.raises(ValueError, match=f"mixed optional topology for {field_name}"):
        _pack_fixed_capacity_local_hypothesis_program(
            changed,
            plan,
            expected_order,
            enabled=True,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("valid_counts", "plan/bucket valid counts"),
        ("image_indices", "plan/bucket mismatch in image_indices"),
        ("row_offsets", "plan/bucket mismatch in row_offsets"),
        ("call_boundaries", "plan/bucket mismatch in call_valid_images"),
        ("image_capacity", "image capacity overflow"),
        ("row_capacity", "candidate-row capacity overflow"),
        ("call_capacity", "call capacity overflow"),
        ("noninteger_capacity", "physical_row_capacity must be an integer"),
    ),
)
def test_fixed_capacity_hypothesis_packer_rejects_plan_bucket_mismatch_or_overflow(change, message):
    buckets, expected_order, plan = _sealed_program()
    if change == "valid_counts":
        plan = replace(plan, valid_row_count=19)
    elif change == "image_indices":
        image_indices = plan.image_indices.copy()
        image_indices[:2] = image_indices[1::-1]
        plan = replace(plan, image_indices=image_indices)
    elif change == "row_offsets":
        row_offsets = plan.row_offsets.copy()
        row_offsets[2] += 1
        plan = replace(plan, row_offsets=row_offsets)
    elif change == "call_boundaries":
        call_valid_images = plan.call_valid_images.copy()
        call_valid_images[:2] = [1, 2]
        plan = replace(plan, call_valid_images=call_valid_images)
    elif change == "image_capacity":
        plan = replace(plan, physical_image_capacity=4)
    elif change == "row_capacity":
        plan = replace(plan, physical_row_capacity=19)
    elif change == "call_capacity":
        plan = replace(plan, physical_call_capacity=2)
    elif change == "noninteger_capacity":
        plan = replace(plan, physical_row_capacity=32.5)

    with pytest.raises(ValueError, match=message):
        _pack_fixed_capacity_local_hypothesis_program(
            buckets,
            plan,
            expected_order,
            enabled=True,
        )


def test_fixed_capacity_hypothesis_packer_rejects_outer_call_chronology_drift():
    buckets, expected_order, plan = _sealed_program()

    with pytest.raises(ValueError, match="chronology does not match the sealed physical order"):
        _pack_fixed_capacity_local_hypothesis_program(
            tuple(reversed(buckets)),
            plan,
            expected_order,
            enabled=True,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "local_rotation_ids",
        "local_rotations",
        "local_mstep_rotations",
        "local_rotation_log_prior",
        "local_rotation_posterior_ids",
        "local_sample_mask",
    ),
)
def test_fixed_capacity_hypothesis_packer_rejects_malformed_source_poison_tails(field_name):
    buckets, expected_order, plan = _sealed_program()
    changed = list(buckets)
    bucket = changed[0]
    values = np.asarray(getattr(bucket, field_name)).copy()
    inactive_row = (0, 2)
    if field_name in {"local_rotations", "local_mstep_rotations"}:
        values[inactive_row] = 0
    elif field_name == "local_sample_mask":
        values[inactive_row] = True
    else:
        values[inactive_row] = 0
    changed[0] = replace(bucket, **{field_name: values})

    with pytest.raises(ValueError, match=f"malformed {field_name} poison tails"):
        _pack_fixed_capacity_local_hypothesis_program(
            changed,
            plan,
            expected_order,
            enabled=True,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "local_rotation_ids",
            np.zeros((2, 3), dtype=np.int32),
            "local_rotation_ids has shape",
        ),
        (
            "local_rotations",
            np.zeros((2, 4, 3, 3), dtype=np.int32),
            "local_rotations must be floating point",
        ),
        (
            "translation_log_prior",
            np.zeros((2,), dtype=np.float32),
            "translation_log_prior must be a nonempty floating-point matrix",
        ),
        (
            "local_sample_mask",
            np.zeros((2, 4, 2), dtype=bool),
            "local_sample_mask has shape",
        ),
    ),
)
def test_fixed_capacity_hypothesis_packer_rejects_malformed_payloads(field_name, value, message):
    buckets, expected_order, plan = _sealed_program()
    changed = list(buckets)
    changed[0] = replace(changed[0], **{field_name: value})

    with pytest.raises(ValueError, match=message):
        _pack_fixed_capacity_local_hypothesis_program(
            changed,
            plan,
            expected_order,
            enabled=True,
        )
