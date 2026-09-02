"""Host contracts for fingerprint-bound fixed-capacity local components."""

from __future__ import annotations

import inspect
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_big_jit, local_em_engine
from recovar.em.dense_single_volume.batch_planning import (
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.fixed_capacity_local import (
    _bind_fixed_capacity_local_execution,
    _materialize_fixed_capacity_active_local_rows,
    _materialize_fixed_capacity_local_call_view,
)
from recovar.em.dense_single_volume.local_caches import _assemble_fixed_capacity_local_operands_once
from recovar.em.dense_single_volume.local_layout import (
    LocalBucketSpec,
    _fixed_capacity_calls_from_local_buckets,
    _pack_fixed_capacity_local_hypothesis_program,
)

pytestmark = pytest.mark.unit


class _IndexedDataset:
    def __init__(self):
        self.images = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
        self.ctf_params = np.arange(3 * 3, dtype=np.float32).reshape(3, 3) + 100

    def iter_batches(self, batch_size, *, indices, by_image):
        assert by_image is False
        indices = np.asarray(indices, dtype=np.int64)
        yield self.images[indices], None, None, self.ctf_params[indices], None, None, indices


def _bucket(image_indices, row_counts, *, radix, image_capacity, include_optional=True):
    image_indices = np.asarray(image_indices, dtype=np.int32)
    row_counts = np.asarray(row_counts, dtype=np.int32)
    n_images = int(image_indices.size)
    rotation_mask = np.arange(radix, dtype=np.int32)[None, :] < row_counts[:, None]
    rotation_ids = np.full((n_images, radix), -1, dtype=np.int32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, radix, 3, 3)).copy()
    mstep_rotations = rotations.copy()
    rotation_log_prior = np.full((n_images, radix), -1e30, dtype=np.float32)
    posterior_ids = np.full((n_images, radix), -1, dtype=np.int32) if include_optional else None
    sample_mask = np.zeros((n_images, radix, 2), dtype=bool) if include_optional else None
    translation_log_prior = np.empty((n_images, 2), dtype=np.float32)
    for row, (image_id, count) in enumerate(zip(image_indices.tolist(), row_counts.tolist(), strict=True)):
        rotation_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
        rotations[row, :count] = np.arange(count * 9, dtype=np.float32).reshape(count, 3, 3) + image_id
        mstep_rotations[row, :count] = rotations[row, :count] + 50
        rotation_log_prior[row, :count] = -np.arange(count, dtype=np.float32)
        if posterior_ids is not None:
            posterior_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
        if sample_mask is not None:
            sample_mask[row, :count] = True
        translation_log_prior[row] = np.asarray([-image_id, -image_id - 0.5], dtype=np.float32)
    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=image_capacity,
        bucket_rotation_count=radix,
        actual_rotation_counts=row_counts,
        local_rotation_ids=rotation_ids,
        local_rotations=rotations,
        local_mstep_rotations=mstep_rotations,
        local_rotation_log_prior=rotation_log_prior,
        local_rotation_mask=rotation_mask,
        translation_log_prior=translation_log_prior,
        local_rotation_posterior_ids=posterior_ids,
        local_sample_mask=sample_mask,
    )


def _components(
    *,
    logical_cutoff=8,
    payload_offset=0.0,
    call0_image_capacity=2,
    call0_row_counts=(2, 3),
    include_pre_shifts=True,
    include_optional=True,
):
    buckets = (
        _bucket(
            [2, 0],
            call0_row_counts,
            radix=4,
            image_capacity=call0_image_capacity,
            include_optional=include_optional,
        ),
        _bucket([1], [1], radix=2, image_capacity=1, include_optional=include_optional),
    )
    if payload_offset:
        changed = []
        for bucket in buckets:
            rotations = bucket.local_rotations.copy()
            rotation_log_prior = bucket.local_rotation_log_prior.copy()
            rotations[bucket.local_rotation_mask] += np.float32(payload_offset)
            rotation_log_prior[bucket.local_rotation_mask] += np.float32(payload_offset)
            changed.append(
                replace(
                    bucket,
                    local_rotations=rotations,
                    local_rotation_log_prior=rotation_log_prior,
                )
            )
        buckets = tuple(changed)
    order = _seal_fixed_capacity_physical_order(np.asarray([2, 0, 1], dtype=np.int32))
    calls = _fixed_capacity_calls_from_local_buckets(buckets, expected_order=order)
    plan = _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=order,
        physical_image_capacity=5,
        physical_row_capacity=10,
        physical_call_capacity=3,
        image_capacity_palette={2: (1,), 4: (call0_image_capacity,)},
        logical_cutoff=logical_cutoff,
        logical_cutoff_capacity=16,
        enabled=True,
    )
    dataset = _IndexedDataset()
    metadata_by_image = {"scale": np.asarray([1.0, 1.5, 2.0], dtype=np.float32)}
    if include_pre_shifts:
        metadata_by_image["image_pre_shifts"] = np.asarray(
            [[0.25, -0.5], [1.25, -1.5], [2.25, -2.5]],
            dtype=np.float32,
        )
    operands = _assemble_fixed_capacity_local_operands_once(
        dataset,
        plan,
        order,
        metadata_by_image=metadata_by_image,
        tail_fill_value=-777,
        enabled=True,
    )
    hypotheses = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        order,
        enabled=True,
    )
    return plan, operands, hypotheses


def test_fixed_capacity_binding_and_materialization_are_default_off_and_inert():
    assert _bind_fixed_capacity_local_execution(None, None, None) is None
    assert _materialize_fixed_capacity_active_local_rows(None) is None


def test_fixed_capacity_binding_seals_one_matching_component_triple():
    plan, operands, hypotheses = _components()
    bundle = _bind_fixed_capacity_local_execution(
        plan,
        operands,
        hypotheses,
        enabled=True,
    )

    assert bundle.plan is plan
    assert bundle.operands is operands
    assert bundle.hypotheses is hypotheses
    assert bundle.descriptor_fingerprint == plan.descriptor_fingerprint
    assert bundle.generation_token is plan.generation_token
    assert operands.plan_fingerprint == plan.descriptor_fingerprint
    assert hypotheses.plan_fingerprint == plan.descriptor_fingerprint
    assert operands.plan_generation_token is plan.generation_token
    assert hypotheses.plan_generation_token is plan.generation_token


@pytest.mark.parametrize("component", ("operands", "hypotheses"))
def test_fixed_capacity_binding_rejects_cross_paired_component_fingerprints(component):
    plan, operands, hypotheses = _components(logical_cutoff=8)
    _, other_operands, other_hypotheses = _components(logical_cutoff=9)
    if component == "operands":
        operands = other_operands
    else:
        hypotheses = other_hypotheses

    with pytest.raises(ValueError, match="cross-paired across plan fingerprints"):
        _bind_fixed_capacity_local_execution(
            plan,
            operands,
            hypotheses,
            enabled=True,
        )


def test_fixed_capacity_binding_rejects_a_forged_component_fingerprint():
    plan, operands, hypotheses = _components()
    hypotheses = replace(hypotheses, plan_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="cross-paired across plan fingerprints"):
        _bind_fixed_capacity_local_execution(plan, operands, hypotheses, enabled=True)


def test_fixed_capacity_binding_rejects_same_descriptor_different_generation_payloads():
    plan, operands, hypotheses = _components(payload_offset=0.0)
    other_plan, _, other_hypotheses = _components(payload_offset=7.0)

    assert plan.descriptor_fingerprint == other_plan.descriptor_fingerprint
    assert plan.generation_token is not other_plan.generation_token
    assert not np.array_equal(
        other_hypotheses.local_rotations[: other_plan.valid_row_count],
        hypotheses.local_rotations[: plan.valid_row_count],
    )
    with pytest.raises(ValueError, match="cross-paired across plan generations"):
        _bind_fixed_capacity_local_execution(
            plan,
            operands,
            other_hypotheses,
            enabled=True,
        )


def test_fixed_capacity_active_row_materialization_structurally_excludes_all_tails():
    plan, operands, hypotheses = _components()
    bundle = _bind_fixed_capacity_local_execution(plan, operands, hypotheses, enabled=True)
    active = _materialize_fixed_capacity_active_local_rows(bundle, enabled=True)

    assert active.descriptor_fingerprint == plan.descriptor_fingerprint
    assert active.generation_token is plan.generation_token
    assert active.excluded_image_tail_count == 2
    assert active.excluded_candidate_tail_count == 4
    assert active.image_indices.shape == (plan.valid_image_count,)
    assert active.row_offsets.shape == (plan.valid_image_count + 1,)
    assert active.raw_images.shape[0] == plan.valid_image_count
    assert active.ctf_params.shape[0] == plan.valid_image_count
    assert active.local_rotation_ids.shape == (plan.valid_row_count,)
    assert active.local_rotations.shape == (plan.valid_row_count, 3, 3)
    assert active.local_mstep_rotations.shape == (plan.valid_row_count, 3, 3)
    assert active.local_rotation_log_prior.shape == (plan.valid_row_count,)
    assert active.translation_log_prior.shape[0] == plan.valid_image_count
    assert active.local_rotation_posterior_ids.shape == (plan.valid_row_count,)
    assert active.local_sample_mask.shape[0] == plan.valid_row_count
    assert np.all(active.local_rotation_ids >= 0)
    assert np.all(np.isfinite(active.local_rotations))
    assert np.all(np.isfinite(active.local_mstep_rotations))
    assert np.all(np.isfinite(active.local_rotation_log_prior))
    assert np.all(np.isfinite(active.translation_log_prior))
    np.testing.assert_array_equal(active.local_rotation_ids, hypotheses.local_rotation_ids[: plan.valid_row_count])
    np.testing.assert_array_equal(active.metadata_by_name["scale"], [2.0, 1.0, 1.5])

    active_arrays = (
        active.image_indices,
        active.row_offsets,
        active.raw_images,
        active.ctf_params,
        active.local_rotation_ids,
        active.local_rotations,
        active.local_mstep_rotations,
        active.local_rotation_log_prior,
        active.translation_log_prior,
        active.local_rotation_posterior_ids,
        active.local_sample_mask,
        active.metadata_by_name["scale"],
    )
    assert all(value.flags.writeable is False for value in active_arrays)


@pytest.mark.parametrize("poison_location", ("inactive_tail", "active_row"))
def test_fixed_capacity_active_row_materialization_rejects_poison_boundary_corruption(poison_location):
    plan, operands, hypotheses = _components()
    rotations = hypotheses.local_rotations.copy()
    if poison_location == "inactive_tail":
        rotations[plan.valid_row_count :] = 0
        message = "lost their NaN poison"
    else:
        rotations[0] = np.nan
        message = "active rotation rows contain NaN/infinite poison"
    rotations.setflags(write=False)
    hypotheses = replace(hypotheses, local_rotations=rotations)
    bundle = _bind_fixed_capacity_local_execution(plan, operands, hypotheses, enabled=True)

    with pytest.raises(ValueError, match=message):
        _materialize_fixed_capacity_active_local_rows(bundle, enabled=True)


def _call0_fixture(
    *,
    include_pre_shifts=True,
    call0_image_capacity=4,
    call0_row_counts=(2, 3),
    include_optional=True,
):
    plan, operands, hypotheses = _components(
        call0_image_capacity=call0_image_capacity,
        call0_row_counts=call0_row_counts,
        include_pre_shifts=include_pre_shifts,
        include_optional=include_optional,
    )
    bundle = _bind_fixed_capacity_local_execution(plan, operands, hypotheses, enabled=True)
    mature_bucket = _bucket(
        [2, 0],
        call0_row_counts,
        radix=4,
        image_capacity=call0_image_capacity,
        include_optional=include_optional,
    )
    image_pre_shifts = np.asarray(
        [[0.25, -0.5], [1.25, -1.5], [2.25, -2.5]],
        dtype=np.float32,
    )
    return plan, operands, hypotheses, bundle, mature_bucket, image_pre_shifts


def _select_call0(bundle, mature_bucket, default_image_pre_shifts, **overrides):
    options = {
        "n_classes": 1,
        "class_log_prior": 0.0,
        "image_pre_shifts": default_image_pre_shifts,
        "image_corrections": None,
        "scale_corrections": None,
        "score_only": True,
        "disable_adjoint_y": True,
        "disable_adjoint_ctf": True,
        "accumulate_noise": False,
        "mstep_requested": False,
        "unsupported_diagnostics": (),
        "enabled": True,
    }
    options.update(overrides)
    return local_em_engine._select_fixed_capacity_call0_score_only_view(
        bundle,
        mature_bucket,
        **options,
    )


def test_fixed_capacity_call0_materialization_and_selection_are_default_off_and_inert():
    assert _materialize_fixed_capacity_local_call_view(None) is None
    assert (
        _select_call0(
            None,
            None,
            None,
            n_classes=None,
            score_only=False,
            enabled=False,
        )
        is None
    )


def test_fixed_capacity_call0_view_is_read_only_call_scoped_and_canonical():
    plan, _, _, bundle, _, _ = _call0_fixture()

    view = _materialize_fixed_capacity_local_call_view(bundle, enabled=True)

    assert view.call_index == 0
    assert view.descriptor_fingerprint == plan.descriptor_fingerprint
    assert view.generation_token is plan.generation_token
    assert view.call_image_offset == int(plan.call_image_offsets[0]) == 0
    assert view.call_row_offset == int(plan.call_row_offsets[0]) == 0
    assert view.valid_image_count == int(plan.call_valid_images[0]) == 2
    assert view.valid_row_count == int(plan.call_valid_rows[0]) == 5
    assert view.physical_image_capacity == int(plan.call_image_capacities[0]) == 4
    assert view.physical_rotation_capacity == int(plan.call_radix_buckets[0]) == 4
    assert view.bucket.image_indices.shape == (2,)
    assert view.bucket.local_rotation_ids.shape == (2, 4)
    assert view.raw_images.shape == (2, 2, 2)
    assert view.ctf_params.shape == (2, 3)
    assert view.metadata_by_name["image_pre_shifts"].shape == (2, 2)
    np.testing.assert_array_equal(view.row_offsets, [0, 2, 5])
    np.testing.assert_array_equal(view.bucket.actual_rotation_counts, [2, 3])
    inactive = ~view.bucket.local_rotation_mask
    assert np.all(view.bucket.local_rotation_ids[inactive] == -1)
    assert np.all(view.bucket.local_rotation_log_prior[inactive] == np.float32(-1e30))
    assert np.all(view.bucket.local_rotation_posterior_ids[inactive] == -1)
    assert not np.any(view.bucket.local_sample_mask[inactive])
    np.testing.assert_array_equal(
        view.bucket.local_rotations[inactive],
        np.broadcast_to(np.eye(3, dtype=np.float32), view.bucket.local_rotations[inactive].shape),
    )
    np.testing.assert_array_equal(view.metadata_by_name["image_pre_shifts"], [[2.25, -2.5], [0.25, -0.5]])
    call_arrays = (
        view.row_offsets,
        view.raw_images,
        view.ctf_params,
        view.bucket.image_indices,
        view.bucket.actual_rotation_counts,
        view.bucket.local_rotation_ids,
        view.bucket.local_rotations,
        view.bucket.local_mstep_rotations,
        view.bucket.local_rotation_log_prior,
        view.bucket.local_rotation_mask,
        view.bucket.translation_log_prior,
        view.bucket.local_rotation_posterior_ids,
        view.bucket.local_sample_mask,
        view.metadata_by_name["image_pre_shifts"],
    )
    assert all(array.flags.writeable is False for array in call_arrays)
    assert all(np.all(np.isfinite(array)) for array in (view.raw_images, view.ctf_params))
    assert not np.any(np.isnan(view.bucket.local_rotation_log_prior))
    assert not np.any(np.isneginf(view.bucket.local_rotation_log_prior))


def test_fixed_capacity_materializes_every_active_call_in_sealed_chronology():
    plan, _, hypotheses, bundle, _, _ = _call0_fixture()

    views = tuple(
        _materialize_fixed_capacity_local_call_view(bundle, call_index=index, enabled=True)
        for index in range(plan.valid_call_count)
    )

    assert [view.call_index for view in views] == [0, 1]
    assert [view.call_image_offset for view in views] == [0, 2]
    assert [view.call_row_offset for view in views] == [0, 5]
    np.testing.assert_array_equal(
        np.concatenate([view.bucket.image_indices for view in views]),
        plan.image_indices[: plan.valid_image_count],
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [
                view.bucket.local_rotation_ids[view.bucket.local_rotation_mask]
                for view in views
            ]
        ),
        hypotheses.local_rotation_ids[: plan.valid_row_count],
    )

    second = views[1]
    assert second.valid_image_count == second.physical_image_capacity == 1
    assert second.valid_row_count == 1
    assert second.physical_rotation_capacity == 2
    np.testing.assert_array_equal(second.row_offsets, [0, 1])
    np.testing.assert_array_equal(second.bucket.actual_rotation_counts, [1])
    np.testing.assert_array_equal(second.bucket.image_indices, [1])
    np.testing.assert_array_equal(second.raw_images, _IndexedDataset().images[[1]])
    np.testing.assert_array_equal(second.ctf_params, _IndexedDataset().ctf_params[[1]])
    np.testing.assert_array_equal(second.metadata_by_name["scale"], [1.5])
    np.testing.assert_array_equal(second.metadata_by_name["image_pre_shifts"], [[1.25, -1.5]])
    assert all(
        value.flags.writeable is False
        for value in (
            second.row_offsets,
            second.bucket.image_indices,
            second.bucket.local_rotation_ids,
            second.raw_images,
            second.ctf_params,
            second.metadata_by_name["scale"],
        )
    )


def test_fixed_capacity_nonzero_call_uses_shared_selection_fetch_and_padding_path():
    _, _, _, bundle, _, image_pre_shifts = _call0_fixture()
    mature_bucket = _bucket(
        [1],
        [1],
        radix=2,
        image_capacity=1,
        include_optional=True,
    )
    view = local_em_engine._select_fixed_capacity_score_only_view(
        bundle,
        mature_bucket,
        call_index=1,
        n_classes=1,
        class_log_prior=0.0,
        image_pre_shifts=image_pre_shifts,
        image_corrections=None,
        scale_corrections=None,
        score_only=True,
        disable_adjoint_y=True,
        disable_adjoint_ctf=True,
        accumulate_noise=False,
        mstep_requested=False,
        enabled=True,
    )
    raw, ctf, fetched_indices = local_em_engine._fetch_and_validate_fixed_capacity_call_operands(
        _IndexedDataset(),
        view,
        mature_bucket,
    )

    padded = local_em_engine._pad_local_big_jit_image_axis(view.bucket, raw, ctf)
    local_em_engine._validate_fixed_capacity_padded_call(view, *padded)

    assert view.call_index == 1
    np.testing.assert_array_equal(fetched_indices, [1])
    _assert_bucket_arrays_equal(padded[0], mature_bucket)
    np.testing.assert_array_equal(padded[1], _IndexedDataset().images[[1]])
    np.testing.assert_array_equal(padded[2], _IndexedDataset().ctf_params[[1]])
    np.testing.assert_array_equal(padded[3], [True])


@pytest.mark.parametrize("call_index", (-1, 2, 3, 100))
def test_fixed_capacity_call_materialization_rejects_inactive_or_out_of_range_calls(call_index):
    _, _, _, bundle, _, _ = _call0_fixture()

    with pytest.raises(ValueError, match=rf"fixed-capacity call {call_index} is not active"):
        _materialize_fixed_capacity_local_call_view(
            bundle,
            call_index=call_index,
            enabled=True,
        )


@pytest.mark.parametrize("call_index", (True, np.bool_(False), 0.5, "1"))
def test_fixed_capacity_call_materialization_rejects_noninteger_call_indices(call_index):
    _, _, _, bundle, _, _ = _call0_fixture()

    with pytest.raises(ValueError, match="call index must be an integer"):
        _materialize_fixed_capacity_local_call_view(
            bundle,
            call_index=call_index,
            enabled=True,
        )


def _assert_bucket_arrays_equal(actual, expected):
    for field_name in (
        "image_indices",
        "actual_rotation_counts",
        "local_rotation_ids",
        "local_rotations",
        "local_mstep_rotations",
        "local_rotation_log_prior",
        "local_rotation_mask",
        "translation_log_prior",
        "local_rotation_posterior_ids",
        "local_sample_mask",
    ):
        np.testing.assert_array_equal(getattr(actual, field_name), getattr(expected, field_name))
    assert actual.bucket_image_count == expected.bucket_image_count
    assert actual.bucket_rotation_count == expected.bucket_rotation_count


def test_fixed_and_mature_call0_use_identical_common_padding_inputs():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()
    mature_raw = dataset.images[mature_bucket.image_indices]
    mature_ctf = dataset.ctf_params[mature_bucket.image_indices]
    fixed_raw, fixed_ctf, fetched_indices = (
        local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
            dataset,
            view,
            mature_bucket,
        )
    )
    np.testing.assert_array_equal(fetched_indices, mature_bucket.image_indices)

    mature_padded = local_em_engine._pad_local_big_jit_image_axis(
        mature_bucket,
        mature_raw,
        mature_ctf,
    )
    fixed_padded = local_em_engine._pad_local_big_jit_image_axis(
        view.bucket,
        fixed_raw,
        fixed_ctf,
    )

    _assert_bucket_arrays_equal(fixed_padded[0], mature_padded[0])
    for fixed_value, mature_value in zip(fixed_padded[1:4], mature_padded[1:4], strict=True):
        np.testing.assert_array_equal(fixed_value, mature_value)
    assert fixed_padded[4] == mature_padded[4] == 4
    local_em_engine._validate_fixed_capacity_padded_call0(view, *fixed_padded)
    padded_bucket, padded_raw, padded_ctf, valid_image_mask, _ = fixed_padded
    np.testing.assert_array_equal(valid_image_mask, [True, True, False, False])
    assert np.all(padded_bucket.translation_log_prior[2:] == 0)
    assert np.all(padded_raw[2:] == 0)
    np.testing.assert_array_equal(padded_ctf[2:], np.broadcast_to(padded_ctf[0], padded_ctf[2:].shape))


def test_fixed_capacity_call0_common_padding_accepts_full_physical_image_capacity():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture(call0_image_capacity=2)
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()
    raw, ctf, fetched_indices = local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
        dataset,
        view,
        mature_bucket,
    )

    padded = local_em_engine._pad_local_big_jit_image_axis(view.bucket, raw, ctf)

    assert view.valid_image_count == view.physical_image_capacity == 2
    assert padded[0] is view.bucket
    assert padded[1] is raw
    assert padded[2] is ctf
    np.testing.assert_array_equal(fetched_indices, view.bucket.image_indices)
    np.testing.assert_array_equal(padded[3], [True, True])
    local_em_engine._validate_fixed_capacity_padded_call0(view, *padded)


def test_fixed_capacity_call0_common_padding_accepts_active_row_equal_to_radix():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture(call0_row_counts=(4, 3))
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()
    raw, ctf, _ = local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
        dataset,
        view,
        mature_bucket,
    )

    padded = local_em_engine._pad_local_big_jit_image_axis(view.bucket, raw, ctf)

    assert view.bucket.actual_rotation_counts[0] == view.physical_rotation_capacity == 4
    assert np.all(view.bucket.local_rotation_mask[0])
    local_em_engine._validate_fixed_capacity_padded_call0(view, *padded)


def test_fixed_capacity_call0_common_padding_accepts_absent_optional_arrays():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture(include_optional=False)
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()
    raw, ctf, _ = local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
        dataset,
        view,
        mature_bucket,
    )

    padded = local_em_engine._pad_local_big_jit_image_axis(view.bucket, raw, ctf)

    assert view.bucket.local_rotation_posterior_ids is None
    assert view.bucket.local_sample_mask is None
    assert padded[0].local_rotation_posterior_ids is None
    assert padded[0].local_sample_mask is None
    local_em_engine._validate_fixed_capacity_padded_call0(view, *padded)


@pytest.mark.parametrize("mutated_field", ("raw_images", "ctf_params"))
def test_fixed_capacity_call0_rejects_current_dataset_operand_mutation_before_jit(
    monkeypatch,
    mutated_field,
):
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()
    if mutated_field == "raw_images":
        dataset.images[2, 0, 0] += np.float32(1.0)
    else:
        dataset.ctf_params[2, 0] += np.float32(1.0)
    jit_calls = []
    monkeypatch.setattr(
        local_em_engine,
        "_invoke_local_bucket_big_jit",
        lambda *args, **kwargs: jit_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=f"current-dataset {mutated_field} does not match"):
        local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
            dataset,
            view,
            mature_bucket,
        )

    assert jit_calls == []


def test_fixed_capacity_call0_rejects_different_current_dataset_with_same_plan_generation_before_jit(
    monkeypatch,
):
    plan, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    different_dataset = _IndexedDataset()
    different_dataset.images += np.float32(100.0)
    jit_calls = []
    monkeypatch.setattr(
        local_em_engine,
        "_invoke_local_bucket_big_jit",
        lambda *args, **kwargs: jit_calls.append((args, kwargs)),
    )

    assert view.descriptor_fingerprint == plan.descriptor_fingerprint
    assert view.generation_token is plan.generation_token
    with pytest.raises(ValueError, match="current-dataset raw_images does not match"):
        local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
            different_dataset,
            view,
            mature_bucket,
        )

    assert jit_calls == []


def test_fixed_capacity_call0_rejects_current_dataset_fetch_order_before_jit(monkeypatch):
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    dataset = _IndexedDataset()

    def reversed_batches(batch_size, *, indices, by_image):
        assert by_image is False
        reversed_indices = np.asarray(indices, dtype=np.int64)[::-1]
        yield (
            dataset.images[reversed_indices],
            None,
            None,
            dataset.ctf_params[reversed_indices],
            None,
            None,
            reversed_indices,
        )

    monkeypatch.setattr(dataset, "iter_batches", reversed_batches)
    jit_calls = []
    monkeypatch.setattr(
        local_em_engine,
        "_invoke_local_bucket_big_jit",
        lambda *args, **kwargs: jit_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="did not preserve the authoritative image order"):
        local_em_engine._fetch_and_validate_fixed_capacity_call0_operands(
            dataset,
            view,
            mature_bucket,
        )

    assert jit_calls == []


@pytest.mark.parametrize("identity_field", ("descriptor_fingerprint", "generation_token"))
def test_fixed_capacity_call0_rejects_bundle_identity_mismatch(identity_field):
    _, _, _, bundle, _, _ = _call0_fixture()
    if identity_field == "descriptor_fingerprint":
        mismatched_bundle = replace(bundle, descriptor_fingerprint="0" * 64)
        message = "bundle fingerprint"
    else:
        other_plan, _, _, _, _, _ = _call0_fixture()
        mismatched_bundle = replace(bundle, generation_token=other_plan.generation_token)
        message = "bundle generation"

    with pytest.raises(ValueError, match=message):
        _materialize_fixed_capacity_local_call_view(mismatched_bundle, enabled=True)


@pytest.mark.parametrize("field_name", ("raw_images", "ctf_params", "local_rotations"))
def test_fixed_capacity_call0_rejects_noncanonical_operand_or_hypothesis_dtype(field_name):
    plan, operands, hypotheses, _, _, _ = _call0_fixture()
    if hasattr(operands, field_name):
        changed = np.asarray(getattr(operands, field_name), dtype=np.float64)
        changed.setflags(write=False)
        operands = replace(operands, **{field_name: changed})
    else:
        changed = np.asarray(getattr(hypotheses, field_name), dtype=np.float64)
        changed.setflags(write=False)
        hypotheses = replace(hypotheses, **{field_name: changed})
    bundle = _bind_fixed_capacity_local_execution(plan, operands, hypotheses, enabled=True)

    with pytest.raises(ValueError, match="canonical dtype float32"):
        _materialize_fixed_capacity_local_call_view(bundle, enabled=True)


def test_fixed_capacity_call0_selection_rejects_missing_required_metadata():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture(include_pre_shifts=False)

    with pytest.raises(ValueError, match="missing required image_pre_shifts metadata"):
        _select_call0(bundle, mature_bucket, image_pre_shifts)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"n_classes": 2}, "supports K=1 only"),
        ({"n_classes": 1.0}, "explicit integer class count"),
        ({"class_log_prior": -0.25}, "zero class log prior"),
        ({"score_only": False}, "supports score-only"),
        ({"disable_adjoint_y": False}, "both adjoints disabled"),
        ({"disable_adjoint_ctf": False}, "both adjoints disabled"),
        ({"accumulate_noise": True}, "does not support noise accumulation"),
        ({"mstep_requested": True}, "does not support an M-step"),
        ({"image_corrections": np.ones(3, dtype=np.float32)}, "image or scale corrections"),
        ({"scale_corrections": np.ones(3, dtype=np.float32)}, "image or scale corrections"),
        ({"image_pre_shifts": None}, "requires explicit image_pre_shifts"),
        ({"unsupported_diagnostics": ("score_dump",)}, "does not support diagnostics: score_dump"),
    ),
)
def test_fixed_capacity_call0_selector_fails_closed_outside_score_only_k1(overrides, message):
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()

    with pytest.raises(ValueError, match=message):
        _select_call0(bundle, mature_bucket, image_pre_shifts, **overrides)


def test_fixed_capacity_call0_selector_rejects_noncanonical_or_mismatched_metadata():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()

    with pytest.raises(ValueError, match="canonical float32 shape"):
        _select_call0(bundle, mature_bucket, image_pre_shifts.astype(np.float64))
    changed_shifts = image_pre_shifts.copy()
    changed_shifts[2, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="image_pre_shifts metadata does not match"):
        _select_call0(bundle, mature_bucket, changed_shifts)


def test_fixed_capacity_call0_selector_rejects_mature_payload_or_shape_mismatch():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    changed_prior = mature_bucket.translation_log_prior.copy()
    changed_prior[0, 0] += np.float32(1.0)

    with pytest.raises(ValueError, match="translation_log_prior does not match"):
        _select_call0(
            bundle,
            replace(mature_bucket, translation_log_prior=changed_prior),
            image_pre_shifts,
        )
    with pytest.raises(ValueError, match="physical B x R shape"):
        _select_call0(
            bundle,
            replace(mature_bucket, bucket_image_count=5),
            image_pre_shifts,
        )


def test_fixed_capacity_call0_padded_validator_rejects_noncanonical_tail():
    _, _, _, bundle, mature_bucket, image_pre_shifts = _call0_fixture()
    view = _select_call0(bundle, mature_bucket, image_pre_shifts)
    padded = list(
        local_em_engine._pad_local_big_jit_image_axis(
            view.bucket,
            view.raw_images,
            view.ctf_params,
        )
    )
    corrupted_prior = padded[0].translation_log_prior.copy()
    corrupted_prior[view.valid_image_count, 0] = np.float32(1.0)
    padded[0] = replace(padded[0], translation_log_prior=corrupted_prior)

    with pytest.raises(ValueError, match="image-tail padding is not zero"):
        local_em_engine._validate_fixed_capacity_padded_call0(view, *padded)


def test_local_big_jit_shared_invocation_forwards_one_call_without_numeric_changes(monkeypatch):
    positional = object()
    keyword = object()
    expected = object()
    calls = []

    def fake_big_jit(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(local_em_engine, "run_local_bucket_big_jit", fake_big_jit)

    result = local_em_engine._invoke_local_bucket_big_jit(positional, marker=keyword)

    assert result is expected
    assert calls == [((positional,), {"marker": keyword})]


def test_local_big_jit_donates_the_two_loop_carried_accumulators_by_signature_index():
    parameter_names = tuple(inspect.signature(local_big_jit.run_local_bucket_big_jit).parameters)
    source = inspect.getsource(local_big_jit.run_local_bucket_big_jit)

    assert parameter_names[7:9] == ("Ft_y", "Ft_ctf")
    assert "donate_argnums=(7, 8)" in source
    assert "donate_argnums=(4, 5)" not in source


def test_local_em_caller_allocates_and_forwards_fresh_donated_accumulators_per_run():
    source = inspect.getsource(local_em_engine.run_local_em_exact)
    allocation_y = source.index("Ft_y = jnp.zeros(")
    allocation_ctf = source.index("Ft_ctf = jnp.zeros(")
    bucket_loop = source.index("for bucket_index, bucket in enumerate(bucket_specs):")
    argument_tuple = source.index("big_jit_arguments = (")
    invocation = source.index("_invoke_local_bucket_big_jit(")

    assert allocation_y < bucket_loop < argument_tuple < invocation
    assert allocation_ctf < bucket_loop < argument_tuple < invocation
    argument_source = source[
        argument_tuple : source.index("big_jit_static_options = dict(", argument_tuple)
    ]
    positional_lines = [
        line.strip().rstrip(",") for line in argument_source.splitlines()[1:11]
    ]
    assert positional_lines[7:9] == ["Ft_y", "Ft_ctf"]


def test_fixed_capacity_selector_is_private_default_off_and_uses_shared_mature_call():
    signature = inspect.signature(local_em_engine.run_local_em_exact)
    assert signature.parameters["_fixed_capacity_enabled"].default is False
    assert signature.parameters["_fixed_capacity_bundle"].default is None
    assert signature.parameters["_fixed_capacity_class_count"].default is None
    assert signature.parameters["_fixed_capacity_whole_boundary_enabled"].default is False
    source = inspect.getsource(local_em_engine.run_local_em_exact)
    assert source.count("_invoke_local_bucket_big_jit(") == 1
    assert "big_jit_result = run_local_bucket_big_jit(" not in source
    assert "fixed_capacity_enabled and not use_big_jit_buckets" in source
    assert "call_index=bucket_index" in source
    assert source.index("_fetch_and_validate_fixed_capacity_call_operands(") < source.index(
        "_invoke_local_bucket_big_jit(",
    )


def test_whole_local_call_preparation_removes_only_invariant_carry_positions():
    positional, _ = local_big_jit._local_bucket_big_jit_signature_parts()
    arguments = tuple(object() for _ in positional)

    prepared = local_big_jit._prepare_fixed_capacity_local_call(*arguments)

    assert prepared.leading_arguments == arguments[:7]
    assert prepared.trailing_arguments == arguments[17:]
    with pytest.raises(ValueError, match="every mature positional argument"):
        local_big_jit._prepare_fixed_capacity_local_call(*arguments[:-1])


def test_whole_local_program_threads_all_ten_state_values_in_call_order():
    call_program = (
        local_big_jit._FixedCapacityPreparedLocalCall(
            leading_arguments=(jnp.asarray(1, dtype=jnp.int32),),
            trailing_arguments=(jnp.asarray(101, dtype=jnp.int32),),
        ),
        local_big_jit._FixedCapacityPreparedLocalCall(
            leading_arguments=(jnp.asarray(2, dtype=jnp.int32),),
            trailing_arguments=(jnp.asarray(202, dtype=jnp.int32),),
        ),
    )
    initial_carry = tuple(jnp.asarray(value, dtype=jnp.int32) for value in range(10))
    received_carries = []

    def fake_numeric_call(delta, *arguments, scale):
        carry = arguments[:10]
        tag = arguments[10]
        received_carries.append(tuple(int(value) for value in carry))
        increment = delta * scale
        next_first_eight = tuple(value + increment for value in carry[:8])
        next_last_two = tuple(value + increment for value in carry[8:])
        return (
            *next_first_eight,
            delta * 100,
            *next_last_two,
            tag,
            delta * 1000,
        )

    final_carry, call_outputs = local_big_jit._run_fixed_capacity_whole_local_program(
        call_program,
        initial_carry,
        (("scale", 3),),
        numeric_call=fake_numeric_call,
    )

    assert received_carries == [tuple(range(10)), tuple(value + 3 for value in range(10))]
    assert tuple(int(value) for value in final_carry) == tuple(
        value + 9 for value in range(10)
    )
    assert tuple(tuple(int(value) for value in output) for output in call_outputs) == (
        (100, 101, 1000),
        (200, 202, 2000),
    )


def test_whole_local_static_options_are_complete_hashable_and_fail_closed():
    _, keyword_only = local_big_jit._local_bucket_big_jit_signature_parts()
    required = {
        parameter.name: False
        for parameter in keyword_only
        if parameter.default is inspect.Parameter.empty
    }

    canonical = local_big_jit._canonicalize_fixed_capacity_static_options(required)

    assert tuple(name for name, _ in canonical) == tuple(
        parameter.name for parameter in keyword_only
    )
    with pytest.raises(ValueError, match="unknown fixed-capacity"):
        local_big_jit._canonicalize_fixed_capacity_static_options(
            {**required, "not_a_mature_option": False}
        )
    missing = dict(required)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing fixed-capacity"):
        local_big_jit._canonicalize_fixed_capacity_static_options(missing)
    with pytest.raises(ValueError, match="recursively hashable"):
        local_big_jit._canonicalize_fixed_capacity_static_options(
            {**required, "mask_mode": []}
        )


def test_whole_local_public_boundary_rejects_empty_or_unsealed_programs_before_jit():
    carry = tuple(jnp.asarray(0, dtype=jnp.float32) for _ in range(10))
    with pytest.raises(ValueError, match="at least one call"):
        local_big_jit.run_fixed_capacity_whole_local((), *carry)
    with pytest.raises(ValueError, match="sealed prepared calls"):
        local_big_jit.run_fixed_capacity_whole_local((((), ()),), *carry)


def test_whole_local_jit_reuses_mature_body_and_donates_only_volume_accumulators():
    source = inspect.getsource(local_big_jit._run_fixed_capacity_whole_local_jit)

    assert "run_local_bucket_big_jit.__wrapped__" in source
    assert "donate_argnums=(1, 2)" in source
    assert "optimization_barrier" in inspect.getsource(
        local_big_jit._run_fixed_capacity_whole_local_program
    )


def test_uniform_local_scan_threads_carry_with_one_stacked_call_axis():
    stacked_program = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(jnp.asarray([1, 2], dtype=jnp.int32),),
        trailing_arguments=(jnp.asarray([101, 202], dtype=jnp.int32),),
    )
    initial_carry = tuple(jnp.asarray(value, dtype=jnp.int32) for value in range(10))

    def fake_numeric_call(delta, *arguments, scale):
        carry = arguments[:10]
        tag = arguments[10]
        increment = delta * scale
        next_first_eight = tuple(value + increment for value in carry[:8])
        next_last_two = tuple(value + increment for value in carry[8:])
        return (
            *next_first_eight,
            delta * 100,
            *next_last_two,
            tag,
            delta * 1000,
        )

    final_carry, stacked_outputs = (
        local_big_jit._run_fixed_capacity_uniform_local_scan_program(
            stacked_program,
            initial_carry,
            (("scale", 3),),
            numeric_call=fake_numeric_call,
        )
    )

    assert tuple(int(value) for value in final_carry) == tuple(
        value + 9 for value in range(10)
    )
    assert tuple(np.asarray(value).tolist() for value in stacked_outputs) == (
        [100, 200],
        [101, 202],
        [1000, 2000],
    )


def test_uniform_local_scan_fails_closed_on_structure_shape_or_dtype_changes():
    call = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float32),),
        trailing_arguments=(),
    )
    changed_structure = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float32), object()),
        trailing_arguments=(),
    )
    changed_shape = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((3,), dtype=np.float32),),
        trailing_arguments=(),
    )
    changed_dtype = local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=(np.zeros((2,), dtype=np.float64),),
        trailing_arguments=(),
    )

    validated = local_big_jit._validate_uniform_fixed_capacity_call_program(
        (call, call)
    )
    assert validated[0] is call and validated[1] is call
    with pytest.raises(ValueError, match="pytree structure"):
        local_big_jit._validate_uniform_fixed_capacity_call_program(
            (call, changed_structure)
        )
    for changed in (changed_shape, changed_dtype):
        with pytest.raises(ValueError, match="leaf shape or dtype"):
            local_big_jit._validate_uniform_fixed_capacity_call_program(
                (call, changed)
            )


def test_uniform_local_scan_reuses_mature_body_inside_lax_scan():
    source = inspect.getsource(local_big_jit._run_fixed_capacity_uniform_local_scan_jit)
    program_source = inspect.getsource(
        local_big_jit._run_fixed_capacity_uniform_local_scan_program
    )

    assert "run_local_bucket_big_jit.__wrapped__" in source
    assert "donate_argnums=(1, 2)" in source
    assert "jax.lax.scan" in program_source
    assert "optimization_barrier" in program_source
