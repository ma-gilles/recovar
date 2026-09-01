"""Host contracts for fingerprint-bound fixed-capacity local components."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.fixed_capacity_local import (
    _bind_fixed_capacity_local_execution,
    _materialize_fixed_capacity_active_local_rows,
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


def _bucket(image_indices, row_counts, *, radix, image_capacity):
    image_indices = np.asarray(image_indices, dtype=np.int32)
    row_counts = np.asarray(row_counts, dtype=np.int32)
    n_images = int(image_indices.size)
    rotation_mask = np.arange(radix, dtype=np.int32)[None, :] < row_counts[:, None]
    rotation_ids = np.full((n_images, radix), -1, dtype=np.int32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, radix, 3, 3)).copy()
    mstep_rotations = rotations.copy()
    rotation_log_prior = np.full((n_images, radix), -1e30, dtype=np.float32)
    posterior_ids = np.full((n_images, radix), -1, dtype=np.int32)
    sample_mask = np.zeros((n_images, radix, 2), dtype=bool)
    translation_log_prior = np.empty((n_images, 2), dtype=np.float32)
    for row, (image_id, count) in enumerate(zip(image_indices.tolist(), row_counts.tolist(), strict=True)):
        rotation_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
        rotations[row, :count] = np.arange(count * 9, dtype=np.float32).reshape(count, 3, 3) + image_id
        mstep_rotations[row, :count] = rotations[row, :count] + 50
        rotation_log_prior[row, :count] = -np.arange(count, dtype=np.float32)
        posterior_ids[row, :count] = image_id * 10 + np.arange(count, dtype=np.int32)
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


def _components(*, logical_cutoff=8, payload_offset=0.0):
    buckets = (
        _bucket([2, 0], [2, 3], radix=4, image_capacity=2),
        _bucket([1], [1], radix=2, image_capacity=1),
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
        image_capacity_palette={2: (1,), 4: (2,)},
        logical_cutoff=logical_cutoff,
        logical_cutoff_capacity=16,
        enabled=True,
    )
    dataset = _IndexedDataset()
    operands = _assemble_fixed_capacity_local_operands_once(
        dataset,
        plan,
        order,
        metadata_by_image={"scale": np.asarray([1.0, 1.5, 2.0], dtype=np.float32)},
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
