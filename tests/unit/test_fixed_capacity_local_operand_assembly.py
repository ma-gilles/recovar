"""Host contracts for sealed cache-once fixed-capacity operands."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.batch_planning import (
    _FixedCapacityLocalCall,
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.local_caches import (
    _assemble_fixed_capacity_local_operands_once,
    _build_local_raw_cache,
)

pytestmark = pytest.mark.unit


class _IndexedDataset:
    def __init__(self, *, returned_indices=None, truncate_raw=False):
        self.images = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        self.ctf_params = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) + 100
        self.returned_indices = returned_indices
        self.truncate_raw = truncate_raw
        self.calls = 0
        self.requested_indices = []

    def iter_batches(self, batch_size, *, indices, by_image):
        assert by_image is False
        requested = np.asarray(indices, dtype=np.int64)
        self.calls += 1
        self.requested_indices.append(requested.copy())
        returned = requested if self.returned_indices is None else np.asarray(self.returned_indices, dtype=np.int64)
        raw = self.images[returned]
        if self.truncate_raw:
            raw = raw[:-1]
        yield (
            raw,
            None,
            None,
            self.ctf_params[returned],
            None,
            None,
            returned,
        )


def _sealed_plan():
    expected_order = _seal_fixed_capacity_physical_order(
        np.asarray([2, 0, 3, 1], dtype=np.int32),
    )
    calls = (
        _FixedCapacityLocalCall(
            image_indices=np.asarray([2, 0, 3], dtype=np.int32),
            row_counts=np.asarray([1, 2, 1], dtype=np.int32),
            radix_bucket=2,
            image_capacity=3,
        ),
        _FixedCapacityLocalCall(
            image_indices=np.asarray([1], dtype=np.int32),
            row_counts=np.asarray([2], dtype=np.int32),
            radix_bucket=2,
            image_capacity=3,
        ),
    )
    plan = _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=expected_order,
        physical_image_capacity=6,
        physical_row_capacity=8,
        physical_call_capacity=3,
        image_capacity_palette={2: (3,)},
        logical_cutoff=16,
        logical_cutoff_capacity=32,
        enabled=True,
    )
    return expected_order, plan


def test_fixed_capacity_operand_assembly_is_default_off_without_fetching():
    expected_order, plan = _sealed_plan()
    dataset = _IndexedDataset()

    result = _assemble_fixed_capacity_local_operands_once(
        dataset,
        plan,
        expected_order,
    )

    assert result is None
    assert dataset.calls == 0


def test_existing_shared_raw_cache_still_indexes_rows_by_returned_image_id():
    dataset = _IndexedDataset(returned_indices=(2, 0, 3, 1))
    expected_raw = dataset.images.copy()
    expected_ctf = dataset.ctf_params.copy()

    raw_cache, ctf_cache = _build_local_raw_cache(dataset, 4)

    assert dataset.calls == 1
    np.testing.assert_array_equal(dataset.requested_indices[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(raw_cache, expected_raw)
    np.testing.assert_array_equal(ctf_cache, expected_ctf)


def test_fixed_capacity_operand_assembly_fetches_once_snapshots_and_poisons_tails():
    expected_order, plan = _sealed_plan()
    dataset = _IndexedDataset()
    scale = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    group = np.asarray([10, 11, 12, 13], dtype=np.int32)
    expected_raw = dataset.images[[2, 0, 3, 1]].copy()
    expected_ctf = dataset.ctf_params[[2, 0, 3, 1]].copy()

    operands = _assemble_fixed_capacity_local_operands_once(
        dataset,
        plan,
        expected_order,
        metadata_by_image={"scale": scale, "group": group},
        tail_fill_value=-777,
        enabled=True,
    )

    assert dataset.calls == 1
    np.testing.assert_array_equal(dataset.requested_indices[0], [2, 0, 3, 1])
    np.testing.assert_array_equal(operands.image_indices, [2, 0, 3, 1, -1, -1])
    np.testing.assert_array_equal(operands.valid_image_mask, [True, True, True, True, False, False])
    np.testing.assert_array_equal(operands.raw_images[:4], expected_raw)
    np.testing.assert_array_equal(operands.ctf_params[:4], expected_ctf)
    assert np.all(operands.raw_images[4:] == -777)
    assert np.all(operands.ctf_params[4:] == -777)
    np.testing.assert_array_equal(operands.metadata_by_name["scale"][:4], scale[[2, 0, 3, 1]])
    np.testing.assert_array_equal(operands.metadata_by_name["group"][:4], group[[2, 0, 3, 1]])
    assert np.all(operands.metadata_by_name["scale"][4:] == -777)
    assert np.all(operands.metadata_by_name["group"][4:] == -777)
    assert dict(operands.physical_position_by_image_id) == {2: 0, 0: 1, 3: 2, 1: 3}

    dataset.images[:] = 0
    dataset.ctf_params[:] = 0
    scale[:] = 0
    group[:] = 0
    np.testing.assert_array_equal(operands.raw_images[:4], expected_raw)
    np.testing.assert_array_equal(operands.ctf_params[:4], expected_ctf)
    assert operands.raw_images.flags.writeable is False
    assert operands.ctf_params.flags.writeable is False
    assert operands.metadata_by_name["scale"].flags.writeable is False


@pytest.mark.parametrize(
    ("returned_indices", "message"),
    (
        ((2, 0, 0, 1), "duplicate image IDs"),
        ((2, 0, 3), "missing or unexpected image IDs"),
        ((0, 2, 3, 1), "image IDs out of order"),
    ),
)
def test_fixed_capacity_operand_assembly_rejects_fetch_id_topology(returned_indices, message):
    expected_order, plan = _sealed_plan()
    dataset = _IndexedDataset(returned_indices=returned_indices)

    with pytest.raises(ValueError, match=message):
        _assemble_fixed_capacity_local_operands_once(
            dataset,
            plan,
            expected_order,
            enabled=True,
        )


def test_fixed_capacity_operand_assembly_rejects_plan_order_before_fetching():
    _, plan = _sealed_plan()
    wrong_order = _seal_fixed_capacity_physical_order(
        np.asarray([2, 0, 1, 3], dtype=np.int32),
    )
    dataset = _IndexedDataset()

    with pytest.raises(ValueError, match="plan chronology"):
        _assemble_fixed_capacity_local_operands_once(
            dataset,
            plan,
            wrong_order,
            enabled=True,
        )
    assert dataset.calls == 0


def test_fixed_capacity_operand_assembly_rejects_operand_or_metadata_row_gaps():
    expected_order, plan = _sealed_plan()
    dataset = _IndexedDataset(truncate_raw=True)

    with pytest.raises(ValueError, match="raw image rows"):
        _assemble_fixed_capacity_local_operands_once(
            dataset,
            plan,
            expected_order,
            enabled=True,
        )

    dataset = _IndexedDataset()
    with pytest.raises(ValueError, match="does not cover every sealed image ID"):
        _assemble_fixed_capacity_local_operands_once(
            dataset,
            plan,
            expected_order,
            metadata_by_image={"short": np.zeros(3, dtype=np.float32)},
            enabled=True,
        )

    dataset = _IndexedDataset()
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        _assemble_fixed_capacity_local_operands_once(
            dataset,
            plan,
            expected_order,
            metadata_by_image=[np.zeros(4, dtype=np.float32)],
            enabled=True,
        )
