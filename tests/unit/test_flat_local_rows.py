from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine
from recovar.em.dense_single_volume.helpers.flat_local_rows import (
    build_pool_flat_local_row_plan,
    encode_flat_local_row_plan,
    gather_flat_local_rows,
    scatter_flat_local_rows,
)


@pytest.mark.unit
def test_pool_flat_rows_preserve_source_chronology_and_static_tail():
    counts = np.asarray([3, 17, 5, 65, 9], dtype=np.int32)
    plan = build_pool_flat_local_row_plan(
        counts,
        256,
        pool_size=3,
        exact_local_bucket_radix=4,
        packed_row_count=3 * 64 + 2 * 256 + 7,
        dense_batch_size=8,
    )

    expected = []
    for image_index, bucket_size in ((0, 64), (1, 64), (2, 64), (3, 256), (4, 256)):
        expected.extend((image_index, rotation) for rotation in range(bucket_size))
    present = plan.present_mask
    assert list(zip(plan.image_indices[present], plan.rotation_rows[present], strict=True)) == expected
    assert np.array_equal(
        plan.valid_mask[present],
        np.asarray([rotation < counts[image] for image, rotation in expected]),
    )
    assert not np.any(plan.present_mask[-7:])
    assert not np.any(plan.valid_mask[-7:])
    assert plan.physical_image_count == 5
    assert plan.batch_size == 8
    encoded = encode_flat_local_row_plan(plan)
    assert encoded.dtype == np.int32
    assert encoded.shape == (plan.packed_row_count, 3)
    assert np.array_equal(encoded[:, 0], plan.image_indices)
    assert np.array_equal(encoded[:, 1], plan.rotation_rows)
    assert np.array_equal(encoded[:, 2] != 0, plan.present_mask)


@pytest.mark.unit
def test_flat_row_gather_and_scatter_restore_present_dense_rows_exactly():
    counts = np.asarray([3, 17, 5, 65, 9], dtype=np.int32)
    dense_rotation_count = 256
    plan = build_pool_flat_local_row_plan(
        counts,
        dense_rotation_count,
        pool_size=3,
        exact_local_bucket_radix=4,
        packed_row_count=712,
    )
    dense = np.arange(5 * dense_rotation_count * 2, dtype=np.float32).reshape(5, dense_rotation_count, 2)
    gathered = gather_flat_local_rows(
        jnp.asarray(dense),
        jnp.asarray(plan.image_indices),
        jnp.asarray(plan.rotation_rows),
    )
    gathered = gathered.at[jnp.logical_not(jnp.asarray(plan.present_mask))].set(
        jnp.nan,
    )
    restored = np.asarray(
        scatter_flat_local_rows(
            gathered,
            plan.image_indices,
            plan.rotation_rows,
            plan.present_mask,
            batch_size=plan.batch_size,
            dense_rotation_count=plan.dense_rotation_count,
            fill_value=-1.0,
        ),
    )

    expected_present = np.zeros(dense.shape[:2], dtype=bool)
    expected_present[
        plan.image_indices[plan.present_mask],
        plan.rotation_rows[plan.present_mask],
    ] = True
    assert np.array_equal(restored[expected_present], dense[expected_present])
    assert np.all(restored[~expected_present] == -1.0)
    assert not np.any(np.isnan(restored))


@pytest.mark.unit
def test_flat_row_capacity_reuses_one_shape_per_dense_bucket_abi():
    first = SimpleNamespace(
        image_indices=np.arange(3, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([3, 17, 5], dtype=np.int32),
    )
    second = SimpleNamespace(
        image_indices=np.arange(2, dtype=np.int32),
        bucket_image_count=4,
        bucket_rotation_count=256,
        actual_rotation_counts=np.asarray([65, 9], dtype=np.int32),
    )

    capacities = local_em_engine._plan_flat_local_row_capacities(
        (first, second),
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )
    encoded = local_em_engine._build_flat_local_row_argument(
        first,
        capacities,
        dense_batch_size=4,
        rotation_block_size=128,
        exact_local_bucket_radix=4,
    )

    assert capacities == {(4, 256): 256}
    assert encoded.shape == (256, 3)
    assert np.count_nonzero(encoded[:, 2]) == 192


@pytest.mark.unit
@pytest.mark.parametrize(
    ("counts", "dense_rotation_count", "kwargs", "message"),
    [
        ([], 16, {}, "at least one image"),
        ([1], 0, {}, "dense_rotation_count"),
        ([0], 16, {}, "rotation counts"),
        ([17], 16, {}, "rotation counts"),
        ([1], 16, {"pool_size": 0}, "pool_size"),
        ([1, 1], 16, {"dense_batch_size": 1}, "dense_batch_size"),
        ([17], 64, {"packed_row_count": 63}, "smaller than required"),
    ],
)
def test_pool_flat_row_plan_rejects_invalid_shapes(counts, dense_rotation_count, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_pool_flat_local_row_plan(
            counts,
            dense_rotation_count,
            exact_local_bucket_radix=4,
            **kwargs,
        )
