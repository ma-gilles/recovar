"""Image padding must preserve real rows and mask every added candidate."""

import numpy as np
import pytest

from recovar.em.scoring.sparse_bucket_arrays import (
    _build_k_class_bucket_arrays,
    _build_compact_pair_bucket_arrays_from_per_image_inputs,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dense_fields", [False, True])
def test_capacity_allocations_preserve_real_rows(dtype, dense_fields):
    rotations = [np.repeat(np.eye(3, dtype=dtype)[None], n, axis=0) for n in [3, 1]]
    inputs = dict(
        oversampled_rots=rotations,
        oversampled_mstep_rots=rotations,
        log_prior=[np.arange(n, dtype=dtype) for n in [3, 1]],
        candidate_mask=[np.ones((n, 2), dtype=bool) for n in [3, 1]],
        parent_map=[np.arange(n, dtype=np.int32) for n in [3, 1]],
        oversampled_rot_indices=[np.arange(n, dtype=np.int64) for n in [3, 1]],
    )
    bucket = dict(image_indices=np.array([1, 0]), bucket_size=4, pair_bucket_size=8)
    base = _build_k_class_bucket_arrays(bucket, [inputs, inputs], 2, include_dense_score_fields=dense_fields)
    padded = _build_k_class_bucket_arrays(
        bucket,
        [inputs, inputs],
        2,
        include_dense_score_fields=dense_fields,
        capacity_rows=4,
    )
    for before, after in zip(base, padded, strict=True):
        assert after["rotations"] is after["mstep_rotations"]
        for name, value in before.items():
            if not isinstance(value, np.ndarray):
                assert after[name] == value
                continue
            assert value.dtype == after[name].dtype
            np.testing.assert_array_equal(after[name][:2], value)
        np.testing.assert_array_equal(after["actual_counts"][2:], 0)
        np.testing.assert_array_equal(after["rotation_indices"][2:], 0)
        np.testing.assert_array_equal(after["rotations"][2:], np.broadcast_to(np.eye(3, dtype=dtype), (2, 4, 3, 3)))
        np.testing.assert_array_equal(after["log_prior"][2:], dtype(-1e30))
        np.testing.assert_array_equal(after["image_indices"], [1, 0, 0, 0])
        if dense_fields:
            assert not after["candidate_mask"][2:].any()
            np.testing.assert_array_equal(after["parent_map"][2:], -1)
    pairs = _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, inputs)
    padded_pairs = _build_compact_pair_bucket_arrays_from_per_image_inputs(bucket, inputs, capacity_rows=4)
    for name, value in pairs.items():
        if isinstance(value, np.ndarray):
            assert padded_pairs[name].dtype == value.dtype
            np.testing.assert_array_equal(padded_pairs[name][:2], value)
    np.testing.assert_array_equal(padded_pairs["pair_counts"][2:], 0)
    assert not padded_pairs["pair_mask"][2:].any()
    for name in ["local_rotation_row", "translation_idx"]:
        np.testing.assert_array_equal(padded_pairs[name][2:], -1)


def test_capacity_respects_byte_and_growth_limits():
    from recovar.em.sparse_pass2.sparse_pass2_budget import quantized_image_capacity

    for n in range(1, 600):
        for limit in [1, 16, 64, 128, 512, 1000, None]:
            capacity = quantized_image_capacity(n, max_images=limit)
            assert n <= capacity <= 2 * n
            if capacity > n:
                assert limit is not None and capacity <= limit
                assert capacity >= 16 and capacity & (capacity - 1) == 0
    assert quantized_image_capacity(20, max_images=24) == 20
    assert quantized_image_capacity(20, max_images=32) == 32
    assert quantized_image_capacity(0, max_images=64) == 0


def test_fetch_reordering_preserves_padding_rows():
    from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _reorder_to_indices

    values = np.array([[10, 11], [20, 21], [30, 31], [-1, -1], [-2, -2]])
    (reordered,) = _reorder_to_indices(np.array([8, 3, 1]), np.array([1, 8, 3]), values)
    np.testing.assert_array_equal(reordered, [[20, 21], [30, 31], [10, 11], [-1, -1], [-2, -2]])
