"""Contracts for the explicit exact-local bucket radix."""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.local_batch_planning import (
    EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV,
    _exact_local_planned_hypotheses_floor,
    _exact_local_xhalf_projection_microbatch_cap,
)
from recovar.em.dense_single_volume.local_em_engine import (
    run_local_em_exact,
)
from recovar.em.dense_single_volume.local_bucket_stages import (
    _build_reconstruction_pack_indices,
)
from recovar.em.dense_single_volume.local_layout import (
    EXACT_LOCAL_BUCKET_RADIX_ENV,
    LocalHypothesisLayout,
    _exact_bucket_rotation_size,
    bucket_local_hypothesis_layout,
)


def _make_layout(rotation_counts: np.ndarray) -> LocalHypothesisLayout:
    rotation_counts = np.asarray(rotation_counts, dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total_rotations = int(rotation_offsets[-1])
    return LocalHypothesisLayout(
        n_global_rotations=total_rotations,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=np.arange(1000, 1000 + total_rotations, dtype=np.int32),
        rotations_flat=np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (total_rotations, 3, 3),
        ).copy(),
        rotation_log_priors_flat=np.zeros(total_rotations, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((rotation_counts.size, 1), dtype=np.float32),
    )


@pytest.mark.unit
def test_exact_local_bucket_radix_none_preserves_default_and_environment(monkeypatch):
    monkeypatch.delenv(EXACT_LOCAL_BUCKET_RADIX_ENV, raising=False)
    assert _exact_bucket_rotation_size(32, 5000) == 32

    monkeypatch.setenv(EXACT_LOCAL_BUCKET_RADIX_ENV, "4")
    assert _exact_bucket_rotation_size(32, 5000) == 64


@pytest.mark.unit
def test_exact_local_bucket_radix_explicit_value_overrides_environment(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_BUCKET_RADIX_ENV, "4")
    assert (
        _exact_bucket_rotation_size(
            32,
            5000,
            exact_local_bucket_radix=2,
        )
        == 32
    )

    monkeypatch.setenv(EXACT_LOCAL_BUCKET_RADIX_ENV, "not-an-integer")
    assert (
        _exact_bucket_rotation_size(
            32,
            5000,
            exact_local_bucket_radix=4,
        )
        == 64
    )


@pytest.mark.unit
@pytest.mark.parametrize("radix", [0, 1])
def test_run_local_em_exact_rejects_invalid_explicit_bucket_radix(radix):
    with pytest.raises(ValueError, match="exact_local_bucket_radix must be at least 2"):
        run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=1,
            exact_local_bucket_radix=radix,
        )


@pytest.mark.unit
def test_explicit_bucket_radix_is_consistent_across_exact_local_topology(monkeypatch):
    monkeypatch.setenv(EXACT_LOCAL_BUCKET_RADIX_ENV, "2")
    monkeypatch.setenv(EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS_ENV, "1")
    layout = _make_layout(np.asarray([32, 129], dtype=np.int32))

    buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=3,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=4096,
        exact_local_bucket_radix=4,
    )
    planned_floor = _exact_local_planned_hypotheses_floor(
        layout,
        image_batch_size=3,
        rotation_block_size=5000,
        exact_local_bucket_radix=4,
    )
    projection_cap = _exact_local_xhalf_projection_microbatch_cap(
        4096,
        layout,
        n_projection_pixels=1,
        rotation_block_size=5000,
        exact_local_bucket_radix=4,
    )

    significant_mask = np.zeros((2, 129), dtype=bool)
    significant_mask[0, :32] = True
    significant_mask[1, :] = True
    take_indices, pack_mask, actual_counts, row_count = _build_reconstruction_pack_indices(
        significant_mask,
        significant_mask,
        rotation_block_size=5000,
        exact_local_bucket_radix=4,
    )

    assert {int(bucket.bucket_rotation_count) for bucket in buckets} == {64, 256}
    assert planned_floor == 3 * 256
    assert projection_cap == 256
    assert take_indices.shape == (2, 256)
    assert pack_mask.shape == (2, 256)
    np.testing.assert_array_equal(actual_counts, np.asarray([32, 129], dtype=np.int32))
    assert row_count == 161

    for bucket in buckets:
        for row, image_index in enumerate(bucket.image_indices):
            image_index = int(image_index)
            count = int(layout.rotation_counts[image_index])
            start = int(layout.rotation_offsets[image_index])
            stop = start + count
            np.testing.assert_array_equal(
                bucket.local_rotation_ids[row, :count],
                layout.rotation_ids_flat[start:stop],
            )
            assert bucket.local_rotation_mask[row, :count].all()
            assert not bucket.local_rotation_mask[row, count:].any()

    np.testing.assert_array_equal(take_indices[0, :32], np.arange(32, dtype=np.int32))
    np.testing.assert_array_equal(take_indices[1, :129], np.arange(129, dtype=np.int32))
    assert pack_mask[0, :32].all()
    assert not pack_mask[0, 32:].any()
    assert pack_mask[1, :129].all()
    assert not pack_mask[1, 129:].any()


@pytest.mark.unit
def test_physical_order_chunks_reduce_padding_without_changing_candidates():
    counts = np.asarray(
        [17, 18, 31, 32, 17, 18, 129, 130, 200, 129, 130, 200, 17],
        dtype=np.int32,
    )
    layout = _make_layout(counts)

    global_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=13,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=100_000,
        unify_bucket_sizes=True,
        preserve_image_order=True,
        exact_local_bucket_radix=2,
    )
    chunked_buckets = bucket_local_hypothesis_layout(
        layout,
        image_batch_size=13,
        rotation_block_size=5000,
        max_hypotheses_per_microbatch=100_000,
        unify_bucket_sizes=False,
        preserve_image_order=True,
        exact_local_bucket_radix=2,
        consecutive_mixed_bucket_size=6,
    )

    assert [bucket.image_indices.tolist() for bucket in chunked_buckets] == [
        list(range(6)),
        list(range(6, 12)),
        [12],
    ]
    assert [bucket.bucket_rotation_count for bucket in chunked_buckets] == [32, 256, 32]
    assert [bucket.bucket_image_count for bucket in chunked_buckets] == [6, 6, 6]
    global_padded_rows = sum(
        bucket.bucket_image_count * bucket.bucket_rotation_count
        for bucket in global_buckets
    )
    chunked_padded_rows = sum(
        bucket.bucket_image_count * bucket.bucket_rotation_count
        for bucket in chunked_buckets
    )
    assert chunked_padded_rows == 1920
    assert chunked_padded_rows < global_padded_rows

    observed_ids = []
    for bucket in chunked_buckets:
        assert bucket.bucket_image_count % 3 == 0
        for row, image_index in enumerate(bucket.image_indices.tolist()):
            start = int(layout.rotation_offsets[image_index])
            stop = int(layout.rotation_offsets[image_index + 1])
            count = stop - start
            np.testing.assert_array_equal(
                bucket.local_rotation_ids[row, :count],
                layout.rotation_ids_flat[start:stop],
            )
            assert bucket.local_rotation_mask[row, :count].all()
            assert not bucket.local_rotation_mask[row, count:].any()
            observed_ids.extend(bucket.local_rotation_ids[row, :count].tolist())
    assert observed_ids == layout.rotation_ids_flat.tolist()


@pytest.mark.unit
def test_physical_order_chunks_require_order_preservation_and_no_global_unify():
    layout = _make_layout(np.asarray([17, 65, 33], dtype=np.int32))
    with pytest.raises(ValueError, match="require preserved image order"):
        bucket_local_hypothesis_layout(
            layout,
            image_batch_size=3,
            rotation_block_size=5000,
            max_hypotheses_per_microbatch=4096,
            consecutive_mixed_bucket_size=3,
        )
    with pytest.raises(ValueError, match="cannot use run-global"):
        bucket_local_hypothesis_layout(
            layout,
            image_batch_size=3,
            rotation_block_size=5000,
            max_hypotheses_per_microbatch=4096,
            preserve_image_order=True,
            unify_bucket_sizes=True,
            consecutive_mixed_bucket_size=3,
        )
