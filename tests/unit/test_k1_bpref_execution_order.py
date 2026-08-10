from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV,
    _BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV,
    _bucket_pass2_inputs,
    _env_flag_enabled,
    _load_bpref_execution_order_local_override,
    _resolve_bpref_processing_order,
)
from recovar.em.dense_single_volume.k_class import (
    _apply_bpref_particle_order_policy,
)


def test_sparse_pass2_execution_order_override_is_exact_and_single_particle():
    counts = [16, 64, 32, 16]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [2],
        [0],
        [3],
        [1],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [32, 16, 16, 64]

    with pytest.raises(ValueError, match="must be a permutation"):
        _bucket_pass2_inputs(
            per_image,
            n_fine_trans=4,
            processing_order_override=np.asarray([0, 0, 2, 3]),
        )


def test_sparse_pass2_execution_order_override_chunks_only_adjacent_particles():
    counts = [16, 64, 32, 16, 48]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_chunk_size=2,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [2, 0],
        [3, 1],
        [4],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [32, 64, 64]
    with pytest.raises(ValueError, match="must be positive"):
        _bucket_pass2_inputs(
            per_image,
            n_fine_trans=4,
            processing_order_override=order,
            processing_order_chunk_size=0,
        )


def test_sparse_pass2_execution_order_can_stay_stable_within_size_buckets():
    counts = [16, 64, 32, 16, 48]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_group_by_bucket_size=True,
        max_images_per_microbatch=8,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [0, 3],
        [2],
        [1, 4],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 32, 64]


def test_grouped_execution_order_environment_flag_uses_module_parser(monkeypatch):
    monkeypatch.setenv(_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV, "1")
    assert _env_flag_enabled(_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV)


def test_execution_order_file_is_fail_closed(monkeypatch, tmp_path):
    order_path = tmp_path / "order.txt"
    order_path.write_text("2\n0\n1\n")
    monkeypatch.setenv(_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV, str(order_path))
    assert np.array_equal(
        _load_bpref_execution_order_local_override(3),
        np.asarray([2, 0, 1]),
    )

    order_path.write_text("2\n0\n0\n")
    with pytest.raises(ValueError, match="must contain a permutation"):
        _load_bpref_execution_order_local_override(3)


def test_production_execution_order_is_identity_and_rejects_diagnostic_override(
    monkeypatch,
    tmp_path,
):
    assert np.array_equal(
        _resolve_bpref_processing_order(
            4,
            preserve_bpref_particle_order=True,
        ),
        np.arange(4, dtype=np.int64),
    )

    order_path = tmp_path / "order.txt"
    order_path.write_text("2\n0\n1\n")
    monkeypatch.setenv(_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV, str(order_path))
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_bpref_processing_order(
            3,
            preserve_bpref_particle_order=True,
        )


def test_sparse_pass_order_policy_is_k1_only_and_dormant_by_default():
    common = {"sentinel": object()}
    assert not _apply_bpref_particle_order_policy(
        common,
        {},
        n_classes=4,
    )
    assert "preserve_bpref_particle_order" not in common

    assert _apply_bpref_particle_order_policy(
        common,
        {"preserve_bpref_particle_order": True},
        n_classes=1,
    )
    assert common["preserve_bpref_particle_order"] is True

    with pytest.raises(ValueError, match="K=1-only"):
        _apply_bpref_particle_order_policy(
            {},
            {"preserve_bpref_particle_order": True},
            n_classes=4,
        )
