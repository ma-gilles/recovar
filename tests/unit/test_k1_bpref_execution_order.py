from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV,
    _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV,
    _BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV,
    _BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV,
    _BPREF_REVERSE_PHYSICAL_ORDER_ENV,
    _bucket_pass2_inputs,
    _env_flag_enabled,
    _load_bpref_execution_order_local_override,
    _normalize_pass2_bucket,
    _resolve_bpref_execution_bucket_policy,
    _resolve_bpref_processing_order,
)
from recovar.em.dense_single_volume.iteration_loop import (
    _validate_bpref_particle_order_scope,
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


def test_sparse_pass2_ordered_chunks_respect_hypothesis_and_image_caps():
    counts = [16, 16, 256, 16, 16, 16]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=np.arange(len(counts), dtype=np.int64),
        processing_order_chunk_size=4,
        max_hypotheses_per_microbatch=2048,
        max_images_per_microbatch=3,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 256, 16]
    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == list(
        range(len(counts))
    )


def test_sparse_pass2_ordered_chunks_do_not_scale_with_support_runs():
    counts = ([16, 32] * 500) + [256] + ([16, 32] * 100)
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    max_hypotheses = 814_509
    n_fine_trans = 116

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        processing_order_override=np.arange(len(counts), dtype=np.int64),
        processing_order_chunk_size=220,
        max_hypotheses_per_microbatch=max_hypotheses,
        max_images_per_microbatch=156,
    )

    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == list(
        range(len(counts))
    )
    assert len(buckets) < 20
    for bucket in buckets:
        image_count = len(bucket["image_indices"])
        assert image_count <= 156
        assert image_count * int(bucket["bucket_size"]) * n_fine_trans <= max_hypotheses


def test_sparse_pass2_execution_order_batches_only_consecutive_equal_sizes():
    counts = [16, 16, 64, 64, 16, 32, 32, 32]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([4, 0, 1, 2, 3, 7, 5, 6], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_batch_consecutive_bucket_sizes=True,
        max_hypotheses_per_microbatch=256,
        max_images_per_microbatch=8,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [4, 0, 1],
        [2],
        [3],
        [7, 5],
        [6],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 64, 64, 32, 32]
    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == order.tolist()


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


def test_fresh_k1_defaults_to_bounded_mixed_support_buckets(monkeypatch):
    monkeypatch.delenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV, raising=False)
    monkeypatch.delenv(
        _BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV,
        raising=False,
    )
    assert _resolve_bpref_execution_bucket_policy(
        preserve_bpref_particle_order=True,
        processing_order_group_by_bucket_size=False,
    ) == (220, False)

    monkeypatch.setenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV, "4")
    assert _resolve_bpref_execution_bucket_policy(
        preserve_bpref_particle_order=True,
        processing_order_group_by_bucket_size=False,
    ) == (4, False)


def test_consecutive_equal_support_batching_is_an_explicit_diagnostic(monkeypatch):
    monkeypatch.delenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV, raising=False)
    monkeypatch.setenv(_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV, "1")
    assert _resolve_bpref_execution_bucket_policy(
        preserve_bpref_particle_order=True,
        processing_order_group_by_bucket_size=False,
    ) == (1, True)

    monkeypatch.setenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV, "4")
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_bpref_execution_bucket_policy(
            preserve_bpref_particle_order=True,
            processing_order_group_by_bucket_size=False,
        )

    monkeypatch.delenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV)
    with pytest.raises(ValueError, match="requires the guarded fresh K=1"):
        _resolve_bpref_execution_bucket_policy(
            preserve_bpref_particle_order=False,
            processing_order_group_by_bucket_size=False,
        )


def test_bounded_fresh_k1_default_does_not_expand_to_other_order_policies(monkeypatch):
    monkeypatch.delenv(_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV, raising=False)
    monkeypatch.delenv(
        _BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV,
        raising=False,
    )
    assert _resolve_bpref_execution_bucket_policy(
        preserve_bpref_particle_order=False,
        processing_order_group_by_bucket_size=False,
    ) == (1, False)
    assert _resolve_bpref_execution_bucket_policy(
        preserve_bpref_particle_order=True,
        processing_order_group_by_bucket_size=True,
    ) == (1, False)


def test_mixed_support_padding_keeps_float_outputs_within_four_ulps():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(1, 16, 116)).astype(np.float32)
    padded_scores = np.full((1, 32, 116), -np.inf, dtype=np.float32)
    padded_scores[:, :16] = scores

    unpadded = _normalize_pass2_bucket(jnp.asarray(scores))
    padded = _normalize_pass2_bucket(jnp.asarray(padded_scores))
    for field_index, (unpadded_field, padded_field) in enumerate(zip(unpadded, padded)):
        padded_array = np.asarray(padded_field)
        if field_index == 1:
            padded_array = padded_array[:, :16]
        unpadded_array = np.asarray(unpadded_field)
        if np.issubdtype(unpadded_array.dtype, np.floating):
            # Hopper may select a different float64 reduction tree when the
            # all-zero padded tail changes shape.  Bound that hardware-level
            # effect tightly while keeping discrete winners exactly equal.
            np.testing.assert_array_max_ulp(unpadded_array, padded_array, maxulp=4)
        else:
            np.testing.assert_array_equal(unpadded_array, padded_array)


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


def test_reverse_physical_execution_order_is_narrow_and_exact(monkeypatch):
    monkeypatch.setenv(_BPREF_REVERSE_PHYSICAL_ORDER_ENV, "1")
    assert np.array_equal(
        _resolve_bpref_processing_order(
            5,
            preserve_bpref_particle_order=True,
        ),
        np.asarray([4, 3, 2, 1, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="requires the guarded fresh K=1"):
        _resolve_bpref_processing_order(
            5,
            preserve_bpref_particle_order=False,
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


def test_fresh_k1_bpref_order_scope_accepts_only_unsealed_iteration_zero():
    kwargs = {
        "preserve_bpref_particle_order": True,
        "n_classes": 1,
        "init_relion_iteration": 0,
        "perturb_replay_relion_dir": None,
        "replay_iteration_overrides": [None],
        "sealed_sampling_state": None,
        "sealed_scoring_context": None,
    }
    _validate_bpref_particle_order_scope(**kwargs)

    for override, match in (
        ({"n_classes": 4}, "K=1-only"),
        ({"init_relion_iteration": 1}, "fresh iteration-0"),
        ({"perturb_replay_relion_dir": "relion"}, "perturbation replay"),
        ({"replay_iteration_overrides": [None, {"state": 1}]}, "numbered replay"),
        ({"sealed_sampling_state": object()}, "sealed boundary"),
        ({"sealed_scoring_context": object()}, "sealed boundary"),
    ):
        invalid = dict(kwargs)
        invalid.update(override)
        with pytest.raises(ValueError, match=match):
            _validate_bpref_particle_order_scope(**invalid)


def test_bpref_order_scope_is_dormant_when_preservation_is_disabled():
    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=False,
        n_classes=4,
        init_relion_iteration=9,
        perturb_replay_relion_dir="relion",
        replay_iteration_overrides=[None, {"state": 1}],
        sealed_sampling_state=object(),
        sealed_scoring_context=object(),
    )


def test_state_swap_fresh_k1_bpref_order_scope_requires_complete_unsealed_replay():
    kwargs = {
        "preserve_bpref_particle_order": True,
        "n_classes": 1,
        "init_relion_iteration": 0,
        "perturb_replay_relion_dir": "relion",
        "replay_iteration_overrides": [None, {"state": 1}],
        "sealed_sampling_state": None,
        "sealed_scoring_context": None,
        "allow_state_swap_fresh_bpref_particle_order": True,
    }
    _validate_bpref_particle_order_scope(**kwargs)

    for override, match in (
        ({"init_relion_iteration": 1}, "fresh iteration-0"),
        ({"perturb_replay_relion_dir": None}, "requires perturbation replay"),
        ({"replay_iteration_overrides": [None]}, "requires numbered replay state"),
        ({"sealed_sampling_state": object()}, "cannot alter a sealed boundary"),
    ):
        invalid = dict(kwargs)
        invalid.update(override)
        with pytest.raises(ValueError, match=match):
            _validate_bpref_particle_order_scope(**invalid)
