"""Source Euler publication preserves RFLOAT metadata without changing score inputs."""

import copy
import json
import logging
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em import sampling
from recovar.em.dense_single_volume import k_class, k_class_results
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _prepare_per_image_pass2_inputs,
)
from recovar.em.dense_single_volume.helpers.types import LocalEMResult, make_relion_stats
from recovar.em.dense_single_volume.local_layout import (
    LocalHypothesisLayout,
    bucket_local_hypothesis_layout,
    build_pass2_hypothesis_layout,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_source", [False, True])
def test_class_prior_override_preserves_source_eulers(dtype, with_source):
    source = (
        np.array([[159.3271497477632, 126.91279408422895, 85.75518260708287]]) if with_source else None
    )
    layout = LocalHypothesisLayout(
        n_global_rotations=1, n_pixels=1, n_psi=1,
        rotation_offsets=np.array([0, 1]), rotation_ids_flat=np.array([0]),
        rotations_flat=np.eye(3, dtype=dtype)[None],
        rotation_log_priors_flat=np.zeros(1, dtype), rotation_counts=np.ones(1, np.int32),
        translation_grid=np.zeros((1, 2), dtype), translation_log_priors=np.zeros((1, 1), dtype),
        source_eulers_flat=source,
    )
    assert k_class._local_layout_for_class(layout, None, 0, 4) is layout
    priors = np.arange(4, dtype=dtype).reshape(4, 1)
    for class_id in range(4):
        result = k_class._local_layout_for_class(layout, priors, class_id, 4)
        assert result.source_eulers_flat is source
        assert result.rotations_flat is layout.rotations_flat
        np.testing.assert_array_equal(result.rotation_log_priors_flat, priors[class_id])
        assert result.rotation_log_priors_flat.dtype == dtype
        buckets = bucket_local_hypothesis_layout(result, 1, 4)
        assert len(buckets) == 1
        if source is None:
            assert buckets[0].local_source_eulers is None
        else:
            np.testing.assert_array_equal(buckets[0].local_source_eulers[0, :1], source)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_source_triplet_and_legacy_arrays(dtype, tmp_path):
    binding = pytest.importorskip("recovar.relion_bind._relion_bind_core")
    # Captured native trial16 last won child4 of parent(direction37, psi3).
    kwargs = dict(
        oversampling_order=1,
        random_perturbation=0.18350759148597717,
        return_rotation_indices=True,
        return_mstep_rotations=True,
        dtype=dtype,
    )
    old = sampling.get_oversampled_rotation_grid_from_samples([37 + 3 * 48], 1, **kwargs)
    new = sampling.get_oversampled_rotation_grid_from_samples([37 + 3 * 48], 1, return_source_eulers=True, **kwargs)
    assert len(old) == 4 and len(new) == 5
    for a, b in zip(old, new[:4], strict=True):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    expected = np.array([159.3271497477632, 126.91279408422895, 85.75518260708287])
    assert new[-1].dtype == np.float64
    np.testing.assert_array_equal(new[-1][4], expected)
    exact_native = binding.get_oversampled_orientations(1, 1, 37, 3, kwargs["random_perturbation"])
    np.testing.assert_array_equal(new[-1], exact_native)
    metrics = dict(matrix_bytes_unchanged=True, source_maxabs=float(np.max(abs(new[-1][4] - expected))))
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    logging.info("%s: %s", dtype, metrics)


@pytest.mark.parametrize("order", [False, True])
def test_sparse_override_source_follows_true_child_permutation(order):
    # IDs repeat by coarse parent; source rows must follow the chosen child, not a global nearest-grid ID.
    parent = np.array([1, 0, 1, 0])
    eulers = np.arange(12, dtype=np.float64).reshape(4, 3) + 2**-35
    matrices = np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()
    kwargs = dict(
        n_coarse_rot=72,
        n_coarse_trans=1,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.zeros(1, np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=matrices,
        fine_rotation_parent_override=parent,
        relion_parent_execution_order=order,
    )
    inputs = _prepare_per_image_pass2_inputs(
        [np.array([0, 1]), np.array([1])], fine_source_eulers_override=eulers, **kwargs
    )
    legacy = _prepare_per_image_pass2_inputs([np.array([0, 1]), np.array([1])], **kwargs)
    for i in range(2):
        ids = inputs["oversampled_rot_indices"][i]
        np.testing.assert_array_equal(inputs["source_eulers"][i], eulers[ids])
        for key in legacy:
            if key != "source_eulers":
                np.testing.assert_array_equal(inputs[key][i], legacy[key][i])
        assert legacy["source_eulers"][i] is None


def test_local_union_and_bucket_source_alignment():
    pytest.importorskip("recovar.relion_bind._relion_bind_core")
    layout = build_pass2_hypothesis_layout(
        [np.array([7, 3]), np.array([3])],
        72,
        1,
        0,
        np.zeros((1, 2), np.float32),
        translation_step=1.0,
        oversampling_order=1,
        random_perturbation=0.13,
        rotation_index_order="relion_hidden",
    )
    for i in range(2):
        parents = np.unique([7, 3] if i == 0 else [3])
        expected = sampling.get_oversampled_rotation_grid_from_samples(
            parents, 0, random_perturbation=0.13, return_source_eulers=True, rotation_index_order="relion_hidden"
        )[-1]
        start, stop = layout.rotation_offsets[i : i + 2]
        np.testing.assert_array_equal(layout.source_eulers_flat[start:stop], expected)
    for bucket in bucket_local_hypothesis_layout(layout, 2, 32):
        for row, idx in enumerate(bucket.image_indices):
            start, stop = layout.rotation_offsets[idx : idx + 2]
            np.testing.assert_array_equal(
                bucket.local_source_eulers[row, : stop - start], layout.source_eulers_flat[start:stop]
            )


def _stats(n):
    return make_relion_stats(
        log_evidence_per_image=np.zeros(n),
        best_log_score_per_image=np.zeros(n),
        max_posterior_per_image=np.ones(n, np.float32),
        rotation_posterior_sums=np.ones(1),
    )


def test_direct_k1_result_preserves_host_eulers(monkeypatch):
    eulers = np.array([[13.0 + 2**-35, 27.0, 41.0]])
    output = LocalEMResult(
        np.zeros(1, np.complex64), np.ones(1, np.float32), np.zeros(1, np.int32), _stats(1), best_pose_eulers_deg=eulers
    )
    monkeypatch.setattr(k_class, "run_local_em_exact", lambda *a, **kw: output)
    result = k_class.run_local_k_class_em(
        SimpleNamespace(n_images=1),
        np.zeros((1, 8), np.complex64),
        np.ones(8),
        np.ones(4),
        SimpleNamespace(n_images=1),
        "linear_interp",
        return_best_pose_details=True,
    )
    assert isinstance(result.best_pose_eulers_deg, np.ndarray)
    np.testing.assert_array_equal(result.best_pose_eulers_deg, eulers)


def test_inactive_class_without_metadata_does_not_erase_winner():
    eulers = np.array([[1.0 + 2**-40, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = k_class_results._assemble_result(
        class_log_evidence=np.array([[0.0, -20.0], [-20.0, 0.0], [-np.inf, -np.inf], [-np.inf, -np.inf]]),
        new_means=None,
        Ft_y=[np.zeros(1, np.complex64)] * 4,
        Ft_ctf=[np.ones(1, np.float32)] * 4,
        per_class_hard_assignments=np.zeros((4, 2), np.int32),
        per_class_stats=tuple(
            _stats(2)._replace(best_log_score_per_image=score)
            for score in np.array([[0.0, -20.0], [-20.0, 0.0], [-np.inf, -np.inf], [-np.inf, -np.inf]])
        ),
        noise_stats=None,
        per_class_best_pose_eulers_deg=[eulers, eulers + 10, None, None],
    )
    np.testing.assert_array_equal(result.best_pose_eulers_deg, np.stack([eulers[0], eulers[1] + 10]))


def test_true_local_winner_gathers_eulers_without_changing_other_buffers():
    n = 3
    buffers = engine._LocalPostprocessBuffers(
        hard_assignment=np.zeros(n, np.int32),
        log_evidence_per_image=np.zeros(n),
        best_log_score_per_image=np.zeros(n),
        max_posterior_per_image=np.zeros(n),
        rotation_posterior_sums=np.zeros(8),
        transfer_profile=defaultdict(float),
        chunk_nonzero_posterior_rows=[],
        chunk_significant_samples=[],
        chunk_reconstruction_rows=[],
        seen_global_rotations=np.zeros(0, bool),
        seen_nonzero_global_rotations=np.zeros(0, bool),
        seen_reconstruction_global_rotations=np.zeros(0, bool),
        best_pose_rotations=np.zeros((n, 3, 3), np.float32),
        best_pose_translations=np.zeros((n, 2), np.float32),
        best_pose_rotation_ids=np.zeros(n, np.int32),
    )
    old = copy.deepcopy(buffers)
    buffers.best_pose_eulers_deg = np.zeros((n, 3), np.float64)
    eulers = np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 2**-37
    kwargs = dict(
        image_indices=np.array([2, 0]),
        local_rotation_ids=np.array([[4, 4], [7, 7]]),
        local_rotation_mask=np.ones((2, 2), bool),
        local_rotations=np.tile(np.eye(3, dtype=np.float32), (2, 2, 1, 1)),
        local_rotation_posterior_ids=None,
        translation_grid=np.zeros((2, 2), np.float32),
        n_trans=2,
        best_argmax=np.array([2, 0]),
        batch_norm=np.zeros((2, 1)),
        log_Z=np.zeros(2),
        best_log_score=np.zeros(2),
        max_posterior=np.ones(2),
        probs_sum_t=np.ones((2, 2)),
        n_significant_samples=None,
        reconstruction_sample_mask=None,
        collect_profile_stats=False,
        reconstruction_row_count=0,
        reconstruction_take_indices=None,
        reconstruction_pack_mask=None,
        host_prefix=True,
    )
    engine._postprocess_local_bucket(**kwargs, buffers=old)
    engine._postprocess_local_bucket(**kwargs, buffers=buffers, local_source_eulers=eulers)
    np.testing.assert_array_equal(buffers.best_pose_eulers_deg[[2, 0]], np.stack([eulers[0, 1], eulers[1, 0]]))
    for name, value in vars(old).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(buffers, name), value)
    np.testing.assert_array_equal(eulers, np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 2**-37)


def test_loader_reordering_preserves_source_rows_and_legacy_unavailability():
    from recovar.em.dense_single_volume.local_layout import LocalBucketSpec

    eulers = np.arange(18, dtype=np.float64).reshape(3, 2, 3) + 2**-37
    bucket = LocalBucketSpec(
        image_indices=np.array([9, 4, 7]),
        bucket_image_count=3,
        bucket_rotation_count=2,
        actual_rotation_counts=np.array([2, 2, 2]),
        local_rotation_ids=np.tile([1, 2], (3, 1)),
        local_rotations=np.tile(np.eye(3, dtype=np.float32), (3, 2, 1, 1)),
        local_rotation_log_prior=np.zeros((3, 2), np.float32),
        local_rotation_mask=np.ones((3, 2), bool),
        translation_log_prior=np.zeros((3, 1), np.float32),
        local_source_eulers=eulers,
    )
    reordered = engine._reorder_bucket_to_indices(bucket, np.array([7, 4, 9]))
    np.testing.assert_array_equal(reordered.local_source_eulers, eulers[[2, 1, 0]])
    np.testing.assert_array_equal(reordered.local_rotations, bucket.local_rotations[[2, 1, 0]])
    assert reordered.local_source_eulers.dtype == np.float64
    assert engine._reorder_bucket_to_indices(bucket, bucket.image_indices) is bucket
    from dataclasses import replace

    legacy = engine._reorder_bucket_to_indices(replace(bucket, local_source_eulers=None), np.array([7, 4, 9]))
    assert legacy.local_source_eulers is None
