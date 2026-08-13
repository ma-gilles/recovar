import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import compact_candidate_capture as capture
from scripts.analyze_k1_partial_fine_topology import (
    load_recovar_candidate_table,
    partial_rotation_map,
)


@pytest.mark.unit
def test_partial_rotation_map_reports_exact_overlap_and_unmatched_rows():
    recovar = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
        ]
    )
    native_matrices = np.asarray(
        [
            recovar[2],
            np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
            recovar[0],
        ]
    )
    factor = np.empty(3, dtype=[("matrix", np.float32, (9,))])
    factor["matrix"] = native_matrices.transpose(0, 2, 1).reshape(3, 9)

    mapping, native_only, recovar_only = partial_rotation_map(factor, recovar)

    np.testing.assert_array_equal(mapping, np.asarray([2, -1, 0]))
    np.testing.assert_array_equal(native_only, np.asarray([1]))
    np.testing.assert_array_equal(recovar_only, np.asarray([1]))


@pytest.mark.unit
def test_production_shard_normalizes_to_dense_partial_topology(tmp_path, monkeypatch):
    rotations = np.stack(
        [np.eye(3, dtype=np.float32), np.diag([-1.0, -1.0, 1.0]).astype(np.float32)]
    )
    scores = np.asarray([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32)
    probs = np.exp(scores, dtype=np.float32)
    probs /= probs.sum(axis=(1, 2), keepdims=True, dtype=np.float32)
    significant = probs >= np.sort(probs.reshape(1, -1), axis=1)[:, -2][:, None, None]
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(capture.CAPTURE_ITERATION_ENV, "3")
    monkeypatch.setattr(capture, "_capture_counter", 0)

    assert capture.maybe_capture_k1_production_bucket(
        iteration=3,
        half=1,
        image_indices=np.asarray([0], dtype=np.int64),
        original_indices=np.asarray([231], dtype=np.int64),
        per_image_inputs={
            "oversampled_rots": [rotations],
            "oversampled_rot_indices": [np.asarray([20, 21], dtype=np.int64)],
            "parent_map": [np.asarray([0, 1], dtype=np.int32)],
            "unique_rot": [np.asarray([100, 101], dtype=np.int32)],
        },
        current_size=64,
        fine_translations=np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
        fine_translation_parent=np.asarray([0, 0], dtype=np.int32),
        scores=scores,
        probs=probs,
        rotation_log_prior=np.asarray([[0.25, 0.5]], dtype=np.float32),
        translation_log_prior=np.asarray([[0.75, 1.0]], dtype=np.float32),
        candidate_mask=np.ones_like(scores, dtype=bool),
        reconstruction_mask=significant,
        log_z=np.log(np.exp(scores, dtype=np.float32).sum(axis=(1, 2), dtype=np.float32)),
        best_log_score=scores[:, 0, 0],
        best_argmax=np.zeros(1, dtype=np.int32),
        max_posterior=probs[:, 0, 0],
    ) == 1

    normalized = load_recovar_candidate_table(next(tmp_path.glob("raw_k1_*.npz")))
    assert int(normalized["original_index"]) == 231
    np.testing.assert_array_equal(normalized["rotations"], rotations)
    np.testing.assert_array_equal(normalized["candidate_mask"], np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(normalized["probs"], probs[0])
    np.testing.assert_array_equal(normalized["production_combined_score"], scores[0])
    np.testing.assert_array_equal(normalized["production_significant"], significant[0])
