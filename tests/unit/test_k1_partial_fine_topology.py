from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import compact_candidate_capture as capture
from scripts.analyze_k1_partial_fine_topology import (
    _native_significant_count,
    _tuple_sequence_report,
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
def test_native_significant_count_accepts_full_operand_capture():
    header = [0] * 64
    header[45] = 229
    factor = SimpleNamespace(header=tuple(header), geometry_only=False)

    assert _native_significant_count(factor, 1664) == 229


@pytest.mark.unit
def test_tuple_sequence_report_distinguishes_set_from_order():
    native = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    recovar = np.asarray([[1, 2], [5, 6], [3, 4]], dtype=np.int64)

    report = _tuple_sequence_report(native, recovar)

    assert not report["exact"]
    assert report["equal_position_count"] == 1
    assert report["first_mismatch"] == {
        "position": 1,
        "native_key": [3, 4],
        "recovar_key": [5, 6],
    }

    exact = _tuple_sequence_report(native[[0, 2]], recovar[[0, 1]])
    assert exact["exact"]
    assert exact["equal_position_count"] == 2


@pytest.mark.unit
def test_legacy_loader_skips_unused_large_operand_fields(tmp_path):
    path = tmp_path / "pass2_orig000001_cs060.npz"
    np.savez(
        path,
        original_index=np.int64(1),
        rotations=np.eye(3, dtype=np.float32)[None],
        fine_translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        candidate_mask=np.asarray([[True, False]], dtype=bool),
        probs=np.asarray([[1.0, 0.0]], dtype=np.float32),
        # Loading this object array with allow_pickle=False raises.  It models
        # an unused projected-reference field and proves the selective path.
        proj_half=np.asarray([object()], dtype=object),
    )

    normalized = load_recovar_candidate_table(path)

    np.testing.assert_array_equal(normalized["candidate_sequence"], [[0, 0]])
    assert "proj_half" not in normalized


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
    np.testing.assert_array_equal(normalized["rotation_global_index"], [20, 21])
    np.testing.assert_array_equal(normalized["rotation_parent_global"], [100, 101])
    np.testing.assert_array_equal(normalized["candidate_mask"], np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(
        normalized["candidate_sequence"],
        [[0, 0], [0, 1], [1, 0], [1, 1]],
    )
    np.testing.assert_array_equal(normalized["probs"], probs[0])
    np.testing.assert_array_equal(normalized["production_combined_score"], scores[0])
    np.testing.assert_array_equal(normalized["production_significant"], significant[0])
