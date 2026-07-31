from pathlib import Path

import numpy as np

from scripts.analyze_em_k4_raw_operand_repeatability import analyze


def _write_archive(
    path: Path,
    projection_offset: float,
    high_shell_offset: float = 0.0,
) -> None:
    n_rot = 2
    n_trans = 3
    shifted = np.arange(n_trans * 2, dtype=np.float32).reshape(n_trans, 2).astype(np.complex64)
    projection = (
        np.arange(n_rot * 2, dtype=np.float32).reshape(n_rot, 2)
        + np.float32(projection_offset)
    ).astype(np.complex64)
    values = {
        "original_index": np.int64(42),
        "class_index": np.int64(0),
        "current_size": np.int64(14),
        "candidate_mask": np.ones((n_rot, n_trans), dtype=bool),
        "rotations": np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1)),
        "oversampled_rot_indices": np.arange(n_rot, dtype=np.int64),
        "parent_map": np.arange(n_rot, dtype=np.int32),
        "fine_translations": np.zeros((n_trans, 2), dtype=np.float32),
        "fine_translation_parent": np.arange(n_trans, dtype=np.int32),
        "raw_operand_relion_full_to_compact": np.arange(2, dtype=np.int32),
        "raw_operand_actual_rotation_count": np.int64(n_rot),
        "raw_operand_pair_mask": np.empty((0,), dtype=bool),
        "raw_operand_pair_rotation_row": np.empty((0,), dtype=np.int32),
        "raw_operand_pair_translation_idx": np.empty((0,), dtype=np.int32),
        "raw_operand_shifted_corrected": shifted,
        "raw_operand_proj_half": projection,
        "raw_operand_corr_img_score": np.ones(2, dtype=np.float32),
        "raw_operand_half_weights": np.ones(2, dtype=np.float32),
        "raw_operand_highres_xi2_half": np.float32(high_shell_offset),
    }
    values["relion_raw_diff2"] = _fake_replay(values)
    values["raw_operand_raw_diff2"] = values["relion_raw_diff2"]
    np.savez(path, **values)


def _fake_replay(values: dict[str, np.ndarray]) -> np.ndarray:
    projection = np.asarray(values["raw_operand_proj_half"]).real
    shifted = np.asarray(values["raw_operand_shifted_corrected"]).real
    return (
        projection[:, :1]
        + shifted[None, :, 0]
        + np.asarray(values["raw_operand_highres_xi2_half"], dtype=np.float32)
    ).astype(np.float32)


def test_analyzer_finds_unique_projection_divergence(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_archive(first, 0.0)
    _write_archive(second, 1.0)

    result = analyze(first, second, replay_fn=_fake_replay)

    assert result["identity_equal"]
    assert result["topology_equal"]
    assert result["self_replay_passed"]
    assert not result["repeatable"]
    assert result["differing_operand_families"] == ["projection"]
    assert result["classification"] == "first_captured_divergence_projection"
    assert result["single_family_substitutions"]["projection"]["mismatch_count"] == 0
    assert result["maximum_mismatch_reduction_families"] == ["projection"]
    assert result["scorecard_change_admissible"] is False
    assert result["correlation_used"] is False


def test_analyzer_reports_all_exact_substitution_ties(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_archive(first, 0.0, 0.0)
    _write_archive(second, 1.0, 1.0)

    result = analyze(first, second, replay_fn=_fake_replay)

    assert result["differing_operand_families"] == [
        "projection",
        "high_shell_scalar",
    ]
    assert result["classification"] == "multiple_operands_differ_attribution_tie"
    assert result["maximum_mismatch_reduction_families"] == [
        "projection",
        "high_shell_scalar",
    ]
