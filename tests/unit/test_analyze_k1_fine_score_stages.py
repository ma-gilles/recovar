from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import analyze_k1_fine_score_stages as analyzer
from scripts.validate_relion_fine_score_capture import ACTIVE, CANDIDATE_DTYPE


def test_complete_stage_comparison_closes_on_additively_shifted_scores(tmp_path: Path, monkeypatch) -> None:
    rotation_dtype = np.dtype([("matrix", "<f4", (3, 3))])
    native_rotations = np.zeros(2, dtype=rotation_dtype)
    native_rotations[0]["matrix"] = np.eye(3, dtype=np.float32)
    native_rotations[1]["matrix"] = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    recovar_rotations = native_rotations["matrix"].transpose(0, 2, 1).copy()

    translation_dtype = np.dtype([("x", "<f8"), ("y", "<f8")])
    native_translations = np.zeros(2, dtype=translation_dtype)
    native_translations[1]["x"] = -2.0 * np.pi / 128.0
    recovar_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    candidates = np.zeros(2, dtype=CANDIDATE_DTYPE)
    candidates["rotation_local"] = [0, 1]
    candidates["translation_id"] = [0, 1]
    candidates["flags"] = ACTIVE
    candidates["raw_diff2"] = [10.0, 11.0]
    candidates["combined_preexponent"] = [-10.0, -11.0]
    candidates["post_exponent_weight"] = [2.0, 1.0]

    header = [0] * 48
    header[32] = int(np.asarray(np.float32(3.0)).view(np.uint32))
    header[35] = 1
    factor = SimpleNamespace(
        stack_index=1204,
        rotations=native_rotations,
        translations=native_translations,
        geometry_only=True,
        header=tuple(header[:45] + [1] + header[46:]),
        sha256="factor-sha",
    )
    score = SimpleNamespace(
        stack_index=1204,
        candidates=candidates,
        header=tuple(header),
        sha256="score-sha",
    )
    monkeypatch.setattr(analyzer, "load_factor_capture", lambda _path: factor)
    monkeypatch.setattr(analyzer, "load_fine_score_capture", lambda _path: score)

    candidate_mask = np.asarray([[True, False], [False, True]])
    scores_pre_prior = np.full((2, 2), -np.inf, dtype=np.float32)
    scores_pre_prior[0, 0] = 90.0
    scores_pre_prior[1, 1] = 89.0
    scores_with_prior = scores_pre_prior.copy()
    probs = np.zeros((2, 2), dtype=np.float64)
    probs[0, 0] = 2.0 / 3.0
    probs[1, 1] = 1.0 / 3.0
    reconstruction_mask = np.asarray([[True, False], [False, False]])
    raw_diff2 = np.full((2, 2), np.nan, dtype=np.float32)
    raw_diff2[0, 0] = 10.0
    raw_diff2[1, 1] = 11.0
    capture = tmp_path / "pass2.npz"
    np.savez(
        capture,
        rotations=recovar_rotations,
        fine_translations=recovar_translations,
        candidate_mask=candidate_mask,
        scores_pre_prior=scores_pre_prior,
        rotation_log_prior=np.zeros(2, dtype=np.float32),
        translation_log_prior=np.zeros(2, dtype=np.float32),
        scores_with_prior=scores_with_prior,
        probs=probs,
        reconstruction_mask=reconstruction_mask,
        raw_operand_raw_diff2=raw_diff2,
    )

    report = analyzer.analyze(
        native_factor=tmp_path / "factor.bin",
        native_fine_score=tmp_path / "score.bin",
        recovar_capture=capture,
        physical_image_size=128,
        top_count=2,
    )

    assert report["first_exact_unequal_boundary"] == "all_stages_exact"
    assert all(report["stage_exact"].values())
    assert report["native_winner"]["recovar_rotation_row"] == 0
    assert report["recovar_dense_winner"] == [0, 0]
    assert report["native_active_missing_candidate_rotation_count"] == 0
    assert report["native_active_missing_candidate_groups"] == []
    assert report["recovar_only_candidate_tuple_count"] == 0
    assert report["recovar_only_candidate_groups"] == []
    assert report["support_boundary"]["native_ranked"]["selected_count"] == 1
    assert report["support_boundary"]["native_ranked"]["records"] == [
        {
            "rank_one_based": 1,
            "tuple_key": [0, 0],
            "native_weight_float32": 2.0,
            "native_weight_float32_bits": int(np.float32(2.0).view(np.uint32)),
            "native_posterior": 2.0 / 3.0,
            "recovar_posterior": 2.0 / 3.0,
            "native_cumulative_mass": 2.0 / 3.0,
            "recovar_cumulative_mass": 2.0 / 3.0,
            "native_selected": True,
            "recovar_selected": True,
        },
        {
            "rank_one_based": 2,
            "tuple_key": [1, 1],
            "native_weight_float32": 1.0,
            "native_weight_float32_bits": int(np.float32(1.0).view(np.uint32)),
            "native_posterior": 1.0 / 3.0,
            "recovar_posterior": 1.0 / 3.0,
            "native_cumulative_mass": 1.0,
            "recovar_cumulative_mass": 1.0,
            "native_selected": False,
            "recovar_selected": False,
        },
    ]

    raw_diff2[1, 1] = 11.25
    np.savez(
        capture,
        rotations=recovar_rotations,
        fine_translations=recovar_translations,
        candidate_mask=candidate_mask,
        scores_pre_prior=scores_pre_prior,
        rotation_log_prior=np.zeros(2, dtype=np.float32),
        translation_log_prior=np.zeros(2, dtype=np.float32),
        scores_with_prior=scores_with_prior,
        probs=probs,
        reconstruction_mask=reconstruction_mask,
        raw_operand_raw_diff2=raw_diff2,
    )
    raw_report = analyzer.analyze(
        native_factor=tmp_path / "factor.bin",
        native_fine_score=tmp_path / "score.bin",
        recovar_capture=capture,
        physical_image_size=128,
        top_count=2,
    )
    assert raw_report["first_exact_unequal_boundary"] == "raw_diff2"
    assert raw_report["first_mismatch"]["raw_diff2"]["recovar_rotation_row"] == 1
    assert raw_report["first_mismatch"]["raw_diff2"]["recovar_translation_row"] == 1
    raw_diff2[1, 1] = 11.0

    candidate_mask[0, 1] = True
    scores_pre_prior[0, 1] = 88.0
    scores_with_prior[0, 1] = 88.0
    probs[0, 1] = 0.01
    np.savez(
        capture,
        rotations=recovar_rotations,
        fine_translations=recovar_translations,
        candidate_mask=candidate_mask,
        scores_pre_prior=scores_pre_prior,
        rotation_log_prior=np.zeros(2, dtype=np.float32),
        translation_log_prior=np.zeros(2, dtype=np.float32),
        scores_with_prior=scores_with_prior,
        probs=probs,
        reconstruction_mask=reconstruction_mask,
        oversampled_rot_indices=np.asarray([10, 11], dtype=np.int64),
        parent_map=np.asarray([4, 5], dtype=np.int32),
        raw_operand_raw_diff2=raw_diff2,
    )

    extra_report = analyzer.analyze(
        native_factor=tmp_path / "factor.bin",
        native_fine_score=tmp_path / "score.bin",
        recovar_capture=capture,
        physical_image_size=128,
        top_count=2,
    )

    assert extra_report["first_exact_unequal_boundary"] == "candidate_tuple_presence"
    assert extra_report["stage_exact"]["candidate_tuple_presence"] is False
    assert extra_report["recovar_active_candidate_count"] == 3
    assert extra_report["recovar_only_candidate_tuple_count"] == 1
    assert extra_report["recovar_only_candidate_groups"] == [
        {
            "recovar_rotation_row": 0,
            "recovar_global_rotation_id": 10,
            "recovar_parent_row": 4,
            "extra_translation_count": 1,
            "extra_translation_rows_recovar": [1],
        }
    ]
