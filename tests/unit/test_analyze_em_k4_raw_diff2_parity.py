from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import analyze_em_k4_raw_diff2_parity as analyzer


def _classification(**overrides: bool) -> str:
    gates = {
        "support_exact": True,
        "common_min_bitwise_exact": True,
        "raw_diff2_bitwise_exact": True,
        "centered_pre_prior_bitwise_exact": True,
        "native_target_tied": True,
        "recovar_target_tied": True,
    }
    gates.update(overrides)
    return analyzer.classify_raw_diff2_parity(**gates)


def _score_classification(**overrides: bool) -> str:
    gates = {
        "support_exact": True,
        "rotation_prior_bitwise_exact": True,
        "translation_prior_bitwise_exact": True,
        "saved_score_replay_bitwise_exact": True,
        "combined_score_bitwise_exact": True,
        "maximum_tie_sets_exact": True,
    }
    gates.update(overrides)
    return analyzer.classify_score_path_parity(**gates)


def test_classifies_exact_raw_diff2_parity() -> None:
    assert _classification() == analyzer.PASS_CLASSIFICATION


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("support_exact", "support"),
        ("common_min_bitwise_exact", "common_min"),
        ("raw_diff2_bitwise_exact", "raw_diff2"),
        ("centered_pre_prior_bitwise_exact", "centered_pre_prior"),
        ("native_target_tied", "native_target_tie"),
        ("recovar_target_tied", "recovar_target_tie"),
    ],
)
def test_classifies_each_raw_diff2_failure(field: str, suffix: str) -> None:
    assert _classification(**{field: False}).endswith(suffix)


def test_classifies_exact_raw_score_path_parity() -> None:
    assert _score_classification() == analyzer.PASS_SCORE_CLASSIFICATION


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("support_exact", "support"),
        ("rotation_prior_bitwise_exact", "rotation_prior"),
        ("translation_prior_bitwise_exact", "translation_prior"),
        ("saved_score_replay_bitwise_exact", "saved_score_replay"),
        ("combined_score_bitwise_exact", "combined_score"),
        ("maximum_tie_sets_exact", "maximum_tie_sets"),
    ],
)
def test_classifies_each_raw_score_path_failure(
    field: str,
    suffix: str,
) -> None:
    assert _score_classification(**{field: False}).endswith(suffix)


def test_relion_score_replay_preserves_float32_operation_order() -> None:
    raw = np.asarray([501.4734191894531], dtype=np.float32)
    rotation = np.asarray([-4.860062599182129], dtype=np.float32)
    translation = np.asarray([-0.05005118250846863], dtype=np.float32)
    minimum = np.float32(500.6817321777344)

    observed = analyzer._relion_score_replay(
        raw,
        rotation,
        translation,
        minimum,
    )
    expected = np.subtract(
        np.add(
            np.add(rotation, translation, dtype=np.float32),
            minimum,
            dtype=np.float32,
        ),
        raw,
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        observed.view(np.uint32),
        expected.view(np.uint32),
    )
    assert observed[0] == np.float32(-5.701812744140625)


def test_float32_from_bits_round_trips() -> None:
    value = np.float32(500.6817321777344)

    observed = analyzer._float32_from_bits(int(value.view(np.uint32)))

    assert observed.view(np.uint32) == value.view(np.uint32)


def _raw_values_with_ulp_offsets(offsets: list[int]) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    native = np.full(len(offsets), np.float32(1.0), dtype=np.float32)
    recovar_bits = (
        native.view(np.uint32).astype(np.int64)
        + np.asarray(offsets, dtype=np.int64)
    ).astype(np.uint32)
    return native, recovar_bits.view(np.float32)


def test_raw_mismatch_strata_replay_and_select_fixed_representative() -> None:
    native, recovar = _raw_values_with_ulp_offsets([3, 3, 2, -4, 0, 1])

    report = analyzer._raw_mismatch_strata(
        native_raw=native,
        recovar_raw=recovar,
        native_candidate_index=np.asarray([40, 10, 30, 20, 50, 60]),
        native_rotation_local=np.asarray([7, 7, 8, 8, 9, 9]),
        mapped_recovar_rotation=np.asarray([2, 2, 1, 1, 3, 3]),
        translation_id=np.asarray([0, 1, 0, 1, 0, 1]),
    )

    assert report["mismatch_count"] == 5
    assert report["positive_mismatch_count"] == 4
    assert report["negative_mismatch_count"] == 1
    assert report["zero_delta_bitwise_mismatch_count"] == 0
    assert report["partition_replay_exact"] is True
    for key in ("rotation_strata", "translation_strata"):
        partition = report[key]
        assert partition["mismatch_count"] == 5
        assert partition["flattened_partition_signed_replay"] == report[
            "signed_raw_delta"
        ]
        assert partition["flattened_partition_l1_replay"] == report[
            "raw_delta_l1"
        ]
        assert partition["rounded_group_signed_replay_residual"] == 0.0
        assert partition["rounded_group_l1_replay_residual"] == 0.0

    assert report["rotation_strata"]["top_10"][0][
        "mapped_recovar_rotation"
    ] == 2
    assert report["translation_strata"]["top_10"][0][
        "translation_id"
    ] == 1
    assert report["selected_representative"] == {
        "native_candidate_index": 10,
        "native_rotation_local": 7,
        "mapped_recovar_rotation": 2,
        "translation_id": 1,
        "native_raw_diff2": 1.0,
        "native_raw_diff2_bits": int(np.float32(1.0).view(np.uint32)),
        "recovar_raw_diff2": float(recovar[1]),
        "recovar_raw_diff2_bits": int(recovar[1].view(np.uint32)),
        "delta_recovar_minus_native": float(
            np.float64(recovar[1]) - np.float64(native[1])
        ),
        "absolute_delta": float(
            np.float64(recovar[1]) - np.float64(native[1])
        ),
        "ulp_distance": 3,
        "selection_rule": (
            "top_rotation_by_descending_mismatch_raw_delta_l1_then_"
            "ascending_rotation; within_rotation_largest_absolute_raw_"
            "delta_then_lowest_native_candidate_index"
        ),
    }


def test_raw_mismatch_rotation_rank_tie_uses_ascending_identity() -> None:
    native, recovar = _raw_values_with_ulp_offsets([1, 0, 1, 0])

    report = analyzer._raw_mismatch_strata(
        native_raw=native,
        recovar_raw=recovar,
        native_candidate_index=np.asarray([0, 1, 2, 3]),
        native_rotation_local=np.asarray([10, 10, 20, 20]),
        mapped_recovar_rotation=np.asarray([5, 5, 2, 2]),
        translation_id=np.asarray([0, 1, 0, 1]),
    )

    assert report["rotation_strata"]["top_10"][0][
        "mapped_recovar_rotation"
    ] == 2
    assert report["selected_representative"][
        "mapped_recovar_rotation"
    ] == 2


def test_raw_mismatch_strata_handles_exact_raw_parity() -> None:
    native, recovar = _raw_values_with_ulp_offsets([0, 0, 0])

    report = analyzer._raw_mismatch_strata(
        native_raw=native,
        recovar_raw=recovar,
        native_candidate_index=np.asarray([0, 1, 2]),
        native_rotation_local=np.asarray([0, 1, 2]),
        mapped_recovar_rotation=np.asarray([2, 0, 1]),
        translation_id=np.asarray([0, 0, 0]),
    )

    assert report["mismatch_count"] == 0
    assert report["raw_delta_l1"] == 0.0
    assert report["selected_representative"] is None
    assert report["partition_replay_exact"] is True


def test_recovar_completion_requires_fixed_capture_contract(tmp_path) -> None:
    path = tmp_path / "complete.json"
    path.write_text(
        json.dumps(
            {
                "schema": analyzer.RECOVAR_CAPTURE_SCHEMA,
                "status": "complete",
                "slurm_job_id": 123,
                "integration_head": analyzer.RECOVAR_CAPTURE_HEAD,
                "gpu_uuid": analyzer.TARGET_GPU_UUID,
                "grid_correction": "unset_default_off",
                "final_all_data_after_max_iter": "unset",
                "scorecard_change_admissible": False,
            }
        )
    )

    report = analyzer._validate_recovar_completion(
        path,
        expected_job_id=123,
    )

    assert report["status"] == "complete"
    with pytest.raises(ValueError, match="Slurm identity"):
        analyzer._validate_recovar_completion(path, expected_job_id=124)
