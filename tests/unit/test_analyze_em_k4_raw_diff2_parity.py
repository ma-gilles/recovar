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
