from __future__ import annotations

import json

import pytest

from scripts import analyze_em_k4_native_target_operands as analyzer


def _classification(**overrides: bool) -> str:
    gates = {
        "rotation_frame_exact": True,
        "translations_exact": True,
        "score_raw_diff2_bitwise_exact": True,
        "production_replay_bitwise_exact": True,
        "target_tie_exact": True,
    }
    gates.update(overrides)
    return analyzer.classify_native_target_operands(**gates)


def test_classifies_exact_native_target_operands() -> None:
    assert _classification() == analyzer.PASS_CLASSIFICATION


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("rotation_frame_exact", "rotation_frame"),
        ("translations_exact", "translations"),
        ("score_raw_diff2_bitwise_exact", "score_raw_diff2"),
        ("production_replay_bitwise_exact", "production_replay"),
        ("target_tie_exact", "target_tie"),
    ],
)
def test_classifies_each_native_target_operand_failure(
    field: str,
    suffix: str,
) -> None:
    assert _classification(**{field: False}).endswith(suffix)


def _completion() -> dict[str, object]:
    return {
        "schema": analyzer.COMPLETION_SCHEMA,
        "status": "complete",
        "slurm_job_id": 123,
        "native_rotation_local": analyzer.EXPECTED_NATIVE_TARGET_ROTATION,
        "target_translations": list(analyzer.TARGET_TRANSLATIONS),
        "grid_correction": "unset_default_off",
        "final_all_data_after_max_iter": "unset",
        "scorecard_change_admissible": False,
    }


def test_target_operand_completion_requires_fixed_contract(tmp_path) -> None:
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(_completion()))

    report = analyzer._validate_target_operand_completion(
        path,
        expected_job_id=123,
    )

    assert report["status"] == "complete"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("slurm_job_id", 124, "Slurm identity"),
        ("native_rotation_local", 2_626, "rotation changed"),
        ("target_translations", [80, 81], "translations changed"),
        ("grid_correction", "on", "grid/finalization"),
        ("scorecard_change_admissible", True, "incorrectly permits"),
    ],
)
def test_target_operand_completion_rejects_contract_changes(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _completion()
    payload[field] = value
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        analyzer._validate_target_operand_completion(
            path,
            expected_job_id=123,
        )
