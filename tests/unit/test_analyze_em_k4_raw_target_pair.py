from __future__ import annotations

import hashlib
import json

import pytest

from scripts import analyze_em_k4_raw_target_pair as analyzer


@pytest.mark.parametrize(
    (
        "operand",
        "raw",
        "score",
        "target",
        "classification",
        "next_boundary",
    ),
    [
        (
            False,
            False,
            False,
            False,
            "native_target_operand_gate_rejected",
            "repair_or_repeat_native_target_operand_capture",
        ),
        (
            True,
            True,
            True,
            True,
            "native_operand_raw_and_combined_score_paths_bitwise_close",
            "continue_downstream_of_combined_score_generation",
        ),
        (
            True,
            True,
            False,
            True,
            "raw_costs_close_but_combined_score_path_differs",
            "localize_prior_minimum_and_saved_score_operation_order",
        ),
        (
            True,
            False,
            True,
            False,
            "raw_costs_differ_but_combined_score_path_bitwise_closes",
            "treat_raw_residual_as_combined_score_inert",
        ),
        (
            True,
            False,
            False,
            True,
            "global_raw_and_score_paths_differ_but_fixed_target_closes",
            "stratify_global_raw_mismatches",
        ),
        (
            True,
            False,
            False,
            False,
            "fixed_target_raw_cost_differs_after_native_operand_replay",
            "capture_recovar_per_pixel_operands",
        ),
    ],
)
def test_classifies_joint_boundary(
    operand: bool,
    raw: bool,
    score: bool,
    target: bool,
    classification: str,
    next_boundary: str,
) -> None:
    observed, observed_next = analyzer.classify_raw_target_pair(
        native_operand_accepted=operand,
        global_raw_accepted=raw,
        global_score_accepted=score,
        target_raw_bitwise_exact=target,
    )

    assert observed == classification
    assert next_boundary in observed_next


def test_rejects_inconsistent_global_and_target_raw_gate() -> None:
    with pytest.raises(ValueError, match="fixed target differs"):
        analyzer.classify_raw_target_pair(
            native_operand_accepted=True,
            global_raw_accepted=True,
            global_score_accepted=False,
            target_raw_bitwise_exact=False,
        )


def _input(path, label: str) -> dict[str, str]:
    input_path = path / label
    input_path.write_text(label)
    return {
        "path": str(input_path.resolve()),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }


def _reports(
    tmp_path,
    *,
    recovar_job_id: int = analyzer.EXPECTED_RECOVAR_JOB_ID,
    operand_job_id: int = analyzer.EXPECTED_OPERAND_JOB_ID,
):
    shared = {
        name: _input(tmp_path, name) for name in analyzer.SHARED_INPUTS
    }
    raw = {
        "schema": analyzer.RAW_SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": analyzer.RAW_PASS_CLASSIFICATION,
        "accepted": True,
        "score_path": {
            "classification": analyzer.PASS_SCORE_CLASSIFICATION,
            "accepted": True,
        },
        "fixed_contract": {
            "native_slurm_job_id": analyzer.EXPECTED_NATIVE_JOB_ID,
            "recovar_slurm_job_id": recovar_job_id,
            "target_gpu_uuid": analyzer.TARGET_GPU_UUID,
            "expected_support": analyzer.EXPECTED_SUPPORT,
            "target_recovar_rotation": analyzer.TARGET_RECOVAR_ROTATION,
            "target_translations": list(analyzer.TARGET_TRANSLATIONS),
        },
        "target": {
            "native_raw_diff2_tied": True,
            "recovar_raw_diff2_tied": True,
            "records": [
                {
                    "translation_id": translation,
                    "native_raw_diff2_bits": 123,
                    "recovar_raw_diff2_bits": 123,
                }
                for translation in analyzer.TARGET_TRANSLATIONS
            ],
        },
        "inputs": shared,
    }
    operand = {
        "schema": analyzer.OPERAND_SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": analyzer.OPERAND_PASS_CLASSIFICATION,
        "accepted": True,
        "fixed_contract": {
            "native_slurm_job_id": analyzer.EXPECTED_NATIVE_JOB_ID,
            "operand_slurm_job_id": operand_job_id,
            "target_gpu_uuid": analyzer.TARGET_GPU_UUID,
            "target_recovar_rotation": analyzer.TARGET_RECOVAR_ROTATION,
            "target_translations": list(analyzer.TARGET_TRANSLATIONS),
        },
        "inputs": dict(shared),
    }
    raw_path = tmp_path / "raw.json"
    operand_path = tmp_path / "operand.json"
    raw_path.write_text(json.dumps(raw))
    operand_path.write_text(json.dumps(operand))
    return raw_path, operand_path


def test_build_report_preserves_fixed_four_boundary_denominator(
    tmp_path,
) -> None:
    raw_path, operand_path = _reports(tmp_path)

    report = analyzer.build_report(
        raw_report_path=raw_path,
        operand_report_path=operand_path,
    )

    assert (
        report["classification"]
        == "native_operand_raw_and_combined_score_paths_bitwise_close"
    )
    assert report["fixed_metric"] == {
        "evaluated_boundaries": 4,
        "expected_boundaries": 4,
        "passed_boundaries": 4,
        "gates": {
            "native_target_operand_replay": True,
            "global_raw_diff2": True,
            "global_combined_score": True,
            "fixed_target_raw_diff2": True,
        },
    }
    assert report["scorecard_change_admissible"] is False


def test_build_report_explicitly_binds_alternate_owner_jobs(tmp_path) -> None:
    raw_path, operand_path = _reports(
        tmp_path,
        recovar_job_id=11793813,
        operand_job_id=11793814,
    )

    report = analyzer.build_report(
        raw_report_path=raw_path,
        operand_report_path=operand_path,
        expected_recovar_job_id=11793813,
        expected_operand_job_id=11793814,
    )

    assert report["fixed_contract"] == {
        "native_slurm_job_id": analyzer.EXPECTED_NATIVE_JOB_ID,
        "recovar_raw_diff2_slurm_job_id": 11793813,
        "native_target_operand_slurm_job_id": 11793814,
        "target_gpu_uuid": analyzer.TARGET_GPU_UUID,
        "expected_support": analyzer.EXPECTED_SUPPORT,
        "target_recovar_rotation": analyzer.TARGET_RECOVAR_ROTATION,
        "target_translations": list(analyzer.TARGET_TRANSLATIONS),
    }


def test_build_report_rejects_unbound_alternate_owner_jobs(tmp_path) -> None:
    raw_path, operand_path = _reports(
        tmp_path,
        recovar_job_id=11793813,
        operand_job_id=11793814,
    )

    with pytest.raises(ValueError, match="raw-score fixed contract"):
        analyzer.build_report(
            raw_report_path=raw_path,
            operand_report_path=operand_path,
        )


def test_rejects_shared_input_hash_mismatch(tmp_path) -> None:
    raw_path, operand_path = _reports(tmp_path)
    operand = json.loads(operand_path.read_text())
    operand["inputs"]["fine_score"]["sha256"] = "0" * 64
    operand_path.write_text(json.dumps(operand))

    with pytest.raises(ValueError, match="hash-linked fine_score"):
        analyzer.build_report(
            raw_report_path=raw_path,
            operand_report_path=operand_path,
        )


def test_rejects_raw_acceptance_that_does_not_replay(tmp_path) -> None:
    raw_path, operand_path = _reports(tmp_path)
    raw = json.loads(raw_path.read_text())
    raw["accepted"] = False
    raw_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="raw-score acceptance"):
        analyzer.build_report(
            raw_report_path=raw_path,
            operand_report_path=operand_path,
        )


def test_rejects_target_tie_summary_that_does_not_replay(tmp_path) -> None:
    raw_path, operand_path = _reports(tmp_path)
    raw = json.loads(raw_path.read_text())
    raw["target"]["native_raw_diff2_tied"] = False
    raw_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="tie summary"):
        analyzer.build_report(
            raw_report_path=raw_path,
            operand_report_path=operand_path,
        )
