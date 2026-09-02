from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_relion_k4_coarse_operand_repeatability as audit


def _write_repeat(root: Path, job_id: int, *, inert: bool = True) -> None:
    (root / "analysis").mkdir(parents=True)
    (root / "provenance").mkdir()
    operand = {
        "schema": audit.OPERAND_SCHEMA,
        "status": "pass",
        "capture_ready": True,
        "particle_count": 16,
        "class_count": 4,
        "artifact_count": 64,
        "iteration": 1,
        "fixed_gates": {"cross_replay_max_abs": 5e-4},
        "fixed_metric": {"evaluated_artifacts": 64, "passed_artifacts": 64},
    }
    inertness = {
        "schema": audit.INERTNESS_SCHEMA,
        "status": "pass" if inert else "rejected",
        "classification": "scientifically_inert_at_recorded_floor_but_not_bitwise_inert",
        "strict_gate": {
            "capture_validated": True,
            "iteration_zero_numeric_state_exact": True,
            "iteration_one_decision_fields_exact": inert,
            "iteration_one_maps_within_floor": inert,
        },
        "iterations": [
            {
                "iteration": 1,
                "maps": [
                    {"fsc_auc": 0.9999999, "relative_l2": 1e-6}
                    for _ in range(4)
                ],
            }
        ],
    }
    (root / "analysis" / "operand_capture_validation.json").write_text(
        json.dumps(operand)
    )
    (root / "analysis" / "capture_inertness.json").write_text(
        json.dumps(inertness)
    )
    tres = "cpu=8,mem=192G,node=1,billing=15,gres/gpu=1"
    (root / "provenance" / f"scontrol_{job_id}.txt").write_text(
        f"ReqTRES={tres} AllocTRES={tres}\n"
    )
    (root / "provenance" / f"science_outputs_{job_id}.sha256").write_text(
        "abc  artifact\n"
    )


def test_admits_three_complete_inert_repeats(tmp_path: Path) -> None:
    roots = tuple(tmp_path / f"repeat-{index}" for index in range(3))
    for index, root in enumerate(roots):
        _write_repeat(root, 100 + index)
    report = audit.build_report(roots)
    assert report["status"] == "pass"
    assert report["repeat_count"] == 3
    assert report["summary"] == {
        "passed_repeats": 3,
        "class_map_fsc_auc_minimum": 0.9999999,
        "class_map_relative_l2_maximum": 1e-6,
        "iteration_one_decision_exact_repeats": 3,
    }


def test_rejects_fewer_than_three_repeats(tmp_path: Path) -> None:
    roots = tuple(tmp_path / f"repeat-{index}" for index in range(2))
    for index, root in enumerate(roots):
        _write_repeat(root, 200 + index)
    with pytest.raises(audit.AuditError, match="at least three"):
        audit.build_report(roots)


def test_rejects_one_noninert_repeat(tmp_path: Path) -> None:
    roots = tuple(tmp_path / f"repeat-{index}" for index in range(3))
    for index, root in enumerate(roots):
        _write_repeat(root, 300 + index, inert=index != 1)
    with pytest.raises(audit.AuditError, match="capture inertness failed"):
        audit.build_report(roots)
