from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_vdam_repeat_state_panel as audit_module
from scripts.audit_vdam_repeat_panel import TRAJECTORY_SCHEMA, RepeatPanelError


def _write_panel(tmp_path: Path, *, source_heads: tuple[str, str] = ("a" * 40, "a" * 40)):
    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "suite_id": "suite",
                "acceptance_contract": {"required_checkpoints": [0, 1]},
                "cases": [{"id": "case"}],
            }
        )
    )
    panel = tmp_path / "panel"
    for index, source_head in enumerate(source_heads, start=1):
        root = panel / f"repeat-{index:02d}" / "case"
        root.mkdir(parents=True)
        (root / "trajectory_audit.json").write_text(
            json.dumps(
                {
                    "schema": TRAJECTORY_SCHEMA,
                    "suite_id": "suite",
                    "case_id": "case",
                    "artifact_topology_exact": True,
                    "checkpoints": [{"iteration": 0}, {"iteration": 1}],
                }
            )
        )
        (root / "run_provenance.json").write_text(
            json.dumps(
                {
                    "git_head": source_head,
                    "relion_reference": {"executable_sha256": "b" * 64},
                    "recovar_native_extensions": {
                        "cuda_backproject": {"sha256": "c" * 64}
                    },
                }
            )
        )
        (root / "paired_gpu_uuid.json").write_text(
            json.dumps(
                {
                    "physical_gpu_uuid": "GPU-one",
                    "relion_gpu_uuid": "GPU-one",
                    "recovar_gpu_uuid": "GPU-one",
                }
            )
        )
        meta = root / "recovar"
        meta.mkdir()
        (meta / "run_it001_recovar_meta.json").write_text(
            json.dumps({"selected_particle_ids": [0]})
        )
    return scorecard, panel


def _table():
    return pd.DataFrame({"_rlnImageName": ["1@stack.mrcs"]})


def test_state_panel_requires_every_candidate_particle_and_schedule_mode(tmp_path, monkeypatch):
    scorecard, panel = _write_panel(tmp_path)
    monkeypatch.setattr(audit_module, "_pixel_size", lambda root: 1.5)
    monkeypatch.setattr(audit_module, "read_star", lambda path: (_table(), None))
    monkeypatch.setattr(
        audit_module,
        "audit_sampling_trajectory",
        lambda *args, **kwargs: {"iterations": [{"iteration": 1}]},
    )
    monkeypatch.setattr(
        audit_module,
        "compare_particle_tables_to_native_set",
        lambda *args, **kwargs: {"pass": True, "evaluated_particle_count": 1},
    )
    monkeypatch.setattr(
        audit_module,
        "classify_schedule_mode_envelope",
        lambda rows: {"pass": True, "matching_native_repeat_indices": [1]},
    )
    monkeypatch.setattr(
        audit_module,
        "classify_schedule_distribution_envelope",
        lambda rows: {
            "pass": True,
            "candidate_validity_pass": True,
            "reverse_native_coverage_pass": True,
        },
    )

    report = audit_module.audit_state_panel(
        scorecard_path=scorecard, case_id="case", panel_root=panel, repeat_count=2
    )

    assert report["result"] == "pass"
    assert report["scoring"] is False
    assert [row["result"] for row in report["candidate_repeats"]] == ["pass", "pass"]
    assert report["schedule_distribution_result"] == "pass"
    assert report["provenance"]["cuda_library_sha256"] == "c" * 64


def test_state_panel_reports_candidate_failure(tmp_path, monkeypatch):
    scorecard, panel = _write_panel(tmp_path)
    monkeypatch.setattr(audit_module, "_pixel_size", lambda root: 1.5)
    monkeypatch.setattr(audit_module, "read_star", lambda path: (_table(), None))
    monkeypatch.setattr(
        audit_module,
        "audit_sampling_trajectory",
        lambda *args, **kwargs: {"iterations": [{"iteration": 1}]},
    )
    monkeypatch.setattr(
        audit_module,
        "compare_particle_tables_to_native_set",
        lambda *args, **kwargs: {"pass": False, "evaluated_particle_count": 1},
    )
    monkeypatch.setattr(
        audit_module,
        "classify_schedule_mode_envelope",
        lambda rows: {"pass": True, "matching_native_repeat_indices": [1]},
    )
    monkeypatch.setattr(
        audit_module,
        "classify_schedule_distribution_envelope",
        lambda rows: {
            "pass": False,
            "candidate_validity_pass": False,
            "reverse_native_coverage_pass": False,
        },
    )

    report = audit_module.audit_state_panel(
        scorecard_path=scorecard, case_id="case", panel_root=panel, repeat_count=2
    )

    assert report["result"] == "fail"
    assert all(row["first_particle_failure_iteration"] == 1 for row in report["candidate_repeats"])
    assert report["schedule_distribution_result"] == "fail"
    assert report["first_schedule_distribution_failure_iteration"] == 1


def test_state_panel_rejects_mixed_source_heads(tmp_path):
    scorecard, panel = _write_panel(tmp_path, source_heads=("a" * 40, "d" * 40))
    scorecard_payload = json.loads(scorecard.read_text())

    with pytest.raises(RepeatPanelError, match="mixed or invalid source heads"):
        audit_module._validated_roots(
            scorecard=scorecard_payload,
            case_id="case",
            panel_root=panel,
            repeat_count=2,
        )
