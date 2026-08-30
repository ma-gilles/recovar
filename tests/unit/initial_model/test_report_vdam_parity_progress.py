from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import report_vdam_parity_progress as progress_mod


@pytest.fixture
def scorecard() -> dict:
    return json.loads(progress_mod.DEFAULT_SCORECARD.read_text())


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "vdam-progress.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def test_checked_v3_scorecard_replays_frozen_panels() -> None:
    loaded = progress_mod.load_and_validate()
    progress = progress_mod.build_progress(loaded)
    assert progress["accepted_cases"] == ["vdam-gf44", "vdam-gf45"]
    assert progress["failure_counts"] == {"map": 15, "particle": 14, "schedule": 7}
    assert {row["id"]: row["passed"] for row in loaded["panels"]} == {
        "k1_strict_correctness": 2,
        "map_trajectory": 5,
        "particle_trajectory": 6,
        "predivergence_schedule": 13,
        "runtime_comparable": 0,
        "terminal_audits": 20,
    }


def test_suite_definition_identity_and_bytes_are_frozen() -> None:
    assert progress_mod.sha256_file(progress_mod.REPO_ROOT / progress_mod.SUITE_DEFINITION_PATH) == (
        progress_mod.SUITE_DEFINITION_SHA256
    )
    loaded = progress_mod.load_and_validate()
    assert loaded["suite_id"] == "vdam-k1-full-trajectory-expansion-v3"
    assert loaded["frozen_denominator"] == 20


def test_scientific_evidence_identity_is_immutable(tmp_path: Path, scorecard: dict) -> None:
    scorecard["evidence_sources"]["gf43_seeded"]["map_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence identity changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_false_strict_pass_is_rejected(tmp_path: Path, scorecard: dict) -> None:
    failed = next(case for case in scorecard["cases"] if case["id"] == "vdam-gf43")
    failed["strict_result"] = "pass"
    with pytest.raises(ValueError, match="strict result does not replay its gates"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_passing_gate_cannot_retain_failure_iteration(tmp_path: Path, scorecard: dict) -> None:
    passed = next(case for case in scorecard["cases"] if case["id"] == "vdam-gf44")
    passed["map"]["first_failure_iteration"] = 200
    with pytest.raises(ValueError, match="passing map gate has a failure iteration"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_failing_gate_needs_bounded_failure_iteration(tmp_path: Path, scorecard: dict) -> None:
    failed = next(case for case in scorecard["cases"] if case["id"] == "vdam-gf43")
    failed["map"]["first_failure_iteration"] = 201
    with pytest.raises(ValueError, match=r"needs an iteration in \[0, 200\]"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_panel_cannot_inflate_or_change_denominator(tmp_path: Path, scorecard: dict) -> None:
    inflated = copy.deepcopy(scorecard)
    inflated["panels"][0]["passed"] = 3
    with pytest.raises(ValueError, match="count does not replay cases"):
        progress_mod.load_and_validate(_write(tmp_path, inflated))
    scorecard["panels"][0]["denominator"] = 21
    with pytest.raises(ValueError, match="denominator changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_runtime_cannot_change_strict_correctness(tmp_path: Path, scorecard: dict) -> None:
    accepted = next(case for case in scorecard["cases"] if case["id"] == "vdam-gf44")
    accepted["runtime"] = {"ratio_vs_relion": 1.0, "result": "pass"}
    scorecard["panels"][4]["passed"] = 1
    loaded = progress_mod.load_and_validate(_write(tmp_path, scorecard))
    assert loaded["cases"][1]["strict_result"] == "pass"
    assert loaded["panels"][0]["passed"] == 2
    assert loaded["panels"][4]["passed"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("role", "release_gate"),
        ("status", "pass"),
        ("scientific_outcome", "fail"),
        ("scheduler_state", "completed"),
        ("score_impact", "strict"),
    ],
)
def test_diagnostic_roles_and_outcomes_are_independent(
    tmp_path: Path, scorecard: dict, field: str, replacement: str
) -> None:
    job = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13206294")
    job[field] = replacement
    with pytest.raises(ValueError):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_terminal_cache_attempts_render_as_invalid_without_parity_inference(scorecard: dict) -> None:
    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "`13208186` | `diagnostic` | **INVALID** | INVALID | cancelled" in rendered
    assert "`13208265` | `diagnostic` | **INVALID** | INVALID | cancelled" in rendered
    assert "`13208734` | `diagnostic` | **INVALID** | INVALID | cancelled" in rendered
    assert "`13208735` | `diagnostic` | **INVALID** | INVALID | cancelled" in rendered
    assert "`13209422` | `diagnostic` | **INVALID** | INVALID | cancelled" in rendered
    assert "fresh_a science PASS through iteration 4" in rendered
    assert "`run_local_bucket_big_jit`" in rendered
    assert "`relion_vdam_mstep_fused_projector_x_half`" in rendered
    assert "INVALID attempts and expected hypothesis rejections" in rendered


def test_profile_matched_cache_evidence_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    job = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13209422")
    job["warm80_a"]["cache_entries_added"] = 0
    with pytest.raises(ValueError, match="profile-matched evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_pair_stable_cache_evidence_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    job = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13210232")
    job["pairs"][1]["warm"]["result"] = "pass"
    with pytest.raises(ValueError, match="pair-stable evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_pair_stable_hypothesis_rejection_is_non_scoring() -> None:
    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "`13210232` | `diagnostic` | **EXPECTED FAIL-CLOSED** | HYPOTHESIS REJECTED" in rendered
    assert "`pair_a` | `cold_a` PASS | `warm_a` PASS | 0->4823; 4823->4823 | byte-stable" in rendered
    assert "`pair_b` | `cold_b` FAIL@4 | `warm_b` FAIL@4 | 0->4676; 4676->4676 | byte-stable" in rendered
    assert "exact historical graph-pair red pose" in rendered
    assert "not a new scorecard failure" in rendered
    assert "analysis/cache_history_summary.json" in rendered
    assert "long-run cache reuse/deserialization is not necessary" in rendered
    assert "compile or autotune variant versus runtime reduction" in rendered
    assert "`13211317` is **INVALID/SUPERSEDED**" in rendered


def test_ordered_shell_same_cache_evidence_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    job = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13211719")
    job["cache_validation"]["manifest_sha256"]["ordered_b_before"] = "0" * 64
    with pytest.raises(ValueError, match="13211719: ordered-shell evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_ordered_shell_hypothesis_rejection_is_valid_and_non_scoring() -> None:
    loaded = progress_mod.load_and_validate()
    job = next(row for row in loaded["active_diagnostics"] if row["job_id"] == "13211719")
    rendered = progress_mod.render_markdown(loaded)

    assert job["execution_valid"] is True
    assert job["arm_success_markers"] is True
    assert job["score_impact"] == "none"
    assert "`13211317` | `diagnostic` | **INVALID/SUPERSEDED** | INVALID | failed | none" in rendered
    assert (
        "`13211719` | `diagnostic` | **EXPECTED HYPOTHESIS REJECTION** | HYPOTHESIS REJECTED | failed | none"
        in rendered
    )
    assert "### Valid same-cache ordered-shell result: job `13211719`" in rendered
    assert "| A before | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |" in rendered
    assert "| A after | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |" in rendered
    assert "| B before | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |" in rendered
    assert "| B after | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |" in rendered
    assert "B added 0 files and changed 0 files" in rendered
    assert "particle ID `2896` / selected index `178`" in rendered
    assert "1.1641532182693481e-10 (1 float32 ULP)" in rendered
    assert "pose / rotation / translation / class exact" in rendered
    assert "| ordered image power | 27/65 shells | 512.0 |" in rendered
    assert "| ordered sigma numerator | 1/65 shells | 0.00390625 |" in rendered
    assert "| final noise | 24/65 shells | 0.001562500001455192 |" in rendered
    assert "| live BPref | both halves; 12 fields | h0 0.0234375; h1 0.0107421875" in rendered
    assert "target stack `286` exact" in rendered
    assert "expected scientific-gate hypothesis rejection" in rendered
    assert "not a frozen-score failure" in rendered
    assert job["repeatability_report_sha256"] in rendered
    assert job["cache_report_sha256"] in rendered
    assert job["evidence_sha256"] in rendered


def test_invalid_speed_harness_evidence_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    job = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13212500")
    job["promotion_authorized"] = True
    with pytest.raises(ValueError, match="13212500: invalid speed-harness evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_invalid_speed_harness_is_visible_and_non_scoring() -> None:
    loaded = progress_mod.load_and_validate()
    job = next(row for row in loaded["active_diagnostics"] if row["job_id"] == "13212500")
    rendered = progress_mod.render_markdown(loaded)

    assert job["science_result_recorded"] is False
    assert job["runtime_result_recorded"] is False
    assert job["promotion_authorized"] is False
    assert "`13212500` | `diagnostic` | **INVALID HARNESS** | INVALID | cancelled | none" in rendered
    assert "### Invalid speed-gate attempt: job `13212500`" in rendered
    assert "before any A/B science or timing result" in rendered
    assert "launched `make`/`nvcc` against the source artifact" in rendered
    assert job["source_cuda_library"]["sha256"] in rendered
    assert job["runner_log_sha256"] in rendered
    assert "authorizes no 80-iteration promotion" in rendered


def test_legacy_v2_track_is_non_scoring(scorecard: dict) -> None:
    loaded = progress_mod.load_and_validate()
    v2 = next(row for row in loaded["secondary_tracks"] if row["id"] == "legacy_parameter_expansion_v2")
    assert (v2["passed"], v2["denominator"], v2["score_impact"]) == (6, 15, "none")
    assert next(row for row in loaded["panels"] if row["id"] == "k1_strict_correctness")["passed"] == 2


def test_dashboard_is_compact_and_exposes_shared_em_reuse() -> None:
    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert len(rendered.splitlines()) < 200
    assert "Authoritative v3 status — NOT READY" in rendered
    assert "Strict K=1 correctness is **2 / 20**" in rendered
    assert "runtime parity is **0 / 20**" in rendered
    assert "coarse projector/scorer | yes" in rendered
    assert "does not carry duplicate projector" in rendered


def test_cli_check_accepts_generated_dashboard() -> None:
    result = subprocess.run(
        [sys.executable, str(progress_mod.__file__), "--check"],
        cwd=progress_mod.REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
