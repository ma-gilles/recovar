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


def test_remaining_profile_targets_shared_scheduling_not_new_math(scorecard: dict) -> None:
    profile = scorecard["performance_gate_updates"]["remaining_profile_decomposition"]
    assert profile["status"] == "SEALED_READ_ONLY_SCHEDULING_TARGET"
    assert profile["kernel_work"]["recovar_overhead_percent"] == pytest.approx(3.051596)
    assert profile["measured_headroom"]["excess_idle_seconds"] == pytest.approx(3.343912411)
    assert profile["measured_headroom"]["excess_idle_percent_warm_wall"] > 20.0
    assert profile["coarse_topology"]["recovar_per_image_kernel_count"] == 1000
    assert "shared EM planner/executor" in profile["next_candidate"]
    assert "material end-to-end runtime gain" in profile["acceptance_rule"]

    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "SEALED PROFILE / LARGE LEVER IDENTIFIED" in rendered
    assert "3.344 s" in rendered
    assert profile["report_sha256"] in rendered


def test_compile_profile_targets_shared_fixed_whole_local_executor(scorecard: dict) -> None:
    profile = scorecard["performance_gate_updates"]["local_compile_shape_decomposition"]
    assert profile["status"] == "SEALED_READ_ONLY_SHARED_FIXED_EXECUTOR_TARGET"
    assert profile["local_executor_compile_seconds"] == pytest.approx(53.547)
    assert profile["local_share_of_compile_percent"] == pytest.approx(92.6)
    assert profile["compile_iterations"] == [62, 63, 67, 72, 73]
    assert profile["forecast"]["compile_only_saving_percent_full_run"] > 10.0
    assert profile["forecast"]["packed_static_row_reduction_percent_range"] == [38, 79]
    assert "fixed-capacity packed whole-local executor" in profile["next_candidate"]
    assert "repeat-bounded" in profile["acceptance_rule"]
    assert profile["slurm_job"] == "13248509"
    assert profile["profiled_source_commit"] == "6b5e6568a4d2d05f9ac70f4a6bee9cfb450e94a8"
    assert profile["sealed_analysis_report_sha256"] == (
        "fd33935f77c275615249e6149abfdf9abf50d4b1c8463aa4dcbe11f2ea176def"
    )
    assert profile["nsight_sqlite_sha256"] == (
        "b73f3ecb660d59484b280e2b0021763c5583c01106ed839efef71f5d50a55845"
    )
    implementation = profile["implementation_progress"]
    assert implementation["commit"] == "793e3bb12a2e9e367ed7d1a302db95575b1ebb63"
    assert implementation["phase_1d_commit"] == "41c1cdbd104cd85cf8f433080889ac6be3a5ef21"
    assert implementation["focused_tests"] == "146 passed"
    assert implementation["independent_cpu_review"] == "GO"
    assert implementation["shared_em_vdam_path"] is True
    assert implementation["single_shared_numeric_wrapper"] is True
    assert implementation["authoritative_dataset_operand_check"] is True
    assert implementation["production_invoked"] is False
    assert implementation["gpu_evaluated"] is True
    assert implementation["independent_h100_review"] == "GO"
    score_gate = implementation["h100_score_correctness_gate"]
    assert score_gate["classification"] == "correctness_only"
    assert score_gate["result"] == "PASS_INDEPENDENT_REVIEW_GO"
    assert score_gate["job_id"] == "13288282"
    assert score_gate["source_head"] == "d880da3d0f169f7e898d5b0fcf8c3a5cf3122000"
    assert score_gate["audit_docs_commit"] == "9669350003b9aed1509513824741d79308d8885d"
    assert score_gate["hardware"] == {
        "node": "della-h21g4",
        "gpu": "NVIDIA H100 80GB HBM3",
        "gpu_uuid": "GPU-099c0d77-bb85-f2e9-f628-148b733c9176",
    }
    assert score_gate["focused_gpu_tests"] == "7/7"
    assert score_gate["precision_lanes"] == ["float32", "float64"]
    assert (score_gate["captured_comparisons"], score_gate["production_comparisons"]) == (8, 12)
    assert set(score_gate["metrics_max_abs"]) == {
        "score",
        "centered_score",
        "log_z",
        "best_score",
        "max_posterior",
        "posterior",
        "posterior_mass",
    }
    assert all(value == 0.0 for value in score_gate["metrics_max_abs"].values())
    assert score_gate["discrete_and_support_exact"] is True
    assert score_gate["significant_counts"] == [2, 4, 4]
    assert score_gate["reconstruction_row_count"] == 6
    assert score_gate["source_manifest_sha256"] == (
        "986f6c733672425e87c8de6b8c7dec18e5d4085c663145d5e2510af6d0a72e6c"
    )
    assert score_gate["cuda_sha256"] == (
        "948a728b98e2d38c882a6832abba991cbbcb4ae87474b849f109166dd7158db6"
    )
    assert score_gate["diagnostics_sha256"] == (
        "f9b73b6facd9f74b41c4e7c76a46f6b47fb6d45734cc032f5a07f8fcadf64d25"
    )
    assert score_gate["gate_json_sha256"] == (
        "065e2901accefe57e59e61e6048a3ae5e66d929e4ce470527c640bec19849f7f"
    )
    assert score_gate["junit_sha256"] == (
        "96dc875b31f7c0a5bbe61f3344cd9f07745a92bd0b1b8ee8ddb5408185d2082f"
    )
    assert score_gate["speed_claim_allowed"] is False
    assert score_gate["default_promotion_allowed"] is False

    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "H100 SCORE CORRECTNESS PASS / RUNTIME UNQUALIFIED" in rendered
    assert "53.547 of 57.833 s" in rendered
    assert profile["runtime_profile_report_sha256"] in rendered
    assert profile["sealed_analysis_report_sha256"] in rendered
    assert "Phase 1e `793e3bb12a`" in rendered
    assert "7/7 focused tests and all 8+12 float32/float64 comparisons" in rendered
    assert "every score, centered-score, logZ, best-score, Pmax, posterior, and mass delta is exactly zero" in rendered
    assert score_gate["source_manifest_sha256"] in rendered
    assert score_gate["cuda_sha256"] in rendered
    assert score_gate["diagnostics_sha256"] in rendered
    assert score_gate["gate_json_sha256"] in rendered
    assert score_gate["junit_sha256"] in rendered
    assert "no speed or default promotion" in rendered


def test_numerical_acceptance_requires_stability_and_large_runtime_advantage(scorecard: dict) -> None:
    policy = scorecard["performance_gate_updates"]["numerical_policy"]
    assert policy == progress_mod.NUMERICAL_ACCEPTANCE_POLICY
    assert "bitwise identity is not a universal requirement" in policy
    assert "stable numerical noise" in policy
    assert "repeat-bounded, unbiased, non-growing" in policy
    assert "preserve discrete choices and the optimization basin" in policy
    assert "no material final-quality loss" in policy
    assert "material, large, reproducible end-to-end runtime advantage" in policy
    assert "unstable, biased, or growing numerics fail" in policy

    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "bitwise identity is not a universal requirement" in rendered
    assert "material, large, reproducible end-to-end runtime advantage" in rendered
    assert "Unstable numerics fail" in rendered


def test_fixed_local_h100_score_gate_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    gate = scorecard["performance_gate_updates"]["local_compile_shape_decomposition"][
        "implementation_progress"
    ]["h100_score_correctness_gate"]
    gate["gate_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current performance gate updates changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_clean_batched_lane_gate_preserves_strict_failure_and_runtime_gain(scorecard: dict) -> None:
    gate = scorecard["performance_gate_updates"]["coarse_atomic_batched_lanes"]
    assert gate["implementation_commit"] == "6f10c2d3f075654a94caa6e44249a5b1275b48f6"
    assert gate["focused_gpu_job"] == "13285438"
    assert gate["crossed_job"] == "13285647"
    assert gate["crossed_result"] == "COMPLETED 0:0"
    assert gate["performance"]["change_percent"]["warm_wall"] == pytest.approx(-9.031075)
    assert gate["performance"]["change_percent"]["coarse_kernel_sum"] == pytest.approx(-10.137424)
    assert gate["performance"]["atomic_batched_lanes_median"]["coarse_kernel_launches_per_repeat"] == 48
    assert gate["numerical_result"]["all_discrete_metadata_exact"] is True
    assert gate["numerical_result"]["cold_cross_map_relative_l2_max"] > gate["numerical_result"][
        "cold_repeat_envelope"
    ]
    assert gate["numerical_result"]["warm_cross_map_relative_l2_max"] > gate["numerical_result"][
        "warm_repeat_envelope"
    ]
    assert gate["numerical_result"]["strict_two_repeat_result"] == "fail"
    assert gate["numerical_result"]["nondirectional_noise_gate_result"] == "pass"
    repeat_panel = gate["replicated_roundoff_equivalence_gate"]
    assert repeat_panel["status"] == "RECOVERED_PASS_INDEPENDENT_REVIEW_GO"
    assert repeat_panel["qualification_overlay_commit"] == (
        "fb9aaf2717c17e67c94a0ea5f54cb6e2cfc9f8cf"
    )
    assert repeat_panel["acceptance_config_sha256"] == (
        "4f266733a4a7ffbc79c06d340918a4389c79d7c5cf34401031d07a64c38f258d"
    )
    assert repeat_panel["total_fresh_process_runs"] == 16
    assert repeat_panel["blocked_exact_label_permutations"] == 1296
    assert repeat_panel["job_id"] == "13286397"
    assert repeat_panel["all_runs_completed"] is True
    assert repeat_panel["original_completed_marker_present"] is False
    assert repeat_panel["recovery_completed"] is True
    assert repeat_panel["result"] == (
        "all_predeclared_repeat_panel_gates_pass_after_independent_serializer_recovery_review"
    )
    assert repeat_panel["independent_recovery_review"] == "GO"
    assert repeat_panel["report_json_sha256"] == (
        "66d52235cf429f952b5db5bf087fdd72f51cc82210abbd8dcd494eaedf12aaf1"
    )
    assert repeat_panel["science"]["all_star_rows_and_discrete_metadata_exact"] is True
    assert repeat_panel["science"]["warm"]["maximum_pairwise_relative_l2"] < 1e-8
    assert repeat_panel["science"]["warm"]["configuration_effect_joint_permutation_p"] >= 0.05
    assert repeat_panel["science"]["cache_state_amplification_pass"] is True
    assert repeat_panel["performance"]["replicated_warm_wall_change_percent"] == pytest.approx(
        -9.316965
    )
    assert repeat_panel["performance"]["material_runtime_gate_pass"] is True
    assert repeat_panel["long_trajectory_authorized"] is False
    incremental = gate["incremental_atomic_serial_gate"]
    assert gate["incremental_atomic_baseline_evaluated"] is True
    assert gate["tracking_derived_port_evaluated"] is True
    assert incremental["job_id"] == "13284897"
    assert incremental["result"] == "COMPLETED 0:0"
    assert incremental["performance"]["warm_wall_change_percent"] == pytest.approx(-8.890551)
    assert incremental["performance"]["coarse_kernel_union_change_percent"] == pytest.approx(-10.86529)
    assert incremental["numerical"]["all_discrete_metadata_exact"] is True
    assert incremental["numerical"]["cold_cross_map_relative_l2_max"] <= incremental["numerical"][
        "cold_repeat_envelope"
    ]
    assert incremental["numerical"]["warm_cross_map_relative_l2_max"] <= incremental["numerical"][
        "warm_repeat_envelope"
    ]
    assert "immutable report remains a strict FAIL" in gate["integration_caveat"]

    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert "MATERIAL RUNTIME PASS; FROZEN TWO-REPEAT NUMERIC GATE FAIL" in rendered
    assert "ALL PREDECLARED NUMERICAL AND RUNTIME GATES PASS AFTER INDEPENDENTLY REVIEWED" in rendered
    assert gate["report_json_sha256"] in rendered
    assert repeat_panel["acceptance_config_sha256"] in rendered
    assert repeat_panel["analyzer_source_sha256"] in rendered
    assert "The frozen n=2 rejection is not overwritten" in rendered


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


def test_current_engineering_snapshot_is_visible_and_non_scoring() -> None:
    loaded = progress_mod.load_and_validate()
    snapshot = loaded["engineering_snapshot"]
    rendered = progress_mod.render_markdown(loaded)

    assert snapshot["frozen_scores_changed"] is False
    assert snapshot["score_impact"] == "none"
    gate = snapshot["active_gate"]
    assert gate["status"] == "completed_mixed_gate"
    assert gate["typed_policy"]["result"] == "pass"
    assert gate["typed_policy"]["iterations_per_arm"] == 80
    assert gate["same_gpu_map_envelope"]["result"] == "pass"
    assert gate["same_gpu_map_envelope"]["checkpoints_passed"] == 81
    assert gate["active_particle_envelope"]["result"] == "fail"
    assert gate["active_particle_envelope"]["first_failure_iteration"] == 37
    assert gate["runtime"]["result"] == "inconclusive"
    assert gate["runtime"]["claim_authorized"] is False
    assert gate["original_profile_repeat"]["status"] == "invalid_harness"
    assert gate["original_profile_repeat"]["stopped_after_iteration"] == 64
    assert gate["corrected_profile_repeat"]["comparison_role"] == "cross_gpu_diagnostic_only"
    assert snapshot["short_gate"]["first_divergent_iteration"] is None
    assert snapshot["short_gate"]["exact_particle_state_iterations"] == 20
    assert snapshot["typed_runtime_controls"]["defaults"] == {
        "relion_wavg_sequential_cuda": True,
        "exact_local_bucket_radix": 4,
    }
    assert "`13252518`: 20-iteration `vdam-gf46` | **PARTICLE TRAJECTORY EXACT**" in rendered
    assert "**PASS 80/80 in both arms**" in rendered
    assert "**PASS 81/81**" in rendered
    assert "**FAIL@37 — OPEN**" in rendered
    assert "**INCONCLUSIVE**" in rendered
    assert "Original profiled repeat stopped at iteration 64 (invalid harness)" in rendered
    assert "corrected job `13254470` completed, with direct map/particle FAIL@4" in rendered
    assert "cross-GPU diagnostic only" in rendered
    assert gate["reports"]["typed_policy"]["sha256"] in rendered
    assert gate["reports"]["same_gpu_map"]["sha256"] in rendered
    assert gate["reports"]["same_gpu_particle"]["sha256"] in rendered
    assert gate["reports"]["runtime"]["sha256"] in rendered
    assert "cross-run H100 observations, not a paired speed result" in rendered
    assert "Warm H100 profile: job `13248509`" in rendered
    assert "inline indexed fine projection | **REJECTED**" in rendered
    assert "float32 coarse scorer | **REJECTED**" in rendered
    assert next(row for row in loaded["panels"] if row["id"] == "k1_strict_correctness")["passed"] == 2
    assert next(row for row in loaded["panels"] if row["id"] == "runtime_comparable")["passed"] == 0


def test_current_engineering_snapshot_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    scorecard["engineering_snapshot"]["active_gate"]["reports"]["runtime"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current engineering snapshot changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_new_performance_diagnostics_are_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    palette = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13254010")
    palette["warm_wall_change_percent"] = -4.1
    with pytest.raises(ValueError, match="13254010: coarse-tail palette evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))

    scorecard = json.loads(progress_mod.DEFAULT_SCORECARD.read_text())
    tail = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13257087")
    tail["promotion_authorized"] = True
    with pytest.raises(ValueError, match="13257087: dynamic-tail paired evidence changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_tail_mask_microgate_is_superseded_by_full_pair() -> None:
    loaded = progress_mod.load_and_validate()
    rendered = progress_mod.render_markdown(loaded)

    micro = next(row for row in loaded["active_diagnostics"] if row["job_id"] == "13256612")
    full = next(row for row in loaded["active_diagnostics"] if row["job_id"] == "13257087")
    active200 = next(row for row in loaded["active_diagnostics"] if row["job_id"] == "13257182")
    assert micro["superseded_by"] == "13257087"
    assert full["promotion_authorized"] is False
    assert full["execution_scope"]["path_role"] == "nondefault_path_diagnostic_only"
    assert active200["active_rows_bitwise_equal"] is False
    assert "`13256612` | `diagnostic` | **MICROGATE ONLY**" in rendered
    assert "`13257087` | `diagnostic` | **VALID SCIENCE FAIL / DO NOT PROMOTE**" in rendered
    assert "`13257182` | `diagnostic` | **VALID EXACTNESS FAIL**" in rendered
    assert "Forced nondefault native-texture path" in rendered
    assert "120/120 strict artifacts differ" in rendered
    assert "**VALID SCIENCE FAIL / DO NOT PROMOTE**" in rendered


def test_legacy_v2_track_is_non_scoring(scorecard: dict) -> None:
    loaded = progress_mod.load_and_validate()
    v2 = next(row for row in loaded["secondary_tracks"] if row["id"] == "legacy_parameter_expansion_v2")
    assert (v2["passed"], v2["denominator"], v2["score_impact"]) == (6, 15, "none")
    assert next(row for row in loaded["panels"] if row["id"] == "k1_strict_correctness")["passed"] == 2


def test_runtime_workboard_is_evidence_bound_and_non_scoring() -> None:
    loaded = progress_mod.load_and_validate()
    workboard = loaded["runtime_lane_workboard"]
    lanes = {row["id"]: row for row in workboard["lanes"]}

    assert workboard["frozen_scores_changed"] is False
    assert workboard["production_default_changed"] is False
    assert workboard["score_impact"] == "none"
    numerical_policy = workboard["numerical_equivalence_policy"]
    assert numerical_policy["zero_two_control_map_diameter_required"] is False
    assert numerical_policy["roundoff_scale_difference_allowed"] is True
    assert numerical_policy["isolated_primitive_speedup_is_insufficient"] is True
    assert numerical_policy["performance_requirement"] == "large reproducible end-to-end runtime gain"
    assert "no_discrete_state_or_schedule_escape" in numerical_policy["required_properties"]
    assert "no_final_quality_regression" in numerical_policy["required_properties"]
    assert list(lanes) == progress_mod.RUNTIME_LANE_IDS
    assert lanes["call_neutral_flat_row"]["status"] == "qualified_microbenchmark_default_off"
    assert lanes["call_neutral_flat_row"]["combined_call_reduction_percent"] == pytest.approx(
        [32.87868950661776, 35.76924670326675, 54.525674298292294, 32.682870365453375]
    )
    assert lanes["call_neutral_flat_row"]["outer_calls_control"] == [7, 3, 4, 5]
    assert lanes["call_neutral_flat_row"]["outer_calls_candidate"] == [7, 3, 4, 5]
    assert lanes["stable_fine_window"]["forecast_only"] is True
    assert lanes["stable_fine_window"]["trajectory_run"] is False
    assert lanes["batched_cub_sort_scan"]["sort_bitwise"] is True
    assert lanes["batched_cub_sort_scan"]["scalar_control_repeat_scan_mismatch_entries"] == 442353
    assert lanes["batched_cub_sort_scan"]["posterior_boundary"]["support_mask_bitwise"] is True
    assert lanes["batched_cub_sort_scan"]["posterior_boundary"]["minimum_speedup"] == pytest.approx(
        1.2565006073036646
    )
    cub_trajectory = lanes["batched_cub_sort_scan"]["trajectory_ab"]
    assert lanes["batched_cub_sort_scan"]["status"] == "rejected_trajectory_science_fail_default_off"
    assert cub_trajectory["job_id"] == "13268653"
    assert cub_trajectory["result"] == "fail"
    assert cub_trajectory["particle_state_escapes"] == [
        {"candidate_repeat": 1, "iteration": 4, "particle_id": 285},
        {"candidate_repeat": 1, "iteration": 16, "particle_id": 2902},
        {"candidate_repeat": 1, "iteration": 18, "particle_id": 902},
    ]
    assert cub_trajectory["schedule_escapes"] == [
        {
            "candidate_repeat": 1,
            "iteration": 18,
            "field": "current_changes_optimal_offsets_angstrom",
        }
    ]
    assert cub_trajectory["map_outside_candidate_checkpoint_count"] == 21
    assert cub_trajectory["worst_map_escape"]["nearest_over_control_diameter"] == pytest.approx(
        87.94603038052571
    )
    assert cub_trajectory["report_sha256"] == (
        "a67f6c969e84da096c70d88219ddb4e6962ecd13266814743d992099be7b172d"
    )
    assert cub_trajectory["runtime"]["scoring"] is False
    assert lanes["batched_cub_sort_scan"]["promotion_authorized"] is False

    elementwise = lanes["batched_posterior_elementwise"]
    assert elementwise["status"] == "numerically_equivalent_end_to_end_gain_inconclusive_default_off"
    assert elementwise["primitive_gate"]["job_id"] == "13269547"
    assert elementwise["primitive_gate"]["combined_bitwise"] is True
    assert elementwise["primitive_gate"]["minimum_speedup"] == pytest.approx(7.18388202434731)
    assert elementwise["trajectory_ab"]["job_id"] == "13269681"
    assert elementwise["trajectory_ab"]["result_role"] == "raw_strict_two_control_map_diameter_diagnostic"
    assert elementwise["trajectory_ab"]["acceptance_result"] == "numerically_equivalent"
    assert elementwise["trajectory_ab"]["particle_state_failed_iteration_count"] == 0
    assert elementwise["trajectory_ab"]["schedule_failed_iteration_count"] == 0
    assert elementwise["trajectory_ab"]["map_escape"]["iteration"] == 20
    assert elementwise["trajectory_ab"]["map_escape"]["nearest_over_control_diameter"] == pytest.approx(
        1.0739597400924568
    )
    assert elementwise["trajectory_ab"]["report_sha256"] == (
        "42544acfe0ae193022808abdbdf56639f418f6102d4e66b9a44ddc1a0aa1ff56"
    )
    assert elementwise["trajectory_ab"]["runtime"]["scoring"] is False
    assert elementwise["numerical_equivalence"]["result"] == "pass"
    assert elementwise["numerical_equivalence"]["fixed_operand_elementwise_bitwise"] is True
    assert elementwise["numerical_equivalence"]["basin_change_observed"] is False
    assert elementwise["numerical_equivalence_gate_passed"] is True
    assert elementwise["end_to_end_runtime_status"] == "inconclusive"
    assert elementwise["promotion_authorized"] is False

    same_binary = lanes["elementwise_same_binary_causal"]
    assert same_binary["status"] == "numerical_equivalence_evidence_end_to_end_gain_immaterial_default_off"
    assert same_binary["job_id"] == "13271166"
    assert same_binary["control_repo_head"] == "4ee4383ed57ccddc4e56575a965f318c3fb737d4"
    assert same_binary["shared_cuda_library_sha256"] == (
        "6210cdb1cc97aa72fbdf80b36b501ad48c8d1d1e4866f4a2c11889076e1bff53"
    )
    assert same_binary["science_result"]["particle_state_failed_iteration_count"] == 0
    assert same_binary["science_result"]["schedule_failed_iteration_count"] == 0
    assert same_binary["science_result"]["acceptance_result"] == "numerically_equivalent"
    assert [row["iteration"] for row in same_binary["science_result"]["map_escapes"]] == [2, 3]
    assert [row["nearest_over_control_diameter"] for row in same_binary["science_result"]["map_escapes"]] == pytest.approx(
        [1.219577024094196, 1.1443311717746543]
    )
    assert same_binary["report_sha256"] == (
        "ccb9d9cc4f4ee949aabbfa2c6045aea5b6c2007bcdbcd871e0e1df246d0c3db0"
    )
    assert [row["job_id"] for row in same_binary["invalid_preflight_attempts"]] == ["13270868", "13270984"]
    assert all(row["science_started"] is False for row in same_binary["invalid_preflight_attempts"])
    assert same_binary["runtime_result"]["scoring"] is False
    assert same_binary["runtime_result"]["qualification"] == "immaterial"
    assert same_binary["numerical_equivalence"]["result"] == "pass"
    assert same_binary["numerical_equivalence"]["different_cuda_binary_ruled_out"] is True
    assert same_binary["numerical_equivalence_gate_passed"] is True
    assert same_binary["promotion_authorized"] is False

    assert lanes["xhalf_projection_cap"]["preboundary_invariant_passed"] is False
    assert lanes["xhalf_projection_cap"]["causal_projection_effect_established"] is False
    assert lanes["xhalf_projection_cap"]["promotion_authorized"] is False
    assert [row["lane_state"] for row in workboard["lanes"]] == [
        "accepted_primitive",
        "accepted_primitive",
        "rejected",
        "accepted_numerical_equivalence",
        "accepted_numerical_equivalence",
        "rejected",
    ]
    assert next(row for row in loaded["panels"] if row["id"] == "k1_strict_correctness")["passed"] == 2
    assert next(row for row in loaded["panels"] if row["id"] == "runtime_comparable")["passed"] == 0


def test_runtime_workboard_is_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    scorecard["runtime_lane_workboard"]["lanes"][0]["promotion_authorized"] = True
    with pytest.raises(ValueError, match="runtime lane workboard changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_performance_gate_updates_are_fail_closed(tmp_path: Path, scorecard: dict) -> None:
    scorecard["performance_gate_updates"]["single_lane_applicability"]["actual_coarse_translation_count"] = 116
    with pytest.raises(ValueError, match="current performance gate updates changed"):
        progress_mod.load_and_validate(_write(tmp_path, scorecard))


def test_runtime_workboard_is_easy_to_scan() -> None:
    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())

    assert "## Performance lanes" in rendered
    assert "**ACCEPTED PRIMITIVE ONLY** | Flat-row scorer `13266322/13266460`" in rendered
    assert "32.88%, 35.77%, 54.53%, 32.68%" in rendered
    assert "**ACCEPTED PRIMITIVE ONLY** | Stable fine window `13264981/13265301`" in rendered
    assert "**REJECTED** | Batched CUB trajectory `13268653`" in rendered
    assert "it4/p285, it16/p2902, it18/p902" in rendered
    assert "87.9460303805" in rendered
    assert "a67f6c969e84da096c70d88219ddb4e6962ecd13266814743d992099be7b172d" in rendered
    assert "**NUMERICALLY EQUIVALENT / E2E INCONCLUSIVE** | Elementwise primitive `13269547`" in rendered
    assert "only roundoff-scale terminal relative-L2 `5.452e-07`" in rendered
    assert "42544acfe0ae193022808abdbdf56639f418f6102d4e66b9a44ddc1a0aa1ff56" in rendered
    assert "Same-binary ABBA `13271166` | **NUMERICALLY EQUIVALENT; END-TO-END GAIN IMMATERIAL**" in rendered
    assert "warm speedup is `1.0091x`" in rendered
    assert "strict two-control map-diameter flag alone is not a scientific rejection" in rendered
    assert "**NUMERICALLY EQUIVALENT / E2E IMMATERIAL** | Same-binary causal `13271166`" in rendered
    assert "ccb9d9cc4f4ee949aabbfa2c6045aea5b6c2007bcdbcd871e0e1df246d0c3db0" in rendered
    assert "Invalid jobs `13270868`, `13270984`, `13285416`, and `13285596`" in rendered
    assert "**REJECTED FOR GF46 / RETAINED PRIMITIVE** | Single-lane coarse" in rendered
    assert "13280613/13280655" in rendered
    assert "ALL PREDECLARED NUMERICAL AND RUNTIME GATES PASS AFTER INDEPENDENTLY REVIEWED" in rendered
    assert "`13285438/13285647/13286397`" in rendered
    assert "improves warm wall 9.03%, expectation 10.26%, GPU union 10.45%, and coarse union 10.83%" in rendered
    assert "frozen two-repeat max-envelope rule fails narrowly" in rendered
    assert "**MATH ACCEPTED / PERFORMANCE REJECTED** | Direct RELION x-half BPref" in rendered
    assert "`13281684/13282815`" in rendered
    assert "Finalize improves 85.40%, but warm wall improves only 2.12%" in rendered
    assert "13281914/13281950/13282022" in rendered
    assert "**MATH ACCEPTED / PERFORMANCE REJECTED** | Shared posterior executor" in rendered
    assert "`13280796/13281970`" in rendered
    assert "posterior-kernel time regresses 36.86%" in rendered
    assert "**PENDING** | None" not in rendered
    assert "trajectory next" not in rendered.lower()
    assert "default-off/unwired" in rendered
    assert "no impact" in rendered


def test_dashboard_is_compact_and_exposes_shared_em_reuse() -> None:
    rendered = progress_mod.render_markdown(progress_mod.load_and_validate())
    assert len(rendered.splitlines()) < 260
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
