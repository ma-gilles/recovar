#!/usr/bin/env python3
"""Validate and render the authoritative VDAM v3 parity dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs/math/vdam_relion_parity_scorecard_v3.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/math/vdam_relion_parity_dashboard.md"
SCHEMA = "recovar.vdam_relion_parity_progress.v1"
SUITE_ID = "vdam-k1-full-trajectory-expansion-v3"
SUITE_VERSION = 3
FROZEN_DENOMINATOR = 20
SUITE_DEFINITION_PATH = "docs/math/vdam_k1_full_trajectory_expansion_v3.json"
SUITE_DEFINITION_SHA256 = "9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091"
VALID_RESULTS = frozenset({"pass", "fail"})
METRIC_POLICY = (
    "Strict K=1 correctness requires the map, particle-state, and pre-divergence schedule gates to pass "
    "across every numbered checkpoint 0--200. Runtime is an independent gate and cannot change correctness. "
    "Correlation is not computed or gated."
)
PANEL_POLICY = {
    "k1_strict_correctness": ("K=1 strict full-trajectory correctness", "release_gate"),
    "map_trajectory": ("Map trajectory", "diagnostic_component"),
    "particle_trajectory": ("Particle-state trajectory", "diagnostic_component"),
    "predivergence_schedule": ("Pre-divergence schedule", "diagnostic_component"),
    "runtime_comparable": ("Runtime within 1.10x RELION", "independent_release_gate"),
    "terminal_audits": ("Complete terminal audits", "coverage"),
}
ACTIVE_DIAGNOSTIC_POLICY = {
    "13206294": ("diagnostic", "pass", "expected_contract_failure", "DIAGNOSTIC"),
    "13207483": ("invalid", "invalid", "cancelled", "INVALID"),
    "13207996": ("invalid_setup", "invalid", "failed_setup", "INVALID setup"),
    "13208089": ("invalid_setup", "invalid", "failed_setup", "INVALID setup"),
    "13208186": ("invalid", "invalid", "cancelled", "INVALID"),
    "13208265": ("invalid", "invalid", "cancelled", "INVALID"),
    "13208734": ("invalid", "invalid", "cancelled", "INVALID"),
    "13208735": ("invalid", "invalid", "cancelled", "INVALID"),
    "13209422": ("invalid", "invalid", "cancelled", "INVALID"),
    "13210232": ("expected_fail_closed", "hypothesis_rejected", "failed", "EXPECTED FAIL-CLOSED"),
    "13211317": ("invalid_superseded", "invalid", "failed", "INVALID/SUPERSEDED"),
    "13211719": (
        "expected_hypothesis_rejection",
        "hypothesis_rejected",
        "failed",
        "EXPECTED HYPOTHESIS REJECTION",
    ),
    "13212500": ("invalid_harness", "invalid", "cancelled", "INVALID HARNESS"),
}
CRITICAL_CACHE_MISS_JOBS = frozenset({"13208734", "13208735", "13209422"})
PROFILE_MATCHED_EVIDENCE = {
    "diagnostic_head": "39a38f9d6",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cache-history-13209422.out",
    "evidence_sha256": "4a73824d224b89e05cb98a2887ac024ed1ad40afe96ce0e416d4b1fa5a7cf3d4",
    "evidence_root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_jax_cache_history_profilematched_39a38f9d6_20260830",
    "arms_report": "analysis/arms.tsv",
    "arms_report_sha256": "5ecba0675af414edd40720b8635c440a349965a1fe365411e245891bbb72c413",
    "fresh_a_native_envelope_report": "analysis/fresh_a_native_particle_envelope.json",
    "fresh_a_native_envelope_sha256": "a75464214b57024c76fc39eea41a08c6bd1e0a6ed67ad38ea56c838521c099ac",
    "warm80_a_native_envelope_report": "analysis/warm80_a_native_particle_envelope.json",
    "warm80_a_native_envelope_sha256": "d830c3c65fac69d1a97d48d6d2e06afa296e827252a9aefed60ec05a7a34f9c3",
    "elapsed": "2:23",
    "fresh_a": {
        "scientific_outcome": "pass",
        "native_envelope": "pass_through_iteration_4",
        "audit_status": 0,
        "cache_entries_before": 0,
        "cache_entries_after": 435,
    },
    "warm80_a": {
        "scientific_outcome": "invalid",
        "cache_entries_before": 5037,
        "cache_entries_added": 435,
        "critical_keys_compiled": [
            "run_local_bucket_big_jit",
            "relion_coarse_diff2_projector_f32",
            "coarse posterior",
            "relion_vdam_mstep_fused_projector_x_half",
        ],
    },
}
PAIR_STABLE_EVIDENCE = {
    "wrapper_outcome": "failed",
    "wrapper_failure_expected": True,
    "scorecard_failure": False,
    "hypothesis": "long_run_cache_reuse_or_deserialization_is_necessary",
    "hypothesis_result": "rejected",
    "source_head": "ee673be1f3c6a5182a917d97691efe07ffbd5e4d",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cache-history-13210232.out",
    "evidence_sha256": "3ab442553e78735d03feac43237e91b32caec70381507b50faab5994031f2016",
    "evidence_root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_jax_cache_history_samepath_ee673be1f_20260830",
    "summary_report": "analysis/cache_history_summary.json",
    "summary_report_sha256": "cb13a0d710936a6234dfa242e392b021d0b7f52565eb287b0d32dd14fb8a4782",
    "arms_report": "analysis/arms.tsv",
    "arms_report_sha256": "28a5d2518ee7a46972d4d3456f0ef288b66fdab58d1e30806d06c52731a78ed9",
    "native_envelope_reports": {
        "cold_a": {
            "path": "analysis/cold_a_native_particle_envelope.json",
            "sha256": "a09c0f28cbbdfd26afab46a90782216cc9eab7fe5567de08837513e5b873f00a",
        },
        "warm_a": {
            "path": "analysis/warm_a_native_particle_envelope.json",
            "sha256": "332ab30f757a4d0ac59317fa0f2a2410e7bd17a41d9123b3fb20557937d0064d",
        },
        "cold_b": {
            "path": "analysis/cold_b_native_particle_envelope.json",
            "sha256": "4840df859813fd758e1dfc8e00788a14bef7769512153a758b21979c71b34fbd",
        },
        "warm_b": {
            "path": "analysis/warm_b_native_particle_envelope.json",
            "sha256": "418dc0eff9c0ec5ffa3376826acbd6a1955ca45d30a3a2c2c601bd57f5dceb38",
        },
    },
    "pairs": [
        {
            "id": "pair_a",
            "cold": {
                "arm": "cold_a",
                "result": "pass",
                "first_failure_iteration": None,
                "audit_status": 0,
                "cache_entries_before": 0,
                "cache_entries_after": 4823,
            },
            "warm": {
                "arm": "warm_a",
                "result": "pass",
                "first_failure_iteration": None,
                "audit_status": 0,
                "cache_entries_before": 4823,
                "cache_entries_after": 4823,
                "cache_byte_stable": True,
            },
        },
        {
            "id": "pair_b",
            "cold": {
                "arm": "cold_b",
                "result": "fail",
                "first_failure_iteration": 4,
                "audit_status": 1,
                "cache_entries_before": 0,
                "cache_entries_after": 4676,
            },
            "warm": {
                "arm": "warm_b",
                "result": "fail",
                "first_failure_iteration": 4,
                "audit_status": 1,
                "cache_entries_before": 4676,
                "cache_entries_after": 4676,
                "cache_byte_stable": True,
            },
        },
    ],
    "particle_boundary": {
        "particle": "286@particles.128.mrcs",
        "red_pose_exact_match_historical_graph_pair": True,
    },
    "conclusion": {
        "long_run_cache_reuse_or_deserialization_necessary": False,
        "remaining_boundary": ["compile_or_autotune_variant", "runtime_reduction"],
        "interpretation": (
            "Pair-stable independently compiled cache outcomes reject long-run reuse/deserialization as necessary "
            "while leaving compile/autotune variant versus runtime reduction unresolved."
        ),
    },
}
SUPERSEDED_ORDERED_SHELL_EVIDENCE = {
    "invalid_reason": "superseded_by_valid_same_cache_job",
    "superseded_by": "13211719",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-ordered-noise-13211317.out",
    "evidence_sha256": "fedbccb441958eb01a4609c6e7ea12d062d5724afcb706c7a86c04869780dbb4",
}
ORDERED_SHELL_EVIDENCE = {
    "wrapper_outcome": "failed_expected_scientific_gate",
    "execution_valid": True,
    "arm_success_markers": True,
    "scorecard_failure": False,
    "hypothesis": "same_cache_ordered_shell_repeatability",
    "hypothesis_result": "rejected",
    "source_head": "3b5afd98e8be29a7c631bf7116f0345834274ceb",
    "pre_diagnostic_head": "94bc7d890472641b02491ec6ef746677dd0f05d8",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-ordered-noise-13211719.out",
    "evidence_sha256": "fedbccb441958eb01a4609c6e7ea12d062d5724afcb706c7a86c04869780dbb4",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_gf46_ordered_noise_shell_samecache_3b5afd98e_20260830T1847ET"
    ),
    "repeatability_report": "analysis/repeatability.json",
    "repeatability_report_sha256": "a3c544602c9845e7c514393e6e20b47d153d23ec8b8757c0a60d1b17d5fe2619",
    "cache_report": "analysis/jax_cache_validation.json",
    "cache_report_sha256": "da25be1dbd38b4940fa5e673081136c808d8240fb9e939fd6863408fdaa24794",
    "slurm": {
        "terminal_state": "FAILED 1:0",
        "elapsed": "00:04:32",
        "max_rss": "2421072K",
    },
    "cache_validation": {
        "result": "pass",
        "canonical_path": (
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_gf46_ordered_noise_shell_samecache_3b5afd98e_20260830T1847ET/jax_cache"
        ),
        "manifest_line_counts": {
            "ordered_a_before": 0,
            "ordered_a_after": 377,
            "ordered_b_before": 377,
            "ordered_b_after": 377,
        },
        "manifest_sha256": {
            "ordered_a_before": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "ordered_a_after": "2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5",
            "ordered_b_before": "2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5",
            "ordered_b_after": "2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5",
        },
        "arm_paths_match": True,
        "arm_b_added_files": 0,
        "arm_b_changed_files": 0,
        "arm_b_byte_stable": True,
    },
    "earliest_captured_nonexact": {
        "stage": "E-step",
        "field": "max_posterior_per_image",
        "dtype": "float32",
        "ulps": 1,
        "max_abs": 1.1641532182693481e-10,
        "particle_id": 2896,
        "selected_index": 178,
        "pose_exact": True,
        "rotation_exact": True,
        "translation_exact": True,
        "class_exact": True,
    },
    "ordered_noise": {
        "image_power": {"nonexact_shells": 27, "shells": 65, "max_abs": 512.0},
        "sigma2_noise_numerator": {"nonexact_shells": 1, "shells": 65, "max_abs": 0.00390625},
        "final_sigma2_noise": {
            "nonexact_shells": 24,
            "shells": 65,
            "max_abs": 0.0015625000014551915,
        },
    },
    "bpref": {
        "exact": False,
        "nonexact_field_count": 12,
        "nonexact_halves": [0, 1],
        "per_half": {
            "0": {"nonexact_fields": 6, "max_abs": 0.0234375},
            "1": {"nonexact_fields": 6, "max_abs": 0.0107421875},
        },
    },
    "mstep": {"exact": False, "nonexact_field_count": 16},
    "target_stack": {"stack_index": 286, "exact": True},
}
INVALID_SPEED_HARNESS_EVIDENCE = {
    "invalid_reason": "automatic_cuda_rebuild_targeted_qualified_source_artifact",
    "source_head": "73945d69f6e7c609e6830dc65d5de84c63434d5b",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-sig-bucket-ab-13212500.out",
    "evidence_sha256": "f39303615d8f692e242f0df8116649139e9d47d4bc78ad5186b2a2f82c47eed3",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_gf01_sig_bucket_ab_73945d69f_20260830T1900ET"
    ),
    "runner_log": "trials/cold/control/runner.log",
    "runner_log_sha256": "693ca77cb8a01ad7dc78b282df19f7df8c5ef4bc624ad2c9f99ef365b508fc04",
    "command_report": "trials/cold/control/command.json",
    "command_report_sha256": "d2461dc96dc202fcd5765a274d44bb1985f087b1559534965391f7115e1ff0ab",
    "native_provenance": "provenance/native_extensions.json",
    "native_provenance_sha256": "3290d654e5b143d719fa90734ffa66cce9f0d7d495097c6fd57c5df78cb2e65c",
    "source_cuda_library": {
        "path": (
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_ordered_scatter_graph_gate_6b5e6568a_20260830/libcuda_backproject.so"
        ),
        "sha256": "a548e44d81adcad7d0356ad369d8cfd23aae7404c1383b1ca2cf85967e77241b",
        "size": 8403456,
        "inode": 239857888,
        "mtime": "2026-08-30T13:10:53.397799229-04:00",
        "bytes_unchanged_after_cancellation": True,
    },
    "source_build_lock": {
        "path": (
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_ordered_scatter_graph_gate_6b5e6568a_20260830/.build.lock"
        ),
        "created_at": "2026-08-30T19:04:15.109070000-04:00",
    },
    "slurm": {
        "terminal_state": "CANCELLED by 230216",
        "exit_code": "0:0",
        "elapsed": "00:04:23",
        "max_rss": "2439592K",
    },
    "science_result_recorded": False,
    "runtime_result_recorded": False,
    "promotion_authorized": False,
}
ENGINEERING_SNAPSHOT_POLICY = {
    "as_of": "2026-08-31",
    "frozen_scores_changed": False,
    "score_impact": "none",
    "active_gate": {
        "job_id": "13253088",
        "status": "active_result_pending",
        "source_head": "6f39ad52e",
        "case_id": "vdam-gf46",
        "iterations": 80,
        "hardware": "H100",
        "purpose": "Run the two-repeat warm trajectory and Nsight gate for the audited typed GUI defaults.",
        "score_impact": "none_until_audited",
    },
    "short_gate": {
        "job_id": "13252518",
        "status": "completed_audited",
        "source_head": "6f39ad52e",
        "case_id": "vdam-gf46",
        "iterations": 20,
        "hardware": "H100",
        "recovar_wall_seconds": 177,
        "cumulative_pre_artifact_seconds": 158.846,
        "particle_count": 3000,
        "exact_particle_state_iterations": 20,
        "first_divergent_iteration": None,
        "requested_effective_wavg_sequential_cuda": True,
        "requested_effective_exact_local_bucket_radix": 4,
        "comparison_note": (
            "177 s is below the 182 s accepted short baseline but above the 169.40 s prior exact-control result; "
            "these are cross-run H100 observations, not a paired speed result."
        ),
        "evidence_root": (
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_gf46_typed_runtime20_6f39ad52e_20260831"
        ),
        "particle_audit": "provenance/particle_state_audit_it001_020.json",
        "score_impact": "none",
    },
    "typed_runtime_controls": {
        "integration_head": "6f39ad52e",
        "integration_commits": ["59c0abde9", "4672232a3", "6f39ad52e"],
        "focused_checks": "18/18",
        "defaults": {
            "relion_wavg_sequential_cuda": True,
            "exact_local_bucket_radix": 4,
        },
    },
    "warm_profile": {
        "job_id": "13248509",
        "source_head": "6b5e6568a",
        "case_id": "vdam-gf46",
        "iteration_window": "47--80",
        "wall_span_seconds": 240.36,
        "gpu_kernel_seconds": 44.58,
        "coarse_projector_seconds": 35.899,
        "coarse_projector_gpu_percent": 80.53,
        "local_exact_wall_seconds": 130.09,
        "pass1_seconds": 71.39,
        "xla_compile_seconds": 57.83,
        "dataset_getitem_seconds": 31.90,
        "disk_read_seconds": 25.36,
        "fine_fused_seconds": 1.897,
        "weighted_average_seconds": 0.556,
        "evidence_root": (
            "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
            "vdam_gf46_warm_nsys_v3_6b5e6568a_20260831"
        ),
    },
    "decisions": [
        {
            "candidate": "shared EM arithmetic primitives",
            "status": "KEEP",
            "evidence": (
                "Coarse scoring, compact planning/fine posterior, Wavg, radix buckets, and ordered "
                "accumulation already use the mature EM implementations."
            ),
        },
        {
            "candidate": "ordered-scatter CUDA Graph",
            "status": "QUALIFIED CANDIDATE",
            "evidence": (
                "Job 13203664 was quality-neutral (80/80 particles; 81/81 maps) and reduced wall "
                "293->291 s."
            ),
        },
        {
            "candidate": "eager shared raw-image cache",
            "status": "REJECTED",
            "evidence": (
                "Exact-control wall regressed 169.40->172.20 s (+1.7%); 9cb34ddf2 was reverted by "
                "274e4062d."
            ),
        },
        {
            "candidate": "inline indexed fine projection",
            "status": "REJECTED",
            "evidence": "Job 13249200 was bitwise exact but 2--4x slower because it repeated projections.",
        },
        {
            "candidate": "float32 coarse scorer",
            "status": "REJECTED",
            "evidence": (
                "788,541/802,681 coarse scans changed; commit 31953d remains a regression-only prototype."
            ),
        },
        {
            "candidate": "typed Wavg/radix defaults",
            "status": "SHORT GATE EXACT; WARM80 ACTIVE",
            "evidence": (
                "Job 13252518 retained all 3,000 particle states through iteration 20; two-repeat warm job "
                "13253088 is pending."
            ),
        },
    ],
}
EVIDENCE_SOURCE_POLICY = {
    "v3_original": {
        "root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826",
        "source_head": "984637b7db95f1ca6f5800c08ea14c1e32c82c2e",
        "analysis_tag": "945a4201f0",
        "cuda_library_sha256": "87274beac3a7b5af59947199588955366485d22780239f4c94fd5afc13f8e337",
        "role": "primary_scientific_evidence",
    },
    "gf43_seeded": {
        "root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf43_full_seeded_accuracy_580477763_87274be_20260826",
        "source_head": "580477763f0f95f028841b074210c4eba34fd24b",
        "map_report_sha256": "493ed7391766378593805c42525c34591e1f802289d937f71190761b09996aeb",
        "particle_report_sha256": "2566afcb182f58c17bd2c40cd604d680050907cf46503d2975a58ac318b0f8c2",
        "schedule_report_sha256": "cd810a7cedc925982bce525a536a7ed1478c509bd0aff367901831e165d73219",
        "role": "superseding_scientific_evidence",
    },
    "gf45_seeded": {
        "root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf45_full_seeded_accuracy_580477763_87274be_20260826",
        "source_head": "580477763f0f95f028841b074210c4eba34fd24b",
        "map_report_sha256": "01f56dd2d66b11349bcbe05a903fbff73af2e95bfd6919537cbbda2637c74dbf",
        "particle_report_sha256": "7670bafc27ea9157d7bf6780e338e95025d8579707ea1e3da6c41bb15ffd29e6",
        "schedule_report_sha256": "d66c6aeab17b11e3467068a4b83e9eac6b87c31f9087afe713df2363e382392f",
        "role": "superseding_scientific_evidence",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_suite_definition(scorecard: dict[str, Any]) -> dict[str, Any]:
    expected_ref = {"path": SUITE_DEFINITION_PATH, "sha256": SUITE_DEFINITION_SHA256}
    _require(scorecard.get("suite_definition") == expected_ref, "v3 suite-definition identity changed")
    path = REPO_ROOT / SUITE_DEFINITION_PATH
    _require(path.is_file(), f"missing v3 suite definition: {path}")
    _require(sha256_file(path) == SUITE_DEFINITION_SHA256, "v3 suite-definition bytes changed")
    suite = json.loads(path.read_text())
    _require(suite.get("suite_id") == SUITE_ID, "v3 suite ID differs from scorecard")
    suite_cases = suite.get("cases")
    _require(isinstance(suite_cases, list) and len(suite_cases) == FROZEN_DENOMINATOR, "v3 suite denominator changed")
    checkpoints = suite.get("acceptance_contract", {}).get("required_checkpoints")
    _require(checkpoints == list(range(201)), "v3 checkpoint range changed")
    return suite


def _validate_gate(case_id: str, label: str, gate: object) -> str:
    _require(isinstance(gate, dict), f"{case_id}: {label} gate must be an object")
    result = gate.get("result")
    _require(result in VALID_RESULTS, f"{case_id}: invalid {label} result")
    first_failure = gate.get("first_failure_iteration")
    if result == "pass":
        _require(first_failure is None, f"{case_id}: passing {label} gate has a failure iteration")
    else:
        _require(
            isinstance(first_failure, int) and not isinstance(first_failure, bool) and 0 <= first_failure <= 200,
            f"{case_id}: failing {label} gate needs an iteration in [0, 200]",
        )
    return str(result)


def _derived_panel_counts(cases: list[dict[str, Any]], runtime_ratio_max: float) -> dict[str, int]:
    return {
        "k1_strict_correctness": sum(case["strict_result"] == "pass" for case in cases),
        "map_trajectory": sum(case["map"]["result"] == "pass" for case in cases),
        "particle_trajectory": sum(case["particle"]["result"] == "pass" for case in cases),
        "predivergence_schedule": sum(case["schedule"]["result"] == "pass" for case in cases),
        "runtime_comparable": sum(case["runtime"]["ratio_vs_relion"] <= runtime_ratio_max for case in cases),
        "terminal_audits": sum(case["terminal_audit"] is True for case in cases),
    }


def _validate_cases(scorecard: dict[str, Any], suite: dict[str, Any]) -> dict[str, int]:
    cases = scorecard.get("cases")
    _require(isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR, "v3 scorecard denominator changed")
    suite_cases = suite["cases"]
    expected_identity = [
        (case["id"], case["name"], case["definition"]["random_seed"]) for case in suite_cases
    ]
    actual_identity = [(case.get("id"), case.get("name"), case.get("seed")) for case in cases]
    _require(actual_identity == expected_identity, "v3 case identity, order, or seed changed")
    evidence_sources = scorecard.get("evidence_sources")
    _require(isinstance(evidence_sources, dict), "evidence sources must be an object")
    runtime_ratio_max = float(scorecard["acceptance_contract"]["runtime_ratio_max"])
    for case in cases:
        case_id = str(case["id"])
        gate_results = [_validate_gate(case_id, label, case[label]) for label in ("map", "particle", "schedule")]
        expected_strict = "pass" if all(result == "pass" for result in gate_results) else "fail"
        _require(case.get("strict_result") == expected_strict, f"{case_id}: strict result does not replay its gates")
        runtime = case.get("runtime")
        _require(isinstance(runtime, dict), f"{case_id}: runtime must be an object")
        ratio = runtime.get("ratio_vs_relion")
        _require(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0, f"{case_id}: invalid runtime ratio")
        expected_runtime = "pass" if float(ratio) <= runtime_ratio_max else "fail"
        _require(runtime.get("result") == expected_runtime, f"{case_id}: runtime result does not replay its gate")
        _require(case.get("terminal_audit") is True, f"{case_id}: terminal audit is incomplete")
        _require(case.get("evidence_source") in evidence_sources, f"{case_id}: unknown evidence source")
        _require(str(case.get("original_science_job", "")).isdigit(), f"{case_id}: invalid science job")
    return _derived_panel_counts(cases, runtime_ratio_max)


def _validate_panels(scorecard: dict[str, Any], derived_counts: dict[str, int]) -> None:
    panels = scorecard.get("panels")
    _require(isinstance(panels, list), "panels must be a list")
    _require([panel.get("id") for panel in panels] == list(PANEL_POLICY), "panel identity or order changed")
    for panel in panels:
        panel_id = str(panel["id"])
        label, role = PANEL_POLICY[panel_id]
        _require(panel.get("label") == label and panel.get("role") == role, f"{panel_id}: panel policy changed")
        _require(panel.get("passed") == derived_counts[panel_id], f"{panel_id}: count does not replay cases")
        _require(panel.get("evaluated") == FROZEN_DENOMINATOR, f"{panel_id}: evaluated count changed")
        _require(panel.get("denominator") == FROZEN_DENOMINATOR, f"{panel_id}: denominator changed")


def _validate_active_diagnostics(scorecard: dict[str, Any]) -> None:
    diagnostics = scorecard.get("active_diagnostics")
    _require(isinstance(diagnostics, list), "active diagnostics must be a list")
    _require([row.get("job_id") for row in diagnostics] == list(ACTIVE_DIAGNOSTIC_POLICY), "active job set changed")
    for row in diagnostics:
        job_id = str(row["job_id"])
        status, science, scheduler, _ = ACTIVE_DIAGNOSTIC_POLICY[job_id]
        _require(row.get("role") == "diagnostic", f"{job_id}: role must remain diagnostic")
        _require(row.get("status") == status, f"{job_id}: status changed without an evidence update")
        _require(row.get("scientific_outcome") == science, f"{job_id}: scientific outcome changed")
        _require(row.get("scheduler_state") == scheduler, f"{job_id}: scheduler state changed")
        _require(row.get("score_impact") == "none", f"{job_id}: diagnostic cannot affect score")
        if job_id in CRITICAL_CACHE_MISS_JOBS:
            _require(
                row.get("invalid_reason") == "warm_cache_compiled_critical_science_keys",
                f"{job_id}: invalid cache-miss reason changed",
            )
    profile_matched = next(row for row in diagnostics if row["job_id"] == "13209422")
    _require(
        {key: profile_matched.get(key) for key in PROFILE_MATCHED_EVIDENCE} == PROFILE_MATCHED_EVIDENCE,
        "13209422: profile-matched evidence changed",
    )
    pair_stable = next(row for row in diagnostics if row["job_id"] == "13210232")
    _require(
        {key: pair_stable.get(key) for key in PAIR_STABLE_EVIDENCE} == PAIR_STABLE_EVIDENCE,
        "13210232: pair-stable evidence changed",
    )
    superseded_ordered_shell = next(row for row in diagnostics if row["job_id"] == "13211317")
    _require(
        {
            key: superseded_ordered_shell.get(key)
            for key in SUPERSEDED_ORDERED_SHELL_EVIDENCE
        }
        == SUPERSEDED_ORDERED_SHELL_EVIDENCE,
        "13211317: superseded ordered-shell evidence changed",
    )
    ordered_shell = next(row for row in diagnostics if row["job_id"] == "13211719")
    _require(
        {key: ordered_shell.get(key) for key in ORDERED_SHELL_EVIDENCE} == ORDERED_SHELL_EVIDENCE,
        "13211719: ordered-shell evidence changed",
    )
    invalid_speed = next(row for row in diagnostics if row["job_id"] == "13212500")
    _require(
        {key: invalid_speed.get(key) for key in INVALID_SPEED_HARNESS_EVIDENCE}
        == INVALID_SPEED_HARNESS_EVIDENCE,
        "13212500: invalid speed-harness evidence changed",
    )


def load_and_validate(path: Path = DEFAULT_SCORECARD) -> dict[str, Any]:
    scorecard = json.loads(Path(path).read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported VDAM progress schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "VDAM v3 suite identity changed")
    _require(scorecard.get("suite_version") == SUITE_VERSION, "VDAM suite version changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "VDAM v3 denominator changed")
    _require(scorecard.get("metric_policy") == METRIC_POLICY, "VDAM v3 metric policy changed")
    policy = scorecard.get("score_policy")
    _require(
        policy
        == {
            "strict_case_gate": ["map", "particle", "schedule"],
            "runtime_is_independent": True,
            "diagnostics_have_score_impact": False,
            "secondary_tracks_have_score_impact": False,
        },
        "VDAM v3 score policy changed",
    )
    suite = _load_suite_definition(scorecard)
    contract = scorecard.get("acceptance_contract")
    suite_contract = suite["acceptance_contract"]
    expected_contract = {
        "cross_engine_fsc_auc_min": suite_contract["cross_engine_fsc_auc_min"],
        "recovar_minus_relion_gt_fsc_auc_min": suite_contract["recovar_minus_relion_gt_fsc_auc_min"],
        "required_checkpoints_start": 0,
        "required_checkpoints_stop": 200,
        "exact_schedule": suite_contract["exact_schedule"],
        "exact_artifact_topology": suite_contract["exact_artifact_topology"],
        "same_physical_gpu_per_pair": suite_contract["same_physical_gpu_per_pair"],
        "correlation_used": suite_contract["correlation_used"],
        "runtime_ratio_max": 1.1,
    }
    _require(contract == expected_contract, "VDAM v3 acceptance contract changed")
    _require(scorecard.get("evidence_sources") == EVIDENCE_SOURCE_POLICY, "VDAM v3 evidence identity changed")
    derived_counts = _validate_cases(scorecard, suite)
    _validate_panels(scorecard, derived_counts)
    _validate_active_diagnostics(scorecard)
    _require(
        scorecard.get("engineering_snapshot") == ENGINEERING_SNAPSHOT_POLICY,
        "current engineering snapshot changed without an evidence update",
    )
    accepted = [case["id"] for case in scorecard["cases"] if case["strict_result"] == "pass"]
    _require(accepted == ["vdam-gf44", "vdam-gf45"], "accepted v3 case set changed")
    secondary = scorecard.get("secondary_tracks")
    _require(isinstance(secondary, list) and all(row.get("score_impact") == "none" for row in secondary), "secondary track changed score")
    _require([row.get("id") for row in secondary] == ["legacy_parameter_expansion_v2", "k_greater_than_one", "real_data"], "secondary track identity changed")
    _require(
        (secondary[0].get("passed"), secondary[0].get("evaluated"), secondary[0].get("denominator"))
        == (6, 15, 15),
        "legacy v2 snapshot changed",
    )
    interface = scorecard.get("interface_policy")
    _require(
        interface
        == {
            "policy_commit": "36103aaa2",
            "focused_tests": "28/28",
            "cli_default": "relion_fast",
            "gui_default": "relion_fast",
            "reference_mode": "diagnostic",
            "k_greater_than_one": "unqualified",
        },
        "CLI/GUI policy snapshot changed",
    )
    shared = scorecard.get("shared_em_reuse")
    expected_shared_components = [
        "coarse projector/scorer",
        "compact active-row planner, fine scorer, and posterior",
        "sequential weighted-average accumulation",
        "radix buckets",
        "ordered-scatter CUDA Graph",
    ]
    _require(
        isinstance(shared, list)
        and [row.get("component") for row in shared] == expected_shared_components
        and all(row.get("shared_with_em") is True for row in shared),
        "shared EM implementation inventory changed",
    )
    speed = scorecard.get("speed_snapshot")
    _require(
        isinstance(speed, dict)
        and speed.get("role") == "diagnostic_performance"
        and speed.get("score_impact") == "none",
        "speed diagnostic changed role or score impact",
    )
    next_gate = scorecard.get("next_gate")
    _require(
        isinstance(next_gate, dict)
        and next_gate.get("harness_fix_commit") == "381bf7949"
        and next_gate.get("prior_profile_matched_head") == "39a38f9d6"
        and next_gate.get("pair_stable_head") == "ee673be1f"
        and next_gate.get("ordered_shell_job") == "13211719"
        and next_gate.get("ordered_shell_head") == "3b5afd98e"
        and next_gate.get("pre_diagnostic_head") == "94bc7d890"
        and next_gate.get("production_change_authorized") is False,
        "cache discriminator gate changed",
    )
    history = scorecard.get("history")
    _require(isinstance(history, list) and history[-1].get("strict_passed") == 2, "current history score changed")
    _require(history[-1].get("denominator") == FROZEN_DENOMINATOR, "current history denominator changed")
    return scorecard


def build_progress(scorecard: dict[str, Any]) -> dict[str, Any]:
    cases = scorecard["cases"]
    return {
        "panels": scorecard["panels"],
        "accepted_cases": [case["id"] for case in cases if case["strict_result"] == "pass"],
        "remaining_cases": [case["id"] for case in cases if case["strict_result"] == "fail"],
        "failure_counts": {
            label: Counter(case[label]["result"] for case in cases)["fail"]
            for label in ("map", "particle", "schedule")
        },
        "scorecard_sha256": sha256_file(DEFAULT_SCORECARD),
        "suite_definition_sha256": sha256_file(REPO_ROOT / SUITE_DEFINITION_PATH),
    }


def _gate_text(gate: dict[str, Any]) -> str:
    if gate["result"] == "pass":
        return "PASS"
    return f"FAIL@{gate['first_failure_iteration']}"


def render_markdown(scorecard: dict[str, Any]) -> str:
    progress = build_progress(scorecard)
    strict = next(panel for panel in scorecard["panels"] if panel["id"] == "k1_strict_correctness")
    runtime = next(panel for panel in scorecard["panels"] if panel["id"] == "runtime_comparable")
    engineering = scorecard["engineering_snapshot"]
    active_gate = engineering["active_gate"]
    short_gate = engineering["short_gate"]
    runtime_ratios = [case["runtime"]["ratio_vs_relion"] for case in scorecard["cases"]]
    lines = [
        "# RECOVAR / RELION VDAM parity dashboard",
        "",
        "> **Authoritative v3 status — NOT READY.** Strict K=1 correctness is "
        f"**{strict['passed']} / {strict['denominator']}** and runtime parity is "
        f"**{runtime['passed']} / {runtime['denominator']}**. The only accepted cases are "
        f"`{', '.join(progress['accepted_cases'])}`.",
        ">",
        "> This page is generated from the frozen 20-case, iteration 0--200 scorecard. "
        "Scheduler diagnostics, the legacy v1/v2 tracks, K>1, and real data cannot change this score.",
        "",
        "## At a glance",
        "",
        "| Axis | Authoritative state | Current engineering read |",
        "|---|---|---|",
        f"| K=1 correctness | **{strict['passed']}/{strict['denominator']}** strict cases pass | "
        "Frozen; no recent diagnostic changes this score. |",
        f"| Runtime | **{runtime['passed']}/{runtime['denominator']}** cases meet 1.10x; "
        f"observed {min(runtime_ratios):.2f}--{max(runtime_ratios):.2f}x | "
        "The shared coarse pass dominates GPU time; compile, rectangular scheduling, and input latency dominate wall. |",
        "| EM reuse | Shared production arithmetic | The remaining gap is execution topology, not duplicate scoring math. |",
        "| Later gates | K>1 unqualified; real data not scored | Kept separate until K=1 correctness and runtime close. |",
        "",
        "### Gate progression",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
        f"| `{short_gate['job_id']}`: {short_gate['iterations']}-iteration `{short_gate['case_id']}` | "
        "**PARTICLE TRAJECTORY EXACT** | All "
        f"{short_gate['particle_count']:,}/{short_gate['particle_count']:,} pose/translation states match at every "
        f"iteration; first divergence is `null`; requested/effective Wavg=`true`, radix=`4`. Wall "
        f"{short_gate['recovar_wall_seconds']} s; pre-artifact {short_gate['cumulative_pre_artifact_seconds']:.3f} s. "
        f"{short_gate['comparison_note']} Evidence: `{short_gate['evidence_root']}/{short_gate['particle_audit']}`. |",
        f"| `{active_gate['job_id']}`: {active_gate['iterations']}-iteration warm two-repeat | "
        f"**RESULT PENDING** | {active_gate['purpose']} No partial result or frozen-score impact is claimed. |",
        "",
        "## Primary panels",
        "",
        "| Gate | Passed | Evaluated | Denominator | Role |",
        "|---|---:|---:|---:|---|",
    ]
    for panel in scorecard["panels"]:
        lines.append(
            f"| {panel['label']} | **{panel['passed']}** | {panel['evaluated']} | "
            f"{panel['denominator']} | `{panel['role']}` |"
        )
    lines.extend(
        [
            "",
            "Strict correctness is the conjunction of map, particle-state, and pre-divergence schedule gates. "
            "Runtime is an independent release gate; diagnostics have no score impact.",
            "",
            "## Frozen v3 case matrix",
            "",
            "| Case | Seed | Stress | Map | Particle | Schedule | Runtime | Strict | Evidence |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in scorecard["cases"]:
        runtime_text = f"{case['runtime']['ratio_vs_relion']:.2f}x FAIL"
        lines.append(
            f"| `{case['id']}` | {case['seed']} | {case['stress']} | {_gate_text(case['map'])} | "
            f"{_gate_text(case['particle'])} | {_gate_text(case['schedule'])} | {runtime_text} | "
            f"{case['strict_result'].upper()} | `{case['evidence_source']}` |"
        )
    lines.extend(
        [
            "",
            "## Active boundary diagnostics",
            "",
            "These are scheduler/causal diagnostics, not v3 score entries. INVALID attempts and expected "
            "hypothesis rejections have no score impact.",
            "",
            "| Job | Role | Status | Scientific outcome | Scheduler state | Score impact | Interpretation |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in scorecard["active_diagnostics"]:
        display_status = ACTIVE_DIAGNOSTIC_POLICY[row["job_id"]][3]
        lines.append(
            f"| `{row['job_id']}` | `{row['role']}` | **{display_status}** | "
            f"{row['scientific_outcome'].replace('_', ' ').upper()} | "
            f"{row['scheduler_state'].replace('_', '-')} | "
            f"{row['score_impact']} | {row['note']} |"
        )
    profile_matched = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13209422")
    critical_keys = ", ".join(f"`{key}`" for key in profile_matched["warm80_a"]["critical_keys_compiled"])
    lines.extend(
        [
            "",
            f"Job `13209422` warm-cache additions included: {critical_keys}. Evidence: "
            f"`{profile_matched['evidence_root']}/{profile_matched['arms_report']}`.",
        ]
    )
    pair_stable = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13210232")
    lines.extend(
        [
            "",
            "### Pair-stable cache result: job `13210232`",
            "",
            "| Pair | Cold arm | Warm arm | Cache transitions | Warm bytes |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for pair in pair_stable["pairs"]:
        cold = pair["cold"]
        warm = pair["warm"]
        lines.append(
            f"| `{pair['id']}` | `{cold['arm']}` {_gate_text(cold)} | `{warm['arm']}` {_gate_text(warm)} | "
            f"{cold['cache_entries_before']}->{cold['cache_entries_after']}; "
            f"{warm['cache_entries_before']}->{warm['cache_entries_after']} | "
            f"{'byte-stable' if warm['cache_byte_stable'] else 'changed'} |"
        )
    particle = pair_stable["particle_boundary"]
    conclusion = pair_stable["conclusion"]
    remaining = " versus ".join(value.replace("_", " ") for value in conclusion["remaining_boundary"])
    lines.extend(
        [
            "",
            f"Particle `{particle['particle']}` has the exact historical graph-pair red pose. The wrapper "
            "**FAILED as an expected fail-closed hypothesis rejection**; this is not a new scorecard failure.",
            "",
            f"Evidence: `{pair_stable['evidence_root']}/{pair_stable['summary_report']}` "
            f"(SHA-256 `{pair_stable['summary_report_sha256']}`).",
            "",
            "**Conclusion:** long-run cache reuse/deserialization is not necessary. Pair-stable independently "
            f"compiled cache outcomes narrow the unresolved boundary to **{remaining}**. Prior ordered-shell "
            "attempt `13211317` is **INVALID/SUPERSEDED** by the valid result below.",
        ]
    )
    ordered_shell = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13211719")
    cache = ordered_shell["cache_validation"]
    counts = cache["manifest_line_counts"]
    manifests = cache["manifest_sha256"]
    earliest = ordered_shell["earliest_captured_nonexact"]
    noise = ordered_shell["ordered_noise"]
    bpref = ordered_shell["bpref"]
    lines.extend(
        [
            "",
            "### Valid same-cache ordered-shell result: job `13211719`",
            "",
            "| Cache snapshot | Files | Manifest SHA-256 |",
            "|---|---:|---|",
            f"| A before | {counts['ordered_a_before']} | `{manifests['ordered_a_before']}` |",
            f"| A after | {counts['ordered_a_after']} | `{manifests['ordered_a_after']}` |",
            f"| B before | {counts['ordered_b_before']} | `{manifests['ordered_b_before']}` |",
            f"| B after | {counts['ordered_b_after']} | `{manifests['ordered_b_after']}` |",
            "",
            f"Both arms used one canonical cache; B added {cache['arm_b_added_files']} files and changed "
            f"{cache['arm_b_changed_files']} files. A-after, B-before, and B-after are byte-stable.",
            "",
            "| Earliest / downstream boundary | Nonexact extent | Maximum absolute difference | Exact companion |",
            "|---|---:|---:|---|",
            f"| E-step `{earliest['field']}` | particle ID `{earliest['particle_id']}` / selected index "
            f"`{earliest['selected_index']}` | {earliest['max_abs']:.17g} "
            f"({earliest['ulps']} float32 ULP) | pose / rotation / translation / class exact |",
            f"| ordered image power | {noise['image_power']['nonexact_shells']}/"
            f"{noise['image_power']['shells']} shells | {noise['image_power']['max_abs']:.1f} | - |",
            f"| ordered sigma numerator | {noise['sigma2_noise_numerator']['nonexact_shells']}/"
            f"{noise['sigma2_noise_numerator']['shells']} shells | "
            f"{noise['sigma2_noise_numerator']['max_abs']:.8f} | - |",
            f"| final noise | {noise['final_sigma2_noise']['nonexact_shells']}/"
            f"{noise['final_sigma2_noise']['shells']} shells | "
            f"{noise['final_sigma2_noise']['max_abs']:.16g} | - |",
            f"| live BPref | both halves; {bpref['nonexact_field_count']} fields | "
            f"h0 {bpref['per_half']['0']['max_abs']}; h1 {bpref['per_half']['1']['max_abs']} | "
            f"target stack `{ordered_shell['target_stack']['stack_index']}` exact |",
            "",
            "The Slurm `FAILED 1:0` terminal state is an **expected scientific-gate hypothesis rejection**: "
            "both arms completed, the execution and cache proof are valid, and this is not a frozen-score failure.",
            "",
            f"Evidence: `{ordered_shell['evidence_root']}/{ordered_shell['repeatability_report']}` "
            f"(SHA-256 `{ordered_shell['repeatability_report_sha256']}`); "
            f"`{ordered_shell['cache_report']}` (SHA-256 `{ordered_shell['cache_report_sha256']}`); "
            f"log SHA-256 `{ordered_shell['evidence_sha256']}`.",
            "",
            "**Conclusion:** identical persistent-cache bytes do not guarantee exact ordered-shell replay. "
            "The first captured upstream difference is one float32 ULP in an E-step posterior scalar; discrete "
            "selection is still exact before differences become visible in ordered noise and both-half BPref.",
        ]
    )
    speed = scorecard["speed_snapshot"]
    invalid_speed = next(row for row in scorecard["active_diagnostics"] if row["job_id"] == "13212500")
    source_cuda = invalid_speed["source_cuda_library"]
    source_lock = invalid_speed["source_build_lock"]
    lines.extend(
        [
            "",
            "### Invalid speed-gate attempt: job `13212500`",
            "",
            f"Cancelled after {invalid_speed['slurm']['elapsed']} before any A/B science or timing result. "
            "The loader treated the qualified CUDA library as stale and launched `make`/`nvcc` against the "
            "source artifact. Its bytes remained unchanged at "
            f"SHA-256 `{source_cuda['sha256']}`, but it created `{source_lock['path']}`. "
            "The attempt is **INVALID HARNESS** and authorizes no 80-iteration promotion.",
            "",
            f"Evidence: `{invalid_speed['evidence_root']}/{invalid_speed['runner_log']}` "
            f"(SHA-256 `{invalid_speed['runner_log_sha256']}`); Slurm log SHA-256 "
            f"`{invalid_speed['evidence_sha256']}`.",
            "",
            "## Speed snapshot",
            "",
            f"Ordered-scatter CUDA Graph candidate `{speed['candidate_commit']}` (paired job `{speed['paired_job']}`) "
            f"ran in {speed['candidate_seconds']} s versus {speed['control_seconds']} s "
            f"({speed['wall_time_change_percent']:.2f}%). Ordered backprojection fell from "
            f"{speed['ordered_backprojection_control_seconds']:.3f} s to "
            f"{speed['ordered_backprojection_candidate_seconds']:.3f} s "
            f"({speed['ordered_backprojection_change_percent']:.2f}%).",
            "",
            f"Quality-neutral candidate/control checks: particles {speed['candidate_control_particle_passed']}/"
            f"{speed['candidate_control_particle_denominator']}; maps {speed['candidate_control_map_passed']}/"
            f"{speed['candidate_control_map_denominator']}. The separate native-particle envelope remains "
            f"{speed['native_particle_envelope_passed']}/{speed['native_particle_envelope_denominator']}. "
            "This performance snapshot cannot change the frozen correctness or runtime panels.",
            "",
            f"### Warm H100 profile: job `{engineering['warm_profile']['job_id']}`",
            "",
            "| Profile slice | Time | Readout |",
            "|---|---:|---|",
            f"| iterations {engineering['warm_profile']['iteration_window']} | "
            f"{engineering['warm_profile']['wall_span_seconds']:.2f} s wall / "
            f"{engineering['warm_profile']['gpu_kernel_seconds']:.2f} s kernels | GPU kernels explain only "
            f"part of wall time. Artifact: `{engineering['warm_profile']['evidence_root']}`. |",
            f"| shared coarse projector | {engineering['warm_profile']['coarse_projector_seconds']:.3f} s | "
            f"{engineering['warm_profile']['coarse_projector_gpu_percent']:.2f}% of GPU kernel time; primary "
            "GPU target. |",
            f"| exact-local / pass 1 wall | {engineering['warm_profile']['local_exact_wall_seconds']:.2f} / "
            f"{engineering['warm_profile']['pass1_seconds']:.2f} s | Globally padded rectangular scheduling is "
            "the main topology target. |",
            f"| XLA compile ranges | {engineering['warm_profile']['xla_compile_seconds']:.2f} s | Shape-policy "
            "stability matters before arithmetic changes. |",
            f"| dataset getitem / disk read | {engineering['warm_profile']['dataset_getitem_seconds']:.2f} / "
            f"{engineering['warm_profile']['disk_read_seconds']:.2f} s | Input latency is material, but the exact "
            "eager cache experiment regressed. |",
            f"| fine fused / Wavg | {engineering['warm_profile']['fine_fused_seconds']:.3f} / "
            f"{engineering['warm_profile']['weighted_average_seconds']:.3f} s | Already small; optimize only without "
            "repeating projections. |",
            "",
            "### Engineering decision ledger",
            "",
            "| Track | Decision | Evidence |",
            "|---|---|---|",
        ]
    )
    for row in engineering["decisions"]:
        lines.append(f"| {row['candidate']} | **{row['status']}** | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Shared EM implementation",
            "",
            "| Component | Shared with EM | Qualification |",
            "|---|---:|---|",
        ]
    )
    for row in scorecard["shared_em_reuse"]:
        lines.append(
            f"| {row['component']} | {'yes' if row['shared_with_em'] else 'no'} | {row['qualification']} |"
        )
    interface = scorecard["interface_policy"]
    typed = engineering["typed_runtime_controls"]
    lines.extend(
        [
            "",
            "VDAM calls the shared EM primitives above; it does not carry duplicate projector, scoring, posterior, "
            "or ordered-accumulation algorithms.",
            "",
            "## Interface and secondary gates",
            "",
            f"CLI and GUI both default to `{interface['cli_default']}`; `reference` remains "
            f"{interface['reference_mode']}. Current typed runtime-control integration `{typed['integration_head']}` "
            f"passed {typed['focused_checks']} focused checks and defaults sequential CUDA Wavg to "
            f"`{str(typed['defaults']['relion_wavg_sequential_cuda']).lower()}` and exact-local radix to "
            f"`{typed['defaults']['exact_local_bucket_radix']}`. K>1 remains {interface['k_greater_than_one']}.",
            "",
            "| Track | Result | Role | v3 score impact |",
            "|---|---:|---|---:|",
        ]
    )
    for row in scorecard["secondary_tracks"]:
        result = (
            f"{row['passed']}/{row['denominator']}"
            if "passed" in row
            else str(row["status"]).replace("_", " ")
        )
        lines.append(f"| `{row['id']}` | {result} | `{row['role']}` | {row['score_impact']} |")
    lines.extend(
        [
            "",
            "## Next gates",
            "",
            f"1. Audit both repeats from active job `{active_gate['job_id']}` through iteration "
            f"{active_gate['iterations']}; require trajectory stability and a same-artifact wall comparison.",
            "2. If green, qualify the same typed defaults on additional frozen K=1 stress cases.",
            "3. Prototype a preprojected indexed-job fine scorer that preserves the exact CUDA reduction tree; the "
            "rejected inline version repeated projections.",
            "4. Attack coarse-pass topology with exact arithmetic and stable shapes; do not revive the rejected "
            "float32 scorer or eager raw cache.",
            "5. Re-run the frozen K=1 matrix before allowing K>1, real-data, or runtime results to affect their own "
            "separate gates.",
            "",
            "## Evidence and reproducibility",
            "",
            f"Frozen suite definition: `{scorecard['suite_definition']['path']}` "
            f"(`{scorecard['suite_definition']['sha256']}`).",
            "",
        ]
    )
    for source_id, source in scorecard["evidence_sources"].items():
        lines.append(f"- `{source_id}` ({source['role']}): `{source['root']}` at `{source['source_head']}`")
    lines.extend(
        [
            "",
            "Detailed chronological notes remain in `docs/math/em_parity_program.md`; they are not the score source.",
            "",
            "Regenerate or validate this dashboard with:",
            "",
            "```bash",
            "pixi run python scripts/report_vdam_parity_progress.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scorecard = load_and_validate(args.scorecard)
    rendered = render_markdown(scorecard)
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"stale generated VDAM dashboard: {args.output}")
    else:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
