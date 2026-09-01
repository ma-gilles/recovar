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
    "13254010": ("expected_fail_closed", "hypothesis_rejected", "failed", "EXPECTED FAIL-CLOSED"),
    "13256612": ("microgate_only", "bitwise_pass_microcase", "completed", "MICROGATE ONLY"),
    "13257087": (
        "valid_science_fail",
        "hypothesis_rejected",
        "failed",
        "VALID SCIENCE FAIL / DO NOT PROMOTE",
    ),
    "13257182": ("valid_exactness_fail", "hypothesis_rejected", "failed", "VALID EXACTNESS FAIL"),
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
COARSE_TAIL_PALETTE_EVIDENCE = {
    "wrapper_outcome": "failed_expected_cache_contract",
    "wrapper_failure_expected": True,
    "scorecard_failure": False,
    "hypothesis": "coarse_128_tail_palette_is_promotable",
    "hypothesis_result": "rejected",
    "source_head": "4846bd5c5bdc35a0f6d232450023082d007a44ff",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-sig-bucket-ab-13254010.out",
    "evidence_sha256": "324bb62a9e257fb8b9871462d99bcb2cbda9750a021db6809b63f85ea088fc7a",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_gf46_sig_bucket_ab_4846bd5c5_20260831"
    ),
    "failure_report": "FAILED.json",
    "failure_report_sha256": "91754058b814633ea39fd3f8f958a3765d48bd21575cc69be489765657da92af",
    "particle_report": "analysis/manual/warm1_candidate_vs_control_particle.json",
    "particle_report_sha256": "f5b71f0d9aa32dcf59e97cf6806a5205907e17d267a66f69c9ba80f663c95a82",
    "warm_particle_result": "pass_20_of_20",
    "warm_control_wall_seconds": 84.2343172990004,
    "warm_candidate_wall_seconds": 87.68975677600065,
    "warm_wall_change_percent": 4.1,
    "warm_control_pass1_seconds": 8.130380868911743,
    "warm_candidate_pass1_seconds": 8.72612977027893,
    "warm_pass1_change_percent": 7.3,
    "cache_contract": {
        "result": "fail",
        "candidate_files_before": 1277,
        "candidate_files_after": 1280,
        "candidate_bytes_before": 9670745,
        "candidate_bytes_after": 9706866,
    },
    "default_enabled": False,
    "promotion_authorized": False,
}
DYNAMIC_TAIL_MICRO_EVIDENCE = {
    "superseded_by": "13257087",
    "source_head": "e5b90c0c6e642b5a98d0b546f6c07e864546719a",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-coarse-tail-mask-13256612.out",
    "evidence_sha256": "56ab0d0871c72e8c09d6e2768cefccc6db5ad549dedb878b04b5c52a0c921d84",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_coarse_dynamic_tail_mask_e5b90c0c6_20260831"
    ),
    "benchmark_report": "benchmark.log",
    "benchmark_report_sha256": "2a1e5eadccc4c79f08ead171412b4fc7b25ea52293d99fa810afeb6aba8f01cf",
    "microgate": {
        "active_batch_size": 3,
        "batch_size": 500,
        "rotation_count": 510,
        "translation_count": 29,
        "active_rows_bitwise_equal": True,
        "inactive_rows_equal_initial_diff2": True,
        "all_active_median_seconds": 0.03901593100090395,
        "tail_masked_median_seconds": 0.001631766001082724,
        "speedup": 23.91024875810364,
    },
    "default_enabled": False,
    "whole_trajectory_qualified": False,
    "promotion_authorized": False,
}
DYNAMIC_TAIL_FULL_EVIDENCE = {
    "wrapper_outcome": "failed_expected_strict_artifact_contract",
    "execution_valid": True,
    "all_arm_success_markers": True,
    "scorecard_failure": False,
    "hypothesis": "dynamic_coarse_tail_mask_is_production_promotable",
    "hypothesis_result": "rejected",
    "source_head": "51357fbecd1a6e977bb8875e7be9a0609ed547c9",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-tail-mask-pair-13257087.out",
    "evidence_sha256": "8e27a0f48d42c072604b326d4ee96e6052b1a794715795f5a9662e5f17f38699",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_gf46_coarse_tail_pair_51357fbec_20260831"
    ),
    "summary_report": "pair_summary.json",
    "summary_report_sha256": "fe00d3936fefa751a676ee8b3b52b262770e50e865d2716a1512a26076d4b1ba",
    "same_physical_gpu": True,
    "iterations": 20,
    "active_rows_per_iteration": 200,
    "static_batch_size": 500,
    "padding_row_fraction": 0.6,
    "candidate_median_wall_seconds": 65.1455342735,
    "control_median_wall_seconds": 65.5981299435,
    "candidate_speedup": 1.0069474550335233,
    "material_speedup": False,
    "particle_pose_translation_exact_iterations": 20,
    "artifact_byte_mismatches": 120,
    "artifact_count": 120,
    "execution_scope": {
        "relion_coarse_gaussian_native_texture": True,
        "relion_coarse_fused_projector": False,
        "path_role": "nondefault_path_diagnostic_only",
        "accepted_warm80_dominant_production_kernel": "relion_coarse_diff2_projector_f32_kernel",
        "accepted_warm80_kernel_seconds": 36.15,
        "accepted_warm80_kernel_calls": 65,
    },
    "default_enabled": False,
    "promotion_authorized": False,
}
DYNAMIC_TAIL_ACTIVE200_EVIDENCE = {
    "source_head": "51357fbecd1a6e977bb8875e7be9a0609ed547c9",
    "evidence": "/scratch/gpfs/GILLES/mg6942/slurmo/vdam-tail-micro200-13257182.out",
    "evidence_sha256": "8f157a19adeedc7dc51cd0785fcd9a3fdff69fd6c209ad8d1466478aca29ed84",
    "evidence_root": (
        "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
        "vdam_coarse_tail_micro_active200_51357fbec_20260831"
    ),
    "benchmark_report": "benchmark.log",
    "benchmark_report_sha256": "b17c74081b77dd0142f5be0e9d97577277bd2ebc874065ddab802ef61b5c1493",
    "active_batch_size": 200,
    "batch_size": 500,
    "speedup": 2.3173440825230447,
    "active_rows_bitwise_equal": False,
    "inactive_rows_equal_initial_diff2": True,
    "default_enabled": False,
    "promotion_authorized": False,
}
ENGINEERING_SNAPSHOT_SHA256 = "7a3818973db45ef0bb3cb84689c6bd9765897b7553b8338d8819d9a21e7c37aa"
RUNTIME_LANE_WORKBOARD_SHA256 = "4aa6666ba0c47c2bce67f5a27e073f145c088c433563e805a77417f67ee03287"
LATE_ITERATION_FACTORIAL_GATE_SHA256 = "24027e3b0bd98e449eb99570e2712cc0c14a3fd9d87f9c77a2a056c32c07946c"
PERFORMANCE_GATE_UPDATES_SHA256 = "0ee14daa8a31dd07deaa2f0313391aa1ba247a5929369daf9b03fde491c814d8"
RUNTIME_LANE_IDS = [
    "call_neutral_flat_row",
    "stable_fine_window",
    "batched_cub_sort_scan",
    "batched_posterior_elementwise",
    "elementwise_same_binary_causal",
    "xhalf_projection_cap",
]
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


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    palette = next(row for row in diagnostics if row["job_id"] == "13254010")
    _require(
        {key: palette.get(key) for key in COARSE_TAIL_PALETTE_EVIDENCE}
        == COARSE_TAIL_PALETTE_EVIDENCE,
        "13254010: coarse-tail palette evidence changed",
    )
    tail_micro = next(row for row in diagnostics if row["job_id"] == "13256612")
    _require(
        {key: tail_micro.get(key) for key in DYNAMIC_TAIL_MICRO_EVIDENCE}
        == DYNAMIC_TAIL_MICRO_EVIDENCE,
        "13256612: dynamic-tail microgate evidence changed",
    )
    tail_full = next(row for row in diagnostics if row["job_id"] == "13257087")
    _require(
        {key: tail_full.get(key) for key in DYNAMIC_TAIL_FULL_EVIDENCE}
        == DYNAMIC_TAIL_FULL_EVIDENCE,
        "13257087: dynamic-tail paired evidence changed",
    )
    tail_active200 = next(row for row in diagnostics if row["job_id"] == "13257182")
    _require(
        {key: tail_active200.get(key) for key in DYNAMIC_TAIL_ACTIVE200_EVIDENCE}
        == DYNAMIC_TAIL_ACTIVE200_EVIDENCE,
        "13257182: dynamic-tail active-200 evidence changed",
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
        _sha256_json(scorecard.get("engineering_snapshot")) == ENGINEERING_SNAPSHOT_SHA256,
        "current engineering snapshot changed without an evidence update",
    )
    workboard = scorecard.get("runtime_lane_workboard")
    _require(
        isinstance(workboard, dict)
        and workboard.get("role") == "diagnostic_performance"
        and workboard.get("score_impact") == "none"
        and workboard.get("frozen_scores_changed") is False
        and workboard.get("production_default_changed") is False
        and [row.get("id") for row in workboard.get("lanes", [])] == RUNTIME_LANE_IDS
        and _sha256_json(workboard) == RUNTIME_LANE_WORKBOARD_SHA256,
        "runtime lane workboard changed without an evidence update",
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
        and speed.get("score_impact") == "none"
        and speed.get("current_audit_job") == "13256248"
        and speed.get("current_result") == "inconclusive"
        and speed.get("current_claim_authorized") is False,
        "speed diagnostic changed role or score impact",
    )
    late_gate = scorecard.get("late_iteration_factorial_gate")
    _require(
        isinstance(late_gate, dict)
        and late_gate.get("role") == "diagnostic_performance"
        and late_gate.get("score_impact") == "none"
        and late_gate.get("frozen_scores_changed") is False
        and late_gate.get("production_change_authorized") is False
        and late_gate.get("case_id") == "vdam-gf46"
        and late_gate.get("transition") == "iteration_180_to_181"
        and late_gate.get("source_head") == "f61808a0e6649aa1f53e77ea52d7ce067ff0f817"
        and late_gate.get("hardware", {}).get("node") == "della-h21g4"
        and late_gate.get("hardware", {}).get("gpu_uuid")
        == "GPU-099c0d77-bb85-f2e9-f628-148b733c9176"
        and late_gate.get("factorial_control") == "A"
        and [row.get("id") for row in late_gate.get("factorial_arms", [])] == ["A", "B", "C", "D"]
        and [row.get("job_id") for row in late_gate.get("factorial_arms", [])]
        == ["13276891", "13276923", "13277456", "13277457"]
        and all(row.get("tracked_discrete_invariants_exact") is True for row in late_gate["factorial_arms"])
        and [row.get("decision") for row in late_gate["factorial_arms"]]
        == ["CONTROL", "HOLD_REJECT_DEFAULT", "RETAIN_FOR_REPEAT_AND_SCALE_GATE_NOT_PROMOTED", "REJECT"]
        and _sha256_json(late_gate) == LATE_ITERATION_FACTORIAL_GATE_SHA256,
        "late-iteration factorial gate changed without an evidence update",
    )
    numerical_policy = late_gate["numerical_acceptance_policy"]
    _require(
        numerical_policy
        == {
            "mathematical_equivalence_required": True,
            "stable_repeat_envelope_bounded_floating_point_noise_accepted": True,
            "bitwise_identity_required": False,
            "discrete_changes_measured": True,
            "rare_marginal_discrete_changes_may_be_accepted_only_with_control_or_native_repeat_variability": True,
            "accepted_noise_must_be_unbiased_and_non_growing": True,
            "same_basin_required": True,
            "no_material_final_quality_loss_required": True,
            "slight_quality_change_within_stable_repeat_envelope_allowed_for_large_runtime_gain": True,
            "unstable_numerics_rejected": True,
            "meaningful_runtime_gain_required": True,
        },
        "late-iteration numerical acceptance policy changed",
    )
    gate_updates = scorecard.get("performance_gate_updates")
    _require(
        isinstance(gate_updates, dict)
        and gate_updates.get("role") == "diagnostic_performance"
        and gate_updates.get("score_impact") == "none"
        and gate_updates.get("frozen_scores_changed") is False
        and gate_updates.get("production_default_changed") is False
        and gate_updates.get("single_lane_applicability", {}).get("late_factorial_job") == "13280655"
        and gate_updates.get("single_lane_applicability", {}).get("actual_coarse_translation_count") == 29
        and gate_updates.get("atomic_t29_reduction", {}).get("job_id") == "13281836"
        and gate_updates.get("atomic_t29_reduction", {}).get("numerical_result", {}).get(
            "winner_euler_translation_pmax_exact"
        )
        == "3000/3000"
        and gate_updates.get("atomic_t29_reduction", {}).get("multistream_crossed_gate", {}).get(
            "crossed_job"
        )
        == "13283759"
        and gate_updates.get("atomic_t29_reduction", {}).get("multistream_crossed_gate", {}).get(
            "report_json_sha256"
        )
        == "c843158cfc6bdbd1a7e5d0c59325c98a0f0f5ec647d1bd458911b9d991a55a79"
        and gate_updates.get("atomic_t29_reduction", {}).get("multistream_crossed_gate", {}).get(
            "default_enablement_allowed"
        )
        is False
        and gate_updates.get("atomic_t29_reduction", {}).get("decision")
        == "ACCEPT_DEFAULT_OFF_TRAJECTORY_CANDIDATE_NOT_DEFAULT"
        and gate_updates.get("direct_relion_xhalf", {}).get("qualified_job") == "13281684"
        and gate_updates.get("direct_relion_xhalf", {}).get("direct_vs_legacy_bpref_bitwise") is True
        and gate_updates.get("direct_relion_xhalf", {}).get("crossed_live_job") == "13282815"
        and gate_updates.get("direct_relion_xhalf", {}).get("decision")
        == "REJECT_NO_MATERIAL_WARM_RUNTIME_WIN"
        and gate_updates.get("shared_posterior_executor", {}).get("qualified_gpu_job") == "13280796"
        and gate_updates.get("shared_posterior_executor", {}).get("crossed_live_job") == "13281970"
        and gate_updates.get("shared_posterior_executor", {}).get("decision")
        == "REJECT_NO_MATERIAL_WARM_RUNTIME_WIN"
        and gate_updates.get("remaining_profile_decomposition", {}).get("status")
        == "SEALED_READ_ONLY_SCHEDULING_TARGET"
        and gate_updates.get("remaining_profile_decomposition", {}).get("report_sha256")
        == "079f52c02b20128902e99461770092fbadd6aa35f30b7c0b1ad209b53ff3658b"
        and gate_updates.get("remaining_profile_decomposition", {}).get("kernel_work", {}).get(
            "recovar_overhead_percent"
        )
        == 3.051596
        and gate_updates.get("remaining_profile_decomposition", {}).get("measured_headroom", {}).get(
            "excess_idle_seconds"
        )
        == 3.343912411
        and gate_updates.get("local_compile_shape_decomposition", {}).get("status")
        == "SEALED_READ_ONLY_SHARED_FIXED_EXECUTOR_TARGET"
        and gate_updates.get("local_compile_shape_decomposition", {}).get("local_executor_compile_seconds")
        == 53.547
        and gate_updates.get("local_compile_shape_decomposition", {}).get("runtime_profile_report_sha256")
        == "5ffa9ae565466aa924eccd1a836a3a5af1613df40cf4c584b344ec3dd4494165"
        and gate_updates.get("local_compile_shape_decomposition", {}).get("pool_layout_report_sha256")
        == "aaa931b08e0decbbda8c91004fb7a6dde5e3e65d9d647466bb43247fb4a2946e"
        and _sha256_json(gate_updates) == PERFORMANCE_GATE_UPDATES_SHA256,
        "current performance gate updates changed without an evidence update",
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
        and next_gate.get("typed_warm80_audit_job") == "13256248"
        and next_gate.get("same_binary_causal_job") == "13271166"
        and next_gate.get("late_factorial_jobs") == ["13276891", "13276923", "13277456", "13277457"]
        and next_gate.get("current_blocker")
        == "serial_padded_coarse_scheduling_vs_native_eight_stream_overlap"
        and next_gate.get("runtime_status")
        == "cache_only_retained_for_repeat_scale_gate_chunking_rejected"
        and next_gate.get("production_change_authorized") is False,
        "current causal gate changed",
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
    typed_policy = active_gate["typed_policy"]
    map_gate = active_gate["same_gpu_map_envelope"]
    particle_gate = active_gate["active_particle_envelope"]
    current_runtime = active_gate["runtime"]
    workboard = scorecard["runtime_lane_workboard"]
    runtime_lanes = {row["id"]: row for row in workboard["lanes"]}
    flat_lane = runtime_lanes["call_neutral_flat_row"]
    stable_lane = runtime_lanes["stable_fine_window"]
    cub_lane = runtime_lanes["batched_cub_sort_scan"]
    elementwise_lane = runtime_lanes["batched_posterior_elementwise"]
    same_binary_lane = runtime_lanes["elementwise_same_binary_causal"]
    xhalf_lane = runtime_lanes["xhalf_projection_cap"]
    late_gate = scorecard["late_iteration_factorial_gate"]
    late_arms = {row["id"]: row for row in late_gate["factorial_arms"]}
    late_reference = late_gate["reference_profile"]
    trace = late_gate["trace_decomposition"]
    coarse_gate = scorecard["coarse_multistream_gate"]
    coarse_means = coarse_gate["abba_means"]
    coarse_nsight = coarse_gate["nsight"]
    gate_updates = scorecard["performance_gate_updates"]
    single_lane_gate = gate_updates["single_lane_applicability"]
    atomic_gate = gate_updates["atomic_t29_reduction"]
    atomic_cross = atomic_gate["multistream_crossed_gate"]
    atomic_cross_change = atomic_cross["atomic_multistream_change_percent"]
    atomic_cross_numerical = atomic_cross["numerical_result"]
    direct_xhalf_gate = gate_updates["direct_relion_xhalf"]
    posterior_executor_gate = gate_updates["shared_posterior_executor"]
    remaining_profile = gate_updates["remaining_profile_decomposition"]
    remaining_kernel = remaining_profile["kernel_work"]
    remaining_headroom = remaining_profile["measured_headroom"]
    remaining_coarse = remaining_profile["coarse_topology"]
    compile_profile = gate_updates["local_compile_shape_decomposition"]
    compile_forecast = compile_profile["forecast"]
    compile_big_jit_xhalf_seconds = sum(
        compile_profile["dominant_identities"][name]["seconds"]
        for name in (
            "jit_run_local_bucket_big_jit",
            "jit_relion_vdam_mstep_fused_projector_x_half",
        )
    )
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
        "| Axis | Frozen score | Current read |",
        "|---|---|---|",
        f"| K=1 correctness | **{strict['passed']}/{strict['denominator']}** | Unchanged; accepted cases are "
        f"`{', '.join(progress['accepted_cases'])}`. |",
        f"| Runtime | **{runtime['passed']}/{runtime['denominator']}** | Unchanged; observed suite range "
        f"{min(runtime_ratios):.2f}--{max(runtime_ratios):.2f}x. Promotion requires a large reproducible gain "
        "without instability or quality loss. |",
        "| Performance lanes | non-scoring | Cache-only is retained for repeat/scale; chunking is rejected. The "
        "shared eight-stream coarse scheduler is mathematically accepted and measurably faster, but held below the "
        "runtime target. The 65--128 single-lane specialization is inapplicable to GF46's actual T=29 coarse call; "
        "native-atomic T=29 plus eight per-particle streams passes its one-iteration math/runtime gate, but remains "
        "default-off while long-run no-growth and the coarse-grained batched-lane gate are active. |",
        "| Numerical policy | non-scoring | Require mathematical equivalence plus stable, unbiased, non-growing "
        "repeat-bounded noise. Bitwise identity is not required. Measure discrete changes; rare marginal changes may "
        "be accepted only within control/native repeat variability, with the same basin and no material final-quality "
        "loss. A slight stable-envelope quality change is allowed only for a large runtime gain. |",
        "| EM reuse | shared production primitives | The remaining boundary is execution topology/variability, "
        "not duplicate projector or scorer math. |",
        "| Later gates | separate | K>1 remains unqualified; real data remains unscored. |",
        "",
        "## Current focus",
        "",
        "| Evidence | Result | What it rules out | Explicit next gate |",
        "|---|---|---|---|",
        f"| Same-H100 GF46 it180->181 factorial `{'/'.join(row['job_id'] for row in late_gate['factorial_arms'])}` | "
        f"**CACHE-ONLY RETAINED FOR REPEAT/SCALE; CHUNKING REJECTED.** Cache-only warm wall "
        f"{late_arms['C']['warm_wall_change_percent']:.2f}% and cold wall "
        f"{late_arms['C']['cold_wall_change_percent']:.2f}%, with +{late_arms['C']['warm_hwm_change_gib']:.3f} GiB "
        "HWM; all tracked discrete invariants exact. | Padding reduction alone does not close the gap: chunk-only "
        f"was {late_arms['D']['cold_wall_change_percent']:+.2f}% cold and "
        f"{late_arms['D']['warm_wall_change_percent']:+.2f}% warm. | Repeat cache-only across scales; do not promote "
        "it yet. |",
        f"| Production coarse trace `{late_reference['job_id']}` | **SCHEDULING/OVERLAP IS THE MEASURED GAP.** "
        f"RECOVAR coarse union {trace['recovar_coarse_union_seconds']:.3f} s versus native "
        f"{trace['native_coarse_union_seconds']:.3f} s, despite "
        f"{trace['recovar_milliseconds_per_executed_slot']:.3f} ms per RECOVAR slot versus "
        f"{trace['native_milliseconds_per_particle']:.3f} ms per native particle. | The shared production kernel "
        "is not slower per unit; six serial padded batches lose to native RELION's eight-stream overlap. | "
        f"{trace['next_focus']} |",
        f"| Shared coarse multistream primitive `{coarse_gate['primitive_job_id']}` + crossed ABBA "
        f"`{coarse_gate['late_pair_job_id']}` | **MATH ACCEPTED; PERFORMANCE HOLD.** Warm expectation "
        f"{coarse_means['serial_expectation_seconds']:.3f}->{coarse_means['multistream_expectation_seconds']:.3f} s "
        f"({coarse_means['expectation_change_percent']:.2f}%) and pass 1 "
        f"{coarse_means['serial_pass1_seconds']:.3f}->{coarse_means['multistream_pass1_seconds']:.3f} s "
        f"({coarse_means['pass1_change_percent']:.2f}%); every tracked discrete state is exact and map/BPref deltas "
        "remain at repeat scale. | Scheduling is only part of the gap: the per-particle RECOVAR kernel is "
        f"{coarse_nsight['multistream_kernel_slowdown_percent_vs_native']:.1f}% slower than native and carries "
        f"{coarse_nsight['excess_static_shared_bytes']} excess shared bytes plus "
        f"{coarse_nsight['recovar_registers_per_thread']} versus {coarse_nsight['native_registers_per_thread']} "
        f"registers/thread. | {coarse_gate['next_gate']} |",
        f"| T=29 applicability factorial `{single_lane_gate['late_factorial_job']}` | **NEGATIVE / REDIRECTED.** "
        f"All {single_lane_gate['numerical_result']['winner_euler_translation_pmax_exact']} winners are exact and "
        f"map rel-L2 is at most `{single_lane_gate['numerical_result']['map_relative_l2_max']:.3e}`, but all arms "
        f"ran the generic kernel because the live coarse operand has {single_lane_gate['actual_coarse_translation_count']} "
        "translations. | The earlier 116 count was oversampled/fine, not coarse; active-lanes=1 cannot accelerate "
        f"GF46. | {single_lane_gate['next_gate']} |",
        f"| Native-atomic T=29 x eight streams `{atomic_gate['job_id']}/{atomic_cross['crossed_job']}` | "
        f"**ONE-ITERATION MATH/RUNTIME PASS; DEFAULT OFF.** Warm wall "
        f"{atomic_cross_change['warm_wall']:.2f}%, expectation {atomic_cross_change['expectation']:.2f}%, pass 1 "
        f"{atomic_cross_change['pass1']:.2f}%, GPU union {atomic_cross_change['gpu_kernel_union']:.2f}%, and coarse "
        f"union {atomic_cross_change['coarse_union']:.2f}%. | All {atomic_cross_numerical['star_rows_and_all_columns_exact']} "
        f"STAR rows/columns and discrete metadata are exact; warm cross-map rel-L2 "
        f"`{atomic_cross_numerical['warm_cross_map_relative_l2_max']:.3e}` is below repeat envelope "
        f"`{atomic_cross_numerical['warm_repeat_envelope']:.3e}`. Long-run no-growth is untested and per-image "
        f"launches still inflate coarse work {atomic_cross_change['coarse_kernel_sum']:.2f}%. | "
        f"{atomic_gate['next_gate']} |",
        f"| Same-binary ABBA `{same_binary_lane['job_id']}` | **NUMERICALLY EQUIVALENT; END-TO-END GAIN "
        f"IMMATERIAL**; zero particle-state/schedule escapes; relative-L2 map differences remain ~1e-7 and warm "
        f"speedup is `{same_binary_lane['runtime_result']['warm_speedup']:.4f}x` | All four arms loaded CUDA SHA "
        f"`{same_binary_lane['shared_cuda_library_sha256']}`; different CUDA libraries are not the cause, and the "
        "strict two-control map-diameter flag alone is not a scientific rejection. | "
        f"{same_binary_lane['next_gate']} |",
        "",
        "## Late-iteration same-H100 factorial",
        "",
        f"GF46 iteration 180->181 ran at source `{late_gate['source_head'][:10]}` on "
        f"`{late_gate['hardware']['node']}` / `{late_gate['hardware']['gpu_uuid']}`. This is a one-transition "
        "diagnostic gate only; it cannot change frozen correctness **2/20** or runtime **0/20**, and no production "
        "default is authorized.",
        "",
        "| Arm / job | Cache / radix / chunk | Cold wall | Warm wall | Warm expectation | Warm HWM | Numerical read | Decision |",
        "|---|---|---:|---:|---:|---:|---|---|",
        f"| A / `{late_arms['A']['job_id']}` | off / 4 / 0 | {late_arms['A']['cold_wall_seconds']:.3f} s "
        f"(control) | {late_arms['A']['warm_wall_seconds']:.3f} s (control) | "
        f"{late_arms['A']['warm_expectation_seconds']:.3f} s (control) | "
        f"{late_arms['A']['warm_hwm_gib']:.3f} GiB (control) | discrete exact; control repeat map rel-L2 "
        f"`{late_arms['A']['control_repeat_map_relative_l2']:.6g}` | **CONTROL** |",
        f"| B / `{late_arms['B']['job_id']}` | auto / 2 / 220 | {late_arms['B']['cold_wall_seconds']:.3f} s "
        f"({late_arms['B']['cold_wall_change_percent']:+.2f}%) | {late_arms['B']['warm_wall_seconds']:.3f} s "
        f"({late_arms['B']['warm_wall_change_percent']:+.2f}%) | "
        f"{late_arms['B']['warm_expectation_seconds']:.3f} s "
        f"({late_arms['B']['warm_expectation_change_percent']:+.2f}%) | "
        f"{late_arms['B']['warm_hwm_gib']:.3f} GiB ({late_arms['B']['warm_hwm_change_percent']:+.2f}%) | "
        f"discrete exact; map rel-L2 `{late_arms['B']['map_relative_l2_vs_control']:.6g}` | "
        "**HOLD / REJECT AS DEFAULT** |",
        f"| C / `{late_arms['C']['job_id']}` | auto / 4 / 0 | {late_arms['C']['cold_wall_seconds']:.3f} s "
        f"({late_arms['C']['cold_wall_change_percent']:+.2f}%) | {late_arms['C']['warm_wall_seconds']:.3f} s "
        f"({late_arms['C']['warm_wall_change_percent']:+.2f}%) | "
        f"{late_arms['C']['warm_expectation_seconds']:.3f} s "
        f"({late_arms['C']['warm_expectation_change_percent']:+.2f}%) | "
        f"{late_arms['C']['warm_hwm_gib']:.3f} GiB (+{late_arms['C']['warm_hwm_change_gib']:.3f} GiB) | "
        f"discrete exact; map rel-L2 `{late_arms['C']['map_relative_l2_vs_control']:.6g}`, within repeat envelope | "
        "**RETAIN FOR REPEAT/SCALE; NOT PROMOTED** |",
        f"| D / `{late_arms['D']['job_id']}` | off / 2 / 220 | {late_arms['D']['cold_wall_seconds']:.3f} s "
        f"({late_arms['D']['cold_wall_change_percent']:+.2f}%) | {late_arms['D']['warm_wall_seconds']:.3f} s "
        f"({late_arms['D']['warm_wall_change_percent']:+.2f}%) | "
        f"{late_arms['D']['warm_expectation_seconds']:.3f} s "
        f"({late_arms['D']['warm_expectation_change_percent']:+.2f}%) | "
        f"{late_arms['D']['warm_hwm_gib']:.3f} GiB ({late_arms['D']['warm_hwm_change_percent']:+.2f}%) | "
        f"discrete exact; map rel-L2 `{late_arms['D']['map_relative_l2_vs_control']:.6g}` | **REJECT** |",
        "",
        f"Reference job `{late_reference['job_id']}` measured native RELION at "
        f"{late_reference['native_unprofiled']['process_seconds']:.3f} s process / "
        f"{late_reference['native_unprofiled']['expectation_seconds']:.3f} s expectation, versus RECOVAR "
        f"{late_reference['recovar']['warm_wall_seconds']:.3f} s warm wall / "
        f"{late_reference['recovar']['warm_expectation_seconds']:.3f} s expectation. Nsight resolves the coarse "
        f"path as `{trace['recovar_launches']}` versus `{trace['native_launches']}`. {trace['conclusion']}",
        "",
        "The numerical gate is scientific, not bitwise. Mathematical equivalence and stable, unbiased, non-growing "
        "repeat-bounded noise are mandatory. Discrete changes are measured, not universally forbidden: rare marginal "
        "changes may be accepted only when consistent with control/native repeat variability, remain in the same "
        "basin, and cause no material final-quality loss. A slight quality change inside that stable envelope is "
        "acceptable only when paired with a large runtime gain. This factorial's exact tracked decisions are strong "
        "evidence, not the universal acceptance definition.",
        "",
        "## Performance lanes",
        "",
        "| Bucket | Lane | Evidence | Explicit next gate |",
        "|---|---|---|---|",
        f"| **ACCEPTED PRIMITIVE ONLY** | Flat-row scorer `{'/'.join(flat_lane['jobs'])}` | Active raw/dense "
        f"scores, posterior {flat_lane['exactness']['posterior_outputs_bitwise']}, poisoned tail, and call count are "
        f"bitwise; isolated combined-call reduction "
        f"{', '.join(f'{value:.2f}%' for value in flat_lane['combined_call_reduction_percent'])}. Default-off. | "
        f"{flat_lane['next_gate']} |",
        f"| **ACCEPTED PRIMITIVE ONLY** | Stable fine window `{'/'.join(stable_lane['jobs'])}` | Logical "
        f"{stable_lane['logical_sizes']} under physical {stable_lane['physical_size']} is bitwise; compile identities "
        f"{stable_lane['compile_identities_control']}->{stable_lane['compile_identities_candidate']}; "
        f"{stable_lane['gf46_net_wall_gain_forecast_percent'][0]:.1f}--"
        f"{stable_lane['gf46_net_wall_gain_forecast_percent'][1]:.1f}% is forecast-only. Default-off. | "
        f"{stable_lane['next_gate']} |",
        f"| **REJECTED** | Batched CUB trajectory `{cub_lane['trajectory_ab']['job_id']}` | State escapes "
        "it4/p285, it16/p2902, it18/p902; schedule escape it18; 21 candidate map checkpoints outside, worst "
        f"ratio `{cub_lane['trajectory_ab']['worst_map_escape']['nearest_over_control_diameter']:.10f}`. Cold/warm "
        f"speedups `{cub_lane['trajectory_ab']['runtime']['cold_speedup']:.4f}x`/"
        f"`{cub_lane['trajectory_ab']['runtime']['warm_speedup']:.4f}x` are non-scoring. Raw report SHA-256 "
        f"`{cub_lane['trajectory_ab']['report_sha256']}`. | {cub_lane['next_gate']} |",
        f"| **NUMERICALLY EQUIVALENT / E2E INCONCLUSIVE** | Elementwise primitive "
        f"`{elementwise_lane['primitive_gate']['job_id']}` + trajectory `{elementwise_lane['trajectory_ab']['job_id']}` "
        f"| Primitive is bitwise at it20/40/60/80 and "
        f"{elementwise_lane['primitive_gate']['minimum_speedup']:.4f}--"
        f"{max(elementwise_lane['primitive_gate']['speedups']):.4f}x faster; trajectory has zero state/schedule "
        f"escapes and only roundoff-scale terminal relative-L2 "
        f"`{elementwise_lane['trajectory_ab']['map_escape']['nearest_control_relative_l2']:.3e}`. "
        f"Raw report SHA-256 `{elementwise_lane['trajectory_ab']['report_sha256']}`. | "
        f"{elementwise_lane['next_gate']} |",
        f"| **NUMERICALLY EQUIVALENT / E2E IMMATERIAL** | Same-binary causal `{same_binary_lane['job_id']}` | "
        "Zero state/schedule escapes; roundoff-scale relative-L2 differences at it2/3 with identical CUDA SHA. "
        "Cold/warm speedups "
        f"`{same_binary_lane['runtime_result']['cold_speedup']:.4f}x`/"
        f"`{same_binary_lane['runtime_result']['warm_speedup']:.4f}x`; the warm gain is immaterial. Raw report SHA-256 "
        f"`{same_binary_lane['report_sha256']}`. | {same_binary_lane['next_gate']} |",
        f"| **REJECTED** | 80M x-half `{xhalf_lane['performance_job']}/{xhalf_lane['operand_job']}` | "
        "Iteration-1 topology is identical but 3/3 artifacts differ; causal projection effect was not proved. "
        f"Prior {xhalf_lane['same_h100_wall_reduction_percent']:.2f}% wall reduction is unusable. | "
        f"{xhalf_lane['next_gate']} |",
        f"| **MATH ACCEPTED / PERFORMANCE HOLD** | Shared 8-stream coarse scheduler "
        f"`{coarse_gate['primitive_job_id']}/{coarse_gate['late_pair_job_id']}` | All tracked discrete state is exact; "
        f"warm expectation improves {abs(coarse_means['expectation_change_percent']):.2f}%, but coarse union remains "
        f"{coarse_nsight['multistream_coarse_union_seconds']:.3f} s versus the 3.0 s gate. | "
        f"{coarse_gate['next_gate']} |",
        f"| **REJECTED FOR GF46 / RETAINED PRIMITIVE** | Single-lane coarse "
        f"`{single_lane_gate['focused_requalification_job']}/{single_lane_gate['late_factorial_job']}` | Primitive "
        f"2/2 passes, but GF46 is T={single_lane_gate['actual_coarse_translation_count']}; requested-vs-generic "
        f"coarse union changed only {single_lane_gate['nsight']['single_lane_coarse_union_effect_percent_multistream']:.4f}%. "
        f"No discrete/basin effect. | {single_lane_gate['next_gate']} |",
        f"| **ONE-ITERATION MATH/RUNTIME PASS; DEFAULT OFF** | Native-atomic x eight streams "
        f"`{atomic_gate['job_id']}/{atomic_cross['crossed_job']}` | Warm wall improves "
        f"{abs(atomic_cross_change['warm_wall']):.2f}%, expectation {abs(atomic_cross_change['expectation']):.2f}%, "
        f"and coarse union {abs(atomic_cross_change['coarse_union']):.2f}%; all 3,000 rows/decisions are exact and "
        "maps remain inside non-directional repeat noise. Long-run growth remains untested. | "
        f"{atomic_gate['next_gate']} |",
        f"| **MATH ACCEPTED / PERFORMANCE REJECTED** | Direct RELION x-half BPref "
        f"`{direct_xhalf_gate['qualified_job']}/{direct_xhalf_gate['crossed_live_job']}` | Actual CUDA K=1/K=3 "
        f"primitive outputs are bitwise; crossed GF46 decisions are exact and map/BPref remain in repeat noise. "
        f"Finalize improves {direct_xhalf_gate['performance_result']['finalize_speedup_percent']:.2f}%, but warm "
        f"wall improves only {direct_xhalf_gate['performance_result']['warm_wall_speedup_percent']:.2f}%. "
        f"`{direct_xhalf_gate['invalid_harness_job']}` is invalid collection-only; jobs "
        f"`{'/'.join(row['job_id'] for row in direct_xhalf_gate['invalid_live_jobs'])}` stopped before science. | "
        f"{direct_xhalf_gate['next_gate']} |",
        f"| **MATH ACCEPTED / PERFORMANCE REJECTED** | Shared posterior executor "
        f"`{posterior_executor_gate['qualified_gpu_job']}/{posterior_executor_gate['crossed_live_job']}` | Every "
        f"discrete result is exact and map rel-L2 is at most "
        f"`{posterior_executor_gate['numerical_result']['map_relative_l2_max']:.3e}`; warm wall improves only "
        f"{abs(posterior_executor_gate['crossed_medians']['warm_wall_change_percent']):.2f}% while posterior-kernel "
        f"time regresses {posterior_executor_gate['crossed_medians']['posterior_kernel_time_change_percent']:.2f}%. | "
        f"{posterior_executor_gate['next_gate']} |",
        f"| **SEALED PROFILE / LARGE LEVER IDENTIFIED** | Native-vs-RECOVAR decomposition | RECOVAR summed GPU "
        f"kernel work is only {remaining_kernel['recovar_overhead_percent']:.2f}% above native, but overlap is "
        f"{remaining_kernel['recovar_baseline_overlap_ratio']:.3f} versus {remaining_kernel['native_overlap_ratio']:.3f}; "
        f"measured excess idle is {remaining_headroom['excess_idle_seconds']:.3f} s "
        f"({remaining_headroom['excess_idle_percent_warm_wall']:.2f}% of warm wall). Six serial coarse kernels "
        f"underlap, while {remaining_coarse['recovar_per_image_kernel_count']} per-image launches inflate coarse "
        f"work {remaining_coarse['per_image_work_inflation_vs_serial_percent']:.2f}%. | "
        f"{remaining_profile['next_candidate']} Report SHA-256 `{remaining_profile['report_sha256']}`. |",
        f"| **SEALED COMPILE PROFILE / SHARED FIXED EXECUTOR TARGET** | Iterations 47--80 | "
        f"`local.run_local_em_exact` accounts for {compile_profile['local_executor_compile_seconds']:.3f} of "
        f"{compile_profile['xla_compile_seconds']:.3f} s XLA compile "
        f"({compile_profile['local_share_of_compile_percent']:.1f}%). Big-JIT plus fused x-half alone consume "
        f"{compile_big_jit_xhalf_seconds:.3f} s. "
        f"The compile-only forecast is {compile_forecast['compile_only_saving_percent_full_run']:.1f}% of the "
        f"423 s run, before packed-work savings. | {compile_profile['next_candidate']} Runtime report SHA-256 "
        f"`{compile_profile['runtime_profile_report_sha256']}`. |",
        "",
        "Invalid jobs `13270868` and `13270984` stopped in preflight before science and are not evidence. All listed "
        "lanes are diagnostic and default-off/unwired, with **no impact** on frozen correctness "
        f"{strict['passed']}/{strict['denominator']} or runtime {runtime['passed']}/{runtime['denominator']}. "
        "Forced nondefault native-texture path remains rejected after 120/120 strict artifacts differed.",
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
        f"| `{active_gate['science_job']}` + audit `{active_gate['job_id']}`: "
        f"{active_gate['iterations']}-iteration typed gate | **MIXED: POLICY/MAP PASS; PARTICLE FAIL@"
        f"{particle_gate['first_failure_iteration']}; RUNTIME INCONCLUSIVE** | Typed policy **PASS "
        f"{typed_policy['iterations_per_arm']}/{typed_policy['iterations_per_arm']} in both arms**. "
        f"Same-GPU map **PASS {map_gate['checkpoints_passed']}/{map_gate['checkpoints']}**; active-particle "
        f"**FAIL@{particle_gate['first_failure_iteration']} — OPEN**; runtime **INCONCLUSIVE**. Original profiled "
        "repeat stopped at "
        f"iteration {active_gate['original_profile_repeat']['stopped_after_iteration']} (invalid harness); corrected "
        f"job `{active_gate['corrected_profile_job']}` completed, with direct map/particle FAIL@4, but is cross-GPU "
        "diagnostic only. Frozen scores stay "
        "2/20 correctness and 0/20 runtime. Evidence SHA-256: policy "
        f"`{active_gate['reports']['typed_policy']['sha256']}`, runtime "
        f"`{active_gate['reports']['runtime']['sha256']}`, map "
        f"`{active_gate['reports']['same_gpu_map']['sha256']}`, particle "
        f"`{active_gate['reports']['same_gpu_particle']['sha256']}`. |",
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
            f"### Warm H100 profile: job `{engineering['warm_profile']['job_id']}`",
            "",
            f"Iterations {engineering['warm_profile']['iteration_window']}: "
            f"{engineering['warm_profile']['wall_span_seconds']:.2f} s wall, "
            f"{engineering['warm_profile']['gpu_kernel_seconds']:.2f} s kernels, "
            f"{engineering['warm_profile']['coarse_projector_seconds']:.3f} s coarse projector, "
            f"{engineering['warm_profile']['xla_compile_seconds']:.2f} s XLA compile, and "
            f"{engineering['warm_profile']['dataset_getitem_seconds']:.2f} s dataset getitem. The profile points to "
            "execution topology and shape churn; full artifact paths and hashes remain bound in the JSON ledger.",
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
            "1. Complete the crossed shared eight-stream canonical/atomic gate at T=29. The sealed single-stream ABBA "
            "is mathematically accepted and cuts the hot coarse kernel 8.00%; require stable repeat-bounded unbiased "
            "noise plus a material combined end-to-end gain before any default change.",
            "2. Keep direct x-half default-off: it is mathematically qualified and makes finalization 85.40% faster, "
            "but finalization is too small and warm wall improves only 2.12%. Preserve the primitive for a future "
            "larger fused finalization redesign; do not spend a trajectory on it alone.",
            "3. Keep the shared posterior executor rejected: it is mathematically qualified but saves only 3.37% "
            "warm wall while its posterior kernels regress 36.86%.",
            "4. Implement the sealed profile's large shared lever: approximately eight coarse-grained batched lanes "
            "plus pipelined data/layout and coarse-to-fine preparation. Target the measured 3.344 s excess idle while "
            "avoiding both six serial monolithic kernels and 1000 work-inflating per-image launches.",
            "5. Repeat cache-only arm C across seeds, scales, and representative trajectory checkpoints. Track the "
            "0.365 GiB HWM cost and promote only if the cold/warm gain is reproducible; keep physical-order chunking "
            "and the combined B arm out of the production default.",
            "6. Keep the profiler unset. Wire the qualified flat-row scorer behind an explicit default-off typed "
            "control, reuse shared compact-pair packing/projection, and time the complete live call. Repeat the raw, "
            "dense-score, six-posterior, poisoned-tail, and outer-call exactness audit before any trajectory.",
            "7. Integrate stable physical score, Wavg, and BPref shapes together while retaining the logical cutoff "
            "as the runtime bound. Poison-test padded tails and replace the 5.2--5.4% forecast with a live exact "
            "compile-amortized trajectory measurement; the isolated runtime-bound BPref primitive is numerically "
            "qualified but 14.8--17.7% slower per steady call, so it cannot stand alone.",
            "8. Keep batched CUB and the 80M x-half cap rejected. Keep elementwise default-off and unpromoted: it is "
            "numerically equivalent, but the same-binary 1.0091x warm result is immaterial and the isolated primitive "
            "speedup cannot authorize promotion.",
            "9. Keep the audited typed Wavg/radix defaults. Preserve the active-particle FAIL@"
            f"{particle_gate['first_failure_iteration']} boundary, but defer its arithmetic investigation while "
            "the explicitly requested performance-first phase is active.",
            "10. Do not revive the rejected 128-tail palette, float32 scorer, eager whole-stack raw cache, shared "
            "coarse-projection cache, literal pool-per-call layout, physical-order chunking, or dynamic tail mask "
            "without new evidence. After speed closes, isolate correctness and then expand the frozen K=1 matrix "
            "before K>1 or real-data promotion.",
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
