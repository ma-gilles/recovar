#!/usr/bin/env python3
"""Validate the sealed, rejected K=4 compact-pair threshold campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em.real_kclass_compact_pair_threshold_campaign.v1"
DIAGNOSTIC_ID = "real-k4-10345-native10k-compact-pair-threshold128-default512-campaign-2f6759608-20260903"
LEDGER_FILENAME = f"{DIAGNOSTIC_ID}.json"
CAMPAIGN_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_threshold128_default512_seedpair_42002_42003_"
    "10345_native10k_2f6759608_20260903"
)
SOURCE = {
    "branch": "codex/pr158-10345-native10k-source-2f6759608",
    "commit": "2f6759608c82356dd5c24e7b149002df8cc84f28",
    "tree": "b2e9b35f3988ae63aae45e1dac90c0386099d97f",
    "clean": True,
}
AGGREGATE_SHA256 = "19a341f196c955ba6fdb19ac975caee412c66ccec88cdfab5719858527807c96"
ATTEMPT2_CONTRACT_SHA256 = "a38953a6a8b1d7365d9d790f694fbed80f1fb572714be85a5eb9219f6b413e7f"
ORIGINAL_CONTRACT_SHA256 = "96124778fda722b06a0741e99d433bdfa833ee5806bd75618294abd0ebb69934"
AGGREGATE_MARKDOWN_SHA256 = "6d98231b644796c035c8ac59a8b41ea39d139ed55ac7ff29e0ee4b48a7cf29e0"
SCIENCE_JOB_TRES = "cpu=8,mem=128G,node=1,billing=10,gres/gpu=1"
AUDIT_PAIR_TRES = "cpu=4,mem=64G,node=1,billing=16"
AUDIT_AGGREGATE_TRES = "cpu=1,mem=4G,node=1,billing=1"
PAIR_REPORT_SHA256 = {
    42001: "60684e10414fd83109f4aa807a33c4baeb313338dd8adae2b29b9c3fb6bf45a8",
    42002: "29f2b481b4d428d83168f13548e033a0454d233387946d68ec8aecf878e3ec63",
    42003: "3c8ca12be45a97562b6680a8ebfc678286104d7893942bece42d131ea1bc5992",
}
PAIR_JOB_IDS = {
    42001: {"baseline": 13377738, "candidate": 13377631},
    42002: {"baseline": 13378734, "candidate": 13378735},
    42003: {"baseline": 13378736, "candidate": 13378737},
}
EXPECTED_PAIR_DECISIONS = {
    42001: ("reject", "accept", "retrospective"),
    42002: ("reject", "reject", "prospective"),
    42003: ("reject", "accept", "prospective"),
}
FAILED_ATTEMPT_JOBS = {
    13378526: (42002, "default512", "00:01:48", "della-h19g1"),
    13378527: (42002, "threshold128", "00:01:48", "della-h19g3"),
    13378528: (42003, "default512", "00:01:48", "della-h19g3"),
    13378529: (42003, "threshold128", "00:01:35", "della-h19g4"),
}
AUDIT_JOBS = {
    13378859: ("seed42001", "00:00:49", "della-r3c3n3", AUDIT_PAIR_TRES),
    13378860: ("seed42002", "00:00:24", "della-r3c2n14", AUDIT_PAIR_TRES),
    13378861: ("seed42003", "00:00:35", "della-r3c2n16", AUDIT_PAIR_TRES),
    13378887: ("aggregate", "00:00:04", "della-r3c4n16", AUDIT_AGGREGATE_TRES),
}
FORMAL_THRESHOLDS = {
    "all_class_assignments_exact": True,
    "all_controller_arrays_exact": True,
    "final_map_max_relative_l2": 1.02e-5,
    "final_map_min_signed_non_dc_fsc_auc": 0.99999999,
    "max_sampled_hbm_ratio": 1.05,
    "max_sparse_group_wall_ratio": 0.95,
}
SCIENCE_THRESHOLDS = {
    "all_class_assignments_exact": True,
    "all_controller_arrays_exact": True,
    "all_pose_arrays_exact": True,
    "final_map_max_relative_l2": 1.0e-4,
    "final_map_min_signed_non_dc_fsc_auc": 0.999999,
    "max_sampled_hbm_ratio": 1.05,
    "max_sparse_group_wall_ratio": 0.95,
    "no_class_collapse": ("all four class labels must be occupied in both arms at every saved iteration"),
}
EXPECTED_PHYSICAL_ROTATION_DEG = 47.374945112723495
EXPECTED_CANONICAL_SHA256 = "73b01f961272fea990736c6966df69834b4a825c04a772391676be77f0faa874"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-f-]+$")
ELAPSED_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


class CampaignValidationError(ValueError):
    """Raised when the campaign ledger could support an unsealed claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignValidationError(message)


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing and not extra, f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def _expect(value: Any, expected: Any, label: str) -> None:
    _require(
        type(value) is type(expected) and value == expected,
        f"{label} must be {expected!r}, got {value!r}",
    )


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    numeric = float(value)
    _require(math.isfinite(numeric), f"{label} must be finite")
    if minimum is not None:
        _require(numeric >= minimum, f"{label} must be at least {minimum}")
    return numeric


def _absolute_path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and Path(value).is_absolute(), f"{label} must be absolute")
    return Path(value)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _file_ref(value: Any, label: str, *, expected_sha256: str | None = None) -> None:
    reference = _exact_keys(value, {"path", "sha256"}, label)
    _absolute_path(reference["path"], f"{label}.path")
    digest = _sha256(reference["sha256"], f"{label}.sha256")
    if expected_sha256 is not None:
        _expect(digest, expected_sha256, f"{label}.sha256")


def _canonical_sha256(data: dict[str, Any]) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_workload(value: Any) -> None:
    workload = _exact_keys(
        value,
        {
            "dataset",
            "K",
            "particles_total",
            "particles_compared_half1",
            "native_grid_size",
            "full_iterations",
            "half",
            "seeds",
            "initial_reference_voxel_arrays_equal_across_seeds",
            "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT",
            "exclusive_requested",
        },
        "workload",
    )
    expected = {
        "dataset": "EMPIAR-10345",
        "K": 4,
        "particles_total": 10_000,
        "particles_compared_half1": 5_000,
        "native_grid_size": 256,
        "full_iterations": 8,
        "half": 1,
        "seeds": [42001, 42002, 42003],
        "initial_reference_voxel_arrays_equal_across_seeds": True,
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT": "unset",
        "exclusive_requested": False,
    }
    _expect(workload, expected, "workload")


def _validate_treatment(value: Any) -> None:
    treatment = _exact_keys(
        value,
        {
            "environment_variable",
            "source_default_constant",
            "source_path",
            "baseline",
            "candidate",
            "production_default_min_bucket_size",
            "production_default_changed",
            "disposition",
        },
        "treatment",
    )
    _expect(
        treatment["environment_variable"],
        "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE",
        "treatment.environment_variable",
    )
    _expect(
        treatment["source_default_constant"],
        "_DEFAULT_COMPACT_PAIR_MIN_BUCKET_SIZE",
        "treatment.source_default_constant",
    )
    _expect(
        treatment["source_path"],
        "recovar/em/dense_single_volume/helpers/sparse_pass2_bucketed.py",
        "treatment.source_path",
    )
    baseline = _exact_keys(
        treatment["baseline"],
        {"label", "environment_value", "resolved_min_bucket_size"},
        "treatment.baseline",
    )
    candidate = _exact_keys(
        treatment["candidate"],
        {"label", "environment_value", "resolved_min_bucket_size"},
        "treatment.candidate",
    )
    _expect(
        baseline,
        {"label": "default512", "environment_value": None, "resolved_min_bucket_size": 512},
        "treatment.baseline",
    )
    _expect(
        candidate,
        {"label": "threshold128", "environment_value": "128", "resolved_min_bucket_size": 128},
        "treatment.candidate",
    )
    _expect(treatment["production_default_min_bucket_size"], 512, "production default")
    _expect(treatment["production_default_changed"], False, "production_default_changed")
    _expect(treatment["disposition"], "retain_default_512", "treatment.disposition")


def _validate_sealed_evidence(value: Any) -> None:
    evidence = _exact_keys(
        value,
        {"campaign_root", "aggregate", "aggregate_markdown", "attempt2_contract", "original_contract"},
        "sealed_evidence",
    )
    _expect(evidence["campaign_root"], str(CAMPAIGN_ROOT), "sealed_evidence.campaign_root")
    aggregate = _exact_keys(evidence["aggregate"], {"path", "sha256", "schema"}, "sealed_evidence.aggregate")
    _expect(
        aggregate["path"],
        str(CAMPAIGN_ROOT / "analysis/attempt2_pair_audits/aggregate.json"),
        "sealed_evidence.aggregate.path",
    )
    _expect(aggregate["sha256"], AGGREGATE_SHA256, "sealed_evidence.aggregate.sha256")
    _expect(
        aggregate["schema"],
        "recovar.em.kclass_default_threshold_pair_aggregate.v1",
        "sealed_evidence.aggregate.schema",
    )
    references = {
        "aggregate_markdown": (
            CAMPAIGN_ROOT / "analysis/attempt2_pair_audits/aggregate.md",
            AGGREGATE_MARKDOWN_SHA256,
        ),
        "attempt2_contract": (
            CAMPAIGN_ROOT / "analysis/predeclared_two_level_acceptance_attempt2.json",
            ATTEMPT2_CONTRACT_SHA256,
        ),
        "original_contract": (
            CAMPAIGN_ROOT / "analysis/predeclared_two_level_acceptance.json",
            ORIGINAL_CONTRACT_SHA256,
        ),
    }
    for name, (path, digest) in references.items():
        _file_ref(evidence[name], f"sealed_evidence.{name}", expected_sha256=digest)
        _expect(evidence[name]["path"], str(path), f"sealed_evidence.{name}.path")


def _validate_decision_contract(value: Any) -> None:
    contract = _exact_keys(
        value,
        {
            "aggregation_rule",
            "tiers_are_independent",
            "scientific_equivalence_cannot_rewrite_formal",
            "formal_thresholds",
            "scientific_equivalence_thresholds",
        },
        "decision_contract",
    )
    _expect(contract["aggregation_rule"], "all_pair_conjunction_no_averaging", "aggregation rule")
    _expect(contract["tiers_are_independent"], True, "tiers_are_independent")
    _expect(
        contract["scientific_equivalence_cannot_rewrite_formal"],
        True,
        "scientific_equivalence_cannot_rewrite_formal",
    )
    _expect(contract["formal_thresholds"], FORMAL_THRESHOLDS, "formal thresholds")
    _expect(
        contract["scientific_equivalence_thresholds"],
        SCIENCE_THRESHOLDS,
        "scientific-equivalence thresholds",
    )


def _validate_job(value: Any, label: str, *, expected_job_id: int) -> None:
    job = _exact_keys(
        value,
        {
            "job_id",
            "state",
            "exit_code",
            "elapsed",
            "node",
            "gpu",
            "gpu_uuid",
            "req_tres",
            "alloc_tres",
            "nonexclusive",
            "root",
            "gpu_monitor",
            "walltime",
        },
        label,
    )
    _expect(job["job_id"], expected_job_id, f"{label}.job_id")
    _expect(job["state"], "COMPLETED", f"{label}.state")
    _expect(job["exit_code"], "0:0", f"{label}.exit_code")
    _require(
        isinstance(job["elapsed"], str) and ELAPSED_RE.fullmatch(job["elapsed"]) is not None,
        f"{label}.elapsed must be HH:MM:SS",
    )
    _require(
        isinstance(job["node"], str) and job["node"].startswith("della-h19g"),
        f"{label}.node must identify the recorded H100 node",
    )
    _expect(job["gpu"], "NVIDIA H100 80GB HBM3", f"{label}.gpu")
    _require(
        isinstance(job["gpu_uuid"], str) and GPU_UUID_RE.fullmatch(job["gpu_uuid"]) is not None,
        f"{label}.gpu_uuid must be recorded",
    )
    _expect(job["req_tres"], SCIENCE_JOB_TRES, f"{label}.req_tres")
    _expect(job["alloc_tres"], SCIENCE_JOB_TRES, f"{label}.alloc_tres")
    _expect(job["nonexclusive"], True, f"{label}.nonexclusive")
    root = _absolute_path(job["root"], f"{label}.root")
    monitor = _absolute_path(job["gpu_monitor"], f"{label}.gpu_monitor")
    walltime = _absolute_path(job["walltime"], f"{label}.walltime")
    _require(monitor.is_relative_to(root), f"{label}.gpu_monitor must be inside the run root")
    _require(walltime.is_relative_to(root), f"{label}.walltime must be inside the run root")


def _validate_summary(value: Any, label: str) -> dict[str, Any]:
    summary = _exact_keys(
        value,
        {
            "assignment_mismatch_count",
            "controller_exact",
            "poses_exact",
            "max_euler_absolute_delta",
            "max_translation_absolute_delta",
            "max_pmax_absolute_delta",
            "significant_count_mismatch_count",
            "no_class_collapse",
            "min_final_map_signed_non_dc_fsc_auc",
            "max_final_map_relative_l2",
        },
        label,
    )
    for key in ("assignment_mismatch_count", "significant_count_mismatch_count"):
        _require(
            isinstance(summary[key], int) and not isinstance(summary[key], bool) and summary[key] >= 0,
            f"{label}.{key} must be a nonnegative integer",
        )
    for key in ("controller_exact", "poses_exact", "no_class_collapse"):
        _require(type(summary[key]) is bool, f"{label}.{key} must be Boolean")
    for key in (
        "max_euler_absolute_delta",
        "max_translation_absolute_delta",
        "max_pmax_absolute_delta",
        "min_final_map_signed_non_dc_fsc_auc",
        "max_final_map_relative_l2",
    ):
        _number(summary[key], f"{label}.{key}", minimum=0.0)
    return summary


def _validate_performance(value: Any, label: str) -> dict[str, Any]:
    performance = _exact_keys(
        value,
        {
            "baseline_peak_sampled_hbm_mib",
            "candidate_peak_sampled_hbm_mib",
            "sampled_hbm_ratio",
            "baseline_sparse_group_wall_sum_s",
            "candidate_sparse_group_wall_sum_s",
            "sparse_group_wall_ratio",
            "baseline_external_wall_s",
            "candidate_external_wall_s",
            "external_wall_ratio_descriptive",
        },
        label,
    )
    for key, raw in performance.items():
        _number(raw, f"{label}.{key}", minimum=0.0)
    ratios = {
        "sampled_hbm_ratio": (
            performance["candidate_peak_sampled_hbm_mib"] / performance["baseline_peak_sampled_hbm_mib"]
        ),
        "sparse_group_wall_ratio": (
            performance["candidate_sparse_group_wall_sum_s"] / performance["baseline_sparse_group_wall_sum_s"]
        ),
        "external_wall_ratio_descriptive": (
            performance["candidate_external_wall_s"] / performance["baseline_external_wall_s"]
        ),
    }
    for key, recomputed in ratios.items():
        _require(
            math.isclose(performance[key], recomputed, rel_tol=1e-15, abs_tol=1e-15),
            f"{label}.{key} does not match its raw metrics",
        )
    return performance


def _formal_decision(summary: dict[str, Any], performance: dict[str, Any]) -> str:
    gates = (
        summary["assignment_mismatch_count"] == 0,
        summary["controller_exact"] is True,
        summary["max_final_map_relative_l2"] <= FORMAL_THRESHOLDS["final_map_max_relative_l2"],
        summary["min_final_map_signed_non_dc_fsc_auc"] >= FORMAL_THRESHOLDS["final_map_min_signed_non_dc_fsc_auc"],
        performance["sampled_hbm_ratio"] <= FORMAL_THRESHOLDS["max_sampled_hbm_ratio"],
        performance["sparse_group_wall_ratio"] <= FORMAL_THRESHOLDS["max_sparse_group_wall_ratio"],
    )
    return "accept" if all(gates) else "reject"


def _science_decision(summary: dict[str, Any], performance: dict[str, Any]) -> str:
    gates = (
        summary["assignment_mismatch_count"] == 0,
        summary["controller_exact"] is True,
        summary["poses_exact"] is True,
        summary["max_final_map_relative_l2"] <= SCIENCE_THRESHOLDS["final_map_max_relative_l2"],
        summary["min_final_map_signed_non_dc_fsc_auc"] >= SCIENCE_THRESHOLDS["final_map_min_signed_non_dc_fsc_auc"],
        performance["sampled_hbm_ratio"] <= SCIENCE_THRESHOLDS["max_sampled_hbm_ratio"],
        performance["sparse_group_wall_ratio"] <= SCIENCE_THRESHOLDS["max_sparse_group_wall_ratio"],
        summary["no_class_collapse"] is True,
    )
    return "accept" if all(gates) else "reject"


def _validate_artifact_hashes(value: Any, label: str) -> None:
    hashes = _exact_keys(
        value,
        {
            "baseline_gpu_monitor",
            "baseline_log",
            "baseline_results",
            "candidate_gpu_monitor",
            "candidate_log",
            "candidate_results",
        },
        label,
    )
    for name, digest in hashes.items():
        _sha256(digest, f"{label}.{name}")


def _validate_pair(value: Any, index: int) -> tuple[str, str]:
    label = f"pairs[{index}]"
    pair = _exact_keys(
        value,
        {
            "seed",
            "label",
            "formal_contract_timing",
            "scientific_equivalence_contract_timing",
            "formal_decision",
            "scientific_equivalence_decision",
            "pair_report",
            "jobs",
            "summary",
            "performance",
            "artifact_sha256",
        },
        label,
    )
    seed = pair["seed"]
    _require(seed in EXPECTED_PAIR_DECISIONS, f"{label}.seed is not sealed")
    _expect(pair["label"], f"seed{seed}", f"{label}.label")
    expected_formal, expected_science, expected_science_timing = EXPECTED_PAIR_DECISIONS[seed]
    _expect(pair["formal_contract_timing"], "prospective", f"{label}.formal_contract_timing")
    _expect(
        pair["scientific_equivalence_contract_timing"],
        expected_science_timing,
        f"{label}.scientific_equivalence_contract_timing",
    )
    report = pair["pair_report"]
    _file_ref(report, f"{label}.pair_report", expected_sha256=PAIR_REPORT_SHA256[seed])
    _expect(
        report["path"],
        str(CAMPAIGN_ROOT / f"analysis/attempt2_pair_audits/seed{seed}.json"),
        f"{label}.pair_report.path",
    )
    jobs = _exact_keys(pair["jobs"], {"baseline", "candidate"}, f"{label}.jobs")
    for arm in ("baseline", "candidate"):
        _validate_job(
            jobs[arm],
            f"{label}.jobs.{arm}",
            expected_job_id=PAIR_JOB_IDS[seed][arm],
        )
    summary = _validate_summary(pair["summary"], f"{label}.summary")
    performance = _validate_performance(pair["performance"], f"{label}.performance")
    _validate_artifact_hashes(pair["artifact_sha256"], f"{label}.artifact_sha256")
    recomputed_formal = _formal_decision(summary, performance)
    recomputed_science = _science_decision(summary, performance)
    _expect(pair["formal_decision"], recomputed_formal, f"{label}.formal_decision")
    _expect(
        pair["scientific_equivalence_decision"],
        recomputed_science,
        f"{label}.scientific_equivalence_decision",
    )
    _expect(pair["formal_decision"], expected_formal, f"{label} sealed formal decision")
    _expect(
        pair["scientific_equivalence_decision"],
        expected_science,
        f"{label} sealed scientific-equivalence decision",
    )
    return recomputed_formal, recomputed_science


def _relion_euler_matrix(eulers_deg: list[float]) -> tuple[tuple[float, ...], ...]:
    alpha, beta, gamma = (math.radians(value) for value in eulers_deg)
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return (
        (cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb),
        (-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb),
        (sc, ss, cb),
    )


def _relative_rotation_deg(baseline_eulers: list[float], candidate_eulers: list[float]) -> float:
    baseline = _relion_euler_matrix(baseline_eulers)
    candidate = _relion_euler_matrix(candidate_eulers)
    trace = sum(candidate[row][column] * baseline[row][column] for row in range(3) for column in range(3))
    cosine = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _validate_physical_pose(value: Any, seed42002: dict[str, Any]) -> None:
    pose = _exact_keys(
        value,
        {
            "metric_provenance",
            "seed",
            "iteration",
            "differing_pose_row_count",
            "row_zero_based",
            "baseline_euler_deg",
            "candidate_euler_deg",
            "euler_component_max_abs_deg",
            "physical_relative_rotation_deg",
            "genuinely_distinct_physical_pose",
            "assignments_exact_all_iterations",
            "controller_exact",
            "translations_exact_all_iterations",
            "maps_pass_scientific_fsc_and_l2_gates",
            "classification",
        },
        "physical_pose_discriminator",
    )
    provenance = _exact_keys(
        pose["metric_provenance"],
        {
            "aggregate_field_present",
            "pair_report_field_present",
            "raw_euler_row_is_immutable_source",
            "derivation_status",
            "conversion",
            "baseline_refinement_results",
            "candidate_refinement_results",
        },
        "physical_pose_discriminator.metric_provenance",
    )
    _expect(provenance["aggregate_field_present"], False, "physical metric aggregate provenance")
    _expect(provenance["pair_report_field_present"], False, "physical metric pair-report provenance")
    _expect(
        provenance["raw_euler_row_is_immutable_source"],
        True,
        "physical metric raw-row provenance",
    )
    _expect(
        provenance["derivation_status"],
        "checked_derived_diagnostic",
        "physical metric derivation status",
    )
    _expect(
        provenance["conversion"],
        "RELION rot/tilt/psi Euler matrices; degrees(acos(clamp((trace(R_candidate @ R_baseline.T) - 1) / 2, -1, 1)))",
        "physical metric conversion",
    )
    hashes = seed42002["artifact_sha256"]
    roots = seed42002["jobs"]
    for arm in ("baseline", "candidate"):
        key = f"{arm}_refinement_results"
        reference = provenance[key]
        _file_ref(reference, f"physical_pose_discriminator.metric_provenance.{key}")
        expected_path = Path(roots[arm]["root"]) / "half1/recovar/refinement_results.npz"
        _expect(reference["path"], str(expected_path), f"{key}.path")
        _expect(reference["sha256"], hashes[f"{arm}_results"], f"{key}.sha256")
    _expect(pose["seed"], 42002, "physical_pose_discriminator.seed")
    _expect(pose["iteration"], 8, "physical_pose_discriminator.iteration")
    _expect(
        pose["differing_pose_row_count"],
        1,
        "physical_pose_discriminator.differing_pose_row_count",
    )
    _expect(pose["row_zero_based"], 3280, "physical_pose_discriminator.row_zero_based")
    baseline_eulers = pose["baseline_euler_deg"]
    candidate_eulers = pose["candidate_euler_deg"]
    _expect(
        baseline_eulers,
        [-142.99462890625, 150.5906524658203, 2.571798324584961],
        "physical_pose_discriminator.baseline_euler_deg",
    )
    _expect(
        candidate_eulers,
        [13.690024375915527, 161.02734375, 157.18435668945312],
        "physical_pose_discriminator.candidate_euler_deg",
    )
    component_delta = max(
        abs(candidate - baseline) for baseline, candidate in zip(baseline_eulers, candidate_eulers, strict=True)
    )
    _require(
        math.isclose(pose["euler_component_max_abs_deg"], component_delta, rel_tol=0.0, abs_tol=1e-12),
        "physical_pose_discriminator Euler component delta is not derived from the raw row",
    )
    relative_rotation = _relative_rotation_deg(baseline_eulers, candidate_eulers)
    _require(
        math.isclose(
            pose["physical_relative_rotation_deg"],
            relative_rotation,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "physical relative rotation is not the checked SO(3) derivation from the raw Euler row",
    )
    _require(
        math.isclose(relative_rotation, EXPECTED_PHYSICAL_ROTATION_DEG, rel_tol=0.0, abs_tol=1e-12),
        "physical relative rotation does not match the sealed 47.374945-degree discriminator",
    )
    _expect(pose["genuinely_distinct_physical_pose"], True, "genuinely_distinct_physical_pose")
    _require(relative_rotation > 1.0, "the physical pose discriminator must remain nontrivial")
    _expect(pose["assignments_exact_all_iterations"], True, "physical assignments exact")
    _expect(pose["controller_exact"], True, "physical controller exact")
    _expect(pose["translations_exact_all_iterations"], True, "physical translations exact")
    _expect(pose["maps_pass_scientific_fsc_and_l2_gates"], True, "physical map gates")
    _expect(
        pose["classification"],
        "single_physical_pose_difference_rejects_scientific_equivalence",
        "physical discriminator classification",
    )
    summary = seed42002["summary"]
    _expect(summary["assignment_mismatch_count"], 0, "seed42002 assignment mismatch count")
    _expect(summary["controller_exact"], True, "seed42002 controller exact")
    _expect(summary["poses_exact"], False, "seed42002 poses exact")
    _expect(summary["max_translation_absolute_delta"], 0.0, "seed42002 translation delta")
    _require(
        math.isclose(summary["max_euler_absolute_delta"], component_delta, rel_tol=0.0, abs_tol=1e-12),
        "seed42002 summary Euler delta conflicts with the immutable row",
    )
    maps_pass = (
        summary["min_final_map_signed_non_dc_fsc_auc"] >= SCIENCE_THRESHOLDS["final_map_min_signed_non_dc_fsc_auc"]
        and summary["max_final_map_relative_l2"] <= SCIENCE_THRESHOLDS["final_map_max_relative_l2"]
    )
    _expect(maps_pass, True, "seed42002 scientific map gates")


def _validate_failed_attempt(value: Any) -> None:
    failed = _exact_keys(
        value,
        {"accepted_evidence", "root", "failure_stage", "reason", "jobs"},
        "failed_first_attempt",
    )
    _expect(failed["accepted_evidence"], False, "failed_first_attempt.accepted_evidence")
    _expect(failed["root"], str(CAMPAIGN_ROOT / "runs"), "failed_first_attempt.root")
    _expect(failed["failure_stage"], "before_iteration_1", "failed_first_attempt.failure_stage")
    _expect(
        failed["reason"],
        "seed42001 dispatch_schedule.npz encoded random_seed=42001 and rejected refinement seeds 42002/42003",
        "failed_first_attempt.reason",
    )
    jobs = failed["jobs"]
    _require(isinstance(jobs, list) and len(jobs) == 4, "failed_first_attempt.jobs must have four rows")
    _expect({job.get("job_id") for job in jobs}, set(FAILED_ATTEMPT_JOBS), "failed job identities")
    for index, job in enumerate(jobs):
        label = f"failed_first_attempt.jobs[{index}]"
        row = _exact_keys(
            job,
            {
                "job_id",
                "seed",
                "arm",
                "state",
                "exit_code",
                "elapsed",
                "node",
                "req_tres",
                "alloc_tres",
                "completed_science_iterations",
            },
            label,
        )
        seed, arm, elapsed, node = FAILED_ATTEMPT_JOBS[row["job_id"]]
        _expect(row["seed"], seed, f"{label}.seed")
        _expect(row["arm"], arm, f"{label}.arm")
        _expect(row["state"], "FAILED", f"{label}.state")
        _expect(row["exit_code"], "1:0", f"{label}.exit_code")
        _expect(row["elapsed"], elapsed, f"{label}.elapsed")
        _expect(row["node"], node, f"{label}.node")
        _expect(row["req_tres"], SCIENCE_JOB_TRES, f"{label}.req_tres")
        _expect(row["alloc_tres"], SCIENCE_JOB_TRES, f"{label}.alloc_tres")
        _expect(row["completed_science_iterations"], 0, f"{label}.completed_science_iterations")


def _validate_audit_jobs(value: Any) -> None:
    _require(isinstance(value, list) and len(value) == 4, "audit_jobs must contain four rows")
    _expect({job.get("job_id") for job in value}, set(AUDIT_JOBS), "audit job identities")
    for index, job in enumerate(value):
        label = f"audit_jobs[{index}]"
        row = _exact_keys(
            job,
            {
                "job_id",
                "scope",
                "state",
                "exit_code",
                "elapsed",
                "node",
                "req_tres",
                "alloc_tres",
                "nonexclusive",
                "compute_failure",
                "classification",
            },
            label,
        )
        scope, elapsed, node, tres = AUDIT_JOBS[row["job_id"]]
        _expect(row["scope"], scope, f"{label}.scope")
        _expect(row["state"], "FAILED", f"{label}.state")
        _expect(row["exit_code"], "3:0", f"{label}.exit_code")
        _expect(row["elapsed"], elapsed, f"{label}.elapsed")
        _expect(row["node"], node, f"{label}.node")
        _expect(row["req_tres"], tres, f"{label}.req_tres")
        _expect(row["alloc_tres"], tres, f"{label}.alloc_tres")
        _expect(row["nonexclusive"], True, f"{label}.nonexclusive")
        _expect(row["compute_failure"], False, f"{label}.compute_failure")
        classification = (
            "expected_aggregate_rejection_exit" if scope == "aggregate" else "expected_formal_rejection_exit"
        )
        _expect(row["classification"], classification, f"{label}.classification")


def _validate_aggregate(value: Any, formal: list[str], science: list[str]) -> None:
    aggregate = _exact_keys(
        value,
        {
            "pair_count",
            "formal",
            "scientific_equivalence",
            "aggregation_rule",
            "production_disposition",
        },
        "aggregate",
    )
    _expect(aggregate["pair_count"], 3, "aggregate.pair_count")
    _expect(aggregate["aggregation_rule"], "all_pair_conjunction_no_averaging", "aggregate rule")
    _expect(aggregate["production_disposition"], "retain_default_512", "aggregate disposition")
    for tier_name, decisions in (("formal", formal), ("scientific_equivalence", science)):
        tier = _exact_keys(
            aggregate[tier_name],
            {"accept_pair_count", "reject_pair_count", "decision"},
            f"aggregate.{tier_name}",
        )
        accepts = decisions.count("accept")
        rejects = decisions.count("reject")
        decision = "accept" if all(item == "accept" for item in decisions) else "reject"
        _expect(tier["accept_pair_count"], accepts, f"aggregate.{tier_name}.accept_pair_count")
        _expect(tier["reject_pair_count"], rejects, f"aggregate.{tier_name}.reject_pair_count")
        _expect(tier["decision"], decision, f"aggregate.{tier_name}.decision")
    _expect(
        aggregate["formal"], {"accept_pair_count": 0, "reject_pair_count": 3, "decision": "reject"}, "formal aggregate"
    )
    _expect(
        aggregate["scientific_equivalence"],
        {"accept_pair_count": 2, "reject_pair_count": 1, "decision": "reject"},
        "scientific-equivalence aggregate",
    )


def validate_campaign(data: dict[str, Any]) -> None:
    """Validate campaign semantics and the exact checked-in evidence seal."""

    ledger = _exact_keys(
        data,
        {
            "schema",
            "diagnostic_id",
            "recorded_at",
            "admission_status",
            "accepted_result",
            "status",
            "workload",
            "source",
            "treatment",
            "sealed_evidence",
            "decision_contract",
            "aggregate",
            "pairs",
            "physical_pose_discriminator",
            "failed_first_attempt",
            "audit_jobs",
            "strict_claim_boundary",
        },
        "ledger",
    )
    _expect(ledger["schema"], SCHEMA, "schema")
    _expect(ledger["diagnostic_id"], DIAGNOSTIC_ID, "diagnostic_id")
    _expect(ledger["recorded_at"], "2026-09-03T08:02:43.978073Z", "recorded_at")
    _expect(ledger["admission_status"], "DIAGNOSTIC_ONLY_REJECTED", "admission_status")
    _expect(ledger["accepted_result"], False, "accepted_result")
    _expect(
        ledger["status"],
        "complete_formal_and_scientific_aggregate_rejected",
        "status",
    )
    _validate_workload(ledger["workload"])
    _expect(ledger["source"], SOURCE, "source")
    _validate_treatment(ledger["treatment"])
    _validate_sealed_evidence(ledger["sealed_evidence"])
    _validate_decision_contract(ledger["decision_contract"])
    pairs = ledger["pairs"]
    _require(isinstance(pairs, list) and len(pairs) == 3, "pairs must contain three seeds")
    _expect([pair.get("seed") for pair in pairs], [42001, 42002, 42003], "pair order and seeds")
    decisions = [_validate_pair(pair, index) for index, pair in enumerate(pairs)]
    formal = [item[0] for item in decisions]
    science = [item[1] for item in decisions]
    _validate_aggregate(ledger["aggregate"], formal, science)
    _validate_physical_pose(ledger["physical_pose_discriminator"], pairs[1])
    _validate_failed_attempt(ledger["failed_first_attempt"])
    _validate_audit_jobs(ledger["audit_jobs"])
    _expect(
        ledger["strict_claim_boundary"],
        "Threshold 128 remains diagnostic-only and must not replace production default 512. "
        "Scientific equivalence cannot rewrite formal rejection, and no averaging or close-map "
        "summary may hide seed42002's distinct physical pose.",
        "strict_claim_boundary",
    )
    digest = _canonical_sha256(ledger)
    _expect(digest, EXPECTED_CANONICAL_SHA256, "canonical campaign SHA-256")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(data: dict[str, Any]) -> None:
    """Verify source reports and hashed run artifacts on shared scratch."""

    references: list[tuple[Path, str]] = []
    evidence = data["sealed_evidence"]
    for name in ("aggregate", "aggregate_markdown", "attempt2_contract", "original_contract"):
        references.append((Path(evidence[name]["path"]), evidence[name]["sha256"]))
    for pair in data["pairs"]:
        references.append((Path(pair["pair_report"]["path"]), pair["pair_report"]["sha256"]))
        hashes = pair["artifact_sha256"]
        for arm in ("baseline", "candidate"):
            job = pair["jobs"][arm]
            root = Path(job["root"])
            references.extend(
                [
                    (Path(job["gpu_monitor"]), hashes[f"{arm}_gpu_monitor"]),
                    (root / "half1/recovar/run.log", hashes[f"{arm}_log"]),
                    (root / "half1/recovar/refinement_results.npz", hashes[f"{arm}_results"]),
                ]
            )
            walltime = Path(job["walltime"])
            _require(walltime.is_file(), f"missing recorded walltime file: {walltime}")
    cached: dict[Path, str] = {}
    for path, expected in references:
        _require(path.is_file(), f"missing sealed file: {path}")
        if path not in cached:
            cached[path] = _hash_file(path)
        _require(cached[path] == expected, f"checksum mismatch: {path}")


def default_ledger_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "em" / "diagnostics" / LEDGER_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=default_ledger_path())
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.ledger.read_text())
        validate_campaign(data)
        if args.verify_files:
            verify_files(data)
    except (OSError, json.JSONDecodeError, CampaignValidationError) as error:
        parser.exit(1, f"K=4 compact-pair threshold campaign validation failed:\n{error}\n")
    print(f"Validated rejected K=4 threshold-128/default-512 campaign ({len(data['pairs'])} pairs): {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
