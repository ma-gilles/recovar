#!/usr/bin/env python3
"""Validate the fixed K=1 real-data science-equivalence scorecard.

This reporter consumes the signed FSC curves emitted by the external
RECOVAR--RELION real-data collector.  Comparable independent half-map quality
is mandatory.  Cross-engine equivalence may be established either directly in
the canonical frame or by one separately pinned continuous proper-SO(3) rigid
transform fitted at low frequency and applied unchanged to both half maps.
Reflections, sign changes, and scale fitting are forbidden.  A common-mask FSC
is reported only as supporting evidence and can never rescue a failure.

See docs/math/em_k1_realdata_science_equivalence_scorecard.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k1_realdata_science_equivalence_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k1_realdata_science_equivalence_scorecard.md"
DIAGNOSTIC_PRODUCER_RELATIVE_PATH = Path("scripts/collect_em_k1_science_diagnostics.py")

SCORECARD_SCHEMA = "recovar.em_k1_realdata_science_equivalence_scorecard.v1"
REPORT_SCHEMA = "recovar.em_k1_realdata_science_equivalence_report.v1"
EVIDENCE_SCHEMA = "recovar.em_k1_realdata_science_equivalence_case_evidence.v1"
COLLECTOR_SCHEMA = "recovar-relion-realdata-metrics-v2"
EXECUTION_BINDING_SCHEMA = "recovar.em_k1_execution_binding.v1"
LAUNCH_MANIFEST_SCHEMA = "recovar.empiar10202_set6_i1_matched_launch.v1"
NATIVE_LAUNCH_MANIFEST_SCHEMA = "recovar.em.matched_launch_harness.v1"
NATIVE_EXECUTION_AUDIT_SCHEMA = "recovar.em.native_matched_execution_audit.v1"
SUITE_ID = "pr158-k1-realdata-science-equivalence-v1"
TARGET_CASE_ID = "empiar-10202-set06-k1-I1"
CALIBRATION_CASE_IDS = (
    "empiar-10073-native-c1",
    "empiar-10345-native-c1",
    "empiar-10097-native-c1",
)
MASKED_SUPPORT_DATASET_IDS = ("10073", "10345", "10097")
MASKED_SUPPORT_ARTIFACT_KEYS = (
    "aggregate_summary",
    "all_fsc_curves",
    "corrected_masked_fsc_summary",
    "provenance",
    "readme",
)
MASKED_SUPPORT_REPLAY_ATOL = 5.0e-13
CURVE_KEYS = (
    "relion_final_half_fsc",
    "recovar_final_half_fsc",
    "final_cross_engine_raw",
    "final_cross_engine_half1",
    "final_cross_engine_half2",
)
SCIENCE_DIAGNOSTICS_SCHEMA = "recovar.em_k1_science_diagnostics.v1"
FINALIZER_RELATIVE_PATH = Path("scripts/finalize_empiar10202_set6_i1_evidence.py")
ALIGNED_CURVE_KEYS = (
    "final_cross_engine_proper_aligned",
    "final_cross_engine_half1_proper_aligned",
    "final_cross_engine_half2_proper_aligned",
)
MASKED_CURVE_KEYS = (
    "relion_final_half_fsc_common_masked",
    "recovar_final_half_fsc_common_masked",
    "final_cross_engine_proper_aligned_common_masked",
    "final_cross_engine_half1_proper_aligned_common_masked",
    "final_cross_engine_half2_proper_aligned_common_masked",
)
SO3_DETERMINANT_ATOL = 1.0e-6
SO3_ORTHOGONALITY_FROBENIUS_MAX = 1.0e-6
PROPER_ALIGNMENT_FIT_MAX_SHELL_FULL_BOX = 32
PROPER_ALIGNMENT_FIT_BOX_SIZE = 65
PROPER_ALIGNMENT_SEED_HEALPIX_ORDER = 1
PROPER_ALIGNMENT_REFINE_HEALPIX_ORDERS = [2]
TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM = 3.0
TARGET_REFERENCE_ENTRY = "EMD-9012"
TARGET_DEPOSITED_VALIDATION_RESOLUTION_ANGSTROM = 1.86
TARGET_EMDB_UNMASKED_HALF_MAP_RESOLUTION_ANGSTROM = 1.94
TARGET_RELION_UNMASKED_RESOLUTION_ANGSTROM = 2.511554
TARGET_RELION_CORRECTED_MASKED_RESOLUTION_ANGSTROM = 2.122559
TARGET_RELION_REFINEMENT_JOB_ID = 13217551
TARGET_RELION_POSTPROCESS_JOB_ID = 13254149
TARGET_INTERIM_REPORTED_ITERATION = 11
TARGET_INTERIM_RECOVAR_SOURCE_COMMIT = "4ea288467debbcba97cfaee6d2742ebc66c637f0"
TARGET_INTERIM_RECOVAR_TRAJECTORY_JOB_ID = 13339556
TARGET_INTERIM_RECOVAR_POSTPROCESS_JOB_ID = 13355910
TARGET_INTERIM_RELION_POSTPROCESS_JOB_ID = 13356820
TARGET_INTERIM_RAW_RESOLUTION_ANGSTROM = 2.521599769592285
TARGET_INTERIM_CORRECTED_MASKED_RESOLUTION_ANGSTROM = 2.541935251605126
TARGET_REPLACEMENT_JOB_ID = 13356985
TARGET_REPLACEMENT_CAPTURE_UTC = "2026-09-03T04:55:17Z"
TARGET_REPLACEMENT_ARTIFACT_KEYS = ("stderr", "hbm_trace", "scontrol", "command")
TARGET_COMPACT_SUCCESSOR_COMMIT = "8069ac01508d57bfd74a7686930ebcea66b6e328"
TARGET_COMPACT_SUCCESSOR_JOB_ID = 13363818
TARGET_INTERIM_ARTIFACT_KEYS = (
    "recovar_partial_summary",
    "recovar_postprocess_star",
    "matched_relion_summary",
    "matched_relion_curve_comparison",
    "matched_relion_postprocess_star",
    "matched_relion_job_script",
    "matched_relion_scontrol",
)
TARGET_PARTIAL_ENGINE_IDS = ("relion", "recovar")
TARGET_RELION_PARTIAL_ARTIFACT_KEYS = (
    "refinement_stdout",
    "launch_manifest",
    "refinement_command",
    "postprocess_stdout",
    "postprocess_result_manifest",
    "common_mask",
)
REPRODUCTION_REPLAY_COMMAND = (
    "pixi run python scripts/summarize_em_k1_realdata_science_equivalence.py "
    "--verify-calibrations --verify-masked-support --verify-target-partial --check-markdown"
)
REPRODUCTION_UNMASKED_GROUPS = ("10073_10345", "10097")
REPRODUCTION_MASKED_GROUPS = ("10073_10345", "10097", "aggregate")
SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT = {
    "recovar_merged": "recovar_final_sha256",
    "recovar_half1": "recovar_half1_sha256",
    "recovar_half2": "recovar_half2_sha256",
    "relion_merged": "relion_final_sha256",
    "relion_half1": "relion_half1_sha256",
    "relion_half2": "relion_half2_sha256",
}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CALIBRATION_REPLAY_ATOL = 5.0e-13
TARGET_SOURCE_STAR = Path(
    "/home/mg6942/mytigress/10202/06_Final_Stack/2017-12-27_MagCorrect_Frames05-19_Numbered_adjusted.star"
)
TARGET_PARTICLE_STACK = Path("/home/mg6942/mytigress/10202/06_Final_Stack/2017-12-27_MagCorrect_Frames05-19.mrcs")
TARGET_POSES_PKL = Path("/home/mg6942/mytigress/10202/06_Final_Stack/poses.pkl")
TARGET_CTF_PKL = Path("/home/mg6942/mytigress/10202/06_Final_Stack/ctf.pkl")
REJECTED_PREPARED_STAR_SHA256 = "c13cb927bb9cdffbd85fc2909a79196e44f5ce1836f8ba74b1d1b33c144edb11"
EXPECTED_THRESHOLDS = {
    "fsc_threshold": 1.0 / 7.0,
    "crossing_consecutive_shells": 3,
    "half_resolution_ratio_max": 1.05,
    "half_curve_rmse_max": 0.02,
    "half_band_auc_abs_delta_max": 0.02,
    "merged_cross_engine_band_auc_min": 0.95,
    "each_half_cross_engine_band_auc_min": 0.90,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of ``path`` without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of the canonical compact JSON representation."""

    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and GIT_SHA_RE.fullmatch(value) is not None


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    _require(math.isfinite(parsed), f"{label} must be finite")
    return parsed


def _as_curve(value: Any, label: str) -> np.ndarray:
    curve = np.asarray(value, dtype=np.float64).reshape(-1)
    _require(curve.size >= 4, f"{label} must contain DC and at least three non-DC shells")
    return curve


def first_sustained_below(curve: Any, threshold: float, consecutive: int) -> int | None:
    """Return the first shell starting ``consecutive`` values below threshold."""

    values = _as_curve(curve, "FSC curve")
    _require(consecutive >= 1, "consecutive-shell count must be positive")
    for shell in range(1, values.size - consecutive + 1):
        window = values[shell : shell + consecutive]
        if np.all(np.isfinite(window)) and np.all(window < threshold):
            return int(shell)
    return None


def normalized_band_auc(curve: Any, first_shell: int, last_shell: int) -> float:
    """Return trapezoidal FSC-AUC normalized over an inclusive shell band."""

    values = _as_curve(curve, "FSC curve")
    _require(1 <= first_shell <= last_shell < values.size, "invalid FSC band")
    band = values[first_shell : last_shell + 1]
    _require(np.all(np.isfinite(band)), "FSC band contains non-finite values")
    if band.size == 1:
        return float(band[0])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(band) / (band.size - 1))


def jointly_resolved_band(
    recovar_half_fsc: Any,
    relion_half_fsc: Any,
    *,
    threshold: float,
    consecutive: int,
) -> dict[str, int | bool]:
    """Derive the shared band from independent within-engine half-map FSCs."""

    recovar_curve = _as_curve(recovar_half_fsc, "RECOVAR half-map FSC")
    relion_curve = _as_curve(relion_half_fsc, "RELION half-map FSC")
    _require(recovar_curve.size == relion_curve.size, "half-map FSC curve lengths differ")
    last_common_shell = recovar_curve.size - 1
    recovar_crossing = first_sustained_below(recovar_curve, threshold, consecutive)
    relion_crossing = first_sustained_below(relion_curve, threshold, consecutive)
    recovar_missing = recovar_crossing is None
    relion_missing = relion_crossing is None
    if recovar_crossing is None:
        recovar_crossing = last_common_shell + 1
    if relion_crossing is None:
        relion_crossing = last_common_shell + 1
    last_shell = min(recovar_crossing, relion_crossing) - 1
    _require(last_shell >= 2, "jointly resolved FSC band has fewer than two non-DC shells")
    return {
        "first_shell": 1,
        "last_shell": int(last_shell),
        "recovar_crossing_shell": int(recovar_crossing),
        "relion_crossing_shell": int(relion_crossing),
        "recovar_crossing_beyond_measured_range": recovar_missing,
        "relion_crossing_beyond_measured_range": relion_missing,
    }


def apply_half_map_quality_gates(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    """Return failures in the mandatory independent half-map quality gates."""

    failures: list[str] = []
    checks = (
        (
            "half_resolution_ratio",
            float(metrics["half_resolution_ratio"]) <= float(thresholds["half_resolution_ratio_max"]),
        ),
        (
            "half_curve_rmse",
            float(metrics["half_curve_rmse"]) <= float(thresholds["half_curve_rmse_max"]),
        ),
        (
            "half_band_auc_abs_delta",
            float(metrics["half_band_auc_abs_delta"]) <= float(thresholds["half_band_auc_abs_delta_max"]),
        ),
    )
    for name, passed in checks:
        if not passed:
            failures.append(name)
    return failures


def apply_cross_engine_gates(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    """Return failures for one raw or proper-aligned cross-engine route."""

    failures: list[str] = []
    if float(metrics["merged_cross_engine_band_auc"]) < float(thresholds["merged_cross_engine_band_auc_min"]):
        failures.append("merged_cross_engine_band_auc")
    if min(
        float(metrics["half1_cross_engine_band_auc"]),
        float(metrics["half2_cross_engine_band_auc"]),
    ) < float(thresholds["each_half_cross_engine_band_auc_min"]):
        failures.append("each_half_cross_engine_band_auc")
    return failures


def apply_primary_gates(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    """Return legacy canonical-route failures (half quality plus raw cross-engine)."""

    return [
        *apply_half_map_quality_gates(metrics, thresholds),
        *apply_cross_engine_gates(metrics, thresholds),
    ]


def score_curves(
    curves: Mapping[str, Any],
    *,
    box_size: int,
    voxel_size_angstrom: float,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the five primary gates from external-collector FSC curves."""

    _require(box_size > 0, "box size must be positive")
    _require(voxel_size_angstrom > 0.0, "voxel size must be positive")
    missing = [key for key in CURVE_KEYS if key not in curves]
    _require(not missing, f"missing FSC curves: {', '.join(missing)}")
    parsed = {key: _as_curve(curves[key], key) for key in CURVE_KEYS}
    lengths = {curve.size for curve in parsed.values()}
    _require(len(lengths) == 1, "FSC curve lengths differ")

    band = jointly_resolved_band(
        parsed["recovar_final_half_fsc"],
        parsed["relion_final_half_fsc"],
        threshold=float(thresholds["fsc_threshold"]),
        consecutive=int(thresholds["crossing_consecutive_shells"]),
    )
    first_shell = int(band["first_shell"])
    last_shell = int(band["last_shell"])
    selection = slice(first_shell, last_shell + 1)
    recovar_half = parsed["recovar_final_half_fsc"][selection]
    relion_half = parsed["relion_final_half_fsc"][selection]
    delta = recovar_half - relion_half

    recovar_resolution = box_size * voxel_size_angstrom / int(band["recovar_crossing_shell"])
    relion_resolution = box_size * voxel_size_angstrom / int(band["relion_crossing_shell"])
    resolution_ratio = max(recovar_resolution, relion_resolution) / min(
        recovar_resolution,
        relion_resolution,
    )
    half_recovar_auc = normalized_band_auc(parsed["recovar_final_half_fsc"], first_shell, last_shell)
    half_relion_auc = normalized_band_auc(parsed["relion_final_half_fsc"], first_shell, last_shell)
    primary = {
        "half_resolution_ratio": float(resolution_ratio),
        "half_curve_rmse": float(np.sqrt(np.mean(delta * delta))),
        "half_curve_p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
        "half_band_auc_abs_delta": abs(half_recovar_auc - half_relion_auc),
        "recovar_half_band_auc": half_recovar_auc,
        "relion_half_band_auc": half_relion_auc,
        "merged_cross_engine_band_auc": normalized_band_auc(parsed["final_cross_engine_raw"], first_shell, last_shell),
        "half1_cross_engine_band_auc": normalized_band_auc(parsed["final_cross_engine_half1"], first_shell, last_shell),
        "half2_cross_engine_band_auc": normalized_band_auc(parsed["final_cross_engine_half2"], first_shell, last_shell),
    }
    half_map_failures = apply_half_map_quality_gates(primary, thresholds)
    raw_cross_engine_failures = apply_cross_engine_gates(primary, thresholds)
    failed_gates = [*half_map_failures, *raw_cross_engine_failures]
    legacy_last_shell = parsed["final_cross_engine_raw"].size - 1
    return {
        "jointly_resolved_band": {
            **band,
            "box_size": int(box_size),
            "voxel_size_angstrom": float(voxel_size_angstrom),
            "recovar_resolution_angstrom": float(recovar_resolution),
            "relion_resolution_angstrom": float(relion_resolution),
            "resolution_formula": "box_size * voxel_size_angstrom / crossing_shell",
        },
        "primary_metrics": primary,
        "primary_pass": not failed_gates,
        "failed_gates": failed_gates,
        "half_map_quality_pass": not half_map_failures,
        "half_map_quality_failed_gates": half_map_failures,
        "raw_cross_engine_pass": not raw_cross_engine_failures,
        "raw_cross_engine_failed_gates": raw_cross_engine_failures,
        "legacy_full_spectrum": {
            "canonical_merged_normalized_non_dc_fsc_auc": normalized_band_auc(
                parsed["final_cross_engine_raw"], 1, legacy_last_shell
            ),
            "strict_parity_gate": 0.995,
            "strict_parity_pass": normalized_band_auc(parsed["final_cross_engine_raw"], 1, legacy_last_shell) >= 0.995,
            "acceptance_metric": False,
        },
    }


def _validate_thresholds(thresholds: Mapping[str, Any]) -> None:
    _require(set(thresholds) == set(EXPECTED_THRESHOLDS), "scorecard threshold keys changed")
    for key, expected in EXPECTED_THRESHOLDS.items():
        observed = thresholds[key]
        if isinstance(expected, int):
            _require(observed == expected, f"scorecard threshold {key} changed")
        else:
            _require(float(observed) == expected, f"scorecard threshold {key} changed")


def _validate_masked_fsc_support_contract(support: Mapping[str, Any]) -> None:
    """Validate the frozen, non-acceptance RELION masked-FSC manifest."""

    _require(support.get("role") == "supporting_only", "masked FSC role changed")
    _require(support.get("acceptance_metric") is False, "masked FSC became an acceptance metric")
    _require(support.get("can_rescue") is False, "masked FSC became a rescue route")
    _require(tuple(support.get("dataset_ids", ())) == MASKED_SUPPORT_DATASET_IDS, "masked FSC dataset set changed")
    artifacts = support.get("artifacts", {})
    _require(set(artifacts) == set(MASKED_SUPPORT_ARTIFACT_KEYS), "masked FSC artifact set changed")
    for name, artifact in artifacts.items():
        _require(Path(artifact.get("path", "")).is_absolute(), f"masked FSC {name} path is not absolute")
        _require(_is_sha256(artifact.get("sha256")), f"masked FSC {name} SHA-256 is invalid")

    jobs = support.get("producer_jobs", {})
    _require(set(jobs) == {"13273806", "13274377"}, "masked FSC producer jobs changed")
    _require(jobs["13273806"].get("datasets") == ["10073", "10345"], "masked FSC job 13273806 scope changed")
    _require(jobs["13274377"].get("datasets") == ["10097"], "masked FSC job 13274377 scope changed")
    for job_id, job in jobs.items():
        _require(job.get("ReqTRES") == job.get("AllocTRES"), f"masked FSC job {job_id} allocation changed")

    binaries = support.get("relion_binaries", {})
    _require(
        set(binaries) == {"relion_image_handler", "relion_mask_create", "relion_postprocess"},
        "masked FSC RELION binary set changed",
    )
    for name, binary in binaries.items():
        _require(Path(binary.get("path", "")).is_absolute(), f"masked FSC {name} path is not absolute")
        _require(_is_sha256(binary.get("sha256")), f"masked FSC {name} SHA-256 is invalid")

    _require(
        support.get("postprocess_policy")
        == {
            "same_mask_for_both_engines": True,
            "force_mask": True,
            "skip_fsc_weighting": True,
            "low_pass_angstrom": 0,
            "randomize_at_fsc": 0.8,
            "random_seed": 42,
        },
        "masked FSC postprocess policy changed",
    )
    mask_policy = support.get("mask_policy", {})
    _require(mask_policy.get("source") == "RELION merged map in native RELION file frame", "mask source changed")
    _require(mask_policy.get("lowpass_angstrom") == 15, "mask low-pass changed")
    _require(mask_policy.get("extend_pixels") == 5, "mask extension changed")
    _require(mask_policy.get("soft_edge_pixels") == 8, "mask soft edge changed")
    header_audit = support.get("mask_header_nondeterminism", {})
    _require(header_audit.get("dataset") == "10097", "mask header audit dataset changed")
    _require(header_audit.get("postprocess_used_literal_audited_mask") is True, "literal 10097 mask was not used")
    _require(
        header_audit.get("differing_one_based_file_bytes") == [249, 250, 253],
        "10097 mask header-byte audit changed",
    )

    expected = support.get("expected_corrected_metrics", {})
    _require(tuple(expected) == MASKED_SUPPORT_DATASET_IDS, "masked FSC expected dataset order changed")
    for dataset, metrics in expected.items():
        _require(int(metrics.get("recovar_crossing_shell", 0)) > 0, f"{dataset} RECOVAR masked crossing is invalid")
        _require(int(metrics.get("relion_crossing_shell", 0)) > 0, f"{dataset} RELION masked crossing is invalid")
        for key in (
            "recovar_resolution_angstrom",
            "relion_resolution_angstrom",
            "curve_rmse",
            "band_auc_absolute_delta",
        ):
            _finite_float(metrics.get(key), f"{dataset} {key}")
        _require(_is_sha256(metrics.get("mask_sha256")), f"{dataset} mask SHA-256 is invalid")


def _validate_frozen_artifact(artifact: Mapping[str, Any], label: str) -> None:
    """Validate one absolute-path/SHA-256 pair without touching the artifact."""

    _require(Path(artifact.get("path", "")).is_absolute(), f"{label} path is not absolute")
    _require(_is_sha256(artifact.get("sha256")), f"{label} SHA-256 is invalid")


def _validate_target_partial_engine_results(case: Mapping[str, Any]) -> None:
    """Validate the sealed RELION-only result without scoring the pending case."""

    partial = case.get("partial_engine_results", {})
    _require(tuple(partial) == TARGET_PARTIAL_ENGINE_IDS, "target partial-engine set changed")
    relion = partial["relion"]
    recovar = partial["recovar"]
    _require(relion.get("status") == "complete", "target RELION partial status changed")
    _require(recovar == {"status": "pending"}, "target RECOVAR partial result is not pending")

    unmasked = relion.get("unmasked", {})
    _require(
        unmasked.get("metric")
        == "RELION final unmasked FSC=0.143 estimate; result_manifest.source_final_unmasked_resolution_angstrom",
        "target RELION unmasked metric changed",
    )
    _require(
        float(unmasked.get("resolution_angstrom", float("nan"))) == TARGET_RELION_UNMASKED_RESOLUTION_ANGSTROM,
        "target RELION unmasked resolution changed",
    )
    _require(
        unmasked.get("high_resolution_gate_pass")
        is (TARGET_RELION_UNMASKED_RESOLUTION_ANGSTROM <= TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM),
        "target RELION high-resolution partial gate changed",
    )

    masked = relion.get("corrected_masked", {})
    _require(masked.get("role") == "supporting_only", "target RELION masked role changed")
    _require(masked.get("acceptance_metric") is False, "target RELION masked result became an acceptance metric")
    _require(masked.get("can_rescue") is False, "target RELION masked result became a rescue route")
    _require(
        float(masked.get("resolution_angstrom", float("nan"))) == TARGET_RELION_CORRECTED_MASKED_RESOLUTION_ANGSTROM,
        "target RELION corrected-masked resolution changed",
    )

    jobs = relion.get("jobs", {})
    _require(set(jobs) == {"refinement", "postprocess"}, "target RELION partial job set changed")
    _require(jobs["refinement"].get("job_id") == TARGET_RELION_REFINEMENT_JOB_ID, "target RELION job changed")
    _require(
        jobs["postprocess"].get("job_id") == TARGET_RELION_POSTPROCESS_JOB_ID,
        "target RELION postprocess job changed",
    )
    for name, job in jobs.items():
        _require(job.get("state") == "COMPLETED", f"target RELION {name} job did not complete")
        _require(job.get("exit_code") == "0:0", f"target RELION {name} exit code changed")
        _require(job.get("ReqTRES") == job.get("AllocTRES"), f"target RELION {name} allocation changed")

    artifacts = relion.get("artifacts", {})
    _require(
        tuple(artifacts) == TARGET_RELION_PARTIAL_ARTIFACT_KEYS,
        "target RELION partial artifact set changed",
    )
    for name, artifact in artifacts.items():
        _validate_frozen_artifact(artifact, f"target RELION {name}")


def _validate_target_interim_iteration11_diagnostic(case: Mapping[str, Any]) -> None:
    """Validate the non-scoring matched-iteration high-resolution checkpoint."""

    diagnostic = case.get("interim_iteration11_diagnostic", {})
    _require(diagnostic.get("role") == "diagnostic_only", "target interim role changed")
    _require(diagnostic.get("can_score_case") is False, "target interim result became scoring evidence")
    _require(
        diagnostic.get("reported_iteration") == TARGET_INTERIM_REPORTED_ITERATION,
        "target interim iteration changed",
    )

    recovar = diagnostic.get("recovar", {})
    _require(
        recovar.get("source_commit") == TARGET_INTERIM_RECOVAR_SOURCE_COMMIT,
        "target interim RECOVAR source changed",
    )
    _require(
        recovar.get("trajectory_job_id") == TARGET_INTERIM_RECOVAR_TRAJECTORY_JOB_ID,
        "target interim RECOVAR trajectory job changed",
    )
    _require(
        recovar.get("postprocess_job_id") == TARGET_INTERIM_RECOVAR_POSTPROCESS_JOB_ID,
        "target interim RECOVAR postprocess job changed",
    )
    _require(
        recovar.get("last_complete_iteration") == TARGET_INTERIM_REPORTED_ITERATION,
        "target interim RECOVAR checkpoint changed",
    )
    _require(
        recovar.get("trajectory_terminal_state") == "FAILED_CUDA_OOM_DURING_ITERATION_12",
        "target interim RECOVAR terminal state changed",
    )

    relion = diagnostic.get("relion", {})
    _require(
        relion.get("source_refinement_job_id") == TARGET_RELION_REFINEMENT_JOB_ID,
        "target interim RELION source job changed",
    )
    _require(
        relion.get("postprocess_job_id") == TARGET_INTERIM_RELION_POSTPROCESS_JOB_ID,
        "target interim RELION postprocess job changed",
    )
    _require(relion.get("state") == "COMPLETED", "target interim RELION postprocess did not complete")
    _require(relion.get("exit_code") == "0:0", "target interim RELION postprocess exit code changed")
    _require(
        relion.get("requested_tres") == relion.get("allocated_tres"),
        "target interim RELION allocation changed",
    )

    resolutions = diagnostic.get("same_iteration_resolutions", {})
    _require(set(resolutions) == {"raw_unmasked", "corrected_masked"}, "target interim resolution set changed")
    expected_resolutions = {
        "raw_unmasked": TARGET_INTERIM_RAW_RESOLUTION_ANGSTROM,
        "corrected_masked": TARGET_INTERIM_CORRECTED_MASKED_RESOLUTION_ANGSTROM,
    }
    for name, expected in expected_resolutions.items():
        result = resolutions[name]
        _require(float(result.get("recovar_angstrom", float("nan"))) == expected, f"target interim {name} RECOVAR resolution changed")
        _require(float(result.get("relion_angstrom", float("nan"))) == expected, f"target interim {name} RELION resolution changed")
        _require(float(result.get("absolute_delta_angstrom", float("nan"))) == 0.0, f"target interim {name} resolution delta changed")
        _require(result.get("crossing_shell_equal") is True, f"target interim {name} crossing changed")
    _require(
        resolutions["raw_unmasked"].get("acceptance_role") == "mandatory_unmasked_diagnostic",
        "target interim raw role changed",
    )
    _require(
        resolutions["corrected_masked"].get("acceptance_role") == "supporting_only",
        "target interim masked role changed",
    )

    curves = diagnostic.get("resolved_band_curve_comparison", {})
    _require(set(curves) == {"raw_unmasked", "corrected_masked"}, "target interim curve set changed")
    raw = curves["raw_unmasked"]
    masked = curves["corrected_masked"]
    _require(raw.get("first_shell") == 1 and raw.get("last_shell") == 249, "target interim raw band changed")
    _require(masked.get("first_shell") == 1 and masked.get("last_shell") == 247, "target interim masked band changed")
    for name, result in curves.items():
        rmse = _finite_float(result.get("rmse"), f"target interim {name} RMSE")
        auc_delta = _finite_float(result.get("normalized_auc_absolute_delta"), f"target interim {name} AUC delta")
        _require(rmse <= EXPECTED_THRESHOLDS["half_curve_rmse_max"], f"target interim {name} RMSE no longer passes")
        _require(
            auc_delta <= EXPECTED_THRESHOLDS["half_band_auc_abs_delta_max"],
            f"target interim {name} AUC delta no longer passes",
        )

    replacement = diagnostic.get("replacement_full_run", {})
    _require(replacement.get("subject_commit") == case.get("expected_subject_commit"), "target replacement subject changed")
    _require(replacement.get("job_id") == TARGET_REPLACEMENT_JOB_ID, "target replacement job changed")
    _require(replacement.get("status_at_capture") == "FAILED", "target replacement capture status changed")
    _require(replacement.get("captured_at_utc") == TARGET_REPLACEMENT_CAPTURE_UTC, "target replacement capture time changed")
    _require(
        replacement.get("terminal_state") == "FAILED_CUDA_OOM_DURING_ITERATION_12",
        "target replacement terminal state changed",
    )
    _require(replacement.get("exit_code") == "1:0", "target replacement exit code changed")
    _require(replacement.get("elapsed") == "08:43:50", "target replacement elapsed time changed")
    _require(replacement.get("last_complete_numbered_iteration") == 11, "target replacement checkpoint changed")
    _require(replacement.get("failed_numbered_iteration") == 12, "target replacement failed iteration changed")
    _require(replacement.get("failed_current_size") == 564, "target replacement failed size changed")
    _require(
        replacement.get("requested_tres") == replacement.get("allocated_tres"),
        "target replacement allocation changed",
    )
    _require(Path(replacement.get("run_root", "")).is_absolute(), "target replacement run root is not absolute")

    boundary = replacement.get("failure_boundary", {})
    _require(boundary.get("planner_mode") == "historical_full_cube", "target replacement planner mode changed")
    _require(float(boundary.get("persistent_estimated_gb", float("nan"))) == 81.92, "target replacement memory estimate changed")
    _require(boundary.get("image_batch_size") == 1, "target replacement image batch changed")
    _require(boundary.get("rotation_block_size") == 4, "target replacement rotation block changed")
    _require(boundary.get("wide_tail_bucket_rotations") == 512, "target replacement wide-tail shape changed")
    _require(boundary.get("wide_tail_bucket_count") == 5, "target replacement wide-tail count changed")
    _require(boundary.get("sampled_peak_hbm_used_mib") == 76329, "target replacement peak HBM changed")
    _require(boundary.get("sampled_minimum_hbm_free_mib") == 4744, "target replacement free HBM changed")
    _require(
        boundary.get("failing_operation") == "relion_projector_half_texture_f32",
        "target replacement failing operation changed",
    )

    replacement_artifacts = replacement.get("artifacts", {})
    _require(
        tuple(replacement_artifacts) == TARGET_REPLACEMENT_ARTIFACT_KEYS,
        "target replacement artifact set changed",
    )
    for name, artifact in replacement_artifacts.items():
        _validate_frozen_artifact(artifact, f"target replacement {name}")

    successor = replacement.get("successor_validation", {})
    _require(successor.get("subject_commit") == TARGET_COMPACT_SUCCESSOR_COMMIT, "target successor subject changed")
    _require(successor.get("job_id") == TARGET_COMPACT_SUCCESSOR_JOB_ID, "target successor job changed")
    _require(successor.get("status_at_capture") == "RUNNING", "target successor capture status changed")
    _require(successor.get("last_complete_numbered_iteration") == 8, "target successor checkpoint changed")
    _require(successor.get("current_numbered_iteration") == 9, "target successor iteration changed")
    _require(successor.get("current_size") == 516, "target successor size changed")
    _require(Path(successor.get("run_root", "")).is_absolute(), "target successor run root is not absolute")

    artifacts = diagnostic.get("artifacts", {})
    _require(tuple(artifacts) == TARGET_INTERIM_ARTIFACT_KEYS, "target interim artifact set changed")
    for name, artifact in artifacts.items():
        _validate_frozen_artifact(artifact, f"target interim {name}")


def _validate_reproduction_contract(reproduction: Mapping[str, Any]) -> None:
    """Validate exact recorded producer references and the repository replay command."""

    _require(
        reproduction.get("artifact_replay_command") == REPRODUCTION_REPLAY_COMMAND,
        "real-data artifact replay command changed",
    )
    unmasked = reproduction.get("unmasked", {})
    _require(tuple(unmasked) == REPRODUCTION_UNMASKED_GROUPS, "unmasked reproduction groups changed")
    expected_commands = {
        "10073_10345": [
            "sbatch --export=ALL,DATASET_ID=10073 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch",
            "sbatch --export=ALL,DATASET_ID=10345 /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution/scripts/run_dataset_native.sbatch",
        ],
        "10097": [
            "sbatch --parsable --export=NONE /home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/full_dataset_native_resolution_replacement_10097_20260828T211155EDT/scripts/run_dataset_native_10097.sbatch"
        ],
    }
    for group, commands in expected_commands.items():
        spec = unmasked[group]
        _require(spec.get("recorded_submission_commands") == commands, f"{group} submission commands changed")
        for name in ("submission_record", "launcher", "collector"):
            _validate_frozen_artifact(spec.get(name, {}), f"{group} {name}")

    masked = reproduction.get("masked", {})
    _require(tuple(masked) == REPRODUCTION_MASKED_GROUPS, "masked reproduction groups changed")
    for group in ("10073_10345", "10097"):
        spec = masked[group]
        _require(
            set(spec) == {"original_submission_command_recorded", "launcher", "driver"},
            f"{group} masked reproduction fields changed",
        )
        _require(
            spec.get("original_submission_command_recorded") is False,
            f"{group} masked reproduction invents an original submission command",
        )
        for name in ("launcher", "driver"):
            _validate_frozen_artifact(spec.get(name, {}), f"{group} masked {name}")
    aggregate = masked["aggregate"]
    _require(set(aggregate) == {"builder", "durable_provenance"}, "masked aggregate fields changed")
    for name in ("builder", "durable_provenance"):
        _validate_frozen_artifact(aggregate.get(name, {}), f"masked aggregate {name}")


def replay_reproduction_contract(reproduction: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash every external producer, collector, and submission reference."""

    _validate_reproduction_contract(reproduction)
    artifacts: list[tuple[str, Mapping[str, Any]]] = []
    for group, spec in reproduction["unmasked"].items():
        artifacts.extend((f"{group} {name}", spec[name]) for name in ("submission_record", "launcher", "collector"))
    for group in ("10073_10345", "10097"):
        spec = reproduction["masked"][group]
        artifacts.extend((f"{group} masked {name}", spec[name]) for name in ("launcher", "driver"))
    aggregate = reproduction["masked"]["aggregate"]
    artifacts.extend((f"masked aggregate {name}", aggregate[name]) for name in ("builder", "durable_provenance"))
    for label, artifact in artifacts:
        path = Path(artifact["path"])
        _require(path.is_file(), f"missing reproduction artifact {label}: {path}")
        _require(sha256_file(path) == artifact["sha256"], f"reproduction artifact {label} SHA-256 changed")
    return {"verification_status": "verified", **reproduction}


def replay_target_partial_engine_results(case: Mapping[str, Any]) -> dict[str, Any]:
    """Hash and parse the frozen RELION-only target evidence."""

    _validate_target_partial_engine_results(case)
    partial = case["partial_engine_results"]
    relion = partial["relion"]
    artifacts = relion["artifacts"]
    for name, artifact in artifacts.items():
        path = Path(artifact["path"])
        _require(path.is_file(), f"missing target RELION {name}: {path}")
        _require(sha256_file(path) == artifact["sha256"], f"target RELION {name} SHA-256 changed")

    launch = json.loads(Path(artifacts["launch_manifest"]["path"]).read_text())
    _require(launch.get("schema") == NATIVE_LAUNCH_MANIFEST_SCHEMA, "target RELION launch schema changed")
    result = json.loads(Path(artifacts["postprocess_result_manifest"]["path"]).read_text())
    _require(result.get("schema") == "recovar.em.relion_postprocess_evidence.v1", "target RELION result schema changed")
    _require(result.get("dataset") == "EMPIAR-10202 image set 6", "target RELION result dataset changed")
    _require(result.get("symmetry") == "I1", "target RELION result symmetry changed")
    _require(
        result.get("source_refinement_job_id") == TARGET_RELION_REFINEMENT_JOB_ID,
        "target RELION result refinement job changed",
    )
    _require(
        result.get("postprocess_job_id") == TARGET_RELION_POSTPROCESS_JOB_ID,
        "target RELION result postprocess job changed",
    )
    _require(
        float(result.get("source_final_unmasked_resolution_angstrom", float("nan")))
        == TARGET_RELION_UNMASKED_RESOLUTION_ANGSTROM,
        "target RELION result unmasked resolution changed",
    )
    _require(
        float(result.get("postprocess_corrected_masked_resolution_angstrom", float("nan")))
        == TARGET_RELION_CORRECTED_MASKED_RESOLUTION_ANGSTROM,
        "target RELION result corrected-masked resolution changed",
    )
    _require(result.get("mask", {}).get("path") == artifacts["common_mask"]["path"], "target RELION mask path changed")
    _require(
        result.get("mask", {}).get("sha256") == artifacts["common_mask"]["sha256"],
        "target RELION mask digest changed",
    )
    return {
        "verification_status": "verified",
        "relion": dict(relion),
        "recovar": dict(partial["recovar"]),
    }


def replay_target_interim_iteration11_diagnostic(case: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash the non-scoring matched iteration-11 checkpoint evidence."""

    _validate_target_interim_iteration11_diagnostic(case)
    diagnostic = case["interim_iteration11_diagnostic"]
    for name, artifact in diagnostic["artifacts"].items():
        path = Path(artifact["path"])
        _require(path.is_file(), f"missing target interim {name}: {path}")
        _require(sha256_file(path) == artifact["sha256"], f"target interim {name} SHA-256 changed")

    relion_summary = json.loads(Path(diagnostic["artifacts"]["matched_relion_summary"]["path"]).read_text())
    _require(
        relion_summary.get("schema") == "recovar.em.10202.relion_it011_matched_fsc.v1",
        "target interim RELION summary schema changed",
    )
    _require(relion_summary.get("status") == "PASS", "target interim RELION summary no longer passes")
    _require(
        relion_summary.get("reported_iteration") == TARGET_INTERIM_REPORTED_ITERATION,
        "target interim RELION summary iteration changed",
    )
    relion_resolutions = relion_summary.get("relion_it011_resolutions_angstrom", {})
    _require(
        float(relion_resolutions.get("raw_unmasked_fsc", {}).get("resolution", float("nan")))
        == TARGET_INTERIM_RAW_RESOLUTION_ANGSTROM,
        "target interim RELION summary raw resolution changed",
    )
    _require(
        float(relion_resolutions.get("corrected_masked_fsc", {}).get("resolution", float("nan")))
        == TARGET_INTERIM_CORRECTED_MASKED_RESOLUTION_ANGSTROM,
        "target interim RELION summary masked resolution changed",
    )
    crossing_comparison = relion_summary.get("comparison_to_recovar_it011_crossings", {})
    _require(crossing_comparison.get("raw_crossing_shell_equal") is True, "target interim raw crossing diverged")
    _require(
        crossing_comparison.get("corrected_masked_crossing_shell_equal") is True,
        "target interim masked crossing diverged",
    )

    curve_summary = json.loads(
        Path(diagnostic["artifacts"]["matched_relion_curve_comparison"]["path"]).read_text()
    )
    _require(
        curve_summary.get("schema") == "recovar.em.fsc_curve_comparison.v1",
        "target interim curve summary schema changed",
    )
    curve_pairs = {
        "raw_unmasked": curve_summary.get("raw_unmasked_fsc", {}).get("shared_resolved_band", {}),
        "corrected_masked": curve_summary.get("corrected_masked_fsc", {}).get("shared_resolved_band", {}),
    }
    for name, observed in curve_pairs.items():
        expected = diagnostic["resolved_band_curve_comparison"][name]
        _require(
            float(observed.get("rmse", float("nan"))) == float(expected["rmse"]),
            f"target interim {name} replay RMSE changed",
        )
        _require(
            float(observed.get("normalized_auc_absolute_delta", float("nan")))
            == float(expected["normalized_auc_absolute_delta"]),
            f"target interim {name} replay AUC delta changed",
        )
    return {"verification_status": "verified", **diagnostic}


def _validate_calibration_route_contract(
    case: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> None:
    """Validate one calibration without making cross-engine parity mandatory."""

    case_id = str(case["id"])
    expected = case.get("expected_metrics", {})
    _require(
        not apply_half_map_quality_gates(expected, thresholds),
        f"{case_id} no longer passes mandatory half-map calibration",
    )
    raw_failures = apply_cross_engine_gates(expected, thresholds)
    route = case.get("expected_cross_engine_route")
    _require(
        route in {"raw_canonical", "continuous_proper_so3_rigid", "none_unqualified"}, f"{case_id} route is invalid"
    )
    diagnostics = case.get("proper_so3_diagnostics")
    if route == "raw_canonical":
        _require(not raw_failures, f"{case_id} raw route no longer passes")
        _require(diagnostics is None, f"{case_id} has unnecessary proper-SO3 diagnostics")
        return

    _require(raw_failures, f"{case_id} proper-SO3 route supplied despite a passing raw route")
    _require(isinstance(diagnostics, Mapping), f"{case_id} proper-SO3 diagnostics are missing")
    _require(_is_git_sha(diagnostics.get("subject_commit")), f"{case_id} diagnostic commit is invalid")
    _require(
        _is_sha256(diagnostics.get("producer_sha256")),
        f"{case_id} diagnostic producer SHA-256 is invalid",
    )
    _require(int(diagnostics.get("slurm_job_id", 0)) > 0, f"{case_id} diagnostic job is invalid")
    for name in ("launch_script", "slurm_stdout", "slurm_stderr"):
        artifact = diagnostics.get(name, {})
        _require(Path(artifact.get("path", "")).is_absolute(), f"{case_id} {name} path is not absolute")
        _require(_is_sha256(artifact.get("sha256")), f"{case_id} {name} SHA-256 is invalid")
    resources = diagnostics.get("resources", {})
    _require(resources.get("state") == "COMPLETED", f"{case_id} diagnostic job did not complete")
    _require(resources.get("exit_code") == "0:0", f"{case_id} diagnostic job exit code changed")
    _require(resources.get("ReqTRES") == resources.get("AllocTRES"), f"{case_id} diagnostic allocation changed")
    artifacts = diagnostics.get("artifacts", {})
    _require(
        set(artifacts) == {"science_diagnostics", "curve_archive", "common_mask"},
        f"{case_id} diagnostic artifact set changed",
    )
    for name, artifact in artifacts.items():
        _require(Path(artifact.get("path", "")).is_absolute(), f"{case_id} diagnostic {name} path is not absolute")
        _require(_is_sha256(artifact.get("sha256")), f"{case_id} diagnostic {name} SHA-256 is invalid")
    _require(
        artifacts["science_diagnostics"].get("schema") == SCIENCE_DIAGNOSTICS_SCHEMA,
        f"{case_id} diagnostic schema changed",
    )
    expected_aligned = diagnostics.get("expected_metrics", {})
    _require(
        set(expected_aligned)
        == {
            "merged_cross_engine_band_auc",
            "half1_cross_engine_band_auc",
            "half2_cross_engine_band_auc",
        },
        f"{case_id} diagnostic metric set changed",
    )
    for key, value in expected_aligned.items():
        _finite_float(value, f"{case_id} {key}")
    aligned_pass = not apply_cross_engine_gates(expected_aligned, thresholds)
    _require(
        diagnostics.get("expected_can_rescue_cross_engine") is aligned_pass,
        f"{case_id} diagnostic rescue expectation changed",
    )
    expected_route = "continuous_proper_so3_rigid" if aligned_pass else "none_unqualified"
    _require(route == expected_route, f"{case_id} expected route disagrees with diagnostic gates")
    superseded = diagnostics.get("superseded_diagnostic")
    if superseded is not None:
        _require(superseded.get("slurm_job_id") == 13275901, f"{case_id} superseded job changed")
        _require(
            _is_sha256(superseded.get("science_diagnostics_sha256")),
            f"{case_id} superseded diagnostic SHA-256 is invalid",
        )
        _require(
            _is_sha256(superseded.get("curve_archive_sha256")),
            f"{case_id} superseded curve SHA-256 is invalid",
        )
        _require(
            superseded.get("reason")
            == "invalid alignment seed search omitted canonical identity; retained for audit and excluded from all scorecard metrics",
            f"{case_id} superseded diagnostic reason changed",
        )


def load_and_validate_scorecard(path: Path = DEFAULT_SCORECARD) -> dict[str, Any]:
    """Load the frozen manifest and validate its denominator and policy."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCORECARD_SCHEMA, "wrong scorecard schema")
    suite = scorecard.get("suite", {})
    _require(suite.get("id") == SUITE_ID, "wrong scorecard suite id")
    _require(suite.get("subject_pr") == 158, "scorecard must remain bound to PR 158")
    _require(suite.get("scoring_denominator") == 1, "scoring denominator must remain one")
    _require(suite.get("scoring_case_ids") == [TARGET_CASE_ID], "fixed scoring case changed")
    _require(
        tuple(suite.get("calibration_case_ids", ())) == CALIBRATION_CASE_IDS,
        "calibration cases changed",
    )
    _require(_is_git_sha(suite.get("subject_commit")), "subject commit is not a full git SHA")
    _validate_thresholds(scorecard.get("thresholds", {}))

    metric_policy = scorecard.get("metric_policy", {})
    _require(
        metric_policy.get("half_map_quality") == "mandatory unmasked unaligned within-engine split-half FSC",
        "half-map quality policy changed",
    )
    _require(
        metric_policy.get("cross_engine_acceptance")
        == "raw canonical FSC gates OR pinned continuous proper-SO(3)+translation FSC gates",
        "cross-engine acceptance policy changed",
    )
    _require(metric_policy.get("proper_alignment_can_rescue_cross_engine_only") is True, "alignment scope changed")
    _require(metric_policy.get("masked_fsc_is_supporting_only") is True, "masked FSC policy changed")
    _require(metric_policy.get("mirror_sign_scale_forbidden") is True, "alignment ambiguity policy changed")
    _require(metric_policy.get("strict_full_spectrum_parity_is_separate") is True, "strict parity policy changed")
    _require(
        metric_policy.get("target_high_resolution_is_independent") is True,
        "target high-resolution policy changed",
    )
    _require(
        metric_policy.get("target_overall_requires_equivalence_and_high_resolution") is True,
        "target overall policy changed",
    )

    high_resolution_gate = scorecard.get("target_high_resolution_gate", {})
    _require(high_resolution_gate.get("case_id") == TARGET_CASE_ID, "wrong high-resolution target")
    _require(
        float(high_resolution_gate.get("each_engine_resolution_angstrom_max", float("nan")))
        == TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM,
        "target high-resolution ceiling changed",
    )
    reference = high_resolution_gate.get("reference", {})
    _require(reference.get("entry") == TARGET_REFERENCE_ENTRY, "target reference entry changed")
    _require(
        float(reference.get("deposited_validation_resolution_angstrom", float("nan")))
        == TARGET_DEPOSITED_VALIDATION_RESOLUTION_ANGSTROM,
        "target deposited resolution changed",
    )
    _require(
        float(reference.get("emdb_calculated_unmasked_half_map_resolution_angstrom", float("nan")))
        == TARGET_EMDB_UNMASKED_HALF_MAP_RESOLUTION_ANGSTROM,
        "target EMDB half-map resolution changed",
    )

    collector = scorecard.get("collector", {})
    _require(collector.get("schema") == COLLECTOR_SCHEMA, "wrong external collector schema")
    _require(Path(collector.get("path", "")).is_absolute(), "collector path is not absolute")
    _require(_is_sha256(collector.get("sha256")), "invalid external collector SHA-256")

    diagnostic_producer = scorecard.get("diagnostic_producer", {})
    _require(
        Path(diagnostic_producer.get("path", "")) == DIAGNOSTIC_PRODUCER_RELATIVE_PATH,
        "wrong proper-SO3 diagnostic producer path",
    )
    _require(
        diagnostic_producer.get("schema") == SCIENCE_DIAGNOSTICS_SCHEMA,
        "wrong proper-SO3 diagnostic schema",
    )
    _require(_is_sha256(diagnostic_producer.get("sha256")), "invalid proper-SO3 producer SHA-256")
    producer_path = REPO_ROOT / DIAGNOSTIC_PRODUCER_RELATIVE_PATH
    _require(producer_path.is_file(), f"missing proper-SO3 diagnostic producer: {producer_path}")
    _require(
        sha256_file(producer_path) == diagnostic_producer["sha256"],
        "proper-SO3 diagnostic producer SHA-256 changed",
    )
    _validate_masked_fsc_support_contract(scorecard.get("masked_fsc_support", {}))

    cases = scorecard.get("cases", [])
    ids = [case.get("id") for case in cases]
    _require(len(ids) == len(set(ids)), "duplicate scorecard case id")
    _require(set(ids) == {TARGET_CASE_ID, *CALIBRATION_CASE_IDS}, "scorecard case set changed")
    by_id = {case["id"]: case for case in cases}
    target = by_id[TARGET_CASE_ID]
    _require(target.get("role") == "scoring", "target case is not scoring")
    _require(target.get("requires_execution_binding") is True, "target execution binding requirement changed")
    _require(target.get("dataset") == "10202", "target dataset changed")
    _require(
        target.get("status") == "relion_complete_recovar_pending",
        "target partial-completion status changed",
    )
    _require(
        target.get("expected_subject_commit") == suite["subject_commit"],
        "target subject commit changed",
    )
    rejected_artifacts = target.get("rejected_artifacts", [])
    _require(
        len(rejected_artifacts) == 1 and rejected_artifacts[0].get("sha256") == REJECTED_PREPARED_STAR_SHA256,
        "target rejected-preparation audit changed",
    )
    _require(
        target.get("refinement_contract")
        == {
            "k": 1,
            "same_particles_and_order": True,
            "same_metadata": True,
            "same_initial_reference_canonical_array": True,
            "same_half_assignment": True,
            "prepared_star_preserves_source_particle_order": True,
            "prepared_star_preserves_source_half_assignment": True,
            "prepared_star_preserves_source_pose_shift_metadata": True,
            "autonomous_refinement": True,
            "fixed_poses": False,
            "local_only": False,
            "forced_final_after_nonconvergence": False,
            "same_exact_symmetry_operator_set": True,
        },
        "target refinement contract changed",
    )
    target_contract = target.get("input_contract", {})
    prepared_star_sha256 = target_contract.get("prepared_star_sha256")
    _require(
        prepared_star_sha256 is None
        or (_is_sha256(prepared_star_sha256) and prepared_star_sha256 != REJECTED_PREPARED_STAR_SHA256),
        "target prepared STAR is invalid or explicitly rejected",
    )
    _require(target_contract.get("particle_count") == 30_515, "target particle count changed")
    _require(target_contract.get("box_size") == 800, "target box size changed")
    _require(target_contract.get("voxel_size_angstrom") == 0.788, "target voxel size changed")
    source_star = target_contract.get("source_star", {})
    particle_stack = target_contract.get("particle_stack", {})
    _require(Path(source_star.get("path", "")) == TARGET_SOURCE_STAR, "target STAR is not set 6")
    _require(
        Path(particle_stack.get("path", "")) == TARGET_PARTICLE_STACK,
        "target stack is not set 6",
    )
    _require(
        source_star.get("sha256") == "5b89aa3f2f4c66c40820952982f67dd0ba3d46864c65b5eb35e16b1fe4d11cdb",
        "target STAR digest changed",
    )
    for name, expected_path, expected_sha256 in (
        (
            "poses_pkl",
            TARGET_POSES_PKL,
            "f52394a1ea41f442fdcbf37e4808aa878f52ab033f875aa81d623c5bc9a2b238",
        ),
        (
            "ctf_pkl",
            TARGET_CTF_PKL,
            "62ad1941816e33668051230faa30f970630de4e57508a943f3dd962f87e2fcbb",
        ),
    ):
        artifact = target_contract.get(name, {})
        _require(
            Path(artifact.get("path", "")) == expected_path,
            f"target {name} path changed",
        )
        _require(artifact.get("sha256") == expected_sha256, f"target {name} digest changed")
    _require(particle_stack.get("size_bytes") == 78_118_401_024, "target stack size changed")
    _require(
        particle_stack.get("sha256") == "8eecf0fbf8e645ac51feff278a86e43e7e4be117921333dc6d3e22e52a628453",
        "target full stack digest changed",
    )
    _require(
        _is_sha256(particle_stack.get("header_1mib_sha256")),
        "target stack header digest is invalid",
    )
    _require(
        target_contract.get("half_assignment_sha256")
        == "1a06a8e0fccc553a99ecbdd24b21115b2b20b14cd1ae267310c6d4fe449d054f",
        "target half assignment changed",
    )
    _require(
        target_contract.get("half_assignment_encoding")
        == "bytes(int(rlnRandomSubset)) in source STAR row order; one uint8 byte per particle",
        "target half-assignment encoding changed",
    )
    _require(target_contract.get("half_assignment_bytes") == 30_515, "target half byte count changed")
    _require(target_contract.get("half1_count") == 15_258, "target half-1 count changed")
    _require(target_contract.get("half2_count") == 15_257, "target half-2 count changed")
    initial_reference = target_contract.get("initial_reference", {})
    _require(
        initial_reference.get("exact_gate")
        == "numpy.array_equal(load_mrc(recovar_frame_file), load_relion_volume(relion_file))",
        "target initial-reference exact gate changed",
    )
    symmetry = target_contract.get("symmetry", {})
    _require(symmetry.get("family") == "icosahedral", "target symmetry family changed")
    _require(symmetry.get("requested_label") == "I1", "target requested symmetry label changed")
    _require(symmetry.get("label") == "I1", "target canonical symmetry label changed")
    _require(
        symmetry.get("forbidden_bare_alias") == "I (RELION canonicalizes it to I2)",
        "target forbidden symmetry alias changed",
    )
    _require(symmetry.get("operator_count") == 60, "target symmetry order changed")
    _require(
        symmetry.get("operator_order") == "identity at index 0, then RELION SymList::get_matrices(isym) for isym=0..58",
        "target symmetry operator order changed",
    )
    _require(
        symmetry.get("left_operator_policy")
        == "all left operators are identity and are not applied in BPref reconstruction",
        "target symmetry left-operator policy changed",
    )
    _require(
        symmetry.get("hash_encoding")
        == "float64 rounded to 12 decimals, stacked [left,right], little-endian C-order bytes",
        "target symmetry hash encoding changed",
    )
    _require(
        symmetry.get("operators_sha256") == "093a0876b93610ec141c87840ae3ff4dc4491b27dec87143558358ef556557b8",
        "target symmetry operator digest changed",
    )
    _require(
        target.get("pending_contract_fields") == _required_pending_contract_fields(target),
        "target pending-contract list is stale",
    )
    _validate_target_partial_engine_results(target)
    _validate_target_interim_iteration11_diagnostic(target)
    _validate_reproduction_contract(scorecard.get("reproduction", {}))
    for case_id in CALIBRATION_CASE_IDS:
        case = by_id[case_id]
        _require(case.get("role") == "calibration", f"{case_id} is not calibration-only")
        _require(_is_git_sha(case.get("subject_commit")), f"{case_id} subject commit is invalid")
        _validate_calibration_route_contract(case, scorecard["thresholds"])
        for artifact in case.get("artifacts", {}).values():
            _require(Path(artifact["path"]).is_absolute(), f"{case_id} artifact path is not absolute")
            _require(_is_sha256(artifact.get("sha256")), f"{case_id} artifact digest is invalid")
    return scorecard


def _validate_collector_metrics(case: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    check(metrics.get("schema") == COLLECTOR_SCHEMA, "collector_schema")
    check(str(metrics.get("dataset")) == str(case.get("dataset")), "dataset_identity")
    check(metrics.get("scientifically_valid") is True, "collector_scientifically_valid")
    check(metrics.get("initial_reference_frame_gate", {}).get("pass") is True, "initial_reference")
    identity = metrics.get("particle_identity_gate", {})
    check(identity.get("pass") is True, "particle_identity")
    expected_count = int(case["input_contract"]["particle_count"])
    check(identity.get("prepared_count") == expected_count, "prepared_particle_count")
    check(identity.get("relion_count") == expected_count, "relion_particle_count")
    check(identity.get("recovar_count") == expected_count, "recovar_particle_count")
    convergence = metrics.get("convergence_and_topology", {})
    check(convergence.get("relion_converged") is True, "relion_convergence")
    check(convergence.get("recovar_converged") is True, "recovar_convergence")
    return failures


def _compare_calibration_metrics(case: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    expected = case["expected_metrics"]
    for key, expected_value in expected.items():
        actual = _finite_float(observed[key], key)
        _require(
            math.isclose(actual, float(expected_value), rel_tol=0.0, abs_tol=CALIBRATION_REPLAY_ATOL),
            f"{case['id']} calibration metric {key} changed: {actual} != {expected_value}",
        )


def replay_calibration_case(
    scorecard: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a calibration row from its frozen external JSON and NPZ."""

    collector_path = Path(scorecard["collector"]["path"])
    _require(collector_path.is_file(), f"missing external collector: {collector_path}")
    _require(
        sha256_file(collector_path) == scorecard["collector"]["sha256"],
        "external collector SHA-256 changed",
    )
    metrics_artifact = case["artifacts"]["metrics_json"]
    curves_artifact = case["artifacts"]["fsc_curves_npz"]
    metrics_path = Path(metrics_artifact["path"])
    curves_path = Path(curves_artifact["path"])
    _require(metrics_path.is_file(), f"missing calibration metrics: {metrics_path}")
    _require(curves_path.is_file(), f"missing calibration curves: {curves_path}")
    _require(sha256_file(metrics_path) == metrics_artifact["sha256"], "calibration metrics SHA-256 changed")
    _require(sha256_file(curves_path) == curves_artifact["sha256"], "calibration curves SHA-256 changed")
    metrics = json.loads(metrics_path.read_text())
    provenance_failures = _validate_collector_metrics(case, metrics)
    _require(not provenance_failures, f"calibration validity failed: {', '.join(provenance_failures)}")
    with np.load(curves_path, allow_pickle=False) as archive:
        curves = {key: np.asarray(archive[key], dtype=np.float64) for key in CURVE_KEYS}
    scored = score_curves(
        curves,
        box_size=int(case["input_contract"]["box_size"]),
        voxel_size_angstrom=float(case["input_contract"]["voxel_size_angstrom"]),
        thresholds=scorecard["thresholds"],
    )
    _compare_calibration_metrics(case, scored["primary_metrics"])
    _require(scored["half_map_quality_pass"], f"{case['id']} half-map calibration no longer passes")

    science_diagnostics: dict[str, Any] = {"status": "not_supplied"}
    diagnostic_spec = case.get("proper_so3_diagnostics")
    if diagnostic_spec is not None:
        for name in ("launch_script", "slurm_stdout", "slurm_stderr"):
            artifact = diagnostic_spec[name]
            path = Path(artifact["path"])
            _require(path.is_file(), f"missing calibration {name}: {path}")
            _require(sha256_file(path) == artifact["sha256"], f"calibration {name} SHA-256 changed")
        science_diagnostics, science_failures = _validate_science_diagnostics(
            scorecard,
            case,
            {"analysis_artifacts": diagnostic_spec["artifacts"]},
            metrics,
            expected_producer_sha256=diagnostic_spec["producer_sha256"],
            first_shell=int(scored["jointly_resolved_band"]["first_shell"]),
            last_shell=int(scored["jointly_resolved_band"]["last_shell"]),
            expected_curve_length=int(curves[CURVE_KEYS[0]].size),
        )
        _require(not science_failures, f"calibration diagnostics invalid: {', '.join(science_failures)}")
        aligned_metrics = science_diagnostics["proper_so3_alignment"]["metrics"]
        for key, expected_value in diagnostic_spec["expected_metrics"].items():
            actual = _finite_float(aligned_metrics[key], key)
            _require(
                math.isclose(actual, float(expected_value), rel_tol=0.0, abs_tol=CALIBRATION_REPLAY_ATOL),
                f"{case['id']} diagnostic metric {key} changed: {actual} != {expected_value}",
            )
        _require(
            science_diagnostics["proper_so3_alignment"]["can_rescue_cross_engine"]
            is diagnostic_spec["expected_can_rescue_cross_engine"],
            f"{case['id']} diagnostic rescue result changed",
        )

    if scored["raw_cross_engine_pass"]:
        observed_route = "raw_canonical"
    elif science_diagnostics.get("proper_so3_alignment", {}).get("can_rescue_cross_engine") is True:
        observed_route = "continuous_proper_so3_rigid"
    else:
        observed_route = "none_unqualified"
    _require(observed_route == case["expected_cross_engine_route"], f"{case['id']} cross-engine route changed")
    science_equivalence_pass = observed_route != "none_unqualified"
    return {
        "id": case["id"],
        "role": "calibration",
        "status": "pass",
        "subject_commit": case["subject_commit"],
        **scored,
        "cross_engine_route": observed_route,
        "science_equivalence_pass": science_equivalence_pass,
        "science_diagnostics": science_diagnostics,
        "artifacts": case["artifacts"],
    }


def replay_masked_fsc_support(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the sealed RELION corrected-masked FSC bundle without scoring it."""

    support = scorecard["masked_fsc_support"]
    _validate_masked_fsc_support_contract(support)
    for name, artifact in support["artifacts"].items():
        path = Path(artifact["path"])
        _require(path.is_file(), f"missing masked FSC {name}: {path}")
        _require(sha256_file(path) == artifact["sha256"], f"masked FSC {name} SHA-256 changed")
    for name, binary in support["relion_binaries"].items():
        path = Path(binary["path"])
        _require(path.is_file(), f"missing masked FSC RELION binary {name}: {path}")
        _require(sha256_file(path) == binary["sha256"], f"masked FSC RELION binary {name} SHA-256 changed")

    aggregate = json.loads(Path(support["artifacts"]["aggregate_summary"]["path"]).read_text())
    _require(
        tuple(aggregate.get("datasets", {})) == MASKED_SUPPORT_DATASET_IDS, "masked FSC aggregate dataset order changed"
    )
    observed_jobs = aggregate.get("producer_jobs", {})
    for job_id, expected_job in support["producer_jobs"].items():
        observed_job = observed_jobs.get(job_id, {})
        for key, value in expected_job.items():
            _require(observed_job.get(key) == value, f"masked FSC job {job_id} {key} changed")
    for name, expected_binary in support["relion_binaries"].items():
        observed_binary = aggregate.get("relion_binaries", {}).get(name, {})
        _require(observed_binary.get("path") == expected_binary["path"], f"masked FSC {name} path changed")
        _require(observed_binary.get("sha256") == expected_binary["sha256"], f"masked FSC {name} digest changed")
    _require(
        aggregate.get("postprocess_policy") == support["postprocess_policy"], "masked FSC aggregate policy changed"
    )
    for key, value in support["mask_policy"].items():
        _require(aggregate.get("mask_policy", {}).get(key) == value, f"masked FSC mask policy {key} changed")
    header_audit = aggregate.get("mask_header_nondeterminism", {})
    expected_header = support["mask_header_nondeterminism"]
    for expected_key, aggregate_key in (
        ("audited_mask_sha256", "audited_10097_mask_sha256"),
        ("regenerated_file_sha256", "regenerated_file_sha256"),
        ("identical_voxel_payload_sha256", "identical_voxel_payload_sha256"),
        ("differing_one_based_file_bytes", "differing_one_based_file_bytes"),
        ("postprocess_used_literal_audited_mask", "postprocess_used_literal_audited_mask"),
    ):
        _require(
            header_audit.get(aggregate_key) == expected_header[expected_key],
            f"masked FSC 10097 header audit {expected_key} changed",
        )

    for dataset, expected in support["expected_corrected_metrics"].items():
        observed_dataset = aggregate["datasets"][dataset]
        _require(observed_dataset.get("producer_job") in (13273806, 13274377), f"{dataset} producer job changed")
        _require(
            observed_dataset.get("same_literal_mask_path_for_both_engines") is True, f"{dataset} mask reuse changed"
        )
        _require(
            observed_dataset.get("same_postprocess_policy_for_both_engines") is True, f"{dataset} policy reuse changed"
        )
        _require(observed_dataset.get("mask", {}).get("sha256") == expected["mask_sha256"], f"{dataset} mask changed")
        corrected = observed_dataset.get("curve_comparisons", {}).get("corrected_masked_fsc", {})
        crossings = corrected.get("threshold_crossings", {}).get("0.143", {})
        observed = {
            "recovar_crossing_shell": crossings.get("recovar", {}).get("first_below_shell"),
            "recovar_resolution_angstrom": crossings.get("recovar", {}).get("first_below_resolution_angstrom"),
            "relion_crossing_shell": crossings.get("relion", {}).get("first_below_shell"),
            "relion_resolution_angstrom": crossings.get("relion", {}).get("first_below_resolution_angstrom"),
            "curve_rmse": corrected.get("curve_rmse"),
            "band_auc_absolute_delta": corrected.get("band_auc_absolute_delta"),
        }
        for key in ("recovar_crossing_shell", "relion_crossing_shell"):
            _require(observed[key] == expected[key], f"{dataset} masked FSC {key} changed")
        for key in (
            "recovar_resolution_angstrom",
            "relion_resolution_angstrom",
            "curve_rmse",
            "band_auc_absolute_delta",
        ):
            _require(
                math.isclose(
                    float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=MASKED_SUPPORT_REPLAY_ATOL
                ),
                f"{dataset} masked FSC {key} changed",
            )
    return {
        "status": "verified",
        "role": "supporting_only",
        "acceptance_metric": False,
        "can_rescue": False,
        "artifacts": support["artifacts"],
        "producer_jobs": support["producer_jobs"],
        "expected_corrected_metrics": support["expected_corrected_metrics"],
    }


def _required_pending_contract_fields(case: Mapping[str, Any]) -> list[str]:
    contract = case["input_contract"]
    symmetry = contract["symmetry"]
    initial_reference = contract["initial_reference"]
    required = {
        "input_contract.prepared_star_sha256": contract.get("prepared_star_sha256"),
        "input_contract.particle_stack.sha256": contract["particle_stack"].get("sha256"),
        "input_contract.initial_reference.relion_file_sha256": initial_reference.get("relion_file_sha256"),
        "input_contract.initial_reference.recovar_frame_file_sha256": initial_reference.get(
            "recovar_frame_file_sha256"
        ),
        "input_contract.initial_reference.canonical_array_sha256": initial_reference.get("canonical_array_sha256"),
        "input_contract.symmetry.label": symmetry.get("label"),
        "input_contract.symmetry.operators_sha256": symmetry.get("operators_sha256"),
    }
    pending: list[str] = []
    for name, value in required.items():
        if name.endswith("sha256"):
            if not _is_sha256(value):
                pending.append(name)
        elif not isinstance(value, str) or not value:
            pending.append(name)
    return pending


def _validate_execution_binding(
    evidence: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    """Validate the immutable launcher/finalizer binding when it is supplied."""

    binding = evidence.get("execution_binding")
    if binding is None:
        return ["execution_binding:missing"] if required else []
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(f"execution_binding:{label}")

    check(isinstance(binding, Mapping), "object")
    if not isinstance(binding, Mapping):
        return failures
    check(binding.get("schema") == EXECUTION_BINDING_SCHEMA, "schema")

    paths: dict[str, Path] = {}
    for stem in ("launch_manifest", "finalizer_command", "evidence_envelope"):
        raw_path = binding.get(f"{stem}_path", "")
        path = Path(raw_path) if isinstance(raw_path, str) else Path("")
        paths[stem] = path
        digest = binding.get(f"{stem}_sha256")
        check(path.is_absolute(), f"{stem}_path_not_absolute")
        check(_is_sha256(digest), f"{stem}_sha256_format")
        check(path.is_file(), f"{stem}_missing")
        if path.is_file() and _is_sha256(digest):
            check(sha256_file(path) == digest, f"{stem}_sha256")

    expected_finalizer = (REPO_ROOT / FINALIZER_RELATIVE_PATH).resolve()
    if paths["finalizer_command"].is_absolute():
        check(paths["finalizer_command"].resolve() == expected_finalizer, "finalizer_command_path")

    native_binding = binding.get("launch_manifest_schema") == NATIVE_LAUNCH_MANIFEST_SCHEMA
    replacement_records = binding.get("replacement_records", []) if native_binding else []
    if native_binding:
        check(isinstance(replacement_records, list), "replacement_records")
        if not isinstance(replacement_records, list):
            replacement_records = []
    else:
        check("replacement_records" not in binding, "unexpected_replacement_records")
    argv = binding.get("finalizer_argv")
    expected_argv_length = 4 + 2 * len(replacement_records)
    check(
        isinstance(argv, list) and len(argv) == expected_argv_length and all(isinstance(value, str) for value in argv),
        "finalizer_argv",
    )
    check(_is_sha256(binding.get("finalizer_argv_sha256")), "finalizer_argv_sha256_format")
    if isinstance(argv, list) and all(isinstance(value, str) for value in argv):
        check(sha256_json(argv) == binding.get("finalizer_argv_sha256"), "finalizer_argv_sha256")
        if len(argv) == expected_argv_length:
            check(Path(argv[1]).resolve() == expected_finalizer, "finalizer_argv_command")
            check(argv[2] == "--launch-manifest", "finalizer_argv_flag")
            check(Path(argv[3]).resolve() == paths["launch_manifest"].resolve(), "finalizer_argv_manifest")
            for index, record in enumerate(replacement_records):
                offset = 4 + 2 * index
                check(argv[offset] == "--replacement-record", f"replacement_{index}_argv_flag")
                if not isinstance(record, Mapping):
                    check(False, f"replacement_{index}_record")
                    continue
                replacement_path = Path(record.get("path", ""))
                replacement_digest = record.get("sha256")
                check(replacement_path.is_absolute(), f"replacement_{index}_path")
                check(_is_sha256(replacement_digest), f"replacement_{index}_sha256_format")
                check(replacement_path.is_file(), f"replacement_{index}_missing")
                check(Path(argv[offset + 1]).resolve() == replacement_path.resolve(), f"replacement_{index}_argv")
                if replacement_path.is_file() and _is_sha256(replacement_digest):
                    check(sha256_file(replacement_path) == replacement_digest, f"replacement_{index}_sha256")

    launch_path = paths["launch_manifest"]
    if launch_path.is_file():
        try:
            launch = json.loads(launch_path.read_text())
        except (json.JSONDecodeError, OSError):
            check(False, "launch_manifest_json")
        else:
            allowed_schema = NATIVE_LAUNCH_MANIFEST_SCHEMA if native_binding else LAUNCH_MANIFEST_SCHEMA
            check(launch.get("schema") == allowed_schema, "launch_manifest_schema")
            if not native_binding:
                check(launch.get("subject") == evidence.get("subject"), "launch_manifest_subject")
            check(Path(launch.get("run_root", "")).resolve() == launch_path.parent.resolve(), "launch_manifest_root")

    envelope_path = paths["evidence_envelope"]
    envelope: Mapping[str, Any] | None = None
    if envelope_path.is_file():
        try:
            loaded_envelope = json.loads(envelope_path.read_text())
        except (json.JSONDecodeError, OSError):
            check(False, "evidence_envelope_json")
        else:
            check(isinstance(loaded_envelope, Mapping), "evidence_envelope_object")
            envelope = loaded_envelope if isinstance(loaded_envelope, Mapping) else None
        if envelope is not None:
            check(envelope.get("schema") == EXECUTION_BINDING_SCHEMA, "evidence_envelope_schema")
            envelope_manifest = envelope.get("launch", {}).get("launch_manifest", {})
            check(
                Path(envelope_manifest.get("path", "")).resolve() == launch_path.resolve(),
                "evidence_envelope_manifest_path",
            )
            check(
                envelope_manifest.get("sha256") == binding.get("launch_manifest_sha256"),
                "evidence_envelope_manifest_sha256",
            )
            for field in ("collector", "analysis_commands", "analysis_artifacts"):
                check(evidence.get(field) == envelope.get(field), f"evidence_envelope_{field}")

    embedded_launch = evidence.get("launch")
    if embedded_launch is not None:
        check(isinstance(embedded_launch, Mapping), "embedded_launch")
        if isinstance(embedded_launch, Mapping):
            embedded_manifest = embedded_launch.get("launch_manifest", {})
            check(
                Path(embedded_manifest.get("path", "")).resolve() == paths["launch_manifest"].resolve(),
                "embedded_launch_manifest_path",
            )
            check(
                embedded_manifest.get("sha256") == binding.get("launch_manifest_sha256"),
                "embedded_launch_manifest_sha256",
            )
            if envelope is not None:
                check(embedded_launch == envelope.get("launch"), "evidence_envelope_launch")
            if native_binding:
                check(embedded_launch.get("schema") == NATIVE_EXECUTION_AUDIT_SCHEMA, "native_audit_schema")
                science = embedded_launch.get("science_scoring", {})
                check(science.get("eligible") is True, "native_science_eligible")
                check(
                    science.get("smoke_only_promotion_forbidden") is True,
                    "native_smoke_promotion_forbidden",
                )
                check(
                    embedded_launch.get("science_subject") == evidence.get("subject"),
                    "native_science_subject",
                )
                embedded_replacements = embedded_launch.get("replacement_records", [])
                check(
                    isinstance(embedded_replacements, list)
                    and [row.get("artifact") for row in embedded_replacements if isinstance(row, Mapping)]
                    == replacement_records,
                    "native_replacement_binding",
                )
                jobs = embedded_launch.get("jobs", {})
                for key in ("recovar_smoke", "relion_smoke"):
                    job = jobs.get(key, {}) if isinstance(jobs, Mapping) else {}
                    check(job.get("science_role") == "capability_only", f"native_{key}_capability_only")
                    check(job.get("smoke_can_promote") is False, f"native_{key}_cannot_promote")
                for key in ("recovar_full", "relion_full"):
                    job = jobs.get(key, {}) if isinstance(jobs, Mapping) else {}
                    check(job.get("phase") == "full", f"native_{key}_phase")
                    check(job.get("science_role") == "science_candidate", f"native_{key}_science_role")
                    check(job.get("classification") == "completed", f"native_{key}_completed")
                    check(job.get("provenance_complete") is True, f"native_{key}_provenance")
                    check(bool(job.get("expected_outputs")), f"native_{key}_outputs")
    return failures


def validate_case_evidence_provenance(
    scorecard: Mapping[str, Any],
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Return fail-closed provenance failures for a scoring-case envelope."""

    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    check(evidence.get("schema") == EVIDENCE_SCHEMA, "evidence_schema")
    check(evidence.get("case_id") == case["id"], "case_id")
    failures.extend(
        _validate_execution_binding(
            evidence,
            required=bool(case.get("requires_execution_binding")),
        )
    )
    pending = _required_pending_contract_fields(case)
    if pending:
        failures.append("scorecard_contract_not_frozen")

    subject = evidence.get("subject", {})
    subject_matches_contract = subject.get("commit") == case.get("expected_subject_commit")
    check(subject_matches_contract, "subject_commit")
    launch_schema = evidence.get("execution_binding", {}).get("launch_manifest_schema")
    if launch_schema == NATIVE_LAUNCH_MANIFEST_SCHEMA and not subject_matches_contract:
        # A native retry may intentionally exercise a newer fix, but that does
        # not silently redefine the frozen science subject.  Promotion requires
        # a reviewed scorecard contract change that names the exact new commit.
        failures.append("native_subject_requires_deliberate_scorecard_contract_update")
    check(subject.get("commit") == scorecard["suite"]["subject_commit"], "suite_subject_commit")
    check(subject.get("tree_clean") is True, "subject_tree_clean")
    check(subject.get("diff_sha256") == EMPTY_SHA256, "subject_diff_sha256")

    expected = case["input_contract"]
    observed = evidence.get("input_contract", {})
    for key in ("particle_count", "box_size", "voxel_size_angstrom", "prepared_star_sha256"):
        check(observed.get(key) == expected.get(key), key)
    check(observed.get("poses_pkl_sha256") == expected["poses_pkl"]["sha256"], "poses_pkl_sha256")
    check(observed.get("ctf_pkl_sha256") == expected["ctf_pkl"]["sha256"], "ctf_pkl_sha256")
    check(
        observed.get("prepared_star_preserves_source_particle_order") is True,
        "prepared_star_particle_order",
    )
    check(
        observed.get("prepared_star_preserves_source_half_assignment") is True,
        "prepared_star_half_assignment",
    )
    check(
        observed.get("prepared_star_preserves_source_pose_shift_metadata") is True,
        "prepared_star_pose_shift_metadata",
    )
    check(observed.get("particle_stack_size_bytes") == expected["particle_stack"]["size_bytes"], "particle_stack_size")
    check(observed.get("particle_stack_sha256") == expected["particle_stack"].get("sha256"), "particle_stack_sha256")
    check(observed.get("half_assignment_sha256") == expected.get("half_assignment_sha256"), "half_assignment_sha256")
    for key in (
        "half_assignment_encoding",
        "half_assignment_bytes",
        "half1_count",
        "half2_count",
    ):
        check(observed.get(key) == expected.get(key), key)
    expected_reference = expected["initial_reference"]
    check(observed.get("initial_reference_canonical_exact") is True, "initial_reference_canonical_exact")
    for key in (
        "relion_file_sha256",
        "recovar_frame_file_sha256",
        "canonical_array_sha256",
    ):
        evidence_key = f"initial_reference_{key}"
        check(observed.get(evidence_key) == expected_reference.get(key), evidence_key)
        check(_is_sha256(observed.get(evidence_key)), f"{evidence_key}_format")
    check(observed.get("k") == 1, "k_is_one")
    check(observed.get("autonomous_refinement") is True, "autonomous_refinement")
    check(observed.get("fixed_poses") is False, "fixed_poses_disabled")
    check(observed.get("local_only") is False, "local_only_disabled")
    check(observed.get("forced_final_after_nonconvergence") is False, "forced_final_disabled")

    expected_symmetry = expected["symmetry"]
    observed_symmetry = observed.get("symmetry", {})
    check(observed_symmetry.get("family") == expected_symmetry.get("family"), "symmetry_family")
    check(
        observed_symmetry.get("requested_label") == expected_symmetry.get("requested_label"),
        "symmetry_requested_label",
    )
    check(observed_symmetry.get("relion_label") == expected_symmetry.get("label"), "relion_symmetry_label")
    check(observed_symmetry.get("recovar_label") == expected_symmetry.get("label"), "recovar_symmetry_label")
    check(observed_symmetry.get("operator_count") == expected_symmetry.get("operator_count"), "symmetry_operator_count")
    for key in ("operator_order", "left_operator_policy", "hash_encoding"):
        check(observed_symmetry.get(key) == expected_symmetry.get(key), f"symmetry_{key}")
    check(
        observed_symmetry.get("operators_sha256") == expected_symmetry.get("operators_sha256"),
        "symmetry_operators_sha256",
    )

    collector = evidence.get("collector", {})
    check(collector.get("schema") == COLLECTOR_SCHEMA, "collector_schema")
    check(collector.get("collector_sha256") == scorecard["collector"]["sha256"], "collector_sha256")
    return failures


def _validate_science_diagnostics(
    scorecard: Mapping[str, Any],
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
    collector_metrics: Mapping[str, Any],
    *,
    expected_producer_sha256: str,
    first_shell: int,
    last_shell: int,
    expected_curve_length: int,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and score an optional proper-SO3/common-mask artifact bundle."""

    analysis_artifacts = evidence.get("analysis_artifacts")
    if analysis_artifacts is None:
        return {
            "status": "not_supplied",
            "proper_so3_alignment": {"status": "not_supplied", "can_rescue_cross_engine": False},
            "common_mask": {"status": "not_supplied", "can_rescue": False},
        }, []
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(f"science_diagnostics:{label}")

    science_spec = analysis_artifacts.get("science_diagnostics", {})
    curve_spec = analysis_artifacts.get("curve_archive", {})
    mask_spec = analysis_artifacts.get("common_mask", {})
    science_path = Path(science_spec.get("path", ""))
    curves_path = Path(curve_spec.get("path", ""))
    mask_path = Path(mask_spec.get("path", ""))
    check(science_path.is_absolute(), "json_path_not_absolute")
    check(curves_path.is_absolute(), "curve_path_not_absolute")
    check(mask_path.is_absolute(), "mask_path_not_absolute")
    check(science_spec.get("schema") == SCIENCE_DIAGNOSTICS_SCHEMA, "json_schema_envelope")
    check(_is_sha256(science_spec.get("sha256")), "json_sha256_format")
    check(_is_sha256(curve_spec.get("sha256")), "curve_sha256_format")
    check(_is_sha256(mask_spec.get("sha256")), "mask_sha256_format")
    check(science_path.is_file(), "json_missing")
    check(curves_path.is_file(), "curve_archive_missing")
    check(mask_path.is_file(), "mask_missing")
    if science_path.is_file():
        check(sha256_file(science_path) == science_spec.get("sha256"), "json_sha256")
    if curves_path.is_file():
        check(sha256_file(curves_path) == curve_spec.get("sha256"), "curve_sha256")
    if mask_path.is_file():
        check(sha256_file(mask_path) == mask_spec.get("sha256"), "mask_sha256")
    if failures:
        return {
            "status": "invalid",
            "proper_so3_alignment": {"status": "invalid", "can_rescue_cross_engine": False},
            "common_mask": {"status": "invalid", "can_rescue": False},
        }, failures

    payload = json.loads(science_path.read_text())
    check(payload.get("schema") == SCIENCE_DIAGNOSTICS_SCHEMA, "json_schema")
    check(payload.get("case_id") == case["id"], "case_id")
    producer = payload.get("producer", {})
    check(
        producer.get("sha256") == expected_producer_sha256,
        "producer_sha256",
    )
    check(_is_sha256(producer.get("sha256")), "producer_sha256_format")
    producer_path = Path(producer.get("path", ""))
    check(producer_path.is_absolute(), "producer_path_not_absolute")
    check(producer_path.is_file(), "producer_missing")
    if producer_path.is_file():
        check(sha256_file(producer_path) == producer.get("sha256"), "producer_file_sha256")
    input_artifacts = payload.get("inputs", {})
    expected_input_names = set(SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT)
    check(set(input_artifacts) == expected_input_names, "input_artifact_names")
    collector_artifacts = collector_metrics.get("artifacts", {})
    for name in sorted(expected_input_names):
        artifact = input_artifacts.get(name, {})
        artifact_path = Path(artifact.get("path", ""))
        check(artifact_path.is_absolute(), f"{name}_path_not_absolute")
        check(_is_sha256(artifact.get("sha256")), f"{name}_sha256_format")
        check(artifact_path.is_file(), f"{name}_missing")
        if artifact_path.is_file():
            check(sha256_file(artifact_path) == artifact.get("sha256"), f"{name}_sha256")
        collector_key = SCIENCE_INPUT_TO_COLLECTOR_ARTIFACT[name]
        check(_is_sha256(collector_artifacts.get(collector_key)), f"collector_{collector_key}_format")
        check(
            artifact.get("sha256") == collector_artifacts.get(collector_key),
            f"{name}_collector_artifact_sha256",
        )
    check(payload.get("box_size") == case["input_contract"]["box_size"], "box_size")
    check(
        math.isclose(
            float(payload.get("voxel_size_angstrom", float("nan"))),
            float(case["input_contract"]["voxel_size_angstrom"]),
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ),
        "voxel_size_angstrom",
    )
    internal_curves = payload.get("artifacts", {}).get("curve_archive", {})
    internal_mask = payload.get("artifacts", {}).get("common_mask", {})
    check(Path(internal_curves.get("path", "")) == curves_path, "internal_curve_path")
    check(internal_curves.get("sha256") == curve_spec.get("sha256"), "internal_curve_sha256")
    check(Path(internal_mask.get("path", "")) == mask_path, "internal_mask_path")
    check(internal_mask.get("sha256") == mask_spec.get("sha256"), "internal_mask_sha256")

    alignment = payload.get("diagnostics", {}).get("proper_so3_alignment", {})
    check(alignment.get("fit_source") == "merged_low_frequency", "fit_source")
    check(
        alignment.get("method")
        == "identity-augmented HEALPix proper-rotation seed plus continuous scipy rotvec Powell and subpixel translation",
        "alignment_method",
    )
    check(alignment.get("seed_source") == "identity_augmented_RELION_HEALPix_grid", "alignment_seed_source")
    check(alignment.get("continuous_so3_refinement") is True, "continuous_so3_refinement")
    check(alignment.get("translation_subpixel") is True, "translation_subpixel")
    check(
        alignment.get("fit_max_shell_full_box") == PROPER_ALIGNMENT_FIT_MAX_SHELL_FULL_BOX,
        "fit_max_shell_full_box",
    )
    check(alignment.get("fit_box_size") == PROPER_ALIGNMENT_FIT_BOX_SIZE, "fit_box_size")
    check(
        alignment.get("seed_healpix_order") == PROPER_ALIGNMENT_SEED_HEALPIX_ORDER,
        "seed_healpix_order",
    )
    check(
        alignment.get("refine_healpix_orders") == PROPER_ALIGNMENT_REFINE_HEALPIX_ORDERS,
        "refine_healpix_orders",
    )
    check(alignment.get("applied_unchanged_to") == ["merged", "half1", "half2"], "transform_reuse")
    check(alignment.get("no_reflection") is True, "reflection_forbidden")
    check(alignment.get("sign_fit") is False, "sign_fit_forbidden")
    check(alignment.get("scale_fit") is False, "scale_fit_forbidden")
    check(alignment.get("optimizer_success") is True, "optimizer_success")
    expected_symmetry = case["input_contract"]["symmetry"]
    check(alignment.get("symmetry_label") == expected_symmetry["label"], "symmetry_label")
    check(
        alignment.get("symmetry_operators_sha256") == expected_symmetry["operators_sha256"],
        "symmetry_operators_sha256",
    )
    rotation = np.asarray(alignment.get("rotation_matrix_recovar_to_relion"), dtype=np.float64)
    translation = np.asarray(alignment.get("translation_recovar_to_relion_zyx"), dtype=np.float64)
    check(rotation.shape == (3, 3) and np.all(np.isfinite(rotation)), "rotation_matrix")
    check(translation.shape == (3,) and np.all(np.isfinite(translation)), "translation")
    if rotation.shape == (3, 3) and np.all(np.isfinite(rotation)):
        determinant = float(np.linalg.det(rotation))
        orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        check(abs(determinant - 1.0) <= SO3_DETERMINANT_ATOL, "rotation_determinant")
        check(orthogonality <= SO3_ORTHOGONALITY_FROBENIUS_MAX, "rotation_orthogonality")
        check(
            math.isclose(float(alignment.get("determinant", float("nan"))), determinant, rel_tol=0.0, abs_tol=1.0e-10),
            "reported_determinant",
        )
        check(
            math.isclose(
                float(alignment.get("orthogonality_frobenius", float("nan"))),
                orthogonality,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            "reported_orthogonality",
        )

    common_mask = payload.get("diagnostics", {}).get("common_mask", {})
    check(Path(common_mask.get("path", "")) == mask_path, "common_mask_path")
    check(common_mask.get("sha256") == mask_spec.get("sha256"), "common_mask_sha256")
    check(common_mask.get("engine_symmetric") is True, "common_mask_engine_symmetry")
    check(common_mask.get("applied_identically_to_both_engines") is True, "common_mask_identical_application")
    check(common_mask.get("acceptance_metric") is False, "common_mask_nonacceptance")
    policy = payload.get("acceptance_policy", {})
    check(policy.get("proper_alignment_can_replace_only_failed_raw_cross_engine_gates") is True, "alignment_scope")
    check(policy.get("within_engine_half_map_quality_remains_mandatory") is True, "half_quality_mandatory")
    check(policy.get("common_mask_can_rescue") is False, "common_mask_nonrescue")

    aligned_metrics: dict[str, float] | None = None
    masked_metrics: dict[str, float] | None = None
    if curves_path.is_file():
        with np.load(curves_path, allow_pickle=False) as archive:
            archive_fields = set(archive.files)
            check(set(ALIGNED_CURVE_KEYS).issubset(archive_fields), "aligned_curve_fields")
            check(set(MASKED_CURVE_KEYS).issubset(archive_fields), "masked_curve_fields")
            check(set(curve_spec.get("fields", ())) == archive_fields, "curve_fields_envelope")
            check(set(internal_curves.get("fields", ())) == archive_fields, "curve_fields_internal")
            if set((*ALIGNED_CURVE_KEYS, *MASKED_CURVE_KEYS)).issubset(archive_fields):
                curves = {
                    key: _as_curve(np.asarray(archive[key], dtype=np.float64), key)
                    for key in (*ALIGNED_CURVE_KEYS, *MASKED_CURVE_KEYS)
                }
                check(all(curve.size == expected_curve_length for curve in curves.values()), "curve_length")
                if all(curve.size == expected_curve_length for curve in curves.values()):
                    aligned_metrics = {
                        "merged_cross_engine_band_auc": normalized_band_auc(
                            curves[ALIGNED_CURVE_KEYS[0]], first_shell, last_shell
                        ),
                        "half1_cross_engine_band_auc": normalized_band_auc(
                            curves[ALIGNED_CURVE_KEYS[1]], first_shell, last_shell
                        ),
                        "half2_cross_engine_band_auc": normalized_band_auc(
                            curves[ALIGNED_CURVE_KEYS[2]], first_shell, last_shell
                        ),
                    }
                    recovar_masked = curves[MASKED_CURVE_KEYS[1]][first_shell : last_shell + 1]
                    relion_masked = curves[MASKED_CURVE_KEYS[0]][first_shell : last_shell + 1]
                    masked_metrics = {
                        "recovar_half_band_auc": normalized_band_auc(
                            curves[MASKED_CURVE_KEYS[1]], first_shell, last_shell
                        ),
                        "relion_half_band_auc": normalized_band_auc(
                            curves[MASKED_CURVE_KEYS[0]], first_shell, last_shell
                        ),
                        "half_curve_rmse": float(np.sqrt(np.mean((recovar_masked - relion_masked) ** 2))),
                        "half_band_auc_abs_delta": abs(
                            normalized_band_auc(curves[MASKED_CURVE_KEYS[1]], first_shell, last_shell)
                            - normalized_band_auc(curves[MASKED_CURVE_KEYS[0]], first_shell, last_shell)
                        ),
                        "merged_cross_engine_band_auc": normalized_band_auc(
                            curves[MASKED_CURVE_KEYS[2]], first_shell, last_shell
                        ),
                        "half1_cross_engine_band_auc": normalized_band_auc(
                            curves[MASKED_CURVE_KEYS[3]], first_shell, last_shell
                        ),
                        "half2_cross_engine_band_auc": normalized_band_auc(
                            curves[MASKED_CURVE_KEYS[4]], first_shell, last_shell
                        ),
                    }
    aligned_failures: list[str] = []
    if aligned_metrics is not None:
        aligned_failures = apply_cross_engine_gates(aligned_metrics, scorecard["thresholds"])
    if failures:
        status = "invalid"
        can_rescue = False
    elif aligned_metrics is None:
        status = "invalid"
        can_rescue = False
        failures.append("science_diagnostics:aligned_metrics_missing")
    else:
        status = "pass" if not aligned_failures else "fail"
        can_rescue = not aligned_failures
    return {
        "status": status,
        "proper_so3_alignment": {
            "status": status,
            "can_rescue_cross_engine": can_rescue,
            "metrics": aligned_metrics,
            "failed_gates": aligned_failures,
            "rotation_matrix_recovar_to_relion": rotation.tolist() if rotation.shape == (3, 3) else None,
            "translation_recovar_to_relion_zyx": translation.tolist() if translation.shape == (3,) else None,
            "no_reflection": alignment.get("no_reflection"),
            "sign_fit": alignment.get("sign_fit"),
            "scale_fit": alignment.get("scale_fit"),
        },
        "common_mask": {
            "status": "reported" if masked_metrics is not None and not failures else "invalid",
            "metrics": masked_metrics,
            "path": str(mask_path),
            "sha256": mask_spec.get("sha256"),
            "can_rescue": False,
        },
        "artifacts": {
            "science_diagnostics": dict(science_spec),
            "curve_archive": dict(curve_spec),
            "common_mask": dict(mask_spec),
        },
    }, failures


def score_case_evidence(
    scorecard: Mapping[str, Any],
    evidence_path: Path,
) -> dict[str, Any]:
    """Validate and score one target-case evidence envelope."""

    evidence = json.loads(evidence_path.read_text())
    cases = {case["id"]: case for case in scorecard["cases"]}
    case_id = evidence.get("case_id")
    _require(case_id in cases, f"evidence names unknown case {case_id!r}")
    case = cases[case_id]
    _require(case.get("role") == "scoring", "calibration evidence cannot enter scoring denominator")
    provenance_failures = validate_case_evidence_provenance(scorecard, case, evidence)

    collector = evidence.get("collector", {})
    metrics_path = Path(collector.get("metrics_json", ""))
    curves_path = Path(collector.get("fsc_curves_npz", ""))
    if not metrics_path.is_file():
        provenance_failures.append("metrics_json_missing")
    elif sha256_file(metrics_path) != collector.get("metrics_sha256"):
        provenance_failures.append("metrics_json_sha256")
    if not curves_path.is_file():
        provenance_failures.append("fsc_curves_npz_missing")
    elif sha256_file(curves_path) != collector.get("curves_sha256"):
        provenance_failures.append("fsc_curves_npz_sha256")

    scored: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    curve_length: int | None = None
    if metrics_path.is_file() and curves_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        provenance_failures.extend(_validate_collector_metrics(case, metrics))
        with np.load(curves_path, allow_pickle=False) as archive:
            try:
                curves = {key: np.asarray(archive[key], dtype=np.float64) for key in CURVE_KEYS}
                curve_length = int(curves[CURVE_KEYS[0]].size)
            except KeyError as exc:
                provenance_failures.append(f"missing_curve:{exc.args[0]}")
                curves = None
        if curves is not None:
            try:
                scored = score_curves(
                    curves,
                    box_size=int(case["input_contract"]["box_size"]),
                    voxel_size_angstrom=float(case["input_contract"]["voxel_size_angstrom"]),
                    thresholds=scorecard["thresholds"],
                )
            except ValueError as exc:
                provenance_failures.append(f"metric_input:{exc}")

    if scored is not None and curve_length is not None and metrics is not None:
        try:
            science_diagnostics, science_failures = _validate_science_diagnostics(
                scorecard,
                case,
                evidence,
                metrics,
                expected_producer_sha256=scorecard["diagnostic_producer"]["sha256"],
                first_shell=int(scored["jointly_resolved_band"]["first_shell"]),
                last_shell=int(scored["jointly_resolved_band"]["last_shell"]),
                expected_curve_length=curve_length,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            science_diagnostics = {
                "status": "invalid",
                "proper_so3_alignment": {"status": "invalid", "can_rescue_cross_engine": False},
                "common_mask": {"status": "invalid", "can_rescue": False},
            }
            science_failures = [f"science_diagnostics:malformed:{exc}"]
        provenance_failures.extend(science_failures)
    else:
        science_diagnostics = {
            "status": "not_scored",
            "proper_so3_alignment": {"status": "not_scored", "can_rescue_cross_engine": False},
            "common_mask": {"status": "not_scored", "can_rescue": False},
        }
    legacy_diagnostics = evidence.get("diagnostics")
    if legacy_diagnostics is None:
        legacy_diagnostics = {"status": "not_supplied"}
    legacy_diagnostics = {**legacy_diagnostics, "can_rescue_cross_engine": False}
    provenance_failures = list(dict.fromkeys(provenance_failures))
    high_resolution_achievement: dict[str, Any]
    if scored is None:
        high_resolution_achievement = {
            "status": "not_scored",
            "pass": False,
            "each_engine_resolution_angstrom_max": TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM,
            "reference": dict(scorecard["target_high_resolution_gate"]["reference"]),
        }
    else:
        band = scored["jointly_resolved_band"]
        recovar_resolution = float(band["recovar_resolution_angstrom"])
        relion_resolution = float(band["relion_resolution_angstrom"])
        recovar_pass = recovar_resolution <= TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM
        relion_pass = relion_resolution <= TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM
        high_resolution_pass = recovar_pass and relion_pass
        high_resolution_achievement = {
            "status": "pass" if high_resolution_pass else "fail",
            "pass": high_resolution_pass,
            "each_engine_resolution_angstrom_max": TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM,
            "recovar": {"resolution_angstrom": recovar_resolution, "pass": recovar_pass},
            "relion": {"resolution_angstrom": relion_resolution, "pass": relion_pass},
            "reference": dict(scorecard["target_high_resolution_gate"]["reference"]),
            "independent_of_science_equivalence": True,
        }

    if provenance_failures or scored is None:
        status = "invalid"
        failed_gates: list[str] = [] if scored is None else list(scored["half_map_quality_failed_gates"])
        equivalence_route = None
        science_equivalence_pass: bool | None = None
    else:
        half_map_failures = list(scored["half_map_quality_failed_gates"])
        raw_cross_engine_pass = bool(scored["raw_cross_engine_pass"])
        aligned_cross_engine_pass = bool(
            science_diagnostics["proper_so3_alignment"].get("can_rescue_cross_engine") is True
        )
        if raw_cross_engine_pass:
            equivalence_route = "raw_canonical"
        elif aligned_cross_engine_pass:
            equivalence_route = "continuous_proper_so3_rigid"
        else:
            equivalence_route = None
        failed_gates = [*half_map_failures]
        if equivalence_route is None:
            failed_gates.append("cross_engine_equivalence")
        science_equivalence_pass = not failed_gates
        if not high_resolution_achievement["pass"]:
            failed_gates.append("target_high_resolution")
        status = "pass" if not failed_gates else "fail"
    result = {
        "id": case["id"],
        "role": "scoring",
        "provenance_failures": provenance_failures,
        "science_diagnostics": science_diagnostics,
        "legacy_collector_diagnostics": legacy_diagnostics,
        "equivalence_route": equivalence_route,
        "science_equivalence_pass": science_equivalence_pass,
        "high_resolution_achievement": high_resolution_achievement,
        "target_overall_requires_equivalence_and_high_resolution": True,
        "half_map_quality_is_mandatory": True,
        "masked_fsc_can_rescue": False,
        **({} if scored is None else scored),
        "evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": sha256_file(evidence_path),
        },
    }
    result.update(
        status=status,
        primary_pass=status == "pass",
        failed_gates=failed_gates,
    )
    return result


def _frozen_case_rows(scorecard: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in scorecard["cases"]:
        if case["role"] == "calibration":
            half_failures = apply_half_map_quality_gates(case["expected_metrics"], scorecard["thresholds"])
            route = case["expected_cross_engine_route"]
            diagnostic_spec = case.get("proper_so3_diagnostics")
            science_diagnostics = (
                {
                    "status": "frozen_not_replayed",
                    "proper_so3_alignment": {
                        "status": "frozen_not_replayed",
                        "can_rescue_cross_engine": diagnostic_spec["expected_can_rescue_cross_engine"],
                        "metrics": diagnostic_spec["expected_metrics"],
                    },
                    "artifacts": diagnostic_spec["artifacts"],
                }
                if diagnostic_spec is not None
                else {"status": "not_supplied"}
            )
            rows.append(
                {
                    "id": case["id"],
                    "role": "calibration",
                    "status": "pass" if not half_failures else "fail",
                    "subject_commit": case["subject_commit"],
                    "primary_metrics": case["expected_metrics"],
                    "primary_pass": not half_failures and route != "none_unqualified",
                    "half_map_quality_pass": not half_failures,
                    "failed_gates": [
                        *half_failures,
                        *([] if route != "none_unqualified" else ["cross_engine_equivalence"]),
                    ],
                    "cross_engine_route": route,
                    "science_equivalence_pass": route != "none_unqualified",
                    "science_diagnostics": science_diagnostics,
                }
            )
        else:
            partial = case["partial_engine_results"]
            relion_partial = partial["relion"]
            rows.append(
                {
                    "id": case["id"],
                    "role": "scoring",
                    "status": "pending",
                    "pending_contract_fields": _required_pending_contract_fields(case),
                    "primary_metrics": None,
                    "failed_gates": [],
                    "science_diagnostics": {"status": "not_supplied"},
                    "equivalence_route": None,
                    "science_equivalence_pass": None,
                    "high_resolution_achievement": {
                        "status": "pending",
                        "pass": None,
                        "each_engine_resolution_angstrom_max": TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM,
                        "relion": {
                            "status": "complete",
                            "resolution_angstrom": relion_partial["unmasked"]["resolution_angstrom"],
                            "pass": relion_partial["unmasked"]["high_resolution_gate_pass"],
                        },
                        "recovar": {"status": "pending", "resolution_angstrom": None, "pass": None},
                        "partial_result_does_not_score_case": True,
                    },
                    "partial_engine_results": {
                        "verification_status": "frozen_not_replayed",
                        "relion": dict(relion_partial),
                        "recovar": dict(partial["recovar"]),
                    },
                    "interim_iteration11_diagnostic": {
                        "verification_status": "frozen_not_replayed",
                        **dict(case["interim_iteration11_diagnostic"]),
                    },
                }
            )
    return rows


def build_report(
    scorecard: Mapping[str, Any],
    *,
    scorecard_path: Path,
    verify_calibrations: bool = False,
    verify_masked_support: bool = False,
    verify_target_partial: bool = False,
    evidence_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Build a fixed-denominator report from frozen and optionally live evidence."""

    rows = _frozen_case_rows(scorecard)
    by_id = {row["id"]: row for row in rows}
    case_defs = {case["id"]: case for case in scorecard["cases"]}
    if verify_calibrations:
        for case_id in CALIBRATION_CASE_IDS:
            by_id[case_id] = replay_calibration_case(scorecard, case_defs[case_id])
    if verify_target_partial:
        by_id[TARGET_CASE_ID]["partial_engine_results"] = replay_target_partial_engine_results(
            case_defs[TARGET_CASE_ID]
        )
        by_id[TARGET_CASE_ID]["interim_iteration11_diagnostic"] = replay_target_interim_iteration11_diagnostic(
            case_defs[TARGET_CASE_ID]
        )
    for evidence_path in evidence_paths:
        scored = score_case_evidence(scorecard, evidence_path)
        if scored["id"] == TARGET_CASE_ID:
            scored["partial_engine_results"] = by_id[TARGET_CASE_ID]["partial_engine_results"]
            scored["interim_iteration11_diagnostic"] = by_id[TARGET_CASE_ID]["interim_iteration11_diagnostic"]
        by_id[scored["id"]] = scored
    rows = [by_id[case["id"]] for case in scorecard["cases"]]

    scoring = [row for row in rows if row["role"] == "scoring"]
    aggregate = {
        "scoring_denominator": 1,
        "passed": sum(row["status"] == "pass" for row in scoring),
        "failed": sum(row["status"] == "fail" for row in scoring),
        "pending": sum(row["status"] == "pending" for row in scoring),
        "invalid": sum(row["status"] == "invalid" for row in scoring),
    }
    masked_support = (
        replay_masked_fsc_support(scorecard)
        if verify_masked_support
        else {
            "status": "frozen_not_replayed",
            "role": "supporting_only",
            "acceptance_metric": False,
            "can_rescue": False,
            "artifacts": scorecard["masked_fsc_support"]["artifacts"],
            "producer_jobs": scorecard["masked_fsc_support"]["producer_jobs"],
            "expected_corrected_metrics": scorecard["masked_fsc_support"]["expected_corrected_metrics"],
        }
    )
    reproduction = (
        replay_reproduction_contract(scorecard["reproduction"])
        if verify_calibrations and verify_masked_support
        else {"verification_status": "frozen_not_replayed", **scorecard["reproduction"]}
    )
    return {
        "schema": REPORT_SCHEMA,
        "suite": scorecard["suite"],
        "scorecard": {
            "path": str(scorecard_path.resolve()),
            "sha256": sha256_file(scorecard_path),
        },
        "metric_policy": {
            "half_map_quality": "mandatory unmasked unaligned within-engine split-half FSC",
            "cross_engine_acceptance": "raw canonical FSC OR pinned continuous proper-SO(3)+translation FSC",
            "proper_alignment_can_rescue_cross_engine_only": True,
            "masked_fsc_is_supporting_only": True,
            "mirror_sign_scale_forbidden": True,
            "strict_full_spectrum_parity_is_separate": True,
            "target_high_resolution_is_independent": True,
            "target_overall_requires_equivalence_and_high_resolution": True,
        },
        "thresholds": scorecard["thresholds"],
        "target_high_resolution_gate": scorecard["target_high_resolution_gate"],
        "cases": rows,
        "aggregate": aggregate,
        "masked_fsc_support": masked_support,
        "reproduction": reproduction,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact checked scorecard and its rigid-alignment policy."""

    thresholds = report["thresholds"]
    cases = {case["id"]: case for case in report["cases"]}
    lines = [
        "# K=1 real-data science-equivalence scorecard",
        "",
        "This fixed scorecard is separate from strict full-spectrum RELION numerical",
        "parity. Comparable unmasked, unaligned within-engine half-map FSC is always",
        "mandatory. Cross-engine equivalence can pass directly in the canonical frame",
        "or through one pinned continuous proper-SO(3) rotation and translation fitted",
        "from low-frequency merged maps and applied unchanged to both split halves.",
        "Reflection, density-sign, and scale fitting are forbidden. A common-mask FSC",
        "is reported as supporting evidence only and can never rescue a failure.",
        "",
        "## Frozen primary gates",
        "",
        "| Gate | Threshold |",
        "| --- | ---: |",
        f"| Half-map resolution ratio | <= {thresholds['half_resolution_ratio_max']:.2f} |",
        f"| Half-FSC curve RMSE in the jointly resolved band | <= {thresholds['half_curve_rmse_max']:.2f} |",
        f"| Absolute half-FSC band-AUC difference | <= {thresholds['half_band_auc_abs_delta_max']:.2f} |",
        f"| Merged cross-engine band FSC-AUC, raw canonical **or** proper-rigid route | >= {thresholds['merged_cross_engine_band_auc_min']:.2f} |",
        f"| Each cross-engine half-map band FSC-AUC on the same route | >= {thresholds['each_half_cross_engine_band_auc_min']:.2f} |",
        "",
        "The joint band is shells 1 through one shell before the earlier first",
        "three-shell-sustained crossing below `1/7`. Resolution uses",
        "`box_size * voxel_size_angstrom / crossing_shell`.",
        "",
        "## Frozen calibration replay",
        "",
        "The completed 10073, 10345, and 10097 runs calibrate the metric only. They",
        "ran descendant commits and do not count in the PR #158 scoring denominator.",
        "A calibration status is determined only by the mandatory unmasked, unaligned",
        "within-engine half-map gates; cross-engine qualification is reported separately.",
        "",
        "| Case | Half-map calibration | Half RMSE | Half AUC delta | Raw merged/min-half AUC | Proper merged/min-half AUC | Qualified route |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case_id in CALIBRATION_CASE_IDS:
        row = cases[case_id]
        metrics = row["primary_metrics"]
        alignment = row.get("science_diagnostics", {}).get("proper_so3_alignment", {})
        aligned_metrics = alignment.get("metrics") or {}
        aligned_merged = aligned_metrics.get("merged_cross_engine_band_auc")
        aligned_min_half = (
            min(
                aligned_metrics["half1_cross_engine_band_auc"],
                aligned_metrics["half2_cross_engine_band_auc"],
            )
            if aligned_metrics
            else None
        )
        proper_text = "--" if aligned_merged is None else f"{_fmt(aligned_merged)}/{_fmt(aligned_min_half)}"
        lines.append(
            f"| `{case_id}` | {row['status']} | {_fmt(metrics['half_curve_rmse'])} | "
            f"{_fmt(metrics['half_band_auc_abs_delta'])} | "
            f"{_fmt(metrics['merged_cross_engine_band_auc'])}/"
            f"{_fmt(min(metrics['half1_cross_engine_band_auc'], metrics['half2_cross_engine_band_auc']))} | "
            f"{proper_text} | `{row['cross_engine_route']}` |"
        )
    lines.extend(
        [
            "",
            "These are deliberately parity-calibration refinements, not reproductions of",
            "the deposited publication workflows. Both engines crossed the",
            "three-consecutive-shell unmasked half-map FSC 1/7 threshold at shell 81 on",
            "10073 and shell 49 on 10345. Under the frozen resolution formula, these are",
            "6.568 A and 8.235 A, respectively. On 10097, RECOVAR and RELION cross at",
            "shells 44 and 45 (7.622 A and 7.452 A). These values are worse than the",
            "3.7 A deposited",
            "resolution for [EMD-8012](https://www.ebi.ac.uk/emdb/EMD-8012) and the 3.51 A",
            "focused resolution for [EMD-20795](https://www.ebi.ac.uk/emdb/EMD-20795),",
            "which also deposits a 3.8 A sharpened full-complex map.",
            "",
            "The absolute gap is expected from the frozen calibration protocol. For 10073,",
            "normalization intentionally drops the supplied refined Euler angles and",
            "origins. The 10345 source STAR has no Euler columns and only zero or invalid",
            "in-plane origins. Both engines therefore start from the same newly generated",
            "de-novo K=1 model, and the reported maps and FSCs are unmasked, unsharpened,",
            "and unpostprocessed. The deposited 10073 workflow instead used EMD-2966",
            "low-pass filtered to 60 A. The exact published 10345 complex did not use 3D",
            "classification, but its final maps used non-uniform and local-resolution",
            "refinement, local-resolution estimation, sharpening, and local filtering;",
            "the deposited primary map is a focused refinement. C1 is the appropriate",
            "symmetry for 10073 and 10345 and is not the cause of that gap.",
            "",
            "The 10073 and 10345 cases establish that RECOVAR and RELION reach essentially",
            "the same reconstruction under the matched protocol. The 10097 within-engine",
            "half-map comparison also passes strongly, but its raw cross-engine AUCs",
            "(0.935736 merged; 0.885803/0.888938 halves) miss the frozen cross-engine",
            "gates. Corrected job 13276576 tested the allowed proper-SO(3)+translation",
            "route after explicitly adding canonical identity to the HEALPix seed set.",
            "It fitted only a 0.244-degree rotation and 0.084-voxel translation, but its",
            "0.935427/0.885517/0.888483 aligned AUCs still miss the same gates. The route",
            "is therefore recorded as unqualified and does not rescue 10097; a small",
            "global rigid drift does not explain the residual cross-engine difference.",
            "Job 13275901 is retained only as a superseded audit artifact because its",
            "seed search omitted identity and selected a false distant orientation.",
            "",
            "These calibration results do not establish",
            "that this intentionally stripped-down protocol reproduces the published",
            "reconstruction. Absolute high-resolution achievement is tested separately by",
            "the frozen 10202 case below.",
        ]
    )
    masked_support = report["masked_fsc_support"]
    masked_metrics = masked_support["expected_corrected_metrics"]
    aggregate_artifact = masked_support["artifacts"]["aggregate_summary"]
    lines.extend(
        [
            "",
            "## Supporting RELION corrected-masked FSC",
            "",
            "These measurements are supporting-only: they do not enter any acceptance",
            "gate, cannot rescue an unmasked failure, and do not change the scoring",
            "denominator. Each mask was generated only from the RELION merged map and",
            "then passed byte-for-byte to both engines' independent half-map postprocess.",
            "",
            "| Dataset | RECOVAR corrected masked (A; shell) | RELION corrected masked (A; shell) | Curve RMSE | AUC delta | Mask SHA-256 prefix |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for dataset in MASKED_SUPPORT_DATASET_IDS:
        metrics = masked_metrics[dataset]
        lines.append(
            f"| {dataset} | {_fmt(metrics['recovar_resolution_angstrom'], 3)}; {metrics['recovar_crossing_shell']} | "
            f"{_fmt(metrics['relion_resolution_angstrom'], 3)}; {metrics['relion_crossing_shell']} | "
            f"{_fmt(metrics['curve_rmse'])} | {_fmt(metrics['band_auc_absolute_delta'])} | "
            f"`{metrics['mask_sha256'][:12]}` |"
        )
    lines.extend(
        [
            "",
            "RELION first low-pass filtered each merged map to 15 A, then used",
            "`relion_mask_create --extend_inimask 5 --width_soft_edge 8`. Both",
            "postprocess calls used `--force_mask --skip_fsc_weighting --low_pass 0",
            "--randomize_at_fsc 0.8 --random_seed 42`. Exact per-dataset argv, input",
            "hashes, mask thresholds, and output hashes are sealed in the aggregate",
            f"artifact with SHA-256 `{aggregate_artifact['sha256']}`.",
            "",
            "Producer jobs were `13273806` for 10073/10345 and `13274377` for",
            "10097; requested and allocated resources matched. The first job's nonzero",
            "state occurred only after its two retained datasets, at the later 10097",
            "component audit. The isolated second job reused the literal audited 10097",
            "mask. Regenerating that mask changed only MRC header-statistic bytes 249,",
            "250, and 253; the voxel payload was identical.",
        ]
    )
    target = cases[TARGET_CASE_ID]
    reproduction = report["reproduction"]
    unmasked_reproduction = reproduction["unmasked"]
    masked_reproduction = reproduction["masked"]
    if target["status"] == "pending":
        partial = target["partial_engine_results"]
        interim = target["interim_iteration11_diagnostic"]
        partial_relion = partial["relion"]
        partial_recovar = partial["recovar"]
        partial_jobs = partial_relion["jobs"]
        partial_artifacts = partial_relion["artifacts"]
        lines.extend(
            [
                "",
                "## Available EMPIAR-10202 per-engine evidence",
                "",
                "This is a deliberately partial report. RELION is sealed and complete;",
                "RECOVAR and every cross-engine acceptance metric remain pending.",
                "The RELION-only result cannot pass the fixed scoring case.",
                "",
                "| Engine | Status | Unmasked FSC=0.143 (A) | Corrected masked FSC=0.143 (A) | <= 3.0 A arm | Jobs |",
                "| --- | --- | ---: | ---: | --- | --- |",
                f"| RELION | {partial_relion['status']} | {_fmt(partial_relion['unmasked']['resolution_angstrom'])} | "
                f"{_fmt(partial_relion['corrected_masked']['resolution_angstrom'])} | "
                f"{'pass' if partial_relion['unmasked']['high_resolution_gate_pass'] else 'fail'} | "
                f"`{partial_jobs['refinement']['job_id']}` / `{partial_jobs['postprocess']['job_id']}` |",
                f"| RECOVAR | {partial_recovar['status']} | -- | -- | pending | -- |",
                "",
                "The 2.511554-A value is RELION's final unmasked FSC estimate sealed by",
                "the postprocess result manifest. The 2.122559-A corrected masked value is",
                "supporting-only and cannot rescue an unmasked or cross-engine failure.",
                "The fixed scorecard's three-consecutive-shell joint-band metric is still",
                "unavailable until RECOVAR supplies its independent half maps.",
                "",
                f"RELION refinement stdout SHA-256: `{partial_artifacts['refinement_stdout']['sha256']}`.",
                f"Matched harness manifest SHA-256: `{partial_artifacts['launch_manifest']['sha256']}`.",
                f"Postprocess result-manifest SHA-256: `{partial_artifacts['postprocess_result_manifest']['sha256']}`.",
                f"Common mask SHA-256: `{partial_artifacts['common_mask']['sha256']}`.",
                "",
                "### Non-scoring matched iteration-11 checkpoint",
                "",
                "The interrupted RECOVAR trajectory completed iteration 11 before a CUDA",
                "out-of-memory failure in iteration 12. This checkpoint cannot score the",
                "case, but RELION iteration 11 was postprocessed with the identical mask,",
                "executable, and FSC convention. The same-iteration resolution crossings",
                "are identical:",
                "",
                "| FSC | RECOVAR (A; shell) | RELION (A; shell) | Resolved-band RMSE | Resolved-band AUC delta |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Raw unmasked | {_fmt(interim['same_iteration_resolutions']['raw_unmasked']['recovar_angstrom'])}; 250 | "
                f"{_fmt(interim['same_iteration_resolutions']['raw_unmasked']['relion_angstrom'])}; 250 | "
                f"{_fmt(interim['resolved_band_curve_comparison']['raw_unmasked']['rmse'])} | "
                f"{_fmt(interim['resolved_band_curve_comparison']['raw_unmasked']['normalized_auc_absolute_delta'])} |",
                f"| Corrected masked (supporting only) | {_fmt(interim['same_iteration_resolutions']['corrected_masked']['recovar_angstrom'])}; 248 | "
                f"{_fmt(interim['same_iteration_resolutions']['corrected_masked']['relion_angstrom'])}; 248 | "
                f"{_fmt(interim['resolved_band_curve_comparison']['corrected_masked']['rmse'])} | "
                f"{_fmt(interim['resolved_band_curve_comparison']['corrected_masked']['normalized_auc_absolute_delta'])} |",
                "",
                "The raw comparison uses shells 1--249, ending immediately before the",
                "shared three-shell-sustained crossing. The corrected-masked comparison",
                "uses its own shells 1--247 band. This is strong evidence that RECOVAR had",
                "already reached RELION's same-iteration half-map quality, but terminal",
                "equivalence and the <=3.0-A gate remain pending until an uninterrupted",
                "trajectory produces sealed final half maps.",
                "",
                f"RECOVAR checkpoint/postprocess jobs: `{interim['recovar']['trajectory_job_id']}` / "
                f"`{interim['recovar']['postprocess_job_id']}`. Matched RELION iteration-11",
                f"postprocess job: `{interim['relion']['postprocess_job_id']}`. Replacement",
                f"full-run job `{interim['replacement_full_run']['job_id']}` at subject commit "
                f"`{interim['replacement_full_run']['subject_commit'][:12]}` failed in numbered iteration "
                f"{interim['replacement_full_run']['failed_numbered_iteration']} at current size "
                f"{interim['replacement_full_run']['failed_current_size']} with a CUDA OOM after completing "
                f"numbered iteration {interim['replacement_full_run']['last_complete_numbered_iteration']}.",
                "The failure is recorded as a compact-planner routing/headroom defect,",
                "not as a scientific-resolution failure. The fail-closed compact-planner",
                f"successor was captured running in job `{interim['replacement_full_run']['successor_validation']['job_id']}`",
                f"at commit `{interim['replacement_full_run']['successor_validation']['subject_commit'][:12]}`, "
                f"after completing numbered iteration {interim['replacement_full_run']['successor_validation']['last_complete_numbered_iteration']}.",
                "",
                f"Matched iteration-11 summary SHA-256: `{interim['artifacts']['matched_relion_summary']['sha256']}`.",
                f"Resolved curve comparison SHA-256: `{interim['artifacts']['matched_relion_curve_comparison']['sha256']}`.",
            ]
        )
    elif target["status"] in {"pass", "fail"}:
        lines.extend(
            [
                "",
                "## Submitted EMPIAR-10202 scoring evidence",
                "",
                "Both engines are represented by the submitted target evidence. Its",
                f"terminal status is `{target['status']}`; the scored unmasked half-map",
                "resolutions and cross-engine FSC metrics are reported below. The frozen",
                "RELION-only partial record is retained for provenance but is not rendered",
                "as the current result.",
            ]
        )
    else:
        provenance_failures = target.get("provenance_failures") or []
        failure_text = ", ".join(f"`{failure}`" for failure in provenance_failures)
        if not failure_text:
            failure_text = "`unspecified_fail_closed_validation_error`"
        lines.extend(
            [
                "",
                "## Rejected EMPIAR-10202 scoring evidence",
                "",
                "The submitted target evidence failed fail-closed validation and was",
                "not admitted as a two-engine scientific result. Any populated diagnostic",
                "values below are non-scoring. The frozen RELION-only partial record is",
                "retained for provenance but is not rendered as the current result.",
                "",
                f"Validation failures: {failure_text}.",
            ]
        )
    target_metrics = target.get("primary_metrics") or {}
    target_band = target.get("jointly_resolved_band") or {}
    target_alignment = target.get("science_diagnostics", {}).get("proper_so3_alignment", {})
    target_aligned_metrics = target_alignment.get("metrics") or {}
    high_resolution = target.get("high_resolution_achievement") or {}
    equivalence_status = target.get("science_equivalence_pass")
    equivalence_text = "--" if equivalence_status is None else ("pass" if equivalence_status else "fail")
    route = target.get("equivalence_route") or "--"
    high_resolution_status = high_resolution.get("status", "pending")
    if target["status"] == "pending":
        target_completion_lines = [
            "ancestry is insufficient. The RELION arm is complete; the case remains",
            "pending until the RECOVAR full refinement and sealed two-engine FSC",
            "analysis complete.",
        ]
    elif target["status"] == "invalid":
        target_completion_lines = [
            "ancestry is insufficient. The RELION arm is complete, but the submitted",
            "RECOVAR/two-engine evidence was rejected by the fail-closed validation",
            "reported above.",
        ]
    else:
        target_completion_lines = [
            "ancestry is insufficient. Both engine arms and the sealed two-engine FSC",
            f"analysis are complete; the terminal scoring status is `{target['status']}`.",
        ]
    lines.extend(
        [
            "",
            "## Fixed scoring case",
            "",
            "Equivalence and absolute high-resolution achievement are reported",
            "separately. The target passes overall only when both pass.",
            "",
            "| Case | Overall | Equivalence | Route | High resolution | RECOVAR half FSC (A) | RELION half FSC (A) | Raw merged AUC | Proper merged AUC | Half RMSE |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            f"| `{TARGET_CASE_ID}` | {target['status']} | {equivalence_text} | `{route}` | {high_resolution_status} | "
            f"{_fmt(target_band.get('recovar_resolution_angstrom'), 3)} | "
            f"{_fmt(target_band.get('relion_resolution_angstrom'), 3)} | "
            f"{_fmt(target_metrics.get('merged_cross_engine_band_auc'))} | "
            f"{_fmt(target_aligned_metrics.get('merged_cross_engine_band_auc'))} | "
            f"{_fmt(target_metrics.get('half_curve_rmse'))} |",
            "",
            "The independent high-resolution gate requires **each** engine's unmasked",
            f"half-map FSC resolution to be <= {TARGET_HIGH_RESOLUTION_CEILING_ANGSTROM:.1f} A. For context,",
            f"{TARGET_REFERENCE_ENTRY} records {TARGET_DEPOSITED_VALIDATION_RESOLUTION_ANGSTROM:.2f} A deposited validation and",
            f"{TARGET_EMDB_UNMASKED_HALF_MAP_RESOLUTION_ANGSTROM:.2f} A EMDB-calculated unmasked half-map resolution.",
            "These reference values provide context; they are not substituted for either",
            "engine's measured result.",
            "",
            "The deposited 30,515-particle half split is already frozen: 15,258 in",
            "half 1 and 15,257 in half 2, hashed as one `uint8` `rlnRandomSubset`",
            "value per source-STAR row. STAR normalization must preserve particle",
            "order, deposited halves, Euler angles, and origins.",
            "The complete 78,118,401,024-byte stack is pinned by SHA-256 prefix",
            "`8eecf0fb` (not merely by its MRC header).",
            "",
            "The older `particles.native.star` artifact with SHA-256 prefix",
            "`c13cb927` is explicitly rejected because it rerandomized halves and",
            "dropped deposited pose/shift metadata.",
            "",
            "This deposited/FREALIGN frame requires explicit `I1`; bare RELION `I`",
            "canonicalizes to `I2` and is forbidden here. The frozen operator sequence",
            "is identity followed by RELION `SymList` order; its rounded little-endian",
            "`(left, right)` float64 digest begins `093a0876`.",
            "",
            "The preparation contract is fully frozen: normalized STAR SHA-256 prefix",
            "`d66afb30`; RELION/RECOVAR initial-map file prefixes `4f83710c` and",
            "`d77516a0`; exact shared canonical-array prefix `b617f90d`. There are no",
            "pending preparation hashes. The subject commit must match exactly;",
            *target_completion_lines,
            "",
            "## Reproduction and artifact replay",
            "",
            "From the repository root, this command re-hashes and replays every completed",
            "10073/10345/10097 unmasked and masked artifact, verifies the partial and",
            "matched-iteration 10202 records, and checks that this generated Markdown is fresh:",
            "",
            "```bash",
            reproduction["artifact_replay_command"],
            "```",
            "",
            "The original unmasked producer submissions are recorded verbatim in their",
            "sealed `SUBMITTED_JOBS.md` files:",
            "",
            "```bash",
            *unmasked_reproduction["10073_10345"]["recorded_submission_commands"],
            *unmasked_reproduction["10097"]["recorded_submission_commands"],
            "```",
            "",
            "| Evidence | Frozen producer/collector reference | SHA-256 prefix |",
            "| --- | --- | --- |",
            f"| 10073/10345 unmasked launcher | `{unmasked_reproduction['10073_10345']['launcher']['path']}` | "
            f"`{unmasked_reproduction['10073_10345']['launcher']['sha256'][:12]}` |",
            f"| 10073/10345 submission record | `{unmasked_reproduction['10073_10345']['submission_record']['path']}` | "
            f"`{unmasked_reproduction['10073_10345']['submission_record']['sha256'][:12]}` |",
            f"| 10097 unmasked launcher | `{unmasked_reproduction['10097']['launcher']['path']}` | "
            f"`{unmasked_reproduction['10097']['launcher']['sha256'][:12]}` |",
            f"| 10097 submission record | `{unmasked_reproduction['10097']['submission_record']['path']}` | "
            f"`{unmasked_reproduction['10097']['submission_record']['sha256'][:12]}` |",
            f"| Signed-FSC collector | `{unmasked_reproduction['10073_10345']['collector']['path']}` | "
            f"`{unmasked_reproduction['10073_10345']['collector']['sha256'][:12]}` |",
            f"| 10073/10345 masked launcher | `{masked_reproduction['10073_10345']['launcher']['path']}` | "
            f"`{masked_reproduction['10073_10345']['launcher']['sha256'][:12]}` |",
            f"| 10073/10345 masked driver | `{masked_reproduction['10073_10345']['driver']['path']}` | "
            f"`{masked_reproduction['10073_10345']['driver']['sha256'][:12]}` |",
            f"| 10097 masked launcher | `{masked_reproduction['10097']['launcher']['path']}` | "
            f"`{masked_reproduction['10097']['launcher']['sha256'][:12]}` |",
            f"| 10097 masked driver | `{masked_reproduction['10097']['driver']['path']}` | "
            f"`{masked_reproduction['10097']['driver']['sha256'][:12]}` |",
            f"| Masked aggregate builder | `{masked_reproduction['aggregate']['builder']['path']}` | "
            f"`{masked_reproduction['aggregate']['builder']['sha256'][:12]}` |",
            "",
            "No original `sbatch` argv was separately sealed for the two masked-FSC jobs,",
            "so none is reconstructed here. Their exact launchers and Python drivers are",
            "pinned above, while the repository replay command verifies their retained",
            "outputs without launching new science jobs.",
            "",
            "## Diagnostics",
            "",
            "The pinned producer searches a HEALPix order-1 proper-rotation grid, refines",
            "at order 2, and then continuously refines a rotation vector and subpixel",
            "translation on a 65-cubed compact fit using full-box shells through 32.",
            "Its single transform is reported and",
            "applied unchanged to merged, half-1, and half-2 maps. Aligned unmasked FSC",
            "may replace only failed raw cross-engine gates; it cannot change the three",
            "mandatory half-map-quality gates.",
            "",
            "The producer also constructs one engine-symmetric soft mask from the two",
            "aligned merged maps, hashes it, and applies it identically to both engines.",
            "Masked within-engine and cross-engine FSC values are included in the report",
            "but are never read by any acceptance gate, so masking cannot conceal poor",
            "independent half-map quality.",
            "All six diagnostic input-map hashes must exactly match the six corresponding",
            "hashes in the external collector. Production evidence also binds and hashes",
            "the launch manifest, finalizer command and canonical argv, and a separate",
            "immutable execution envelope; any supplied binding is checked fail-closed.",
            "",
            "## Code references",
            "",
            "- `scripts/summarize_em_k1_realdata_science_equivalence.py`: scorecard validation, FSC band metrics, provenance gates, and rendering.",
            "- `scripts/collect_em_k1_science_diagnostics.py`: continuous proper-SO(3)+translation fitting and common-mask FSC artifacts.",
            "- `tests/unit/test_summarize_em_k1_realdata_science_equivalence.py`: deterministic metric, provenance, calibration, and non-rescue tests.",
            "- `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/scripts/collect_metrics.py`: external signed-FSC artifact collector pinned by SHA-256 in the manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--verify-calibrations", action="store_true")
    parser.add_argument("--verify-masked-support", action="store_true")
    parser.add_argument("--verify-target-partial", action="store_true")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--check-markdown", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scorecard_path = args.scorecard.resolve()
    scorecard = load_and_validate_scorecard(scorecard_path)
    report = build_report(
        scorecard,
        scorecard_path=scorecard_path,
        verify_calibrations=bool(args.verify_calibrations),
        verify_masked_support=bool(args.verify_masked_support),
        verify_target_partial=bool(args.verify_target_partial),
        evidence_paths=tuple(path.resolve() for path in args.evidence),
    )
    markdown = render_markdown(report)
    if args.check_markdown:
        _require(DEFAULT_MARKDOWN.read_text() == markdown, f"checked Markdown is stale: {DEFAULT_MARKDOWN}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)
    if args.output_json is None and args.output_markdown is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    supplied_scoring = [case for case in report["cases"] if case["role"] == "scoring" and args.evidence]
    return 0 if all(case["status"] == "pass" for case in supplied_scoring) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
