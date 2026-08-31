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
)
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
    for case_id in CALIBRATION_CASE_IDS:
        case = by_id[case_id]
        _require(case.get("role") == "calibration", f"{case_id} is not calibration-only")
        _require(_is_git_sha(case.get("subject_commit")), f"{case_id} subject commit is invalid")
        expected = case.get("expected_metrics", {})
        _require(not apply_primary_gates(expected, scorecard["thresholds"]), f"{case_id} no longer passes")
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
    _require(scored["primary_pass"], f"{case['id']} calibration no longer passes")
    return {
        "id": case["id"],
        "role": "calibration",
        "status": "pass",
        "subject_commit": case["subject_commit"],
        **scored,
        "science_diagnostics": {"status": "not_supplied"},
        "artifacts": case["artifacts"],
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
        producer.get("sha256") == scorecard["diagnostic_producer"]["sha256"],
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
        == "HEALPix proper-rotation seed plus continuous scipy rotvec Powell and subpixel translation",
        "alignment_method",
    )
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
            failed = apply_primary_gates(case["expected_metrics"], scorecard["thresholds"])
            rows.append(
                {
                    "id": case["id"],
                    "role": "calibration",
                    "status": "pass" if not failed else "fail",
                    "subject_commit": case["subject_commit"],
                    "primary_metrics": case["expected_metrics"],
                    "failed_gates": failed,
                    "science_diagnostics": {"status": "not_supplied"},
                }
            )
        else:
            rows.append(
                {
                    "id": case["id"],
                    "role": "scoring",
                    "status": "pending",
                    "pending_contract_fields": _required_pending_contract_fields(case),
                    "primary_metrics": None,
                    "failed_gates": [],
                    "science_diagnostics": {"status": "not_supplied"},
                }
            )
    return rows


def build_report(
    scorecard: Mapping[str, Any],
    *,
    scorecard_path: Path,
    verify_calibrations: bool = False,
    evidence_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Build a fixed-denominator report from frozen and optionally live evidence."""

    rows = _frozen_case_rows(scorecard)
    by_id = {row["id"]: row for row in rows}
    case_defs = {case["id"]: case for case in scorecard["cases"]}
    if verify_calibrations:
        for case_id in CALIBRATION_CASE_IDS:
            by_id[case_id] = replay_calibration_case(scorecard, case_defs[case_id])
    for evidence_path in evidence_paths:
        scored = score_case_evidence(scorecard, evidence_path)
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
        "The completed 10073 and 10345 runs calibrate the metric only. They ran a",
        "descendant commit and do not count in the PR #158 scoring denominator.",
        "",
        "| Case | Status | Half RMSE | Half AUC delta | Merged band AUC | Minimum half band AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case_id in CALIBRATION_CASE_IDS:
        row = cases[case_id]
        metrics = row["primary_metrics"]
        lines.append(
            f"| `{case_id}` | {row['status']} | {_fmt(metrics['half_curve_rmse'])} | "
            f"{_fmt(metrics['half_band_auc_abs_delta'])} | "
            f"{_fmt(metrics['merged_cross_engine_band_auc'])} | "
            f"{_fmt(min(metrics['half1_cross_engine_band_auc'], metrics['half2_cross_engine_band_auc']))} |"
        )
    lines.extend(
        [
            "",
            "These are deliberately parity-calibration refinements, not reproductions of",
            "the deposited publication workflows. Both engines crossed the",
            "three-consecutive-shell unmasked half-map FSC 1/7 threshold at shell 81 on",
            "10073 and shell 49 on 10345. Under the frozen resolution formula, these are",
            "6.568 A and 8.235 A, respectively. They are worse than the 3.7 A deposited",
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
            "symmetry for both calibration datasets and is not the cause of the gap.",
            "",
            "Accordingly, these cases establish that RECOVAR and RELION reach essentially",
            "the same reconstruction under the matched protocol. They do not establish",
            "that this intentionally stripped-down protocol reproduces the published",
            "reconstruction. Absolute high-resolution achievement is tested separately by",
            "the frozen 10202 case below.",
        ]
    )
    target = cases[TARGET_CASE_ID]
    target_metrics = target.get("primary_metrics") or {}
    target_band = target.get("jointly_resolved_band") or {}
    target_alignment = target.get("science_diagnostics", {}).get("proper_so3_alignment", {})
    target_aligned_metrics = target_alignment.get("metrics") or {}
    high_resolution = target.get("high_resolution_achievement") or {}
    equivalence_status = target.get("science_equivalence_pass")
    equivalence_text = "--" if equivalence_status is None else ("pass" if equivalence_status else "fail")
    route = target.get("equivalence_route") or "--"
    high_resolution_status = high_resolution.get("status", "pending")
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
            "ancestry is insufficient. The case remains pending only until both full",
            "refinements and their sealed FSC analysis complete.",
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
