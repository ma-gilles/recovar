#!/usr/bin/env python3
"""Validate checked-in RECOVAR EM benchmark evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class RegistryValidationError(ValueError):
    """Raised when a benchmark registry record is malformed or inconsistent."""


_SYNTHETIC_NEGATIVE_RECORD_TYPE = "synthetic_kclass_negative_campaign_collection"
_K1_DIRECT_FSC_SCHEMA = "recovar.em.k1_empiar10202_it011_direct_fsc.v1"
_REAL_KCLASS_SINGLE_SEED_SCHEMA = "recovar.em.real_kclass_native_single_seed_diagnostic.v1"
_INLINE_DIAGNOSTIC_ROUTES = {
    _K1_DIRECT_FSC_SCHEMA: "k1_direct_fsc",
    _REAL_KCLASS_SINGLE_SEED_SCHEMA: "real_kclass_single_seed",
}
_INLINE_DIAGNOSTIC_FILENAMES = {
    _K1_DIRECT_FSC_SCHEMA: "k1-empiar10202-it011-direct-fsc-20260903",
    _REAL_KCLASS_SINGLE_SEED_SCHEMA: "real-k4-10345-native10k-seed42001-2f6759608-20260903",
}
_SEPARATELY_VALIDATED_DIAGNOSTIC_SCHEMAS = {
    "recovar.em.real_kclass_compact_pair_threshold_diagnostic.v1",
    "recovar.em.real_kclass_halfmap_multiseed_stability.v1",
    "recovar.em.real_kclass_selected_fine_diagnostic.v1",
    "recovar.em_real_k4_native_coarse_component_diagnostic.v1",
    "recovar.em_real_k4_native_coarse_operand_diagnostic.v1",
    "recovar.em_real_k4_native_coarse_score_diagnostic.v1",
    "recovar.em_real_k4_native_signfix_causal_diagnostic.v1",
    "recovar.em_real_kclass_initialmodel_diagnostics.v1",
    "recovar.em_real_kclass_offset_prior_fullpairs.v1",
}


def _diagnostic_registry_route(diagnostic: dict[str, Any]) -> str:
    """Route each diagnostic family to exactly one fail-closed validator."""

    if diagnostic.get("record_type") == _SYNTHETIC_NEGATIVE_RECORD_TYPE:
        return "synthetic_negative"
    external_schema = diagnostic.get("schema")
    if external_schema in _INLINE_DIAGNOSTIC_ROUTES:
        return _INLINE_DIAGNOSTIC_ROUTES[external_schema]
    if external_schema in _SEPARATELY_VALIDATED_DIAGNOSTIC_SCHEMAS:
        return "separate"
    raise RegistryValidationError(
        "unrecognized diagnostic family: expected record_type "
        f"{_SYNTHETIC_NEGATIVE_RECORD_TYPE!r}, one of the inline schemas "
        f"{sorted(_INLINE_DIAGNOSTIC_ROUTES)!r}, or one of the separately "
        f"validated schemas {sorted(_SEPARATELY_VALIDATED_DIAGNOSTIC_SCHEMAS)!r}"
    )


def _json_path(parts: list[Any]) -> str:
    return ".".join(str(part) for part in parts) or "<record>"


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_K1_SOURCE = {
    "recovar_trajectory_commit": "4ea288467debbcba97cfaee6d2742ebc66c637f0",
    "analysis_commit": "47f8fe79d5556cf43ad921db503afa59929a6d56",
    "analysis_tree": "cb36870934dd846c10af646f400e508b6ba51c6a",
    "analysis_clean": True,
}
_K1_RUN_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "10202_it11_direct_fsc_20260903T040436Z"
)
_K1_ARTIFACT_SEALS = {
    "full_metrics": (
        "outputs/raw/raw_direct_metrics.json",
        "a412be7ebcb713a01a577548c92115a3c7bb192fbff23106cf4cf8b479c642ca",
    ),
    "full_curves": (
        "outputs/raw/raw_direct_curves.npz",
        "0d6fb5112fd5c2fff887753003ac6041640b57b672684870229d70cd430de7fc",
    ),
    "half_spectrum_metrics": (
        "outputs/raw_rfft_preview/raw_rfft_preview_metrics.json",
        "f6aa025df37c68e5f89c1e5264759981d0a0a5bfccee28599ce230ee1faf7b54",
    ),
    "half_spectrum_curves": (
        "outputs/raw_rfft_preview/raw_rfft_preview_curves.npz",
        "1111a203290596ce997de6b79896577a69c0c95e7d30f450711f35c144d257dc",
    ),
    "crosscheck": (
        "outputs/raw_fft_crosscheck_v2.json",
        "0215314416f1946a3f096867aeccc7fd7dd038b0e333456df30a689f6e4c6c46",
    ),
}
_REAL_K4_SOURCE = {
    "branch": "codex/pr158-10345-native10k-source-2f6759608",
    "commit": "2f6759608c82356dd5c24e7b149002df8cc84f28",
    "tree": "b2e9b35f3988ae63aae45e1dac90c0386099d97f",
    "clean": True,
}
_REAL_K4_RUN_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_halfmap_10345_native10k_seed42001_2f6759608_20260903"
)
_REAL_K4_ARTIFACT_SEALS = {
    "submission_manifest_sha256": "9931f94c44fa7ff5627e17c04eefeee04f7aa1b64c587af7bb1783cbcf5a184c",
    "audit_sha256": "84217ddbc0f5cf20967a14d2a1d29e12f1a5006114200e31659b4490c8ae9e5a",
    "curve_archive_sha256": "cd497fe6c7731a9d97ff8c86fb4faf55917dd2d5bc68a8fbb71541124a23a620",
    "common_mask_sha256": "a8256cd679a9c4fad3044c4e98cfb1517eef287bf1f9c7095d99b4fdfce8cd9d",
    "qualification_stdout_sha256": "35d2483aaabb0ed6a76fb62f978d233529fd19dd55914f9f07d85623d98fdcd4",
    "qualification_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
_REAL_K4_GATE_FAILURES = [
    "class002:unmasked:cross_merged_band_fsc_auc",
    "class004:common_masked:half_band_auc_drop",
    "class004:unmasked:cross_merged_band_fsc_auc",
    "half1:class_assignment_agreement",
    "half2:class_assignment_agreement",
]


def _require_diagnostic(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryValidationError(message)


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require_diagnostic(isinstance(value, dict), f"{label} must be an object")
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    _require_diagnostic(not missing, f"{label} is missing keys: {missing}")
    _require_diagnostic(not unexpected, f"{label} has unexpected keys: {unexpected}")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    _require_diagnostic(
        type(value) is type(expected) and value == expected,
        f"{label} must equal the sealed value {expected!r}",
    )


def _require_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    _require_diagnostic(valid and math.isfinite(value), f"{label} must be a finite number")
    numeric = float(value)
    if minimum is not None:
        _require_diagnostic(numeric >= minimum, f"{label} must be >= {minimum}")
    if maximum is not None:
        _require_diagnostic(numeric <= maximum, f"{label} must be <= {maximum}")
    return numeric


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require_diagnostic(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def _require_numeric_list(
    value: Any,
    length: int,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    _require_diagnostic(isinstance(value, list) and len(value) == length, f"{label} must contain {length} values")
    return [
        _require_number(item, f"{label}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def _require_integer_list(value: Any, length: int, label: str) -> list[int]:
    _require_diagnostic(isinstance(value, list) and len(value) == length, f"{label} must contain {length} values")
    return [_require_integer(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _require_sealed_hash(value: Any, expected: str, label: str, *, git: bool = False) -> None:
    pattern = _GIT_SHA_RE if git else _SHA256_RE
    kind = "Git SHA" if git else "SHA-256"
    _require_diagnostic(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{label} must be a lowercase {kind}")
    _require_diagnostic(value == expected, f"{label} differs from the sealed {kind}")


def _validate_sealed_source(value: Any, expected: dict[str, Any], label: str) -> None:
    source = _require_exact_keys(value, set(expected), label)
    for key, sealed_value in expected.items():
        if key in {"commit", "tree", "recovar_trajectory_commit", "analysis_commit", "analysis_tree"}:
            _require_sealed_hash(source[key], sealed_value, f"{label}.{key}", git=True)
        else:
            _require_exact(source[key], sealed_value, f"{label}.{key}")


def _validate_k1_direct_fsc_diagnostic(diagnostic: dict[str, Any]) -> None:
    top = _require_exact_keys(
        diagnostic,
        {
            "schema",
            "admission_status",
            "status",
            "case_id",
            "reported_iteration",
            "fitted_operations",
            "comparison_frame",
            "source",
            "geometry",
            "within_engine_halfmap",
            "direct_cross_engine",
            "diagnostic_centered_correlation",
            "full_vs_hermitian_half_spectrum",
            "slurm",
            "artifacts",
        },
        "K1 direct-FSC diagnostic",
    )
    for key, expected in {
        "schema": _K1_DIRECT_FSC_SCHEMA,
        "admission_status": "INTERIM_DIAGNOSTIC_ONLY_FINAL_REFINEMENT_PENDING",
        "status": "pass",
        "case_id": "empiar-10202-set06-k1-I1",
        "reported_iteration": 11,
        "fitted_operations": [],
        "comparison_frame": "shared RELION on-disk file frame",
    }.items():
        _require_exact(top[key], expected, f"K1 direct-FSC diagnostic.{key}")
    _validate_sealed_source(top["source"], _K1_SOURCE, "K1 direct-FSC diagnostic.source")

    geometry = _require_exact_keys(
        top["geometry"],
        {"box_size", "voxel_size_angstrom", "symmetry", "joint_band_first_shell", "joint_band_last_shell"},
        "K1 direct-FSC diagnostic.geometry",
    )
    _require_exact(geometry["box_size"], 800, "K1 direct-FSC diagnostic.geometry.box_size")
    voxel_size = _require_number(
        geometry["voxel_size_angstrom"],
        "K1 direct-FSC diagnostic.geometry.voxel_size_angstrom",
        minimum=0.0,
    )
    _require_diagnostic(voxel_size > 0.0, "K1 direct-FSC diagnostic.geometry.voxel_size_angstrom must be positive")
    _require_exact(geometry["symmetry"], "I1", "K1 direct-FSC diagnostic.geometry.symmetry")
    _require_exact(geometry["joint_band_first_shell"], 1, "K1 direct-FSC diagnostic.geometry.joint_band_first_shell")
    _require_exact(geometry["joint_band_last_shell"], 249, "K1 direct-FSC diagnostic.geometry.joint_band_last_shell")

    halfmap = _require_exact_keys(
        top["within_engine_halfmap"],
        {
            "recovar_crossing_shell",
            "relion_crossing_shell",
            "recovar_resolution_angstrom",
            "relion_resolution_angstrom",
            "resolution_ratio",
            "curve_rmse",
            "band_auc_abs_delta",
            "pass",
        },
        "K1 direct-FSC diagnostic.within_engine_halfmap",
    )
    recovar_shell = _require_integer(halfmap["recovar_crossing_shell"], "within_engine_halfmap.recovar_crossing_shell", minimum=1)
    relion_shell = _require_integer(halfmap["relion_crossing_shell"], "within_engine_halfmap.relion_crossing_shell", minimum=1)
    _require_diagnostic(
        recovar_shell == relion_shell == geometry["joint_band_last_shell"] + 1,
        "within-engine crossing shells must both terminate the frozen joint band",
    )
    recovar_resolution = _require_number(halfmap["recovar_resolution_angstrom"], "within_engine_halfmap.recovar_resolution_angstrom", minimum=0.0)
    relion_resolution = _require_number(halfmap["relion_resolution_angstrom"], "within_engine_halfmap.relion_resolution_angstrom", minimum=0.0)
    _require_diagnostic(
        recovar_resolution > 0.0 and relion_resolution > 0.0,
        "within-engine resolutions must be positive",
    )
    for label, resolution, shell in (
        ("recovar", recovar_resolution, recovar_shell),
        ("relion", relion_resolution, relion_shell),
    ):
        expected_resolution = geometry["box_size"] * voxel_size / shell
        _require_diagnostic(
            math.isclose(resolution, expected_resolution, rel_tol=0.0, abs_tol=1e-12),
            f"within_engine_halfmap.{label}_resolution_angstrom conflicts with geometry",
        )
    ratio = _require_number(halfmap["resolution_ratio"], "within_engine_halfmap.resolution_ratio", minimum=1.0)
    expected_ratio = max(recovar_resolution, relion_resolution) / min(recovar_resolution, relion_resolution)
    _require_diagnostic(
        math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12),
        "within_engine_halfmap.resolution_ratio is inconsistent",
    )
    curve_rmse = _require_number(halfmap["curve_rmse"], "within_engine_halfmap.curve_rmse", minimum=0.0)
    auc_delta = _require_number(halfmap["band_auc_abs_delta"], "within_engine_halfmap.band_auc_abs_delta", minimum=0.0)
    halfmap_pass = ratio <= 1.05 and curve_rmse <= 0.02 and auc_delta <= 0.02
    _require_diagnostic(halfmap["pass"] is halfmap_pass, "within_engine_halfmap.pass conflicts with frozen thresholds")
    _require_diagnostic(max(recovar_resolution, relion_resolution) <= 3.0, "interim K1 record no longer passes its high-resolution gate")

    direct = _require_exact_keys(
        top["direct_cross_engine"],
        {"merged_band_fsc_auc", "half1_band_fsc_auc", "half2_band_fsc_auc", "raw_unaligned_pass", "proper_rigid_rescue_run"},
        "K1 direct-FSC diagnostic.direct_cross_engine",
    )
    merged = _require_number(direct["merged_band_fsc_auc"], "direct_cross_engine.merged_band_fsc_auc", minimum=-1.0, maximum=1.0)
    half1 = _require_number(direct["half1_band_fsc_auc"], "direct_cross_engine.half1_band_fsc_auc", minimum=-1.0, maximum=1.0)
    half2 = _require_number(direct["half2_band_fsc_auc"], "direct_cross_engine.half2_band_fsc_auc", minimum=-1.0, maximum=1.0)
    raw_pass = merged >= 0.95 and min(half1, half2) >= 0.90
    _require_diagnostic(direct["raw_unaligned_pass"] is raw_pass, "direct_cross_engine.raw_unaligned_pass conflicts with frozen thresholds")
    _require_diagnostic(direct["proper_rigid_rescue_run"] is False, "a passing raw comparison cannot claim a fitted rigid rescue")
    _require_diagnostic(halfmap_pass and raw_pass, "status=pass requires every primary K1 gate to pass")

    correlations = _require_exact_keys(
        top["diagnostic_centered_correlation"],
        {"merged", "half1", "half2", "acceptance_metric"},
        "K1 direct-FSC diagnostic.diagnostic_centered_correlation",
    )
    for key in ("merged", "half1", "half2"):
        _require_number(correlations[key], f"diagnostic_centered_correlation.{key}", minimum=-1.0, maximum=1.0)
    _require_exact(correlations["acceptance_metric"], False, "diagnostic_centered_correlation.acceptance_metric")

    crosscheck = _require_exact_keys(
        top["full_vs_hermitian_half_spectrum"],
        {
            "maximum_joint_band_curve_abs_delta",
            "maximum_all_shell_curve_abs_delta",
            "maximum_primary_metric_abs_delta",
            "crossing_shells_identical",
            "selected_band_identical",
            "all_gates_identical",
            "full_fft_batch_max_rss_kib",
            "half_spectrum_batch_max_rss_kib",
        },
        "K1 direct-FSC diagnostic.full_vs_hermitian_half_spectrum",
    )
    for key in (
        "maximum_joint_band_curve_abs_delta",
        "maximum_all_shell_curve_abs_delta",
        "maximum_primary_metric_abs_delta",
    ):
        _require_number(crosscheck[key], f"full_vs_hermitian_half_spectrum.{key}", minimum=0.0)
    for key in ("crossing_shells_identical", "selected_band_identical", "all_gates_identical"):
        _require_exact(crosscheck[key], True, f"full_vs_hermitian_half_spectrum.{key}")
    full_rss = _require_integer(crosscheck["full_fft_batch_max_rss_kib"], "full_vs_hermitian_half_spectrum.full_fft_batch_max_rss_kib", minimum=1)
    half_rss = _require_integer(crosscheck["half_spectrum_batch_max_rss_kib"], "full_vs_hermitian_half_spectrum.half_spectrum_batch_max_rss_kib", minimum=1)
    _require_diagnostic(half_rss < full_rss, "Hermitian half-spectrum RSS must remain below full-FFT RSS")

    slurm = _require_exact_keys(
        top["slurm"],
        {
            "half_spectrum_job_id",
            "half_spectrum_state",
            "half_spectrum_exit_code",
            "half_spectrum_elapsed",
            "full_fft_job_id",
            "full_fft_state",
            "full_fft_exit_code",
            "full_fft_elapsed",
            "requested_tres",
            "allocated_tres",
            "nonexclusive",
            "gpu_count",
        },
        "K1 direct-FSC diagnostic.slurm",
    )
    for key, expected in {
        "half_spectrum_job_id": 13373204,
        "half_spectrum_state": "COMPLETED",
        "half_spectrum_exit_code": "0:0",
        "half_spectrum_elapsed": "00:01:22",
        "full_fft_job_id": 13373359,
        "full_fft_state": "COMPLETED",
        "full_fft_exit_code": "0:0",
        "full_fft_elapsed": "00:02:04",
        "requested_tres": "cpu=8,mem=64G,node=1,billing=16",
        "allocated_tres": "cpu=8,mem=64G,node=1,billing=16",
        "nonexclusive": True,
        "gpu_count": 0,
    }.items():
        _require_exact(slurm[key], expected, f"K1 direct-FSC diagnostic.slurm.{key}")

    artifacts = _require_exact_keys(
        top["artifacts"], {"run_root", *_K1_ARTIFACT_SEALS}, "K1 direct-FSC diagnostic.artifacts"
    )
    _require_exact(artifacts["run_root"], str(_K1_RUN_ROOT), "K1 direct-FSC diagnostic.artifacts.run_root")
    for role, (relative_path, sealed_hash) in _K1_ARTIFACT_SEALS.items():
        reference = _require_exact_keys(artifacts[role], {"path", "sha256"}, f"artifacts.{role}")
        _require_exact(reference["path"], str(_K1_RUN_ROOT / relative_path), f"artifacts.{role}.path")
        _require_sealed_hash(reference["sha256"], sealed_hash, f"artifacts.{role}.sha256")


def _validate_real_kclass_single_seed_diagnostic(diagnostic: dict[str, Any]) -> None:
    top = _require_exact_keys(
        diagnostic,
        {
            "schema",
            "admission_status",
            "accepted_result",
            "status",
            "dataset",
            "profile",
            "source",
            "configuration",
            "class_matching",
            "assignment",
            "frozen_relion_unmasked_band_last_shell",
            "classes",
            "resolution",
            "prospective_science_gate",
            "performance",
            "slurm",
            "artifacts",
        },
        "real K-class single-seed diagnostic",
    )
    for key, expected in {
        "schema": _REAL_KCLASS_SINGLE_SEED_SCHEMA,
        "admission_status": "DIAGNOSTIC_ONLY_MULTI_SEED_REQUIRED",
        "accepted_result": False,
        "status": "complete_prospective_science_gate_rejected",
        "dataset": "EMPIAR-10345",
        "profile": "native10k-256",
    }.items():
        _require_exact(top[key], expected, f"real K-class single-seed diagnostic.{key}")
    _validate_sealed_source(top["source"], _REAL_K4_SOURCE, "real K-class single-seed diagnostic.source")

    configuration = _require_exact_keys(
        top["configuration"],
        {
            "K",
            "particles",
            "half_counts",
            "grid_size",
            "max_iter",
            "symmetry",
            "seed",
            "initial_lowpass_angstrom",
            "fourier_backend",
            "same_job_serial",
            "absolute_resolution_claim",
            "phase_randomization_corrected",
        },
        "real K-class single-seed diagnostic.configuration",
    )
    for key, expected in {
        "K": 4,
        "particles": 10000,
        "grid_size": 256,
        "max_iter": 8,
        "symmetry": "C1",
        "seed": 42001,
        "initial_lowpass_angstrom": 30,
        "fourier_backend": "relion_cuda",
        "same_job_serial": True,
        "absolute_resolution_claim": False,
        "phase_randomization_corrected": False,
    }.items():
        _require_exact(configuration[key], expected, f"configuration.{key}")
    half_counts = _require_integer_list(configuration["half_counts"], 2, "configuration.half_counts")
    _require_diagnostic(sum(half_counts) == configuration["particles"], "configuration.half_counts must sum to particles")
    k = configuration["K"]

    matching = _require_exact_keys(
        top["class_matching"],
        {
            "recovar_half1_to_relion_half1",
            "recovar_half2_to_relion_half1",
            "relion_half2_to_relion_half1",
            "unique_exact_optimum",
            "all_frozen_objective_margin_gates_pass",
        },
        "real K-class single-seed diagnostic.class_matching",
    )
    expected_classes = list(range(1, k + 1))
    for key in (
        "recovar_half1_to_relion_half1",
        "recovar_half2_to_relion_half1",
        "relion_half2_to_relion_half1",
    ):
        values = _require_integer_list(matching[key], k, f"class_matching.{key}")
        _require_diagnostic(sorted(values) == expected_classes, f"class_matching.{key} must be a permutation of 1..K")
    _require_exact(matching["unique_exact_optimum"], True, "class_matching.unique_exact_optimum")
    _require_exact(
        matching["all_frozen_objective_margin_gates_pass"],
        True,
        "class_matching.all_frozen_objective_margin_gates_pass",
    )

    assignment = _require_exact_keys(
        top["assignment"],
        {
            "half1_agreement",
            "half2_agreement",
            "threshold",
            "half1_relion_counts",
            "half1_recovar_counts",
            "half2_relion_counts",
            "half2_recovar_counts",
            "class_collapse",
        },
        "real K-class single-seed diagnostic.assignment",
    )
    _require_exact(assignment["threshold"], 0.99, "assignment.threshold")
    agreements = {
        half: _require_number(assignment[f"half{half}_agreement"], f"assignment.half{half}_agreement", minimum=0.0, maximum=1.0)
        for half in (1, 2)
    }
    all_counts: dict[tuple[int, str], list[int]] = {}
    for half in (1, 2):
        for engine in ("relion", "recovar"):
            key = f"half{half}_{engine}_counts"
            counts = _require_integer_list(assignment[key], k, f"assignment.{key}")
            _require_diagnostic(sum(counts) == half_counts[half - 1], f"assignment.{key} must sum to its frozen half count")
            all_counts[(half, engine)] = counts
    expected_collapse = any(min(counts) < 1 for counts in all_counts.values())
    _require_diagnostic(assignment["class_collapse"] is expected_collapse, "assignment.class_collapse conflicts with hard counts")

    _require_exact(
        top["frozen_relion_unmasked_band_last_shell"],
        configuration["grid_size"] // 2 - 2,
        "real K-class single-seed diagnostic.frozen_relion_unmasked_band_last_shell",
    )
    classes = top["classes"]
    _require_diagnostic(isinstance(classes, list) and len(classes) == k, "classes must contain exactly K rows")
    failures: list[str] = []
    class_ids: list[int] = []
    shell_pairs: list[list[int]] = []
    for index, row_value in enumerate(classes):
        row = _require_exact_keys(
            row_value,
            {
                "class",
                "unmasked_half_auc_relion_recovar_delta",
                "unmasked_cross_merged_half1_half2_auc",
                "masked_half_auc_relion_recovar_delta",
                "masked_cross_merged_half1_half2_auc",
                "unmasked_fsc_0p5_shell_relion_recovar",
            },
            f"classes[{index}]",
        )
        class_id = _require_integer(row["class"], f"classes[{index}].class", minimum=1)
        class_ids.append(class_id)
        for route in ("unmasked", "masked"):
            key = f"{route}_half_auc_relion_recovar_delta"
            relion_auc, recovar_auc, delta = _require_numeric_list(
                row[key], 3, f"classes[{index}].{key}", minimum=-1.0, maximum=1.0
            )
            _require_diagnostic(
                math.isclose(delta, recovar_auc - relion_auc, rel_tol=0.0, abs_tol=1e-12),
                f"classes[{index}].{key} signed delta is inconsistent",
            )
            if delta < -0.01:
                gate_route = "common_masked" if route == "masked" else route
                failures.append(f"class{class_id:03d}:{gate_route}:half_band_auc_drop")
        unmasked_cross = _require_numeric_list(
            row["unmasked_cross_merged_half1_half2_auc"],
            3,
            f"classes[{index}].unmasked_cross_merged_half1_half2_auc",
            minimum=-1.0,
            maximum=1.0,
        )
        _require_numeric_list(
            row["masked_cross_merged_half1_half2_auc"],
            3,
            f"classes[{index}].masked_cross_merged_half1_half2_auc",
            minimum=-1.0,
            maximum=1.0,
        )
        if unmasked_cross[0] < 0.99:
            failures.append(f"class{class_id:03d}:unmasked:cross_merged_band_fsc_auc")
        if min(unmasked_cross[1:]) < 0.90:
            failures.append(f"class{class_id:03d}:unmasked:cross_each_half_band_fsc_auc")
        shells = _require_integer_list(
            row["unmasked_fsc_0p5_shell_relion_recovar"],
            2,
            f"classes[{index}].unmasked_fsc_0p5_shell_relion_recovar",
        )
        _require_diagnostic(shells[0] == shells[1] > 0, f"classes[{index}] must retain equal positive FSC=0.5 shells")
        shell_pairs.append(shells)
    _require_diagnostic(sorted(class_ids) == expected_classes, "classes.class must be a permutation of 1..K")
    for half, agreement in agreements.items():
        if agreement < assignment["threshold"]:
            failures.append(f"half{half}:class_assignment_agreement")
    _require_diagnostic(not expected_collapse, "single-seed diagnostic must retain its no-collapse boundary")

    resolution = _require_exact_keys(
        top["resolution"],
        {
            "registered_unmasked_fsc_0p143",
            "registered_common_masked_fsc_0p143",
            "better_than_angstrom",
            "unmasked_fsc_0p5_angstrom_relion_equals_recovar",
        },
        "real K-class single-seed diagnostic.resolution",
    )
    beyond_range = "all RELION and RECOVAR classes beyond measured range"
    _require_exact(resolution["registered_unmasked_fsc_0p143"], beyond_range, "resolution.registered_unmasked_fsc_0p143")
    _require_exact(resolution["registered_common_masked_fsc_0p143"], beyond_range, "resolution.registered_common_masked_fsc_0p143")
    better_than = _require_number(resolution["better_than_angstrom"], "resolution.better_than_angstrom", minimum=0.0)
    fsc_0p5_resolutions = _require_numeric_list(
        resolution["unmasked_fsc_0p5_angstrom_relion_equals_recovar"],
        k,
        "resolution.unmasked_fsc_0p5_angstrom_relion_equals_recovar",
        minimum=0.0,
    )
    voxel_size = better_than * top["frozen_relion_unmasked_band_last_shell"] / configuration["grid_size"]
    for index, (shells, observed) in enumerate(zip(shell_pairs, fsc_0p5_resolutions, strict=True)):
        expected = configuration["grid_size"] * voxel_size / shells[0]
        _require_diagnostic(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6),
            f"resolution.unmasked_fsc_0p5_angstrom_relion_equals_recovar[{index}] conflicts with its shell",
        )

    gate = _require_exact_keys(
        top["prospective_science_gate"],
        {"accepted", "failures", "qualification_exit_code", "threshold_rejection_only"},
        "real K-class single-seed diagnostic.prospective_science_gate",
    )
    _require_exact(gate["accepted"], False, "prospective_science_gate.accepted")
    _require_exact(gate["failures"], _REAL_K4_GATE_FAILURES, "prospective_science_gate.failures")
    _require_diagnostic(failures == gate["failures"], "prospective_science_gate.failures conflict with frozen thresholds")
    _require_exact(gate["qualification_exit_code"], "3:0", "prospective_science_gate.qualification_exit_code")
    _require_exact(gate["threshold_rejection_only"], True, "prospective_science_gate.threshold_rejection_only")

    performance = _require_exact_keys(
        top["performance"],
        {
            "relion_wall_seconds_half1_half2",
            "recovar_wall_seconds_half1_half2",
            "relion_peak_hbm_mib_half1_half2",
            "recovar_peak_hbm_mib_half1_half2",
            "relion_max_rss_kib_half1_half2",
            "recovar_max_rss_kib_half1_half2",
        },
        "real K-class single-seed diagnostic.performance",
    )
    for key in performance:
        _require_numeric_list(performance[key], 2, f"performance.{key}", minimum=0.0)

    slurm = _require_exact_keys(
        top["slurm"],
        {
            "setup_job_id",
            "setup_state",
            "setup_exit_code",
            "qualification_job_id",
            "qualification_state",
            "qualification_exit_code",
            "qualification_elapsed",
            "qualification_requested_tres",
            "qualification_allocated_tres",
            "nonexclusive",
            "compute_failures",
        },
        "real K-class single-seed diagnostic.slurm",
    )
    for key, expected in {
        "setup_job_id": 13371068,
        "setup_state": "COMPLETED",
        "setup_exit_code": "0:0",
        "qualification_job_id": 13371069,
        "qualification_state": "FAILED",
        "qualification_exit_code": gate["qualification_exit_code"],
        "qualification_elapsed": "01:04:34",
        "qualification_requested_tres": "cpu=24,mem=256G,node=1,billing=24,gres/gpu=1",
        "qualification_allocated_tres": "cpu=24,mem=256G,node=1,billing=24,gres/gpu=1",
        "nonexclusive": True,
        "compute_failures": False,
    }.items():
        _require_exact(slurm[key], expected, f"real K-class single-seed diagnostic.slurm.{key}")

    artifacts = _require_exact_keys(
        top["artifacts"],
        {"run_root", "audit_path", *_REAL_K4_ARTIFACT_SEALS},
        "real K-class single-seed diagnostic.artifacts",
    )
    _require_exact(artifacts["run_root"], str(_REAL_K4_RUN_ROOT), "artifacts.run_root")
    _require_exact(artifacts["audit_path"], str(_REAL_K4_RUN_ROOT / "audit/halfmap_audit.json"), "artifacts.audit_path")
    for key, sealed_hash in _REAL_K4_ARTIFACT_SEALS.items():
        _require_sealed_hash(artifacts[key], sealed_hash, f"artifacts.{key}")


def validate_registered_diagnostic(diagnostic: dict[str, Any]) -> None:
    """Validate diagnostic schemas owned directly by this registry validator."""

    schema = diagnostic.get("schema") if isinstance(diagnostic, dict) else None
    if schema == _K1_DIRECT_FSC_SCHEMA:
        _validate_k1_direct_fsc_diagnostic(diagnostic)
    elif schema == _REAL_KCLASS_SINGLE_SEED_SCHEMA:
        _validate_real_kclass_single_seed_diagnostic(diagnostic)
    else:
        raise RegistryValidationError(f"unsupported inline diagnostic schema: {schema!r}")


def _semantic_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    k = record["subject"]["k"]
    n_particles = record["subject"]["particles"]
    quality = record["quality"]

    matching = quality["matching"]
    expected_classes = list(range(1, k + 1))
    for key in ("recovar_to_relion", "matched_pair_to_gt"):
        assignment = matching[key]
        if sorted(assignment) != expected_classes:
            errors.append(f"quality.matching.{key} must be a permutation of 1..K")

    final_classes = quality["final_classes"]
    if len(final_classes) != k:
        errors.append("quality.final_classes must contain exactly K rows")
    for key in ("recovar_class", "relion_class", "gt_class"):
        values = [row[key] for row in final_classes]
        if key == "gt_class" and not quality["ground_truth_available"]:
            continue
        if sorted(values) != expected_classes:
            errors.append(f"quality.final_classes.{key} must be a permutation of 1..K")

    if quality["ground_truth_available"]:
        gt_keys = ("gt_class", "recovar_gt_fsc_auc", "relion_gt_fsc_auc", "gt_fsc_auc_delta")
        if any(row[key] is None for row in final_classes for key in gt_keys):
            errors.append("ground_truth_available=true requires every per-engine GT metric")

    populations = quality["final_class_populations"]
    for key in ("recovar", "relion", "recovar_posterior_fractions"):
        if len(populations[key]) != k:
            errors.append(f"quality.final_class_populations.{key} must contain K values")
    for key in ("recovar", "relion"):
        values = populations[key]
        if all(value is not None for value in values) and sum(values) != n_particles:
            errors.append(f"quality.final_class_populations.{key} must sum to particle count")
    posterior = populations["recovar_posterior_fractions"]
    if all(value is not None for value in posterior) and not math.isclose(
        sum(posterior), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        errors.append("quality.final_class_populations.recovar_posterior_fractions must sum to 1")
    population_values = populations["recovar"] + populations["relion"] + posterior
    if any(value is None for value in population_values) and not populations["missing_reason"]:
        errors.append("missing class-population values require missing_reason")

    class_collapse = quality.get("class_collapse")
    if class_collapse is not None:
        threshold = class_collapse["threshold"]
        expected_flags: dict[str, list[int]] = {}
        for engine in ("recovar", "relion"):
            values = populations[engine]
            if any(value is None for value in values):
                errors.append("quality.class_collapse requires complete hard class populations")
                continue
            expected_flags[engine] = [
                class_id
                for class_id, count in enumerate(values, start=1)
                if count / n_particles < threshold
            ]
            if class_collapse[f"{engine}_classes"] != expected_flags[engine]:
                errors.append(
                    f"quality.class_collapse.{engine}_classes must match the hard-population threshold"
                )
        if len(expected_flags) == 2:
            expected_status = "FLAGGED" if any(expected_flags.values()) else "CLEAR"
            if class_collapse["status"] != expected_status:
                errors.append("quality.class_collapse.status conflicts with the flagged classes")

    halfmap = quality["halfmap_fsc"]
    halfmap_values: list[float | None] = []
    for engine in ("recovar", "relion"):
        for key in (
            "unmasked_fsc_auc",
            "masked_fsc_auc",
            "masked_resolution_0_143_angstrom",
        ):
            values = halfmap[engine][key]
            halfmap_values.extend(values)
            if len(values) != k:
                errors.append(f"quality.halfmap_fsc.{engine}.{key} must contain K values")
    masked_values = halfmap["recovar"]["masked_fsc_auc"] + halfmap["relion"]["masked_fsc_auc"]
    if any(value is not None for value in masked_values) and halfmap["mask"] is None:
        errors.append("masked half-map FSC values require a hashed mask")
    if any(value is None for value in halfmap_values) and not halfmap["missing_reason"]:
        errors.append("missing half-map values require missing_reason")

    trajectory = quality["trajectory"]
    expected_cells = trajectory["evaluated_iterations"] * k
    if trajectory["evaluated_class_cells"] != expected_cells:
        errors.append("evaluated_class_cells must equal evaluated_iterations * K")
    if trajectory["passing_class_cells"] > trajectory["evaluated_class_cells"]:
        errors.append("passing_class_cells cannot exceed evaluated_class_cells")
    if trajectory["all_class_passing_iterations"] > trajectory["evaluated_iterations"]:
        errors.append("all_class_passing_iterations cannot exceed evaluated_iterations")

    gate = record["gate"]
    if gate["formal_status"] == "PASS" and trajectory["earliest_failure"] is not None:
        errors.append("formal_status=PASS conflicts with a trajectory failure")
    if gate["formal_status"] == "FAIL" and trajectory["earliest_failure"] is None:
        errors.append("formal_status=FAIL requires a recorded earliest trajectory failure")
    if gate["classification"] == "TRAJECTORY_EXACT":
        if not gate["trajectory_exact"]:
            errors.append("TRAJECTORY_EXACT requires trajectory_exact=true")
        if gate["formal_status"] != "PASS" or gate["science_status"] != "PASS":
            errors.append("TRAJECTORY_EXACT requires both formal and science PASS")
    elif gate["classification"] == "SCIENCE_EQUIVALENT":
        if gate["trajectory_exact"]:
            errors.append("SCIENCE_EQUIVALENT requires trajectory_exact=false")
        if gate["science_status"] != "PASS":
            errors.append("SCIENCE_EQUIVALENT requires science_status=PASS")
        if record["record_type"] == "candidate" and gate["comparator_record_id"] is None:
            errors.append("candidate SCIENCE_EQUIVALENT records require comparator_record_id")
    elif gate["science_status"] != "FAIL":
        errors.append("NOT_EQUIVALENT requires science_status=FAIL")

    comparison = record["performance"]["comparison"]
    for engine in ("recovar", "relion"):
        measurement = record["performance"][engine]
        measured_values = [
            measurement["wall_s"],
            measurement["iteration_sum_s"],
            measurement["peak_hbm_mib"],
            measurement["max_rss_kib"],
        ]
        if any(value is None for value in measured_values) and not measurement["missing_reason"]:
            errors.append(f"performance.{engine} missing values require missing_reason")
    if comparison["hardware_comparable"] and comparison["formal_speedup"] is None:
        errors.append("hardware_comparable=true requires formal_speedup")
    if not comparison["hardware_comparable"] and comparison["formal_speedup"] is not None:
        errors.append("cross-hardware comparisons cannot report formal_speedup")

    for job in record["execution"]["jobs"]:
        if job["req_tres"] != job["alloc_tres"]:
            errors.append(f"job {job['job_id']} ReqTRES and AllocTRES differ")

    for collection in (record["inputs"], record["artifacts"]):
        roles = [item["role"] for item in collection]
        if len(roles) != len(set(roles)):
            errors.append("input and artifact roles must be unique within each collection")

    return errors


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate one record against JSON Schema and cross-field invariants."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    errors = [f"{_json_path(list(error.path))}: {error.message}" for error in schema_errors]
    errors.extend(_semantic_errors(record) if not schema_errors else [])
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _campaign_semantic_errors(campaign: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    k = campaign["subject"]["k"]
    expected_classes = list(range(1, k + 1))
    expected_iterations = campaign["subject"]["iterations"]

    case_ids = [case["case_id"] for case in campaign["cases"]]
    case_names = [case["name"] for case in campaign["cases"]]
    if len(case_ids) != len(set(case_ids)):
        errors.append("cases.case_id values must be unique")
    if len(case_names) != len(set(case_names)):
        errors.append("cases.name values must be unique")

    for collection_name in ("shared_jobs", "shared_artifacts"):
        roles = [item["role"] for item in campaign[collection_name]]
        if len(roles) != len(set(roles)):
            errors.append(f"{collection_name} roles must be unique")

    for job in campaign["shared_jobs"]:
        if job["req_tres"] != job["alloc_tres"]:
            errors.append(f"job {job['job_id']} ReqTRES and AllocTRES differ")
        if (job["status"] == "COMPLETED") != (job["exit_code"] == "0:0"):
            errors.append(f"job {job['job_id']} status conflicts with exit_code")

    equivalence_groups: dict[str, list[dict[str, Any]]] = {}
    for case in campaign["cases"]:
        prefix = f"case {case['case_id']}"
        if case["execution"]["req_tres"] != case["execution"]["alloc_tres"]:
            errors.append(f"{prefix} job ReqTRES and AllocTRES differ")
        if (case["execution"]["status"] == "COMPLETED") != (
            case["execution"]["exit_code"] == "0:0"
        ):
            errors.append(f"{prefix} job status conflicts with exit_code")

        for collection_name in ("inputs", "artifacts"):
            roles = [item["role"] for item in case[collection_name]]
            if len(roles) != len(set(roles)):
                errors.append(f"{prefix} {collection_name} roles must be unique")
        if "particles" not in {item["role"] for item in case["inputs"]}:
            errors.append(f"{prefix} inputs must contain the particle stack with role=particles")

        equivalence = case["input_equivalence"]
        if equivalence["group"] is not None:
            equivalence_groups.setdefault(equivalence["group"], []).append(case)

        quality = case["quality"]
        trajectory = quality["trajectory"]
        outcome = case["outcome"]
        complete_endpoint = outcome["endpoint_status"] == "COMPLETE"
        expected_cells = trajectory["evaluated_iterations"] * k
        if trajectory["evaluated_class_cells"] != expected_cells:
            errors.append(f"{prefix} evaluated_class_cells must equal evaluated_iterations * K")
        if trajectory["passing_class_cells"] > trajectory["evaluated_class_cells"]:
            errors.append(f"{prefix} passing_class_cells cannot exceed evaluated_class_cells")
        if outcome["formal_status"] != trajectory["status"]:
            errors.append(f"{prefix} formal_status must equal trajectory status")
        if trajectory["status"] == "PASS":
            if trajectory["evaluated_iterations"] != expected_iterations:
                errors.append(f"{prefix} PASS must evaluate every configured iteration")
            if trajectory["passing_class_cells"] != trajectory["evaluated_class_cells"]:
                errors.append(f"{prefix} PASS requires every trajectory class cell to pass")
            if trajectory["earliest_failure"] is not None:
                errors.append(f"{prefix} PASS cannot record earliest_failure")
        elif trajectory["status"] == "FAIL":
            if trajectory["earliest_failure"] is None:
                errors.append(f"{prefix} FAIL requires earliest_failure")
        elif any(
            (
                trajectory["report"] is not None,
                trajectory["evaluated_iterations"] != 0,
                trajectory["evaluated_class_cells"] != 0,
                trajectory["passing_class_cells"] != 0,
                trajectory["earliest_failure"] is None,
            )
        ):
            errors.append(f"{prefix} NOT_EVALUABLE must retain only its failure reason")

        final_classes = quality["final_classes"]
        mean_keys = (
            "mean_recovar_gt_fsc_auc",
            "mean_relion_gt_fsc_auc",
            "mean_gt_fsc_auc_delta",
        )
        if complete_endpoint:
            if final_classes is None or len(final_classes) != k:
                errors.append(f"{prefix} complete endpoint requires exactly K final class rows")
            elif any(quality[key] is None for key in mean_keys):
                errors.append(f"{prefix} complete endpoint requires all mean GT FSC-AUC values")
            else:
                for key in ("recovar_class", "relion_class", "gt_class"):
                    if sorted(row[key] for row in final_classes) != expected_classes:
                        errors.append(f"{prefix} final_classes.{key} must be a permutation of 1..K")
                for row in final_classes:
                    expected_delta = row["recovar_gt_fsc_auc"] - row["relion_gt_fsc_auc"]
                    if not math.isclose(
                        row["gt_fsc_auc_delta"], expected_delta, rel_tol=0.0, abs_tol=1e-12
                    ):
                        errors.append(f"{prefix} per-class signed GT FSC-AUC delta is inconsistent")
                    resolution_keys = (
                        "recovar_gt_resolution_0_143_angstrom",
                        "relion_gt_resolution_0_143_angstrom",
                    )
                    if sum(key in row for key in resolution_keys) == 1:
                        errors.append(
                            f"{prefix} per-class 0.143 GT resolutions must be recorded for both engines"
                        )
                expected_means = {
                    "mean_recovar_gt_fsc_auc": sum(
                        row["recovar_gt_fsc_auc"] for row in final_classes
                    )
                    / k,
                    "mean_relion_gt_fsc_auc": sum(
                        row["relion_gt_fsc_auc"] for row in final_classes
                    )
                    / k,
                    "mean_gt_fsc_auc_delta": sum(
                        row["gt_fsc_auc_delta"] for row in final_classes
                    )
                    / k,
                }
                for key, expected in expected_means.items():
                    if not math.isclose(quality[key], expected, rel_tol=0.0, abs_tol=1e-12):
                        errors.append(f"{prefix} {key} is inconsistent with the per-class rows")
            if quality["matching_method"] != "hungarian_max_fsc_auc":
                errors.append(f"{prefix} complete endpoint requires Hungarian FSC matching")
            if quality["class_assignment_agreement"] is None:
                errors.append(f"{prefix} complete endpoint requires class-assignment agreement")
        elif any(
            value is not None
            for value in (final_classes, quality["class_assignment_agreement"], *(quality[key] for key in mean_keys))
        ):
            errors.append(f"{prefix} non-evaluable endpoint cannot report final quality metrics")

        occupancy = quality["occupancy"]
        relion_counts = occupancy["relion_counts"]
        occupancy_not_evaluated = occupancy["recovar_metric"] == "not_evaluated"
        if occupancy_not_evaluated:
            if any(
                value is not None
                for value in (
                    occupancy["source_iteration"],
                    occupancy["recovar_counts"],
                    occupancy["recovar_fractions"],
                    relion_counts,
                )
            ) or any(
                (
                    occupancy["recovar_flagged_classes"],
                    occupancy["relion_flagged_classes"],
                )
            ):
                errors.append(f"{prefix} unevaluated occupancy cannot report populations or flags")
            if occupancy["status"] != "NOT_EVALUATED":
                errors.append(f"{prefix} unevaluated occupancy requires NOT_EVALUATED status")
            recovar_values = None
        elif relion_counts is None or len(relion_counts) != k or sum(relion_counts) != case["particles"]:
            errors.append(f"{prefix} RELION occupancy must contain K counts summing to particles")
            recovar_values = None
        elif occupancy["source_iteration"] is None:
            errors.append(f"{prefix} evaluated occupancy requires a source iteration")
            recovar_values = None
        elif occupancy["recovar_metric"] == "hard_assignments":
            counts = occupancy["recovar_counts"]
            if counts is None or len(counts) != k or sum(counts) != case["particles"]:
                errors.append(f"{prefix} RECOVAR hard occupancy must contain K counts summing to particles")
            if occupancy["recovar_fractions"] is not None:
                errors.append(f"{prefix} hard occupancy must not also report posterior fractions")
            recovar_values = counts
        elif occupancy["recovar_metric"] == "posterior_fraction":
            fractions = occupancy["recovar_fractions"]
            if occupancy["recovar_counts"] is not None:
                errors.append(f"{prefix} posterior occupancy cannot report hard counts")
            if fractions is None or len(fractions) != k or not math.isclose(
                sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-3
            ):
                errors.append(f"{prefix} RECOVAR posterior occupancy must contain K fractions summing to one")
            recovar_values = (
                None
                if fractions is None
                else [fraction * case["particles"] for fraction in fractions]
            )

        if not occupancy_not_evaluated and relion_counts is not None:
            threshold_count = occupancy["collapse_threshold_fraction"] * case["particles"]
            expected_relion_flags = [
                class_id
                for class_id, count in enumerate(relion_counts, start=1)
                if count < threshold_count
            ]
            if occupancy["relion_flagged_classes"] != expected_relion_flags:
                errors.append(f"{prefix} RELION collapse flags conflict with its occupancy")
            if recovar_values is not None:
                expected_recovar_flags = [
                    class_id
                    for class_id, count in enumerate(recovar_values, start=1)
                    if count < threshold_count
                ]
                if occupancy["recovar_flagged_classes"] != expected_recovar_flags:
                    errors.append(f"{prefix} RECOVAR collapse flags conflict with its occupancy")
            zero_class = (
                0 in relion_counts
                or (occupancy["recovar_counts"] is not None and 0 in occupancy["recovar_counts"])
                or (
                    occupancy["recovar_fractions"] is not None
                    and 0 in occupancy["recovar_fractions"]
                )
            )
            flagged = bool(occupancy["recovar_flagged_classes"] or expected_relion_flags)
            expected_occupancy_status = "ZERO_CLASS" if zero_class else "NEAR_COLLAPSE" if flagged else "CLEAR"
            if occupancy["status"] != expected_occupancy_status:
                errors.append(f"{prefix} occupancy status conflicts with the recorded populations")

        ratio = case["performance"]["recovar_wall_s"] / case["performance"]["relion_wall_s"]
        if not math.isclose(
            case["performance"]["recovar_over_relion_wall_ratio"],
            ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{prefix} RECOVAR/RELION wall ratio is inconsistent")

        classification = outcome["classification"]
        if classification == "TRAJECTORY_EXACT":
            if (
                outcome["endpoint_status"],
                outcome["formal_status"],
                outcome["science_status"],
                occupancy["status"],
            ) != (
                "COMPLETE",
                "PASS",
                "PASS",
                "CLEAR",
            ):
                errors.append(f"{prefix} TRAJECTORY_EXACT requires PASS/PASS and clear occupancy")
        elif classification == "SCIENCE_EQUIVALENT":
            if (outcome["endpoint_status"], outcome["formal_status"], outcome["science_status"]) != (
                "COMPLETE",
                "FAIL",
                "PASS",
            ):
                errors.append(f"{prefix} SCIENCE_EQUIVALENT requires a complete FAIL/PASS endpoint")
        elif classification == "TRAJECTORY_EXACT_NEAR_COLLAPSE":
            if (
                outcome["endpoint_status"],
                outcome["formal_status"],
                outcome["science_status"],
                occupancy["status"],
            ) != (
                "COMPLETE",
                "PASS",
                "BOUNDARY",
                "NEAR_COLLAPSE",
            ):
                errors.append(f"{prefix} near-collapse classification conflicts with status")
        elif classification == "NEGATIVE_ZERO_CLASS_BOUNDARY":
            if (outcome["endpoint_status"], outcome["formal_status"], occupancy["status"]) != (
                "NOT_EVALUABLE",
                "NOT_EVALUABLE",
                "ZERO_CLASS",
            ) or outcome["science_status"] != "BOUNDARY":
                errors.append(f"{prefix} zero-class boundary classification conflicts with status")
        elif classification == "NEGATIVE_RELION_CLASS_COLLAPSE":
            if (
                outcome["endpoint_status"],
                outcome["formal_status"],
                outcome["science_status"],
            ) != ("NOT_EVALUABLE", "NOT_EVALUABLE", "BOUNDARY") or occupancy["status"] not in {
                "NEAR_COLLAPSE",
                "ZERO_CLASS",
            }:
                errors.append(f"{prefix} RELION-collapse classification conflicts with status")
        elif classification == "RECOVAR_IMPLEMENTATION_FAILURE":
            if (
                outcome["endpoint_status"],
                outcome["formal_status"],
                outcome["science_status"],
                occupancy["status"],
            ) != ("NOT_EVALUABLE", "NOT_EVALUABLE", "UNRESOLVED", "NOT_EVALUATED"):
                errors.append(f"{prefix} implementation-failure classification conflicts with status")
        elif (
            outcome["endpoint_status"],
            outcome["formal_status"],
            outcome["science_status"],
        ) != ("COMPLETE", "FAIL", "UNRESOLVED"):
            errors.append(f"{prefix} unresolved classification requires FAIL/UNRESOLVED")

    for group, cases in equivalence_groups.items():
        if len(cases) < 2:
            errors.append(f"input-equivalence group {group} must contain at least two cases")
            continue
        particle_hashes = {
            next(
                (item["sha256"] for item in case["inputs"] if item["role"] == "particles"),
                f"missing:{case['case_id']}",
            )
            for case in cases
        }
        if len(particle_hashes) != 1 and any(
            case["input_equivalence"]["admissible_for_execution_invariance"] for case in cases
        ):
            errors.append(
                f"input-equivalence group {group} has differing particle hashes and is inadmissible"
            )

    return errors


def validate_campaign(campaign: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate one compact campaign scorecard and its cross-field invariants."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(campaign), key=lambda error: list(error.path))
    errors = [f"{_json_path(list(error.path))}: {error.message}" for error in schema_errors]
    errors.extend(_campaign_semantic_errors(campaign) if not schema_errors else [])
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _requested_gpu_count(tres: str) -> int | None:
    total = 0
    found = False
    for component in tres.split(","):
        key, separator, raw_value = component.partition("=")
        if not separator or (key != "gres/gpu" and not key.startswith("gres/gpu:")):
            continue
        found = True
        try:
            total += int(raw_value)
        except ValueError:
            return None
    return total if found else 0


def _diagnostic_semantic_errors(diagnostic: dict[str, Any]) -> list[str]:
    """Validate invariants specific to excluded negative diagnostic evidence."""
    errors: list[str] = []
    k = diagnostic["subject"]["k"]
    expected_seeds = diagnostic["expected_seeds"]

    run_ids = [run["case_id"] for run in diagnostic["runs"]]
    run_names = [run["name"] for run in diagnostic["runs"]]
    if len(run_ids) != len(set(run_ids)):
        errors.append("runs.case_id values must be unique")
    if len(run_names) != len(set(run_names)):
        errors.append("runs.name values must be unique")

    source_roles = [item["role"] for item in diagnostic["source_structures"]]
    expected_source_roles = [f"pdb_class_{class_id}" for class_id in range(1, k + 1)]
    if source_roles != expected_source_roles:
        errors.append("source_structures roles must be ordered pdb_class_1..pdb_class_K")

    for run in diagnostic["runs"]:
        prefix = f"diagnostic case {run['case_id']}"
        configuration = run["configuration"]
        if configuration.get("n_classes") != k:
            errors.append(f"{prefix} n_classes must equal subject.k")
        if configuration.get("case_id") != run["case_id"]:
            errors.append(f"{prefix} configuration.case_id must match case_id")
        if configuration.get("base_name") != run["name"]:
            errors.append(f"{prefix} configuration.base_name must match name")
        if configuration.get("symmetry") != diagnostic["subject"]["symmetry"]:
            errors.append(f"{prefix} configuration.symmetry must match subject.symmetry")

        support_roles = [job["role"] for job in run["support_jobs"]]
        if sorted(support_roles) != ["setup", "summary"]:
            errors.append(f"{prefix} support_jobs must contain exactly setup and summary")
        for job in run["support_jobs"]:
            if job["req_tres"] != job["alloc_tres"]:
                errors.append(f"{prefix} support job {job['job_id']} ReqTRES and AllocTRES differ")

        shared_roles = [item["role"] for item in run["shared_artifacts"]]
        if len(shared_roles) != len(set(shared_roles)):
            errors.append(f"{prefix} shared_artifacts roles must be unique")
        required_shared_roles = {
            "safe_to_delete_marker",
            "case_table",
            "submission_environment",
            "setup_stdout",
            "setup_stderr",
            "summary_stdout",
            "summary_stderr",
            "multiseed_summary_json",
            "matrix_summary_json",
            "historical_accounting_snapshot_may_be_stale",
        }
        if not required_shared_roles.issubset(shared_roles):
            errors.append(f"{prefix} shared_artifacts omit required sealed evidence")

        seeds = [replicate["seed"] for replicate in run["replicates"]]
        if seeds != expected_seeds:
            errors.append(f"{prefix} replicate seeds must exactly match expected_seeds in order")
        names = [replicate["name"] for replicate in run["replicates"]]
        if len(names) != len(set(names)):
            errors.append(f"{prefix} replicate names must be unique")

        for replicate in run["replicates"]:
            replicate_prefix = f"{prefix} seed {replicate['seed']}"
            job = replicate["job"]
            if job["req_tres"] != job["alloc_tres"]:
                errors.append(f"{replicate_prefix} job ReqTRES and AllocTRES differ")
            if _requested_gpu_count(job["req_tres"]) != 1:
                errors.append(f"{replicate_prefix} must request exactly one GPU")

            for collection_name in ("inputs", "artifacts"):
                roles = [item["role"] for item in replicate[collection_name]]
                if len(roles) != len(set(roles)):
                    errors.append(f"{replicate_prefix} {collection_name} roles must be unique")
            input_roles = {item["role"] for item in replicate["inputs"]}
            for required_role in ("particles", "poses", "ctf", "generation_config", "class_manifest"):
                if required_role not in input_roles:
                    errors.append(f"{replicate_prefix} inputs must contain role={required_role}")
            artifact_roles = {item["role"] for item in replicate["artifacts"]}
            required_artifact_roles = {
                "relion_class_population_audit",
                "relion_walltime",
                "relion_gpu_monitor",
                "gpu_inventory",
                "gpu_uuid",
                "job_stdout",
                "job_stderr",
            }
            if not required_artifact_roles.issubset(artifact_roles):
                errors.append(f"{replicate_prefix} artifacts omit required sealed evidence")

            collapse = replicate["relion_collapse"]
            distributions = collapse["final_class_distributions"]
            orientation_masses = collapse["final_orientation_masses"]
            if len(distributions) != k or len(orientation_masses) != k:
                errors.append(f"{replicate_prefix} final RELION populations must contain K values")
            if len(distributions) == k and not math.isclose(
                sum(distributions), 1.0, rel_tol=0.0, abs_tol=2e-6
            ):
                errors.append(f"{replicate_prefix} final class distributions must sum to one")
            if len(orientation_masses) == k and not math.isclose(
                sum(orientation_masses), 1.0, rel_tol=0.0, abs_tol=2e-6
            ):
                errors.append(f"{replicate_prefix} final orientation masses must sum to one")
            expected_zero_classes = [
                class_id
                for class_id, (distribution, mass) in enumerate(
                    zip(distributions, orientation_masses, strict=True), start=1
                )
                if distribution == 0 or mass == 0
            ]
            if collapse["zero_classes"] != expected_zero_classes:
                errors.append(f"{replicate_prefix} zero_classes must match final RELION populations")
            events = collapse["collapsed_events"]
            if collapse["collapsed_event_count"] != len(events):
                errors.append(f"{replicate_prefix} collapsed_event_count must match collapsed_events")
            if collapse["earliest_iteration"] != min(event["iteration"] for event in events):
                errors.append(f"{replicate_prefix} earliest_iteration must match collapsed_events")
            numbered_iterations = collapse["numbered_iterations"]
            max_iter = configuration.get("max_iter")
            if isinstance(max_iter, int) and numbered_iterations != list(range(1, max_iter + 1)):
                errors.append(f"{replicate_prefix} numbered_iterations must cover 1..max_iter")
            if any(event["iteration"] not in numbered_iterations for event in events):
                errors.append(f"{replicate_prefix} collapsed event lies outside numbered_iterations")
            event_keys = [(event["iteration"], event["class"]) for event in events]
            if len(event_keys) != len(set(event_keys)):
                errors.append(f"{replicate_prefix} collapsed_events contain duplicate cells")
            if any(event["class"] > k for event in events) or any(
                class_id > k for class_id in collapse["zero_classes"]
            ):
                errors.append(f"{replicate_prefix} collapse class lies outside 1..K")
            final_zero_classes = sorted(
                event["class"]
                for event in events
                if event["iteration"] == numbered_iterations[-1]
            )
            if final_zero_classes != collapse["zero_classes"]:
                errors.append(f"{replicate_prefix} final collapse events must match zero_classes")

            if replicate["outcome"]["recovar_started"]:
                errors.append(f"{replicate_prefix} negative boundary cannot claim RECOVAR started")
            performance = replicate["performance"]
            if performance["recovar_wall_s"] is not None or performance["recovar_peak_hbm_mib"] is not None:
                errors.append(f"{replicate_prefix} cannot report RECOVAR performance before it ran")
            if performance["relion_wall_s"] > job["elapsed_s"]:
                errors.append(f"{replicate_prefix} RELION wall time cannot exceed Slurm elapsed time")

    return errors


def validate_diagnostic(diagnostic: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate excluded negative evidence without admitting it as an accepted result."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(diagnostic), key=lambda error: list(error.path))
    errors = [f"{_json_path(list(error.path))}: {error.message}" for error in schema_errors]
    errors.extend(_diagnostic_semantic_errors(diagnostic) if not schema_errors else [])
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_references(record: dict[str, Any]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for key in ("generation_config", "class_manifest"):
        if record["dataset"][key] is not None:
            references.append(record["dataset"][key])
    references.append(record["source"]["relion"]["executable"])
    references.extend(record["inputs"])
    references.extend(record["artifacts"])
    mask = record["quality"]["halfmap_fsc"]["mask"]
    if mask is not None:
        references.append(mask)
    return references


def verify_record_files(record: dict[str, Any], digest_cache: dict[Path, str]) -> None:
    """Verify every external path and checksum, caching shared large inputs."""
    errors: list[str] = []
    for reference in _file_references(record):
        path = Path(reference["path"])
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        if "size_bytes" in reference and path.stat().st_size != reference["size_bytes"]:
            errors.append(
                f"size mismatch: {path}: expected {reference['size_bytes']}, got {path.stat().st_size}"
            )
            continue
        if path not in digest_cache:
            digest_cache[path] = _sha256(path)
        digest = digest_cache[path]
        if digest != reference["sha256"]:
            errors.append(f"checksum mismatch: {path}: expected {reference['sha256']}, got {digest}")
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _campaign_file_references(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = [
        campaign["source"]["relion"]["executable"],
        campaign["source"]["custom_cuda"],
        *campaign["shared_artifacts"],
    ]
    for case in campaign["cases"]:
        references.extend((case["case_config"], case["launcher"]))
        references.extend(case["inputs"])
        references.extend(case["artifacts"])
        report = case["quality"]["trajectory"]["report"]
        if report is not None:
            references.append(report)
    return references


def verify_campaign_files(campaign: dict[str, Any], digest_cache: dict[Path, str]) -> None:
    """Verify every external campaign artifact; missing files fail closed."""
    errors: list[str] = []
    for reference in _campaign_file_references(campaign):
        path = Path(reference["path"])
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != reference["size_bytes"]:
            errors.append(
                f"size mismatch: {path}: expected {reference['size_bytes']}, got {actual_size}"
            )
            continue
        if path not in digest_cache:
            digest_cache[path] = _sha256(path)
        digest = digest_cache[path]
        if digest != reference["sha256"]:
            errors.append(f"checksum mismatch: {path}: expected {reference['sha256']}, got {digest}")
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _diagnostic_file_references(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = [*diagnostic["source_structures"]]
    for run in diagnostic["runs"]:
        references.extend(
            (
                run["source"]["relion"]["executable"],
                run["source"]["custom_cuda"],
                *run["shared_artifacts"],
            )
        )
        for replicate in run["replicates"]:
            references.extend((replicate["launcher"], replicate["case_config"]))
            references.extend(replicate["inputs"])
            references.extend(replicate["artifacts"])
    return references


def verify_diagnostic_files(diagnostic: dict[str, Any], digest_cache: dict[Path, str]) -> None:
    """Verify external files for negative diagnostics, sharing the registry hash cache."""
    errors: list[str] = []
    for reference in _diagnostic_file_references(diagnostic):
        path = Path(reference["path"])
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != reference["size_bytes"]:
            errors.append(
                f"size mismatch: {path}: expected {reference['size_bytes']}, got {actual_size}"
            )
            continue
        if path not in digest_cache:
            digest_cache[path] = _sha256(path)
        digest = digest_cache[path]
        if digest != reference["sha256"]:
            errors.append(f"checksum mismatch: {path}: expected {reference['sha256']}, got {digest}")
    if errors:
        raise RegistryValidationError("\n".join(errors))


def _registered_diagnostic_file_references(diagnostic: dict[str, Any]) -> list[dict[str, str]]:
    schema = diagnostic["schema"]
    artifacts = diagnostic["artifacts"]
    if schema == _K1_DIRECT_FSC_SCHEMA:
        return [artifacts[role] for role in _K1_ARTIFACT_SEALS]
    if schema == _REAL_KCLASS_SINGLE_SEED_SCHEMA:
        run_root = Path(artifacts["run_root"])
        return [
            {
                "path": str(run_root / relative_path),
                "sha256": artifacts[hash_key],
            }
            for hash_key, relative_path in (
                ("submission_manifest_sha256", "submission_manifest.json"),
                ("audit_sha256", "audit/halfmap_audit.json"),
                ("curve_archive_sha256", "audit/halfmap_fsc_curves.npz"),
                ("common_mask_sha256", "audit/common_soft_mask.mrc"),
                ("qualification_stdout_sha256", "logs/qualification-13371069.out"),
                ("qualification_stderr_sha256", "logs/qualification-13371069.err"),
            )
        ]
    raise RegistryValidationError(f"unsupported inline diagnostic schema: {schema!r}")


def verify_registered_diagnostic_files(
    diagnostic: dict[str, Any], digest_cache: dict[Path, str]
) -> None:
    """Verify compact artifacts for an inline-validated diagnostic."""

    errors: list[str] = []
    for reference in _registered_diagnostic_file_references(diagnostic):
        path = Path(reference["path"])
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        if path not in digest_cache:
            digest_cache[path] = _sha256(path)
        digest = digest_cache[path]
        if digest != reference["sha256"]:
            errors.append(f"checksum mismatch: {path}: expected {reference['sha256']}, got {digest}")
    if errors:
        raise RegistryValidationError("\n".join(errors))


def validate_registry(registry_root: Path, *, verify_files: bool = False) -> list[Path]:
    """Validate all registry entries and return their paths."""
    schema_path = registry_root / "schema_v1.json"
    campaign_schema_path = registry_root / "campaign_schema_v1.json"
    diagnostic_schema_path = registry_root / "diagnostic_schema_v1.json"
    entries_dir = registry_root / "entries"
    campaigns_dir = registry_root / "campaigns"
    diagnostics_dir = registry_root / "diagnostics"
    schema = json.loads(schema_path.read_text())
    campaign_schema = json.loads(campaign_schema_path.read_text())
    diagnostic_schema = json.loads(diagnostic_schema_path.read_text())
    entry_paths = sorted(entries_dir.glob("*.json"))
    if not entry_paths:
        raise RegistryValidationError(f"no records found under {entries_dir}")

    records: dict[str, dict[str, Any]] = {}
    digest_cache: dict[Path, str] = {}
    for path in entry_paths:
        record = json.loads(path.read_text())
        try:
            validate_record(record, schema)
            if verify_files:
                verify_record_files(record, digest_cache)
        except RegistryValidationError as error:
            raise RegistryValidationError(f"{path}:\n{error}") from error
        record_id = record["record_id"]
        if path.stem != record_id:
            raise RegistryValidationError(f"{path}: filename must equal record_id")
        if record_id in records:
            raise RegistryValidationError(f"duplicate record_id: {record_id}")
        records[record_id] = record

    for record_id, record in records.items():
        comparator = record["gate"]["comparator_record_id"]
        if comparator is not None and comparator not in records:
            raise RegistryValidationError(f"{record_id}: unknown comparator_record_id {comparator}")
        if comparator is not None and records[comparator]["subject"] != record["subject"]:
            raise RegistryValidationError(f"{record_id}: comparator subject differs")

    campaign_paths = sorted(campaigns_dir.glob("*.json"))
    if not campaign_paths:
        raise RegistryValidationError(f"no campaign scorecards found under {campaigns_dir}")
    campaign_ids: set[str] = set()
    for path in campaign_paths:
        campaign = json.loads(path.read_text())
        try:
            validate_campaign(campaign, campaign_schema)
            if verify_files:
                verify_campaign_files(campaign, digest_cache)
        except RegistryValidationError as error:
            raise RegistryValidationError(f"{path}:\n{error}") from error
        campaign_id = campaign["campaign_id"]
        if path.stem != campaign_id:
            raise RegistryValidationError(f"{path}: filename must equal campaign_id")
        if campaign_id in campaign_ids:
            raise RegistryValidationError(f"duplicate campaign_id: {campaign_id}")
        campaign_ids.add(campaign_id)

    discovered_diagnostic_paths = sorted(diagnostics_dir.glob("*.json"))
    if not discovered_diagnostic_paths:
        raise RegistryValidationError(f"no negative diagnostic records found under {diagnostics_dir}")
    diagnostic_paths: list[Path] = []
    diagnostic_ids: set[str] = set()
    for path in discovered_diagnostic_paths:
        diagnostic = json.loads(path.read_text())
        try:
            route = _diagnostic_registry_route(diagnostic)
        except RegistryValidationError as error:
            raise RegistryValidationError(f"{path}:\n{error}") from error
        if route == "separate":
            continue
        try:
            if route == "synthetic_negative":
                validate_diagnostic(diagnostic, diagnostic_schema)
                if verify_files:
                    verify_diagnostic_files(diagnostic, digest_cache)
                diagnostic_id = diagnostic["diagnostic_id"]
                if path.stem != diagnostic_id:
                    raise RegistryValidationError(f"{path}: filename must equal diagnostic_id")
            else:
                validate_registered_diagnostic(diagnostic)
                if verify_files:
                    verify_registered_diagnostic_files(diagnostic, digest_cache)
                diagnostic_id = _INLINE_DIAGNOSTIC_FILENAMES[diagnostic["schema"]]
                if path.stem != diagnostic_id:
                    raise RegistryValidationError(
                        f"{path}: filename must equal the sealed diagnostic filename {diagnostic_id}"
                    )
        except RegistryValidationError as error:
            raise RegistryValidationError(f"{path}:\n{error}") from error
        if diagnostic_id in diagnostic_ids:
            raise RegistryValidationError(f"duplicate diagnostic_id: {diagnostic_id}")
        diagnostic_ids.add(diagnostic_id)
        diagnostic_paths.append(path)
    if not diagnostic_paths:
        raise RegistryValidationError(
            f"no synthetic negative diagnostic records found under {diagnostics_dir}"
        )
    return [*entry_paths, *campaign_paths, *diagnostic_paths]


def _default_registry_root() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "em"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry_root", nargs="?", type=Path, default=_default_registry_root())
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="rehash all external artifacts and inputs (the current stack is 26 GB)",
    )
    args = parser.parse_args()
    try:
        paths = validate_registry(args.registry_root.resolve(), verify_files=args.verify_files)
    except (OSError, json.JSONDecodeError, RegistryValidationError) as error:
        parser.exit(1, f"EM benchmark registry validation failed:\n{error}\n")
    print(f"Validated {len(paths)} EM benchmark record(s) under {args.registry_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
