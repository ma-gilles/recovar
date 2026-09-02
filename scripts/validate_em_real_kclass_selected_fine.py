#!/usr/bin/env python3
"""Validate the sealed historical K=4 selected-fine diagnostic ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "recovar.em.real_kclass_selected_fine_diagnostic.v1"
EXPECTED_SOURCE_COMMIT = "e05c63a33b588c6bf965634b4c063d2ecde53f72"
EXPECTED_ROWS = [114, 132]
EXPECTED_STACK_INDICES = [1598, 1839]
EXPECTED_FAILED_JOBS = {"13308306", "13308620", "13309898"}
INCORRECT_COARSE_REPORT_SHA256 = "f03f2385c01a1b8f67e67215909a72db69dca95b95f468766f940861553a5722"
CORRECT_COARSE_REPORT_SHA256 = "f03f2385971bc93403b10b8f75c94a421e237aa5c6b51c31b8297e0765a74586"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SelectedFineEvidenceValidationError(ValueError):
    """Raised when the ledger could support a misleading scientific claim."""


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise SelectedFineEvidenceValidationError(f"{label} is missing keys: {missing}")


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise SelectedFineEvidenceValidationError(f"{label} must be an absolute path")
    return Path(value)


def _positive_number(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise SelectedFineEvidenceValidationError(f"{label} must be finite and positive")


def _validate_file_ref(value: dict[str, Any], label: str) -> None:
    _require_keys(value, {"path", "sha256"}, label)
    _absolute_path(value["path"], f"{label}.path")
    if not isinstance(value["sha256"], str) or SHA256_RE.fullmatch(value["sha256"]) is None:
        raise SelectedFineEvidenceValidationError(f"{label}.sha256 must be a lowercase SHA-256")


def _validate_mass(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise SelectedFineEvidenceValidationError(f"{label} must contain four class masses")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(item)
        or item < 0
        for item in value
    ):
        raise SelectedFineEvidenceValidationError(f"{label} must contain finite nonnegative masses")
    if sum(value) <= 0:
        raise SelectedFineEvidenceValidationError(f"{label} must contain positive total mass")
    return [float(item) for item in value]


def _validate_particle(value: dict[str, Any], index: int) -> None:
    label = f"particles[{index}]"
    _require_keys(
        value,
        {
            "recovar_source_row_zero_based",
            "stack_index_one_based",
            "classification",
            "first_observed_nonidentical_boundary",
            "coarse_significance_support_mismatch_count",
            "fine_support_symmetric_difference_count",
            "relion_class_probability_mass",
            "recovar_captured_class_probability_mass",
            "relion_class_winner_zero_based",
            "recovar_class_winner_zero_based",
            "common_centered_raw_score_max_abs",
            "common_centered_rotation_log_prior_max_abs",
            "common_centered_translation_log_prior_max_abs",
            "class_candidate_counts_relion_recovar_common",
            "interpretation",
        },
        label,
    )
    if value["classification"] not in {
        "fine_class_winner_exact_with_support_and_prior_differences",
        "fine_class_winner_mismatch_with_support_and_prior_differences",
    }:
        raise SelectedFineEvidenceValidationError(f"{label} has an unsupported classification")
    if value["first_observed_nonidentical_boundary"] != "coarse_significance_support":
        raise SelectedFineEvidenceValidationError(
            f"{label} must retain coarse significance support as the first observed boundary"
        )
    for key in ("coarse_significance_support_mismatch_count", "fine_support_symmetric_difference_count"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0:
            raise SelectedFineEvidenceValidationError(f"{label}.{key} must be a positive integer")

    relion_mass = _validate_mass(value["relion_class_probability_mass"], f"{label}.relion_class_probability_mass")
    recovar_mass = _validate_mass(
        value["recovar_captured_class_probability_mass"],
        f"{label}.recovar_captured_class_probability_mass",
    )
    relion_winner = value["relion_class_winner_zero_based"]
    recovar_winner = value["recovar_class_winner_zero_based"]
    if relion_winner != max(range(4), key=relion_mass.__getitem__):
        raise SelectedFineEvidenceValidationError(f"{label} RELION winner disagrees with its masses")
    if recovar_winner != max(range(4), key=recovar_mass.__getitem__):
        raise SelectedFineEvidenceValidationError(f"{label} RECOVAR winner disagrees with its masses")
    winners_match = relion_winner == recovar_winner
    classification_says_match = value["classification"].startswith("fine_class_winner_exact")
    if winners_match != classification_says_match:
        raise SelectedFineEvidenceValidationError(f"{label} classification disagrees with its class winners")

    _positive_number(value["common_centered_raw_score_max_abs"], f"{label}.common_centered_raw_score_max_abs")
    if value["common_centered_rotation_log_prior_max_abs"] != 0:
        raise SelectedFineEvidenceValidationError(f"{label} must retain the exact common rotation-prior match")
    _positive_number(
        value["common_centered_translation_log_prior_max_abs"],
        f"{label}.common_centered_translation_log_prior_max_abs",
    )

    counts = value["class_candidate_counts_relion_recovar_common"]
    if not isinstance(counts, list) or len(counts) != 4:
        raise SelectedFineEvidenceValidationError(f"{label} must contain four candidate-count rows")
    for class_index, row in enumerate(counts):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in row)
        ):
            raise SelectedFineEvidenceValidationError(
                f"{label}.class_candidate_counts_relion_recovar_common[{class_index}] must be three nonnegative integers"
            )
        relion_count, recovar_count, common_count = row
        if common_count > min(relion_count, recovar_count):
            raise SelectedFineEvidenceValidationError(f"{label} common candidate count exceeds an engine count")
    if not isinstance(value["interpretation"], str) or not value["interpretation"]:
        raise SelectedFineEvidenceValidationError(f"{label}.interpretation must be non-empty")


def validate_evidence(data: dict[str, Any]) -> None:
    """Validate the checked ledger without reading external artifacts."""

    _require_keys(
        data,
        {
            "schema",
            "recorded_at",
            "provenance_correction",
            "admission_status",
            "dataset",
            "scope",
            "source",
            "capture",
            "artifacts",
            "particles",
            "excluded_outputs",
            "failed_or_cancelled_capture_attempts",
            "strict_claim_boundary",
        },
        "ledger",
    )
    if data["schema"] != SCHEMA:
        raise SelectedFineEvidenceValidationError(f"unexpected schema: {data['schema']!r}")
    if data["admission_status"] != "DIAGNOSTIC_ONLY":
        raise SelectedFineEvidenceValidationError("selected-fine evidence cannot claim benchmark admission")
    if "EMPIAR-10076" not in str(data["dataset"]):
        raise SelectedFineEvidenceValidationError("ledger must retain the frozen EMPIAR-10076 fixture")

    correction = data["provenance_correction"]
    _require_keys(
        correction,
        {"corrected_at", "field", "previous_value", "corrected_value", "basis"},
        "provenance_correction",
    )
    if (
        correction["field"] != "artifacts.coarse_report.sha256"
        or correction["previous_value"] != INCORRECT_COARSE_REPORT_SHA256
        or correction["corrected_value"] != CORRECT_COARSE_REPORT_SHA256
        or "transcription error" not in str(correction["basis"])
    ):
        raise SelectedFineEvidenceValidationError("ledger must retain the audited coarse-report hash correction")

    scope = data["scope"]
    _require_keys(
        scope,
        {
            "K",
            "iteration",
            "selected_recovar_source_rows_zero_based",
            "selected_stack_indices_one_based",
            "current_size",
            "match_policy",
            "rotation_frobenius_tolerance",
            "translation_max_abs_tolerance_pixels",
            "unmatched_policy",
        },
        "scope",
    )
    if scope["K"] != 4 or scope["iteration"] != 1 or scope["current_size"] != 56:
        raise SelectedFineEvidenceValidationError("scope must retain K=4, iteration 1, and current_size=56")
    if scope["selected_recovar_source_rows_zero_based"] != EXPECTED_ROWS:
        raise SelectedFineEvidenceValidationError("scope must retain the two frozen RECOVAR source rows")
    if scope["selected_stack_indices_one_based"] != EXPECTED_STACK_INDICES:
        raise SelectedFineEvidenceValidationError("scope must retain the two frozen stack indices")
    _positive_number(scope["rotation_frobenius_tolerance"], "scope.rotation_frobenius_tolerance")
    _positive_number(
        scope["translation_max_abs_tolerance_pixels"],
        "scope.translation_max_abs_tolerance_pixels",
    )
    if "physical rotation matrices" not in str(scope["match_policy"]):
        raise SelectedFineEvidenceValidationError("scope must retain physical-coordinate matching")
    if "never map to the nearest row" not in str(scope["unmatched_policy"]):
        raise SelectedFineEvidenceValidationError("scope must forbid nearest-row joins")

    source = data["source"]
    _require_keys(source, {"commit", "immutable_checkout", "analysis_checkout"}, "source")
    if source["commit"] != EXPECTED_SOURCE_COMMIT:
        raise SelectedFineEvidenceValidationError("ledger source is not the frozen historical treatment commit")
    _absolute_path(source["immutable_checkout"], "source.immutable_checkout")
    _absolute_path(source["analysis_checkout"], "source.analysis_checkout")

    capture = data["capture"]
    _require_keys(
        capture,
        {
            "job_id",
            "state",
            "exit_code",
            "elapsed_s",
            "node_observed_live",
            "gpu",
            "gpu_uuid",
            "live_scontrol_req_tres",
            "live_scontrol_alloc_tres",
            "accounting_warning",
            "run_root",
            "peak_hbm_mib",
            "max_rss_kib",
            "formal_performance_result",
        },
        "capture",
    )
    if not str(capture["job_id"]).isdigit() or capture["state"] != "COMPLETED" or capture["exit_code"] != "0:0":
        raise SelectedFineEvidenceValidationError("capture must retain the completed zero-exit Slurm job")
    _positive_number(capture["elapsed_s"], "capture.elapsed_s")
    if capture["live_scontrol_req_tres"] != capture["live_scontrol_alloc_tres"]:
        raise SelectedFineEvidenceValidationError("capture ReqTRES and AllocTRES differ")
    if "gres/gpu=1" not in capture["live_scontrol_req_tres"]:
        raise SelectedFineEvidenceValidationError("capture must retain the exact one-GPU allocation")
    if not str(capture["gpu"]).startswith("NVIDIA H100") or not str(capture["gpu_uuid"]).startswith("GPU-"):
        raise SelectedFineEvidenceValidationError("capture must retain its observed H100 identity")
    _absolute_path(capture["run_root"], "capture.run_root")
    if capture["peak_hbm_mib"] is not None or capture["max_rss_kib"] is not None:
        raise SelectedFineEvidenceValidationError("capture must not invent unavailable HBM or RSS measurements")
    if capture["formal_performance_result"] is not False:
        raise SelectedFineEvidenceValidationError("diagnostic capture cannot claim a formal performance result")
    if "unrelated CPU job" not in str(capture["accounting_warning"]):
        raise SelectedFineEvidenceValidationError("capture must retain the Slurm accounting-ID warning")

    artifacts = data["artifacts"]
    required_roles = {"selection", "coarse_report", "fine_report", "capture_manifest", "sbatch", "stdout", "stderr"}
    if not isinstance(artifacts, dict):
        raise SelectedFineEvidenceValidationError("artifacts must be keyed by role")
    _require_keys(artifacts, required_roles, "artifacts")
    for role, reference in artifacts.items():
        _validate_file_ref(reference, f"artifacts.{role}")
    if artifacts["coarse_report"]["sha256"] != correction["corrected_value"]:
        raise SelectedFineEvidenceValidationError("coarse-report hash disagrees with its provenance correction")

    particles = data["particles"]
    if not isinstance(particles, list) or len(particles) != 2:
        raise SelectedFineEvidenceValidationError("ledger must retain exactly two selected particles")
    for index, particle in enumerate(particles):
        _validate_particle(particle, index)
    if [particle["recovar_source_row_zero_based"] for particle in particles] != EXPECTED_ROWS:
        raise SelectedFineEvidenceValidationError("particle identities disagree with the frozen source rows")
    if [particle["stack_index_one_based"] for particle in particles] != EXPECTED_STACK_INDICES:
        raise SelectedFineEvidenceValidationError("particle identities disagree with the frozen stack indices")
    if sum(
        particle["relion_class_winner_zero_based"] == particle["recovar_class_winner_zero_based"]
        for particle in particles
    ) != 1:
        raise SelectedFineEvidenceValidationError("ledger must retain one matching and one mismatching class winner")

    excluded = data["excluded_outputs"]
    _require_keys(excluded, {"path", "reason"}, "excluded_outputs")
    _absolute_path(excluded["path"], "excluded_outputs.path")
    if "nearest-matrix join" not in str(excluded["reason"]) or "not evidence" not in str(excluded["reason"]):
        raise SelectedFineEvidenceValidationError("invalid nearest-row outputs must remain explicitly excluded")

    failed = data["failed_or_cancelled_capture_attempts"]
    if not isinstance(failed, list) or {str(item.get("job_id")) for item in failed} != EXPECTED_FAILED_JOBS:
        raise SelectedFineEvidenceValidationError("ledger must retain all three excluded capture attempts")
    if any(not isinstance(item.get("reason"), str) or not item["reason"] for item in failed):
        raise SelectedFineEvidenceValidationError("excluded capture attempts require reasons")

    claim = data["strict_claim_boundary"]
    if not isinstance(claim, str) or not all(
        phrase in claim for phrase in ("does not establish", "half-map resolution", "formal speed ratio")
    ):
        raise SelectedFineEvidenceValidationError(
            "strict claim boundary must deny map-quality, half-map, and speed conclusions"
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(data: dict[str, Any]) -> None:
    """Verify every compact artifact retained by the checked ledger."""

    for role, reference in data["artifacts"].items():
        path = Path(reference["path"])
        if not path.is_file():
            raise SelectedFineEvidenceValidationError(f"missing sealed artifact {role}: {path}")
        if _hash_file(path) != reference["sha256"]:
            raise SelectedFineEvidenceValidationError(f"checksum mismatch for {role}: {path}")


def default_ledger_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "em"
        / "diagnostics"
        / "real-kclass-selected-fine-10076-20260901.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=default_ledger_path())
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.ledger.read_text())
        validate_evidence(data)
        if args.verify_files:
            verify_files(data)
    except (OSError, json.JSONDecodeError, SelectedFineEvidenceValidationError) as error:
        parser.exit(1, f"K=4 selected-fine evidence validation failed:\n{error}\n")
    print(f"Validated {len(data['particles'])} historical selected-fine particle(s): {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
