#!/usr/bin/env python3
"""Validate checked-in RECOVAR EM benchmark evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class RegistryValidationError(ValueError):
    """Raised when a benchmark registry record is malformed or inconsistent."""


def _json_path(parts: list[Any]) -> str:
    return ".".join(str(part) for part in parts) or "<record>"


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


def validate_registry(registry_root: Path, *, verify_files: bool = False) -> list[Path]:
    """Validate all registry entries and return their paths."""
    schema_path = registry_root / "schema_v1.json"
    campaign_schema_path = registry_root / "campaign_schema_v1.json"
    entries_dir = registry_root / "entries"
    campaigns_dir = registry_root / "campaigns"
    schema = json.loads(schema_path.read_text())
    campaign_schema = json.loads(campaign_schema_path.read_text())
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
    return [*entry_paths, *campaign_paths]


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
