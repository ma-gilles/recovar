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
    entries_dir = registry_root / "entries"
    schema = json.loads(schema_path.read_text())
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
    return entry_paths


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
