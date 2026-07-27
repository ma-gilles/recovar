#!/usr/bin/env python3
"""Measure exact repeatability of two fixed-scope K=4 pass-2 NPZ panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "recovar.k4_preprocess_repeatability.v1"
PROBABILITY_FIELDS = frozenset({"probs", "reconstruction_probs"})
FILENAME_PATTERN = re.compile(
    r"pass2_orig(?P<target>[0-9]{6})_class(?P<class>[0-9]{3})_cs(?P<size>[0-9]{3})[.]npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _discover(
    directory: Path,
    *,
    expected_target_count: int,
    expected_class_count: int,
    expected_current_size: int,
) -> dict[tuple[int, int], Path]:
    _require(directory.is_dir(), f"pass-2 directory does not exist: {directory}")
    paths = sorted(directory.glob("*.npz"))
    expected_count = expected_target_count * expected_class_count
    _require(
        len(paths) == expected_count,
        f"{directory} must contain exactly {expected_count} NPZ files; found {len(paths)}",
    )
    result: dict[tuple[int, int], Path] = {}
    for path in paths:
        match = FILENAME_PATTERN.fullmatch(path.name)
        _require(match is not None, f"unexpected pass-2 filename: {path.name}")
        target = int(match.group("target"))
        class_one_based = int(match.group("class"))
        current_size = int(match.group("size"))
        _require(current_size == expected_current_size, f"current size changed in {path.name}")
        _require(
            1 <= class_one_based <= expected_class_count,
            f"class index is out of scope in {path.name}",
        )
        key = (target, class_one_based)
        _require(key not in result, f"duplicate target/class key: {key}")
        result[key] = path
    targets = sorted({target for target, _class in result})
    _require(
        len(targets) == expected_target_count,
        f"expected {expected_target_count} targets; found {len(targets)}",
    )
    expected_keys = {
        (target, class_one_based)
        for target in targets
        for class_one_based in range(1, expected_class_count + 1)
    }
    _require(set(result) == expected_keys, "target/class panel is incomplete")
    return result


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _field_stats(parts: list[np.ndarray], *, element_count: int, exact: bool) -> dict[str, Any]:
    values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    return {
        "exact": exact,
        "element_count": element_count,
        "finite_pair_count": int(values.size),
        "nonzero_finite_delta_count": int(np.count_nonzero(values)),
        "max_abs_finite_delta": float(np.max(np.abs(values))) if values.size else 0.0,
        "finite_delta_l2": float(np.linalg.norm(values)),
        "finite_delta_residual_energy": float(np.vdot(values, values).real),
    }


def analyze(
    *,
    reference_directory: Path,
    repeat_directory: Path,
    expected_target_count: int,
    expected_class_count: int,
    expected_current_size: int,
) -> dict[str, Any]:
    for value, label in (
        (expected_target_count, "target count"),
        (expected_class_count, "class count"),
        (expected_current_size, "current size"),
    ):
        _require(value > 0, f"expected {label} must be positive")
    reference_paths = _discover(
        reference_directory,
        expected_target_count=expected_target_count,
        expected_class_count=expected_class_count,
        expected_current_size=expected_current_size,
    )
    repeat_paths = _discover(
        repeat_directory,
        expected_target_count=expected_target_count,
        expected_class_count=expected_class_count,
        expected_current_size=expected_current_size,
    )
    _require(set(reference_paths) == set(repeat_paths), "panel target/class keys differ")

    field_delta_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    field_element_counts: dict[str, int] = defaultdict(int)
    field_exact: dict[str, bool] = defaultdict(lambda: True)
    mismatch_field_files: dict[str, list[str]] = defaultdict(list)
    files = []
    byte_exact_count = 0
    array_exact_count = 0
    inputs = []
    probability_masses: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for key in sorted(reference_paths):
        reference_path = reference_paths[key]
        repeat_path = repeat_paths[key]
        reference_sha = _sha256(reference_path)
        repeat_sha = _sha256(repeat_path)
        byte_exact = reference_sha == repeat_sha
        byte_exact_count += int(byte_exact)
        reference = _load(reference_path)
        repeat = _load(repeat_path)
        _require(
            set(reference) == set(repeat),
            f"NPZ field sets differ for {reference_path.name}",
        )
        mismatch_fields = []
        for field in sorted(reference):
            lhs = np.asarray(reference[field])
            rhs = np.asarray(repeat[field])
            _require(
                lhs.shape == rhs.shape and lhs.dtype == rhs.dtype,
                f"field shape/dtype changed for {reference_path.name}:{field}",
            )
            exact = bool(np.array_equal(lhs, rhs, equal_nan=True))
            field_exact[field] &= exact
            field_element_counts[field] += int(lhs.size)
            if not exact:
                mismatch_fields.append(field)
                mismatch_field_files[field].append(reference_path.name)
            if np.issubdtype(lhs.dtype, np.number):
                lhs64 = lhs.astype(np.float64)
                rhs64 = rhs.astype(np.float64)
                finite = np.isfinite(lhs64) & np.isfinite(rhs64)
                _require(
                    np.array_equal(np.isfinite(lhs64), np.isfinite(rhs64)),
                    f"finite-value pattern changed for {reference_path.name}:{field}",
                )
                field_delta_parts[field].append((rhs64[finite] - lhs64[finite]).reshape(-1))
        array_exact = not mismatch_fields
        array_exact_count += int(array_exact)
        target, class_one_based = key
        reference_mass = float(np.sum(reference["probs"], dtype=np.float64))
        repeat_mass = float(np.sum(repeat["probs"], dtype=np.float64))
        probability_masses[target].append(
            {
                "class_one_based": class_one_based,
                "reference": reference_mass,
                "repeat": repeat_mass,
                "repeat_minus_reference": repeat_mass - reference_mass,
            }
        )
        files.append(
            {
                "name": reference_path.name,
                "target_original_index": target,
                "class_one_based": class_one_based,
                "byte_exact": byte_exact,
                "array_exact": array_exact,
                "mismatch_fields": mismatch_fields,
                "reference_sha256": reference_sha,
                "repeat_sha256": repeat_sha,
            }
        )
        inputs.append(
            {
                "name": reference_path.name,
                "reference_path": str(reference_path.resolve()),
                "reference_sha256": reference_sha,
                "repeat_path": str(repeat_path.resolve()),
                "repeat_sha256": repeat_sha,
            }
        )

    field_stats = {
        field: _field_stats(
            field_delta_parts[field],
            element_count=field_element_counts[field],
            exact=field_exact[field],
        )
        for field in sorted(field_element_counts)
    }
    nonprobability_mismatch_fields = sorted(
        field for field, exact in field_exact.items() if not exact and field not in PROBABILITY_FIELDS
    )
    class_rows = []
    class_prediction_exact = True
    for target in sorted(probability_masses):
        masses = sorted(probability_masses[target], key=lambda row: row["class_one_based"])
        reference_prediction = int(np.argmax([row["reference"] for row in masses]) + 1)
        repeat_prediction = int(np.argmax([row["repeat"] for row in masses]) + 1)
        exact = reference_prediction == repeat_prediction
        class_prediction_exact &= exact
        class_rows.append(
            {
                "target_original_index": target,
                "reference_prediction_class_one_based": reference_prediction,
                "repeat_prediction_class_one_based": repeat_prediction,
                "prediction_exact": exact,
                "class_probability_mass": masses,
            }
        )

    all_arrays_exact = array_exact_count == len(files)
    score_topology_exact = not nonprobability_mismatch_fields
    if all_arrays_exact:
        classification = "all_arrays_exact"
        status = "complete"
    elif score_topology_exact and class_prediction_exact:
        classification = "exact_score_topology_posterior_roundoff_only"
        status = "complete"
    else:
        classification = "score_topology_or_class_repeatability_failure"
        status = "rejected"
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "classification": classification,
        "classification_rule": (
            "complete iff every array is exact, or every non-probability field and every "
            "class prediction are exact; probability deltas are measured without tolerance"
        ),
        "scorecard_change_admissible": False,
        "scope": {
            "target_count": expected_target_count,
            "class_count": expected_class_count,
            "current_size": expected_current_size,
            "file_count": len(files),
            "byte_exact_file_count": byte_exact_count,
            "array_exact_file_count": array_exact_count,
            "class_prediction_exact_count": sum(row["prediction_exact"] for row in class_rows),
        },
        "strict_gates": {
            "score_and_topology_fields_exact": score_topology_exact,
            "class_predictions_exact": class_prediction_exact,
            "nonprobability_mismatch_fields": nonprobability_mismatch_fields,
        },
        "field_mismatch_file_count": {
            field: len(paths) for field, paths in sorted(mismatch_field_files.items())
        },
        "field_stats": field_stats,
        "class_predictions": class_rows,
        "files": files,
        "inputs": inputs,
        "quality_metric_policy": {
            "map_gate": "not evaluated by this preprocessing repeatability diagnostic",
            "correlation_computed": False,
            "exact_score_topology_required": True,
            "posterior_deltas_reported_without_tolerance": True,
        },
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status, "analyzer repository is dirty")
    return head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--repeat-directory", required=True, type=Path)
    parser.add_argument("--expected-target-count", required=True, type=int)
    parser.add_argument("--expected-class-count", required=True, type=int)
    parser.add_argument("--expected-current-size", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        reference_directory=args.reference_directory,
        repeat_directory=args.repeat_directory,
        expected_target_count=args.expected_target_count,
        expected_class_count=args.expected_class_count,
        expected_current_size=args.expected_current_size,
    )
    report["analyzer_repo_head"] = _clean_repo_head(args.repo)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "classification": report["classification"], **report["scope"]}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
