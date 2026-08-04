#!/usr/bin/env python3
"""Run the K=4 fine-operand comparator only on two admitted boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .compare_k4_relion_recovar_fine_operands import compare
else:
    from compare_k4_relion_recovar_fine_operands import compare  # type: ignore[no-redef]

SCHEMA = "recovar.em_k4_admitted_fine_operand_comparison.v1"
NATIVE_SCHEMA = "recovar.em_k4_native_class1_fine_operand_admission.v1"
RECOVAR_SCHEMA = "recovar.em_k4_contribution_repeatability.v1"
COMPARISON_SCHEMA = "k4_relion_recovar_fine_operand_comparison_v8"
EXPECTED_SCOPE = "iteration2_half1_source53722_class1_only"
EXPECTED_NATIVE_GATES = 7
EXPECTED_RECOVAR_GATES = 3
EXPECTED_NATIVE_ROTATION = 1790
EXPECTED_RECOVAR_GLOBAL_ROTATION = 4446
EXPECTED_CANDIDATES = 96


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _passed_fixed_metric(
    report: dict[str, Any],
    *,
    expected: int,
) -> bool:
    metric = report.get("fixed_metric")
    return bool(
        isinstance(metric, dict)
        and metric.get("passing") == expected
        and metric.get("evaluated") == expected
        and isinstance(metric.get("gates"), dict)
        and len(metric["gates"]) == expected
        and all(metric["gates"].values())
    )


def validate_admissions(
    native: dict[str, Any],
    recovar: dict[str, Any],
) -> None:
    """Reject any broad, incomplete, or cross-device artifact admission."""

    native_scope = native.get("scope")
    _require(
        native.get("schema") == NATIVE_SCHEMA
        and native.get("status") == "complete"
        and native.get("accepted") is True
        and native.get("target_class_local_operand_use_allowed") is True
        and native.get("allclass_cross_engine_attribution_allowed") is False
        and native.get("scorecard_change_admissible") is False
        and native.get("correlation_used") is False
        and _passed_fixed_metric(native, expected=EXPECTED_NATIVE_GATES)
        and isinstance(native_scope, dict)
        and native_scope.get("iteration") == 2
        and native_scope.get("source_row_zero_based") == 53722
        and native_scope.get("stack_index_one_based") == 53723
        and native_scope.get("class_one_based") == 1
        and native_scope.get("rotation_local") == EXPECTED_NATIVE_ROTATION
        and native_scope.get("candidate_count") == EXPECTED_CANDIDATES,
        "native operand admission did not pass the fixed target-local 7/7 gate",
    )
    _require(
        recovar.get("schema") == RECOVAR_SCHEMA
        and recovar.get("status") == "complete"
        and recovar.get("accepted") is True
        and recovar.get("cross_engine_attribution_allowed") is True
        and recovar.get("cross_engine_scope") == EXPECTED_SCOPE
        and recovar.get("allclass_cross_engine_attribution_allowed") is False
        and recovar.get("scorecard_change_admissible") is False
        and recovar.get("correlation_used") is False
        and _passed_fixed_metric(recovar, expected=EXPECTED_RECOVAR_GATES),
        "RECOVAR contribution admission did not pass the fixed target-local 3/3 gate",
    )
    _require(
        isinstance(native.get("gpu_uuid"), str)
        and native["gpu_uuid"]
        and native["gpu_uuid"] == recovar.get("gpu_uuid"),
        "native and RECOVAR admissions are not from the same physical GPU",
    )


def _validated_artifact(record: dict[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    _require(path.is_absolute() and path.is_file(), f"{label} path is invalid")
    _require(record.get("sha256") == _sha256(path), f"{label} hash changed")
    return path


def build_report(
    *,
    native_admission_path: Path,
    recovar_admission_path: Path,
    reference_path: Path,
    particle_diameter_angstrom: float,
    mask_edge_pixels: float,
    compare_fn: Callable[..., dict[str, Any]] = compare,
) -> dict[str, Any]:
    native = _load_json(native_admission_path)
    recovar = _load_json(recovar_admission_path)
    validate_admissions(native, recovar)

    native_operand = _validated_artifact(
        native.get("artifacts", {}).get("fine_operand", {}),
        label="native fine operand",
    )
    recovar_contribution = _validated_artifact(
        recovar.get("comparisons", {}).get("contribution", {}).get("reference", {}),
        label="RECOVAR contribution",
    )
    _require(
        reference_path.is_absolute() and reference_path.is_file(),
        "reference path is invalid",
    )

    comparison = compare_fn(
        native_operand,
        recovar_contribution,
        reference_path,
        recovar_global_rotation=EXPECTED_RECOVAR_GLOBAL_ROTATION,
        particle_diameter_angstrom=particle_diameter_angstrom,
        mask_edge_pixels=mask_edge_pixels,
    )
    validation = comparison.get("capture_validation", {})
    scope = comparison.get("scope", {})
    _require(
        comparison.get("schema") == COMPARISON_SCHEMA
        and comparison.get("status") == "complete"
        and validation.get("status") == "accepted"
        and validation.get("candidate_count") == EXPECTED_CANDIDATES
        and validation.get("exact_production_replay_count", 0) >= 2
        and len(comparison.get("candidates", [])) == EXPECTED_CANDIDATES
        and scope.get("original_index_zero_based") == 53722
        and scope.get("stack_index_one_based") == 53723
        and scope.get("relion_rotation_local") == EXPECTED_NATIVE_ROTATION
        and scope.get("recovar_global_rotation")
        == EXPECTED_RECOVAR_GLOBAL_ROTATION,
        "fine-operand comparison escaped the admitted multi-candidate scope",
    )
    _require(
        comparison.get("classification_basis") in {"centered_raw_diff2", "none"},
        "multi-candidate classification did not use the centered score boundary",
    )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": comparison["classification"],
        "classification_basis": comparison["classification_basis"],
        "scope": {
            "target_class_local_only": True,
            "iteration": 2,
            "source_row_zero_based": 53722,
            "stack_index_one_based": 53723,
            "class_one_based": 1,
            "native_rotation_local": EXPECTED_NATIVE_ROTATION,
            "recovar_global_rotation": EXPECTED_RECOVAR_GLOBAL_ROTATION,
            "candidate_count": EXPECTED_CANDIDATES,
            "gpu_uuid": native["gpu_uuid"],
            "allclass_cross_engine_attribution_allowed": False,
        },
        "comparison": comparison,
        "inputs": {
            "native_admission": {
                "path": str(native_admission_path.resolve()),
                "sha256": _sha256(native_admission_path),
            },
            "recovar_admission": {
                "path": str(recovar_admission_path.resolve()),
                "sha256": _sha256(recovar_admission_path),
            },
            "reference": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256(reference_path),
            },
        },
        "metric_policy": (
            "admitted original float32 operand bits and candidate-relative centered "
            "raw diff2; no all-class promotion, map-score promotion, or correlation"
        ),
        "scorecard_change_admissible": False,
        "correlation_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-admission", type=Path, required=True)
    parser.add_argument("--recovar-admission", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        native_admission_path=args.native_admission,
        recovar_admission_path=args.recovar_admission,
        reference_path=args.reference,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        mask_edge_pixels=args.mask_edge_pixels,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "classification_basis": report["classification_basis"],
                "scope": report["scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
