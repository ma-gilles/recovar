#!/usr/bin/env python3
"""Audit scientific inertness of the native K=4 coarse-score capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import pandas as pd
import starfile

from scripts.analyze_relion_control_capture_inertness import _fsc_auc

SCHEMA = "recovar.relion_k4_coarse_score_capture_inertness.v1"
MAP_FSC_AUC_FLOOR = 0.999999
MAP_RELATIVE_L2_CEILING = 1e-5
DECISION_FIELDS = (
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnClassNumber",
    "rlnNrOfSignificantSamples",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_table(path: Path) -> pd.DataFrame:
    document = starfile.read(path)
    if isinstance(document, pd.DataFrame):
        candidates = [document] if "rlnImageName" in document.columns else []
    else:
        candidates = [
            table
            for table in document.values()
            if isinstance(table, pd.DataFrame) and "rlnImageName" in table.columns
        ]
    _require(len(candidates) == 1, f"{path} must have one particle table")
    return candidates[0]


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _compare_particle_tables(control: pd.DataFrame, instrumented: pd.DataFrame) -> dict[str, Any]:
    identity = "rlnImageName"
    _require(identity in control and identity in instrumented, "particle identity is absent")
    _require(set(control.columns) == set(instrumented.columns), "particle columns differ")
    control_ids = control[identity].astype(str).to_numpy()
    instrumented_ids = instrumented[identity].astype(str).to_numpy()
    _require(np.unique(control_ids).size == control_ids.size, "control particle identities repeat")
    _require(
        np.unique(instrumented_ids).size == instrumented_ids.size,
        "instrumented particle identities repeat",
    )
    _require(set(control_ids) == set(instrumented_ids), "particle identity sets differ")
    left = control.set_index(identity, drop=False)
    right = instrumented.set_index(identity, drop=False).loc[left.index]

    fields = {}
    for field in left.columns:
        lhs = left[field].to_numpy()
        rhs = right[field].to_numpy()
        mismatch = np.asarray(lhs != rhs)
        positions = np.flatnonzero(mismatch)
        numeric = np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number)
        fields[field] = {
            "exact": bool(not positions.size),
            "mismatch_count": int(positions.size),
            "max_abs": float(np.max(np.abs(lhs - rhs))) if numeric and positions.size else 0.0 if numeric else None,
            "examples": [
                {
                    "image_identity": str(left.iloc[position][identity]),
                    "control": _json_scalar(lhs[position]),
                    "instrumented": _json_scalar(rhs[position]),
                }
                for position in positions[:8]
            ],
        }
    return {
        "particle_count": int(len(left)),
        "raw_row_order_exact": bool(np.array_equal(control_ids, instrumented_ids)),
        "exact_field_count": sum(field["exact"] for field in fields.values()),
        "field_count": len(fields),
        "fields": fields,
    }


def _map_metrics(control_path: Path, instrumented_path: Path) -> dict[str, Any]:
    with mrcfile.open(control_path, permissive=False) as handle:
        control = np.asarray(handle.data, dtype=np.float32).copy()
    with mrcfile.open(instrumented_path, permissive=False) as handle:
        instrumented = np.asarray(handle.data, dtype=np.float32).copy()
    _require(control.shape == instrumented.shape and control.ndim == 3, "map shapes differ")
    difference = instrumented.astype(np.float64) - control.astype(np.float64)
    denominator = float(np.linalg.norm(control.astype(np.float64).reshape(-1)))
    _require(denominator > 0, "control map norm is zero")
    return {
        "shape": list(control.shape),
        "byte_exact": bool(control_path.read_bytes() == instrumented_path.read_bytes()),
        "array_exact": bool(np.array_equal(control, instrumented)),
        "fsc_auc": _fsc_auc(control, instrumented),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(difference))),
        "control_sha256": _sha256(control_path),
        "instrumented_sha256": _sha256(instrumented_path),
    }


def _iteration_report(
    control_dir: Path,
    instrumented_dir: Path,
    *,
    iteration: int,
    n_classes: int,
) -> dict[str, Any]:
    stem = f"run_it{iteration:03d}"
    control_star = control_dir / f"{stem}_data.star"
    instrumented_star = instrumented_dir / f"{stem}_data.star"
    tables = _compare_particle_tables(
        _particle_table(control_star),
        _particle_table(instrumented_star),
    )
    maps = []
    for class_id in range(1, n_classes + 1):
        control_map = control_dir / f"{stem}_class{class_id:03d}.mrc"
        instrumented_map = instrumented_dir / f"{stem}_class{class_id:03d}.mrc"
        maps.append(
            {
                "class_id_one_based": class_id,
                "control_path": str(control_map.resolve()),
                "instrumented_path": str(instrumented_map.resolve()),
                **_map_metrics(control_map, instrumented_map),
            }
        )
    return {
        "iteration": iteration,
        "control_data_star": str(control_star.resolve()),
        "instrumented_data_star": str(instrumented_star.resolve()),
        "data_star_byte_exact": bool(control_star.read_bytes() == instrumented_star.read_bytes()),
        "particle_table": tables,
        "maps": maps,
    }


def build_report(
    *,
    control_dir: Path,
    instrumented_dir: Path,
    capture_validation_path: Path,
    n_classes: int,
) -> dict[str, Any]:
    validation = json.loads(capture_validation_path.read_text())
    _require(validation.get("capture_ready") is True, "native capture did not validate")
    _require(n_classes > 0, "class count must be positive")
    iteration_zero = _iteration_report(
        control_dir,
        instrumented_dir,
        iteration=0,
        n_classes=n_classes,
    )
    iteration_one = _iteration_report(
        control_dir,
        instrumented_dir,
        iteration=1,
        n_classes=n_classes,
    )

    iteration_zero_exact = (
        all(field["exact"] for field in iteration_zero["particle_table"]["fields"].values())
        and all(row["array_exact"] for row in iteration_zero["maps"])
    )
    decision_fields_exact = all(
        iteration_one["particle_table"]["fields"][field]["exact"]
        for field in DECISION_FIELDS
    )
    maps_within_floor = all(
        row["fsc_auc"] >= MAP_FSC_AUC_FLOOR
        and row["relative_l2"] <= MAP_RELATIVE_L2_CEILING
        for row in iteration_one["maps"]
    )
    bitwise_inert = (
        iteration_one["data_star_byte_exact"]
        and all(row["byte_exact"] for row in iteration_one["maps"])
    )
    accepted = iteration_zero_exact and decision_fields_exact and maps_within_floor
    return {
        "schema": SCHEMA,
        "status": "pass" if accepted else "rejected",
        "classification": (
            "bitwise_inert"
            if accepted and bitwise_inert
            else "scientifically_inert_at_recorded_floor_but_not_bitwise_inert"
            if accepted
            else "capture_inertness_not_established"
        ),
        "metric_policy": (
            "iteration-0 arrays and particle state must be exact; iteration-1 pose/class/support "
            "decisions must be exact; every class-map FSC-AUC must be at least 0.999999 and "
            "relative L2 at most 1e-5"
        ),
        "thresholds": {
            "map_fsc_auc_floor": MAP_FSC_AUC_FLOOR,
            "map_relative_l2_ceiling": MAP_RELATIVE_L2_CEILING,
        },
        "strict_gate": {
            "capture_validated": True,
            "iteration_zero_numeric_state_exact": iteration_zero_exact,
            "iteration_one_decision_fields_exact": decision_fields_exact,
            "iteration_one_maps_within_floor": maps_within_floor,
            "bitwise_inert": bitwise_inert,
        },
        "capture_validation": {
            "path": str(capture_validation_path.resolve()),
            "sha256": _sha256(capture_validation_path),
            "particle_count": validation["particle_count"],
            "candidate_count": validation["candidate_count"],
        },
        "iterations": [iteration_zero, iteration_one],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--instrumented-dir", required=True, type=Path)
    parser.add_argument("--capture-validation", required=True, type=Path)
    parser.add_argument("--n-classes", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        control_dir=args.control_dir,
        instrumented_dir=args.instrumented_dir,
        capture_validation_path=args.capture_validation,
        n_classes=args.n_classes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
