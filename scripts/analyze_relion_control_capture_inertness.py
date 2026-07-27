#!/usr/bin/env python3
"""Audit whether a passive RELION particle capture changed iteration-1 output.

Particle rows in RELION output are not dataset-order rows.  The target is
therefore supplied and validated as an exact ``rlnImageName`` identity rather
than recovered with a positional ``iloc`` lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import pandas as pd
import starfile

SCHEMA = "em_relion_iteration1_particle_state_inertness_v3"
IDENTITY_FIELD = "rlnImageName"
PARTICLE_FIELDS = (
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnClassNumber",
    "rlnMaxValueProbDistribution",
    "rlnNrOfSignificantSamples",
)
MAP_FSC_AUC_THRESHOLD = 0.999999
MAX_MISMATCH_EXAMPLES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def _particle_table(path: Path) -> pd.DataFrame:
    document = starfile.read(path)
    if isinstance(document, pd.DataFrame):
        candidates = [document] if IDENTITY_FIELD in document.columns else []
    else:
        candidates = [
            table
            for table in document.values()
            if isinstance(table, pd.DataFrame) and IDENTITY_FIELD in table.columns
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"{path} must contain exactly one table with {IDENTITY_FIELD}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _validate_particle_table(table: pd.DataFrame, *, source: str) -> np.ndarray:
    missing = [field for field in (IDENTITY_FIELD, *PARTICLE_FIELDS) if field not in table.columns]
    if missing:
        raise ValueError(f"{source} is missing particle fields: {missing}")
    identities = table[IDENTITY_FIELD].astype(str).to_numpy()
    if np.any(identities == ""):
        raise ValueError(f"{source} contains an empty {IDENTITY_FIELD}")
    unique, counts = np.unique(identities, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        raise ValueError(f"{source} contains duplicate identities: {duplicates[:4].tolist()}")
    return identities


def _validate_target_identity(target_image_identity: str, target_original_index: int) -> int:
    if target_original_index < 0:
        raise ValueError("target original index must be nonnegative")
    match = re.fullmatch(r"([1-9][0-9]*)@(.+)", target_image_identity)
    if match is None:
        raise ValueError("target image identity must have a positive one-based stack prefix")
    stack_index = int(match.group(1))
    if stack_index != target_original_index + 1:
        raise ValueError(
            "target image identity stack prefix does not equal target original index plus one"
        )
    return stack_index


def compare_particle_tables(
    control: pd.DataFrame,
    capture: pd.DataFrame,
    *,
    target_original_index: int,
    target_image_identity: str,
) -> dict[str, Any]:
    """Compare all audited fields after exact particle-identity alignment."""

    stack_index = _validate_target_identity(target_image_identity, target_original_index)
    control_ids = _validate_particle_table(control, source="control particle table")
    capture_ids = _validate_particle_table(capture, source="capture particle table")
    if set(control_ids) != set(capture_ids):
        missing_capture = sorted(set(control_ids) - set(capture_ids))
        missing_control = sorted(set(capture_ids) - set(control_ids))
        raise ValueError(
            "control and capture particle identity sets differ; "
            f"missing_capture={missing_capture[:4]}, missing_control={missing_control[:4]}"
        )
    if target_image_identity not in set(control_ids):
        raise ValueError(f"target image identity is absent: {target_image_identity}")

    control_rows = control.set_index(IDENTITY_FIELD, drop=False)
    capture_rows = capture.set_index(IDENTITY_FIELD, drop=False).loc[control_rows.index]
    raw_control_positions = {identity: position for position, identity in enumerate(control_ids)}
    raw_capture_positions = {identity: position for position, identity in enumerate(capture_ids)}

    fields: dict[str, Any] = {}
    for field in PARTICLE_FIELDS:
        lhs = control_rows[field].to_numpy()
        rhs = capture_rows[field].to_numpy()
        equal = np.asarray(lhs == rhs)
        mismatch_positions = np.flatnonzero(~equal)
        examples = []
        for position in mismatch_positions[:MAX_MISMATCH_EXAMPLES]:
            identity = str(control_rows.iloc[position][IDENTITY_FIELD])
            examples.append(
                {
                    "image_identity": identity,
                    "control_value": _json_scalar(lhs[position]),
                    "capture_value": _json_scalar(rhs[position]),
                    "control_raw_row": raw_control_positions[identity],
                    "capture_raw_row": raw_capture_positions[identity],
                }
            )
        numeric = np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number)
        fields[field] = {
            "exact": bool(np.all(equal)),
            "mismatch_count": int(mismatch_positions.size),
            "max_abs": float(np.max(np.abs(lhs - rhs))) if numeric else None,
            "mismatch_examples": examples,
            "mismatch_examples_truncated": bool(
                mismatch_positions.size > MAX_MISMATCH_EXAMPLES
            ),
        }

    target_control = control_rows.loc[target_image_identity]
    target_capture = capture_rows.loc[target_image_identity]
    target_control_fields = {
        field: _json_scalar(target_control[field]) for field in PARTICLE_FIELDS
    }
    target_capture_fields = {
        field: _json_scalar(target_capture[field]) for field in PARTICLE_FIELDS
    }
    return {
        "particle_count": int(len(control_rows)),
        "identity_alignment": "exact rlnImageName set; capture rows reordered to control identity order",
        "raw_row_order_exact": bool(np.array_equal(control_ids, capture_ids)),
        "fields": fields,
        "target": {
            "original_index_zero_based": target_original_index,
            "stack_index_one_based": stack_index,
            "image_identity": target_image_identity,
            "control_raw_row": raw_control_positions[target_image_identity],
            "capture_raw_row": raw_capture_positions[target_image_identity],
            "control_fields": target_control_fields,
            "capture_fields": target_capture_fields,
            "mismatch_fields": [
                field
                for field in PARTICLE_FIELDS
                if target_control_fields[field] != target_capture_fields[field]
            ],
        },
    }


def _sampling_perturbation(path: Path) -> float:
    match = re.search(
        r"_rlnSamplingPerturbInstance\s+([-+0-9.eE]+)",
        path.read_text(errors="replace"),
    )
    if match is None:
        raise ValueError(f"missing sampling perturbation in {path}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"non-finite sampling perturbation in {path}")
    return value


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float32).copy()


def _fsc_auc(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape or lhs.ndim != 3:
        raise ValueError(f"map shapes must be equal 3-D arrays, got {lhs.shape} and {rhs.shape}")
    lhs_ft = np.fft.fftn(np.asarray(lhs, dtype=np.float64))
    rhs_ft = np.fft.fftn(np.asarray(rhs, dtype=np.float64))
    coordinates = np.meshgrid(
        *(np.fft.fftfreq(size) * size for size in lhs.shape),
        indexing="ij",
    )
    shells = np.floor(np.sqrt(sum(value * value for value in coordinates))).astype(np.int32)
    curve = []
    for shell in range(1, min(lhs.shape) // 2):
        selected = shells == shell
        numerator = np.sum(lhs_ft[selected] * np.conj(rhs_ft[selected])).real
        denominator = np.sqrt(
            np.sum(np.abs(lhs_ft[selected]) ** 2)
            * np.sum(np.abs(rhs_ft[selected]) ** 2)
        )
        curve.append(float(numerator / denominator) if denominator > 0 else float("nan"))
    value = float(np.nanmean(np.asarray(curve, dtype=np.float64)))
    if not math.isfinite(value):
        raise ValueError("map FSC-AUC is non-finite")
    return value


def build_report(
    *,
    control_root: Path,
    capture_root: Path,
    target_original_index: int,
    target_image_identity: str,
    expected_particle_count: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    if expected_particle_count <= 0:
        raise ValueError("expected particle count must be positive")
    relative_star = Path("relion/run_it001_data.star")
    relative_sampling = Path("relion/run_it001_sampling.star")
    control_star = control_root / relative_star
    capture_star = capture_root / relative_star
    control_sampling = control_root / relative_sampling
    capture_sampling = capture_root / relative_sampling
    required_paths = [control_star, capture_star, control_sampling, capture_sampling]
    for half in (1, 2):
        relative_map = Path(f"relion/run_it001_half{half}_class001.mrc")
        required_paths.extend((control_root / relative_map, capture_root / relative_map))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    particle_report = compare_particle_tables(
        _particle_table(control_star),
        _particle_table(capture_star),
        target_original_index=target_original_index,
        target_image_identity=target_image_identity,
    )
    control_perturbation = _sampling_perturbation(control_sampling)
    capture_perturbation = _sampling_perturbation(capture_sampling)
    map_results = []
    for half in (1, 2):
        relative_map = Path(f"relion/run_it001_half{half}_class001.mrc")
        control_map_path = control_root / relative_map
        capture_map_path = capture_root / relative_map
        control_map = _load_mrc(control_map_path)
        capture_map = _load_mrc(capture_map_path)
        difference = capture_map.astype(np.float64) - control_map.astype(np.float64)
        map_results.append(
            {
                "half": half,
                "fsc_auc": _fsc_auc(control_map, capture_map),
                "relative_l2": float(
                    np.linalg.norm(difference.reshape(-1))
                    / np.linalg.norm(control_map.astype(np.float64).reshape(-1))
                ),
                "max_abs": float(np.max(np.abs(difference))),
            }
        )

    exact_fields = all(result["exact"] for result in particle_report["fields"].values())
    particle_count_exact = particle_report["particle_count"] == expected_particle_count
    perturbation_exact = control_perturbation == capture_perturbation
    maps_pass = all(row["fsc_auc"] >= MAP_FSC_AUC_THRESHOLD for row in map_results)
    accepted = particle_count_exact and exact_fields and perturbation_exact and maps_pass
    return {
        "schema": SCHEMA,
        "status": "pass" if accepted else "rejected",
        "gpu_uuid": gpu_uuid,
        **particle_report,
        "expected_particle_count": expected_particle_count,
        "control_perturbation": control_perturbation,
        "capture_perturbation": capture_perturbation,
        "perturbation_exact": perturbation_exact,
        "half_map_fsc_auc_threshold": MAP_FSC_AUC_THRESHOLD,
        "half_map_comparison": map_results,
        "strict_gate": {
            "particle_count_exact": particle_count_exact,
            "all_particle_fields_exact": exact_fields,
            "sampling_perturbation_exact": perturbation_exact,
            "all_half_map_fsc_auc_at_least_threshold": maps_pass,
        },
        "scorecard_change_admissible": False,
        "inputs": {
            str(path.resolve()): _sha256(path)
            for path in required_paths
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--target-original-index", required=True, type=int)
    parser.add_argument("--target-image-identity", required=True)
    parser.add_argument("--expected-particle-count", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite inertness artifact: {args.output_json}")
    report = build_report(
        control_root=args.control_root,
        capture_root=args.capture_root,
        target_original_index=args.target_original_index,
        target_image_identity=args.target_image_identity,
        expected_particle_count=args.expected_particle_count,
        gpu_uuid=args.gpu_uuid,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
