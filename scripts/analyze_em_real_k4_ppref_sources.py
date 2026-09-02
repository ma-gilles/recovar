#!/usr/bin/env python3
"""Compare native and RECOVAR-rebuilt class-resolved RELION PPref inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_relion_k4_ppref_capture import load_ppref, validate_directory

SCHEMA = "recovar.em_real_k4_ppref_sources.v1"
SOURCE_RELATIVE_L2_CEILING = 1.0e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.complex64)
    reference = np.asarray(reference, dtype=np.complex64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "PPref shapes differ")
    difference = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = float(np.linalg.norm(reference.astype(np.complex128)))
    _require(denominator > 0, "native PPref has zero norm")
    return {
        "complex_count": int(candidate.size),
        "bitwise_equal": bool(np.array_equal(candidate, reference)),
        "bitwise_equal_count": int(np.count_nonzero(candidate == reference)),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "max_abs": float(np.max(np.abs(difference))),
    }


def classify(maximum_relative_l2: float) -> str:
    _require(maximum_relative_l2 >= 0, "negative PPref error")
    if maximum_relative_l2 <= SOURCE_RELATIVE_L2_CEILING:
        return "map_to_ppref_matches_native"
    return "map_to_ppref_difference_remains"


def build_report(
    *,
    native_dir: Path,
    rebuilt_dir: Path,
    source_maps: list[Path],
    expected_iteration: int,
    expected_rank: int,
    n_classes: int,
    expected_current_size: int,
) -> dict[str, Any]:
    _require(len(source_maps) == n_classes, "source-map count differs from class count")
    native_gate = validate_directory(
        native_dir,
        expected_iteration=expected_iteration,
        expected_rank=expected_rank,
        n_classes=n_classes,
        expected_current_size=expected_current_size,
    )
    rebuilt_gate = validate_directory(
        rebuilt_dir,
        expected_iteration=expected_iteration,
        expected_rank=expected_rank,
        n_classes=n_classes,
        expected_current_size=expected_current_size,
    )
    records = []
    for model in range(n_classes):
        name = (
            f"ppref_iter{expected_iteration:03d}_rank{expected_rank:03d}_"
            f"model{model:03d}.bin"
        )
        native_path = native_dir / name
        rebuilt_path = rebuilt_dir / name
        native, native_metadata = load_ppref(native_path)
        rebuilt, rebuilt_metadata = load_ppref(rebuilt_path)
        _require(native_metadata == rebuilt_metadata, f"class {model} PPref metadata differs")
        records.append(
            {
                "model_zero_based": model,
                "comparison": _comparison(rebuilt, native),
                "native": {
                    "path": str(native_path.resolve()),
                    "sha256": _sha256(native_path),
                },
                "rebuilt": {
                    "path": str(rebuilt_path.resolve()),
                    "sha256": _sha256(rebuilt_path),
                },
            }
        )
    maximum_relative_l2 = max(
        float(record["comparison"]["relative_l2"]) for record in records
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classify(maximum_relative_l2),
        "metric_policy": "complex64 bitwise equality and direct relative L2; no correlation",
        "fixed_gates": {
            "maximum_relative_l2": SOURCE_RELATIVE_L2_CEILING,
        },
        "summary": {
            "class_count": n_classes,
            "bitwise_equal_classes": sum(
                bool(record["comparison"]["bitwise_equal"]) for record in records
            ),
            "maximum_relative_l2": maximum_relative_l2,
            "maximum_abs": max(float(record["comparison"]["max_abs"]) for record in records),
        },
        "native_validation": native_gate,
        "rebuilt_validation": rebuilt_gate,
        "source_maps": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in source_maps
        ],
        "classes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--rebuilt-dir", required=True, type=Path)
    parser.add_argument("--source-map", required=True, type=Path, action="append")
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-rank", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--expected-current-size", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = build_report(
        native_dir=args.native_dir,
        rebuilt_dir=args.rebuilt_dir,
        source_maps=args.source_map,
        expected_iteration=args.expected_iteration,
        expected_rank=args.expected_rank,
        n_classes=args.n_classes,
        expected_current_size=args.expected_current_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
