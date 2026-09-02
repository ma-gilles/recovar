#!/usr/bin/env python3
"""Audit repeatability of two bounded K-class replay outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc

SCHEMA = "recovar.em_k4_replay_repeatability.v1"
MAP_PREFIXES = ("recovar_class", "recovar_best_variant_class")
CONFIGURATION_KEYS = (
    "n_classes",
    "n_images",
    "current_size",
    "healpix_order",
    "n_rotations",
    "n_translations",
    "random_perturbation",
    "coarse_rotation_source",
    "image_fourier_backend",
)


class AuditError(RuntimeError):
    """Raised when a replay artifact is absent or structurally incompatible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_summary(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "output" / "summary.json"
    _require(path.is_file(), f"missing replay summary: {path}")
    return path, json.loads(path.read_text())


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left.ravel()))
    numerator = float(np.linalg.norm((left - right).ravel()))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _map_metrics(left_path: Path, right_path: Path) -> dict[str, Any]:
    _require(left_path.is_file(), f"missing baseline map: {left_path}")
    _require(right_path.is_file(), f"missing candidate map: {right_path}")
    with mrcfile.open(left_path, permissive=True) as stream:
        left = np.asarray(stream.data, dtype=np.float64).copy()
    with mrcfile.open(right_path, permissive=True) as stream:
        right = np.asarray(stream.data, dtype=np.float64).copy()
    _require(left.shape == right.shape, f"map shape differs: {left_path.name}")
    difference = np.abs(left - right)
    fsc_auc = float(
        normalized_fsc_auc(np.asarray(shell_fsc(left, right), dtype=np.float64))
    )
    relative_l2 = _relative_l2(left, right)
    _require(np.isfinite(fsc_auc), f"nonfinite FSC-AUC: {left_path.name}")
    _require(np.isfinite(relative_l2), f"nonfinite relative L2: {left_path.name}")
    return {
        "map": left_path.name,
        "shape": list(left.shape),
        "bitwise_equal": bool(np.array_equal(left, right)),
        "fsc_auc": fsc_auc,
        "relative_l2": relative_l2,
        "max_abs": float(np.max(difference, initial=0.0)),
        "baseline_sha256": _sha256(left_path),
        "candidate_sha256": _sha256(right_path),
    }


def build_report(
    baseline_root: Path,
    candidate_root: Path,
    *,
    expected_k: int = 4,
    minimum_fsc_auc: float = 0.999999,
    maximum_relative_l2: float = 1e-5,
) -> dict[str, Any]:
    """Compare deterministic replay state exactly and maps at an atomic floor."""

    baseline_root = baseline_root.resolve()
    candidate_root = candidate_root.resolve()
    _require(baseline_root != candidate_root, "baseline and candidate roots must differ")
    _require(expected_k >= 1, "expected K must be positive")

    baseline_summary_path, baseline_summary = _load_summary(baseline_root)
    candidate_summary_path, candidate_summary = _load_summary(candidate_root)
    configuration = {
        key: {
            "baseline": baseline_summary.get(key),
            "candidate": candidate_summary.get(key),
            "exact": baseline_summary.get(key) == candidate_summary.get(key),
        }
        for key in CONFIGURATION_KEYS
    }
    configuration_exact = all(item["exact"] for item in configuration.values())
    _require(
        baseline_summary.get("n_classes") == expected_k
        and candidate_summary.get("n_classes") == expected_k,
        "summary K differs from the expected value",
    )

    baseline_arrays_path = baseline_root / "output" / "k_class_parity_arrays.npz"
    candidate_arrays_path = candidate_root / "output" / "k_class_parity_arrays.npz"
    _require(baseline_arrays_path.is_file(), f"missing parity arrays: {baseline_arrays_path}")
    _require(candidate_arrays_path.is_file(), f"missing parity arrays: {candidate_arrays_path}")
    with np.load(baseline_arrays_path) as baseline_arrays, np.load(
        candidate_arrays_path
    ) as candidate_arrays:
        _require(
            baseline_arrays.files == candidate_arrays.files,
            "parity-array keys or order differ",
        )
        parity_array_results = {
            name: bool(np.array_equal(baseline_arrays[name], candidate_arrays[name]))
            for name in baseline_arrays.files
        }
    parity_arrays_bitwise = all(parity_array_results.values())

    maps = [
        _map_metrics(
            baseline_root / "output" / f"{prefix}{class_one_based:03d}.mrc",
            candidate_root / "output" / f"{prefix}{class_one_based:03d}.mrc",
        )
        for prefix in MAP_PREFIXES
        for class_one_based in range(1, expected_k + 1)
    ]
    map_fsc_auc_minimum = min(item["fsc_auc"] for item in maps)
    map_relative_l2_maximum = max(item["relative_l2"] for item in maps)
    passed = bool(
        configuration_exact
        and parity_arrays_bitwise
        and map_fsc_auc_minimum >= minimum_fsc_auc
        and map_relative_l2_maximum <= maximum_relative_l2
    )
    return {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "classification": (
            "replay_state_exact_and_maps_within_atomic_floor"
            if passed
            else "replay_repeatability_gate_failed"
        ),
        "metric_policy": (
            "configuration and parity arrays must be exact; GPU atomic reconstruction "
            "maps use predeclared normalized FSC-AUC and relative-L2 floors"
        ),
        "roots": {
            "baseline": str(baseline_root),
            "candidate": str(candidate_root),
        },
        "artifacts": {
            "baseline_summary": {
                "path": str(baseline_summary_path),
                "sha256": _sha256(baseline_summary_path),
            },
            "candidate_summary": {
                "path": str(candidate_summary_path),
                "sha256": _sha256(candidate_summary_path),
            },
            "baseline_parity_arrays": {
                "path": str(baseline_arrays_path),
                "sha256": _sha256(baseline_arrays_path),
            },
            "candidate_parity_arrays": {
                "path": str(candidate_arrays_path),
                "sha256": _sha256(candidate_arrays_path),
            },
        },
        "gates": {
            "configuration_exact": True,
            "parity_arrays_bitwise": True,
            "map_fsc_auc_minimum": minimum_fsc_auc,
            "map_relative_l2_maximum": maximum_relative_l2,
        },
        "configuration": configuration,
        "configuration_exact": configuration_exact,
        "parity_array_results": parity_array_results,
        "parity_arrays_bitwise": parity_arrays_bitwise,
        "maps": maps,
        "summary": {
            "map_count": len(maps),
            "map_bitwise_count": sum(item["bitwise_equal"] for item in maps),
            "map_fsc_auc_minimum": map_fsc_auc_minimum,
            "map_relative_l2_maximum": map_relative_l2_maximum,
            "map_max_abs_maximum": max(item["max_abs"] for item in maps),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument("--minimum-fsc-auc", type=float, default=0.999999)
    parser.add_argument("--maximum-relative-l2", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        args.baseline_root,
        args.candidate_root,
        expected_k=args.expected_k,
        minimum_fsc_auc=args.minimum_fsc_auc,
        maximum_relative_l2=args.maximum_relative_l2,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
