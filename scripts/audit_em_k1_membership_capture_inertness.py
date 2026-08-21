#!/usr/bin/env python3
"""Gate K=1 membership diagnostics on shellwise map inertness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

SCHEMA = "em-k1-membership-capture-inertness-v1"
DEFAULT_FSC_AUC_THRESHOLD = 0.999999


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_map(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with mrcfile.open(path, permissive=False) as handle:
        volume = np.asarray(handle.data, dtype=np.float32).copy()
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"expected a cubic 3-D map, got {volume.shape}: {path}")
    if not np.all(np.isfinite(volume)):
        raise ValueError(f"map contains non-finite values: {path}")
    return volume


def _shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.shape != rhs.shape:
        raise ValueError(f"map shape mismatch: {lhs.shape} != {rhs.shape}")
    size = lhs.shape[0]
    lhs_ft = np.fft.rfftn(np.asarray(lhs, dtype=np.float64))
    rhs_ft = np.fft.rfftn(np.asarray(rhs, dtype=np.float64))
    full = np.fft.fftfreq(size) * size
    half = np.fft.rfftfreq(size) * size
    yy, xx = np.meshgrid(full, half, indexing="ij")
    packed_weights = np.full(half.shape, 2.0, dtype=np.float64)
    packed_weights[0] = 1.0
    if size % 2 == 0:
        packed_weights[-1] = 1.0
    _, shell_weights = np.meshgrid(full, packed_weights, indexing="ij")
    numerator = np.zeros(size // 2 + 1, dtype=np.float64)
    lhs_power = np.zeros_like(numerator)
    rhs_power = np.zeros_like(numerator)
    for z_index, z_frequency in enumerate(full):
        shell = np.rint(
            np.sqrt(z_frequency**2 + yy**2 + xx**2)
        ).astype(np.int32)
        keep = shell <= size // 2
        indices = shell[keep].reshape(-1)
        weights = shell_weights[keep].reshape(-1)
        cross = (lhs_ft[z_index] * np.conj(rhs_ft[z_index]))[keep].reshape(-1)
        numerator += np.bincount(
            indices,
            weights=cross.real * weights,
            minlength=size // 2 + 1,
        )
        lhs_power += np.bincount(
            indices,
            weights=(np.abs(lhs_ft[z_index][keep]) ** 2).reshape(-1) * weights,
            minlength=size // 2 + 1,
        )
        rhs_power += np.bincount(
            indices,
            weights=(np.abs(rhs_ft[z_index][keep]) ** 2).reshape(-1) * weights,
            minlength=size // 2 + 1,
        )
    denominator = np.sqrt(lhs_power * rhs_power)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        ),
        -1.0,
        1.0,
    )


def _metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    fsc = _shell_fsc(lhs, rhs)
    finite = fsc[1:][np.isfinite(fsc[1:])]
    if finite.size != fsc.size - 1:
        raise ValueError("map FSC has a non-finite non-DC shell")
    denominator = float(np.linalg.norm(lhs.astype(np.float64).reshape(-1)))
    if denominator == 0.0:
        raise ValueError("control map has zero norm")
    delta = rhs.astype(np.float64) - lhs.astype(np.float64)
    auc = float(np.mean(finite))
    if not math.isfinite(auc):
        raise ValueError("map FSC-AUC is non-finite")
    return {
        "shape": list(lhs.shape),
        "fsc": fsc.tolist(),
        "fsc_auc_non_dc": auc,
        "fsc_min_non_dc": float(np.min(finite)),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _map_paths(
    root: Path,
    *,
    engine: str,
    half: int,
    relion_iteration: int,
) -> Path:
    if engine == "relion":
        return (
            root
            / "relion"
            / f"run_it{relion_iteration:03d}_half{half}_class001.mrc"
        )
    if engine == "recovar":
        return root / "recovar" / f"final_half{half}.mrc"
    raise ValueError(f"unsupported engine: {engine}")


def build_report(
    *,
    control_root: Path,
    capture_root: Path,
    relion_iteration: int,
    fsc_auc_threshold: float,
) -> dict[str, Any]:
    if relion_iteration < 1:
        raise ValueError("RELION iteration must be positive")
    if not 0.0 < fsc_auc_threshold <= 1.0:
        raise ValueError("FSC-AUC threshold must be in (0, 1]")

    comparisons: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    for engine in ("relion", "recovar"):
        loaded: dict[str, list[np.ndarray]] = {"control": [], "capture": []}
        for half in (1, 2):
            paths = {
                arm: _map_paths(
                    root,
                    engine=engine,
                    half=half,
                    relion_iteration=relion_iteration,
                )
                for arm, root in (
                    ("control", control_root),
                    ("capture", capture_root),
                )
            }
            volumes = {arm: _load_map(path) for arm, path in paths.items()}
            for arm, path in paths.items():
                artifact_hashes[str(path.resolve())] = _sha256(path)
                loaded[arm].append(volumes[arm])
            current = _metrics(volumes["control"], volumes["capture"])
            current["passed"] = (
                current["fsc_auc_non_dc"] >= fsc_auc_threshold
            )
            comparisons[f"{engine}_half{half}"] = current

        control_merged = 0.5 * (loaded["control"][0] + loaded["control"][1])
        capture_merged = 0.5 * (loaded["capture"][0] + loaded["capture"][1])
        merged = _metrics(control_merged, capture_merged)
        merged["passed"] = merged["fsc_auc_non_dc"] >= fsc_auc_threshold
        comparisons[f"{engine}_merged"] = merged

    qualified = all(bool(row["passed"]) for row in comparisons.values())
    return {
        "schema": SCHEMA,
        "status": "pass" if qualified else "rejected",
        "metric_policy": (
            "shellwise FSC/FSC-AUC only for acceptance; "
            "relative L2 and max-absolute error are secondary; no correlation"
        ),
        "control_root": str(control_root.resolve()),
        "capture_root": str(capture_root.resolve()),
        "relion_iteration": relion_iteration,
        "fsc_auc_non_dc_threshold": fsc_auc_threshold,
        "comparisons": comparisons,
        "strict_gate": {
            "comparison_count": len(comparisons),
            "expected_comparison_count": 6,
            "all_fsc_auc_at_least_threshold": qualified,
        },
        "capture_inertness_qualified": qualified,
        "artifact_sha256": artifact_hashes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--relion-iteration", type=int, default=2)
    parser.add_argument(
        "--fsc-auc-threshold",
        type=float,
        default=DEFAULT_FSC_AUC_THRESHOLD,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(
            f"refusing to overwrite inertness report: {args.output_json}"
        )
    report = build_report(
        control_root=args.control_root,
        capture_root=args.capture_root,
        relion_iteration=args.relion_iteration,
        fsc_auc_threshold=args.fsc_auc_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
