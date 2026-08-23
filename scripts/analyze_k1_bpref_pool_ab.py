#!/usr/bin/env python3
"""Compare serial and pooled K=1 BPref accumulators with native RELION."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_bpref_substitution import relion_raw_to_recovar_full
from scripts.analyze_k1_half1_raw_accumulator import _load_native_bpref


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(source: np.ndarray, target: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(source).reshape(-1)
    rhs = np.asarray(target).reshape(-1)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError(f"metric topology mismatch: {lhs.shape} versus {rhs.shape}")
    dtype = np.complex128 if np.iscomplexobj(lhs) or np.iscomplexobj(rhs) else np.float64
    lhs_metric = lhs.astype(dtype)
    rhs_metric = rhs.astype(dtype)
    if not np.all(np.isfinite(lhs_metric)) or not np.all(np.isfinite(rhs_metric)):
        raise ValueError("metric contains nonfinite values")
    residual = lhs_metric - rhs_metric
    target_norm = float(np.linalg.norm(rhs_metric))
    if target_norm == 0.0:
        raise ValueError("metric target has zero norm")
    unequal = np.flatnonzero(lhs != rhs)
    return {
        "relative_l2": float(np.linalg.norm(residual) / target_norm),
        "max_absolute": float(np.max(np.abs(residual))),
        "value_equal_fraction": float(np.mean(lhs == rhs)),
        "value_unequal_count": int(unequal.size),
        "first_unequal_index": None if unequal.size == 0 else int(unequal[0]),
    }


def _movement(control: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    control_error = float(control["relative_l2"])
    candidate_error = float(candidate["relative_l2"])
    delta = candidate_error - control_error
    return {
        "control_relative_l2": control_error,
        "candidate_relative_l2": candidate_error,
        "candidate_minus_control_relative_l2": delta,
        "candidate_over_control_relative_l2": (
            candidate_error / control_error if control_error > 0.0 else None
        ),
        "classification": "improved" if delta < 0.0 else "regressed" if delta > 0.0 else "exact_tie",
    }


def _spherical_mask(shape: tuple[int, ...], radius: int) -> np.ndarray:
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError(f"expected a cubic accumulator, got {shape}")
    axis = np.arange(shape[0], dtype=np.int64) - shape[0] // 2
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return (x * x + y * y + z * z <= int(radius) ** 2).reshape(-1)


def analyze(
    *,
    control_intermediates: Path,
    candidate_intermediates: Path,
    native_directory: Path,
    grid_size: int,
    current_size: int,
    padding_factor: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "recovar.em.k1_bpref_pool_ab.v1",
        "status": "complete",
        "metric_policy": "scale-sensitive relative-L2 on native spherical BPref support; no correlation",
        "grid_size": int(grid_size),
        "current_size": int(current_size),
        "padding_factor": int(padding_factor),
        "halves": {},
        "artifacts": {},
    }
    for half in (1, 2):
        control_paths = {
            "numerator": control_intermediates / f"it000_Ft_y_{half - 1}.npy",
            "denominator": control_intermediates / f"it000_Ft_ctf_{half - 1}.npy",
        }
        candidate_paths = {
            "numerator": candidate_intermediates / f"it000_Ft_y_{half - 1}.npy",
            "denominator": candidate_intermediates / f"it000_Ft_ctf_{half - 1}.npy",
        }
        native_data_path = native_directory / f"bpref_iter001_rank{half}_data.bin"
        native_weight_path = native_directory / f"bpref_iter001_rank{half}_weight.bin"
        _, native_data = _load_native_bpref(native_data_path, value_dtype=np.complex128)
        _, native_weight = _load_native_bpref(native_weight_path, value_dtype=np.float64)
        native_numerator, native_denominator = relion_raw_to_recovar_full(
            native_data,
            native_weight,
            grid_size=int(grid_size),
        )
        accumulator_size = int(np.asarray(native_numerator).size)
        accumulator_side = round(accumulator_size ** (1.0 / 3.0))
        if accumulator_side**3 != accumulator_size:
            raise ValueError(
                f"native accumulator is not cubic: {accumulator_size} elements"
            )
        accumulator_shape = (accumulator_side,) * 3
        mask = _spherical_mask(
            accumulator_shape,
            radius=int(current_size) * int(padding_factor) // 2,
        )
        native_values = {
            "numerator": -np.asarray(native_numerator).reshape(-1)[mask],
            "denominator": np.asarray(native_denominator).reshape(-1)[mask],
        }
        half_report: dict[str, object] = {
            "spherical_support_count": int(np.count_nonzero(mask)),
            "fields": {},
        }
        for field in ("numerator", "denominator"):
            control = np.load(control_paths[field], mmap_mode="r").reshape(-1)[mask]
            candidate = np.load(candidate_paths[field], mmap_mode="r").reshape(-1)[mask]
            native = native_values[field]
            control_native = _metric(control, native)
            candidate_native = _metric(candidate, native)
            half_report["fields"][field] = {
                "control_vs_native": control_native,
                "candidate_vs_native": candidate_native,
                "candidate_vs_control": _metric(candidate, control),
                "movement": _movement(control_native, candidate_native),
            }
            for path in (control_paths[field], candidate_paths[field]):
                result["artifacts"][str(path.resolve())] = _sha256(path)
        for path in (native_data_path, native_weight_path):
            result["artifacts"][str(path.resolve())] = _sha256(path)
        result["halves"][str(half)] = half_report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-intermediates", type=Path, required=True)
    parser.add_argument("--candidate-intermediates", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--padding-factor", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        control_intermediates=args.control_intermediates,
        candidate_intermediates=args.candidate_intermediates,
        native_directory=args.native_directory,
        grid_size=args.grid_size,
        current_size=args.current_size,
        padding_factor=args.padding_factor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
