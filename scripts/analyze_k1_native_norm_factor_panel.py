#!/usr/bin/env python3
"""Compare native RELION and RECOVAR iteration-2 normalization factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import starfile

if __package__:
    from .validate_relion_preprocess_capture import load_artifact
else:
    from validate_relion_preprocess_capture import load_artifact  # type: ignore[no-redef]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float32_ulp_distance(lhs: float, rhs: float) -> int:
    values = np.asarray([lhs, rhs], dtype=np.float32)
    _require(np.all(np.isfinite(values)) and np.all(values > 0), "ULP inputs must be finite and positive")
    bits = values.view(np.uint32).astype(np.int64)
    return int(abs(bits[0] - bits[1]))


def _particle_table(path: Path):
    document = starfile.read(path)
    return document["particles"] if isinstance(document, dict) else document


def _stack_index(identity: str) -> int:
    try:
        return int(str(identity).split("@", maxsplit=1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid RELION image identity {identity!r}") from error


def _half_arrays(
    iteration: np.lib.npyio.NpzFile,
    results: np.lib.npyio.NpzFile,
    half: int,
) -> tuple[np.ndarray, ...]:
    prefix = f"half{half}"
    source_indices = np.asarray(results[f"{prefix}_indices"], dtype=np.int64)
    image_corrections = np.asarray(iteration[f"{prefix}_image_corrections"], dtype=np.float32)
    scale_corrections = np.asarray(iteration[f"{prefix}_scale_corrections"], dtype=np.float32)
    norm_corrections = np.asarray(iteration[f"{prefix}_norm_corrections"], dtype=np.float32)
    _require(
        source_indices.shape
        == image_corrections.shape
        == scale_corrections.shape
        == norm_corrections.shape,
        f"half {half}: particle-aligned state shapes changed",
    )
    return source_indices, image_corrections, scale_corrections, norm_corrections


def analyze(
    *,
    capture_dir: Path,
    recovar_iteration: Path,
    recovar_results: Path,
    relion_data_star: Path,
) -> dict[str, object]:
    table = _particle_table(relion_data_star)
    captures = tuple(load_artifact(path) for path in sorted(capture_dir.glob("*.preprocess-v1.bin")))
    _require(captures, "native preprocess capture directory is empty")

    with (
        np.load(recovar_iteration, allow_pickle=False) as iteration,
        np.load(recovar_results, allow_pickle=False) as results,
    ):
        half_state = {half: _half_arrays(iteration, results, half) for half in (1, 2)}
        average_norm = {
            half: float(np.asarray(iteration[f"half{half}_avg_norm_correction"]).item())
            for half in (1, 2)
        }

    records: list[dict[str, object]] = []
    for capture in captures:
        _require(0 <= capture.part_id < len(table), f"part ID {capture.part_id} is outside the data STAR")
        row = table.iloc[capture.part_id]
        stack = _stack_index(row["rlnImageName"])
        _require(stack == capture.stack_index, f"part {capture.part_id}: stack identity changed")
        half = int(row["rlnRandomSubset"])
        _require(half in (1, 2), f"part {capture.part_id}: invalid random subset {half}")
        source_index = stack - 1
        source_indices, image_corrections, scale_corrections, norm_corrections = half_state[half]
        physical_rows = np.flatnonzero(source_indices == source_index)
        _require(physical_rows.size == 1, f"source index {source_index}: RECOVAR physical row is not unique")
        physical_row = int(physical_rows[0])
        scale = np.float32(scale_corrections[physical_row])
        _require(np.isfinite(scale) and scale > 0, f"source index {source_index}: invalid RECOVAR scale")
        recovar_factor = np.float32(image_corrections[physical_row] / scale)
        native_factor = np.float32(capture.norm_correction)
        ratio = float(np.float64(native_factor) / np.float64(recovar_factor))
        records.append(
            {
                "part_id": int(capture.part_id),
                "stack_index_one_based": stack,
                "source_index_zero_based": source_index,
                "half_one_based": half,
                "recovar_physical_row": physical_row,
                "native_factor_float32": float(native_factor),
                "native_factor_bits": f"0x{native_factor.view(np.uint32).item():08x}",
                "recovar_factor_float32": float(recovar_factor),
                "recovar_factor_bits": f"0x{recovar_factor.view(np.uint32).item():08x}",
                "native_minus_recovar": float(np.float64(native_factor) - np.float64(recovar_factor)),
                "native_over_recovar": ratio,
                "float32_ulp_distance": _float32_ulp_distance(native_factor, recovar_factor),
                "recovar_norm_correction_float32": float(norm_corrections[physical_row]),
                "recovar_average_norm_correction_float64": average_norm[half],
                "capture_path": str(capture.path.resolve()),
                "capture_sha256": capture.sha256,
            }
        )

    summaries: dict[str, object] = {}
    for half in (1, 2):
        selected = [record for record in records if record["half_one_based"] == half]
        _require(selected, f"capture panel contains no half-{half} particles")
        ratios = np.asarray([record["native_over_recovar"] for record in selected], dtype=np.float64)
        ulps = np.asarray([record["float32_ulp_distance"] for record in selected], dtype=np.int64)
        summaries[f"half{half}"] = {
            "particle_count": len(selected),
            "native_over_recovar_min": float(np.min(ratios)),
            "native_over_recovar_max": float(np.max(ratios)),
            "native_over_recovar_spread": float(np.ptp(ratios)),
            "native_over_recovar_median": float(np.median(ratios)),
            "ulp_distance_min": int(np.min(ulps)),
            "ulp_distance_max": int(np.max(ulps)),
            "ulp_distance_counts": {
                str(int(value)): int(np.count_nonzero(ulps == value)) for value in np.unique(ulps)
            },
        }

    return {
        "schema": "recovar.em.k1_native_norm_factor_panel.v1",
        "status": "complete",
        "capture_dir": str(capture_dir.resolve()),
        "recovar_iteration": str(recovar_iteration.resolve()),
        "recovar_results": str(recovar_results.resolve()),
        "relion_data_star": str(relion_data_star.resolve()),
        "particle_count": len(records),
        "half_summaries": summaries,
        "particles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--recovar-iteration", type=Path, required=True)
    parser.add_argument("--recovar-results", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        capture_dir=args.capture_dir,
        recovar_iteration=args.recovar_iteration,
        recovar_results=args.recovar_results,
        relion_data_star=args.relion_data_star,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
