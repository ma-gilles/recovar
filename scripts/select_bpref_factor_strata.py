#!/usr/bin/env python3
"""Select deterministic error-stratified particles for BPref factor capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    left = np.asarray(lhs, dtype=np.complex128)
    right = np.asarray(rhs, dtype=np.complex128)
    _require(left.ndim == 2 and left.shape == right.shape, "operand arrays must be matching matrices")
    numerator = np.linalg.norm(right - left, axis=1)
    denominator = np.maximum(np.linalg.norm(left, axis=1), np.finfo(np.float64).tiny)
    return numerator / denominator


def select_strata(
    arrays: dict[str, np.ndarray],
    *,
    excluded_stack: int,
    stratum_count: int = 8,
    per_stratum: int = 4,
) -> dict[str, object]:
    """Return a stable 8x4 selection after excluding one tracked outlier."""

    required = {
        "stack_indices_1based",
        "recovar_data",
        "relion_data_recovar_units",
        "recovar_device_support_mask",
        "recovar_global_rotation_indices",
        "relion_orientation_class_keys",
        "relion_oversampled_rotations",
    }
    missing = required.difference(arrays)
    _require(not missing, "missing aligned arrays: " + ", ".join(sorted(missing)))
    stacks = np.asarray(arrays["stack_indices_1based"], dtype=np.int64)
    _require(stacks.ndim == 1 and np.unique(stacks).size == stacks.size, "stack identities must be unique")
    _require(np.count_nonzero(stacks == excluded_stack) == 1, "excluded stack must occur exactly once")

    support = np.asarray(arrays["recovar_device_support_mask"], dtype=bool)
    recovar_data = np.where(support, np.asarray(arrays["recovar_data"]), 0)
    relion_data = np.asarray(arrays["relion_data_recovar_units"])
    _require(recovar_data.shape == relion_data.shape == support.shape, "operand/support shapes differ")
    errors = _relative_l2(recovar_data, relion_data)
    _require(np.all(np.isfinite(errors)), "per-particle errors must be finite")

    cohort = np.flatnonzero(stacks != excluded_stack)
    _require(cohort.size >= stratum_count * per_stratum, "cohort is too small for requested strata")
    order = cohort[np.lexsort((stacks[cohort], errors[cohort]))]
    strata = np.array_split(order, stratum_count)
    selected: list[dict[str, object]] = []
    for stratum_index, indices in enumerate(strata):
        _require(indices.size >= per_stratum, f"stratum {stratum_index} is too small")
        # Midpoints of four equal subintervals: stable and avoids over-sampling
        # exact bin boundaries when the cohort size is not divisible by eight.
        positions = np.floor((2 * np.arange(per_stratum) + 1) * indices.size / (2 * per_stratum)).astype(int)
        positions = np.minimum(positions, indices.size - 1)
        _require(np.unique(positions).size == per_stratum, f"stratum {stratum_index} selections collide")
        for within_index, position in enumerate(positions):
            row = int(indices[position])
            selected.append(
                {
                    "stack_index_1based": int(stacks[row]),
                    "source_row": row,
                    "cohort_error_rank_zero_based": int(np.flatnonzero(order == row)[0]),
                    "stratum_zero_based": stratum_index,
                    "within_stratum_zero_based": within_index,
                    "data_relative_l2": float(errors[row]),
                    "support_pixel_count": int(np.count_nonzero(support[row])),
                    "recovar_global_rotation_index": int(
                        np.asarray(arrays["recovar_global_rotation_indices"])[row]
                    ),
                    "relion_orientation_class_key": int(
                        np.asarray(arrays["relion_orientation_class_keys"])[row]
                    ),
                    "relion_oversampled_rotation": int(
                        np.asarray(arrays["relion_oversampled_rotations"])[row]
                    ),
                }
            )
    selected_stacks = [int(record["stack_index_1based"]) for record in selected]
    _require(len(selected) == stratum_count * per_stratum, "selection count changed")
    _require(len(set(selected_stacks)) == len(selected_stacks), "selected stack identities are duplicated")
    return {
        "schema": "bpref-factor-stratification-v1",
        "metric_policy": "exact/array metrics for intermediate operands; no correlation",
        "selection_policy": "eight equal-count stable error-order strata; four subinterval midpoints per stratum",
        "cohort": {
            "input_particle_count": int(stacks.size),
            "systematic_particle_count": int(cohort.size),
            "excluded_stack_index_1based": int(excluded_stack),
            "excluded_data_relative_l2": float(errors[stacks == excluded_stack][0]),
            "stratum_count": int(stratum_count),
            "particles_per_stratum": int(per_stratum),
            "selected_particle_count": len(selected),
        },
        "selected": selected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned_operands", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-stack-list", required=True, type=Path)
    parser.add_argument("--excluded-stack", type=int, default=111721)
    parser.add_argument("--expected-sha256", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for output in (args.output_json, args.output_stack_list):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite selection artifact: {output}")
    actual_sha256 = _sha256(args.aligned_operands)
    _require(actual_sha256 == args.expected_sha256, "aligned-operands SHA-256 does not match")
    with np.load(args.aligned_operands, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    report = select_strata(arrays, excluded_stack=args.excluded_stack)
    report["source"] = {
        "aligned_operands": str(args.aligned_operands.resolve()),
        "sha256": actual_sha256,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_stack_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_stack_list.write_text(
        ",".join(str(record["stack_index_1based"]) for record in report["selected"]) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
