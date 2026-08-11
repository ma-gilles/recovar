#!/usr/bin/env python3
"""Compare one RECOVAR post-M-step norm state with serialized RELION state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import starfile


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_float32_ulp_distance(left: float, right: float) -> int:
    values = np.asarray([left, right], dtype=np.float32)
    _require(np.all(np.isfinite(values)) and np.all(values > 0), "ULP inputs must be finite and positive")
    bits = values.view(np.uint32)
    return abs(int(bits[1]) - int(bits[0]))


def _complex_scale(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference128 = np.asarray(reference, dtype=np.complex128).reshape(-1)
    candidate128 = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    _require(reference128.shape == candidate128.shape, "operand shapes changed")
    denominator = float(np.vdot(candidate128, candidate128).real)
    _require(denominator > 0.0, "candidate operand has zero norm")
    scale = float(np.vdot(candidate128, reference128).real / denominator)
    before = float(np.vdot(candidate128 - reference128, candidate128 - reference128).real)
    after_delta = scale * candidate128 - reference128
    after = float(np.vdot(after_delta, after_delta).real)
    return {
        "native_over_recovar_optimal_real_scale": scale,
        "scale_delta": scale - 1.0,
        "scale_error_energy_removal_fraction": 0.0 if before == 0.0 else 1.0 - after / before,
    }


def _particle_table(path: Path):
    document = starfile.read(path)
    return document["particles"] if isinstance(document, dict) else document


def _model_tables(path: Path) -> tuple[dict[str, Any], Any]:
    document = starfile.read(path)
    _require(isinstance(document, dict), "RELION model STAR must contain named tables")
    return document["model_general"], document["model_groups"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovar-iteration-npz", type=Path, required=True)
    parser.add_argument("--recovar-results-npz", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--relion-model-star", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--operand-dump", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    stack_index = args.source_index + 1
    with np.load(args.recovar_results_npz, allow_pickle=False) as results:
        half_indices = [
            np.asarray(results["half1_indices"], dtype=np.int64),
            np.asarray(results["half2_indices"], dtype=np.int64),
        ]
    matches = [np.flatnonzero(indices == args.source_index) for indices in half_indices]
    matched_halves = [half for half, rows in enumerate(matches) if rows.size]
    _require(len(matched_halves) == 1, "source particle must occur in exactly one RECOVAR half")
    half = matched_halves[0]
    physical_row = int(matches[half][0])

    with np.load(args.recovar_iteration_npz, allow_pickle=False) as iteration:
        image_corrections = np.asarray(iteration[f"half{half + 1}_image_corrections"], dtype=np.float32)
        scale_corrections = np.asarray(iteration[f"half{half + 1}_scale_corrections"], dtype=np.float32)
        norm_corrections = np.asarray(iteration[f"half{half + 1}_norm_corrections"], dtype=np.float32)
        average_norm = float(iteration[f"half{half + 1}_avg_norm_correction"])
        wsum_norm_correction = np.asarray(
            iteration[f"half{half + 1}_wsum_norm_correction"],
            dtype=np.float64,
        )
        retained_sum_weight = float(iteration[f"half{half + 1}_sumw"])
        recovar_relion_iteration = int(iteration["relion_iteration"])
    _require(
        image_corrections.shape == scale_corrections.shape == norm_corrections.shape == half_indices[half].shape,
        "RECOVAR correction-array shape changed",
    )
    recovar_image_correction = np.float32(image_corrections[physical_row])
    recovar_scale_correction = np.float32(scale_corrections[physical_row])
    recovar_norm_correction = np.float32(norm_corrections[physical_row])
    recovar_normalization_factor = np.divide(
        recovar_image_correction,
        recovar_scale_correction,
        dtype=np.float32,
    )

    particles = _particle_table(args.relion_data_star)
    identities = particles["rlnImageName"].astype(str)
    relion_rows = np.flatnonzero(identities.str.startswith(f"{stack_index}@").to_numpy())
    _require(relion_rows.size == 1, "RELION particle identity is not unique")
    relion_particle = particles.iloc[int(relion_rows[0])]
    relion_half = int(relion_particle["rlnRandomSubset"])
    _require(relion_half == half + 1, "RELION and RECOVAR half identities differ")

    model_general, model_groups = _model_tables(args.relion_model_star)
    image_size = int(model_general["rlnOriginalImageSize"])
    n2 = float(image_size**2)
    n4 = n2 * n2
    group_number = int(relion_particle["rlnGroupNumber"])
    group_rows = np.flatnonzero(
        np.asarray(model_groups["rlnGroupNumber"], dtype=np.int64) == group_number
    )
    _require(group_rows.size == 1, "RELION particle group is not unique")
    relion_scale = np.float32(model_groups.iloc[int(group_rows[0])]["rlnGroupScaleCorrection"])
    relion_average_norm = float(model_general["rlnNormCorrectionAverage"])
    relion_norm = float(relion_particle["rlnNormCorrection"])
    relion_serialized_normalization = np.float32(relion_average_norm / relion_norm)
    relion_serialized_image_correction = np.multiply(
        relion_serialized_normalization,
        relion_scale,
        dtype=np.float32,
    )
    recovar_norm_correction_relion_units = float(recovar_norm_correction) / n2
    recovar_average_norm_relion_units = average_norm / n2
    recovar_norm_power_relion_units = float(wsum_norm_correction[physical_row]) / n4
    relion_serialized_norm_power = 0.5 * relion_norm * relion_norm

    operand_scale = None
    if args.operand_dump is not None:
        with np.load(args.operand_dump, allow_pickle=False) as operands:
            operand_scale = _complex_scale(
                operands["native_processed_image"],
                operands["recovar_processed_image"],
            )

    state_ratio = float(
        np.float64(relion_serialized_normalization)
        / np.float64(recovar_normalization_factor)
    )
    report = {
        "schema": "recovar.em.k1_norm_state_boundary.v1",
        "status": "complete",
        "source_index_zero_based": args.source_index,
        "stack_index_one_based": stack_index,
        "half_one_based": half + 1,
        "recovar_physical_half_row": physical_row,
        "recovar_relion_iteration": recovar_relion_iteration,
        "recovar": {
            "image_correction_float32": float(recovar_image_correction),
            "scale_correction_float32": float(recovar_scale_correction),
            "normalization_factor_float32": float(recovar_normalization_factor),
            "norm_correction_float32": float(recovar_norm_correction),
            "average_norm_correction_float64": average_norm,
            "norm_correction_relion_units": recovar_norm_correction_relion_units,
            "average_norm_correction_relion_units": recovar_average_norm_relion_units,
            "wsum_norm_correction_float64": float(wsum_norm_correction[physical_row]),
            "wsum_norm_correction_relion_units": recovar_norm_power_relion_units,
            "retained_sum_weight_float64": retained_sum_weight,
        },
        "relion_serialized": {
            "image_correction_float32": float(relion_serialized_image_correction),
            "scale_correction_float32": float(relion_scale),
            "normalization_factor_float32": float(relion_serialized_normalization),
            "norm_correction_decimal": relion_norm,
            "average_norm_correction_decimal": relion_average_norm,
            "norm_power_inferred_from_serialized_norm": relion_serialized_norm_power,
        },
        "comparison": {
            "serialized_relion_over_recovar_normalization": state_ratio,
            "normalization_float32_ulp_distance": _positive_float32_ulp_distance(
                recovar_normalization_factor,
                relion_serialized_normalization,
            ),
            "image_correction_float32_ulp_distance": _positive_float32_ulp_distance(
                recovar_image_correction,
                relion_serialized_image_correction,
            ),
            "norm_correction_relion_units_delta": (
                recovar_norm_correction_relion_units - relion_norm
            ),
            "average_norm_relion_units_delta": (
                recovar_average_norm_relion_units - relion_average_norm
            ),
            "norm_power_relion_units_delta": (
                recovar_norm_power_relion_units - relion_serialized_norm_power
            ),
            "operand_processed_fourier_scale": operand_scale,
            "serialized_state_ratio_minus_operand_optimal_scale": (
                None
                if operand_scale is None
                else state_ratio - operand_scale["native_over_recovar_optimal_real_scale"]
            ),
        },
        "inputs": {
            "recovar_iteration_npz": str(args.recovar_iteration_npz.resolve()),
            "recovar_iteration_sha256": _sha256(args.recovar_iteration_npz),
            "recovar_results_npz": str(args.recovar_results_npz.resolve()),
            "recovar_results_sha256": _sha256(args.recovar_results_npz),
            "relion_data_star": str(args.relion_data_star.resolve()),
            "relion_data_sha256": _sha256(args.relion_data_star),
            "relion_model_star": str(args.relion_model_star.resolve()),
            "relion_model_sha256": _sha256(args.relion_model_star),
            "operand_dump": None if args.operand_dump is None else str(args.operand_dump.resolve()),
            "operand_dump_sha256": None if args.operand_dump is None else _sha256(args.operand_dump),
        },
        "interpretation_limit": (
            "RELION norm and average-norm fields are decimal STAR serializations; "
            "the comparison does not claim access to RELION's private in-memory norm state."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
