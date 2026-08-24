#!/usr/bin/env python3
"""Compare one RECOVAR post-M-step norm state with serialized RELION state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import starfile

if __package__:
    from .validate_relion_preprocess_capture import load_artifact as load_preprocess_capture
else:
    from validate_relion_preprocess_capture import (  # type: ignore[no-redef]
        load_artifact as load_preprocess_capture,
    )


_NATIVE_EXPECTATION_PATTERN = re.compile(
    r"RELION_P1_NORMALIZATION_OPERANDS_V1 part_id=(?P<part_id>\d+) "
    r"avg_norm=(?P<avg_norm>\S+) particle_norm=(?P<particle_norm>\S+) "
    r"quotient=(?P<quotient>\S+) quotient_f32_bits=(?P<quotient_bits>[0-9a-fA-F]{8})"
)
_NATIVE_UPDATE_PATTERN = re.compile(
    r"RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=(?P<iteration>\d+) "
    r"part_id=(?P<part_id>\d+) previous_norm=(?P<previous_norm>\S+) "
    r"previous_avg=(?P<previous_avg>\S+) old_norm_over_avg=(?P<old_norm_over_avg>\S+) "
    r"wsum_norm=(?P<wsum_norm>\S+) sqrt_2_wsum=(?P<sqrt_2_wsum>\S+) "
    r"new_norm=(?P<new_norm>\S+)"
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


def _parse_native_norm_operands(path: Path, *, part_id: int) -> dict[str, object]:
    expectation_records: list[dict[str, object]] = []
    update_records: list[dict[str, object]] = []
    for line in path.read_text(errors="replace").splitlines():
        expectation = _NATIVE_EXPECTATION_PATTERN.search(line)
        if expectation is not None and int(expectation["part_id"]) == part_id:
            expectation_records.append(
                {
                    "part_id": part_id,
                    "avg_norm_float64": float.fromhex(expectation["avg_norm"]),
                    "particle_norm_float64": float.fromhex(expectation["particle_norm"]),
                    "quotient_float64": float.fromhex(expectation["quotient"]),
                    "quotient_float32_bits": f"0x{expectation['quotient_bits'].lower()}",
                }
            )
        update = _NATIVE_UPDATE_PATTERN.search(line)
        if update is not None and int(update["part_id"]) == part_id:
            update_records.append(
                {
                    "iteration": int(update["iteration"]),
                    "part_id": part_id,
                    **{
                        name: float.fromhex(update[name])
                        for name in (
                            "previous_norm",
                            "previous_avg",
                            "old_norm_over_avg",
                            "wsum_norm",
                            "sqrt_2_wsum",
                            "new_norm",
                        )
                    },
                }
            )
    _require(
        len(expectation_records) == 1,
        f"expected exactly one native expectation operand record for part {part_id}, "
        f"found {len(expectation_records)}",
    )
    return {
        "expectation": expectation_records[0],
        "updates": update_records,
        "log": str(path.resolve()),
        "log_sha256": _sha256(path),
    }


def _load_native_factor_report(path: Path, *, source_index: int) -> dict[str, object]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "recovar.em.k1_native_normalization_factor.v1",
        "native factor report schema changed",
    )
    _require(
        int(report["source_index_zero_based"]) == source_index,
        "native factor report source identity changed",
    )
    recovered = report["recovered"]
    factor = np.float32(recovered["factor_float32"])
    bits = f"0x{factor.view(np.uint32).item():08x}"
    _require(bits == recovered["factor_float32_bits"], "native factor value and bits differ")
    _require(int(recovered["mismatch_count"]) == 0, "native factor report is not byte exact")
    return {
        "normalization_factor_float32": float(factor),
        "normalization_factor_bits": bits,
        "factor_report": str(path.resolve()),
        "factor_report_sha256": _sha256(path),
    }


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
    parser.add_argument("--relion-native-preprocess-capture", type=Path)
    parser.add_argument("--relion-native-operands-log", type=Path)
    parser.add_argument("--relion-native-factor-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if (args.relion_native_preprocess_capture is None) != (
        args.relion_native_operands_log is None
    ):
        raise ValueError(
            "--relion-native-preprocess-capture and --relion-native-operands-log "
            "must be supplied together"
        )
    if args.relion_native_factor_report is not None and args.relion_native_preprocess_capture is not None:
        raise ValueError(
            "--relion-native-factor-report is mutually exclusive with the native capture/log pair"
        )

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

    relion_native = None
    if args.relion_native_factor_report is not None:
        relion_native = _load_native_factor_report(
            args.relion_native_factor_report,
            source_index=args.source_index,
        )
    elif args.relion_native_preprocess_capture is not None:
        native_capture = load_preprocess_capture(args.relion_native_preprocess_capture)
        _require(native_capture.stack_index == stack_index, "native capture stack identity changed")
        native_operands = _parse_native_norm_operands(
            args.relion_native_operands_log,
            part_id=native_capture.part_id,
        )
        native_factor = np.float32(native_capture.norm_correction)
        expectation = native_operands["expectation"]
        _require(
            expectation["quotient_float32_bits"]
            == f"0x{native_factor.view(np.uint32).item():08x}",
            "native capture factor and logged quotient bits differ",
        )
        relion_native = {
            "normalization_factor_float32": float(native_factor),
            "normalization_factor_bits": f"0x{native_factor.view(np.uint32).item():08x}",
            "preprocess_capture": str(args.relion_native_preprocess_capture.resolve()),
            "preprocess_capture_sha256": native_capture.sha256,
            "operands": native_operands,
        }

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
        "relion_native": relion_native,
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
            "native_normalization_float32_ulp_distance": (
                None
                if relion_native is None
                else _positive_float32_ulp_distance(
                    recovar_normalization_factor,
                    relion_native["normalization_factor_float32"],
                )
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
            "relion_native_preprocess_capture": (
                None
                if args.relion_native_preprocess_capture is None
                else str(args.relion_native_preprocess_capture.resolve())
            ),
            "relion_native_operands_log": (
                None
                if args.relion_native_operands_log is None
                else str(args.relion_native_operands_log.resolve())
            ),
            "relion_native_factor_report": (
                None
                if args.relion_native_factor_report is None
                else str(args.relion_native_factor_report.resolve())
            ),
        },
        "interpretation_limit": (
            "Serialized RELION norm fields remain decimal STAR values. "
            + (
                "No native in-memory operand capture was supplied."
                if relion_native is None
                else (
                    "The optional native section contains the byte-exact float32 factor recovered "
                    "from RELION's captured normalized real image; it does not contain the host "
                    "update numerator or denominator."
                    if args.relion_native_factor_report is not None
                    else "The optional selected-particle native section contains exact hex-float "
                    "in-memory operands and the captured float32 accelerator quotient."
                )
            )
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
