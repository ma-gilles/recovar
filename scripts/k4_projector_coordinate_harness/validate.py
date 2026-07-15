#!/usr/bin/env python3
"""Fail-closed validation and array report for the K4 projector microharness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VARIANTS = (
    "current_source",
    "relion_source",
    "explicit_fma_current",
    "explicit_fma_relion",
    "noncontracted_relion",
    "relion_direct_source",
)
N_ROTATIONS = 8
N_PIXELS = 840
PPREF_COUNT = 83 * 83 * 42
COORDINATE_FIELDS = 9
SPECIAL_ROTATION = 4
SPECIAL_PIXEL = 242


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"shape mismatch: {actual_array.shape} versus {expected_array.shape}")
    delta = np.abs(actual_array.astype(np.complex128) - expected_array.astype(np.complex128))
    denominator = float(np.sum(np.abs(expected_array), dtype=np.float64))
    numerator = float(np.sum(delta, dtype=np.float64))
    return {
        "exact_equal": bool(np.array_equal(actual_array, expected_array)),
        "different_elements": int(np.count_nonzero(actual_array != expected_array)),
        "max_abs": float(np.max(delta, initial=0.0)),
        "mean_abs": float(np.mean(delta)) if delta.size else 0.0,
        "rel_l1": numerator / denominator if denominator else (0.0 if numerator == 0.0 else None),
    }


def uint32_bits(values: np.ndarray) -> list[int]:
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64).tolist()


def validate_inputs(root: Path, manifest: dict) -> dict:
    inputs = root / "inputs"
    observed = {}
    for name, expected_hash in manifest["input_hashes"].items():
        path = inputs / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"frozen input hash mismatch for {path}")
        observed[name] = actual_hash
    for raw_path, expected_hash in manifest["source_hashes"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"immutable source artifact drifted: {path}")

    ppref_real = np.fromfile(inputs / "ppref_real.f32", dtype=np.float32)
    ppref_imag = np.fromfile(inputs / "ppref_imag.f32", dtype=np.float32)
    eulers = np.fromfile(inputs / "eulers.f32", dtype=np.float32)
    relion_reference = np.fromfile(inputs / "relion_reference.f32x2", dtype=np.complex64)
    recovar_reference = np.fromfile(inputs / "recovar_reference.f32x2", dtype=np.complex64)
    mapping = np.fromfile(inputs / "relion_to_recovar_column.i32", dtype=np.int32)
    expected_shapes = {
        "ppref_real": (PPREF_COUNT,),
        "ppref_imag": (PPREF_COUNT,),
        "eulers": (N_ROTATIONS * 9,),
        "relion_reference": (N_ROTATIONS * N_PIXELS,),
        "recovar_reference": (N_ROTATIONS * N_PIXELS,),
        "mapping": (N_PIXELS,),
    }
    arrays = {
        "ppref_real": ppref_real,
        "ppref_imag": ppref_imag,
        "eulers": eulers,
        "relion_reference": relion_reference,
        "recovar_reference": recovar_reference,
        "mapping": mapping,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"input topology mismatch for {name}: {arrays[name].shape}")
    if not all(np.all(np.isfinite(array)) for name, array in arrays.items() if name != "mapping"):
        raise ValueError("non-finite frozen input")
    if sorted(mapping.tolist()) != list(range(N_PIXELS)) or int(mapping[SPECIAL_PIXEL]) != 641:
        raise ValueError("special RELION-to-RECOVAR pixel identity drifted")
    special = manifest["special_identity"]
    relion_reference = relion_reference.reshape(N_ROTATIONS, N_PIXELS)
    recovar_reference = recovar_reference.reshape(N_ROTATIONS, N_PIXELS)
    if (
        uint32_bits(
            [
                relion_reference[SPECIAL_ROTATION, SPECIAL_PIXEL].real,
                relion_reference[SPECIAL_ROTATION, SPECIAL_PIXEL].imag,
            ]
        )
        != special["relion_reference_bits"]
    ):
        raise ValueError("special RELION reference bits drifted")
    if (
        uint32_bits(
            [
                recovar_reference[SPECIAL_ROTATION, SPECIAL_PIXEL].real,
                recovar_reference[SPECIAL_ROTATION, SPECIAL_PIXEL].imag,
            ]
        )
        != special["recovar_reference_bits"]
    ):
        raise ValueError("special RECOVAR reference bits drifted")
    return {"status": "pass", "hashes": observed}


def read_projection(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.complex64)
    if values.shape != (N_ROTATIONS * N_PIXELS,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid projection output: {path}")
    return values.reshape(N_ROTATIONS, N_PIXELS)


def read_coordinates(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = N_ROTATIONS * N_PIXELS * COORDINATE_FIELDS
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid coordinate output: {path}")
    return values.reshape(N_ROTATIONS, N_PIXELS, COORDINATE_FIELDS)


def special_coordinate_record(coordinates: np.ndarray) -> dict:
    values = coordinates[SPECIAL_ROTATION, SPECIAL_PIXEL]
    post_y = np.float32(values[4])
    fraction = np.float32(post_y - np.floor(post_y))
    scaled = np.float32(fraction * np.float32(256.0))
    lower = np.float32(np.floor(scaled))
    upper = np.float32(np.ceil(scaled))
    return {
        "values": [float(value) for value in values],
        "bits": uint32_bits(values),
        "post_hermitian_y_fraction": float(fraction),
        "post_hermitian_y_fraction_times_256": float(scaled),
        "adjacent_bins": [int(lower), int(upper)],
        "distance_from_half_step": float(abs(float(scaled) - (math.floor(float(scaled)) + 0.5))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "input_manifest.json").read_text())
    input_gate = validate_inputs(root, manifest)
    if args.inputs_only:
        print(json.dumps({"schema": "k4_projector_inputs_gate_v1", **input_gate}, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --inputs-only is set")

    results = root / "results"
    ppref_real = np.fromfile(root / "inputs/ppref_real.f32", dtype=np.float32)
    ppref_imag = np.fromfile(root / "inputs/ppref_imag.f32", dtype=np.float32)
    staged_direct_real = np.fromfile(results / "staged_relion_direct_real.f32", dtype=np.float32)
    staged_direct_imag = np.fromfile(results / "staged_relion_direct_imag.f32", dtype=np.float32)
    staged_recovar_real = np.fromfile(results / "staged_recovar_real.f32", dtype=np.float32)
    staged_recovar_imag = np.fromfile(results / "staged_recovar_imag.f32", dtype=np.float32)
    for name, array in {
        "staged_direct_real": staged_direct_real,
        "staged_direct_imag": staged_direct_imag,
        "staged_recovar_real": staged_recovar_real,
        "staged_recovar_imag": staged_recovar_imag,
    }.items():
        if array.shape != (PPREF_COUNT,) or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid staged texture source: {name}")
    if not np.array_equal(staged_direct_real, ppref_real) or not np.array_equal(staged_direct_imag, ppref_imag):
        raise ValueError("RELION-direct stage output is not the frozen PPref input")
    staged_real_metrics = exact_metrics(staged_recovar_real, staged_direct_real)
    staged_imag_metrics = exact_metrics(staged_recovar_imag, staged_direct_imag)
    stage_exact = staged_real_metrics["exact_equal"] and staged_imag_metrics["exact_equal"]

    relion_reference = np.fromfile(root / "inputs/relion_reference.f32x2", dtype=np.complex64).reshape(
        N_ROTATIONS, N_PIXELS
    )
    recovar_reference = np.fromfile(root / "inputs/recovar_reference.f32x2", dtype=np.complex64).reshape(
        N_ROTATIONS, N_PIXELS
    )
    projections = {name: read_projection(results / f"projection_{name}.f32x2") for name in VARIANTS}
    coordinates = {name: read_coordinates(results / f"coordinates_{name}.f32") for name in VARIANTS}
    lower = read_projection(results / "projection_relion_y_bin_lower.f32x2")
    upper = read_projection(results / "projection_relion_y_bin_upper.f32x2")

    variant_metrics = {}
    for name in VARIANTS:
        variant_metrics[name] = {
            "versus_relion_capture": exact_metrics(projections[name], relion_reference),
            "versus_recovar_capture": exact_metrics(projections[name], recovar_reference),
        }
    coordinate_metrics = {name: exact_metrics(coordinates[name], coordinates["relion_source"]) for name in VARIANTS}

    target = (SPECIAL_ROTATION, SPECIAL_PIXEL)
    special_projections = {
        name: {
            "real_imag": [float(projections[name][target].real), float(projections[name][target].imag)],
            "bits": uint32_bits([projections[name][target].real, projections[name][target].imag]),
            "abs_gap_to_relion_capture": float(abs(projections[name][target] - relion_reference[target])),
            "abs_gap_to_recovar_capture": float(abs(projections[name][target] - recovar_reference[target])),
        }
        for name in VARIANTS
    }
    special_projections["relion_y_bin_lower"] = {
        "real_imag": [float(lower[target].real), float(lower[target].imag)],
        "bits": uint32_bits([lower[target].real, lower[target].imag]),
    }
    special_projections["relion_y_bin_upper"] = {
        "real_imag": [float(upper[target].real), float(upper[target].imag)],
        "bits": uint32_bits([upper[target].real, upper[target].imag]),
    }

    special_coordinates = {name: special_coordinate_record(coordinates[name]) for name in VARIANTS}
    near_half = min(record["distance_from_half_step"] for record in special_coordinates.values())
    if near_half >= 0.01:
        raise ValueError(f"special coordinate is no longer near the 1/256 half-step: {near_half}")

    if not stage_exact:
        classification = "staged_texture_source_mismatch"
    elif variant_metrics["relion_direct_source"]["versus_relion_capture"]["exact_equal"]:
        current_exact = variant_metrics["current_source"]["versus_recovar_capture"]["exact_equal"]
        relion_exact = variant_metrics["relion_source"]["versus_relion_capture"]["exact_equal"]
        if current_exact and relion_exact:
            classification = "coordinate_arithmetic"
        elif relion_exact:
            classification = "recovar_replay_mismatch"
        else:
            classification = "coordinate_or_compiler_unresolved"
    else:
        classification = "relion_direct_replay_mismatch"

    output_hashes = {path.name: sha256(path) for path in sorted(results.iterdir()) if path.is_file()}
    report = {
        "schema": "k4_p3591_projector_coordinate_report_v1",
        "status": "pass",
        "classification": classification,
        "metrics_policy": "exact/array metrics only; no correlation and no map-quality claim",
        "input_gate": input_gate,
        "staged_texture_source": {
            "captured_before_cudaMemcpy3D": True,
            "exact_equal": stage_exact,
            "real": staged_real_metrics,
            "imag": staged_imag_metrics,
            "hashes": {
                "relion_direct_real": sha256(results / "staged_relion_direct_real.f32"),
                "relion_direct_imag": sha256(results / "staged_relion_direct_imag.f32"),
                "recovar_real": sha256(results / "staged_recovar_real.f32"),
                "recovar_imag": sha256(results / "staged_recovar_imag.f32"),
            },
        },
        "variant_metrics": variant_metrics,
        "coordinate_metrics_versus_relion_source": coordinate_metrics,
        "special_identity": manifest["special_identity"],
        "special_coordinates": special_coordinates,
        "special_projections": special_projections,
        "adjacent_y_bin_metrics": {
            "lower_versus_relion_capture": exact_metrics(lower, relion_reference),
            "upper_versus_relion_capture": exact_metrics(upper, relion_reference),
        },
        "output_hashes": output_hashes,
        "limitations": [
            "Frozen device inputs are replayed in a target-only kernel; production registers are not intercepted.",
            "A standalone NVCC translation unit may choose different contraction than either full application.",
            "The adjacent 1/256-bin samples are diagnostics for texture quantization, not proposed production behavior.",
            "This one-particle projection boundary cannot establish FSC/FSC-AUC map quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
