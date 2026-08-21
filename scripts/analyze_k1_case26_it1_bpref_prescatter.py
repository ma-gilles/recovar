#!/usr/bin/env python3
"""Compare case-26 iteration-1 RELION and RECOVAR BPref source operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .validate_relion_bpref_prescatter import load_artifact
else:
    from validate_relion_bpref_prescatter import (  # type: ignore[no-redef]
        load_artifact,
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


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "comparison shape changed or is empty")
    dtype = np.complex128 if np.iscomplexobj(left) or np.iscomplexobj(right) else np.float64
    delta = right.astype(dtype, copy=False) - left.astype(dtype, copy=False)
    denominator = max(float(np.linalg.norm(left.reshape(-1))), np.finfo(np.float64).tiny)
    unequal = np.flatnonzero((left != right).reshape(-1))
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(unequal.size == 0),
        "mismatch_count": int(unequal.size),
        "relative_l2_over_reference": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)),
        "reference_sha256": _array_sha256(left),
        "candidate_sha256": _array_sha256(right),
    }


def _rotation_map(native: np.ndarray, recovar: np.ndarray) -> tuple[np.ndarray, float]:
    native_matrix = np.asarray(native["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    recovar_matrix = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    error = np.max(
        np.abs(native_matrix.transpose(0, 2, 1)[:, None] - recovar_matrix[None]),
        axis=(2, 3),
    )
    nearest = np.argmin(error, axis=1)
    nearest_error = error[np.arange(error.shape[0]), nearest]
    _require(
        np.max(nearest_error, initial=0.0) <= 1.0e-6
        and np.unique(nearest).size == nearest.size,
        "native rotations do not map one-to-one to RECOVAR",
    )
    return nearest.astype(np.int64), float(np.max(nearest_error, initial=0.0))


def _recovar_coordinates(indices: np.ndarray, physical_image_size: int) -> list[tuple[int, int]]:
    half_width = physical_image_size // 2 + 1
    return [
        (
            int(index % half_width),
            int(index // half_width - physical_image_size // 2),
        )
        for index in np.asarray(indices, dtype=np.int64)
    ]


def _compare_particle(
    artifact,
    recovar_path: Path,
    *,
    half: int,
    source_index: int,
    physical_image_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: archive[name] for name in archive.files}
    _require(int(recovar["original_index"]) == source_index, "RECOVAR source identity changed")
    _require(int(artifact.stack_index) == source_index + 1, "RELION stack/source identity changed")
    _require(int(artifact.header[5]) == 1, "native physical iteration changed")
    _require(int(recovar["current_size"]) == int(artifact.header[13]), "current size changed")

    rotation_map, rotation_error = _rotation_map(artifact.rotations, recovar["rotations"])
    active_native_orientations = np.unique(artifact.rows["orientation_local"])
    _require(active_native_orientations.size == 1, "expected one native firstiter-CC winner")
    native_orientation = int(active_native_orientations[0])

    reconstruction_mask = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    reconstruction_probs = np.asarray(recovar["reconstruction_probs"], dtype=np.float32)
    winner_rows = np.argwhere(reconstruction_mask)
    _require(winner_rows.shape == (1, 2), "expected one RECOVAR firstiter-CC winner")
    recovar_rotation, recovar_translation = (int(value) for value in winner_rows[0])
    _require(
        int(rotation_map[native_orientation]) == recovar_rotation,
        "native and RECOVAR winning rotations changed",
    )
    posterior = np.float32(reconstruction_probs[recovar_rotation, recovar_translation])
    _require(posterior == np.float32(1.0), "RECOVAR firstiter-CC posterior is not exactly one")

    coordinates = _recovar_coordinates(recovar["recon_window_indices"], physical_image_size)
    coordinate_to_column = {coordinate: column for column, coordinate in enumerate(coordinates)}
    native_coordinates = list(
        zip(
            np.asarray(artifact.rows["x"], dtype=np.int64).tolist(),
            np.asarray(artifact.rows["y"], dtype=np.int64).tolist(),
            strict=True,
        )
    )
    _require(
        len(set(native_coordinates)) == len(native_coordinates),
        "native pre-scatter coordinates are duplicated",
    )
    _require(
        all(coordinate in coordinate_to_column for coordinate in native_coordinates),
        "native pre-scatter support extends outside RECOVAR reconstruction support",
    )
    columns = np.asarray(
        [coordinate_to_column[coordinate] for coordinate in native_coordinates],
        dtype=np.int64,
    )

    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    native_numerator = (
        -(
            np.asarray(artifact.rows["source_re"], dtype=np.float32)
            + np.complex64(1j) * np.asarray(artifact.rows["source_im"], dtype=np.float32)
        )
        / n2
    ).astype(np.complex64)
    native_weight = (
        np.asarray(artifact.rows["source_weight"], dtype=np.float32) / n4
    ).astype(np.float32)
    recovar_numerator = (
        posterior
        * np.asarray(recovar["shifted_recon"], dtype=np.complex64)[
            recovar_translation, columns
        ]
    ).astype(np.complex64)
    recovar_weight = (
        posterior
        * np.asarray(recovar["ctf2_over_nv_recon"], dtype=np.float32)[columns]
    ).astype(np.float32)
    comparisons = {
        "numerator_operand": _metric(native_numerator, recovar_numerator),
        "weight_operand": _metric(native_weight, recovar_weight),
    }
    arrays = {
        "source_index": np.asarray(source_index, dtype=np.int64),
        "stack_index_one_based": np.asarray(artifact.stack_index, dtype=np.int64),
        "half": np.asarray(half, dtype=np.int32),
        "x": np.asarray(artifact.rows["x"], dtype=np.int32),
        "y": np.asarray(artifact.rows["y"], dtype=np.int32),
        "recovar_columns": columns,
        "native_numerator": native_numerator,
        "recovar_numerator": recovar_numerator,
        "native_weight": native_weight,
        "recovar_weight": recovar_weight,
    }
    return (
        {
            "source_index": source_index,
            "stack_index_one_based": int(artifact.stack_index),
            "part_id": int(artifact.part_id),
            "half": half,
            "native_capture": str(artifact.path.resolve()),
            "native_capture_sha256": artifact.sha256,
            "recovar_capture": str(recovar_path.resolve()),
            "recovar_capture_sha256": _sha256(recovar_path),
            "native_orientation_local": native_orientation,
            "recovar_rotation_row": recovar_rotation,
            "recovar_translation_row": recovar_translation,
            "rotation_map_max_abs": rotation_error,
            "support_pixel_count": int(columns.size),
            "comparisons": comparisons,
        },
        arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--wavg-report", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    _require(not args.output_arrays.exists(), f"refusing to overwrite {args.output_arrays}")

    paths = sorted(args.native_directory.glob("*.bpre-v1.bin"))
    _require(len(paths) == 17, "native pre-scatter artifact count changed")
    _require(not list(args.native_directory.glob("*.tmp.*")), "incomplete native artifact remains")
    artifacts = tuple(load_artifact(path) for path in paths)
    _require(
        len({artifact.part_id for artifact in artifacts}) == len(artifacts)
        and len({artifact.stack_index for artifact in artifacts}) == len(artifacts),
        "native identities are duplicated",
    )
    validation = {
        "structural_validation": "passed_by_load_artifact_for_every_file",
        "artifact_count": len(artifacts),
        "iteration_values": sorted({int(artifact.header[5]) for artifact in artifacts}),
        "class_values": sorted({int(artifact.header[6]) for artifact in artifacts}),
        "mpi_rank_values": sorted({int(artifact.mpi_rank) for artifact in artifacts}),
        "mpi_rank_note": (
            "OMPI_COMM_WORLD_RANK is absent under the pinned srun launcher; "
            "the passive records retain complete part/stack identities"
        ),
        "artifact_sha256": {
            artifact.path.name: artifact.sha256 for artifact in artifacts
        },
    }
    wavg = json.loads(args.wavg_report.read_text())
    _require(
        wavg.get("schema") == "recovar.em.k1_native_wavg_pixel_comparison.v1",
        "Wavg schema changed",
    )
    wavg_by_source = {int(row["source_index"]): row for row in wavg["particles"]}
    _require(len(wavg_by_source) == len(artifacts), "Wavg/native particle count changed")

    particle_reports = []
    particle_arrays = []
    for artifact in sorted(artifacts, key=lambda item: item.stack_index):
        source_index = int(artifact.stack_index) - 1
        wavg_row = wavg_by_source[source_index]
        _require(int(wavg_row["part_id"]) == artifact.part_id, "RELION part identity changed")
        recovar_path = Path(wavg_row["recovar_pass2_capture"])
        _require(
            recovar_path.resolve().parent == args.recovar_directory.resolve(),
            "Wavg report points outside the declared RECOVAR capture directory",
        )
        report, arrays = _compare_particle(
            artifact,
            recovar_path,
            half=int(wavg_row["half"]),
            source_index=source_index,
            physical_image_size=args.physical_image_size,
        )
        report["upstream_wavg_input_comparisons"] = wavg_row["native_input_comparisons"]
        report["upstream_masked_fourier_pretranslation"] = wavg_row[
            "masked_fourier_pretranslation"
        ]
        particle_reports.append(report)
        particle_arrays.append(arrays)

    aggregate = {}
    output_arrays: dict[str, np.ndarray] = {}
    for half in (1, 2):
        selected = [
            (report, arrays)
            for report, arrays in zip(particle_reports, particle_arrays, strict=True)
            if report["half"] == half
        ]
        _require(bool(selected), f"half {half} panel is empty")
        native_numerator = np.concatenate([arrays["native_numerator"] for _, arrays in selected])
        recovar_numerator = np.concatenate([arrays["recovar_numerator"] for _, arrays in selected])
        native_weight = np.concatenate([arrays["native_weight"] for _, arrays in selected])
        recovar_weight = np.concatenate([arrays["recovar_weight"] for _, arrays in selected])
        aggregate[f"half{half}"] = {
            "particle_count": len(selected),
            "support_pixel_count": int(native_weight.size),
            "numerator_operand": _metric(native_numerator, recovar_numerator),
            "weight_operand": _metric(native_weight, recovar_weight),
            "per_particle_numerator_relative_l2": [
                report["comparisons"]["numerator_operand"]["relative_l2_over_reference"]
                for report, _ in selected
            ],
            "per_particle_weight_relative_l2": [
                report["comparisons"]["weight_operand"]["relative_l2_over_reference"]
                for report, _ in selected
            ],
        }
        output_arrays[f"half{half}_native_numerator"] = native_numerator
        output_arrays[f"half{half}_recovar_numerator"] = recovar_numerator
        output_arrays[f"half{half}_native_weight"] = native_weight
        output_arrays[f"half{half}_recovar_weight"] = recovar_weight

    report = {
        "schema": "recovar.em.k1_case26_it1_bpref_prescatter_boundary.v1",
        "status": "complete",
        "metric_policy": "exact and relative-L2 operand metrics; no correlation",
        "native_validation": validation,
        "wavg_report": str(args.wavg_report.resolve()),
        "wavg_report_sha256": _sha256(args.wavg_report),
        "particle_count": len(particle_reports),
        "aggregate": aggregate,
        "particles": particle_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_arrays, **output_arrays)
    report["output_arrays"] = str(args.output_arrays.resolve())
    report["output_arrays_sha256"] = _sha256(args.output_arrays)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
