#!/usr/bin/env python3
"""Compare complete RELION and RECOVAR BPref pre-scatter operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_prescatter import (
        CaptureArtifact,
        load_recovar_stack_indices,
        validate_directory,
    )
else:
    from validate_relion_bpref_prescatter import (  # type: ignore[no-redef]
        CaptureArtifact,
        load_recovar_stack_indices,
        validate_directory,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_gate(path: Path, field: str) -> dict[str, object]:
    report = json.loads(Path(path).read_text())
    _require(report.get(field) is True, f"required gate {field!r} did not pass: {path}")
    return report


def _array_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    _require(lhs.shape == rhs.shape, f"array shape mismatch: {lhs.shape} != {rhs.shape}")
    left = lhs.astype(np.complex128, copy=False).reshape(-1)
    right = rhs.astype(np.complex128, copy=False).reshape(-1)
    delta = right - left
    absolute = np.abs(delta)
    reference = np.abs(left)
    reference_l2 = max(float(np.linalg.norm(left)), np.finfo(np.float64).tiny)
    significant = reference > max(float(reference.max(initial=0.0)) * 1e-8, 1e-30)
    relative = absolute[significant] / reference[significant]
    relative_quantiles = {
        "relative_significant_p50": None,
        "relative_significant_p95": None,
        "relative_significant_p99": None,
    }
    if relative.size:
        relative_quantiles = {
            "relative_significant_p50": float(np.quantile(relative, 0.50)),
            "relative_significant_p95": float(np.quantile(relative, 0.95)),
            "relative_significant_p99": float(np.quantile(relative, 0.99)),
        }
    return {
        "shape": list(lhs.shape),
        "lhs_dtype": str(lhs.dtype),
        "rhs_dtype": str(rhs.dtype),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "mismatch_count": int(np.count_nonzero(lhs != rhs)),
        "relative_l2_over_lhs": float(np.linalg.norm(delta) / reference_l2),
        "delta_mean_abs": float(np.mean(absolute)),
        "delta_p50_abs": float(np.quantile(absolute, 0.50)),
        "delta_p95_abs": float(np.quantile(absolute, 0.95)),
        "delta_p99_abs": float(np.quantile(absolute, 0.99)),
        "delta_max_abs": float(np.max(absolute)),
        **relative_quantiles,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    _require(values.size > 0 and np.all(np.isfinite(values)), "quantile input is empty or non-finite")
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _per_particle_relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    left = lhs.astype(np.complex128, copy=False)
    right = rhs.astype(np.complex128, copy=False)
    numerator = np.linalg.norm(right - left, axis=1)
    denominator = np.maximum(np.linalg.norm(left, axis=1), np.finfo(np.float64).tiny)
    return numerator / denominator


def _shell_metrics(lhs: np.ndarray, rhs: np.ndarray, window_indices: np.ndarray) -> list[dict[str, object]]:
    x = window_indices % 129
    y = window_indices // 129
    y = np.where(y > 128, y - 256, y)
    shell = np.rint(np.sqrt(x * x + y * y)).astype(np.int32)
    records = []
    for radius in np.unique(shell):
        keep = shell == radius
        records.append({"shell": int(radius), **_array_metrics(lhs[:, keep], rhs[:, keep])})
    return records


def _load_device_support(
    geometry_directory: Path,
    stack_indices: np.ndarray,
    window_indices: np.ndarray,
) -> np.ndarray:
    records: dict[int, np.ndarray] = {}
    paths = sorted(Path(geometry_directory).glob("*.npz"))
    _require(bool(paths), f"no RECOVAR device geometry shards: {geometry_directory}")
    for path in paths:
        with np.load(path, allow_pickle=False) as geometry:
            companion = Path(str(geometry["companion_contribution_path"]))
            particle_rows = np.asarray(geometry["signature_particle_rows"], dtype=np.int64)
            pixels = np.asarray(geometry["signature_pixel_indices"], dtype=np.int32)
            flags = np.asarray(geometry["signature_row_flags"], dtype=np.uint32)
        with np.load(companion, allow_pickle=False) as contribution:
            companion_stacks = np.asarray(contribution["stack_indices_1based"], dtype=np.int64)
        selected_stacks = companion_stacks[particle_rows]
        _require(
            np.array_equal(pixels, np.broadcast_to(window_indices, pixels.shape)),
            f"device geometry pixel identities changed: {path}",
        )
        for row, stack in enumerate(selected_stacks):
            key = int(stack)
            _require(key not in records, f"duplicate device support for stack {key}")
            records[key] = (flags[row] & np.uint32(64)) != 0
    _require(set(records) == set(stack_indices.tolist()), "device support stack identities are incomplete")
    return np.stack([records[int(stack)] for stack in stack_indices])


def _load_recovar(
    contribution_directory: Path, geometry_directory: Path
) -> dict[str, np.ndarray]:
    paths = sorted(Path(contribution_directory).glob("*.npz"))
    _require(bool(paths), f"no RECOVAR contribution shards: {contribution_directory}")
    stacks = []
    data = []
    weight = []
    rotations = []
    global_rotations = []
    window_reference = None
    zero_valid_rows = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            _require(not bool(archive["shadow_only_mode"]), f"shadow-only shard is inadmissible: {path}")
            stack = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
            particle_rows = np.asarray(archive["active_particle_rows"], dtype=np.int64)
            summed = np.asarray(archive["active_summed"], dtype=np.complex64)
            ctf = np.asarray(archive["active_ctf_probs"], dtype=np.float32)
            active_rotations = np.asarray(archive["active_rotations"], dtype=np.float32)
            active_global = np.asarray(archive["active_global_rotation_indices"], dtype=np.int64)
            window = np.asarray(archive["window_indices"], dtype=np.int32)
        if window_reference is None:
            window_reference = window
        else:
            _require(np.array_equal(window_reference, window), f"RECOVAR support changed: {path}")
        selected = []
        for particle in range(stack.size):
            rows = np.flatnonzero(particle_rows == particle)
            _require(rows.size == 8, f"expected eight valid RECOVAR rotation rows: {path}, particle={particle}")
            positive = rows[np.any(ctf[rows] > 0, axis=1)]
            _require(positive.size == 1, f"expected one positive RECOVAR winner row: {path}, particle={particle}")
            others = rows[rows != positive[0]]
            _require(
                np.all(summed[others] == 0) and np.all(ctf[others] == 0),
                f"nonwinner RECOVAR rows are not exact zero: {path}, particle={particle}",
            )
            zero_valid_rows += others.size
            selected.append(int(positive[0]))
        selected_array = np.asarray(selected, dtype=np.int64)
        stacks.append(stack)
        data.append(summed[selected_array])
        weight.append(ctf[selected_array])
        rotations.append(active_rotations[selected_array])
        global_rotations.append(active_global[selected_array])
    stack_array = np.concatenate(stacks)
    order = np.argsort(stack_array)
    _require(np.unique(stack_array).size == stack_array.size, "duplicate RECOVAR stack identities")
    assert window_reference is not None
    sorted_stacks = stack_array[order]
    support_mask = _load_device_support(geometry_directory, sorted_stacks, window_reference)
    return {
        "stack_indices": sorted_stacks,
        "data": np.concatenate(data)[order],
        "weight": np.concatenate(weight)[order],
        "rotations": np.concatenate(rotations)[order],
        "global_rotations": np.concatenate(global_rotations)[order],
        "window_indices": window_reference,
        "support_mask": support_mask,
        "exact_zero_nonwinner_rows": np.asarray(zero_valid_rows, dtype=np.int64),
        "shard_count": np.asarray(len(paths), dtype=np.int64),
    }


def _align_relion(
    artifacts: tuple[CaptureArtifact, ...],
    recovar: dict[str, np.ndarray],
    *,
    mpi_rank: int,
) -> dict[str, np.ndarray]:
    selected = sorted(
        (artifact for artifact in artifacts if artifact.mpi_rank == mpi_rank),
        key=lambda artifact: artifact.stack_index,
    )
    stacks = np.asarray([artifact.stack_index for artifact in selected], dtype=np.int64)
    _require(np.array_equal(stacks, recovar["stack_indices"]), "sorted stack identities changed after validation")
    window = recovar["window_indices"]
    relion_data = np.empty((stacks.size, window.size), dtype=np.complex64)
    relion_weight = np.empty((stacks.size, window.size), dtype=np.float32)
    relion_rotations = np.empty((stacks.size, 3, 3), dtype=np.float32)
    orientation_keys = np.empty(stacks.size, dtype=np.uint64)
    oversampled_rotations = np.empty(stacks.size, dtype=np.uint64)
    supported_rows = np.empty(stacks.size, dtype=np.int64)
    positive_candidates = np.empty(stacks.size, dtype=np.int64)
    radius_excluded = np.empty(stacks.size, dtype=np.int64)
    for index, artifact in enumerate(selected):
        rows = artifact.rows
        box_size = int(artifact.header[13])
        _require(
            int(artifact.header[12]) == box_size // 2 + 1 and int(artifact.header[14]) == 1,
            f"unexpected RELION half-spectrum shape: {artifact.path}",
        )
        orientations = np.unique(rows["orientation_local"])
        _require(orientations.size == 1, f"expected one RELION active winner orientation: {artifact.path}")
        orientation = int(orientations[0])
        rotation = artifact.rotations[orientation]
        full_indices = (
            (rows["y"].astype(np.int64) % box_size) * (box_size // 2 + 1)
            + rows["x"]
        )
        _require(np.unique(full_indices).size == full_indices.size, f"duplicate RELION support pixel: {artifact.path}")
        window_order = np.argsort(window)
        positions = np.searchsorted(window[window_order], full_indices)
        _require(
            np.all(positions < window.size)
            and np.array_equal(window[window_order][positions], full_indices),
            f"RELION emitted a pixel outside the RECOVAR source window: {artifact.path}",
        )
        output_columns = window_order[positions]
        expected_columns = np.flatnonzero(recovar["support_mask"][index])
        _require(
            np.array_equal(np.sort(output_columns), expected_columns),
            f"RELION/RECOVAR device support sets differ: {artifact.path}",
        )
        # The capture stores RELION's native pre-scatter values. Use the same
        # qualified RELION-to-RECOVAR conversion as the sealed aggregate BPref
        # comparison: -1/N^2 for data and 1/N^4 for weight. Derive N from the
        # capture header without attributing it to either forward-FFT convention.
        relion_data[index] = 0
        relion_weight[index] = 0
        relion_data[index, output_columns] = (
            rows["source_re"] + 1j * rows["source_im"]
        ) * np.float32(-1.0 / box_size**2)
        relion_weight[index, output_columns] = rows["source_weight"] * np.float32(
            1.0 / box_size**4
        )
        relion_rotations[index] = rotation["matrix"].reshape(3, 3)
        orientation_keys[index] = rotation["orientation_class_key"]
        oversampled_rotations[index] = rotation["oversampled_rotation"]
        supported_rows[index] = rows.size
        positive_candidates[index] = artifact.header[38]
        radius_excluded[index] = artifact.header[39]
    return {
        "stack_indices": stacks,
        "data": relion_data,
        "weight": relion_weight,
        "rotations": relion_rotations,
        "orientation_class_keys": orientation_keys,
        "oversampled_rotations": oversampled_rotations,
        "supported_rows": supported_rows,
        "positive_candidates": positive_candidates,
        "radius_excluded": radius_excluded,
        "image_box_size": np.asarray(int(selected[0].header[13]), dtype=np.int64),
    }


def compare(
    capture_directory: Path,
    contribution_directory: Path,
    geometry_directory: Path,
    *,
    validation_json: Path,
    inertness_json: Path,
    mpi_rank: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    validation = _load_gate(validation_json, "classification_ready")
    inertness = _load_gate(inertness_json, "capture_inertness_qualified")
    recovar = _load_recovar(contribution_directory, geometry_directory)
    expected = load_recovar_stack_indices((contribution_directory,))
    artifacts, current_validation = validate_directory(
        capture_directory,
        expected_particles=int(validation["particle_count"]),
        expected_stack_indices=expected,
        expected_stack_mpi_rank=mpi_rank,
    )
    _require(current_validation["classification_ready"] is True, "fresh validation did not pass")
    relion = _align_relion(artifacts, recovar, mpi_rank=mpi_rank)

    transpose_delta = relion["rotations"].transpose(0, 2, 1) - recovar["rotations"]
    rotation_metrics = _array_metrics(
        recovar["rotations"], relion["rotations"].transpose(0, 2, 1)
    )
    oversampled_exact = relion["oversampled_rotations"] == (recovar["global_rotations"] % 8)
    recovar_data_active = np.where(recovar["support_mask"], recovar["data"], 0)
    recovar_weight_active = np.where(recovar["support_mask"], recovar["weight"], 0)
    data_particle = _per_particle_relative_l2(recovar_data_active, relion["data"])
    weight_particle = _per_particle_relative_l2(recovar_weight_active, relion["weight"])
    data_delta_l2 = np.linalg.norm(
        relion["data"].astype(np.complex128) - recovar_data_active.astype(np.complex128),
        axis=1,
    )
    outlier_index = int(np.argmax(data_delta_l2))
    without_outlier = np.arange(data_delta_l2.size) != outlier_index
    delta_energy = data_delta_l2**2
    total_delta_energy = float(np.sum(delta_energy))
    support_counts = np.count_nonzero(recovar["support_mask"], axis=1)
    support_exact = bool(np.array_equal(relion["supported_rows"], support_counts))
    geometry_aligned = bool(np.max(np.abs(transpose_delta)) <= 1e-6 and support_exact)
    report: dict[str, object] = {
        "schema": "relion-recovar-bpref-prescatter-comparison-v1",
        "metric_policy": "exact/array metrics for intermediate operands; no correlation",
        "gates": {
            "fresh_capture_validation": current_validation["classification_ready"],
            "capture_inertness": inertness["capture_inertness_qualified"],
            "comparison_ready": geometry_aligned,
        },
        "scope": {
            "mpi_rank": mpi_rank,
            "particle_count": int(recovar["stack_indices"].size),
            "pixels_per_particle": int(recovar["window_indices"].size),
            "relion_image_box_size": int(relion["image_box_size"]),
            "recovar_shard_count": int(recovar["shard_count"]),
            "recovar_exact_zero_nonwinner_rows": int(recovar["exact_zero_nonwinner_rows"]),
            "relion_supported_rows": int(np.sum(relion["supported_rows"])),
            "relion_positive_candidates": int(np.sum(relion["positive_candidates"])),
            "relion_radius_excluded_candidates": int(np.sum(relion["radius_excluded"])),
        },
        "identity_and_geometry": {
            "stack_identities_exact": bool(np.array_equal(recovar["stack_indices"], relion["stack_indices"])),
            "support_set_exact_for_every_particle": support_exact,
            "supported_rows_per_particle": _quantiles(relion["supported_rows"]),
            "rotation_relion_transpose_vs_recovar": rotation_metrics,
            "oversampled_rotation_identity_match_count": int(np.count_nonzero(oversampled_exact)),
            "oversampled_rotation_identity_mismatch_count": int(np.count_nonzero(~oversampled_exact)),
        },
        "operands": {
            "data_numerator_recovar_vs_scaled_negative_relion": _array_metrics(
                recovar_data_active, relion["data"]
            ),
            "real_weight_recovar_vs_scaled_relion": _array_metrics(
                recovar_weight_active, relion["weight"]
            ),
            "data_per_particle_relative_l2": _quantiles(data_particle),
            "weight_per_particle_relative_l2": _quantiles(weight_particle),
            "largest_data_delta_particle": {
                "stack_index_1based": int(recovar["stack_indices"][outlier_index]),
                "per_particle_relative_l2": float(data_particle[outlier_index]),
                "fraction_of_total_data_delta_l2_squared": (
                    float(delta_energy[outlier_index] / total_delta_energy)
                    if total_delta_energy > 0
                    else 0.0
                ),
                "all_other_particles": (
                    _array_metrics(
                        recovar_data_active[without_outlier], relion["data"][without_outlier]
                    )
                    if np.count_nonzero(without_outlier)
                    else None
                ),
            },
            "data_shellwise": _shell_metrics(
                recovar_data_active, relion["data"], recovar["window_indices"]
            ),
            "weight_shellwise": _shell_metrics(
                recovar_weight_active, relion["weight"], recovar["window_indices"]
            ),
        },
        "classification": (
            "pre_scatter_operand_generation_difference"
            if geometry_aligned
            else "unresolved_geometry_or_support_difference"
        ),
        "qualification": (
            "The classification localises the first aggregate difference at the captured pre-scatter "
            "operands after exact stack/support alignment and the known RELION-transpose rotation "
            "convention. Precision versus formulation requires float64/order controls and controlled "
            "accumulator substitution; this report alone does not make that second classification."
        ),
    }
    arrays = {
        "stack_indices_1based": recovar["stack_indices"],
        "window_indices": recovar["window_indices"],
        "recovar_data": recovar["data"],
        "recovar_device_support_mask": recovar["support_mask"],
        "relion_data_recovar_units": relion["data"],
        "recovar_weight": recovar["weight"],
        "relion_weight_recovar_units": relion["weight"],
        "recovar_rotations": recovar["rotations"],
        "relion_rotations": relion["rotations"],
        "recovar_global_rotation_indices": recovar["global_rotations"],
        "relion_orientation_class_keys": relion["orientation_class_keys"],
        "relion_oversampled_rotations": relion["oversampled_rotations"],
    }
    return report, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("contribution_directory", type=Path)
    parser.add_argument("geometry_directory", type=Path)
    parser.add_argument("--validation-json", required=True, type=Path)
    parser.add_argument("--inertness-json", required=True, type=Path)
    parser.add_argument("--mpi-rank", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-arrays", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report, arrays = compare(
        args.capture_directory,
        args.contribution_directory,
        args.geometry_directory,
        validation_json=args.validation_json,
        inertness_json=args.inertness_json,
        mpi_rank=args.mpi_rank,
    )
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.output_arrays, **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gates"]["comparison_ready"]:
        raise RuntimeError("RELION/RECOVAR pre-scatter comparison failed geometry alignment")


if __name__ == "__main__":
    main()
