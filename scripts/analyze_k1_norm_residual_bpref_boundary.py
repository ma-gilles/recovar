#!/usr/bin/env python3
"""Localize a K=1 norm-residual mismatch at the BPref operand boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_prescatter import load_artifact
else:
    from validate_relion_bpref_prescatter import load_artifact  # type: ignore[no-redef]


SCHEMA = "recovar.em.k1_norm_residual_bpref_boundary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape, f"shape mismatch: {left.shape} != {right.shape}")
    promoted_left = left.astype(np.complex128, copy=False).reshape(-1)
    promoted_right = right.astype(np.complex128, copy=False).reshape(-1)
    delta = promoted_right - promoted_left
    denominator = max(float(np.linalg.norm(promoted_left)), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _rotation_map(native: np.ndarray, recovar: np.ndarray) -> tuple[np.ndarray, float]:
    native_matrices = np.asarray(native["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    recovar_matrices = np.asarray(recovar, dtype=np.float32).reshape(-1, 3, 3)
    distances = np.max(
        np.abs(native_matrices.transpose(0, 2, 1)[:, None] - recovar_matrices[None]),
        axis=(2, 3),
    )
    nearest = np.argmin(distances, axis=1)
    error = distances[np.arange(nearest.size), nearest]
    _require(np.all(error <= 1.0e-6), "native/RECOVAR rotations do not match within 1e-6")
    _require(np.unique(nearest).size == nearest.size, "native rotation mapping is not one-to-one")
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _centered_coordinates(indices: np.ndarray, physical_image_size: int) -> list[tuple[int, int]]:
    packed = np.asarray(indices, dtype=np.int64).reshape(-1)
    half_width = physical_image_size // 2 + 1
    return [
        (int(index % half_width), int(index // half_width - physical_image_size // 2))
        for index in packed
    ]


def _dense_native_operands(
    *,
    rows: np.ndarray,
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
    recon_window_indices: np.ndarray,
    physical_image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rotation_rows, rotation_error = _rotation_map(native_rotations, recovar_rotations)
    coordinates = _centered_coordinates(recon_window_indices, physical_image_size)
    coordinate_to_pixel = {coordinate: index for index, coordinate in enumerate(coordinates)}
    _require(len(coordinate_to_pixel) == len(coordinates), "RECOVAR reconstruction pixels are duplicated")
    native_coordinates = list(zip(rows["x"].astype(int), rows["y"].astype(int)))
    _require(
        all(coordinate in coordinate_to_pixel for coordinate in native_coordinates),
        "native BPref emitted a pixel outside the RECOVAR reconstruction window",
    )
    shape = (np.asarray(recovar_rotations).shape[0], len(coordinates))
    data = np.zeros(shape, dtype=np.complex64)
    weight = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    data_scale = np.float32(-1.0 / physical_image_size**2)
    weight_scale = np.float32(1.0 / physical_image_size**4)
    for row in rows:
        rotation = int(rotation_rows[int(row["orientation_local"])])
        pixel = coordinate_to_pixel[(int(row["x"]), int(row["y"]))]
        key = (rotation, pixel)
        _require(key not in seen, "native BPref rotation/pixel row is duplicated")
        seen.add(key)
        data[rotation, pixel] = np.complex64(row["source_re"] + 1j * row["source_im"]) * data_scale
        weight[rotation, pixel] = np.float32(row["source_weight"]) * weight_scale
    return data, weight, {
        "rotation_max_abs": rotation_error,
        "native_supported_rows": len(seen),
        "recovar_rotation_count": int(shape[0]),
        "reconstruction_pixel_count": int(shape[1]),
        "data_scale": float(data_scale),
        "weight_scale": float(weight_scale),
    }


def _norm_terms(
    projection: np.ndarray,
    projection_abs2: np.ndarray,
    summed: np.ndarray,
    ctf_prob: np.ndarray,
    noise_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    proj = np.asarray(projection, dtype=np.complex64)
    proj_abs2 = np.asarray(projection_abs2, dtype=np.float32)
    data = np.asarray(summed, dtype=np.complex64)
    weight = np.asarray(ctf_prob, dtype=np.float32)
    noise = np.asarray(noise_variance, dtype=np.float32).reshape(-1)
    _require(proj.shape == proj_abs2.shape == data.shape == weight.shape, "norm operand shapes differ")
    _require(proj.shape[1] == noise.size, "noise/pixel dimensions differ")
    has_mass = weight != 0.0
    raw_weight = np.where(has_mass, weight * noise[None, :], np.float32(0.0)).astype(np.float32)
    a2 = np.where(has_mass, proj_abs2 * raw_weight, np.float32(0.0)).astype(np.float32)
    cross = np.where(data != 0.0, proj * np.conj(data), np.complex64(0.0)).astype(np.complex64)
    xa = (noise[None, :] * cross.real).astype(np.float32)
    return {
        "ctf_has_mass": has_mass,
        "ctf_probs_raw": raw_weight,
        "a2": a2,
        "cross": cross,
        "xa": xa,
    }


def _float64_scalar_summary(terms: dict[str, np.ndarray]) -> dict[str, float]:
    a2 = float(np.sum(terms["a2"], dtype=np.float64))
    xa = float(np.sum(terms["xa"], dtype=np.float64))
    return {"a2": a2, "xa": xa, "residual_a2_minus_2xa": a2 - 2.0 * xa}


def _load_native_norm_panel(path: Path, original_index: int) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as panel:
        _require(panel["schema"].item() == "relion-k1-wavg-direct-norm-v1", "native norm schema changed")
        rows = np.flatnonzero(np.asarray(panel["input_row"], dtype=np.int64) == original_index)
        _require(rows.size == 1, "native norm panel does not contain exactly one target row")
        row = int(rows[0])
        return {
            "direct_current_size": float(panel["direct_current_size"][row]),
            "powerclass_high_shell": float(panel["powerclass_high_shell"][row]),
            "total": float(panel["total"][row]),
        }


def analyze(
    native_path: Path,
    recovar_path: Path,
    *,
    physical_image_size: int,
    native_norm_panel: Path | None = None,
) -> dict[str, object]:
    native = load_artifact(native_path)
    with np.load(recovar_path, allow_pickle=False) as capture:
        _require(capture["schema"].item() == "recovar-k1-norm-residual-inputs-v3", "RECOVAR schema changed")
        recovar = {name: np.asarray(capture[name]) for name in capture.files}
    original_index = int(recovar["original_index"])
    _require(native.stack_index == original_index + 1, "stack identity changed")
    _require(int(native.header[5]) == int(recovar["iteration"]), "iteration identity changed")
    current_size = int(recovar["current_size"])
    _require(
        int(native.header[12]) == current_size // 2 + 1
        and int(native.header[13]) == current_size
        and int(native.header[14]) == 1,
        "native/RECOVAR current-size layouts differ",
    )
    native_data, native_weight, identity = _dense_native_operands(
        rows=native.rows,
        native_rotations=native.rotations,
        recovar_rotations=recovar["rotations_for_noise"],
        recon_window_indices=recovar["recon_window_indices"],
        physical_image_size=physical_image_size,
    )
    recovar_data = np.asarray(recovar["summed_masked_noise"], dtype=np.complex64)
    recovar_weight = np.asarray(recovar["ctf_probs"], dtype=np.float32)
    _require(native_data.shape == recovar_data.shape, "aligned native/RECOVAR data shapes differ")
    recovar_terms = _norm_terms(
        recovar["proj_for_noise"],
        recovar["proj_abs2_for_noise"],
        recovar_data,
        recovar_weight,
        recovar["noise_variance_for_noise"],
    )
    native_operand_terms = _norm_terms(
        recovar["proj_for_noise"],
        recovar["proj_abs2_for_noise"],
        native_data,
        native_weight,
        recovar["noise_variance_for_noise"],
    )
    recovar_scalar = _float64_scalar_summary(recovar_terms)
    native_operand_scalar = _float64_scalar_summary(native_operand_terms)
    captured_residual = float(recovar["block_norm_residual"])
    weighted_image = float(recovar["weighted_img_per_image"])
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": {
            "native_capture": str(Path(native_path).resolve()),
            "recovar_capture": str(Path(recovar_path).resolve()),
            "original_index_zero_based": original_index,
            "stack_index_one_based": native.stack_index,
            "iteration": int(recovar["iteration"]),
            "half": int(recovar["half"]),
            "current_size": current_size,
            "physical_image_size": physical_image_size,
        },
        "identity": identity,
        "support": {
            "native_positive_count": int(np.count_nonzero(native_weight)),
            "recovar_positive_count": int(np.count_nonzero(recovar_weight)),
            "mask_exact": bool(np.array_equal(native_weight != 0.0, recovar_weight != 0.0)),
        },
        "bpref_operands": {
            "summed_data_recovar_vs_native": _metric(recovar_data, native_data),
            "ctf_probability_recovar_vs_native": _metric(recovar_weight, native_weight),
        },
        "recovar_replay_closure": {
            "ctf_has_mass": _metric(recovar["norm_ctf_has_mass"], recovar_terms["ctf_has_mass"]),
            "ctf_probs_raw": _metric(recovar["norm_ctf_probs_raw"], recovar_terms["ctf_probs_raw"]),
            "a2_terms": _metric(recovar["norm_a2_terms"], recovar_terms["a2"]),
            "cross_terms": _metric(recovar["norm_cross_terms"], recovar_terms["cross"]),
            "xa_terms": _metric(recovar["norm_xa_terms"], recovar_terms["xa"]),
            "captured_a2_per_image": float(recovar["norm_a2_per_image"]),
            "captured_xa_per_image": float(recovar["norm_xa_per_image"]),
            "captured_residual": captured_residual,
            "host_float64": recovar_scalar,
        },
        "native_bpref_substitution": {
            "host_float64": native_operand_scalar,
            "recovar_weighted_image_held_fixed": weighted_image,
            "counterfactual_total": weighted_image
            + native_operand_scalar["residual_a2_minus_2xa"],
            "recovar_total": weighted_image + captured_residual,
        },
    }
    if native_norm_panel is not None:
        native_norm = _load_native_norm_panel(native_norm_panel, original_index)
        recovar_total = weighted_image + captured_residual
        counterfactual_total = weighted_image + native_operand_scalar["residual_a2_minus_2xa"]
        original_gap = abs(recovar_total - native_norm["total"])
        counterfactual_gap = abs(counterfactual_total - native_norm["total"])
        report["native_norm_target"] = {
            **native_norm,
            "recovar_total_abs_error": original_gap,
            "native_bpref_substitution_abs_error": counterfactual_gap,
            "absolute_gap_closure_fraction": (
                (original_gap - counterfactual_gap) / original_gap if original_gap else 0.0
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bpref", type=Path, required=True)
    parser.add_argument("--recovar-norm", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--native-norm-panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.native_bpref,
        args.recovar_norm,
        physical_image_size=args.physical_image_size,
        native_norm_panel=args.native_norm_panel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
