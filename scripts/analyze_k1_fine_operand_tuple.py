#!/usr/bin/env python3
"""Compare one matched K=1 fine-score tuple at the pixel/reduction boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_k4_relion_recovar_fine_operands import (
    _infer_current_size,
    _metric,
    _translation_alignment,
    _zero_dc_compact_score_weight,
)
from scripts.validate_relion_fine_operand_capture import (
    _cuda_fine_contribution,
    _cuda_fine_production_lanes,
    _replay_lanes,
    _reduce_lanes,
    load_fine_operand_capture,
    validate_capture,
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


def _sass_tree_raw_diff2(
    reference: np.ndarray,
    shifted: np.ndarray,
    correction: np.ndarray,
    sum_init: np.float32,
) -> tuple[np.float32, np.ndarray, np.ndarray]:
    """Replay the demonstrated native SM80/SM90 fine-score arithmetic."""

    reference = np.asarray(reference, dtype=np.complex64)
    shifted = np.asarray(shifted, dtype=np.complex64)
    correction = np.asarray(correction, dtype=np.float32)
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float32)
    contribution = _cuda_fine_contribution(diff_real, diff_imag, correction)
    lanes = _cuda_fine_production_lanes(diff_real, diff_imag, correction)
    raw_diff2 = np.float32(_reduce_lanes(lanes) + np.float32(sum_init))
    return raw_diff2, contribution, lanes


def _largest_mismatches(
    relion: np.ndarray,
    recovar: np.ndarray,
    *,
    flat_indices: np.ndarray | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    left = np.asarray(relion)
    right = np.asarray(recovar)
    _require(left.shape == right.shape, "operand shapes differ")
    if flat_indices is None:
        selected_indices = np.arange(left.size, dtype=np.int64)
    else:
        selected_indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        _require(
            np.all((selected_indices >= 0) & (selected_indices < left.size)),
            "mismatch indices are outside the flattened operand",
        )
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    delta = np.abs(
        right_flat[selected_indices].astype(
            np.complex128 if np.iscomplexobj(right) else np.float64
        )
        - left_flat[selected_indices].astype(
            np.complex128 if np.iscomplexobj(left) else np.float64
        )
    )
    order = np.argsort(delta, kind="stable")[::-1][:limit]
    return [
        {
            "flat_index": int(selected_indices[index]),
            "abs_delta": float(delta[index]),
            "relion": (
                [
                    float(left_flat[selected_indices[index]].real),
                    float(left_flat[selected_indices[index]].imag),
                ]
                if np.iscomplexobj(left)
                else float(left_flat[selected_indices[index]])
            ),
            "recovar": (
                [
                    float(right_flat[selected_indices[index]].real),
                    float(right_flat[selected_indices[index]].imag),
                ]
                if np.iscomplexobj(right)
                else float(right_flat[selected_indices[index]])
            ),
        }
        for index in order
        if delta[index] != 0
    ]


def _score_window_rows_from_relion_full(
    *,
    supported_full: np.ndarray,
    window_indices: np.ndarray,
    image_shape: tuple[int, int],
    current_size: int,
) -> np.ndarray:
    """Map native current-size FFT rows to RECOVAR score-window rows.

    Projection operands use the compact lookup stored in
    ``raw_operand_relion_full_to_compact``.  Shifted-image and correction
    operands instead use ``window_indices`` order.  Those coordinate systems
    contain the same Fourier pixels but are not the same gather.  In
    particular, the even-size y-Nyquist row is represented as ``+N/2`` by the
    score window and ``-N/2`` by the projector lookup.
    """

    supported_full = np.asarray(supported_full, dtype=np.int64).reshape(-1)
    window_indices = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    current_half_width = int(current_size) // 2 + 1
    fftw_rows = supported_full // current_half_width
    columns = supported_full % current_half_width
    ky = np.where(
        fftw_rows <= int(current_size) // 2,
        fftw_rows,
        fftw_rows - int(current_size),
    )
    physical_half_width = int(image_shape[1]) // 2 + 1
    physical_indices = (
        (ky + int(image_shape[0]) // 2) * physical_half_width + columns
    ).astype(np.int64)
    _require(
        np.unique(window_indices).size == window_indices.size,
        "RECOVAR score window contains duplicate physical pixels",
    )
    window_order = np.argsort(window_indices, kind="stable")
    sorted_window = window_indices[window_order]
    positions = np.searchsorted(sorted_window, physical_indices)
    _require(
        np.all(positions < sorted_window.size)
        and np.array_equal(sorted_window[positions], physical_indices),
        "native score pixels and RECOVAR score window differ",
    )
    rows = window_order[positions]
    _require(
        np.unique(rows).size == rows.size,
        "native score pixels do not map one-to-one to RECOVAR score rows",
    )
    return rows.astype(np.int64, copy=False)


def analyze(
    capture_path: Path,
    pass2_path: Path,
    *,
    recovar_global_rotation: int,
    physical_image_size: int,
) -> dict[str, Any]:
    capture_path = capture_path.resolve()
    pass2_path = pass2_path.resolve()
    capture = load_fine_operand_capture(capture_path)
    validation = validate_capture(capture)
    _require(capture.candidates.size == 1, "capture must contain exactly one tuple")
    candidate = capture.candidates[0]
    pixels = capture.pixels.reshape(1, capture.image_size)[0]

    with np.load(pass2_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "fine_translations",
        "oversampled_rot_indices",
        "rotations",
        "candidate_mask",
        "relion_raw_diff2",
        "raw_operand_actual_rotation_count",
        "raw_operand_proj_half",
        "raw_operand_shifted_corrected",
        "raw_operand_corr_img_score",
        "raw_operand_half_weights",
        "raw_operand_relion_full_to_compact",
        "raw_operand_highres_xi2_half",
    }
    _require(required <= set(recovar), f"pass-2 dump misses {sorted(required - set(recovar))}")

    current_size = _infer_current_size(capture.image_size)
    _require(int(np.asarray(recovar["current_size"]).item()) == current_size, "current size differs")
    image_shape = (physical_image_size, physical_image_size)
    lookup = np.asarray(recovar["raw_operand_relion_full_to_compact"], dtype=np.int32)
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    _require("window_indices" in recovar, "pass-2 dump misses score-window indices")
    supported_score_rows = _score_window_rows_from_relion_full(
        supported_full=supported_full,
        window_indices=recovar["window_indices"],
        image_shape=image_shape,
        current_size=current_size,
    )

    rotation_rows = np.flatnonzero(
        np.asarray(recovar["oversampled_rot_indices"], dtype=np.int64)
        == int(recovar_global_rotation)
    )
    _require(rotation_rows.size == 1, "RECOVAR global fine rotation is not unique")
    rotation_row = int(rotation_rows[0])
    actual_rotation_count = int(np.asarray(recovar["raw_operand_actual_rotation_count"]).item())
    _require(rotation_row < actual_rotation_count, "target rotation is outside active raw operands")
    native_matrix = np.asarray(candidate["matrix"], dtype=np.float32).reshape(3, 3)
    recovar_matrix = np.asarray(recovar["rotations"][rotation_row], dtype=np.float32)
    direct_matrix_error = float(np.max(np.abs(native_matrix - recovar_matrix)))
    transpose_matrix_error = float(np.max(np.abs(native_matrix.T - recovar_matrix)))
    _require(min(direct_matrix_error, transpose_matrix_error) <= 1e-6, "rotation matrix differs")

    translation_row, translation_error = _translation_alignment(
        candidate["translation"],
        recovar["fine_translations"],
        physical_image_size,
    )
    _require(translation_error <= 1e-6, "translation differs")
    _require(bool(recovar["candidate_mask"][rotation_row, translation_row]), "tuple is masked")

    recovar_weight = np.multiply(
        np.asarray(recovar["raw_operand_corr_img_score"], dtype=np.float32),
        np.asarray(recovar["raw_operand_half_weights"], dtype=np.float32),
        dtype=np.float32,
    )
    recovar_weight, dc_mask = _zero_dc_compact_score_weight(
        recovar_weight,
        recovar["window_indices"],
        image_shape,
    )
    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    recovar_reference = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_shifted = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_corr = np.zeros(capture.image_size, dtype=np.float32)
    recovar_reference[supported_full] = -np.asarray(
        recovar["raw_operand_proj_half"][rotation_row, supported_compact],
        dtype=np.complex64,
    ) / n2
    recovar_shifted[supported_full] = -np.asarray(
        recovar["raw_operand_shifted_corrected"][translation_row, supported_score_rows],
        dtype=np.complex64,
    ) / n2
    recovar_corr[supported_full] = recovar_weight[supported_score_rows] * n4

    relion_reference = (
        np.asarray(pixels["reference_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["reference_imag"], dtype=np.float32)
    ).astype(np.complex64)
    relion_shifted = (
        np.asarray(pixels["shifted_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["shifted_imag"], dtype=np.float32)
    ).astype(np.complex64)
    relion_corr = np.asarray(pixels["corr"], dtype=np.float32)
    relion_contribution = np.asarray(pixels["contribution"], dtype=np.float32)
    relion_sum = np.float32(candidate["sum_init"])
    recovar_sum = np.float32(np.asarray(recovar["raw_operand_highres_xi2_half"]).item())

    recovar_raw_replay, recovar_contribution, recovar_lanes = _sass_tree_raw_diff2(
        recovar_reference,
        recovar_shifted,
        recovar_corr,
        recovar_sum,
    )
    native_raw_replay, native_contribution, native_lanes = _sass_tree_raw_diff2(
        relion_reference,
        relion_shifted,
        relion_corr,
        relion_sum,
    )
    native_production_raw = np.float32(candidate["production_raw_diff2"])
    recovar_production_raw = np.float32(
        recovar["relion_raw_diff2"][rotation_row, translation_row]
    )
    _require(native_raw_replay == native_production_raw, "native host replay differs from production")
    _require(np.array_equal(native_contribution, relion_contribution), "native contribution replay differs")
    native_captured_lanes = _replay_lanes(relion_contribution)
    recovar_captured_lanes = _replay_lanes(recovar_contribution)
    _require(
        np.array_equal(native_captured_lanes, candidate["lane_partials"]),
        "native captured lanes replay differs",
    )

    substitutions = {}
    for name, operands in {
        "native": (relion_reference, relion_shifted, relion_corr, relion_sum),
        "recovar": (recovar_reference, recovar_shifted, recovar_corr, recovar_sum),
        "native_reference_only": (relion_reference, recovar_shifted, recovar_corr, recovar_sum),
        "native_shifted_only": (recovar_reference, relion_shifted, recovar_corr, recovar_sum),
        "native_corr_only": (recovar_reference, recovar_shifted, relion_corr, recovar_sum),
        "native_highres_only": (recovar_reference, recovar_shifted, recovar_corr, relion_sum),
    }.items():
        substitutions[name] = float(_sass_tree_raw_diff2(*operands)[0])

    stage_arrays = {
        "projected_reference": (relion_reference, recovar_reference),
        "shifted_image": (relion_shifted, recovar_shifted),
        "correction_weight": (relion_corr, recovar_corr),
        "highres_sum": (np.asarray([relion_sum]), np.asarray([recovar_sum])),
        "pixel_contribution": (relion_contribution, recovar_contribution),
        "lane_partial": (candidate["lane_partials"], recovar_captured_lanes),
        "production_lane_partial": (native_lanes, recovar_lanes),
        "raw_diff2_replay": (
            np.asarray([native_raw_replay]),
            np.asarray([recovar_raw_replay]),
        ),
        "raw_diff2_production": (
            np.asarray([native_production_raw]),
            np.asarray([recovar_production_raw]),
        ),
    }
    stage_metrics = {
        name: _metric(left, right)
        for name, (left, right) in stage_arrays.items()
    }
    pixel_stage_names = (
        "projected_reference",
        "shifted_image",
        "correction_weight",
        "pixel_contribution",
    )
    score_active_stage_arrays = {
        name: (
            np.asarray(stage_arrays[name][0]).reshape(-1)[supported_full],
            np.asarray(stage_arrays[name][1]).reshape(-1)[supported_full],
        )
        for name in pixel_stage_names
    }
    score_active_stage_metrics = {
        name: _metric(left, right)
        for name, (left, right) in score_active_stage_arrays.items()
    }
    causal_stage_metrics = {
        name: score_active_stage_metrics.get(name, stage_metrics[name])
        for name in stage_arrays
    }
    first_unequal = next(
        name for name, metric in causal_stage_metrics.items() if not metric["exact_equal"]
    )
    return {
        "schema": "recovar.em.k1_fine_operand_tuple.v2",
        "identity": {
            "stack_index_one_based": capture.stack_index,
            "original_index_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "native_particle_id": capture.particle_id,
            "native_rotation_local": int(candidate["rotation_local"]),
            "recovar_global_rotation": int(recovar_global_rotation),
            "recovar_rotation_row": rotation_row,
            "native_translation": int(candidate["translation_id"]),
            "recovar_translation_row": translation_row,
        },
        "alignment": {
            "native_to_recovar_rotation_transform": (
                "identity" if direct_matrix_error <= transpose_matrix_error else "transpose"
            ),
            "rotation_max_abs": min(direct_matrix_error, transpose_matrix_error),
            "translation_max_abs": translation_error,
            "supported_pixel_count": int(supported_full.size),
            "dc_present_in_compact_support": bool(np.any(dc_mask)),
        },
        "first_exact_unequal_boundary": first_unequal,
        "first_exact_unequal_boundary_domain": (
            "score-active pixels for pixel operands; complete arrays for reductions and scalars"
        ),
        "stage_metrics": stage_metrics,
        "stage_metrics_domain": "complete RELION current-size FFT rectangle",
        "score_active_pixel_stage_metrics": score_active_stage_metrics,
        "score_active_pixel_stage_metrics_domain": (
            "RECOVAR compact support embedded in the RELION current-size FFT rectangle"
        ),
        "largest_pixel_mismatches": {
            name: _largest_mismatches(left, right)
            for name, (left, right) in stage_arrays.items()
            if np.asarray(left).size > 1
        },
        "largest_score_active_pixel_mismatches": {
            name: _largest_mismatches(
                stage_arrays[name][0],
                stage_arrays[name][1],
                flat_indices=supported_full,
            )
            for name in pixel_stage_names
        },
        "raw_scores": {
            "native_production": float(native_production_raw),
            "native_host_replay": float(native_raw_replay),
            "recovar_production": float(recovar_production_raw),
            "recovar_host_replay": float(recovar_raw_replay),
            "substitutions": substitutions,
        },
        "native_capture_validation": validation,
        "artifacts": {
            "native_capture": str(capture_path),
            "native_capture_sha256": _sha256(capture_path),
            "recovar_pass2": str(pass2_path),
            "recovar_pass2_sha256": _sha256(pass2_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--recovar-global-rotation", type=int, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.capture,
        args.pass2,
        recovar_global_rotation=args.recovar_global_rotation,
        physical_image_size=args.physical_image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
