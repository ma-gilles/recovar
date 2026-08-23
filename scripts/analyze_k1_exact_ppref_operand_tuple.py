#!/usr/bin/env python3
"""Attribute one K=1 fine tuple after substituting exact RELION PPref."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
)
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref
from scripts.analyze_k1_fine_operand_tuple import (
    _largest_mismatches,
    _metric,
    _translation_alignment,
    _zero_dc_compact_score_weight,
)
from scripts.validate_relion_fine_operand_capture import (
    _cuda_fine_production_lanes,
    _reduce_lanes,
    load_fine_operand_capture,
    validate_capture,
)
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_ulp_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Summarize component-wise float32 ULP distances, including complex arrays."""
    left = np.ascontiguousarray(reference).view(np.float32).reshape(-1)
    right = np.ascontiguousarray(candidate).view(np.float32).reshape(-1)
    _require(left.shape == right.shape, "ULP topology mismatch")
    finite = np.isfinite(left) & np.isfinite(right)

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = values.view(np.uint32)
        return np.where(
            (bits & np.uint32(0x80000000)) != 0,
            ~bits,
            bits | np.uint32(0x80000000),
        ).astype(np.uint32)

    distance = np.abs(
        ordered(left[finite]).astype(np.int64) - ordered(right[finite]).astype(np.int64)
    )
    return {
        "component_count": int(left.size),
        "finite_component_count": int(np.count_nonzero(finite)),
        "exact_component_count": int(np.count_nonzero(distance == 0)),
        "one_ulp_component_count": int(np.count_nonzero(distance == 1)),
        "two_ulp_component_count": int(np.count_nonzero(distance == 2)),
        "more_than_two_ulp_component_count": int(np.count_nonzero(distance > 2)),
        "median_ulp": float(np.median(distance)) if distance.size else None,
        "p95_ulp": float(np.percentile(distance, 95)) if distance.size else None,
        "max_ulp": int(np.max(distance)) if distance.size else None,
    }


def _production_raw_diff2(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.float32,
) -> np.float32:
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float32)
    lanes = _cuda_fine_production_lanes(diff_real, diff_imag, corr)
    return np.float32(_reduce_lanes(lanes) + np.float32(sum_init))


def _relion_current_image_shells(current_size: int) -> np.ndarray:
    """Return RELION's rounded shell index for a flattened FFTW half image."""
    half_width = current_size // 2 + 1
    flat = np.arange(current_size * half_width, dtype=np.int64)
    y = flat // half_width
    x = flat % half_width
    y = np.where(y <= current_size // 2, y, y - current_size)
    return np.floor(np.sqrt(x * x + y * y) + 0.5).astype(np.int32)


def analyze(
    *,
    ppref_path: Path,
    capture_path: Path,
    recovar_path: Path,
    recovar_rotation_row: int,
    physical_image_size: int,
    fine_score_path: Path | None = None,
) -> dict[str, Any]:
    _require(jax.default_backend() == "gpu", "exact PPref tuple replay requires a GPU")
    ppref, ppref_metadata = _load_ppref(ppref_path)
    capture = load_fine_operand_capture(capture_path)
    capture_validation = validate_capture(capture)
    _require(capture.candidates.size == 1, "capture must contain one candidate")
    candidate = capture.candidates[0]
    pixels = capture.pixels.reshape(capture.candidates.size, capture.image_size)[0]
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "fine_translations",
        "rotations",
        "candidate_mask",
        "shifted_corrected",
        "ctf2_over_nv_score",
        "half_weights",
        "window_indices",
        "proj_half",
        "relion_highres_xi2_half",
    }
    _require(required <= set(recovar), f"RECOVAR dump misses {sorted(required - set(recovar))}")
    current_size = int(np.asarray(recovar["current_size"]).item())
    _require(current_size == ppref_metadata["current_size"], "current-size mismatch")
    _require(capture.image_size == current_size * (current_size // 2 + 1), "capture geometry mismatch")

    rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    recovar_rotation_row_global = int(recovar_rotation_row)
    if "rotation_rows_global" in recovar:
        global_rows = np.asarray(recovar["rotation_rows_global"], dtype=np.int64)
        matches = np.flatnonzero(global_rows == recovar_rotation_row_global)
        _require(matches.size == 1, "requested global rotation row is absent from compact RECOVAR dump")
        recovar_rotation_row = int(matches[0])
    _require(0 <= recovar_rotation_row < rotations.shape[0], "rotation row is outside RECOVAR table")
    native_matrix = np.asarray(candidate["matrix"], dtype=np.float32).reshape(3, 3)
    recovar_matrix = rotations[recovar_rotation_row]
    direct_matrix_error = float(np.max(np.abs(native_matrix - recovar_matrix)))
    transpose_matrix_error = float(np.max(np.abs(native_matrix.T - recovar_matrix)))
    _require(min(direct_matrix_error, transpose_matrix_error) == 0.0, "rotation matrix differs")
    translation_row, translation_error = _translation_alignment(
        candidate["translation"], recovar["fine_translations"], physical_image_size
    )
    _require(translation_error <= 1.0e-6, "translation differs")
    _require(bool(recovar["candidate_mask"][recovar_rotation_row, translation_row]), "tuple is masked")

    window_indices = np.asarray(recovar["window_indices"], dtype=np.int32)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        (physical_image_size, physical_image_size), current_size, window_indices
    )
    supported_full = np.flatnonzero(full_to_compact >= 0)
    supported_compact = full_to_compact[supported_full]
    exact_projection, _ = compute_relion_projector_projections_block(
        jnp.asarray(ppref),
        jnp.asarray(recovar_matrix[None, ...]),
        (physical_image_size, physical_image_size),
        r_max=ppref_metadata["r_max"],
        padding_factor=2,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(window_indices),
        relion_texture_interp=True,
    )
    exact_projection = np.asarray(jax.block_until_ready(exact_projection[0]), dtype=np.complex64)
    recovar_projection = np.asarray(recovar["proj_half"][recovar_rotation_row], dtype=np.complex64)

    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    native_reference = (
        pixels["reference_real"].astype(np.float32)
        + 1j * pixels["reference_imag"].astype(np.float32)
    ).astype(np.complex64)
    native_shifted = (
        pixels["shifted_real"].astype(np.float32)
        + 1j * pixels["shifted_imag"].astype(np.float32)
    ).astype(np.complex64)
    native_corr = np.asarray(pixels["corr"], dtype=np.float32)
    native_sum = np.float32(candidate["sum_init"])

    exact_reference = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_reference = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_shifted = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_corr = np.zeros(capture.image_size, dtype=np.float32)
    exact_reference[supported_full] = -exact_projection[supported_compact] / n2
    recovar_reference[supported_full] = -recovar_projection[supported_compact] / n2
    recovar_shifted[supported_full] = -np.asarray(
        recovar["shifted_corrected"][translation_row, supported_compact], dtype=np.complex64
    ) / n2
    compact_weight = np.multiply(
        np.asarray(recovar["ctf2_over_nv_score"], dtype=np.float32),
        np.asarray(recovar["half_weights"], dtype=np.float32),
        dtype=np.float32,
    )
    compact_weight, _ = _zero_dc_compact_score_weight(
        compact_weight, window_indices, (physical_image_size, physical_image_size)
    )
    recovar_corr[supported_full] = compact_weight[supported_compact] * n4
    recovar_sum = np.float32(np.asarray(recovar["relion_highres_xi2_half"]).item())

    shell_indices = _relion_current_image_shells(current_size)
    low_shell = (shell_indices >= 1) & (shell_indices <= 4)
    recovar_corr_with_native_low = recovar_corr.copy()
    recovar_corr_with_native_low[low_shell] = native_corr[low_shell]
    recovar_corr_with_native_high = recovar_corr.copy()
    recovar_corr_with_native_high[~low_shell] = native_corr[~low_shell]
    native_corr_with_recovar_low = native_corr.copy()
    native_corr_with_recovar_low[low_shell] = recovar_corr[low_shell]
    native_corr_with_recovar_high = native_corr.copy()
    native_corr_with_recovar_high[~low_shell] = recovar_corr[~low_shell]

    operand_sets = {
        "native": (native_reference, native_shifted, native_corr, native_sum),
        "exact_ppref_with_native_other_operands": (
            exact_reference,
            native_shifted,
            native_corr,
            native_sum,
        ),
        "exact_ppref_with_recovar_other_operands": (
            exact_reference,
            recovar_shifted,
            recovar_corr,
            recovar_sum,
        ),
        "recovar": (recovar_reference, recovar_shifted, recovar_corr, recovar_sum),
        "native_with_recovar_reference_only": (
            recovar_reference,
            native_shifted,
            native_corr,
            native_sum,
        ),
        "native_with_recovar_shifted_only": (
            native_reference,
            recovar_shifted,
            native_corr,
            native_sum,
        ),
        "native_with_recovar_corr_only": (
            native_reference,
            native_shifted,
            recovar_corr,
            native_sum,
        ),
        "native_with_recovar_sum_only": (
            native_reference,
            native_shifted,
            native_corr,
            recovar_sum,
        ),
        "recovar_with_exact_ppref_only": (
            exact_reference,
            recovar_shifted,
            recovar_corr,
            recovar_sum,
        ),
        "recovar_with_exact_ppref_and_native_corr": (
            exact_reference,
            recovar_shifted,
            native_corr,
            recovar_sum,
        ),
        "recovar_with_exact_ppref_and_native_corr_shells_1_through_4": (
            exact_reference,
            recovar_shifted,
            recovar_corr_with_native_low,
            recovar_sum,
        ),
        "recovar_with_exact_ppref_and_native_corr_shells_5_plus": (
            exact_reference,
            recovar_shifted,
            recovar_corr_with_native_high,
            recovar_sum,
        ),
        "recovar_with_native_shifted_only": (
            recovar_reference,
            native_shifted,
            recovar_corr,
            recovar_sum,
        ),
        "recovar_with_native_corr_only": (
            recovar_reference,
            recovar_shifted,
            native_corr,
            recovar_sum,
        ),
        "recovar_with_native_corr_shells_1_through_4": (
            recovar_reference,
            recovar_shifted,
            recovar_corr_with_native_low,
            recovar_sum,
        ),
        "recovar_with_native_corr_shells_5_plus": (
            recovar_reference,
            recovar_shifted,
            recovar_corr_with_native_high,
            recovar_sum,
        ),
        "native_with_recovar_corr_shells_1_through_4": (
            native_reference,
            native_shifted,
            native_corr_with_recovar_low,
            native_sum,
        ),
        "native_with_recovar_corr_shells_5_plus": (
            native_reference,
            native_shifted,
            native_corr_with_recovar_high,
            native_sum,
        ),
        "recovar_with_native_sum_only": (
            recovar_reference,
            recovar_shifted,
            recovar_corr,
            native_sum,
        ),
    }
    raw = {name: float(_production_raw_diff2(*operands)) for name, operands in operand_sets.items()}
    native_production_raw = float(np.float32(candidate["production_raw_diff2"]))
    _require(raw["native"] == native_production_raw, "native host replay differs from production")

    stage_arrays = {
        "exact_ppref_projected_reference": (
            native_reference[supported_full],
            exact_reference[supported_full],
        ),
        "recovar_projected_reference": (
            native_reference[supported_full],
            recovar_reference[supported_full],
        ),
        "translated_image": (
            native_shifted[supported_full],
            recovar_shifted[supported_full],
        ),
        "correction_weight": (
            native_corr[supported_full],
            recovar_corr[supported_full],
        ),
        "highres_sum": (np.asarray([native_sum]), np.asarray([recovar_sum])),
    }
    stage_metrics = {name: _metric(left, right) for name, (left, right) in stage_arrays.items()}
    first_unequal = next(
        (name for name, metric in stage_metrics.items() if not metric["exact_equal"]),
        None,
    )
    posterior_boundary = None
    if fine_score_path is not None:
        fine_score = load_fine_score_capture(fine_score_path)
        _require(fine_score.stack_index == capture.stack_index, "fine-score particle identity differs")
        native_rows = np.flatnonzero(
            (fine_score.candidates["rotation_local"] == candidate["rotation_local"])
            & (fine_score.candidates["translation_id"] == candidate["translation_id"])
        )
        _require(native_rows.size == 1, "native fine tuple is absent or duplicated")
        native_row = fine_score.candidates[int(native_rows[0])]
        _require(bool(native_row["flags"] & ACTIVE), "native fine tuple is inactive")
        _require(
            float(np.float32(native_row["raw_diff2"])) == native_production_raw,
            "fine-score and operand captures disagree on native raw diff2",
        )
        active = (fine_score.candidates["flags"] & ACTIVE) != 0
        native_weights = np.asarray(
            fine_score.candidates["post_exponent_weight"][active], dtype=np.float64
        )
        native_weight_sum = float(np.sum(native_weights, dtype=np.float64))
        _require(native_weight_sum > 0.0 and np.isfinite(native_weight_sum), "invalid native posterior sum")
        native_probability = float(np.float64(native_row["post_exponent_weight"]) / native_weight_sum)
        recovar_probability = float(
            np.asarray(recovar["probs"], dtype=np.float64)[
                recovar_rotation_row, translation_row
            ]
        )
        native_values = {
            "raw_diff2": float(np.float32(native_row["raw_diff2"])),
            "orientation_log_prior": float(np.float32(native_row["orientation_log_prior"])),
            "translation_log_prior": float(np.float32(native_row["translation_log_prior"])),
            "combined_preexponent": float(np.float32(native_row["combined_preexponent"])),
            "normalized_posterior": native_probability,
        }
        recovar_values = {
            "raw_diff2": raw["recovar"],
            "orientation_log_prior": float(
                np.asarray(recovar["rotation_log_prior"], dtype=np.float64)[recovar_rotation_row]
            ),
            "translation_log_prior": float(
                np.asarray(recovar["translation_log_prior"], dtype=np.float64)[translation_row]
            ),
            "combined_preexponent": float(
                np.asarray(recovar["scores_with_prior"], dtype=np.float64)[recovar_rotation_row, translation_row]
            ),
            "normalized_posterior": recovar_probability,
        }
        ordered_boundaries = (
            "raw_diff2",
            "orientation_log_prior",
            "translation_log_prior",
            "combined_preexponent",
            "normalized_posterior",
        )
        comparisons = {
            name: _metric(
                np.asarray([native_values[name]], dtype=np.float64),
                np.asarray([recovar_values[name]], dtype=np.float64),
            )
            for name in ordered_boundaries
        }
        posterior_boundary = {
            "first_exact_unequal_boundary": next(
                (name for name in ordered_boundaries if not comparisons[name]["exact_equal"]),
                None,
            ),
            "native": native_values,
            "recovar": recovar_values,
            "comparisons": comparisons,
            "native_active_candidate_count": int(np.count_nonzero(active)),
            "native_active_weight_sum": native_weight_sum,
            "fine_score_path": str(fine_score_path.resolve()),
            "fine_score_sha256": _sha256(fine_score_path),
        }
    return {
        "schema": "recovar.em.k1_exact_ppref_operand_tuple.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": capture.stack_index,
            "original_index_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "native_rotation_local": int(candidate["rotation_local"]),
            "recovar_rotation_row": recovar_rotation_row_global,
            "recovar_rotation_row_in_dump": recovar_rotation_row,
            "native_translation": int(candidate["translation_id"]),
            "recovar_translation_row": translation_row,
        },
        "alignment": {
            "rotation_transform": "identity" if direct_matrix_error <= transpose_matrix_error else "transpose",
            "rotation_max_abs": min(direct_matrix_error, transpose_matrix_error),
            "translation_max_abs": translation_error,
            "supported_pixel_count": int(supported_full.size),
        },
        "first_exact_unequal_boundary": first_unequal,
        "stage_metrics": stage_metrics,
        "stage_float32_ulp_metrics": {
            name: _float32_ulp_stats(left, right)
            for name, (left, right) in stage_arrays.items()
        },
        "largest_pixel_mismatches": {
            name: _largest_mismatches(left, right)
            for name, (left, right) in stage_arrays.items()
            if np.asarray(left).size > 1
        },
        "raw_scores": {
            "native_production": native_production_raw,
            "replays": raw,
            "exact_ppref_rescues_native_raw": raw["exact_ppref_with_native_other_operands"]
            == native_production_raw,
            "exact_ppref_with_recovar_other_operands_rescues_native_raw": raw[
                "exact_ppref_with_recovar_other_operands"
            ]
            == native_production_raw,
            "exact_ppref_and_native_corr_rescue_recovar_raw": raw[
                "recovar_with_exact_ppref_and_native_corr"
            ]
            == native_production_raw,
            "exact_ppref_and_native_corr_shells_1_through_4_rescue_recovar_raw": raw[
                "recovar_with_exact_ppref_and_native_corr_shells_1_through_4"
            ]
            == native_production_raw,
            "exact_ppref_and_native_corr_shells_5_plus_rescue_recovar_raw": raw[
                "recovar_with_exact_ppref_and_native_corr_shells_5_plus"
            ]
            == native_production_raw,
            "native_corr_shells_1_through_4_rescue_recovar_raw": raw[
                "recovar_with_native_corr_shells_1_through_4"
            ]
            == native_production_raw,
            "native_corr_shells_5_plus_rescue_recovar_raw": raw[
                "recovar_with_native_corr_shells_5_plus"
            ]
            == native_production_raw,
        },
        "posterior_boundary": posterior_boundary,
        "ppref": ppref_metadata,
        "native_capture_validation": capture_validation,
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "native_capture": str(capture_path.resolve()),
            "native_capture_sha256": _sha256(capture_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--recovar-rotation-row", type=int, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--fine-score", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        ppref_path=args.ppref,
        capture_path=args.capture,
        recovar_path=args.recovar,
        recovar_rotation_row=args.recovar_rotation_row,
        physical_image_size=args.physical_image_size,
        fine_score_path=args.fine_score,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
