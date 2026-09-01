#!/usr/bin/env python3
"""Compare one fresh RECOVAR firstiter-CC operand panel with native RELION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_native_cc_translation_tie import _pass_field, _sha256
from scripts.analyze_k1_native_fine_operand_boundary import (
    _center,
    _complex_metric,
    _metric,
    _native_to_recovar_compact,
    _relion_tree_sum,
)
from scripts.analyze_k1_pose_winner_map_counterfactual import _rotation_distances_deg
from scripts.parse_relion_dump_dir import parse_dump_dir


SCHEMA = "recovar.em.k1_firstiter_cc_native_operands.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_cc_components(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.complex64)
    shifted = np.asarray(shifted, dtype=np.complex64)
    corr_img = np.asarray(corr_img, dtype=np.float32)
    _require(reference.shape == shifted.shape, "reference and shifted shapes differ")
    _require(
        reference.ndim == 2 and corr_img.shape == (reference.shape[1],),
        "invalid normalized-CC operand shapes",
    )
    numerator_terms = np.multiply(
        np.add(
            np.multiply(reference.real, shifted.real, dtype=np.float32),
            np.multiply(reference.imag, shifted.imag, dtype=np.float32),
            dtype=np.float32,
        ),
        corr_img[None, :],
        dtype=np.float32,
    )
    norm_terms = np.multiply(
        np.add(
            np.multiply(reference.real, reference.real, dtype=np.float32),
            np.multiply(reference.imag, reference.imag, dtype=np.float32),
            dtype=np.float32,
        ),
        corr_img[None, :],
        dtype=np.float32,
    )
    numerator_lanes, numerator = _relion_tree_sum(numerator_terms)
    norm_lanes, norm = _relion_tree_sum(norm_terms)
    score = np.divide(
        numerator,
        np.sqrt(np.maximum(norm, np.float32(1e-30)), dtype=np.float32),
        dtype=np.float32,
    )
    return numerator_terms, norm_terms, numerator_lanes, norm_lanes, score


def _rotation_map(native: np.ndarray, recovar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    native = np.asarray(native, dtype=np.float64).reshape(-1, 3, 3)
    recovar = np.asarray(recovar, dtype=np.float64).reshape(-1, 3, 3)
    mapped = []
    errors = []
    for rotation in native:
        distances = _rotation_distances_deg(recovar, rotation)
        row = int(np.argmin(distances))
        mapped.append(row)
        errors.append(float(distances[row]))
    return np.asarray(mapped, dtype=np.int64), np.asarray(errors, dtype=np.float64)


def analyze(
    *,
    relion_dump_dir: Path,
    recovar_capture: Path,
    physical_image_size: int,
    rotation_gate_deg: float,
) -> dict[str, Any]:
    payload = parse_dump_dir(relion_dump_dir)
    native_score = -_pass_field(
        payload, "firstiter_cc_exp_Mweight_raw_preonehot"
    ).reshape(-1).astype(np.float32)
    native_rotation_rows = _pass_field(payload, "firstiter_cc_raw_rot_idx").reshape(-1)
    native_translations = _pass_field(payload, "firstiter_cc_raw_trans_idx").reshape(-1)
    candidate_count = int(native_score.size)
    _require(
        native_rotation_rows.shape == native_translations.shape == (candidate_count,),
        "native candidate identities are misaligned",
    )
    native_corr = _pass_field(payload, "corr_img").reshape(-1).astype(np.float32)
    pixel_count = int(native_corr.size)

    def native_complex(stem: str) -> np.ndarray:
        real = _pass_field(payload, f"{stem}_real").reshape(candidate_count, pixel_count)
        imag = _pass_field(payload, f"{stem}_imag").reshape(candidate_count, pixel_count)
        return (real + np.complex64(1j) * imag).astype(np.complex64)

    native_reference = native_complex("fine_ref")
    native_shifted = native_complex("fine_shifted")
    # RELION's CUDA-side Euler dump is the transpose of RECOVAR's projector
    # matrix convention.
    native_rotations = np.transpose(
        _pass_field(payload, "fine_eulers").reshape(-1, 3, 3),
        (0, 2, 1),
    )
    _require(
        np.max(native_rotation_rows, initial=-1) < native_rotations.shape[0],
        "native compact rotation lies outside the matrix panel",
    )

    with np.load(recovar_capture, allow_pickle=False) as archive:
        _require("raw_operand_schema" in archive, "RECOVAR capture lacks raw operands")
        recovar_rotations = np.asarray(archive["rotations"], dtype=np.float32)
        recovar_translations = np.asarray(archive["fine_translations"], dtype=np.float32)
        recovar_reference_compact = np.asarray(
            archive["raw_operand_proj_half"], dtype=np.complex64
        )
        recovar_shifted_compact = np.asarray(
            archive["raw_operand_shifted_corrected"], dtype=np.complex64
        )
        recovar_corr_compact = np.asarray(
            archive["raw_operand_corr_img_score"], dtype=np.float32
        )
        recovar_score_dense = np.asarray(
            archive["raw_operand_raw_diff2"], dtype=np.float32
        )
        full_to_compact = np.asarray(
            archive["raw_operand_relion_full_to_compact"], dtype=np.int64
        )
        candidate_mask = np.asarray(archive["candidate_mask"], dtype=bool)
        source_row = int(archive["original_index"])

    rotation_map, rotation_errors = _rotation_map(native_rotations, recovar_rotations)
    _require(
        float(np.max(rotation_errors, initial=0.0)) <= rotation_gate_deg,
        f"rotation mapping exceeds {rotation_gate_deg} degrees",
    )
    mapped_rotation = rotation_map[np.asarray(native_rotation_rows, dtype=np.int64)]
    mapped_translation = np.asarray(native_translations, dtype=np.int64)
    _require(
        np.all((mapped_translation >= 0) & (mapped_translation < recovar_translations.shape[0])),
        "native translation lies outside RECOVAR's fine grid",
    )
    _require(
        np.all(candidate_mask[mapped_rotation, mapped_translation]),
        "native candidate panel is not active in RECOVAR",
    )
    recovar_score = recovar_score_dense[mapped_rotation, mapped_translation]

    native_to_compact = _native_to_recovar_compact(
        native_image_size=pixel_count,
        recovar_full_to_compact=full_to_compact,
    )
    valid = native_to_compact >= 0
    compact_rows = native_to_compact[valid]
    recovar_reference = np.zeros_like(native_reference)
    recovar_shifted = np.zeros_like(native_shifted)
    recovar_corr = np.zeros_like(native_corr)
    recovar_reference[:, valid] = recovar_reference_compact[mapped_rotation][:, compact_rows]
    recovar_shifted[:, valid] = recovar_shifted_compact[mapped_translation][:, compact_rows]
    recovar_corr[valid] = recovar_corr_compact[compact_rows]

    fft_scale = np.float32(physical_image_size * physical_image_size)
    recovar_reference = np.negative(
        np.divide(recovar_reference, fft_scale, dtype=np.complex64),
        dtype=np.complex64,
    )
    recovar_shifted = np.negative(
        np.divide(recovar_shifted, fft_scale, dtype=np.complex64),
        dtype=np.complex64,
    )
    recovar_corr = np.multiply(
        recovar_corr,
        np.multiply(fft_scale, fft_scale, dtype=np.float32),
        dtype=np.float32,
    )

    native_components = _normalized_cc_components(
        native_reference, native_shifted, native_corr
    )
    recovar_components = _normalized_cc_components(
        recovar_reference, recovar_shifted, recovar_corr
    )
    component_names = (
        "numerator_pixel_terms",
        "norm_pixel_terms",
        "numerator_pre_tree_lanes",
        "norm_pre_tree_lanes",
        "source_order_score",
    )
    boundaries = {
        "corr_img_valid_pixels": _metric(native_corr[valid], recovar_corr[valid]),
        "projected_reference_valid_pixels": _complex_metric(
            native_reference[:, valid], recovar_reference[:, valid]
        ),
        "shifted_image_valid_pixels": _complex_metric(
            native_shifted[:, valid], recovar_shifted[:, valid]
        ),
        **{
            name: _metric(native_value, recovar_value)
            for name, native_value, recovar_value in zip(
                component_names, native_components, recovar_components, strict=True
            )
        },
        "captured_centered_score": _metric(
            _center(native_score), _center(recovar_score)
        ),
        "native_source_replay_score": _metric(native_score, native_components[-1]),
        "recovar_source_replay_score": _metric(recovar_score, recovar_components[-1]),
    }
    ordered = (
        "corr_img_valid_pixels",
        "projected_reference_valid_pixels",
        "shifted_image_valid_pixels",
        "numerator_pixel_terms",
        "norm_pixel_terms",
        "numerator_pre_tree_lanes",
        "norm_pre_tree_lanes",
        "source_order_score",
        "captured_centered_score",
    )
    first_unequal = next(
        (
            name
            for name in ordered
            if boundaries[name].get("bit_equal_fraction") != 1.0
        ),
        None,
    )

    arms = {
        "recovar_all": (recovar_reference, recovar_shifted, recovar_corr),
        "native_corr_only": (recovar_reference, recovar_shifted, native_corr),
        "native_reference_only": (native_reference, recovar_shifted, recovar_corr),
        "native_shifted_only": (recovar_reference, native_shifted, recovar_corr),
        "native_reference_and_shifted": (
            native_reference,
            native_shifted,
            recovar_corr,
        ),
        "native_all": (native_reference, native_shifted, native_corr),
    }
    counterfactuals = {
        name: _metric(
            _center(native_score),
            _center(_normalized_cc_components(*operands)[-1]),
        )
        for name, operands in arms.items()
    }
    native_winner = int(np.argmax(native_score))
    recovar_winner = int(np.argmax(recovar_score))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "exact bytes and relative L2; no correlation",
        "source_row_zero_based": source_row,
        "candidate_count": candidate_count,
        "native_pixel_count": pixel_count,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "rotation_map_max_error_deg": float(np.max(rotation_errors, initial=0.0)),
        "first_non_bit_exact_boundary": first_unequal,
        "ordered_boundaries": list(ordered),
        "winner": {
            "native_candidate_row": native_winner,
            "native_rotation_row": int(native_rotation_rows[native_winner]),
            "native_translation": int(native_translations[native_winner]),
            "recovar_candidate_row_within_native_panel": recovar_winner,
            "recovar_rotation_row": int(mapped_rotation[recovar_winner]),
            "recovar_translation": int(mapped_translation[recovar_winner]),
        },
        "boundaries": boundaries,
        "counterfactual_centered_score_metrics": counterfactuals,
        "artifacts": {
            "relion_dump_dir": str(relion_dump_dir.resolve()),
            "relion_dump_sha256": {
                path.name: _sha256(path) for path in sorted(relion_dump_dir.glob("*.bin"))
            },
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-capture", required=True, type=Path)
    parser.add_argument("--physical-image-size", required=True, type=int)
    parser.add_argument("--rotation-gate-deg", type=float, default=1e-3)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        relion_dump_dir=args.relion_dump_dir,
        recovar_capture=args.recovar_capture,
        physical_image_size=args.physical_image_size,
        rotation_gate_deg=args.rotation_gate_deg,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
