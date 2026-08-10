#!/usr/bin/env python3
"""Locate the first per-pixel K=1 fine-score operand mismatch."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.core.ctf import _compute_spa_ctf
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _half_translation_phase_table_for_indices,
    _relion_cuda_fine_full_to_compact_lookup,
)

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_operand_capture import (
        _cuda_fine_contribution,
        _cuda_fine_production_lanes,
        _reduce_lanes,
        _replay_lanes,
        load_fine_operand_capture,
        validate_capture,
    )
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_operand_capture import (  # type: ignore[no-redef]
        _cuda_fine_contribution,
        _cuda_fine_production_lanes,
        _reduce_lanes,
        _replay_lanes,
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


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _json_value(value: Any) -> Any:
    scalar = np.asarray(value).item()
    if isinstance(scalar, complex):
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(scalar, (np.floating, float)):
        return float(scalar)
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    return scalar


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(reference)
    rhs = np.asarray(candidate)
    _require(lhs.shape == rhs.shape, f"operand shape changed: {lhs.shape} != {rhs.shape}")
    promoted_dtype = np.complex128 if np.iscomplexobj(lhs) or np.iscomplexobj(rhs) else np.float64
    delta = rhs.astype(promoted_dtype, copy=False) - lhs.astype(promoted_dtype, copy=False)
    denominator = max(float(np.linalg.norm(lhs.reshape(-1))), np.finfo(np.float64).tiny)
    unequal = np.flatnonzero((lhs != rhs).reshape(-1))
    report: dict[str, Any] = {
        "shape": list(lhs.shape),
        "reference_dtype": str(lhs.dtype),
        "candidate_dtype": str(rhs.dtype),
        "exact_equal": bool(unequal.size == 0),
        "mismatch_count": int(unequal.size),
        "relative_l2_over_reference": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
        "reference_sha256": _array_sha256(lhs),
        "candidate_sha256": _array_sha256(rhs),
    }
    if unequal.size:
        flat_index = int(unequal[0])
        report["first_mismatch"] = {
            "flat_index": flat_index,
            "index": list(np.unravel_index(flat_index, lhs.shape)),
            "reference": _json_value(lhs.reshape(-1)[flat_index]),
            "candidate": _json_value(rhs.reshape(-1)[flat_index]),
            "absolute_delta": float(abs(delta.reshape(-1)[flat_index])),
        }
    return report


def _infer_current_size(packed_size: int) -> int:
    matches = [size for size in range(2, 4097, 2) if size * (size // 2 + 1) == packed_size]
    _require(len(matches) == 1, f"cannot infer current size from packed size {packed_size}")
    return matches[0]


def _rotation_alignment(matrix: np.ndarray, rotations: np.ndarray) -> tuple[int, float]:
    native = np.asarray(matrix, dtype=np.float32).reshape(3, 3).T
    recovar = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
    error = np.max(np.abs(recovar - native[None]), axis=(1, 2))
    row = int(np.argmin(error))
    return row, float(error[row])


def _translation_alignment(
    translation: np.ndarray,
    translations: np.ndarray,
    physical_image_size: int,
) -> tuple[int, float]:
    native = np.asarray(translation, dtype=np.float64).reshape(3)[:2]
    recovar = -2.0 * np.pi * np.asarray(translations, dtype=np.float64) / physical_image_size
    error = np.max(np.abs(recovar - native[None]), axis=1)
    row = int(np.argmin(error))
    return row, float(error[row])


def _score_terms(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.float32,
) -> dict[str, np.ndarray | np.float32]:
    reference = np.asarray(reference, dtype=np.complex64)
    shifted = np.asarray(shifted, dtype=np.complex64)
    corr = np.asarray(corr, dtype=np.float32)
    diff_real = np.subtract(reference.real, shifted.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted.imag, dtype=np.float32)
    contribution = _cuda_fine_contribution(diff_real, diff_imag, corr)
    replay_lanes = _replay_lanes(contribution)
    production_lanes = _cuda_fine_production_lanes(diff_real, diff_imag, corr)
    replay_reduced = _reduce_lanes(replay_lanes)
    production_reduced = _reduce_lanes(production_lanes)
    replay_raw = np.add(replay_reduced, np.float32(sum_init), dtype=np.float32)
    production_raw = np.add(production_reduced, np.float32(sum_init), dtype=np.float32)
    return {
        "diff_real": diff_real,
        "diff_imag": diff_imag,
        "contribution": contribution,
        "replay_lanes": replay_lanes,
        "production_lanes": production_lanes,
        "replay_reduced": replay_reduced,
        "production_reduced": production_reduced,
        "replay_raw": replay_raw,
        "production_raw": production_raw,
    }


def _raw_with(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.float32,
) -> float:
    return float(_score_terms(reference, shifted, corr, sum_init)["production_raw"])


def _expanded_score_components(
    reference: np.ndarray,
    shifted: np.ndarray,
    corr: np.ndarray,
    sum_init: np.float32,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    reference128 = np.asarray(reference, dtype=np.complex128)
    shifted128 = np.asarray(shifted, dtype=np.complex128)
    corr64 = np.asarray(corr, dtype=np.float64)
    image_norm = 0.5 * corr64 * np.abs(shifted128) ** 2
    reference_norm = 0.5 * corr64 * np.abs(reference128) ** 2
    cross = -corr64 * np.real(reference128 * np.conj(shifted128))
    pixel_total = image_norm + reference_norm + cross
    components = {
        "image_norm": float(np.sum(image_norm, dtype=np.float64)),
        "reference_norm": float(np.sum(reference_norm, dtype=np.float64)),
        "cross": float(np.sum(cross, dtype=np.float64)),
        "pixel_total": float(np.sum(pixel_total, dtype=np.float64)),
        "highres_sum": float(np.float32(sum_init)),
        "total": float(np.sum(pixel_total, dtype=np.float64) + np.float64(sum_init)),
    }
    return components, {
        "image_norm": image_norm,
        "reference_norm": reference_norm,
        "cross": cross,
        "pixel_total": pixel_total,
    }


def _compare_particle(
    capture_path: Path,
    recovar_path: Path,
    *,
    expected_stack: int,
    physical_image_size: int,
    dump_path: Path,
    ctf_params: np.ndarray | None,
    voxel_size: float | None,
) -> dict[str, Any]:
    capture = load_fine_operand_capture(capture_path)
    validation = validate_capture(
        capture,
        expected_stack=expected_stack,
        expected_class=1,
        expected_rotation_local=0,
        expected_translations=(56,),
    )
    _require(capture.candidates.size == 1, "focused capture must contain one candidate")
    candidate = capture.candidates[0]
    pixels = capture.pixels.reshape(1, capture.image_size)[0]
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: archive[name] for name in archive.files}

    current_size = _infer_current_size(capture.image_size)
    _require(int(recovar["current_size"]) == current_size, "current size changed")
    rotation_row, rotation_error = _rotation_alignment(candidate["matrix"], recovar["rotations"])
    translation_row, translation_error = _translation_alignment(
        candidate["translation"], recovar["fine_translations"], physical_image_size
    )
    _require(rotation_error <= 1.0e-6, "rotation alignment exceeds tolerance")
    _require(translation_error <= 1.0e-6, "translation alignment exceeds tolerance")
    _require(bool(recovar["candidate_mask"][rotation_row, translation_row]), "target tuple is masked")

    factor_matches = sorted(
        capture_path.parent.glob(f"*_stack{expected_stack}_img0_class1.bpre-v2.bin")
    )
    _require(len(factor_matches) == 1, "expected one colocated native BPref factor capture")
    factor = load_factor_capture(factor_matches[0])
    _require(
        factor.pixels.shape == pixels.shape
        and np.array_equal(factor.pixels["x"], pixels["x"])
        and np.array_equal(factor.pixels["y"], pixels["y"]),
        "native BPref and fine-score pixel coordinates changed",
    )

    lookup = np.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            (physical_image_size, physical_image_size),
            current_size,
            np.asarray(recovar["window_indices"], dtype=np.int32),
        ),
        dtype=np.int32,
    )
    _require(lookup.shape == (capture.image_size,), "packed pixel lookup shape changed")
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    _require(
        np.array_equal(np.sort(supported_compact), np.arange(recovar["window_indices"].size)),
        "packed pixel lookup is not a complete compact gather",
    )

    native_reference = (
        np.asarray(pixels["reference_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["reference_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_image = (
        np.asarray(pixels["image_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["image_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_shifted = (
        np.asarray(pixels["shifted_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["shifted_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_corr = np.asarray(pixels["corr"], dtype=np.float32)
    native_contribution = np.asarray(pixels["contribution"], dtype=np.float32)
    _require(
        not np.any(native_corr[lookup < 0]) and not np.any(native_contribution[lookup < 0]),
        "native score support extends outside RECOVAR compact support",
    )

    n2 = np.float32(physical_image_size**2)
    n4 = np.float32(physical_image_size**4)
    recovar_reference = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_image = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_shifted = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_corr = np.zeros(capture.image_size, dtype=np.float32)
    recovar_reference[supported_full] = (
        np.asarray(recovar["proj_half"][rotation_row, supported_compact], dtype=np.complex64) / n2
    ).astype(np.complex64)
    recovar_shifted[supported_full] = (
        np.asarray(
            recovar["shifted_corrected"][translation_row, supported_compact],
            dtype=np.complex64,
        )
        / n2
    ).astype(np.complex64)
    recovar_phases = np.asarray(
        _half_translation_phase_table_for_indices(
            np.asarray(recovar["fine_translations"], dtype=np.float32),
            (physical_image_size, physical_image_size),
            np.asarray(recovar["window_indices"], dtype=np.int32),
        ),
        dtype=np.complex64,
    )
    recovar_phase = np.zeros(capture.image_size, dtype=np.complex64)
    recovar_phase[supported_full] = recovar_phases[translation_row, supported_compact]
    recovar_image[supported_full] = (
        recovar_shifted[supported_full] / recovar_phase[supported_full]
    ).astype(np.complex64)
    recovar_corr[supported_full] = (
        np.asarray(recovar["ctf2_over_nv_score"][supported_compact], dtype=np.float32)
        * np.asarray(recovar["half_weights"][supported_compact], dtype=np.float32)
        * n4
    ).astype(np.float32)

    positive_reference_error = _metric(
        native_reference[supported_full], recovar_reference[supported_full]
    )["relative_l2_over_reference"]
    negative_reference_error = _metric(
        native_reference[supported_full], -recovar_reference[supported_full]
    )["relative_l2_over_reference"]
    sign = -1 if negative_reference_error < positive_reference_error else 1
    _require(sign == -1, "expected RELION/RECOVAR Fourier sign convention changed")
    recovar_reference *= np.complex64(sign)
    recovar_image *= np.complex64(sign)
    recovar_shifted *= np.complex64(sign)
    phase_valid = np.zeros(capture.image_size, dtype=bool)
    phase_valid[supported_full] = (
        np.abs(native_image[supported_full]) > np.float32(1.0e-8)
    )
    native_phase = np.zeros(capture.image_size, dtype=np.complex64)
    native_phase[phase_valid] = (
        native_shifted[phase_valid] / native_image[phase_valid]
    ).astype(np.complex64)

    native_terms = _score_terms(
        native_reference,
        native_shifted,
        native_corr,
        np.float32(candidate["sum_init"]),
    )
    recovar_terms = _score_terms(
        recovar_reference,
        recovar_shifted,
        recovar_corr,
        np.float32(recovar["relion_highres_xi2_half"]),
    )
    native_expanded, native_expanded_pixels = _expanded_score_components(
        native_reference, native_shifted, native_corr, np.float32(candidate["sum_init"])
    )
    recovar_expanded, recovar_expanded_pixels = _expanded_score_components(
        recovar_reference,
        recovar_shifted,
        recovar_corr,
        np.float32(recovar["relion_highres_xi2_half"]),
    )
    expanded_comparison = {
        name: {
            "native": native_expanded[name],
            "recovar": recovar_expanded[name],
            "recovar_minus_native": recovar_expanded[name] - native_expanded[name],
        }
        for name in native_expanded
    }
    ctf_noise_preprocess_attribution = None
    ctf_attribution_arrays: dict[str, np.ndarray] = {}
    if ctf_params is not None:
        _require(voxel_size is not None, "voxel size is required with CTF parameters")
        original_index = int(recovar["original_index"])
        recovar_ctf_physical = np.asarray(
            _compute_spa_ctf(
                jnp.asarray(ctf_params[original_index : original_index + 1], dtype=jnp.float32),
                (physical_image_size, physical_image_size),
                voxel_size,
                half_image=True,
            ),
            dtype=np.float32,
        )[0].reshape(-1)
        recovar_ctf = -recovar_ctf_physical[
            np.asarray(recovar["window_indices"], dtype=np.int32)[supported_compact]
        ]
        native_ctf = np.asarray(factor.pixels["ctf"][supported_full], dtype=np.float32)
        native_inverse_noise = np.asarray(
            factor.pixels["minvsigma2"][supported_full], dtype=np.float32
        )
        safe_ctf = np.abs(recovar_ctf) > np.float32(0.05)
        recovar_inverse_noise = np.divide(
            recovar_corr[supported_full],
            recovar_ctf * recovar_ctf,
            dtype=np.float32,
        )
        native_processed = (
            native_image[supported_full] * native_ctf
        ).astype(np.complex64)
        recovar_processed = (
            recovar_image[supported_full] * recovar_ctf
        ).astype(np.complex64)
        native_image_norm_same_noise = float(
            0.5
            * np.sum(
                native_inverse_noise.astype(np.float64)
                * np.abs(native_processed.astype(np.complex128)) ** 2,
                dtype=np.float64,
            )
        )
        recovar_image_norm_same_noise = float(
            0.5
            * np.sum(
                native_inverse_noise.astype(np.float64)
                * np.abs(recovar_processed.astype(np.complex128)) ** 2,
                dtype=np.float64,
            )
        )
        ctf_noise_preprocess_attribution = {
            "ctf_sign_alignment_multiplier": -1,
            "ctf": _metric(native_ctf, recovar_ctf),
            "inverse_noise_safe_abs_ctf_gt_0p05": _metric(
                native_inverse_noise[safe_ctf], recovar_inverse_noise[safe_ctf]
            ),
            "safe_inverse_noise_pixel_count": int(np.count_nonzero(safe_ctf)),
            "processed_fourier_image": _metric(native_processed, recovar_processed),
            "image_norm_with_native_inverse_noise": {
                "native": native_image_norm_same_noise,
                "recovar_processed_image": recovar_image_norm_same_noise,
                "recovar_minus_native": (
                    recovar_image_norm_same_noise - native_image_norm_same_noise
                ),
            },
        }
        ctf_attribution_arrays = {
            "native_ctf": native_ctf,
            "recovar_ctf": recovar_ctf,
            "native_inverse_noise": native_inverse_noise,
            "recovar_inverse_noise": recovar_inverse_noise,
            "safe_inverse_noise_mask": safe_ctf,
            "native_processed_image": native_processed,
            "recovar_processed_image": recovar_processed,
        }
    _require(
        np.array_equal(native_terms["contribution"], native_contribution),
        "host SASS replay does not reproduce captured native contributions",
    )
    _require(
        np.array_equal(native_terms["replay_lanes"], candidate["lane_partials"]),
        "host replay does not reproduce captured native diagnostic lanes",
    )
    _require(
        np.asarray(native_terms["production_raw"], dtype=np.float32)
        == np.float32(candidate["production_raw_diff2"]),
        "host SASS replay does not reproduce native production raw diff2",
    )

    native_sum = np.float32(candidate["sum_init"])
    recovar_sum = np.float32(recovar["relion_highres_xi2_half"])
    native_raw = float(np.float32(candidate["production_raw_diff2"]))
    recovar_raw = float(np.float32(recovar_terms["production_raw"]))
    substitutions = {
        "native_baseline": native_raw,
        "replace_reference_only": _raw_with(
            recovar_reference, native_shifted, native_corr, native_sum
        ),
        "replace_shifted_image_only": _raw_with(
            native_reference, recovar_shifted, native_corr, native_sum
        ),
        "replace_corr_only": _raw_with(
            native_reference, native_shifted, recovar_corr, native_sum
        ),
        "replace_highres_sum_only": _raw_with(
            native_reference, native_shifted, native_corr, recovar_sum
        ),
        "recovar_all": recovar_raw,
        "recovar_with_native_reference": _raw_with(
            native_reference, recovar_shifted, recovar_corr, recovar_sum
        ),
        "recovar_with_native_shifted_image": _raw_with(
            recovar_reference, native_shifted, recovar_corr, recovar_sum
        ),
        "recovar_with_native_corr": _raw_with(
            recovar_reference, recovar_shifted, native_corr, recovar_sum
        ),
        "recovar_with_native_highres_sum": _raw_with(
            recovar_reference, recovar_shifted, recovar_corr, native_sum
        ),
    }
    rescue_names = (
        "recovar_with_native_reference",
        "recovar_with_native_shifted_image",
        "recovar_with_native_corr",
        "recovar_with_native_highres_sum",
    )
    rescue_ranking = sorted(
        (
            {
                "intervention": name,
                "absolute_residual_to_native_raw": abs(substitutions[name] - native_raw),
            }
            for name in rescue_names
        ),
        key=lambda row: row["absolute_residual_to_native_raw"],
    )
    exhaustive_rescues = []
    component_names = ("reference", "shifted_image", "corr", "highres_sum")
    for native_count in range(len(component_names) + 1):
        for native_components in itertools.combinations(component_names, native_count):
            native_set = set(native_components)
            raw = _raw_with(
                native_reference if "reference" in native_set else recovar_reference,
                native_shifted if "shifted_image" in native_set else recovar_shifted,
                native_corr if "corr" in native_set else recovar_corr,
                native_sum if "highres_sum" in native_set else recovar_sum,
            )
            exhaustive_rescues.append(
                {
                    "native_components": list(native_components),
                    "native_component_count": native_count,
                    "raw_diff2": raw,
                    "absolute_residual_to_native_raw": abs(raw - native_raw),
                    "exact_native_raw": raw == native_raw,
                }
            )
    exhaustive_rescues.sort(
        key=lambda row: (
            row["absolute_residual_to_native_raw"],
            row["native_component_count"],
            row["native_components"],
        )
    )

    stage_arrays = {
        "reference": (native_reference[supported_full], recovar_reference[supported_full]),
        "unshifted_image": (native_image[supported_full], recovar_image[supported_full]),
        "translation_phase": (native_phase[phase_valid], recovar_phase[phase_valid]),
        "shifted_image": (native_shifted[supported_full], recovar_shifted[supported_full]),
        "corr": (native_corr, recovar_corr),
        "diff_real": (
            native_terms["diff_real"][supported_full],
            recovar_terms["diff_real"][supported_full],
        ),
        "diff_imag": (
            native_terms["diff_imag"][supported_full],
            recovar_terms["diff_imag"][supported_full],
        ),
        "contribution": (native_terms["contribution"], recovar_terms["contribution"]),
        "diagnostic_lane_partial": (
            native_terms["replay_lanes"],
            recovar_terms["replay_lanes"],
        ),
        "production_lane_partial": (
            native_terms["production_lanes"],
            recovar_terms["production_lanes"],
        ),
        "highres_sum": (np.asarray([native_sum]), np.asarray([recovar_sum])),
        "raw_diff2": (
            np.asarray([native_terms["production_raw"]], dtype=np.float32),
            np.asarray([recovar_terms["production_raw"]], dtype=np.float32),
        ),
    }
    stage_metrics = {name: _metric(lhs, rhs) for name, (lhs, rhs) in stage_arrays.items()}
    first_exact_unequal = next(
        (name for name, metrics in stage_metrics.items() if not metrics["exact_equal"]),
        None,
    )

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_path,
        native_pixel_x=np.asarray(pixels["x"], dtype=np.int32),
        native_pixel_y=np.asarray(pixels["y"], dtype=np.int32),
        native_pixel_flags=np.asarray(pixels["flags"], dtype=np.uint32),
        supported_full=supported_full,
        supported_compact=supported_compact,
        native_reference=native_reference,
        recovar_reference=recovar_reference,
        native_image=native_image,
        recovar_image=recovar_image,
        native_phase=native_phase,
        recovar_phase=recovar_phase,
        phase_valid=phase_valid,
        native_shifted=native_shifted,
        recovar_shifted=recovar_shifted,
        native_corr=native_corr,
        recovar_corr=recovar_corr,
        native_diff_real=native_terms["diff_real"],
        recovar_diff_real=recovar_terms["diff_real"],
        native_diff_imag=native_terms["diff_imag"],
        recovar_diff_imag=recovar_terms["diff_imag"],
        native_contribution=native_terms["contribution"],
        recovar_contribution=recovar_terms["contribution"],
        native_expanded_image_norm=native_expanded_pixels["image_norm"],
        recovar_expanded_image_norm=recovar_expanded_pixels["image_norm"],
        native_expanded_reference_norm=native_expanded_pixels["reference_norm"],
        recovar_expanded_reference_norm=recovar_expanded_pixels["reference_norm"],
        native_expanded_cross=native_expanded_pixels["cross"],
        recovar_expanded_cross=recovar_expanded_pixels["cross"],
        native_diagnostic_lanes=native_terms["replay_lanes"],
        recovar_diagnostic_lanes=recovar_terms["replay_lanes"],
        native_production_lanes=native_terms["production_lanes"],
        recovar_production_lanes=recovar_terms["production_lanes"],
        native_diagnostic_raw=np.asarray(native_terms["replay_raw"], dtype=np.float32),
        recovar_diagnostic_raw=np.asarray(recovar_terms["replay_raw"], dtype=np.float32),
        native_raw=np.asarray(native_terms["production_raw"], dtype=np.float32),
        recovar_raw=np.asarray(recovar_terms["production_raw"], dtype=np.float32),
        **ctf_attribution_arrays,
    )
    return {
        "stack_index_one_based": expected_stack,
        "capture_path": str(capture_path.resolve()),
        "capture_sha256": _sha256(capture_path),
        "recovar_path": str(recovar_path.resolve()),
        "recovar_sha256": _sha256(recovar_path),
        "dump_path": str(dump_path.resolve()),
        "dump_sha256": _sha256(dump_path),
        "validation": validation,
        "current_size": current_size,
        "physical_image_size": physical_image_size,
        "rotation_row_recovar": rotation_row,
        "rotation_alignment_max_abs": rotation_error,
        "translation_row_recovar": translation_row,
        "translation_alignment_max_abs": translation_error,
        "packed_pixel_count": int(capture.image_size),
        "supported_pixel_count": int(supported_full.size),
        "recovar_fourier_alignment_multiplier": sign,
        "first_exact_unequal_boundary": first_exact_unequal,
        "stage_metrics": stage_metrics,
        "expanded_score_components_float64": expanded_comparison,
        "ctf_noise_preprocess_attribution": ctf_noise_preprocess_attribution,
        "raw_substitutions": substitutions,
        "single_native_operand_rescue_ranking": rescue_ranking,
        "exhaustive_native_operand_rescue_ranking": exhaustive_rescues,
    }


def _parse_particle(value: str) -> tuple[int, int]:
    try:
        stack_text, original_text = value.split(":", maxsplit=1)
        stack = int(stack_text)
        original = int(original_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("particle must be STACK:ZERO_BASED_ORIGINAL") from error
    if stack <= 0 or original < 0:
        raise argparse.ArgumentTypeError("particle indices must be nonnegative and stack is one-based")
    return stack, original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--particle", type=_parse_particle, action="append", required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--ctf-pkl", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    args = parser.parse_args()

    ctf_params = None
    voxel_size = None
    if args.ctf_pkl is not None:
        with args.ctf_pkl.open("rb") as stream:
            legacy_ctf = np.asarray(pickle.load(stream), dtype=np.float32)
        _require(
            legacy_ctf.ndim == 2 and legacy_ctf.shape[1] == 9,
            "expected cryoDRGN CTF array with D, pixel size, and seven CTF fields",
        )
        voxel_size = float(legacy_ctf[0, 1])
        _require(
            np.all(legacy_ctf[:, 0] == args.physical_image_size)
            and np.all(legacy_ctf[:, 1] == voxel_size),
            "CTF image size or pixel size is not constant",
        )
        ctf_params = np.concatenate(
            (
                legacy_ctf[:, 2:],
                np.zeros((legacy_ctf.shape[0], 1), dtype=np.float32),
                np.ones((legacy_ctf.shape[0], 1), dtype=np.float32),
            ),
            axis=1,
        )

    reports = []
    for stack, original in args.particle:
        capture_matches = sorted(
            args.capture_dir.glob(f"*_stack{stack}_class1.fine-operand-v1.bin")
        )
        recovar_matches = sorted(args.recovar_dir.glob(f"pass2_orig{original:06d}_cs*.npz"))
        _require(len(capture_matches) == 1, f"expected one fine operand capture for stack {stack}")
        _require(len(recovar_matches) == 1, f"expected one RECOVAR capture for row {original}")
        reports.append(
            _compare_particle(
                capture_matches[0],
                recovar_matches[0],
                expected_stack=stack,
                physical_image_size=args.physical_image_size,
                dump_path=args.dump_dir / f"stack{stack}_fine_operand_boundary.npz",
                ctf_params=ctf_params,
                voxel_size=voxel_size,
            )
        )
    report = {
        "schema": "recovar.em.k1_relion_fine_operand_boundary.v1",
        "status": "complete",
        "particle_count": len(reports),
        "first_exact_unequal_boundary_counts": {
            stage: sum(row["first_exact_unequal_boundary"] == stage for row in reports)
            for stage in sorted({row["first_exact_unequal_boundary"] for row in reports})
        },
        "particles": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
