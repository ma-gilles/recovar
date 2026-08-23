#!/usr/bin/env python3
"""Compare bounded native/RECOVAR K=1 coarse components and live operands."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    POSTERIOR_TV_GATE,
    SCORE_MAX_GATE,
    SCORE_P95_GATE,
    _map_relion_table,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    recovar_score_components,
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.analyze_k1_coarse_live_operands import (
    load_live_artifact,
    replay_live_lanes,
)
from scripts.validate_relion_coarse_lane_capture import load_artifact as load_lanes
from scripts.validate_relion_coarse_operand_capture import (
    load_artifact as load_operand,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
)
from scripts.validate_relion_coarse_pass1_components import (
    load_artifact as load_components,
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


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.complex128).reshape(-1)
    right = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "operand topology mismatch")
    return float(np.linalg.norm(right - left) / max(np.linalg.norm(left), np.finfo(float).tiny))


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    _require(absolute.size > 0, "cannot summarize an empty residual")
    _require(np.all(np.isfinite(absolute)), "residual contains non-finite values")
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _operand_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare original binary32 operand values without discarding complex phase."""

    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "operand topology mismatch")
    dtype = np.complex64 if np.iscomplexobj(left) or np.iscomplexobj(right) else np.float32
    left = np.ascontiguousarray(left, dtype=dtype)
    right = np.ascontiguousarray(right, dtype=dtype)
    if dtype == np.complex64:
        exact = np.all(
            left.view(np.float32).view(np.uint32).reshape(-1, 2)
            == right.view(np.float32).view(np.uint32).reshape(-1, 2),
            axis=1,
        )
    else:
        exact = left.view(np.uint32).reshape(-1) == right.view(np.uint32).reshape(-1)
    residual = np.abs(right.astype(np.complex128) - left.astype(np.complex128)).reshape(-1)
    left_flat = left.astype(np.complex128).reshape(-1)
    right_flat = right.astype(np.complex128).reshape(-1)
    denominator = np.vdot(left_flat, left_flat).real
    _require(denominator > 0.0, "cannot fit a zero operand")
    fitted_scalar = np.vdot(left_flat, right_flat) / denominator
    fitted_residual = right_flat - fitted_scalar * left_flat
    return {
        "shape": list(left.shape),
        "bitwise_equal_count": int(np.count_nonzero(exact)),
        "value_count": int(exact.size),
        "bitwise_equal_fraction": float(np.mean(exact)),
        "relative_l2": _relative_l2(left, right),
        "p95_abs": float(np.percentile(residual, 95)),
        "max_abs": float(np.max(residual)),
        "diagnostic_least_squares_scalar": {
            "real": float(fitted_scalar.real),
            "imag": float(fitted_scalar.imag),
        },
        "diagnostic_after_scalar_fit_relative_l2": float(
            np.linalg.norm(fitted_residual)
            / max(np.linalg.norm(right_flat), np.finfo(float).tiny)
        ),
    }


def _active_pixel_operand_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    active_pixels: np.ndarray,
) -> dict[str, Any]:
    """Compare only pixels that can contribute to the native score.

    Native live captures retain values outside the circular score mask, while
    RECOVAR's exact-operand path explicitly zeros those unused pixels.  Those
    values are multiplied by a zero pixel weight and must not be counted as an
    operand mismatch.
    """

    left = np.asarray(reference)
    right = np.asarray(candidate)
    mask = np.asarray(active_pixels, dtype=bool)
    _require(left.shape == right.shape, "active operand topology mismatch")
    _require(left.shape[-1] == mask.size, "active pixel mask topology mismatch")
    _require(np.any(mask), "active pixel mask is empty")
    return _operand_comparison(left[..., mask], right[..., mask])


def _center_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.max(values)


def _component_decomposition(
    total_residual: np.ndarray,
    norm_residual: np.ndarray,
    cross_residual: np.ndarray,
) -> dict[str, Any]:
    total = np.asarray(total_residual, dtype=np.float64)
    norm = np.asarray(norm_residual, dtype=np.float64)
    cross = np.asarray(cross_residual, dtype=np.float64)
    _require(total.shape == norm.shape == cross.shape, "component topology mismatch")
    _require(total.ndim == 2 and total.size > 1, "component panel is too small")

    def center(values: np.ndarray) -> np.ndarray:
        return values - np.mean(values)

    centered_total = center(total)
    total_energy = float(np.sum(centered_total**2))
    _require(total_energy > 0.0, "component residual has zero centered energy")
    without_norm = float(np.sum(center(total - norm) ** 2))
    without_cross = float(np.sum(center(total - cross) ** 2))
    closure = centered_total - center(norm + cross)
    return {
        "total_centered_energy": total_energy,
        "counterfactual_energy_removal_fraction": {
            "reference_norm": float(1.0 - without_norm / total_energy),
            "cross": float(1.0 - without_cross / total_energy),
        },
        "closure": {
            "p95_abs": float(np.percentile(np.abs(closure), 95)),
            "max_abs": float(np.max(np.abs(closure))),
        },
    }


def _candidate_panel_counterfactual(
    total_residual: np.ndarray,
    counterfactual_residual: np.ndarray,
) -> dict[str, Any]:
    """Measure residual-energy removal over any nontrivial candidate panel."""

    total = np.asarray(total_residual, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_residual, dtype=np.float64)
    _require(total.shape == counterfactual.shape, "counterfactual shape mismatch")
    _require(total.ndim == 2 and total.size > 1, "candidate panel is too small")
    baseline = total - np.mean(total)
    swapped = counterfactual - np.mean(counterfactual)
    baseline_energy = float(np.sum(baseline**2))
    _require(baseline_energy > 0.0, "baseline residual has zero centered energy")
    swapped_energy = float(np.sum(swapped**2))
    return {
        "baseline_centered_energy": baseline_energy,
        "swapped_centered_energy": swapped_energy,
        "counterfactual_energy_removal_fraction": float(
            1.0 - swapped_energy / baseline_energy
        ),
        "swapped_centered_p95_abs": float(np.percentile(np.abs(swapped), 95)),
        "swapped_centered_max_abs": float(np.max(np.abs(swapped))),
    }


def _support_mismatch_panel(
    *,
    rotation_ids: np.ndarray,
    native_raw: np.ndarray,
    native_norm: np.ndarray,
    native_cross: np.ndarray,
    native_significant: np.ndarray,
    recovar_raw: np.ndarray,
    recovar_norm: np.ndarray,
    recovar_cross: np.ndarray,
    recovar_significant: np.ndarray,
    support_mismatches_only: bool = True,
) -> list[dict[str, Any]]:
    """Emit captured component values, optionally only at support mismatches."""

    arrays = (
        native_raw,
        native_norm,
        native_cross,
        native_significant,
        recovar_raw,
        recovar_norm,
        recovar_cross,
        recovar_significant,
    )
    expected_shape = (len(rotation_ids), native_raw.shape[1])
    _require(all(np.asarray(value).shape == expected_shape for value in arrays), "support panel topology mismatch")
    selected_positions = (
        np.argwhere(native_significant != recovar_significant)
        if support_mismatches_only
        else np.argwhere(np.ones(expected_shape, dtype=bool))
    )
    rows = []
    for rotation_position, translation_id in selected_positions:
        rows.append(
            {
                "rotation_id": int(rotation_ids[rotation_position]),
                "translation_id": int(translation_id),
                "native_significant": bool(native_significant[rotation_position, translation_id]),
                "recovar_significant": bool(recovar_significant[rotation_position, translation_id]),
                "native": {
                    "raw_score": float(-native_raw[rotation_position, translation_id]),
                    "reference_norm_score": float(-native_norm[rotation_position, translation_id]),
                    "cross_score": float(-native_cross[rotation_position, translation_id]),
                },
                "recovar": {
                    "raw_score": float(recovar_raw[rotation_position, translation_id]),
                    "reference_norm_score": float(recovar_norm[rotation_position, translation_id]),
                    "cross_score": float(recovar_cross[rotation_position, translation_id]),
                },
                "recovar_minus_native": {
                    "raw_score": float(
                        recovar_raw[rotation_position, translation_id]
                        + native_raw[rotation_position, translation_id]
                    ),
                    "reference_norm_score": float(
                        recovar_norm[rotation_position, translation_id]
                        + native_norm[rotation_position, translation_id]
                    ),
                    "cross_score": float(
                        recovar_cross[rotation_position, translation_id]
                        + native_cross[rotation_position, translation_id]
                    ),
                },
            }
        )
    return rows


def _complete_support_mismatch_records(
    *,
    native_significant: np.ndarray,
    recovar_significant: np.ndarray,
    native_raw_diff2: np.ndarray,
    recovar_raw_score: np.ndarray,
    native_posterior: np.ndarray,
    recovar_posterior: np.ndarray,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """Describe support flips without requiring optional projection captures."""

    arrays = (
        native_significant,
        recovar_significant,
        native_raw_diff2,
        recovar_raw_score,
        native_posterior,
        recovar_posterior,
    )
    shape = np.asarray(native_significant).shape
    _require(len(shape) == 2, "coarse support table must be rotation by translation")
    _require(all(np.asarray(value).shape == shape for value in arrays), "coarse support topology mismatch")
    _require(limit > 0, "support mismatch record limit must be positive")
    positions = np.argwhere(native_significant != recovar_significant)
    records = []
    for rotation_id, translation_id in positions[:limit]:
        records.append(
            {
                "rotation_id": int(rotation_id),
                "translation_id": int(translation_id),
                "native_significant": bool(native_significant[rotation_id, translation_id]),
                "recovar_significant": bool(recovar_significant[rotation_id, translation_id]),
                "native_raw_score": float(-native_raw_diff2[rotation_id, translation_id]),
                "recovar_raw_score": float(recovar_raw_score[rotation_id, translation_id]),
                "native_posterior": float(native_posterior[rotation_id, translation_id]),
                "recovar_posterior": float(recovar_posterior[rotation_id, translation_id]),
            }
        )
    return records


def _top_posterior_records(
    *,
    native_raw_score: np.ndarray,
    recovar_raw_score: np.ndarray,
    native_log_weight: np.ndarray,
    recovar_log_weight: np.ndarray,
    native_posterior: np.ndarray,
    recovar_posterior: np.ndarray,
    native_significant: np.ndarray,
    recovar_significant: np.ndarray,
    limit: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    """Record the leading cutoff operands in each engine's posterior order."""

    arrays = [
        np.asarray(value)
        for value in (
            native_raw_score,
            recovar_raw_score,
            native_log_weight,
            recovar_log_weight,
            native_posterior,
            recovar_posterior,
            native_significant,
            recovar_significant,
        )
    ]
    _require(all(value.shape == arrays[0].shape for value in arrays), "top-posterior topology mismatch")
    _require(arrays[0].ndim == 2, "top-posterior tables must be rotation x translation")
    native_prob = arrays[4].reshape(-1)
    recovar_prob = arrays[5].reshape(-1)
    native_order = np.argsort(-native_prob, kind="stable")
    recovar_order = np.argsort(-recovar_prob, kind="stable")
    n_translations = arrays[0].shape[1]

    def ordered_rows(order: np.ndarray, engine: str) -> list[dict[str, Any]]:
        cumulative = np.float64(0.0)
        rows = []
        for rank, flat_index in enumerate(order[: max(1, int(limit))], start=1):
            rotation_id, translation_id = divmod(int(flat_index), n_translations)
            cumulative += np.float64(
                native_prob[flat_index] if engine == "native" else recovar_prob[flat_index]
            )
            rows.append(
                {
                    "rank": rank,
                    "rotation_id": rotation_id,
                    "translation_id": translation_id,
                    "ordered_engine_cumulative_posterior": float(cumulative),
                    "native": {
                        "raw_score": float(arrays[0][rotation_id, translation_id]),
                        "log_weight": float(arrays[2][rotation_id, translation_id]),
                        "posterior": float(arrays[4][rotation_id, translation_id]),
                        "significant": bool(arrays[6][rotation_id, translation_id]),
                    },
                    "recovar": {
                        "raw_score": float(arrays[1][rotation_id, translation_id]),
                        "log_weight": float(arrays[3][rotation_id, translation_id]),
                        "posterior": float(arrays[5][rotation_id, translation_id]),
                        "significant": bool(arrays[7][rotation_id, translation_id]),
                    },
                }
            )
        return rows

    return {
        "native_order": ordered_rows(native_order, "native"),
        "recovar_order": ordered_rows(recovar_order, "recovar"),
    }


def _float32_table_bitwise_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Report exact binary32 equality and the first unequal table coordinate."""

    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    _require(left.shape == right.shape == selected.shape, "float32 table topology mismatch")
    _require(np.any(selected), "float32 table comparison mask is empty")
    unequal = selected & (left.view(np.uint32) != right.view(np.uint32))
    first = None
    if np.any(unequal):
        coordinate = tuple(int(value) for value in np.argwhere(unequal)[0])
        first = {
            "coordinate": list(coordinate),
            "reference": float(left[coordinate]),
            "reference_bits": int(left[coordinate].view(np.uint32)),
            "candidate": float(right[coordinate]),
            "candidate_bits": int(right[coordinate].view(np.uint32)),
        }
    selected_count = int(np.count_nonzero(selected))
    unequal_count = int(np.count_nonzero(unequal))
    return {
        "selected_count": selected_count,
        "bitwise_equal_count": selected_count - unequal_count,
        "bitwise_unequal_count": unequal_count,
        "bitwise_equal_fraction": float((selected_count - unequal_count) / selected_count),
        "first_bitwise_unequal": first,
    }


def _rotation_key_to_recovar(rotation_key: int, n_directions: int, n_psi: int) -> int:
    direction, psi = divmod(int(rotation_key), int(n_psi))
    _require(direction < n_directions, "native rotation key is out of range")
    return int(psi * n_directions + direction)


def _compact_to_native_full_order(
    values: np.ndarray,
    score_indices: np.ndarray,
    *,
    physical_image_size: int,
    current_size: int,
) -> np.ndarray:
    """Scatter RECOVAR compact operands into RELION's current-size FFTW order."""

    values = np.asarray(values)
    score_indices = np.asarray(score_indices, dtype=np.int64)
    _require(values.shape[-1] == score_indices.size, "compact operand topology mismatch")
    physical_half_width = physical_image_size // 2 + 1
    current_half_width = current_size // 2 + 1
    rows = score_indices // physical_half_width
    columns = score_indices % physical_half_width
    ky = rows - physical_image_size // 2
    inside = (
        (ky > -(current_size // 2))
        & (ky <= current_size // 2)
        & (columns <= current_size // 2)
    )
    if np.any(~inside):
        _require(
            np.all(values[..., ~inside] == 0),
            "nonzero compact operands lie outside the native scoring grid",
        )
    native_rows = np.where(ky[inside] >= 0, ky[inside], current_size + ky[inside])
    native_positions = native_rows * current_half_width + columns[inside]
    native_pixel_count = current_size * current_half_width
    _require(
        np.array_equal(np.sort(native_positions), np.arange(native_pixel_count)),
        "score indices do not bijectively cover the native current-size grid",
    )
    result = np.empty((*values.shape[:-1], native_pixel_count), dtype=values.dtype)
    result[..., native_positions] = values[..., inside]
    return result


def _native_fftw_values_on_recovar_grid(
    values: np.ndarray,
    score_indices: np.ndarray,
    *,
    physical_image_size: int,
    native_current_size: int,
) -> np.ndarray:
    """Gather a native FFTW grid onto RECOVAR's possibly +2 optics grid.

    The extra outer row/column in RECOVAR's even optics FFT size is outside
    RELION's scored 56x29 grid and is represented by exact zeros.  The active
    circular score support is wholly contained in the common coordinates.
    """

    values = np.asarray(values)
    score_indices = np.asarray(score_indices, dtype=np.int64)
    _require(values.ndim >= 1, "native FFTW values must have a pixel axis")
    physical_half_width = physical_image_size // 2 + 1
    native_half_width = native_current_size // 2 + 1
    native_pixel_count = native_current_size * native_half_width
    _require(
        values.shape[-1] == native_pixel_count,
        "native FFTW value topology mismatch",
    )
    rows = score_indices // physical_half_width
    columns = score_indices % physical_half_width
    ky = rows - physical_image_size // 2
    inside = (
        (ky > -(native_current_size // 2))
        & (ky <= native_current_size // 2)
        & (columns <= native_current_size // 2)
    )
    native_rows = np.where(ky[inside] >= 0, ky[inside], native_current_size + ky[inside])
    native_positions = native_rows * native_half_width + columns[inside]
    result = np.zeros((*values.shape[:-1], score_indices.size), dtype=values.dtype)
    result[..., inside] = values[..., native_positions]
    return result


def _log_scores_from_lane_partials(
    lane_partials: np.ndarray,
    *,
    translation_count: int,
) -> np.ndarray:
    """Apply the deterministic thread-order sum used for operand factorials."""

    lanes = np.asarray(lane_partials, dtype=np.float32).reshape(-1)
    _require(translation_count > 0, "translation count must be positive")
    _require(lanes.size >= translation_count, "lane panel is too small")
    diff2 = np.zeros(translation_count, dtype=np.float32)
    for thread, value in enumerate(lanes):
        translation = thread % translation_count
        diff2[translation] = np.float32(diff2[translation] + value)
    return -diff2


def _atomic_add_log_score_values(
    lane_partials: np.ndarray,
    *,
    translation_count: int,
    translation: int,
    initial_diff2: np.float32,
) -> np.ndarray:
    """Enumerate binary32 scores allowed by native coarse atomic ordering.

    RELION initializes each hypothesis with ``highres_Xi2 / 2`` and then one
    thread per active pixel lane atomically adds its lane partial.  CUDA may
    serialize those atomics in any lane order, so a deterministic thread-order
    sum is only one member of the legal result set.
    """

    lanes = np.asarray(lane_partials, dtype=np.float32).reshape(-1)
    _require(translation_count > 0, "translation count must be positive")
    _require(0 <= int(translation) < translation_count, "translation is out of range")
    _require(lanes.size >= translation_count, "lane panel is too small")
    active_lane_count = lanes.size // translation_count
    _require(0 < active_lane_count <= 8, "atomic-order enumeration requires 1--8 active lanes")
    terms = lanes[
        int(translation) : active_lane_count * translation_count : translation_count
    ]
    _require(terms.size == active_lane_count, "active lane topology mismatch")
    values: dict[int, np.float32] = {}
    for order in itertools.permutations(range(active_lane_count)):
        accumulator = np.float32(initial_diff2)
        for lane in order:
            accumulator = np.float32(
                np.float64(accumulator) + np.float64(terms[lane])
            )
        score = np.float32(-accumulator)
        values[int(score.view(np.uint32))] = score
    return np.asarray(sorted(values.values()), dtype=np.float32)


def _atomic_relative_score_envelope(
    lane_partials: np.ndarray,
    *,
    translation_count: int,
    first_translation: int,
    second_translation: int,
    initial_diff2: np.float32,
) -> tuple[dict[str, Any], np.ndarray]:
    """Summarize all legal second-minus-first native atomic score margins."""

    first = _atomic_add_log_score_values(
        lane_partials,
        translation_count=translation_count,
        translation=first_translation,
        initial_diff2=initial_diff2,
    )
    second = _atomic_add_log_score_values(
        lane_partials,
        translation_count=translation_count,
        translation=second_translation,
        initial_diff2=initial_diff2,
    )
    margins = np.unique(
        second.astype(np.float64)[:, np.newaxis]
        - first.astype(np.float64)[np.newaxis, :]
    )
    return (
        {
            "first_translation": int(first_translation),
            "second_translation": int(second_translation),
            "active_lane_count": int(np.asarray(lane_partials).size // translation_count),
            "first_unique_log_score_count": int(first.size),
            "second_unique_log_score_count": int(second.size),
            "relative_log_score_unique_count": int(margins.size),
            "relative_log_score_min": float(np.min(margins)),
            "relative_log_score_max": float(np.max(margins)),
        },
        margins,
    )


def _native_coarse_image_size(
    component_header: np.ndarray,
    operand_header: np.ndarray,
) -> int:
    """Return the scored image size while checking native model consistency."""

    model_current_size = int(component_header[27])
    _require(
        int(operand_header[12]) == model_current_size,
        "native component/operand model current-size mismatch",
    )
    return int(operand_header[18])


def _score_size_relation(native_coarse_image_size: int, recovar_score_size: int) -> str:
    """Classify the exact or optics-FFT-equivalent coarse score window."""

    if native_coarse_image_size == recovar_score_size:
        return "exact"
    if recovar_score_size == native_coarse_image_size + 2:
        return "recovar_optics_fft_size_plus_two"
    return "mismatch"


def _matched_operand_rotation_panel(
    rotation_ids: np.ndarray,
    mapped_operand_ids: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Join a requested RECOVAR panel to the available native operand rows."""

    requested = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    captured = np.asarray(mapped_operand_ids, dtype=np.int64).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    _require(requested.shape == active.shape, "requested/active rotation topology mismatch")
    _require(np.unique(requested).size == requested.size, "duplicate requested rotations")
    _require(np.unique(captured).size == captured.size, "duplicate operand rotations")
    operand_index = {int(key): index for index, key in enumerate(captured)}
    matched = np.isin(requested, captured) & active
    _require(np.any(matched), "no common active requested operand rotations")
    request_positions = np.flatnonzero(matched)
    selected_ids = requested[request_positions]
    operand_order = np.asarray(
        [operand_index[int(key)] for key in selected_ids],
        dtype=np.int64,
    )
    return (
        selected_ids,
        request_positions,
        operand_order,
        sorted(set(requested.tolist()) - set(captured.tolist())),
        sorted(set(captured.tolist()) - set(requested.tolist())),
    )


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "current_size",
            "scores_pre_prior_per_class",
            "scores_with_prior_per_class",
            "weights_per_class",
            "significant_mask",
            "translations",
            "window_indices",
            "shifted_data",
            "ctf2_data",
            "half_weights",
            "coarse_gaussian_shifted_corrected",
            "coarse_gaussian_unshifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_initial_diff2",
            "coarse_gaussian_score_indices",
        }
        _require(required <= set(payload.files), f"missing RECOVAR fields: {path}")
        projection_fields = {
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        projection_capture_available = projection_fields <= set(payload.files)
        return {
            "path": path.resolve(),
            "sha256": _sha256(path),
            "original_index": int(payload["original_index"]),
            "current_size": int(payload["current_size"]),
            "scores": np.asarray(payload["scores_pre_prior_per_class"][0], dtype=np.float64),
            "scores_with_prior": np.asarray(
                payload["scores_with_prior_per_class"][0], dtype=np.float64
            ),
            "weights": np.asarray(payload["weights_per_class"][0], dtype=np.float64),
            "significant_mask": np.asarray(payload["significant_mask"], dtype=bool),
            "translations": np.asarray(payload["translations"], dtype=np.float64),
            "window_indices": np.asarray(payload["window_indices"], dtype=np.int64),
            "shifted": np.asarray(payload["shifted_data"], dtype=np.complex128),
            "ctf2": np.asarray(payload["ctf2_data"][0], dtype=np.float64),
            "half_weights": np.asarray(payload["half_weights"], dtype=np.float64),
            "projection_capture_available": projection_capture_available,
            "missing_projection_fields": sorted(projection_fields - set(payload.files)),
            "rotation_ids": (
                np.asarray(payload["projected_reference_rotation_ids"], dtype=np.int64)
                if projection_capture_available
                else None
            ),
            "references": (
                np.asarray(payload["projected_reference_per_class"][0], dtype=np.complex128)
                if projection_capture_available
                else None
            ),
            "norms": (
                np.asarray(
                    payload["projected_reference_norm_score_per_class"][0],
                    dtype=np.float64,
                )
                if projection_capture_available
                else None
            ),
            "crosses": (
                np.asarray(payload["projected_cross_score_per_class"][0], dtype=np.float64)
                if projection_capture_available
                else None
            ),
            "exact_shifted": np.asarray(
                payload["coarse_gaussian_shifted_corrected"], dtype=np.complex64
            ),
            "exact_unshifted": np.asarray(
                payload["coarse_gaussian_unshifted_corrected"], dtype=np.complex64
            ),
            "exact_pixel_weight": np.asarray(
                payload["coarse_gaussian_pixel_weight"], dtype=np.float32
            ),
            "exact_initial_diff2": np.float32(payload["coarse_gaussian_initial_diff2"]),
            "exact_score_indices": np.asarray(
                payload["coarse_gaussian_score_indices"], dtype=np.int64
            ),
        }


def _compare(
    components_path: Path,
    operand_path: Path,
    lane_path: Path,
    live_path: Path,
    recovar_path: Path,
    *,
    physical_iteration: int,
    physical_image_size: int,
    translation_pair_recovar: tuple[int, int] | None = None,
) -> dict[str, Any]:
    _require(
        int(physical_image_size) > 0 and int(physical_image_size) % 2 == 0,
        "physical image size must be a positive even integer",
    )
    components = load_components(components_path)
    operand = load_operand(operand_path)
    lanes = load_lanes(lane_path)
    live = load_live_artifact(live_path)
    recovar = _load_recovar(recovar_path)
    _require(components.stack_index == operand.stack_index, "native stack mismatch")
    _require(components.part_id == operand.part_id, "native part mismatch")
    _require(lanes.part_id == operand.part_id, "native lane/operand part mismatch")
    _require(lanes.stack_index == operand.stack_index, "native lane/operand stack mismatch")
    _require(live.part_id == operand.part_id, "native live/operand part mismatch")
    _require(live.stack_index == operand.stack_index, "native live/operand stack mismatch")
    _require(components.stack_index - 1 == recovar["original_index"], "cross-engine identity mismatch")
    _require(
        components.header[5] == int(physical_iteration),
        "native component physical iteration mismatch",
    )
    _require(
        operand.header[5] == int(physical_iteration),
        "native operand physical iteration mismatch",
    )
    native_coarse_image_size = _native_coarse_image_size(
        components.header,
        operand.header,
    )
    recovar_score_size = int(recovar["current_size"])
    score_size_relation = _score_size_relation(
        native_coarse_image_size,
        recovar_score_size,
    )
    _require(
        score_size_relation != "mismatch",
        "native coarse image/RECOVAR scoring-size mismatch",
    )
    n_directions, n_psi, _ = components.header[10:13]
    translation_permutation, translation_mapping = _translation_permutation(
        components.translations,
        recovar["translations"],
    )
    mapped_raw = _map_relion_table(
        components.raw_diff2,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_norm = _map_relion_table(
        components.reference_norms,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_cross = _map_relion_table(
        components.cross_terms,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    )
    mapped_weights = _map_relion_table(
        components.weights,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(np.float64)
    mapped_significant = _map_relion_table(
        components.significant_mask,
        n_directions=n_directions,
        n_psi=n_psi,
        relion_to_recovar_translation=translation_permutation,
    ).astype(bool)
    _require(mapped_raw.shape == recovar["scores"].shape, "candidate topology mismatch")
    relion_prior_support = mapped_raw != RELION_INVALID_DIFF2
    recovar_prior_support = np.isfinite(recovar["scores_with_prior"])
    common_prior_support = relion_prior_support & recovar_prior_support
    _require(np.any(common_prior_support), "no common finite-prior support")
    raw_residual = (
        _center_max(recovar["scores"][common_prior_support])
        - _center_max(-mapped_raw[common_prior_support])
    )
    raw_stats = _stats(raw_residual)
    raw_bitwise = _float32_table_bitwise_report(
        -mapped_raw,
        recovar["scores"],
        common_prior_support,
    )
    native_raw_score_f32 = np.asarray(-mapped_raw, dtype=np.float32)
    recovar_raw_score_f32 = np.asarray(recovar["scores"], dtype=np.float32)
    native_raw_centered_f32 = np.subtract(
        native_raw_score_f32,
        np.max(native_raw_score_f32[common_prior_support]),
        dtype=np.float32,
    )
    recovar_raw_centered_f32 = np.subtract(
        recovar_raw_score_f32,
        np.max(recovar_raw_score_f32[common_prior_support]),
        dtype=np.float32,
    )
    raw_centered_bitwise = _float32_table_bitwise_report(
        native_raw_centered_f32,
        recovar_raw_centered_f32,
        common_prior_support,
    )
    relion_probabilities = mapped_weights / np.sum(mapped_weights)
    recovar_probabilities = recovar["weights"].reshape(mapped_weights.shape)
    recovar_probabilities = recovar_probabilities / np.sum(recovar_probabilities)
    posterior_tv = float(
        0.5 * np.sum(np.abs(recovar_probabilities - relion_probabilities))
    )
    relion_with_prior = np.full(mapped_weights.shape, -np.inf, dtype=np.float64)
    positive = mapped_weights > 0.0
    relion_with_prior[positive] = np.log(mapped_weights[positive])
    common_positive = positive & np.isfinite(recovar["scores_with_prior"])
    _require(np.any(common_positive), "no common positive posterior support")
    with_prior_stats = _stats(
        _center_max(recovar["scores_with_prior"][common_positive])
        - _center_max(relion_with_prior[common_positive])
    )
    recovar_significant = recovar["significant_mask"].reshape(mapped_significant.shape)
    significant_mismatch_count = int(
        np.count_nonzero(mapped_significant != recovar_significant)
    )
    parent_mismatch_count = int(
        np.count_nonzero(
            np.any(mapped_significant, axis=1)
            != np.any(recovar_significant, axis=1)
        )
    )
    raw_pass = raw_stats["p95_abs"] <= SCORE_P95_GATE and raw_stats["max_abs"] < SCORE_MAX_GATE
    prior_pass = with_prior_stats["p95_abs"] <= SCORE_P95_GATE and with_prior_stats["max_abs"] < SCORE_MAX_GATE
    stage_exact = {
        "current_size": True,
        "prior_support": bool(np.array_equal(relion_prior_support, recovar_prior_support)),
        "raw_scores": bool(raw_pass),
        "scores_with_prior": bool(prior_pass),
        "posterior": bool(posterior_tv <= POSTERIOR_TV_GATE),
        "significant_support": significant_mismatch_count == 0,
        "winner": int(np.argmax(mapped_weights)) == int(np.argmax(recovar["weights"])),
    }
    first_unequal = next((name for name, equal in stage_exact.items() if not equal), "coarse_boundary_exact")
    complete_coarse_boundary = {
        "first_unequal_stage": first_unequal,
        "stage_exact": stage_exact,
        "prior_support_mismatch_count": int(
            np.count_nonzero(relion_prior_support != recovar_prior_support)
        ),
        "raw_centered_score_diff": raw_stats,
        "raw_score_bitwise": raw_bitwise,
        "raw_score_centered_bitwise": raw_centered_bitwise,
        "with_prior_centered_score_diff": with_prior_stats,
        "posterior_total_variation": posterior_tv,
        "posterior_max_abs": float(
            np.max(np.abs(recovar_probabilities - relion_probabilities))
        ),
        "significant_candidate_mismatch_count": significant_mismatch_count,
        "significant_parent_mismatch_count": parent_mismatch_count,
        "relion_significant_count": int(np.count_nonzero(mapped_significant)),
        "recovar_significant_count": int(np.count_nonzero(recovar_significant)),
        "top_posterior_records": _top_posterior_records(
            native_raw_score=-mapped_raw,
            recovar_raw_score=recovar["scores"],
            native_log_weight=relion_with_prior,
            recovar_log_weight=recovar["scores_with_prior"],
            native_posterior=relion_probabilities,
            recovar_posterior=recovar_probabilities,
            native_significant=mapped_significant,
            recovar_significant=recovar_significant,
        ),
        "significant_mismatch_records_first_64": _complete_support_mismatch_records(
            native_significant=mapped_significant,
            recovar_significant=recovar_significant,
            native_raw_diff2=mapped_raw,
            recovar_raw_score=recovar["scores"],
            native_posterior=relion_probabilities,
            recovar_posterior=recovar_probabilities,
        ),
    }

    # These live inputs are present even when the optional RECOVAR projected-
    # reference quartet was not captured.  Compare them before that optional
    # gate so a bounded run can still localize a discrepancy to particle
    # preprocessing, translation, pixel weighting, or the initial diff2 term.
    score_indices = recovar["exact_score_indices"]
    native_exact_shifted = -np.float64(physical_image_size**2) * _native_fftw_values_on_recovar_grid(
        live.shifted_real.astype(np.float64)
        + 1j * live.shifted_imag.astype(np.float64),
        score_indices,
        physical_image_size=physical_image_size,
        native_current_size=native_coarse_image_size,
    ).astype(np.complex64)
    native_exact_shifted_ordered = np.empty_like(native_exact_shifted)
    native_exact_shifted_ordered[translation_permutation] = native_exact_shifted
    native_exact_unshifted = (
        _native_fftw_values_on_recovar_grid(
            (
                operand.image_real.astype(np.float64)
                + 1j * operand.image_imag.astype(np.float64)
            )[np.newaxis, :],
            score_indices,
            physical_image_size=physical_image_size,
            native_current_size=native_coarse_image_size,
        )[0].astype(np.complex64)
        * np.float32(-(physical_image_size**2))
    )
    native_exact_pixel_weight = _native_fftw_values_on_recovar_grid(
        (np.float32(2.0) * live.correction_half)[np.newaxis, :],
        score_indices,
        physical_image_size=physical_image_size,
        native_current_size=native_coarse_image_size,
    )[0].real.astype(np.float32) / np.float32(physical_image_size**4)
    recovar_exact_shifted = recovar["exact_shifted"]
    recovar_exact_pixel_weight = recovar["exact_pixel_weight"]
    _require(
        recovar_exact_shifted.shape == native_exact_shifted_ordered.shape,
        "exact shifted-image topology mismatch",
    )
    _require(
        recovar["exact_unshifted"].shape == native_exact_unshifted.shape,
        "exact unshifted-image topology mismatch",
    )
    _require(
        recovar_exact_pixel_weight.shape == native_exact_pixel_weight.shape,
        "exact pixel-weight topology mismatch",
    )
    active_exact_pixels = recovar_exact_pixel_weight != 0.0
    _require(np.any(active_exact_pixels), "exact pixel-weight operand is empty")
    shifted_comparison = _active_pixel_operand_comparison(
        native_exact_shifted_ordered,
        recovar_exact_shifted,
        active_exact_pixels,
    )
    unshifted_comparison = _active_pixel_operand_comparison(
        native_exact_unshifted,
        recovar["exact_unshifted"],
        active_exact_pixels,
    )
    pixel_weight_comparison = _operand_comparison(
        native_exact_pixel_weight,
        recovar_exact_pixel_weight,
    )
    native_initial_diff2 = np.asarray(np.uint32(lanes.header[20])).view(np.float32).item()
    recovar_initial_diff2 = np.float32(recovar["exact_initial_diff2"])
    exact_input_stage = {
        "unshifted_corrected_active_pixels": (
            unshifted_comparison["bitwise_equal_count"] == unshifted_comparison["value_count"]
        ),
        "shifted_corrected_active_pixels": (
            shifted_comparison["bitwise_equal_count"] == shifted_comparison["value_count"]
        ),
        "pixel_weight": (
            pixel_weight_comparison["bitwise_equal_count"]
            == pixel_weight_comparison["value_count"]
        ),
        "initial_diff2": bool(
            np.asarray(native_initial_diff2, dtype=np.float32).view(np.uint32)
            == np.asarray(recovar_initial_diff2, dtype=np.float32).view(np.uint32)
        ),
    }
    exact_input_boundary = {
        "stage_order": list(exact_input_stage),
        "stage_exact": exact_input_stage,
        "first_exact_unequal_boundary": next(
            (name for name, equal in exact_input_stage.items() if not equal),
            "all_exact_input_operands_equal",
        ),
        "unshifted_corrected_active_pixels": unshifted_comparison,
        "shifted_corrected_active_pixels": shifted_comparison,
        "pixel_weight": pixel_weight_comparison,
        "initial_diff2": {
            "native": float(native_initial_diff2),
            "native_bits": int(lanes.header[20]),
            "recovar": float(recovar_initial_diff2),
            "recovar_bits": int(np.asarray(recovar_initial_diff2).view(np.uint32)),
        },
    }
    if not recovar["projection_capture_available"]:
        return {
            "stack_index_one_based": components.stack_index,
            "original_index_zero_based": recovar["original_index"],
            "relion_part_id": components.part_id,
            "complete_coarse_boundary": complete_coarse_boundary,
            "exact_input_operand_boundary": exact_input_boundary,
            "operand_decomposition": {
                "status": "not_captured",
                "reason": (
                    "passive native-texture capture omitted optional "
                    "preprojected-reference diagnostic arrays"
                ),
                "missing_fields": recovar["missing_projection_fields"],
            },
            "artifacts": {
                "components": str(components_path.resolve()),
                "components_sha256": components.sha256,
                "operands": str(operand_path.resolve()),
                "operands_sha256": operand.sha256,
                "lanes": str(lane_path.resolve()),
                "lanes_sha256": lanes.sha256,
                "live": str(live_path.resolve()),
                "live_sha256": live.sha256,
                "recovar": str(recovar["path"]),
                "recovar_sha256": recovar["sha256"],
            },
        }
    rotation_ids = recovar["rotation_ids"]
    active = np.all(mapped_raw[rotation_ids] != RELION_INVALID_DIFF2, axis=1)
    _require(np.any(active), "no active requested component rotations")
    component_request_positions = np.flatnonzero(active)
    component_ids = rotation_ids[component_request_positions]
    component_recovar_norm = recovar["norms"][component_request_positions]
    component_recovar_cross = recovar["crosses"][component_request_positions]
    component_recovar_total = recovar["scores"][component_ids]
    component_decomposition = _component_decomposition(
        component_recovar_total + mapped_raw[component_ids],
        component_recovar_norm + mapped_norm[component_ids],
        component_recovar_cross + mapped_cross[component_ids],
    )
    component_support_mismatches = _support_mismatch_panel(
        rotation_ids=component_ids,
        native_raw=mapped_raw[component_ids],
        native_norm=mapped_norm[component_ids],
        native_cross=mapped_cross[component_ids],
        native_significant=mapped_significant[component_ids],
        recovar_raw=component_recovar_total,
        recovar_norm=component_recovar_norm,
        recovar_cross=component_recovar_cross,
        recovar_significant=recovar_significant[component_ids],
    )
    component_rows = _support_mismatch_panel(
        rotation_ids=component_ids,
        native_raw=mapped_raw[component_ids],
        native_norm=mapped_norm[component_ids],
        native_cross=mapped_cross[component_ids],
        native_significant=mapped_significant[component_ids],
        recovar_raw=component_recovar_total,
        recovar_norm=component_recovar_norm,
        recovar_cross=component_recovar_cross,
        recovar_significant=recovar_significant[component_ids],
        support_mismatches_only=False,
    )
    mapped_operand_ids = np.asarray(
        [
            _rotation_key_to_recovar(key, n_directions, n_psi)
            for key in operand.rotation_keys
        ],
        dtype=np.int64,
    )
    active_operand_overlap = np.isin(rotation_ids, mapped_operand_ids) & active
    if not np.any(active_operand_overlap):
        return {
            "stack_index_one_based": components.stack_index,
            "original_index_zero_based": recovar["original_index"],
            "relion_part_id": components.part_id,
            "active_requested_rotation_count": int(np.count_nonzero(active)),
            "requested_rotation_count": int(rotation_ids.size),
            "matched_active_operand_rotation_count": 0,
            "matched_recovar_rotation_ids": [],
            "recovar_only_requested_rotation_ids": sorted(rotation_ids.tolist()),
            "native_only_captured_rotation_ids": sorted(mapped_operand_ids.tolist()),
            "component_rotation_ids": component_ids.tolist(),
            "complete_coarse_boundary": complete_coarse_boundary,
            "exact_input_operand_boundary": exact_input_boundary,
            "component_decomposition": component_decomposition,
            "captured_component_rows": component_rows,
            "captured_support_mismatches": component_support_mismatches,
            "operand_texture_panel": {
                "status": "not_captured",
                "reason": (
                    "native texture rotation keys do not overlap the active "
                    "RECOVAR projected-reference panel"
                ),
            },
            "artifacts": {
                "components": str(components_path.resolve()),
                "components_sha256": components.sha256,
                "operands": str(operand_path.resolve()),
                "operands_sha256": operand.sha256,
                "lanes": str(lane_path.resolve()),
                "lanes_sha256": lanes.sha256,
                "live": str(live_path.resolve()),
                "live_sha256": live.sha256,
                "recovar": str(recovar["path"]),
                "recovar_sha256": recovar["sha256"],
            },
        }
    (
        selected_ids,
        selected_request_positions,
        operand_order,
        recovar_only_rotation_ids,
        native_only_rotation_ids,
    ) = _matched_operand_rotation_panel(rotation_ids, mapped_operand_ids, active)
    recovar_norm = recovar["norms"][selected_request_positions]
    recovar_cross = recovar["crosses"][selected_request_positions]
    recovar_references = recovar["references"][selected_request_positions]
    recovar_total = recovar["scores"][selected_ids]
    native_reference = relion_reference_on_recovar_window(
        (
            operand.reference_real.astype(np.float64)
            + 1j * operand.reference_imag.astype(np.float64)
        )[operand_order],
        recovar["window_indices"],
        full_image_size=physical_image_size,
        current_size=native_coarse_image_size,
    )
    native_shifted = relion_values_on_recovar_window(
        operand.shifted_real.astype(np.float64)
        + 1j * operand.shifted_imag.astype(np.float64),
        recovar["window_indices"],
        full_image_size=physical_image_size,
        current_size=native_coarse_image_size,
    )
    native_shifted_ordered = np.empty_like(native_shifted)
    native_shifted_ordered[translation_permutation] = native_shifted
    native_correction = relion_values_on_recovar_window(
        operand.correction[np.newaxis, :],
        recovar["window_indices"],
        full_image_size=physical_image_size,
        current_size=native_coarse_image_size,
    )[0].real
    native_exact_reference = -np.float64(physical_image_size**2) * _native_fftw_values_on_recovar_grid(
        (
            live.reference_real.astype(np.float64)
            + 1j * live.reference_imag.astype(np.float64)
        )[np.newaxis, :],
        score_indices,
        physical_image_size=physical_image_size,
        native_current_size=native_coarse_image_size,
    ).astype(np.complex64)
    score_position = {int(index): position for position, index in enumerate(score_indices)}
    _require(
        all(int(index) in score_position for index in recovar["window_indices"]),
        "window indices are not a subset of exact score indices",
    )
    window_positions = np.asarray(
        [score_position[int(index)] for index in recovar["window_indices"]],
        dtype=np.int64,
    )
    recovar_exact_reference = np.zeros_like(native_exact_reference)
    recovar_exact_reference[:, window_positions] = recovar_references
    _require(
        np.all(active_exact_pixels[window_positions]),
        "window includes an inactive exact-score pixel",
    )
    _require(
        np.count_nonzero(active_exact_pixels) == window_positions.size,
        "active exact-score pixels do not match the projection window",
    )

    lane_replay: dict[str, Any]
    if (
        selected_ids.size == operand.rotation_keys.size == 1
        and np.array_equal(operand_order, np.asarray([0]))
        and np.array_equal(lanes.rotation_keys, operand.rotation_keys)
        and int(live.header[14]) == int(operand.rotation_keys[0])
    ):
        recovar_reference_native_order = _compact_to_native_full_order(
            recovar_exact_reference,
            score_indices,
            physical_image_size=physical_image_size,
            current_size=native_coarse_image_size,
        )
        recovar_shifted_native_order = _compact_to_native_full_order(
            recovar_exact_shifted,
            score_indices,
            physical_image_size=physical_image_size,
            current_size=native_coarse_image_size,
        )
        recovar_shifted_native_order = recovar_shifted_native_order[
            translation_permutation
        ]
        recovar_weight_native_order = _compact_to_native_full_order(
            recovar_exact_pixel_weight,
            score_indices,
            physical_image_size=physical_image_size,
            current_size=native_coarse_image_size,
        )
        image_normalization_f32 = np.float32(physical_image_size**2)
        recovar_reference_live_units = np.asarray(
            recovar_reference_native_order[0] / -image_normalization_f32,
            dtype=np.complex64,
        )
        recovar_shifted_live_units = np.asarray(
            recovar_shifted_native_order / -image_normalization_f32,
            dtype=np.complex64,
        )
        recovar_correction_half_live_units = np.asarray(
            recovar_weight_native_order
            * np.float32(physical_image_size**4)
            * np.float32(0.5),
            dtype=np.float32,
        )
        native_initial_diff2 = np.asarray(
            np.uint32(lanes.header[20])
        ).view(np.float32).item()
        recovar_initial_diff2 = np.float32(recovar["exact_initial_diff2"])
        live_variants = {
            "native_all": (live, native_initial_diff2),
            "recovar_initial_diff2_only": (live, recovar_initial_diff2),
            "recovar_projected_reference_only": (
                replace(
                    live,
                    reference_real=recovar_reference_live_units.real,
                    reference_imag=recovar_reference_live_units.imag,
                ),
                native_initial_diff2,
            ),
            "recovar_shifted_image_only": (
                replace(
                    live,
                    shifted_real=recovar_shifted_live_units.real,
                    shifted_imag=recovar_shifted_live_units.imag,
                ),
                native_initial_diff2,
            ),
            "recovar_pixel_weight_only": (
                replace(
                    live,
                    correction_half=recovar_correction_half_live_units,
                ),
                native_initial_diff2,
            ),
            "recovar_all": (
                replace(
                    live,
                    reference_real=recovar_reference_live_units.real,
                    reference_imag=recovar_reference_live_units.imag,
                    shifted_real=recovar_shifted_live_units.real,
                    shifted_imag=recovar_shifted_live_units.imag,
                    correction_half=recovar_correction_half_live_units,
                ),
                recovar_initial_diff2,
            ),
        }
        mismatch_positions = np.argwhere(
            mapped_significant[selected_ids] != recovar_significant[selected_ids]
        )
        mismatch_translation_ids = sorted(
            {int(position[1]) for position in mismatch_positions}
        )
        selected_translation_pair = translation_pair_recovar
        if selected_translation_pair is None and len(mismatch_translation_ids) == 2:
            selected_translation_pair = (
                mismatch_translation_ids[0],
                mismatch_translation_ids[1],
            )
        if selected_translation_pair is not None:
            first_selected, second_selected = selected_translation_pair
            _require(
                0 <= first_selected < recovar_total.shape[1]
                and 0 <= second_selected < recovar_total.shape[1]
                and first_selected != second_selected,
                "selected RECOVAR translation pair is invalid",
            )
        selected_translation_ids = sorted(
            set(mismatch_translation_ids)
            | (set(selected_translation_pair) if selected_translation_pair is not None else set())
        )
        lane_replay = {}
        inverse_translation_permutation = np.argsort(translation_permutation)
        for label, (variant, initial_diff2) in live_variants.items():
            replayed_lanes = replay_live_lanes(variant)
            log_scores_native_order = _log_scores_from_lane_partials(
                replayed_lanes,
                translation_count=int(live.header[13]),
            )
            log_scores_recovar_order = np.empty_like(log_scores_native_order)
            log_scores_recovar_order[translation_permutation] = log_scores_native_order
            selected_scores = {
                str(translation): float(log_scores_recovar_order[translation])
                for translation in selected_translation_ids
            }
            row: dict[str, Any] = {
                "versus_native_production_lanes": _operand_comparison(
                    lanes.lane_partials[0],
                    replayed_lanes,
                ),
                "initial_diff2": float(initial_diff2),
                "selected_translation_log_scores_without_initial_term": selected_scores,
            }
            if selected_translation_pair is not None:
                first, second = selected_translation_pair
                row["selected_translation_pair_recovar"] = [int(first), int(second)]
                row["second_minus_first_selected_translation_log_score"] = float(
                    log_scores_recovar_order[second]
                    - log_scores_recovar_order[first]
                )
                atomic_report, possible_margins = _atomic_relative_score_envelope(
                    replayed_lanes,
                    translation_count=int(live.header[13]),
                    first_translation=int(inverse_translation_permutation[first]),
                    second_translation=int(inverse_translation_permutation[second]),
                    initial_diff2=np.float32(initial_diff2),
                )
                native_margin = float(
                    -mapped_raw[selected_ids[0], second]
                    + mapped_raw[selected_ids[0], first]
                )
                recovar_margin = float(
                    recovar_total[0, second] - recovar_total[0, first]
                )
                atomic_report.update(
                    {
                        "first_translation_recovar": int(first),
                        "second_translation_recovar": int(second),
                        "native_production_relative_log_score": native_margin,
                        "native_production_is_legal_atomic_order": bool(
                            np.any(possible_margins == native_margin)
                        ),
                        "recovar_production_relative_log_score": recovar_margin,
                        "recovar_production_is_legal_atomic_order": bool(
                            np.any(possible_margins == recovar_margin)
                        ),
                    }
                )
                row["atomic_add_order_envelope"] = atomic_report
            lane_replay[label] = row
    else:
        lane_replay = {
            "status": "not computed: captured rotation order is not a complete lane panel"
        }
    image_normalization = float(physical_image_size**2)
    native_weighted_shifted = (
        -native_shifted_ordered
        * native_correction[np.newaxis, :]
        / (image_normalization * recovar["half_weights"][np.newaxis, :])
    )
    native_ctf2 = native_correction / (
        image_normalization**2 * recovar["half_weights"]
    )
    configurations = {
        "projected_reference": (native_reference, recovar["shifted"], recovar["ctf2"]),
        "weighted_shifted_image": (recovar_references, native_weighted_shifted, recovar["ctf2"]),
        "correction": (recovar_references, recovar["shifted"], native_ctf2),
        "all_native": (native_reference, native_weighted_shifted, native_ctf2),
    }
    total_residual = recovar_total + mapped_raw[selected_ids]
    counterfactuals: dict[str, Any] = {}
    for label, (reference, shifted, ctf2) in configurations.items():
        norm, cross = recovar_score_components(
            reference,
            shifted,
            ctf2,
            recovar["half_weights"],
        )
        counterfactuals[label] = _candidate_panel_counterfactual(
            total_residual,
            norm + cross + mapped_raw[selected_ids],
        )
    replay_norm, replay_cross = recovar_score_components(
        recovar_references,
        recovar["shifted"],
        recovar["ctf2"],
        recovar["half_weights"],
    )
    replay_error = replay_norm + replay_cross - recovar_total
    return {
        "stack_index_one_based": components.stack_index,
        "original_index_zero_based": recovar["original_index"],
        "relion_part_id": components.part_id,
        "native_coarse_image_size": native_coarse_image_size,
        "recovar_score_size": recovar_score_size,
        "score_size_relation": score_size_relation,
        "active_requested_rotation_count": int(np.count_nonzero(active)),
        "requested_rotation_count": int(rotation_ids.size),
        "matched_active_operand_rotation_count": int(selected_ids.size),
        "matched_recovar_rotation_ids": selected_ids.tolist(),
        "recovar_only_requested_rotation_ids": recovar_only_rotation_ids,
        "native_only_captured_rotation_ids": native_only_rotation_ids,
        "component_rotation_ids": component_ids.tolist(),
        "complete_coarse_boundary": complete_coarse_boundary,
        "exact_input_operand_boundary": exact_input_boundary,
        "component_decomposition": component_decomposition,
        "captured_component_rows": component_rows,
        "captured_support_mismatches": component_support_mismatches,
        "recovar_component_replay": {
            "p95_abs": float(np.percentile(np.abs(replay_error), 95)),
            "max_abs": float(np.max(np.abs(replay_error))),
        },
        "operand_relative_l2": {
            "projected_reference": _relative_l2(native_reference, recovar_references),
            "weighted_shifted_image": _relative_l2(native_weighted_shifted, recovar["shifted"]),
            "correction": _relative_l2(native_ctf2, recovar["ctf2"]),
        },
        "exact_coarse_operands": {
            "projected_reference_active_pixels": _active_pixel_operand_comparison(
                native_exact_reference,
                recovar_exact_reference,
                active_exact_pixels,
            ),
            "shifted_corrected_active_pixels": _active_pixel_operand_comparison(
                native_exact_shifted_ordered,
                recovar_exact_shifted,
                active_exact_pixels,
            ),
            "unshifted_corrected_active_pixels": _active_pixel_operand_comparison(
                native_exact_unshifted,
                recovar["exact_unshifted"],
                active_exact_pixels,
            ),
            "pixel_weight": _operand_comparison(
                native_exact_pixel_weight,
                recovar_exact_pixel_weight,
            ),
            "initial_diff2": {
                "native_bits": int(lanes.header[20]),
                "recovar": float(recovar["exact_initial_diff2"]),
            },
            "recovar_operands_replayed_with_native_lane_topology": lane_replay,
        },
        "counterfactuals": counterfactuals,
        "translation_mapping": translation_mapping,
        "artifacts": {
            "components": str(components_path.resolve()),
            "components_sha256": components.sha256,
            "operands": str(operand_path.resolve()),
            "operands_sha256": operand.sha256,
            "lanes": str(lane_path.resolve()),
            "lanes_sha256": lanes.sha256,
            "live": str(live_path.resolve()),
            "live_sha256": live.sha256,
            "recovar": str(recovar["path"]),
            "recovar_sha256": recovar["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())
    physical_iteration = int(selection["physical_iteration"])
    rows = []
    for target in selection["targets"]:
        stack = int(target["stack_index_one_based"])
        original = int(target["original_index_zero_based"])
        component_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-v2.bin"))
        operand_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-op-v2.bin"))
        lane_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-lane-v1.bin"))
        live_paths = list(args.native_directory.glob(f"part*_stack{stack}.p1-live-v1.bin"))
        recovar_paths = list(args.recovar_directory.glob(f"significance_orig{original:06d}*_cs*.npz"))
        _require(
            len(component_paths)
            == len(operand_paths)
            == len(lane_paths)
            == len(live_paths)
            == len(recovar_paths)
            == 1,
            f"artifact lookup failed for stack {stack}",
        )
        rows.append(
            _compare(
                component_paths[0],
                operand_paths[0],
                lane_paths[0],
                live_paths[0],
                recovar_paths[0],
                physical_iteration=physical_iteration,
                physical_image_size=args.physical_image_size,
                translation_pair_recovar=(
                    tuple(int(value) for value in target["translation_pair_recovar"])
                    if "translation_pair_recovar" in target
                    else None
                ),
            )
        )
    validation_dir = args.native_directory.resolve().parent / "analysis"
    validation_candidates = {
        "components": (validation_dir / "components_validation.json",),
        "operands": (
            validation_dir / "operand_validation.json",
            validation_dir / "operands_validation.json",
        ),
        "lanes": (
            validation_dir / "lane_validation.json",
            validation_dir / "lanes_validation.json",
        ),
    }
    capture_validation: dict[str, Any] = {}
    classification_ready = True
    for label, candidates in validation_candidates.items():
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if path.is_file():
            payload = json.loads(path.read_text())
            ready = bool(payload.get("classification_ready", False))
            capture_validation[label] = {
                "path": str(path),
                "sha256": _sha256(path),
                "status": payload.get("status"),
                "classification_ready": ready,
            }
            classification_ready = classification_ready and ready
        else:
            capture_validation[label] = {
                "path": str(path),
                "status": "missing",
                "classification_ready": False,
            }
            classification_ready = False
    report = {
        "schema": "recovar.em.k1_coarse_operand_boundary.v3",
        "case_id": int(selection["case_id"]),
        "physical_iteration": physical_iteration,
        "physical_image_size": int(args.physical_image_size),
        "metric_policy": "scale-sensitive relative-L2 and centered residual energy; no correlation",
        "classification_ready": classification_ready,
        "capture_validation": capture_validation,
        "particles": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
