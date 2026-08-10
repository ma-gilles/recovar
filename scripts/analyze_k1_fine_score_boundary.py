#!/usr/bin/env python3
"""Localize a K=1 RELION/RECOVAR fine-score boundary on a fixed panel.

The native capture contains every sparse fine-score tuple but no dense BPref
pixel factors.  This makes it suitable for particles with million-sample
significant supports.  Intermediate acceptance uses exact equality and
relative L2 only; signed shellwise FSC/FSC-AUC remains the map metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import starfile

if __package__:
    from .validate_relion_bpref_factor_capture import fnv1a64, load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        fnv1a64,
        load_factor_capture,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )


SELECTION_SCHEMAS = {
    "recovar.em.k1_fine_score_panel.v1",
    "recovar.em.k1_bpref_factor_panel.v1",
}
INERTNESS_SCHEMAS = {
    "recovar.em.k1_fine_score_capture_inertness.v1",
    "recovar.em.k1_bpref_factor_capture_inertness.v1",
}
REPORT_SCHEMA = "recovar.em.k1_fine_score_boundary.v2"
RELATIVE_L2_BOUND = 1.0e-5
EXACT_BOUNDARY_ORDER = (
    "active_candidate_tuples",
    "raw_diff2",
    "orientation_log_prior",
    "translation_log_prior",
    "combined_log_weight_centered",
    "normalized_posterior_native_active",
    "significant_support",
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


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    _require(left.shape == right.shape and left.size > 0, "comparison shape changed or is empty")
    delta = right.astype(np.float64) - left.astype(np.float64)
    denominator = max(float(np.linalg.norm(left.astype(np.float64))), np.finfo(np.float64).tiny)
    return {
        "shape": list(left.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _float32_ulp_distance(reference: float, candidate: float) -> int | None:
    values = np.asarray([reference, candidate], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return None
    bits = values.view(np.uint32).astype(np.int64)
    ordered = np.where(
        (bits & 0x80000000) != 0,
        0x80000000 - (bits & 0x7FFFFFFF),
        0x80000000 + bits,
    )
    return int(abs(ordered[1] - ordered[0]))


def _first_mismatch_record(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tuple_keys: np.ndarray,
) -> dict[str, Any] | None:
    """Return the first exact mismatch with its stable class-pose key."""

    left = np.asarray(reference).reshape(-1)
    right = np.asarray(candidate).reshape(-1)
    keys = np.asarray(tuple_keys, dtype=np.int64).reshape(-1, 2)
    _require(left.shape == right.shape, "first-mismatch arrays changed shape")
    _require(left.size == keys.shape[0], "first-mismatch tuple key count changed")
    mismatch = np.flatnonzero(left != right)
    if mismatch.size == 0:
        return None
    index = int(mismatch[0])
    reference_value = float(left[index])
    candidate_value = float(right[index])
    return {
        "flat_index": index,
        "recovar_rotation_row": int(keys[index, 0]),
        "recovar_translation_row": int(keys[index, 1]),
        "reference_value": reference_value,
        "candidate_value": candidate_value,
        "absolute_delta": abs(candidate_value - reference_value),
        "reference_float32_bits": int(np.asarray(reference_value, dtype=np.float32).view(np.uint32)),
        "candidate_float32_bits": int(np.asarray(candidate_value, dtype=np.float32).view(np.uint32)),
        "float32_ulp_distance": _float32_ulp_distance(reference_value, candidate_value),
    }


def _first_exact_boundary(stage_equal: dict[str, bool]) -> str:
    _require(
        set(stage_equal) == set(EXACT_BOUNDARY_ORDER),
        "exact-boundary stage set changed",
    )
    return next(
        (name for name in EXACT_BOUNDARY_ORDER if not stage_equal[name]),
        "fine_score_boundary_exact",
    )


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "cannot center empty or non-finite values")
    return array - np.max(array)


def _relion_full_to_compact_lookup(
    *,
    physical_image_size: int,
    current_size: int,
    compact_indices: np.ndarray,
) -> np.ndarray:
    original_half_width = physical_image_size // 2 + 1
    current_half_width = current_size // 2 + 1
    indices = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    centered_rows = indices // original_half_width
    columns = indices % original_half_width
    ky = centered_rows - physical_image_size // 2
    fftw_rows = np.where(ky < 0, ky + current_size, ky)
    _require(
        np.all((0 <= fftw_rows) & (fftw_rows < current_size)),
        "compact score rows fall outside the RELION current-size crop",
    )
    _require(
        np.all((0 <= columns) & (columns < current_half_width)),
        "compact score columns fall outside the RELION current-size crop",
    )
    relion_indices = fftw_rows * current_half_width + columns
    _require(
        np.unique(relion_indices).size == relion_indices.size,
        "compact score rows do not map uniquely to RELION packed order",
    )
    lookup = np.full(current_size * current_half_width, -1, dtype=np.int32)
    lookup[relion_indices] = np.arange(indices.size, dtype=np.int32)
    return lookup


def _raw_diff2_terms(
    reference: np.ndarray,
    shifted_image: np.ndarray,
    pixel_weight: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.complex64)
    shifted_image = np.asarray(shifted_image, dtype=np.complex64)
    pixel_weight = np.asarray(pixel_weight, dtype=np.float32)
    diff_real = np.subtract(reference.real, shifted_image.real, dtype=np.float32)
    diff_imag = np.subtract(reference.imag, shifted_image.imag, dtype=np.float32)
    imag_squared = np.multiply(diff_imag, diff_imag, dtype=np.float32)
    squared = np.asarray(
        diff_real.astype(np.float64) * diff_real.astype(np.float64)
        + imag_squared.astype(np.float64),
        dtype=np.float32,
    )
    half_squared = np.multiply(squared, np.float32(0.5), dtype=np.float32)
    return np.multiply(half_squared, pixel_weight, dtype=np.float32)


def _reduce_relion_fine_lanes(lanes: np.ndarray) -> tuple[np.float32, list[np.ndarray]]:
    values = np.asarray(lanes, dtype=np.float32).reshape(256)
    levels = []
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        values = np.add(values[:width], values[width : 2 * width], dtype=np.float32)
        levels.append(values.copy())
    return np.float32(values[0]), levels


def _replay_raw_diff2(
    *,
    rotations: np.ndarray,
    translations: np.ndarray,
    projected_references: np.ndarray,
    shifted_images: np.ndarray,
    ctf2_over_nv: np.ndarray,
    half_weights: np.ndarray,
    full_to_compact: np.ndarray,
    highres_xi2_half: float,
) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.int64).reshape(-1)
    translations = np.asarray(translations, dtype=np.int64).reshape(-1)
    _require(rotations.shape == translations.shape, "raw replay tuple arrays changed shape")
    references = np.asarray(projected_references, dtype=np.complex64)
    shifted = np.asarray(shifted_images, dtype=np.complex64)
    weights = np.multiply(
        np.asarray(ctf2_over_nv, dtype=np.float32),
        np.asarray(half_weights, dtype=np.float32),
        dtype=np.float32,
    )
    lookup = np.asarray(full_to_compact, dtype=np.int32).reshape(-1)
    block_size = 256
    n_passes = (lookup.size + block_size - 1) // block_size
    padded = np.pad(lookup, (0, n_passes * block_size - lookup.size), constant_values=-1)
    output = np.empty(rotations.size, dtype=np.float32)
    highres = np.float32(highres_xi2_half)
    for candidate, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        lanes = np.zeros(block_size, dtype=np.float32)
        for pass_index in range(n_passes):
            compact = padded[pass_index * block_size : (pass_index + 1) * block_size]
            valid = compact >= 0
            safe = np.where(valid, compact, 0)
            terms = _raw_diff2_terms(
                references[rotation, safe],
                shifted[translation, safe],
                weights[safe],
            )
            terms = np.where(valid, terms, np.float32(0.0)).astype(np.float32, copy=False)
            lanes = np.add(lanes, terms, dtype=np.float32)
        reduced, _ = _reduce_relion_fine_lanes(lanes)
        output[candidate] = np.add(reduced, highres, dtype=np.float32)
    return output


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _raw_diff2_trace(
    *,
    rotation: int,
    translation: int,
    projected_references: np.ndarray,
    shifted_images: np.ndarray,
    ctf2_over_nv: np.ndarray,
    half_weights: np.ndarray,
    full_to_compact: np.ndarray,
    highres_xi2_half: float,
) -> dict[str, Any]:
    references = np.asarray(projected_references, dtype=np.complex64)
    shifted = np.asarray(shifted_images, dtype=np.complex64)
    ctf2 = np.asarray(ctf2_over_nv, dtype=np.float32)
    multiplicity = np.asarray(half_weights, dtype=np.float32)
    weights = np.multiply(ctf2, multiplicity, dtype=np.float32)
    lookup = np.asarray(full_to_compact, dtype=np.int32).reshape(-1)
    block_size = 256
    n_passes = (lookup.size + block_size - 1) // block_size
    padded = np.pad(lookup, (0, n_passes * block_size - lookup.size), constant_values=-1)
    lanes = np.zeros(block_size, dtype=np.float32)
    pixel_term_hash = hashlib.sha256()
    for pass_index in range(n_passes):
        compact = padded[pass_index * block_size : (pass_index + 1) * block_size]
        valid = compact >= 0
        safe = np.where(valid, compact, 0)
        terms = _raw_diff2_terms(
            references[rotation, safe],
            shifted[translation, safe],
            weights[safe],
        )
        terms = np.where(valid, terms, np.float32(0.0)).astype(np.float32, copy=False)
        pixel_term_hash.update(memoryview(np.ascontiguousarray(terms)).cast("B"))
        lanes = np.add(lanes, terms, dtype=np.float32)
    reduced, levels = _reduce_relion_fine_lanes(lanes)
    highres = np.float32(highres_xi2_half)
    final = np.add(reduced, highres, dtype=np.float32)
    return {
        "recovar_rotation_row": int(rotation),
        "recovar_translation_row": int(translation),
        "full_packed_pixel_count": int(lookup.size),
        "compact_pixel_count": int(np.count_nonzero(lookup >= 0)),
        "pass_count": n_passes,
        "projected_reference_sha256": _array_sha256(references[rotation]),
        "shifted_image_sha256": _array_sha256(shifted[translation]),
        "ctf2_over_nv_sha256": _array_sha256(ctf2),
        "half_weights_sha256": _array_sha256(multiplicity),
        "pixel_weights_sha256": _array_sha256(weights),
        "full_to_compact_sha256": _array_sha256(lookup),
        "pixel_terms_by_pass_sha256": pixel_term_hash.hexdigest(),
        "lane_sums_before_tree": [float(value) for value in lanes],
        "tree_levels": [[float(value) for value in level] for level in levels],
        "raw_diff2_before_highres": float(reduced),
        "highres_xi2_half": float(highres),
        "raw_diff2_after_highres": float(final),
        "raw_diff2_after_highres_float32_bits": int(final.view(np.uint32)),
    }


def _stable_top_n_mask(weights: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    _require(np.all(np.isfinite(values)) and np.all(values >= 0), "support weights are invalid")
    _require(0 <= count <= values.size, "significant-support count exceeds active candidates")
    order = np.argsort(-values, kind="stable")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _rotation_map(factor_rotations: np.ndarray, recovar_rotations: np.ndarray) -> tuple[np.ndarray, float]:
    native = np.asarray(factor_rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    native = native.transpose(0, 2, 1)
    recovar = np.asarray(recovar_rotations, dtype=np.float32).reshape(-1, 3, 3)
    distance = np.max(np.abs(native[:, None] - recovar[None]), axis=(2, 3))
    nearest = np.argmin(distance, axis=1)
    error = distance[np.arange(native.shape[0]), nearest]
    _require(
        np.all(error <= 1.0e-6) and np.unique(nearest).size == nearest.size,
        "native and RECOVAR rotations do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _translation_map(
    factor_translations: np.ndarray,
    recovar_translations: np.ndarray,
    *,
    physical_image_size: int,
) -> tuple[np.ndarray, float]:
    native = np.column_stack((factor_translations["x"], factor_translations["y"])).astype(np.float64)
    recovar = -2.0 * np.pi * np.asarray(recovar_translations, dtype=np.float64) / physical_image_size
    distance = np.max(np.abs(native[:, None] - recovar[None]), axis=2)
    nearest = np.argmin(distance, axis=1)
    error = distance[np.arange(native.shape[0]), nearest]
    _require(
        np.all(error <= 1.0e-6) and np.unique(nearest).size == nearest.size,
        "native and RECOVAR translations do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(error, initial=0.0))


def _particle_table(path: Path):
    document = starfile.read(path)
    tables = [document] if hasattr(document, "columns") else list(document.values())
    matches = [table for table in tables if hasattr(table, "columns") and "rlnImageName" in table.columns]
    _require(len(matches) == 1, f"expected one particle table in {path}")
    table = matches[0].copy()
    identities = table["rlnImageName"].astype(str)
    _require(not identities.duplicated().any(), f"duplicate particle identity in {path}")
    return table.set_index(identities)


def _recovar_capture_path(directory: Path, original_index: int, current_size: int) -> Path:
    matches = sorted(directory.glob(f"pass2_orig{original_index:06d}_cs{current_size:03d}.npz"))
    matches += sorted(
        directory.glob(f"pass2_orig{original_index:06d}_class001_cs{current_size:03d}.npz")
    )
    _require(len(matches) == 1, f"expected one RECOVAR pass-2 capture for row {original_index}")
    return matches[0]


def _classify_particle(
    *,
    active_tuple_subset: bool,
    raw_diff2_close: bool,
    priors_close: bool,
    centered_log_weight_close: bool,
    posterior_close: bool,
    support_exact: bool,
) -> str:
    if not active_tuple_subset:
        return "active_candidate_tuple_mismatch"
    if not raw_diff2_close:
        return "raw_fine_diff2_mismatch"
    if not priors_close:
        return "fine_prior_mismatch"
    if not centered_log_weight_close:
        return "fine_log_weight_arithmetic_mismatch"
    if not posterior_close:
        return "fine_normalized_posterior_mismatch"
    if not support_exact:
        return "fine_significant_support_mismatch"
    return "fine_score_boundary_closes"


def _compare_particle(
    *,
    target: dict[str, Any],
    factor,
    score,
    recovar: dict[str, np.ndarray],
    native_state_row,
    physical_image_size: int,
    current_size: int,
) -> dict[str, Any]:
    stack = int(target["stack_index_one_based"])
    original = int(target["original_index_zero_based"])
    _require(factor.stack_index == score.stack_index == stack, f"stack {stack}: native identity changed")
    _require(int(recovar["original_index"]) == original, f"stack {stack}: RECOVAR identity changed")
    _require(int(recovar["current_size"]) == current_size, f"stack {stack}: current size changed")

    rotation_map, rotation_error = _rotation_map(factor.rotations, recovar["rotations"])
    translation_map, translation_error = _translation_map(
        factor.translations,
        recovar["fine_translations"],
        physical_image_size=physical_image_size,
    )
    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    active_rows = np.flatnonzero(active)
    _require(active_rows.size > 0, f"stack {stack}: native active fine support is empty")
    selected = candidates[active_rows]
    native_rotation = np.asarray(selected["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(selected["translation_id"], dtype=np.int64)
    _require(
        np.all(native_rotation < rotation_map.size) and np.all(native_translation < translation_map.size),
        f"stack {stack}: native tuple index exceeds geometry panel",
    )
    mapped_rotation = rotation_map[native_rotation]
    mapped_translation = translation_map[native_translation]
    mapped_keys = np.column_stack((mapped_rotation, mapped_translation))
    _require(
        np.unique(mapped_keys, axis=0).shape[0] == mapped_keys.shape[0],
        f"stack {stack}: mapped native tuples are duplicated",
    )

    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    captured_raw_diff2 = (
        None
        if "relion_raw_diff2" not in recovar
        else np.asarray(recovar["relion_raw_diff2"], dtype=np.float32)
    )
    scores_pre_prior = np.asarray(recovar["scores_pre_prior"], dtype=np.float64)
    scores_with_prior = np.asarray(recovar["scores_with_prior"], dtype=np.float64)
    recovar_probs = np.asarray(recovar["probs"], dtype=np.float64)
    reconstruction_mask = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    rotation_prior = np.asarray(recovar["rotation_log_prior"], dtype=np.float64)
    translation_prior = np.asarray(recovar["translation_log_prior"], dtype=np.float64)
    expected_shape = candidate_mask.shape
    for name, array in (
        ("scores_pre_prior", scores_pre_prior),
        ("scores_with_prior", scores_with_prior),
        ("probs", recovar_probs),
        ("reconstruction_mask", reconstruction_mask),
    ):
        _require(array.shape == expected_shape, f"stack {stack}: {name} shape changed")

    tuple_present = candidate_mask[mapped_rotation, mapped_translation]
    finite_preprior = np.isfinite(scores_pre_prior[mapped_rotation, mapped_translation])
    if captured_raw_diff2 is not None:
        _require(captured_raw_diff2.shape == expected_shape, f"stack {stack}: raw_diff2 shape changed")
        finite_preprior &= np.isfinite(captured_raw_diff2[mapped_rotation, mapped_translation])
    active_tuple_subset = bool(np.all(tuple_present & finite_preprior))
    comparable = tuple_present & finite_preprior
    _require(np.any(comparable), f"stack {stack}: no comparable native fine tuples")
    rr = mapped_rotation[comparable]
    tt = mapped_translation[comparable]
    native_rows = selected[comparable]
    comparable_keys = mapped_keys[comparable]

    native_raw = np.asarray(native_rows["raw_diff2"], dtype=np.float32)
    full_to_compact = _relion_full_to_compact_lookup(
        physical_image_size=physical_image_size,
        current_size=current_size,
        compact_indices=recovar["window_indices"],
    )
    replay_raw = _replay_raw_diff2(
        rotations=rr,
        translations=tt,
        projected_references=recovar["proj_half"],
        shifted_images=recovar["shifted_corrected"],
        ctf2_over_nv=recovar["ctf2_over_nv_score"],
        half_weights=recovar["half_weights"],
        full_to_compact=full_to_compact,
        highres_xi2_half=float(recovar["relion_highres_xi2_half"]),
    )
    native_preprior = -native_raw
    recovar_preprior = -replay_raw
    dumped_preprior = scores_pre_prior[rr, tt]
    native_rot_prior = np.asarray(native_rows["orientation_log_prior"], dtype=np.float32)
    recovar_rot_prior = rotation_prior[rr]
    native_trans_prior = np.asarray(native_rows["translation_log_prior"], dtype=np.float32)
    recovar_trans_prior = translation_prior[tt]
    native_log = np.asarray(native_rows["combined_preexponent"], dtype=np.float32)
    recovar_log = scores_with_prior[rr, tt]

    native_weight = np.asarray(selected["post_exponent_weight"], dtype=np.float64)
    native_weight_sum = float(np.sum(native_weight, dtype=np.float64))
    _require(native_weight_sum > 0 and np.isfinite(native_weight_sum), f"stack {stack}: invalid native weight sum")
    native_prob = native_weight / native_weight_sum
    recovar_prob = recovar_probs[mapped_rotation, mapped_translation]
    _require(np.all(np.isfinite(recovar_prob)), f"stack {stack}: RECOVAR posterior is non-finite")

    if factor.geometry_only:
        significant_count = int(native_state_row["rlnNrOfSignificantSamples"])
        native_support_flat = _stable_top_n_mask(native_weight, significant_count)
        native_support_keys = {
            (int(rotation), int(translation))
            for rotation, translation in mapped_keys[native_support_flat]
        }
        support_source = "native_state_significant_count_stable_top_n"
    else:
        accepted = factor.hypotheses[(factor.hypotheses["flags"] & 1) != 0]
        accepted_rotation = rotation_map[np.asarray(accepted["orientation_local"], dtype=np.int64)]
        accepted_translation = translation_map[np.asarray(accepted["translation"], dtype=np.int64)]
        native_support_keys = {
            (int(rotation), int(translation))
            for rotation, translation in zip(accepted_rotation, accepted_translation)
        }
        significant_count = len(native_support_keys)
        _require(
            significant_count == accepted.size,
            f"stack {stack}: full factor support contains duplicate tuples",
        )
        support_source = "full_prescatter_factor_accepted_hypotheses"
    recovar_support_keys = {
        (int(rotation), int(translation))
        for rotation, translation in np.argwhere(reconstruction_mask)
    }
    support_exact = native_support_keys == recovar_support_keys

    comparisons = {
        "raw_diff2": _metric(native_raw, replay_raw),
        "preprior_score_centered": _metric(_center(native_preprior), _center(recovar_preprior)),
        "dumped_preprior_score_centered": _metric(
            _center(recovar_preprior), _center(dumped_preprior)
        ),
        "orientation_log_prior": _metric(native_rot_prior, recovar_rot_prior),
        "translation_log_prior": _metric(native_trans_prior, recovar_trans_prior),
        "combined_log_weight_centered": _metric(_center(native_log), _center(recovar_log)),
        "normalized_posterior_native_active": _metric(native_prob, recovar_prob),
    }
    if captured_raw_diff2 is not None:
        comparisons["captured_recovar_raw_diff2_to_replay"] = _metric(
            captured_raw_diff2[rr, tt], replay_raw
        )
    raw_diff2_close = comparisons["raw_diff2"]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
    priors_close = all(
        comparisons[name]["relative_l2_over_reference"] <= RELATIVE_L2_BOUND
        for name in ("orientation_log_prior", "translation_log_prior")
    )
    centered_log_weight_close = (
        comparisons["combined_log_weight_centered"]["relative_l2_over_reference"]
        <= RELATIVE_L2_BOUND
    )
    posterior_close = (
        comparisons["normalized_posterior_native_active"]["relative_l2_over_reference"]
        <= RELATIVE_L2_BOUND
    )
    classification = _classify_particle(
        active_tuple_subset=active_tuple_subset,
        raw_diff2_close=raw_diff2_close,
        priors_close=priors_close,
        centered_log_weight_close=centered_log_weight_close,
        posterior_close=posterior_close,
        support_exact=support_exact,
    )
    first_missing = np.flatnonzero(~tuple_present | ~finite_preprior)
    first_missing_record = None
    if first_missing.size:
        missing_index = int(first_missing[0])
        first_missing_record = {
            "native_active_row": missing_index,
            "recovar_rotation_row": int(mapped_keys[missing_index, 0]),
            "recovar_translation_row": int(mapped_keys[missing_index, 1]),
            "candidate_present": bool(tuple_present[missing_index]),
            "finite_preprior_score": bool(finite_preprior[missing_index]),
        }
    native_support_only = sorted(native_support_keys - recovar_support_keys)
    recovar_support_only = sorted(recovar_support_keys - native_support_keys)
    first_mismatches = {
        "active_candidate_tuples": first_missing_record,
        "raw_diff2": _first_mismatch_record(
            native_raw,
            replay_raw,
            tuple_keys=comparable_keys,
        ),
        "orientation_log_prior": _first_mismatch_record(
            native_rot_prior,
            recovar_rot_prior,
            tuple_keys=comparable_keys,
        ),
        "translation_log_prior": _first_mismatch_record(
            native_trans_prior,
            recovar_trans_prior,
            tuple_keys=comparable_keys,
        ),
        "combined_log_weight_centered": _first_mismatch_record(
            _center(native_log),
            _center(recovar_log),
            tuple_keys=comparable_keys,
        ),
        "normalized_posterior_native_active": _first_mismatch_record(
            native_prob,
            recovar_prob,
            tuple_keys=mapped_keys,
        ),
        "significant_support": (
            None
            if support_exact
            else {
                "first_native_only_key": list(native_support_only[0]) if native_support_only else None,
                "first_recovar_only_key": list(recovar_support_only[0]) if recovar_support_only else None,
                "native_only_count": len(native_support_only),
                "recovar_only_count": len(recovar_support_only),
            }
        ),
    }
    stage_exact = {
        "active_candidate_tuples": active_tuple_subset,
        "raw_diff2": comparisons["raw_diff2"]["exact_equal"],
        "orientation_log_prior": comparisons["orientation_log_prior"]["exact_equal"],
        "translation_log_prior": comparisons["translation_log_prior"]["exact_equal"],
        "combined_log_weight_centered": comparisons["combined_log_weight_centered"]["exact_equal"],
        "normalized_posterior_native_active": comparisons["normalized_posterior_native_active"]["exact_equal"],
        "significant_support": support_exact,
    }
    first_raw_mismatch = first_mismatches["raw_diff2"]
    raw_diff2_trace = None
    if first_raw_mismatch is not None:
        raw_diff2_trace = _raw_diff2_trace(
            rotation=int(first_raw_mismatch["recovar_rotation_row"]),
            translation=int(first_raw_mismatch["recovar_translation_row"]),
            projected_references=recovar["proj_half"],
            shifted_images=recovar["shifted_corrected"],
            ctf2_over_nv=recovar["ctf2_over_nv_score"],
            half_weights=recovar["half_weights"],
            full_to_compact=full_to_compact,
            highres_xi2_half=float(recovar["relion_highres_xi2_half"]),
        )
    return {
        "original_index_zero_based": original,
        "stack_index_one_based": stack,
        "role": str(target["role"]),
        "factor_capture_kind": "geometry_only" if factor.geometry_only else "full_prescatter",
        "factor_path": str(factor.path.resolve()),
        "factor_sha256": factor.sha256,
        "fine_score_path": str(score.path.resolve()),
        "fine_score_sha256": score.sha256,
        "recovar_capture_path": str(Path(str(recovar["_path"])).resolve()),
        "recovar_capture_sha256": _sha256(Path(str(recovar["_path"]))),
        "rotation_map_max_abs": rotation_error,
        "translation_map_max_abs": translation_error,
        "native_candidate_count": int(candidates.size),
        "native_active_candidate_count": int(active_rows.size),
        "native_significant_count": significant_count,
        "native_support_source": support_source,
        "recovar_significant_count": int(np.count_nonzero(reconstruction_mask)),
        "native_active_tuple_missing_from_recovar_count": int(
            np.count_nonzero(~tuple_present | ~finite_preprior)
        ),
        "native_active_posterior_mass": float(np.sum(native_prob)),
        "recovar_mass_on_native_active": float(np.sum(recovar_prob)),
        "support_intersection_count": len(native_support_keys & recovar_support_keys),
        "support_union_count": len(native_support_keys | recovar_support_keys),
        "support_exact": support_exact,
        "comparisons": comparisons,
        "exact_boundary_order": list(EXACT_BOUNDARY_ORDER),
        "stage_exact": stage_exact,
        "first_exact_unequal_boundary": _first_exact_boundary(stage_exact),
        "first_mismatch_records": first_mismatches,
        "first_raw_diff2_recovar_operand_trace": raw_diff2_trace,
        "classification": classification,
    }


def analyze(
    *,
    capture_directory: Path,
    recovar_capture_directory: Path,
    selection_json: Path,
    capture_inertness_json: Path,
    native_state_star: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_json.read_text())
    selection_schema = selection.get("schema")
    _require(selection_schema in SELECTION_SCHEMAS, "selection schema changed")
    targets = selection.get("targets")
    _require(isinstance(targets, list) and targets, "fine-score panel is empty")
    stacks = [int(target["stack_index_one_based"]) for target in targets]
    _require(len(stacks) == len(set(stacks)), "fine-score panel contains duplicate stacks")
    physical_iteration = int(selection["physical_iteration"])
    physical_image_size = int(selection["physical_image_size"])
    current_size = int(selection["current_size"])

    inertness = json.loads(capture_inertness_json.read_text())
    inertness_schema = inertness.get("schema")
    _require(inertness_schema in INERTNESS_SCHEMAS, "capture inertness schema changed")
    _require(inertness.get("qualified") is True, "fine-score capture inertness did not qualify")
    _require(int(inertness["case_id"]) == int(selection["case_id"]), "inertness case changed")
    _require(int(inertness["physical_iteration"]) == physical_iteration, "inertness iteration changed")
    _require(
        sorted(int(value) for value in inertness["target_stack_indices_one_based"]) == sorted(stacks),
        "inertness target set changed",
    )

    factor_paths = sorted(capture_directory.glob("*.bpre-v2.bin"))
    score_paths = sorted(capture_directory.glob("*.fine-score-v1.bin"))
    _require(len(factor_paths) == len(score_paths) == len(stacks), "native capture file count changed")
    factors = {capture.stack_index: capture for capture in map(load_factor_capture, factor_paths)}
    scores = {capture.stack_index: capture for capture in map(load_fine_score_capture, score_paths)}
    _require(set(factors) == set(scores) == set(stacks), "native capture stack set changed")
    selected_hash = fnv1a64(",".join(str(value) for value in stacks))
    expected_rank = {int(target["stack_index_one_based"]): int(target["expected_mpi_rank"]) for target in targets}
    _require(
        all(factors[stack].header[36] == selected_hash for stack in stacks)
        and all(scores[stack].header[28] == selected_hash for stack in stacks),
        "native selected-set hash changed",
    )
    _require(
        all(factors[stack].header[14] == expected_rank[stack] for stack in stacks)
        and all(scores[stack].header[8] == expected_rank[stack] for stack in stacks),
        "native MPI rank changed",
    )

    state = _particle_table(native_state_star)
    particles = []
    for target in targets:
        stack = int(target["stack_index_one_based"])
        identity = str(target["image_identity"])
        _require(identity in state.index, f"native state is missing {identity}")
        recovar_path = _recovar_capture_path(
            recovar_capture_directory,
            int(target["original_index_zero_based"]),
            current_size,
        )
        with np.load(recovar_path, allow_pickle=False) as archive:
            recovar = {name: archive[name] for name in archive.files}
        recovar["_path"] = str(recovar_path)
        particles.append(
            _compare_particle(
                target=target,
                factor=factors[stack],
                score=scores[stack],
                recovar=recovar,
                native_state_row=state.loc[identity],
                physical_image_size=physical_image_size,
                current_size=current_size,
            )
        )

    classifications = [particle["classification"] for particle in particles]
    exact_boundaries = [particle["first_exact_unequal_boundary"] for particle in particles]
    aggregate = (
        "fixed_panel_fine_score_boundary_closes"
        if all(value == "fine_score_boundary_closes" for value in classifications)
        else next(value for value in classifications if value != "fine_score_boundary_closes")
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": (
            "exact and relative-L2 intermediate metrics; no correlation; "
            "signed shellwise FSC/FSC-AUC remains the map acceptance metric"
        ),
        "relative_l2_bound": RELATIVE_L2_BOUND,
        "case_id": int(selection["case_id"]),
        "physical_iteration": physical_iteration,
        "physical_image_size": physical_image_size,
        "current_size": current_size,
        "particle_count": len(particles),
        "selected_stack_text": ",".join(str(value) for value in stacks),
        "selected_stack_fnv1a64": selected_hash,
        "selection_schema": selection_schema,
        "selection_json": str(selection_json.resolve()),
        "selection_sha256": _sha256(selection_json),
        "capture_inertness_json": str(capture_inertness_json.resolve()),
        "capture_inertness_sha256": _sha256(capture_inertness_json),
        "capture_inertness_schema": inertness_schema,
        "native_state_star": str(native_state_star.resolve()),
        "native_state_sha256": _sha256(native_state_star),
        "particle_classification_counts": {
            value: classifications.count(value) for value in sorted(set(classifications))
        },
        "first_exact_unequal_boundary_counts": {
            value: exact_boundaries.count(value) for value in sorted(set(exact_boundaries))
        },
        "support_exact_count": sum(bool(particle["support_exact"]) for particle in particles),
        "classification": aggregate,
        "next_boundary": (
            "bpref_base_operands_and_translation_reduction"
            if aggregate == "fixed_panel_fine_score_boundary_closes"
            else "resolve_reported_first_unequal_fine_score_boundary"
        ),
        "production_authorized": False,
        "fixed_scorecard_changed": False,
        "particles": particles,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--recovar-capture-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--capture-inertness-json", type=Path, required=True)
    parser.add_argument("--native-state-star", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        capture_directory=args.capture_directory,
        recovar_capture_directory=args.recovar_capture_directory,
        selection_json=args.selection_json,
        capture_inertness_json=args.capture_inertness_json,
        native_state_star=args.native_state_star,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "next_boundary": report["next_boundary"],
                "particle_count": report["particle_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
