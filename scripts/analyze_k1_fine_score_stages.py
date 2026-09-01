#!/usr/bin/env python3
"""Compare a complete K=1 fine-score table without loading projection operands.

This diagnostic is intentionally limited to the ordered boundaries already
materialized by the native RELION fine-score capture and the RECOVAR pass-2
NPZ: candidate tuples, raw scores, priors, combined weights, normalized
posterior, significant support, and hard winner.  It avoids loading the dense
projection panel, which is both unnecessary for this attribution and very
large for the case-22 iteration-2 surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .analyze_k1_fine_score_boundary import (
        _first_mismatch_record,
        _geometry_only_significant_count,
        _metric,
        _rotation_map,
        _translation_map,
    )
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from analyze_k1_fine_score_boundary import (  # type: ignore[no-redef]
        _first_mismatch_record,
        _geometry_only_significant_count,
        _metric,
        _rotation_map,
        _translation_map,
    )
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )


REPORT_SCHEMA = "recovar.em.k1_fine_score_stages.v1"
STAGES = (
    "candidate_tuple_presence",
    "preprior_score_finite",
    "preprior_score_centered",
    "orientation_log_prior",
    "translation_log_prior",
    "combined_log_weight_centered",
    "normalized_posterior",
    "significant_support",
    "hard_winner",
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


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "cannot center non-finite scores")
    return array - np.max(array)


def _support_boundary_window(
    *,
    order: np.ndarray,
    selected_count: int,
    tuple_keys: np.ndarray,
    native_weight: np.ndarray,
    native_posterior: np.ndarray,
    recovar_posterior: np.ndarray,
    native_support: np.ndarray,
    recovar_support: np.ndarray,
    radius: int = 3,
) -> dict[str, Any]:
    """Describe the exact ranked neighborhood around a support cutoff."""

    ranked = np.asarray(order, dtype=np.int64).reshape(-1)
    _require(ranked.size == tuple_keys.shape[0], "support rank is incomplete")
    _require(0 <= selected_count <= ranked.size, "support count is outside the ranked table")
    cumulative_native = np.cumsum(native_posterior[ranked], dtype=np.float64)
    cumulative_recovar = np.cumsum(recovar_posterior[ranked], dtype=np.float64)
    start = max(0, selected_count - radius)
    stop = min(ranked.size, selected_count + radius)
    records = []
    for rank_zero_based in range(start, stop):
        index = int(ranked[rank_zero_based])
        native_weight_f32 = np.float32(native_weight[index])
        records.append(
            {
                "rank_one_based": rank_zero_based + 1,
                "tuple_key": [int(value) for value in tuple_keys[index]],
                "native_weight_float32": float(native_weight_f32),
                "native_weight_float32_bits": int(native_weight_f32.view(np.uint32)),
                "native_posterior": float(native_posterior[index]),
                "recovar_posterior": float(recovar_posterior[index]),
                "native_cumulative_mass": float(cumulative_native[rank_zero_based]),
                "recovar_cumulative_mass": float(cumulative_recovar[rank_zero_based]),
                "native_selected": bool(native_support[index]),
                "recovar_selected": bool(recovar_support[index]),
            }
        )
    return {
        "selected_count": int(selected_count),
        "table_count": int(ranked.size),
        "window_radius": int(radius),
        "records": records,
    }


def _candidate_record(
    index: int,
    *,
    native_rows: np.ndarray,
    mapped_rotation: np.ndarray,
    mapped_translation: np.ndarray,
    recovar_combined: np.ndarray,
) -> dict[str, Any]:
    row = native_rows[index]
    return {
        "native_active_index": int(index),
        "native_rotation_local": int(row["rotation_local"]),
        "native_translation_id": int(row["translation_id"]),
        "recovar_rotation_row": int(mapped_rotation[index]),
        "recovar_translation_row": int(mapped_translation[index]),
        "native_raw_diff2": float(row["raw_diff2"]),
        "native_combined_log_weight": float(row["combined_preexponent"]),
        "recovar_combined_log_weight": float(recovar_combined[index]),
    }


def _recovar_only_candidate_groups(
    *,
    recovar_active_mask: np.ndarray,
    mapped_rotation: np.ndarray,
    mapped_translation: np.ndarray,
    oversampled_rotation_ids: np.ndarray,
    parent_map: np.ndarray,
) -> tuple[int, list[dict[str, Any]]]:
    """Describe finite RECOVAR tuples absent from native RELION's active set."""

    native_active_mask = np.zeros(recovar_active_mask.shape, dtype=bool)
    native_active_mask[mapped_rotation, mapped_translation] = True
    recovar_only_rows = np.argwhere(recovar_active_mask & ~native_active_mask)
    groups = []
    for rotation_row in np.unique(recovar_only_rows[:, 0]):
        rows = recovar_only_rows[recovar_only_rows[:, 0] == rotation_row]
        groups.append(
            {
                "recovar_rotation_row": int(rotation_row),
                "recovar_global_rotation_id": int(oversampled_rotation_ids[rotation_row]),
                "recovar_parent_row": int(parent_map[rotation_row]),
                "extra_translation_count": int(rows.shape[0]),
                "extra_translation_rows_recovar": [int(value) for value in rows[:, 1]],
            }
        )
    return int(recovar_only_rows.shape[0]), groups


def _logsumexp_float64(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    _require(finite.size > 0, "cannot normalize an empty score table")
    maximum = np.max(finite)
    return float(maximum + np.log(np.sum(np.exp(finite - maximum), dtype=np.float64)))


def _softmax_float64(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.all(np.isfinite(array)), "cannot normalize non-finite scores")
    shifted = array - np.max(array)
    weights = np.exp(shifted)
    return weights / np.sum(weights, dtype=np.float64)


def analyze(
    *,
    native_factor: Path,
    native_fine_score: Path,
    recovar_capture: Path,
    physical_image_size: int,
    top_count: int,
) -> dict[str, Any]:
    factor = load_factor_capture(native_factor)
    score = load_fine_score_capture(native_fine_score)
    _require(factor.stack_index == score.stack_index, "native capture identity changed")
    _require(top_count >= 2, "top_count must be at least two")

    active_mask = (score.candidates["flags"] & ACTIVE) != 0
    native = score.candidates[active_mask]
    _require(native.size >= top_count, "native active table is smaller than top_count")

    with np.load(recovar_capture, allow_pickle=False) as archive:
        rotations = archive["rotations"]
        translations = archive["fine_translations"]
        oversampled_rotation_ids = (
            np.asarray(archive["oversampled_rot_indices"], dtype=np.int64)
            if "oversampled_rot_indices" in archive.files
            else np.arange(rotations.shape[0], dtype=np.int64)
        )
        parent_map = (
            np.asarray(archive["parent_map"], dtype=np.int64)
            if "parent_map" in archive.files
            else np.full(rotations.shape[0], -1, dtype=np.int64)
        )
    rotation_map, rotation_error = _rotation_map(factor.rotations, rotations)
    translation_map, translation_error = _translation_map(
        factor.translations,
        translations,
        physical_image_size=physical_image_size,
    )
    native_rotation = np.asarray(native["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(native["translation_id"], dtype=np.int64)
    mapped_rotation = rotation_map[native_rotation]
    mapped_translation = translation_map[native_translation]
    tuple_keys = np.column_stack((mapped_rotation, mapped_translation))
    _require(
        np.unique(tuple_keys, axis=0).shape[0] == tuple_keys.shape[0],
        "mapped native active tuples are not unique",
    )

    with np.load(recovar_capture, allow_pickle=False) as archive:
        candidate_mask = np.asarray(archive["candidate_mask"], dtype=bool)
    tuple_present = candidate_mask[mapped_rotation, mapped_translation]

    with np.load(recovar_capture, allow_pickle=False) as archive:
        dense_preprior = archive["scores_pre_prior"]
        recovar_preprior = np.asarray(dense_preprior[mapped_rotation, mapped_translation], dtype=np.float32)
        del dense_preprior
    finite_score = np.isfinite(recovar_preprior)
    preprior_finite_equal = bool(np.all(finite_score))
    native_raw = np.asarray(native["raw_diff2"], dtype=np.float32)
    native_preprior = np.negative(native_raw, dtype=np.float32)

    with np.load(recovar_capture, allow_pickle=False) as archive:
        rotation_prior = np.asarray(archive["rotation_log_prior"], dtype=np.float32)
        translation_prior = np.asarray(archive["translation_log_prior"], dtype=np.float32)
    native_rotation_prior = np.asarray(native["orientation_log_prior"], dtype=np.float32)
    native_translation_prior = np.asarray(native["translation_log_prior"], dtype=np.float32)
    recovar_rotation_prior = rotation_prior[mapped_rotation]
    recovar_translation_prior = translation_prior[mapped_translation]

    with np.load(recovar_capture, allow_pickle=False) as archive:
        dense_combined = archive["scores_with_prior"]
        dense_combined_float64 = np.asarray(dense_combined, dtype=np.float64)
        recovar_combined_float64 = dense_combined_float64[mapped_rotation, mapped_translation]
        recovar_combined = np.asarray(dense_combined[mapped_rotation, mapped_translation], dtype=np.float32)
        recovar_dense_winner_flat = int(np.nanargmax(dense_combined))
        recovar_dense_winner = tuple(
            int(value) for value in np.unravel_index(recovar_dense_winner_flat, dense_combined.shape)
        )
        del dense_combined
    recovar_active_mask = candidate_mask & np.isfinite(dense_combined_float64)
    recovar_only_candidate_count, recovar_only_candidate_groups = _recovar_only_candidate_groups(
        recovar_active_mask=recovar_active_mask,
        mapped_rotation=mapped_rotation,
        mapped_translation=mapped_translation,
        oversampled_rotation_ids=oversampled_rotation_ids,
        parent_map=parent_map,
    )
    recovar_active_candidate_count = int(np.count_nonzero(recovar_active_mask))
    candidate_set_equal = bool(np.all(tuple_present) and recovar_only_candidate_count == 0)
    native_combined = np.asarray(native["combined_preexponent"], dtype=np.float32)
    comparable = tuple_present & finite_score & np.isfinite(recovar_combined)
    _require(np.any(comparable), "native and RECOVAR tables have no comparable active tuple")
    comparable_keys = tuple_keys[comparable]

    native_weight = np.asarray(native["post_exponent_weight"], dtype=np.float64)
    native_sum_weight = (
        float(np.asarray(np.uint32(score.header[32])).view(np.float32))
        if int(score.header[35]) == 1
        else float(np.sum(native_weight, dtype=np.float64))
    )
    native_posterior = native_weight / native_sum_weight
    with np.load(recovar_capture, allow_pickle=False) as archive:
        dense_posterior = archive["probs"]
        dense_posterior_float64 = np.asarray(dense_posterior, dtype=np.float64)
        recovar_posterior = np.asarray(dense_posterior[mapped_rotation, mapped_translation], dtype=np.float64)
        del dense_posterior

    native_exponent_shift = float(
        np.asarray(np.uint32(score.header[20]), dtype=np.uint32).view(np.float32)
    )
    native_log_z = float(np.log(native_sum_weight) - native_exponent_shift)
    recovar_log_z = _logsumexp_float64(dense_combined_float64[recovar_active_mask])
    common_log_z = _logsumexp_float64(recovar_combined_float64[comparable])
    native_active_in_recovar = np.zeros(recovar_active_mask.shape, dtype=bool)
    native_active_in_recovar[mapped_rotation, mapped_translation] = True
    recovar_only_probability_mass = float(
        np.sum(
            dense_posterior_float64[recovar_active_mask & ~native_active_in_recovar],
            dtype=np.float64,
        )
    )
    recovar_total_probability_mass = float(
        np.sum(dense_posterior_float64[recovar_active_mask], dtype=np.float64)
    )
    native_common_conditional = native_posterior[comparable]
    native_common_conditional /= np.sum(native_common_conditional, dtype=np.float64)
    recovar_common_conditional = recovar_posterior[comparable]
    recovar_common_conditional /= np.sum(recovar_common_conditional, dtype=np.float64)
    posterior_algorithm_decomposition = {
        "qualified_candidate_set_exact": candidate_set_equal and bool(np.all(comparable)),
        "production_posterior_conditional_on_common_candidates": _metric(
            native_common_conditional,
            recovar_common_conditional,
        ),
        "float64_softmax_conditional_on_common_candidates": _metric(
            _softmax_float64(native_combined[comparable]),
            _softmax_float64(recovar_combined_float64[comparable]),
        ),
    }
    if candidate_set_equal and bool(np.all(comparable)):
        native_mathematical_posterior = _softmax_float64(native_combined)
        recovar_mathematical_posterior = _softmax_float64(recovar_combined_float64)
        posterior_algorithm_decomposition.update({
            "native_production_vs_native_float64_softmax": _metric(
                native_posterior,
                native_mathematical_posterior,
            ),
            "native_float64_softmax_vs_recovar_float64_softmax": _metric(
                native_mathematical_posterior,
                recovar_mathematical_posterior,
            ),
            "recovar_float64_softmax_vs_recovar_production": _metric(
                recovar_mathematical_posterior,
                recovar_posterior,
            ),
        })
    del (
        candidate_mask,
        dense_combined_float64,
        dense_posterior_float64,
        recovar_active_mask,
        recovar_combined_float64,
    )

    significant_count = _geometry_only_significant_count(factor)
    native_support_order = np.argsort(-native_weight, kind="stable")[:significant_count]
    native_support = np.zeros(native.size, dtype=bool)
    native_support[native_support_order] = True
    with np.load(recovar_capture, allow_pickle=False) as archive:
        reconstruction_mask = np.asarray(archive["reconstruction_mask"], dtype=bool)
    recovar_support_on_native = reconstruction_mask[mapped_rotation, mapped_translation]
    recovar_significant_count = int(np.count_nonzero(reconstruction_mask))
    del reconstruction_mask
    native_support_keys = {(int(rotation), int(translation)) for rotation, translation in tuple_keys[native_support]}
    recovar_support_keys = {
        (int(rotation), int(translation)) for rotation, translation in tuple_keys[recovar_support_on_native]
    }
    support_exact = significant_count == recovar_significant_count and native_support_keys == recovar_support_keys
    recovar_support_order = np.argsort(-recovar_posterior, kind="stable")
    recovar_common_significant_count = int(np.count_nonzero(recovar_support_on_native))
    support_boundary = {
        "native_ranked": _support_boundary_window(
            order=np.argsort(-native_weight, kind="stable"),
            selected_count=significant_count,
            tuple_keys=tuple_keys,
            native_weight=native_weight,
            native_posterior=native_posterior,
            recovar_posterior=recovar_posterior,
            native_support=native_support,
            recovar_support=recovar_support_on_native,
        ),
        "recovar_ranked": _support_boundary_window(
            order=recovar_support_order,
            selected_count=recovar_common_significant_count,
            tuple_keys=tuple_keys,
            native_weight=native_weight,
            native_posterior=native_posterior,
            recovar_posterior=recovar_posterior,
            native_support=native_support,
            recovar_support=recovar_support_on_native,
        ),
    }

    native_winner = int(np.argmax(native_combined))
    recovar_winner_on_native = int(np.nanargmax(recovar_combined))
    native_winner_key = tuple(int(value) for value in tuple_keys[native_winner])
    recovar_winner_key = tuple(int(value) for value in tuple_keys[recovar_winner_on_native])
    winner_exact = native_winner_key == recovar_dense_winner == recovar_winner_key

    comparisons = {
        # RECOVAR stores the pre-prior score after a particle-global additive
        # offset.  Centering is therefore the exact observable comparison; it
        # preserves every pairwise score difference and the hard winner.
        "preprior_score_centered": _metric(
            _center(native_preprior[comparable]),
            _center(recovar_preprior[comparable]),
        ),
        "orientation_log_prior": _metric(native_rotation_prior[comparable], recovar_rotation_prior[comparable]),
        "translation_log_prior": _metric(native_translation_prior[comparable], recovar_translation_prior[comparable]),
        "combined_log_weight_centered": _metric(
            _center(native_combined[comparable]), _center(recovar_combined[comparable])
        ),
        "normalized_posterior": _metric(native_posterior[comparable], recovar_posterior[comparable]),
    }
    stage_exact = {
        "candidate_tuple_presence": candidate_set_equal,
        "preprior_score_finite": preprior_finite_equal,
        "preprior_score_centered": comparisons["preprior_score_centered"]["exact_equal"],
        "orientation_log_prior": comparisons["orientation_log_prior"]["exact_equal"],
        "translation_log_prior": comparisons["translation_log_prior"]["exact_equal"],
        "combined_log_weight_centered": comparisons["combined_log_weight_centered"]["exact_equal"],
        "normalized_posterior": comparisons["normalized_posterior"]["exact_equal"],
        "significant_support": support_exact,
        "hard_winner": winner_exact,
    }
    first_unequal = next((stage for stage in STAGES if not stage_exact[stage]), "all_stages_exact")

    native_rank = np.argsort(-native_combined, kind="stable")[:top_count]
    recovar_rank = np.argsort(-recovar_combined, kind="stable")[:top_count]
    missing_tuple = np.flatnonzero(~tuple_present)
    nonfinite_preprior = np.flatnonzero(~finite_score)
    missing_tuple_groups = []
    if missing_tuple.size:
        for rotation_row in np.unique(mapped_rotation[missing_tuple]):
            group = missing_tuple[mapped_rotation[missing_tuple] == rotation_row]
            native_rotation_ids = np.unique(native_rotation[group])
            _require(
                native_rotation_ids.size == 1,
                "one RECOVAR rotation row maps to multiple native rotations",
            )
            missing_tuple_groups.append(
                {
                    "recovar_rotation_row": int(rotation_row),
                    "recovar_global_rotation_id": int(oversampled_rotation_ids[rotation_row]),
                    "recovar_parent_row": int(parent_map[rotation_row]),
                    "native_rotation_local": int(native_rotation_ids[0]),
                    "missing_translation_count": int(group.size),
                    "missing_translation_ids_native": [
                        int(value) for value in native_translation[group]
                    ],
                    "missing_translation_rows_recovar": [
                        int(value) for value in mapped_translation[group]
                    ],
                }
            )
    native_only = sorted(native_support_keys - recovar_support_keys)
    recovar_only = sorted(recovar_support_keys - native_support_keys)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": "exact and relative-L2 intermediates; no correlation",
        "stack_index_one_based": factor.stack_index,
        "physical_image_size": physical_image_size,
        "native_factor": str(native_factor.resolve()),
        "native_factor_sha256": factor.sha256,
        "native_fine_score": str(native_fine_score.resolve()),
        "native_fine_score_sha256": score.sha256,
        "recovar_capture": str(recovar_capture.resolve()),
        "recovar_capture_sha256": _sha256(recovar_capture),
        "rotation_map_max_abs": rotation_error,
        "translation_map_max_abs": translation_error,
        "native_active_count": int(native.size),
        "comparable_active_count": int(np.count_nonzero(comparable)),
        "native_active_missing_candidate_tuple_count": int(missing_tuple.size),
        "native_active_missing_candidate_rotation_count": len(missing_tuple_groups),
        "native_active_missing_candidate_groups": missing_tuple_groups,
        "recovar_active_candidate_count": recovar_active_candidate_count,
        "recovar_only_candidate_tuple_count": recovar_only_candidate_count,
        "recovar_only_candidate_rotation_count": len(recovar_only_candidate_groups),
        "recovar_only_candidate_groups": recovar_only_candidate_groups,
        "native_active_nonfinite_preprior_count": int(nonfinite_preprior.size),
        "native_significant_count": significant_count,
        "recovar_significant_count": recovar_significant_count,
        "support_intersection_count": len(native_support_keys & recovar_support_keys),
        "support_native_only_count": len(native_only),
        "support_recovar_only_count": len(recovar_only),
        "first_support_native_only_key": list(native_only[0]) if native_only else None,
        "first_support_recovar_only_key": list(recovar_only[0]) if recovar_only else None,
        "support_boundary": support_boundary,
        "comparisons": comparisons,
        "normalization_decomposition": {
            "native_log_z_from_captured_exp_weights": native_log_z,
            "recovar_log_z_all_finite_candidates": recovar_log_z,
            "recovar_log_z_native_common_candidates": common_log_z,
            "recovar_minus_native_log_z": recovar_log_z - native_log_z,
            "common_score_log_z_delta": common_log_z - native_log_z,
            "recovar_only_candidate_log_z_increment": recovar_log_z - common_log_z,
            "recovar_only_candidate_probability_mass": recovar_only_probability_mass,
            "recovar_total_probability_mass": recovar_total_probability_mass,
            "posterior_algorithm_decomposition": posterior_algorithm_decomposition,
        },
        "stage_order": list(STAGES),
        "stage_exact": stage_exact,
        "first_exact_unequal_boundary": first_unequal,
        "first_mismatch": {
            "candidate_tuple_presence": (
                None
                if missing_tuple.size == 0
                else {
                    "native_active_index": int(missing_tuple[0]),
                    "recovar_rotation_row": int(tuple_keys[missing_tuple[0], 0]),
                    "recovar_translation_row": int(tuple_keys[missing_tuple[0], 1]),
                }
            ),
            "preprior_score_finite": (
                None
                if nonfinite_preprior.size == 0
                else {
                    "native_active_index": int(nonfinite_preprior[0]),
                    "recovar_rotation_row": int(tuple_keys[nonfinite_preprior[0], 0]),
                    "recovar_translation_row": int(tuple_keys[nonfinite_preprior[0], 1]),
                }
            ),
            "preprior_score_centered": _first_mismatch_record(
                _center(native_preprior[comparable]),
                _center(recovar_preprior[comparable]),
                tuple_keys=comparable_keys,
            ),
            "orientation_log_prior": _first_mismatch_record(
                native_rotation_prior[comparable],
                recovar_rotation_prior[comparable],
                tuple_keys=comparable_keys,
            ),
            "translation_log_prior": _first_mismatch_record(
                native_translation_prior[comparable],
                recovar_translation_prior[comparable],
                tuple_keys=comparable_keys,
            ),
            "combined_log_weight_centered": _first_mismatch_record(
                _center(native_combined[comparable]),
                _center(recovar_combined[comparable]),
                tuple_keys=comparable_keys,
            ),
            "normalized_posterior": _first_mismatch_record(
                native_posterior[comparable],
                recovar_posterior[comparable],
                tuple_keys=comparable_keys,
            ),
        },
        "native_winner": _candidate_record(
            native_winner,
            native_rows=native,
            mapped_rotation=mapped_rotation,
            mapped_translation=mapped_translation,
            recovar_combined=recovar_combined,
        ),
        "recovar_winner_on_native": _candidate_record(
            recovar_winner_on_native,
            native_rows=native,
            mapped_rotation=mapped_rotation,
            mapped_translation=mapped_translation,
            recovar_combined=recovar_combined,
        ),
        "recovar_dense_winner": list(recovar_dense_winner),
        "native_winner_vs_recovar_winner_margin_attribution": {
            "native_preprior_margin": float(native_preprior[native_winner] - native_preprior[recovar_winner_on_native]),
            "recovar_preprior_margin": float(
                recovar_preprior[native_winner] - recovar_preprior[recovar_winner_on_native]
            ),
            "preprior_margin_delta": float(
                (recovar_preprior[native_winner] - recovar_preprior[recovar_winner_on_native])
                - (native_preprior[native_winner] - native_preprior[recovar_winner_on_native])
            ),
            "native_orientation_prior_margin": float(
                native_rotation_prior[native_winner] - native_rotation_prior[recovar_winner_on_native]
            ),
            "recovar_orientation_prior_margin": float(
                recovar_rotation_prior[native_winner] - recovar_rotation_prior[recovar_winner_on_native]
            ),
            "orientation_prior_margin_delta": float(
                (recovar_rotation_prior[native_winner] - recovar_rotation_prior[recovar_winner_on_native])
                - (native_rotation_prior[native_winner] - native_rotation_prior[recovar_winner_on_native])
            ),
            "native_translation_prior_margin": float(
                native_translation_prior[native_winner] - native_translation_prior[recovar_winner_on_native]
            ),
            "recovar_translation_prior_margin": float(
                recovar_translation_prior[native_winner] - recovar_translation_prior[recovar_winner_on_native]
            ),
            "translation_prior_margin_delta": float(
                (recovar_translation_prior[native_winner] - recovar_translation_prior[recovar_winner_on_native])
                - (native_translation_prior[native_winner] - native_translation_prior[recovar_winner_on_native])
            ),
            "native_combined_margin": float(native_combined[native_winner] - native_combined[recovar_winner_on_native]),
            "recovar_combined_margin": float(
                recovar_combined[native_winner] - recovar_combined[recovar_winner_on_native]
            ),
        },
        "native_top": [
            _candidate_record(
                int(index),
                native_rows=native,
                mapped_rotation=mapped_rotation,
                mapped_translation=mapped_translation,
                recovar_combined=recovar_combined,
            )
            for index in native_rank
        ],
        "recovar_top_on_native": [
            _candidate_record(
                int(index),
                native_rows=native,
                mapped_rotation=mapped_rotation,
                mapped_translation=mapped_translation,
                recovar_combined=recovar_combined,
            )
            for index in recovar_rank
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-factor", type=Path, required=True)
    parser.add_argument("--native-fine-score", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--top-count", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        native_factor=args.native_factor,
        native_fine_score=args.native_fine_score,
        recovar_capture=args.recovar_capture,
        physical_image_size=args.physical_image_size,
        top_count=args.top_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
