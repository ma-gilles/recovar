#!/usr/bin/env python3
"""Audit the exact-device authoritative K=4 iteration-2 native score capture.

This gate compares RELION's native fine-score table to the frozen RECOVAR
pass-2 table after constructing a bitwise-exact rotation permutation.  It is
a fixed-target operand diagnostic, not a map-quality or scorecard gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-exact-device-native-score-audit-v8"
PASS_CLASSIFICATION = "exact_device_authoritative_native_and_recovar_target_match_after_exact_rotation_permutation"
TARGET_OFFSET_CLASSIFICATION = (
    "exact_device_target_absolute_score_offset_is_preprior_plus_float32_order_and_decision_inert"
)
GLOBAL_OFFSET_CLASSIFICATION = (
    "global_absolute_score_residual_is_preprior_data_path_dominated_with_exact_telescoping_closure"
)
TARGET_GPU_UUID = "GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30"
RECOVAR_PASS2_SHA256 = "3c4c566b6f2fce613f4d5869d2d3ccf53a2bcd1b3c26e5a32138588464049485"
EXPECTED_SUPPORT = 109_184
EXPECTED_ROTATIONS = 2_968
EXPECTED_STACK = 53_723
EXPECTED_PARTICLE_ID = 48_584
EXPECTED_CLASS = 1
EXPECTED_ITERATION = 2
EXPECTED_CURRENT_SIZE = 38
TARGET_RECOVAR_ROTATION = 2_626
TARGET_TRANSLATIONS = (80, 82)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_l2(values: np.ndarray) -> float:
    """Return an order-fixed float64 L2 norm without a threaded BLAS reduction."""

    flat_values = np.asarray(values, dtype=np.float64).ravel(order="C")
    squared_sum = math.fsum(
        float(value) * float(value)
        for value in flat_values
    )
    return math.sqrt(squared_sum)


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    """Normalize one captured score table with fixed-order scalar reductions."""

    flat_values = np.asarray(values, dtype=np.float64).ravel(order="C")
    _require(flat_values.size > 0, "cannot normalize an empty score table")
    maximum = float(np.max(flat_values))
    _require(math.isfinite(maximum), "score-table maximum is not finite")
    weights = np.fromiter(
        (
            math.exp(float(value) - maximum)
            for value in flat_values
        ),
        dtype=np.float64,
        count=flat_values.size,
    )
    denominator = math.fsum(float(weight) for weight in weights)
    _require(
        denominator > 0.0 and math.isfinite(denominator),
        "score-table normalization is not finite and positive",
    )
    return weights / denominator


def _normalized_mass_strata(
    *,
    stratum_ids: np.ndarray,
    stratum_name: str,
    native_score_mass: np.ndarray,
    recovar_score_mass: np.ndarray,
    native_candidate_index: np.ndarray,
    selected_stratum_ids: tuple[int, ...] = (),
    paired_ids: np.ndarray | None = None,
    paired_name: str | None = None,
) -> dict[str, Any]:
    """Partition candidate-level score-mass TV into fixed identity strata."""

    stratum_ids = np.asarray(stratum_ids, dtype=np.int64)
    native_score_mass = np.asarray(native_score_mass, dtype=np.float64)
    recovar_score_mass = np.asarray(recovar_score_mass, dtype=np.float64)
    native_candidate_index = np.asarray(
        native_candidate_index,
        dtype=np.int64,
    )
    arrays = (
        stratum_ids,
        native_score_mass,
        recovar_score_mass,
        native_candidate_index,
    )
    _require(
        len({array.shape for array in arrays}) == 1,
        f"{stratum_name} score-mass stratum array shapes differ",
    )
    _require(stratum_ids.size > 0, f"{stratum_name} strata are empty")
    _require(
        np.all(stratum_ids >= 0),
        f"{stratum_name} stratum identities must be nonnegative",
    )
    if paired_ids is not None:
        paired_ids = np.asarray(paired_ids, dtype=np.int64)
        _require(
            paired_ids.shape == stratum_ids.shape,
            f"{stratum_name} paired identity shape differs",
        )
        _require(
            paired_name is not None and np.all(paired_ids >= 0),
            f"{stratum_name} paired identity contract is invalid",
        )
    else:
        _require(
            paired_name is None,
            f"{stratum_name} paired identity name lacks values",
        )

    score_mass_delta = recovar_score_mass - native_score_mass
    score_mass_abs_delta = np.abs(score_mass_delta)
    total_variation = 0.5 * math.fsum(
        float(value) for value in score_mass_abs_delta
    )
    records = []
    for stratum_id in np.unique(stratum_ids):
        rows = np.flatnonzero(stratum_ids == stratum_id)
        row_abs_delta = score_mass_abs_delta[rows]
        maximum_abs_delta = float(np.max(row_abs_delta))
        representative_rows = rows[
            row_abs_delta == maximum_abs_delta
        ]
        representative_row = int(
            representative_rows[
                np.argmin(native_candidate_index[representative_rows])
            ]
        )
        candidate_l1 = math.fsum(
            float(value) for value in row_abs_delta
        )
        record = {
            stratum_name: int(stratum_id),
            "candidate_count": int(rows.size),
            "native_normalized_mass": float(
                math.fsum(
                    float(native_score_mass[row])
                    for row in rows
                )
            ),
            "recovar_normalized_mass": float(
                math.fsum(
                    float(recovar_score_mass[row])
                    for row in rows
                )
            ),
            "marginal_mass_delta_recovar_minus_native": float(
                math.fsum(
                    float(score_mass_delta[row])
                    for row in rows
                )
            ),
            "candidate_level_l1": float(candidate_l1),
            "candidate_level_tv_contribution": float(
                0.5 * candidate_l1
            ),
            "share_of_total_candidate_level_tv": float(
                0.0
                if total_variation == 0.0
                else 0.5 * candidate_l1 / total_variation
            ),
            "max_absolute_candidate_mass_delta": maximum_abs_delta,
            "max_absolute_delta_representative": {
                "selection_rule": (
                    "maximum_absolute_normalized_score_mass_delta_then_"
                    "lowest_native_candidate_index"
                ),
                "aligned_table_index": representative_row,
                "native_candidate_index": int(
                    native_candidate_index[representative_row]
                ),
                "delta_recovar_minus_native": float(
                    score_mass_delta[representative_row]
                ),
            },
        }
        if paired_ids is not None:
            paired_values = np.unique(paired_ids[rows])
            _require(
                paired_values.size == 1,
                f"{stratum_name} does not map to exactly one {paired_name}",
            )
            record[paired_name] = int(paired_values[0])
        records.append(record)

    ranked_records = sorted(
        records,
        key=lambda record: (
            -record["candidate_level_tv_contribution"],
            record[stratum_name],
        ),
    )
    for rank, record in enumerate(ranked_records, start=1):
        record["candidate_level_tv_rank_1based"] = rank
    record_by_id = {
        record[stratum_name]: record
        for record in ranked_records
    }
    selected_ids = tuple(
        sorted(set(int(value) for value in selected_stratum_ids))
    )
    stratum_tv_sum = math.fsum(
        float(record["candidate_level_tv_contribution"])
        for record in records
    )
    marginal_l1 = math.fsum(
        abs(float(record["marginal_mass_delta_recovar_minus_native"]))
        for record in records
    )
    marginal_total_variation = 0.5 * marginal_l1
    within_stratum_cancellation = (
        total_variation - marginal_total_variation
    )
    return {
        "stratum_identity": stratum_name,
        "partition_metric": (
            "candidate_level_total_variation_without_within_stratum_"
            "cancellation"
        ),
        "group_count": len(records),
        "candidate_count": int(stratum_ids.size),
        "candidate_level_total_variation": float(total_variation),
        "summed_stratum_tv_contributions": float(stratum_tv_sum),
        "partition_replay_residual": float(
            stratum_tv_sum - total_variation
        ),
        "marginal_distribution_l1": float(marginal_l1),
        "marginal_distribution_total_variation": float(
            marginal_total_variation
        ),
        "marginal_tv_fraction_of_candidate_level_tv": float(
            0.0
            if total_variation == 0.0
            else marginal_total_variation / total_variation
        ),
        "within_stratum_cancellation_total_variation": float(
            within_stratum_cancellation
        ),
        "within_stratum_cancellation_fraction_of_candidate_level_tv": (
            float(
                0.0
                if total_variation == 0.0
                else within_stratum_cancellation / total_variation
            )
        ),
        "ranking_rule": (
            "descending_candidate_level_tv_contribution_then_"
            f"ascending_{stratum_name}"
        ),
        "top_10": ranked_records[:10],
        "selected_stratum_ids": list(selected_ids),
        "selected_strata": [
            record_by_id[stratum_id]
            for stratum_id in selected_ids
            if stratum_id in record_by_id
        ],
        "missing_selected_stratum_ids": [
            stratum_id
            for stratum_id in selected_ids
            if stratum_id not in record_by_id
        ],
    }


def float32_metric(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Return fixed scale-sensitive float32 comparison metrics."""

    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    _require(lhs.shape == rhs.shape, "float32 metric shapes differ")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    denominator = max(
        _stable_l2(rhs),
        float(np.finfo(np.float64).tiny),
    )
    return {
        "count": int(lhs.size),
        "bitwise_exact": bool(np.array_equal(lhs.view(np.uint32), rhs.view(np.uint32))),
        "bitwise_mismatch_count": int(np.count_nonzero(lhs.view(np.uint32) != rhs.view(np.uint32))),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_rhs": _stable_l2(delta) / denominator,
        "l2_reduction": "math.fsum_of_float64_squares_in_c_order",
    }


def target_score_offset_attribution(
    *,
    min_diff2: np.float32,
    native_raw_diff2: np.float32,
    native_orientation_prior: np.float32,
    native_translation_prior: np.float32,
    native_combined: np.float32,
    recovar_pre_prior_residual: np.float64,
    recovar_orientation_prior: np.float32,
    recovar_translation_prior: np.float32,
    recovar_combined: np.float32,
    decision_topology_exact: bool,
) -> dict[str, Any]:
    """Decompose one target score offset using each engine's float32 order."""

    min_diff2 = np.float32(min_diff2)
    native_raw_diff2 = np.float32(native_raw_diff2)
    native_orientation_prior = np.float32(native_orientation_prior)
    native_translation_prior = np.float32(native_translation_prior)
    native_combined = np.float32(native_combined)
    recovar_pre_prior = np.float32(recovar_pre_prior_residual)
    recovar_orientation_prior = np.float32(recovar_orientation_prior)
    recovar_translation_prior = np.float32(recovar_translation_prior)
    recovar_combined = np.float32(recovar_combined)

    native_pre_prior = np.subtract(
        min_diff2,
        native_raw_diff2,
        dtype=np.float32,
    )
    native_production_replay = np.subtract(
        np.add(
            np.add(
                native_orientation_prior,
                native_translation_prior,
                dtype=np.float32,
            ),
            min_diff2,
            dtype=np.float32,
        ),
        native_raw_diff2,
        dtype=np.float32,
    )
    native_data_then_prior = np.add(
        np.add(native_pre_prior, native_orientation_prior, dtype=np.float32),
        native_translation_prior,
        dtype=np.float32,
    )
    recovar_data_then_prior = np.add(
        np.add(recovar_pre_prior, recovar_orientation_prior, dtype=np.float32),
        recovar_translation_prior,
        dtype=np.float32,
    )

    native_replay_exact = bool(
        native_production_replay.view(np.uint32) == native_combined.view(np.uint32)
    )
    recovar_replay_exact = bool(
        recovar_data_then_prior.view(np.uint32) == recovar_combined.view(np.uint32)
    )
    priors_bitwise_exact = bool(
        native_orientation_prior.view(np.uint32)
        == recovar_orientation_prior.view(np.uint32)
        and native_translation_prior.view(np.uint32)
        == recovar_translation_prior.view(np.uint32)
    )
    combined_bitwise_exact = bool(
        native_combined.view(np.uint32) == recovar_combined.view(np.uint32)
    )
    pre_prior_bitwise_exact = bool(
        native_pre_prior.view(np.uint32) == recovar_pre_prior.view(np.uint32)
    )

    data_path_contribution = np.float64(recovar_data_then_prior) - np.float64(
        native_data_then_prior
    )
    native_order_contribution = np.float64(native_data_then_prior) - np.float64(
        native_production_replay
    )
    combined_delta = np.float64(recovar_combined) - np.float64(native_combined)
    decomposition_residual = combined_delta - (
        data_path_contribution + native_order_contribution
    )
    attributed = bool(
        decision_topology_exact
        and priors_bitwise_exact
        and native_replay_exact
        and recovar_replay_exact
        and not combined_bitwise_exact
        and not pre_prior_bitwise_exact
        and decomposition_residual == 0.0
    )

    return {
        "classification": (
            TARGET_OFFSET_CLASSIFICATION
            if attributed
            else "exact_device_target_absolute_score_offset_not_fully_attributed"
        ),
        "attributed": attributed,
        "decision_topology_exact": decision_topology_exact,
        "target_priors_bitwise_exact": priors_bitwise_exact,
        "native_production_formula_replay_bitwise_exact": native_replay_exact,
        "recovar_data_then_prior_replay_bitwise_exact": recovar_replay_exact,
        "combined_scores_bitwise_exact": combined_bitwise_exact,
        "pre_prior_scores_bitwise_exact": pre_prior_bitwise_exact,
        "values": {
            "native_pre_prior_min_diff2_minus_raw": float(native_pre_prior),
            "recovar_pre_prior_residual_float32": float(recovar_pre_prior),
            "native_data_then_prior_counterfactual": float(native_data_then_prior),
            "native_production_combined": float(native_combined),
            "recovar_data_then_prior_combined": float(recovar_combined),
        },
        "deltas_recovar_minus_native": {
            "pre_prior": float(
                np.float64(recovar_pre_prior) - np.float64(native_pre_prior)
            ),
            "combined": float(combined_delta),
        },
        "combined_delta_decomposition": {
            "shared_data_then_prior_path": float(data_path_contribution),
            "native_float32_operation_order": float(native_order_contribution),
            "sum": float(data_path_contribution + native_order_contribution),
            "residual": float(decomposition_residual),
        },
    }


def _delta_summary(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    return {
        "count": int(delta.size),
        "nonzero_count": int(np.count_nonzero(delta)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
        "l1": float(np.sum(np.abs(delta))),
    }


def global_score_offset_attribution(
    *,
    min_diff2: np.float32,
    native_raw_diff2: np.ndarray,
    native_orientation_prior: np.ndarray,
    native_translation_prior: np.ndarray,
    native_combined: np.ndarray,
    recovar_pre_prior_residual: np.ndarray,
    recovar_orientation_prior: np.ndarray,
    recovar_translation_prior: np.ndarray,
    recovar_combined: np.ndarray,
    native_candidate_index: np.ndarray,
    native_rotation_local: np.ndarray,
    recovar_rotation_row: np.ndarray,
    translation_id: np.ndarray,
    decision_topology_exact: bool,
) -> dict[str, Any]:
    """Telescope the full active-table score delta through fixed operands."""

    native_raw_diff2 = np.asarray(native_raw_diff2, dtype=np.float32)
    native_orientation_prior = np.asarray(
        native_orientation_prior,
        dtype=np.float32,
    )
    native_translation_prior = np.asarray(
        native_translation_prior,
        dtype=np.float32,
    )
    native_combined = np.asarray(native_combined, dtype=np.float32)
    recovar_pre_prior = np.asarray(
        recovar_pre_prior_residual,
        dtype=np.float64,
    ).astype(np.float32)
    recovar_orientation_prior = np.asarray(
        recovar_orientation_prior,
        dtype=np.float32,
    )
    recovar_translation_prior = np.asarray(
        recovar_translation_prior,
        dtype=np.float32,
    )
    recovar_combined = np.asarray(recovar_combined, dtype=np.float32)
    native_candidate_index = np.asarray(
        native_candidate_index,
        dtype=np.int64,
    )
    native_rotation_local = np.asarray(
        native_rotation_local,
        dtype=np.int64,
    )
    recovar_rotation_row = np.asarray(
        recovar_rotation_row,
        dtype=np.int64,
    )
    translation_id = np.asarray(translation_id, dtype=np.int64)
    shapes = {
        values.shape
        for values in (
            native_raw_diff2,
            native_orientation_prior,
            native_translation_prior,
            native_combined,
            recovar_pre_prior,
            recovar_orientation_prior,
            recovar_translation_prior,
            recovar_combined,
            native_candidate_index,
            native_rotation_local,
            recovar_rotation_row,
            translation_id,
        )
    }
    _require(len(shapes) == 1, "global attribution array shapes differ")
    _require(native_combined.size > 0, "global attribution arrays are empty")
    _require(
        np.all(native_candidate_index >= 0)
        and np.all(native_rotation_local >= 0)
        and np.all(recovar_rotation_row >= 0)
        and np.all(translation_id >= 0),
        "global attribution candidate identities must be nonnegative",
    )
    _require(
        np.unique(native_candidate_index).size == native_candidate_index.size,
        "global attribution native candidate indices must be unique",
    )

    min_diff2 = np.float32(min_diff2)
    native_pre_prior = np.subtract(
        min_diff2,
        native_raw_diff2,
        dtype=np.float32,
    )
    native_production_replay = np.subtract(
        np.add(
            np.add(
                native_orientation_prior,
                native_translation_prior,
                dtype=np.float32,
            ),
            min_diff2,
            dtype=np.float32,
        ),
        native_raw_diff2,
        dtype=np.float32,
    )
    native_data_then_prior = np.add(
        np.add(
            native_pre_prior,
            native_orientation_prior,
            dtype=np.float32,
        ),
        native_translation_prior,
        dtype=np.float32,
    )
    recovar_data_native_priors = np.add(
        np.add(
            recovar_pre_prior,
            native_orientation_prior,
            dtype=np.float32,
        ),
        native_translation_prior,
        dtype=np.float32,
    )
    recovar_data_recovar_orientation = np.add(
        np.add(
            recovar_pre_prior,
            recovar_orientation_prior,
            dtype=np.float32,
        ),
        native_translation_prior,
        dtype=np.float32,
    )
    recovar_data_then_prior = np.add(
        np.add(
            recovar_pre_prior,
            recovar_orientation_prior,
            dtype=np.float32,
        ),
        recovar_translation_prior,
        dtype=np.float32,
    )

    components = {
        "native_float32_operation_order": (
            native_data_then_prior.astype(np.float64)
            - native_production_replay.astype(np.float64)
        ),
        "pre_prior_data_path": (
            recovar_data_native_priors.astype(np.float64)
            - native_data_then_prior.astype(np.float64)
        ),
        "orientation_prior_operand": (
            recovar_data_recovar_orientation.astype(np.float64)
            - recovar_data_native_priors.astype(np.float64)
        ),
        "translation_prior_operand": (
            recovar_data_then_prior.astype(np.float64)
            - recovar_data_recovar_orientation.astype(np.float64)
        ),
        "recovar_dump_replay_residual": (
            recovar_combined.astype(np.float64)
            - recovar_data_then_prior.astype(np.float64)
        ),
    }
    total_delta = (
        recovar_combined.astype(np.float64)
        - native_combined.astype(np.float64)
    )
    component_sum = np.sum(np.stack(tuple(components.values())), axis=0)
    closure = component_sum - total_delta
    summaries = {
        name: _delta_summary(delta)
        for name, delta in components.items()
    }
    total_component_l1 = sum(summary["l1"] for summary in summaries.values())
    _require(total_component_l1 > 0.0, "global attribution component L1 is zero")
    component_l1_fractions = {
        name: float(summary["l1"] / total_component_l1)
        for name, summary in summaries.items()
    }
    native_replay_exact_count = int(
        np.count_nonzero(
            native_production_replay.view(np.uint32)
            == native_combined.view(np.uint32)
        )
    )
    recovar_replay_exact_count = int(
        np.count_nonzero(
            recovar_data_then_prior.view(np.uint32)
            == recovar_combined.view(np.uint32)
        )
    )
    closure_exact = bool(np.all(closure == 0.0))
    data_path_strict_majority = bool(
        component_l1_fractions["pre_prior_data_path"] > 0.5
    )
    pre_prior_component = components["pre_prior_data_path"]
    maximum_abs_pre_prior = np.max(np.abs(pre_prior_component))
    tied_rows = np.flatnonzero(
        np.abs(pre_prior_component) == maximum_abs_pre_prior
    )
    representative_row = int(
        tied_rows[
            np.argmin(native_candidate_index[tied_rows])
        ]
    )
    pre_prior_representative = {
        "selection_rule": (
            "maximum_absolute_pre_prior_data_path_component_then_"
            "lowest_native_candidate_index"
        ),
        "aligned_table_index": representative_row,
        "native_candidate_index": int(
            native_candidate_index[representative_row]
        ),
        "native_rotation_local": int(
            native_rotation_local[representative_row]
        ),
        "recovar_rotation_row": int(
            recovar_rotation_row[representative_row]
        ),
        "translation_id": int(translation_id[representative_row]),
        "native_pre_prior": float(native_pre_prior[representative_row]),
        "recovar_pre_prior": float(recovar_pre_prior[representative_row]),
        "component_delta_recovar_minus_native": float(
            pre_prior_component[representative_row]
        ),
        "component_absolute_delta": float(
            abs(pre_prior_component[representative_row])
        ),
    }
    native_score_mass = _stable_softmax(native_combined)
    recovar_score_mass = _stable_softmax(recovar_combined)
    score_mass_delta = recovar_score_mass - native_score_mass
    score_mass_abs_delta = np.abs(score_mass_delta)
    maximum_abs_score_mass_delta = float(np.max(score_mass_abs_delta))
    score_mass_tied_rows = np.flatnonzero(
        score_mass_abs_delta == maximum_abs_score_mass_delta
    )
    score_mass_representative_row = int(
        score_mass_tied_rows[
            np.argmin(native_candidate_index[score_mass_tied_rows])
        ]
    )
    pre_prior_representative["decision_context"] = {
        "scope": (
            "within_captured_class_normalized_score_mass_only_"
            "not_full_kclass_posterior"
        ),
        "native_combined_score": float(
            native_combined[representative_row]
        ),
        "recovar_combined_score": float(
            recovar_combined[representative_row]
        ),
        "combined_score_delta_recovar_minus_native": float(
            total_delta[representative_row]
        ),
        "native_gap_below_class_max": float(
            np.float64(np.max(native_combined))
            - np.float64(native_combined[representative_row])
        ),
        "recovar_gap_below_class_max": float(
            np.float64(np.max(recovar_combined))
            - np.float64(recovar_combined[representative_row])
        ),
        "native_strict_rank_1based": int(
            np.count_nonzero(
                native_combined > native_combined[representative_row]
            )
            + 1
        ),
        "recovar_strict_rank_1based": int(
            np.count_nonzero(
                recovar_combined > recovar_combined[representative_row]
            )
            + 1
        ),
        "native_normalized_score_mass": float(
            native_score_mass[representative_row]
        ),
        "recovar_normalized_score_mass": float(
            recovar_score_mass[representative_row]
        ),
        "normalized_score_mass_delta_recovar_minus_native": float(
            score_mass_delta[representative_row]
        ),
    }
    normalized_score_mass_effect = {
        "scope": (
            "within_captured_class_normalized_score_mass_only_"
            "not_full_kclass_posterior"
        ),
        "normalization": (
            "math_exp_after_class_max_then_math_fsum_in_"
            "aligned_candidate_order"
        ),
        "native_sum": float(
            math.fsum(float(value) for value in native_score_mass)
        ),
        "recovar_sum": float(
            math.fsum(float(value) for value in recovar_score_mass)
        ),
        "l1": float(
            math.fsum(float(value) for value in score_mass_abs_delta)
        ),
        "total_variation": float(
            0.5
            * math.fsum(
                float(value)
                for value in score_mass_abs_delta
            )
        ),
        "max_absolute_delta": maximum_abs_score_mass_delta,
        "max_absolute_delta_representative": {
            "selection_rule": (
                "maximum_absolute_normalized_score_mass_delta_then_"
                "lowest_native_candidate_index"
            ),
            "aligned_table_index": score_mass_representative_row,
            "native_candidate_index": int(
                native_candidate_index[score_mass_representative_row]
            ),
            "native_rotation_local": int(
                native_rotation_local[score_mass_representative_row]
            ),
            "recovar_rotation_row": int(
                recovar_rotation_row[score_mass_representative_row]
            ),
            "translation_id": int(
                translation_id[score_mass_representative_row]
            ),
            "native_normalized_score_mass": float(
                native_score_mass[score_mass_representative_row]
            ),
            "recovar_normalized_score_mass": float(
                recovar_score_mass[score_mass_representative_row]
            ),
            "delta_recovar_minus_native": float(
                score_mass_delta[score_mass_representative_row]
            ),
        },
    }
    normalized_score_mass_effect["strata"] = {
        "scope": (
            "descriptive_partition_of_within_captured_class_candidate_"
            "level_total_variation_not_full_kclass_posterior"
        ),
        "rotation": _normalized_mass_strata(
            stratum_ids=recovar_rotation_row,
            stratum_name="recovar_rotation_row",
            native_score_mass=native_score_mass,
            recovar_score_mass=recovar_score_mass,
            native_candidate_index=native_candidate_index,
            selected_stratum_ids=(
                TARGET_RECOVAR_ROTATION,
                int(
                    recovar_rotation_row[
                        score_mass_representative_row
                    ]
                ),
            ),
            paired_ids=native_rotation_local,
            paired_name="native_rotation_local",
        ),
        "translation": _normalized_mass_strata(
            stratum_ids=translation_id,
            stratum_name="translation_id",
            native_score_mass=native_score_mass,
            recovar_score_mass=recovar_score_mass,
            native_candidate_index=native_candidate_index,
            selected_stratum_ids=(
                *TARGET_TRANSLATIONS,
                int(translation_id[score_mass_representative_row]),
            ),
        ),
    }
    attributed = bool(
        decision_topology_exact
        and native_replay_exact_count == native_combined.size
        and closure_exact
        and data_path_strict_majority
        and np.count_nonzero(total_delta) > 0
    )
    return {
        "classification": (
            GLOBAL_OFFSET_CLASSIFICATION
            if attributed
            else "global_absolute_score_residual_not_fully_attributed"
        ),
        "attributed": attributed,
        "decision_topology_exact": decision_topology_exact,
        "candidate_count": int(native_combined.size),
        "native_production_replay_bitwise_exact_count": (
            native_replay_exact_count
        ),
        "recovar_dump_replay_bitwise_exact_count": (
            recovar_replay_exact_count
        ),
        "telescoping_closure": {
            "exact": closure_exact,
            **_delta_summary(closure),
        },
        "total_delta_recovar_minus_native": _delta_summary(total_delta),
        "components": summaries,
        "component_l1_fractions": component_l1_fractions,
        "pre_prior_data_path_strict_majority": data_path_strict_majority,
        "pre_prior_data_path_representative": (
            pre_prior_representative
        ),
        "normalized_score_mass_effect": normalized_score_mass_effect,
    }


def classify_target_parity(
    *,
    support_exact: bool,
    winner_exact: bool,
    max_tie_key_sets_exact: bool,
    native_raw_diff2_tied: bool,
    recovar_scores_tied: bool,
    cross_engine_target_scores_bitwise_exact: bool,
) -> str:
    """Classify the six predeclared exact target gates."""

    gates = {
        "support": support_exact,
        "winner": winner_exact,
        "max_ties": max_tie_key_sets_exact,
        "native_raw_tie": native_raw_diff2_tied,
        "recovar_tie": recovar_scores_tied,
        "cross_engine_target": cross_engine_target_scores_bitwise_exact,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_CLASSIFICATION
    if len(failures) == 1:
        return f"exact_device_k4_target_{failures[0]}_mismatch"
    return "exact_device_k4_target_mixed_mismatch__" + "__".join(failures)


def _read_allocation_table(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw_line in path.read_text().splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        _require(len(fields) == 3, f"invalid allocation row: {raw_line}")
        rows.append({"uuid": fields[0], "name": fields[1], "pci_bus_id": fields[2]})
    _require(len(rows) == 2, "exact-device job did not receive exactly two GPUs")
    targets = [row for row in rows if row["uuid"] == TARGET_GPU_UUID]
    _require(len(targets) == 1, "required exact GPU UUID is absent or duplicated")
    _require("A100" in targets[0]["name"], "required exact GPU is not an A100")
    return rows


def _validate_completion(path: Path, *, expected_job_id: int) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "relion_k4_it2_authoritative_native_capture_v1",
        "unexpected science-completion schema",
    )
    _require(report.get("status") == "complete", "science is incomplete")
    _require(
        int(report.get("slurm_job_id")) == expected_job_id,
        "science-completion job identity changed",
    )
    _require(
        report.get("sampling_perturbation") == 0.27053284645080566,
        "authoritative perturbation changed",
    )
    _require(
        report.get("scorecard_change_admissible") is False,
        "science completion incorrectly permits a scorecard change",
    )
    _require(
        report.get("grid_correction") == "unset_default_off" and report.get("final_all_data_after_max_iter") == "unset",
        "grid/finalization contract changed",
    )
    return report


def _validate_state(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == "relion_k4_it2_authoritative_translation_grid_validation_v1",
        "unexpected translation-grid validation schema",
    )
    _require(
        report.get("status") == "accepted"
        and report.get("classification") == "native_capture_matches_uninterrupted_iteration2_translation_grid",
        "authoritative translation-grid validation did not pass",
    )
    _require(
        report.get("translation_ids") == list(TARGET_TRANSLATIONS),
        "target translation IDs changed",
    )
    _require(
        float(report.get("max_abs_pixels")) <= 2.0e-6,
        "native translation grid exceeds the fixed pixel gate",
    )
    _require(
        report.get("phase_capture_sha256") == RECOVAR_PASS2_SHA256,
        "frozen RECOVAR pass-2 hash changed in state validation",
    )
    return report


def _rotation_permutation(
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
) -> np.ndarray:
    native = np.asarray(native_rotations, dtype=np.float32)
    recovar = np.asarray(recovar_rotations, dtype=np.float32)
    _require(native.shape == recovar.shape, "rotation table shapes differ")
    _require(
        native.shape == (EXPECTED_ROTATIONS, 3, 3),
        "fixed K4 rotation topology changed",
    )
    lookup = {matrix.tobytes(): index for index, matrix in enumerate(recovar)}
    _require(
        len(lookup) == EXPECTED_ROTATIONS,
        "RECOVAR rotation matrices are not unique",
    )
    permutation = np.asarray(
        [lookup.get(matrix.tobytes(), -1) for matrix in native],
        dtype=np.int64,
    )
    _require(
        np.all(permutation >= 0),
        "one or more native rotations lack a bitwise RECOVAR match",
    )
    _require(
        np.unique(permutation).size == EXPECTED_ROTATIONS,
        "native-to-RECOVAR rotation mapping is not bijective",
    )
    return permutation


def _comparison(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    _require(factor.geometry_only, "expected geometry-only BPref capture")
    _require(factor.stack_index == EXPECTED_STACK, "factor stack changed")
    _require(
        score.header[4:8]
        == (
            EXPECTED_ITERATION,
            EXPECTED_CLASS,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
        ),
        "fine-score identity changed",
    )
    _require(
        _sha256(recovar_pass2_path) == RECOVAR_PASS2_SHA256,
        "frozen RECOVAR pass-2 artifact hash changed",
    )
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    _require(int(recovar["original_index"]) == 53_722, "RECOVAR particle changed")
    _require(
        int(recovar["class_index"]) == 0 and int(recovar["current_size"]) == EXPECTED_CURRENT_SIZE,
        "RECOVAR class/current-size identity changed",
    )

    native_rotations = np.asarray(factor.rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3).transpose(0, 2, 1)
    recovar_rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    native_to_recovar = _rotation_permutation(
        native_rotations,
        recovar_rotations,
    )

    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    active_indices = np.flatnonzero(active)
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(candidates["translation_id"], dtype=np.int64)
    _require(
        np.all((native_rotation >= 0) & (native_rotation < EXPECTED_ROTATIONS)),
        "native rotation index is out of bounds",
    )
    mapped_rotation = native_to_recovar[native_rotation]
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    _require(
        np.all((native_translation >= 0) & (native_translation < candidate_mask.shape[1])),
        "native translation index is out of bounds",
    )
    mapped_keys = np.column_stack((mapped_rotation[active], native_translation[active]))
    _require(
        np.unique(mapped_keys, axis=0).shape[0] == mapped_keys.shape[0],
        "mapped active native candidates are not unique",
    )
    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_support[mapped_rotation[active], native_translation[active]] = True
    _require(
        np.count_nonzero(native_support) == EXPECTED_SUPPORT and np.count_nonzero(candidate_mask) == EXPECTED_SUPPORT,
        "fixed active-support denominator changed",
    )
    intersection = int(np.count_nonzero(native_support & candidate_mask))
    union = int(np.count_nonzero(native_support | candidate_mask))
    support_exact = bool(np.array_equal(native_support, candidate_mask))

    native_combined = np.asarray(candidates["combined_preexponent"][active], dtype=np.float32)
    recovar_combined = np.asarray(
        recovar["scores_with_prior"][
            mapped_rotation[active],
            native_translation[active],
        ],
        dtype=np.float32,
    )
    native_orientation_prior = np.asarray(candidates["orientation_log_prior"][active], dtype=np.float32)
    recovar_orientation_prior = np.asarray(recovar["rotation_log_prior"][mapped_rotation[active]], dtype=np.float32)
    native_translation_prior = np.asarray(candidates["translation_log_prior"][active], dtype=np.float32)
    recovar_translation_prior = np.asarray(
        recovar["translation_log_prior"][native_translation[active]],
        dtype=np.float32,
    )
    min_diff2 = np.asarray(
        np.frombuffer(
            np.asarray(score.header[18], dtype="<u4").tobytes(),
            dtype="<f4",
        )[0],
        dtype=np.float32,
    )

    native_winner_candidate = int(active_indices[np.argmax(native_combined)])
    native_winner_mapped = (
        int(mapped_rotation[native_winner_candidate]),
        int(native_translation[native_winner_candidate]),
    )
    recovar_scores = np.asarray(recovar["scores_with_prior"], dtype=np.float32)
    recovar_winner = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(np.where(candidate_mask, recovar_scores, -np.inf))),
            candidate_mask.shape,
        )
    )
    winner_exact = native_winner_mapped == recovar_winner

    native_max = np.max(native_combined)
    native_max_keys = {tuple(int(value) for value in key) for key in mapped_keys[native_combined == native_max]}
    recovar_max = np.max(recovar_scores[candidate_mask])
    recovar_max_keys = {
        tuple(int(value) for value in row) for row in np.argwhere(candidate_mask & (recovar_scores == recovar_max))
    }
    max_tie_key_sets_exact = native_max_keys == recovar_max_keys
    decision_topology_exact = bool(
        support_exact
        and winner_exact
        and max_tie_key_sets_exact
    )
    global_attribution = global_score_offset_attribution(
        min_diff2=min_diff2,
        native_raw_diff2=np.asarray(
            candidates["raw_diff2"][active],
            dtype=np.float32,
        ),
        native_orientation_prior=native_orientation_prior,
        native_translation_prior=native_translation_prior,
        native_combined=native_combined,
        recovar_pre_prior_residual=np.asarray(
            recovar["scores_pre_prior"][
                mapped_rotation[active],
                native_translation[active],
            ],
            dtype=np.float64,
        ),
        recovar_orientation_prior=recovar_orientation_prior,
        recovar_translation_prior=recovar_translation_prior,
        recovar_combined=recovar_combined,
        native_candidate_index=active_indices,
        native_rotation_local=native_rotation[active],
        recovar_rotation_row=mapped_rotation[active],
        translation_id=native_translation[active],
        decision_topology_exact=decision_topology_exact,
    )

    inverse_target = np.flatnonzero(native_to_recovar == TARGET_RECOVAR_ROTATION)
    _require(
        inverse_target.size == 1,
        "target RECOVAR rotation does not have one native match",
    )
    target_native_rotation = int(inverse_target[0])
    target_records = []
    target_attributions = []
    for translation in TARGET_TRANSLATIONS:
        matches = np.flatnonzero(
            active & (native_rotation == target_native_rotation) & (native_translation == translation)
        )
        _require(
            matches.size == 1,
            f"target translation {translation} does not have one native row",
        )
        row = candidates[int(matches[0])]
        native_raw = np.asarray(row["raw_diff2"], dtype=np.float32)
        native_total = np.asarray(row["combined_preexponent"], dtype=np.float32)
        recovar_total = np.asarray(
            recovar_scores[TARGET_RECOVAR_ROTATION, translation],
            dtype=np.float32,
        )
        native_rotation_prior = np.asarray(
            row["orientation_log_prior"],
            dtype=np.float32,
        )
        native_translation_prior_target = np.asarray(
            row["translation_log_prior"],
            dtype=np.float32,
        )
        recovar_rotation_prior_target = np.asarray(
            recovar["rotation_log_prior"][TARGET_RECOVAR_ROTATION],
            dtype=np.float32,
        )
        recovar_translation_prior_target = np.asarray(
            recovar["translation_log_prior"][translation],
            dtype=np.float32,
        )
        recovar_pre_prior_residual = np.asarray(
            recovar["scores_pre_prior"][TARGET_RECOVAR_ROTATION, translation],
            dtype=np.float64,
        )
        attribution = target_score_offset_attribution(
            min_diff2=min_diff2,
            native_raw_diff2=native_raw,
            native_orientation_prior=native_rotation_prior,
            native_translation_prior=native_translation_prior_target,
            native_combined=native_total,
            recovar_pre_prior_residual=recovar_pre_prior_residual,
            recovar_orientation_prior=recovar_rotation_prior_target,
            recovar_translation_prior=recovar_translation_prior_target,
            recovar_combined=recovar_total,
            decision_topology_exact=decision_topology_exact,
        )
        target_attributions.append(attribution)
        target_records.append(
            {
                "translation_id": translation,
                "native_sparse_index": int(row["sparse_index"]),
                "native_rotation_local": target_native_rotation,
                "recovar_rotation_row": TARGET_RECOVAR_ROTATION,
                "native_raw_diff2": float(native_raw),
                "native_raw_diff2_bits": int(native_raw.view(np.uint32)),
                "native_combined_preexponent": float(native_total),
                "native_combined_preexponent_bits": int(native_total.view(np.uint32)),
                "native_orientation_log_prior": float(native_rotation_prior),
                "native_orientation_log_prior_bits": int(
                    native_rotation_prior.view(np.uint32)
                ),
                "native_translation_log_prior": float(
                    native_translation_prior_target
                ),
                "native_translation_log_prior_bits": int(
                    native_translation_prior_target.view(np.uint32)
                ),
                "recovar_pre_prior_residual_float64": float(
                    recovar_pre_prior_residual
                ),
                "recovar_pre_prior_residual_float32": float(
                    np.float32(recovar_pre_prior_residual)
                ),
                "recovar_orientation_log_prior": float(
                    recovar_rotation_prior_target
                ),
                "recovar_orientation_log_prior_bits": int(
                    recovar_rotation_prior_target.view(np.uint32)
                ),
                "recovar_translation_log_prior": float(
                    recovar_translation_prior_target
                ),
                "recovar_translation_log_prior_bits": int(
                    recovar_translation_prior_target.view(np.uint32)
                ),
                "recovar_score_with_prior": float(recovar_total),
                "recovar_score_with_prior_bits": int(recovar_total.view(np.uint32)),
            }
        )
    native_raw_tied = target_records[0]["native_raw_diff2_bits"] == target_records[1]["native_raw_diff2_bits"]
    recovar_tied = (
        target_records[0]["recovar_score_with_prior_bits"] == target_records[1]["recovar_score_with_prior_bits"]
    )
    target_cross_exact = all(
        row["native_combined_preexponent_bits"] == row["recovar_score_with_prior_bits"] for row in target_records
    )
    classification = classify_target_parity(
        support_exact=support_exact,
        winner_exact=winner_exact,
        max_tie_key_sets_exact=max_tie_key_sets_exact,
        native_raw_diff2_tied=native_raw_tied,
        recovar_scores_tied=recovar_tied,
        cross_engine_target_scores_bitwise_exact=target_cross_exact,
    )
    return {
        "classification": classification,
        "accepted": classification == PASS_CLASSIFICATION,
        "rotation_mapping": {
            "count": int(native_to_recovar.size),
            "bitwise_exact_bijection": True,
            "native_target_rotation_local": target_native_rotation,
            "recovar_target_rotation_row": TARGET_RECOVAR_ROTATION,
            "native_row_2626_maps_to_recovar_row": int(native_to_recovar[2626]),
        },
        "support": {
            "native_active_count": int(np.count_nonzero(native_support)),
            "recovar_active_count": int(np.count_nonzero(candidate_mask)),
            "intersection": intersection,
            "union": union,
            "jaccard": float(intersection / union),
            "exact": support_exact,
        },
        "scores": {
            "combined_preexponent_vs_scores_with_prior": float32_metric(
                native_combined,
                recovar_combined,
            ),
            "orientation_log_prior": float32_metric(
                native_orientation_prior,
                recovar_orientation_prior,
            ),
            "translation_log_prior": float32_metric(
                native_translation_prior,
                recovar_translation_prior,
            ),
            "global_offset_attribution": global_attribution,
        },
        "winner": {
            "native_rotation_local": int(candidates[native_winner_candidate]["rotation_local"]),
            "native_mapped_recovar_key": list(native_winner_mapped),
            "recovar_key": list(recovar_winner),
            "exact": winner_exact,
            "native_max_tie_count": len(native_max_keys),
            "recovar_max_tie_count": len(recovar_max_keys),
            "max_tie_key_sets_exact": max_tie_key_sets_exact,
        },
        "target": {
            "records": target_records,
            "native_raw_diff2_tied": native_raw_tied,
            "recovar_scores_tied": recovar_tied,
            "cross_engine_combined_scores_bitwise_exact": target_cross_exact,
            "offset_attribution": {
                "classification": (
                    TARGET_OFFSET_CLASSIFICATION
                    if all(record["attributed"] for record in target_attributions)
                    else "exact_device_target_absolute_score_offset_not_fully_attributed"
                ),
                "all_target_rows_attributed": all(
                    record["attributed"] for record in target_attributions
                ),
                "records": target_attributions,
            },
        },
    }


def build_report(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
    state_validation_path: Path,
    allocation_table_path: Path,
    science_completion_path: Path,
    expected_job_id: int,
) -> dict[str, Any]:
    """Build the exact-device fixed-target K4 report."""

    allocation = _read_allocation_table(allocation_table_path)
    completion = _validate_completion(
        science_completion_path,
        expected_job_id=expected_job_id,
    )
    state = _validate_state(state_validation_path)
    comparison = _comparison(
        factor_path=factor_path,
        fine_score_path=fine_score_path,
        recovar_pass2_path=recovar_pass2_path,
    )
    inputs = {
        "factor": factor_path,
        "fine_score": fine_score_path,
        "recovar_pass2": recovar_pass2_path,
        "state_validation": state_validation_path,
        "allocation_table": allocation_table_path,
        "science_completion": science_completion_path,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": comparison.pop("classification"),
        "accepted": comparison.pop("accepted"),
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 iteration-2 candidate diagnostic; "
            "bitwise rotation mapping, exact support/winner/tie sets, and "
            "bitwise target scores; no map acceptance claim; no correlation"
        ),
        "fixed_contract": {
            "slurm_job_id": expected_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_support": EXPECTED_SUPPORT,
            "expected_rotations": EXPECTED_ROTATIONS,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "target_translations": list(TARGET_TRANSLATIONS),
            "recovar_pass2_sha256": RECOVAR_PASS2_SHA256,
        },
        "hardware": {
            "allocation": allocation,
            "target_gpu_present": True,
        },
        "science_completion": completion,
        "state_validation": {
            "classification": state["classification"],
            "max_abs_pixels": state["max_abs_pixels"],
        },
        **comparison,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in inputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--state-validation", type=Path, required=True)
    parser.add_argument("--allocation-table", type=Path, required=True)
    parser.add_argument("--science-completion", type=Path, required=True)
    parser.add_argument("--expected-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_pass2_path=args.recovar_pass2,
        state_validation_path=args.state_validation,
        allocation_table_path=args.allocation_table,
        science_completion_path=args.science_completion,
        expected_job_id=args.expected_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"accepted": report["accepted"], "classification": report["classification"]}))


if __name__ == "__main__":
    main()
