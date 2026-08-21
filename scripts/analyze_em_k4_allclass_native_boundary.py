#!/usr/bin/env python3
"""Join accepted native RELION and RECOVAR K=4 class-pose boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )

SCHEMA = "recovar.em_k4_allclass_native_boundary.v1"
ADMISSION_SCHEMA = "recovar-k4-highres-treatment-allclass-capture-v1"
EXPECTED_CLASSES = 4
EXPECTED_ITERATION = 2
EXPECTED_PARTICLE_ID = 48_584
EXPECTED_STACK = 53_723
EXPECTED_ORIGINAL_INDEX = 53_722
EXPECTED_CURRENT_SIZE = 38
BOUNDARY_ORDER = (
    "candidate_tuple_set",
    "raw_diff2",
    "combined_class_rotation_prior",
    "translation_prior",
    "unnormalized_class_pose_log_weight",
    "joint_class_pose_normalization",
    "global_significant_support",
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


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _ordered_float32_bits(values: np.ndarray) -> np.ndarray:
    """Map float32 bits to monotone integer codes for exact ULP distances."""

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    sign = (bits & np.uint32(0x80000000)) != 0
    return np.where(
        sign,
        np.bitwise_not(bits),
        bits | np.uint32(0x80000000),
    ).astype(np.uint64)


def float32_metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return scale-sensitive float32 metrics without correlation."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    _require(left.shape == right.shape, "float32 metric shapes differ")
    left_bits = left.view(np.uint32)
    right_bits = right.view(np.uint32)
    same_nonfinite = bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    finite = np.isfinite(left) & np.isfinite(right)
    delta = left[finite].astype(np.float64) - right[finite].astype(np.float64)
    denominator = math.sqrt(
        math.fsum(float(value) * float(value) for value in right[finite].astype(np.float64))
    )
    numerator = math.sqrt(
        math.fsum(float(value) * float(value) for value in delta)
    )
    mismatch = left_bits != right_bits
    mismatch_indices = np.flatnonzero(mismatch.reshape(-1))
    finite_mismatch = finite & mismatch
    left_ordered = _ordered_float32_bits(left[finite_mismatch])
    right_ordered = _ordered_float32_bits(right[finite_mismatch])
    mismatch_ulp = np.abs(
        left_ordered.astype(np.int64) - right_ordered.astype(np.int64)
    )
    return {
        "count": int(left.size),
        "bitwise_exact": bool(not np.any(mismatch)),
        "same_nonfinite_mask": same_nonfinite,
        "bitwise_mismatch_count": int(np.count_nonzero(mismatch)),
        "first_mismatch_flat_index": (
            int(mismatch_indices[0]) if mismatch_indices.size else None
        ),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "finite_mismatch_max_ulp": (
            int(np.max(mismatch_ulp)) if mismatch_ulp.size else 0
        ),
        "finite_mismatch_p50_ulp": (
            float(np.quantile(mismatch_ulp, 0.50)) if mismatch_ulp.size else 0.0
        ),
        "finite_mismatch_p95_ulp": (
            float(np.quantile(mismatch_ulp, 0.95)) if mismatch_ulp.size else 0.0
        ),
        "relative_l2_over_right": float(
            numerator / max(denominator, np.finfo(np.float64).tiny)
        ),
        "correlation_used": False,
    }


def nonnegative_support_boundary(
    values: np.ndarray,
    active_mask: np.ndarray,
    selected_mask: np.ndarray,
    *,
    threshold: float | np.floating[Any] | None = None,
) -> dict[str, Any]:
    """Describe a nonnegative support boundary in its original float dtype."""

    values = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    selected = np.asarray(selected_mask, dtype=bool)
    _require(
        values.shape == active.shape == selected.shape,
        "support-boundary shapes differ",
    )
    _require(values.dtype in (np.dtype(np.float32), np.dtype(np.float64)), "unsupported support dtype")
    _require(np.all(~selected | active), "selected support contains an inactive value")
    _require(
        np.all(np.isfinite(values[active])) and np.all(values[active] >= 0),
        "active support values must be finite and nonnegative",
    )
    retained = values[selected]
    excluded = values[active & ~selected]
    _require(retained.size > 0, "selected support is empty")
    bit_dtype = np.uint32 if values.dtype == np.float32 else np.uint64

    minimum = values.dtype.type(np.min(retained))
    minimum_bits = int(np.asarray(minimum).view(bit_dtype))
    record: dict[str, Any] = {
        "dtype": str(values.dtype),
        "selected_count": int(retained.size),
        "excluded_active_count": int(excluded.size),
        "minimum_selected": float(minimum),
        "minimum_selected_bits": minimum_bits,
        "minimum_selected_ties": int(np.count_nonzero(retained == minimum)),
    }
    if excluded.size:
        maximum = values.dtype.type(np.max(excluded))
        maximum_bits = int(np.asarray(maximum).view(bit_dtype))
        _require(minimum >= maximum, "support boundary is inverted")
        record.update(
            {
                "maximum_excluded_active": float(maximum),
                "maximum_excluded_active_bits": maximum_bits,
                "maximum_excluded_active_ties": int(
                    np.count_nonzero(excluded == maximum)
                ),
                "selected_minus_excluded_margin": float(
                    np.float64(minimum) - np.float64(maximum)
                ),
                "selected_minus_excluded_margin_ulps": minimum_bits - maximum_bits,
            }
        )
    else:
        record.update(
            {
                "maximum_excluded_active": None,
                "maximum_excluded_active_bits": None,
                "maximum_excluded_active_ties": None,
                "selected_minus_excluded_margin": None,
                "selected_minus_excluded_margin_ulps": None,
            }
        )

    if threshold is not None:
        threshold_value = values.dtype.type(threshold)
        _require(
            np.isfinite(threshold_value) and threshold_value >= 0,
            "support threshold is invalid",
        )
        threshold_bits = int(np.asarray(threshold_value).view(bit_dtype))
        replayed = active & (values >= threshold_value)
        record["recorded_threshold"] = {
            "value": float(threshold_value),
            "bits": threshold_bits,
            "replays_selection_exact": bool(np.array_equal(replayed, selected)),
            "selected_ties_at_threshold": int(
                np.count_nonzero(selected & (values == threshold_value))
            ),
            "minimum_selected_minus_threshold": float(
                np.float64(minimum) - np.float64(threshold_value)
            ),
            "minimum_selected_minus_threshold_ulps": minimum_bits - threshold_bits,
            "threshold_minus_maximum_excluded": (
                float(np.float64(threshold_value) - np.float64(maximum))
                if excluded.size
                else None
            ),
            "threshold_minus_maximum_excluded_ulps": (
                threshold_bits - maximum_bits if excluded.size else None
            ),
        }
    return record


def classify_first_unequal_boundary(stage_exact: dict[str, bool]) -> str:
    """Return the first failed staged boundary or the next unobserved boundary."""

    _require(tuple(stage_exact) == BOUNDARY_ORDER, "boundary identity/order changed")
    for boundary, exact in stage_exact.items():
        if not exact:
            return boundary
    return "bpref_operands_unobserved"


def candidate_key_from_flat_index(
    flat_index: int | None,
    shape: tuple[int, int],
    *,
    recovar_to_native_rotation: np.ndarray,
) -> dict[str, int] | None:
    """Resolve a RECOVAR rotation-major flat offset to an exact tuple key."""

    if flat_index is None:
        return None
    n_rotations, n_translations = (int(value) for value in shape)
    _require(
        0 <= int(flat_index) < n_rotations * n_translations,
        "candidate flat index is outside the class table",
    )
    inverse = np.asarray(recovar_to_native_rotation, dtype=np.int64).reshape(-1)
    _require(inverse.shape == (n_rotations,), "rotation inverse shape changed")
    recovar_rotation, translation = divmod(int(flat_index), n_translations)
    return {
        "native_rotation_local": int(inverse[recovar_rotation]),
        "recovar_rotation_local": recovar_rotation,
        "translation_id": translation,
    }


def candidate_key_from_sequence_index(
    sequence_index: int | None,
    *,
    native_rotation: np.ndarray,
    recovar_rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, int] | None:
    """Resolve an intersection-order offset to its exact native/RECOVAR key."""

    if sequence_index is None:
        return None
    arrays = tuple(
        np.asarray(values, dtype=np.int64).reshape(-1)
        for values in (native_rotation, recovar_rotation, translation)
    )
    _require(len({values.size for values in arrays}) == 1, "candidate key arrays differ")
    _require(
        0 <= int(sequence_index) < arrays[0].size,
        "candidate sequence index is outside the joined table",
    )
    return {
        "native_rotation_local": int(arrays[0][sequence_index]),
        "recovar_rotation_local": int(arrays[1][sequence_index]),
        "translation_id": int(arrays[2][sequence_index]),
    }


def exact_rotation_permutation(
    native_rotations: np.ndarray,
    recovar_rotations: np.ndarray,
) -> np.ndarray:
    """Map native rotation rows to bitwise-identical RECOVAR rows."""

    native = np.asarray(native_rotations, dtype=np.float32)
    recovar = np.asarray(recovar_rotations, dtype=np.float32)
    _require(
        native.ndim == 3
        and native.shape[1:] == (3, 3)
        and native.shape == recovar.shape
        and native.shape[0] > 0,
        "rotation table shapes differ",
    )
    lookup = {matrix.tobytes(): index for index, matrix in enumerate(recovar)}
    _require(len(lookup) == recovar.shape[0], "RECOVAR rotation matrices are not unique")
    permutation = np.asarray(
        [lookup.get(matrix.tobytes(), -1) for matrix in native],
        dtype=np.int64,
    )
    _require(np.all(permutation >= 0), "native rotation lacks a bitwise RECOVAR match")
    _require(
        np.unique(permutation).size == permutation.size,
        "native-to-RECOVAR rotation mapping is not bijective",
    )
    return permutation


def _load_recovar(path: Path, class_one_based: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        capture = {key: np.asarray(archive[key]) for key in archive.files}
    _require(int(capture["original_index"]) == EXPECTED_ORIGINAL_INDEX, "RECOVAR particle changed")
    _require(int(capture["class_index"]) == class_one_based - 1, "RECOVAR class changed")
    _require(int(capture["current_size"]) == EXPECTED_CURRENT_SIZE, "RECOVAR size changed")
    required = {
        "rotations",
        "candidate_mask",
        "scores_with_prior",
        "probs",
        "rotation_log_prior",
        "translation_log_prior",
        "reconstruction_mask",
        "reconstruction_probs",
        "relion_raw_diff2",
    }
    _require(required.issubset(capture), f"RECOVAR capture missing {sorted(required - set(capture))}")
    return capture


def _class_join(
    *,
    class_one_based: int,
    factor_path: Path,
    fine_score_path: Path,
    recovar_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    recovar = _load_recovar(recovar_path, class_one_based)
    _require(factor.geometry_only, "expected geometry-only BPref capture")
    _require(
        tuple(int(value) for value in score.header[4:10])
        == (
            EXPECTED_ITERATION,
            class_one_based,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
            2,
            1,
        ),
        "native fine-score identity changed",
    )
    _require(
        tuple(int(value) for value in factor.header[9:16])
        == (
            EXPECTED_ITERATION,
            class_one_based,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
            0,
            2,
            1,
        ),
        "native BPref identity changed",
    )

    native_rotations = (
        np.asarray(factor.rotations["matrix"], dtype=np.float32)
        .reshape(-1, 3, 3)
        .transpose(0, 2, 1)
    )
    recovar_rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    native_to_recovar = exact_rotation_permutation(native_rotations, recovar_rotations)
    recovar_to_native = np.empty_like(native_to_recovar)
    recovar_to_native[native_to_recovar] = np.arange(native_to_recovar.size, dtype=np.int64)
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    _require(
        candidate_mask.shape
        == (native_to_recovar.size, int(factor.header[21])),
        "candidate-mask geometry changed",
    )

    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    translation = np.asarray(candidates["translation_id"], dtype=np.int64)
    _require(
        np.all((native_rotation >= 0) & (native_rotation < native_to_recovar.size)),
        "native rotation index is out of range",
    )
    _require(
        np.all((translation >= 0) & (translation < candidate_mask.shape[1])),
        "native translation index is out of range",
    )
    mapped_rotation = native_to_recovar[native_rotation]
    native_tuple_mask = np.zeros(candidate_mask.shape, dtype=bool)
    native_tuple_mask[mapped_rotation[active], translation[active]] = True
    _require(
        int(np.count_nonzero(native_tuple_mask)) == int(np.count_nonzero(active)),
        "native active tuple keys are not unique",
    )
    tuple_exact = bool(np.array_equal(native_tuple_mask, candidate_mask))
    intersection_mask = active & candidate_mask[mapped_rotation, translation]
    intersection_count = int(np.count_nonzero(intersection_mask))
    union_count = int(np.count_nonzero(native_tuple_mask | candidate_mask))

    common_native_rotation = native_rotation[intersection_mask]
    mapped_common_rotation = mapped_rotation[intersection_mask]
    common_translation = translation[intersection_mask]
    raw_metric = float32_metric(
        candidates["raw_diff2"][intersection_mask],
        recovar["relion_raw_diff2"][mapped_common_rotation, common_translation],
    )
    rotation_prior_metric = float32_metric(
        candidates["orientation_log_prior"][intersection_mask],
        recovar["rotation_log_prior"][mapped_common_rotation],
    )
    translation_prior_metric = float32_metric(
        candidates["translation_log_prior"][intersection_mask],
        recovar["translation_log_prior"][common_translation],
    )
    combined_metric = float32_metric(
        candidates["combined_preexponent"][intersection_mask],
        recovar["scores_with_prior"][mapped_common_rotation, common_translation],
    )

    significant_weight = _float32_from_bits(int(factor.header[25]))
    weight_norm = _float32_from_bits(int(factor.header[26]))
    _require(
        np.isfinite(significant_weight)
        and np.isfinite(weight_norm)
        and weight_norm > 0,
        "native posterior scalars are invalid",
    )
    native_posterior = np.zeros(candidate_mask.shape, dtype=np.float32)
    native_posterior[mapped_rotation[active], translation[active]] = np.divide(
        candidates["post_exponent_weight"][active],
        weight_norm,
        dtype=np.float32,
    )
    native_significant = np.zeros(candidate_mask.shape, dtype=bool)
    significant_rows = active & (candidates["post_exponent_weight"] >= significant_weight)
    native_significant[mapped_rotation[significant_rows], translation[significant_rows]] = True
    _require(
        int(np.count_nonzero(native_significant)) == int(factor.header[45]),
        "native significant-support count does not replay BPref header",
    )
    recovar_posterior = np.asarray(recovar["probs"], dtype=np.float64)
    recovar_significant = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    _require(
        recovar_posterior.shape == candidate_mask.shape
        and recovar_significant.shape == candidate_mask.shape,
        "RECOVAR posterior/support geometry changed",
    )
    posterior_metric = float32_metric(native_posterior, recovar_posterior)
    significant_exact = bool(np.array_equal(native_significant, recovar_significant))
    normalized_native_threshold = np.divide(
        significant_weight,
        weight_norm,
        dtype=np.float32,
    )
    support_boundary = {
        "native_raw_weight_float32": nonnegative_support_boundary(
            candidates["post_exponent_weight"],
            active,
            significant_rows,
            threshold=significant_weight,
        ),
        "native_normalized_probability_float32": nonnegative_support_boundary(
            native_posterior,
            native_tuple_mask,
            native_significant,
            threshold=normalized_native_threshold,
        ),
        "recovar_probability_original": nonnegative_support_boundary(
            recovar_posterior,
            candidate_mask,
            recovar_significant,
        ),
    }

    candidate_mismatch = np.flatnonzero(
        (native_tuple_mask ^ candidate_mask).reshape(-1)
    )
    candidate_first_key = candidate_key_from_flat_index(
        int(candidate_mismatch[0]) if candidate_mismatch.size else None,
        candidate_mask.shape,
        recovar_to_native_rotation=recovar_to_native,
    )
    for metric in (
        raw_metric,
        rotation_prior_metric,
        translation_prior_metric,
        combined_metric,
    ):
        metric["first_mismatch_candidate_key"] = candidate_key_from_sequence_index(
            metric["first_mismatch_flat_index"],
            native_rotation=common_native_rotation,
            recovar_rotation=mapped_common_rotation,
            translation=common_translation,
        )
    posterior_metric["first_mismatch_candidate_key"] = candidate_key_from_flat_index(
        posterior_metric["first_mismatch_flat_index"],
        candidate_mask.shape,
        recovar_to_native_rotation=recovar_to_native,
    )
    support_mismatch = np.flatnonzero(
        (native_significant ^ recovar_significant).reshape(-1)
    )
    support_first_key = candidate_key_from_flat_index(
        int(support_mismatch[0]) if support_mismatch.size else None,
        candidate_mask.shape,
        recovar_to_native_rotation=recovar_to_native,
    )

    report = {
        "class_one_based": class_one_based,
        "identity": {
            "native_particle_id_zero_based": EXPECTED_PARTICLE_ID,
            "stack_index_one_based": EXPECTED_STACK,
            "recovar_original_index_zero_based": EXPECTED_ORIGINAL_INDEX,
        },
        "rotation_mapping": {
            "count": int(native_to_recovar.size),
            "bitwise_exact_bijection": True,
        },
        "translation_count": int(candidate_mask.shape[1]),
        "candidate_tuples": {
            "native": int(np.count_nonzero(native_tuple_mask)),
            "recovar": int(np.count_nonzero(candidate_mask)),
            "intersection": intersection_count,
            "union": union_count,
            "exact": tuple_exact,
            "first_mismatch_candidate_key": candidate_first_key,
        },
        "raw_diff2": raw_metric,
        "combined_class_rotation_prior": rotation_prior_metric,
        "translation_prior": translation_prior_metric,
        "unnormalized_class_pose_log_weight": combined_metric,
        "joint_posterior_native_float32_vs_recovar_capture_cast_to_float32": posterior_metric,
        "global_significant_support": {
            "native_count": int(np.count_nonzero(native_significant)),
            "recovar_count": int(np.count_nonzero(recovar_significant)),
            "intersection": int(np.count_nonzero(native_significant & recovar_significant)),
            "union": int(np.count_nonzero(native_significant | recovar_significant)),
            "exact": significant_exact,
            "first_mismatch_candidate_key": support_first_key,
            "boundary": support_boundary,
        },
        "native_global_scalars": {
            "fine_score_min_diff2_bits": int(score.header[18]),
            "fine_score_weights_max_bits": int(score.header[19]),
            "fine_score_exponent_shift_bits": int(score.header[20]),
            "bpref_significant_weight_bits": int(factor.header[25]),
            "bpref_weight_norm_bits": int(factor.header[26]),
            "significant_weight": float(significant_weight),
            "weight_norm": float(weight_norm),
        },
        "posterior_mass": {
            "native_float32_sum": float(native_posterior.sum(dtype=np.float64)),
            "recovar_float64_sum": float(recovar_posterior.sum(dtype=np.float64)),
        },
        "inputs": {
            "factor": {"path": str(factor_path.resolve()), "sha256": factor.sha256},
            "fine_score": {"path": str(fine_score_path.resolve()), "sha256": score.sha256},
            "recovar": {"path": str(recovar_path.resolve()), "sha256": _sha256(recovar_path)},
        },
    }
    arrays = {
        "native_posterior": native_posterior,
        "recovar_posterior": recovar_posterior,
        "native_significant": native_significant,
        "recovar_significant": recovar_significant,
    }
    return report, arrays


def build_report(
    *,
    admission_report_path: Path,
    relion_run_root: Path,
    recovar_capture_root: Path,
) -> dict[str, Any]:
    admission = json.loads(admission_report_path.read_text())
    _require(admission.get("schema") == ADMISSION_SCHEMA, "admission schema changed")
    _require(admission.get("status") == "complete", "admission run is incomplete")
    _require(admission.get("accepted") is True, "native capture admission did not pass")
    _require(
        admission.get("allclass_operand_localization_allowed") is True,
        "native all-class localization was not admitted",
    )
    _require(admission.get("scorecard_change_admissible") is False, "admission permits score change")
    _require(admission.get("correlation_used") is False, "admission used correlation")

    classes = []
    arrays = []
    for class_one_based in range(1, EXPECTED_CLASSES + 1):
        factor_paths = sorted(
            (relion_run_root / f"capture_class{class_one_based}/capture/factors").glob(
                "*.bpre-v2.bin"
            )
        )
        score_paths = sorted(
            (relion_run_root / f"capture_class{class_one_based}/capture/factors").glob(
                "*.fine-score-v1.bin"
            )
        )
        recovar_paths = sorted(
            recovar_capture_root.glob(
                f"pass2_orig{EXPECTED_ORIGINAL_INDEX:06d}_class{class_one_based:03d}_cs{EXPECTED_CURRENT_SIZE:03d}.npz"
            )
        )
        _require(len(factor_paths) == len(score_paths) == len(recovar_paths) == 1, "class input count changed")
        class_report, class_arrays = _class_join(
            class_one_based=class_one_based,
            factor_path=factor_paths[0],
            fine_score_path=score_paths[0],
            recovar_path=recovar_paths[0],
        )
        classes.append(class_report)
        arrays.append(class_arrays)

    native_scalars = [row["native_global_scalars"] for row in classes]
    scalar_keys = (
        "fine_score_min_diff2_bits",
        "fine_score_weights_max_bits",
        "fine_score_exponent_shift_bits",
        "bpref_significant_weight_bits",
        "bpref_weight_norm_bits",
    )
    global_scalars_exact = all(
        len({row[key] for row in native_scalars}) == 1 for key in scalar_keys
    )
    native_posterior = np.concatenate(
        [row["native_posterior"].reshape(-1) for row in arrays]
    ).astype(np.float64)
    recovar_posterior = np.concatenate(
        [row["recovar_posterior"].reshape(-1) for row in arrays]
    ).astype(np.float64)
    posterior_delta = recovar_posterior - native_posterior
    native_significant = np.concatenate(
        [row["native_significant"].reshape(-1) for row in arrays]
    )
    recovar_significant = np.concatenate(
        [row["recovar_significant"].reshape(-1) for row in arrays]
    )

    stage_exact = {
        "candidate_tuple_set": all(row["candidate_tuples"]["exact"] for row in classes),
        "raw_diff2": all(row["raw_diff2"]["bitwise_exact"] for row in classes),
        "combined_class_rotation_prior": all(
            row["combined_class_rotation_prior"]["bitwise_exact"] for row in classes
        ),
        "translation_prior": all(row["translation_prior"]["bitwise_exact"] for row in classes),
        "unnormalized_class_pose_log_weight": all(
            row["unnormalized_class_pose_log_weight"]["bitwise_exact"] for row in classes
        ),
        "joint_class_pose_normalization": bool(
            global_scalars_exact
            and all(
                row["joint_posterior_native_float32_vs_recovar_capture_cast_to_float32"][
                    "bitwise_exact"
                ]
                for row in classes
            )
        ),
        "global_significant_support": bool(np.array_equal(native_significant, recovar_significant)),
    }
    first_unequal = classify_first_unequal_boundary(stage_exact)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": f"first_unequal_boundary__{first_unequal}",
        "first_unequal_boundary": first_unequal,
        "stage_exact": stage_exact,
        "causal_interpretation_allowed_through": (
            BOUNDARY_ORDER[-1]
            if first_unequal == "bpref_operands_unobserved"
            else (
                BOUNDARY_ORDER[BOUNDARY_ORDER.index(first_unequal) - 1]
                if BOUNDARY_ORDER.index(first_unequal) > 0
                else None
            )
        ),
        "bpref_operand_capture_available": False,
        "joint_posterior_bpref_map_parity_established": False,
        "scorecard_change_admissible": False,
        "correlation_used": False,
        "metric_policy": (
            "immutable class/rotation/translation keys; original float32 bits; "
            "global posterior/support comparison; no correlation; geometry-only BPref "
            "cannot close the operand or reduction boundary"
        ),
        "native_global_scalars_exact_across_class_runs": global_scalars_exact,
        "joint_posterior": {
            "native_mass": float(math.fsum(float(value) for value in native_posterior)),
            "recovar_mass": float(math.fsum(float(value) for value in recovar_posterior)),
            "total_variation": float(
                0.5 * math.fsum(abs(float(value)) for value in posterior_delta)
            ),
            "max_abs": float(np.max(np.abs(posterior_delta), initial=0.0)),
        },
        "global_significant_support": {
            "native_count": int(np.count_nonzero(native_significant)),
            "recovar_count": int(np.count_nonzero(recovar_significant)),
            "intersection": int(np.count_nonzero(native_significant & recovar_significant)),
            "union": int(np.count_nonzero(native_significant | recovar_significant)),
            "exact": bool(np.array_equal(native_significant, recovar_significant)),
        },
        "classes": classes,
        "inputs": {
            "admission_report": {
                "path": str(admission_report_path.resolve()),
                "sha256": _sha256(admission_report_path),
            },
            "relion_run_root": str(relion_run_root.resolve()),
            "recovar_capture_root": str(recovar_capture_root.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--relion-run-root", type=Path, required=True)
    parser.add_argument("--recovar-capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        admission_report_path=args.admission_report,
        relion_run_root=args.relion_run_root,
        recovar_capture_root=args.recovar_capture_root,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "first_unequal_boundary": report["first_unequal_boundary"],
                "stage_exact": report["stage_exact"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
