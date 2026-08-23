#!/usr/bin/env python3
"""Locate the first unequal field between two RECOVAR coarse captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ORDERED_FIELDS = (
    "rotations",
    "translations",
    "window_indices",
    "coarse_gaussian_score_indices",
    "translation_phase_source",
    "coarse_gaussian_unshifted_corrected",
    "coarse_gaussian_shifted_corrected",
    "coarse_gaussian_pixel_weight",
    "coarse_gaussian_initial_diff2",
    "shifted_data",
    "ctf2_data",
    "half_weights",
    "projected_reference_rotation_ids",
    "projected_reference_per_class",
    "projected_reference_norm_score_per_class",
    "projected_cross_score_per_class",
    "class_log_priors",
    "rotation_log_prior",
    "translation_log_prior",
    "scores_pre_prior_per_class",
    "scores_with_prior_per_class",
    "best_score",
    "normalization_log_z",
    "class_log_z",
    "weights_per_class",
    "relion_f32_sum_weight",
    "relion_f32_significant_weight",
    "relion_f32_cutoff_count",
    "max_posterior",
    "significant_mask",
    "hard_assignment",
)

OPTIONAL_PROJECTION_FIELDS = (
    "projected_reference_rotation_ids",
    "projected_reference_per_class",
    "projected_reference_norm_score_per_class",
    "projected_cross_score_per_class",
)


def _available_ordered_fields(
    control_files: tuple[str, ...] | list[str],
    candidate_files: tuple[str, ...] | list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return comparable fields while allowing one-sided optional captures."""

    control_set = set(control_files)
    candidate_set = set(candidate_files)
    missing_required = [
        field
        for field in ORDERED_FIELDS
        if field not in OPTIONAL_PROJECTION_FIELDS
        and (field not in control_set or field not in candidate_set)
    ]
    if missing_required:
        raise ValueError(f"coarse capture is missing ordered fields: {missing_required}")
    missing_control = [field for field in OPTIONAL_PROJECTION_FIELDS if field not in control_set]
    missing_candidate = [field for field in OPTIONAL_PROJECTION_FIELDS if field not in candidate_set]
    available = [
        field for field in ORDERED_FIELDS if field in control_set and field in candidate_set
    ]
    return available, missing_control, missing_candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bit_equal(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return per-element byte equality without numeric-equality shortcuts."""
    left = np.ascontiguousarray(lhs)
    right = np.ascontiguousarray(rhs)
    if left.dtype != right.dtype:
        return np.zeros(left.shape, dtype=bool)
    byte_shape = left.shape + (left.dtype.itemsize,)
    return np.all(left.view(np.uint8).reshape(byte_shape) == right.view(np.uint8).reshape(byte_shape), axis=-1)


def _encoded_scalar(value: object, dtype: np.dtype) -> dict[str, object]:
    scalar = np.asarray(value, dtype=dtype).reshape(())
    encoded: dict[str, object] = {
        "dtype": str(dtype),
        "bytes_hex": scalar.tobytes().hex(),
    }
    if np.issubdtype(dtype, np.complexfloating):
        encoded["real"] = float(np.real(scalar))
        encoded["imag"] = float(np.imag(scalar))
    elif np.issubdtype(dtype, np.floating):
        encoded["value"] = float(scalar)
    elif np.issubdtype(dtype, np.integer):
        encoded["value"] = int(scalar)
    elif np.issubdtype(dtype, np.bool_):
        encoded["value"] = bool(scalar)
    else:
        encoded["value"] = str(scalar.item())
    return encoded


def _metrics(control: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(control)
    rhs = np.asarray(candidate)
    if lhs.shape != rhs.shape:
        return {
            "shape_equal": False,
            "control_shape": list(lhs.shape),
            "candidate_shape": list(rhs.shape),
        }
    metric_dtype = np.complex128 if np.iscomplexobj(rhs) or np.iscomplexobj(lhs) else np.float64
    lhs_metric = lhs.astype(metric_dtype)
    rhs_metric = rhs.astype(metric_dtype)
    numeric_equal = lhs == rhs
    bit_equal = _bit_equal(lhs, rhs)
    both_finite = np.isfinite(lhs_metric) & np.isfinite(rhs_metric)
    delta = np.zeros(lhs.shape, dtype=metric_dtype)
    delta[both_finite] = rhs_metric[both_finite] - lhs_metric[both_finite]
    delta[(~both_finite) & (~numeric_equal)] = np.inf
    lhs_finite = np.where(np.isfinite(lhs_metric), lhs_metric, 0.0)
    lhs_norm = float(np.linalg.norm(lhs_finite.reshape(-1)))
    unequal_flat = np.flatnonzero(~bit_equal)
    first_difference = None
    if unequal_flat.size:
        flat_index = int(unequal_flat[0])
        index = np.unravel_index(flat_index, lhs.shape)
        first_difference = {
            "flat_index": flat_index,
            "index": [int(item) for item in index],
            "control": _encoded_scalar(lhs[index], lhs.dtype),
            "candidate": _encoded_scalar(rhs[index], rhs.dtype),
        }
    return {
        "shape_equal": True,
        "shape": list(lhs.shape),
        "dtype_equal": bool(lhs.dtype == rhs.dtype),
        "control_dtype": str(lhs.dtype),
        "candidate_dtype": str(rhs.dtype),
        "bit_equal_fraction": float(np.mean(bit_equal)) if bit_equal.size else 1.0,
        "bit_unequal_count": int(unequal_flat.size),
        "first_bit_difference": first_difference,
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "relative_l2": (
            float(np.linalg.norm(delta.reshape(-1)) / lhs_norm)
            if lhs_norm > 0.0
            else 0.0 if not np.any(delta) else float("inf")
        ),
    }


def _centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    maximum = np.max(array[finite])
    return np.where(finite, array - maximum, -np.inf)


def analyze(*, control_path: Path, candidate_path: Path) -> dict[str, object]:
    with np.load(control_path, allow_pickle=False) as control, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        for scalar in (
            "original_index",
            "debug_iteration",
            "current_size",
            "n_classes",
            "n_rot",
            "n_trans",
        ):
            if int(np.asarray(control[scalar]).item()) != int(np.asarray(candidate[scalar]).item()):
                raise ValueError(f"capture scalar {scalar} differs")
        available_fields, missing_projection_control, missing_projection_candidate = (
            _available_ordered_fields(control.files, candidate.files)
        )

        fields = {
            field: _metrics(control[field], candidate[field])
            for field in available_fields
        }
        centered = {
            field + "_centered": _metrics(_centered(control[field]), _centered(candidate[field]))
            for field in ("scores_pre_prior_per_class", "scores_with_prior_per_class")
        }
        first_unequal = next(
            (
                field
                for field in ORDERED_FIELDS
                if field in fields
                if not fields[field].get("shape_equal", False)
                or fields[field].get("bit_equal_fraction") != 1.0
            ),
            None,
        )
        control_mask = np.asarray(control["significant_mask"], dtype=bool).reshape(
            int(control["n_rot"]), int(control["n_trans"])
        )
        candidate_mask = np.asarray(candidate["significant_mask"], dtype=bool).reshape(
            control_mask.shape
        )
        control_raw = np.asarray(control["scores_pre_prior_per_class"], dtype=np.float32)[0]
        candidate_raw = np.asarray(candidate["scores_pre_prior_per_class"], dtype=np.float32)[0]
        control_pre = np.asarray(control["scores_with_prior_per_class"], dtype=np.float32)[0]
        candidate_pre = np.asarray(candidate["scores_with_prior_per_class"], dtype=np.float32)[0]
        control_weights = np.asarray(control["weights_per_class"], dtype=np.float32)[0].reshape(
            control_mask.shape
        )
        candidate_weights = np.asarray(candidate["weights_per_class"], dtype=np.float32)[0].reshape(
            control_mask.shape
        )
        control_hard = np.unravel_index(
            int(np.asarray(control["hard_assignment"]).item()), control_mask.shape
        )
        candidate_hard = np.unravel_index(
            int(np.asarray(candidate["hard_assignment"]).item()), candidate_mask.shape
        )
        support_changes = [
            {
                "rotation": int(rotation),
                "translation": int(translation),
                "control_selected": bool(control_mask[rotation, translation]),
                "candidate_selected": bool(candidate_mask[rotation, translation]),
                "control_raw_log_score": float(control_raw[rotation, translation]),
                "candidate_raw_log_score": float(candidate_raw[rotation, translation]),
                "candidate_minus_control_raw_log_score": float(
                    candidate_raw[rotation, translation] - control_raw[rotation, translation]
                ),
                "control_centered_preexponent": float(
                    control_pre[rotation, translation] - control_pre[control_hard]
                ),
                "candidate_centered_preexponent": float(
                    candidate_pre[rotation, translation] - candidate_pre[candidate_hard]
                ),
                "control_posterior": float(control_weights[rotation, translation]),
                "candidate_posterior": float(candidate_weights[rotation, translation]),
            }
            for rotation, translation in np.argwhere(control_mask != candidate_mask)
        ]
        summary = {
            "control_n_significant": int(np.asarray(control["n_significant"]).item()),
            "candidate_n_significant": int(np.asarray(candidate["n_significant"]).item()),
            "control_pmax": float(np.asarray(control["max_posterior"]).item()),
            "candidate_pmax": float(np.asarray(candidate["max_posterior"]).item()),
            "control_hard_assignment": int(np.asarray(control["hard_assignment"]).item()),
            "candidate_hard_assignment": int(np.asarray(candidate["hard_assignment"]).item()),
            "control_hard_coordinate": [int(control_hard[0]), int(control_hard[1])],
            "candidate_hard_coordinate": [int(candidate_hard[0]), int(candidate_hard[1])],
            "support_symmetric_difference_count": len(support_changes),
            "support_changes": support_changes,
        }

    return {
        "schema": "recovar.em.k1_coarse_capture_ab.v1",
        "status": "complete",
        "first_non_bit_exact_field": first_unequal,
        "unavailable_optional_fields": sorted(
            set(missing_projection_control) | set(missing_projection_candidate)
        ),
        "unavailable_optional_fields_by_arm": {
            "control": missing_projection_control,
            "candidate": missing_projection_candidate,
        },
        "summary": summary,
        "fields": fields,
        "derived_fields": centered,
        "artifacts": {
            "control": str(control_path.resolve()),
            "control_sha256": _sha256(control_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(control_path=args.control, candidate_path=args.candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
