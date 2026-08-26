#!/usr/bin/env python3
"""Compare two K=1 sparse pass-2 captures in causal boundary order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "recovar.em.k1_pass2_state_ab.v1"

FIELD_STAGES = (
    (
        "identity_and_topology",
        (
            "iteration",
            "half",
            "original_index",
            "local_index",
            "current_size",
            "n_fine_trans",
            "fine_translations",
            "rotations",
            "oversampled_rot_indices",
            "parent_map",
            "candidate_mask",
            "window_indices",
            "recon_window_indices",
        ),
    ),
    (
        "particle_score_inputs",
        (
            "relion_integer_pre_shift",
            "batch_image_correction",
            "batch_scale_correction",
            "relion_preprocess_normalization_factor",
            "direct_score_input",
            "direct_preprocessed_score_input",
            "direct_pixel_correction",
            "direct_inverse_noise_score",
            "direct_ctf_rfloat_score",
            "raw_operand_corr_img_score",
            "shifted_corrected",
            "raw_operand_shifted_corrected",
            "ctf2_over_nv_score",
            "shifted_recon",
            "ctf2_over_nv_recon",
            "half_weights",
            "raw_operand_half_weights",
        ),
    ),
    ("projected_reference", ("proj_half", "raw_operand_proj_half")),
    (
        "raw_score",
        (
            "raw_operand_raw_diff2",
            "relion_raw_diff2",
            "scores_pre_prior",
        ),
    ),
    ("priors", ("rotation_log_prior", "translation_log_prior")),
    ("total_score", ("scores_with_prior",)),
    ("posterior", ("probs",)),
    (
        "reconstruction_support",
        (
            "reconstruction_mask",
            "reconstruction_probs",
            "reconstruction_n_significant",
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(reference)
    rhs = np.asarray(candidate)
    result: dict[str, Any] = {
        "reference_shape": list(lhs.shape),
        "candidate_shape": list(rhs.shape),
        "reference_dtype": str(lhs.dtype),
        "candidate_dtype": str(rhs.dtype),
        "shape_equal": bool(lhs.shape == rhs.shape),
        "dtype_equal": bool(lhs.dtype == rhs.dtype),
    }
    if lhs.shape != rhs.shape:
        result.update({"byte_equal": False, "value_equal": False})
        return result

    lhs_contiguous = np.ascontiguousarray(lhs)
    rhs_contiguous = np.ascontiguousarray(rhs)
    byte_equal = bool(
        lhs.dtype == rhs.dtype
        and np.array_equal(lhs_contiguous.view(np.uint8), rhs_contiguous.view(np.uint8))
    )
    value_equal = bool(np.array_equal(lhs, rhs, equal_nan=True))
    result.update({"byte_equal": byte_equal, "value_equal": value_equal})

    if not (
        np.issubdtype(lhs.dtype, np.number)
        and np.issubdtype(rhs.dtype, np.number)
    ):
        return result

    finite_lhs = np.isfinite(lhs)
    finite_rhs = np.isfinite(rhs)
    finite_mask = finite_lhs & finite_rhs
    result["finite_mask_equal"] = bool(np.array_equal(finite_lhs, finite_rhs))
    result["finite_comparison_count"] = int(np.count_nonzero(finite_mask))
    unequal = np.not_equal(lhs, rhs)
    unequal &= ~(np.isnan(lhs) & np.isnan(rhs))
    result["value_mismatch_count"] = int(np.count_nonzero(unequal))
    if not np.any(finite_mask):
        result.update({"max_abs": None, "relative_l2": None})
        return result

    lhs_finite = lhs[finite_mask].astype(np.complex128 if np.iscomplexobj(lhs) else np.float64)
    rhs_finite = rhs[finite_mask].astype(np.complex128 if np.iscomplexobj(rhs) else np.float64)
    delta = rhs_finite - lhs_finite
    denominator = float(np.linalg.norm(lhs_finite.reshape(-1)))
    delta_norm = float(np.linalg.norm(delta.reshape(-1)))
    result["max_abs"] = float(np.max(np.abs(delta)))
    result["relative_l2"] = delta_norm / denominator if denominator else delta_norm
    mismatch = np.flatnonzero(unequal.reshape(-1))
    result["first_mismatch_flat_index"] = int(mismatch[0]) if mismatch.size else None
    return result


def _centered_score_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(reference)
    rhs = np.asarray(candidate)
    if lhs.shape != rhs.shape:
        return {"shape_equal": False}
    finite_lhs = np.isfinite(lhs)
    finite_rhs = np.isfinite(rhs)
    if not np.array_equal(finite_lhs, finite_rhs):
        return {"shape_equal": True, "finite_mask_equal": False}
    if not np.any(finite_lhs):
        return {"shape_equal": True, "finite_mask_equal": True, "metric": None}
    lhs_centered = lhs[finite_lhs] - np.max(lhs[finite_lhs])
    rhs_centered = rhs[finite_rhs] - np.max(rhs[finite_rhs])
    return {
        "shape_equal": True,
        "finite_mask_equal": True,
        "metric": _metric(lhs_centered, rhs_centered),
    }


def analyze(*, reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        common = set(reference.files) & set(candidate.files)
        stages: dict[str, Any] = {}
        first_unequal: dict[str, str] | None = None
        ordered_names: set[str] = set()
        for stage_name, field_names in FIELD_STAGES:
            stage: dict[str, Any] = {}
            for field_name in field_names:
                if field_name not in common:
                    stage[field_name] = {"present_in_both": False}
                    continue
                ordered_names.add(field_name)
                metric = _metric(reference[field_name], candidate[field_name])
                if field_name in {"raw_operand_raw_diff2", "relion_raw_diff2", "scores_pre_prior", "scores_with_prior"}:
                    metric["centered"] = _centered_score_metric(
                        reference[field_name], candidate[field_name]
                    )
                stage[field_name] = {"present_in_both": True, **metric}
                if first_unequal is None and not bool(metric["byte_equal"]):
                    first_unequal = {"stage": stage_name, "field": field_name}
            stages[stage_name] = stage

        remaining: dict[str, Any] = {}
        for field_name in sorted(common - ordered_names):
            lhs = np.asarray(reference[field_name])
            rhs = np.asarray(candidate[field_name])
            if np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number):
                remaining[field_name] = _metric(lhs, rhs)

        summary: dict[str, Any] = {"first_unequal": first_unequal}
        for label, archive in (("reference", reference), ("candidate", candidate)):
            probs = np.asarray(archive["probs"], dtype=np.float64)
            summary[label] = {
                "pmax": float(np.max(probs)),
                "winner_flat_index": int(np.argmax(probs)),
                "candidate_count": int(np.count_nonzero(archive["candidate_mask"])),
                "reconstruction_support_count": int(
                    np.count_nonzero(archive["reconstruction_mask"])
                ),
            }

    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "summary": summary,
        "stages": stages,
        "remaining_common_numeric_fields": remaining,
        "artifacts": {
            "reference": str(reference_path.resolve()),
            "reference_sha256": _sha256(reference_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(reference_path=args.reference, candidate_path=args.candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
