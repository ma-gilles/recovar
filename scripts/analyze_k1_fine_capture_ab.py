#!/usr/bin/env python3
"""Locate the first unequal field between two RECOVAR K=1 fine captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_k1_coarse_capture_ab import _metrics, _sha256
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from analyze_k1_coarse_capture_ab import _metrics, _sha256


ORDERED_FIELDS = (
    "fine_translations",
    "rotations",
    "oversampled_rot_indices",
    "parent_map",
    "candidate_mask",
    "window_indices",
    "recon_window_indices",
    "relion_preprocess_normalization_factor",
    "relion_integer_pre_shift",
    "batch_image_correction",
    "batch_scale_correction",
    "direct_score_input",
    "direct_preprocessed_score_input",
    "direct_pixel_correction",
    "direct_inverse_noise_score",
    "direct_ctf_rfloat_score",
    "raw_operand_corr_img_score",
    "shifted_corrected",
    "ctf2_over_nv_score",
    "proj_half",
    "half_weights",
    "relion_highres_xi2_half",
    "raw_operand_raw_diff2",
    "relion_raw_diff2",
    "rotation_log_prior",
    "translation_log_prior",
    "scores_pre_prior",
    "scores_with_prior",
    "probs",
    "reconstruction_mask",
    "reconstruction_probs",
    "reconstruction_n_significant",
)


def _scalar(archive, name: str) -> int:
    return int(np.asarray(archive[name]).item())


def _winner(scores: np.ndarray, candidate_mask: np.ndarray) -> list[int]:
    values = np.asarray(scores)
    active = np.asarray(candidate_mask, dtype=bool) & np.isfinite(values)
    if not np.any(active):
        raise ValueError("fine capture contains no active finite candidate")
    masked = np.where(active, values, -np.inf)
    return [int(value) for value in np.unravel_index(int(np.argmax(masked)), values.shape)]


def analyze(
    *,
    control_path: Path,
    candidate_path: Path,
    allow_iteration_mismatch: bool = False,
) -> dict[str, object]:
    with np.load(control_path, allow_pickle=False) as control, np.load(
        candidate_path,
        allow_pickle=False,
    ) as candidate:
        scalar_names = ("iteration", "half", "original_index", "current_size", "n_fine_trans")
        scalar_context = {
            scalar: {
                "control": _scalar(control, scalar),
                "candidate": _scalar(candidate, scalar),
            }
            for scalar in scalar_names
        }
        for scalar in scalar_names:
            if (
                scalar == "iteration"
                and allow_iteration_mismatch
            ):
                continue
            if scalar_context[scalar]["control"] != scalar_context[scalar]["candidate"]:
                raise ValueError(f"capture scalar {scalar} differs")
        missing = [
            field
            for field in ORDERED_FIELDS
            if field not in control.files or field not in candidate.files
        ]
        if missing:
            raise ValueError(f"fine capture is missing ordered fields: {missing}")

        fields = {
            field: _metrics(control[field], candidate[field])
            for field in ORDERED_FIELDS
        }
        first_unequal = next(
            (
                field
                for field in ORDERED_FIELDS
                if not fields[field].get("shape_equal", False)
                or fields[field].get("bit_equal_fraction") != 1.0
            ),
            None,
        )
        control_mask = np.asarray(control["candidate_mask"], dtype=bool)
        candidate_mask = np.asarray(candidate["candidate_mask"], dtype=bool)
        control_support = np.asarray(control["reconstruction_mask"], dtype=bool)
        candidate_support = np.asarray(candidate["reconstruction_mask"], dtype=bool)
        summary = {
            "control_candidate_count": int(np.count_nonzero(control_mask)),
            "candidate_candidate_count": int(np.count_nonzero(candidate_mask)),
            "candidate_symmetric_difference_count": int(
                np.count_nonzero(control_mask != candidate_mask)
            ),
            "control_significant_count": int(np.count_nonzero(control_support)),
            "candidate_significant_count": int(np.count_nonzero(candidate_support)),
            "support_symmetric_difference_count": int(
                np.count_nonzero(control_support != candidate_support)
            ),
            "control_pmax": float(np.max(np.asarray(control["probs"], dtype=np.float64))),
            "candidate_pmax": float(np.max(np.asarray(candidate["probs"], dtype=np.float64))),
            "control_winner": _winner(control["scores_with_prior"], control_mask),
            "candidate_winner": _winner(candidate["scores_with_prior"], candidate_mask),
        }

    return {
        "schema": "recovar.em.k1_fine_capture_ab.v1",
        "status": "complete",
        "metric_policy": "exact ordered intermediates and relative L2; no correlation",
        "scalar_context": scalar_context,
        "iteration_mismatch_allowed": bool(allow_iteration_mismatch),
        "first_non_bit_exact_field": first_unequal,
        "summary": summary,
        "fields": fields,
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
    parser.add_argument(
        "--allow-iteration-mismatch",
        action="store_true",
        help=(
            "Compare a fresh and continuation capture of the same physical boundary "
            "even when their local iteration labels differ"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        control_path=args.control,
        candidate_path=args.candidate,
        allow_iteration_mismatch=args.allow_iteration_mismatch,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
