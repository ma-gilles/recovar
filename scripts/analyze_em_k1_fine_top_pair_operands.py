#!/usr/bin/env python3
"""Attribute one K=1 fine-grid winner flip with a fixed 2x2x2 operand swap.

The two candidates are fixed by a prior matrix-matched E-step comparison:
RELION's top candidate first and RECOVAR's top candidate second.  The analyzer
then swaps projected references, shifted images, and correction weights between
the engines without changing either candidate.  Pair margins are direct score
differences (candidate 1 minus candidate 2); no correlation is used as an
acceptance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_em_k1_live_reference_counterfactual import (  # noqa: E402
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)

SCHEMA = "em-k1-fine-top-pair-operands-v1"
INERTNESS_SCHEMA = "em-recovar-intermediate-capture-inertness-v1"
PRODUCTION_MARGIN_CLOSURE_GATE = 2.0e-4
ROTATION_MATRIX_MAX_FROBENIUS_GATE = 1.0e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source)
    target = np.asarray(target)
    _require(source.shape == target.shape, "relative-L2 shapes differ")
    denominator = float(np.linalg.norm(target))
    _require(denominator > 0.0, "relative-L2 target has zero norm")
    return float(np.linalg.norm(source - target) / denominator)


def score_candidate_components(
    projected_reference: np.ndarray,
    shifted_image: np.ndarray,
    correction_weight: np.ndarray,
    half_weights: np.ndarray,
) -> dict[str, float]:
    """Return direct fine-score components in RECOVAR's common units."""

    projected_reference = np.asarray(projected_reference, dtype=np.complex128)
    shifted_image = np.asarray(shifted_image, dtype=np.complex128)
    correction_weight = np.asarray(correction_weight, dtype=np.float64)
    half_weights = np.asarray(half_weights, dtype=np.float64)
    _require(projected_reference.ndim == 1, "projected reference must be 1D")
    _require(shifted_image.shape == projected_reference.shape, "shifted-image shape mismatch")
    _require(correction_weight.shape == projected_reference.shape, "correction-weight shape mismatch")
    _require(half_weights.shape == projected_reference.shape, "half-weight shape mismatch")
    norm = float(
        -0.5
        * np.sum(
            correction_weight * np.abs(projected_reference) ** 2 * half_weights,
            dtype=np.float64,
        )
    )
    cross = float(
        np.real(
            np.sum(
                np.conj(shifted_image)
                * projected_reference
                * correction_weight
                * half_weights,
                dtype=np.complex128,
            )
        )
    )
    return {"norm": norm, "cross": cross, "score": norm + cross}


def top_pair_margin(
    projected_references: np.ndarray,
    shifted_images: np.ndarray,
    correction_weight: np.ndarray,
    half_weights: np.ndarray,
) -> dict[str, Any]:
    """Score exactly two ordered candidates and return first-minus-second."""

    projected_references = np.asarray(projected_references)
    shifted_images = np.asarray(shifted_images)
    _require(projected_references.ndim == 2, "projected-reference pair must be 2D")
    _require(projected_references.shape[0] == 2, "projected-reference pair must contain two rows")
    _require(shifted_images.shape == projected_references.shape, "shifted-image pair shape mismatch")
    candidates = [
        score_candidate_components(
            projected_references[row],
            shifted_images[row],
            correction_weight,
            half_weights,
        )
        for row in range(2)
    ]
    margins = {
        name: float(candidates[0][name] - candidates[1][name])
        for name in ("norm", "cross", "score")
    }
    return {"candidate_components": candidates, "pair_margin": margins}


def classify_factorial(arms: dict[str, dict[str, Any]]) -> str:
    """Classify whether projected-reference source alone fixes winner sign."""

    expected = {
        f"{projected}{shifted}{correction}"
        for projected in "RC"
        for shifted in "RC"
        for correction in "RC"
    }
    _require(set(arms) == expected, "factorial must contain all eight R/C arms")
    relion_projection_margins = [
        float(arms[f"R{shifted}{correction}"]["pair_margin"]["score"])
        for shifted in "RC"
        for correction in "RC"
    ]
    recovar_projection_margins = [
        float(arms[f"C{shifted}{correction}"]["pair_margin"]["score"])
        for shifted in "RC"
        for correction in "RC"
    ]
    if all(value > 0.0 for value in relion_projection_margins) and all(
        value < 0.0 for value in recovar_projection_margins
    ):
        return "fine_winner_flip_is_projected_reference_determined"
    if all(value < 0.0 for value in relion_projection_margins) and all(
        value > 0.0 for value in recovar_projection_margins
    ):
        return "fine_winner_flip_is_projected_reference_determined_with_reversed_engine_signs"
    return "fine_winner_flip_has_mixed_operand_attribution"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "current_size",
            "rotations",
            "candidate_mask",
            "scores_pre_prior",
            "shifted_corrected",
            "ctf2_over_nv_score",
            "proj_half",
            "half_weights",
            "window_indices",
        }
        _require(required <= set(payload.files), f"missing RECOVAR fields: {sorted(required - set(payload.files))}")
        return {
            "original_index": int(np.asarray(payload["original_index"]).item()),
            "current_size": int(np.asarray(payload["current_size"]).item()),
            "rotations": np.asarray(payload["rotations"], dtype=np.float64),
            "candidate_mask": np.asarray(payload["candidate_mask"], dtype=bool),
            "scores_pre_prior": np.asarray(payload["scores_pre_prior"], dtype=np.float64),
            "shifted": np.asarray(payload["shifted_corrected"], dtype=np.complex128),
            "correction": np.asarray(payload["ctf2_over_nv_score"], dtype=np.float64),
            "references": np.asarray(payload["proj_half"], dtype=np.complex128),
            "half_weights": np.asarray(payload["half_weights"], dtype=np.float64),
            "window_indices": np.asarray(payload["window_indices"], dtype=np.int64),
        }


def _load_relion(path: Path, *, current_size: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "pass1_acc_stack_index",
            "pass1_acc_rot_idx",
            "pass1_acc_trans_idx",
            "pass1_exp_Mweight_raw_preprior",
            "pass1_class0_fine_eulers",
            "pass1_class0_fine_ref_real",
            "pass1_class0_fine_ref_imag",
            "pass1_class0_fine_shifted_real",
            "pass1_class0_fine_shifted_imag",
            "pass1_img0_corr_img",
            "pass1_img0_exp_current_image_size",
        }
        _require(required <= set(payload.files), f"missing RELION fields: {sorted(required - set(payload.files))}")
        relion_current_size = int(np.asarray(payload["pass1_img0_exp_current_image_size"]).item())
        _require(relion_current_size == current_size, "RELION current size does not match RECOVAR")
        n_current_pixels = current_size * (current_size // 2 + 1)
        rotations_flat = np.asarray(payload["pass1_class0_fine_eulers"], dtype=np.float64)
        _require(rotations_flat.size % 9 == 0, "RELION fine Euler matrix payload is malformed")
        rotation_rows = np.asarray(payload["pass1_acc_rot_idx"], dtype=np.int64).reshape(-1)
        translation_rows = np.asarray(payload["pass1_acc_trans_idx"], dtype=np.int64).reshape(-1)
        n_candidates = rotation_rows.size
        _require(translation_rows.size == n_candidates, "RELION candidate key arrays differ in size")

        def candidate_complex(real_name: str, imag_name: str) -> np.ndarray:
            real = np.asarray(payload[real_name], dtype=np.float64)
            imag = np.asarray(payload[imag_name], dtype=np.float64)
            _require(real.size == imag.size, f"RELION {real_name}/{imag_name} sizes differ")
            _require(
                real.size == n_candidates * n_current_pixels,
                f"RELION {real_name} does not contain one current-size row per candidate",
            )
            return (real + 1j * imag).reshape(n_candidates, n_current_pixels)

        raw_preprior = np.asarray(payload["pass1_exp_Mweight_raw_preprior"], dtype=np.float64).reshape(-1)
        _require(raw_preprior.size == n_candidates, "RELION raw-score count differs from candidate count")
        correction = np.asarray(payload["pass1_img0_corr_img"], dtype=np.float64).reshape(-1)
        _require(correction.size == n_current_pixels, "RELION correction-weight topology mismatch")
        return {
            "stack_index": int(np.asarray(payload["pass1_acc_stack_index"]).item()),
            "rotation_rows": rotation_rows,
            "translation_rows": translation_rows,
            "raw_preprior": raw_preprior,
            "rotation_matrices": rotations_flat.reshape(-1, 3, 3),
            "references": candidate_complex(
                "pass1_class0_fine_ref_real",
                "pass1_class0_fine_ref_imag",
            ),
            "shifted": candidate_complex(
                "pass1_class0_fine_shifted_real",
                "pass1_class0_fine_shifted_imag",
            ),
            "correction": correction,
        }


def _comparison_top_details(comparison: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    details = comparison.get("cross_top_candidate_details")
    _require(isinstance(details, list) and len(details) == 2, "comparison must contain two cross-top details")
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in details:
        key = tuple(int(value) for value in row["key"])
        _require(len(key) == 2 and key not in keyed, "comparison contains duplicate or malformed top keys")
        keyed[key] = row
    return keyed


def build_report(
    *,
    relion_npz: Path,
    recovar_npz: Path,
    comparison_json: Path,
    inertness_json: Path,
    expected_original_index: int,
    expected_current_size: int,
    full_image_size: int,
) -> dict[str, Any]:
    """Build a fail-closed, provenance-bound top-pair operand report."""

    relion_npz = Path(relion_npz).resolve()
    recovar_npz = Path(recovar_npz).resolve()
    comparison_json = Path(comparison_json).resolve()
    inertness_json = Path(inertness_json).resolve()
    comparison = _load_json(comparison_json)
    inertness = _load_json(inertness_json)

    _require(comparison.get("match_mode") == "matrix", "comparison must use matrix matching")
    _require(float(comparison.get("candidate_jaccard", -1.0)) == 1.0, "comparison candidate Jaccard must be 1")
    _require(int(comparison.get("relion_only_count", -1)) == 0, "comparison has RELION-only candidates")
    _require(int(comparison.get("recovar_only_count", -1)) == 0, "comparison has RECOVAR-only candidates")
    _require(int(comparison.get("recovar_original_index", -1)) == expected_original_index, "comparison particle mismatch")
    _require(int(comparison.get("recovar_current_size", -1)) == expected_current_size, "comparison current size mismatch")
    _require(inertness.get("schema") == INERTNESS_SCHEMA, "capture-inertness schema mismatch")
    _require(inertness.get("capture_inertness_qualified") is True, "capture inertness is not qualified")
    strict_gate = inertness.get("strict_gate", {})
    _require(
        strict_gate.get("passed") == strict_gate.get("expected") == 3
        and strict_gate.get("evaluated") == 3,
        "capture inertness must pass all three fixed comparisons",
    )

    relion_top_key = tuple(int(value) for value in comparison["relion_top_key"])
    recovar_top_key = tuple(int(value) for value in comparison["recovar_top_key"])
    _require(relion_top_key != recovar_top_key, "comparison does not contain a winner flip")
    top_keys = (relion_top_key, recovar_top_key)
    top_details = _comparison_top_details(comparison)
    _require(set(top_details) == set(top_keys), "comparison top-detail keys do not match winners")

    recovar = _load_recovar(recovar_npz)
    _require(recovar["original_index"] == expected_original_index, "RECOVAR particle mismatch")
    _require(recovar["current_size"] == expected_current_size, "RECOVAR current size mismatch")
    _require(full_image_size > expected_current_size > 0, "full/current image sizes are invalid")
    _require(recovar["rotations"].shape[1:] == (3, 3), "RECOVAR rotation topology mismatch")
    _require(recovar["references"].shape[0] == recovar["rotations"].shape[0], "RECOVAR reference/rotation mismatch")
    _require(recovar["shifted"].shape[0] == recovar["candidate_mask"].shape[1], "RECOVAR shifted/translation mismatch")
    n_score_pixels = recovar["references"].shape[1]
    for name in ("shifted",):
        _require(recovar[name].shape[1] == n_score_pixels, f"RECOVAR {name} pixel mismatch")
    for name in ("correction", "half_weights", "window_indices"):
        _require(recovar[name].shape == (n_score_pixels,), f"RECOVAR {name} pixel mismatch")

    relion = _load_relion(relion_npz, current_size=expected_current_size)
    _require(relion["stack_index"] == expected_original_index + 1, "RELION one-based stack index mismatch")
    nearest_rows, matrix_distance, orientation = _nearest_rotation_rows_by_matrix(
        relion["rotation_matrices"],
        recovar["rotations"],
    )
    _require(np.unique(nearest_rows).size == relion["rotation_matrices"].shape[0], "rotation mapping is not one-to-one")
    _require(
        float(np.max(matrix_distance)) <= ROTATION_MATRIX_MAX_FROBENIUS_GATE,
        "rotation matrix match exceeds fixed Frobenius gate",
    )
    expected_orientation = comparison.get("match_details", {}).get("rotation_matrix_orientation")
    _require(orientation == expected_orientation, "rotation orientation differs from comparison")

    relion_candidate_rows: list[int] = []
    for rotation_row, translation_row in top_keys:
        matching = np.flatnonzero(
            (nearest_rows[relion["rotation_rows"]] == rotation_row)
            & (relion["translation_rows"] == translation_row)
        )
        _require(matching.size == 1, f"RELION top key {(rotation_row, translation_row)} is not unique")
        relion_candidate_rows.append(int(matching[0]))
        _require(
            bool(recovar["candidate_mask"][rotation_row, translation_row]),
            f"RECOVAR top key {(rotation_row, translation_row)} is outside candidate support",
        )

    recovar_rotation_rows = np.asarray([key[0] for key in top_keys], dtype=np.int64)
    recovar_translation_rows = np.asarray([key[1] for key in top_keys], dtype=np.int64)
    relion_reference = relion_reference_on_recovar_window(
        relion["references"][relion_candidate_rows],
        recovar["window_indices"],
        full_image_size=full_image_size,
        current_size=expected_current_size,
    )
    relion_shifted = relion_reference_on_recovar_window(
        relion["shifted"][relion_candidate_rows],
        recovar["window_indices"],
        full_image_size=full_image_size,
        current_size=expected_current_size,
    )
    relion_correction = (
        relion_values_on_recovar_window(
            relion["correction"][np.newaxis, :],
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=expected_current_size,
        )[0]
        / (full_image_size**4)
    )
    recovar_reference = recovar["references"][recovar_rotation_rows]
    recovar_shifted = recovar["shifted"][recovar_translation_rows]
    recovar_correction = recovar["correction"]

    references = {"R": relion_reference, "C": recovar_reference}
    shifted = {"R": relion_shifted, "C": recovar_shifted}
    correction = {"R": relion_correction, "C": recovar_correction}
    arms: dict[str, dict[str, Any]] = {}
    for projected_source in "RC":
        for shifted_source in "RC":
            for correction_source in "RC":
                label = projected_source + shifted_source + correction_source
                arms[label] = top_pair_margin(
                    references[projected_source],
                    shifted[shifted_source],
                    correction[correction_source],
                    recovar["half_weights"],
                )

    production_relion_margin = float(
        top_details[relion_top_key]["relion"]["score_pre_prior"]
        - top_details[recovar_top_key]["relion"]["score_pre_prior"]
    )
    production_recovar_margin = float(
        top_details[relion_top_key]["recovar"]["score_pre_prior"]
        - top_details[recovar_top_key]["recovar"]["score_pre_prior"]
    )
    closure = {
        "relion_rrr_absolute_error": abs(
            float(arms["RRR"]["pair_margin"]["score"]) - production_relion_margin
        ),
        "recovar_ccc_absolute_error": abs(
            float(arms["CCC"]["pair_margin"]["score"]) - production_recovar_margin
        ),
    }
    closure["passed"] = bool(
        max(closure["relion_rrr_absolute_error"], closure["recovar_ccc_absolute_error"])
        <= PRODUCTION_MARGIN_CLOSURE_GATE
    )
    _require(
        closure["passed"],
        "double-precision score recomputation does not close to production margins: "
        f"{closure}",
    )

    projection_source_effects: dict[str, Any] = {}
    for shifted_source in "RC":
        for correction_source in "RC":
            fixed_sources = shifted_source + correction_source
            relion_arm = arms["R" + fixed_sources]["pair_margin"]
            recovar_arm = arms["C" + fixed_sources]["pair_margin"]
            norm_change = float(recovar_arm["norm"] - relion_arm["norm"])
            cross_change = float(recovar_arm["cross"] - relion_arm["cross"])
            projection_source_effects[fixed_sources] = {
                "recovar_minus_relion_norm_pair_margin": norm_change,
                "recovar_minus_relion_cross_pair_margin": cross_change,
                "recovar_minus_relion_score_pair_margin": float(
                    recovar_arm["score"] - relion_arm["score"]
                ),
                "norm_change_dominates_cross_change": bool(abs(norm_change) > abs(cross_change)),
            }

    classification = classify_factorial(arms)
    _require(
        classification == "fine_winner_flip_is_projected_reference_determined",
        f"fixed factorial did not isolate projected reference: {classification}",
    )
    norm_dominates = all(
        row["norm_change_dominates_cross_change"]
        for row in projection_source_effects.values()
    )
    _require(norm_dominates, "projection-norm change does not dominate every fixed shifted/correction arm")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": classification,
        "component_attribution": "projection_norm_change_dominates_cross_change",
        "metric_policy": (
            "direct candidate score margins and operand errors only; "
            "capture acceptance is shellwise FSC/FSC-AUC; no correlation"
        ),
        "identity": {
            "recovar_original_index_zero_based": expected_original_index,
            "relion_stack_index_one_based": relion["stack_index"],
            "current_size": expected_current_size,
            "full_image_size": full_image_size,
            "ordered_top_keys": [list(key) for key in top_keys],
            "first_key_source": "relion_top",
            "second_key_source": "recovar_top",
        },
        "capture_inertness_gate": {
            "schema": inertness["schema"],
            "status": inertness["status"],
            "capture_inertness_qualified": inertness["capture_inertness_qualified"],
            "strict_gate": strict_gate,
            "fsc_auc_non_dc_threshold": inertness["fsc_auc_non_dc_threshold"],
            "fsc_auc_non_dc": {
                name: float(inertness["comparisons"][name]["fsc_auc_non_dc"])
                for name in ("half1", "half2", "merged")
            },
        },
        "candidate_gate": {
            "candidate_jaccard": float(comparison["candidate_jaccard"]),
            "common_candidate_count": int(comparison["common_candidate_count"]),
            "relion_only_count": int(comparison["relion_only_count"]),
            "recovar_only_count": int(comparison["recovar_only_count"]),
        },
        "rotation_matrix_match": {
            "orientation": orientation,
            "median_frobenius": float(np.median(matrix_distance)),
            "max_frobenius": float(np.max(matrix_distance)),
            "max_frobenius_gate": ROTATION_MATRIX_MAX_FROBENIUS_GATE,
            "one_to_one": True,
        },
        "operand_relative_l2_relion_minus_recovar_over_recovar": {
            "projected_reference_top_pair": _relative_l2(relion_reference, recovar_reference),
            "shifted_image_top_pair": _relative_l2(relion_shifted, recovar_shifted),
            "correction_weight": _relative_l2(relion_correction, recovar_correction),
        },
        "production_pair_margins_first_minus_second": {
            "relion": production_relion_margin,
            "recovar": production_recovar_margin,
        },
        "production_margin_closure": {
            **closure,
            "absolute_error_gate": PRODUCTION_MARGIN_CLOSURE_GATE,
        },
        "factorial_source_key": {
            "R": "RELION",
            "C": "RECOVAR",
            "positions": ["projected_reference", "shifted_image", "correction_weight"],
        },
        "factorial": arms,
        "projection_source_effects_at_fixed_shifted_and_correction_sources": projection_source_effects,
        "input_artifacts": {
            str(path): _sha256(path)
            for path in (relion_npz, recovar_npz, comparison_json, inertness_json)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-npz", type=Path, required=True)
    parser.add_argument("--recovar-pass2-npz", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--inertness-json", type=Path, required=True)
    parser.add_argument("--expected-original-index", type=int, required=True)
    parser.add_argument("--expected-current-size", type=int, required=True)
    parser.add_argument("--full-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    report = build_report(
        relion_npz=args.relion_npz,
        recovar_npz=args.recovar_pass2_npz,
        comparison_json=args.comparison_json,
        inertness_json=args.inertness_json,
        expected_original_index=args.expected_original_index,
        expected_current_size=args.expected_current_size,
        full_image_size=args.full_image_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
