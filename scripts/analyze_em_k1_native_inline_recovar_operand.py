#!/usr/bin/env python3
"""Compare one native RELION coarse CUDA operand to RECOVAR's exact operand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _relion_parent_to_recovar,
    _translation_permutation,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    recovar_score_components,
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts import validate_relion_coarse_inline_capture as inline_validator
from scripts import validate_relion_coarse_lane_capture as lane_validator
from scripts import validate_relion_coarse_operand_capture as operand_validator
from scripts import validate_relion_coarse_pass1_components as component_validator

DOMINANCE_FRACTION = 0.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    _require(actual.shape == expected.shape, "operand comparison shapes differ")
    difference = np.abs(actual.astype(np.complex128) - expected.astype(np.complex128))
    denominator = float(np.linalg.norm(expected))
    _require(denominator > 0.0, "operand comparison target has zero norm")
    return {
        "relative_l2": float(np.linalg.norm(actual - expected) / denominator),
        "p95_abs": float(np.percentile(difference, 95)),
        "max_abs": float(np.max(difference)),
    }


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values)


def _score(values: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    norm, cross = values
    _require(norm.shape == cross.shape == (1, 29), "unexpected score topology")
    return norm[0] + cross[0]


def _residual_intervention(
    baseline_residual: np.ndarray,
    actual_recovar_score: np.ndarray,
    recovar_formula_score: np.ndarray,
    arm_formula_score: np.ndarray,
    relion_raw_diff2: np.ndarray,
) -> dict[str, float | bool]:
    delta = arm_formula_score - recovar_formula_score
    residual = _center(actual_recovar_score + delta + relion_raw_diff2)
    baseline_energy = float(np.sum(baseline_residual**2))
    residual_energy = float(np.sum(residual**2))
    _require(baseline_energy > 0.0, "baseline residual has zero centered energy")
    removal = 1.0 - residual_energy / baseline_energy
    absolute = np.abs(residual)
    return {
        "centered_residual_energy": residual_energy,
        "centered_residual_energy_removal_fraction": float(removal),
        "centered_residual_p95_abs": float(np.percentile(absolute, 95)),
        "centered_residual_max_abs": float(np.max(absolute)),
        "dominant": bool(removal > DOMINANCE_FRACTION),
    }


def build_report(
    *,
    inline_path: Path,
    lane_path: Path,
    component_path: Path,
    operand_path: Path,
    recovar_path: Path,
    full_image_size: int,
) -> dict[str, object]:
    inline = inline_validator.load_artifact(inline_path)
    lane = lane_validator.load_artifact(lane_path)
    component = component_validator.load_artifact(component_path)
    operand = operand_validator.load_artifact(operand_path)
    inline_validation = inline_validator.validate_capture(inline, operand, lane)
    lane_validation = lane_validator.validate_capture(lane, operand, component)
    _require(inline_validation["classification_ready"] is True, "native inline capture did not close")
    _require(lane_validation["capture_qualified"] is True, "native lane capture did not close")
    _require(
        lane_validation["fixed_metric"]["atomic_target_exactly_reachable"] == 29,
        "not all native production scores are reachable",
    )
    _require(inline.part_id == component.part_id == operand.part_id, "RELION particle identities differ")
    _require(inline.stack_index == component.stack_index == operand.stack_index == 21875, "stack identity changed")
    _require(inline.header[12] == component.header[27] == 100, "current size changed")

    with np.load(recovar_path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "current_size",
            "translations",
            "window_indices",
            "half_weights",
            "shifted_data",
            "ctf2_data",
            "scores_pre_prior_per_class",
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
        }
        _require(required <= set(payload.files), "RECOVAR operand fields are incomplete")
        _require(int(payload["original_index"]) == 21874, "RECOVAR particle identity changed")
        _require(int(payload["current_size"]) == 100, "RECOVAR current size changed")
        translations = np.asarray(payload["translations"], dtype=np.float64)
        window_indices = np.asarray(payload["window_indices"], dtype=np.int64)
        half_weights = np.asarray(payload["half_weights"], dtype=np.float64)
        recovar_shifted = np.asarray(payload["shifted_data"], dtype=np.complex128)
        recovar_ctf2 = np.asarray(payload["ctf2_data"], dtype=np.float64)[0]
        recovar_scores = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)[0]
        projection_ids = np.asarray(payload["projected_reference_rotation_ids"], dtype=np.int64)
        projections = np.asarray(payload["projected_reference_per_class"], dtype=np.complex128)[0]

    n_directions, n_psi, n_translations = component.header[10:13]
    _require(n_translations == 29, "translation denominator changed")
    recovar_rotation = _relion_parent_to_recovar(
        inline.rotation_key,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    matches = np.flatnonzero(projection_ids == recovar_rotation)
    _require(matches.size == 1, "native rotation is absent from RECOVAR projection capture")
    recovar_reference = projections[int(matches[0])]

    # The operand artifact stores phase arguments, not physical translations.
    # Candidate identity must be joined through the component artifact's
    # source translation table, which is in the audited RELION coordinates.
    relion_translations = np.asarray(component.translations, dtype=np.float64)
    translation_permutation, translation_mapping = _translation_permutation(
        relion_translations,
        translations,
    )
    native_reference_fftw = (
        inline.fields[0, 0].astype(np.float64)
        + 1j * inline.fields[1, 0].astype(np.float64)
    )[np.newaxis, :]
    native_reference = relion_reference_on_recovar_window(
        native_reference_fftw,
        window_indices,
        full_image_size=full_image_size,
        current_size=100,
    )[0]
    native_shifted_fftw = (
        inline.fields[2].astype(np.float64)
        + 1j * inline.fields[3].astype(np.float64)
    )
    native_shifted_window = relion_values_on_recovar_window(
        native_shifted_fftw,
        window_indices,
        full_image_size=full_image_size,
        current_size=100,
    )
    native_shifted_ordered = np.empty_like(native_shifted_window)
    native_shifted_ordered[translation_permutation] = native_shifted_window
    native_correction = relion_values_on_recovar_window(
        (2.0 * inline.fields[4, 0].astype(np.float64))[np.newaxis, :],
        window_indices,
        full_image_size=full_image_size,
        current_size=100,
    )[0].real
    image_normalization = float(full_image_size**2)
    native_shifted = (
        -native_shifted_ordered
        * native_correction[np.newaxis, :]
        / (image_normalization * half_weights[np.newaxis, :])
    )
    native_ctf2 = native_correction / (image_normalization**2 * half_weights)

    recovar_formula = _score(
        recovar_score_components(
            recovar_reference[np.newaxis, :],
            recovar_shifted,
            recovar_ctf2,
            half_weights,
        )
    )
    arms = {
        "reference": _score(
            recovar_score_components(
                native_reference[np.newaxis, :], recovar_shifted, recovar_ctf2, half_weights
            )
        ),
        "shifted_image": _score(
            recovar_score_components(
                recovar_reference[np.newaxis, :], native_shifted, recovar_ctf2, half_weights
            )
        ),
        "correction": _score(
            recovar_score_components(
                recovar_reference[np.newaxis, :], recovar_shifted, native_ctf2, half_weights
            )
        ),
        "all_native": _score(
            recovar_score_components(
                native_reference[np.newaxis, :], native_shifted, native_ctf2, half_weights
            )
        ),
    }
    actual_recovar_score = recovar_scores[recovar_rotation]
    relion_raw = np.asarray(component.raw_diff2, dtype=np.float64).reshape(
        n_directions * n_psi, n_translations
    )[inline.rotation_key]
    relion_raw_ordered = np.empty_like(relion_raw)
    relion_raw_ordered[translation_permutation] = relion_raw
    baseline = _center(actual_recovar_score + relion_raw_ordered)
    interventions = {
        name: _residual_intervention(
            baseline,
            actual_recovar_score,
            recovar_formula,
            score,
            relion_raw_ordered,
        )
        for name, score in arms.items()
    }
    single_removals = {
        name: row["centered_residual_energy_removal_fraction"]
        for name, row in interventions.items()
        if name != "all_native"
    }
    strongest_single = max(single_removals, key=single_removals.get)
    all_native_dominant = bool(interventions["all_native"]["dominant"])
    component_dominant = bool(
        interventions[strongest_single]["dominant"]
        and single_removals[strongest_single]
        > max(value for name, value in single_removals.items() if name != strongest_single)
    )
    if not all_native_dominant:
        classification = "native_relion_operands_do_not_dominate_case04_fixed_candidate_residual"
    elif component_dominant:
        classification = f"native_relion_{strongest_single}_dominates_case04_fixed_candidate_residual"
    else:
        classification = "native_relion_operands_jointly_dominate_case04_fixed_candidate_residual"

    baseline_absolute = np.abs(baseline)
    return {
        "schema": "recovar.em_k1_native_inline_recovar_operand.v1",
        "status": "complete",
        "classification_ready": True,
        "classification": classification,
        "correlation_computed": False,
        "fixed_identity": {
            "original_index_zero_based": 21874,
            "stack_index_one_based": 21875,
            "relion_direction_major_rotation_key": int(inline.rotation_key),
            "recovar_psi_major_rotation_id": int(recovar_rotation),
            "translation_count": 29,
            "score_pixel_count": int(window_indices.size),
        },
        "fixed_gates": {
            "operand_dominance_fraction_strictly_greater_than": DOMINANCE_FRACTION,
        },
        "baseline": {
            "centered_residual_energy": float(np.sum(baseline**2)),
            "centered_residual_p95_abs": float(np.percentile(baseline_absolute, 95)),
            "centered_residual_max_abs": float(np.max(baseline_absolute)),
        },
        "operand_differences": {
            "reference": _difference_metrics(native_reference, recovar_reference),
            "shifted_image": _difference_metrics(native_shifted, recovar_shifted),
            "correction_ctf2": _difference_metrics(native_ctf2, recovar_ctf2),
        },
        "interventions": interventions,
        "strongest_single_component": strongest_single,
        "all_native_dominant": all_native_dominant,
        "single_component_dominant": component_dominant,
        "translation_mapping": translation_mapping,
        "native_validation": {
            "inline": inline_validation,
            "lane": lane_validation,
        },
        "artifacts": {
            "inline": {"path": str(inline_path.resolve()), "sha256": inline.sha256},
            "lane": {"path": str(lane_path.resolve()), "sha256": lane.sha256},
            "component": {"path": str(component_path.resolve()), "sha256": component.sha256},
            "operand": {"path": str(operand_path.resolve()), "sha256": operand.sha256},
            "recovar": {"path": str(recovar_path.resolve()), "sha256": _sha256(recovar_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inline", required=True, type=Path)
    parser.add_argument("--lane", required=True, type=Path)
    parser.add_argument("--component", required=True, type=Path)
    parser.add_argument("--operand", required=True, type=Path)
    parser.add_argument("--recovar", required=True, type=Path)
    parser.add_argument("--full-image-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    report = build_report(
        inline_path=args.inline,
        lane_path=args.lane,
        component_path=args.component,
        operand_path=args.operand,
        recovar_path=args.recovar,
        full_image_size=args.full_image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], "status": report["status"]}))


if __name__ == "__main__":
    main()
