#!/usr/bin/env python3
"""Attribute the fixed K=1 coarse residual using captured score components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    validate_directory,
)

COMPONENT_DOMINANCE_FRACTION = 0.5
RECOVAR_REPLAY_P95_GATE = 5.0e-5
RECOVAR_REPLAY_MAX_GATE = 2.0e-4
CROSS_ENGINE_CLOSURE_P95_GATE = 1.0e-4
CROSS_ENGINE_CLOSURE_MAX_GATE = 5.0e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values)


def decompose_captured_residual(
    total_residual: np.ndarray,
    reference_norm_residual: np.ndarray,
    cross_residual: np.ndarray,
) -> dict[str, Any]:
    """Measure counterfactual residual-energy removal without correlation."""

    total = np.asarray(total_residual, dtype=np.float64)
    norm = np.asarray(reference_norm_residual, dtype=np.float64)
    cross = np.asarray(cross_residual, dtype=np.float64)
    _require(total.shape == norm.shape == cross.shape, "component shapes differ")
    _require(total.ndim == 2 and min(total.shape) > 1, "component panel is too small")
    _require(np.all(np.isfinite(total)), "total residual is not finite")
    _require(np.all(np.isfinite(norm)), "reference-norm residual is not finite")
    _require(np.all(np.isfinite(cross)), "cross residual is not finite")

    total_centered = _center(total)
    norm_centered = _center(norm)
    cross_centered = _center(cross)
    closure = total_centered - _center(norm + cross)
    total_energy = float(np.sum(total_centered**2))
    _require(total_energy > 0.0, "total residual has zero centered energy")
    without_norm_energy = float(np.sum(_center(total - norm) ** 2))
    without_cross_energy = float(np.sum(_center(total - cross) ** 2))
    norm_removal = 1.0 - without_norm_energy / total_energy
    cross_removal = 1.0 - without_cross_energy / total_energy
    absolute_closure = np.abs(closure)
    return {
        "total_centered_energy": total_energy,
        "reference_norm_centered_energy": float(np.sum(norm_centered**2)),
        "cross_centered_energy": float(np.sum(cross_centered**2)),
        "counterfactual_energy_removal_fraction": {
            "reference_norm": float(norm_removal),
            "cross": float(cross_removal),
        },
        "closure": {
            "p95_abs": float(np.percentile(absolute_closure, 95)),
            "max_abs": float(np.max(absolute_closure)),
        },
        "reference_norm_dominated": bool(
            norm_removal > COMPONENT_DOMINANCE_FRACTION
            and norm_removal > cross_removal
        ),
        "cross_dominated": bool(
            cross_removal > COMPONENT_DOMINANCE_FRACTION
            and cross_removal > norm_removal
        ),
    }


def _load_recovar_components(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "scores_pre_prior_per_class",
            "translations",
            "projected_reference_rotation_ids",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        _require(required <= set(payload.files), f"missing component fields: {path}")
        scores = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        ids = np.asarray(payload["projected_reference_rotation_ids"], dtype=np.int64)
        norms = np.asarray(
            payload["projected_reference_norm_score_per_class"],
            dtype=np.float64,
        )
        crosses = np.asarray(
            payload["projected_cross_score_per_class"],
            dtype=np.float64,
        )
        _require(scores.ndim == 3 and scores.shape[0] == 1, "expected K=1 scores")
        _require(
            norms.shape == crosses.shape == (1, ids.size, scores.shape[2]),
            f"RECOVAR component topology mismatch: {path}",
        )
        _require(np.unique(ids).size == ids.size, f"duplicate projection ID: {path}")
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "original_index": int(np.asarray(payload["original_index"]).item()),
            "scores": scores[0],
            "rotation_ids": ids,
            "reference_norm_scores": norms[0],
            "cross_scores": crosses[0],
            "translations": np.asarray(payload["translations"], dtype=np.float64),
        }


def build_report(
    *,
    cohort_json: Path,
    relion_directory: Path,
    recovar_directory: Path,
) -> dict[str, Any]:
    cohort = json.loads(Path(cohort_json).read_text())
    _require(cohort["selected_particle_count"] == 14, "cohort denominator must be 14")
    expected_stacks = np.asarray(
        cohort["selected_stack_indices_one_based"],
        dtype=np.int64,
    )
    relion_artifacts, relion_validation = validate_directory(
        relion_directory,
        expected_particles=14,
        expected_stack_indices=expected_stacks,
        expected_mpi_rank=1,
    )
    relion_by_stack = {item.stack_index: item for item in relion_artifacts}
    recovar_items = [
        _load_recovar_components(path)
        for path in sorted(Path(recovar_directory).glob("*.npz"))
    ]
    _require(len(recovar_items) == 14, "RECOVAR denominator must be 14")
    recovar_by_index = {item["original_index"]: item for item in recovar_items}
    _require(len(recovar_by_index) == 14, "duplicate RECOVAR original index")

    particles = []
    for cohort_row in cohort["rows"]:
        relion = relion_by_stack[cohort_row["stack_index_one_based"]]
        recovar = recovar_by_index[cohort_row["original_index_zero_based"]]
        n_directions, n_psi, _ = relion.header[10:13]
        translation_permutation, translation_mapping = _translation_permutation(
            relion.translations,
            recovar["translations"],
        )
        mapped_raw = _map_relion_table(
            relion.raw_diff2,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        mapped_norm = _map_relion_table(
            relion.reference_norms,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        mapped_cross = _map_relion_table(
            relion.cross_terms,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        rotation_ids = recovar["rotation_ids"]
        _require(
            np.all((rotation_ids >= 0) & (rotation_ids < mapped_raw.shape[0])),
            "projection rotation ID is out of range",
        )
        active = np.all(
            mapped_raw[rotation_ids] != RELION_INVALID_DIFF2,
            axis=1,
        )
        _require(np.count_nonzero(active) > 1, "too few common active rotations")
        selected_ids = rotation_ids[active]
        recovar_norm = recovar["reference_norm_scores"][active]
        recovar_cross = recovar["cross_scores"][active]
        recovar_total = recovar["scores"][selected_ids]
        recovar_replay_error = recovar_norm + recovar_cross - recovar_total
        replay_absolute = np.abs(recovar_replay_error)
        replay = {
            "p95_abs": float(np.percentile(replay_absolute, 95)),
            "max_abs": float(np.max(replay_absolute)),
        }
        total_residual = recovar_total + mapped_raw[selected_ids]
        norm_residual = recovar_norm + mapped_norm[selected_ids]
        cross_residual = recovar_cross + mapped_cross[selected_ids]
        decomposition = decompose_captured_residual(
            total_residual,
            norm_residual,
            cross_residual,
        )
        decomposition["closure_passed"] = bool(
            decomposition["closure"]["p95_abs"] <= CROSS_ENGINE_CLOSURE_P95_GATE
            and decomposition["closure"]["max_abs"] <= CROSS_ENGINE_CLOSURE_MAX_GATE
        )
        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": cohort_row["stack_index_one_based"],
                "original_index_zero_based": cohort_row["original_index_zero_based"],
                "active_requested_rotation_count": int(np.count_nonzero(active)),
                "requested_rotation_count": int(rotation_ids.size),
                "recovar_replay": replay,
                "recovar_replay_passed": bool(
                    replay["p95_abs"] <= RECOVAR_REPLAY_P95_GATE
                    and replay["max_abs"] <= RECOVAR_REPLAY_MAX_GATE
                ),
                "decomposition": decomposition,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    "relion": str(relion.path.resolve()),
                    "recovar": recovar["path"],
                },
                "artifact_sha256": {
                    "relion": relion.sha256,
                    "recovar": recovar["sha256"],
                },
            }
        )

    fixed_metric = {
        "evaluated_particles": len(particles),
        "expected_particles": 14,
        "recovar_replay_passed": sum(
            row["recovar_replay_passed"] for row in particles
        ),
        "cross_engine_closure_passed": sum(
            row["decomposition"]["closure_passed"] for row in particles
        ),
        "reference_norm_dominated": sum(
            row["decomposition"]["reference_norm_dominated"] for row in particles
        ),
        "cross_dominated": sum(
            row["decomposition"]["cross_dominated"] for row in particles
        ),
    }
    captures_qualified = bool(
        relion_validation["status"] == "pass"
        and fixed_metric["recovar_replay_passed"] == 14
        and fixed_metric["cross_engine_closure_passed"] == 14
    )
    if not captures_qualified:
        classification = "component_capture_not_qualified"
    elif fixed_metric["reference_norm_dominated"] == 14:
        classification = "raw_coarse_residual_is_reference_norm_dominated"
    elif fixed_metric["cross_dominated"] == 14:
        classification = "raw_coarse_residual_is_image_reference_cross_dominated"
    else:
        classification = "raw_coarse_residual_has_mixed_component_dominance"
    return {
        "schema": "recovar-k1-case22-captured-score-components-v1",
        "status": "complete",
        "classification": classification,
        "captures_qualified": captures_qualified,
        "metric_policy": (
            "fixed 14 particles and 13 requested canonical rotations; "
            "counterfactual centered residual-energy removal; component dominance "
            "requires >0.5 removal and strictly exceeds the other component; "
            "no correlation"
        ),
        "fixed_gates": {
            "component_dominance_fraction_strictly_above": (
                COMPONENT_DOMINANCE_FRACTION
            ),
            "recovar_replay_p95_abs_max": RECOVAR_REPLAY_P95_GATE,
            "recovar_replay_max_abs_max": RECOVAR_REPLAY_MAX_GATE,
            "cross_engine_closure_p95_abs_max": CROSS_ENGINE_CLOSURE_P95_GATE,
            "cross_engine_closure_max_abs_max": CROSS_ENGINE_CLOSURE_MAX_GATE,
        },
        "fixed_metric": fixed_metric,
        "particles": particles,
        "relion_validation": relion_validation,
        "cohort": str(Path(cohort_json).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--relion-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        cohort_json=args.cohort_json,
        relion_directory=args.relion_directory,
        recovar_directory=args.recovar_directory,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "captures_qualified": report["captures_qualified"],
                "classification": report["classification"],
                "fixed_metric": report["fixed_metric"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
