#!/usr/bin/env python3
"""Decompose the fixed K=1 coarse-score residual by rotation and translation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _load_recovar,
    _map_relion_table,
    _relion_prior_support,
    _translation_permutation,
)
from scripts.validate_relion_coarse_pass1_capture import validate_directory

ROTATION_DOMINANCE_FRACTION = 0.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompose_additive_score_residual(values: np.ndarray) -> dict[str, Any]:
    """Return the orthogonal row, column, and interaction energy split."""

    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 2, "score residual must be a matrix")
    _require(values.shape[0] > 1 and values.shape[1] > 1, "score matrix is too small")
    _require(np.all(np.isfinite(values)), "score residual must be finite")

    centered = values - np.mean(values)
    row_effect = np.mean(centered, axis=1, keepdims=True)
    translation_effect = np.mean(centered, axis=0, keepdims=True)
    interaction = centered - row_effect - translation_effect
    row_component = np.broadcast_to(row_effect, centered.shape)
    translation_component = np.broadcast_to(translation_effect, centered.shape)

    energies = {
        "rotation_only": float(np.sum(row_component**2)),
        "translation_only": float(np.sum(translation_component**2)),
        "interaction": float(np.sum(interaction**2)),
    }
    total = float(np.sum(centered**2))
    _require(total > 0, "score residual has zero centered energy")
    component_sum = sum(energies.values())
    _require(
        np.isclose(component_sum, total, rtol=1e-10, atol=1e-18),
        "additive component energies are not orthogonal",
    )
    return {
        "centered_total_energy": total,
        "energy": energies,
        "energy_fraction": {name: value / total for name, value in energies.items()},
        "interaction_p95_abs": float(np.percentile(np.abs(interaction), 95)),
        "interaction_max_abs": float(np.max(np.abs(interaction))),
    }


def _recovar_by_original_index(directory: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for path in sorted(Path(directory).glob("*.npz")):
        artifact = _load_recovar(path)
        original_index = artifact["original_index"]
        _require(original_index not in result, f"duplicate RECOVAR index {original_index}")
        result[original_index] = artifact
    return result


def build_report(
    *,
    cohort_json: Path,
    relion_directory: Path,
    recovar_directory: Path,
) -> dict[str, Any]:
    cohort = json.loads(Path(cohort_json).read_text())
    _require(cohort["selected_particle_count"] == 14, "cohort denominator must be 14")
    _require(cohort["mismatch_particle_count"] == 10, "mismatch denominator must be 10")
    _require(cohort["control_particle_count"] == 4, "control denominator must be 4")

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
    relion_by_stack = {
        artifact.stack_index: artifact for artifact in relion_artifacts
    }
    recovar_by_index = _recovar_by_original_index(recovar_directory)
    _require(len(recovar_by_index) == 14, "RECOVAR denominator must be 14")

    particles = []
    for cohort_row in cohort["rows"]:
        relion = relion_by_stack[cohort_row["stack_index_one_based"]]
        recovar = recovar_by_index[cohort_row["original_index_zero_based"]]
        n_directions, n_psi, n_translations = relion.header[10:13]
        translation_permutation, translation_mapping = _translation_permutation(
            relion.translations,
            recovar["translations"],
        )
        relion_raw = _map_relion_table(
            relion.raw_diff2,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        common_support = _map_relion_table(
            _relion_prior_support(relion.raw_diff2),
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=translation_permutation,
        )
        common_support &= np.isfinite(recovar["scores_with"])
        fully_active = np.all(common_support, axis=1)
        fully_inactive = np.all(~common_support, axis=1)
        _require(
            np.all(fully_active | fully_inactive),
            "finite-prior support must contain complete translation rows",
        )
        residual = (
            np.asarray(recovar["scores_pre"][fully_active], dtype=np.float64)
            + np.asarray(relion_raw[fully_active], dtype=np.float64)
        )
        decomposition = decompose_additive_score_residual(residual)
        rotation_fraction = decomposition["energy_fraction"]["rotation_only"]
        particles.append(
            {
                "group": cohort_row["group"],
                "stack_index_one_based": cohort_row["stack_index_one_based"],
                "original_index_zero_based": cohort_row["original_index_zero_based"],
                "relion_part_id": cohort_row["relion_part_id"],
                "active_rotation_rows": int(np.count_nonzero(fully_active)),
                "n_translations": int(n_translations),
                "rotation_dominated": bool(
                    rotation_fraction > ROTATION_DOMINANCE_FRACTION
                ),
                "decomposition": decomposition,
                "translation_mapping": translation_mapping,
                "artifact_paths": {
                    "relion": str(relion.path.resolve()),
                    "recovar": recovar["path"],
                },
                "artifact_sha256": {
                    "relion": relion.sha256,
                    "recovar": _sha256(Path(recovar["path"])),
                },
            }
        )

    mismatch_particles = [row for row in particles if row["group"] == "mismatch"]
    control_particles = [row for row in particles if row["group"] == "control"]
    fixed_metric = {
        "all_particles": {
            "rotation_dominated": sum(row["rotation_dominated"] for row in particles),
            "denominator": 14,
        },
        "mismatch_particles": {
            "rotation_dominated": sum(
                row["rotation_dominated"] for row in mismatch_particles
            ),
            "denominator": 10,
        },
        "controls": {
            "rotation_dominated": sum(
                row["rotation_dominated"] for row in control_particles
            ),
            "denominator": 4,
        },
    }
    rotation_fractions = [
        row["decomposition"]["energy_fraction"]["rotation_only"] for row in particles
    ]
    translation_fractions = [
        row["decomposition"]["energy_fraction"]["translation_only"]
        for row in particles
    ]
    interaction_fractions = [
        row["decomposition"]["energy_fraction"]["interaction"] for row in particles
    ]
    if fixed_metric["all_particles"]["rotation_dominated"] == 14:
        classification = (
            "raw_coarse_score_residual_is_translation_independent_rotation_dominated"
        )
    else:
        classification = "raw_coarse_score_residual_has_mixed_component_dominance"

    return {
        "schema": "recovar-k1-case22-coarse-score-components-v1",
        "status": "complete",
        "classification": classification,
        "interpretation_limit": (
            "rotation-only dominance is consistent with a projection-norm or other "
            "translation-independent rotation term, but does not distinguish those "
            "operands without a component capture"
        ),
        "metric_policy": (
            "fixed 10 mismatch particles and 4 controls; additive two-way centered "
            "energy decomposition on exact common finite-prior translation rows; "
            "rotation dominance requires energy fraction >0.5; no correlation"
        ),
        "rotation_dominance_fraction_strictly_above": ROTATION_DOMINANCE_FRACTION,
        "fixed_metric": fixed_metric,
        "energy_fraction_range": {
            "rotation_only": [float(min(rotation_fractions)), float(max(rotation_fractions))],
            "translation_only": [
                float(min(translation_fractions)),
                float(max(translation_fractions)),
            ],
            "interaction": [
                float(min(interaction_fractions)),
                float(max(interaction_fractions)),
            ],
        },
        "particles": particles,
        "relion_validation": relion_validation,
        "cohort": str(Path(cohort_json).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-json", type=Path, required=True)
    parser.add_argument("--relion-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
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
                "classification": report["classification"],
                "energy_fraction_range": report["energy_fraction_range"],
                "fixed_metric": report["fixed_metric"],
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
