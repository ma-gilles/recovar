#!/usr/bin/env python3
"""Localize fixed-panel K=4 numerator residuals before or after exponentiation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .analyze_relion_k4_panel_posterior_decomposition import (
        BACKENDS,
        EXP50_F32,
        EXPECTED_COHORT_COUNTS,
        _energy,
        _float32_from_bits,
        _load_npz,
        _require,
        _residual,
        _rotation_map,
        _sha256,
        _translation_map,
    )
    from .analyze_relion_k4_panel_posterior_decomposition import (
        REPORT_SCHEMA as POSTERIOR_SCHEMA,
    )
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from analyze_relion_k4_panel_posterior_decomposition import (  # type: ignore[no-redef]
        BACKENDS,
        EXP50_F32,
        EXPECTED_COHORT_COUNTS,
        _energy,
        _float32_from_bits,
        _load_npz,
        _require,
        _residual,
        _rotation_map,
        _sha256,
        _translation_map,
    )
    from analyze_relion_k4_panel_posterior_decomposition import (
        REPORT_SCHEMA as POSTERIOR_SCHEMA,
    )
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )

REPORT_SCHEMA = "recovar.k4_iter10_panel12_numerator_boundary.v1"
SCORE_SUBSTITUTION_MIN_REMOVED_FRACTION = 0.99
COMPONENTS = ("data_score", "orientation_prior", "translation_prior")


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and np.all(np.isfinite(array)), "centering needs finite values")
    mean = math.fsum(float(value) for value in array) / array.size
    return array - mean


def _expf_replay(shifted_log_weight: np.ndarray) -> np.ndarray:
    """Replay RELION's float32 underflow predicate and host ``expf`` frame."""

    shifted = np.asarray(shifted_log_weight, dtype=np.float32)
    _require(np.all(np.isfinite(shifted)), "shifted log weights are not finite")
    return np.where(
        shifted < np.float32(-88.0),
        np.float32(0.0),
        np.exp(shifted, dtype=np.float32),
    ).astype(np.float32, copy=False)


def _metric_from_deltas(deltas: list[np.ndarray]) -> dict[str, float | int | bool]:
    values = np.concatenate(
        [np.asarray(delta, dtype=np.float64).reshape(-1) for delta in deltas]
    )
    return _residual(np.zeros_like(values), values)


def _numerator_summary(
    *,
    relion_raw_weight: np.ndarray,
    relion_shifted_log_weight: np.ndarray,
    candidate_probability: np.ndarray,
    candidate_pmax: float,
    candidate_shifted_log_weight: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    relion_raw = np.asarray(relion_raw_weight, dtype=np.float64).reshape(-1)
    relion_shifted = np.asarray(relion_shifted_log_weight, dtype=np.float32).reshape(-1)
    candidate_prob = np.asarray(candidate_probability, dtype=np.float64).reshape(-1)
    candidate_shifted = np.asarray(
        candidate_shifted_log_weight,
        dtype=np.float32,
    ).reshape(-1)
    _require(
        relion_raw.shape
        == relion_shifted.shape
        == candidate_prob.shape
        == candidate_shifted.shape,
        "numerator candidate shape changed",
    )
    _require(
        np.all(np.isfinite(relion_raw))
        and np.all(relion_raw >= 0)
        and np.all(np.isfinite(candidate_prob))
        and np.all(candidate_prob >= 0)
        and math.isfinite(candidate_pmax)
        and 0 < candidate_pmax <= 1,
        "numerator operands are invalid",
    )

    candidate_raw = candidate_prob * (EXP50_F32 / candidate_pmax)
    relion_score_replay = _expf_replay(relion_shifted).astype(np.float64)
    candidate_score_replay = _expf_replay(candidate_shifted).astype(np.float64)
    production = _residual(relion_raw, candidate_raw)
    candidate_replay = _residual(relion_raw, candidate_score_replay)
    relion_roundtrip = _residual(relion_raw, relion_score_replay)
    candidate_production_vs_replay = _residual(
        candidate_score_replay,
        candidate_raw,
    )
    production_energy = float(production["residual_energy"])
    relion_roundtrip_energy = float(relion_roundtrip["residual_energy"])
    candidate_production_vs_replay_energy = float(
        candidate_production_vs_replay["residual_energy"]
    )
    score_removed = (
        1.0 - relion_roundtrip_energy / production_energy
        if production_energy > 0
        else 0.0
    )
    return (
        {
            "production_raw_numerator_residual": production,
            "candidate_f32_score_replay_raw_numerator_residual": candidate_replay,
            "relion_f32_score_replay_roundtrip_residual": relion_roundtrip,
            "candidate_posterior_inferred_vs_f32_score_replay_residual": (
                candidate_production_vs_replay
            ),
            "shifted_log_weight_residual": _residual(
                relion_shifted,
                candidate_shifted,
            ),
            "replace_score_with_relion_score_residual_energy_removed_fraction": (
                score_removed
            ),
            "replace_score_removes_at_least_99_percent": (
                score_removed >= SCORE_SUBSTITUTION_MIN_REMOVED_FRACTION
            ),
            "candidate_posterior_vs_score_replay_to_production_energy_ratio": (
                candidate_production_vs_replay_energy / production_energy
                if production_energy > 0
                else 0.0
            ),
        },
        {
            "production_raw": candidate_raw,
            "candidate_score_replay_raw": candidate_score_replay,
            "relion_raw": relion_raw,
            "relion_score_replay_raw": relion_score_replay,
            "candidate_shifted": candidate_shifted.astype(np.float64),
            "relion_shifted": relion_shifted.astype(np.float64),
        },
    )


def _component_summary(
    *,
    relion_data_score: np.ndarray,
    relion_orientation_prior: np.ndarray,
    relion_translation_prior: np.ndarray,
    relion_combined_score: np.ndarray,
    candidate_data_score: np.ndarray,
    candidate_orientation_prior: np.ndarray,
    candidate_translation_prior: np.ndarray,
    candidate_combined_score: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    component_deltas = {
        "data_score": _center(candidate_data_score) - _center(relion_data_score),
        "orientation_prior": (
            _center(candidate_orientation_prior) - _center(relion_orientation_prior)
        ),
        "translation_prior": (
            _center(candidate_translation_prior) - _center(relion_translation_prior)
        ),
    }
    observed = _center(candidate_combined_score) - _center(relion_combined_score)
    _require(
        all(delta.shape == observed.shape for delta in component_deltas.values()),
        "centered component shape changed",
    )
    predicted = sum(component_deltas.values())
    closure = observed - predicted
    production = _residual(np.zeros_like(observed), observed)
    production_energy = float(production["residual_energy"])
    substitutions = {}
    for name in COMPONENTS:
        after = observed - component_deltas[name]
        metric = _residual(np.zeros_like(after), after)
        substitutions[name] = {
            "after_relion_component_substitution_residual": metric,
            "residual_energy_removed_fraction": (
                1.0 - float(metric["residual_energy"]) / production_energy
                if production_energy > 0
                else 0.0
            ),
        }
    strongest = max(
        COMPONENTS,
        key=lambda name: float(
            substitutions[name]["residual_energy_removed_fraction"]
        ),
    )
    return (
        {
            "centered_combined_score_residual": production,
            "centered_component_residuals": {
                name: _residual(np.zeros_like(delta), delta)
                for name, delta in component_deltas.items()
            },
            "component_substitutions": substitutions,
            "strongest_single_component": strongest,
            "strongest_residual_energy_removed_fraction": substitutions[strongest][
                "residual_energy_removed_fraction"
            ],
            "component_sum_closure_residual": _residual(
                np.zeros_like(closure),
                closure,
            ),
        },
        {
            "combined": observed,
            **component_deltas,
            "closure": closure,
        },
    )


def _cohort_backend_summary(
    rows: list[dict[str, Any]],
    *,
    backend: str,
) -> dict[str, Any]:
    arrays = [row["_arrays"][backend] for row in rows]
    production = _metric_from_deltas(
        [value["production_raw"] - value["relion_raw"] for value in arrays]
    )
    candidate_replay = _metric_from_deltas(
        [value["candidate_score_replay_raw"] - value["relion_raw"] for value in arrays]
    )
    relion_roundtrip = _metric_from_deltas(
        [value["relion_score_replay_raw"] - value["relion_raw"] for value in arrays]
    )
    candidate_production_vs_replay = _metric_from_deltas(
        [
            value["production_raw"] - value["candidate_score_replay_raw"]
            for value in arrays
        ]
    )
    shifted = _metric_from_deltas(
        [value["candidate_shifted"] - value["relion_shifted"] for value in arrays]
    )
    combined_values = np.concatenate([value["combined"] for value in arrays])
    combined_energy = _energy(combined_values)
    component_substitutions = {}
    for name in COMPONENTS:
        component = np.concatenate([value[name] for value in arrays])
        after = combined_values - component
        metric = _residual(np.zeros_like(after), after)
        component_substitutions[name] = {
            "after_relion_component_substitution_residual": metric,
            "residual_energy_removed_fraction": (
                1.0 - float(metric["residual_energy"]) / combined_energy
                if combined_energy > 0
                else 0.0
            ),
        }
    strongest = max(
        COMPONENTS,
        key=lambda name: float(
            component_substitutions[name]["residual_energy_removed_fraction"]
        ),
    )
    production_energy = float(production["residual_energy"])
    roundtrip_energy = float(relion_roundtrip["residual_energy"])
    score_removed = (
        1.0 - roundtrip_energy / production_energy
        if production_energy > 0
        else 0.0
    )
    posterior_vs_replay_energy = float(
        candidate_production_vs_replay["residual_energy"]
    )
    return {
        "production_raw_numerator_residual": production,
        "candidate_f32_score_replay_raw_numerator_residual": candidate_replay,
        "relion_f32_score_replay_roundtrip_residual": relion_roundtrip,
        "candidate_posterior_inferred_vs_f32_score_replay_residual": (
            candidate_production_vs_replay
        ),
        "shifted_log_weight_residual": shifted,
        "replace_score_with_relion_score_residual_energy_removed_fraction": (
            score_removed
        ),
        "replace_score_removes_at_least_99_percent": (
            score_removed >= SCORE_SUBSTITUTION_MIN_REMOVED_FRACTION
        ),
        "candidate_posterior_vs_score_replay_to_production_energy_ratio": (
            posterior_vs_replay_energy / production_energy
            if production_energy > 0
            else 0.0
        ),
        "centered_combined_score_residual": _residual(
            np.zeros_like(combined_values),
            combined_values,
        ),
        "component_substitutions": component_substitutions,
        "strongest_single_component": strongest,
        "strongest_residual_energy_removed_fraction": component_substitutions[
            strongest
        ]["residual_energy_removed_fraction"],
        "component_sum_closure_residual": _metric_from_deltas(
            [value["closure"] for value in arrays]
        ),
    }


def _cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backends = {
        backend: _cohort_backend_summary(rows, backend=backend)
        for backend in BACKENDS
    }
    localized = all(
        report["replace_score_removes_at_least_99_percent"]
        and report["strongest_single_component"] == "data_score"
        for report in backends.values()
    )
    return {
        "target_count": len(rows),
        "candidate_count": sum(int(row["active_candidate_count"]) for row in rows),
        "classification": (
            "numerator_residual_localized_upstream_of_exponentiation_to_data_score"
            if localized
            else "heterogeneous_numerator_boundary"
        ),
        "backends": backends,
    }


def _validate_posterior_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    _require(
        report.get("schema") == POSTERIOR_SCHEMA
        and report.get("status") == "complete"
        and report.get("scorecard_change_admissible") is False,
        "posterior decomposition report changed or is incomplete",
    )
    scope = report.get("scope")
    _require(
        isinstance(scope, dict)
        and scope.get("physical_iteration") == 10
        and scope.get("class_one_based") == 2
        and scope.get("current_size") == 74
        and scope.get("target_count") == 12
        and scope.get("cohort_counts") == EXPECTED_COHORT_COUNTS,
        "posterior decomposition scope changed",
    )
    targets = report.get("targets")
    _require(isinstance(targets, list) and len(targets) == 12, "target panel changed")
    _require(
        {
            name: sum(row["cohort"] == name for row in targets)
            for name in EXPECTED_COHORT_COUNTS
        }
        == EXPECTED_COHORT_COUNTS,
        "target cohort counts changed",
    )
    return targets


def analyze(*, posterior_json: Path) -> dict[str, Any]:
    posterior = json.loads(posterior_json.read_text())
    targets = _validate_posterior_report(posterior)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        original_index = int(target["zero_based_identity_row"])
        stack = original_index + 1
        cohort = target["cohort"]
        artifacts = target["artifacts"]
        factor_path = Path(artifacts["capture_b_bpref"]["path"])
        score_path = Path(artifacts["capture_b_fine_score"]["path"])
        _require(
            _sha256(factor_path) == artifacts["capture_b_bpref"]["sha256"]
            and _sha256(score_path) == artifacts["capture_b_fine_score"]["sha256"],
            f"target {stack}: RELION artifact hash changed",
        )
        factor = load_factor_capture(factor_path)
        score = load_fine_score_capture(score_path)
        _require(
            factor.stack_index == score.stack_index == stack
            and factor.header[9:11] == (10, 2)
            and score.header[4:6] == (10, 2)
            and factor.geometry_only,
            f"target {stack}: RELION capture identity changed",
        )
        active = (score.candidates["flags"] & ACTIVE) != 0
        selected = score.candidates[active]
        active_count = int(selected.size)
        _require(
            active_count == int(target["active_candidate_count"]),
            f"target {stack}: active candidate count changed",
        )
        relion_raw = selected["post_exponent_weight"].astype(np.float64)
        relion_shifted = selected["shifted_log_weight"].astype(np.float32)
        relion_data = np.subtract(
            _float32_from_bits(score.header[18]),
            selected["raw_diff2"],
            dtype=np.float32,
        )
        relion_orientation = selected["orientation_log_prior"].astype(np.float64)
        relion_translation = selected["translation_log_prior"].astype(np.float64)
        relion_combined = selected["combined_preexponent"].astype(np.float64)

        backend_reports = {}
        backend_arrays = {}
        backend_artifacts = {}
        for backend, _directory_name in BACKENDS.items():
            class_artifacts = artifacts["preprocess"][backend]
            _require(
                isinstance(class_artifacts, list) and len(class_artifacts) == 4,
                f"target {stack}: {backend} class artifact set changed",
            )
            class_values = []
            for class_one_based, artifact in enumerate(class_artifacts, start=1):
                path = Path(artifact["path"])
                _require(
                    _sha256(path) == artifact["sha256"],
                    f"target {stack}: {backend} class {class_one_based} hash changed",
                )
                values = _load_npz(path)
                _require(
                    int(values["original_index"]) == original_index
                    and int(values["class_index"]) == class_one_based - 1
                    and int(values["current_size"]) == 74,
                    f"target {stack}: {backend} class identity changed",
                )
                class_values.append(values)

            class2 = class_values[1]
            rotation_map, rotation_error = _rotation_map(
                factor.rotations,
                class2["rotations"],
            )
            translation_map, translation_error = _translation_map(
                factor.translations,
                class2["fine_translations"],
            )
            relion_rotation = selected["rotation_local"].astype(np.int64)
            relion_translation_id = selected["translation_id"].astype(np.int64)
            candidate_rotation = rotation_map[relion_rotation]
            candidate_translation = translation_map[relion_translation_id]
            mapped_mask = np.zeros_like(class2["candidate_mask"], dtype=bool)
            mapped_mask[candidate_rotation, candidate_translation] = True
            _require(
                np.all(class2["candidate_mask"][mapped_mask])
                and np.count_nonzero(class2["candidate_mask"]) == active_count,
                f"target {stack}: {backend} class-2 support changed",
            )
            all_class_probability_mass = math.fsum(
                float(value)
                for values in class_values
                for value in np.asarray(values["probs"], dtype=np.float64).reshape(-1)
            )
            _require(
                abs(all_class_probability_mass - 1.0) <= 1e-12,
                f"target {stack}: {backend} all-class probability mass changed",
            )
            class_pmax = [
                float(np.max(np.asarray(values["probs"], dtype=np.float64), initial=0.0))
                for values in class_values
            ]
            candidate_pmax = max(class_pmax)
            _require(
                candidate_pmax == float(target["backends"][backend]["global_pmax"]),
                f"target {stack}: {backend} global Pmax changed",
            )
            candidate_probability = np.asarray(class2["probs"], dtype=np.float64)[
                candidate_rotation,
                candidate_translation,
            ]
            all_scores_f32 = [
                np.asarray(values["scores_with_prior"], dtype=np.float32)
                for values in class_values
            ]
            best_f32 = np.float32(
                max(
                    float(
                        np.max(
                            np.where(np.isfinite(values), values, -np.inf),
                            initial=-np.inf,
                        )
                    )
                    for values in all_scores_f32
                )
            )
            _require(np.isfinite(best_f32), f"target {stack}: {backend} has no finite score")
            exponent_shift = np.subtract(
                np.float32(50.0),
                best_f32,
                dtype=np.float32,
            )
            candidate_shifted = np.add(
                all_scores_f32[1][candidate_rotation, candidate_translation],
                exponent_shift,
                dtype=np.float32,
            )
            numerator_report, numerator_arrays = _numerator_summary(
                relion_raw_weight=relion_raw,
                relion_shifted_log_weight=relion_shifted,
                candidate_probability=candidate_probability,
                candidate_pmax=candidate_pmax,
                candidate_shifted_log_weight=candidate_shifted,
            )
            component_report, component_arrays = _component_summary(
                relion_data_score=relion_data,
                relion_orientation_prior=relion_orientation,
                relion_translation_prior=relion_translation,
                relion_combined_score=relion_combined,
                candidate_data_score=class2["scores_pre_prior"][
                    candidate_rotation,
                    candidate_translation,
                ],
                candidate_orientation_prior=class2["rotation_log_prior"][
                    candidate_rotation
                ],
                candidate_translation_prior=class2["translation_log_prior"][
                    candidate_translation
                ],
                candidate_combined_score=class2["scores_with_prior"][
                    candidate_rotation,
                    candidate_translation,
                ],
            )
            backend_reports[backend] = {
                "rotation_matrix_map_max_abs": rotation_error,
                "translation_map_max_abs": translation_error,
                "global_pmax": candidate_pmax,
                "all_class_probability_mass": all_class_probability_mass,
                "numerator_boundary": numerator_report,
                "score_component_attribution": component_report,
            }
            backend_arrays[backend] = {**numerator_arrays, **component_arrays}
            backend_artifacts[backend] = class_artifacts

        row = {
            "zero_based_identity_row": original_index,
            "stack_index_one_based": stack,
            "rlnImageName": target["rlnImageName"],
            "cohort": cohort,
            "active_candidate_count": active_count,
            "backends": backend_reports,
            "artifacts": {
                "relion_bpref": artifacts["capture_b_bpref"],
                "relion_fine_score": artifacts["capture_b_fine_score"],
                "preprocess": backend_artifacts,
            },
            "_arrays": backend_arrays,
        }
        rows.append(row)
        grouped[cohort].append(row)

    _require(
        {name: len(values) for name, values in grouped.items()}
        == EXPECTED_COHORT_COUNTS,
        "analyzed cohort counts changed",
    )
    cohorts = {
        name: _cohort_summary(grouped[name])
        for name in sorted(EXPECTED_COHORT_COUNTS)
    }
    classifications = {value["classification"] for value in cohorts.values()}
    classification = (
        next(iter(classifications))
        if len(classifications) == 1
        else "heterogeneous_numerator_boundary"
    )
    for row in rows:
        del row["_arrays"]
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "current_size": 74,
            "target_count": len(rows),
            "cohort_counts": EXPECTED_COHORT_COUNTS,
            "backends": list(BACKENDS),
        },
        "diagnostic_thresholds": {
            "score_substitution_min_removed_fraction": (
                SCORE_SUBSTITUTION_MIN_REMOVED_FRACTION
            ),
            "threshold_changes_fixed_scorecard": False,
        },
        "cohorts": cohorts,
        "targets": rows,
        "quality_metric_policy": {
            "map_gate": "shellwise FSC/FSC-AUC only",
            "correlation_computed": False,
            "numerator_metrics_are_diagnostic": True,
        },
        "next_step": (
            "Do not change exponentiation, posterior normalization, or prior handling. "
            "Trace the native RELION/RECOVAR data-score residual before prior addition, "
            "starting with the persistent identity 64843 and introduced identity 42824."
        ),
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1"],
        text=True,
    )
    _require(not status, "analyzer repository is dirty")
    return head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--posterior-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(posterior_json=args.posterior_json)
    report["inputs"] = {
        "posterior_decomposition": {
            "path": str(args.posterior_json.resolve()),
            "sha256": _sha256(args.posterior_json),
        },
        "analyzer_repo_head": _clean_repo_head(args.repo),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "cohort_classifications": {
                    name: value["classification"]
                    for name, value in report["cohorts"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
