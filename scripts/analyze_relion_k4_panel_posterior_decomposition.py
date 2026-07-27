#!/usr/bin/env python3
"""Decompose fixed-panel K=4 posterior residuals into numerator and normalizer terms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .analyze_relion_k4_panel_threeway import (
        _float32_from_bits,
        _rotation_map,
        _translation_map,
    )
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from analyze_relion_k4_panel_threeway import (  # type: ignore[no-redef]
        _float32_from_bits,
        _rotation_map,
        _translation_map,
    )
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )

PANEL_SCHEMA = "recovar.k4_iter10_class2_residual_target_panel.v1"
THREEWAY_SCHEMA = "recovar.k4_iter10_panel12_threeway_fine_score.v2"
COHORT_SCHEMA = "recovar.k4_iter10_panel12_cohort_calibration.v1"
REPEATABILITY_SCHEMA = "relion.k4_iter10_panel12_capture_repeatability.v1"
REPORT_SCHEMA = "recovar.k4_iter10_panel12_posterior_decomposition.v1"
EXPECTED_COHORT_COUNTS = {
    "corrected_by_relion_cuda": 4,
    "introduced_by_relion_cuda": 4,
    "persistent_class_mismatch": 4,
}
BACKENDS = {
    "host_numpy": "host_numpy",
    "relion_cuda": "jax_gpu",
}
EXP50_F32 = float(np.exp(np.float32(50.0), dtype=np.float32))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _energy(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return math.fsum(float(value) * float(value) for value in array)


def _residual(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    lhs = np.asarray(reference, dtype=np.float64)
    rhs = np.asarray(candidate, dtype=np.float64)
    _require(lhs.shape == rhs.shape, "posterior residual shape changed")
    delta = rhs - lhs
    energy = _energy(delta)
    return {
        "candidate_count": int(delta.size),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "residual_energy": energy,
        "residual_l2": math.sqrt(energy),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _counterfactual_summary(
    relion_raw_weight: np.ndarray,
    relion_weight_normalizer: float,
    candidate_probability: np.ndarray,
    candidate_pmax: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray | float]]:
    """Return exact exp(50)-frame numerator/normalizer counterfactuals."""

    relion_raw = np.asarray(relion_raw_weight, dtype=np.float64)
    candidate_prob = np.asarray(candidate_probability, dtype=np.float64)
    _require(relion_raw.shape == candidate_prob.shape, "posterior candidate shape changed")
    _require(
        np.all(np.isfinite(relion_raw))
        and np.all(relion_raw >= 0)
        and math.isfinite(relion_weight_normalizer)
        and relion_weight_normalizer > 0,
        "RELION posterior operands are invalid",
    )
    _require(
        np.all(np.isfinite(candidate_prob))
        and np.all(candidate_prob >= 0)
        and math.isfinite(candidate_pmax)
        and 0 < candidate_pmax <= 1,
        "RECOVAR posterior operands are invalid",
    )

    candidate_weight_normalizer = EXP50_F32 / candidate_pmax
    candidate_raw = candidate_prob * candidate_weight_normalizer
    relion_probability = relion_raw / relion_weight_normalizer
    reconstructed_candidate = candidate_raw / candidate_weight_normalizer
    _require(
        np.allclose(
            reconstructed_candidate,
            candidate_prob,
            rtol=8 * np.finfo(np.float64).eps,
            atol=0,
        ),
        "RECOVAR exp(50)-frame reconstruction does not close",
    )

    relion_numerator = relion_raw / candidate_weight_normalizer
    relion_normalizer = candidate_raw / relion_weight_normalizer
    production_metric = _residual(relion_probability, candidate_prob)
    numerator_metric = _residual(relion_probability, relion_numerator)
    normalizer_metric = _residual(relion_probability, relion_normalizer)
    production_energy = float(production_metric["residual_energy"])
    return (
        {
            "production_posterior_residual": production_metric,
            "relion_numerator_counterfactual_residual": numerator_metric,
            "relion_normalizer_counterfactual_residual": normalizer_metric,
            "replace_numerator_residual_energy_removed_fraction": (
                1.0 - float(numerator_metric["residual_energy"]) / production_energy
                if production_energy > 0
                else 0.0
            ),
            "replace_normalizer_residual_energy_removed_fraction": (
                1.0 - float(normalizer_metric["residual_energy"]) / production_energy
                if production_energy > 0
                else 0.0
            ),
            "exp50_frame_weight_normalizer": candidate_weight_normalizer,
            "weight_normalizer_relative_error": (
                candidate_weight_normalizer / relion_weight_normalizer - 1.0
            ),
            "class2_probability_mass": math.fsum(float(value) for value in candidate_prob),
            "inferred_exp50_frame_reconstruction_max_abs": float(
                np.max(np.abs(reconstructed_candidate - candidate_prob), initial=0.0)
            ),
        },
        {
            "relion_probability": relion_probability,
            "production": candidate_prob,
            "relion_numerator": relion_numerator,
            "relion_normalizer": relion_normalizer,
            "weight_normalizer_relative_error": (
                candidate_weight_normalizer / relion_weight_normalizer - 1.0
            ),
        },
    )


def _metric_from_deltas(deltas: list[np.ndarray]) -> dict[str, float | int | bool]:
    values = np.concatenate([np.asarray(delta, dtype=np.float64).reshape(-1) for delta in deltas])
    zeros = np.zeros_like(values)
    return _residual(zeros, values)


def _cohort_backend_summary(
    rows: list[dict[str, Any]],
    *,
    backend: str,
) -> dict[str, Any]:
    production = _metric_from_deltas(
        [row["_arrays"][backend]["production"] - row["_arrays"][backend]["relion_probability"] for row in rows]
    )
    relion_numerator = _metric_from_deltas(
        [
            row["_arrays"][backend]["relion_numerator"]
            - row["_arrays"][backend]["relion_probability"]
            for row in rows
        ]
    )
    relion_normalizer = _metric_from_deltas(
        [
            row["_arrays"][backend]["relion_normalizer"]
            - row["_arrays"][backend]["relion_probability"]
            for row in rows
        ]
    )
    production_energy = float(production["residual_energy"])
    normalizer_errors = [
        float(row["_arrays"][backend]["weight_normalizer_relative_error"]) for row in rows
    ]
    normalizer_energy = math.fsum(value * value for value in normalizer_errors)
    numerator_removed = (
        1.0 - float(relion_numerator["residual_energy"]) / production_energy
        if production_energy > 0
        else 0.0
    )
    normalizer_removed = (
        1.0 - float(relion_normalizer["residual_energy"]) / production_energy
        if production_energy > 0
        else 0.0
    )
    if numerator_removed < 0 or normalizer_removed < 0:
        component_classification = "numerator_normalizer_components_counteract"
    elif numerator_removed > normalizer_removed:
        component_classification = "numerator_substitution_removes_more_residual_energy"
    elif normalizer_removed > numerator_removed:
        component_classification = "normalizer_substitution_removes_more_residual_energy"
    else:
        component_classification = "numerator_normalizer_substitutions_remove_equal_energy"
    return {
        "production_posterior_residual": production,
        "relion_numerator_counterfactual_residual": relion_numerator,
        "relion_normalizer_counterfactual_residual": relion_normalizer,
        "replace_numerator_residual_energy_removed_fraction": numerator_removed,
        "replace_normalizer_residual_energy_removed_fraction": normalizer_removed,
        "weight_normalizer_relative_l2": math.sqrt(normalizer_energy),
        "weight_normalizer_relative_max_abs": max(map(abs, normalizer_errors), default=0.0),
        "component_classification": component_classification,
    }


def _cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capture_floor = _metric_from_deltas(
        [row["_arrays"]["capture_b"] - row["_arrays"]["capture_a"] for row in rows]
    )
    backends = {
        backend: _cohort_backend_summary(rows, backend=backend) for backend in BACKENDS
    }
    host_energy = float(
        backends["host_numpy"]["production_posterior_residual"]["residual_energy"]
    )
    relion_cuda_energy = float(
        backends["relion_cuda"]["production_posterior_residual"]["residual_energy"]
    )
    floor_energy = float(capture_floor["residual_energy"])
    improvement = host_energy - relion_cuda_energy
    if improvement > floor_energy:
        classification = (
            "relion_cuda_reduces_posterior_residual_beyond_capture_repeatability_floor"
        )
    elif improvement > 0:
        classification = (
            "relion_cuda_posterior_reduction_is_within_capture_repeatability_floor"
        )
    else:
        classification = "relion_cuda_does_not_reduce_posterior_residual"
    return {
        "target_count": len(rows),
        "candidate_count": int(capture_floor["candidate_count"]),
        "classification": classification,
        "capture_repeatability_posterior_residual": capture_floor,
        "backends": backends,
        "relion_cuda_improvement": {
            "improvement_energy": improvement,
            "capture_repeatability_residual_energy": floor_energy,
            "improvement_positive": improvement > 0,
            "improvement_exceeds_capture_repeatability_floor": improvement > floor_energy,
            "improvement_to_repeatability_energy_ratio": (
                improvement / floor_energy if floor_energy > 0 else None
            ),
        },
    }


def _validate_reports(
    panel: dict[str, Any],
    threeway: dict[str, Any],
    cohort: dict[str, Any],
    repeatability: dict[str, Any],
    *,
    panel_json: Path,
    threeway_json: Path,
    repeatability_json: Path,
    capture_a_directory: Path,
    capture_b_directory: Path,
) -> list[dict[str, Any]]:
    _require(panel.get("schema") == PANEL_SCHEMA and panel.get("status") == "complete", "panel changed")
    _require(
        threeway.get("schema") == THREEWAY_SCHEMA
        and threeway.get("status") == "complete"
        and threeway.get("scorecard_change_admissible") is False,
        "three-way report changed or is incomplete",
    )
    _require(
        cohort.get("schema") == COHORT_SCHEMA
        and cohort.get("status") == "complete"
        and cohort.get("scorecard_change_admissible") is False,
        "cohort report changed or is incomplete",
    )
    _require(
        repeatability.get("schema") == REPEATABILITY_SCHEMA
        and repeatability.get("status") == "complete"
        and repeatability.get("scorecard_change_admissible") is False,
        "capture-repeatability report changed or is incomplete",
    )
    targets = panel.get("targets")
    _require(isinstance(targets, list) and len(targets) == 12, "panel target count changed")
    _require(
        threeway["inputs"]["panel"]["sha256"] == _sha256(panel_json),
        "three-way report does not bind the panel",
    )
    _require(
        cohort["inputs"]["threeway"]["sha256"] == _sha256(threeway_json),
        "cohort report does not bind the three-way report",
    )
    _require(
        threeway["inputs"]["capture_repeatability"]["sha256"] == _sha256(repeatability_json),
        "three-way report does not bind capture repeatability",
    )
    _require(
        Path(repeatability["inputs"]["capture_a"]).resolve() == capture_a_directory.resolve()
        and Path(repeatability["inputs"]["capture_b"]).resolve() == capture_b_directory.resolve(),
        "capture-repeatability directories changed",
    )
    repeatability_scope = repeatability["scope"]
    _require(
        repeatability_scope["physical_iteration"] == 10
        and repeatability_scope["class_one_based"] == 2
        and repeatability_scope["target_count"] == 12
        and repeatability_scope["geometry_exact_all"]
        and repeatability_scope["winners_exact_all"],
        "capture repeatability scope or topology changed",
    )
    cohort_counts = {
        name: int(value["target_count"]) for name, value in cohort["cohorts"].items()
    }
    _require(cohort_counts == EXPECTED_COHORT_COUNTS, "fixed cohort counts changed")
    return targets


def analyze(
    *,
    preprocess_root: Path,
    capture_a_directory: Path,
    capture_b_directory: Path,
    panel_json: Path,
    threeway_json: Path,
    cohort_json: Path,
    repeatability_json: Path,
) -> dict[str, Any]:
    panel = json.loads(panel_json.read_text())
    threeway = json.loads(threeway_json.read_text())
    cohort = json.loads(cohort_json.read_text())
    repeatability = json.loads(repeatability_json.read_text())
    targets = _validate_reports(
        panel,
        threeway,
        cohort,
        repeatability,
        panel_json=panel_json,
        threeway_json=threeway_json,
        repeatability_json=repeatability_json,
        capture_a_directory=capture_a_directory,
        capture_b_directory=capture_b_directory,
    )

    expected_stacks = {int(target["zero_based_identity_row"]) + 1 for target in targets}

    def load_captures(directory: Path) -> tuple[dict[int, Any], dict[int, Any]]:
        factors = {
            capture.stack_index: capture
            for capture in map(load_factor_capture, directory.glob("*.bpre-v2.bin"))
        }
        scores = {
            capture.stack_index: capture
            for capture in map(
                load_fine_score_capture,
                directory.glob("*.fine-score-v1.bin"),
            )
        }
        _require(set(factors) == expected_stacks, f"{directory}: factor target set changed")
        _require(set(scores) == expected_stacks, f"{directory}: score target set changed")
        return factors, scores

    factors_a, scores_a = load_captures(capture_a_directory)
    factors_b, scores_b = load_captures(capture_b_directory)
    threeway_by_identity = {
        int(row["zero_based_identity_row"]): row for row in threeway["targets"]
    }
    repeatability_by_identity = {
        int(row["zero_based_identity_row"]): row for row in repeatability["targets"]
    }

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        original_index = int(target["zero_based_identity_row"])
        stack = original_index + 1
        target_cohort = target["cohort"]
        _require(
            original_index in threeway_by_identity
            and original_index in repeatability_by_identity
            and threeway_by_identity[original_index]["cohort"] == target_cohort
            and repeatability_by_identity[original_index]["cohort"] == target_cohort,
            f"target {stack}: report identity or cohort changed",
        )

        factor_a = factors_a[stack]
        factor_b = factors_b[stack]
        score_a = scores_a[stack]
        score_b = scores_b[stack]
        _require(
            factor_a.geometry_only and factor_b.geometry_only,
            f"target {stack}: factor capture is not geometry-only",
        )
        _require(
            factor_a.header[9:11] == factor_b.header[9:11] == (10, 2)
            and score_a.header[4:6] == score_b.header[4:6] == (10, 2),
            f"target {stack}: iteration or class changed",
        )
        active_a = (score_a.candidates["flags"] & ACTIVE) != 0
        active_b = (score_b.candidates["flags"] & ACTIVE) != 0
        candidate_identity_fields = (
            "sparse_index",
            "rotation_id",
            "rotation_local",
            "translation_id",
            "coarse_translation",
            "flags",
        )
        _require(
            np.array_equal(
                score_a.candidates[list(candidate_identity_fields)],
                score_b.candidates[list(candidate_identity_fields)],
            )
            and np.array_equal(active_a, active_b),
            f"target {stack}: capture candidate topology changed",
        )
        selected_a = score_a.candidates[active_a]
        selected_b = score_b.candidates[active_b]
        active_count = int(selected_b.size)
        _require(
            active_count == int(threeway_by_identity[original_index]["active_candidate_count"])
            and active_count
            == int(repeatability_by_identity[original_index]["active_candidate_count"]),
            f"target {stack}: active candidate count changed",
        )
        normalizer_a = float(_float32_from_bits(factor_a.header[26]))
        normalizer_b = float(_float32_from_bits(factor_b.header[26]))
        _require(
            math.isfinite(normalizer_a)
            and normalizer_a > 0
            and math.isfinite(normalizer_b)
            and normalizer_b > 0,
            f"target {stack}: RELION weight normalizer is invalid",
        )
        relion_raw_a = selected_a["post_exponent_weight"].astype(np.float64)
        relion_raw_b = selected_b["post_exponent_weight"].astype(np.float64)
        relion_probability_a = relion_raw_a / normalizer_a
        relion_probability_b = relion_raw_b / normalizer_b
        capture_repeatability = _residual(relion_probability_a, relion_probability_b)

        backend_reports: dict[str, Any] = {}
        backend_arrays: dict[str, Any] = {
            "capture_a": relion_probability_a,
            "capture_b": relion_probability_b,
        }
        backend_artifacts: dict[str, Any] = {}
        rotation_error = 0.0
        translation_error = 0.0
        for backend, directory_name in BACKENDS.items():
            class_values = []
            class_artifacts = []
            for class_one_based in range(1, 5):
                path = (
                    preprocess_root
                    / directory_name
                    / "pass2"
                    / (
                        f"pass2_orig{original_index:06d}_"
                        f"class{class_one_based:03d}_cs074.npz"
                    )
                )
                values = _load_npz(path)
                _require(
                    int(values["original_index"]) == original_index
                    and int(values["class_index"]) == class_one_based - 1
                    and int(values["current_size"]) == 74,
                    f"target {stack}: {backend} class identity changed",
                )
                probabilities = np.asarray(values["probs"], dtype=np.float64)
                _require(
                    probabilities.shape == values["candidate_mask"].shape
                    and np.all(np.isfinite(probabilities))
                    and np.all(probabilities >= 0),
                    f"target {stack}: {backend} probability array is invalid",
                )
                class_values.append((values, probabilities))
                class_artifacts.append({"path": str(path.resolve()), "sha256": _sha256(path)})

            total_probability_mass = math.fsum(
                float(value)
                for _values, probabilities in class_values
                for value in probabilities.reshape(-1)
            )
            _require(
                abs(total_probability_mass - 1.0) <= 1e-12,
                f"target {stack}: {backend} all-class probability mass changed",
            )
            class_pmax = [
                float(np.max(probabilities, initial=0.0))
                for _values, probabilities in class_values
            ]
            candidate_pmax = max(class_pmax)
            predicted_class = int(np.argmax(class_pmax)) + 1

            class2_values, class2_probabilities = class_values[1]
            rotation_map, backend_rotation_error = _rotation_map(
                factor_b.rotations,
                class2_values["rotations"],
            )
            translation_map, backend_translation_error = _translation_map(
                factor_b.translations,
                class2_values["fine_translations"],
            )
            rotation_error = max(rotation_error, backend_rotation_error)
            translation_error = max(translation_error, backend_translation_error)
            relion_rotation = selected_b["rotation_local"].astype(np.int64)
            relion_translation = selected_b["translation_id"].astype(np.int64)
            recovar_rotation = rotation_map[relion_rotation]
            recovar_translation = translation_map[relion_translation]
            mapped_mask = np.zeros_like(class2_values["candidate_mask"], dtype=bool)
            mapped_mask[recovar_rotation, recovar_translation] = True
            _require(
                np.all(class2_values["candidate_mask"][mapped_mask])
                and np.count_nonzero(class2_values["candidate_mask"]) == active_count
                and np.count_nonzero(class2_probabilities[~mapped_mask]) == 0,
                f"target {stack}: {backend} class-2 support changed",
            )
            candidate_probability = class2_probabilities[
                recovar_rotation,
                recovar_translation,
            ]
            backend_report, arrays = _counterfactual_summary(
                relion_raw_b,
                normalizer_b,
                candidate_probability,
                candidate_pmax,
            )
            backend_report.update(
                {
                    "predicted_class_one_based": predicted_class,
                    "global_pmax": candidate_pmax,
                    "all_class_probability_mass": total_probability_mass,
                    "all_class_probability_mass_error": total_probability_mass - 1.0,
                }
            )
            backend_reports[backend] = backend_report
            backend_arrays[backend] = arrays
            backend_artifacts[backend] = class_artifacts

        row = {
            **target,
            "stack_index_one_based": stack,
            "active_candidate_count": active_count,
            "rotation_matrix_map_max_abs": rotation_error,
            "translation_map_max_abs": translation_error,
            "capture_repeatability_posterior_residual": capture_repeatability,
            "relion_capture": {
                "capture_a": {
                    "exp50_frame_weight_normalizer": normalizer_a,
                    "derived_global_pmax": EXP50_F32 / normalizer_a,
                    "class2_probability_mass": math.fsum(
                        float(value) for value in relion_probability_a
                    ),
                },
                "capture_b": {
                    "exp50_frame_weight_normalizer": normalizer_b,
                    "derived_global_pmax": EXP50_F32 / normalizer_b,
                    "class2_probability_mass": math.fsum(
                        float(value) for value in relion_probability_b
                    ),
                },
            },
            "backends": backend_reports,
            "artifacts": {
                "capture_a_bpref": {
                    "path": str(factor_a.path.resolve()),
                    "sha256": factor_a.sha256,
                },
                "capture_a_fine_score": {
                    "path": str(score_a.path.resolve()),
                    "sha256": score_a.sha256,
                },
                "capture_b_bpref": {
                    "path": str(factor_b.path.resolve()),
                    "sha256": factor_b.sha256,
                },
                "capture_b_fine_score": {
                    "path": str(score_b.path.resolve()),
                    "sha256": score_b.sha256,
                },
                "preprocess": backend_artifacts,
            },
            "_arrays": backend_arrays,
        }
        rows.append(row)
        grouped[target_cohort].append(row)

    _require(
        {name: len(values) for name, values in grouped.items()} == EXPECTED_COHORT_COUNTS,
        "analyzed cohort counts changed",
    )
    cohorts = {name: _cohort_summary(grouped[name]) for name in sorted(grouped)}
    cohort_classifications = {name: value["classification"] for name, value in cohorts.items()}
    classification = (
        "heterogeneous_posterior_arithmetic_response"
        if len(set(cohort_classifications.values())) > 1
        else next(iter(cohort_classifications.values()))
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
        "exp50_frame": {
            "expf_50_float32": EXP50_F32,
            "relion_weight_normalizer_source": "geometry-only BPref header field 26",
            "recovar_weight_normalizer_source": "expf(50) / dumped all-class global Pmax",
        },
        "cohorts": cohorts,
        "targets": rows,
        "quality_metric_policy": {
            "map_gate": "shellwise FSC/FSC-AUC only",
            "correlation_computed": False,
            "posterior_metrics_are_diagnostic": True,
        },
        "next_step": (
            "Do not change the preprocessing or posterior-normalization default. "
            "The persistent and introduced cohorts require a numerator score-arithmetic "
            "discriminator; retain the normalizer as a separate persistent-cohort branch."
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
    parser.add_argument("--preprocess-root", required=True, type=Path)
    parser.add_argument("--capture-a-directory", required=True, type=Path)
    parser.add_argument("--capture-b-directory", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--threeway-json", required=True, type=Path)
    parser.add_argument("--cohort-json", required=True, type=Path)
    parser.add_argument("--capture-repeatability-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        preprocess_root=args.preprocess_root,
        capture_a_directory=args.capture_a_directory,
        capture_b_directory=args.capture_b_directory,
        panel_json=args.panel_json,
        threeway_json=args.threeway_json,
        cohort_json=args.cohort_json,
        repeatability_json=args.capture_repeatability_json,
    )
    report["inputs"] = {
        "panel": {"path": str(args.panel_json.resolve()), "sha256": _sha256(args.panel_json)},
        "threeway": {
            "path": str(args.threeway_json.resolve()),
            "sha256": _sha256(args.threeway_json),
        },
        "cohort": {"path": str(args.cohort_json.resolve()), "sha256": _sha256(args.cohort_json)},
        "capture_repeatability": {
            "path": str(args.capture_repeatability_json.resolve()),
            "sha256": _sha256(args.capture_repeatability_json),
        },
        "capture_a_directory": str(args.capture_a_directory.resolve()),
        "capture_b_directory": str(args.capture_b_directory.resolve()),
        "preprocess_root": str(args.preprocess_root.resolve()),
        "analyzer_repo_head": _clean_repo_head(args.repo),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "cohort_classifications": {
                    name: value["classification"] for name, value in report["cohorts"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
