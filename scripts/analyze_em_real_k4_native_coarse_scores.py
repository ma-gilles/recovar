#!/usr/bin/env python3
"""Localize K=4 coarse-support drift with native RELION score surfaces.

This analyzer joins a bounded, passive RELION coarse-score capture to the
corresponding passive RECOVAR significance dumps.  It compares likelihood and
prior components separately, then swaps the likelihood and prior surfaces.
The score-swap support sets provide a direct causal discriminator: they show
whether the remaining parent-order difference comes from the likelihood or
from the priors without changing either engine's production arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.analyze_em_real_k4_coarse_score_support import (
    _stable_top_count_mask,
    _support_metric,
)
from scripts.validate_relion_coarse_score_capture import (
    SIGNIFICANT,
    CoarseScoreCapture,
    load_coarse_score_capture,
)

SCHEMA = "recovar.em_real_k4_native_coarse_scores.v1"
SUPPORT_SCHEMA = "recovar.em_real_k4_coarse_score_support.v1"


class AnalysisError(RuntimeError):
    """Raised when the joined score evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "quantile input is invalid")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _centered_comparison(recovar: np.ndarray, relion: np.ndarray) -> dict[str, Any]:
    """Compare two surfaces after removing one per-particle additive offset."""

    left = np.asarray(recovar, dtype=np.float64).reshape(-1)
    right = np.asarray(relion, dtype=np.float64).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "score surface shapes differ")
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "score surface is non-finite")
    offset = float(np.mean(left - right))
    residual = left - right - offset
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = max(
        float(np.linalg.norm(left_centered)),
        float(np.linalg.norm(right_centered)),
    )
    residual_norm = float(np.linalg.norm(residual))

    left_order = np.argsort(left, kind="stable")
    right_order = np.argsort(right, kind="stable")
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[left_order] = np.arange(left.size, dtype=np.float64)
    right_rank[right_order] = np.arange(right.size, dtype=np.float64)
    if denominator == 0:
        centered_relative_l2 = 0.0
        rank_correlation = 1.0
    else:
        centered_relative_l2 = residual_norm / denominator
        rank_correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
    _require(math.isfinite(rank_correlation), "rank correlation is non-finite")
    return {
        "value_count": int(left.size),
        "recovar_minus_relion_offset": offset,
        "centered_relative_l2": centered_relative_l2,
        "centered_rms": float(np.sqrt(np.mean(residual * residual))),
        "centered_max_abs": float(np.max(np.abs(residual))),
        "descending_rank_correlation": rank_correlation,
        "winner_exact": bool(np.argmax(left) == np.argmax(right)),
    }


def _native_surface(capture: CoarseScoreCapture) -> dict[str, np.ndarray]:
    """Convert RELION class/direction/psi/translation order to RECOVAR order."""

    class_min, n_classes, n_directions, n_psi, n_translations = capture.header[9:14]
    _require(class_min == 0, "native capture does not start at class zero")
    candidates = capture.candidates
    native_shape = (n_classes, n_directions, n_psi, n_translations)
    recovar_shape = (n_classes, n_psi * n_directions, n_translations)

    def convert(field: str, dtype: np.dtype[Any] | type) -> np.ndarray:
        values = np.asarray(candidates[field], dtype=dtype).reshape(native_shape)
        return values.transpose(0, 2, 1, 3).reshape(recovar_shape)

    raw_score = -convert("raw_diff2", np.float64)
    orientation_prior = convert("orientation_log_prior", np.float64)
    translation_prior = convert("translation_log_prior", np.float64)
    return {
        "raw_score": raw_score,
        "orientation_prior": orientation_prior,
        "translation_prior": translation_prior,
        "total_prior": orientation_prior + translation_prior,
        "combined_score": convert("combined_preexponent", np.float64),
        "significant": (convert("flags", np.uint32) & SIGNIFICANT) != 0,
    }


def _raw_difference_axis_decomposition(
    recovar_raw_score: np.ndarray,
    native_raw_score: np.ndarray,
) -> dict[str, float]:
    """Describe which score axes carry the centered raw-score difference.

    A class/rotation-only contribution is consistent with a projection-norm
    boundary, while the within-class/rotation remainder must vary with the
    translation and therefore also contains cross-term differences.  This is
    a localization diagnostic, not a proof of a particular kernel cause.
    """

    left = np.asarray(recovar_raw_score, dtype=np.float64)
    right = np.asarray(native_raw_score, dtype=np.float64)
    _require(left.shape == right.shape and left.ndim == 3, "raw score topology differs")
    difference = left - right
    difference -= float(np.mean(difference))
    total_squared = float(np.sum(difference * difference))
    _require(total_squared > 0, "raw score difference is constant")

    class_rotation_mean = np.mean(difference, axis=2, keepdims=True)
    within_class_rotation = difference - class_rotation_mean
    class_rotation_squared = float(
        np.sum(class_rotation_mean * class_rotation_mean) * difference.shape[2]
    )
    translation_mean = np.mean(difference, axis=(0, 1), keepdims=True)
    translation_squared = float(
        np.sum(translation_mean * translation_mean)
        * difference.shape[0]
        * difference.shape[1]
    )
    return {
        "centered_rms": float(np.sqrt(np.mean(difference * difference))),
        "class_rotation_mean_fraction_of_centered_squared_difference": (
            class_rotation_squared / total_squared
        ),
        "within_class_rotation_fraction_of_centered_squared_difference": float(
            np.sum(within_class_rotation * within_class_rotation) / total_squared
        ),
        "within_class_rotation_rms": float(
            np.sqrt(np.mean(within_class_rotation * within_class_rotation))
        ),
        "within_class_rotation_max_abs": float(np.max(np.abs(within_class_rotation))),
        "translation_main_effect_fraction_of_centered_squared_difference": (
            translation_squared / total_squared
        ),
    }


def _load_recovar_dump(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "debug_iteration",
            "current_size",
            "n_classes",
            "n_rot",
            "n_trans",
            "scores_pre_prior_per_class",
            "scores_with_prior_per_class",
            "significant_mask",
            "class_log_priors",
            "rotation_log_prior",
            "translation_log_prior",
            "score_capture_mode",
        }
        _require(required <= set(payload.files), f"RECOVAR dump lacks {sorted(required - set(payload.files))}")
        n_classes = int(np.asarray(payload["n_classes"]).item())
        n_rotations = int(np.asarray(payload["n_rot"]).item())
        n_translations = int(np.asarray(payload["n_trans"]).item())
        shape = (n_classes, n_rotations, n_translations)
        raw_score = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        combined_score = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64)
        significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(shape)
        _require(raw_score.shape == combined_score.shape == shape, "RECOVAR score topology differs")
        class_prior = np.asarray(payload["class_log_priors"], dtype=np.float64)
        rotation_prior = np.asarray(payload["rotation_log_prior"], dtype=np.float64)
        translation_prior_vector = np.asarray(payload["translation_log_prior"], dtype=np.float64)
        _require(class_prior.shape == (n_classes,), "RECOVAR class-prior topology differs")
        _require(rotation_prior.shape == (n_classes, n_rotations), "RECOVAR rotation-prior topology differs")
        _require(translation_prior_vector.shape == (n_translations,), "RECOVAR translation-prior topology differs")
        orientation_prior = np.broadcast_to(
            class_prior[:, None, None] + rotation_prior[:, :, None],
            shape,
        ).copy()
        translation_prior = np.broadcast_to(
            translation_prior_vector[None, None, :],
            shape,
        ).copy()
        return {
            "path": path.resolve(),
            "sha256": _sha256(path),
            "dataset_index": int(np.asarray(payload["original_index"]).item()),
            "debug_iteration": int(np.asarray(payload["debug_iteration"]).item()),
            "current_size": int(np.asarray(payload["current_size"]).item()),
            "n_classes": n_classes,
            "n_rotations": n_rotations,
            "n_translations": n_translations,
            "raw_score": raw_score,
            "combined_score": combined_score,
            # This is the exact prior contribution used by the captured
            # production score, including its floating-point addition order.
            "total_prior": combined_score - raw_score,
            "orientation_prior": orientation_prior,
            "translation_prior": translation_prior,
            "significant": significant,
            "score_capture_mode": str(np.asarray(payload["score_capture_mode"]).item()),
        }


def _support_counterfactuals(
    *,
    native: dict[str, np.ndarray],
    recovar: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = native["significant"]
    count = int(np.count_nonzero(reference))
    surfaces = {
        "native_combined": native["combined_score"],
        "recovar_combined": recovar["combined_score"],
        "recovar_raw_native_prior": recovar["raw_score"] + native["total_prior"],
        "native_raw_recovar_prior": native["raw_score"] + recovar["total_prior"],
        "native_raw_without_prior": native["raw_score"],
        "recovar_raw_without_prior": recovar["raw_score"],
    }
    report: dict[str, Any] = {}
    selected_sets: dict[str, np.ndarray] = {}
    for name, surface in surfaces.items():
        selected, cutoff = _stable_top_count_mask(surface, count)
        selected_sets[name] = selected
        report[name] = {**_support_metric(reference, selected), **cutoff}
    report["recovar_observed_support"] = _support_metric(reference, recovar["significant"])
    report["set_identities"] = {
        "recovar_raw_native_prior_equals_recovar_combined": bool(
            np.array_equal(
                selected_sets["recovar_raw_native_prior"],
                selected_sets["recovar_combined"],
            )
        ),
        "native_raw_recovar_prior_equals_native_combined": bool(
            np.array_equal(
                selected_sets["native_raw_recovar_prior"],
                selected_sets["native_combined"],
            )
        ),
    }
    return report, selected_sets


def _validate_previous_support(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in (
        "exact",
        "intersection",
        "union",
        "native_only",
        "recovar_only",
        "native_count",
        "recovar_count",
    ):
        _require(observed[key] == expected[key], f"direct native support changed prior result: {key}")
    _require(
        math.isclose(observed["jaccard"], expected["jaccard"], rel_tol=0.0, abs_tol=1e-15),
        "direct native support changed prior Jaccard",
    )


def _aggregate_support(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    metrics = [record["support_counterfactuals"][field] for record in records]
    intersection = sum(int(metric["intersection"]) for metric in metrics)
    union = sum(int(metric["union"]) for metric in metrics)
    return {
        "records": len(metrics),
        "exact_records": sum(bool(metric["exact"]) for metric in metrics),
        "intersection": intersection,
        "union": union,
        "jaccard": 1.0 if union == 0 else intersection / float(union),
        "native_count": sum(int(metric["native_count"]) for metric in metrics),
        "candidate_count": sum(int(metric["recovar_count"]) for metric in metrics),
        "boundary_tie_records": sum(bool(metric.get("boundary_tie", False)) for metric in metrics),
    }


def _aggregate_comparison(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    metrics = [record["component_comparison"][field] for record in records]
    total = sum(int(metric["value_count"]) for metric in metrics)
    pooled_rms = math.sqrt(
        sum(metric["centered_rms"] ** 2 * metric["value_count"] for metric in metrics)
        / float(total)
    )
    return {
        "records": len(metrics),
        "value_count": total,
        "winner_exact_records": sum(bool(metric["winner_exact"]) for metric in metrics),
        "pooled_centered_rms": pooled_rms,
        "maximum_centered_abs": max(float(metric["centered_max_abs"]) for metric in metrics),
        "centered_relative_l2": _quantiles(
            float(metric["centered_relative_l2"]) for metric in metrics
        ),
        "descending_rank_correlation": _quantiles(
            float(metric["descending_rank_correlation"]) for metric in metrics
        ),
        "recovar_minus_relion_offset": _quantiles(
            float(metric["recovar_minus_relion_offset"]) for metric in metrics
        ),
    }


def _aggregate_axis_decomposition(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["raw_difference_axis_decomposition"] for record in records]
    return {
        "records": len(metrics),
        "class_rotation_mean_fraction_of_centered_squared_difference": _quantiles(
            metric["class_rotation_mean_fraction_of_centered_squared_difference"]
            for metric in metrics
        ),
        "within_class_rotation_fraction_of_centered_squared_difference": _quantiles(
            metric["within_class_rotation_fraction_of_centered_squared_difference"]
            for metric in metrics
        ),
        "within_class_rotation_rms": _quantiles(
            metric["within_class_rotation_rms"] for metric in metrics
        ),
        "within_class_rotation_max_abs": max(
            metric["within_class_rotation_max_abs"] for metric in metrics
        ),
        "translation_main_effect_fraction_of_centered_squared_difference": _quantiles(
            metric[
                "translation_main_effect_fraction_of_centered_squared_difference"
            ]
            for metric in metrics
        ),
    }


def _classification(records: list[dict[str, Any]]) -> str:
    native_self_exact = all(
        record["support_counterfactuals"]["native_combined"]["exact"] for record in records
    )
    native_raw_recovar_prior_exact = all(
        record["support_counterfactuals"]["native_raw_recovar_prior"]["exact"]
        for record in records
    )
    recovar_raw_native_prior_unchanged = all(
        record["support_counterfactuals"]["set_identities"][
            "recovar_raw_native_prior_equals_recovar_combined"
        ]
        for record in records
    )
    if native_self_exact and native_raw_recovar_prior_exact and recovar_raw_native_prior_unchanged:
        return "raw_likelihood_surface_is_first_material_support_difference"
    if native_self_exact:
        return "likelihood_and_prior_contributions_both_affect_support"
    return "native_capture_does_not_reproduce_its_own_support"


def build_report(
    *,
    capture_dir: Path,
    significance_dir: Path,
    support_report_path: Path,
) -> dict[str, Any]:
    support_report = json.loads(support_report_path.read_text())
    _require(support_report.get("schema") == SUPPORT_SCHEMA, "support report schema differs")
    expected_particles = {
        int(record["native_particle_id_zero_based"]): record
        for record in support_report["particles"]
    }
    dataset_by_particle = {
        int(key): int(value)
        for key, value in support_report["inputs"]["dataset_index_by_native_particle_id"].items()
    }
    _require(set(expected_particles) == set(dataset_by_particle), "support report join differs")

    capture_paths = sorted(capture_dir.glob("*.coarse-score-v1.bin"))
    captures = [load_coarse_score_capture(path) for path in capture_paths]
    capture_by_particle = {capture.header[5]: capture for capture in captures}
    _require(len(capture_by_particle) == len(captures), "duplicate native particle capture")
    _require(set(capture_by_particle) == set(expected_particles), "native capture panel differs")

    dump_paths = sorted(significance_dir.glob("significance_*.npz"))
    dumps = [_load_recovar_dump(path) for path in dump_paths]
    dump_by_dataset = {dump["dataset_index"]: dump for dump in dumps}
    _require(len(dump_by_dataset) == len(dumps), "duplicate RECOVAR dataset-index dump")
    _require(set(dump_by_dataset) == set(dataset_by_particle.values()), "RECOVAR dump panel differs")

    records = []
    for particle_id in sorted(expected_particles):
        expected = expected_particles[particle_id]
        capture = capture_by_particle[particle_id]
        recovar = dump_by_dataset[dataset_by_particle[particle_id]]
        native = _native_surface(capture)
        shape = native["combined_score"].shape
        _require(
            shape
            == (
                recovar["n_classes"],
                recovar["n_rotations"],
                recovar["n_translations"],
            ),
            "native/RECOVAR score topology differs",
        )
        _require(capture.header[6] == expected["stack_index_one_based"], "native stack join differs")
        _require(recovar["debug_iteration"] == -1 and recovar["current_size"] == 20, "RECOVAR boundary differs")
        _require(
            recovar["score_capture_mode"] == "passive_cached_after_support",
            "RECOVAR score dump is not passive",
        )
        support_counterfactuals, _ = _support_counterfactuals(native=native, recovar=recovar)
        _validate_previous_support(
            support_counterfactuals["recovar_observed_support"],
            expected["global_across_classes"]["observed"],
        )
        component_comparison = {
            "raw_likelihood_score": _centered_comparison(
                recovar["raw_score"], native["raw_score"]
            ),
            "orientation_prior": _centered_comparison(
                recovar["orientation_prior"], native["orientation_prior"]
            ),
            "translation_prior": _centered_comparison(
                recovar["translation_prior"], native["translation_prior"]
            ),
            "total_prior": _centered_comparison(
                recovar["total_prior"], native["total_prior"]
            ),
            "combined_score": _centered_comparison(
                recovar["combined_score"], native["combined_score"]
            ),
        }
        records.append(
            {
                "native_particle_id_zero_based": particle_id,
                "dataset_index_zero_based": recovar["dataset_index"],
                "stack_index_one_based": capture.header[6],
                "stratum": expected["stratum"],
                "candidate_count": int(np.prod(shape)),
                "native_capture": {
                    "path": str(capture.path.resolve()),
                    "sha256": capture.sha256,
                },
                "recovar_dump": {
                    "path": str(recovar["path"]),
                    "sha256": recovar["sha256"],
                },
                "component_comparison": component_comparison,
                "raw_difference_axis_decomposition": _raw_difference_axis_decomposition(
                    recovar["raw_score"], native["raw_score"]
                ),
                "support_counterfactuals": support_counterfactuals,
            }
        )

    support_fields = (
        "native_combined",
        "recovar_combined",
        "recovar_raw_native_prior",
        "native_raw_recovar_prior",
        "native_raw_without_prior",
        "recovar_raw_without_prior",
    )
    comparison_fields = (
        "raw_likelihood_score",
        "orientation_prior",
        "translation_prior",
        "total_prior",
        "combined_score",
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": (
            "16 frozen EMPIAR-10076 shared-200 particles at K=4 iteration 1/coarse "
            "current-size 20; all 4 classes, 576 rotations, and 29 translations"
        ),
        "metric_policy": (
            "exact support sets and score-swap counterfactuals are causal metrics; component "
            "scores remove only one per-particle additive constant before L2/rank diagnostics"
        ),
        "inputs": {
            "capture_directory": str(capture_dir.resolve()),
            "significance_directory": str(significance_dir.resolve()),
            "support_report": str(support_report_path.resolve()),
            "support_report_sha256": _sha256(support_report_path),
        },
        "summary": {
            "classification": _classification(records),
            "particle_records": len(records),
            "candidate_values": sum(record["candidate_count"] for record in records),
            "support_counterfactuals": {
                field: _aggregate_support(records, field) for field in support_fields
            },
            "component_comparison": {
                field: _aggregate_comparison(records, field) for field in comparison_fields
            },
            "raw_difference_axis_decomposition": _aggregate_axis_decomposition(records),
        },
        "particles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--significance-dir", required=True, type=Path)
    parser.add_argument("--support-report", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        capture_dir=args.capture_dir,
        significance_dir=args.significance_dir,
        support_report_path=args.support_report,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
