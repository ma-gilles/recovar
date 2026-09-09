#!/usr/bin/env python3
"""Compare matched native and live RECOVAR VDAM posterior repeat widths."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_vdam_storewavg_panel import _quantiles
from scripts.audit_vdam_native_operand_replay_panel import (
    _normalized_weights,
    _pairwise_width,
)
from scripts.audit_vdam_repeat_panel import RepeatPanelError, _load_json
from scripts.run_vdam_worker_private_host_replay import _load_native_panels

SCHEMA = "recovar.vdam_live_posterior_repeat_panel.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatPanelError(message)


def _pairwise_rms(values: list[np.ndarray]) -> dict[str, Any]:
    rows = []
    for (left_index, left), (right_index, right) in itertools.combinations(
        enumerate(values, start=1), 2
    ):
        left_array = np.asarray(left, dtype=np.float64).reshape(-1)
        right_array = np.asarray(right, dtype=np.float64).reshape(-1)
        _require(
            left_array.shape == right_array.shape and left_array.size > 0,
            "score repeat topology differs",
        )
        residual = right_array - left_array
        rows.append(
            {
                "left_repeat": left_index,
                "right_repeat": right_index,
                "rms": float(np.sqrt(np.mean(residual * residual))),
                "max_absolute_error": float(np.max(np.abs(residual))),
            }
        )
    _require(bool(rows), "at least two repeats are required")
    return {
        "minimum_pairwise_rms": min(row["rms"] for row in rows),
        "maximum_pairwise_rms": max(row["rms"] for row in rows),
        "pairwise": rows,
    }


def _diameter_ratio(candidate: float, native: float) -> float:
    candidate = float(candidate)
    native = float(native)
    _require(candidate >= 0.0 and np.isfinite(candidate), "candidate diameter is invalid")
    _require(native >= 0.0 and np.isfinite(native), "native diameter is invalid")
    if native == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return candidate / native


def _load_candidate_capture(path: Path, *, iteration: int) -> dict[str, Any]:
    required = {
        "selected_global_image_indices",
        "local_rotation_matrices",
        "translations",
        "posterior",
        "reconstruction_rotation_mask",
        "pass2_scores_total",
        "debug_iteration",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        _require(not missing, f"candidate posterior dump lacks fields {missing}: {path}")
        values = {name: np.asarray(archive[name]) for name in required}
    identities = np.asarray(values["selected_global_image_indices"], dtype=np.int64)
    _require(identities.shape == (1,), f"candidate dump must contain one identity: {path}")
    observed_iteration = np.asarray(values["debug_iteration"], dtype=np.int32).reshape(-1)
    _require(
        observed_iteration.shape == (1,) and int(observed_iteration[0]) == int(iteration),
        f"candidate dump iteration differs: {path}",
    )
    rotations = np.asarray(values["local_rotation_matrices"], dtype=np.float32)
    posterior = np.asarray(values["posterior"], dtype=np.float32)
    scores = np.asarray(values["pass2_scores_total"], dtype=np.float32)
    rotation_mask = np.asarray(values["reconstruction_rotation_mask"], dtype=bool)
    _require(
        posterior.ndim == 3
        and posterior.shape[0] == 1
        and scores.shape == posterior.shape
        and rotation_mask.shape == posterior.shape[:2]
        and rotations.shape == (posterior.shape[1], 3, 3),
        f"candidate posterior topology differs: {path}",
    )
    active = rotation_mask[0] & np.any(posterior[0] > np.float32(0.0), axis=1)
    _require(np.any(active), f"candidate posterior has no active rotations: {path}")
    active_posterior = posterior[0, active]
    active_scores = scores[0, active]
    _require(
        not np.any(np.isnan(active_scores)) and not np.any(np.isposinf(active_scores)),
        f"candidate active scores contain NaN or positive infinity: {path}",
    )
    _require(
        np.all(np.isfinite(active_scores[active_posterior > np.float32(0.0)])),
        f"candidate positive posterior has a nonfinite score: {path}",
    )
    finite_scores = np.isfinite(active_scores)
    _require(np.any(finite_scores), f"candidate active scores are all masked: {path}")
    centered_scores = np.full(active_scores.shape, -np.inf, dtype=np.float32)
    centered_scores[finite_scores] = (
        active_scores[finite_scores] - np.max(active_scores[finite_scores])
    )
    return {
        "path": path.resolve(),
        "original_index": int(identities[0]),
        "rotations": rotations[active],
        "posterior": active_posterior,
        "centered_scores": centered_scores,
        "translation_count": int(np.asarray(values["translations"]).shape[0]),
    }


def _load_candidate_repeat(directory: Path, *, iteration: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"local_fused_posterior_it{iteration:03d}_image_*.npz")):
        capture = _load_candidate_capture(path, iteration=iteration)
        identity = int(capture["original_index"])
        _require(identity not in result, f"duplicate candidate identity {identity}: {directory}")
        result[identity] = capture
    _require(bool(result), f"candidate posterior repeat is empty: {directory}")
    return result


def _load_native_repeat(root: Path, *, iteration: int) -> dict[int, dict[str, Any]]:
    data_star = root / "relion" / f"run_it{iteration:03d}_data.star"
    panels = _load_native_panels(
        root / "native_panels", data_star, iteration=iteration
    )
    result = {}
    for original_index, panel in panels.items():
        orientation_count = int(panel["orientation_count"])
        result[int(original_index)] = {
            "probabilities": _normalized_weights(panel),
            "rotations": np.asarray(panel["eulers"], dtype=np.float32)
            .reshape(orientation_count, 3, 3)
            .transpose(0, 2, 1),
            "raw_weights": np.asarray(panel["weights"], dtype=np.float64),
            "part_id": int(panel["part_id"]),
            "path": str(panel["path"]),
        }
    _require(bool(result), f"native repeat is empty: {root}")
    return result


def _rotation_union_maps(
    rotations_by_repeat: list[np.ndarray],
    *,
    rotation_tolerance: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    union: list[np.ndarray] = []
    exact_lookup: dict[bytes, int] = {}
    mappings: list[np.ndarray] = []
    for repeat_index, rotations in enumerate(rotations_by_repeat):
        rotations = np.asarray(rotations, dtype=np.float32)
        _require(
            rotations.ndim == 3 and rotations.shape[1:] == (3, 3),
            "repeat rotation topology differs",
        )
        mapping = []
        for rotation in rotations:
            exact_key = np.ascontiguousarray(rotation).tobytes()
            nearest = exact_lookup.get(exact_key, -1)
            if nearest < 0 and union and repeat_index > 0:
                errors = np.max(
                    np.abs(np.asarray(union, dtype=np.float32) - rotation), axis=(1, 2)
                )
                nearest = int(np.argmin(errors))
            else:
                errors = np.asarray([], dtype=np.float32)
            if nearest < 0 or (
                errors.size > 0 and float(errors[nearest]) > rotation_tolerance
            ):
                union.append(rotation.copy())
                nearest = len(union) - 1
            exact_lookup[exact_key] = nearest
            mapping.append(nearest)
        mapping_array = np.asarray(mapping, dtype=np.int64)
        _require(
            np.unique(mapping_array).size == mapping_array.size,
            "repeat contains duplicate rotations within tolerance",
        )
        mappings.append(mapping_array)
    _require(bool(union), "repeat family has no active rotations")
    return np.asarray(union, dtype=np.float32), mappings


def _align_repeat_family(
    values: list[dict[str, Any]],
    *,
    posterior_key: str,
    score_key: str,
    scores_are_weights: bool,
    rotation_tolerance: float,
) -> dict[str, Any]:
    active_values = []
    translation_count = None
    for value in values:
        posterior = np.asarray(value[posterior_key], dtype=np.float32)
        scores_or_weights = np.asarray(value[score_key])
        rotations = np.asarray(value["rotations"], dtype=np.float32)
        _require(
            posterior.ndim == 2
            and scores_or_weights.shape == posterior.shape
            and rotations.shape == (posterior.shape[0], 3, 3),
            "repeat posterior/score topology differs",
        )
        if translation_count is None:
            translation_count = posterior.shape[1]
        _require(
            posterior.shape[1] == translation_count,
            "repeat translation topology differs",
        )
        if scores_are_weights:
            finite_score = scores_or_weights > 0.0
            centered_scores = np.full(posterior.shape, -np.inf, dtype=np.float64)
            centered_scores[finite_score] = np.log(scores_or_weights[finite_score])
            centered_scores[finite_score] -= np.max(centered_scores[finite_score])
        else:
            centered_scores = np.asarray(scores_or_weights, dtype=np.float64)
            finite_score = np.isfinite(centered_scores)
        active_rotation = np.logical_or(
            np.any(posterior > np.float32(0.0), axis=1),
            np.any(finite_score, axis=1),
        )
        _require(np.any(active_rotation), "repeat has no active posterior rotations")
        active_values.append(
            {
                "rotations": rotations[active_rotation],
                "posterior": posterior[active_rotation],
                "centered_scores": centered_scores[active_rotation],
            }
        )

    union, mappings = _rotation_union_maps(
        [value["rotations"] for value in active_values],
        rotation_tolerance=rotation_tolerance,
    )
    aligned_posteriors = []
    aligned_scores = []
    for value, mapping in zip(active_values, mappings, strict=True):
        posterior = np.zeros((union.shape[0], int(translation_count)), dtype=np.float32)
        posterior[mapping] = value["posterior"]
        normalization = float(np.sum(posterior, dtype=np.float64))
        _require(
            np.isfinite(normalization) and normalization > 0.0,
            "repeat posterior normalization is invalid",
        )
        posterior /= np.float32(normalization)
        scores = np.full((union.shape[0], int(translation_count)), -np.inf)
        scores[mapping] = value["centered_scores"]
        aligned_posteriors.append(posterior)
        aligned_scores.append(scores)

    shared_score_mask = np.logical_and.reduce(
        [np.isfinite(value) for value in aligned_scores]
    )
    _require(
        np.any(shared_score_mask),
        "repeat family shares no finite score coordinates",
    )
    positive_supports = [value > np.float32(0.0) for value in aligned_posteriors]
    return {
        "posteriors": aligned_posteriors,
        "scores": [value[shared_score_mask] for value in aligned_scores],
        "support_mismatch": int(
            np.count_nonzero(
                np.logical_or.reduce(positive_supports)
                != np.logical_and.reduce(positive_supports)
            )
        ),
    }


def _aligned_particle_panel(
    native_values: list[dict[str, Any]],
    candidate_values: list[dict[str, Any]],
    *,
    rotation_tolerance: float,
) -> dict[str, Any]:
    native = _align_repeat_family(
        native_values,
        posterior_key="probabilities",
        score_key="raw_weights",
        scores_are_weights=True,
        rotation_tolerance=rotation_tolerance,
    )
    candidate = _align_repeat_family(
        candidate_values,
        posterior_key="posterior",
        score_key="centered_scores",
        scores_are_weights=False,
        rotation_tolerance=rotation_tolerance,
    )

    return {
        "native_posteriors": native["posteriors"],
        "candidate_posteriors": candidate["posteriors"],
        "native_scores": native["scores"],
        "candidate_scores": candidate["scores"],
        "native_support_mismatch": native["support_mismatch"],
        "candidate_support_mismatch": candidate["support_mismatch"],
    }


def audit_live_posterior_repeat_panel(
    panel_root: Path,
    *,
    repeat_count: int = 4,
    iteration: int = 1,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    panel_root = panel_root.resolve()
    _require(repeat_count >= 2, "at least two repeats are required")
    submission = _load_json(
        panel_root / "submission_provenance.json", label="submission provenance"
    )
    _require(
        submission.get("execution_policy")
        == "one allocation, one physical H100, sequential native and live candidate arms",
        "matched posterior panel execution policy differs",
    )
    native_repeats = [
        _load_native_repeat(panel_root / f"repeat-{repeat:02d}", iteration=iteration)
        for repeat in range(1, repeat_count + 1)
    ]
    candidate_repeats = [
        _load_candidate_repeat(
            panel_root / f"candidate-repeat-{repeat:02d}" / "posterior",
            iteration=iteration,
        )
        for repeat in range(1, repeat_count + 1)
    ]
    identities = set(native_repeats[0])
    _require(bool(identities), "matched posterior panel has no particles")
    _require(
        all(set(value) == identities for value in native_repeats + candidate_repeats),
        "matched posterior repeat identities differ",
    )

    pooled_native_posteriors = [[] for _ in range(repeat_count)]
    pooled_candidate_posteriors = [[] for _ in range(repeat_count)]
    pooled_native_scores = [[] for _ in range(repeat_count)]
    pooled_candidate_scores = [[] for _ in range(repeat_count)]
    native_particle_widths = []
    candidate_particle_widths = []
    native_support_mismatch = 0
    candidate_support_mismatch = 0
    for identity in sorted(identities):
        aligned = _aligned_particle_panel(
            [value[identity] for value in native_repeats],
            [value[identity] for value in candidate_repeats],
            rotation_tolerance=rotation_tolerance,
        )
        native_posteriors = aligned["native_posteriors"]
        candidate_posteriors = aligned["candidate_posteriors"]
        native_particle_widths.append(
            _pairwise_width(native_posteriors)["maximum_pairwise_relative_l2"]
        )
        candidate_particle_widths.append(
            _pairwise_width(candidate_posteriors)["maximum_pairwise_relative_l2"]
        )
        native_support_mismatch += int(aligned["native_support_mismatch"])
        candidate_support_mismatch += int(aligned["candidate_support_mismatch"])
        for repeat in range(repeat_count):
            pooled_native_posteriors[repeat].append(native_posteriors[repeat].reshape(-1))
            pooled_candidate_posteriors[repeat].append(candidate_posteriors[repeat].reshape(-1))
            pooled_native_scores[repeat].append(aligned["native_scores"][repeat].reshape(-1))
            pooled_candidate_scores[repeat].append(
                aligned["candidate_scores"][repeat].reshape(-1)
            )

    native_posterior_metrics = _pairwise_width(
        [np.concatenate(value) for value in pooled_native_posteriors]
    )
    candidate_posterior_metrics = _pairwise_width(
        [np.concatenate(value) for value in pooled_candidate_posteriors]
    )
    native_score_metrics = _pairwise_rms(
        [np.concatenate(value) for value in pooled_native_scores]
    )
    candidate_score_metrics = _pairwise_rms(
        [np.concatenate(value) for value in pooled_candidate_scores]
    )
    return {
        "schema": SCHEMA,
        "result": "complete",
        "scope": "matched live iteration-1 posterior and centered-score repeat width",
        "iteration": int(iteration),
        "repeat_count": int(repeat_count),
        "particle_count": len(identities),
        "physical_gpu_uuid": submission.get("physical_gpu_uuid"),
        "science_head": submission.get("science_head"),
        "posterior": {
            "native": native_posterior_metrics,
            "candidate": candidate_posterior_metrics,
            "candidate_over_native_maximum_diameter": _diameter_ratio(
                candidate_posterior_metrics["maximum_pairwise_relative_l2"],
                native_posterior_metrics["maximum_pairwise_relative_l2"],
            ),
            "native_particle_maximum_pairwise_relative_l2": _quantiles(
                native_particle_widths
            ),
            "candidate_particle_maximum_pairwise_relative_l2": _quantiles(
                candidate_particle_widths
            ),
            "native_support_mismatch_coordinate_count": native_support_mismatch,
            "candidate_support_mismatch_coordinate_count": candidate_support_mismatch,
        },
        "centered_score": {
            "native": native_score_metrics,
            "candidate": candidate_score_metrics,
            "candidate_over_native_maximum_diameter": _diameter_ratio(
                candidate_score_metrics["maximum_pairwise_rms"],
                native_score_metrics["maximum_pairwise_rms"],
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_live_posterior_repeat_panel(
        args.panel_root,
        repeat_count=args.repeat_count,
        iteration=args.iteration,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
