#!/usr/bin/env python3
"""Compare matched native and live RECOVAR VDAM posterior repeat widths."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import (
    _match_rotations,
    _positive_rotation_mask,
)
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
        np.all(np.isfinite(active_scores)),
        f"candidate active scores contain nonfinite values: {path}",
    )
    return {
        "path": path.resolve(),
        "original_index": int(identities[0]),
        "rotations": rotations[active],
        "posterior": active_posterior,
        "centered_scores": active_scores - np.max(active_scores),
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


def _aligned_particle_panel(
    native_values: list[dict[str, Any]],
    candidate_values: list[dict[str, Any]],
    *,
    rotation_tolerance: float,
) -> dict[str, Any]:
    canonical_native = native_values[0]
    canonical_probabilities_all = np.asarray(
        canonical_native["probabilities"], dtype=np.float32
    )
    canonical_positive = _positive_rotation_mask(canonical_probabilities_all)
    canonical_rotations = np.asarray(canonical_native["rotations"], dtype=np.float32)[
        canonical_positive
    ]

    native_posteriors = []
    native_centered_scores = []
    native_positive_score_masks = []
    for native in native_values:
        rotation_map = _match_rotations(
            canonical_rotations,
            np.asarray(native["rotations"], dtype=np.float32),
            rotation_tolerance,
        )
        probabilities = np.asarray(native["probabilities"], dtype=np.float32)[rotation_map]
        raw_weights = np.asarray(native["raw_weights"], dtype=np.float64)[rotation_map]
        _require(
            probabilities.shape[1] == raw_weights.shape[1],
            "native posterior/score translation topology differs",
        )
        positive = raw_weights > 0.0
        centered = np.full(raw_weights.shape, np.nan, dtype=np.float64)
        centered[positive] = np.log(raw_weights[positive]) - np.max(np.log(raw_weights[positive]))
        native_posteriors.append(probabilities)
        native_centered_scores.append(centered)
        native_positive_score_masks.append(positive)

    candidate_posteriors = []
    candidate_centered_scores = []
    for candidate in candidate_values:
        rotation_map = _match_rotations(
            canonical_rotations,
            np.asarray(candidate["rotations"], dtype=np.float32),
            rotation_tolerance,
        )
        posterior = np.asarray(candidate["posterior"], dtype=np.float32)[rotation_map]
        positive = posterior > np.float32(0.0)
        posterior_norm = float(np.sum(posterior[positive], dtype=np.float64))
        _require(
            np.isfinite(posterior_norm) and posterior_norm > 0.0,
            "candidate posterior normalization is invalid",
        )
        posterior = np.where(positive, posterior / posterior_norm, 0.0).astype(
            np.float32
        )
        centered_scores = np.asarray(candidate["centered_scores"], dtype=np.float64)[
            rotation_map
        ]
        _require(
            posterior.shape == native_posteriors[0].shape
            and centered_scores.shape == posterior.shape,
            "native/candidate posterior topology differs",
        )
        candidate_posteriors.append(posterior)
        candidate_centered_scores.append(centered_scores)

    native_score_mask = np.logical_and.reduce(native_positive_score_masks)
    _require(np.any(native_score_mask), "native repeats share no positive score coordinates")
    return {
        "native_posteriors": native_posteriors,
        "candidate_posteriors": candidate_posteriors,
        "native_scores": [value[native_score_mask] for value in native_centered_scores],
        "candidate_scores": [value[native_score_mask] for value in candidate_centered_scores],
        "native_support_mismatch": int(
            np.count_nonzero(
                np.logical_or.reduce([value > 0.0 for value in native_posteriors])
                != np.logical_and.reduce([value > 0.0 for value in native_posteriors])
            )
        ),
        "candidate_support_mismatch": int(
            np.count_nonzero(
                np.logical_or.reduce([value > 0.0 for value in candidate_posteriors])
                != np.logical_and.reduce([value > 0.0 for value in candidate_posteriors])
            )
        ),
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
