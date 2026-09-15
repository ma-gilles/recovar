#!/usr/bin/env python3
"""Compare complete RECOVAR VDAM posteriors with a native RELION ensemble."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_posterior_repeat_panel import (
    _centered_positive_logs,
    _load_capture,
    _native_raw_weights,
    _particle_locations,
)
from scripts.analyze_vdam_storewavg_boundary import _load_native, _match_rotations, _require
from scripts.analyze_vdam_storewavg_panel import (
    _native_prefixes,
    _part_to_original_indices,
    _quantiles,
)

SCHEMA = "recovar.vdam_posterior_ensemble.v1"


def _ensemble_metrics(
    native_vectors: np.ndarray,
    candidate_vector: np.ndarray,
    *,
    mode: str,
) -> dict[str, object]:
    native = np.asarray(native_vectors, dtype=np.float64)
    candidate = np.asarray(candidate_vector, dtype=np.float64).reshape(-1)
    if native.ndim != 2 or native.shape[0] < 2 or native.shape[1:] != candidate.shape:
        raise ValueError("ensemble vectors must have shape (repeat, coordinate)")
    if mode == "relative_l2":
        scale = max(float(np.linalg.norm(native[0])), np.finfo(np.float64).tiny)
    elif mode == "rms":
        scale = float(np.sqrt(native.shape[1]))
    else:
        raise ValueError(f"unknown ensemble metric mode: {mode}")
    native_pair = np.asarray(
        [np.linalg.norm(native[left] - native[right]) / scale for left, right in combinations(range(native.shape[0]), 2)],
        dtype=np.float64,
    )
    candidate_distance = np.linalg.norm(native - candidate[None, :], axis=1) / scale
    native_max = float(np.max(native_pair))
    candidate_nearest = float(np.min(candidate_distance))
    if native_max == 0.0:
        nearest_over_native_max = 0.0 if candidate_nearest == 0.0 else float("inf")
    else:
        nearest_over_native_max = candidate_nearest / native_max
    lower = np.min(native, axis=0)
    upper = np.max(native, axis=0)
    outside = np.maximum(lower - candidate, candidate - upper)
    outside = np.maximum(outside, 0.0)
    return {
        "mode": mode,
        "coordinate_count": int(candidate.size),
        "native_pair": _quantiles(native_pair.tolist()),
        "candidate_to_native": _quantiles(candidate_distance.tolist()),
        "candidate_nearest_over_native_pair_max": nearest_over_native_max,
        "candidate_within_native_pair_max": bool(candidate_nearest <= native_max),
        "candidate_inside_coordinate_envelope_fraction": float(np.mean(outside == 0.0)),
        "candidate_rms_distance_outside_coordinate_envelope": float(
            np.sqrt(np.mean(outside * outside))
        ),
        "candidate_max_distance_outside_coordinate_envelope": float(np.max(outside)),
    }


def _ratio_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    _require(array.size > 0 and finite.size > 0, "ensemble ratios contain no finite values")
    return {
        "finite": _quantiles(finite.tolist()),
        "infinite_count": int(np.count_nonzero(np.isinf(array))),
    }


def analyze(
    native_directories: list[Path],
    relion_data_star: Path,
    half_capture_paths: dict[int, list[Path]],
    *,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    _require(len(native_directories) >= 2, "at least two native repeats are required")
    prefix_sets = []
    incomplete_by_repeat = []
    for directory in native_directories:
        prefixes, incomplete = _native_prefixes(directory)
        prefix_sets.append(prefixes)
        incomplete_by_repeat.append(incomplete)
    _require(not any(incomplete_by_repeat), "a native ensemble capture is incomplete")
    _require(
        all(set(prefixes) == set(prefix_sets[0]) for prefixes in prefix_sets[1:]),
        "native ensemble particle IDs differ",
    )
    original_by_part_all = _part_to_original_indices(relion_data_star)
    original_by_part = {
        part_id: original_by_part_all[part_id] for part_id in sorted(prefix_sets[0])
    }
    captures_by_half = {
        half: [_load_capture(path, half) for path in paths]
        for half, paths in sorted(half_capture_paths.items())
    }
    _require(set(captures_by_half) == {1, 2}, "both RECOVAR halves are required")
    locations = _particle_locations(captures_by_half)
    _require(set(locations) == set(original_by_part.values()), "particle identities do not close")

    half_rows = {
        half: {
            "posterior_native": [[] for _ in native_directories],
            "posterior_candidate": [],
            "score_native": [[] for _ in native_directories],
            "score_candidate": [],
            "posterior_ratios": [],
            "score_ratios": [],
            "posterior_within_count": 0,
            "score_within_count": 0,
            "raw_support_mismatch_count": 0,
            "per_particle": [],
        }
        for half in (1, 2)
    }

    for part_id, original_index in original_by_part.items():
        half, capture_index, slot = locations[original_index]
        capture = captures_by_half[half][capture_index]
        row_mask = capture.active_particle_rows == slot
        _require(np.any(row_mask), f"candidate capture lacks rows for particle {original_index}")
        native_payloads = [
            _load_native(directory, prefixes[part_id], load_projector=False)
            for directory, prefixes in zip(native_directories, prefix_sets, strict=True)
        ]
        reference_rotations = np.asarray(native_payloads[0]["rotations"], dtype=np.float32)
        native_rotation_maps = [
            np.arange(reference_rotations.shape[0], dtype=np.int64)
        ] + [
            _match_rotations(
                reference_rotations,
                np.asarray(payload["rotations"], dtype=np.float32),
                rotation_tolerance,
            )
            for payload in native_payloads[1:]
        ]
        candidate_rotation_map = _match_rotations(
            reference_rotations,
            capture.active_rotations[row_mask],
            rotation_tolerance,
        )
        active_rotation_rows = capture.active_rotation_rows[row_mask][candidate_rotation_map]
        native_posteriors = np.asarray(
            [
                np.asarray(payload["probabilities"], dtype=np.float32)[mapping]
                for payload, mapping in zip(native_payloads, native_rotation_maps, strict=True)
            ],
            dtype=np.float32,
        )
        candidate_posterior = np.asarray(
            capture.reconstruction_probs[slot], dtype=np.float32
        )[active_rotation_rows]
        _require(
            native_posteriors.shape[1:] == candidate_posterior.shape,
            f"posterior topology differs for part {part_id}",
        )
        posterior_metric = _ensemble_metrics(
            native_posteriors.reshape(len(native_directories), -1),
            candidate_posterior.reshape(-1),
            mode="relative_l2",
        )

        native_raw = [
            _native_raw_weights(directory, prefixes[part_id])[mapping]
            for directory, prefixes, mapping in zip(
                native_directories, prefix_sets, native_rotation_maps, strict=True
            )
        ]
        candidate_raw = np.asarray(
            capture.candidate_raw_exp_weights_f32[slot], dtype=np.float32
        )[active_rotation_rows]
        supports = [np.asarray(values) > 0.0 for values in native_raw] + [candidate_raw > 0.0]
        support_union = np.logical_or.reduce(supports)
        support_intersection = np.logical_and.reduce(supports)
        support_mismatch_count = int(np.count_nonzero(support_union != support_intersection))
        _require(np.any(support_intersection), f"raw score support is empty for part {part_id}")

        def centered_common(values: np.ndarray) -> np.ndarray:
            support, logs = _centered_positive_logs(values)
            full = np.zeros(np.asarray(values).shape, dtype=np.float64)
            full[support] = logs - float(np.mean(logs))
            return full[support_intersection]

        native_scores = np.asarray([centered_common(values) for values in native_raw])
        candidate_score = centered_common(candidate_raw)
        score_metric = _ensemble_metrics(native_scores, candidate_score, mode="rms")

        summary = half_rows[half]
        for repeat_index, values in enumerate(native_posteriors):
            summary["posterior_native"][repeat_index].append(values.reshape(-1))
        summary["posterior_candidate"].append(candidate_posterior.reshape(-1))
        for repeat_index, values in enumerate(native_scores):
            summary["score_native"][repeat_index].append(values.reshape(-1))
        summary["score_candidate"].append(candidate_score.reshape(-1))
        summary["posterior_ratios"].append(
            posterior_metric["candidate_nearest_over_native_pair_max"]
        )
        summary["score_ratios"].append(score_metric["candidate_nearest_over_native_pair_max"])
        summary["posterior_within_count"] += int(
            posterior_metric["candidate_within_native_pair_max"]
        )
        summary["score_within_count"] += int(score_metric["candidate_within_native_pair_max"])
        summary["raw_support_mismatch_count"] += support_mismatch_count
        summary["per_particle"].append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "posterior": posterior_metric,
                "centered_log_weight": score_metric,
                "raw_support_mismatch_count": support_mismatch_count,
            }
        )

    halves = {}
    for half, rows in half_rows.items():
        posterior_native = np.asarray(
            [np.concatenate(chunks) for chunks in rows["posterior_native"]]
        )
        posterior_candidate = np.concatenate(rows["posterior_candidate"])
        score_native = np.asarray([np.concatenate(chunks) for chunks in rows["score_native"]])
        score_candidate = np.concatenate(rows["score_candidate"])
        halves[str(half)] = {
            "particle_count": len(rows["per_particle"]),
            "raw_support_mismatch_count": int(rows["raw_support_mismatch_count"]),
            "posterior_pooled": _ensemble_metrics(
                posterior_native, posterior_candidate, mode="relative_l2"
            ),
            "centered_log_weight_pooled": _ensemble_metrics(
                score_native, score_candidate, mode="rms"
            ),
            "posterior_candidate_within_native_pair_max_count": int(
                rows["posterior_within_count"]
            ),
            "centered_log_weight_candidate_within_native_pair_max_count": int(
                rows["score_within_count"]
            ),
            "posterior_nearest_over_native_pair_max": _ratio_summary(
                rows["posterior_ratios"]
            ),
            "centered_log_weight_nearest_over_native_pair_max": _ratio_summary(
                rows["score_ratios"]
            ),
            "per_particle": rows["per_particle"],
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "native_repeat_count": len(native_directories),
        "coverage": {
            "native_particle_count": len(prefix_sets[0]),
            "recovar_particle_count": len(locations),
            "relion_data_star_particle_count": len(original_by_part_all),
        },
        "halves": halves,
        "artifacts": {
            "native_directories": [str(path.resolve()) for path in native_directories],
            "relion_data_star": str(relion_data_star.resolve()),
            "recovar_half_captures": {
                str(half): [str(path.resolve()) for path in paths]
                for half, paths in half_capture_paths.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", action="append", required=True, type=Path)
    parser.add_argument("--relion-data-star", required=True, type=Path)
    parser.add_argument("--recovar-half1-capture", action="append", required=True, type=Path)
    parser.add_argument("--recovar-half2-capture", action="append", required=True, type=Path)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.relion_data_star,
        {1: args.recovar_half1_capture, 2: args.recovar_half2_capture},
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "native_repeat_count": report["native_repeat_count"],
                "coverage": report["coverage"],
                "halves": {
                    half: {key: value for key, value in rows.items() if key != "per_particle"}
                    for half, rows in report["halves"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
