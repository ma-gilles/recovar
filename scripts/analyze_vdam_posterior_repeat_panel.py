#!/usr/bin/env python3
"""Compare RECOVAR VDAM posteriors with two complete native RELION repeats."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import (
    _flat,
    _load_native,
    _match_rotations,
    _posterior_metric,
    _require,
    _scalar,
)
from scripts.analyze_vdam_storewavg_panel import (
    _native_prefixes,
    _part_to_original_indices,
    _quantiles,
)

SCHEMA = "recovar.vdam_posterior_repeat_panel.v2"


@dataclass(frozen=True)
class _PosteriorCapture:
    path: Path
    half: int
    original_indices: np.ndarray
    reconstruction_probs: np.ndarray
    active_particle_rows: np.ndarray
    active_rotation_rows: np.ndarray
    active_rotations: np.ndarray
    candidate_raw_exp_weights_f32: np.ndarray


def _load_capture(path: Path, expected_half: int) -> _PosteriorCapture:
    required = {
        "half",
        "original_indices",
        "reconstruction_probs",
        "active_particle_rows",
        "active_rotation_rows",
        "active_rotations",
        "candidate_raw_exp_weights_f32",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        _require(not missing, f"posterior capture lacks fields {missing}: {path}")
        values = {name: np.asarray(archive[name]) for name in required}
    half = int(values["half"])
    _require(half == expected_half, f"expected half {expected_half}, got {half}: {path}")
    original_indices = np.asarray(values["original_indices"], dtype=np.int64)
    reconstruction_probs = np.asarray(values["reconstruction_probs"])
    active_particle_rows = np.asarray(values["active_particle_rows"], dtype=np.int32)
    active_rotation_rows = np.asarray(values["active_rotation_rows"], dtype=np.int32)
    active_rotations = np.asarray(values["active_rotations"], dtype=np.float32)
    candidate_raw_exp_weights_f32 = np.asarray(
        values["candidate_raw_exp_weights_f32"], dtype=np.float32
    )
    _require(
        reconstruction_probs.ndim == 3
        and reconstruction_probs.shape[0] == original_indices.size,
        f"posterior topology changed: {path}",
    )
    _require(
        active_particle_rows.shape == active_rotation_rows.shape
        and active_rotations.shape == (active_particle_rows.size, 3, 3),
        f"active rotation topology changed: {path}",
    )
    return _PosteriorCapture(
        path=path.resolve(),
        half=half,
        original_indices=original_indices,
        reconstruction_probs=reconstruction_probs,
        active_particle_rows=active_particle_rows,
        active_rotation_rows=active_rotation_rows,
        active_rotations=active_rotations,
        candidate_raw_exp_weights_f32=candidate_raw_exp_weights_f32,
    )


def _native_raw_weights(directory: Path, prefix: str) -> np.ndarray:
    root = directory / prefix
    orientation_count = int(round(_scalar(Path(f"{root}orientation_num.bin"))))
    translation_count = int(round(_scalar(Path(f"{root}translation_num.bin"))))
    return _flat(Path(f"{root}sorted_weights.bin"), np.dtype("<f8")).reshape(
        orientation_count, translation_count
    )


def _normalize_positive_weights(raw_weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_weights, dtype=np.float64)
    positive = raw > 0.0
    _require(np.any(positive), "raw weight table contains no positive weights")
    total = float(np.sum(raw[positive], dtype=np.float64))
    _require(np.isfinite(total) and total > 0.0, "raw positive weight sum is invalid")
    return np.where(positive, raw / total, 0.0)


def _centered_positive_logs(raw_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_weights, dtype=np.float64)
    positive = raw > 0.0
    _require(np.any(positive), "raw weight table contains no positive weights")
    logs = np.log(raw[positive])
    return positive, logs - np.max(logs)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "cannot compute RMS of an empty array")
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float64)))


def _particle_locations(
    captures_by_half: dict[int, list[_PosteriorCapture]],
) -> dict[int, tuple[int, int, int]]:
    result = {}
    for half, captures in captures_by_half.items():
        for capture_index, capture in enumerate(captures):
            for slot, original_index in enumerate(capture.original_indices):
                identity = int(original_index)
                _require(identity not in result, f"particle {identity} occurs in multiple captures")
                result[identity] = (half, capture_index, slot)
    return result


def _distance_ratio(distance_a: float, distance_b: float, native_repeat_distance: float) -> float:
    nearest = min(float(distance_a), float(distance_b))
    native = float(native_repeat_distance)
    if native == 0.0:
        return 0.0 if nearest == 0.0 else float("inf")
    return nearest / native


def _pooled_relative_l2(reference_norm2: float, residual_norm2: float) -> float:
    return float(np.sqrt(residual_norm2 / max(reference_norm2, np.finfo(np.float64).tiny)))


def _residual_geometry(reference_gap: np.ndarray, candidate_gap: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_gap, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_gap, dtype=np.float64).reshape(-1)
    _require(reference.shape == candidate.shape and reference.size > 0, "gap topology mismatch")
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    denominator = max(reference_norm * candidate_norm, np.finfo(np.float64).tiny)
    projection_denominator = max(reference_norm * reference_norm, np.finfo(np.float64).tiny)
    projection = float(np.vdot(reference, candidate) / projection_denominator)
    orthogonal = candidate - projection * reference
    return {
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "cosine": float(np.vdot(reference, candidate) / denominator),
        "candidate_projection_on_reference": projection,
        "candidate_orthogonal_over_reference": float(
            np.linalg.norm(orthogonal) / max(reference_norm, np.finfo(np.float64).tiny)
        ),
    }


def _ratio_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    _require(array.size > 0 and finite.size > 0, "repeat-distance ratios contain no finite values")
    return {
        "finite": _quantiles(finite.tolist()),
        "infinite_count": int(np.count_nonzero(np.isinf(array))),
    }


def analyze(
    native_a_directory: Path,
    native_b_directory: Path,
    relion_data_star: Path,
    half_capture_paths: dict[int, list[Path]],
    *,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    prefixes_a, incomplete_a = _native_prefixes(native_a_directory)
    prefixes_b, incomplete_b = _native_prefixes(native_b_directory)
    _require(not incomplete_a and not incomplete_b, "native repeat panel is incomplete")
    _require(set(prefixes_a) == set(prefixes_b), "native repeat particle IDs differ")
    all_original_by_part = _part_to_original_indices(relion_data_star)
    _require(set(prefixes_a).issubset(all_original_by_part), "native part ID is absent from STAR")
    original_by_part = {
        part_id: all_original_by_part[part_id] for part_id in sorted(prefixes_a)
    }

    captures_by_half = {
        half: [_load_capture(path, half) for path in paths]
        for half, paths in sorted(half_capture_paths.items())
    }
    _require(set(captures_by_half) == {1, 2}, "both RECOVAR halves are required")
    locations = _particle_locations(captures_by_half)
    _require(set(locations) == set(original_by_part.values()), "particle identities do not close")

    summaries = {
        half: {
            "particle_count": 0,
            "argmax_a_b": 0,
            "argmax_a_candidate": 0,
            "argmax_b_candidate": 0,
            "support_a_b": 0,
            "support_a_candidate": 0,
            "support_b_candidate": 0,
            "reference_norm2": 0.0,
            "native_repeat_norm2": 0.0,
            "candidate_a_norm2": 0.0,
            "candidate_b_norm2": 0.0,
            "raw_reference_norm2": 0.0,
            "raw_native_repeat_norm2": 0.0,
            "raw_candidate_a_norm2": 0.0,
            "raw_candidate_b_norm2": 0.0,
            "native_delta_chunks": [],
            "candidate_delta_chunks": [],
            "log_native_delta_chunks": [],
            "log_candidate_delta_chunks": [],
            "log_a_b_square_sum": 0.0,
            "log_a_candidate_square_sum": 0.0,
            "log_b_candidate_square_sum": 0.0,
            "log_coordinate_count": 0,
            "log_a_b_max": 0.0,
            "log_a_candidate_max": 0.0,
            "log_b_candidate_max": 0.0,
            "raw_support_a_b": 0,
            "raw_support_a_candidate": 0,
            "raw_support_b_candidate": 0,
            "exact_native_raw_candidate_different_count": 0,
            "distance_ratios": [],
            "within_native_repeat_count": 0,
            "coordinate_envelope_fractions": [],
            "per_particle": [],
        }
        for half in (1, 2)
    }

    for part_id, original_index in original_by_part.items():
        half, capture_index, slot = locations[original_index]
        capture = captures_by_half[half][capture_index]
        row_mask = capture.active_particle_rows == slot
        _require(np.any(row_mask), f"candidate capture lacks rows for particle {original_index}")

        native_a = _load_native(native_a_directory, prefixes_a[part_id], load_projector=False)
        native_b = _load_native(native_b_directory, prefixes_b[part_id], load_projector=False)
        rotations_a = np.asarray(native_a["rotations"], dtype=np.float32)
        map_a_to_candidate = _match_rotations(
            rotations_a, capture.active_rotations[row_mask], rotation_tolerance
        )
        map_a_to_b = _match_rotations(
            rotations_a, np.asarray(native_b["rotations"], dtype=np.float32), rotation_tolerance
        )
        probs_a = np.asarray(native_a["probabilities"], dtype=np.float32)
        probs_b = np.asarray(native_b["probabilities"], dtype=np.float32)[map_a_to_b]
        probs_candidate = np.asarray(capture.reconstruction_probs[slot], dtype=np.float32)[
            capture.active_rotation_rows[row_mask]
        ][map_a_to_candidate]
        _require(
            probs_a.shape == probs_b.shape == probs_candidate.shape,
            f"posterior topology differs for part {part_id}",
        )

        metric_a_b = _posterior_metric(probs_a, probs_b)
        metric_a_candidate = _posterior_metric(probs_a, probs_candidate)
        metric_b_candidate = _posterior_metric(probs_b, probs_candidate)
        raw_a = _native_raw_weights(native_a_directory, prefixes_a[part_id])
        raw_b = _native_raw_weights(native_b_directory, prefixes_b[part_id])[map_a_to_b]
        raw_candidate = np.asarray(
            capture.candidate_raw_exp_weights_f32[slot], dtype=np.float32
        )[capture.active_rotation_rows[row_mask]][map_a_to_candidate]
        _require(
            raw_a.shape == raw_b.shape == raw_candidate.shape == probs_a.shape,
            f"raw score-weight topology differs for part {part_id}",
        )
        normalized_raw_a = _normalize_positive_weights(raw_a)
        normalized_raw_b = _normalize_positive_weights(raw_b)
        normalized_raw_candidate = _normalize_positive_weights(raw_candidate)
        raw_metric_a_b = _posterior_metric(normalized_raw_a, normalized_raw_b)
        raw_metric_a_candidate = _posterior_metric(
            normalized_raw_a, normalized_raw_candidate
        )
        raw_metric_b_candidate = _posterior_metric(
            normalized_raw_b, normalized_raw_candidate
        )
        positive_a, log_a = _centered_positive_logs(raw_a)
        positive_b, log_b = _centered_positive_logs(raw_b)
        positive_candidate, log_candidate = _centered_positive_logs(raw_candidate)
        common_positive = positive_a & positive_b & positive_candidate
        log_a_full = np.zeros(raw_a.shape, dtype=np.float64)
        log_b_full = np.zeros(raw_b.shape, dtype=np.float64)
        log_candidate_full = np.zeros(raw_candidate.shape, dtype=np.float64)
        log_a_full[positive_a] = log_a
        log_b_full[positive_b] = log_b
        log_candidate_full[positive_candidate] = log_candidate
        log_a_common = log_a_full[common_positive]
        log_b_common = log_b_full[common_positive]
        log_candidate_common = log_candidate_full[common_positive]
        log_delta_a_b = log_b_common - log_a_common
        log_delta_a_candidate = log_candidate_common - log_a_common
        log_delta_b_candidate = log_candidate_common - log_b_common
        a64 = probs_a.astype(np.float64)
        native_delta = probs_b.astype(np.float64) - a64
        candidate_delta = probs_candidate.astype(np.float64) - a64
        candidate_b_delta = probs_candidate.astype(np.float64) - probs_b.astype(np.float64)
        distance_a_b = float(np.linalg.norm(native_delta))
        distance_a_candidate = float(np.linalg.norm(candidate_delta))
        distance_b_candidate = float(np.linalg.norm(candidate_b_delta))
        distance_ratio = _distance_ratio(
            distance_a_candidate, distance_b_candidate, distance_a_b
        )
        lower = np.minimum(probs_a, probs_b)
        upper = np.maximum(probs_a, probs_b)
        coordinate_envelope_fraction = float(
            np.mean((probs_candidate >= lower) & (probs_candidate <= upper))
        )

        summary = summaries[half]
        summary["particle_count"] += 1
        summary["argmax_a_b"] += int(np.argmax(probs_a) != np.argmax(probs_b))
        summary["argmax_a_candidate"] += int(
            np.argmax(probs_a) != np.argmax(probs_candidate)
        )
        summary["argmax_b_candidate"] += int(
            np.argmax(probs_b) != np.argmax(probs_candidate)
        )
        summary["support_a_b"] += int(metric_a_b["support_mismatch_count"])
        summary["support_a_candidate"] += int(
            metric_a_candidate["support_mismatch_count"]
        )
        summary["support_b_candidate"] += int(
            metric_b_candidate["support_mismatch_count"]
        )
        summary["reference_norm2"] += float(np.vdot(a64, a64).real)
        summary["native_repeat_norm2"] += distance_a_b**2
        summary["candidate_a_norm2"] += distance_a_candidate**2
        summary["candidate_b_norm2"] += distance_b_candidate**2
        raw_a64 = normalized_raw_a.astype(np.float64)
        raw_delta_a_b = normalized_raw_b.astype(np.float64) - raw_a64
        raw_delta_a_candidate = normalized_raw_candidate.astype(np.float64) - raw_a64
        raw_delta_b_candidate = (
            normalized_raw_candidate.astype(np.float64)
            - normalized_raw_b.astype(np.float64)
        )
        summary["raw_reference_norm2"] += float(np.vdot(raw_a64, raw_a64).real)
        summary["raw_native_repeat_norm2"] += float(
            np.vdot(raw_delta_a_b, raw_delta_a_b).real
        )
        summary["raw_candidate_a_norm2"] += float(
            np.vdot(raw_delta_a_candidate, raw_delta_a_candidate).real
        )
        summary["raw_candidate_b_norm2"] += float(
            np.vdot(raw_delta_b_candidate, raw_delta_b_candidate).real
        )
        summary["raw_support_a_b"] += int(raw_metric_a_b["support_mismatch_count"])
        summary["raw_support_a_candidate"] += int(
            raw_metric_a_candidate["support_mismatch_count"]
        )
        summary["raw_support_b_candidate"] += int(
            raw_metric_b_candidate["support_mismatch_count"]
        )
        summary["log_native_delta_chunks"].append(log_delta_a_b)
        summary["log_candidate_delta_chunks"].append(log_delta_a_candidate)
        summary["log_a_b_square_sum"] += float(np.vdot(log_delta_a_b, log_delta_a_b).real)
        summary["log_a_candidate_square_sum"] += float(
            np.vdot(log_delta_a_candidate, log_delta_a_candidate).real
        )
        summary["log_b_candidate_square_sum"] += float(
            np.vdot(log_delta_b_candidate, log_delta_b_candidate).real
        )
        summary["log_coordinate_count"] += int(common_positive.sum())
        summary["log_a_b_max"] = max(
            summary["log_a_b_max"], float(np.max(np.abs(log_delta_a_b)))
        )
        summary["log_a_candidate_max"] = max(
            summary["log_a_candidate_max"], float(np.max(np.abs(log_delta_a_candidate)))
        )
        summary["log_b_candidate_max"] = max(
            summary["log_b_candidate_max"], float(np.max(np.abs(log_delta_b_candidate)))
        )
        summary["exact_native_raw_candidate_different_count"] += int(
            raw_metric_a_b["relative_l2"] == 0.0
            and raw_metric_a_candidate["relative_l2"] > 0.0
        )
        summary["native_delta_chunks"].append(native_delta.reshape(-1))
        summary["candidate_delta_chunks"].append(candidate_delta.reshape(-1))
        summary["distance_ratios"].append(distance_ratio)
        summary["within_native_repeat_count"] += int(
            min(distance_a_candidate, distance_b_candidate) <= distance_a_b
        )
        summary["coordinate_envelope_fractions"].append(coordinate_envelope_fraction)
        summary["per_particle"].append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "native_a_vs_native_b": metric_a_b,
                "native_a_vs_candidate": metric_a_candidate,
                "native_b_vs_candidate": metric_b_candidate,
                "normalized_raw_native_a_vs_native_b": raw_metric_a_b,
                "normalized_raw_native_a_vs_candidate": raw_metric_a_candidate,
                "normalized_raw_native_b_vs_candidate": raw_metric_b_candidate,
                "centered_log_weight_rms": {
                    "native_a_vs_native_b": _rms(log_delta_a_b),
                    "native_a_vs_candidate": _rms(log_delta_a_candidate),
                    "native_b_vs_candidate": _rms(log_delta_b_candidate),
                },
                "candidate_nearest_repeat_distance_over_native_repeat": distance_ratio,
                "candidate_coordinate_envelope_fraction": coordinate_envelope_fraction,
            }
        )

    half_reports = {}
    for half, summary in summaries.items():
        native_delta = np.concatenate(summary.pop("native_delta_chunks"))
        candidate_delta = np.concatenate(summary.pop("candidate_delta_chunks"))
        log_native_delta = np.concatenate(summary.pop("log_native_delta_chunks"))
        log_candidate_delta = np.concatenate(summary.pop("log_candidate_delta_chunks"))
        log_coordinate_count = int(summary["log_coordinate_count"])
        half_reports[str(half)] = {
            "particle_count": int(summary["particle_count"]),
            "argmax_mismatch_counts": {
                "native_a_vs_native_b": int(summary["argmax_a_b"]),
                "native_a_vs_candidate": int(summary["argmax_a_candidate"]),
                "native_b_vs_candidate": int(summary["argmax_b_candidate"]),
            },
            "support_mismatch_counts": {
                "native_a_vs_native_b": int(summary["support_a_b"]),
                "native_a_vs_candidate": int(summary["support_a_candidate"]),
                "native_b_vs_candidate": int(summary["support_b_candidate"]),
            },
            "pooled_relative_l2": {
                "native_a_vs_native_b": _pooled_relative_l2(
                    summary["reference_norm2"], summary["native_repeat_norm2"]
                ),
                "native_a_vs_candidate": _pooled_relative_l2(
                    summary["reference_norm2"], summary["candidate_a_norm2"]
                ),
                "native_b_vs_candidate_over_native_a_norm": _pooled_relative_l2(
                    summary["reference_norm2"], summary["candidate_b_norm2"]
                ),
            },
            "normalized_raw_weight_pooled_relative_l2": {
                "native_a_vs_native_b": _pooled_relative_l2(
                    summary["raw_reference_norm2"], summary["raw_native_repeat_norm2"]
                ),
                "native_a_vs_candidate": _pooled_relative_l2(
                    summary["raw_reference_norm2"], summary["raw_candidate_a_norm2"]
                ),
                "native_b_vs_candidate_over_native_a_norm": _pooled_relative_l2(
                    summary["raw_reference_norm2"], summary["raw_candidate_b_norm2"]
                ),
            },
            "raw_weight_support_mismatch_counts": {
                "native_a_vs_native_b": int(summary["raw_support_a_b"]),
                "native_a_vs_candidate": int(summary["raw_support_a_candidate"]),
                "native_b_vs_candidate": int(summary["raw_support_b_candidate"]),
            },
            "centered_log_weight_residual": {
                "common_positive_coordinate_count": log_coordinate_count,
                "rms": {
                    "native_a_vs_native_b": float(
                        np.sqrt(summary["log_a_b_square_sum"] / log_coordinate_count)
                    ),
                    "native_a_vs_candidate": float(
                        np.sqrt(summary["log_a_candidate_square_sum"] / log_coordinate_count)
                    ),
                    "native_b_vs_candidate": float(
                        np.sqrt(summary["log_b_candidate_square_sum"] / log_coordinate_count)
                    ),
                },
                "maximum_absolute": {
                    "native_a_vs_native_b": float(summary["log_a_b_max"]),
                    "native_a_vs_candidate": float(summary["log_a_candidate_max"]),
                    "native_b_vs_candidate": float(summary["log_b_candidate_max"]),
                },
                "candidate_delta_geometry_on_native_repeat": _residual_geometry(
                    log_native_delta, log_candidate_delta
                ),
            },
            "exact_native_raw_candidate_different_count": int(
                summary["exact_native_raw_candidate_different_count"]
            ),
            "candidate_delta_geometry_on_native_repeat": _residual_geometry(
                native_delta, candidate_delta
            ),
            "candidate_nearest_repeat_distance_over_native_repeat": _ratio_summary(
                summary["distance_ratios"]
            ),
            "candidate_within_native_repeat_distance_count": int(
                summary["within_native_repeat_count"]
            ),
            "candidate_coordinate_envelope_fraction": _quantiles(
                summary["coordinate_envelope_fractions"]
            ),
            "per_particle": summary["per_particle"],
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "coverage": {
            "native_particle_count": len(prefixes_a),
            "recovar_particle_count": len(locations),
            "relion_data_star_particle_count": len(all_original_by_part),
            "native_a_incomplete_capture_count": len(incomplete_a),
            "native_b_incomplete_capture_count": len(incomplete_b),
        },
        "halves": half_reports,
        "artifacts": {
            "native_a_directory": str(native_a_directory.resolve()),
            "native_b_directory": str(native_b_directory.resolve()),
            "relion_data_star": str(relion_data_star.resolve()),
            "recovar_half_captures": {
                str(half): [str(path.resolve()) for path in paths]
                for half, paths in half_capture_paths.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-a-directory", required=True, type=Path)
    parser.add_argument("--native-b-directory", required=True, type=Path)
    parser.add_argument("--relion-data-star", required=True, type=Path)
    parser.add_argument("--recovar-half1-capture", required=True, action="append", type=Path)
    parser.add_argument("--recovar-half2-capture", required=True, action="append", type=Path)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_a_directory,
        args.native_b_directory,
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
                "coverage": report["coverage"],
                "halves": {
                    half: {key: value for key, value in values.items() if key != "per_particle"}
                    for half, values in report["halves"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
