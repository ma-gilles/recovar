#!/usr/bin/env python3
"""Decompose the exact-topology K4 RELION/RECOVAR fine-score residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_bpref_factor_capture import validate_directory as validate_factors
    from .validate_relion_fine_score_capture import (
        ACTIVE,
        load_fine_score_capture,
    )
    from .validate_relion_fine_score_capture import (
        validate_directory as validate_fine_scores,
    )
else:
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_bpref_factor_capture import (
        validate_directory as validate_factors,
    )
    from validate_relion_fine_score_capture import (  # type: ignore[no-redef]
        ACTIVE,
        load_fine_score_capture,
    )
    from validate_relion_fine_score_capture import (
        validate_directory as validate_fine_scores,
    )

PHYSICAL_IMAGE_SIZE = 256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _metric(relion: np.ndarray, recovar: np.ndarray) -> dict[str, object]:
    lhs = np.asarray(relion)
    rhs = np.asarray(recovar)
    _require(lhs.shape == rhs.shape, f"score shape changed: {lhs.shape} != {rhs.shape}")
    delta = rhs.astype(np.float64, copy=False) - lhs.astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(lhs.astype(np.float64, copy=False))), np.finfo(np.float64).tiny)
    return {
        "shape": list(lhs.shape),
        "relion_dtype": str(lhs.dtype),
        "recovar_dtype": str(rhs.dtype),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "mismatch_count": int(np.count_nonzero(lhs != rhs)),
        "relative_l2_over_relion": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _counterfactual_residuals(component_deltas: dict[str, np.ndarray]) -> dict[str, object]:
    total = sum((np.asarray(value, dtype=np.float64) for value in component_deltas.values()))
    total_energy = float(np.vdot(total, total).real)
    records = {}
    for name, delta in component_deltas.items():
        after = total - np.asarray(delta, dtype=np.float64)
        after_energy = float(np.vdot(after, after).real)
        records[name] = {
            "baseline_residual_l2": float(np.sqrt(total_energy)),
            "after_relion_component_substitution_l2": float(np.sqrt(after_energy)),
            "residual_energy_removed_fraction": (
                float(1.0 - after_energy / total_energy) if total_energy > 0 else 0.0
            ),
        }
    strongest = max(records, key=lambda name: records[name]["residual_energy_removed_fraction"])
    return {
        "component_substitution": records,
        "strongest_single_component": strongest,
        "strongest_residual_energy_removed_fraction": records[strongest][
            "residual_energy_removed_fraction"
        ],
    }


def _scalar_rotation_records(path: Path, stacks: list[int]) -> dict[int, tuple[tuple[int, int], ...]]:
    report = json.loads(path.read_text())
    _require(
        report.get("classification")
        == "pixel_varying_source_difference_not_explained_by_per_rotation_scalar",
        "pre-scatter scalar classification changed",
    )
    records: dict[int, tuple[tuple[int, int], ...]] = {}
    for particle in report.get("particles", []):
        stack = int(particle["stack_index_one_based"])
        if stack not in stacks:
            continue
        records[stack] = tuple(
            (
                int(fit["recovar_global_rotation_index"]),
                int(fit["relion_rotation_local_row"]),
            )
            for fit in particle["rotation_scalar_fits"]
        )
        _require(records[stack], f"stack {stack}: no exact contributor rotations")
        _require(len(set(records[stack])) == len(records[stack]), f"stack {stack}: duplicate contributor rotation")
    _require(set(records) == set(stacks), "fine-score contributor rotations are incomplete")
    return records


def _contribution_locations(
    directory: Path, stacks: list[int]
) -> tuple[dict[int, tuple[Path, int]], dict[str, str]]:
    wanted = set(stacks)
    locations: dict[int, tuple[Path, int]] = {}
    hashes: dict[str, str] = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            shard_stacks = np.asarray(archive["stack_indices_1based"], dtype=np.int64)
        matched = wanted.intersection(int(value) for value in shard_stacks)
        if not matched:
            continue
        hashes[path.name] = _sha256(path)
        for stack in matched:
            rows = np.flatnonzero(shard_stacks == stack)
            _require(rows.size == 1 and stack not in locations, f"stack {stack}: duplicate contribution shard")
            locations[stack] = (path, int(rows[0]))
    _require(set(locations) == wanted, "fine-score contribution shards are incomplete")
    return locations, hashes


def _translation_map(factor_capture, fine_translations: np.ndarray) -> dict[int, int]:
    relion = np.column_stack((factor_capture.translations["x"], factor_capture.translations["y"])).astype(
        np.float64
    )
    recovar = -2 * np.pi * np.asarray(fine_translations, dtype=np.float64) / PHYSICAL_IMAGE_SIZE
    distance = np.max(np.abs(relion[:, None, :] - recovar[None, :, :]), axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(nearest_error <= 1e-6) and np.unique(nearest).size == nearest.size,
        f"stack {factor_capture.stack_index}: translation alignment changed",
    )
    return {row: int(candidate) for row, candidate in enumerate(nearest)}


def compare(
    capture_directory: Path,
    selection_json: Path,
    contribution_directory: Path,
    scalar_json: Path,
) -> dict[str, object]:
    factor_validation = validate_factors(capture_directory, selection_json)
    score_validation = validate_fine_scores(capture_directory, selection_json)
    selection = json.loads(selection_json.read_text())
    stacks = [int(record["stack_index_1based"]) for record in selection["selected"]]
    rotations = _scalar_rotation_records(scalar_json, stacks)
    locations, contribution_hashes = _contribution_locations(contribution_directory, stacks)
    factors = {
        capture.stack_index: capture
        for capture in (
            load_factor_capture(path) for path in capture_directory.glob("*.bpre-v2.bin")
        )
    }
    scores = {
        capture.stack_index: capture
        for capture in (
            load_fine_score_capture(path)
            for path in capture_directory.glob("*.fine-score-v1.bin")
        )
    }
    _require(set(factors) == set(stacks) and set(scores) == set(stacks), "validated capture set changed")

    paired: dict[str, list[np.ndarray]] = {
        f"{name}_{engine}": []
        for name in (
            "orientation_log_prior",
            "translation_log_prior",
            "data_log_score_centered",
            "combined_log_score_centered",
            "raw_exp50_weight",
            "fine_post_vs_factor_weight",
        )
        for engine in ("relion", "recovar")
    }
    particles: list[dict[str, object]] = []
    for path in sorted({location[0] for location in locations.values()}):
        selected_stacks = [stack for stack in stacks if locations[stack][0] == path]
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        for stack in selected_stacks:
            particle = locations[stack][1]
            factor = factors[stack]
            score = scores[stack]
            translation_map = _translation_map(factor, values["fine_translations"])
            min_diff2 = _float32_from_bits(score.header[18])
            weights_max = _float32_from_bits(score.header[19])
            recovar_best = np.float64(values["candidate_best_log_score"][particle])
            particle_operands = {
                f"{name}_{engine}": []
                for name in (
                    "orientation_log_prior",
                    "translation_log_prior",
                    "data_log_score_centered",
                    "combined_log_score_centered",
                    "raw_exp50_weight",
                    "fine_post_vs_factor_weight",
                )
                for engine in ("relion", "recovar")
            }
            contributors: list[dict[str, object]] = []
            for global_rotation, relion_orientation in rotations[stack]:
                recovar_rows = np.flatnonzero(
                    values["oversampled_rotation_indices"][particle] == global_rotation
                )
                _require(
                    recovar_rows.size == 1,
                    f"stack {stack}: RECOVAR rotation {global_rotation} is not unique",
                )
                recovar_orientation = int(recovar_rows[0])
                selected = score.candidates[
                    (score.candidates["rotation_local"] == relion_orientation)
                    & ((score.candidates["flags"] & ACTIVE) != 0)
                ]
                _require(selected.size > 0, f"stack {stack}: contributor has no active fine scores")
                _require(
                    np.unique(selected["translation_id"]).size == selected.size,
                    f"stack {stack}: contributor translation identity is duplicated",
                )
                relion_translations = selected["translation_id"].astype(np.int64)
                _require(
                    np.all(relion_translations < factor.translations.size),
                    f"stack {stack}: fine-score translation exceeds factor panel",
                )
                recovar_translations = np.asarray(
                    [translation_map[int(value)] for value in relion_translations],
                    dtype=np.int64,
                )
                _require(
                    np.all(
                        values["candidate_mask"][
                            particle,
                            recovar_orientation,
                            recovar_translations,
                        ]
                    ),
                    f"stack {stack}: RELION active fine support is absent from RECOVAR",
                )
                relion_orientation_prior = selected["orientation_log_prior"]
                relion_translation_prior = selected["translation_log_prior"]
                relion_data_centered = (
                    min_diff2 - selected["raw_diff2"] - weights_max
                ).astype(np.float32)
                relion_combined_centered = (
                    selected["shifted_log_weight"] - np.float32(50.0)
                ).astype(np.float32)
                relion_raw_exp = selected["post_exponent_weight"]
                recovar_orientation_prior = np.asarray(
                    values["candidate_rotation_log_prior"][
                        particle,
                        recovar_orientation,
                    ],
                    dtype=np.float64,
                )
                recovar_orientation_prior = np.full(
                    selected.size, recovar_orientation_prior, dtype=np.float64
                )
                recovar_translation_prior = np.asarray(
                    values["candidate_translation_log_prior"][
                        particle,
                        recovar_translations,
                    ],
                    dtype=np.float64,
                )
                recovar_data_centered = np.asarray(
                    values["candidate_preprior_scores"][
                        particle,
                        recovar_orientation,
                        recovar_translations,
                    ],
                    dtype=np.float64,
                ) - recovar_best
                recovar_combined_centered = np.asarray(
                    values["candidate_combined_scores"][
                        particle,
                        recovar_orientation,
                        recovar_translations,
                    ],
                    dtype=np.float64,
                ) - recovar_best
                recovar_raw_exp = np.asarray(
                    values["candidate_raw_exp_weights_f32"][
                        particle,
                        recovar_orientation,
                        recovar_translations,
                    ],
                    dtype=np.float32,
                )
                factor_hypothesis_indices = (
                    relion_orientation * factor.translations.size + relion_translations
                )
                factor_raw_exp = factor.hypotheses["posterior"][factor_hypothesis_indices]

                values_by_name = {
                    "orientation_log_prior": (
                        relion_orientation_prior,
                        recovar_orientation_prior,
                    ),
                    "translation_log_prior": (
                        relion_translation_prior,
                        recovar_translation_prior,
                    ),
                    "data_log_score_centered": (
                        relion_data_centered,
                        recovar_data_centered,
                    ),
                    "combined_log_score_centered": (
                        relion_combined_centered,
                        recovar_combined_centered,
                    ),
                    "raw_exp50_weight": (relion_raw_exp, recovar_raw_exp),
                    "fine_post_vs_factor_weight": (relion_raw_exp, factor_raw_exp),
                }
                for name, (relion_value, recovar_value) in values_by_name.items():
                    paired[f"{name}_relion"].append(np.asarray(relion_value).reshape(-1))
                    paired[f"{name}_recovar"].append(np.asarray(recovar_value).reshape(-1))
                    particle_operands[f"{name}_relion"].append(np.asarray(relion_value).reshape(-1))
                    particle_operands[f"{name}_recovar"].append(np.asarray(recovar_value).reshape(-1))
                contributors.append(
                    {
                        "recovar_global_rotation_index": global_rotation,
                        "recovar_rotation_local": recovar_orientation,
                        "relion_rotation_local": relion_orientation,
                        "active_translation_count": int(selected.size),
                        "translation_ids_relion": relion_translations.tolist(),
                        "translation_ids_recovar": recovar_translations.tolist(),
                        "metrics": {
                            name: _metric(relion_value, recovar_value)
                            for name, (relion_value, recovar_value) in values_by_name.items()
                        },
                    }
                )
            particle_metrics = {
                name: _metric(
                    np.concatenate(particle_operands[f"{name}_relion"]),
                    np.concatenate(particle_operands[f"{name}_recovar"]),
                )
                for name in (
                    "orientation_log_prior",
                    "translation_log_prior",
                    "data_log_score_centered",
                    "combined_log_score_centered",
                    "raw_exp50_weight",
                    "fine_post_vs_factor_weight",
                )
            }
            component_deltas = {
                name: np.concatenate(particle_operands[f"{name}_recovar"]).astype(np.float64)
                - np.concatenate(particle_operands[f"{name}_relion"]).astype(np.float64)
                for name in (
                    "data_log_score_centered",
                    "orientation_log_prior",
                    "translation_log_prior",
                )
            }
            predicted_combined_delta = sum(component_deltas.values())
            observed_combined_delta = np.concatenate(
                particle_operands["combined_log_score_centered_recovar"]
            ).astype(np.float64) - np.concatenate(
                particle_operands["combined_log_score_centered_relion"]
            ).astype(np.float64)
            contributors_count = len(contributors)
            particles.append(
                {
                    "stack_index_1based": stack,
                    "matched_contributor_count": contributors_count,
                    "active_hypothesis_count": int(
                        sum(item["active_translation_count"] for item in contributors)
                    ),
                    "metrics": particle_metrics,
                    "component_delta_closure": _metric(
                        observed_combined_delta,
                        predicted_combined_delta,
                    ),
                    "component_counterfactual": _counterfactual_residuals(component_deltas),
                    "contributors": contributors,
                }
            )

    aggregate = {
        name: _metric(
            np.concatenate(paired[f"{name}_relion"]),
            np.concatenate(paired[f"{name}_recovar"]),
        )
        for name in (
            "orientation_log_prior",
            "translation_log_prior",
            "data_log_score_centered",
            "combined_log_score_centered",
            "raw_exp50_weight",
            "fine_post_vs_factor_weight",
        )
    }
    component_deltas = {
        name: np.concatenate(paired[f"{name}_recovar"]).astype(np.float64)
        - np.concatenate(paired[f"{name}_relion"]).astype(np.float64)
        for name in (
            "data_log_score_centered",
            "orientation_log_prior",
            "translation_log_prior",
        )
    }
    predicted_combined_delta = sum(component_deltas.values())
    observed_combined_delta = np.concatenate(
        paired["combined_log_score_centered_recovar"]
    ).astype(np.float64) - np.concatenate(
        paired["combined_log_score_centered_relion"]
    ).astype(np.float64)
    return {
        "schema": "k4-relion-recovar-fine-score-decomposition-v1",
        "status": "complete",
        "metric_policy": "exact and scale-aware array metrics only; no correlation",
        "score_convention": (
            "data and combined scores are centered by each engine's global all-class best; "
            "RELION uses captured min_diff2/weights_max and RECOVAR uses candidate_best_log_score"
        ),
        "factor_validation": factor_validation,
        "fine_score_validation": score_validation,
        "selection_sha256": _sha256(selection_json),
        "prescatter_scalar_sha256": _sha256(scalar_json),
        "contribution_artifact_sha256": contribution_hashes,
        "particle_count": len(particles),
        "matched_contributor_count": sum(
            int(particle["matched_contributor_count"]) for particle in particles
        ),
        "active_hypothesis_count": sum(
            int(particle["active_hypothesis_count"]) for particle in particles
        ),
        "aggregate": aggregate,
        "aggregate_component_delta_closure": _metric(
            observed_combined_delta,
            predicted_combined_delta,
        ),
        "aggregate_component_counterfactual": _counterfactual_residuals(component_deltas),
        "particles": particles,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--contribution-directory", required=True, type=Path)
    parser.add_argument("--prescatter-scalar-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {args.output_json}")
    report = compare(
        args.capture_directory,
        args.selection_json,
        args.contribution_directory,
        args.prescatter_scalar_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
