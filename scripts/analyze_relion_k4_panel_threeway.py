#!/usr/bin/env python3
"""Compare a fixed K=4 panel across host, RELION-CUDA, and RELION CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PHYSICAL_IMAGE_SIZE = 256
ACTIVE = np.uint32(8)
PANEL_SCHEMA = "recovar.k4_iter10_class2_residual_target_panel.v1"
CAPTURE_INERTNESS_SCHEMA = "relion_k4_fine_score_panel_capture_inertness_v2"
CAPTURE_REPEATABILITY_SCHEMA = "relion.k4_iter10_panel12_capture_repeatability.v1"
CONTROL_REPEATABILITY_SCHEMA = "relion_k4_uninstrumented_control_repeatability_v1"
SCREEN_SCHEMA = "recovar.k4_iter10_panel12_preprocess_screen.v1"
REPORT_SCHEMA = "recovar.k4_iter10_panel12_threeway_fine_score.v2"
MAP_FSC_AUC_THRESHOLD = 0.999999
SCIENCE_PARTICLE_FIELDS = (
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnClassNumber",
)
ALL_PARTICLE_FIELDS = SCIENCE_PARTICLE_FIELDS + (
    "rlnMaxValueProbDistribution",
    "rlnNrOfSignificantSamples",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in text.encode():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _center(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(np.all(np.isfinite(array)), "score panel contains non-finite values")
    return array - np.mean(array, dtype=np.float64) if array.size else array


def _residual(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    lhs = np.asarray(reference, dtype=np.float64)
    rhs = np.asarray(candidate, dtype=np.float64)
    _require(lhs.shape == rhs.shape, "score residual shape changed")
    delta = rhs - lhs
    energy = math.fsum(float(value) * float(value) for value in delta.reshape(-1))
    return {
        "candidate_count": int(delta.size),
        "empty": not bool(delta.size),
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "residual_l2": float(np.sqrt(energy)),
        "residual_energy": energy,
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "median_abs": float(np.median(np.abs(delta))) if delta.size else 0.0,
        "p95_abs": float(np.quantile(np.abs(delta), 0.95)) if delta.size else 0.0,
    }


def _rotation_map(factor_rotations: np.ndarray, recovar_rotations: np.ndarray) -> tuple[np.ndarray, float]:
    relion = np.asarray(factor_rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3)
    relion = relion.transpose(0, 2, 1)
    recovar = np.asarray(recovar_rotations, dtype=np.float32)
    distance = np.max(np.abs(relion[:, None] - recovar[None]), axis=(2, 3))
    nearest = np.argmin(distance, axis=1)
    nearest_error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(nearest_error <= 1e-6) and np.unique(nearest).size == nearest.size,
        "RELION/RECOVAR fine rotation matrices do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(nearest_error, initial=0.0))


def _translation_map(
    factor_translations: np.ndarray,
    fine_translations: np.ndarray,
) -> tuple[np.ndarray, float]:
    relion = np.column_stack((factor_translations["x"], factor_translations["y"])).astype(np.float64)
    recovar = -2 * np.pi * np.asarray(fine_translations, dtype=np.float64) / PHYSICAL_IMAGE_SIZE
    distance = np.max(np.abs(relion[:, None] - recovar[None]), axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_error = distance[np.arange(relion.shape[0]), nearest]
    _require(
        np.all(nearest_error <= 1e-6) and np.unique(nearest).size == nearest.size,
        "RELION/RECOVAR fine translations do not map one-to-one",
    )
    return nearest.astype(np.int64), float(np.max(nearest_error, initial=0.0))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_calibration_inputs(
    *,
    inertness: dict[str, object],
    screen: dict[str, object],
    capture_repeatability: dict[str, object],
    control_repeatability: dict[str, object],
) -> dict[str, object]:
    _require(
        inertness.get("schema") == CAPTURE_INERTNESS_SCHEMA,
        "RELION control/capture inertness schema changed",
    )
    _require(
        inertness.get("dispatch_exact") is True
        and inertness.get("control_perturbation") == inertness.get("capture_perturbation"),
        "RELION control/capture dispatch or perturbation changed",
    )
    _require(
        inertness.get("threshold") == MAP_FSC_AUC_THRESHOLD,
        "RELION control/capture FSC-AUC threshold changed",
    )
    exact_capture_fields = inertness.get("exact_particle_fields")
    _require(
        isinstance(exact_capture_fields, dict) and set(exact_capture_fields) == set(ALL_PARTICLE_FIELDS),
        "RELION control/capture particle-field schema changed",
    )
    _require(
        len(inertness["class_map_comparison"]) == 4
        and all(
            row["capture_vs_control_fsc_auc"] >= inertness["threshold"]
            for row in inertness["class_map_comparison"]
        ),
        "RELION control/capture map inertness changed",
    )
    _require(
        all(exact_capture_fields[field] for field in SCIENCE_PARTICLE_FIELDS),
        "RELION control/capture changed pose, translation, or class fields",
    )
    _require(
        capture_repeatability.get("schema") == CAPTURE_REPEATABILITY_SCHEMA
        and capture_repeatability.get("status") == "complete",
        "capture repeatability report is incomplete",
    )
    capture_scope = capture_repeatability["scope"]
    _require(
        capture_scope["target_count"] == 12
        and capture_scope["physical_iteration"] == 10
        and capture_scope["class_one_based"] == 2
        and capture_scope["geometry_exact_all"]
        and capture_scope["winners_exact_all"],
        "capture geometry or winners are not repeatable",
    )
    _require(
        all(
            row["factor_rotations_exact"]
            and row["factor_translations_exact"]
            and all(row["candidate_topology_exact"].values())
            for row in capture_repeatability["targets"]
        ),
        "capture-panel topology is not exactly repeatable",
    )
    _require(
        control_repeatability.get("schema") == CONTROL_REPEATABILITY_SCHEMA,
        "control repeatability schema changed",
    )
    _require(
        control_repeatability.get("threshold") == MAP_FSC_AUC_THRESHOLD,
        "control repeatability FSC-AUC threshold changed",
    )
    _require(
        control_repeatability["dispatch_exact"]
        and control_repeatability["perturbation_a"] == control_repeatability["perturbation_b"],
        "uninstrumented control dispatch or perturbation changed",
    )
    exact_control_fields = control_repeatability["exact_particle_fields"]
    _require(
        set(exact_control_fields) == set(ALL_PARTICLE_FIELDS),
        "uninstrumented control particle-field schema changed",
    )
    _require(
        all(exact_control_fields[field] for field in SCIENCE_PARTICLE_FIELDS),
        "uninstrumented controls changed pose, translation, or class fields",
    )
    _require(
        len(control_repeatability["class_map_comparison"]) == 4
        and all(
            row["repeat_fsc_auc"] >= control_repeatability["threshold"]
            for row in control_repeatability["class_map_comparison"]
        ),
        "uninstrumented control map repeatability changed",
    )
    screen_scope = screen.get("scope")
    _require(
        screen.get("schema") == SCREEN_SCHEMA
        and screen.get("status") == "complete"
        and screen.get("topology_exact_all") is True,
        "host/RELION-CUDA preprocessing screen is incomplete",
    )
    _require(
        isinstance(screen_scope, dict)
        and screen_scope.get("physical_iteration") == 10
        and screen_scope.get("current_size") == 74
        and screen_scope.get("target_count") == 12
        and screen_scope.get("classes") == 4,
        "host/RELION-CUDA preprocessing screen scope changed",
    )

    capture_pmax = inertness["particle_fields"]["rlnMaxValueProbDistribution"]
    control_pmax = control_repeatability["particle_fields"]["rlnMaxValueProbDistribution"]
    capture_significant = inertness["particle_fields"]["rlnNrOfSignificantSamples"]
    control_significant = control_repeatability["particle_fields"]["rlnNrOfSignificantSamples"]
    all_exact = all(exact_capture_fields.values())
    expected_capture_status = "pass" if all_exact else "rejected"
    _require(
        inertness.get("status") == expected_capture_status,
        "RELION control/capture status is inconsistent with particle fields",
    )
    expected_control_status = "pass" if all(exact_control_fields.values()) else "rejected"
    _require(
        control_repeatability.get("status") == expected_control_status,
        "control repeatability status is inconsistent with particle fields",
    )
    return {
        "classification": (
            "strict_control_capture_inertness" if all_exact else "repeatability_calibrated_non_scorecard_diagnostic"
        ),
        "all_eight_particle_fields_exact": all_exact,
        "pose_translation_class_fields_exact": True,
        "capture_geometry_topology_and_winners_repeatable": True,
        "uninstrumented_control_floor_measured": True,
        "capture_pmax_mismatch_count": int(capture_pmax["mismatch_count"]),
        "control_repeat_pmax_mismatch_count": int(control_pmax["mismatch_count"]),
        "capture_pmax_max_abs": float(capture_pmax["max_abs"]),
        "control_repeat_pmax_max_abs": float(control_pmax["max_abs"]),
        "capture_significant_sample_mismatch_count": int(capture_significant["mismatch_count"]),
        "control_repeat_significant_sample_mismatch_count": int(control_significant["mismatch_count"]),
    }


def _classify_improvement(
    aggregate: dict[str, dict[str, object]],
    repeatability_floor: dict[str, dict[str, object]],
) -> tuple[str, dict[str, dict[str, float | bool]]]:
    improvement_vs_floor = {}
    uniformly_improved = True
    exceeds_floor = True
    for family in ("data", "combined"):
        host_energy = float(aggregate[family]["host_numpy"]["residual_energy"])
        candidate_energy = float(aggregate[family]["relion_cuda"]["residual_energy"])
        improvement_energy = host_energy - candidate_energy
        floor_energy = float(repeatability_floor[family]["residual_energy"])
        improved = improvement_energy > 0
        exceeds = improvement_energy > floor_energy
        uniformly_improved &= improved
        exceeds_floor &= exceeds
        improvement_vs_floor[family] = {
            "improvement_energy": improvement_energy,
            "capture_repeatability_residual_energy": floor_energy,
            "improvement_to_repeatability_energy_ratio": (
                improvement_energy / floor_energy if floor_energy > 0 else float("inf")
            ),
            "improvement_exceeds_capture_repeatability_floor": exceeds,
        }

    if uniformly_improved and exceeds_floor:
        classification = "relion_cuda_preprocessing_reduces_residual_beyond_capture_repeatability_floor"
    elif uniformly_improved:
        classification = "relion_cuda_preprocessing_reduction_is_within_capture_repeatability_floor"
    else:
        classification = "relion_cuda_preprocessing_does_not_uniformly_reduce_relion_fine_score_residual"
    return classification, improvement_vs_floor


def analyze(
    *,
    repo: Path,
    preprocess_root: Path,
    factor_directory: Path,
    panel_json: Path,
    inertness_json: Path,
    screen_json: Path,
    capture_repeatability_json: Path,
    control_repeatability_json: Path,
) -> dict[str, object]:
    sys.path.insert(0, str(repo / "scripts"))
    from validate_relion_bpref_factor_capture import load_factor_capture
    from validate_relion_fine_score_capture import load_fine_score_capture

    panel = json.loads(panel_json.read_text())
    inertness = json.loads(inertness_json.read_text())
    screen = json.loads(screen_json.read_text())
    capture_repeatability = json.loads(capture_repeatability_json.read_text())
    control_repeatability = json.loads(control_repeatability_json.read_text())
    _require(panel.get("schema") == PANEL_SCHEMA, "panel schema changed")
    targets = panel.get("targets")
    _require(isinstance(targets, list) and len(targets) == 12, "panel target count changed")
    diagnostic_admissibility = _validate_calibration_inputs(
        inertness=inertness,
        screen=screen,
        capture_repeatability=capture_repeatability,
        control_repeatability=control_repeatability,
    )

    expected_stacks = [int(target["zero_based_identity_row"]) + 1 for target in targets]
    _require(len(set(expected_stacks)) == len(expected_stacks), "panel contains duplicate stack identities")
    selected_hash = _fnv1a64(",".join(str(stack) for stack in expected_stacks))
    factor_paths = sorted(factor_directory.glob("*.bpre-v2.bin"))
    score_paths = sorted(factor_directory.glob("*.fine-score-v1.bin"))
    _require(len(factor_paths) == len(targets), "RELION BPref factor count changed")
    _require(len(score_paths) == len(targets), "RELION fine-score count changed")
    factors = {capture.stack_index: capture for capture in map(load_factor_capture, factor_paths)}
    scores = {capture.stack_index: capture for capture in map(load_fine_score_capture, score_paths)}
    _require(set(factors) == set(expected_stacks), "RELION BPref stack set changed")
    _require(set(scores) == set(expected_stacks), "RELION fine-score stack set changed")

    aggregate_values: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = []
    for target in targets:
        original_index = int(target["zero_based_identity_row"])
        stack_index = original_index + 1
        name = f"pass2_orig{original_index:06d}_class002_cs074.npz"
        host_path = preprocess_root / "host_numpy" / "pass2" / name
        relion_cuda_path = preprocess_root / "jax_gpu" / "pass2" / name
        host = _load_npz(host_path)
        relion_cuda = _load_npz(relion_cuda_path)
        factor = factors[stack_index]
        score = scores[stack_index]

        _require(int(host["original_index"]) == original_index, "host identity changed")
        _require(int(relion_cuda["original_index"]) == original_index, "RELION-CUDA identity changed")
        _require(int(host["class_index"]) == 1 and int(relion_cuda["class_index"]) == 1, "class changed")
        _require(int(host["current_size"]) == 74 and int(relion_cuda["current_size"]) == 74, "size changed")
        _require(factor.header[9:11] == (10, 2), "RELION BPref iteration/class changed")
        _require(score.header[4:6] == (10, 2), "RELION fine-score iteration/class changed")
        _require(factor.geometry_only, "RELION BPref capture is not geometry-only")
        _require(factor.header[29] == 12 and score.header[21] == 12, "capture target count changed")
        _require(factor.header[36] == selected_hash, "RELION BPref selected-set hash changed")
        _require(score.header[28] == selected_hash, "RELION fine-score selected-set hash changed")

        topology_fields = (
            "fine_translations",
            "fine_translation_parent",
            "rotations",
            "oversampled_rot_indices",
            "parent_map",
            "candidate_mask",
            "rotation_log_prior",
            "translation_log_prior",
        )
        topology_exact = {
            field: bool(np.array_equal(host[field], relion_cuda[field])) for field in topology_fields
        }
        _require(all(topology_exact.values()), f"target {stack_index}: host/RELION-CUDA topology changed")

        rotation_map, rotation_error = _rotation_map(factor.rotations, host["rotations"])
        translation_map, translation_error = _translation_map(factor.translations, host["fine_translations"])
        active = (score.candidates["flags"] & ACTIVE) != 0
        selected = score.candidates[active]
        relion_rotation = selected["rotation_local"].astype(np.int64)
        relion_translation = selected["translation_id"].astype(np.int64)
        _require(
            np.all(relion_rotation < rotation_map.size) and np.all(relion_translation < translation_map.size),
            f"target {stack_index}: RELION candidate index is outside the factor panel",
        )
        recovar_rotation = rotation_map[relion_rotation]
        recovar_translation = translation_map[relion_translation]
        _require(
            np.all(host["candidate_mask"][recovar_rotation, recovar_translation]),
            f"target {stack_index}: active RELION support is absent from host",
        )
        _require(
            np.all(relion_cuda["candidate_mask"][recovar_rotation, recovar_translation]),
            f"target {stack_index}: active RELION support is absent from RELION-CUDA",
        )
        keys = np.column_stack((recovar_rotation, recovar_translation))
        _require(np.unique(keys, axis=0).shape[0] == keys.shape[0], "mapped RELION candidates duplicate")

        min_diff2 = _float32_from_bits(score.header[18])
        relion_data = _center(min_diff2 - selected["raw_diff2"])
        relion_combined = _center(selected["combined_preexponent"])
        host_data = _center(host["scores_pre_prior"][recovar_rotation, recovar_translation])
        host_combined = _center(host["scores_with_prior"][recovar_rotation, recovar_translation])
        relion_cuda_data = _center(
            relion_cuda["scores_pre_prior"][recovar_rotation, recovar_translation]
        )
        relion_cuda_combined = _center(
            relion_cuda["scores_with_prior"][recovar_rotation, recovar_translation]
        )
        for label, values in (
            ("relion_data", relion_data),
            ("host_data", host_data),
            ("relion_cuda_data", relion_cuda_data),
            ("relion_combined", relion_combined),
            ("host_combined", host_combined),
            ("relion_cuda_combined", relion_cuda_combined),
        ):
            aggregate_values[label].append(values)

        winner_defined = bool(selected.size)
        relion_winner = int(np.argmax(relion_combined)) if winner_defined else None
        host_winner = int(np.argmax(host_combined)) if winner_defined else None
        relion_cuda_winner = int(np.argmax(relion_cuda_combined)) if winner_defined else None
        rows.append(
            {
                **target,
                "stack_index_one_based": stack_index,
                "active_candidate_count": int(selected.size),
                "rotation_matrix_map_max_abs": rotation_error,
                "translation_map_max_abs": translation_error,
                "topology_exact": topology_exact,
                "data_score_residual": {
                    "host_numpy": _residual(relion_data, host_data),
                    "relion_cuda": _residual(relion_data, relion_cuda_data),
                },
                "combined_score_residual": {
                    "host_numpy": _residual(relion_combined, host_combined),
                    "relion_cuda": _residual(relion_combined, relion_cuda_combined),
                },
                "winner": {
                    "winner_defined": winner_defined,
                    "relion_cuda_reference_flat": relion_winner,
                    "host_numpy_flat": host_winner,
                    "recovar_relion_cuda_flat": relion_cuda_winner,
                    "host_matches_relion": host_winner == relion_winner if winner_defined else None,
                    "recovar_relion_cuda_matches_relion": (
                        relion_cuda_winner == relion_winner if winner_defined else None
                    ),
                },
                "artifacts": {
                    "host_numpy": {"path": str(host_path), "sha256": _sha256(host_path)},
                    "relion_cuda": {"path": str(relion_cuda_path), "sha256": _sha256(relion_cuda_path)},
                    "relion_bpref": {
                        "path": str(factor.path),
                        "sha256": factor.sha256,
                        "geometry_only": factor.geometry_only,
                    },
                    "relion_fine_score": {"path": str(score.path), "sha256": score.sha256},
                },
            }
        )

    aggregate = {}
    for family in ("data", "combined"):
        relion = np.concatenate(aggregate_values[f"relion_{family}"])
        host = np.concatenate(aggregate_values[f"host_{family}"])
        relion_cuda = np.concatenate(aggregate_values[f"relion_cuda_{family}"])
        _require(relion.size > 0, f"all RELION {family} candidate sets are empty")
        host_metric = _residual(relion, host)
        relion_cuda_metric = _residual(relion, relion_cuda)
        baseline = float(host_metric["residual_energy"])
        candidate = float(relion_cuda_metric["residual_energy"])
        aggregate[family] = {
            "host_numpy": host_metric,
            "relion_cuda": relion_cuda_metric,
            "relion_cuda_residual_energy_change_vs_host_numpy": (
                candidate / baseline - 1.0 if baseline > 0 else 0.0
            ),
            "relion_cuda_residual_energy_removed_fraction": (
                1.0 - candidate / baseline if baseline > 0 else 0.0
            ),
        }

    repeatability_floor = {
        "data": capture_repeatability["aggregate"]["centered_raw_diff2"],
        "combined": capture_repeatability["aggregate"]["centered_combined"],
    }
    classification, improvement_vs_floor = _classify_improvement(aggregate, repeatability_floor)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "diagnostic_admissibility": diagnostic_admissibility,
        "quality_metric_policy": {
            "map_gate": "shellwise FSC/FSC-AUC only",
            "correlation_computed": False,
            "particle_metrics_are_diagnostic": True,
        },
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "current_size": 74,
            "target_count": len(rows),
            "winner_evaluable_target_count": sum(row["winner"]["winner_defined"] for row in rows),
            "backends": ["host_numpy", "relion_cuda", "RELION CUDA"],
            "host_winner_matches_relion_count": sum(
                row["winner"]["host_matches_relion"] is True for row in rows
            ),
            "relion_cuda_winner_matches_relion_count": sum(
                row["winner"]["recovar_relion_cuda_matches_relion"] is True for row in rows
            ),
        },
        "aggregate": aggregate,
        "capture_repeatability_floor": repeatability_floor,
        "improvement_vs_repeatability_floor": improvement_vs_floor,
        "targets": rows,
        "inputs": {
            "panel": {"path": str(panel_json), "sha256": _sha256(panel_json)},
            "inertness": {"path": str(inertness_json), "sha256": _sha256(inertness_json)},
            "screen": {"path": str(screen_json), "sha256": _sha256(screen_json)},
            "capture_repeatability": {
                "path": str(capture_repeatability_json),
                "sha256": _sha256(capture_repeatability_json),
            },
            "control_repeatability": {
                "path": str(control_repeatability_json),
                "sha256": _sha256(control_repeatability_json),
            },
            "analyzer_repo_head": "",
        },
        "next_step": (
            "Use this exact three-backend classification with the already-complete "
            "15-iteration FSC/FSC-AUC trajectory before changing any K-class default."
        ),
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status, "analyzer repository is dirty")
    return head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--preprocess-root", required=True, type=Path)
    parser.add_argument("--factor-directory", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--inertness-json", required=True, type=Path)
    parser.add_argument("--screen-json", required=True, type=Path)
    parser.add_argument("--capture-repeatability-json", required=True, type=Path)
    parser.add_argument("--control-repeatability-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        repo=args.repo,
        preprocess_root=args.preprocess_root,
        factor_directory=args.factor_directory,
        panel_json=args.panel_json,
        inertness_json=args.inertness_json,
        screen_json=args.screen_json,
        capture_repeatability_json=args.capture_repeatability_json,
        control_repeatability_json=args.control_repeatability_json,
    )
    report["inputs"]["analyzer_repo_head"] = _clean_repo_head(args.repo)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": report["classification"], **report["scope"]}, indent=2))


if __name__ == "__main__":
    main()
