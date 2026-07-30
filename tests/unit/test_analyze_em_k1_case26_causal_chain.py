from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_em_k1_case26_causal_chain import CHAIN_CLASSIFICATION, build_report


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _summary(*, maximum: float, fraction_le: float, n: int = 1000) -> dict:
    return {
        "n": n,
        "max": maximum,
        "threshold_fractions": {"le_0.01": fraction_le},
    }


def _inputs(tmp_path: Path) -> dict[str, Path]:
    trajectory = {
        "schema": "em_k1_fsc_trajectory_audit_v2",
        "numbered_iterations": [
            {
                "relion_iteration": iteration,
                "cross_engine": {"merged": {"fsc_auc": fsc}},
                "merged_gt_fsc_auc_delta": gt_delta,
            }
            for iteration, fsc, gt_delta in [
                (1, 0.9999999999, 1.0e-8),
                (2, 0.9999997, 1.0e-5),
                (3, 0.9999991, 2.0e-5),
            ]
        ],
    }
    trajectory_path = _write(tmp_path / "trajectory.json", trajectory)
    trajectory_sha = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()

    particles = {
        "schema": "em_particle_state_distribution_audit_v1",
        "status": "pass",
        "n_images": 1000,
        "iterations": [],
    }
    for iteration, support_count, pmax, angle_fraction, translation_fraction in [
        (1, 0, 0.0, 1.0, 1.0),
        (2, 87, 3.6e-6, 1.0, 1.0),
        (3, 165, 1.8e-2, 0.998, 0.998),
    ]:
        particles["iterations"].append(
            {
                "relion_iteration": iteration,
                "recovar_vs_relion": {
                    "significant_support": {
                        "different_count": support_count,
                        "absolute": {"max": float(support_count > 0)},
                    },
                    "pmax": {"absolute": {"max": pmax}},
                    "angular_error_deg": _summary(
                        maximum=8.6 if iteration == 3 else 1.0e-5,
                        fraction_le=angle_fraction,
                    ),
                    "translation_error": _summary(
                        maximum=2.125 if iteration == 3 else 4.0e-6,
                        fraction_le=translation_fraction,
                    ),
                },
            }
        )
    particle_path = _write(tmp_path / "particles.json", particles)

    identity = {
        "recovar_original_index_zero_based": 206,
        "relion_stack_index_one_based": 207,
        "ordered_top_keys": [[36, 54], [38, 55]],
        "current_size": 66,
    }
    operand_path = _write(
        tmp_path / "operands.json",
        {
            "schema": "em-k1-fine-top-pair-operands-v1",
            "status": "pass",
            "classification": "fine_winner_flip_is_projected_reference_determined",
            "identity": identity,
        },
    )
    source_path = _write(
        tmp_path / "source.json",
        {
            "schema": "em-k1-fine-ppref-source-boundary-v1",
            "status": "pass",
            "classification": "fine_projection_difference_is_iteration_start_map_state",
            "first_open_boundary": "iteration-start half-map state entering physical iteration 3",
            "closed_boundaries": ["texture", "map-to-PPref"],
            "identity": identity,
        },
    )
    precision_path = _write(
        tmp_path / "precision.json",
        {
            "schema": "em-k1-case26-xhalf-precision-factorial-v1",
            "status": "complete",
            "classification": (
                "double_xhalf_mstep_introduces_numbered_failures_and_worsens_case26_final_parity_on_matched_head"
            ),
            "paths": {"control": {"audit": str(trajectory_path)}},
            "artifact_sha256": {str(trajectory_path): trajectory_sha},
            "fixed_metric": {
                "control_numbered_failures": 0,
                "double_numbered_failures": 3,
                "double_minus_control_final_cross_engine_fsc_auc": -0.08,
            },
        },
    )
    return {
        "trajectory_json": trajectory_path,
        "particle_state_json": particle_path,
        "operand_report_json": operand_path,
        "source_boundary_json": source_path,
        "precision_factorial_json": precision_path,
    }


def test_build_report_closes_expected_temporal_chain(tmp_path: Path) -> None:
    report = build_report(**_inputs(tmp_path), expected_original_index=206)

    assert report["status"] == "pass"
    assert report["classification"] == CHAIN_CLASSIFICATION
    assert report["onsets"] == {
        "cross_engine_map_fsc_nonidentity_physical_iteration": 1,
        "significant_support_count_divergence_physical_iteration": 2,
        "hard_pose_divergence_physical_iteration": 3,
    }
    assert report["early_timeline"][1]["significant_support_different_count"] == 87
    assert report["early_timeline"][2]["angular_error_gt_0p01_deg_count"] == 2
    assert report["early_timeline"][2]["translation_error_gt_0p01_angstrom_count"] == 2


def test_build_report_fails_closed_when_support_onset_changes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = json.loads(inputs["particle_state_json"].read_text())
    payload["iterations"][0]["recovar_vs_relion"]["significant_support"]["different_count"] = 1
    inputs["particle_state_json"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="support divergence onset changed"):
        build_report(**inputs, expected_original_index=206)


def test_build_report_fails_closed_when_source_classification_changes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = json.loads(inputs["source_boundary_json"].read_text())
    payload["classification"] = "texture_projection_boundary_remains_open"
    inputs["source_boundary_json"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="source-boundary classification changed"):
        build_report(**inputs, expected_original_index=206)
