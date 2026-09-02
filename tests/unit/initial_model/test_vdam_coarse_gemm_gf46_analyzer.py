from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from scripts import analyze_vdam_coarse_gemm_gf46_gate as analyzer


def _performance_arms(*, direct: tuple[float, float], gemm: tuple[float, float]) -> dict:
    values = {
        "direct_1": direct[0],
        "direct_2": direct[1],
        "gemm_1": gemm[0],
        "gemm_2": gemm[1],
    }
    return {
        label: {
            "performance": {
                "warm_wall_s": wall,
                "warm_expectation_s": wall * 0.8,
                "warm_pass1_s": wall * 0.5,
                "warm_pass2_s": wall * 0.2,
                "peak_rss_gib": 12.0 if label.startswith("direct") else 13.0,
            }
        }
        for label, wall in values.items()
    }


def _numeric_arms(*, directional: bool = False, oversized: bool = False) -> dict:
    base = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    perturbation = np.full(4, 2e-4, dtype=np.float64)
    direct_1 = base
    direct_2 = base + perturbation
    gemm_1 = direct_1 + 0.5 * perturbation
    if directional:
        gemm_2 = direct_2 + 0.5 * perturbation
    elif oversized:
        gemm_2 = direct_2 - 3.0 * perturbation
    else:
        gemm_2 = direct_2 - 0.5 * perturbation
    values = {
        "direct_1": direct_1,
        "direct_2": direct_2,
        "gemm_1": gemm_1,
        "gemm_2": gemm_2,
    }
    return {
        label: {
            "warm": {
                "map": value.reshape(1, 2, 2),
                "model_state": {"continuous_values": value.copy()},
            }
        }
        for label, value in values.items()
    }


def _discrete_arms() -> dict:
    particles = pd.DataFrame(
        {
            "rlnImageName": ["000001@particles.mrcs", "000002@particles.mrcs"],
            "rlnAngleRot": [1.0, 2.0],
        }
    )
    metadata = {
        "selected_particle_ids": [0, 1],
        "best_pose_rotation_ids": [2, 3],
        "best_pose_rotations": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "best_pose_translations": [[0.0, 0.0], [1.0, -1.0]],
        "class_assignments": [0, 0],
        "max_posterior_per_image": [0.8, 0.9],
        "pose_assignments": [3, 4],
        "halfset_0_class_assignments": [0, 0],
    }
    phase = {
        "metadata": metadata,
        "star": {"particles": particles},
        "model_state": {
            "identity": {"iteration": 181, "reference": "run_it181_class001.mrc"},
            "continuous_keys": ["tau2[0]", "tau2[1]"],
        },
    }
    return {
        label: {
            "cold": {
                "metadata": dict(metadata),
                "star": {"particles": particles.copy()},
                "model_state": dict(phase["model_state"]),
            },
            "warm": {
                "metadata": dict(metadata),
                "star": {"particles": particles.copy()},
                "model_state": dict(phase["model_state"]),
            },
        }
        for label in analyzer.ARM_LABELS
    }


def _write_raw_artifact(
    path: Path,
    *,
    original_index: int,
    run_id: str,
    call_id: str,
    reasons: tuple[str, ...] = (),
    support_equal: bool = True,
    nonfinite: bool = False,
) -> None:
    direct = np.asarray([[[[-4.0, -3.0], [-2.0, -1.0]]]], dtype=np.float32)
    macro = direct.copy()
    if nonfinite:
        macro[0, 0, 0, 0] = np.nan
    direct_support = np.asarray([[[[False, False], [True, True]]]])
    macro_support = direct_support.copy()
    if not support_equal:
        macro_support[0, 0, 0, 0] = True
    delta = macro.astype(np.float64) - direct.astype(np.float64)
    np.savez_compressed(
        path,
        layout=np.asarray("image,class,rotation,translation"),
        qualification_status=np.asarray("NO_GO_UNQUALIFIED"),
        automatic_no_go_reasons=np.asarray(reasons, dtype=np.str_),
        pending_qualification_gates=np.asarray(
            ["repeat_stability", "bounded_non-growing", "material_runtime_win"],
            dtype=np.str_,
        ),
        paired_capture_active=np.asarray(True),
        clean_timing_eligible=np.asarray(False),
        requires_bitwise_score_identity=np.asarray(False),
        requires_exact_discrete_identity=np.asarray(True),
        diagnostic_run_id=np.asarray(run_id),
        diagnostic_call_id=np.asarray(call_id),
        diagnostic_selection_policy=np.asarray("explicit_call_scope_intersection"),
        debug_iteration=np.asarray(181, dtype=np.int64),
        current_size=np.asarray(128, dtype=np.int64),
        direct_scores_pre_prior=direct,
        macro_scores_pre_prior=macro,
        direct_scores_with_prior=direct,
        macro_scores_with_prior=macro,
        direct_support=direct_support,
        macro_support=macro_support,
        score_delta=delta,
        ulp_score_delta=np.zeros_like(direct, dtype=np.uint64),
        argmax_equal=np.asarray([True]),
        support_equal=np.asarray([support_equal]),
        exact_zero_direct_nonzero_macro_per_image=np.asarray([False]),
        original_indices=np.asarray([original_index], dtype=np.int64),
        macro_only_negative_implied_diff2_count=np.asarray(0, dtype=np.int64),
        resource_full_centered_projection_bytes=np.asarray(100, dtype=np.int64),
        resource_compact_projection_bytes=np.asarray(50, dtype=np.int64),
        resource_compact_projection_abs2_bytes=np.asarray(25, dtype=np.int64),
        resource_predicted_peak_projection_bytes=np.asarray(175, dtype=np.int64),
        resource_projected_transient_budget_bytes=np.asarray(200, dtype=np.int64),
        resource_pixel_index_device_to_host_materializations=np.asarray(0, dtype=np.int64),
    )


def _write_raw_tree(tmp_path: Path) -> Path:
    root = tmp_path / "gate"
    diagnostic = root / "raw_score_diagnostic" / "artifacts"
    output = root / "raw_score_diagnostic" / "output"
    diagnostic.mkdir(parents=True)
    output.mkdir(parents=True)
    run_id = "initial_model_it0181_cs0128_k001_fixture"
    call_id = "call0000_group0000_joint_halfsets"
    call_ids = [call_id]
    artifact_names = []
    for original_index, batch_start, batch_end in (
        (1, 0, 500),
        (2160, 500, 1000),
    ):
        artifact_name = (
            f"coarse_gemm_ab_{run_id}_{call_id}_it0181_cs0128_"
            f"batch{batch_start:08d}_{batch_end:08d}.npz"
        )
        _write_raw_artifact(
            diagnostic / artifact_name,
            original_index=original_index,
            run_id=run_id,
            call_id=call_id,
        )
        artifact_names.append(artifact_name)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "call_id": call_id,
        "expected_call_ids": call_ids,
        "selection_policy": "explicit_call_scope_intersection",
        "requested_original_indices": [1, 2160],
        "targets_in_scope": [1, 2160],
        "targets_explicitly_out_of_scope": [],
        "captured_target_counts": {"1": 1, "2160": 1},
        "artifact_paths": artifact_names,
    }
    (diagnostic / f"coarse_gemm_scope_{run_id}_{call_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    aggregate = {
        "schema_version": 1,
        "run_id": run_id,
        "expected_call_ids": call_ids,
        "requested_original_indices": [1, 2160],
        "captured_target_counts": {"1": 1, "2160": 1},
        "all_requested_captured_exactly_once": True,
        "scope_records": [record],
    }
    aggregate_path = diagnostic / f"coarse_gemm_manifest_{run_id}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    selected_particle_ids = list(range(10_000, 11_000))
    selected_particle_ids[72] = 1
    selected_particle_ids[999] = 2160
    (output / "run_it181_recovar_meta.json").write_text(
        json.dumps(
            {
                "coarse_gaussian_gemm_aggregate_manifest_path": str(aggregate_path.resolve()),
                "joint_halfset_particle_stream": True,
                "selected_particle_ids": selected_particle_ids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def test_panel_contract_is_frozen_a_b_b_a_at_point_nine() -> None:
    assert analyzer.ARM_LABELS == ("direct_1", "gemm_1", "gemm_2", "direct_2")
    assert [spec[1] for spec in analyzer.ARM_SPECS] == [0, 1, 1, 0]
    assert analyzer.MATERIAL_END_TO_END_RATIO == 0.90
    assert analyzer.RAW_TARGETS == (1, 2160)
    assert analyzer.RAW_TARGET_POSITIONS == (72, 999)
    assert analyzer.RAW_TARGET_PART_IDS == (1, 2160)
    assert analyzer.RAW_TARGET_HALFSETS == (1, 0)
    assert (
        analyzer.EXPECTED_SELECTED_IDS_INT64_SHA256
        == "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be"
    )
    assert analyzer.EXPECTED_SCHEDULE == {
        "current_size": 128,
        "healpix_order": 3,
        "n_rotations": 294_912,
        "n_translations": 116,
        "subset_size": 1_000,
        "random_perturbation": 0.4751259684562683,
    }


def test_performance_accepts_the_exact_crossed_median_boundary() -> None:
    report = analyzer._performance(
        _performance_arms(direct=(10.0, 12.0), gemm=(9.0, 10.8))
    )

    assert report["medians"]["direct"]["warm_wall_s"] == 11.0
    assert report["medians"]["gemm"]["warm_wall_s"] == 9.9
    assert report["median_warm_wall_ratio"] == pytest.approx(0.9)
    assert report["crossed_median_end_to_end_gate_pass"] is True


def test_performance_rejects_a_small_win_that_misses_the_predeclared_gate() -> None:
    report = analyzer._performance(
        _performance_arms(direct=(10.0, 10.0), gemm=(9.01, 9.01))
    )

    assert report["median_warm_wall_ratio"] == pytest.approx(0.901)
    assert report["pass"] is False


@pytest.mark.parametrize("field", ["map", "model_state"])
def test_numeric_panel_accepts_bounded_repeat_noise_with_opposed_cross_drift(field: str) -> None:
    report = analyzer._numeric_repeat_panel(_numeric_arms(), field=field)

    assert report["gemm_repeat_within_direct_envelope"] is True
    assert report["all_cross_arm_deltas_within_direct_envelope"] is True
    assert report["cross_arm_signed_drift_opposes_or_is_zero"] is True
    assert report["pass"] is True


def test_numeric_panel_rejects_directional_cross_arm_bias_even_inside_envelope() -> None:
    report = analyzer._numeric_repeat_panel(_numeric_arms(directional=True), field="map")

    assert report["all_cross_arm_deltas_within_direct_envelope"] is True
    assert report["cross_arm_signed_drift_opposes_or_is_zero"] is False
    assert report["pass"] is False


def test_numeric_panel_rejects_candidate_repeat_growth_beyond_control() -> None:
    report = analyzer._numeric_repeat_panel(_numeric_arms(oversized=True), field="model_state")

    assert report["gemm_repeat_within_direct_envelope"] is False
    assert report["pass"] is False


def test_discrete_panel_requires_exact_cold_warm_and_cross_arm_state() -> None:
    arms = _discrete_arms()
    assert analyzer._discrete_checks(arms)["pass"] is True

    arms["gemm_2"]["warm"]["metadata"]["pose_assignments"] = [3, 99]
    report = analyzer._discrete_checks(arms)
    assert report["warm_panel"]["direct_1__gemm_2"]["all_exact"] is False
    assert report["pass"] is False


def test_raw_artifact_reports_finite_exact_support_and_argmax(tmp_path: Path) -> None:
    path = tmp_path / "raw.npz"
    _write_raw_artifact(path, original_index=0, run_id="run", call_id="call")

    report = analyzer._raw_artifact_summary(
        path,
        expected_run_id="run",
        expected_call_id="call",
    )

    assert report["original_indices"] == [0]
    assert report["support_exact"] is True
    assert report["argmax_exact"] is True
    assert report["automatic_no_go_reasons"] == []
    assert report["clean_timing_eligible"] is False


def test_raw_artifact_rejects_an_automatic_no_go_reason(tmp_path: Path) -> None:
    path = tmp_path / "raw.npz"
    _write_raw_artifact(
        path,
        original_index=0,
        run_id="run",
        call_id="call",
        reasons=("negative_implied_diff2",),
    )

    with pytest.raises(analyzer.GemmGateSetupError, match="automatic NO-GO"):
        analyzer._raw_artifact_summary(path, expected_run_id="run", expected_call_id="call")


def test_raw_artifact_rejects_support_mismatch_and_nonfinite_scores(tmp_path: Path) -> None:
    support_path = tmp_path / "support.npz"
    _write_raw_artifact(
        support_path,
        original_index=0,
        run_id="run",
        call_id="call",
        support_equal=False,
    )
    with pytest.raises(analyzer.GemmGateSetupError, match="support differs"):
        analyzer._raw_artifact_summary(
            support_path,
            expected_run_id="run",
            expected_call_id="call",
        )

    nonfinite_path = tmp_path / "nonfinite.npz"
    _write_raw_artifact(
        nonfinite_path,
        original_index=0,
        run_id="run",
        call_id="call",
        nonfinite=True,
    )
    with pytest.raises(analyzer.GemmGateSetupError, match="non-finite"):
        analyzer._raw_artifact_summary(
            nonfinite_path,
            expected_run_id="run",
            expected_call_id="call",
        )


def test_raw_tree_requires_one_joint_scope_and_two_exact_once_artifacts(tmp_path: Path) -> None:
    root = _write_raw_tree(tmp_path)

    report = analyzer._validate_raw_score_diagnostic(root)

    assert report["scope_manifest_count"] == 1
    assert report["artifact_count"] == 2
    assert report["captured_original_indices"] == [1, 2160]
    assert report["all_requested_captured_exactly_once"] is True
    assert report["pass"] is True


def test_raw_tree_rejects_duplicate_aggregate_counts(tmp_path: Path) -> None:
    root = _write_raw_tree(tmp_path)
    aggregate_path = next(
        (root / "raw_score_diagnostic" / "artifacts").glob("coarse_gemm_manifest_*.json")
    )
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["captured_target_counts"]["2160"] = 2
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    with pytest.raises(analyzer.GemmGateSetupError, match="target counts"):
        analyzer._validate_raw_score_diagnostic(root)


def test_raw_tree_rejects_a_second_pseudo_half_scope(tmp_path: Path) -> None:
    root = _write_raw_tree(tmp_path)
    diagnostic = root / "raw_score_diagnostic" / "artifacts"
    extra = diagnostic / "coarse_gemm_scope_extra_call.json"
    extra.write_text("{}\n")

    with pytest.raises(analyzer.GemmGateSetupError, match="one joint-halfset scope"):
        analyzer._validate_raw_score_diagnostic(root)


def test_raw_target_preflight_requires_the_hard_pinned_subset_state(tmp_path: Path) -> None:
    root = tmp_path / "gate"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    path = provenance / "raw_target_preflight.json"
    payload = {
        "schema": "recovar.vdam_coarse_gemm_gf46_raw_target_preflight.v1",
        "checkpoint": str(analyzer.EXPECTED_CHECKPOINT_OPTIMISER.resolve()),
        "input_star": str(analyzer.EXPECTED_INPUT_STAR.resolve()),
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "random_seed": 29,
        "native_shuffle_seed": 210,
        "particle_count": 3000,
        "subset_size": 1000,
        "selected_particle_ids_int64_sha256": (
            "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be"
        ),
        "joint_halfset_particle_stream": True,
        "requested_image_batch_size": 500,
        "targets": [
            {
                "original_index": 1,
                "selected_position": 72,
                "part_id": 1,
                "pseudo_halfset_id": 1,
                "requested_batch_index": 0,
            },
            {
                "original_index": 2160,
                "selected_position": 999,
                "part_id": 2160,
                "pseudo_halfset_id": 0,
                "requested_batch_index": 1,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    report = analyzer._validate_raw_target_preflight(root)
    assert report["state"] == payload

    payload["targets"][1]["pseudo_halfset_id"] = 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(analyzer.GemmGateSetupError, match="raw target preflight differs"):
        analyzer._validate_raw_target_preflight(root)


def test_model_state_normalizes_arm_specific_reference_paths(tmp_path: Path) -> None:
    first = tmp_path / "first" / "run_it181_model.star"
    second = tmp_path / "second" / "run_it181_model.star"
    for path in (first, second):
        path.parent.mkdir()
        starfile.write(
            {
                "model_general": {
                    "rlnCurrentIteration": 181,
                    "rlnAveragePmax": 0.8,
                },
                "model_classes": pd.DataFrame(
                    {
                        "rlnReferenceImage": [
                            str(path.parent / "run_it181_class001.mrc")
                        ],
                        "rlnClassDistribution": [1.0],
                    }
                ),
            },
            path,
            overwrite=True,
        )

    left = analyzer._model_state(first, "first")
    right = analyzer._model_state(second, "second")

    assert left["identity"] == right["identity"]
    assert left["continuous_keys"] == right["continuous_keys"]
    np.testing.assert_array_equal(left["continuous_values"], right["continuous_values"])
