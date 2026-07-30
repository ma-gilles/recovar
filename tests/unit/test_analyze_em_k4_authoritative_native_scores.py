from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_em_k4_authoritative_native_scores as analyzer


def _classification(**overrides: bool) -> str:
    values = {
        "support_exact": True,
        "winner_exact": True,
        "max_tie_key_sets_exact": True,
        "native_raw_diff2_tied": True,
        "recovar_scores_tied": True,
        "cross_engine_target_scores_bitwise_exact": True,
    }
    values.update(overrides)
    return analyzer.classify_target_parity(**values)


def test_classifies_complete_exact_target_parity() -> None:
    assert _classification() == analyzer.PASS_CLASSIFICATION


@pytest.mark.parametrize(
    ("field", "suffix"),
    [
        ("support_exact", "support_mismatch"),
        ("winner_exact", "winner_mismatch"),
        ("max_tie_key_sets_exact", "max_ties_mismatch"),
        ("native_raw_diff2_tied", "native_raw_tie_mismatch"),
        ("recovar_scores_tied", "recovar_tie_mismatch"),
        (
            "cross_engine_target_scores_bitwise_exact",
            "cross_engine_target_mismatch",
        ),
    ],
)
def test_classifies_each_single_target_failure(field: str, suffix: str) -> None:
    assert _classification(**{field: False}).endswith(suffix)


def test_classifies_mixed_target_failure() -> None:
    result = _classification(
        support_exact=False,
        cross_engine_target_scores_bitwise_exact=False,
    )
    assert result == ("exact_device_k4_target_mixed_mismatch__support__cross_engine_target")


def test_float32_metric_is_bit_sensitive() -> None:
    lhs = np.asarray([1.0, 2.0], dtype=np.float32)
    rhs = lhs.copy()
    rhs.view(np.uint32)[1] += 1

    metric = analyzer.float32_metric(lhs, rhs)

    assert metric["count"] == 2
    assert metric["bitwise_exact"] is False
    assert metric["bitwise_mismatch_count"] == 1
    assert metric["max_abs"] > 0.0


def test_target_score_offset_attribution_closes_exact_decomposition() -> None:
    report = analyzer.target_score_offset_attribution(
        min_diff2=np.float32(500.6817321777344),
        native_raw_diff2=np.float32(501.4734191894531),
        native_orientation_prior=np.float32(-4.860062599182129),
        native_translation_prior=np.float32(-0.05005118250846863),
        native_combined=np.float32(-5.701812744140625),
        recovar_pre_prior_residual=np.float64(-0.7916684448719025),
        recovar_orientation_prior=np.float32(-4.860062599182129),
        recovar_translation_prior=np.float32(-0.05005118250846863),
        recovar_combined=np.float32(-5.7017822265625),
        decision_topology_exact=True,
    )

    assert report["classification"] == analyzer.TARGET_OFFSET_CLASSIFICATION
    assert report["attributed"] is True
    assert report["target_priors_bitwise_exact"] is True
    assert report["native_production_formula_replay_bitwise_exact"] is True
    assert report["recovar_data_then_prior_replay_bitwise_exact"] is True
    assert report["combined_delta_decomposition"]["residual"] == 0.0
    assert (
        report["combined_delta_decomposition"]["sum"]
        == report["deltas_recovar_minus_native"]["combined"]
    )


def test_target_score_offset_attribution_rejects_prior_mismatch() -> None:
    report = analyzer.target_score_offset_attribution(
        min_diff2=np.float32(500.6817321777344),
        native_raw_diff2=np.float32(501.4734191894531),
        native_orientation_prior=np.float32(-4.860062599182129),
        native_translation_prior=np.float32(-0.05005118250846863),
        native_combined=np.float32(-5.701812744140625),
        recovar_pre_prior_residual=np.float64(-0.7916684448719025),
        recovar_orientation_prior=np.nextafter(
            np.float32(-4.860062599182129),
            np.float32(0.0),
        ),
        recovar_translation_prior=np.float32(-0.05005118250846863),
        recovar_combined=np.float32(-5.7017822265625),
        decision_topology_exact=True,
    )

    assert report["attributed"] is False
    assert report["target_priors_bitwise_exact"] is False


def _global_attribution(decision_topology_exact: bool = True) -> dict:
    count = 2
    min_diff2 = np.float32(500.6817321777344)
    native_raw = np.full(
        count,
        np.float32(501.4734191894531),
        dtype=np.float32,
    )
    native_orientation = np.full(
        count,
        np.float32(-4.860062599182129),
        dtype=np.float32,
    )
    native_translation = np.full(
        count,
        np.float32(-0.05005118250846863),
        dtype=np.float32,
    )
    native_combined = np.subtract(
        np.add(
            np.add(
                native_orientation,
                native_translation,
                dtype=np.float32,
            ),
            min_diff2,
            dtype=np.float32,
        ),
        native_raw,
        dtype=np.float32,
    )
    recovar_pre = np.full(
        count,
        np.float64(-0.7916684448719025),
        dtype=np.float64,
    )
    recovar_combined = np.add(
        np.add(
            recovar_pre.astype(np.float32),
            native_orientation,
            dtype=np.float32,
        ),
        native_translation,
        dtype=np.float32,
    )
    return analyzer.global_score_offset_attribution(
        min_diff2=min_diff2,
        native_raw_diff2=native_raw,
        native_orientation_prior=native_orientation,
        native_translation_prior=native_translation,
        native_combined=native_combined,
        recovar_pre_prior_residual=recovar_pre,
        recovar_orientation_prior=native_orientation,
        recovar_translation_prior=native_translation,
        recovar_combined=recovar_combined,
        native_candidate_index=np.asarray([11, 7]),
        native_rotation_local=np.asarray([5, 3]),
        recovar_rotation_row=np.asarray([2, 4]),
        translation_id=np.asarray([8, 9]),
        decision_topology_exact=decision_topology_exact,
    )


def test_global_score_offset_attribution_closes_and_is_data_dominated() -> None:
    report = _global_attribution()

    assert report["classification"] == analyzer.GLOBAL_OFFSET_CLASSIFICATION
    assert report["attributed"] is True
    assert report["native_production_replay_bitwise_exact_count"] == 2
    assert report["recovar_dump_replay_bitwise_exact_count"] == 2
    assert report["telescoping_closure"]["exact"] is True
    assert report["telescoping_closure"]["max_abs"] == 0.0
    assert report["pre_prior_data_path_strict_majority"] is True
    assert report["component_l1_fractions"]["pre_prior_data_path"] > 0.5
    assert report["pre_prior_data_path_representative"] == {
        "selection_rule": (
            "maximum_absolute_pre_prior_data_path_component_then_"
            "lowest_native_candidate_index"
        ),
        "aligned_table_index": 1,
        "native_candidate_index": 7,
        "native_rotation_local": 3,
        "recovar_rotation_row": 4,
        "translation_id": 9,
        "native_pre_prior": pytest.approx(-0.79168701171875),
        "recovar_pre_prior": pytest.approx(-0.7916684150695801),
        "component_delta_recovar_minus_native": pytest.approx(
            1.8596649169921875e-05
        ),
        "component_absolute_delta": pytest.approx(
            1.8596649169921875e-05
        ),
    }


def test_global_score_offset_attribution_requires_decision_topology() -> None:
    report = _global_attribution(decision_topology_exact=False)

    assert report["attributed"] is False
    assert report["decision_topology_exact"] is False


def test_rotation_permutation_accepts_exact_bijection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "EXPECTED_ROTATIONS", 3)
    recovar = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    native = recovar[[2, 0, 1]]

    permutation = analyzer._rotation_permutation(native, recovar)

    np.testing.assert_array_equal(permutation, np.asarray([2, 0, 1]))


def test_rotation_permutation_rejects_missing_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "EXPECTED_ROTATIONS", 3)
    recovar = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    native = recovar.copy()
    native[2, 0, 0] = -1.0

    with pytest.raises(ValueError, match="lack a bitwise RECOVAR match"):
        analyzer._rotation_permutation(native, recovar)


def test_allocation_requires_target_uuid(tmp_path) -> None:
    table = tmp_path / "allocation.csv"
    table.write_text("GPU-a, NVIDIA A100-SXM4-80GB, 0000:01:00.0\nGPU-b, NVIDIA A100-SXM4-80GB, 0000:02:00.0\n")

    with pytest.raises(ValueError, match="required exact GPU UUID"):
        analyzer._read_allocation_table(table)


def test_allocation_accepts_exact_target_and_peer(tmp_path) -> None:
    table = tmp_path / "allocation.csv"
    table.write_text(
        f"{analyzer.TARGET_GPU_UUID}, NVIDIA A100-SXM4-80GB, 0000:81:00.0\n"
        "GPU-peer, NVIDIA A100-SXM4-80GB, 0000:c1:00.0\n"
    )

    rows = analyzer._read_allocation_table(table)

    assert len(rows) == 2
    assert rows[0]["uuid"] == analyzer.TARGET_GPU_UUID


def test_completion_requires_exact_job_and_contract(tmp_path) -> None:
    completion = tmp_path / "completion.json"
    completion.write_text(
        """
{
  "schema": "relion_k4_it2_authoritative_native_capture_v1",
  "status": "complete",
  "slurm_job_id": 123,
  "sampling_perturbation": 0.27053284645080566,
  "scorecard_change_admissible": false,
  "grid_correction": "unset_default_off",
  "final_all_data_after_max_iter": "unset"
}
""".strip()
        + "\n"
    )

    report = analyzer._validate_completion(completion, expected_job_id=123)

    assert report["status"] == "complete"
    with pytest.raises(ValueError, match="job identity"):
        analyzer._validate_completion(completion, expected_job_id=124)


def test_state_requires_frozen_translation_grid(tmp_path) -> None:
    state = tmp_path / "state.json"
    state.write_text(
        """
{
  "schema": "relion_k4_it2_authoritative_translation_grid_validation_v1",
  "status": "accepted",
  "classification": "native_capture_matches_uninterrupted_iteration2_translation_grid",
  "translation_ids": [80, 82],
  "max_abs_pixels": 0.000001,
  "phase_capture_sha256": "%s"
}
"""
        % analyzer.RECOVAR_PASS2_SHA256
    )

    report = analyzer._validate_state(state)

    assert report["translation_ids"] == [80, 82]
