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


def test_float32_metric_uses_order_fixed_l2_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_blas_norm(*args: object, **kwargs: object) -> None:
        raise AssertionError("float32 metric must not use np.linalg.norm")

    monkeypatch.setattr(np.linalg, "norm", reject_blas_norm)
    metric = analyzer.float32_metric(
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([3.0, 4.0], dtype=np.float32),
    )

    assert metric["relative_l2_over_rhs"] == 1.0
    assert metric["l2_reduction"] == (
        "math.fsum_of_float64_squares_in_c_order"
    )


def test_stable_softmax_uses_fixed_scalar_reductions() -> None:
    normalized = analyzer._stable_softmax(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    np.testing.assert_array_equal(
        normalized,
        np.asarray([0.5, 0.5], dtype=np.float64),
    )


def test_normalized_mass_strata_partition_and_rank_deterministically() -> None:
    report = analyzer._normalized_mass_strata(
        stratum_ids=np.asarray([1, 1, 2, 2]),
        stratum_name="rotation",
        native_score_mass=np.asarray([0.4, 0.3, 0.2, 0.1]),
        recovar_score_mass=np.asarray([0.35, 0.4, 0.2, 0.05]),
        native_candidate_index=np.asarray([30, 10, 40, 20]),
        selected_stratum_ids=(2, 3),
        paired_ids=np.asarray([9, 9, 7, 7]),
        paired_name="native_rotation",
    )

    assert report["group_count"] == 2
    assert report["candidate_count"] == 4
    assert report["candidate_level_total_variation"] == 0.10000000000000003
    assert report["summed_stratum_tv_contributions"] == 0.10000000000000003
    assert report["partition_replay_residual"] == 0.0
    assert report["marginal_distribution_l1"] == 0.09999999999999999
    assert report["marginal_distribution_total_variation"] == (
        0.049999999999999996
    )
    assert report["marginal_tv_fraction_of_candidate_level_tv"] == (
        0.4999999999999998
    )
    assert report["within_stratum_cancellation_total_variation"] == (
        0.05000000000000004
    )
    assert (
        report[
            "within_stratum_cancellation_fraction_of_candidate_level_tv"
        ]
        == 0.5000000000000002
    )
    assert report["summed_marginal_tv_contributions"] == (
        0.049999999999999996
    )
    assert report["marginal_tv_replay_residual"] == 0.0
    assert [
        row["rotation"] for row in report["marginal_top_10"]
    ] == [2, 1]
    assert report["marginal_top_10"][0][
        "marginal_tv_rank_1based"
    ] == 1
    assert report["marginal_tv_concentration"]["top_1"][
        "available_strata_used"
    ] == 1
    assert report["marginal_tv_concentration"]["top_3"][
        "available_strata_used"
    ] == 2
    assert report["marginal_tv_concentration"]["top_10"][
        "share_of_marginal_distribution_tv"
    ] == 1.0
    assert [row["rotation"] for row in report["top_10"]] == [1, 2]
    assert report["top_10"][0]["candidate_level_tv_rank_1based"] == 1
    assert report["top_10"][0]["native_rotation"] == 9
    assert report["top_10"][0]["candidate_level_tv_contribution"] == (
        0.07500000000000004
    )
    assert report["top_10"][0]["share_of_total_candidate_level_tv"] == (
        0.7500000000000001
    )
    assert report["top_10"][0]["max_absolute_delta_representative"][
        "native_candidate_index"
    ] == 10
    assert [row["rotation"] for row in report["selected_strata"]] == [2]
    assert report["selected_strata_marginal_tv_contribution"] == (
        0.025
    )
    assert report[
        "selected_strata_share_of_marginal_distribution_tv"
    ] == 0.5000000000000001
    assert report["missing_selected_stratum_ids"] == [3]


def test_normalized_mass_strata_ranks_marginal_tv_with_exact_ties() -> None:
    report = analyzer._normalized_mass_strata(
        stratum_ids=np.asarray([1, 1, 2, 2, 3, 3]),
        stratum_name="translation",
        native_score_mass=np.asarray(
            [0.25, 0.125, 0.25, 0.125, 0.125, 0.125]
        ),
        recovar_score_mass=np.asarray(
            [0.3125, 0.1875, 0.1875, 0.125, 0.0625, 0.125]
        ),
        native_candidate_index=np.arange(6),
        selected_stratum_ids=(2, 3, 4),
        selected_stratum_sets={
            "queued": (2, 3),
            "with_missing": (4,),
        },
    )

    assert report["marginal_distribution_total_variation"] == 0.125
    assert report["summed_marginal_tv_contributions"] == 0.125
    assert report["marginal_tv_replay_residual"] == 0.0
    assert [
        row["translation"] for row in report["marginal_top_10"]
    ] == [1, 2, 3]
    assert [
        row["marginal_tv_rank_1based"]
        for row in report["marginal_top_10"]
    ] == [1, 2, 3]
    assert report["marginal_tv_concentration"]["top_1"][
        "share_of_marginal_distribution_tv"
    ] == 0.5
    assert report["marginal_tv_concentration"]["top_3"][
        "share_of_marginal_distribution_tv"
    ] == 1.0
    assert report["selected_strata_marginal_tv_contribution"] == 0.0625
    assert report[
        "selected_strata_share_of_marginal_distribution_tv"
    ] == 0.5
    assert report["selected_stratum_set_coverage"]["queued"] == {
        "stratum_ids": [2, 3],
        "present_stratum_ids": [2, 3],
        "missing_stratum_ids": [],
        "marginal_tv_contribution": 0.0625,
        "share_of_marginal_distribution_tv": 0.5,
    }
    assert report["selected_stratum_set_coverage"]["with_missing"] == {
        "stratum_ids": [4],
        "present_stratum_ids": [],
        "missing_stratum_ids": [4],
        "marginal_tv_contribution": 0.0,
        "share_of_marginal_distribution_tv": 0.0,
    }
    assert report["missing_selected_stratum_ids"] == [4]


def test_selected_translation_owner_concentration_closes_exactly() -> None:
    report = analyzer._selected_stratum_owner_concentration(
        stratum_ids=np.asarray([10, 10, 10]),
        stratum_name="translation",
        owner_ids=np.asarray([1, 2, 3]),
        owner_name="rotation",
        paired_owner_ids=np.asarray([11, 22, 33]),
        paired_owner_name="native_rotation",
        score_mass_delta=np.asarray([0.25, -0.125, -0.0625]),
        selected_stratum_ids=(10,),
        selected_owner_ids=(2, 4),
    )

    assert report["selected_stratum_ids"] == [10]
    translation = report["translations"][0]
    assert translation["direct_marginal_mass_delta_recovar_minus_native"] == (
        0.0625
    )
    assert translation[
        "summed_owner_mass_delta_recovar_minus_native"
    ] == 0.0625
    assert translation["owner_delta_replay_residual"] == 0.0
    assert translation["marginal_tv_contribution"] == 0.03125
    assert translation[
        "rotation_component_tv_before_cancellation"
    ] == 0.21875
    assert translation[
        "within_translation_rotation_cancellation_tv"
    ] == 0.1875
    assert translation[
        "within_translation_rotation_cancellation_fraction"
    ] == 6.0 / 7.0
    assert [
        row["rotation"]
        for row in translation["top_10_rotation_owners"]
    ] == [1, 2, 3]
    assert translation["rotation_owner_concentration"]["top_1"][
        "share_of_within_translation_rotation_component_tv"
    ] == 4.0 / 7.0
    assert translation["rotation_owner_concentration"]["top_3"][
        "share_of_within_translation_rotation_component_tv"
    ] == 1.0
    assert translation["selected_owners"][0]["rotation"] == 2
    assert translation["selected_owners"][0]["native_rotation"] == 22
    assert translation["selected_owners"][0][
        "rotation_component_tv_rank_1based"
    ] == 2
    assert translation["missing_selected_owner_ids"] == [4]


def test_selected_candidate_components_close_and_rank() -> None:
    report = analyzer._selected_candidate_component_attribution(
        components={
            "alpha": np.asarray([0.125, 0.0]),
            "beta": np.asarray([-0.0625, 0.0]),
            "zero": np.asarray([0.0, 0.0]),
        },
        total_delta=np.asarray([0.0625, 0.0]),
        native_pre_prior=np.asarray([1.0, 2.0], dtype=np.float32),
        recovar_pre_prior=np.asarray([1.125, 2.0], dtype=np.float32),
        native_combined=np.asarray([3.0, 4.0], dtype=np.float32),
        recovar_combined=np.asarray([3.0625, 4.0], dtype=np.float32),
        native_score_mass=np.asarray([0.25, 0.75]),
        recovar_score_mass=np.asarray([0.375, 0.625]),
        native_orientation_prior=np.asarray([0.5, 0.5], dtype=np.float32),
        recovar_orientation_prior=np.asarray([0.5, 0.5], dtype=np.float32),
        native_translation_prior=np.asarray([0.25, 0.25], dtype=np.float32),
        recovar_translation_prior=np.asarray([0.25, 0.5], dtype=np.float32),
        native_candidate_index=np.asarray([7, 8]),
        native_rotation_local=np.asarray([11, 12]),
        recovar_rotation_row=np.asarray([21, 22]),
        translation_id=np.asarray([31, 32]),
        selected_rotation=21,
        selected_translations=(31, 33),
    )

    assert report["missing_selected_translation_ids"] == [33]
    candidate = report["candidates"][0]
    assert candidate["translation_id"] == 31
    assert candidate["dominant_absolute_component"] == "alpha"
    assert candidate["component_l1"] == 0.1875
    assert candidate["component_sum"] == 0.0625
    assert candidate["combined_score_delta_recovar_minus_native"] == 0.0625
    assert candidate["telescoping_closure_residual"] == 0.0
    assert candidate["orientation_priors_bitwise_exact"] is True
    assert candidate["translation_priors_bitwise_exact"] is True
    assert [
        row["component"] for row in candidate["ranked_components"]
    ] == ["alpha", "beta", "zero"]
    assert candidate["ranked_components"][0][
        "share_of_candidate_component_l1"
    ] == 2.0 / 3.0


def test_softmax_partition_contributions_replay_and_rank() -> None:
    report = analyzer._softmax_partition_contribution_attribution(
        native_combined=np.asarray([0.0, 0.0]),
        recovar_combined=np.asarray([0.0, np.log(2.0)]),
        native_candidate_index=np.asarray([7, 8]),
        native_rotation_local=np.asarray([11, 12]),
        recovar_rotation_row=np.asarray([21, 21]),
        translation_id=np.asarray([31, 32]),
        selected_rotation=21,
        selected_translations=(31, 32),
    )

    assert report["shared_reference_score"] == np.log(2.0)
    assert report["native_partition"] == 1.0
    assert report["recovar_partition"] == 1.5
    assert report["partition_delta_recovar_minus_native"] == 0.5
    assert report["candidate_contribution_sum"] == 0.5
    assert report["partition_delta_replay_residual"] == 0.0
    assert report["absolute_candidate_contribution_sum"] == 0.5
    assert report["signed_cancellation_fraction"] == 0.0
    assert report["top_candidates"][0]["native_candidate_index"] == 8
    assert report["top_candidates"][0][
        "absolute_partition_contribution_rank_1based"
    ] == 1
    assert report["selected_share_of_absolute_partition_contributions"] == 1.0
    selected = report["selected_candidates"]
    assert [candidate["translation_id"] for candidate in selected] == [31, 32]
    assert selected[0]["combined_score_delta_recovar_minus_native"] == 0.0
    assert selected[0]["partition_contribution_recovar_minus_native"] == 0.0
    assert selected[1]["partition_contribution_recovar_minus_native"] == 0.5


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
        "decision_context": {
            "scope": (
                "within_captured_class_normalized_score_mass_only_"
                "not_full_kclass_posterior"
            ),
            "native_combined_score": pytest.approx(-5.701812744140625),
            "recovar_combined_score": pytest.approx(-5.7017822265625),
            "combined_score_delta_recovar_minus_native": pytest.approx(
                3.0517578125e-05
            ),
            "native_gap_below_class_max": 0.0,
            "recovar_gap_below_class_max": 0.0,
            "native_strict_rank_1based": 1,
            "recovar_strict_rank_1based": 1,
            "native_normalized_score_mass": 0.5,
            "recovar_normalized_score_mass": 0.5,
            "normalized_score_mass_delta_recovar_minus_native": 0.0,
        },
    }
    score_mass_effect = report["normalized_score_mass_effect"]
    assert {
        key: value
        for key, value in score_mass_effect.items()
        if key != "strata"
    } == {
        "scope": (
            "within_captured_class_normalized_score_mass_only_"
            "not_full_kclass_posterior"
        ),
        "normalization": (
            "math_exp_after_class_max_then_math_fsum_in_"
            "aligned_candidate_order"
        ),
        "native_sum": 1.0,
        "recovar_sum": 1.0,
        "l1": 0.0,
        "total_variation": 0.0,
        "max_absolute_delta": 0.0,
        "max_absolute_delta_representative": {
            "selection_rule": (
                "maximum_absolute_normalized_score_mass_delta_then_"
                "lowest_native_candidate_index"
            ),
            "aligned_table_index": 1,
            "native_candidate_index": 7,
            "native_rotation_local": 3,
            "recovar_rotation_row": 4,
            "translation_id": 9,
            "native_normalized_score_mass": 0.5,
            "recovar_normalized_score_mass": 0.5,
            "delta_recovar_minus_native": 0.0,
        },
    }
    strata = score_mass_effect["strata"]
    assert strata["scope"] == (
        "descriptive_partition_of_within_captured_class_candidate_"
        "level_total_variation_not_full_kclass_posterior"
    )
    assert strata["rotation"]["group_count"] == 2
    assert strata["rotation"]["candidate_level_total_variation"] == 0.0
    assert [
        row["recovar_rotation_row"]
        for row in strata["rotation"]["top_10"]
    ] == [2, 4]
    assert [
        row["recovar_rotation_row"]
        for row in strata["rotation"]["selected_strata"]
    ] == [4]
    assert strata["rotation"]["missing_selected_stratum_ids"] == [2626]
    assert strata["translation"]["group_count"] == 2
    assert [
        row["translation_id"]
        for row in strata["translation"]["top_10"]
    ] == [8, 9]
    assert [
        row["translation_id"]
        for row in strata["translation"]["selected_strata"]
    ] == [9]
    assert strata["translation"]["missing_selected_stratum_ids"] == [80, 82]


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
