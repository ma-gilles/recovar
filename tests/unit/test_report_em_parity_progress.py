from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.report_em_parity_progress import build_progress, render_markdown


@pytest.mark.unit
def test_reports_all_fixed_em_parity_panels() -> None:
    progress = build_progress()

    assert progress["schema"] == "recovar.em_parity_progress.v15"
    assert progress["scorecard_change_admissible"] is False
    assert progress["k1_strict_history"] == [20, 21, 22, 23, 25, 26, 27, 28]
    assert progress["k1_continuation_initializer_progress"] == {
        "stock_pass": 3,
        "patched_pass": 3,
        "denominator": 21,
        "paired_gain": 0,
    }
    assert progress["k1_sampling_perturbation_progress"] == {
        "geometry_stock_pass": 3,
        "geometry_treatment_pass": 3,
        "geometry_denominator": 5,
        "geometry_gain": 0,
        "score_map_stock_pass": 3,
        "score_map_treatment_pass": 0,
        "score_map_denominator": 21,
        "score_map_gain": -3,
    }
    assert progress["k1_sampling_roundtrip_progress"] == {
        "geometry_stock_pass": 3,
        "geometry_treatment_pass": 5,
        "geometry_denominator": 5,
        "geometry_gain": 2,
        "score_map_stock_pass": 3,
        "score_map_treatment_pass": 17,
        "score_map_denominator": 21,
        "score_map_gain": 14,
    }
    assert progress["k1_norm_roundtrip_progress"] == {
        "preprocess_baseline_pass": 0,
        "preprocess_treatment_pass": 2,
        "preprocess_denominator": 2,
        "preprocess_gain": 2,
        "geometry_baseline_pass": 5,
        "geometry_treatment_pass": 5,
        "geometry_denominator": 5,
        "geometry_gain": 0,
        "score_map_baseline_pass": 17,
        "score_map_treatment_pass": 17,
        "score_map_denominator": 21,
        "score_map_gain": 0,
    }
    assert progress["k1_mask_deterministic_progress"] == {
        "preprocess_baseline_pass": 0,
        "preprocess_treatment_pass": 3,
        "preprocess_denominator": 3,
        "preprocess_gain": 3,
        "geometry_baseline_pass": 5,
        "geometry_treatment_pass": 5,
        "geometry_denominator": 5,
        "geometry_gain": 0,
        "score_map_baseline_pass": 17,
        "score_map_treatment_pass": 17,
        "score_map_denominator": 21,
        "score_map_gain": 0,
    }
    assert [
        (
            panel["id"],
            panel["passed"],
            panel["evaluated"],
            panel["denominator"],
            panel["rate_percent"],
            panel["scoring"],
        )
        for panel in progress["panels"]
    ] == [
        ("k1_strict", 28, 34, 34, 82.4, True),
        ("k1_topology", 32, 34, 34, 94.1, False),
        ("k1_evaluated", 34, 34, 34, 100.0, False),
        ("k1_restart_causal", 24, 42, 42, 57.1, False),
        ("k1_continuation_initializer", 3, 21, 21, 14.3, False),
        ("k1_sampling_perturbation_geometry", 3, 5, 5, 60.0, False),
        ("k1_sampling_perturbation_score_map", 0, 21, 21, 0.0, False),
        ("k1_sampling_roundtrip_geometry", 5, 5, 5, 100.0, False),
        ("k1_sampling_roundtrip_score_map", 17, 21, 21, 81.0, False),
        ("k1_norm_roundtrip_preprocess", 2, 2, 2, 100.0, False),
        ("k1_norm_roundtrip_geometry", 5, 5, 5, 100.0, False),
        ("k1_norm_roundtrip_score_map", 17, 21, 21, 81.0, False),
        ("k1_mask_deterministic_preprocess", 3, 3, 3, 100.0, False),
        ("k1_mask_deterministic_geometry", 5, 5, 5, 100.0, False),
        ("k1_mask_deterministic_score_map", 17, 21, 21, 81.0, False),
        ("k1_exact_initial_noise_counterfactual", 4, 24, 24, 16.7, False),
        ("k1_live_noise_counterfactual", 21, 24, 24, 87.5, False),
        ("k1_reference_roundtrip_rejection", 2, 9, 9, 22.2, False),
        ("k1_shared_checkpoint_fp64", 16, 34, 34, 47.1, False),
        ("k4_direct", 41, 60, 60, 68.3, True),
        ("k4_all_class", 9, 15, 15, 60.0, True),
        ("k4_causal", 2, 4, 4, 50.0, False),
        ("k4_contribution_repeatability", 0, 3, 3, 0.0, False),
        (
            "k4_deterministic_contribution_repeatability_candidate",
            3,
            3,
            3,
            100.0,
            False,
        ),
        ("k4_preprocess_bitwise", 3, 9, 9, 33.3, False),
        ("k4_preprocess_material", 9, 9, 9, 100.0, False),
    ]
    assert set(progress["inputs"]) == {
        "k1_scorecard",
        "k1_fixture_manifest",
        "k1_restart_scorecard",
        "k1_continuation_initializer_scorecard",
        "k1_sampling_perturbation_scorecard",
        "k1_sampling_roundtrip_scorecard",
        "k1_norm_roundtrip_scorecard",
        "k1_mask_deterministic_scorecard",
        "k1_exact_initial_noise_counterfactual_scorecard",
        "k1_live_noise_counterfactual_scorecard",
        "k1_reference_roundtrip_rejection_scorecard",
        "k1_shared_checkpoint_fp64_scorecard",
        "k4_trajectory_snapshot",
        "k4_class_scorecard",
        "k4_causal_scorecard",
        "k4_contribution_repeatability_scorecard",
        "k4_deterministic_contribution_repeatability_candidate_scorecard",
        "k4_preprocess_scorecard",
    }
    assert all(len(record["sha256"]) == 64 for record in progress["inputs"].values())
    assert progress["remaining"] == {
        "k1_strict_failures": [
            {
                "id": "k1-04",
                "name": "high_noise_100k_g256_white_noise3_bf80",
                "intermediate_result": "pass",
            },
            {
                "id": "k1-05",
                "name": "very_high_noise_100k_g256_white_noise10_bf80",
                "intermediate_result": "pass",
            },
            {
                "id": "k1-07",
                "name": "anisotropic_100k_g256_white_noise1_bf80",
                "intermediate_result": "fail",
            },
            {
                "id": "k1-10",
                "name": "high_res_anisotropic_100k_g384_radial_noise3_bf0",
                "intermediate_result": "pass",
            },
            {
                "id": "k1-22",
                "name": "small_severe_outliers_3k_g128_radial_noise5_bf80",
                "intermediate_result": "fail",
            },
            {
                "id": "k1-26",
                "name": "tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80",
                "intermediate_result": "pass",
            },
        ],
        "k1_topology_failures": [
            {
                "id": "k1-07",
                "name": "anisotropic_100k_g256_white_noise1_bf80",
            },
            {
                "id": "k1-22",
                "name": "small_severe_outliers_3k_g128_radial_noise5_bf80",
            },
        ],
        "k1_restart_causal_failures": [
            {
                "id": "iteration1-restart-score-stack-0035",
                "name": "Stack 35 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-0252",
                "name": "Stack 252 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-0348",
                "name": "Stack 348 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-0591",
                "name": "Stack 591 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-0683",
                "name": "Stack 683 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-1100",
                "name": "Stack 1100 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-1522",
                "name": "Stack 1522 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-1640",
                "name": "Stack 1640 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-1767",
                "name": "Stack 1767 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-2124",
                "name": "Stack 2124 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-2322",
                "name": "Stack 2322 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-2330",
                "name": "Stack 2330 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-2846",
                "name": "Stack 2846 score counterfactual",
            },
            {
                "id": "iteration1-restart-score-stack-2994",
                "name": "Stack 2994 score counterfactual",
            },
            {
                "id": "iteration1-restart-map-parity-half1",
                "name": "half1 RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "iteration1-restart-map-parity-half2",
                "name": "half2 RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "iteration1-restart-map-parity-merged",
                "name": "merged RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "iteration1-restart-overall",
                "name": "iteration1-restart overall acceptance",
            },
        ],
        "k1_continuation_initializer_failures": [
            {
                "id": "continuation-init-score-stack-0035",
                "name": "Stack 35 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-0252",
                "name": "Stack 252 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-0348",
                "name": "Stack 348 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-0591",
                "name": "Stack 591 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-0683",
                "name": "Stack 683 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-1100",
                "name": "Stack 1100 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-1522",
                "name": "Stack 1522 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-1640",
                "name": "Stack 1640 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-1767",
                "name": "Stack 1767 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-2124",
                "name": "Stack 2124 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-2322",
                "name": "Stack 2322 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-2330",
                "name": "Stack 2330 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-2846",
                "name": "Stack 2846 score counterfactual",
            },
            {
                "id": "continuation-init-score-stack-2994",
                "name": "Stack 2994 score counterfactual",
            },
            {
                "id": "continuation-init-map-parity-half1",
                "name": "half1 RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "continuation-init-map-parity-half2",
                "name": "half2 RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "continuation-init-map-parity-merged",
                "name": "merged RECOVAR-to-RELION FSC-AUC improvement",
            },
            {
                "id": "continuation-init-overall",
                "name": "Continuation-initializer overall acceptance",
            },
        ],
        "k4_direct_failures_by_iteration": [
            {"iteration": 10, "passed": 3, "failed": 1},
            {"iteration": 11, "passed": 0, "failed": 4},
            {"iteration": 12, "passed": 2, "failed": 2},
            {"iteration": 13, "passed": 0, "failed": 4},
            {"iteration": 14, "passed": 0, "failed": 4},
            {"iteration": 15, "passed": 0, "failed": 4},
        ],
        "k4_direct_failures": [
            {
                "id": "k4-it10-class2",
                "iteration": 10,
                "class": 2,
                "fsc_auc": 0.9948890936244424,
            },
            {
                "id": "k4-it11-class1",
                "iteration": 11,
                "class": 1,
                "fsc_auc": 0.9948252391062554,
            },
            {
                "id": "k4-it11-class2",
                "iteration": 11,
                "class": 2,
                "fsc_auc": 0.9934831677593646,
            },
            {
                "id": "k4-it11-class3",
                "iteration": 11,
                "class": 3,
                "fsc_auc": 0.9933616944645686,
            },
            {
                "id": "k4-it11-class4",
                "iteration": 11,
                "class": 4,
                "fsc_auc": 0.9946532315314668,
            },
            {
                "id": "k4-it12-class2",
                "iteration": 12,
                "class": 2,
                "fsc_auc": 0.9940211272807769,
            },
            {
                "id": "k4-it12-class3",
                "iteration": 12,
                "class": 3,
                "fsc_auc": 0.9936717912244045,
            },
            {
                "id": "k4-it13-class1",
                "iteration": 13,
                "class": 1,
                "fsc_auc": 0.9948790904352033,
            },
            {
                "id": "k4-it13-class2",
                "iteration": 13,
                "class": 2,
                "fsc_auc": 0.9933167865795268,
            },
            {
                "id": "k4-it13-class3",
                "iteration": 13,
                "class": 3,
                "fsc_auc": 0.9924670644050795,
            },
            {
                "id": "k4-it13-class4",
                "iteration": 13,
                "class": 4,
                "fsc_auc": 0.9944513966582168,
            },
            {
                "id": "k4-it14-class1",
                "iteration": 14,
                "class": 1,
                "fsc_auc": 0.9935362674564922,
            },
            {
                "id": "k4-it14-class2",
                "iteration": 14,
                "class": 2,
                "fsc_auc": 0.9919011463930602,
            },
            {
                "id": "k4-it14-class3",
                "iteration": 14,
                "class": 3,
                "fsc_auc": 0.9915812757362203,
            },
            {
                "id": "k4-it14-class4",
                "iteration": 14,
                "class": 4,
                "fsc_auc": 0.9943106599903103,
            },
            {
                "id": "k4-it15-class1",
                "iteration": 15,
                "class": 1,
                "fsc_auc": 0.9936005126730775,
            },
            {
                "id": "k4-it15-class2",
                "iteration": 15,
                "class": 2,
                "fsc_auc": 0.9912818890621345,
            },
            {
                "id": "k4-it15-class3",
                "iteration": 15,
                "class": 3,
                "fsc_auc": 0.9900911277299856,
            },
            {
                "id": "k4-it15-class4",
                "iteration": 15,
                "class": 4,
                "fsc_auc": 0.9938280816577081,
            },
        ],
        "k4_all_class_failure_iterations": [10, 11, 12, 13, 14, 15],
        "k4_causal_failures": [
            {"id": "global-raw-diff2", "name": "Complete active-table raw diff2"},
            {
                "id": "global-combined-score",
                "name": "Complete active-table combined score",
            },
        ],
    }


@pytest.mark.unit
def test_renders_pr_ready_fixed_metric_table() -> None:
    rendered = render_markdown(build_progress())

    assert "| K=1 strict FSC/FSC-AUC | **28** | 34 | 34 | 82.4% | yes |" in rendered
    assert "| K=1 serialized-restart causal gates | **24** | 42 | 42 | 57.1% | no |" in rendered
    assert ("| K=1 continuation initializer patched arm | **3** | 21 | 21 | 14.3% | no |") in rendered
    assert "| K=1 sampling-perturbation geometry identity | **3** | 5 | 5 | 60.0% | no |" in rendered
    assert "| K=1 sampling-perturbation score/map gates | **0** | 21 | 21 | 0.0% | no |" in rendered
    assert "| K=1 sampling-roundtrip geometry identity | **5** | 5 | 5 | 100.0% | no |" in rendered
    assert "| K=1 sampling-roundtrip score/map gates | **17** | 21 | 21 | 81.0% | no |" in rendered
    assert "| K=1 normalization-roundtrip preprocessing exactness | **2** | 2 | 2 | 100.0% | no |" in rendered
    assert "| K=1 normalization-roundtrip geometry identity | **5** | 5 | 5 | 100.0% | no |" in rendered
    assert "| K=1 normalization-roundtrip score/map gates | **17** | 21 | 21 | 81.0% | no |" in rendered
    assert "| K=1 deterministic-mask preprocessing exactness | **3** | 3 | 3 | 100.0% | no |" in rendered
    assert "| K=1 deterministic-mask geometry identity | **5** | 5 | 5 | 100.0% | no |" in rendered
    assert "| K=1 deterministic-mask score/map gates | **17** | 21 | 21 | 81.0% | no |" in rendered
    assert "| K=4 direct per-class FSC-AUC | **41** | 60 | 60 | 68.3% | yes |" in rendered
    assert "| K=4 exact-device causal boundary | **2** | 4 | 4 | 50.0% | no |" in rendered
    assert "| K=4 preprocess bitwise replay | **3** | 9 | 9 | 33.3% | no |" in rendered
    assert ("| K=4 preprocess within fixed material floor | **9** | 9 | 9 | 100.0% | no |") in rendered
    assert "K=1 strict progress on the unchanged denominator: **20 → 21 → 22 → 23 → 25 → 26 → 27 → 28**." in rendered
    assert (
        "K=1 continuation-initializer paired progress on the unchanged denominator: **3/21 stock → 3/21 patched (+0)**."
    ) in rendered
    assert (
        "K=1 sampling-perturbation geometry on the unchanged denominator: **3/5 stock → 3/5 treatment (+0)**."
    ) in rendered
    assert (
        "K=1 sampling-perturbation score/map gates on the unchanged denominator: **3/21 stock → 0/21 treatment (-3)**."
    ) in rendered
    assert (
        "K=1 sampling-roundtrip geometry on the unchanged denominator: **3/5 stock → 5/5 treatment (+2)**." in rendered
    )
    assert (
        "K=1 sampling-roundtrip score/map gates on the unchanged denominator: **3/21 stock → 17/21 treatment (+14)**."
    ) in rendered
    assert (
        "K=1 normalization-roundtrip preprocessing on the unchanged denominator: "
        "**0/2 sampling-only → 2/2 treatment (+2)**."
    ) in rendered
    assert (
        "K=1 normalization-roundtrip geometry on the unchanged denominator: **5/5 sampling-only → 5/5 treatment (+0)**."
    ) in rendered
    assert (
        "K=1 normalization-roundtrip score/map gates on the unchanged denominator: "
        "**17/21 sampling-only → 17/21 treatment (+0)**."
    ) in rendered
    assert (
        "K=1 deterministic-mask preprocessing on the unchanged denominator: "
        "**0/3 normalization-only → 3/3 treatment (+3)**."
    ) in rendered
    assert (
        "K=1 deterministic-mask geometry on the unchanged denominator: **5/5 normalization-only → 5/5 treatment (+0)**."
    ) in rendered
    assert (
        "K=1 deterministic-mask score/map gates on the unchanged denominator: "
        "**17/21 normalization-only → 17/21 treatment (+0)**."
    ) in rendered
    assert "Remaining K=1 strict cases: k1-04, k1-05, k1-07, k1-10, k1-22, k1-26." in rendered
    assert "Remaining K=1 topology cases: k1-07, k1-22." in rendered
    assert ("Remaining K=1 serialized-restart causal cases: iteration1-restart-score-stack-0035") in rendered
    assert ("Remaining K=1 continuation-initializer patched cases: continuation-init-score-stack-0035") in rendered
    assert (
        "Remaining K=4 direct iterations: 10 (1/4 failed), 11 (4/4 failed), "
        "12 (2/4 failed), 13 (4/4 failed), 14 (4/4 failed), 15 (4/4 failed)." in rendered
    )
    assert (
        "Remaining K=4 direct classes: k4-it10-class2, k4-it11-class1, "
        "k4-it11-class2, k4-it11-class3, k4-it11-class4, k4-it12-class2, "
        "k4-it12-class3, k4-it13-class1, k4-it13-class2, k4-it13-class3, "
        "k4-it13-class4, k4-it14-class1, k4-it14-class2, k4-it14-class3, "
        "k4-it14-class4, k4-it15-class1, k4-it15-class2, k4-it15-class3, "
        "k4-it15-class4." in rendered
    )
    assert "Remaining K=4 causal cases: global-raw-diff2, global-combined-score." in rendered
    assert "correlation is not used" in rendered


@pytest.mark.unit
def test_renders_completed_gap_lists_without_empty_punctuation() -> None:
    progress = deepcopy(build_progress())
    for key in progress["remaining"]:
        progress["remaining"][key] = []

    rendered = render_markdown(progress)

    assert "Remaining K=1 strict cases: none." in rendered
    assert "Remaining K=1 topology cases: none." in rendered
    assert "Remaining K=1 serialized-restart causal cases: none." in rendered
    assert "Remaining K=1 continuation-initializer patched cases: none." in rendered
    assert "Remaining K=4 direct iterations: none." in rendered
    assert "Remaining K=4 direct classes: none." in rendered
    assert "Remaining K=4 causal cases: none." in rendered
