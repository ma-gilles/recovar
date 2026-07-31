from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.report_em_parity_progress import build_progress, render_markdown


@pytest.mark.unit
def test_reports_all_fixed_em_parity_panels() -> None:
    progress = build_progress()

    assert progress["schema"] == "recovar.em_parity_progress.v2"
    assert progress["scorecard_change_admissible"] is False
    assert progress["k1_strict_history"] == [20, 21, 22, 23, 25, 26, 27, 28]
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
        ("k4_direct", 41, 60, 60, 68.3, True),
        ("k4_all_class", 9, 15, 15, 60.0, True),
        ("k4_causal", 2, 4, 4, 50.0, False),
    ]
    assert set(progress["inputs"]) == {
        "k1_scorecard",
        "k1_fixture_manifest",
        "k4_trajectory_snapshot",
        "k4_causal_scorecard",
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
        "k4_direct_failures_by_iteration": [
            {"iteration": 10, "passed": 3, "failed": 1},
            {"iteration": 11, "passed": 0, "failed": 4},
            {"iteration": 12, "passed": 2, "failed": 2},
            {"iteration": 13, "passed": 0, "failed": 4},
            {"iteration": 14, "passed": 0, "failed": 4},
            {"iteration": 15, "passed": 0, "failed": 4},
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
    assert "| K=4 direct per-class FSC-AUC | **41** | 60 | 60 | 68.3% | yes |" in rendered
    assert "| K=4 exact-device causal boundary | **2** | 4 | 4 | 50.0% | no |" in rendered
    assert "K=1 strict progress on the unchanged denominator: **20 → 21 → 22 → 23 → 25 → 26 → 27 → 28**." in rendered
    assert "Remaining K=1 strict cases: k1-04, k1-05, k1-07, k1-10, k1-22, k1-26." in rendered
    assert "Remaining K=1 topology cases: k1-07, k1-22." in rendered
    assert (
        "Remaining K=4 direct iterations: 10 (1/4 failed), 11 (4/4 failed), "
        "12 (2/4 failed), 13 (4/4 failed), 14 (4/4 failed), 15 (4/4 failed)." in rendered
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
    assert "Remaining K=4 direct iterations: none." in rendered
    assert "Remaining K=4 causal cases: none." in rendered
