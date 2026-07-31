from __future__ import annotations

import pytest

from scripts.report_em_parity_progress import build_progress, render_markdown


@pytest.mark.unit
def test_reports_all_fixed_em_parity_panels() -> None:
    progress = build_progress()

    assert progress["schema"] == "recovar.em_parity_progress.v1"
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


@pytest.mark.unit
def test_renders_pr_ready_fixed_metric_table() -> None:
    rendered = render_markdown(build_progress())

    assert "| K=1 strict FSC/FSC-AUC | **28** | 34 | 34 | 82.4% | yes |" in rendered
    assert "| K=4 direct per-class FSC-AUC | **41** | 60 | 60 | 68.3% | yes |" in rendered
    assert "| K=4 exact-device causal boundary | **2** | 4 | 4 | 50.0% | no |" in rendered
    assert "K=1 strict progress on the unchanged denominator: **20 → 21 → 22 → 23 → 25 → 26 → 27 → 28**." in rendered
    assert "correlation is not used" in rendered
