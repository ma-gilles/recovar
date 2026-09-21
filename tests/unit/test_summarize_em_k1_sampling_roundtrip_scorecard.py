from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import summarize_em_k1_sampling_roundtrip_scorecard as scorecard


@pytest.mark.unit
def test_validates_fixed_sampling_roundtrip_panels() -> None:
    report = scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD)

    assert report["geometry_summary"] == {
        "baseline_pass": 3,
        "treatment_pass": 5,
        "evaluated": 5,
        "denominator": 5,
        "paired_gain": 2,
    }
    assert report["score_map_summary"] == {
        "baseline_pass": 3,
        "treatment_pass": 17,
        "evaluated": 21,
        "denominator": 21,
        "paired_gain": 14,
    }
    assert sum(case["checked"] for case in report["geometry_cases"]) == 5
    assert sum(case["checked"] for case in report["score_map_cases"]) == 17


@pytest.mark.unit
def test_fails_closed_when_a_fixed_result_changes(tmp_path: Path) -> None:
    report = deepcopy(scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD))
    report["score_map_cases"][0]["treatment_result"] = "fail"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="fixed result pair changed"):
        scorecard.load_and_validate(path)


@pytest.mark.unit
def test_renders_checked_case_tables() -> None:
    rendered = scorecard.render_markdown(scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD))

    assert "| Geometry identity | 3/5 | 5/5 | +2 |" in rendered
    assert "| Score/map gates | 3/21 | 17/21 | +14 |" in rendered
    assert "| [x] | `sampling-roundtrip-geometry-euler-matrices` | fail | pass | improved |" in rendered
    assert "| [ ] | `sampling-roundtrip-map-gt-merged` | pass | fail | regressed |" in rendered
