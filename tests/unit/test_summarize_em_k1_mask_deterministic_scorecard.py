from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import summarize_em_k1_mask_deterministic_scorecard as scorecard


@pytest.mark.unit
def test_validates_fixed_mask_deterministic_panels() -> None:
    report = scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD)

    assert report["preprocess_summary"] == {
        "baseline_pass": 0,
        "treatment_pass": 3,
        "evaluated": 3,
        "denominator": 3,
        "paired_gain": 3,
    }
    assert report["geometry_summary"] == {
        "baseline_pass": 5,
        "treatment_pass": 5,
        "evaluated": 5,
        "denominator": 5,
        "paired_gain": 0,
    }
    assert report["score_map_summary"] == {
        "baseline_pass": 17,
        "treatment_pass": 17,
        "evaluated": 21,
        "denominator": 21,
        "paired_gain": 0,
    }


@pytest.mark.unit
def test_fails_closed_when_a_fixed_result_changes(tmp_path: Path) -> None:
    report = deepcopy(scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD))
    report["preprocess_cases"][0]["treatment_result"] = "fail"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="fixed result pair changed"):
        scorecard.load_and_validate(path)


@pytest.mark.unit
def test_renders_checked_case_tables() -> None:
    rendered = scorecard.render_markdown(scorecard.load_and_validate(scorecard.DEFAULT_SCORECARD))

    assert "| Masked-preprocessing exactness | 0/3 | 3/3 | +3 |" in rendered
    assert "| Geometry identity | 5/5 | 5/5 | +0 |" in rendered
    assert "| Score/map gates | 17/21 | 17/21 | +0 |" in rendered
    assert "| [x] | `mask-deterministic-preprocess-background` | fail | pass | improved |" in rendered
    assert "| [ ] | `mask-deterministic-map-gt-merged` | fail | fail | unchanged-fail |" in rendered
