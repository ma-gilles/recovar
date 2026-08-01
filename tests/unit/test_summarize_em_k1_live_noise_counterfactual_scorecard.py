from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_em_k1_live_noise_counterfactual_scorecard as summary


@pytest.mark.unit
def test_fixed_live_noise_counterfactual_scorecard_is_fresh() -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    rendered = summary.render_markdown(scorecard)
    assert scorecard["summary"] == {"pass": 21, "evaluated": 24}
    assert scorecard["science_summary"] == {"pass": 17, "evaluated": 20}
    assert scorecard["provenance_summary"] == {"pass": 4, "evaluated": 4}
    assert rendered.count("| [x] |") == 21
    assert rendered.count("| [ ] |") == 3
    assert summary.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_result(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["cases"][0]["result"] = "fail"
    scorecard["cases"][0]["checked"] = False
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="fixed case identities or results changed"):
        summary.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_denominator(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 25
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="frozen denominator changed"):
        summary.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["evidence"]["primary_report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="primary_report: evidence digest changed"):
        summary.load_and_validate(path)


@pytest.mark.unit
def test_rejects_false_acceptance(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["acceptance_contract"]["accepted"] = True
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="terminal acceptance contract changed"):
        summary.load_and_validate(path)
