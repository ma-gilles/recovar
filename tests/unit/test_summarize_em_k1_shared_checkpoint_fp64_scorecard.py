from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_em_k1_shared_checkpoint_fp64_scorecard as summary


@pytest.mark.unit
def test_fixed_shared_checkpoint_scorecard_is_fresh() -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    rendered = summary.render_markdown(scorecard)
    assert scorecard["summary"] == {"pass": 16, "evaluated": 34}
    assert rendered.count("| [x] |") == 16
    assert rendered.count("| [ ] |") == 18
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
    scorecard["frozen_denominator"] = 35
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="frozen denominator changed"):
        summary.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["evidence"]["fixed_audit"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="fixed_audit: evidence digest changed"):
        summary.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_case_identity(tmp_path: Path) -> None:
    scorecard = summary.load_and_validate(summary.DEFAULT_SCORECARD)
    scorecard["cases"][0]["id"] = "replacement-case"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    with pytest.raises(ValueError, match="fixed case identities or results changed"):
        summary.load_and_validate(path)
