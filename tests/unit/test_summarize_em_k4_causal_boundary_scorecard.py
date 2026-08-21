from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT / "scripts" / "summarize_em_k4_causal_boundary_scorecard.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k4_causal_boundary_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_causal_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 4
    assert scorecard["summary"] == {"pass": 2, "fail": 2, "evaluated": 4}
    assert "Fixed causal score: **2 / 4 passing** (4 / 4 evaluated)." in rendered
    assert rendered.count("| [x] |") == 2
    assert rendered.count("| [ ] |") == 2
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_silently_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 5
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="frozen denominator changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_case_order_or_identity_drift(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0], scorecard["cases"][1] = (
        scorecard["cases"][1],
        scorecard["cases"][0],
    )
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="fixed case identity/order changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_summary_that_does_not_replay_cases(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["summary"]["pass"] = 3
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(
        ValueError,
        match="recorded summary does not match fixed cases",
    ):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_checkmark_that_disagrees_with_result(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["checked"] = False
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="checkmark disagrees with result"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_invalid_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["completion_report"]["sha256"] = "not-a-digest"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="invalid SHA-256"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_well_formed_but_changed_evidence_digest(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["completion_report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
