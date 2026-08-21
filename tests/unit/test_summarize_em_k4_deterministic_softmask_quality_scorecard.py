from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "summarize_em_k4_deterministic_softmask_quality_scorecard.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k4_deterministic_softmask_quality_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_quality_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 7
    assert scorecard["summary"] == {"pass": 7, "evaluated": 7}
    assert scorecard["arms"]["control"]["direct_fsc_auc"] == {
        "passed": 41,
        "evaluated": 60,
    }
    assert scorecard["arms"]["treatment"]["direct_fsc_auc"] == {
        "passed": 41,
        "evaluated": 60,
    }
    assert "Quality acceptance: **7 / 7**." in rendered
    assert "41/60 | 41/60 | 0" in rendered
    assert rendered.count("| [x] |") == 7
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_integration_without_quality_acceptance(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["acceptance_contract"]["quality_accepted"] = False
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="integration acceptance contract changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_fixed_quality_count(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["arms"]["treatment"]["direct_fsc_auc"]["passed"] = 42
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="fixed arm quality counts changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["quality_report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
