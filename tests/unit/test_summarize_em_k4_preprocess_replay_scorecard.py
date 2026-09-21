from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k4_preprocess_replay_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k4_preprocess_replay_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_preprocess_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 9
    assert scorecard["summary"] == {
        "bitwise_equal": 3,
        "within_material_floor": 9,
        "evaluated": 9,
    }
    assert "Bitwise exact: **3 / 9**. Within fixed material floor: **9 / 9**." in rendered
    assert rendered.count("| [x] |") == 9
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 10
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="frozen denominator changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_material_result_that_disagrees_with_metric(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][3]["within_material_floor"] = False
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="material-floor result disagrees"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_bitwise_result(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][3]["bitwise_equal"] = True
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="bitwise result changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
