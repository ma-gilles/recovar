from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_continuation_initializer_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_continuation_initializer_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_continuation_initializer_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 21
    assert scorecard["baseline_summary"]["pass"] == 3
    assert scorecard["treatment_summary"]["pass"] == 3
    assert scorecard["paired_gain"] == 0
    assert scorecard["two_arm_summary"] == {
        "pass": 6,
        "fail": 36,
        "evaluated": 42,
        "denominator": 42,
    }
    assert scorecard["transition_summary"] == {
        "improved": 0,
        "retained": 3,
        "regressed": 0,
        "unchanged-fail": 18,
    }
    assert "stock **3 / 21** → patched **3 / 21**" in rendered
    assert rendered.count("| [x] |") == 3
    assert rendered.count("| [ ] |") == 18
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


def _write_scorecard(tmp_path: Path, scorecard: dict) -> Path:
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    return path


@pytest.mark.unit
def test_rejects_silently_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 22

    with pytest.raises(ValueError, match="frozen denominator changed"):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_case_order_or_identity_drift(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0], scorecard["cases"][1] = (
        scorecard["cases"][1],
        scorecard["cases"][0],
    )

    with pytest.raises(ValueError, match="fixed case identity/order changed"):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_checkmark_that_disagrees_with_treatment(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["checked"] = True

    with pytest.raises(
        ValueError,
        match="checkmark disagrees with treatment result",
    ):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_transition_that_disagrees_with_arms(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["transition"] = "retained"

    with pytest.raises(
        ValueError,
        match="transition disagrees with arm results",
    ):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_summary_that_does_not_replay_cases(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["treatment_summary"]["pass"] -= 1

    with pytest.raises(
        ValueError,
        match="treatment summary does not replay cases",
    ):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_correlation_metric(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["correlation_used"] = True

    with pytest.raises(ValueError, match="correlation is forbidden"):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))


@pytest.mark.unit
def test_rejects_well_formed_but_changed_evidence_digest(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    first = next(iter(scorecard["evidence"]))
    scorecard["evidence"][first]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(_write_scorecard(tmp_path, scorecard))
