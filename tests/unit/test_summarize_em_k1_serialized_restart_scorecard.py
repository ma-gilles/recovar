from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_serialized_restart_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_serialized_restart_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_serialized_restart_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 42
    assert scorecard["summary"]["evaluated"] == 42
    assert "Fixed causal score:" in rendered
    assert rendered.count("| [x] |") == scorecard["summary"]["pass"]
    assert rendered.count("| [ ] |") == scorecard["summary"]["fail"]
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_silently_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 43
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="frozen denominator changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_group_denominator_drift(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["group_denominators"]["score"] = 29
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="group denominators changed"):
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
def test_rejects_summary_that_does_not_replay_cases(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["summary"]["pass"] += 1
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(
        ValueError,
        match="recorded summary does not match fixed cases",
    ):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_grouped_summary_that_does_not_replay_cases(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["grouped_summary"]["score"]["pass"] += 1
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(
        ValueError,
        match="recorded grouped summary does not replay cases",
    ):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_checkmark_that_disagrees_with_result(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["checked"] = not scorecard["cases"][0]["checked"]
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="checkmark disagrees with result"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_case_group_drift(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["group"] = "map-parity"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="fixed group changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_correlation_metric(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["correlation_used"] = True
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="correlation is forbidden"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_invalid_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    first = next(iter(scorecard["evidence"]))
    scorecard["evidence"][first]["sha256"] = "not-a-digest"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="invalid SHA-256"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_well_formed_but_changed_evidence_digest(
    tmp_path: Path,
) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    first = next(iter(scorecard["evidence"]))
    scorecard["evidence"][first]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
