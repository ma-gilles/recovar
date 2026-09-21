from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_exact_initial_noise_counterfactual_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_exact_initial_noise_counterfactual_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_exact_noise_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 24
    assert scorecard["summary"] == {"pass": 4, "evaluated": 24}
    assert scorecard["science_summary"] == {"pass": 0, "evaluated": 20}
    assert scorecard["provenance_summary"] == {"pass": 4, "evaluated": 4}
    assert "Accepted gates: **4 / 24**." in rendered
    assert rendered.count("| [x] |") == 4
    assert rendered.count("| [ ] |") == 20
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_science_result(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["result"] = "pass"
    scorecard["cases"][0]["checked"] = True
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="fixed exact-noise result changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_terminal_contract(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["acceptance_contract"]["science_owner_state"] = "COMPLETED"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="terminal acceptance contract changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["primary_report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
