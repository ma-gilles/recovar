from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_restart_particle_order_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_restart_particle_order_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_restart_particle_order_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator_per_arm"] == 20
    assert scorecard["summary"]["stock"] == {"pass": 17, "evaluated": 20}
    assert scorecard["summary"]["restored_combined"] == {"pass": 40, "evaluated": 40}
    assert "Stock restart: **17 / 20**." in rendered
    assert "Restored repeats: **40 / 40**." in rendered
    assert rendered.count("| [x] |") == 20
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_restored_result(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["restored_a_result"] = "fail"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="fixed particle-order result changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_dispatch_contract(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["acceptance_contract"]["dispatch_controls_passed"] = 3
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
