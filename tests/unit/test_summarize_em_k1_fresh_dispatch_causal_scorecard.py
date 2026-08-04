from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k1_fresh_dispatch_causal_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k1_fresh_dispatch_causal_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_fresh_dispatch_causal_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 2
    assert scorecard["summary"] == {
        "evaluations_complete": 2,
        "standalone_rescues": 0,
        "not_supported": 2,
        "evaluated": 2,
    }
    assert "Evaluated: **2 / 2**." in rendered
    assert "Standalone rescues: **0 / 2**." in rendered
    assert rendered.count("| [x] |") == 2
    assert all(
        case["dispatch_alignment"]
        == {
            "full_physical_order_exact": True,
            "expected_accuracy_identity_order_exact": True,
            "expected_accuracy_runtime_ctf_rows_exact": True,
            "physical_vs_internal_execution_equivalence_established": False,
            "production_output_restoration_accepted": False,
        }
        for case in scorecard["cases"]
    )
    assert "runtime float64 expected-accuracy CTF rows" in rendered
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_dispatch_treatment_promoted_without_evidence(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["standalone_rescue_result"] = "pass"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="causal or policy result changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_hidden_producer_state_change(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["producer"]["state"] = "COMPLETED"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="producer identity changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][1]["evidence"]["primary_report"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
