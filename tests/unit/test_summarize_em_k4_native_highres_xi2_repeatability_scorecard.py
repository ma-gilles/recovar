from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k4_native_highres_xi2_repeatability_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k4_native_highres_xi2_repeatability_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_fixed_native_highres_xi2_repeatability_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 15
    assert scorecard["summary"] == {"pass": 15, "evaluated": 15}
    assert scorecard["artifact_exactness"]["fine_score"] is True
    assert scorecard["artifact_exactness"]["fine_operand"] is True
    assert scorecard["artifact_exactness"]["bpref"] is True
    assert scorecard["particle_state_telemetry"]["full_particle_state_exact"] is False
    assert scorecard["class_map_fsc_auc"]["threshold"] == 0.999999
    assert "Fixed gates: **15 / 15**." in rendered
    assert rendered.count("| [x] |") == 15
    assert "not joint posterior/BPref/map parity" in rendered
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 16
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="denominator changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_gate_result(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["result"] = "fail"
    scorecard["cases"][0]["checked"] = False
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="artifact_bytes_exact: result changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_overclaimed_particle_state(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["particle_state_telemetry"]["full_particle_state_exact"] = True
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="particle-state telemetry changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_changed_evidence_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["evidence"]["analysis_result"]["sha256"] = "0" * 64
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        MODULE.load_and_validate(path)
