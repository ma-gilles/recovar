from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.summarize_em_k4_allclass_recovar_repeatability_scorecard import (
    DEFAULT_SCORECARD,
    load_and_validate,
    render_markdown,
)


@pytest.mark.unit
def test_loads_fixed_repeatability_scorecard() -> None:
    scorecard = load_and_validate(DEFAULT_SCORECARD)

    assert scorecard["summary"] == {"pass": 9, "evaluated": 9}
    assert scorecard["classification"] == "all_observed_pass2_fields_exact"
    assert scorecard["stable_recovar_boundary_established"] is True
    assert scorecard["cross_engine_parity_established"] is False
    assert "Exact repeatability gates: **9 / 9**" in render_markdown(scorecard)


@pytest.mark.unit
def test_rejects_changed_gate_result(tmp_path) -> None:
    scorecard = json.loads(DEFAULT_SCORECARD.read_text())
    changed = deepcopy(scorecard)
    changed["cases"][-1]["result"] = "fail"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="fixed repeatability result changed"):
        load_and_validate(path)


@pytest.mark.unit
def test_rejects_cross_engine_promotion(tmp_path) -> None:
    scorecard = json.loads(DEFAULT_SCORECARD.read_text())
    changed = deepcopy(scorecard)
    changed["cross_engine_parity_established"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="scope or metric policy changed"):
        load_and_validate(path)
