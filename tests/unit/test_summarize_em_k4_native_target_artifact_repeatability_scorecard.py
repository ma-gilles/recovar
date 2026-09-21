from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.summarize_em_k4_native_target_artifact_repeatability_scorecard import (
    DEFAULT_SCORECARD,
    load_and_validate,
    render_markdown,
)


@pytest.mark.unit
def test_loads_fixed_target_artifact_scorecard() -> None:
    scorecard = load_and_validate(DEFAULT_SCORECARD)

    assert scorecard["summary"] == {"pass": 32, "evaluated": 32}
    assert scorecard["target_local_artifact_use_allowed"] is True
    assert scorecard["allclass_cross_engine_attribution_allowed"] is False
    assert "Fixed admission gates: **32 / 32**" in render_markdown(scorecard)


@pytest.mark.unit
def test_rejects_changed_gate_result(tmp_path) -> None:
    scorecard = json.loads(DEFAULT_SCORECARD.read_text())
    changed = deepcopy(scorecard)
    changed["cases"][0]["result"] = "fail"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="fixed repeatability result changed"):
        load_and_validate(path)


@pytest.mark.unit
def test_rejects_allclass_promotion(tmp_path) -> None:
    scorecard = json.loads(DEFAULT_SCORECARD.read_text())
    changed = deepcopy(scorecard)
    changed["allclass_cross_engine_attribution_allowed"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="scope or metric policy changed"):
        load_and_validate(path)
