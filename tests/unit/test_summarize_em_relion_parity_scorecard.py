import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_relion_parity_scorecard.py"
SPEC = importlib.util.spec_from_file_location("summarize_em_relion_parity_scorecard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_frozen_v1_scorecard_is_valid_and_renders_fixed_denominator():
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["frozen_denominator"] == 34
    assert scorecard["frozen_case_definitions_sha256"] == MODULE.frozen_case_definitions_sha256(
        scorecard["cases"]
    )
    assert scorecard["current_snapshot"]["counts"] == {"pass": 21, "fail": 13, "not_run": 0}
    assert "K=1 fixed-suite score: 21 / 34 passing" in rendered
    assert rendered.count("| [x] |") == 21
    assert rendered.count("| [ ] |") == 13
    assert "| `strict-k1-v1-old-head-20260721`" in rendered
    assert "| 20 | 12 | 2 |" in rendered
    assert "| `strict-k1-v3-20260721`" in rendered
    assert "| 21 | 13 | 0 |" in rendered


@pytest.mark.unit
def test_validation_rejects_a_silently_changed_denominator(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"].pop()
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="frozen_denominator"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_validation_rejects_history_that_moves_the_fixed_denominator(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["history"][0]["counts"]["not_run"] = 1
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="history counts do not preserve frozen denominator"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_validation_rejects_a_silently_changed_case_definition(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["definition"]["n_images"] = "99999"
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="frozen case definitions changed"):
        MODULE.load_and_validate(path)
