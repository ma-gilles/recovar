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
    fixture_manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)
    rendered = MODULE.render_markdown(
        scorecard,
        fixture_manifest,
        MODULE.sha256_file(MODULE.DEFAULT_FIXTURE_MANIFEST),
    )

    assert scorecard["frozen_denominator"] == 34
    assert scorecard["frozen_case_definitions_sha256"] == MODULE.frozen_case_definitions_sha256(scorecard["cases"])
    assert scorecard["current_snapshot"]["counts"] == {"pass": 25, "fail": 9, "not_run": 0}
    assert "K=1 fixed-suite score: 25 / 34 passing" in rendered
    assert "Progress: +5 passing cases since the first frozen snapshot; +2 since the previous snapshot." in rendered
    assert rendered.count("| [x] |") == 25
    assert rendered.count("| [ ] |") == 9
    assert "34 cases (470,170,958,467 bytes)" in rendered
    assert "| `strict-k1-v1-old-head-20260721`" in rendered
    assert "| 20 | — | 12 | 2 |" in rendered
    assert "| `strict-k1-v3-20260721`" in rendered
    assert "| 21 | +1 | 13 | 0 |" in rendered
    assert "| `strict-k1-v4-20260722`" in rendered
    assert "| 22 | +1 | 12 | 0 |" in rendered
    assert "| `strict-k1-v5-20260722`" in rendered
    assert "| 23 | +1 | 11 | 0 |" in rendered
    assert "| `strict-k1-v6-20260724`" in rendered
    assert "| 25 | +2 | 9 | 0 |" in rendered
    assert "Non-scoring regenerated-data diagnostics" in rendered
    assert "| `k1-23` | pass | pass | 0.997483478 |" in rendered


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


@pytest.mark.unit
def test_validation_rejects_changed_definition_even_with_recomputed_digest(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["cases"][0]["definition"]["n_images"] = "99999"
    scorecard["frozen_case_definitions_sha256"] = MODULE.frozen_case_definitions_sha256(scorecard["cases"])
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))

    with pytest.raises(ValueError, match="v1 frozen case-definition digest changed"):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_fixture_manifest_is_pinned_to_scorecard_identity():
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)

    assert len(manifest["cases"]) == 34
    assert [case["id"] for case in manifest["cases"]] == [f"k1-{index:02d}" for index in range(1, 35)]
    assert manifest["frozen_case_definitions_sha256"] == scorecard["frozen_case_definitions_sha256"]


@pytest.mark.unit
def test_fixture_validation_rejects_a_changed_file_digest(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)
    manifest["cases"][0]["files"][0]["sha256"] = "not-a-digest"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="invalid SHA-256"):
        MODULE.load_and_validate_fixture_manifest(path, scorecard)
