import hashlib
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
def test_check_preserves_marked_post_snapshot_diagnostics():
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    fixture_manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)
    rendered = MODULE.render_markdown(
        scorecard,
        fixture_manifest,
        MODULE.sha256_file(MODULE.DEFAULT_FIXTURE_MANIFEST),
    )
    manual = "\n".join(
        (
            MODULE.MANUAL_DIAGNOSTICS_BEGIN,
            "## Post-snapshot fixed-fixture intervention diagnostics",
            "",
            "Pending evidence does not mutate the frozen score.",
            MODULE.MANUAL_DIAGNOSTICS_END,
        )
    )
    checked_text = rendered.replace(
        MODULE.MANUAL_DIAGNOSTICS_ANCHOR,
        f"\n\n{manual}{MODULE.MANUAL_DIAGNOSTICS_ANCHOR}",
        1,
    )

    assert MODULE.preserve_manual_diagnostics(rendered, checked_text) == checked_text


@pytest.mark.unit
def test_check_rejects_unpaired_manual_diagnostics_marker():
    with pytest.raises(ValueError, match="matched pair"):
        MODULE.preserve_manual_diagnostics(
            "generated",
            MODULE.MANUAL_DIAGNOSTICS_BEGIN,
        )


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


@pytest.mark.unit
def test_proposal_fixture_validation_rehashes_materialized_bytes(tmp_path):
    case = {"id": "k1-04", "name": "high_noise_100k_g256_white_noise3_bf80"}
    case_root = tmp_path / "cases" / "4_high_noise_100k_g256_white_noise3_bf80"
    data_root = case_root / "data"
    data_root.mkdir(parents=True)
    fixture_path = data_root / "particles.test.mrcs"
    fixture_path.write_bytes(b"original-bytes")
    expected_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    fixture_manifest_sha256 = "a" * 64
    expected_row = {
        "name": fixture_path.name,
        "size": fixture_path.stat().st_size,
        "sha256": expected_sha256,
    }
    fixture_manifest = {
        "cases": [
            {
                "id": case["id"],
                "name": case["name"],
                "files": [expected_row],
            }
        ]
    }
    materialization = {
        "schema": "recovar.em_k1_fixture_materialization.v1",
        "case_id": case["id"],
        "case_name": case["name"],
        "manifest_sha256": fixture_manifest_sha256,
        "files": [expected_row],
    }
    (data_root / "fixture_materialization.json").write_text(json.dumps(materialization))

    MODULE._validate_materialized_fixture(
        case,
        case_root,
        fixture_manifest,
        fixture_manifest_sha256,
    )
    fixture_path.write_bytes(b"mutated--bytes")

    with pytest.raises(ValueError, match="fixture SHA-256 changed"):
        MODULE._validate_materialized_fixture(
            case,
            case_root,
            fixture_manifest,
            fixture_manifest_sha256,
        )


@pytest.mark.unit
def test_proposal_evidence_parser_requires_absolute_fixed_suite_identity():
    parsed = MODULE.parse_proposal_evidence("k1-04|/scratch/example/cases/4_high_noise|11563827|11563842")

    assert parsed == MODULE.ProposalEvidence(
        "k1-04",
        Path("/scratch/example/cases/4_high_noise"),
        "11563827",
        "11563842",
    )
    with pytest.raises(MODULE.argparse.ArgumentTypeError, match="must be absolute"):
        MODULE.parse_proposal_evidence("k1-04|relative/path|11563827|11563842")
    with pytest.raises(MODULE.argparse.ArgumentTypeError, match="only digits"):
        MODULE.parse_proposal_evidence("k1-04|/scratch/example|science|11563842")


@pytest.mark.unit
def test_superseding_ledger_proposal_preserves_denominator_and_topology_counts(
    tmp_path,
    monkeypatch,
):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)
    previous = tmp_path / "previous.json"
    previous_schema = "em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v6"
    previous.write_text(json.dumps({"schema": previous_schema}) + "\n")
    scorecard["current_snapshot"]["source_ledger"] = {
        "schema": previous_schema,
        "generated_utc": "2026-07-24T01:04:11+00:00",
        "sha256": MODULE.sha256_file(previous),
    }
    evidence = MODULE.ProposalEvidence(
        "k1-04",
        Path("/unused/by-mocked-builder"),
        "11563827",
        "11563842",
    )
    update = {
        "case_id": "k1-04",
        "case_name": "high_noise_100k_g256_white_noise3_bf80",
        "result": "pass",
        "intermediate_result": "pass",
    }
    monkeypatch.setattr(MODULE, "build_proposal_update", lambda *args: update)

    ledger = MODULE.build_superseding_ledger(
        scorecard,
        manifest,
        MODULE.sha256_file(MODULE.DEFAULT_FIXTURE_MANIFEST),
        previous,
        "em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v7",
        "2026-07-24T13:00:00+00:00",
        [evidence],
        "Case k1-04 passed immutable strict evidence.",
    )

    assert ledger["counts"]["strict"] == {"pass": 26, "fail": 8, "not_run": 0}
    assert ledger["counts"]["topology"] == {"pass": 31, "fail": 3, "not_run": 0}
    assert sum(ledger["counts"]["strict"].values()) == 34
    assert ledger["updates"] == [update]
    assert ledger["supersedes"]["sha256"] == MODULE.sha256_file(previous)


@pytest.mark.unit
def test_superseding_ledger_rejects_unpinned_previous_evidence(tmp_path):
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    manifest = MODULE.load_and_validate_fixture_manifest(MODULE.DEFAULT_FIXTURE_MANIFEST, scorecard)
    previous = tmp_path / "previous.json"
    previous.write_text('{"schema":"wrong"}\n')

    with pytest.raises(ValueError, match="SHA-256 differs"):
        MODULE.build_superseding_ledger(
            scorecard,
            manifest,
            MODULE.sha256_file(MODULE.DEFAULT_FIXTURE_MANIFEST),
            previous,
            "em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v7",
            "2026-07-24T13:00:00+00:00",
            [
                MODULE.ProposalEvidence(
                    "k1-04",
                    Path("/unused"),
                    "11563827",
                    "11563842",
                )
            ],
            "This must not be emitted.",
        )
