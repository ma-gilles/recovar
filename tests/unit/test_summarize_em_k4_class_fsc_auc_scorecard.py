from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize_em_k4_class_fsc_auc_scorecard.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_em_k4_class_fsc_auc_scorecard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_mutated(tmp_path: Path, scorecard: dict) -> Path:
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard))
    return path


def _source_trajectory(scorecard: dict) -> dict:
    return {
        "schema": "em_k4_fsc_trajectory_audit_v2",
        "n_classes": 4,
        "numbered_iteration_count": 15,
        "quality_metric_policy": ("shellwise FSC and normalized FSC-AUC only; correlation is not computed"),
        "thresholds": {"per_class_direct_fsc_auc_min": 0.995},
        "numbered_iterations": [
            {
                "relion_iteration": record["iteration"],
                "classes": [
                    {
                        "recovar_class": class_id,
                        "relion_class": class_id,
                        "cross_engine": {"fsc_auc": value},
                    }
                    for class_id, value in enumerate(
                        record["cross_engine_fsc_auc"],
                        start=1,
                    )
                ],
            }
            for record in scorecard["iterations"]
        ],
    }


@pytest.mark.unit
def test_fixed_class_scorecard_is_valid_and_fresh() -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    failures = MODULE.failed_checks(scorecard)
    rendered = MODULE.render_markdown(scorecard)

    assert scorecard["summary"] == {
        "pass": 41,
        "fail": 19,
        "evaluated": 60,
        "iterations_all_classes_passed": 9,
    }
    assert len(failures) == 19
    assert failures[0] == {
        "id": "k4-it10-class2",
        "iteration": 10,
        "class": 2,
        "fsc_auc": 0.9948890936244424,
    }
    assert failures[-1]["id"] == "k4-it15-class4"
    assert rendered.count("[x]") == 41
    assert rendered.count("[ ]") == 19
    assert MODULE.DEFAULT_MARKDOWN.read_text() == rendered


@pytest.mark.unit
def test_rejects_changed_scorecard_bytes_without_new_version(tmp_path: Path) -> None:
    path = tmp_path / "scorecard.json"
    path.write_text(MODULE.DEFAULT_SCORECARD.read_text() + "\n")

    with pytest.raises(
        ValueError,
        match="scorecard bytes changed without a suite-version change",
    ):
        MODULE.load_and_validate(path)


@pytest.mark.unit
def test_rejects_silently_changed_denominator(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["frozen_denominator"] = 61

    with pytest.raises(ValueError, match="frozen denominator changed"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_iteration_order_drift(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["iterations"][0], scorecard["iterations"][1] = (
        scorecard["iterations"][1],
        scorecard["iterations"][0],
    )

    with pytest.raises(ValueError, match="fixed iteration identity/order changed"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_changed_class_result(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["iterations"][9]["cross_engine_fsc_auc"][1] = 0.995

    with pytest.raises(ValueError, match="per-iteration class results changed"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_summary_that_does_not_replay_cells(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["summary"]["pass"] = 42

    with pytest.raises(
        ValueError,
        match="recorded summary does not match fixed class checks",
    ):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_changed_source_trajectory_digest(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["source_trajectory"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source trajectory SHA-256 changed"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_invalid_k_class_finalization_policy(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["forced_final_all_data_after_nonconvergence"] = True

    with pytest.raises(ValueError, match="invalid forced final all-data is enabled"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_rejects_changed_source_commit(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    scorecard["source_commit"] = "0" * 40

    with pytest.raises(ValueError, match="source commit changed"):
        MODULE.load_and_validate(
            _write_mutated(tmp_path, scorecard),
            enforce_pinned_bytes=False,
        )


@pytest.mark.unit
def test_replays_all_cells_against_source_trajectory(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    source = _source_trajectory(scorecard)
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(source))

    replayed = MODULE.verify_source_trajectory(
        scorecard,
        path,
        enforce_pinned_bytes=False,
    )

    assert replayed["numbered_iteration_count"] == 15


@pytest.mark.unit
def test_rejects_cell_value_that_differs_from_source(tmp_path: Path) -> None:
    scorecard = MODULE.load_and_validate(MODULE.DEFAULT_SCORECARD)
    source = _source_trajectory(scorecard)
    source["numbered_iterations"][9]["classes"][1]["cross_engine"]["fsc_auc"] += 1e-6
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(source))

    with pytest.raises(
        ValueError,
        match="iteration 10: checked FSC-AUC values differ from source",
    ):
        MODULE.verify_source_trajectory(
            scorecard,
            path,
            enforce_pinned_bytes=False,
        )
