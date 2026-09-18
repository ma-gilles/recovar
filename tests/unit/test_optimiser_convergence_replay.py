"""Numbered replay controls and the unnumbered final-pass boundary."""

import logging
from dataclasses import asdict

import pytest

from recovar.em.diagnostics import relion_replay
from recovar.em.helpers.convergence import RefinementState

pytestmark = pytest.mark.unit


def _apply(state, metadata, tmp_path, **overrides):
    kwargs = dict(
        metadata=metadata,
        optimiser_star=str(tmp_path / "run_it020_optimiser.star"),
        optimiser_iteration=20,
        replay_dir=str(tmp_path),
        replay_prefix="run",
        sealed_sampling_state=None,
        logger=logging.getLogger(__name__),
    )
    kwargs.update(overrides)
    return relion_replay.apply_optimiser_convergence_replay(state, **kwargs)


def test_numbered_metadata_updates_only_supplied_controls(tmp_path):
    state = RefinementState(acc_rot=1.5, acc_trans=2.5, current_resolution=8.0, fraction_changed=0.375)
    before = asdict(state)
    metadata = {
        "number_iter_without_resolution_gain": "2",
        "number_iter_without_changing_assignments": "3",
        "changes_optimal_orientations": "0.5",
        "changes_optimal_offsets": "0.25",
        "changes_optimal_classes": "4",
        "smallest_changes_orientations": "0.125",
        "smallest_changes_offsets": "0.0625",
        "smallest_changes_classes": "1",
        "has_converged": "0",
    }
    expected = dict(
        before,
        nr_iter_wo_resol_gain=2,
        nr_iter_wo_large_hidden_variable_changes=3,
        nr_iter_wo_assignment_changes=3,
        current_changes_optimal_orientations=0.5,
        current_changes_optimal_offsets_angstrom=0.25,
        current_changes_optimal_classes=4.0,
        smallest_changes_optimal_orientations=0.125,
        smallest_changes_optimal_offsets_angstrom=0.0625,
        smallest_changes_optimal_classes=1.0,
        has_converged=False,
    )
    assert _apply(state, metadata, tmp_path) is None
    assert asdict(state) == expected
    assert metadata["number_iter_without_resolution_gain"] == "2"


@pytest.mark.parametrize("metadata", [{}, {"has_converged": None, "changes_optimal_offsets": None}])
def test_missing_controls_preserve_computed_state(tmp_path, metadata):
    state = RefinementState(nr_iter_wo_resol_gain=7, has_converged=True)
    before = asdict(state)
    _apply(state, metadata, tmp_path)
    assert asdict(state) == before


@pytest.mark.parametrize("boundary", ["final", "next", "missing_sampling", "missing_optimiser", "sealed", "no_replay"])
def test_final_convergence_requires_end_of_numbered_replay(tmp_path, boundary):
    if boundary != "missing_sampling":
        (tmp_path / "run_sampling.star").touch()
    if boundary != "missing_optimiser":
        (tmp_path / "run_optimiser.star").write_text("_rlnHasConverged 1\n")
    if boundary == "next":
        (tmp_path / "run_it021_sampling.star").touch()
    state = RefinementState()
    overrides = {}
    if boundary == "sealed":
        overrides["sealed_sampling_state"] = {}
    if boundary == "no_replay":
        overrides["replay_dir"] = None
    _apply(state, {}, tmp_path, **overrides)
    assert state.has_converged is (boundary == "final")


def test_invalid_numbered_counter_preserves_prior_mutation_order(tmp_path):
    state = RefinementState(nr_iter_wo_assignment_changes=9)
    with pytest.raises(ValueError):
        _apply(
            state,
            {"number_iter_without_resolution_gain": "2", "number_iter_without_changing_assignments": "invalid"},
            tmp_path,
        )
    assert state.nr_iter_wo_resol_gain == 2
    assert state.nr_iter_wo_assignment_changes == 9


def test_final_read_failure_is_logged_after_numbered_controls(tmp_path, monkeypatch, caplog):
    (tmp_path / "run_sampling.star").touch()
    (tmp_path / "run_optimiser.star").touch()

    def failed_read(path):
        raise OSError("test unreadable final optimiser")

    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", failed_read)
    state = RefinementState()
    _apply(state, {"number_iter_without_resolution_gain": 4}, tmp_path)
    assert state.nr_iter_wo_resol_gain == 4
    assert not state.has_converged
    assert "failed to read final optimiser metadata" in caplog.text


def test_numbered_convergence_does_not_read_final_metadata(tmp_path, monkeypatch):
    def forbidden_read(path):
        pytest.fail("already converged numbered state must not read final optimiser")

    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", forbidden_read)
    state = RefinementState()
    _apply(state, {"has_converged": 1}, tmp_path)
    assert state.has_converged


def _read_accuracy(tmp_path, **overrides):
    kwargs = dict(
        replay_dir=str(tmp_path),
        replay_prefix="run",
        init_relion_iteration=3,
        iteration=4,
        sealed_sampling_state=None,
        acc_rot=1.5,
        acc_trans=None,
        convergence_acc_rot=None,
        convergence_acc_trans=None,
        logger=logging.getLogger(__name__),
    )
    kwargs.update(overrides)
    return relion_replay.read_optimiser_accuracy_replay(**kwargs)


def _unchanged(result, **expected):
    values = dict(acc_rot=1.5, acc_trans=None, convergence_acc_rot=None, convergence_acc_trans=None)
    values.update(expected)
    assert (result.acc_rot, result.acc_trans, result.convergence_acc_rot, result.convergence_acc_trans) == (
        values["acc_rot"],
        values["acc_trans"],
        values["convergence_acc_rot"],
        values["convergence_acc_trans"],
    )


@pytest.mark.parametrize("override", [dict(replay_dir=None), dict(sealed_sampling_state={"healpix_order_original": 2})])
def test_accuracy_replay_is_inactive_without_replay_dir_or_with_sealed_state(tmp_path, override):
    result = _read_accuracy(tmp_path, **override)
    assert result.metadata is None and result.optimiser_star is None and result.optimiser_iteration is None
    _unchanged(result)


def test_accuracy_replay_selects_numbered_star_and_tolerates_missing_file(tmp_path, monkeypatch):
    def forbidden_read(path):
        raise AssertionError(f"unexpected read of {path}")

    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", forbidden_read)
    result = _read_accuracy(tmp_path)
    assert result.optimiser_iteration == 8
    assert result.optimiser_star == str(tmp_path / "run_it008_optimiser.star")
    assert result.metadata is None
    _unchanged(result)


def test_accuracy_replay_substitutes_finite_relion_accuracies(tmp_path, monkeypatch, caplog):
    star = tmp_path / "run_it008_optimiser.star"
    star.write_text("data_\n")
    metadata = {"overall_accuracy_rotations": "0.75", "overall_accuracy_translations_angst": "2.5"}
    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", lambda path: metadata)
    with caplog.at_level(logging.INFO, logger=__name__):
        result = _read_accuracy(tmp_path)
    assert result.metadata is metadata
    assert result.optimiser_star == str(star)
    _unchanged(result, acc_rot=0.75, acc_trans=2.5, convergence_acc_rot=0.75, convergence_acc_trans=2.5)
    assert type(result.acc_rot) is float and type(result.acc_trans) is float
    assert "Replay override: optimiser accuracy <- " in caplog.text
    assert "acc_rot=0.750 deg, acc_trans=2.500" in caplog.text


def test_accuracy_replay_ignores_missing_or_nonfinite_fields(tmp_path, monkeypatch, caplog):
    (tmp_path / "run_it008_optimiser.star").write_text("data_\n")
    metadata = {"overall_accuracy_rotations": "inf", "overall_accuracy_translations_angst": None}
    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", lambda path: metadata)
    with caplog.at_level(logging.INFO, logger=__name__):
        result = _read_accuracy(tmp_path)
    assert result.metadata is metadata
    _unchanged(result)
    assert "acc_rot=1.500 deg, acc_trans=unset" in caplog.text


def test_accuracy_replay_read_failure_warns_and_keeps_inputs(tmp_path, monkeypatch, caplog):
    (tmp_path / "run_it008_optimiser.star").write_text("data_\n")

    def failed_read(path):
        raise OSError("truncated")

    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", failed_read)
    with caplog.at_level(logging.WARNING, logger=__name__):
        result = _read_accuracy(tmp_path)
    assert result.metadata is None
    assert result.optimiser_iteration == 8
    _unchanged(result)
    assert "failed to read optimiser metadata" in caplog.text


def test_accuracy_replay_parse_failure_keeps_metadata_and_partial_override(tmp_path, monkeypatch, caplog):
    """A non-numeric translation accuracy fails after the rotation override, as the inline code did."""

    (tmp_path / "run_it008_optimiser.star").write_text("data_\n")
    metadata = {"overall_accuracy_rotations": "0.5", "overall_accuracy_translations_angst": "n/a"}
    monkeypatch.setattr(relion_replay, "read_relion_optimiser_metadata", lambda path: metadata)
    with caplog.at_level(logging.WARNING, logger=__name__):
        result = _read_accuracy(tmp_path)
    assert result.metadata is metadata
    _unchanged(result, acc_rot=0.5, convergence_acc_rot=0.5)
    assert "failed to read optimiser metadata" in caplog.text
