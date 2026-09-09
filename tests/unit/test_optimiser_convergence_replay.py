"""Numbered replay controls and the unnumbered final-pass boundary."""

import logging
from dataclasses import asdict

import pytest

from recovar.em.dense_single_volume import relion_replay
from recovar.em.dense_single_volume.helpers.convergence import RefinementState

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
    state = RefinementState(acc_rot=1.5, acc_trans=2.5, current_resolution=8.0)
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
