from __future__ import annotations

import pytest

from scripts.audit_vdam_candidate_state_envelope import (
    CandidateStateEnvelopeError,
    classify_schedule_distribution_envelope,
)

pytestmark = pytest.mark.unit


def _state(**updates):
    state = {
        "healpix_order": 3,
        "n_translations": 84,
        "current_size": 36,
        "orientational_prior_mode": 0,
        "sampling_updated": False,
        "offset_range_angstrom": 8.118967,
        "offset_step_angstrom": 3.0,
        "random_perturbation": 0.235074,
        "sampling_acc_rot": 8.065,
        "sampling_acc_trans_angstrom": 5.39325,
        "current_changes_optimal_offsets_angstrom": 1.208444,
        "current_resolution_angstrom": 68.0,
    }
    state.update(updates)
    return state


def _candidate(*, estimated=True, **updates):
    return {
        **_state(**updates),
        "sampling_accuracy_estimated": estimated,
    }


def _matrix(candidates, natives, *, iteration=30):
    return [
        [
            {"iteration": iteration, "candidate": candidate, "native": native}
            for native in natives
        ]
        for candidate in candidates
    ]


def test_schedule_distribution_accepts_joint_state_inside_native_repeat_spread():
    natives = [
        _state(),
        _state(
            offset_range_angstrom=7.913957,
            sampling_acc_rot=7.985,
            sampling_acc_trans_angstrom=5.37625,
            current_changes_optimal_offsets_angstrom=1.195827,
        ),
    ]
    candidates = [
        _candidate(
            offset_range_angstrom=8.1189667,
            current_changes_optimal_offsets_angstrom=1.207909,
        ),
        _candidate(
            offset_range_angstrom=7.914,
            sampling_acc_rot=7.985,
            sampling_acc_trans_angstrom=5.37625,
            current_changes_optimal_offsets_angstrom=1.196,
        ),
    ]

    report = classify_schedule_distribution_envelope(_matrix(candidates, natives))

    assert report["pass"] is True
    assert report["candidate_validity_pass"] is True
    assert report["reverse_native_coverage_pass"] is True
    assert 1 in report["candidate_repeats"][0]["matching_native_repeat_indices"]
    assert 2 in report["candidate_repeats"][1]["matching_native_repeat_indices"]
    assert report["candidate_repeats"][0]["best_native_repeat_index"] == 1
    assert report["candidate_repeats"][1]["best_native_repeat_index"] == 2


def test_schedule_distribution_rejects_continuous_state_outside_native_spread():
    natives = [
        _state(),
        _state(current_changes_optimal_offsets_angstrom=1.195827),
    ]
    candidates = [
        _candidate(current_changes_optimal_offsets_angstrom=1.3),
        _candidate(current_changes_optimal_offsets_angstrom=1.31),
    ]

    report = classify_schedule_distribution_envelope(_matrix(candidates, natives))

    assert report["pass"] is False
    assert report["candidate_validity_pass"] is False
    assert all(
        not row["matching_native_repeat_indices"]
        for row in report["candidate_repeats"]
    )


def test_schedule_distribution_preserves_categorical_modes_and_reverse_coverage():
    native_mode_one = _state(n_translations=84)
    native_mode_two = _state(n_translations=52)
    candidates = [
        _candidate(n_translations=84),
        _candidate(n_translations=84),
    ]

    report = classify_schedule_distribution_envelope(
        _matrix(candidates, [native_mode_one, native_mode_two])
    )

    assert report["candidate_validity_pass"] is True
    assert report["reverse_native_coverage_pass"] is False
    assert report["native_repeats"][0]["matching_candidate_repeat_indices"] == [1, 2]
    assert report["native_repeats"][1]["matching_candidate_repeat_indices"] == []


def test_schedule_distribution_ignores_unestimated_accuracy_but_gates_estimated_accuracy():
    natives = [
        _state(sampling_acc_rot=24.0, sampling_acc_trans_angstrom=15.0),
        _state(sampling_acc_rot=25.0, sampling_acc_trans_angstrom=16.0),
    ]
    unestimated = [
        _candidate(estimated=False, sampling_acc_rot=0.0, sampling_acc_trans_angstrom=999.0),
        _candidate(estimated=False, sampling_acc_rot=0.0, sampling_acc_trans_angstrom=999.0),
    ]
    estimated = [
        _candidate(estimated=True, sampling_acc_rot=0.0, sampling_acc_trans_angstrom=999.0),
        _candidate(estimated=True, sampling_acc_rot=0.0, sampling_acc_trans_angstrom=999.0),
    ]

    assert classify_schedule_distribution_envelope(_matrix(unestimated, natives))["pass"] is True
    estimated_report = classify_schedule_distribution_envelope(_matrix(estimated, natives))
    assert estimated_report["candidate_validity_pass"] is False


def test_schedule_distribution_accepts_rectangular_candidate_native_panel():
    candidates = [_candidate(), _candidate(offset_range_angstrom=7.9)]
    natives = [
        _state(),
        _state(offset_range_angstrom=7.9),
        _state(offset_range_angstrom=8.0),
    ]

    report = classify_schedule_distribution_envelope(_matrix(candidates, natives))

    assert report["candidate_validity_pass"] is True
    assert len(report["candidate_repeats"]) == 2
    assert len(report["native_repeats"]) == 3


def test_schedule_distribution_rejects_incomplete_evidence():
    row = {"iteration": 1, "candidate": _candidate(), "native": _state()}
    with pytest.raises(CandidateStateEnvelopeError, match="two candidate and two native"):
        classify_schedule_distribution_envelope([[row], [row]])

    incomplete = _candidate()
    del incomplete["current_size"]
    with pytest.raises(CandidateStateEnvelopeError, match="missing fields"):
        classify_schedule_distribution_envelope(
            _matrix([incomplete, _candidate()], [_state(), _state()])
        )
