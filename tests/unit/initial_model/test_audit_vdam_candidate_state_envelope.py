from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_vdam_candidate_state_envelope import (
    CandidateStateEnvelopeError,
    classify_schedule_mode_envelope,
    compare_particle_tables_to_native_set,
)


def _particles(states):
    return pd.DataFrame(
        {
            "_rlnImageName": [row[0] for row in states],
            "_rlnAngleRot": [row[1] for row in states],
            "_rlnAngleTilt": 0.0,
            "_rlnAnglePsi": 0.0,
            "_rlnOriginXAngst": [row[2] for row in states],
            "_rlnOriginYAngst": 0.0,
            "_rlnMaxValueProbDistribution": [row[3] for row in states],
        }
    )


def _schedule_row(
    iteration,
    checks,
    *,
    estimated=True,
    candidate_counter=None,
    native_counter=None,
):
    row = {
        "iteration": iteration,
        "candidate": {"sampling_accuracy_estimated": estimated},
        "native": {},
        "checks": checks,
    }
    if candidate_counter is not None:
        row["candidate"]["nr_iter_without_resolution_gain"] = candidate_counter
    if native_counter is not None:
        row["native"]["nr_iter_without_resolution_gain"] = native_counter
    return row


def test_particle_envelope_accepts_per_particle_native_modes():
    candidate = _particles((("a", 0.0, 0.0, 0.4), ("b", 10.0, 2.0, 0.6)))
    native1 = _particles((("b", 40.0, 8.0, 0.2), ("a", 0.0, 0.0, 0.5)))
    native2 = _particles((("a", 30.0, 9.0, 0.1), ("b", 10.0, 2.0, 0.7)))
    report = compare_particle_tables_to_native_set(
        candidate, [native1, native2], active_image_ids={"a", "b"}
    )
    assert report["pass"] is True
    assert report["candidate_vs_each_native_mismatch_count"] == [1, 1]
    assert report["candidate_vs_native_envelope_mismatch_count"] == 0


def test_particle_envelope_rejects_state_outside_all_native_modes():
    candidate = _particles((("a", 5.0, 0.0, 0.4),))
    native1 = _particles((("a", 0.0, 0.0, 0.4),))
    native2 = _particles((("a", 10.0, 0.0, 0.4),))
    report = compare_particle_tables_to_native_set(
        candidate, [native1, native2], active_image_ids={"a"}
    )
    assert report["pass"] is False
    assert report["first_particles_matching_no_native_repeat"] == ["a"]


def test_particle_envelope_rejects_mixed_identity_sets():
    candidate = _particles((("a", 0.0, 0.0, 0.4),))
    native = _particles((("b", 0.0, 0.0, 0.4),))
    with pytest.raises(CandidateStateEnvelopeError, match="identity sets differ"):
        compare_particle_tables_to_native_set(
            candidate, [native, native], active_image_ids={"a"}
        )


def test_particle_envelope_rejects_incomplete_active_identity_set():
    table = _particles((("a", 0.0, 0.0, 0.4),))
    with pytest.raises(CandidateStateEnvelopeError, match="exact nonempty subset"):
        compare_particle_tables_to_native_set(
            table, [table, table], active_image_ids={"a", "missing"}
        )


def test_schedule_envelope_accepts_one_complete_native_mode():
    report = classify_schedule_mode_envelope(
        [
            _schedule_row(90, {"healpix_order": False, "offset_step": True}),
            _schedule_row(90, {"healpix_order": True, "offset_step": True}),
        ]
    )
    assert report["pass"] is True
    assert report["matching_native_repeat_indices"] == [2]


def test_schedule_envelope_rejects_cross_repeat_field_chimera():
    report = classify_schedule_mode_envelope(
        [
            _schedule_row(90, {"healpix_order": True, "offset_step": False}),
            _schedule_row(90, {"healpix_order": False, "offset_step": True}),
        ]
    )
    assert report["pass"] is False
    assert all(report["checks_matching_at_least_one_native"].values())


def test_schedule_envelope_reports_pre_post_mstep_counter_without_gating_it():
    report = classify_schedule_mode_envelope(
        [
            _schedule_row(
                1,
                {"healpix_order": True},
                candidate_counter=0,
                native_counter=1,
            ),
            _schedule_row(
                1,
                {"healpix_order": True},
                candidate_counter=0,
                native_counter=1,
            ),
        ]
    )

    assert report["pass"] is True
    assert "nr_iter_without_resolution_gain" not in report["active_checks"]
    assert report["diagnostic_only_fields"] == ["nr_iter_without_resolution_gain"]
    assert report["diagnostics_matching_at_least_one_native"] == {
        "nr_iter_without_resolution_gain": False
    }


def test_schedule_envelope_ignores_unused_accuracy_until_estimated():
    report = classify_schedule_mode_envelope(
        [
            _schedule_row(
                1,
                {"healpix_order": True, "accuracy_rotation": False, "accuracy_translation": False},
                estimated=False,
            ),
            _schedule_row(
                1,
                {"healpix_order": False, "accuracy_rotation": True, "accuracy_translation": True},
                estimated=False,
            ),
        ]
    )
    assert report["pass"] is True
    assert report["matching_native_repeat_indices"] == [1]
    assert "accuracy_rotation" not in report["active_checks"]


def test_schedule_envelope_requires_accuracy_after_estimation():
    report = classify_schedule_mode_envelope(
        [
            _schedule_row(
                90,
                {"healpix_order": True, "accuracy_rotation": False, "accuracy_translation": False},
            ),
            _schedule_row(
                90,
                {"healpix_order": False, "accuracy_rotation": True, "accuracy_translation": True},
            ),
        ]
    )
    assert report["pass"] is False
    assert "accuracy_rotation" in report["active_checks"]


def test_schedule_envelope_rejects_mixed_iterations():
    with pytest.raises(CandidateStateEnvelopeError, match="one iteration"):
        classify_schedule_mode_envelope(
            [
                _schedule_row(89, {"healpix_order": True}),
                _schedule_row(90, {"healpix_order": True}),
            ]
        )
