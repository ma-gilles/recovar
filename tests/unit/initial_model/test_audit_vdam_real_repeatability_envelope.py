from __future__ import annotations

import pandas as pd

from scripts.audit_vdam_real_repeatability_envelope import (
    compare_particle_tables_to_reference_set,
)


def _table(rows):
    return pd.DataFrame(
        rows,
        columns=(
            "_rlnImageName",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnAnglePsi",
            "_rlnOriginXAngst",
            "_rlnOriginYAngst",
            "_rlnMaxValueProbDistribution",
        ),
    )


def test_repeatability_envelope_accepts_state_and_pmax_from_either_reference():
    canonical = _table(
        [("1@stack.mrcs", 0, 0, 0, 0, 0, 0.4), ("2@stack.mrcs", 0, 0, 0, 0, 0, 0.8)]
    )
    repeat = _table(
        [("2@stack.mrcs", 0, 0, 0, 1, 0, 0.7), ("1@stack.mrcs", 0, 0, 0, 1, 0, 0.41)]
    )
    recovar = _table(
        [("1@stack.mrcs", 0, 0, 0, 0, 0, 0.405), ("2@stack.mrcs", 0, 0, 0, 1, 0, 0.705)]
    )

    report = compare_particle_tables_to_reference_set(
        recovar,
        canonical,
        repeat,
        active_image_ids={"1@stack.mrcs", "2@stack.mrcs"},
        pose_tolerance_deg=1e-3,
        translation_tolerance_angst=1e-4,
        pmax_absolute_error_p95_max=0.01,
        pmax_absolute_error_max=0.01,
    )

    assert report["recovar_vs_canonical_state_mismatch_count"] == 1
    assert report["recovar_vs_repeat_state_mismatch_count"] == 1
    assert report["recovar_vs_either_state_mismatch_count"] == 0
    assert report["pass"] is True


def test_repeatability_envelope_rejects_state_outside_both_references():
    canonical = _table([("1@stack.mrcs", 0, 0, 0, 0, 0, 0.4)])
    repeat = _table([("1@stack.mrcs", 0, 0, 0, 1, 0, 0.5)])
    recovar = _table([("1@stack.mrcs", 0, 0, 0, 2, 0, 0.45)])

    report = compare_particle_tables_to_reference_set(
        recovar,
        canonical,
        repeat,
        active_image_ids={"1@stack.mrcs"},
        pose_tolerance_deg=1e-3,
        translation_tolerance_angst=1e-4,
        pmax_absolute_error_p95_max=0.1,
        pmax_absolute_error_max=0.1,
    )

    assert report["recovar_vs_either_state_mismatch_count"] == 1
    assert report["first_particles_matching_neither"] == ["1@stack.mrcs"]
    assert report["pass"] is False
