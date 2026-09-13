from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_vdam_particle_state_trajectory import AuditError, compare_particle_tables


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


def test_particle_state_audit_aligns_identity_and_reports_first_divergence():
    recovar = _table(
        [
            ("2@stack.mrcs", 0, 0, 0, 0, 0, 0.5),
            ("1@stack.mrcs", 0, 0, 0, 0, 1, 0.4),
        ]
    )
    relion = _table(
        [
            ("1@stack.mrcs", 0, 0, 0, 0, 2, 0.3),
            ("2@stack.mrcs", 0, 0, 0, 0, 0, 0.51),
        ]
    )

    report = compare_particle_tables(
        recovar,
        relion,
        iteration=2,
        pose_tolerance_deg=1e-3,
        translation_tolerance_angst=1e-4,
    )

    assert report["identity_alignment_exact"] is True
    assert report["divergent_particle_count"] == 1
    assert report["pose_match_fraction"] == 1.0
    assert report["translation_match_fraction"] == 0.5
    assert report["pmax_absolute_error"]["mean"] == pytest.approx(0.055)
    assert report["first_divergent_particles"][0]["image_name"] == "1@stack.mrcs"
    assert report["first_divergent_particles"][0]["recovar_row"] == 1


def test_particle_state_audit_rejects_identity_set_drift():
    recovar = _table([("1@stack.mrcs", 0, 0, 0, 0, 0, 0.5)])
    relion = _table([("2@stack.mrcs", 0, 0, 0, 0, 0, 0.5)])

    with pytest.raises(AuditError, match="identity sets differ"):
        compare_particle_tables(
            recovar,
            relion,
            iteration=1,
            pose_tolerance_deg=1e-3,
            translation_tolerance_angst=1e-4,
        )


@pytest.mark.parametrize("identity", [None, float("nan"), pd.NA, "", "  "])
def test_particle_state_audit_rejects_matching_missing_identities(identity):
    table = _table([(identity, 0, 0, 0, 0, 0, 0.5)])
    with pytest.raises(AuditError, match="identities"):
        compare_particle_tables(
            table, table.copy(), iteration=1,
            pose_tolerance_deg=1e-3, translation_tolerance_angst=1e-4,
        )


@pytest.mark.parametrize("name", ["pose_tolerance_deg", "translation_tolerance_angst"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_particle_state_audit_rejects_invalid_tolerances(name, value):
    recovar = _table([("1@stack.mrcs", 0, 0, 0, 0, 0, 0.5)])
    relion = recovar.copy()
    column = "_rlnAngleTilt" if name == "pose_tolerance_deg" else "_rlnOriginYAngst"
    relion.loc[0, column] = 5.0
    kwargs = dict(pose_tolerance_deg=1e-3, translation_tolerance_angst=1e-4)
    kwargs[name] = value
    with pytest.raises(AuditError, match=name):
        compare_particle_tables(recovar, relion, iteration=1, **kwargs)


def test_particle_state_audit_accepts_zero_tolerances():
    table = _table([("1@stack.mrcs", 0, 0, 0, 0, 0, 0.5)])
    report = compare_particle_tables(
        table, table.copy(), iteration=1,
        pose_tolerance_deg=0.0, translation_tolerance_angst=0.0,
    )
    assert report["divergent_particle_count"] == 0


def test_empty_trajectory_is_not_reported_as_no_divergence(tmp_path):
    from scripts.audit_vdam_particle_state_trajectory import audit_trajectory

    with pytest.raises(AuditError, match="at least one iteration"):
        audit_trajectory(tmp_path, tmp_path, iterations=())


@pytest.mark.parametrize("changed,expected", [((2, 8), 2), ((8,), 8), ((), None)])
def test_trajectory_reports_earliest_checked_divergence_in_any_request_order(tmp_path, changed, expected):
    from recovar.data_io.starfile import write_star
    from scripts.audit_vdam_particle_state_trajectory import audit_trajectory

    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    for iteration in (2, 8):
        table = _table([("1@stack.mrcs", 0, 0, 0, 0, 0, 0.5)])
        write_star(str(reference / f"run_it{iteration:03d}_data.star"), table)
        if iteration in changed:
            table.loc[0, "_rlnOriginYAngst"] = 1.0
        write_star(str(candidate / f"run_it{iteration:03d}_data.star"), table)
    report = audit_trajectory(candidate, reference, iterations=(8, 2))
    assert [row["iteration"] for row in report["iterations"]] == [8, 2]
    assert report["first_divergent_iteration"] == expected
