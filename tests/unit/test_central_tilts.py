"""Focused regressions for strict common-position ``--central-tilts`` selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from recovar.commands import pipeline as pipeline_cmd
from recovar.data_io import halfsets, starfile

pytestmark = pytest.mark.unit


def _write_mrcs(path: Path, data: np.ndarray) -> None:
    import mrcfile

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))


def _make_global_dose_tilt_fixture(tmp_path: Path):
    """Make four particles over shared doses; gB lacks central dose 6."""
    rows = [
        ("gB", 0.0),
        ("gA", 9.0),
        ("gC", 12.0),
        ("gD", 6.0),
        ("gA", 0.0),
        ("gB", 3.0),
        ("gD", 0.0),
        ("gC", 6.0),
        ("gB", 9.0),
        ("gA", 6.0),
        ("gC", 0.0),
        ("gD", 12.0),
        ("gB", 12.0),
        ("gA", 3.0),
        ("gC", 3.0),
        ("gD", 3.0),
        ("gA", 12.0),
        ("gB", 15.0),
        ("gC", 9.0),
        ("gD", 9.0),
        ("gA", 15.0),
        ("gB", 18.0),
        ("gC", 15.0),
        ("gD", 15.0),
    ]
    n_images = len(rows)
    box_size = 8
    mrcs_path = tmp_path / "global_dose_stack.mrcs"
    _write_mrcs(
        mrcs_path,
        np.arange(n_images * box_size * box_size, dtype=np.float32).reshape(n_images, box_size, box_size),
    )

    particle_df = pd.DataFrame(
        {
            "_rlnImageName": [f"{idx + 1}@{mrcs_path.name}" for idx in range(n_images)],
            "_rlnGroupName": [group for group, _dose in rows],
            "_rlnTiltName": [f"{int(dose)}@fixture_tomo.mrcs" for _group, dose in rows],
            "_rlnMicrographPreExposure": [dose for _group, dose in rows],
            # Some extracted STARs have an uninformative all-zero angle column.
            # The strict position identity must still work from global dose.
            "_rlnTomoNominalStageTiltAngle": np.zeros(n_images, dtype=np.float32),
            "_rlnCtfScalefactor": np.ones(n_images, dtype=np.float32),
            "_rlnCtfBfactor": -np.ones(n_images, dtype=np.float32),
        }
    )
    star_path = tmp_path / "global_dose_particles.star"
    starfile.write_star(str(star_path), data=particle_df)
    return star_path, particle_df


def _selected_image_union(split):
    return np.sort(np.concatenate([np.asarray(half, dtype=np.int32) for half in split]))


def _all_particles_in_first_halfset():
    return [np.arange(4, dtype=np.int32), np.array([], dtype=np.int32)]


def _pipeline_parser():
    parser = argparse.ArgumentParser()
    pipeline_cmd.add_args(parser)
    return parser


def test_pipeline_parser_registers_central_tilts():
    parser = _pipeline_parser()
    action = parser._option_string_actions["--central-tilts"]

    assert action.dest == "central_tilts"
    assert action.default is None
    args = parser.parse_args(["particles.star", "--mask", "sphere", "--central-tilts", "5"])
    assert args.central_tilts == 5


@pytest.mark.parametrize("value", ["0", "-1"])
def test_pipeline_parser_rejects_nonpositive_central_tilts(value):
    with pytest.raises(SystemExit):
        _pipeline_parser().parse_args(["particles.star", "--mask", "sphere", "--central-tilts", value])


def test_pipeline_parser_rejects_central_tilts_with_ntilts():
    with pytest.raises(SystemExit):
        _pipeline_parser().parse_args(
            [
                "particles.star",
                "--mask",
                "sphere",
                "--central-tilts",
                "5",
                "--ntilts",
                "5",
            ]
        )


def test_pipeline_validation_requires_tilt_series():
    args = SimpleNamespace(central_tilts=5, ntilts=None, tilt_series=False)
    with pytest.raises(ValueError, match="requires --tilt-series"):
        pipeline_cmd._validate_tilt_selection_args(args)


def test_pipeline_validation_rejects_programmatic_ntilts_conflict():
    args = SimpleNamespace(central_tilts=5, ntilts=5, tilt_series=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline_cmd._validate_tilt_selection_args(args)


def test_central_tilts_requires_full_global_dose_set_without_substitution(tmp_path):
    star_path, particle_df = _make_global_dose_tilt_fixture(tmp_path)
    split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        central_tilts=5,
        datadir=str(tmp_path),
        particle_halfset_indices_file=_all_particles_in_first_halfset(),
    )

    kept = _selected_image_union(split)
    # Global targets are doses 0, 3, 6, 9, 12. gB lacks dose 6 and must be
    # removed; its later dose-15 image must not substitute for the missing view.
    expected = np.array([1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 18, 19], dtype=np.int32)
    np.testing.assert_array_equal(kept, expected)
    assert set(particle_df.iloc[kept]["_rlnGroupName"]) == {"gA", "gC", "gD"}
    for _group, particle_rows in particle_df.iloc[kept].groupby("_rlnGroupName"):
        np.testing.assert_array_equal(
            np.sort(particle_rows["_rlnMicrographPreExposure"].to_numpy(dtype=float)),
            np.array([0.0, 3.0, 6.0, 9.0, 12.0]),
        )

    ordinary_split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        ntilts=5,
        datadir=str(tmp_path),
        particle_halfset_indices_file=_all_particles_in_first_halfset(),
    )
    assert 17 in _selected_image_union(ordinary_split)  # gB's dose-15 substitution under --ntilts.


def test_central_tilts_applies_ind_before_particle_completeness(tmp_path):
    star_path, particle_df = _make_global_dose_tilt_fixture(tmp_path)
    # Remove gC's central dose-6 image but retain its later dose-15 image.
    image_ind = np.setdiff1d(np.arange(len(particle_df), dtype=np.int32), np.array([7], dtype=np.int32))
    split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        ind_file=image_ind,
        central_tilts=5,
        datadir=str(tmp_path),
        particle_halfset_indices_file=_all_particles_in_first_halfset(),
    )

    kept = _selected_image_union(split)
    assert set(particle_df.iloc[kept]["_rlnGroupName"]) == {"gA", "gD"}
    assert 22 not in kept
    assert kept.size == 10


def test_central_tilts_composes_with_particle_ind(tmp_path):
    star_path, particle_df = _make_global_dose_tilt_fixture(tmp_path)
    split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        tilt_ind_file=np.array([1, 2], dtype=np.int32),  # canonical gB and gC
        central_tilts=5,
        datadir=str(tmp_path),
        particle_halfset_indices_file=[np.array([1, 2], dtype=np.int32), np.array([], dtype=np.int32)],
    )

    kept = _selected_image_union(split)
    assert set(particle_df.iloc[kept]["_rlnGroupName"]) == {"gC"}
    assert kept.size == 5


def test_central_tilts_allows_different_dose_schedules_per_tomogram(tmp_path):
    rows = []
    schedules = {
        "tomoA": [0.0, 3.0, 6.0, 9.0, 12.0, 15.0],
        "tomoB": [0.2, 3.1, 6.2, 9.3, 12.4, 15.5],
    }
    for tomo_name, doses in schedules.items():
        for particle_name, frames in (("full", range(6)), ("missing", [0, 1, 3, 4, 5])):
            for frame in frames:
                rows.append((f"{tomo_name}/{particle_name}", f"{frame + 1}@{tomo_name}.mrcs", doses[frame]))

    particle_df = pd.DataFrame(
        {
            "_rlnImageName": [f"{idx + 1}@variable_schedule.mrcs" for idx in range(len(rows))],
            "_rlnGroupName": [row[0] for row in rows],
            "_rlnTiltName": [row[1] for row in rows],
            "_rlnMicrographPreExposure": [row[2] for row in rows],
            "_rlnTomoNominalStageTiltAngle": np.zeros(len(rows), dtype=np.float32),
            "_rlnCtfScalefactor": np.ones(len(rows), dtype=np.float32),
            "_rlnCtfBfactor": -np.ones(len(rows), dtype=np.float32),
        }
    )
    star_path = tmp_path / "variable_schedule.star"
    starfile.write_star(str(star_path), data=particle_df)

    split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        central_tilts=5,
        particle_halfset_indices_file=[np.arange(4, dtype=np.int32), np.array([], dtype=np.int32)],
    )
    kept = _selected_image_union(split)
    assert set(particle_df.iloc[kept]["_rlnGroupName"]) == {"tomoA/full", "tomoB/full"}
    assert kept.size == 10


def test_central_tilts_prefers_informative_nominal_angles(tmp_path):
    angles = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0]
    rows = []
    for particle_name in ("particle1", "particle2"):
        for frame, angle in enumerate(angles):
            rows.append((particle_name, f"{frame + 1}@angle_tomo.mrcs", float(frame), angle))

    particle_df = pd.DataFrame(
        {
            "_rlnImageName": [f"{idx + 1}@angle_stack.mrcs" for idx in range(len(rows))],
            "_rlnGroupName": [row[0] for row in rows],
            "_rlnTiltName": [row[1] for row in rows],
            "_rlnMicrographPreExposure": [row[2] for row in rows],
            "_rlnTomoNominalStageTiltAngle": [row[3] for row in rows],
            "_rlnCtfScalefactor": np.ones(len(rows), dtype=np.float32),
            "_rlnCtfBfactor": -np.ones(len(rows), dtype=np.float32),
        }
    )
    star_path = tmp_path / "angle_schedule.star"
    starfile.write_star(str(star_path), data=particle_df)

    split = halfsets.get_split_tilt_indices(
        particles_file=str(star_path),
        central_tilts=5,
        particle_halfset_indices_file=[np.arange(2, dtype=np.int32), np.array([], dtype=np.int32)],
    )
    kept = _selected_image_union(split)
    for _group, particle_rows in particle_df.iloc[kept].groupby("_rlnGroupName"):
        np.testing.assert_array_equal(
            np.sort(particle_rows["_rlnTomoNominalStageTiltAngle"].to_numpy(dtype=float)),
            np.array([-6.0, -3.0, 0.0, 3.0, 6.0]),
        )


@pytest.mark.parametrize("central_tilts", [0, -1, 8])
def test_get_split_tilt_indices_rejects_invalid_central_tilts(tmp_path, central_tilts):
    star_path, _particle_df = _make_global_dose_tilt_fixture(tmp_path)
    with pytest.raises(ValueError, match="central_tilts"):
        halfsets.get_split_tilt_indices(
            particles_file=str(star_path),
            central_tilts=central_tilts,
            datadir=str(tmp_path),
        )


def test_get_split_tilt_indices_rejects_central_tilts_with_ntilts(tmp_path):
    star_path, _particle_df = _make_global_dose_tilt_fixture(tmp_path)
    with pytest.raises(ValueError, match="central_tilts.*ntilts|ntilts.*central_tilts"):
        halfsets.get_split_tilt_indices(
            particles_file=str(star_path),
            central_tilts=5,
            ntilts=5,
            datadir=str(tmp_path),
        )


def test_resolve_halfset_indices_propagates_central_tilts(monkeypatch):
    captured = {}

    def fake_get_split_tilt_indices(*args, **kwargs):
        captured.update(kwargs)
        return [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]

    args = SimpleNamespace(
        halfsets=None,
        tilt_series=True,
        tilt_series_ctf="relion5",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        central_tilts=5,
        datadir="/tmp/data",
        strip_prefix=None,
        n_images=-1,
    )
    monkeypatch.setattr(halfsets, "get_split_tilt_indices", fake_get_split_tilt_indices)

    halfsets.resolve_halfset_indices(args)

    assert captured["central_tilts"] == 5


def test_resolve_halfset_indices_rejects_n_images_with_central_tilts(monkeypatch):
    args = SimpleNamespace(
        halfsets=None,
        tilt_series=True,
        tilt_series_ctf="relion5",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        central_tilts=5,
        datadir="/tmp/data",
        strip_prefix=None,
        n_images=10,
    )
    monkeypatch.setattr(
        halfsets,
        "get_split_tilt_indices",
        lambda *args, **kwargs: [np.arange(10, dtype=np.int32), np.arange(10, 20, dtype=np.int32)],
    )

    with pytest.raises(ValueError, match="n-images.*central-tilts"):
        halfsets.resolve_halfset_indices(args)


def _write_identity_star(tmp_path, name, rows, *, include_dose=True):
    particle_df = pd.DataFrame(
        {
            "_rlnImageName": [f"{idx + 1}@{name}_images.mrcs" for idx in range(len(rows))],
            "_rlnGroupName": [row["particle"] for row in rows],
            "_rlnTiltName": [row["tilt_name"] for row in rows],
            "_rlnTomoNominalStageTiltAngle": [row["angle"] for row in rows],
            "_rlnCtfScalefactor": np.ones(len(rows), dtype=np.float32),
            "_rlnCtfBfactor": -np.ones(len(rows), dtype=np.float32),
        }
    )
    if include_dose:
        particle_df["_rlnMicrographPreExposure"] = [row["dose"] for row in rows]
    star_path = tmp_path / f"{name}.star"
    starfile.write_star(str(star_path), data=particle_df)
    return star_path, particle_df


def test_central_tilts_accepts_separate_stack_per_physical_tilt(tmp_path):
    rows = []
    angles = [-6.0, -3.0, 0.0, 3.0, 6.0]
    for particle in ("p0", "p1"):
        for frame, angle in enumerate(angles):
            rows.append(
                {
                    "particle": particle,
                    "tilt_name": f"1@physical_tilt_{frame}.mrcs",
                    "angle": angle,
                    "dose": float(frame),
                }
            )
    star_path, _particle_df = _write_identity_star(tmp_path, "separate_stacks", rows)

    split = halfsets.get_split_tilt_indices(str(star_path), central_tilts=5)
    assert _selected_image_union(split).size == 10


def test_central_tilts_uses_informative_angles_without_dose_column(tmp_path):
    rows = []
    angles = [-6.0, -3.0, 0.0, 3.0, 6.0]
    for particle in ("p0", "p1"):
        for frame, angle in enumerate(angles):
            rows.append(
                {
                    "particle": particle,
                    "tilt_name": f"{frame + 1}@angle_only_tomo.mrcs",
                    "angle": angle,
                }
            )
    star_path, particle_df = _write_identity_star(tmp_path, "angle_only", rows, include_dose=False)

    split = halfsets.get_split_tilt_indices(str(star_path), central_tilts=5)
    kept = _selected_image_union(split)
    assert kept.size == 10
    assert "_rlnMicrographPreExposure" not in particle_df.columns


def test_nan_angle_in_one_tomogram_does_not_disable_other_tomograms(tmp_path):
    rows = []
    angles = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0]
    for tomo in ("good", "bad"):
        for frame, angle in enumerate(angles):
            rows.append(
                {
                    "particle": f"{tomo}/particle",
                    "tilt_name": f"{frame + 1}@{tomo}_tomo.mrcs",
                    "angle": np.nan if tomo == "bad" and frame == 0 else angle,
                    "dose": float(frame),
                }
            )
    star_path, particle_df = _write_identity_star(tmp_path, "isolated_nan", rows)

    split = halfsets.get_split_tilt_indices(str(star_path), central_tilts=5)
    kept_rows = particle_df.iloc[_selected_image_union(split)]
    good_angles = np.sort(
        kept_rows.loc[
            kept_rows["_rlnGroupName"] == "good/particle",
            "_rlnTomoNominalStageTiltAngle",
        ].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(good_angles, np.array([-6.0, -3.0, 0.0, 3.0, 6.0]))


def test_central_tilts_drops_tomogram_whose_zero_position_is_absent(tmp_path):
    rows = []
    schedules = {
        "complete": [-6.0, -3.0, 0.0, 3.0, 6.0],
        "missing_zero": [-9.0, -6.0, -3.0, 3.0, 6.0, 9.0],
    }
    for tomo, angles in schedules.items():
        for frame, angle in enumerate(angles):
            rows.append(
                {
                    "particle": f"{tomo}/particle",
                    "tilt_name": f"{frame + 1}@{tomo}_tomo.mrcs",
                    "angle": angle,
                    "dose": float(frame),
                }
            )
    star_path, particle_df = _write_identity_star(tmp_path, "missing_zero", rows)

    split = halfsets.get_split_tilt_indices(str(star_path), central_tilts=5)
    kept = _selected_image_union(split)
    assert set(particle_df.iloc[kept]["_rlnGroupName"]) == {"complete/particle"}
    assert kept.size == 5
