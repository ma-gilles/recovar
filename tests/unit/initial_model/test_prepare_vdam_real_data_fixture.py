from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import starfile

from scripts.prepare_vdam_real_data_fixture import (
    materialize_cryodrgn_source_star,
    promote_legacy_optics,
    select_balanced_half_indices,
    select_synthetic_half_indices,
)


def _particles(halves):
    return pd.DataFrame(
        {
            "rlnImageName": [f"{index + 1}@particles.mrcs" for index in range(len(halves))],
            "rlnRandomSubset": halves,
        }
    )


def test_balanced_half_selection_is_deterministic_sorted_and_exactly_balanced():
    particles = _particles([1] * 8 + [2] * 8)

    first = select_balanced_half_indices(particles, particles_per_half=4, seed=23)
    second = select_balanced_half_indices(particles, particles_per_half=4, seed=23)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) > 0)
    selected_halves = np.asarray(particles.iloc[first]["rlnRandomSubset"])
    assert np.count_nonzero(selected_halves == 1) == 4
    assert np.count_nonzero(selected_halves == 2) == 4


def test_balanced_half_selection_rejects_insufficient_half():
    particles = _particles([1] * 5 + [2] * 2)

    with pytest.raises(ValueError, match="half 2 contains 2 particles"):
        select_balanced_half_indices(particles, particles_per_half=3, seed=0)


def test_balanced_half_selection_rejects_invalid_subset_identifiers():
    particles = _particles([1, 1, 2, 3])

    with pytest.raises(ValueError, match="only RELION half identifiers 1 and 2"):
        select_balanced_half_indices(particles, particles_per_half=1, seed=0)


def test_synthetic_half_selection_is_deterministic_sorted_and_balanced():
    first_indices, first_halves = select_synthetic_half_indices(20, particles_per_half=4, seed=23)
    second_indices, second_halves = select_synthetic_half_indices(20, particles_per_half=4, seed=23)

    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_halves, second_halves)
    assert np.all(np.diff(first_indices) > 0)
    assert np.count_nonzero(first_halves == 1) == 4
    assert np.count_nonzero(first_halves == 2) == 4


def test_promote_legacy_optics_preserves_particles_and_computes_pixel_size():
    particles = _particles([1, 2])
    particles["rlnVoltage"] = 300.0
    particles["rlnSphericalAberration"] = 2.7
    particles["rlnAmplitudeContrast"] = 0.1
    particles["rlnDetectorPixelSize"] = 5.0
    particles["rlnMagnification"] = 35714.0
    particles["rlnDefocusU"] = [10_000.0, 11_000.0]
    particles["rlnOriginX"] = [2.0, -3.0]
    particles["rlnOriginY"] = [-1.0, 4.0]
    particles["rlnMaxValueProbDistribution"] = [0.9, 0.7]

    promoted = promote_legacy_optics(particles, image_size=256)

    optics = promoted["optics"]
    output_particles = promoted["particles"]
    assert optics.loc[0, "rlnImagePixelSize"] == pytest.approx(5.0 * 10_000.0 / 35714.0)
    assert optics.loc[0, "rlnImageSize"] == 256
    np.testing.assert_array_equal(output_particles["rlnOpticsGroup"], [1, 1])
    np.testing.assert_array_equal(output_particles["rlnPhaseShift"], [0.0, 0.0])
    np.testing.assert_array_equal(output_particles["rlnDefocusU"], [10_000.0, 11_000.0])
    np.testing.assert_allclose(
        output_particles["rlnOriginXAngst"],
        np.asarray([2.0, -3.0]) * optics.loc[0, "rlnImagePixelSize"],
    )
    np.testing.assert_allclose(
        output_particles["rlnOriginYAngst"],
        np.asarray([-1.0, 4.0]) * optics.loc[0, "rlnImagePixelSize"],
    )
    np.testing.assert_array_equal(output_particles["rlnMaxValueProbDistribution"], [0.0, 0.0])
    assert "rlnDetectorPixelSize" not in output_particles
    assert "rlnMagnification" not in output_particles


def test_promote_legacy_optics_rejects_unrepresented_microscope_groups():
    particles = _particles([1, 2])
    particles["rlnVoltage"] = [300.0, 200.0]
    particles["rlnSphericalAberration"] = 2.7
    particles["rlnAmplitudeContrast"] = 0.1
    particles["rlnDetectorPixelSize"] = 5.0
    particles["rlnMagnification"] = 35714.0

    with pytest.raises(ValueError, match="rlnVoltage varies"):
        promote_legacy_optics(particles, image_size=256)


def test_promote_legacy_optics_rejects_incomplete_origin_pairs():
    particles = _particles([1, 2])
    particles["rlnVoltage"] = 300.0
    particles["rlnSphericalAberration"] = 2.7
    particles["rlnAmplitudeContrast"] = 0.1
    particles["rlnDetectorPixelSize"] = 5.0
    particles["rlnMagnification"] = 35714.0
    particles["rlnOriginX"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="both rlnOriginX and rlnOriginY"):
        promote_legacy_optics(particles, image_size=256)


def test_promote_legacy_optics_accepts_explicit_pixel_size_when_scale_columns_are_absent():
    particles = _particles([1, 2])
    particles["rlnVoltage"] = 300.0
    particles["rlnSphericalAberration"] = 2.7
    particles["rlnAmplitudeContrast"] = 0.1

    promoted = promote_legacy_optics(particles, image_size=256, pixel_size=1.345)

    assert promoted["optics"].loc[0, "rlnImagePixelSize"] == pytest.approx(1.345)


def test_promote_legacy_optics_requires_explicit_pixel_size_without_scale_columns():
    particles = _particles([1, 2])
    particles["rlnVoltage"] = 300.0
    particles["rlnSphericalAberration"] = 2.7
    particles["rlnAmplitudeContrast"] = 0.1

    with pytest.raises(ValueError, match="provide an explicit positive pixel size"):
        promote_legacy_optics(particles, image_size=256)


def test_materialize_cryodrgn_source_star_preserves_indexed_identity_and_units(tmp_path):
    n_images = 5
    grid_size = 64
    voxel_size = 1.5
    particles_path = tmp_path / "particles.mrcs"
    particles_path.touch()
    poses_path = tmp_path / "poses.pkl"
    ctf_path = tmp_path / "ctf.pkl"
    indices_path = tmp_path / "ind.pkl"
    output_star = tmp_path / "source" / "particles.star"

    rotations = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    translations = np.array(
        [[0.0, 0.0], [0.125, -0.25], [0.0, 0.0], [-0.0625, 0.03125], [0.0, 0.0]],
        dtype=np.float32,
    )
    ctf = np.zeros((n_images, 9), dtype=np.float32)
    ctf[:, 0] = grid_size
    ctf[:, 1] = voxel_size
    ctf[:, 2:5] = np.array([10_000.0, 11_000.0, 15.0], dtype=np.float32)
    ctf[:, 5:8] = np.array([300.0, 2.7, 0.1], dtype=np.float32)
    with poses_path.open("wb") as handle:
        pickle.dump((rotations, translations), handle)
    with ctf_path.open("wb") as handle:
        pickle.dump(ctf, handle)
    with indices_path.open("wb") as handle:
        pickle.dump(np.array([3, 1], dtype=np.int64), handle)

    manifest = materialize_cryodrgn_source_star(
        particles_path=particles_path,
        poses_path=poses_path,
        ctf_path=ctf_path,
        output_star=output_star,
        source_indices_path=indices_path,
    )

    particles = starfile.read(output_star)["particles"]
    assert particles["rlnImageName"].tolist() == [
        f"4@{particles_path}",
        f"2@{particles_path}",
    ]
    expected_angstrom = translations[[3, 1]] * grid_size * voxel_size
    np.testing.assert_allclose(particles["rlnOriginXAngst"], expected_angstrom[:, 0])
    np.testing.assert_allclose(particles["rlnOriginYAngst"], expected_angstrom[:, 1])
    assert manifest["source_particle_count"] == 5
    assert manifest["selected_particle_count"] == 2
    assert manifest["output_star_sha256"]
    assert output_star.with_suffix(".materialization.json").is_file()
