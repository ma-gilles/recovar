"""Production input-STAR pose initialization for full EM refinement."""

import argparse

import numpy as np
import pandas as pd
import pytest

from recovar.em.relion.input_poses import (
    _add_initial_pose_source_argument,
    _load_input_star_previous_best_poses,
    _resolve_input_star_pose_seed,
)

pytestmark = pytest.mark.unit


def _particle_tables(*, angstrom_origins=False):
    input_particles = pd.DataFrame(
        {
            "rlnImageName": ["1@stack.mrcs", "2@stack.mrcs", "3@stack.mrcs", "4@stack.mrcs"],
            "rlnAngleRot": [10.0, 20.0, 30.0, 40.0],
            "rlnAngleTilt": [11.0, 21.0, 31.0, 41.0],
            "rlnAnglePsi": [12.0, 22.0, 32.0, 42.0],
        }
    )
    origin_scale = 2.0 if angstrom_origins else 1.0
    suffix = "Angst" if angstrom_origins else ""
    input_particles[f"rlnOriginX{suffix}"] = origin_scale * np.asarray([0.5, 1.5, 2.5, 3.5])
    input_particles[f"rlnOriginY{suffix}"] = origin_scale * np.asarray([-0.5, -1.5, -2.5, -3.5])
    # Deliberately permute the half-set STAR; production mapping must use the
    # complete image identity, not the DataFrame row number.
    halfset_particles = pd.DataFrame(
        {
            "rlnImageName": ["3@stack.mrcs", "1@stack.mrcs", "4@stack.mrcs", "2@stack.mrcs"],
            "rlnRandomSubset": [1, 1, 2, 2],
        }
    )
    return input_particles, halfset_particles


@pytest.mark.parametrize("angstrom_origins", [False, True])
def test_input_star_pose_seed_follows_half_local_order_and_converts_origins(
    angstrom_origins,
):
    input_particles, halfset_particles = _particle_tables(
        angstrom_origins=angstrom_origins,
    )

    seed = _load_input_star_previous_best_poses(
        input_particles,
        halfset_particles,
        half1_idx=np.asarray([2, 0]),
        half2_idx=np.asarray([3, 1]),
        voxel_size=2.0,
    )

    assert seed["iteration"] == "input_star"
    assert seed["translation_units"] == ("angstrom" if angstrom_origins else "pixel")
    np.testing.assert_array_equal(
        seed["previous_best_rotation_eulers"][0],
        np.asarray([[30.0, 31.0, 32.0], [10.0, 11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_rotation_eulers"][1],
        np.asarray([[40.0, 41.0, 42.0], [20.0, 21.0, 22.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_translations"][0],
        np.asarray([[2.5, -2.5], [0.5, -0.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        seed["previous_best_translations"][1],
        np.asarray([[3.5, -3.5], [1.5, -1.5]], dtype=np.float32),
    )


def test_input_star_pose_seed_rejects_stale_half_layout():
    input_particles, halfset_particles = _particle_tables()

    with pytest.raises(ValueError, match="disagrees with RELION rlnRandomSubset"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 1]),
            half2_idx=np.asarray([3, 0]),
            voxel_size=1.0,
        )


def test_input_star_pose_seed_rejects_nonpartition_and_nonfinite_pose():
    input_particles, halfset_particles = _particle_tables()
    with pytest.raises(ValueError, match="exact partition"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 0]),
            half2_idx=np.asarray([3]),
            voxel_size=1.0,
        )

    input_particles.loc[2, "rlnAnglePsi"] = np.nan
    with pytest.raises(ValueError, match="Euler-angle values must be finite"):
        _load_input_star_previous_best_poses(
            input_particles,
            halfset_particles,
            half1_idx=np.asarray([2, 0]),
            half2_idx=np.asarray([3, 1]),
            voxel_size=1.0,
        )


def test_initial_pose_source_cli_defaults_to_fresh_k1_halfset_auto():
    parser = argparse.ArgumentParser()
    _add_initial_pose_source_argument(parser)

    assert parser.parse_args([]).initial_pose_source == "auto"
    assert parser.parse_args(["--initial-pose-source", "input-star"]).initial_pose_source == "input-star"
    assert _resolve_input_star_pose_seed(
        "auto",
        n_classes=1,
        init_relion_iteration=0,
        has_relion_half_sets=True,
        has_competing_pose_source=False,
        diagnostic_single_half=False,
    )
    assert not _resolve_input_star_pose_seed(
        "auto",
        n_classes=4,
        init_relion_iteration=0,
        has_relion_half_sets=True,
        has_competing_pose_source=False,
        diagnostic_single_half=False,
    )


def test_explicit_input_star_pose_source_fails_closed_on_competing_state():
    with pytest.raises(ValueError, match="already owns initialization"):
        _resolve_input_star_pose_seed(
            "input-star",
            n_classes=1,
            init_relion_iteration=0,
            has_relion_half_sets=True,
            has_competing_pose_source=True,
            diagnostic_single_half=False,
        )
