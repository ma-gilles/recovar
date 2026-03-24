"""
Unit tests for recovar.commands.parse_relion5_tomo.

Covers:
  add_args()               – CLI argument registration
  _parse_visible_frames()  – frame selection parsing
  Tomogram geometry        – rotation matrices, local defocus
"""
import argparse

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from recovar.commands import parse_relion5_tomo as tomo_cmd

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# add_args – argument registration
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    tomo_cmd.add_args(parser)
    return parser


def test_registers_tomograms_flag():
    actions = _parser()._option_string_actions
    assert "-t" in actions or "--tomograms" in actions


def test_registers_particles_flag():
    actions = _parser()._option_string_actions
    assert "-p" in actions or "--particles" in actions


def test_registers_output_flag():
    actions = _parser()._option_string_actions
    assert "-o" in actions or "--output" in actions


def test_output_default_is_particles_2d_star():
    action = _parser()._option_string_actions["--output"]
    assert action.default == "particles_2d.star"


def test_registers_verbose_flag():
    actions = _parser()._option_string_actions
    assert "-v" in actions or "--verbose" in actions


# ---------------------------------------------------------------------------
# _parse_visible_frames
# ---------------------------------------------------------------------------

def test_parse_visible_frames_basic():
    result = tomo_cmd._parse_visible_frames("[1,1,0,1,0,1]")
    assert result == [0, 1, 3, 5]


def test_parse_visible_frames_all_visible():
    result = tomo_cmd._parse_visible_frames("[1,1,1,1]")
    assert result == [0, 1, 2, 3]


def test_parse_visible_frames_none_visible():
    result = tomo_cmd._parse_visible_frames("[0,0,0]")
    assert result == []


def test_parse_visible_frames_single():
    result = tomo_cmd._parse_visible_frames("[1]")
    assert result == [0]


# ---------------------------------------------------------------------------
# Tomogram – defocus and rotation
# ---------------------------------------------------------------------------

def _make_simple_tomogram(n_tilts=3):
    """Create a Tomogram with simple geometry for testing."""
    return tomo_cmd.Tomogram(
        pixel_size=1.0,
        defocus_u=np.array([10000.0] * n_tilts),
        defocus_v=np.array([10000.0] * n_tilts),
        defocus_angle=np.zeros(n_tilts),
        x_tilts=np.zeros(n_tilts),
        y_tilts=np.zeros(n_tilts),
        z_rots=np.zeros(n_tilts),
        x_shifts=np.zeros(n_tilts),
        y_shifts=np.zeros(n_tilts),
        hand=1,
    )


def test_tomogram_init_creates_rotation_matrices():
    tomo = _make_simple_tomogram(n_tilts=5)
    assert tomo.n_tilts == 5
    assert tomo._rzyx_matrices.shape == (5, 3, 3)
    assert tomo._tilt_rots.shape == (5, 3, 3)


def test_local_defocus_at_origin_returns_nominal():
    """At the origin with no tilt, local defocus should equal nominal."""
    tomo = _make_simple_tomogram(n_tilts=1)
    dfu, dfv, dfa = tomo.local_defocus(0, np.array([0.0, 0.0, 0.0]))
    assert dfu == pytest.approx(10000.0)
    assert dfv == pytest.approx(10000.0)
    assert dfa == pytest.approx(0.0)


def test_local_defocus_varies_with_depth():
    """A tilted specimen should produce depth-dependent defocus changes."""
    tomo = tomo_cmd.Tomogram(
        pixel_size=1.0,
        defocus_u=np.array([20000.0]),
        defocus_v=np.array([20000.0]),
        defocus_angle=np.zeros(1),
        x_tilts=np.array([30.0]),  # 30° tilt
        y_tilts=np.zeros(1),
        z_rots=np.zeros(1),
        x_shifts=np.zeros(1),
        y_shifts=np.zeros(1),
        hand=1,
    )
    # Two points at different Z positions
    dfu_near, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 100.0]))
    dfu_far, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, -100.0]))
    # They should differ due to depth
    assert dfu_near != pytest.approx(dfu_far, abs=1.0)


# ---------------------------------------------------------------------------
# Tomogram.expand_particles_batch – basic structure
# ---------------------------------------------------------------------------

def test_expand_particles_batch_returns_correct_number_of_rows():
    """expand_particles_batch should produce M * n_tilts rows."""
    n_tilts = 3
    tomo = _make_simple_tomogram(n_tilts=n_tilts)

    tilt_df = pd.DataFrame({
        '_rlnMicrographName': [f'mic_{i}.mrc' for i in range(n_tilts)],
        '_rlnMicrographPreExposure': np.arange(n_tilts, dtype=float),
        '_rlnCtfScalefactor': np.ones(n_tilts),
        '_rlnTomoNominalStageTiltAngle': np.zeros(n_tilts),
    })

    df_2d = tomo.expand_particles_batch(
        points_3d=np.array([[0.0, 0.0, 0.0]]),
        image_names=["particles.mrcs"],
        tilt_df=tilt_df,
        group_names=["particle_001"],
        base_orientations=None,
        random_subsets=np.array([1]),
    )

    assert len(df_2d) == n_tilts
    assert '_rlnDefocusU' in df_2d.columns
    assert '_rlnImageName' in df_2d.columns
    assert '_rlnGroupName' in df_2d.columns
    assert all(df_2d['_rlnGroupName'] == "particle_001")


def test_expand_particles_batch_multiple_particles():
    """Batch of 3 particles should produce 3 * n_tilts rows."""
    n_tilts = 4
    tomo = _make_simple_tomogram(n_tilts=n_tilts)

    tilt_df = pd.DataFrame({
        '_rlnMicrographName': [f'mic_{i}.mrc' for i in range(n_tilts)],
        '_rlnMicrographPreExposure': np.arange(n_tilts, dtype=float),
        '_rlnCtfScalefactor': np.ones(n_tilts),
        '_rlnTomoNominalStageTiltAngle': np.zeros(n_tilts),
    })

    M = 3
    R_batch = R.from_matrix(np.broadcast_to(np.eye(3), (M, 3, 3)).copy())

    df_2d = tomo.expand_particles_batch(
        points_3d=np.zeros((M, 3)),
        image_names=[f"p{i}.mrcs" for i in range(M)],
        tilt_df=tilt_df,
        group_names=[f"group_{i}" for i in range(M)],
        base_orientations=R_batch,
        random_subsets=np.array([1, 2, 1]),
    )

    assert len(df_2d) == M * n_tilts
    # Check each group appears n_tilts times
    for i in range(M):
        assert (df_2d['_rlnGroupName'] == f"group_{i}").sum() == n_tilts
    # Check random subsets preserved
    for i in range(M):
        subset_vals = df_2d[df_2d['_rlnGroupName'] == f"group_{i}"]['_rlnRandomSubset']
        assert all(subset_vals == [1, 2, 1][i])
