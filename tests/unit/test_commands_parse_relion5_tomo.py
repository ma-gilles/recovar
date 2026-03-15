"""
Unit tests for recovar.commands.parse_relion5_tomo.

Covers:
  add_args()               – CLI argument registration
  _parse_visible_frames()  – frame selection parsing
  Tomogram geometry        – translation/rotation matrices, projection, local defocus
"""
import argparse

import numpy as np
import pytest

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


def test_registers_tilt_dim():
    actions = _parser()._option_string_actions
    assert "--tilt-dim" in actions


def test_tilt_dim_accepts_two_ints():
    parser = _parser()
    args = parser.parse_args([
        "-t", "tomo.star",
        "-p", "particles.star",
        "--tilt-dim", "3710", "3838",
    ])
    assert args.tilt_dim == [3710, 3838]


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
# Tomogram – geometry helpers
# ---------------------------------------------------------------------------

def test_translation_matrix_identity_at_zero():
    mat = tomo_cmd.Tomogram._translation_matrix(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(mat, np.eye(4))


def test_translation_matrix_shifts_correctly():
    shift = np.array([10.0, -5.0, 3.0])
    mat = tomo_cmd.Tomogram._translation_matrix(shift)
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat[:3, :3], np.eye(3))
    np.testing.assert_allclose(mat[:3, 3], shift)
    np.testing.assert_allclose(mat[3, :], [0, 0, 0, 1])


def test_rotation_matrix_identity_at_zero_angle():
    mat = tomo_cmd.Tomogram._rotation_matrix([1, 0, 0], 0.0)
    np.testing.assert_allclose(mat, np.eye(4), atol=1e-10)


def test_rotation_matrix_90_degrees_around_z():
    mat = tomo_cmd.Tomogram._rotation_matrix([0, 0, 1], 90.0)
    assert mat.shape == (4, 4)
    # After 90° around z: x -> y, y -> -x
    point = mat[:3, :3] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(point, [0.0, 1.0, 0.0], atol=1e-10)


def test_rotation_matrix_preserves_homogeneous_structure():
    mat = tomo_cmd.Tomogram._rotation_matrix([1, 0, 0], 45.0)
    np.testing.assert_allclose(mat[3, :], [0, 0, 0, 1])
    np.testing.assert_allclose(mat[:3, 3], [0, 0, 0])


# ---------------------------------------------------------------------------
# Tomogram – projection and defocus
# ---------------------------------------------------------------------------

def _make_simple_tomogram(n_tilts=3):
    """Create a Tomogram with simple geometry for testing."""
    return tomo_cmd.Tomogram(
        tilt_image_dims=(100, 100),
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


def test_tomogram_init_creates_projection_matrices():
    tomo = _make_simple_tomogram(n_tilts=5)
    assert tomo.n_tilts == 5
    assert len(tomo.projection_matrices) == 5
    for i in range(5):
        assert tomo.projection_matrices[i].shape == (4, 4)


def test_project_point_at_origin_maps_to_image_center():
    """With zero tilts/shifts, the origin should project to the image center."""
    tomo = _make_simple_tomogram(n_tilts=1)
    pt_2d = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
    # Image center = tilt_image_dims / 2 = (50, 50)
    np.testing.assert_allclose(pt_2d, [50.0, 50.0], atol=1e-6)


def test_project_point_with_shift():
    """X/Y shifts should offset the projected position."""
    tomo = tomo_cmd.Tomogram(
        tilt_image_dims=(200, 200),
        pixel_size=1.0,
        defocus_u=np.array([10000.0]),
        defocus_v=np.array([10000.0]),
        defocus_angle=np.zeros(1),
        x_tilts=np.zeros(1),
        y_tilts=np.zeros(1),
        z_rots=np.zeros(1),
        x_shifts=np.array([10.0]),
        y_shifts=np.array([-5.0]),
        hand=1,
    )
    pt_2d = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
    np.testing.assert_allclose(pt_2d, [110.0, 95.0], atol=1e-6)


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
        tilt_image_dims=(100, 100),
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
# Tomogram.expand_particle – basic structure
# ---------------------------------------------------------------------------

def test_expand_particle_returns_correct_number_of_rows():
    """expand_particle should produce one row per tilt."""
    import pandas as pd

    n_tilts = 3
    tomo = _make_simple_tomogram(n_tilts=n_tilts)

    tilt_df = pd.DataFrame({
        '_rlnMicrographName': [f'mic_{i}.mrc' for i in range(n_tilts)],
        '_rlnMicrographPreExposure': np.arange(n_tilts, dtype=float),
        '_rlnCtfScalefactor': np.ones(n_tilts),
        '_rlnTomoNominalStageTiltAngle': np.zeros(n_tilts),
    })

    df_2d = tomo.expand_particle(
        point_3d=np.array([0.0, 0.0, 0.0]),
        image_name="particles.mrcs",
        tilt_df=tilt_df,
        group_name="particle_001",
        base_orientation=None,
    )

    assert len(df_2d) == n_tilts
    assert '_rlnDefocusU' in df_2d.columns
    assert '_rlnImageName' in df_2d.columns
    assert '_rlnGroupName' in df_2d.columns
    assert all(df_2d['_rlnGroupName'] == "particle_001")
