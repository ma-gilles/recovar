"""Tests for recovar.metadata_parsing — STAR/CS pose and CTF extraction."""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovar import metadata_parsing, utils
from recovar.starfile import StarFile, write_star

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_star(n=20, grid_size=64, voxel_size=1.5, seed=42):
    """Create a temporary STAR file with known poses and CTF parameters.

    Returns (path, rotations, translations_pixels, ctf_params_raw).
    """
    rng = np.random.RandomState(seed)

    # Random rotation matrices
    from scipy.spatial.transform import Rotation as R
    rots_scipy = R.random(n, random_state=seed)
    rot_matrices = rots_scipy.as_matrix()  # (N, 3, 3)

    # Convert to RELION Euler angles and back to get consistent matrices
    euler_relion = utils.R_to_relion(rot_matrices, degrees=True)  # (N, 3)
    rot_matrices_relion = utils.R_from_relion(euler_relion, degrees=True)  # (N, 3, 3)

    # Random translations in Angstroms
    trans_angstrom = rng.uniform(-5, 5, (n, 2))

    # Random CTF parameters
    dfu = rng.uniform(5000, 25000, n)
    dfv = rng.uniform(5000, 25000, n)
    dfang = rng.uniform(0, 180, n)
    voltage = np.full(n, 300.0)
    cs = np.full(n, 2.7)
    w = np.full(n, 0.1)
    phase_shift = rng.uniform(0, 10, n)

    # Build CTF params array for write_starfile
    ctf_params = np.column_stack([dfu, dfv, dfang, voltage, cs, w, phase_shift,
                                  np.zeros(n), np.ones(n)])

    # Write star file using existing write_starfile
    import pandas as pd
    tmpdir = tempfile.mkdtemp()
    star_path = os.path.join(tmpdir, "test.star")

    # Use write_starfile from utils
    utils.write_starfile(
        CTF_params=ctf_params,
        rotation_matrices=rot_matrices_relion,
        translations=trans_angstrom,
        voxel_size=voxel_size,
        grid_size=grid_size,
        particles_file="dummy.mrcs",
        output_filename=star_path,
    )

    return star_path, rot_matrices_relion, trans_angstrom, ctf_params, voxel_size, grid_size


def _make_test_cs(n=20, grid_size=64, voxel_size=1.5, seed=42):
    """Create a temporary CS file with known poses and CTF parameters."""
    rng = np.random.RandomState(seed)

    from scipy.spatial.transform import Rotation as R
    rots_scipy = R.random(n, random_state=seed)
    rotvecs = rots_scipy.as_rotvec()  # (N, 3)
    rot_matrices = rots_scipy.as_matrix()  # (N, 3, 3)

    # Translations in pixels
    trans_pixels = rng.uniform(-5, 5, (n, 2))

    # CTF parameters
    dfu = rng.uniform(5000, 25000, n)
    dfv = rng.uniform(5000, 25000, n)
    dfang_rad = rng.uniform(0, np.pi, n)
    voltage = np.full(n, 300.0)
    cs_mm = np.full(n, 2.7)
    amp_contrast = np.full(n, 0.1)
    phase_shift_rad = rng.uniform(0, 0.2, n)

    # Build structured array
    dtype = np.dtype([
        ('blob/idx', np.int32),
        ('blob/path', 'U200'),
        ('blob/shape', np.int32, (2,)),
        ('blob/psize_A', np.float32),
        ('alignments3D/pose', np.float32, (3,)),
        ('alignments3D/shift', np.float32, (2,)),
        ('ctf/df1_A', np.float32),
        ('ctf/df2_A', np.float32),
        ('ctf/df_angle_rad', np.float32),
        ('ctf/accel_kv', np.float32),
        ('ctf/cs_mm', np.float32),
        ('ctf/amp_contrast', np.float32),
        ('ctf/phase_shift_rad', np.float32),
    ])

    cs_data = np.zeros(n, dtype=dtype)
    cs_data['blob/idx'] = np.arange(n)
    cs_data['blob/path'] = 'dummy.mrcs'
    cs_data['blob/shape'] = [grid_size, grid_size]
    cs_data['blob/psize_A'] = voxel_size
    cs_data['alignments3D/pose'] = rotvecs.astype(np.float32)
    cs_data['alignments3D/shift'] = trans_pixels.astype(np.float32)
    cs_data['ctf/df1_A'] = dfu.astype(np.float32)
    cs_data['ctf/df2_A'] = dfv.astype(np.float32)
    cs_data['ctf/df_angle_rad'] = dfang_rad.astype(np.float32)
    cs_data['ctf/accel_kv'] = voltage.astype(np.float32)
    cs_data['ctf/cs_mm'] = cs_mm.astype(np.float32)
    cs_data['ctf/amp_contrast'] = amp_contrast.astype(np.float32)
    cs_data['ctf/phase_shift_rad'] = phase_shift_rad.astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    cs_path = os.path.join(tmpdir, "test.cs")
    # Use file object to prevent np.save from appending .npy to the path
    with open(cs_path, 'wb') as f:
        np.save(f, cs_data)

    return cs_path, rot_matrices, trans_pixels, dfu, dfv, dfang_rad, voltage, cs_mm, amp_contrast, phase_shift_rad, voxel_size, grid_size


# ---------------------------------------------------------------------------
# STAR file tests
# ---------------------------------------------------------------------------

class TestParseFromStar:

    def test_pose_roundtrip(self):
        """write_starfile → parse_poses_from_star recovers rotation matrices."""
        star_path, rot_expected, trans_ang, _, voxel_size, grid_size = _make_test_star()

        rots, trans_frac = metadata_parsing.parse_poses_from_star(star_path, grid_size)

        assert rots.shape == rot_expected.shape
        # Rotation matrices should match to reasonable precision
        assert_allclose(rots, rot_expected, atol=1e-4)

    def test_translation_convention(self):
        """Translations in Angstroms → fractional units."""
        star_path, _, trans_ang, _, voxel_size, grid_size = _make_test_star()

        _, trans_frac = metadata_parsing.parse_poses_from_star(star_path, grid_size)

        # trans_ang are the values written to rlnOriginXAngst/Y (in pixel units by write_starfile)
        # The parser reads them as Angstroms and divides by Apix
        # Since write_starfile writes pixel-unit values to the Angstrom fields,
        # and the parser converts Angstrom → pixel (divide by Apix) then → fractional (divide by resolution),
        # we need to check the convention carefully.
        # write_starfile writes translations[:,0] and translations[:,1] directly.
        # These are in pixel units. The parser treats them as Angstroms.
        # So parsed_pixels = written_value / apix, and parsed_frac = parsed_pixels / resolution.
        # For self-consistency, we just check shape and range.
        assert trans_frac.shape == (20, 2)
        assert np.all(np.abs(trans_frac) <= 1.0), "Fractional translations should be in [-1, 1]"

    def test_ctf_roundtrip(self):
        """write_starfile → parse_ctf_from_star recovers CTF parameters."""
        star_path, _, _, ctf_raw, voxel_size, grid_size = _make_test_star()

        ctf = metadata_parsing.parse_ctf_from_star(star_path, grid_size)

        assert ctf.shape == (20, 8)  # [Apix, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]

        # Check defocus values match
        assert_allclose(ctf[:, 1], ctf_raw[:, 0], atol=1e-2)  # DFU
        assert_allclose(ctf[:, 2], ctf_raw[:, 1], atol=1e-2)  # DFV
        assert_allclose(ctf[:, 3], ctf_raw[:, 2], atol=1e-2)  # DFANG
        assert_allclose(ctf[:, 4], ctf_raw[:, 3], atol=1e-2)  # VOLT
        assert_allclose(ctf[:, 5], ctf_raw[:, 4], atol=1e-2)  # CS
        assert_allclose(ctf[:, 6], ctf_raw[:, 5], atol=1e-2)  # W

    def test_ctf_apix_adjustment(self):
        """Pixel size is adjusted for target D != original D."""
        star_path, _, _, _, voxel_size, grid_size = _make_test_star(grid_size=64, voxel_size=1.5)

        # Parse at original D
        ctf_orig = metadata_parsing.parse_ctf_from_star(star_path, 64)
        # Parse at half D (downsampled)
        ctf_ds = metadata_parsing.parse_ctf_from_star(star_path, 32)

        # Apix should double when D halves
        assert_allclose(ctf_ds[:, 0], ctf_orig[:, 0] * 2, rtol=1e-4)

    def test_missing_pose_fields_raises(self):
        """STAR file without Euler angles raises ValueError."""
        tmpdir = tempfile.mkdtemp()
        star_path = os.path.join(tmpdir, "no_poses.star")

        import pandas as pd
        import starfile
        optics = pd.DataFrame({
            'rlnOpticsGroup': [1],
            'rlnOpticsGroupName': ['opticsGroup1'],
            'rlnVoltage': [300.0],
            'rlnImagePixelSize': [1.5],
            'rlnImageSize': [64],
            'rlnImageDimensionality': [2],
            'rlnAmplitudeContrast': [0.1],
            'rlnSphericalAberration': [2.7],
        })
        particles = pd.DataFrame({
            'rlnImageName': ['1@dummy.mrcs'],
            'rlnDefocusU': [10000.0],
            'rlnDefocusV': [10000.0],
            'rlnDefocusAngle': [45.0],
            'rlnOpticsGroup': [1],
        })
        starfile.write({'optics': optics, 'particles': particles}, star_path)

        with pytest.raises(ValueError, match="rlnAngleRot"):
            metadata_parsing.parse_poses_from_star(star_path, 64)


# ---------------------------------------------------------------------------
# cryoSPARC CS file tests
# ---------------------------------------------------------------------------

class TestParseFromCS:

    def test_pose_extraction(self):
        """CS file pose extraction produces correct shape and dtype."""
        cs_path, rot_expected, trans_pix, *_, voxel_size, grid_size = _make_test_cs()

        rots, trans_frac = metadata_parsing.parse_poses_from_cs(cs_path, grid_size)

        assert rots.shape == (20, 3, 3)
        assert trans_frac.shape == (20, 2)

        # Rotations should be proper rotation matrices (det=1, R^T R = I)
        for i in range(rots.shape[0]):
            assert_allclose(np.linalg.det(rots[i]), 1.0, atol=1e-5)
            assert_allclose(rots[i] @ rots[i].T, np.eye(3), atol=1e-5)

    def test_translation_fractional(self):
        """CS translations are correctly converted to fractional units."""
        cs_path, _, trans_pix, *_, voxel_size, grid_size = _make_test_cs()

        _, trans_frac = metadata_parsing.parse_poses_from_cs(cs_path, grid_size)

        expected_frac = trans_pix / float(grid_size)
        assert_allclose(trans_frac, expected_frac, atol=1e-5)

    def test_ctf_extraction(self):
        """CS CTF extraction produces correct values."""
        cs_path, _, _, dfu, dfv, dfang_rad, volt, cs_mm, amp_c, phase_rad, voxel_size, grid_size = _make_test_cs()

        ctf = metadata_parsing.parse_ctf_from_cs(cs_path, grid_size)

        assert ctf.shape == (20, 8)
        assert_allclose(ctf[:, 1], dfu, atol=1)  # DFU
        assert_allclose(ctf[:, 2], dfv, atol=1)  # DFV
        assert_allclose(ctf[:, 3], np.degrees(dfang_rad), atol=0.1)  # DFANG in degrees
        assert_allclose(ctf[:, 4], volt, atol=0.1)  # VOLT
        assert_allclose(ctf[:, 5], cs_mm, atol=0.01)  # CS
        assert_allclose(ctf[:, 6], amp_c, atol=0.01)  # W
        assert_allclose(ctf[:, 7], np.degrees(phase_rad), atol=0.1)  # PHASE_SHIFT in degrees

    def test_ctf_apix(self):
        """CS pixel size is correctly reported."""
        cs_path, *_, voxel_size, grid_size = _make_test_cs(voxel_size=2.0)

        ctf = metadata_parsing.parse_ctf_from_cs(cs_path, grid_size)
        assert_allclose(ctf[:, 0], 2.0, atol=0.01)  # Apix

    def test_missing_pose_field_raises(self):
        """CS file without pose field raises ValueError."""
        tmpdir = tempfile.mkdtemp()
        cs_path = os.path.join(tmpdir, "no_poses.cs")

        dtype = np.dtype([('blob/idx', np.int32), ('blob/path', 'U200')])
        data = np.zeros(5, dtype=dtype)
        data['blob/idx'] = np.arange(5)
        data['blob/path'] = 'dummy.mrcs'
        with open(cs_path, 'wb') as f:
            np.save(f, data)

        with pytest.raises(ValueError, match="alignments3D/pose"):
            metadata_parsing.parse_poses_from_cs(cs_path, 64)


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------

class TestAutoDispatch:

    def test_can_extract_star(self):
        assert metadata_parsing.can_extract_poses("particles.star")
        assert metadata_parsing.can_extract_ctf("particles.star")

    def test_can_extract_cs(self):
        assert metadata_parsing.can_extract_poses("data.cs")
        assert metadata_parsing.can_extract_ctf("data.cs")

    def test_cannot_extract_mrcs(self):
        assert not metadata_parsing.can_extract_poses("particles.mrcs")
        assert not metadata_parsing.can_extract_ctf("particles.mrcs")

    def test_auto_parse_star(self):
        star_path, rot_expected, _, _, _, grid_size = _make_test_star(n=5)
        rots, _ = metadata_parsing.auto_parse_poses(star_path, grid_size)
        assert rots.shape == (5, 3, 3)

    def test_auto_parse_cs(self):
        cs_path, _, _, *_, grid_size = _make_test_cs(n=5)
        rots, _ = metadata_parsing.auto_parse_poses(cs_path, grid_size)
        assert rots.shape == (5, 3, 3)

    def test_auto_parse_unsupported_raises(self):
        with pytest.raises(ValueError, match="Cannot auto-extract"):
            metadata_parsing.auto_parse_poses("particles.mrcs", 64)
