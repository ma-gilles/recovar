"""Regression test for cryoSPARC .cs file parsing using real data subset.

Tests that parse_poses_from_cs and parse_ctf_from_cs produce identical
output on a small fixture extracted from a real HRA 2025 dataset.

Fixture: tests/fixtures/cryosparc_cs_subset/
  - particles_20.cs.npy (20 particles, 54 fields from J877 exported)
  - reference_values.npz (expected rotations, translations, CTF)
"""

import os

import numpy as np
import pytest

from recovar.data_io.metadata_readers import (
    parse_poses_from_cs,
    parse_ctf_from_cs,
    _load_cs,
)

pytestmark = pytest.mark.unit

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "cryosparc_cs_subset")

CS_PATH = os.path.join(FIXTURE_DIR, "particles_20.cs.npy")
REF_PATH = os.path.join(FIXTURE_DIR, "reference_values.npz")


@pytest.fixture
def ref():
    return np.load(REF_PATH)


def test_fixture_exists():
    assert os.path.isfile(CS_PATH), f"Missing fixture: {CS_PATH}"
    assert os.path.isfile(REF_PATH), f"Missing fixture: {REF_PATH}"


def test_load_cs_structured_array():
    """_load_cs should return a structured array with named fields."""
    data = _load_cs(CS_PATH)
    assert data.dtype.names is not None
    assert len(data) == 20


def test_cs_has_required_pose_fields():
    data = _load_cs(CS_PATH)
    assert "alignments3D/pose" in data.dtype.names
    assert "alignments3D/shift" in data.dtype.names


def test_cs_has_required_ctf_fields():
    data = _load_cs(CS_PATH)
    for field in ("ctf/df1_A", "ctf/df2_A", "ctf/df_angle_rad", "ctf/accel_kv", "ctf/cs_mm", "ctf/amp_contrast"):
        assert field in data.dtype.names, f"Missing CTF field: {field}"


def test_parse_poses_shape(ref):
    D = int(ref["D"])
    rots, trans = parse_poses_from_cs(CS_PATH, D)
    assert rots.shape == (20, 3, 3)
    assert trans.shape == (20, 2)


def test_parse_poses_rotation_regression(ref):
    """Rotation matrices should match reference to machine precision."""
    D = int(ref["D"])
    rots, _ = parse_poses_from_cs(CS_PATH, D)
    np.testing.assert_allclose(
        rots,
        ref["rotations"],
        atol=1e-10,
        rtol=1e-12,
        err_msg="Rotation matrices differ from reference",
    )


def test_parse_poses_translation_regression(ref):
    """Translations should match reference to machine precision."""
    D = int(ref["D"])
    _, trans = parse_poses_from_cs(CS_PATH, D)
    np.testing.assert_allclose(
        trans,
        ref["translations"],
        atol=1e-10,
        rtol=1e-12,
        err_msg="Translations differ from reference",
    )


def test_parse_poses_translation_downsample_invariant(ref):
    """Fractional translations should not change when target D is downsampled."""
    _, trans = parse_poses_from_cs(CS_PATH, 128)
    np.testing.assert_allclose(
        trans,
        ref["translations"],
        atol=1e-10,
        rtol=1e-12,
        err_msg="Translations should use the original CS image size, not target D",
    )


def test_parse_poses_rotations_are_orthogonal(ref):
    """All rotation matrices should be proper rotations (det=1, R^T R = I)."""
    D = int(ref["D"])
    rots, _ = parse_poses_from_cs(CS_PATH, D)
    for i in range(len(rots)):
        R = rots[i]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_parse_ctf_shape(ref):
    D = int(ref["D"])
    ctf = parse_ctf_from_cs(CS_PATH, D)
    assert ctf.shape == (20, 8)


def test_parse_ctf_regression(ref):
    """CTF parameters should match reference to machine precision."""
    D = int(ref["D"])
    ctf = parse_ctf_from_cs(CS_PATH, D)
    np.testing.assert_allclose(
        ctf,
        ref["ctf"],
        atol=1e-10,
        rtol=1e-12,
        err_msg="CTF parameters differ from reference",
    )


def test_parse_ctf_values_reasonable(ref):
    """CTF values should be in physically reasonable ranges."""
    D = int(ref["D"])
    ctf = parse_ctf_from_cs(CS_PATH, D)
    apix = ctf[:, 0]
    dfu = ctf[:, 1]
    dfv = ctf[:, 2]
    volt = ctf[:, 4]
    cs_mm = ctf[:, 5]
    amp = ctf[:, 6]

    assert (apix > 0.5).all() and (apix < 10).all(), f"Apix out of range: {apix}"
    assert (dfu > 1000).all() and (dfu < 100000).all(), f"DFU out of range"
    assert (dfv > 1000).all() and (dfv < 100000).all(), f"DFV out of range"
    assert (volt > 100).all() and (volt < 400).all(), f"Voltage out of range"
    assert (cs_mm >= 0).all() and (cs_mm < 5).all(), f"Cs out of range"
    assert (amp > 0).all() and (amp < 0.5).all(), f"Amp contrast out of range"
