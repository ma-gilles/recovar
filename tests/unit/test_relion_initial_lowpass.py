"""Numerical contracts for RELION-compatible initial low-pass filtering."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.em.refinement.mean_helpers import (
    _apply_relion_initial_lowpass_filter,
    initial_low_pass_filter_references,
)
from scripts.run_full_refinement import _read_relion_mrc_model_pixel_size

pytestmark = pytest.mark.unit


def test_model_pixel_size_uses_binary64_header_division(tmp_path):
    import mrcfile

    path = tmp_path / "model.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(np.zeros((6, 6, 6), dtype=np.float32))
        handle.header.cella.x = 17.0
        handle.header.cella.y = 17.0
        handle.header.cella.z = 17.0
        handle.header.mx = 12
        handle.header.my = 12
        handle.header.mz = 12

    expected = 17.0 / 12.0
    assert _read_relion_mrc_model_pixel_size(path) == expected
    with mrcfile.open(path, permissive=False) as handle:
        assert float(handle.voxel_size.x) != expected


def test_centered_fft_wrapper_matches_relion_lowpass():
    rng = np.random.default_rng(17)
    volume = rng.normal(size=(1, 8, 8, 8)).astype(np.float32)
    volume_ft = np.asarray(ftu.get_dft3(jnp.asarray(volume[0]))).reshape(-1)

    actual_ft = _apply_relion_initial_lowpass_filter(
        volume_ft,
        (8, 8, 8),
        voxel_size=1.25,
        ini_high_angstrom=4.0,
        filter_edgewidth=2.0,
    )
    actual = np.asarray(ftu.get_idft3(jnp.asarray(actual_ft).reshape(8, 8, 8))).real
    expected = initial_low_pass_filter_references(
        volume.astype(np.float64),
        ori_size=8,
        pixel_size=1.25,
        ini_high_ang=4.0,
        filter_edgewidth=2.0,
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
