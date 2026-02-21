import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import ewald

pytestmark = pytest.mark.unit


def test_parse_disc_type_values_and_error():
    assert ewald.parse_disc_type("nearest") == 0
    assert ewald.parse_disc_type("linear_interp") == 1
    assert ewald.parse_disc_type("cubic") == 3
    with pytest.raises(ValueError, match="not recognized"):
        ewald.parse_disc_type("bad")


def test_get_flipped_indices_shape_and_bad_nyquist_markers():
    idx = np.asarray(ewald.get_flipped_indices((4, 4)))
    assert idx.shape == (16,)
    # Implementation marks ambiguous Nyquist flips as -1.
    assert np.any(idx == -1)


def test_volt_to_wavelength_positive_and_decreases_with_voltage():
    lam_200 = float(ewald.volt_to_wavelength(np.array(200.0, dtype=np.float32)))
    lam_300 = float(ewald.volt_to_wavelength(np.array(300.0, dtype=np.float32)))
    assert lam_200 > 0 and lam_300 > 0
    assert lam_300 < lam_200


def test_vec_unvec_masked_roundtrip_on_masked_entries():
    shape = (4, 4, 4)
    vol_size = np.prod(shape)
    vr = np.linspace(0.1, 2.0, vol_size).astype(np.float32)
    vi = np.linspace(-1.0, 1.0, vol_size).astype(np.float32)
    x = ewald.vec_masked(vr, vi, shape)
    mask_real_idx, mask_imag_idx = ewald.get_good_idx_mask(shape)
    mask_size = int(np.asarray(mask_real_idx[0]).size)
    vr2, vi2 = ewald.unvec_masked(x, shape, mask_size)
    vr2 = np.asarray(vr2)
    vi2 = np.asarray(vi2)
    np.testing.assert_allclose(vr2[np.asarray(mask_real_idx[0])], vr[np.asarray(mask_real_idx[0])])
    np.testing.assert_allclose(vi2[np.asarray(mask_imag_idx[0])], vi[np.asarray(mask_imag_idx[0])])
