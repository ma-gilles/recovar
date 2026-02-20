import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.mask as mask

pytestmark = pytest.mark.unit


def test_get_radial_mask_and_window_mask():
    m = np.asarray(mask.get_radial_mask((8, 8, 8)))
    assert m.shape == (8, 8, 8)
    assert m.dtype == np.bool_
    assert m.any()

    w = mask.window_mask(8, in_rad=0.2, out_rad=0.8)
    assert w.shape == (8, 8)
    assert np.all((w >= 0) & (w <= 1))


def test_raised_cosine_mask_bounds():
    rc = np.asarray(mask.raised_cosine_mask((8, 8, 8), radius=2, radius_p=4, offset=-1))
    assert rc.shape == (8, 8, 8)
    assert np.all((rc >= 0) & (rc <= 1))
    assert np.max(rc) <= 1.0 + 1e-6


def test_soften_volume_mask_new_range():
    binary = np.zeros((8, 8, 8), dtype=np.float32)
    binary[2:6, 2:6, 2:6] = 1.0
    out = mask.soften_volume_mask_new(binary, kernel_size=2)
    assert out.shape == binary.shape
    assert out.dtype == np.float32
    assert np.all((out >= 0) & (out <= 1))
