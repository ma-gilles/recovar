import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import constants
from recovar import relion_functions as rf


def test_gridding_correct_invalid_order_raises():
    vol = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        rf.griddingCorrect(vol, ori_size=4, padding_factor=2, order=9)


def test_gridding_correct_variants_return_finite():
    vol = np.ones((6, 6, 6), dtype=np.float32)
    out0, s0 = rf.griddingCorrect(vol, ori_size=6, padding_factor=2, order=0)
    out1, s1 = rf.griddingCorrect_square(vol, ori_size=6, padding_factor=2, order=1)
    assert out0.shape == vol.shape
    assert out1.shape == vol.shape
    assert np.isfinite(np.asarray(out0)).all()
    assert np.isfinite(np.asarray(s0)).all()
    assert np.isfinite(np.asarray(out1)).all()
    assert np.isfinite(np.asarray(s1)).all()


def test_upscale_tau_shape_and_values():
    tau_1d = np.linspace(1.0, 2.0, 16, dtype=np.float32)
    out = rf.upscale_tau(tau_1d, padding_factor=2, volume_shape=(4, 4, 4), tau_is_1d=True)
    assert out.shape == (8 * 8 * 8,)
    assert float(np.min(np.asarray(out))) >= 1.0
    assert float(np.max(np.asarray(out))) <= 2.0


def test_adjust_regularization_relion_style_lower_bounded():
    filt = np.zeros((4, 4, 4), dtype=np.float32)
    reg = rf.adjust_regularization_relion_style(filt, volume_shape=(4, 4, 4))
    reg_np = np.asarray(reg)
    assert reg_np.shape == (4, 4, 4)
    assert (reg_np >= constants.EPSILON).all()

