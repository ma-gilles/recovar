import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar import constants
from recovar import core
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


def test_relion_style_kernel_batch_normalizes_noise_variance_shapes(monkeypatch):
    # Keep this test focused on noise-shape normalization behavior.
    monkeypatch.setattr(core, "translate_images", lambda images, translations, image_shape: images)
    monkeypatch.setattr(
        core,
        "adjoint_forward_model_from_map",
        lambda images, *args, **kwargs: jnp.ones((64,), dtype=jnp.complex64) * jnp.sum(images),
    )

    def ctf_fun(params, image_shape, voxel_size):
        return jnp.ones((params.shape[0], image_shape[0] * image_shape[1]), dtype=jnp.float32)

    bsz = 5
    images = jnp.ones((bsz, 16), dtype=jnp.complex64)
    ctf_params = jnp.zeros((bsz, 9), dtype=jnp.float32)
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.zeros((bsz, 2), dtype=jnp.float32)

    candidate_noises = [
        jnp.ones((16,), dtype=jnp.float32),
        jnp.ones((4, 4), dtype=jnp.float32),
        jnp.ones((1, 4, 4), dtype=jnp.float32),
        jnp.ones((bsz, 16), dtype=jnp.float32),
    ]
    for noise_var in candidate_noises:
        ft_y, ft_ctf = rf.relion_style_triangular_kernel_batch(
            images,
            ctf_params,
            rots,
            trans,
            (4, 4),
            (4, 4, 4),
            1.5,
            ctf_fun,
            "linear_interp",
            noise_var,
            False,
            False,
        )
        assert np.asarray(ft_y).shape == (64,)
        assert np.asarray(ft_ctf).shape == (64,)
        assert np.isfinite(np.asarray(ft_y)).all()
