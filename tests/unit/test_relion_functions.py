import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar import jax_config
from recovar import core
from recovar.reconstruction import relion_functions as rf
from recovar.core.configs import BatchData


def _make_batch_data(images, ctf_params, rots, trans, noise_var):
    return BatchData(
        images=images,
        ctf_params=ctf_params,
        rotation_matrices=rots,
        translations=trans,
        noise_variance=noise_var,
    )


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
    assert (reg_np >= jax_config.EPSILON).all()


def test_relion_kernel_batch_normalizes_noise_variance_shapes():
    from recovar.core.configs import ForwardModelConfig

    def ctf_fun(params, image_shape, voxel_size):
        return jnp.ones((params.shape[0], image_shape[0] * image_shape[1]), dtype=jnp.float32)

    config = ForwardModelConfig(
        image_shape=(4, 4), volume_shape=(4, 4, 4),
        grid_size=4, voxel_size=1.5,
        padding=0, disc_type="linear_interp", CTF_fun=ctf_fun,
        premultiplied_ctf=False, volume_mask_threshold=0.0,
    )

    bsz = 5
    images = jnp.ones((bsz, 16), dtype=jnp.complex64)
    ctf_params = jnp.zeros((bsz, 9), dtype=jnp.float32)
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.zeros((bsz, 2), dtype=jnp.float32)

    # relion_kernel_batch_from_fft accumulates in half-volume layout: 4*4*(4//2+1) = 48
    half_vol_size = 4 * 4 * (4 // 2 + 1)
    candidate_noises = [
        jnp.ones((16,), dtype=jnp.float32),
        jnp.ones((4, 4), dtype=jnp.float32),
        jnp.ones((1, 4, 4), dtype=jnp.float32),
        jnp.ones((bsz, 16), dtype=jnp.float32),
    ]
    for noise_var in candidate_noises:
        ft_y, ft_ctf = rf.relion_kernel_batch_from_fft(
            config, _make_batch_data(images, ctf_params, rots, trans, noise_var),
        )
        assert np.asarray(ft_y).shape == (half_vol_size,)
        assert np.asarray(ft_ctf).shape == (half_vol_size,)
        assert np.isfinite(np.asarray(ft_y)).all()


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax


@pytest.mark.gpu
def test_gridding_correct_gpu(gpu_device):
    vol = np.ones((6, 6, 6), dtype=np.float32)

    cpu_out0, cpu_s0 = rf.griddingCorrect(vol, ori_size=6, padding_factor=2, order=0)
    cpu_out0, cpu_s0 = np.asarray(cpu_out0), np.asarray(cpu_s0)

    with jax.default_device(gpu_device):
        vol_g = jax.device_put(jnp.array(vol), gpu_device)
        gpu_out0, gpu_s0 = rf.griddingCorrect(vol_g, ori_size=6, padding_factor=2, order=0)
        gpu_out0, gpu_s0 = np.asarray(gpu_out0), np.asarray(gpu_s0)

    np.testing.assert_allclose(cpu_out0, gpu_out0, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_s0, gpu_s0, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_gridding_correct_square_gpu(gpu_device):
    vol = np.ones((6, 6, 6), dtype=np.float32)

    cpu_out1, cpu_s1 = rf.griddingCorrect_square(vol, ori_size=6, padding_factor=2, order=1)
    cpu_out1, cpu_s1 = np.asarray(cpu_out1), np.asarray(cpu_s1)

    with jax.default_device(gpu_device):
        vol_g = jax.device_put(jnp.array(vol), gpu_device)
        gpu_out1, gpu_s1 = rf.griddingCorrect_square(vol_g, ori_size=6, padding_factor=2, order=1)
        gpu_out1, gpu_s1 = np.asarray(gpu_out1), np.asarray(gpu_s1)

    np.testing.assert_allclose(cpu_out1, gpu_out1, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_s1, gpu_s1, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_upscale_tau_gpu(gpu_device):
    tau_1d = np.linspace(1.0, 2.0, 16, dtype=np.float32)

    cpu_out = np.asarray(rf.upscale_tau(tau_1d, padding_factor=2, volume_shape=(4, 4, 4), tau_is_1d=True))

    with jax.default_device(gpu_device):
        tau_g = jax.device_put(jnp.array(tau_1d), gpu_device)
        gpu_out = np.asarray(rf.upscale_tau(tau_g, padding_factor=2, volume_shape=(4, 4, 4), tau_is_1d=True))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


def test_relion_kernel_batch_half_image_matches_full_reference():
    """New half-image relion_kernel_batch produces identical output to
    the old full-image path (padded_dft + translate + full adjoint)."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import cryodrgn_CTF
    from recovar.core import padding
    import recovar.core.forward as core_forward
    from recovar.reconstruction import noise as noise_mod

    rng = np.random.default_rng(42)
    grid_size = 8
    pad = 4
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3
    voxel_size = 1.5
    bsz = 3
    data_multiplier = -1.0

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=grid_size, voxel_size=voxel_size,
        padding=pad, disc_type="linear_interp", CTF_fun=cryodrgn_CTF,
        premultiplied_ctf=False, volume_mask_threshold=0.0,
        data_multiplier=data_multiplier,
    )

    raw_images = rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32)
    rots = np.stack([
        np.eye(3, dtype=np.float32),
        rng.standard_normal((3, 3)).astype(np.float32),
        rng.standard_normal((3, 3)).astype(np.float32),
    ])
    for i in range(1, bsz):
        q, _ = np.linalg.qr(rots[i])
        rots[i] = q * np.sign(np.linalg.det(q))

    trans = rng.standard_normal((bsz, 2)).astype(np.float32) * 0.5
    ctf_params = np.zeros((bsz, 9), dtype=np.float32)
    ctf_params[:, 0] = 300.0
    ctf_params[:, 1] = 2.7
    ctf_params[:, 2] = 0.1
    ctf_params[:, 3] = rng.uniform(1.0, 3.0, bsz)
    ctf_params[:, 4] = ctf_params[:, 3]
    ctf_params[:, 6] = 1.0
    ctf_params[:, 7] = 10.0
    noise_var = rng.uniform(0.5, 2.0, (bsz, np.prod(image_shape))).astype(np.float32)

    # --- Reference: old full-image path ---
    full_images = padding.padded_dft(jnp.array(raw_images) * data_multiplier, grid_size, pad)
    full_images = core.translate_images(full_images, jnp.array(trans), image_shape)
    noise_full = noise_mod.to_batched_pixel_noise(jnp.array(noise_var), image_shape, batch_size=bsz)
    full_images_normed = full_images / noise_full
    ref_Ft_y = np.asarray(core_forward.adjoint_forward_model(
        config, full_images_normed, jnp.array(ctf_params), jnp.array(rots),
        skip_ctf=False, volume=None, half_image=False,
    ))
    ctf_full = config.compute_ctf(jnp.array(ctf_params)) / noise_full
    ref_Ft_ctf = np.asarray(core_forward.adjoint_forward_model(
        config, ctf_full, jnp.array(ctf_params), jnp.array(rots),
        volume=None, half_image=False,
    ))

    # --- New: half-image path via relion_kernel_batch (outputs half-volume) ---
    import recovar.core.fourier_transform_utils as ftu
    half_Ft_y, half_Ft_ctf = rf.relion_kernel_batch(
        config, _make_batch_data(
            jnp.array(raw_images), jnp.array(ctf_params),
            jnp.array(rots), jnp.array(trans), jnp.array(noise_var),
        ),
    )
    new_Ft_y = np.asarray(ftu.half_volume_to_full_volume(half_Ft_y, volume_shape)).reshape(-1)
    new_Ft_ctf = np.asarray(ftu.half_volume_to_full_volume(half_Ft_ctf, volume_shape)).reshape(-1)
    np.testing.assert_allclose(new_Ft_y, ref_Ft_y, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(new_Ft_ctf, ref_Ft_ctf, atol=1e-4, rtol=1e-4)


def test_relion_kernel_batch_complex_input_matches_full_reference():
    """relion_kernel_batch_from_fft with pre-processed complex images
    matches the full-image reference."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import cryodrgn_CTF
    from recovar.core import padding
    import recovar.core.forward as core_forward
    from recovar.reconstruction import noise as noise_mod

    rng = np.random.default_rng(99)
    grid_size = 8
    pad = 4
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3
    bsz = 3

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=grid_size, voxel_size=1.5,
        padding=pad, disc_type="linear_interp", CTF_fun=cryodrgn_CTF,
        premultiplied_ctf=False, volume_mask_threshold=0.0,
    )

    raw_images = rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32)
    complex_images = padding.padded_dft(jnp.array(raw_images), grid_size, pad)
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.zeros((bsz, 2), dtype=jnp.float32)
    ctf_params = np.zeros((bsz, 9), dtype=np.float32)
    ctf_params[:, 0] = 300.0
    ctf_params[:, 1] = 2.7
    ctf_params[:, 2] = 0.1
    ctf_params[:, 3] = 2.0
    ctf_params[:, 4] = 2.0
    noise_var = np.ones((1, np.prod(image_shape)), dtype=np.float32)

    # --- Reference: full-image path ---
    full_images = core.translate_images(complex_images, trans, image_shape)
    noise_full = noise_mod.to_batched_pixel_noise(jnp.array(noise_var), image_shape, batch_size=bsz)
    full_normed = full_images / noise_full
    ref_Ft_y = np.asarray(core_forward.adjoint_forward_model(
        config, full_normed, jnp.array(ctf_params), rots,
        skip_ctf=False, volume=None, half_image=False,
    ))
    ctf_full = config.compute_ctf(jnp.array(ctf_params)) / noise_full
    ref_Ft_ctf = np.asarray(core_forward.adjoint_forward_model(
        config, ctf_full, jnp.array(ctf_params), rots, volume=None, half_image=False,
    ))

    # --- New: half-image path (outputs half-volume) ---
    import recovar.core.fourier_transform_utils as ftu
    half_Ft_y, half_Ft_ctf = rf.relion_kernel_batch_from_fft(
        config, _make_batch_data(complex_images, jnp.array(ctf_params), rots, trans, jnp.array(noise_var)),
    )
    new_Ft_y = np.asarray(ftu.half_volume_to_full_volume(half_Ft_y, volume_shape)).reshape(-1)
    new_Ft_ctf = np.asarray(ftu.half_volume_to_full_volume(half_Ft_ctf, volume_shape)).reshape(-1)
    np.testing.assert_allclose(new_Ft_y, ref_Ft_y, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(new_Ft_ctf, ref_Ft_ctf, atol=1e-4, rtol=1e-4)


def test_relion_kernel_batch_accumulator_matches_sequential():
    """Calling relion_kernel_batch twice with accumulators matches summing two independent calls."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import cryodrgn_CTF

    rng = np.random.default_rng(77)
    grid_size = 6
    pad = 2
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=grid_size, voxel_size=1.0,
        padding=pad, disc_type="linear_interp", CTF_fun=cryodrgn_CTF,
        premultiplied_ctf=False, volume_mask_threshold=0.0,
    )

    bsz = 2
    imgs1 = jnp.array(rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32))
    imgs2 = jnp.array(rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32))
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.zeros((bsz, 2), dtype=jnp.float32)
    ctf_params = jnp.zeros((bsz, 9), dtype=jnp.float32)
    noise_var = jnp.ones((1, np.prod(image_shape)), dtype=jnp.float32)

    bd1 = _make_batch_data(imgs1, ctf_params, rots, trans, noise_var)
    bd2 = _make_batch_data(imgs2, ctf_params, rots, trans, noise_var)

    # Two independent calls, sum result
    y1, c1 = rf.relion_kernel_batch(config, bd1)
    y2, c2 = rf.relion_kernel_batch(config, bd2)
    sum_Ft_y = np.asarray(y1) + np.asarray(y2)
    sum_Ft_ctf = np.asarray(c1) + np.asarray(c2)

    # Accumulated version
    acc_y, acc_c = rf.relion_kernel_batch(config, bd1)
    acc_y, acc_c = rf.relion_kernel_batch(config, bd2, Ft_y=acc_y, Ft_ctf=acc_c)

    np.testing.assert_allclose(np.asarray(acc_y), sum_Ft_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(acc_c), sum_Ft_ctf, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_relion_kernel_batch_half_volume_output_on_gpu(gpu_device):
    """relion_kernel_batch on GPU produces finite half-volume outputs."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import cryodrgn_CTF
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(55)
    grid_size = 8
    pad = 4
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3
    bsz = 3

    config = ForwardModelConfig(
        image_shape=image_shape, volume_shape=volume_shape,
        grid_size=grid_size, voxel_size=1.5,
        padding=pad, disc_type="linear_interp", CTF_fun=cryodrgn_CTF,
        premultiplied_ctf=False, volume_mask_threshold=0.0,
        data_multiplier=-1.0,
    )

    raw_images = jnp.array(rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32))
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.array(rng.standard_normal((bsz, 2)).astype(np.float32) * 0.3)
    ctf_params = jnp.zeros((bsz, 9), dtype=jnp.float32)
    ctf_params = ctf_params.at[:, 0].set(300.0)
    ctf_params = ctf_params.at[:, 1].set(2.7)
    ctf_params = ctf_params.at[:, 2].set(0.1)
    ctf_params = ctf_params.at[:, 3].set(2.0)
    ctf_params = ctf_params.at[:, 4].set(2.0)
    noise_var = jnp.ones((1, np.prod(image_shape)), dtype=jnp.float32)
    bd = _make_batch_data(raw_images, ctf_params, rots, trans, noise_var)

    with jax.default_device(gpu_device):
        half_y, half_c = rf.relion_kernel_batch(config, bd)

    # Expand and verify shape + finiteness
    full_y = np.asarray(ftu.half_volume_to_full_volume(half_y, volume_shape)).reshape(-1)
    full_c = np.asarray(ftu.half_volume_to_full_volume(half_c, volume_shape)).reshape(-1)
    assert full_y.shape == (int(np.prod(volume_shape)),)
    assert np.isfinite(full_y).all()
    assert np.isfinite(full_c).all()


@pytest.mark.gpu
def test_adjust_regularization_relion_style_gpu(gpu_device):
    filt = np.zeros((4, 4, 4), dtype=np.float32)

    cpu_out = np.asarray(rf.adjust_regularization_relion_style(filt, volume_shape=(4, 4, 4)))

    with jax.default_device(gpu_device):
        filt_g = jax.device_put(jnp.array(filt), gpu_device)
        gpu_out = np.asarray(rf.adjust_regularization_relion_style(filt_g, volume_shape=(4, 4, 4)))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)
