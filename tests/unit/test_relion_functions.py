import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar import jax_config
from recovar import core
from recovar.reconstruction import regularization
from recovar.reconstruction import relion_functions as rf


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


def test_zero_pad_fourier_volume_maps_centered_frequencies():
    native_shape = (4, 4, 4)
    padding_factor = 2
    padded_shape = tuple(dim * padding_factor for dim in native_shape)

    real = np.arange(np.prod(native_shape), dtype=np.float32).reshape(native_shape)
    imag = (1000 + np.arange(np.prod(native_shape), dtype=np.float32)).reshape(native_shape)
    native = real + 1j * imag

    padded = np.asarray(
        rf.zero_pad_fourier_volume(jnp.array(native.reshape(-1)), native_shape, padding_factor)
    ).reshape(padded_shape)

    mapped_axes = []
    for native_dim, padded_dim in zip(native_shape, padded_shape):
        start = padded_dim // 2 - padding_factor * (native_dim // 2)
        mapped_axes.append(start + padding_factor * np.arange(native_dim))

    recovered = padded[np.ix_(*mapped_axes)]
    np.testing.assert_array_equal(recovered, native)

    mask = np.zeros(padded_shape, dtype=bool)
    mask[np.ix_(*mapped_axes)] = True
    assert np.count_nonzero(padded[~mask]) == 0


def test_projection_padding_host_path_matches_device_path(monkeypatch):
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(141)
    volume_shape = (6, 6, 6)
    real = rng.standard_normal(volume_shape).astype(np.float32)
    ft = ftu.get_dft3(jnp.asarray(real)).reshape(-1)

    device_padded, device_shape = rf.pad_volume_for_projection(
        ft,
        volume_shape,
        2,
        do_gridding_correction=True,
        current_size=4,
    )
    monkeypatch.setattr(rf, "_RELION_PROJECTION_PAD_HOST_FFT_MIN_VOXELS", 1)
    host_padded, host_shape = rf.pad_volume_for_projection(
        ft,
        volume_shape,
        2,
        do_gridding_correction=True,
        current_size=4,
    )

    assert host_shape == device_shape
    np.testing.assert_allclose(
        np.asarray(host_padded),
        np.asarray(device_padded),
        atol=2e-4,
        rtol=2e-4,
    )


def test_adjust_regularization_relion_style_lower_bounded():
    filt = np.zeros((4, 4, 4), dtype=np.float32)
    reg = rf.adjust_regularization_relion_style(filt, volume_shape=(4, 4, 4))
    reg_np = np.asarray(reg)
    assert reg_np.shape == (4, 4, 4)
    assert (reg_np >= jax_config.EPSILON).all()


def test_adjust_regularization_relion_style_respects_minres_map():
    import recovar.core.fourier_transform_utils as ftu

    volume_shape = (8, 8, 8)
    filt = np.ones(int(np.prod(volume_shape)), dtype=np.float32)
    tau = np.full(int(np.prod(volume_shape)), 0.5, dtype=np.float32)

    reg = np.asarray(
        rf.adjust_regularization_relion_style(
            filt,
            volume_shape=volume_shape,
            tau=tau,
            minres_map=2,
        )
    ).reshape(-1)
    shell = np.rint(
        np.asarray(ftu.get_grid_of_radial_distances(volume_shape, scaled=False, frequency_shift=0))
    ).astype(int).reshape(-1)

    np.testing.assert_allclose(reg[shell < 2], 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reg[shell >= 2], 3.0, rtol=1e-6, atol=1e-6)


def test_adjust_regularization_relion_style_half_matches_full():
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(123)
    volume_shape = (6, 6, 6)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))
    full_size = int(np.prod(volume_shape))

    filt_half = rng.random(half_size).astype(np.float32) + 0.1
    tau = rng.random(full_size).astype(np.float32) + 0.2

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        filt_full = (
            np.asarray(ftu.half_volume_to_full_volume(jnp.array(filt_half), volume_shape))
            .reshape(-1)
            .real.astype(np.float32)
        )
        reg_full = np.asarray(
            rf.adjust_regularization_relion_style(jnp.array(filt_full), volume_shape=volume_shape, tau=jnp.array(tau))
        ).reshape(-1)
        reg_half = np.asarray(
            rf.adjust_regularization_relion_style(
                jnp.array(filt_half), volume_shape=volume_shape, tau=jnp.array(tau), half_volume=True
            )
        ).reshape(-1)
        reg_full_to_half = np.asarray(ftu.full_volume_to_half_volume(jnp.array(reg_full), volume_shape)).reshape(-1)

    np.testing.assert_allclose(reg_half, reg_full_to_half, atol=1e-5, rtol=1e-5)


def test_post_process_from_filter_v2_half_matches_full():
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(124)
    volume_shape = (6, 6, 6)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))

    ft_ctf_half = rng.random(half_size).astype(np.float32) + 0.2
    tau = rng.random(int(np.prod(volume_shape))).astype(np.float32) + 0.3

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        ft_ctf_full = (
            np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_ctf_half), volume_shape))
            .reshape(-1)
            .real.astype(np.float32)
        )
        spatial_rhs = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
        f_ty_half = np.asarray(ftu.get_dft3_real(spatial_rhs)).reshape(-1).astype(np.complex64)
        f_ty_full = (
            np.asarray(ftu.half_volume_to_full_volume(jnp.array(f_ty_half), volume_shape))
            .reshape(-1)
            .astype(np.complex64)
        )

        out_full = np.asarray(
            rf.post_process_from_filter_v2(
                jnp.array(ft_ctf_full),
                jnp.array(f_ty_full),
                volume_shape,
                1,
                tau=jnp.array(tau),
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=False,
            )
        ).reshape(-1)
        out_half = np.asarray(
            rf.post_process_from_filter_v2(
                jnp.array(ft_ctf_half),
                jnp.array(f_ty_half),
                volume_shape,
                1,
                tau=jnp.array(tau),
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=True,
            )
        ).reshape(-1)
        out_half_packed = np.asarray(
            rf.post_process_from_filter_v2(
                jnp.array(ft_ctf_half),
                jnp.array(f_ty_half),
                volume_shape,
                1,
                tau=jnp.array(tau),
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=True,
                return_half_volume=True,
            )
        ).reshape(-1)
        out_real_full = np.asarray(
            rf.post_process_from_filter_v2(
                jnp.array(ft_ctf_full),
                jnp.array(f_ty_full),
                volume_shape,
                1,
                tau=jnp.array(tau),
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=False,
                return_real_space=True,
            )
        )
        out_real_half = np.asarray(
            rf.post_process_from_filter_v2(
                jnp.array(ft_ctf_half),
                jnp.array(f_ty_half),
                volume_shape,
                1,
                tau=jnp.array(tau),
                kernel="triangular",
                use_spherical_mask=False,
                grid_correct=False,
                input_half_volume=True,
                return_real_space=True,
            )
        )
        out_real_from_packed = np.asarray(ftu.get_idft3_real(jnp.array(out_half_packed.reshape(half_shape)), volume_shape))

    np.testing.assert_allclose(out_real_half, out_real_full, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out_real_from_packed, out_real_full, atol=1e-4, rtol=1e-4)


def test_join_halves_at_low_resolution_half_matches_full():
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(125)
    volume_shape = (6, 6, 6)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))

    ft_y_0_half = (
        rng.standard_normal(half_size).astype(np.float32)
        + 1j * rng.standard_normal(half_size).astype(np.float32)
    ).astype(np.complex64)
    ft_y_1_half = (
        rng.standard_normal(half_size).astype(np.float32)
        + 1j * rng.standard_normal(half_size).astype(np.float32)
    ).astype(np.complex64)
    ft_ctf_0_half = rng.random(half_size).astype(np.float32) + 0.1
    ft_ctf_1_half = rng.random(half_size).astype(np.float32) + 0.1

    ft_y_0_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_y_0_half), volume_shape)).reshape(-1)
    ft_y_1_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_y_1_half), volume_shape)).reshape(-1)
    ft_ctf_0_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_ctf_0_half), volume_shape)).reshape(-1).real
    ft_ctf_1_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_ctf_1_half), volume_shape)).reshape(-1).real

    full_joined = regularization.join_halves_at_low_resolution(
        jnp.array(ft_y_0_full),
        jnp.array(ft_y_1_full),
        jnp.array(ft_ctf_0_full),
        jnp.array(ft_ctf_1_full),
        volume_shape,
        voxel_size=1.2,
        grid_size=volume_shape[0],
        low_resol_join_halves_angstrom=12.0,
        current_resolution_angstrom=18.0,
    )
    half_joined = regularization.join_halves_at_low_resolution(
        jnp.array(ft_y_0_half),
        jnp.array(ft_y_1_half),
        jnp.array(ft_ctf_0_half),
        jnp.array(ft_ctf_1_half),
        volume_shape,
        voxel_size=1.2,
        grid_size=volume_shape[0],
        low_resol_join_halves_angstrom=12.0,
        current_resolution_angstrom=18.0,
    )

    for full_arr, half_arr in zip(full_joined, half_joined):
        restored_half = np.asarray(ftu.half_volume_to_full_volume(jnp.asarray(half_arr), volume_shape)).reshape(-1)
        np.testing.assert_allclose(restored_half, np.asarray(full_arr).reshape(-1), atol=1e-5, rtol=1e-5)


def test_join_halves_at_low_resolution_uses_relion_squared_radius_boundary():
    volume_shape = (8, 8, 8)
    idx_inside = (6, 4, 4)  # frequency (2, 0, 0), included for radius^2 <= 2^2.
    idx_boundary = (6, 5, 4)  # frequency (2, 1, 0), rounded shell 2 but radius^2 > 2^2.

    ft_y_0 = np.zeros(volume_shape, dtype=np.complex64)
    ft_y_1 = np.zeros(volume_shape, dtype=np.complex64)
    ft_ctf_0 = np.ones(volume_shape, dtype=np.float32)
    ft_ctf_1 = np.ones(volume_shape, dtype=np.float32)

    ft_y_0[idx_inside] = 10.0
    ft_y_1[idx_inside] = 2.0
    ft_y_0[idx_boundary] = 20.0
    ft_y_1[idx_boundary] = 4.0

    joined0, joined1, _, _ = regularization.join_halves_at_low_resolution(
        jnp.array(ft_y_0.reshape(-1)),
        jnp.array(ft_y_1.reshape(-1)),
        jnp.array(ft_ctf_0.reshape(-1)),
        jnp.array(ft_ctf_1.reshape(-1)),
        volume_shape,
        voxel_size=10.0,
        grid_size=4,
        low_resol_join_halves_angstrom=40.0,
    )

    joined0 = np.asarray(joined0).reshape(volume_shape)
    joined1 = np.asarray(joined1).reshape(volume_shape)

    np.testing.assert_allclose(joined0[idx_inside], 6.0, atol=1e-6)
    np.testing.assert_allclose(joined1[idx_inside], 6.0, atol=1e-6)
    np.testing.assert_allclose(joined0[idx_boundary], 20.0, atol=1e-6)
    np.testing.assert_allclose(joined1[idx_boundary], 4.0, atol=1e-6)


def test_relion_weight_shell_stats_half_matches_full():
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(126)
    volume_shape = (6, 6, 6)
    padding_factor = 2
    padded_shape = tuple(d * padding_factor for d in volume_shape)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(padded_shape)))

    weight_half = rng.random(half_size).astype(np.float32) + 0.1
    weight_full = np.asarray(ftu.half_volume_to_full_volume(jnp.array(weight_half), padded_shape)).reshape(-1).real
    fsc = np.linspace(0.95, 0.2, volume_shape[0] // 2 + 1, dtype=np.float32)
    tau2 = np.linspace(0.5, 1.0, volume_shape[0] // 2 + 1, dtype=np.float32)

    prior_full, fsc_full = regularization.compute_relion_tau2_from_weights(
        jnp.array(weight_full),
        jnp.array(weight_full * 1.1),
        jnp.array(fsc),
        volume_shape,
        padding_factor=padding_factor,
    )
    prior_half, fsc_half = regularization.compute_relion_tau2_from_weights(
        jnp.array(weight_half),
        jnp.array(weight_half * 1.1),
        jnp.array(fsc),
        volume_shape,
        padding_factor=padding_factor,
    )
    dvp_full = regularization.compute_data_vs_prior(
        jnp.array(weight_full),
        jnp.array(tau2),
        volume_shape,
        padding_factor=padding_factor,
    )
    dvp_half = regularization.compute_data_vs_prior(
        jnp.array(weight_half),
        jnp.array(tau2),
        volume_shape,
        padding_factor=padding_factor,
    )

    np.testing.assert_allclose(np.asarray(prior_half), np.asarray(prior_full), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(fsc_half), np.asarray(fsc_full), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dvp_half), np.asarray(dvp_full), atol=1e-5, rtol=1e-5)


def test_relion_kernel_batch_normalizes_noise_variance_shapes():
    from recovar.core.configs import ForwardModelConfig

    def ctf_fun(params, image_shape, voxel_size, *, half_image=False, **kwargs):
        w = image_shape[1] // 2 + 1 if half_image else image_shape[1]
        return jnp.ones((params.shape[0], image_shape[0] * w), dtype=jnp.float32)

    config = ForwardModelConfig(
        image_shape=(4, 4),
        volume_shape=(4, 4, 4),
        grid_size=4,
        voxel_size=1.5,
        padding=0,
        disc_type="linear_interp",
        ctf=core.as_ctf_evaluator(ctf_fun),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
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
            config,
            images,
            ctf_params,
            rots,
            trans,
            noise_var,
        )
        assert np.asarray(ft_y).shape == (half_vol_size,)
        assert np.asarray(ft_ctf).shape == (half_vol_size,)
        assert np.isfinite(np.asarray(ft_y)).all()


def test_relion_style_triangular_kernel_respects_by_image_flag(monkeypatch):
    calls = {}

    class _FakeDataset:
        noise = None

        def iter_batches(self, batch_size, *, noise_model=None, indices=None, by_image=True):
            calls["batch_size"] = batch_size
            calls["noise_model"] = noise_model
            calls["indices"] = np.asarray(indices, dtype=np.int32)
            calls["by_image"] = by_image
            yield (
                np.ones((1, 4, 4), dtype=np.float32),
                np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 9), dtype=np.float32),
                None,
                np.array([0], dtype=np.int32),
                np.array([3], dtype=np.int32),
            )

    monkeypatch.setattr(
        rf.ForwardModelConfig,
        "from_dataset",
        staticmethod(
            lambda *_args, **_kwargs: SimpleNamespace(
                image_shape=(4, 4),
                volume_shape=(4, 4, 4),
                premultiplied_ctf=False,
                compute_ctf_half=lambda ctf_params: jnp.ones((ctf_params.shape[0], 12), dtype=jnp.float32),
                data_multiplier=1.0,
                grid_size=4,
                padding=0,
                disc_type="linear_interp",
            )
        ),
    )
    monkeypatch.setattr(
        rf,
        "relion_kernel_batch",
        lambda _config, _images, _ctf_params, _rots, _trans, _noise_variance, Ft_y=None, Ft_ctf=None: (
            jnp.ones((4 * 4 * (4 // 2 + 1),), dtype=jnp.complex64),
            jnp.ones((4 * 4 * (4 // 2 + 1),), dtype=jnp.float32),
        ),
    )

    ft_ctf, ft_y = rf.relion_style_triangular_kernel(
        _FakeDataset(),
        cov_noise=None,
        batch_size=7,
        index_subset=np.array([1, 5], dtype=np.int32),
        by_image=False,
    )

    np.testing.assert_array_equal(calls["indices"], np.array([1, 5], dtype=np.int32))
    assert calls["by_image"] is False
    assert calls["batch_size"] == 7
    assert np.asarray(ft_ctf).shape == (4 * 4 * 4,)
    assert np.asarray(ft_y).shape == (4 * 4 * 4,)


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
    from recovar.core.ctf import CTFEvaluator
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
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=voxel_size,
        padding=pad,
        disc_type="linear_interp",
        ctf=CTFEvaluator(),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
        data_multiplier=data_multiplier,
    )

    raw_images = rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32)
    rots = np.stack(
        [
            np.eye(3, dtype=np.float32),
            rng.standard_normal((3, 3)).astype(np.float32),
            rng.standard_normal((3, 3)).astype(np.float32),
        ]
    )
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
    ref_Ft_y = np.asarray(
        core_forward.adjoint_forward_model(
            config,
            full_images_normed,
            jnp.array(ctf_params),
            jnp.array(rots),
            skip_ctf=False,
            volume=None,
            half_image=False,
        )
    )
    ctf_full = config.compute_ctf(jnp.array(ctf_params)) / noise_full
    ref_Ft_ctf = np.asarray(
        core_forward.adjoint_forward_model(
            config,
            ctf_full,
            jnp.array(ctf_params),
            jnp.array(rots),
            volume=None,
            half_image=False,
        )
    )

    # --- New: half-image path via relion_kernel_batch (outputs half-volume) ---
    import recovar.core.fourier_transform_utils as ftu

    half_Ft_y, half_Ft_ctf = rf.relion_kernel_batch(
        config,
        jnp.array(raw_images),
        jnp.array(ctf_params),
        jnp.array(rots),
        jnp.array(trans),
        jnp.array(noise_var),
    )
    new_Ft_y = np.asarray(ftu.half_volume_to_full_volume(half_Ft_y, volume_shape)).reshape(-1)
    new_Ft_ctf = np.asarray(ftu.half_volume_to_full_volume(half_Ft_ctf, volume_shape)).reshape(-1)
    np.testing.assert_allclose(new_Ft_y, ref_Ft_y, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(new_Ft_ctf, ref_Ft_ctf, atol=1e-4, rtol=1e-4)


def test_relion_kernel_batch_complex_input_matches_full_reference():
    """relion_kernel_batch_from_fft with pre-processed complex images
    matches the full-image reference."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import CTFEvaluator
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
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=1.5,
        padding=pad,
        disc_type="linear_interp",
        ctf=CTFEvaluator(),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
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
    ref_Ft_y = np.asarray(
        core_forward.adjoint_forward_model(
            config,
            full_normed,
            jnp.array(ctf_params),
            rots,
            skip_ctf=False,
            volume=None,
            half_image=False,
        )
    )
    ctf_full = config.compute_ctf(jnp.array(ctf_params)) / noise_full
    ref_Ft_ctf = np.asarray(
        core_forward.adjoint_forward_model(
            config,
            ctf_full,
            jnp.array(ctf_params),
            rots,
            volume=None,
            half_image=False,
        )
    )

    # --- New: half-image path (outputs half-volume) ---
    import recovar.core.fourier_transform_utils as ftu

    half_Ft_y, half_Ft_ctf = rf.relion_kernel_batch_from_fft(
        config,
        complex_images,
        jnp.array(ctf_params),
        rots,
        trans,
        jnp.array(noise_var),
    )
    new_Ft_y = np.asarray(ftu.half_volume_to_full_volume(half_Ft_y, volume_shape)).reshape(-1)
    new_Ft_ctf = np.asarray(ftu.half_volume_to_full_volume(half_Ft_ctf, volume_shape)).reshape(-1)
    np.testing.assert_allclose(new_Ft_y, ref_Ft_y, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(new_Ft_ctf, ref_Ft_ctf, atol=1e-4, rtol=1e-4)


def test_relion_kernel_batch_accumulator_matches_sequential():
    """Calling relion_kernel_batch twice with accumulators matches summing two independent calls."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import CTFEvaluator

    rng = np.random.default_rng(77)
    grid_size = 6
    pad = 2
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3

    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=1.0,
        padding=pad,
        disc_type="linear_interp",
        ctf=CTFEvaluator(),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )

    bsz = 2
    imgs1 = jnp.array(rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32))
    imgs2 = jnp.array(rng.standard_normal((bsz, grid_size, grid_size)).astype(np.float32))
    rots = jnp.tile(jnp.eye(3, dtype=jnp.float32), (bsz, 1, 1))
    trans = jnp.zeros((bsz, 2), dtype=jnp.float32)
    ctf_params = jnp.zeros((bsz, 9), dtype=jnp.float32)
    noise_var = jnp.ones((1, np.prod(image_shape)), dtype=jnp.float32)

    # Two independent calls, sum result
    y1, c1 = rf.relion_kernel_batch(config, imgs1, ctf_params, rots, trans, noise_var)
    y2, c2 = rf.relion_kernel_batch(config, imgs2, ctf_params, rots, trans, noise_var)
    sum_Ft_y = np.asarray(y1) + np.asarray(y2)
    sum_Ft_ctf = np.asarray(c1) + np.asarray(c2)

    # Accumulated version
    acc_y, acc_c = rf.relion_kernel_batch(config, imgs1, ctf_params, rots, trans, noise_var)
    acc_y, acc_c = rf.relion_kernel_batch(config, imgs2, ctf_params, rots, trans, noise_var, Ft_y=acc_y, Ft_ctf=acc_c)

    np.testing.assert_allclose(np.asarray(acc_y), sum_Ft_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(acc_c), sum_Ft_ctf, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_relion_kernel_batch_half_volume_output_on_gpu(gpu_device):
    """relion_kernel_batch on GPU produces finite half-volume outputs."""
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.ctf import CTFEvaluator
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(55)
    grid_size = 8
    pad = 4
    image_shape = (grid_size + pad, grid_size + pad)
    volume_shape = (grid_size + pad,) * 3
    bsz = 3

    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=1.5,
        padding=pad,
        disc_type="linear_interp",
        ctf=CTFEvaluator(),
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
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
    with jax.default_device(gpu_device):
        half_y, half_c = rf.relion_kernel_batch(config, raw_images, ctf_params, rots, trans, noise_var)

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


@pytest.mark.gpu
def test_post_process_from_filter_v2_half_matches_full_gpu(gpu_device):
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(125)
    volume_shape = (8, 8, 8)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))

    ft_ctf_half = rng.random(half_size).astype(np.float32) + 0.2
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        ft_ctf_full = (
            np.asarray(ftu.half_volume_to_full_volume(jnp.array(ft_ctf_half), volume_shape))
            .reshape(-1)
            .real.astype(np.float32)
        )

        spatial_rhs = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
        f_ty_half = np.asarray(ftu.get_dft3_real(spatial_rhs)).reshape(-1).astype(np.complex64)
        f_ty_full = (
            np.asarray(ftu.half_volume_to_full_volume(jnp.array(f_ty_half), volume_shape))
            .reshape(-1)
            .astype(np.complex64)
        )

    tau = rng.random(int(np.prod(volume_shape))).astype(np.float32) + 0.3

    try:
        with jax.default_device(gpu_device):
            out_full = np.asarray(
                rf.post_process_from_filter_v2(
                    jax.device_put(jnp.array(ft_ctf_full), gpu_device),
                    jax.device_put(jnp.array(f_ty_full), gpu_device),
                    volume_shape,
                    1,
                    tau=jax.device_put(jnp.array(tau), gpu_device),
                    kernel="triangular",
                    use_spherical_mask=False,
                    grid_correct=False,
                    input_half_volume=False,
                )
            )
            out_half = np.asarray(
                rf.post_process_from_filter_v2(
                    jax.device_put(jnp.array(ft_ctf_half), gpu_device),
                    jax.device_put(jnp.array(f_ty_half), gpu_device),
                    volume_shape,
                    1,
                    tau=jax.device_put(jnp.array(tau), gpu_device),
                    kernel="triangular",
                    use_spherical_mask=False,
                    grid_correct=False,
                    input_half_volume=True,
                )
            )
    except jax.errors.JaxRuntimeError as exc:
        if "Failed to create cuFFT batched plan" in str(exc):
            pytest.skip("cuFFT plan allocation failed on this GPU runner")
        raise

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        vol_full_real = np.asarray(ftu.get_idft3(jnp.array(out_full.reshape(volume_shape)))).real
        vol_half_real = np.asarray(ftu.get_idft3(jnp.array(out_half.reshape(volume_shape)))).real
    np.testing.assert_allclose(vol_half_real, vol_full_real, atol=1e-4, rtol=1e-4)
