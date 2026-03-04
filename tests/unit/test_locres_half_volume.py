import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.core import mask as mask_fn
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.heterogeneity import locres
from recovar.reconstruction import regularization

pytestmark = pytest.mark.unit


def _legacy_sampling_points(grid_size, locres_sampling, locres_maskrad, voxel_size):
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    maskrad_pix = np.round(locres_maskrad / voxel_size).astype(int)
    step_size = np.round(locres_sampling / voxel_size).astype(int)
    myrad = grid_size // 2 - maskrad_pix

    sampling_points = []
    grid = np.array(
        fourier_transform_utils.get_1d_frequency_grid(grid_size, 1, scaled=False)[::step_size]
    )
    for kk in grid:
        for ii in grid:
            for jj in grid:
                rad = np.sqrt(kk * kk + ii * ii + jj * jj)
                if rad < myrad:
                    sampling_points.append((kk, ii, jj))
    return np.asarray(sampling_points, dtype=np.int32)


def _legacy_local_error(map1, map2, voxel_size, locres_sampling, locres_maskrad, locres_edgwidth, low_pass_filter_res):
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)

    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)
    mask = mask / np.linalg.norm(mask)

    if low_pass_filter_res is not None:
        map1_ft = fourier_transform_utils.get_dft3(map1)
        map1_ft = locres.low_pass_filter_map(
            map1_ft,
            map1_ft.shape[0],
            low_pass_filter_res,
            voxel_size,
            edgewidth_pix,
            do_highpass_instead=False,
        )
        map1 = fourier_transform_utils.get_idft3(map1_ft).real

        map2_ft = fourier_transform_utils.get_dft3(map2)
        map2_ft = locres.low_pass_filter_map(
            map2_ft,
            map2_ft.shape[0],
            low_pass_filter_res,
            voxel_size,
            edgewidth_pix,
            do_highpass_instead=False,
        )
        map2 = fourier_transform_utils.get_idft3(map2_ft).real

    mask_ft = fourier_transform_utils.get_dft3(mask)
    map1_square_ft = fourier_transform_utils.get_dft3(map1 * map1)
    map2_square_ft = fourier_transform_utils.get_dft3(map2 * map2)
    map1map2_ft = fourier_transform_utils.get_dft3(map1 * map2)

    local_errors = (
        fourier_transform_utils.get_idft3(map1_square_ft * mask_ft).real
        - 2 * fourier_transform_utils.get_idft3(map1map2_ft * mask_ft)
        + fourier_transform_utils.get_idft3(map2_square_ft * mask_ft)
    )
    return np.asarray(local_errors.real)


def _legacy_local_error_with_cov(
    map1,
    map2,
    voxel_size,
    locres_sampling,
    locres_maskrad,
    locres_edgwidth,
    low_pass_filter_res,
    noise_variance,
):
    locres_maskrad = 0.5 * locres_sampling if locres_maskrad is None else locres_maskrad
    locres_edgwidth = locres_sampling if locres_edgwidth is None else locres_edgwidth
    edgewidth_pix = np.round(locres_edgwidth / voxel_size).astype(int)

    mask = mask_fn.raised_cosine_mask(map1.shape, locres_maskrad, locres_maskrad + edgewidth_pix, -1)

    if low_pass_filter_res is not None:
        map1_ft = fourier_transform_utils.get_dft3(map1)
        map1_ft = locres.low_pass_filter_map(
            map1_ft,
            map1_ft.shape[0],
            low_pass_filter_res,
            voxel_size,
            edgewidth_pix,
            do_highpass_instead=False,
        )
        map1 = fourier_transform_utils.get_idft3(map1_ft).real

        map2_ft = fourier_transform_utils.get_dft3(map2)
        map2_ft = locres.low_pass_filter_map(
            map2_ft,
            map2_ft.shape[0],
            low_pass_filter_res,
            voxel_size,
            edgewidth_pix,
            do_highpass_instead=False,
        )
        map2 = fourier_transform_utils.get_idft3(map2_ft).real

    if noise_variance is not None:
        noise_variance = noise_variance.reshape(map1.shape)
        noise_scale = np.sqrt(noise_variance).reshape(map1.shape)
        map1 = fourier_transform_utils.get_idft3(
            fourier_transform_utils.get_dft3(map1) * noise_scale
        ).real
        map2 = fourier_transform_utils.get_idft3(
            fourier_transform_utils.get_dft3(map2) * noise_scale
        ).real

    mask_ft = fourier_transform_utils.get_dft3(mask)
    map1_square_ft = fourier_transform_utils.get_dft3(map1 * map1)
    map2_square_ft = fourier_transform_utils.get_dft3(map2 * map2)
    map1map2_ft = fourier_transform_utils.get_dft3(map1 * map2)

    local_errors = (
        fourier_transform_utils.get_idft3(map1_square_ft * mask_ft).real
        - 2 * fourier_transform_utils.get_idft3(map1map2_ft * mask_ft)
        + fourier_transform_utils.get_idft3(map2_square_ft * mask_ft)
    )
    return np.asarray(local_errors.real)


def test_get_sampling_points_matches_legacy_loop_order_and_values():
    got = np.asarray(locres.get_sampling_points(20, 5, 2.0, 1.0))
    expected = _legacy_sampling_points(20, 5, 2.0, 1.0)
    np.testing.assert_array_equal(got, expected)


def test_filter_with_local_fsc_half_matches_full():
    rng = np.random.default_rng(0)
    volume_shape = (12, 12, 12)
    map1 = rng.standard_normal(volume_shape).astype(np.float32)
    map2 = rng.standard_normal(volume_shape).astype(np.float32)

    fsc = regularization.get_fsc(
        fourier_transform_utils.get_dft3(map1),
        fourier_transform_utils.get_dft3(map2),
        volume_shape=volume_shape,
    )
    ft_full = fourier_transform_utils.get_dft3(0.5 * (map1 + map2))
    ft_half = fourier_transform_utils.get_dft3_real(0.5 * (map1 + map2))

    out_full = np.asarray(
        locres.filter_with_local_fsc(
            ft_full,
            fsc,
            local_resol=8.0,
            voxel_size=1.5,
            filter_edgewidth=2,
            volume_shape=volume_shape,
        )
    )
    out_half = np.asarray(
        locres.filter_with_local_fsc(
            ft_half,
            fsc,
            local_resol=8.0,
            voxel_size=1.5,
            filter_edgewidth=2,
            volume_shape=volume_shape,
        )
    )
    np.testing.assert_allclose(out_half, out_full, rtol=1e-5, atol=1e-5)


def test_local_error_matches_legacy_full_fft_implementation():
    rng = np.random.default_rng(1)
    volume_shape = (12, 12, 12)
    map1 = rng.standard_normal(volume_shape).astype(np.float32)
    map2 = rng.standard_normal(volume_shape).astype(np.float32)

    kwargs = dict(
        voxel_size=1.5,
        locres_sampling=6,
        locres_maskrad=3.0,
        locres_edgwidth=4.5,
        low_pass_filter_res=8.0,
    )

    expected = _legacy_local_error(map1.copy(), map2.copy(), **kwargs)
    got = np.asarray(locres.local_error(map1.copy(), map2.copy(), **kwargs))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_local_error_with_cov_matches_legacy_full_fft_implementation():
    rng = np.random.default_rng(2)
    volume_shape = (12, 12, 12)
    map1 = rng.standard_normal(volume_shape).astype(np.float32)
    map2 = rng.standard_normal(volume_shape).astype(np.float32)
    noise_variance = (0.1 + rng.random(volume_shape)).astype(np.float32)

    kwargs = dict(
        voxel_size=1.5,
        locres_sampling=6,
        locres_maskrad=3.0,
        locres_edgwidth=4.5,
        low_pass_filter_res=7.0,
    )

    expected = _legacy_local_error_with_cov(
        map1.copy(),
        map2.copy(),
        noise_variance=noise_variance.copy(),
        **kwargs,
    )
    got = np.asarray(
        locres.local_error_with_cov(
            map1.copy(),
            map2.copy(),
            noise_variance=noise_variance.copy(),
            **kwargs,
        )
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_filter_with_global_fsc_half_matches_full():
    rng = np.random.default_rng(3)
    volume_shape = (12, 12, 12)
    map1 = rng.standard_normal(volume_shape).astype(np.float32)
    map2 = rng.standard_normal(volume_shape).astype(np.float32)

    fsc = regularization.get_fsc(
        fourier_transform_utils.get_dft3(map1),
        fourier_transform_utils.get_dft3(map2),
        volume_shape=volume_shape,
    )
    mask = mask_fn.raised_cosine_mask(volume_shape, 4, 5, -1)

    ft_full = 0.5 * (
        fourier_transform_utils.get_dft3(map1)
        + fourier_transform_utils.get_dft3(map2)
    )
    ft_half = 0.5 * (
        fourier_transform_utils.get_dft3_real(map1)
        + fourier_transform_utils.get_dft3_real(map2)
    )

    out_full = np.asarray(
        locres.filter_with_global_fsc(
            ft_full,
            fsc,
            voxel_size=1.5,
            filter_edgewidth=2,
            mask=mask,
            B_factor=20.0,
            volume_shape=volume_shape,
        )
    )
    out_half = np.asarray(
        locres.filter_with_global_fsc(
            ft_half,
            fsc,
            voxel_size=1.5,
            filter_edgewidth=2,
            mask=mask,
            B_factor=20.0,
            volume_shape=volume_shape,
        )
    )
    np.testing.assert_allclose(out_half, out_full, rtol=1e-5, atol=1e-5)
