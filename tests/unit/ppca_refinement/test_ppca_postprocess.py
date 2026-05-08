import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.postprocess import (
    PPCA_POSTPROCESS_HEURISTIC_WARNING,
    postprocess_ppca_half_volumes,
)


pytestmark = pytest.mark.unit


def _real_to_half(vol):
    return ftu.get_dft3_real(jnp.asarray(vol).real).reshape(-1)


def _half_to_real(half, volume_shape):
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    return np.asarray(ftu.get_idft3_real(jnp.asarray(half).reshape(half_shape), volume_shape=volume_shape))


def test_ppca_postprocess_mean_background_fill_and_w_zero_mask():
    volume_shape = (8, 8, 8)
    radius = 2.0
    width = 1.0
    coords = np.asarray(ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, 1, scaled=False)).reshape(
        volume_shape + (3,)
    )
    r = np.linalg.norm(coords, axis=-1)
    outside = r > radius + width

    mu_real = np.where(r < radius, 5.0, 2.0).astype(np.float32)
    W_real = np.ones(volume_shape, dtype=np.float32)
    mu_half = _real_to_half(mu_real)
    W_half = _real_to_half(W_real)[:, None]

    result = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        strategy="mean_and_w_mask",
        mask_radius_px=radius,
        cosine_width_px=width,
        grid_correct=False,
        cap_W_shell_power=False,
    )

    mu_processed = _half_to_real(result.mu_half, volume_shape)
    W_processed = _half_to_real(result.W_half[:, 0], volume_shape)
    np.testing.assert_allclose(mu_processed[outside], 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(W_processed[outside], 0.0, rtol=1e-5, atol=1e-5)
    assert result.diagnostics["postprocess_warning"] == PPCA_POSTPROCESS_HEURISTIC_WARNING
    assert result.diagnostics["postprocess_strategy"] == "mean_and_w_mask"


def test_ppca_postprocess_modes_leave_requested_columns_raw():
    volume_shape = (8, 8, 8)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    mu_half = jnp.arange(half_size, dtype=jnp.float32).astype(jnp.complex64)
    W_half = jnp.stack([mu_half * 0.1, mu_half * 0.2], axis=1)

    none_result = postprocess_ppca_half_volumes(mu_half, W_half, volume_shape, strategy="none")
    np.testing.assert_allclose(np.asarray(none_result.mu_half), np.asarray(mu_half))
    np.testing.assert_allclose(np.asarray(none_result.W_half), np.asarray(W_half))

    mean_only = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        strategy="mean_only",
        grid_correct=False,
    )
    np.testing.assert_allclose(np.asarray(mean_only.W_half), np.asarray(W_half))


def test_ppca_postprocess_bandlimits_heuristic_output():
    volume_shape = (8, 8, 8)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    mu_half = jnp.ones((half_size,), dtype=jnp.complex64)
    W_half = jnp.ones((half_size, 2), dtype=jnp.complex64)
    max_r = 2.0

    result = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        strategy="mean_and_w_mask",
        grid_correct=False,
        bandlimit_max_r=max_r,
    )

    coords = np.asarray(ftu.get_k_coordinate_of_each_pixel_3d_real(volume_shape, 1, scaled=False))
    outside = np.sum(coords**2, axis=-1) > max_r**2
    np.testing.assert_allclose(np.asarray(result.mu_half)[outside], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(result.W_half)[outside], 0.0, rtol=0.0, atol=0.0)
    assert result.diagnostics["postprocess_bandlimit_max_r"] == pytest.approx(max_r)
    assert 0.0 < result.diagnostics["postprocess_bandlimit_fraction"] < 1.0


def test_ppca_postprocess_caps_W_shell_power_after_masking():
    volume_shape = (8, 8, 8)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    mu_half = jnp.zeros((half_size,), dtype=jnp.complex64)
    W_half = jnp.zeros((half_size, 1), dtype=jnp.complex64)
    shells = np.asarray(
        ftu.get_grid_of_radial_distances_real(volume_shape, scaled=False, frequency_shift=0),
        dtype=np.int64,
    ).reshape(-1)
    dc_idx = int(np.flatnonzero(shells == 0)[0])
    W_half = W_half.at[dc_idx, 0].set(1.0 + 0.0j)

    result = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        strategy="mean_and_w_mask",
        mask_radius_px=2.0,
        cosine_width_px=1.0,
        grid_correct=False,
    )

    output_power_by_shell = np.bincount(
        shells,
        weights=np.sum(np.abs(np.asarray(result.W_half)) ** 2, axis=1),
    )
    assert output_power_by_shell[0] > 0.0
    np.testing.assert_allclose(output_power_by_shell[1:], 0.0, rtol=0.0, atol=1.0e-7)
    assert result.diagnostics["postprocess_cap_W_shell_power"] is True
    assert result.diagnostics["postprocess_W_shell_power_scale_min"] == pytest.approx(0.0)
