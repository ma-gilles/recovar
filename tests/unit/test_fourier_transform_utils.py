import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def test_get_1d_frequency_grid_even_odd_and_scaled():
    even = fourier_transform_utils.get_1d_frequency_grid(4)
    odd_scaled = fourier_transform_utils.get_1d_frequency_grid(5, voxel_size=2, scaled=True)

    np.testing.assert_array_equal(even, np.array([-2, -1, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(
        odd_scaled,
        np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32),
    )


def test_get_1d_frequency_grid_singleton():
    grid = fourier_transform_utils.get_1d_frequency_grid(1, scaled=False)
    np.testing.assert_array_equal(grid, np.array([0.0], dtype=np.float32))


def test_get_k_coordinate_of_each_pixel_shapes():
    coords_2d = fourier_transform_utils.get_k_coordinate_of_each_pixel((2, 3), voxel_size=1, scaled=False)
    coords_3d = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d((2, 3, 4), voxel_size=1, scaled=False)

    assert coords_2d.shape == (6, 2)
    assert coords_3d.shape == (24, 3)
    assert coords_2d.dtype == np.float32
    assert coords_3d.dtype == np.float32


def test_get_grid_of_radial_distances_current_behavior():
    r = fourier_transform_utils.get_grid_of_radial_distances((3, 3), scaled=False)

    expected = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(r, expected)
    assert np.asarray(r).dtype == np.int32


def test_get_grid_of_radial_distances_frequency_shift_scalar_and_vector():
    r_scalar = fourier_transform_utils.get_grid_of_radial_distances((3, 3, 3), frequency_shift=1.0, scaled=False, rounded=False)
    r_vector = fourier_transform_utils.get_grid_of_radial_distances(
        (3, 3, 3), frequency_shift=np.array([1.0, 1.0, 1.0], dtype=np.float32), scaled=False, rounded=False
    )
    np.testing.assert_allclose(r_scalar, r_vector, atol=1e-7, rtol=1e-7)


def test_get_grid_of_radial_distances_scaled_ignores_rounding():
    r_scaled_int = fourier_transform_utils.get_grid_of_radial_distances((4, 4), scaled=True, rounded=True)
    r_scaled_float = fourier_transform_utils.get_grid_of_radial_distances((4, 4), scaled=True, rounded=False)
    np.testing.assert_allclose(r_scaled_int, r_scaled_float)
    assert np.issubdtype(np.asarray(r_scaled_int).dtype, np.floating)


def test_dft_idft_roundtrip_1d_2d_3d():
    rng = np.random.default_rng(0)

    x1 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    x2 = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    x3 = rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4))

    np.testing.assert_allclose(fourier_transform_utils.get_idft(fourier_transform_utils.get_dft(x1)), x1, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(fourier_transform_utils.get_idft2(fourier_transform_utils.get_dft2(x2)), x2, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(fourier_transform_utils.get_idft3(fourier_transform_utils.get_dft3(x3)), x3, atol=1e-10, rtol=1e-10)


def test_dft3_idft3_custom_axes_roundtrip():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5))
    y = fourier_transform_utils.get_idft3(
        fourier_transform_utils.get_dft3(x, axes=(0, 1, 2)),
        axes=(0, 1, 2),
    )
    np.testing.assert_allclose(y, x, atol=1e-10, rtol=1e-10)
