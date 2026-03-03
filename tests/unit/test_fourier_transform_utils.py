import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = pytest.mark.unit


def _full_last_axis_indices_for_rfft(n: int) -> np.ndarray:
    """Map rfft last-axis bins to shifted full-spectrum indices."""
    n = int(n)
    half = n // 2
    if n % 2 == 0:
        # [0, 1, ..., half-1, half] -> [half, half+1, ..., n-1, 0]
        return np.array(list(range(half, n)) + [0], dtype=np.int32)
    # odd n has no Nyquist singleton
    return np.array(list(range(half, n)), dtype=np.int32)


def _partner_indices_in_shifted_axis(n: int) -> np.ndarray:
    n = int(n)
    half = n // 2
    i = np.arange(n, dtype=np.int64)
    u = (i + half) % n
    u_partner = (-u) % n
    return ((u_partner - half) % n).astype(np.int32)


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


def test_get_1d_frequency_grid_rfft_basic_and_scaled():
    grid = fourier_transform_utils.get_1d_frequency_grid_rfft(5, scaled=False)
    scaled = fourier_transform_utils.get_1d_frequency_grid_rfft(4, voxel_size=2, scaled=True)
    np.testing.assert_array_equal(grid, np.array([0, 1, 2], dtype=np.float32))
    np.testing.assert_allclose(scaled, np.array([0.0, 0.125, 0.25], dtype=np.float32))


def test_get_1d_frequency_grid_rfft_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.get_1d_frequency_grid_rfft(0)


def test_get_real_fft_packed_shape_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="at least one dimension"):
        fourier_transform_utils.get_real_fft_packed_shape(())
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.get_real_fft_packed_shape((8, 0, 8))


def test_packed_last_axis_indices_match_expected_helper():
    np.testing.assert_array_equal(
        np.asarray(fourier_transform_utils.get_real_fft_packed_last_axis_indices(8)),
        _full_last_axis_indices_for_rfft(8),
    )
    np.testing.assert_array_equal(
        np.asarray(fourier_transform_utils.get_real_fft_packed_last_axis_indices(7)),
        _full_last_axis_indices_for_rfft(7),
    )


def test_shifted_conjugate_partner_indices_match_reference():
    np.testing.assert_array_equal(
        np.asarray(fourier_transform_utils.get_shifted_conjugate_partner_indices(8)),
        _partner_indices_in_shifted_axis(8),
    )
    np.testing.assert_array_equal(
        np.asarray(fourier_transform_utils.get_shifted_conjugate_partner_indices(7)),
        _partner_indices_in_shifted_axis(7),
    )


def test_real_fft_index_helpers_reject_nonpositive_n():
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.get_real_fft_packed_last_axis_indices(0)
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.get_shifted_conjugate_partner_indices(0)


def test_volume_shape_to_half_volume_shape_helper():
    assert fourier_transform_utils.volume_shape_to_half_volume_shape((8, 8, 8)) == (8, 8, 5)
    assert fourier_transform_utils.volume_shape_to_half_volume_shape((7, 9, 11)) == (7, 9, 6)


def test_image_shape_to_half_image_shape_helper():
    assert fourier_transform_utils.image_shape_to_half_image_shape((8, 8)) == (8, 5)
    assert fourier_transform_utils.image_shape_to_half_image_shape((7, 9)) == (7, 5)


def test_volume_shape_to_half_volume_shape_rejects_bad_shape():
    with pytest.raises(ValueError, match="must have 3 dims"):
        fourier_transform_utils.volume_shape_to_half_volume_shape((8, 8))
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.volume_shape_to_half_volume_shape((8, 0, 8))


def test_image_shape_to_half_image_shape_rejects_bad_shape():
    with pytest.raises(ValueError, match="must have 2 dims"):
        fourier_transform_utils.image_shape_to_half_image_shape((8, 8, 8))
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.image_shape_to_half_image_shape((8, 0))


def test_get_k_coordinate_of_each_pixel_shapes():
    coords_2d = fourier_transform_utils.get_k_coordinate_of_each_pixel((2, 3), voxel_size=1, scaled=False)
    coords_3d = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d((2, 3, 4), voxel_size=1, scaled=False)

    assert coords_2d.shape == (6, 2)
    assert coords_3d.shape == (24, 3)
    assert coords_2d.dtype == np.float32
    assert coords_3d.dtype == np.float32


def test_get_k_coordinate_of_each_pixel_real_shapes():
    coords_2d = fourier_transform_utils.get_k_coordinate_of_each_pixel_real((6, 10), voxel_size=1, scaled=False)
    coords_3d = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d_real((4, 6, 10), voxel_size=1, scaled=False)
    assert coords_2d.shape == (6 * (10 // 2 + 1), 2)
    assert coords_3d.shape == (4 * 6 * (10 // 2 + 1), 3)
    assert coords_2d.dtype == np.float32
    assert coords_3d.dtype == np.float32


def test_get_k_coordinate_of_each_pixel_real_last_axis_frequencies_are_nonnegative():
    coords = np.asarray(
        fourier_transform_utils.get_k_coordinate_of_each_pixel_3d_real((4, 6, 10), voxel_size=1, scaled=False)
    )
    kz_unique = np.unique(coords[:, 2])
    np.testing.assert_array_equal(kz_unique, np.arange(0, 10 // 2 + 1, dtype=np.float32))


def test_get_k_coordinate_of_each_pixel_real_rejects_bad_dims():
    with pytest.raises(ValueError, match="must have 2 dims"):
        fourier_transform_utils.get_k_coordinate_of_each_pixel_real((4, 4, 4), voxel_size=1, scaled=False)
    with pytest.raises(ValueError, match="must have 3 dims"):
        fourier_transform_utils.get_k_coordinate_of_each_pixel_3d_real((4, 4), voxel_size=1, scaled=False)


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


def test_get_grid_of_radial_distances_real_shape_and_values():
    r = fourier_transform_utils.get_grid_of_radial_distances_real((3, 3), scaled=False)
    expected = np.array(
        [
            [1, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(r, expected)
    assert np.asarray(r).dtype == np.int32


def test_get_grid_of_radial_distances_real_scaled_and_shift_validation():
    r_scaled = fourier_transform_utils.get_grid_of_radial_distances_real(
        (4, 6, 8),
        voxel_size=2.0,
        scaled=True,
        rounded=True,  # ignored when scaled=True
    )
    assert np.issubdtype(np.asarray(r_scaled).dtype, np.floating)

    r_scalar = fourier_transform_utils.get_grid_of_radial_distances_real((5, 7), frequency_shift=1.0, scaled=False, rounded=False)
    r_vector = fourier_transform_utils.get_grid_of_radial_distances_real(
        (5, 7),
        frequency_shift=np.array([1.0, 1.0], dtype=np.float32),
        scaled=False,
        rounded=False,
    )
    np.testing.assert_allclose(r_scalar, r_vector, atol=1e-7, rtol=1e-7)

    with pytest.raises(ValueError, match="frequency_shift must be scalar or shape"):
        fourier_transform_utils.get_grid_of_radial_distances_real((4, 4, 4), frequency_shift=np.array([0.0, 0.0], dtype=np.float32))


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


def test_get_grid_of_radial_distances_real_matches_full_grid_mapped_last_axis_when_unshifted():
    shape = (5, 7, 8)
    full = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances(shape, scaled=False, frequency_shift=0.0, rounded=False)
    )
    real_packed = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances_real(
            shape, scaled=False, frequency_shift=0.0, rounded=False
        )
    )
    last_map = _full_last_axis_indices_for_rfft(shape[-1])
    # Nyquist is represented with opposite sign in centered full grid; compare squared radii.
    np.testing.assert_allclose(real_packed**2, full[..., last_map] ** 2, atol=1e-6, rtol=1e-6)


def test_half_full_volume_mapping_roundtrip_grid_representation():
    rng = np.random.default_rng(61)
    volume_shape = (6, 5, 8)
    x = rng.standard_normal(volume_shape).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)

    full = fourier_transform_utils.half_volume_to_full_volume(half, volume_shape)
    half_back = fourier_transform_utils.full_volume_to_half_volume(full, volume_shape)
    np.testing.assert_allclose(np.asarray(half_back), np.asarray(half), atol=5e-6, rtol=5e-6)


def test_half_full_image_mapping_roundtrip_grid_representation():
    rng = np.random.default_rng(62)
    image_shape = (6, 8)
    x = rng.standard_normal(image_shape).astype(np.float32)
    half = fourier_transform_utils.get_dft2_real(x)

    full = fourier_transform_utils.half_image_to_full_image(half, image_shape)
    half_back = fourier_transform_utils.full_image_to_half_image(full, image_shape)
    np.testing.assert_allclose(np.asarray(half_back), np.asarray(half), atol=5e-6, rtol=5e-6)


def test_full_volume_to_half_volume_is_direct_last_axis_mapping():
    rng = np.random.default_rng(63)
    volume_shape = (6, 5, 8)
    full = (
        rng.standard_normal(volume_shape).astype(np.float32)
        + 1j * rng.standard_normal(volume_shape).astype(np.float32)
    ).astype(np.complex64)
    idx = _full_last_axis_indices_for_rfft(volume_shape[-1])

    expected_grid = np.take(full, idx, axis=-1)
    mapped_grid = fourier_transform_utils.full_volume_to_half_volume(full, volume_shape)
    np.testing.assert_allclose(np.asarray(mapped_grid), expected_grid, atol=1e-6, rtol=1e-6)

    full_flat = full.reshape(-1)
    expected_flat = expected_grid.reshape(-1)
    mapped_flat = fourier_transform_utils.full_volume_to_half_volume(full_flat, volume_shape)
    np.testing.assert_allclose(np.asarray(mapped_flat), expected_flat, atol=1e-6, rtol=1e-6)


def test_full_image_to_half_image_is_direct_last_axis_mapping():
    rng = np.random.default_rng(64)
    image_shape = (6, 8)
    full = (
        rng.standard_normal(image_shape).astype(np.float32)
        + 1j * rng.standard_normal(image_shape).astype(np.float32)
    ).astype(np.complex64)
    idx = _full_last_axis_indices_for_rfft(image_shape[-1])

    expected_grid = np.take(full, idx, axis=-1)
    mapped_grid = fourier_transform_utils.full_image_to_half_image(full, image_shape)
    np.testing.assert_allclose(np.asarray(mapped_grid), expected_grid, atol=1e-6, rtol=1e-6)

    full_flat = full.reshape(-1)
    expected_flat = expected_grid.reshape(-1)
    mapped_flat = fourier_transform_utils.full_image_to_half_image(full_flat, image_shape)
    np.testing.assert_allclose(np.asarray(mapped_flat), expected_flat, atol=1e-6, rtol=1e-6)


def test_half_full_volume_mapping_roundtrip_flat_representation():
    rng = np.random.default_rng(67)
    volume_shape = (5, 4, 7)
    x = rng.standard_normal(volume_shape).astype(np.float32)
    half_flat = np.asarray(fourier_transform_utils.get_dft3_real(x)).reshape(-1)

    full_flat = fourier_transform_utils.half_volume_to_full_volume(half_flat, volume_shape)
    half_flat_back = fourier_transform_utils.full_volume_to_half_volume(full_flat, volume_shape)
    np.testing.assert_allclose(np.asarray(half_flat_back), half_flat, atol=5e-6, rtol=5e-6)


def test_half_full_image_mapping_roundtrip_flat_representation():
    rng = np.random.default_rng(68)
    image_shape = (5, 7)
    x = rng.standard_normal(image_shape).astype(np.float32)
    half_flat = np.asarray(fourier_transform_utils.get_dft2_real(x)).reshape(-1)

    full_flat = fourier_transform_utils.half_image_to_full_image(half_flat, image_shape)
    half_flat_back = fourier_transform_utils.full_image_to_half_image(full_flat, image_shape)
    np.testing.assert_allclose(np.asarray(half_flat_back), half_flat, atol=5e-6, rtol=5e-6)


def test_half_full_volume_mapping_roundtrip_batched_flat_representation():
    rng = np.random.default_rng(71)
    volume_shape = (4, 6, 8)
    x = rng.standard_normal((3,) + volume_shape).astype(np.float32)
    half = np.asarray(fourier_transform_utils.get_dft3_real(x, axes=(-3, -2, -1)))
    half_flat = half.reshape(half.shape[0], -1)

    full_flat = fourier_transform_utils.half_volume_to_full_volume(half_flat, volume_shape)
    assert np.asarray(full_flat).shape == (3, int(np.prod(volume_shape)))
    half_flat_back = fourier_transform_utils.full_volume_to_half_volume(full_flat, volume_shape)
    np.testing.assert_allclose(np.asarray(half_flat_back), half_flat, atol=5e-6, rtol=5e-6)


def test_half_volume_to_full_volume_enforces_hermitian_symmetry():
    rng = np.random.default_rng(73)
    volume_shape = (4, 4, 8)
    x = rng.standard_normal(volume_shape).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)
    full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(half, volume_shape))

    px = _partner_indices_in_shifted_axis(volume_shape[0])
    py = _partner_indices_in_shifted_axis(volume_shape[1])
    pz = _partner_indices_in_shifted_axis(volume_shape[2])
    for ix in range(volume_shape[0]):
        for iy in range(volume_shape[1]):
            for iz in range(volume_shape[2]):
                np.testing.assert_allclose(
                    full[ix, iy, iz],
                    np.conj(full[px[ix], py[iy], pz[iz]]),
                    atol=1e-5,
                    rtol=1e-5,
                )


def test_half_full_volume_mapping_matches_real_fft_helpers_for_real_input():
    rng = np.random.default_rng(79)
    x = rng.standard_normal((5, 6, 8)).astype(np.float32)
    full = fourier_transform_utils.get_dft3(x)
    half_direct = fourier_transform_utils.get_dft3_real(x)
    half_mapped = fourier_transform_utils.full_volume_to_half_volume(full, x.shape)
    np.testing.assert_allclose(np.asarray(half_mapped), np.asarray(half_direct), atol=1e-5, rtol=1e-5)

    full_from_half = fourier_transform_utils.half_volume_to_full_volume(half_direct, x.shape)
    np.testing.assert_allclose(np.asarray(full_from_half), np.asarray(full), atol=1e-5, rtol=1e-5)


def test_half_full_image_mapping_matches_real_fft_helpers_for_real_input():
    rng = np.random.default_rng(83)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    full = fourier_transform_utils.get_dft2(x)
    half_direct = fourier_transform_utils.get_dft2_real(x)
    half_mapped = fourier_transform_utils.full_image_to_half_image(full, x.shape)
    np.testing.assert_allclose(np.asarray(half_mapped), np.asarray(half_direct), atol=1e-5, rtol=1e-5)

    full_from_half = fourier_transform_utils.half_image_to_full_image(half_direct, x.shape)
    np.testing.assert_allclose(np.asarray(full_from_half), np.asarray(full), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [
    (4, 4, 8),   # all even
    (5, 6, 8),   # odd × even × even
    (5, 7, 9),   # all odd
    (7, 5, 11),  # odd × odd × odd
    (3, 3, 3),   # small odd
    (6, 6, 6),   # small even
    (4, 5, 6),   # mixed
    (8, 7, 5),   # mixed reversed
])
def test_half_to_full_volume_matches_fft_diverse_shapes(shape):
    """half_volume_to_full(rfft3(vol)) == fft3(vol) for diverse even/odd shapes."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape).astype(np.float32)
    full_ref = np.asarray(fourier_transform_utils.get_dft3(x))
    half = fourier_transform_utils.get_dft3_real(x)
    full_from_half = np.asarray(fourier_transform_utils.half_volume_to_full_volume(half, shape))
    np.testing.assert_allclose(full_from_half, full_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [
    (4, 6),   # even × even
    (5, 8),   # odd × even
    (7, 9),   # odd × odd
    (5, 5),   # small odd
    (3, 3),   # tiny odd
    (6, 7),   # even × odd
])
def test_half_to_full_image_matches_fft_diverse_shapes(shape):
    """half_image_to_full(rfft2(img)) == fft2(img) for diverse even/odd shapes."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape).astype(np.float32)
    full_ref = np.asarray(fourier_transform_utils.get_dft2(x))
    half = fourier_transform_utils.get_dft2_real(x)
    full_from_half = np.asarray(fourier_transform_utils.half_image_to_full_image(half, shape))
    np.testing.assert_allclose(full_from_half, full_ref, atol=1e-5, rtol=1e-5)


def test_half_full_volume_mapping_rejects_bad_input_shapes():
    volume_shape = (4, 4, 8)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    with pytest.raises(ValueError, match="must have trailing shape"):
        fourier_transform_utils.full_volume_to_half_volume(np.zeros((3, 3), dtype=np.complex64), volume_shape)
    with pytest.raises(ValueError, match="must have trailing shape"):
        fourier_transform_utils.half_volume_to_full_volume(np.zeros((3, 3), dtype=np.complex64), volume_shape)
    with pytest.raises(ValueError, match="must have 3 dims"):
        fourier_transform_utils.full_volume_to_half_volume(np.zeros(volume_shape, dtype=np.complex64), (4, 4))
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.half_volume_to_full_volume(np.zeros(half_shape, dtype=np.complex64), (4, 4, 0))


def test_half_full_image_mapping_rejects_bad_input_shapes():
    image_shape = (4, 8)
    half_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    with pytest.raises(ValueError, match="must have trailing shape"):
        fourier_transform_utils.full_image_to_half_image(np.zeros((3, 3), dtype=np.complex64), image_shape)
    with pytest.raises(ValueError, match="must have trailing shape"):
        fourier_transform_utils.half_image_to_full_image(np.zeros((3, 3), dtype=np.complex64), image_shape)
    with pytest.raises(ValueError, match="must have 2 dims"):
        fourier_transform_utils.full_image_to_half_image(np.zeros(image_shape, dtype=np.complex64), (4, 8, 2))
    with pytest.raises(ValueError, match="must be positive"):
        fourier_transform_utils.half_image_to_full_image(np.zeros(half_shape, dtype=np.complex64), (4, 0))


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


def test_real_fft_packed_shape_and_memory_ratio():
    shape = (8, 8, 8)
    packed = fourier_transform_utils.get_real_fft_packed_shape(shape)
    assert packed == (8, 8, 5)
    ratio = fourier_transform_utils.get_real_fft_memory_saving_ratio(shape)
    assert np.isclose(ratio, (8 * 8 * 5) / (8 * 8 * 8))
    assert ratio < 1.0


def test_dft2_real_matches_full_dft2_mapped_last_axis():
    rng = np.random.default_rng(23)
    x = rng.standard_normal((6, 10)).astype(np.float32)
    half = np.asarray(fourier_transform_utils.get_dft2_real(x))
    full = np.asarray(fourier_transform_utils.get_dft2(x))
    idx = _full_last_axis_indices_for_rfft(x.shape[-1])
    np.testing.assert_allclose(half, full[..., idx], atol=1e-5, rtol=1e-5)


def test_dft2_real_idft2_real_roundtrip_matches_real_input():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((6, 10)).astype(np.float32)
    x_hat_half = fourier_transform_utils.get_dft2_real(x)
    x_back = fourier_transform_utils.get_idft2_real(x_hat_half, image_shape=x.shape)
    np.testing.assert_allclose(np.asarray(x_back), x, atol=1e-5, rtol=1e-5)


def test_idft2_real_infers_shape_when_not_provided():
    rng = np.random.default_rng(29)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    half = fourier_transform_utils.get_dft2_real(x)
    x_back = fourier_transform_utils.get_idft2_real(half)
    assert np.asarray(x_back).shape == x.shape
    np.testing.assert_allclose(np.asarray(x_back), x, atol=1e-5, rtol=1e-5)


def test_idft2_real_reconstructs_odd_length_when_shape_is_explicit():
    rng = np.random.default_rng(30)
    x = rng.standard_normal((5, 9)).astype(np.float32)
    half = fourier_transform_utils.get_dft2_real(x)
    x_back = fourier_transform_utils.get_idft2_real(half, image_shape=x.shape)
    assert np.asarray(x_back).shape == x.shape
    np.testing.assert_allclose(np.asarray(x_back), x, atol=1e-5, rtol=1e-5)


def test_idft2_real_rejects_bad_image_shape():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    half = fourier_transform_utils.get_dft2_real(x)
    with pytest.raises(ValueError, match="must have 2 dims"):
        fourier_transform_utils.get_idft2_real(half, image_shape=(4, 8, 1))


def test_dft3_real_matches_full_dft3_mapped_last_axis():
    rng = np.random.default_rng(37)
    x = rng.standard_normal((4, 6, 10)).astype(np.float32)
    half = np.asarray(fourier_transform_utils.get_dft3_real(x))
    full = np.asarray(fourier_transform_utils.get_dft3(x))
    idx = _full_last_axis_indices_for_rfft(x.shape[-1])
    np.testing.assert_allclose(half, full[..., idx], atol=1e-5, rtol=1e-5)


def test_dft3_real_idft3_real_roundtrip_matches_real_input():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((6, 6, 10)).astype(np.float32)
    x_hat_half = fourier_transform_utils.get_dft3_real(x)
    x_back = fourier_transform_utils.get_idft3_real(x_hat_half, volume_shape=x.shape)
    np.testing.assert_allclose(np.asarray(x_back), x, atol=1e-5, rtol=1e-5)


def test_dft3_real_idft3_real_custom_axes_roundtrip():
    rng = np.random.default_rng(41)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x, axes=(0, 1, 2))
    back = fourier_transform_utils.get_idft3_real(half, volume_shape=x.shape, axes=(0, 1, 2))
    np.testing.assert_allclose(np.asarray(back), x, atol=1e-5, rtol=1e-5)


def test_idft3_real_infers_shape_for_default_axes():
    rng = np.random.default_rng(43)
    x = rng.standard_normal((5, 6, 8)).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)
    back = fourier_transform_utils.get_idft3_real(half)
    assert np.asarray(back).shape == x.shape
    np.testing.assert_allclose(np.asarray(back), x, atol=1e-5, rtol=1e-5)


def test_idft3_real_reconstructs_odd_length_when_shape_is_explicit():
    rng = np.random.default_rng(44)
    x = rng.standard_normal((5, 6, 7)).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)
    back = fourier_transform_utils.get_idft3_real(half, volume_shape=x.shape)
    assert np.asarray(back).shape == x.shape
    np.testing.assert_allclose(np.asarray(back), x, atol=1e-5, rtol=1e-5)


def test_idft3_real_requires_shape_for_nondefault_axes():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    x_hat_half = fourier_transform_utils.get_dft3_real(x, axes=(0, 1, 2))
    with pytest.raises(ValueError, match="volume_shape is required"):
        fourier_transform_utils.get_idft3_real(x_hat_half, axes=(0, 1, 2))


def test_dft3_real_and_idft3_real_reject_invalid_axes_length():
    rng = np.random.default_rng(47)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    with pytest.raises(ValueError, match="length 3"):
        fourier_transform_utils.get_dft3_real(x, axes=(0, 1))
    half = fourier_transform_utils.get_dft3_real(x)
    with pytest.raises(ValueError, match="length 3"):
        fourier_transform_utils.get_idft3_real(half, volume_shape=x.shape, axes=(0, 1))


def test_idft3_real_rejects_bad_volume_shape_rank():
    rng = np.random.default_rng(53)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)
    with pytest.raises(ValueError, match="must have 3 dims"):
        fourier_transform_utils.get_idft3_real(half, volume_shape=(4, 5))


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_dft_idft_roundtrip_gpu(gpu_device):
    rng = np.random.default_rng(0)
    x2 = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex64)

    cpu_out = np.asarray(fourier_transform_utils.get_idft2(fourier_transform_utils.get_dft2(x2)))

    with jax.default_device(gpu_device):
        x2_g = jax.device_put(jnp.array(x2), gpu_device)
        gpu_out = np.asarray(fourier_transform_utils.get_idft2(fourier_transform_utils.get_dft2(x2_g)))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_dft3_roundtrip_gpu(gpu_device):
    rng = np.random.default_rng(7)
    x3 = (rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4))).astype(np.complex64)

    cpu_out = np.asarray(fourier_transform_utils.get_idft3(fourier_transform_utils.get_dft3(x3)))

    with jax.default_device(gpu_device):
        x3_g = jax.device_put(jnp.array(x3), gpu_device)
        gpu_out = np.asarray(fourier_transform_utils.get_idft3(fourier_transform_utils.get_dft3(x3_g)))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_dft2_real_idft2_real_roundtrip_gpu(gpu_device):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((6, 10)).astype(np.float32)

    cpu_half = np.asarray(fourier_transform_utils.get_dft2_real(x))
    cpu_back = np.asarray(fourier_transform_utils.get_idft2_real(
        fourier_transform_utils.get_dft2_real(x), image_shape=x.shape
    ))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_half = np.asarray(fourier_transform_utils.get_dft2_real(x_g))
        gpu_back = np.asarray(fourier_transform_utils.get_idft2_real(
            fourier_transform_utils.get_dft2_real(x_g), image_shape=x.shape
        ))

    np.testing.assert_allclose(cpu_half, gpu_half, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_back, gpu_back, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_dft3_real_idft3_real_roundtrip_gpu(gpu_device):
    rng = np.random.default_rng(13)
    x = rng.standard_normal((6, 6, 10)).astype(np.float32)

    cpu_back = np.asarray(fourier_transform_utils.get_idft3_real(
        fourier_transform_utils.get_dft3_real(x), volume_shape=x.shape
    ))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_back = np.asarray(fourier_transform_utils.get_idft3_real(
            fourier_transform_utils.get_dft3_real(x_g), volume_shape=x.shape
        ))

    np.testing.assert_allclose(cpu_back, gpu_back, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_half_full_volume_roundtrip_gpu(gpu_device):
    rng = np.random.default_rng(61)
    volume_shape = (6, 5, 8)
    x = rng.standard_normal(volume_shape).astype(np.float32)
    half = fourier_transform_utils.get_dft3_real(x)

    cpu_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(half, volume_shape))
    cpu_half_back = np.asarray(fourier_transform_utils.full_volume_to_half_volume(
        fourier_transform_utils.half_volume_to_full_volume(half, volume_shape), volume_shape
    ))

    with jax.default_device(gpu_device):
        half_g = jax.device_put(jnp.array(np.asarray(half)), gpu_device)
        gpu_full = np.asarray(fourier_transform_utils.half_volume_to_full_volume(half_g, volume_shape))
        gpu_half_back = np.asarray(fourier_transform_utils.full_volume_to_half_volume(
            fourier_transform_utils.half_volume_to_full_volume(half_g, volume_shape), volume_shape
        ))

    np.testing.assert_allclose(cpu_full, gpu_full, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_half_back, gpu_half_back, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_get_k_coordinate_of_each_pixel_gpu(gpu_device):
    cpu_coords = np.asarray(fourier_transform_utils.get_k_coordinate_of_each_pixel((4, 6), voxel_size=1.5, scaled=True))

    with jax.default_device(gpu_device):
        gpu_coords = np.asarray(fourier_transform_utils.get_k_coordinate_of_each_pixel((4, 6), voxel_size=1.5, scaled=True))

    np.testing.assert_allclose(cpu_coords, gpu_coords, atol=1e-5, rtol=1e-5)
