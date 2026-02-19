import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core

pytestmark = pytest.mark.unit


def test_vol_vec_index_roundtrip():
    vol_shape = (4, 5, 6)
    vol_indices = np.array(
        [
            [[0, 0, 0], [1, 2, 3]],
            [[3, 4, 5], [2, 1, 0]],
        ],
        dtype=np.int32,
    )
    vec = core.vol_indices_to_vec_indices(vol_indices, vol_shape)
    back = core.vec_indices_to_vol_indices(vec, vol_shape)
    np.testing.assert_array_equal(np.asarray(back), vol_indices)


def test_vol_indices_to_vec_indices_clips_oob():
    vol_shape = (4, 4, 4)
    vol_indices = np.array([[-1, 10, 2], [3, 3, 8]], dtype=np.int32)
    vec = np.asarray(core.vol_indices_to_vec_indices(vol_indices, vol_shape))
    clipped = np.clip(vol_indices, [0, 0, 0], np.array(vol_shape) - 1)
    expected = np.ravel_multi_index(clipped.T, vol_shape)
    np.testing.assert_array_equal(vec, expected)


def test_frequency_conversions_roundtrip():
    vol_shape = (5, 5, 5)
    vol_indices = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]], dtype=np.int32)
    freqs = core.vol_indices_to_frequencies(vol_indices, vol_shape)
    back = core.frequencies_to_vol_indices(freqs, vol_shape)
    np.testing.assert_array_equal(np.asarray(back), vol_indices)

    vec = core.vol_indices_to_vec_indices(vol_indices, vol_shape)
    freqs2 = core.vec_indices_to_frequencies(vec, vol_shape)
    vec2 = core.frequencies_to_vec_indices(freqs2, vol_shape)
    np.testing.assert_array_equal(np.asarray(vec2), np.asarray(vec))


def test_bound_checks():
    grid_size = 4
    freqs = np.array([[-2, 0, 1], [2, 0, 0], [-2.1, 0, 0]])
    vol_idx = np.array([[0, 1, 2], [3, 3, 3], [4, 0, 0], [-1, 0, 0]])
    vec_idx = np.array([0, 63, 64, -1])

    np.testing.assert_array_equal(
        np.asarray(core.check_frequencies_in_bound(freqs, grid_size)),
        np.array([True, False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(core.check_vol_indices_in_bound(vol_idx, grid_size)),
        np.array([True, True, False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(core.check_vec_indices_in_bound(vec_idx, grid_size)),
        np.array([True, True, False, False]),
    )


def test_distance_and_round_helpers():
    d = np.array([0.0, 0.1, 1.0, 1.2, 2.9])
    np.testing.assert_array_equal(core.distance_to_max_grid_dist(d), np.array([0, 1, 1, 2, 3]))

    rounded = core.round_to_int(np.array([1.2, -1.8, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(rounded), np.array([1, -2, 0], dtype=np.int32))


def test_find_frequencies_within_grid_dist_integer_coords():
    coords = np.array([1, 2, 3], dtype=np.int32)
    out = np.asarray(core.find_frequencies_within_grid_dist(coords, 1))
    assert out.shape == (27, 3)
    assert np.any(np.all(out == np.array([1, 2, 3]), axis=1))


def test_find_frequencies_within_grid_dist_float_coords_rounds_to_int():
    coords = np.array([0.2, -0.7], dtype=np.float32)
    out = np.asarray(core.find_frequencies_within_grid_dist(coords, 1))
    assert out.shape == (9, 2)
    assert out.dtype == np.int32
    assert np.any(np.all(out == np.array([0, -1], dtype=np.int32), axis=1))


def test_unrotated_plane_helpers():
    pts_grid = np.asarray(core.get_unrotated_plane_grid_points((2, 2), three_d_upsampling_factor=2))
    expected_grid = np.array(
        [
            [-2, -2, 0],
            [0, -2, 0],
            [-2, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(pts_grid, expected_grid)

    pts_scaled = np.asarray(core.get_unrotated_plane_coords((2, 2), voxel_size=2.0, scaled=True))
    expected_scaled = np.array(
        [
            [-0.25, -0.25, 0.0],
            [0.0, -0.25, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pts_scaled, expected_scaled)


def test_gridpoint_coords_and_indices_identity_rotation():
    rot = np.eye(3, dtype=np.float32)
    image_shape = (2, 2)
    volume_shape = (4, 4, 4)
    coords = np.asarray(core.get_gridpoint_coords(rot, image_shape, volume_shape))
    expected_coords = np.array(
        [
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(coords, expected_coords)

    vec = np.asarray(core.get_nearest_gridpoint_indices(rot, image_shape, volume_shape))
    expected_vec = np.ravel_multi_index(expected_coords.astype(np.int32).T, volume_shape)
    np.testing.assert_array_equal(vec, expected_vec)


def test_decide_order():
    assert core.decide_order("nearest") == 0
    assert core.decide_order("linear_interp") == 1
    assert core.decide_order("cubic") == 3
    with pytest.raises(ValueError):
        core.decide_order("bad")


def test_ctf_param_index_values_are_stable():
    assert core.CTFParamIndex.DFU == 0
    assert core.CTFParamIndex.PHASE_SHIFT == 6
    assert core.CTFParamIndex.TILT_ANGLE == 10


def test_core_exports_include_expected_symbols():
    assert "forward_model_from_map" in core.__all__
    assert "evaluate_ctf_wrapper" in core.__all__
    assert "vol_indices_to_vec_indices" in core.__all__
