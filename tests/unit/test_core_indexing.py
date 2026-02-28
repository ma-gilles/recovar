import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core_indexing as core_indexing

pytestmark = pytest.mark.unit


def test_core_reexports_indexing_api():
    assert core.vol_indices_to_vec_indices is core_indexing.vol_indices_to_vec_indices
    assert core.vec_indices_to_vol_indices is core_indexing.vec_indices_to_vol_indices
    assert core.check_vec_indices_in_bound is core_indexing.check_vec_indices_in_bound


def test_core_indexing_vec_bound_checks():
    grid_size = 4
    vec_idx = np.array([0, 63, 64, -1])
    np.testing.assert_array_equal(
        np.asarray(core_indexing.check_vec_indices_in_bound(vec_idx, grid_size)),
        np.array([True, True, False, False]),
    )


def test_core_indexing_frequency_and_volume_bounds():
    freqs = np.array([[-2, 0, 1], [2, 0, 0], [-2.1, 0, 0]])
    vol_idx = np.array([[0, 1, 2], [3, 3, 3], [4, 0, 0], [-1, 0, 0]])
    np.testing.assert_array_equal(
        np.asarray(core_indexing.check_frequencies_in_bound(freqs, 4)),
        np.array([True, False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(core_indexing.check_vol_indices_in_bound(vol_idx, 4)),
        np.array([True, True, False, False]),
    )


def test_core_indexing_distance_helper():
    d = np.array([0.0, 0.1, 1.0, 1.2, 2.9])
    np.testing.assert_array_equal(core_indexing.distance_to_max_grid_dist(d), np.array([0, 1, 1, 2, 3]))


def test_core_indexing_frequency_roundtrip_even_and_odd_grids():
    for vol_shape in [(4, 4, 4), (5, 5, 5)]:
        vol_indices = np.array([[0, 0, 0], [1, 2, 3], [vol_shape[0] - 1, vol_shape[1] - 1, vol_shape[2] - 1]])
        freqs = core_indexing.vol_indices_to_frequencies(vol_indices, vol_shape)
        back = core_indexing.frequencies_to_vol_indices(freqs, vol_shape)
        np.testing.assert_array_equal(np.asarray(back), vol_indices)


def test_core_indexing_frequency_vec_conversion_roundtrip():
    vol_shape = (4, 4, 4)
    vec = np.array([0, 5, 17, 63], dtype=np.int32)
    freqs = core_indexing.vec_indices_to_frequencies(vec, vol_shape)
    vec_back = core_indexing.frequencies_to_vec_indices(freqs, vol_shape)
    np.testing.assert_array_equal(np.asarray(vec_back), vec)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------
import jax


@pytest.mark.gpu
def test_vol_vec_index_roundtrip_on_gpu(gpu_device):
    vol_shape = (4, 5, 6)
    vol_indices = np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]], dtype=np.int32)
    with jax.default_device(gpu_device):
        vec = core_indexing.vol_indices_to_vec_indices(jax.device_put(vol_indices), vol_shape)
        back = np.asarray(core_indexing.vec_indices_to_vol_indices(vec, vol_shape))
    np.testing.assert_array_equal(back, vol_indices)


@pytest.mark.gpu
def test_frequency_roundtrip_on_gpu(gpu_device):
    for vol_shape in [(4, 4, 4), (5, 5, 5)]:
        vol_indices = np.array(
            [[0, 0, 0], [1, 2, 3], [vol_shape[0] - 1, vol_shape[1] - 1, vol_shape[2] - 1]], dtype=np.int32
        )
        with jax.default_device(gpu_device):
            freqs = core_indexing.vol_indices_to_frequencies(jax.device_put(vol_indices), vol_shape)
            back = np.asarray(core_indexing.frequencies_to_vol_indices(freqs, vol_shape))
        np.testing.assert_array_equal(back, vol_indices)


@pytest.mark.gpu
def test_frequency_vec_roundtrip_on_gpu(gpu_device):
    vol_shape = (4, 4, 4)
    vec = np.array([0, 5, 17, 63], dtype=np.int32)
    with jax.default_device(gpu_device):
        freqs = core_indexing.vec_indices_to_frequencies(jax.device_put(vec), vol_shape)
        vec_back = np.asarray(core_indexing.frequencies_to_vec_indices(freqs, vol_shape))
    np.testing.assert_array_equal(vec_back, vec)


@pytest.mark.gpu
def test_check_frequencies_in_bound_on_gpu(gpu_device):
    freqs = np.array([[-2, 0, 1], [2, 0, 0], [-2.1, 0, 0]], dtype=np.float32)
    cpu_out = np.asarray(core_indexing.check_frequencies_in_bound(freqs, 4))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_indexing.check_frequencies_in_bound(jax.device_put(freqs), 4))
    np.testing.assert_array_equal(gpu_out, cpu_out)


@pytest.mark.gpu
def test_check_vol_indices_in_bound_on_gpu(gpu_device):
    vol_idx = np.array([[0, 1, 2], [3, 3, 3], [4, 0, 0], [-1, 0, 0]], dtype=np.int32)
    cpu_out = np.asarray(core_indexing.check_vol_indices_in_bound(vol_idx, 4))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_indexing.check_vol_indices_in_bound(jax.device_put(vol_idx), 4))
    np.testing.assert_array_equal(gpu_out, cpu_out)


@pytest.mark.gpu
def test_check_vec_indices_in_bound_on_gpu(gpu_device):
    vec_idx = np.array([0, 63, 64, -1], dtype=np.int32)
    cpu_out = np.asarray(core_indexing.check_vec_indices_in_bound(vec_idx, 4))
    with jax.default_device(gpu_device):
        gpu_out = np.asarray(core_indexing.check_vec_indices_in_bound(jax.device_put(vec_idx), 4))
    np.testing.assert_array_equal(gpu_out, cpu_out)
