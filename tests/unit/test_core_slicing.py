import numpy as np
import pytest

pytest.importorskip("jax")
import recovar.core as core
import recovar.core_slicing as core_slicing

pytestmark = pytest.mark.unit


def test_core_reexports_slicing_api():
    assert core.decide_order is core_slicing.decide_order
    assert core.slice_volume_by_nearest is core_slicing.slice_volume_by_nearest
    assert core.get_trilinear_weights_and_vol_indices is core_slicing.get_trilinear_weights_and_vol_indices


def test_decide_order_values():
    assert core_slicing.decide_order("nearest") == 0
    assert core_slicing.decide_order("linear_interp") == 1
    assert core_slicing.decide_order("cubic") == 3
    with pytest.raises(ValueError):
        core_slicing.decide_order("bad")


def test_slice_volume_by_nearest_and_forward_model():
    volume = np.array([10, 20, 30, 40], dtype=np.complex64)
    idx = np.array([[0, 2], [1, 3]], dtype=np.int32)
    sliced = np.asarray(core_slicing.slice_volume_by_nearest(volume, idx))
    np.testing.assert_array_equal(sliced, np.array([[10, 30], [20, 40]], dtype=np.complex64))

    ctf = np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]], dtype=np.complex64)
    forward = np.asarray(core_slicing.forward_model(volume, ctf, idx))
    np.testing.assert_array_equal(forward, sliced * ctf)


def test_summed_adjoint_slice_by_nearest_accumulates():
    volume_size = 4
    image_vecs = np.array([[1, 2], [3, 4]], dtype=np.float32)
    idx = np.array([[0, 2], [0, 2]], dtype=np.int32)
    out = np.asarray(core_slicing.summed_adjoint_slice_by_nearest(volume_size, image_vecs, idx))
    np.testing.assert_array_equal(out, np.array([4, 0, 6, 0], dtype=np.float32))


def test_get_trilinear_weights_and_vol_indices_simple_cases():
    # Integer coordinate should place full weight on one grid point.
    grid_coords = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    points, weights = core_slicing.get_trilinear_weights_and_vol_indices(grid_coords, (4, 4, 4))
    points = np.asarray(points)
    weights = np.asarray(weights)
    assert points.shape == (1, 8, 3)
    assert weights.shape == (1, 8)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights.max(), 1.0)

    # Out-of-bounds coordinate should have zero total weight after masking.
    grid_coords_oob = np.array([[-2.0, -2.0, -2.0]], dtype=np.float32)
    _, weights_oob = core_slicing.get_trilinear_weights_and_vol_indices(grid_coords_oob, (4, 4, 4))
    assert np.isclose(np.asarray(weights_oob).sum(), 0.0)


def test_adjoint_slice_volume_by_trilinear_from_weights_accumulates():
    images = np.array([2.0, 3.0], dtype=np.float32)
    grid_vec_indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    weights = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    out = np.asarray(
        core_slicing.adjoint_slice_volume_by_trilinear_from_weights(
            images, grid_vec_indices, weights, volume_shape=(3, 1, 1)
        )
    )
    expected = np.array([1.0, 1.0 + 0.75, 2.25], dtype=np.float32)
    np.testing.assert_allclose(out[:3], expected)
