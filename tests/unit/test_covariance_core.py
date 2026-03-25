import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.heterogeneity import covariance_core as cc
from recovar import core

pytestmark = pytest.mark.unit


def test_pick_frequencies_half_and_full():
    freq = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [3, 0, 0],
        ],
        dtype=np.float32,
    )
    half = np.asarray(cc.pick_frequencies(freq, radius=2, use_half=True))
    full = np.asarray(cc.pick_frequencies(freq, radius=2, use_half=False))
    assert np.all(full[:3])
    assert not full[3]
    assert half[0] and half[1]
    assert not half[2]  # filtered by freq[...,0] >= 0 in half mode


def test_get_picked_frequencies_matches_radius_filter():
    shape = (4, 4, 4)
    picked = np.asarray(cc.get_picked_frequencies(shape, radius=1, use_half=True))
    freqs = np.asarray(core.vec_indices_to_frequencies(picked, shape))
    assert np.all(np.linalg.norm(freqs, axis=-1) <= 1 + 1e-6)
    assert np.all(freqs[:, 0] >= 0)


def test_check_mask_accepts_none_and_all_ones():
    assert cc.check_mask(None) is True
    assert cc.check_mask(np.ones((2, 2), dtype=np.float32)) is True
    assert cc.check_mask(np.array([[1.0, 0.0]], dtype=np.float32)) is False


def test_evaluate_kernel_on_grid_triangular_and_square():
    grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    tgt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tri = np.asarray(cc.evaluate_kernel_on_grid(grid, tgt, kernel="triangular", kernel_width=1.0))
    sq = np.asarray(cc.evaluate_kernel_on_grid(grid, tgt, kernel="square", kernel_width=1.0))
    assert tri[0] > tri[1] > tri[2]
    assert tri[2] == 0.0
    assert sq[0] > 0.0 and sq[1] == 0.0 and sq[2] == 0.0


def test_sum_up_over_near_grid_points_weighted_sum():
    image = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    grid = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    tgt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out_tri = float(cc.sum_up_over_near_grid_points(image, grid, tgt, kernel="triangular", kernel_width=1.0))
    out_sq = float(cc.sum_up_over_near_grid_points(image, grid, tgt, kernel="square", kernel_width=1.0))
    assert out_tri > out_sq  # triangular includes first two points, square includes only first


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_evaluate_kernel_on_grid_gpu(gpu_device):
    grid = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    tgt = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cpu_tri = np.asarray(cc.evaluate_kernel_on_grid(grid, tgt, kernel="triangular", kernel_width=1.0))

    with jax.default_device(gpu_device):
        grid_g = jax.device_put(jnp.array(grid), gpu_device)
        tgt_g = jax.device_put(jnp.array(tgt), gpu_device)
        gpu_tri = np.asarray(cc.evaluate_kernel_on_grid(grid_g, tgt_g, kernel="triangular", kernel_width=1.0))

    np.testing.assert_allclose(cpu_tri, gpu_tri, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_sum_up_over_near_grid_points_gpu(gpu_device):
    image = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    grid = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    tgt = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cpu_out = float(cc.sum_up_over_near_grid_points(image, grid, tgt, kernel="triangular", kernel_width=1.0))

    with jax.default_device(gpu_device):
        image_g = jax.device_put(jnp.array(image), gpu_device)
        grid_g = jax.device_put(jnp.array(grid), gpu_device)
        tgt_g = jax.device_put(jnp.array(tgt), gpu_device)
        gpu_out = float(cc.sum_up_over_near_grid_points(image_g, grid_g, tgt_g, kernel="triangular", kernel_width=1.0))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_pick_frequencies_gpu(gpu_device):
    freq = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [3, 0, 0]], dtype=np.float32)

    cpu_half = np.asarray(cc.pick_frequencies(freq, radius=2, use_half=True))

    with jax.default_device(gpu_device):
        freq_g = jax.device_put(jnp.array(freq), gpu_device)
        gpu_half = np.asarray(cc.pick_frequencies(freq_g, radius=2, use_half=True))

    np.testing.assert_array_equal(cpu_half, gpu_half)
