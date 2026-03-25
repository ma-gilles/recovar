import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.core.mask as mask

pytestmark = pytest.mark.unit


def test_get_radial_mask_and_window_mask():
    m = np.asarray(mask.get_radial_mask((8, 8, 8)))
    assert m.shape == (8, 8, 8)
    assert m.dtype == np.bool_
    assert m.any()

    w = mask.window_mask(8, in_rad=0.2, out_rad=0.8)
    assert w.shape == (8, 8)
    assert np.all((w >= 0) & (w <= 1))


def test_raised_cosine_mask_bounds():
    rc = np.asarray(mask.raised_cosine_mask((8, 8, 8), radius=2, radius_p=4, offset=-1))
    assert rc.shape == (8, 8, 8)
    assert np.all((rc >= 0) & (rc <= 1))
    assert np.max(rc) <= 1.0 + 1e-6


def test_soften_volume_mask_range():
    binary = np.zeros((8, 8, 8), dtype=np.float32)
    binary[2:6, 2:6, 2:6] = 1.0
    out = mask.soften_volume_mask(binary, kern_rad=2)
    assert out.shape == binary.shape
    assert out.dtype == np.float32
    assert np.all((out >= 0) & (out <= 1))


def test_smooth_circular_mask_shape_and_range():
    m = mask.smooth_circular_mask(image_size=16, radius=4, thickness=2)
    assert m.shape == (16, 16)
    assert np.all((m >= 0.0) & (m <= 1.0))


def test_smooth_circular_mask_interior_is_one():
    # Center pixel is at coords (0, 0) so r=0, which is < radius
    m = mask.smooth_circular_mask(image_size=10, radius=3, thickness=2)
    half = 10 // 2
    # coords[half] = 0 → r = 0 < radius
    assert m[half, half] == 1.0


def test_smooth_circular_mask_exterior_is_zero():
    m = mask.smooth_circular_mask(image_size=10, radius=3, thickness=2)
    # corner pixel: coords = (-5, -5), r = 5*sqrt(2) ≈ 7.07 >> radius+thickness=5
    assert m[0, 0] == 0.0


def test_smooth_circular_mask_exact_formula():
    """Verify every pixel matches the raised-cosine formula exactly."""
    image_size = 14
    radius = 4.0
    thickness = 3.0
    m = mask.smooth_circular_mask(image_size, radius=radius, thickness=thickness)

    half = image_size // 2
    coords = np.arange(-half, image_size - half, dtype=float)
    gx, gy = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(gx**2 + gy**2)

    for i in range(image_size):
        for j in range(image_size):
            ri = r[i, j]
            if ri < radius:
                expected = 1.0
            elif ri <= radius + thickness:
                expected = 0.5 + 0.5 * np.cos(np.pi * (ri - radius) / thickness)
            else:
                expected = 0.0
            assert abs(m[i, j] - expected) < 1e-10, (
                f"Mismatch at [{i},{j}]: r={ri:.4f}, expected={expected:.6f}, got={m[i, j]:.6f}"
            )


def test_smooth_circular_mask_symmetric():
    """The mask must be centrosymmetric (same value at (i,j) and (j,i))."""
    m = mask.smooth_circular_mask(image_size=12, radius=3, thickness=2)
    np.testing.assert_allclose(m, m.T, atol=1e-12)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_soften_volume_mask_gpu(gpu_device):
    binary = np.zeros((8, 8, 8), dtype=np.float32)
    binary[2:6, 2:6, 2:6] = 1.0

    cpu_out = np.asarray(mask.soften_volume_mask(binary, kern_rad=2))

    with jax.default_device(gpu_device):
        binary_g = jax.device_put(jnp.array(binary), gpu_device)
        gpu_out = np.asarray(mask.soften_volume_mask(binary_g, kern_rad=2))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)
