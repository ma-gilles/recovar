"""Indexed backprojection with a traced radius inside a larger capacity cube.

RELION's backprojector keeps a sample when its image radius and its rotated,
volume-grid radius are inside r_max, and skips a sample whose trilinear stencil
leaves the compact r_max cube (acc/cuda/cuda_kernels/BP.cuh). A caller whose
volume is a stable capacity class larger than RELION's current-size cube passes
the capacity radius as the static ``max_r`` and RELION's radius as the traced
``runtime_max_r``: every cutoff must then be RELION's, so the capacity cube
cropped to RELION's cube equals the logical backprojection.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.core import slicing

pytestmark = pytest.mark.unit

PAD = 2
LOGICAL_R, CAPACITY_R = 28, 32
LOGICAL_SIZE = 2 * (int(PAD * LOGICAL_R + 0.5) + 1) + 1  # 115
CAPACITY_SIZE = 2 * (int(PAD * CAPACITY_R + 0.5) + 1) + 1  # 131
IMAGE_SHAPE = (128, 128)


def _crop_x_half(values, capacity_size, logical_size):
    """Centered z/y crop and x-half prefix of a RELION (z, y, x>=0) accumulator."""

    start = (capacity_size - logical_size) // 2
    grid = np.asarray(values).reshape(capacity_size, capacity_size, capacity_size // 2 + 1)
    return grid[start : start + logical_size, start : start + logical_size, : logical_size // 2 + 1].reshape(-1)


def _volume(size, dtype):
    return jnp.zeros(size * size * (size // 2 + 1), dtype)


def test_jax_path_refuses_a_runtime_radius(monkeypatch):
    monkeypatch.setattr(slicing, "_use_cuda_backproject", lambda order: False)
    with pytest.raises(NotImplementedError, match="runtime backprojection radius"):
        slicing.adjoint_slice_volume_indexed(
            jnp.zeros((1, 4), jnp.complex64),
            jnp.arange(4),
            jnp.eye(3)[None],
            IMAGE_SHAPE,
            (CAPACITY_SIZE,) * 3,
            "linear_interp",
            half_image=True,
            half_volume=True,
            max_r=float(CAPACITY_R),
            runtime_max_r=jnp.float32(LOGICAL_R),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("radius,inside", [(27, True), (28, True), (29, False)])
def test_runtime_radius_keeps_relions_edge_samples(radius, inside):
    """Pixel (0, radius) is kept exactly when radius <= r_max, in the capacity cube as in RELION's."""

    assert jax.default_backend() == "gpu"
    weight = cb.backproject_indexed(
        _volume(CAPACITY_SIZE, jnp.float32),
        jnp.ones((1, 1), jnp.float32),
        jnp.asarray([radius], dtype=jnp.int32),  # packed half-image pixel (ky, kx) = (0, radius)
        jnp.asarray(np.eye(3, dtype=np.float32)[None]),
        IMAGE_SHAPE,
        (CAPACITY_SIZE,) * 3,
        order=1,
        half_volume=True,
        half_image=True,
        max_r=float(CAPACITY_R),
        relion_x_half=True,
        runtime_max_r=jnp.float32(LOGICAL_R),
    )
    total = float(jnp.sum(jax.block_until_ready(weight)))
    assert (total > 0.5) is inside, total


@pytest.mark.gpu
def test_capacity_cube_with_runtime_radius_crops_to_the_logical_backprojection():
    """Random rotations of every pixel out to the capacity radius, edge samples included."""

    assert jax.default_backend() == "gpu"
    rng = np.random.default_rng(11)
    half_width = IMAGE_SHAPE[1] // 2 + 1
    rows, kx = np.meshgrid(np.arange(IMAGE_SHAPE[0]), np.arange(half_width), indexing="ij")
    ky = np.where(rows <= IMAGE_SHAPE[0] // 2, rows, rows - IMAGE_SHAPE[0])
    pixels = (rows * half_width + kx)[ky**2 + kx**2 <= CAPACITY_R**2].astype(np.int32)
    values = (rng.normal(size=(8, pixels.size)) + 1j * rng.normal(size=(8, pixels.size))).astype(np.complex64)
    q = rng.normal(size=(8, 4))
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix().astype(np.float32)
    common = dict(order=1, half_volume=True, half_image=True, relion_x_half=True)
    operands = (jnp.asarray(values), jnp.asarray(pixels), jnp.asarray(rotations))
    logical = np.asarray(
        cb.backproject_indexed(
            _volume(LOGICAL_SIZE, jnp.complex64), *operands, IMAGE_SHAPE, (LOGICAL_SIZE,) * 3,
            max_r=float(LOGICAL_R), **common,
        )
    )
    capacity = np.asarray(
        cb.backproject_indexed(
            _volume(CAPACITY_SIZE, jnp.complex64), *operands, IMAGE_SHAPE, (CAPACITY_SIZE,) * 3,
            max_r=float(CAPACITY_R), runtime_max_r=jnp.float32(LOGICAL_R), **common,
        )
    )
    cropped = _crop_x_half(capacity, CAPACITY_SIZE, LOGICAL_SIZE)
    # Nothing lands outside RELION's cube, and the capacity cube receives exactly
    # RELION's samples: the same voxels are written.
    assert np.abs(capacity).sum() == pytest.approx(np.abs(cropped).sum(), rel=1e-6)
    assert np.array_equal(cropped != 0, logical != 0)
    # The values agree to float32 rounding: the grid centre moves from 57 to 65,
    # which rounds the trilinear weights (rk + c) differently, and the atomics
    # add in another order. Measured on an H100 (14475451): 13 of 767050 voxels
    # beyond rtol 1e-5 / atol 1e-6 * max, by at most 1.9e-5 at max |x| = 6.
    np.testing.assert_allclose(cropped, logical, rtol=1e-5, atol=1e-5 * np.abs(logical).max())
    # The static capacity radius alone would keep the samples RELION skips.
    wide = np.asarray(
        cb.backproject_indexed(
            _volume(CAPACITY_SIZE, jnp.complex64), *operands, IMAGE_SHAPE, (CAPACITY_SIZE,) * 3,
            max_r=float(CAPACITY_R), **common,
        )
    )
    assert not np.allclose(_crop_x_half(wide, CAPACITY_SIZE, LOGICAL_SIZE), logical, rtol=1e-5, atol=1e-6 * np.abs(logical).max())
