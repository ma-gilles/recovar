"""Explicit volume padding for indexed backprojection of images on another grid.

RELION's backprojector keeps a sample when its rotated, volume-grid radius is inside
r_max (acc/cuda/cuda_kernels/BP.cuh, ``xp*xp+yp*yp+zp*zp > max_r2_vol``). An optics
group on another pixel size or box projects with rotations scaled by 1/s, so its image
pixel |k| lands at |k|/s: the image-side clip radius is r_max*s, and the padding can no
longer be read off the image shape and that radius.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.core import slicing

pytestmark = pytest.mark.unit

# RELION BackProjector at current size 56, padding 2: pad_size = 2*(int(2*28+0.5)+1)+1.
R_MAX, PAD, PAD_SIZE = 28, 2, 115
SCALE = 112 * 5.44 / (128 * 4.25)  # a 112 px / 5.44 A group against a 128 px / 4.25 A reference


def test_default_inference_is_unchanged():
    assert cb._infer_backproject_upsampling((128, 128), (PAD_SIZE,) * 3, max_r=R_MAX) == PAD
    assert cb._infer_backproject_upsampling((64, 64), (128,) * 3) == 2


def test_class_radius_breaks_inference_and_explicit_upsampling_restores_it():
    # The image-side radius r_max*s does not reproduce the pad size; the image-size
    # fallback would silently return 1 for a 112 px image on a 115 grid.
    assert cb._infer_backproject_upsampling((112, 112), (PAD_SIZE,) * 3, max_r=R_MAX * SCALE) == 1
    assert cb._infer_backproject_upsampling((112, 112), (PAD_SIZE,) * 3, max_r=R_MAX * SCALE, upsampling=PAD) == PAD
    with pytest.raises(ValueError, match="positive integer"):
        cb._infer_backproject_upsampling((112, 112), (PAD_SIZE,) * 3, upsampling=0)
    with pytest.raises(ValueError, match="positive integer"):
        cb._infer_backproject_upsampling((112, 112), (PAD_SIZE,) * 3, upsampling=1.5)


def test_jax_path_refuses_explicit_upsampling(monkeypatch):
    monkeypatch.setattr(slicing, "_use_cuda_backproject", lambda order: False)
    with pytest.raises(NotImplementedError, match="explicit upsampling"):
        slicing.adjoint_slice_volume_indexed(
            jnp.zeros((1, 4), jnp.complex64),
            jnp.arange(4),
            jnp.eye(3)[None],
            (112, 112),
            (PAD_SIZE,) * 3,
            "linear_interp",
            half_image=True,
            half_volume=True,
            max_r=R_MAX * SCALE,
            upsampling=PAD,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("radius,inside", [(27, True), (31, True), (32, False)])
def test_class_pixels_follow_relions_rotated_radius(radius, inside):
    """Image pixel |k| is kept exactly when |k|/s <= r_max (31/1.12 = 27.7, 32/1.12 = 28.6)."""

    assert jax.default_backend() == "gpu"
    image_shape, volume_shape = (112, 112), (PAD_SIZE,) * 3
    rotation = (np.eye(3, dtype=np.float32) / np.float32(SCALE))[None]
    volume_size = PAD_SIZE * PAD_SIZE * (PAD_SIZE // 2 + 1)
    weight = cb.backproject_indexed(
        jnp.zeros(volume_size, jnp.float32),
        jnp.ones((1, 1), jnp.float32),
        jnp.asarray([radius], dtype=jnp.int32),  # packed half-image pixel (ky, kx) = (0, radius)
        jnp.asarray(rotation),
        image_shape,
        volume_shape,
        order=1,
        half_volume=True,
        half_image=True,
        max_r=R_MAX * SCALE,
        relion_x_half=True,
        upsampling=PAD,
    )
    total = float(jnp.sum(jax.block_until_ready(weight)))
    assert (total > 0.5) is inside, total


@pytest.mark.gpu
def test_explicit_upsampling_reproduces_the_inferred_geometry():
    """On one grid, passing the inferred padding explicitly gives the default output."""

    assert jax.default_backend() == "gpu"
    rng = np.random.default_rng(7)
    image_shape, volume_shape = (128, 128), (PAD_SIZE,) * 3
    half_width = image_shape[1] // 2 + 1
    ky, kx = np.meshgrid(np.arange(image_shape[0]), np.arange(half_width), indexing="ij")
    ky = np.where(ky <= image_shape[0] // 2, ky, ky - image_shape[0])
    pixels = (np.meshgrid(np.arange(image_shape[0]), np.arange(half_width), indexing="ij")[0] * half_width + kx)[
        ky**2 + kx**2 <= R_MAX**2
    ].astype(np.int32)
    values = (rng.normal(size=(6, pixels.size)) + 1j * rng.normal(size=(6, pixels.size))).astype(np.complex64)
    q = rng.normal(size=(6, 4))
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix().astype(np.float32)
    volume_size = PAD_SIZE * PAD_SIZE * (PAD_SIZE // 2 + 1)
    kwargs = dict(order=1, half_volume=True, half_image=True, max_r=float(R_MAX), relion_x_half=True)
    args = (jnp.zeros(volume_size, jnp.complex64), jnp.asarray(values), jnp.asarray(pixels), jnp.asarray(rotations))
    inferred = np.asarray(cb.backproject_indexed(*args, image_shape, volume_shape, **kwargs))
    explicit = np.asarray(cb.backproject_indexed(*args, image_shape, volume_shape, **kwargs, upsampling=PAD))
    np.testing.assert_allclose(explicit, inferred, rtol=1e-5, atol=1e-6 * np.abs(inferred).max())
