"""Padding-factor plumbing of the fused RELION coarse projector kernel.

RELION's ``AccProjectorKernel::project3Dmodel`` multiplies the rotated pixel
coordinates by the Projector padding factor and tests them against
``maxR2_padded`` before the texture fetch.  Scaling by 2 is exact in IEEE
arithmetic, so a padding factor of 2 with rotation ``R`` and radius ``r`` must
reproduce, bitwise, a padding factor of 1 with rotation ``2 R`` and radius
``2 r`` on the same padded volume, provided the score radius is not clipped by
``current_size / 2``.  A padding factor of 1 must reproduce the pre-existing
contract (default argument).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def _inputs(rng, *, current_size, model_max_r, padding_factor, rotation_count, translation_count, batch):
    projector_size = 2 * model_max_r * padding_factor + 3
    compact_pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0.0, 0.02, (projector_size,) * 3)
        + 1j * rng.normal(0.0, 0.02, (projector_size,) * 3)
    ).astype(np.complex64)
    angles = rng.uniform(0.0, 2 * np.pi, (rotation_count, 3))
    rotations = np.stack([_euler_matrix(*a) for a in angles]).astype(np.float32)
    images = (
        rng.normal(0.0, 0.02, (batch, compact_pixel_count))
        + 1j * rng.normal(0.0, 0.02, (batch, compact_pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.uniform(-0.2, 0.2, (translation_count, 2)).astype(np.float32)
    weight = rng.uniform(0.1, 3.0, images.shape).astype(np.float32)
    initial_diff2 = rng.uniform(5.0, 15.0, (batch,)).astype(np.float32)
    lookup = np.arange(compact_pixel_count, dtype=np.int32)
    return projector, rotations, images, translation_angles, weight, initial_diff2, lookup


def _euler_matrix(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz2 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz1 @ ry @ rz2


@pytest.mark.gpu
def test_padding_factor_two_matches_scaled_rotations_at_padding_one(monkeypatch, custom_cuda_lib, gpu_device):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1234)
    model_max_r = 3
    current_size = 16  # current_size / 2 = 8 >= 2 * model_max_r, so no score-radius clipping
    projector, rotations, images, angles, weight, initial, lookup = _inputs(
        rng, current_size=current_size, model_max_r=model_max_r, padding_factor=2,
        rotation_count=131, translation_count=21, batch=3,
    )
    with jax.default_device(gpu_device):
        padded = cuda_backproject.relion_coarse_diff2_projector_f32(
            jnp.asarray(projector), jnp.asarray(rotations), jnp.asarray(images), jnp.asarray(angles),
            jnp.asarray(weight), jnp.asarray(initial), jnp.asarray(lookup),
            current_size=current_size, physical_image_size=current_size, model_max_r=model_max_r, padding_factor=2,
        )
        scaled = cuda_backproject.relion_coarse_diff2_projector_f32(
            jnp.asarray(projector), jnp.asarray(2.0 * rotations), jnp.asarray(images), jnp.asarray(angles),
            jnp.asarray(weight), jnp.asarray(initial), jnp.asarray(lookup),
            current_size=current_size, physical_image_size=current_size, model_max_r=2 * model_max_r, padding_factor=1,
        )
        padded = np.asarray(padded)
        scaled = np.asarray(scaled)
    assert padded.shape == (3, 131, 21)
    assert np.isfinite(padded).all()
    # The padded run must not be a trivial copy of the initial diff2 (projection reached the texture).
    assert np.any(np.abs(padded - initial[:, None, None]) > 1e-6)
    np.testing.assert_array_equal(padded, scaled)


@pytest.mark.gpu
def test_padding_factor_default_is_one(monkeypatch, custom_cuda_lib, gpu_device):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(99)
    projector, rotations, images, angles, weight, initial, lookup = _inputs(
        rng, current_size=8, model_max_r=2, padding_factor=1, rotation_count=129, translation_count=13, batch=2,
    )
    with jax.default_device(gpu_device):
        common = dict(current_size=8, physical_image_size=8, model_max_r=2)
        args = [jnp.asarray(x) for x in (projector, rotations, images, angles, weight, initial, lookup)]
        default = np.asarray(cuda_backproject.relion_coarse_diff2_projector_f32(*args, **common))
        explicit = np.asarray(cuda_backproject.relion_coarse_diff2_projector_f32(*args, padding_factor=1, **common))
    np.testing.assert_array_equal(default, explicit)


def test_padding_factor_must_be_positive():
    from recovar.em.scoring import significance

    assert significance._k1_coarse_fused_projector_supports_padding(1)
    assert significance._k1_coarse_fused_projector_supports_padding(2)
    assert not significance._k1_coarse_fused_projector_supports_padding(0)
