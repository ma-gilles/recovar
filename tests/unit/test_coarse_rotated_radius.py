"""RELION coarse image clipping retains rotated float32 boundary samples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.dense_single_volume.helpers import projection, significance

pytestmark = pytest.mark.unit

# Actual GF46 iteration2 coarse row70, captured in job13571299. With this
# rotation, (x,y)=(7,-11) has integer rotated float32 squared radius169,
# despite exact source-pixel squared radius170. This changed top100 support.
_ROTATION = np.array(
    [
        [-0.16886791586875916, 0.9150263071060181, -0.36634740233421326],
        [-0.725357711315155, -0.36702489852905273, -0.5823649168014526],
        [-0.6673378348350525, 0.16739018261432648, 0.7257001399993896],
    ],
    dtype=np.float32,
)[None]


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True)])
def test_optin(monkeypatch, token, expected):
    monkeypatch.delenv("RECOVAR_K1_COARSE_ROTATED_RADIUS", raising=False)
    if token is not None:
        monkeypatch.setenv("RECOVAR_K1_COARSE_ROTATED_RADIUS", token)
    assert significance._coarse_rotated_radius_enabled() is expected


def test_invalid_optin(monkeypatch):
    monkeypatch.setenv("RECOVAR_K1_COARSE_ROTATED_RADIUS", "typo")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        significance._coarse_rotated_radius_enabled()


@pytest.mark.parametrize("radius", [np.float32(13), np.array([13], np.int32)])
def test_image_radius_validated_before_loading_cuda(monkeypatch, radius):
    monkeypatch.setattr(cb, "_ensure_projector_image_radius_ffi", lambda: pytest.fail("loaded before validation"))
    with pytest.raises(ValueError, match="image_r_max must be an S32 scalar"):
        cb.project_relion_half_capacity(
            jnp.ones((33, 33, 17), jnp.complex64),
            jnp.asarray(_ROTATION),
            jnp.int32(15),
            image_shape=(32, 32),
            image_r_max=radius,
        )


def test_exact_mask_cannot_erase_rotated_boundary(monkeypatch):
    with pytest.raises(ValueError, match="cannot be combined"):
        projection._project_relion_projector_texture(
            None,
            None,
            (128, 128),
            r_max=15,
            projector_output_size=32,
            image_r_max=jnp.int32(13),
            mask_current_image_disk=True,
        )


def test_compact_boundary_survives_runtime_image_radius(monkeypatch):
    calls = []

    def projected(half, rotations, radius, **kw):
        calls.append((half.shape, radius.aval, kw["image_r_max"].aval))
        return jnp.ones((1, 32 * 17), jnp.complex64)

    monkeypatch.setattr(cb, "project_relion_half_capacity", projected)

    @jax.jit
    def run(radius):
        return projection._project_relion_projector_texture(
            jnp.ones((33, 33, 17), jnp.complex64),
            jnp.asarray(_ROTATION),
            (128, 128),
            r_max=15,
            projector_output_size=32,
            image_r_max=radius,
            mask_current_image_disk=False,
            pixel_indices=np.array([(64 - 11) * 65 + 7]),
        )

    for radius in (12, 13):
        np.testing.assert_array_equal(run(jnp.int32(radius)), [[1 + 0j]])
    assert run._cache_size() == 1 and len(calls) == 1
    assert calls[0][0] == (33, 33, 17)
    assert calls[0][1].shape == calls[0][2].shape == ()


@pytest.mark.gpu
def test_gpu_rotated_boundary_and_original_texture():
    assert jax.default_backend() == "gpu"
    values = np.arange(33 * 33 * 17, dtype=np.float32).reshape(33, 33, 17)
    volume = jnp.asarray((values + 1j * (values + 1)).astype(np.complex64))
    rotations = jnp.asarray(_ROTATION)
    old = cb.project_relion_half_capacity(volume, rotations, jnp.int32(15), image_shape=(32, 32))
    clipped = cb.project_relion_half_capacity(
        volume,
        rotations,
        jnp.int32(15),
        image_shape=(32, 32),
        image_r_max=jnp.int32(13),
    )
    equal_radius = cb.project_relion_half_capacity(
        volume,
        rotations,
        jnp.int32(15),
        image_shape=(32, 32),
        image_r_max=jnp.int32(15),
    )
    np.testing.assert_array_equal(np.asarray(equal_radius).view(np.uint32), np.asarray(old).view(np.uint32))
    old, clipped = np.asarray(old).reshape(32, 17), np.asarray(clipped).reshape(32, 17)
    kept = clipped != 0
    np.testing.assert_array_equal(clipped[kept].view(np.uint32), old[kept].view(np.uint32))
    assert clipped[16 - 11, 7] != 0  # Exact image-disk masking loses this sample.
    assert clipped[16 + 12, 12] == 0
    assert clipped[16, 0] == old[16, 0]


@pytest.mark.gpu
@pytest.mark.parametrize("radius", [-1, 17])
def test_gpu_invalid_image_radius_is_nan(radius):
    assert jax.default_backend() == "gpu"
    result = cb.project_relion_half_capacity(
        jnp.ones((33, 33, 17), jnp.complex64),
        jnp.asarray(_ROTATION),
        jnp.int32(15),
        image_shape=(32, 32),
        image_r_max=jnp.int32(radius),
    )
    assert np.isnan(np.asarray(result)).all()
