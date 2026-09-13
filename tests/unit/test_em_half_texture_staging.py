"""Remove cubic staging without changing the current projection postprocessing."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.helpers import projection as p

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("padding_factor", [1, 2])
@pytest.mark.parametrize("mask_disk", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dense_scale", [False, True])
def test_half_staging_preserves_current_crop_mask_and_scaling(
    monkeypatch,
    padding_factor,
    mask_disk,
    compact,
    dense_scale,
):
    size = 32 * padding_factor + 3
    slab = jnp.zeros((size, size, size // 2 + 1), dtype=jnp.complex64)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    raw = jnp.asarray(np.arange(32 * 17).reshape(1, -1), dtype=jnp.complex64)
    indices = jnp.asarray([40 * 25, 9 * 25 + 1, 24 * 25, 39 * 25 + 16], jnp.int32) if compact else None
    calls = []
    monkeypatch.setattr(p, "_cuda_projection_available", lambda: True)

    def direct(half, rows, radius, *, image_shape, padding_factor):
        assert half is slab and rows is rotations
        assert radius.shape == () and radius.dtype == jnp.int32 and int(radius) == 16
        assert image_shape == (32, 32)
        calls.append(padding_factor)
        return raw

    monkeypatch.setattr(cb, "project_relion_half_capacity", direct)
    monkeypatch.setattr(
        p, "relion_projector_half_to_texture_full", lambda *_: pytest.fail("full cubic staging on eligible half input")
    )
    # Independent use of the pre-existing output transformation: the staging
    # change must not bypass masks, current-image radius, gather or scaling.
    output_kwargs = dict(
        image_shape=(48, 48), projector_output_size=32, mask_current_image_disk=mask_disk, current_image_mask_size=28
    )
    if compact:
        expected = p._texture_centered_crop_at_indices(raw, indices, **output_kwargs)
    else:
        expected = p._texture_centered_crop_to_full(raw, **output_kwargs)
    monkeypatch.delenv("RECOVAR_DENSE_MEANS_SCALE", raising=False)
    if dense_scale:
        expected = expected * -(48**2)
    projected, abs2 = p.compute_relion_projector_projections_block(
        slab,
        rotations,
        (48, 48),
        r_max=16,
        padding_factor=padding_factor,
        centered_rows=True,
        dense_scale=dense_scale,
        return_abs2=True,
        projector_output_size=32,
        pixel_indices=indices,
        relion_texture_interp=True,
        mask_current_image_disk=mask_disk,
        current_image_mask_size=28,
    )
    np.testing.assert_array_equal(projected, expected)
    np.testing.assert_array_equal(abs2, jnp.abs(expected) ** 2)
    assert calls == [padding_factor]


@pytest.mark.parametrize(
    "rotation_dtype,output_size,padding_factor",
    [
        (jnp.float64, 32, 2),
        (jnp.float32, 28, 2),
        (jnp.float32, 32, 3),
    ],
)
def test_nonqualified_geometry_keeps_existing_staging(
    monkeypatch,
    rotation_dtype,
    output_size,
    padding_factor,
):
    size = 32 * padding_factor + 3
    slab = jnp.zeros((size, size, size // 2 + 1), dtype=jnp.complex64)
    rows = jnp.eye(3, dtype=rotation_dtype)[None]
    calls = []

    def full_stage(value):
        assert value is slab
        calls.append("full")
        return jnp.zeros((size, size, size), dtype=jnp.complex64)

    monkeypatch.setattr(p, "relion_projector_half_to_texture_full", full_stage)
    monkeypatch.setattr(cb, "project_relion_half_capacity", lambda *a, **kw: pytest.fail("unqualified route"))
    monkeypatch.setattr(
        p,
        "project_half_spectrum",
        lambda *a, **kw: jnp.zeros((1, output_size * (output_size // 2 + 1)), dtype=jnp.complex64),
    )
    p._project_relion_projector_texture(
        slab, rows, (32, 32), r_max=16, projector_output_size=output_size, padding_factor=padding_factor
    )
    assert calls == ["full"]


@pytest.mark.gpu
@pytest.mark.parametrize("padding_factor", [1, 2])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("mask_disk", [False, True])
def test_gpu_half_staging_matches_previous_full_staging(padding_factor, compact, mask_disk):
    import jax

    assert jax.default_backend() == "gpu"
    rng = np.random.default_rng(913)
    size = 32 * padding_factor + 3
    shape = (size, size, size // 2 + 1)
    slab = jnp.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64))
    angles = np.array([0.0, 0.137, -0.219, np.pi / 2], dtype=np.float32)
    rotations = np.stack(
        [
            np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            for a in angles
        ]
    )
    rotations = jnp.asarray(rotations)
    full = p.relion_projector_half_to_texture_full(slab)
    old_crop = p.project_half_spectrum(
        full.reshape(-1), rotations, (32, 32), full.shape, "linear_interp", max_r=16.0, relion_texture_interp=True
    )
    indices = jnp.asarray([40 * 25, 9 * 25 + 1, 24 * 25, 39 * 25 + 16], jnp.int32) if compact else None
    kwargs = dict(
        image_shape=(48, 48), projector_output_size=32, mask_current_image_disk=mask_disk, current_image_mask_size=28
    )
    if compact:
        expected = p._texture_centered_crop_at_indices(old_crop, indices, **kwargs)
    else:
        expected = p._texture_centered_crop_to_full(old_crop, **kwargs)
    got = p._project_relion_projector_texture(
        slab,
        rotations,
        (48, 48),
        r_max=16,
        padding_factor=padding_factor,
        projector_output_size=32,
        pixel_indices=indices,
        mask_current_image_disk=mask_disk,
        current_image_mask_size=28,
    )
    np.testing.assert_array_equal(np.asarray(got).view(np.uint32), np.asarray(expected).view(np.uint32))
