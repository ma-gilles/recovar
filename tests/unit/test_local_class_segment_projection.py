"""Class-segmented projection: each class segment is projected from its own volume."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.local.local_big_jit import (
    _class_volume,
    _project_local_class_segments,
    _project_local_half_spectrum,
)

pytestmark = pytest.mark.unit


def _rotations(rng, batch_size, rows):
    q = rng.standard_normal((batch_size, rows, 3, 3))
    # Orthonormalize so the rotations are valid for the projector.
    u, _, vt = np.linalg.svd(q)
    r = u @ vt
    flip = np.linalg.det(r) < 0
    r[flip, :, 0] *= -1.0
    return jnp.asarray(r, dtype=jnp.float32)


def _volumes(rng, n_classes, volume_shape):
    size = int(np.prod(volume_shape))
    return jnp.asarray(
        rng.standard_normal((n_classes, size)) + 1j * rng.standard_normal((n_classes, size)),
        dtype=jnp.complex64,
    )


def _project(volume, rotations, image_shape, volume_shape):
    return _project_local_half_spectrum(
        volume,
        None,
        rotations,
        None,
        image_shape,
        volume_shape,
        "linear_interp",
        projection_half_volume=False,
        projection_max_r=None,
        relion_projector_output_size=0,
        projection_relion_texture_interp=False,
        projection_force_jax=True,
        use_relion_projector=False,
        relion_projector_r_max=0,
        projection_padding_factor=1,
    )


@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_class_segments_match_projecting_each_class_separately(n_classes):
    rng = np.random.default_rng(19)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    batch_size, seg = 3, 4
    rows = n_classes * seg
    local_rotations = _rotations(rng, batch_size, rows)
    volumes = _volumes(rng, n_classes, volume_shape)

    segmented = _project_local_class_segments(
        volumes,
        None,
        local_rotations,
        lambda volume, projector_half, rotations: _project(volume, rotations, image_shape, volume_shape),
        n_classes=n_classes,
        segment_rotation_count=seg,
    )
    assert segmented.shape[0] == batch_size * rows

    per_row = np.asarray(segmented).reshape(batch_size, rows, -1)
    for k in range(n_classes):
        expected = _project(
            volumes[k],
            local_rotations[:, k * seg : (k + 1) * seg].reshape(batch_size * seg, 3, 3),
            image_shape,
            volume_shape,
        )
        np.testing.assert_array_equal(
            per_row[:, k * seg : (k + 1) * seg].reshape(batch_size * seg, -1),
            np.asarray(expected),
        )


def test_one_class_reproduces_the_unsegmented_projection_exactly():
    rng = np.random.default_rng(23)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    batch_size, rows = 4, 6
    local_rotations = _rotations(rng, batch_size, rows)
    volume = _volumes(rng, 1, volume_shape)

    segmented = _project_local_class_segments(
        volume,
        None,
        local_rotations,
        lambda v, _p, r: _project(v, r, image_shape, volume_shape),
        n_classes=1,
        segment_rotation_count=rows,
    )
    flat = _project(volume[0], local_rotations.reshape(batch_size * rows, 3, 3), image_shape, volume_shape)
    np.testing.assert_array_equal(np.asarray(segmented), np.asarray(flat))


def test_segment_placement_is_class_major_and_row_preserving():
    """A projector that reports (class marker, rotation trace) pins where each row lands."""
    rng = np.random.default_rng(29)
    batch_size, seg, n_classes = 2, 3, 4
    rows = n_classes * seg
    local_rotations = _rotations(rng, batch_size, rows)
    markers = jnp.asarray(np.arange(n_classes, dtype=np.float32)[:, None] + 100.0)

    def fake_project(volume, _projector_half, rotations):
        traces = jnp.trace(rotations, axis1=1, axis2=2)[:, None]
        return jnp.concatenate([jnp.broadcast_to(volume, (rotations.shape[0], 1)), traces], axis=1)

    out = np.asarray(
        _project_local_class_segments(
            markers, None, local_rotations, fake_project,
            n_classes=n_classes, segment_rotation_count=seg,
        )
    ).reshape(batch_size, rows, 2)
    expected_traces = np.asarray(jnp.trace(local_rotations, axis1=2, axis2=3))
    for k in range(n_classes):
        np.testing.assert_array_equal(out[:, k * seg : (k + 1) * seg, 0], np.full((batch_size, seg), 100.0 + k))
    np.testing.assert_allclose(out[:, :, 1], expected_traces, rtol=0, atol=0)


def test_class_volume_rejects_a_mismatched_stack():
    volumes = jnp.zeros((2, 5))
    assert _class_volume(None, 0, 2) is None
    np.testing.assert_array_equal(np.asarray(_class_volume(volumes, 1, 2)), np.zeros(5))
    with pytest.raises(ValueError, match="leading class axis"):
        _class_volume(volumes, 0, 3)


def test_segmented_rows_must_factor_into_classes():
    rng = np.random.default_rng(31)
    local_rotations = _rotations(rng, 2, 7)
    with pytest.raises(ValueError, match="class-segmented rows must be"):
        _project_local_class_segments(
            jnp.zeros((2, 4)), None, local_rotations, lambda v, p, r: r.reshape(r.shape[0], -1),
            n_classes=2, segment_rotation_count=3,
        )
