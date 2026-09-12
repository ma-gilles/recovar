"""RELION local pass-2 projector crop must be the particle-image window.

RELION remaps the particle-image ``current_size`` per optics group while the
Projector keeps the model sphere (``mymodel.current_size``).  When the two
differ (EMPIAR-10073: image window 202 for model size 200), the local exact
pass-2 path used to size its projector crop from ``2 * max_r`` (the model
size) while gathering projections at indices from the larger image window.
Indices outside the crop then alias onto other projection pixels: the
``kx = crop // 2 + 1`` column reads the next row's ``kx = 0`` value.

These tests pin the spec-level fix and reproduce the aliasing numerically on
the CPU JAX projector fallback.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.em.helpers import fourier_window, projection  # noqa: E402

IMAGE = 64
MODEL_CURRENT = 32          # model / reconstruction current size
IMAGE_CURRENT = 34          # remapped particle-image window (optics pixel size > model)


def _spec():
    n_half = IMAGE * (IMAGE // 2 + 1)
    return fourier_window.make_fourier_window_spec(
        (IMAGE, IMAGE),
        IMAGE_CURRENT,
        n_half,
        reconstruction_current_size=MODEL_CURRENT,
        square=False,
        include_recon_window=True,
    )


def test_spec_records_particle_image_window_and_projector_crop():
    spec = _spec()
    assert spec.use_window
    assert spec.max_r == MODEL_CURRENT // 2
    assert spec.image_current_size == IMAGE_CURRENT
    assert spec.relion_projector_output_size() == IMAGE_CURRENT
    # The model-size crop cannot hold the image-window indices ...
    with pytest.raises(ValueError, match="exceed projector crop"):
        projection._validate_centered_relion_projector_pixel_indices(
            spec.projection_indices_np,
            image_shape=(IMAGE, IMAGE),
            projector_output_size=2 * int(spec.max_r),
        )
    # ... while the particle-image crop does.
    projection._validate_centered_relion_projector_pixel_indices(
        spec.projection_indices_np,
        image_shape=(IMAGE, IMAGE),
        projector_output_size=spec.relion_projector_output_size(),
    )


def test_unwindowed_spec_has_no_projector_crop():
    n_half = IMAGE * (IMAGE // 2 + 1)
    spec = fourier_window.make_fourier_window_spec((IMAGE, IMAGE), None, n_half)
    assert not spec.use_window
    assert spec.image_current_size is None
    assert spec.relion_projector_output_size() is None


def _synthetic_relion_half_projector(r_max, padding_factor):
    """Small Hermitian-consistent PPref-like half volume with structure."""

    from recovar.em.vdam.dense_adapter import reference_to_relion_projector_half_maps

    rng = np.random.default_rng(7)
    grid = np.indices((IMAGE, IMAGE, IMAGE)) - IMAGE // 2
    real = np.zeros((IMAGE, IMAGE, IMAGE), dtype=np.float64)
    for _ in range(6):
        center = rng.uniform(-10, 10, size=3)
        width = rng.uniform(2.0, 4.0)
        real += np.exp(-np.sum((grid.T - center) ** 2, axis=-1).T / (2 * width**2))
    half, projector_r_max = reference_to_relion_projector_half_maps(
        real[None, ...],
        current_size=2 * r_max,
        padding_factor=padding_factor,
    )
    return np.asarray(half)[0] if np.asarray(half).ndim == 4 else np.asarray(half), int(projector_r_max)


def test_gather_at_image_window_indices_matches_full_projection_and_model_crop_aliases():
    spec = _spec()
    r_max = MODEL_CURRENT // 2
    try:
        half, projector_r_max = _synthetic_relion_half_projector(r_max, padding_factor=2)
    except Exception as exc:  # pragma: no cover - environment without the projector builder
        pytest.skip(f"RELION projector builder unavailable on this backend: {exc}")
    rotations = jnp.asarray(
        np.stack([np.eye(3), _rotation_z(np.deg2rad(37.0)), _rotation_x(np.deg2rad(61.0))]),
        dtype=jnp.float32,
    )
    full = projection.project_relion_projector_half_spectrum_centered_rows(
        jnp.asarray(half),
        rotations,
        (IMAGE, IMAGE),
        projector_r_max,
        2,
        None,
    )
    full = np.asarray(full).reshape((rotations.shape[0], -1))
    expected = full[:, spec.projection_indices_np]

    gathered_image_crop = np.asarray(
        projection.project_relion_projector_half_spectrum_centered_rows_at_indices(
            jnp.asarray(half),
            rotations,
            (IMAGE, IMAGE),
            projector_r_max,
            2,
            spec.relion_projector_output_size(),
            spec.projection_indices_np,
        )
    )
    np.testing.assert_allclose(gathered_image_crop, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max())

    gathered_model_crop = np.asarray(
        projection.project_relion_projector_half_spectrum_centered_rows_at_indices(
            jnp.asarray(half),
            rotations,
            (IMAGE, IMAGE),
            projector_r_max,
            2,
            2 * r_max,
            spec.projection_indices_np,
        )
    )
    # The model-size crop aliases indices outside it: the gather disagrees.
    mismatch = np.abs(gathered_model_crop - expected) > 1e-6 * np.abs(expected).max()
    assert mismatch.any(), "model-size crop unexpectedly reproduced the image-window gather"
    # and the disagreement is confined to pixels outside the model crop
    rows = spec.projection_indices_np // (IMAGE // 2 + 1)
    cols = spec.projection_indices_np - rows * (IMAGE // 2 + 1)
    ky = rows - IMAGE // 2
    outside = (np.abs(ky) > r_max) | (cols > r_max) | (ky == -r_max)
    assert not mismatch[:, ~outside].any()


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_validate_local_relion_projector_window_accepts_image_window_and_rejects_model_window():
    """The host-side local check must reject the pre-fix ``2 * max_r`` crop and accept the image window."""

    from dataclasses import replace

    from recovar.em.local import local_bucket_stages
    spec = _spec()
    assert local_bucket_stages.validate_local_relion_projector_window(spec, (IMAGE, IMAGE)) == IMAGE_CURRENT
    model_window_spec = replace(spec, image_current_size=MODEL_CURRENT)
    assert model_window_spec.relion_projector_output_size() == MODEL_CURRENT
    with pytest.raises(ValueError, match="exceed projector crop"):
        local_bucket_stages.validate_local_relion_projector_window(model_window_spec, (IMAGE, IMAGE))
