"""Local projection backends preserve score/noise views and precision boundaries."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.fourier_window import make_fourier_window_spec
from recovar.em.local import local_bucket_stages
from recovar.em.local import local_em_engine as engine

pytestmark = pytest.mark.unit


def projection_case(monkeypatch, backend, current_size, reconstruct, input_dtype, double_score):
    values = np.arange(6 * 40).reshape(6, 40)
    full = (values + 1j * (values + 3)).astype(input_dtype)
    weights = ((np.arange(40) % 3 + 1) * 0.5).astype(np.float32)
    window = make_fourier_window_spec((8, 8), current_size, 40, include_recon_window=True)
    calls = []

    def native(slab, rotations, image_shape, **kwargs):
        calls.append(("relion", tuple(slab.shape), tuple(rotations.shape), kwargs))
        indices = kwargs.get("pixel_indices")
        return jnp.asarray(full if indices is None else full[:, np.asarray(indices)]), None

    def indexed(mean, indices, rotations, image_shape, volume_shape, disc_type, **kwargs):
        calls.append(("indexed", tuple(rotations.shape), np.asarray(indices).tolist(), kwargs))
        return jnp.asarray(full[:, np.asarray(indices)])

    def ordinary(mean, rotations, image_shape, volume_shape, disc_type, **kwargs):
        calls.append(("ordinary", tuple(rotations.shape), kwargs))
        return jnp.asarray(full), None

    monkeypatch.setattr(local_bucket_stages, "_compute_relion_projector_projections_block", native)
    monkeypatch.setattr(local_bucket_stages, "_project_indexed_half_spectrum", indexed)
    monkeypatch.setattr(local_bucket_stages, "_compute_projections_block", ordinary)
    monkeypatch.setattr(local_bucket_stages, "_indexed_projection_available", lambda: backend != "unavailable")
    kwargs = dict(
        mean_for_proj=jnp.zeros((8, 8, 8), dtype=jnp.complex64),
        bucket=SimpleNamespace(
            image_indices=np.array([0, 1]),
            bucket_rotation_count=3,
            local_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3, 3)),
        ),
        image_shape=(8, 8),
        proj_volume_shape=(8, 8, 8),
        disc_type="linear_interp",
        projection_kwargs={
            "force_jax": backend == "jax",
            "relion_texture_interp": backend == "texture",
            "mask_current_image_disk": False,
            "max_r": 3,
        },
        window_spec=window,
        n_half=40,
        half_weights=jnp.asarray(weights),
        precision_policy=DensePrecisionPolicy(use_float64_scoring=double_score),
        relion_projector_half=jnp.ones((1, 8, 8, 5)) if backend == "relion" else None,
        relion_projector_r_max=4,
        materialize_recon_projection=reconstruct,
    )
    return kwargs, full, weights, calls


@pytest.mark.parametrize("backend", ["relion", "indexed", "jax", "texture", "unavailable"])
@pytest.mark.parametrize("current_size", [6, 8])
@pytest.mark.parametrize("reconstruct", [False, True])
@pytest.mark.parametrize("input_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("double_score", [False, True])
def test_local_projection_views(monkeypatch, backend, current_size, reconstruct, input_dtype, double_score):
    kwargs, full, weights, calls = projection_case(
        monkeypatch,
        backend,
        current_size,
        reconstruct,
        input_dtype,
        double_score,
    )
    result = engine._project_local_bucket(**kwargs)
    window = kwargs["window_spec"]
    score_indices = np.arange(40) if not window.use_window else window.score_indices_np
    recon_indices = np.arange(40) if not window.use_window else window.recon_indices_np
    expected_score = (full[:, score_indices] * weights[score_indices]).reshape(2, 3, -1)
    expected_score = expected_score.astype(np.complex128 if double_score else np.complex64)
    assert result.proj_weighted.dtype == expected_score.dtype
    np.testing.assert_array_equal(result.proj_weighted, expected_score)
    if reconstruct:
        expected_noise = full[:, recon_indices].reshape(2, 3, -1)
        if double_score:
            expected_noise = expected_noise.astype(np.complex128)
        assert result.proj_for_noise.dtype == expected_noise.dtype
        np.testing.assert_array_equal(result.proj_for_noise, expected_noise)
    else:
        assert result.proj_for_noise is None
    expected_backend = (
        "relion" if backend == "relion" else ("indexed" if backend == "indexed" and window.use_window else "ordinary")
    )
    assert len(calls) == 1
    assert calls[0][0] == expected_backend
    if expected_backend == "ordinary":
        assert "mask_current_image_disk" not in calls[0][-1]
    elif expected_backend == "relion":
        assert calls[0][-1]["mask_current_image_disk"] is False
