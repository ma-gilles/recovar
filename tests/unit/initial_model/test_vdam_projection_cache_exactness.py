"""Exactness guards for reusing EM projection batching in VDAM."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.helpers.fourier_window import make_fourier_window_spec
from recovar.em.helpers.projection import compute_relion_projector_projections_block

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_relion_projection_rows_are_bitwise_invariant_to_duplicate_batching(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """A cached unique row must equal every directly projected duplicate row."""

    from recovar import cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    cuda_backproject._cuda_ok = None

    image_shape = (32, 32)
    current_size = 20
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        include_recon_window=True,
    )
    rng = np.random.default_rng(20260830)
    projector_half = (
        rng.standard_normal((33, 33, 17))
        + 1j * rng.standard_normal((33, 33, 17))
    ).astype(np.complex64)
    unique_rotations = np.stack(
        [_rotation_z(float(angle)) for angle in np.linspace(0.0, 1.3, 5)],
        axis=0,
    )
    duplicate_ids = np.resize(np.asarray([4, 1, 3, 0, 2, 3, 1], dtype=np.int32), 7048)
    duplicate_rotations = unique_rotations[duplicate_ids]

    common = dict(
        image_shape=image_shape,
        r_max=16,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=window.projection_indices,
        relion_texture_interp=True,
    )
    direct, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        jnp.asarray(duplicate_rotations),
        **common,
    )
    unique, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        jnp.asarray(unique_rotations),
        **common,
    )

    np.testing.assert_array_equal(np.asarray(direct), np.asarray(unique)[duplicate_ids])
