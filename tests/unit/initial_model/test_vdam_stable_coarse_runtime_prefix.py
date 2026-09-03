"""GPU contract for stable-capacity coarse scoring over a logical prefix."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_posterior,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _plan_coarse_gaussian_square_layout,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def test_stable_coarse_runtime_prefix_preserves_scores_and_support_wordwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Physical padding must not execute or perturb logical RELION pixels."""

    from recovar import cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM", "32")
    cuda_backproject._cuda_ok = None

    image_shape = (128, 128)
    logical_size = 86
    logical_indices, logical_count = make_fourier_window_indices_np(
        image_shape,
        logical_size,
        square=True,
        include_dc=True,
    )
    logical_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=False,
    )
    stable_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=True,
    )
    assert stable_layout.physical_square_count > logical_count

    rng = np.random.default_rng(20260903)
    batch_size, rotation_count, translation_count = 1, 16, 29
    physical_count = stable_layout.physical_square_count
    reference = (
        rng.normal(0.0, 0.02, (rotation_count, physical_count))
        + 1j * rng.normal(0.0, 0.02, (rotation_count, physical_count))
    ).astype(np.complex64)
    shifted = (
        rng.normal(
            0.0,
            0.02,
            (batch_size, translation_count, physical_count),
        )
        + 1j
        * rng.normal(
            0.0,
            0.02,
            (batch_size, translation_count, physical_count),
        )
    ).astype(np.complex64)
    weight = rng.uniform(0.0, 150_000.0, (batch_size, physical_count)).astype(
        np.float32
    )
    initial_diff2 = rng.uniform(10_000.0, 20_000.0, batch_size).astype(
        np.float32
    )

    with jax.default_device(gpu_device):
        direct = cuda_backproject.relion_coarse_diff2_rectangular_f32(
            jnp.asarray(reference[:, :logical_count]),
            jnp.asarray(shifted[:, :, :logical_count]),
            jnp.asarray(weight[:, :logical_count]),
            jnp.asarray(initial_diff2),
            jnp.asarray(logical_layout.full_to_compact_np),
        )
        stable = cuda_backproject.relion_coarse_diff2_rectangular_runtime_f32(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(stable_layout.full_to_compact_np),
            jnp.asarray(logical_count, dtype=jnp.int32),
        )
        direct_posterior = relion_cuda_f32_coarse_posterior(
            -direct.reshape(batch_size, -1),
            adaptive_fraction=0.999,
            max_significants=500,
        )
        stable_posterior = relion_cuda_f32_coarse_posterior(
            -stable.reshape(batch_size, -1),
            adaptive_fraction=0.999,
            max_significants=500,
        )

    np.testing.assert_array_equal(np.asarray(stable), np.asarray(direct))
    for stable_value, direct_value in zip(stable_posterior, direct_posterior):
        np.testing.assert_array_equal(
            np.asarray(stable_value),
            np.asarray(direct_value),
        )


def test_stable_coarse_runtime_prefix_preserves_source16_scores_wordwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Selected source-16 rescoring obeys the same logical-prefix contract."""

    from recovar import cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM", "32")
    cuda_backproject._cuda_ok = None

    image_shape = (128, 128)
    logical_size = 86
    logical_indices, logical_count = make_fourier_window_indices_np(
        image_shape,
        logical_size,
        square=True,
        include_dc=True,
    )
    logical_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=False,
    )
    stable_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=True,
    )
    rng = np.random.default_rng(20260904)
    physical_count = stable_layout.physical_square_count
    reference = (
        rng.normal(0.0, 0.02, (16, physical_count))
        + 1j * rng.normal(0.0, 0.02, (16, physical_count))
    ).astype(np.complex64)
    shifted = (
        rng.normal(0.0, 0.02, (1, 29, physical_count))
        + 1j * rng.normal(0.0, 0.02, (1, 29, physical_count))
    ).astype(np.complex64)
    weight = rng.uniform(0.0, 150_000.0, (1, physical_count)).astype(np.float32)
    initial_diff2 = rng.uniform(10_000.0, 20_000.0, 1).astype(np.float32)
    block_ids = np.zeros((1, 1), dtype=np.int32)

    with jax.default_device(gpu_device):
        direct = cuda_backproject.relion_coarse_diff2_rotation_blocks_f32(
            jnp.asarray(reference[:, :logical_count]),
            jnp.asarray(shifted[:, :, :logical_count]),
            jnp.asarray(weight[:, :logical_count]),
            jnp.asarray(initial_diff2),
            jnp.asarray(block_ids),
            jnp.asarray(logical_layout.full_to_compact_np),
        )
        stable = cuda_backproject.relion_coarse_diff2_rotation_blocks_runtime_f32(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(block_ids),
            jnp.asarray(stable_layout.full_to_compact_np),
            jnp.asarray(logical_count, dtype=jnp.int32),
        )

    np.testing.assert_array_equal(np.asarray(stable), np.asarray(direct))
