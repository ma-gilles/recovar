"""Focused tests for the guarded native VDAM weighted-sum boundary."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def test_dual_weighted_sums_cuda_source_has_one_ordered_translation_loop():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index("dual_weighted_fma")
    end = source.index("ffi::Error DualWeightedSumsF32Impl", start)
    kernel = source[start:end]
    assert "translation = 0; translation < translation_count; ++translation" in kernel
    assert "if (weight == 0.0f) continue;" in kernel
    assert "return __fmaf_rn(weight, value, accumulator)" in kernel
    assert "sum_real = dual_weighted_fma(weight, value.x, sum_real)" in kernel
    assert "sum_imag = dual_weighted_fma(weight, value.y, sum_imag)" in kernel
    assert "__fma_rn(static_cast<double>(weight), value, accumulator)" in kernel


@pytest.mark.gpu
@pytest.mark.parametrize("value_dtype", [np.complex64, np.complex128])
def test_dual_weighted_sums_matches_sparse_jax_matmul(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    value_dtype,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    rng = np.random.default_rng(17)
    # Representative compact K=4 bucket axes: 116 fine translations and a
    # current-size reconstruction/noise window near 600 half-spectrum pixels.
    probabilities = np.zeros((5, 128, 116), dtype=np.float32)
    for batch in range(probabilities.shape[0]):
        for rotation in range(probabilities.shape[1]):
            active = rng.choice(probabilities.shape[2], size=8, replace=False)
            probabilities[batch, rotation, active] = rng.random(8, dtype=np.float32)
    probabilities /= np.sum(probabilities, axis=(1, 2), keepdims=True, dtype=np.float32)
    value_real_dtype = np.float32 if value_dtype == np.complex64 else np.float64
    first_values = (
        rng.standard_normal((5, 116, 565)).astype(value_real_dtype)
        + 1j * rng.standard_normal((5, 116, 565)).astype(value_real_dtype)
    ).astype(value_dtype)
    second_values = (
        rng.standard_normal((5, 116, 596)).astype(value_real_dtype)
        + 1j * rng.standard_normal((5, 116, 596)).astype(value_real_dtype)
    ).astype(value_dtype)

    with jax.default_device(gpu_device):
        probabilities_jax = jnp.asarray(probabilities)
        first_jax = jnp.asarray(first_values)
        second_jax = jnp.asarray(second_values)
        expected = (
            jnp.matmul(probabilities_jax, first_jax, precision=jax.lax.Precision.HIGHEST),
            jnp.matmul(probabilities_jax, second_jax, precision=jax.lax.Precision.HIGHEST),
        )
        actual = cuda_backproject.dual_weighted_sums_f32(
            probabilities_jax,
            first_jax,
            second_jax,
        )

    expected_np = tuple(np.asarray(value) for value in expected)
    actual_np = tuple(np.asarray(value) for value in actual)
    for native, reference in zip(actual_np, expected_np):
        np.testing.assert_array_equal(native.view(np.uint32), reference.view(np.uint32))


@pytest.mark.gpu
def test_compact_weighted_sums_and_noise_wrapper_matches_composed_boundaries(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    rng = np.random.default_rng(41)
    batch, rotation_count, translation_count = 3, 32, 11
    pair_count, recon_pixels = 96, 43
    noise_pixels = recon_pixels
    pair_probabilities = np.zeros((batch, pair_count), dtype=np.float32)
    rotation_rows = np.full((batch, pair_count), -1, dtype=np.int32)
    translation_indices = np.full((batch, pair_count), -1, dtype=np.int32)
    pair_mask = np.zeros((batch, pair_count), dtype=bool)
    for batch_index in range(batch):
        flat = np.sort(
            rng.choice(
                rotation_count * translation_count,
                size=pair_count - 7,
                replace=False,
            )
        )
        valid_count = flat.size
        rotation_rows[batch_index, :valid_count] = flat // translation_count
        translation_indices[batch_index, :valid_count] = flat % translation_count
        pair_probabilities[batch_index, :valid_count] = rng.random(
            valid_count,
            dtype=np.float32,
        )
        pair_probabilities[batch_index] /= np.sum(
            pair_probabilities[batch_index],
            dtype=np.float32,
        )
        pair_mask[batch_index, :valid_count] = True

    def _complex_values(shape):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ).astype(np.complex128)

    shifted_recon = _complex_values((batch, translation_count, recon_pixels))
    shifted_noise = _complex_values((batch, translation_count, noise_pixels))
    ctf2_over_noise = rng.random((batch, recon_pixels)).astype(np.float64)
    proj_for_noise = _complex_values((batch, rotation_count, noise_pixels))
    proj_abs2_for_noise = np.abs(proj_for_noise).astype(np.float32) ** 2
    noise_variance = (0.5 + rng.random(noise_pixels)).astype(np.float64)
    shell_count = 9
    shell_indices = (np.arange(noise_pixels) % shell_count).astype(np.int32)

    with jax.default_device(gpu_device):
        args = (
            jnp.asarray(pair_probabilities),
            jnp.asarray(rotation_rows),
            jnp.asarray(translation_indices),
            jnp.asarray(pair_mask),
            jnp.asarray(shifted_recon),
            jnp.asarray(shifted_noise),
            jnp.asarray(ctf2_over_noise),
        )
        expected_sums = (
            sparse_pass2_bucketed._compact_pair_weighted_rotation_and_image_sums_native(
                *args,
                n_rotation_rows=rotation_count,
            )
        )
        flat_image_indices = jnp.broadcast_to(
            jnp.arange(batch, dtype=jnp.int32)[:, None],
            (batch, rotation_count),
        ).reshape(-1)
        expected_noise = sparse_pass2_bucketed._compute_noise_block_and_norm_residual_from_flat_rows_residual_terms(
            jnp.asarray(proj_for_noise).reshape((-1, noise_pixels)),
            jnp.asarray(proj_abs2_for_noise).reshape((-1, noise_pixels)),
            expected_sums[1].reshape((-1, noise_pixels)),
            expected_sums[2].reshape((-1, recon_pixels)),
            jnp.asarray(noise_variance),
            jnp.asarray(shell_indices),
            flat_image_indices,
            shell_count=shell_count,
            batch_size=batch,
        )
        actual = sparse_pass2_bucketed._compact_pair_weighted_sums_and_noise_native(
            *args,
            jnp.asarray(proj_for_noise),
            jnp.asarray(proj_abs2_for_noise),
            jnp.asarray(noise_variance),
            jnp.asarray(shell_indices),
            n_rotation_rows=rotation_count,
            shell_count=shell_count,
            batch_size=batch,
        )

    expected = (*expected_sums, *expected_noise)
    expected_np = tuple(np.asarray(value) for value in expected)
    actual_np = tuple(np.asarray(value) for value in actual)
    for fused, composed in zip(actual_np, expected_np):
        np.testing.assert_array_equal(fused, composed)
