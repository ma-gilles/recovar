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
