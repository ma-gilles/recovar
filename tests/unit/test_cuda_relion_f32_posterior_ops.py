"""Focused tests for RELION's float32 fine-posterior CUDA primitives."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def test_relion_f32_posterior_cuda_source_pins_deployed_arithmetic():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    exponentiate_start = source.index("relion_exponentiate_f32_kernel")
    exponentiate_end = source.index("ffi::Error RelionExponentiateF32Impl", exponentiate_start)
    exponentiate = source[exponentiate_start:exponentiate_end]
    assert "const float exponent = values[index] + add[0];" in exponentiate
    assert "exponent < -88.0f ? 0.0f : expf(exponent)" in exponentiate

    divide_start = source.index("relion_divide_f32_kernel")
    divide_end = source.index("ffi::Error RelionDivideF32Impl", divide_start)
    divide = source[divide_start:divide_end]
    assert "output[index] = values[index] / divisor[0];" in divide

    assert "typename cub::DeviceScanPolicy<float, cub::Sum>::Policy800" in source
    assert "#if CUB_VERSION < 300000" in source


@pytest.mark.gpu
def test_relion_f32_posterior_cuda_primitives_preserve_float32_chain(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    # The non-round values ensure the test exercises expf, the CUB scan, and
    # CUDA float division rather than only exact powers of two.
    values = np.asarray([-53.920059, -54.920059, -55.920059, -200.0], dtype=np.float32)
    add = np.asarray(53.920059, dtype=np.float32)
    with jax.default_device(gpu_device):
        raw = cuda_backproject.relion_exponentiate_f32(
            jnp.asarray(values),
            jnp.asarray(add),
        )
        sorted_weights, cumulative = cuda_backproject.relion_cub_sort_scan_f32(raw)
        normalized = cuda_backproject.relion_divide_f32(raw, cumulative[-1])

    raw = np.asarray(raw)
    sorted_weights = np.asarray(sorted_weights)
    cumulative = np.asarray(cumulative)
    normalized = np.asarray(normalized)

    assert raw.dtype == np.float32
    assert sorted_weights.dtype == np.float32
    assert cumulative.dtype == np.float32
    assert normalized.dtype == np.float32
    assert raw[-1].view(np.uint32) == 0
    np.testing.assert_array_equal(sorted_weights, np.sort(raw))
    assert cumulative[-1] > np.float32(0.0)
    np.testing.assert_allclose(
        np.sum(normalized, dtype=np.float32),
        np.float32(1.0),
        rtol=2e-7,
        atol=0.0,
    )


def test_relion_f32_posterior_cuda_primitives_fail_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    values = jnp.asarray([0.0, -1.0], dtype=jnp.float32)
    scalar = jnp.asarray(1.0, dtype=jnp.float32)

    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_exponentiate_f32.__wrapped__(values, scalar)
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_divide_f32.__wrapped__(values, scalar)
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_cub_sort_scan_f32.__wrapped__(values)
