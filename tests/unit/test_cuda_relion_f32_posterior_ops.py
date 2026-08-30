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
    assert "const float exponent = values[index] + add[index / row_size];" in exponentiate
    assert "exponent < -88.0f ? 0.0f : expf(exponent)" in exponentiate

    divide_start = source.index("relion_divide_f32_kernel")
    divide_end = source.index("ffi::Error RelionDivideF32Impl", divide_start)
    divide = source[divide_start:divide_end]
    assert "output[index] = values[index] / divisor[index / row_size];" in divide

    assert "typename cub::DeviceScanPolicy<float, cub::Sum>::Policy800" in source
    assert "#if CUB_VERSION < 300000" in source
    assert "for (int64_t row = 0; row < row_count && err == cudaSuccess; ++row)" in source


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


@pytest.mark.gpu
@pytest.mark.parametrize("row_count,row_size", [(1, 1), (3, 17), (5, 257)])
def test_relion_f32_posterior_batched_cuda_is_bitwise_sequential(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    row_count,
    row_size,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    rng = np.random.default_rng(1701 + row_count * 1000 + row_size)
    values = rng.uniform(-150.0, -45.0, size=(row_count, row_size)).astype(np.float32)
    add = rng.uniform(48.0, 56.0, size=row_count).astype(np.float32)
    values[:, 0] = np.float32(-np.inf)
    with jax.default_device(gpu_device):
        values_jax = jnp.asarray(values)
        add_jax = jnp.asarray(add)
        batched_raw = cuda_backproject.relion_exponentiate_f32(values_jax, add_jax)
        batched_sorted, batched_cumulative = (
            cuda_backproject.relion_cub_sort_scan_f32(batched_raw)
        )
        batched_normalized = cuda_backproject.relion_divide_f32(
            batched_raw,
            batched_cumulative[:, -1],
        )

        sequential_raw = jnp.stack(
            [
                cuda_backproject.relion_exponentiate_f32(values_jax[row], add_jax[row])
                for row in range(row_count)
            ]
        )
        sequential_pairs = [
            cuda_backproject.relion_cub_sort_scan_f32(sequential_raw[row])
            for row in range(row_count)
        ]
        sequential_sorted = jnp.stack([pair[0] for pair in sequential_pairs])
        sequential_cumulative = jnp.stack([pair[1] for pair in sequential_pairs])
        sequential_normalized = jnp.stack(
            [
                cuda_backproject.relion_divide_f32(
                    sequential_raw[row], sequential_cumulative[row, -1]
                )
                for row in range(row_count)
            ]
        )

    for batched, sequential in (
        (batched_raw, sequential_raw),
        (batched_sorted, sequential_sorted),
        (batched_cumulative, sequential_cumulative),
        (batched_normalized, sequential_normalized),
    ):
        np.testing.assert_array_equal(
            np.asarray(batched).view(np.uint32),
            np.asarray(sequential).view(np.uint32),
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


def test_relion_f32_posterior_cuda_primitives_validate_batched_shapes():
    import recovar.cuda_backproject as cuda_backproject

    values = jnp.ones((3, 5), dtype=jnp.float32)
    scalar = jnp.asarray(1.0, dtype=jnp.float32)

    with pytest.raises(TypeError, match="one value per row"):
        cuda_backproject.relion_exponentiate_f32.__wrapped__(values, scalar)
    with pytest.raises(TypeError, match="one value per row"):
        cuda_backproject.relion_divide_f32.__wrapped__(values, scalar)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        cuda_backproject.relion_cub_sort_scan_f32.__wrapped__(
            jnp.ones((2, 3, 5), dtype=jnp.float32)
        )
