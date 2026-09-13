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

    positive_start = source.index("cudaError_t relion_cub_positive_sort_scan_f32")
    positive_end = source.index(
        "ffi::Error RelionCubPositiveSortScanF32Impl",
        positive_start,
    )
    positive_sort_scan = source[positive_start:positive_end]
    assert "cub::DeviceSelect::If(" in positive_sort_scan
    assert "return value > 0.0f;" in source
    assert "const int output_offset = count - selected_count;" in positive_sort_scan
    assert "sorted + output_offset" in positive_sort_scan
    assert "cumulative + output_offset" in positive_sort_scan

    batched_exponentiate_start = source.index(
        "relion_exponentiate_batched_f32_kernel"
    )
    batched_exponentiate_end = source.index(
        "ffi::Error RelionExponentiateBatchedF32Impl",
        batched_exponentiate_start,
    )
    batched_exponentiate = source[
        batched_exponentiate_start:batched_exponentiate_end
    ]
    assert "values[index] + add[index / row_size]" in batched_exponentiate
    assert "exponent < -88.0f ? 0.0f : expf(exponent)" in batched_exponentiate

    batched_divide_start = source.index("relion_divide_batched_f32_kernel")
    batched_divide_end = source.index(
        "ffi::Error RelionDivideBatchedF32Impl",
        batched_divide_start,
    )
    batched_divide = source[batched_divide_start:batched_divide_end]
    assert "values[index] / divisor[index / row_size]" in batched_divide

    batched_sort_start = source.index("ffi::Error RelionCubSortScanBatchedF32Impl")
    batched_sort_end = source.index("struct RelionPositiveF32", batched_sort_start)
    batched_sort = source[batched_sort_start:batched_sort_end]
    assert "for (int64_t row = 0; row < row_count" in batched_sort
    assert "cudaMallocAsync(&temporary, temporary_bytes, stream)" in batched_sort
    assert "cudaFreeAsync(temporary, stream)" in batched_sort
    assert "cudaStreamSynchronize" not in batched_sort


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
def test_batched_relion_f32_posterior_primitives_are_wordwise_row_exact(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    values = np.asarray(
        [
            [-53.920059, -54.920059, -55.920059, -200.0, -np.inf, -53.920059],
            [-1.0, -1.0, -1.0000001, -2.0, -88.0, -89.0],
            [0.0, -0.0, 1.0, -3.5, -87.99999, -88.00001],
        ],
        dtype=np.float32,
    )
    add = np.asarray([53.920059, 2.0, -1.0], dtype=np.float32)

    with jax.default_device(gpu_device):
        values_jax = jnp.asarray(values)
        add_jax = jnp.asarray(add)
        scalar_raw = jax.vmap(cuda_backproject.relion_exponentiate_f32)(
            values_jax,
            add_jax,
        )
        scalar_sorted, scalar_cumulative = jax.vmap(
            cuda_backproject.relion_cub_sort_scan_f32
        )(scalar_raw)
        scalar_normalized = jax.vmap(cuda_backproject.relion_divide_f32)(
            scalar_raw,
            scalar_cumulative[:, -1],
        )

        batched_raw = cuda_backproject.relion_exponentiate_batched_f32(
            values_jax,
            add_jax,
        )
        batched_sorted, batched_cumulative = (
            cuda_backproject.relion_cub_sort_scan_batched_f32(batched_raw)
        )
        batched_normalized = cuda_backproject.relion_divide_batched_f32(
            batched_raw,
            batched_cumulative[:, -1],
        )

    for scalar, batched in (
        (scalar_raw, batched_raw),
        (scalar_sorted, batched_sorted),
        (scalar_cumulative, batched_cumulative),
        (scalar_normalized, batched_normalized),
    ):
        np.testing.assert_array_equal(
            np.asarray(batched).view(np.uint32),
            np.asarray(scalar).view(np.uint32),
        )


@pytest.mark.gpu
def test_relion_positive_cub_sort_scan_matches_native_sized_input_and_right_aligns(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    values = np.asarray(
        [0.0, 3.25, -0.0, 0.5, -1.0, 1.75, 0.25],
        dtype=np.float32,
    )
    positive = values[values > np.float32(0.0)]
    prefix_size = values.size - positive.size
    with jax.default_device(gpu_device):
        sorted_full, cumulative_full = (
            cuda_backproject.relion_cub_positive_sort_scan_f32(
                jnp.asarray(values),
            )
        )
        sorted_native, cumulative_native = cuda_backproject.relion_cub_sort_scan_f32(
            jnp.asarray(positive),
        )

    sorted_full = np.asarray(sorted_full)
    cumulative_full = np.asarray(cumulative_full)
    sorted_native = np.asarray(sorted_native)
    cumulative_native = np.asarray(cumulative_native)
    np.testing.assert_array_equal(sorted_full[:prefix_size], np.float32(0.0))
    np.testing.assert_array_equal(cumulative_full[:prefix_size], np.float32(0.0))
    np.testing.assert_array_equal(
        sorted_full[prefix_size:].view(np.uint32),
        sorted_native.view(np.uint32),
    )
    np.testing.assert_array_equal(
        cumulative_full[prefix_size:].view(np.uint32),
        cumulative_native.view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_positive_cub_sort_scan_zero_mass_is_all_zero(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        sorted_weights, cumulative = (
            cuda_backproject.relion_cub_positive_sort_scan_f32(
                jnp.asarray([0.0, -0.0, 0.0], dtype=jnp.float32),
            )
        )
    np.testing.assert_array_equal(np.asarray(sorted_weights), np.float32(0.0))
    np.testing.assert_array_equal(np.asarray(cumulative), np.float32(0.0))


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
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_cub_positive_sort_scan_f32.__wrapped__(values)

    matrix = values[None, :]
    vector = scalar[None]
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_exponentiate_batched_f32.__wrapped__(matrix, vector)
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_divide_batched_f32.__wrapped__(matrix, vector)
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_cub_sort_scan_batched_f32.__wrapped__(matrix)


def test_relion_batched_posterior_primitive_selector_is_explicit(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.delenv("RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES", raising=False)
    assert not cuda_backproject.relion_batched_posterior_primitives_requested()
    monkeypatch.setenv("RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES", "1")
    assert cuda_backproject.relion_batched_posterior_primitives_requested()
