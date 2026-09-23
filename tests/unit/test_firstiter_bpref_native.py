"""Native firstiter contracts retained from Q 72e785f3f."""
from pathlib import Path
import inspect
import numpy as np
import pytest
import jax.numpy as jnp
import recovar.cuda_backproject as cuda_backproject
from recovar.em.cuda import kernels as em_cuda_kernels
pytestmark = pytest.mark.unit

def test_relion_firstiter_bpref_cuda_source_preserves_native_interface_and_pass_loop():
    cuda_source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    )
    text = cuda_source.read_text()
    kernel_start = text.index("relion_firstiter_bpref_fused_x_half_kernel(")
    declaration_start = text.rfind("__global__", 0, kernel_start)
    kernel_end = text.index("/* ================================================================== */", kernel_start)
    kernel = text[declaration_start:kernel_end]

    assert "__launch_bounds__" not in kernel[: kernel.index("{")]
    assert "const int pixel_pass_num = (int)ceilf((float)img_xyz / 128.0f);" in kernel
    assert "for (unsigned pass = 0; pass < (unsigned)pixel_pass_num; ++pass)" in kernel
    assert "const unsigned pixel = pass * 128U + tid;" in kernel
    assert "pixel += 128" not in kernel
    assert "const float* __restrict__ image_real" in kernel
    assert "const float* __restrict__ image_imag" in kernel
    assert "const float* __restrict__ translation_x" in kernel
    assert "const float* __restrict__ translation_y" in kernel
    assert "float* __restrict__ model_real" in kernel
    assert "float* __restrict__ model_imag" in kernel
    assert "unsigned long translation_num" in kernel

    handler_start = text.index("ffi::Error RelionFirstiterBprefFusedXHalfImpl(")
    handler_end = text.index("ffi::Error RelionFusedXHalfBackprojectSignatureImpl(")
    handler = text[handler_start:handler_end]
    assert "float significant_weight" in handler
    assert "float weight_norm" in handler
    assert "cudaMemcpy" not in handler


def test_relion_firstiter_bpref_wrapper_uses_split_native_operands_and_static_scalars(
    monkeypatch,
):
    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            return args[8], args[9], args[10]

        return call

    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    data_volume = jnp.zeros(volume_size, dtype=jnp.complex64)
    weight_volume = jnp.zeros(volume_size, dtype=jnp.float32)
    image = jnp.arange(12, dtype=jnp.float32).astype(jnp.complex64) * (1.0 + 2.0j)
    ctf = jnp.ones(12, dtype=jnp.float32)
    minvsigma2 = jnp.full(12, 2.0, dtype=jnp.float32)
    posterior = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    translation_angles = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    native_eulers = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))

    data_out, weight_out = (
        em_cuda_kernels._relion_firstiter_bpref_fused_x_half_static.__wrapped__(
            data_volume,
            weight_volume,
            image,
            ctf,
            minvsigma2,
            posterior,
            translation_angles,
            native_eulers,
            0.125,
            1.0,
            image_shape,
            volume_shape,
            2.0,
        )
    )

    assert observed["target"] == cuda_backproject._TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF
    assert observed["options"]["input_output_aliases"] == {8: 0, 9: 1, 10: 2}
    assert observed["options"]["vmap_method"] == "sequential"
    assert [item.dtype for item in observed["result_types"]] == [
        jnp.float32,
        jnp.float32,
        jnp.float32,
    ]
    args = observed["args"]
    np.testing.assert_array_equal(np.asarray(args[0]), np.asarray(jnp.real(image)))
    np.testing.assert_array_equal(np.asarray(args[1]), np.asarray(jnp.imag(image)))
    np.testing.assert_array_equal(np.asarray(args[5]), np.asarray(translation_angles[:, 0]))
    np.testing.assert_array_equal(np.asarray(args[6]), np.asarray(translation_angles[:, 1]))
    assert args[8].dtype == jnp.float32 and args[9].dtype == jnp.float32
    assert observed["attrs"]["significant_weight"] == np.float32(0.125)
    assert observed["attrs"]["weight_norm"] == np.float32(1.0)
    assert data_out.dtype == jnp.complex64
    assert weight_out is weight_volume


def test_relion_firstiter_bpref_split_wrapper_preserves_three_ffi_aliases(
    monkeypatch,
):
    split_jit_source = inspect.getsource(
        em_cuda_kernels._relion_firstiter_bpref_fused_x_half_split_static,
    )
    assert "donate_argnums=(0, 1, 2)" in split_jit_source

    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            return args[8], args[9], args[10]

        return call

    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    monkeypatch.setattr(
        cuda_backproject.jax.lax,
        "complex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("split wrapper interleaved its accumulators")
        ),
    )

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    data_real = jnp.arange(volume_size, dtype=jnp.float32)
    data_imag = -data_real
    weight = jnp.full(volume_size, 3.0, dtype=jnp.float32)
    image = jnp.arange(12, dtype=jnp.float32).astype(jnp.complex64) * (1.0 + 2.0j)
    ctf = jnp.ones(12, dtype=jnp.float32)
    minvsigma2 = jnp.full(12, 2.0, dtype=jnp.float32)
    posterior = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    translation_angles = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    native_eulers = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))

    outputs = (
        em_cuda_kernels._relion_firstiter_bpref_fused_x_half_split_static.__wrapped__(
            data_real,
            data_imag,
            weight,
            image,
            ctf,
            minvsigma2,
            posterior,
            translation_angles,
            native_eulers,
            0.125,
            1.0,
            image_shape,
            volume_shape,
            2.0,
        )
    )

    assert observed["target"] == cuda_backproject._TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF
    assert observed["options"]["input_output_aliases"] == {8: 0, 9: 1, 10: 2}
    assert observed["options"]["vmap_method"] == "sequential"
    assert [item.dtype for item in observed["result_types"]] == [
        jnp.float32,
        jnp.float32,
        jnp.float32,
    ]
    assert outputs[0] is data_real
    assert outputs[1] is data_imag
    assert outputs[2] is weight
    assert observed["args"][8] is data_real
    assert observed["args"][9] is data_imag
    assert observed["args"][10] is weight
    assert observed["attrs"]["significant_weight"] == np.float32(0.125)
    assert observed["attrs"]["weight_norm"] == np.float32(1.0)


@pytest.mark.gpu
def test_relion_firstiter_bpref_exact_native_ffi_smoke(
    monkeypatch, custom_cuda_lib, gpu_device
):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    image = np.zeros(12, dtype=np.complex64)
    image[0] = np.complex64(2.0 + 3.0j)
    ctf = np.zeros(12, dtype=np.float32)
    ctf[0] = np.float32(2.0)
    minvsigma2 = np.full(12, np.float32(0.5), dtype=np.float32)

    with cuda_backproject.jax.default_device(gpu_device):
        data_out, weight_out = em_cuda_kernels.relion_firstiter_bpref_fused_x_half(
            jnp.zeros(volume_size, dtype=jnp.complex64),
            jnp.zeros(volume_size, dtype=jnp.float32),
            jnp.asarray(image),
            jnp.asarray(ctf),
            jnp.asarray(minvsigma2),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32)[None],
            0.5,
            1.0,
            image_shape,
            volume_shape,
            2.0,
        )

    data_out = np.asarray(data_out)
    weight_out = np.asarray(weight_out)
    expected_offset = 3 * (7 * 4) + 3 * 4
    expected_data = np.zeros(volume_size, dtype=np.complex64)
    expected_weight = np.zeros(volume_size, dtype=np.float32)
    expected_data[expected_offset] = np.complex64(2.0 + 3.0j)
    expected_weight[expected_offset] = np.float32(2.0)
    np.testing.assert_array_equal(data_out, expected_data)
    np.testing.assert_array_equal(weight_out, expected_weight)


@pytest.mark.gpu
@pytest.mark.parametrize('threshold, expected_scale', [(0.5, 1.0), (2.0, 0.0)])
def test_split_firstiter_repeated_launches_consume_owned_accumulators(
    monkeypatch, custom_cuda_lib, gpu_device, threshold, expected_scale
):
    """Two exact DC contributions exercise aliasing, threshold and replay order."""
    import jax
    monkeypatch.setenv('RECOVAR_CUDA_LIB', str(custom_cuda_lib))
    monkeypatch.delenv('RECOVAR_DISABLE_CUDA', raising=False)
    shape = (7, 7, 7)
    size = 7 * 7 * 4
    offset = 3 * 7 * 4 + 3 * 4
    with jax.default_device(gpu_device):
        accumulators = tuple(jnp.zeros(size, dtype=jnp.float32) for _ in range(3))
        pointers = tuple(x.unsafe_buffer_pointer() for x in accumulators)
        ctf = jnp.zeros(12, dtype=jnp.float32).at[0].set(2)
        noise = jnp.full(12, 0.5, dtype=jnp.float32)
        for value in [2 + 3j, 4 - 1j]:
            image = jnp.zeros(12, dtype=jnp.complex64).at[0].set(value)
            old = accumulators
            accumulators = em_cuda_kernels.relion_firstiter_bpref_fused_x_half_split(
                *old, image, ctf, noise,
                jnp.ones((1, 1), dtype=jnp.float32),
                jnp.zeros((1, 2), dtype=jnp.float32),
                jnp.eye(3, dtype=jnp.float32)[None],
                threshold, 1.0, (4, 4), shape, 2.0,
            )
            jax.block_until_ready(accumulators)
            assert all(x.is_deleted() for x in old)
            assert tuple(x.unsafe_buffer_pointer() for x in accumulators) == pointers
    for actual, value in zip(accumulators, [6.0, 2.0, 4.0]):
        expected = np.zeros(size, dtype=np.float32)
        expected[offset] = value * expected_scale
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_relion_split_symmetry_range_has_bounded_outputs_and_no_aliases(monkeypatch):
    observed = {}

    def fake_ffi_call(target, result_types, **options):
        observed["target"] = target
        observed["result_types"] = result_types
        observed["options"] = options

        def call(*args, **attrs):
            observed["args"] = args
            observed["attrs"] = attrs
            return (
                jnp.zeros(result_types[0].shape, dtype=result_types[0].dtype),
                jnp.zeros(result_types[1].shape, dtype=result_types[1].dtype),
            )

        return call

    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(em_cuda_kernels, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    range_voxels = 17
    outputs = (
        em_cuda_kernels._relion_point_group_symmetrise_bpref_split_range_static.__wrapped__(
            jnp.zeros(volume_size, dtype=jnp.float32),
            jnp.zeros(volume_size, dtype=jnp.float32),
            jnp.zeros(volume_size, dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32)[None],
            jnp.asarray([volume_size - 5], dtype=jnp.int64),
            volume_shape,
            2,
            range_voxels,
        )
    )

    assert (
        observed["target"]
        == cuda_backproject._TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_SPLIT_RANGE
    )
    assert "input_output_aliases" not in observed["options"]
    assert [result.shape for result in observed["result_types"]] == [
        (range_voxels,),
        (range_voxels,),
    ]
    assert [result.dtype for result in observed["result_types"]] == [
        jnp.complex64,
        jnp.float32,
    ]
    assert [value.shape for value in observed["args"][:3]] == [
        (volume_size,),
        (volume_size,),
        (volume_size,),
    ]
    assert observed["args"][4].shape == (1,)
    assert observed["attrs"]["support_radius"] == np.int64(2)
    assert outputs[0].shape == (range_voxels,)
    assert outputs[1].shape == (range_voxels,)
