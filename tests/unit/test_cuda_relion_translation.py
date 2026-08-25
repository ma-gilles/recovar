"""Focused tests for RELION's accelerated fine-score translation FFI."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def test_relion_translation_angles_match_captured_float32_bits():
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    translations = np.asarray(
        [
            [-0.7461191415786743, -0.7461191415786743],
            [-0.7461191415786743, 0.2538808584213257],
        ],
        dtype=np.float32,
    )
    angles = _relion_translation_angles_f32(translations, (256, 256))

    np.testing.assert_array_equal(
        angles.view(np.uint32),
        np.asarray(
            [
                [1016464419, 1016464419],
                [1016464419, 3150720736],
            ],
            dtype=np.uint32,
        ),
    )


def test_relion_translation_cuda_source_preserves_explicit_arithmetic():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    assert "relion_score_translate_f32" in source
    assert "__fmaf_rn(" in source
    assert "__fmul_rn(static_cast<float>(y), ty)" in source
    assert "const float translated_real = __fmaf_rn(" in source
    assert "cosine, value.x" in source
    assert "__fmul_rn(sine, value.y)" in source
    assert "const float translated_imag = __fmaf_rn(" in source
    assert "__fmul_rn(cosine, value.y)" in source
    assert "float translated_real = cosine * value.x - sine * value.y;" in source
    assert "translated_real * factor" in source
    assert "translated_imag * factor" in source


def test_relion_vdam_fused_source_uses_native_separate_accumulator_storage():
    import inspect

    from recovar import cuda_backproject

    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()
    assert "bool SEPARATE_DATA = false" in source
    assert "atomicAdd(&data_real_volume[off], sre);" in source
    assert "atomicAdd(&data_imag_volume[off], sim);" in source
    assert "float* data_real_volume" in source
    assert "float* data_imag_volume" in source
    fused_kernel = source.split(
        "__global__ void relion_vdam_mstep_fused_x_half_kernel(", 1
    )[1].split("cudaError_t launch_relion_vdam_mstep_fused_x_half", 1)[0]
    assert "template <bool INLINE_PROJECTOR>" in source
    assert "__shared__ float R[9];" in fused_kernel
    assert "__shared__ float E" not in fused_kernel
    assert "rk0 = (R[6] * x_unscaled + R[7] * y_unscaled)" in fused_kernel

    native_kernel = source.split(
        "__global__ void relion_vdam_native_sgd_f32_kernel(", 1
    )[1].split("__global__ void relion_vdam_denominator_after_sgd_f32_kernel", 1)[0]
    assert "RelionVdamProjectorKernel projector" in native_kernel
    assert "float* image_real" in native_kernel
    assert "float* image_imag" in native_kernel
    assert "float* translation_x" in native_kernel
    assert "float* translation_y" in native_kernel
    assert "if (weight >= significant_weight)" in native_kernel
    assert "weight = (weight / weight_norm) * ctf * minvsigma2;" in native_kernel
    assert "RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z1, y1, x1, dd111);" in native_kernel
    assert "denominator" not in native_kernel

    projector_launcher = source.split(
        "cudaError_t launch_relion_vdam_mstep_fused_projector_x_half(", 1
    )[1].split("__device__ __forceinline__ float relion_fine_diff2_update_f32", 1)[0]
    assert "relion_vdam_native_sgd_f32_kernel<<<rotation_count, 128" in projector_launcher
    assert "relion_vdam_denominator_after_sgd_f32_kernel<<<" in projector_launcher

    wrapper = inspect.getsource(cuda_backproject.relion_vdam_mstep_fused_x_half)
    assert "data_real_volume = jnp.asarray(data_volume.real" in wrapper
    assert "data_imag_volume = jnp.asarray(data_volume.imag" in wrapper
    assert "fused_data = jax.lax.complex(fused_real, fused_imag)" in wrapper

    projector_wrapper = inspect.getsource(
        cuda_backproject.relion_vdam_mstep_fused_projector_x_half
    )
    assert "data_real_volume = jnp.asarray(data_volume.real" in projector_wrapper
    assert "data_imag_volume = jnp.asarray(data_volume.imag" in projector_wrapper
    assert "fused_data = jax.lax.complex(fused_real, fused_imag)" in projector_wrapper


@pytest.mark.gpu
def test_relion_translate_score_f32_matches_float32_reference(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray(
        [0, 1, half_width - 1, 3 * half_width + 2, 8 * half_width, 15 * half_width + 7],
        dtype=np.int32,
    )
    images = np.asarray(
        [
            [1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, 3.0 + 2.0j, -0.75 - 0.5j, 8.0 - 3.0j],
            [-1.5 + 1.0j, 0.5 - 0.125j, 2.5 + 7.0j, -4.0 + 0.75j, 0.25 + 0.5j, -6.0 - 2.0j],
        ],
        dtype=np.complex64,
    )
    angles = np.asarray(
        [[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual = np.asarray(actual)

    expected = np.empty(
        (images.shape[0] * angles.shape[0], images.shape[1]),
        dtype=np.complex64,
    )
    for image_index, image in enumerate(images):
        for translation_index, (tx, ty) in enumerate(angles):
            output_row = image_index * angles.shape[0] + translation_index
            for pixel_row, pixel_index in enumerate(pixel_indices):
                x = int(pixel_index % half_width)
                y = int(pixel_index // half_width - image_shape[0] // 2)
                phase = np.float32(
                    np.float32(x) * tx + np.float32(y) * ty
                )
                sine = np.float32(np.sin(phase))
                cosine = np.float32(np.cos(phase))
                real = np.float32(
                    cosine * image[pixel_row].real
                    - sine * image[pixel_row].imag
                )
                imag = np.float32(
                    cosine * image[pixel_row].imag
                    + sine * image[pixel_row].real
                )
                expected[output_row, pixel_row] = np.complex64(real + 1j * imag)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


@pytest.mark.gpu
def test_relion_translate_score_f32_matches_sealed_relion_bits(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Match a fixed RELION 5.0 stack-42988 translation sample bitwise."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    # Eight spread-out pixels from the sealed RELION fine-operand capture
    # a81cf6c18e9ce47864c119ae3d827e3aeb64121bf8d071e01176e4bc350e1102.
    # Keeping the raw float32 words makes this a stable repository-owned
    # parity case rather than a tolerance-based numerical comparison.
    pixel_indices = np.asarray(
        [16512, 17823, 19226, 20537, 12394, 13705, 15108, 16420],
        dtype=np.int32,
    )
    image_words = np.asarray(
        [
            [3171083599, 0],
            [981717925, 988820227],
            [964390543, 987107983],
            [3148702193, 3140350956],
            [3130951531, 986468636],
            [3127477872, 986896523],
            [3136130221, 991661190],
            [3116871779, 3131905678],
        ],
        dtype=np.uint32,
    )
    expected_words = np.asarray(
        [
            [3171083599, 0],
            [3104834649, 990426048],
            [3121372780, 986491646],
            [3116391690, 3150027316],
            [3120789091, 989469894],
            [3130377719, 985453807],
            [3137459980, 991064183],
            [973414023, 3131532664],
        ],
        dtype=np.uint32,
    )
    images = (
        image_words[:, 0].view(np.float32)
        + np.complex64(1j) * image_words[:, 1].view(np.float32)
    ).astype(np.complex64)[None, :]
    angles = np.asarray([[1016464419, 1016464419]], dtype=np.uint32).view(
        np.float32
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            (256, 256),
        )
    actual_words = np.asarray(actual)[0].view(np.float32).view(np.uint32).reshape(-1, 2)

    np.testing.assert_array_equal(actual_words, expected_words)


def test_relion_translate_score_f32_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_translate_score_f32.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            (8, 8),
        )


def test_relion_translate_bpref_f32_validates_weight_shape(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(ValueError, match="weighted_ctf must have shape"):
        cuda_backproject.relion_translate_bpref_f32.__wrapped__(
            jnp.zeros((2, 3), dtype=jnp.complex64),
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
            (8, 8),
        )


def test_relion_vdam_mstep_sums_f32_validates_reference_shape():
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(ValueError, match="reference must have shape"):
        cuda_backproject.relion_vdam_mstep_sums_f32.__wrapped__(
            jnp.zeros((2, 3), dtype=jnp.complex64),
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((2, 4, 5), dtype=jnp.float32),
            jnp.zeros((5, 2), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
            jnp.zeros((2, 3, 3), dtype=jnp.complex64),
            (8, 8),
        )


def test_relion_vdam_mstep_fused_x_half_validates_reference_shape():
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(ValueError, match="reference must have shape"):
        cuda_backproject.relion_vdam_mstep_fused_x_half.__wrapped__(
            jnp.zeros((196,), dtype=jnp.complex64),
            jnp.zeros((196,), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.complex64),
            jnp.ones((1, 3), dtype=jnp.float32),
            jnp.ones((1, 3), dtype=jnp.float32),
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
            jnp.zeros((1, 2, 2), dtype=jnp.complex64),
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 2, 3, 3)),
            (8, 8),
            (7, 7, 7),
            2.0,
        )


def test_relion_vdam_mstep_fused_projector_x_half_validates_projector_shape():
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(TypeError, match="projector_full must be a nonempty complex64 cube"):
        cuda_backproject.relion_vdam_mstep_fused_projector_x_half.__wrapped__(
            jnp.zeros((196,), dtype=jnp.complex64),
            jnp.zeros((196,), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.complex64),
            jnp.ones((1, 3), dtype=jnp.float32),
            jnp.ones((1, 3), dtype=jnp.float32),
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
            jnp.zeros((3, 3), dtype=jnp.complex64),
            jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 2, 3, 3)),
            (8, 8),
            (7, 7, 7),
            2.0,
            2,
            1,
        )


@pytest.mark.gpu
def test_relion_vdam_mstep_sums_f32_matches_source_order_and_translation(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    images = np.asarray(
        [[1.25 - 0.5j, -2.0 + 0.125j, 0.75 + 3.0j]],
        dtype=np.complex64,
    )
    ctf = np.asarray([[0.75, -1.5, 0.25]], dtype=np.float32)
    minvsigma2 = np.asarray([[2.0, 0.125, 4.0]], dtype=np.float32)
    posterior = np.asarray(
        [[[0.125, 0.25, 0.5], [0.0, 0.75, 0.0625]]],
        dtype=np.float32,
    )
    reference = np.asarray(
        [[[0.5 + 0.25j, 1.0 - 2.0j, -0.125 + 0.5j],
          [-1.0 + 0.75j, 0.25 + 0.5j, 2.0 - 0.25j]]],
        dtype=np.complex64,
    )
    translation_angles = np.asarray(
        [[0.0, 0.0], [0.01831252, -0.006231174], [-0.03125, 0.015625]],
        dtype=np.float32,
    )
    pixel_indices = np.asarray([0, 1, 2], dtype=np.int32)

    with jax.default_device(gpu_device):
        actual_num, actual_den = cuda_backproject.relion_vdam_mstep_sums_f32(
            jnp.asarray(images),
            jnp.asarray(ctf),
            jnp.asarray(minvsigma2),
            jnp.asarray(posterior),
            jnp.asarray(translation_angles),
            jnp.asarray(pixel_indices),
            jnp.asarray(reference),
            (8, 8),
        )
        translated = cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(images),
            jnp.ones_like(jnp.asarray(ctf)),
            jnp.asarray(translation_angles),
            jnp.asarray(pixel_indices),
            (8, 8),
        )
    translated = np.asarray(translated).reshape(1, posterior.shape[2], -1)

    expected_num = np.zeros_like(reference)
    expected_den = np.zeros(reference.shape, dtype=np.float32)
    for rotation in range(posterior.shape[1]):
        for pixel in range(images.shape[1]):
            ref_real = np.float32(reference[0, rotation, pixel].real * ctf[0, pixel])
            ref_imag = np.float32(reference[0, rotation, pixel].imag * ctf[0, pixel])
            sum_real = np.float32(0.0)
            sum_imag = np.float32(0.0)
            fweight = np.float32(0.0)
            for translation in range(posterior.shape[2]):
                weight = np.float32(posterior[0, rotation, translation] * ctf[0, pixel])
                weight = np.float32(weight * minvsigma2[0, pixel])
                fweight = np.float32(fweight + np.float32(weight * ctf[0, pixel]))
                sum_real = np.float32(
                    sum_real
                    + np.float32(
                        (translated[0, translation, pixel].real - ref_real) * weight
                    )
                )
                sum_imag = np.float32(
                    sum_imag
                    + np.float32(
                        (translated[0, translation, pixel].imag - ref_imag) * weight
                    )
                )
            expected_num[0, rotation, pixel] = np.complex64(sum_real + 1j * sum_imag)
            expected_den[0, rotation, pixel] = fweight

    np.testing.assert_allclose(np.asarray(actual_num), expected_num, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual_den), expected_den, rtol=0.0, atol=2e-6)


@pytest.mark.gpu
def test_relion_vdam_mstep_fused_x_half_matches_two_stage_interior_source_order(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(6217)
    image_shape = (8, 8)
    max_r = 4.0
    volume_shape = (11, 11, 11)
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    n_particles, n_rotations, n_translations = 1, 2, 3
    images = (
        rng.normal(size=(n_particles, n_pixels))
        + 1j * rng.normal(size=(n_particles, n_pixels))
    ).astype(np.complex64)
    ctf = rng.uniform(0.25, 1.5, size=images.shape).astype(np.float32)
    # The generic EM x-half scatter and VDAM's SGD kernel intentionally have
    # different y-Nyquist and negative-y/x=0 boundary conventions. Compare
    # their shared interior here; the VDAM-native boundaries have a dedicated
    # behavioral test below.
    ctf[:, [0, 1, 2, 3, 4, 5, 10, 15]] = 0.0
    minvsigma2 = rng.uniform(0.5, 2.0, size=images.shape).astype(np.float32)
    posterior = rng.uniform(
        0.0, 0.5, size=(n_particles, n_rotations, n_translations)
    ).astype(np.float32)
    angles = np.asarray(
        [[0.0, 0.0], [0.01831252, -0.006231174], [-0.03125, 0.015625]],
        dtype=np.float32,
    )
    reference = (
        rng.normal(size=(n_particles, n_rotations, n_pixels))
        + 1j * rng.normal(size=(n_particles, n_rotations, n_pixels))
    ).astype(np.complex64)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (n_particles, n_rotations, 3, 3),
    ).copy()
    rotations[0, 1] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    centered_indices = np.arange(n_pixels, dtype=np.int32)
    centered_rows = centered_indices // (image_shape[1] // 2 + 1)
    columns = centered_indices % (image_shape[1] // 2 + 1)
    fftw_rows = (centered_rows - image_shape[0] // 2) % image_shape[0]
    pixel_indices = (fftw_rows * (image_shape[1] // 2 + 1) + columns).astype(np.int32)
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    expected_data0 = jnp.zeros((volume_size,), dtype=jnp.complex64)
    expected_weight0 = jnp.zeros((volume_size,), dtype=jnp.float32)
    actual_data0 = jnp.zeros((volume_size,), dtype=jnp.complex64)
    actual_weight0 = jnp.zeros((volume_size,), dtype=jnp.float32)

    with jax.default_device(gpu_device):
        sums, denominator = cuda_backproject.relion_vdam_mstep_sums_f32(
            jnp.asarray(images),
            jnp.asarray(ctf),
            jnp.asarray(minvsigma2),
            jnp.asarray(posterior),
            jnp.asarray(angles),
            jnp.asarray(centered_indices),
            jnp.asarray(reference),
            image_shape,
        )
        expected_data, expected_weight = (
            cuda_backproject.relion_fused_x_half_backproject_particle_grid_indexed(
                expected_data0,
                expected_weight0,
                sums,
                denominator,
                jnp.asarray(pixel_indices),
                jnp.asarray(rotations),
                image_shape,
                volume_shape,
                max_r,
            )
        )
        jax.block_until_ready((expected_data, expected_weight, denominator))
        actual_data, actual_weight, actual_denominator = (
            cuda_backproject.relion_vdam_mstep_fused_x_half(
                actual_data0,
                actual_weight0,
                jnp.asarray(images),
                jnp.asarray(ctf),
                jnp.asarray(minvsigma2),
                jnp.asarray(posterior),
                jnp.asarray(angles),
                jnp.asarray(pixel_indices),
                jnp.asarray(reference),
                jnp.asarray(rotations),
                image_shape,
                volume_shape,
                max_r,
            )
        )
    actual_data_np = np.asarray(actual_data)
    expected_data_np = np.asarray(expected_data)
    close_data = np.isclose(actual_data_np, expected_data_np, rtol=2e-6, atol=2e-5)
    if not np.all(close_data):
        bad = np.flatnonzero(~close_data)
        diagnostic = [
            (
                tuple(np.unravel_index(int(index), (volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1))),
                actual_data_np[index],
                expected_data_np[index],
            )
            for index in bad
        ]
        print(f"fused VDAM data mismatches: {diagnostic}")
    np.testing.assert_allclose(np.asarray(actual_denominator), np.asarray(denominator), rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(actual_data_np, expected_data_np, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_weight), np.asarray(expected_weight), rtol=2e-6, atol=2e-5)


@pytest.mark.gpu
def test_relion_vdam_mstep_fused_x_half_uses_native_sgd_y_boundaries(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_shape = (8, 8)
    volume_shape = (11, 11, 11)
    half_width = image_shape[1] // 2 + 1
    fftw_rows = np.asarray([image_shape[0] // 2, image_shape[0] // 2 + 1], dtype=np.int32)
    pixel_indices = fftw_rows * half_width
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)

    with jax.default_device(gpu_device):
        actual_data, actual_weight, actual_denominator = (
            cuda_backproject.relion_vdam_mstep_fused_x_half(
                jnp.zeros((volume_size,), dtype=jnp.complex64),
                jnp.zeros((volume_size,), dtype=jnp.float32),
                jnp.ones((1, 2), dtype=jnp.complex64),
                jnp.ones((1, 2), dtype=jnp.float32),
                jnp.ones((1, 2), dtype=jnp.float32),
                jnp.ones((1, 1, 1), dtype=jnp.float32),
                jnp.asarray([[0.0, 0.1]], dtype=jnp.float32),
                jnp.asarray(pixel_indices),
                jnp.zeros((1, 1, 2), dtype=jnp.complex64),
                jnp.eye(3, dtype=jnp.float32)[None, None],
                image_shape,
                volume_shape,
                4.0,
            )
        )

    actual_data_np = np.asarray(actual_data).reshape(11, 11, 6)
    actual_weight_np = np.asarray(actual_weight).reshape(11, 11, 6)
    np.testing.assert_allclose(actual_denominator, np.ones((1, 1, 2), dtype=np.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_data_np[5, 9, 0], np.exp(0.4j), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_data_np[5, 2, 0], np.exp(-0.3j), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_weight_np[[5, 5], [9, 2], [0, 0]], np.ones(2), rtol=0.0, atol=0.0)
    assert np.count_nonzero(actual_data_np) == 2


@pytest.mark.gpu
def test_relion_vdam_mstep_fused_projector_zero_matches_preprojected_zero(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_shape = (8, 8)
    volume_shape = (11, 11, 11)
    max_r = 4.0
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.arange(image_shape[0] * half_width, dtype=np.int32)
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    rng = np.random.default_rng(9017)
    images = (rng.normal(size=(1, pixel_indices.size)) + 1j * rng.normal(size=(1, pixel_indices.size))).astype(
        np.complex64
    )
    ctf = rng.uniform(0.25, 1.25, size=images.shape).astype(np.float32)
    minvsigma2 = rng.uniform(0.5, 2.0, size=images.shape).astype(np.float32)
    posterior = rng.uniform(0.0, 0.5, size=(1, 2, 3)).astype(np.float32)
    angles = np.asarray([[0.0, 0.0], [0.01, -0.02], [-0.03, 0.015]], dtype=np.float32)
    rotations = np.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 2, 3, 3))

    with jax.default_device(gpu_device):
        common = (
            jnp.zeros((volume_size,), dtype=jnp.complex64),
            jnp.zeros((volume_size,), dtype=jnp.float32),
            jnp.asarray(images),
            jnp.asarray(ctf),
            jnp.asarray(minvsigma2),
            jnp.asarray(posterior),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
        )
        expected = cuda_backproject.relion_vdam_mstep_fused_x_half(
            *common,
            jnp.zeros((1, 2, pixel_indices.size), dtype=jnp.complex64),
            rotations,
            image_shape,
            volume_shape,
            max_r,
        )
        actual = cuda_backproject.relion_vdam_mstep_fused_projector_x_half(
            *common,
            jnp.zeros((11, 11, 11), dtype=jnp.complex64),
            rotations,
            image_shape,
            volume_shape,
            max_r,
            4,
            1,
        )
        jax.block_until_ready((expected, actual))

    for expected_value, actual_value in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=0.0)


@pytest.mark.gpu
def test_relion_translate_bpref_f32_matches_translate_then_weight(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    half_width = image_shape[1] // 2 + 1
    pixel_indices = np.asarray(
        [0, 1, 3 * half_width + 2, 8 * half_width, 15 * half_width + 7],
        dtype=np.int32,
    )
    images = np.asarray(
        [[1.0 + 0.5j, -2.0 + 0.25j, 0.125 - 4.0j, 3.0 + 2.0j, -0.75 - 0.5j]],
        dtype=np.complex64,
    )
    weighted_ctf = np.asarray(
        [[2.0, -0.25, 1.5, 1000.0, -3.0]],
        dtype=np.float32,
    )
    angles = np.asarray(
        [[0.0, 0.0], [0.018312519416213036, -0.006231173872947693]],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(images),
            jnp.asarray(weighted_ctf),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
        translated = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual = np.asarray(actual)
    translated = np.asarray(translated).reshape(1, angles.shape[0], -1)
    expected = np.empty_like(translated)
    for translation_index in range(angles.shape[0]):
        for pixel_index in range(images.shape[1]):
            expected[0, translation_index, pixel_index] = np.complex64(
                np.float32(translated[0, translation_index, pixel_index].real * weighted_ctf[0, pixel_index])
                + 1j
                * np.float32(translated[0, translation_index, pixel_index].imag * weighted_ctf[0, pixel_index])
            )

    np.testing.assert_array_equal(actual.reshape(expected.shape), expected)
