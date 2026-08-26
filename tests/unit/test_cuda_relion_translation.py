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


def test_relion_translation_angle_scale_changes_only_final_angle_operand():
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    translations = np.asarray([[0.25, -1.75]], dtype=np.float64)
    baseline_translations = translations.copy()
    angle_scale = np.float64("0.99999976470593788")
    angles = _relion_translation_angles_f32(
        translations,
        (384, 384),
        angle_scale=angle_scale,
    )
    expected = np.asarray(
        -2.0 * np.pi * translations * angle_scale / 384.0,
        dtype=np.float32,
    )

    np.testing.assert_array_equal(angles.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(translations, baseline_translations)


def test_relion_k1_translation_angle_scale_uses_model_over_optics_pixel_size():
    from recovar.em.dense_single_volume.iteration_loop import (
        _relion_k1_translation_angle_scale,
    )

    scale = _relion_k1_translation_angle_scale(
        n_classes=1,
        model_pixel_size=544.0 / 384.0,
        optics_pixel_sizes=np.asarray([1.416667], dtype=np.float64),
    )
    assert scale == pytest.approx(0.99999976470593788, rel=0.0, abs=1e-16)
    assert _relion_k1_translation_angle_scale(
        n_classes=4,
        model_pixel_size=544.0 / 384.0,
        optics_pixel_sizes=np.asarray([1.416667], dtype=np.float64),
    ) == 1.0


def test_relion_k1_translation_angle_scale_rejects_heterogeneous_optics():
    from recovar.em.dense_single_volume.iteration_loop import (
        _relion_k1_translation_angle_scale,
    )

    with pytest.raises(NotImplementedError, match="one shared optics pixel size"):
        _relion_k1_translation_angle_scale(
            n_classes=1,
            model_pixel_size=1.5,
            optics_pixel_sizes=np.asarray([1.5, 1.6], dtype=np.float64),
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


@pytest.mark.gpu
def test_relion_translate_bpref_f32_matches_native_captured_bits(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (256, 256)
    pixel_indices = np.asarray(
        [13031, 13160, 13161, 13167],
        dtype=np.int32,
    )
    image_words = np.asarray(
        [
            [3263777648, 1133111626],
            [3272549528, 1090742966],
            [1127815781, 1125080211],
            [3252159688, 1125857107],
        ],
        dtype=np.uint32,
    )
    images = (
        image_words[:, 0].view(np.float32)
        + np.complex64(1j) * image_words[:, 1].view(np.float32)
    ).astype(np.complex64)[None, :]
    weighted_ctf = np.asarray(
        [[3106311266, 3108593774, 3108351983, 3104897938]],
        dtype=np.uint32,
    ).view(np.float32)
    angles = np.asarray([[3168013433, 3176042026]], dtype=np.uint32).view(
        np.float32
    )
    expected_words = np.asarray(
        [
            [1027138631, 3125411712],
            [1008901170, 1020422283],
            [1013229036, 3173746131],
            [1017732809, 3152071391],
        ],
        dtype=np.uint32,
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(images),
            jnp.asarray(weighted_ctf),
            jnp.asarray(angles),
            jnp.asarray(pixel_indices),
            image_shape,
        )
    actual_words = (
        np.asarray(actual)[0].view(np.float32).view(np.uint32).reshape(-1, 2)
    )

    np.testing.assert_array_equal(actual_words, expected_words)
