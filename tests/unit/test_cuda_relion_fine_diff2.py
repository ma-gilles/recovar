"""Focused tests for RELION's fused fine-Gaussian CUDA FFI."""

from itertools import permutations
from pathlib import Path
from decimal import Decimal, localcontext

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def _fma32(left, right, addend):
    return np.asarray(
        np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _fma64(left, right, addend):
    with localcontext() as context:
        context.prec = 200
        exact = (
            Decimal.from_float(float(left)) * Decimal.from_float(float(right))
            + Decimal.from_float(float(addend))
        )
    return np.float64(float(exact))


def _production_reference(reference, shifted, weight, lookup):
    lanes = np.zeros(256, dtype=np.float32)
    for full_pixel, compact_pixel in enumerate(lookup):
        if compact_pixel < 0:
            continue
        diff_real = np.float32(
            reference[compact_pixel].real - shifted[compact_pixel].real
        )
        diff_imag = np.float32(
            reference[compact_pixel].imag - shifted[compact_pixel].imag
        )
        imag_square = np.float32(diff_imag * diff_imag)
        square_sum = _fma32(diff_real, diff_real, imag_square)
        half_square_sum = np.float32(square_sum * np.float32(0.5))
        lane = full_pixel % 256
        lanes[lane] = _fma32(half_square_sum, weight[compact_pixel], lanes[lane])
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)
    return np.float32(lanes[0])


def _production_reference_f64(reference, shifted, weight, lookup):
    lanes = np.zeros(256, dtype=np.float64)
    for full_pixel, compact_pixel in enumerate(lookup):
        if compact_pixel < 0:
            continue
        diff_real = np.float64(reference[compact_pixel].real - shifted[compact_pixel].real)
        diff_imag = np.float64(reference[compact_pixel].imag - shifted[compact_pixel].imag)
        imag_square = np.float64(diff_imag * diff_imag)
        square_sum = _fma64(diff_real, diff_real, imag_square)
        half_square_sum = np.float64(square_sum * np.float64(0.5))
        lane = full_pixel % 256
        lanes[lane] = _fma64(half_square_sum, weight[compact_pixel], lanes[lane])
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float64)
    return np.float64(lanes[0])


def _coarse_production_results(
    reference,
    shifted,
    weight,
    lookup,
    *,
    translation_count,
    initial_diff2=np.float32(0),
):
    active_lanes = 128 // translation_count
    lanes = np.zeros(active_lanes, dtype=np.float32)
    for chunk_start in range(0, lookup.size, 32):
        for lane in range(active_lanes):
            for pixel_in_chunk in range(lane, 32, active_lanes):
                full_pixel = chunk_start + pixel_in_chunk
                if full_pixel >= lookup.size:
                    break
                compact_pixel = lookup[full_pixel]
                if compact_pixel < 0:
                    continue
                diff_real = np.float32(
                    reference[compact_pixel].real - shifted[compact_pixel].real
                )
                diff_imag = np.float32(
                    reference[compact_pixel].imag - shifted[compact_pixel].imag
                )
                imag_square = np.float32(diff_imag * diff_imag)
                square_sum = _fma32(diff_real, diff_real, imag_square)
                half_square_sum = np.float32(square_sum * np.float32(0.5))
                lanes[lane] = _fma32(
                    half_square_sum,
                    weight[compact_pixel],
                    lanes[lane],
                )
    possible = set()
    for order in permutations(range(active_lanes)):
        total = np.float32(initial_diff2)
        for lane in order:
            total = np.float32(total + lanes[lane])
        possible.add(int(total.view(np.uint32)))
    return possible


def _operands():
    rng = np.random.default_rng(20)
    pixel_count = 513
    reference = (
        rng.normal(0, 0.02, pixel_count)
        + 1j * rng.normal(0, 0.02, pixel_count)
    ).astype(np.complex64)
    shifted = (
        rng.normal(0, 0.02, pixel_count)
        + 1j * rng.normal(0, 0.02, pixel_count)
    ).astype(np.complex64)
    weight = rng.uniform(0, 150_000, pixel_count).astype(np.float32)
    weight[rng.random(pixel_count) < 0.2] = 0
    lookup = np.arange(pixel_count, dtype=np.int32)
    return reference, shifted, weight, lookup


def test_relion_fine_diff2_cuda_source_pins_production_rounding_order():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index("relion_fine_diff2_update_f32")
    block = source[start : source.index("__global__", start)]
    assert "__fsub_rn(reference.x, shifted_image.x)" in block
    assert "__fmul_rn(diff_imag, diff_imag)" in block
    assert "__fmaf_rn(diff_real, diff_real, imag_square)" in block
    assert "__fmul_rn(square_sum, 0.5f)" in block
    assert "__fmaf_rn(half_square_sum, weight, lane_sum)" in block


def test_relion_coarse_diff2_cuda_source_pins_production_topology():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index("relion_coarse_diff2_rectangular_f32_kernel")
    block = source[start : source.index("cudaError_t", start)]
    assert "kRelionCoarseDiff2BlockSize = 128" in source
    assert "kRelionCoarseEulersPerBlock = 16" in source
    assert "kRelionCoarsePrefetchFraction = 4" in source
    assert "threadIdx.x % translation_count" in block
    assert "threadIdx.x / translation_count" in block
    assert "pixel_in_chunk += active_lanes" in block
    assert "atomicAdd(" in block
    f64_start = source.index("relion_coarse_diff2_rectangular_f64_kernel")
    f64_block = source[f64_start : source.index("cudaError_t", f64_start)]
    assert "double lane_sums" in f64_block
    assert "relion_fine_diff2_update_f64" in f64_block
    assert "atomicAdd(" in f64_block


def test_k1_coarse_gaussian_flag_is_off_by_default_and_k1_only(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_GAUSSIAN_FFI", raising=False)
    assert not significance._k1_coarse_gaussian_ffi_enabled()
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_FFI", "1")
    assert significance._k1_coarse_gaussian_ffi_enabled()

    source = Path(significance.__file__).read_text()
    start = source.index("if coarse_gaussian_ffi_enabled:")
    guard = source[start : source.index("tree_rescore_fftw_order", start)]
    assert "if n_classes != 1:" in guard
    assert "restricted to K=1" in guard
    assert "square_score_indices_np" in guard
    assert "square=True" in guard
    assert "include_dc=True" in guard


def test_k1_coarse_gaussian_sincosf_flag_is_off_and_requires_ffi(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", raising=False)
    assert not significance._k1_coarse_gaussian_sincosf_enabled()
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", "1")
    assert significance._k1_coarse_gaussian_sincosf_enabled()

    source = Path(significance.__file__).read_text()
    assert "coarse_gaussian_sincosf_enabled and not coarse_gaussian_ffi_enabled" in source
    assert "return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled" in source


def test_coarse_gaussian_square_operands_reuse_weighted_score_inputs():
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_gaussian_square_operands,
    )

    shifted = jnp.asarray(
        [
            [2 + 4j, 12 + 6j, 99 + 3j, -8 + 16j],
            [4 + 2j, 6 + 18j, 77 + 5j, 12 - 4j],
        ],
        dtype=jnp.complex64,
    )
    score_weight = jnp.asarray([[2.0, 3.0, 0.0, 4.0]], dtype=jnp.float32)
    half_weights = jnp.asarray([1.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    score_indices = jnp.asarray([3, 1, 2], dtype=jnp.int32)
    score_active_mask = jnp.asarray([True, False, True])

    corrected, pixel_weight = _relion_coarse_gaussian_square_operands(
        shifted,
        score_weight,
        half_weights,
        score_indices,
        score_active_mask,
        batch_size=1,
        n_trans=2,
    )

    np.testing.assert_allclose(
        np.asarray(corrected),
        np.asarray(
            [
                [
                    [(-8 + 16j) / 4, 0, 0],
                    [(12 - 4j) / 4, 0, 0],
                ]
            ],
            dtype=np.complex64,
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        np.asarray(pixel_weight),
        np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_coarse_gaussian_sincosf_operands_reuse_unshifted_weighted_input(
    monkeypatch,
):
    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_gaussian_square_operands_sincosf,
    )

    captured = {}

    def fake_translate(images, translation_angles, pixel_indices, image_shape):
        captured.update(
            images=np.asarray(images),
            translation_angles=np.asarray(translation_angles),
            pixel_indices=np.asarray(pixel_indices),
            image_shape=tuple(image_shape),
        )
        return jnp.repeat(images[:, None, :], 2, axis=1).reshape(2, -1)

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        fake_translate,
    )
    unshifted_weighted = jnp.asarray(
        [[2 + 4j, 12 + 6j, 99 + 3j, -8 + 16j]],
        dtype=jnp.complex64,
    )
    score_weight = jnp.asarray([[2.0, 3.0, 0.0, 4.0]], dtype=jnp.float32)
    half_weights = jnp.asarray([1.0, 2.0, 2.0, 1.0], dtype=jnp.float32)
    score_indices = jnp.asarray([3, 1, 2], dtype=jnp.int32)
    score_active_mask = jnp.asarray([True, False, True])
    translations = np.asarray([[0.0, 0.0], [1.0, -2.0]], dtype=np.float32)

    corrected, pixel_weight = _relion_coarse_gaussian_square_operands_sincosf(
        unshifted_weighted,
        score_weight,
        half_weights,
        score_indices,
        score_active_mask,
        translations,
        (8, 8),
    )

    expected_base = np.asarray(
        [[(-8 + 16j) / 4, 0, 0]],
        dtype=np.complex64,
    )
    np.testing.assert_array_equal(captured["images"], expected_base)
    np.testing.assert_array_equal(captured["pixel_indices"], np.asarray([3, 1, 2]))
    np.testing.assert_allclose(
        captured["translation_angles"],
        -2.0 * np.pi * translations / 8.0,
        rtol=0,
        atol=np.finfo(np.float32).eps,
    )
    assert captured["image_shape"] == (8, 8)
    np.testing.assert_array_equal(
        np.asarray(corrected),
        np.repeat(expected_base[:, None, :], 2, axis=1),
    )
    np.testing.assert_array_equal(
        np.asarray(pixel_weight),
        np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_coarse_gaussian_sincosf_operands_preserve_float64(monkeypatch):
    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_gaussian_square_operands_sincosf,
    )

    captured = {}

    def fake_translate(images, translation_angles, pixel_indices, image_shape):
        captured.update(
            images=np.asarray(images),
            translation_angles=np.asarray(translation_angles),
            pixel_indices=np.asarray(pixel_indices),
            image_shape=tuple(image_shape),
        )
        return jnp.repeat(images[:, None, :], 2, axis=1).reshape(2, -1)

    monkeypatch.setattr(cuda_backproject, "relion_translate_score_f64", fake_translate)
    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        lambda *args, **kwargs: pytest.fail("float32 translation target was called"),
    )
    corrected, pixel_weight, unshifted = _relion_coarse_gaussian_square_operands_sincosf(
        jnp.asarray([[2 + 4j, -8 + 16j]], dtype=jnp.complex128),
        jnp.asarray([[2.0, 4.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        jnp.asarray([1, 0], dtype=jnp.int32),
        jnp.asarray([True, True]),
        np.asarray([[0.0, 0.0], [1.0, -2.0]], dtype=np.float64),
        (8, 8),
        return_unshifted=True,
    )

    assert captured["images"].dtype == np.complex128
    assert captured["translation_angles"].dtype == np.float64
    assert np.asarray(corrected).dtype == np.complex128
    assert np.asarray(pixel_weight).dtype == np.float64
    assert np.asarray(unshifted).dtype == np.complex128


@pytest.mark.gpu
def test_coarse_gaussian_sincosf_operands_run_cuda_translation(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_gaussian_square_operands_sincosf,
    )
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    image_shape = (16, 16)
    score_indices = jnp.asarray([0, 1, 8, 17, 46, 88], dtype=jnp.int32)
    translations = np.asarray([[0.0, 0.0], [1.25, -0.75]], dtype=np.float32)
    half_size = image_shape[0] * (image_shape[1] // 2 + 1)
    score_weight = jnp.zeros((1, half_size), dtype=jnp.float32).at[:, score_indices].set(
        jnp.asarray([[2.0, 4.0, 3.0, 5.0, 0.0, 2.0]], dtype=jnp.float32),
    )
    half_weights = jnp.zeros(half_size, dtype=jnp.float32).at[score_indices].set(
        jnp.asarray([1.0, 2.0, 1.0, 2.0, 2.0, 1.0], dtype=jnp.float32),
    )
    unshifted_weighted = jnp.zeros((1, half_size), dtype=jnp.complex64).at[:, score_indices].set(
        jnp.asarray(
            [[2 + 1j, -4 + 2j, 3 - 6j, 10 + 5j, 7 + 9j, -2 - 4j]],
            dtype=jnp.complex64,
        ),
    )
    expected_input = jnp.asarray(
        [[1 + 0.5j, -1 + 0.5j, 1 - 2j, 2 + 1j, 0, -1 - 2j]],
        dtype=jnp.complex64,
    )

    with jax.default_device(gpu_device):
        actual, actual_weight = _relion_coarse_gaussian_square_operands_sincosf(
            unshifted_weighted,
            score_weight,
            half_weights,
            score_indices,
            jnp.ones(score_indices.shape, dtype=jnp.bool_),
            translations,
            image_shape,
        )
        expected = cuda_backproject.relion_translate_score_f32(
            expected_input,
            jnp.asarray(
                _relion_translation_angles_f32(translations, image_shape),
                dtype=jnp.float32,
            ),
            score_indices,
            image_shape,
        ).reshape(1, 2, 6)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(actual_weight),
        np.asarray([[2.0, 8.0, 3.0, 10.0, 0.0, 2.0]], dtype=np.float32),
    )


@pytest.mark.gpu
def test_relion_coarse_diff2_rectangular_matches_atomic_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(29)
    batch_size, rotation_count, translation_count = 2, 17, 29
    compact_pixel_count, full_pixel_count = 421, 513
    reference = (
        rng.normal(0, 0.02, (rotation_count, compact_pixel_count))
        + 1j * rng.normal(0, 0.02, (rotation_count, compact_pixel_count))
    ).astype(np.complex64)
    shifted = (
        rng.normal(
            0,
            0.02,
            (batch_size, translation_count, compact_pixel_count),
        )
        + 1j
        * rng.normal(
            0,
            0.02,
            (batch_size, translation_count, compact_pixel_count),
        )
    ).astype(np.complex64)
    weight = rng.uniform(0, 150_000, (batch_size, compact_pixel_count)).astype(
        np.float32
    )
    initial_diff2 = rng.uniform(10_000, 20_000, batch_size).astype(np.float32)
    retained = np.sort(
        rng.choice(full_pixel_count, compact_pixel_count, replace=False)
    )
    lookup = np.full(full_pixel_count, -1, dtype=np.int32)
    lookup[retained] = np.arange(compact_pixel_count, dtype=np.int32)

    with jax.default_device(gpu_device):
        actual = np.asarray(
            cuda_backproject.relion_coarse_diff2_rectangular_f32(
                jnp.asarray(reference),
                jnp.asarray(shifted),
                jnp.asarray(weight),
                jnp.asarray(initial_diff2),
                jnp.asarray(lookup),
            )
        )

    for batch in range(batch_size):
        for rotation in range(rotation_count):
            for translation in range(translation_count):
                possible = _coarse_production_results(
                    reference[rotation],
                    shifted[batch, translation],
                    weight[batch],
                    lookup,
                    translation_count=translation_count,
                    initial_diff2=initial_diff2[batch],
                )
                actual_bits = int(
                    actual[batch, rotation, translation].view(np.uint32)
                )
                assert actual_bits in possible


@pytest.mark.gpu
def test_relion_fine_diff2_rectangular_matches_production_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference, shifted, weight, lookup = _operands()
    expected = _production_reference(reference, shifted, weight, lookup)

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_rectangular_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
        )

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray([[[expected]]], dtype=np.float32).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_fine_diff2_pairs_matches_production_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference, shifted, weight, lookup = _operands()
    expected = _production_reference(reference, shifted, weight, lookup)

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_pairs_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
        )

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray([[expected]], dtype=np.float32).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_fine_diff2_rectangular_f64_matches_acc_double_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference32, shifted32, weight32, lookup = _operands()
    reference = reference32.astype(np.complex128)
    shifted = shifted32.astype(np.complex128)
    weight = weight32.astype(np.float64)
    expected = _production_reference_f64(reference, shifted, weight, lookup)

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_rectangular_f64(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
        )

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint64),
        np.asarray([[[expected]]], dtype=np.float64).view(np.uint64),
    )


@pytest.mark.parametrize(
    "function_name,expected_target,expected_shape",
    [
        (
            "relion_fine_diff2_rectangular_f64",
            "cuda_relion_fine_diff2_rectangular_f64",
            (1, 2, 3),
        ),
        (
            "relion_fine_diff2_pairs_f64",
            "cuda_relion_fine_diff2_pairs_f64",
            (1, 2),
        ),
    ],
)
def test_relion_fine_diff2_f64_uses_double_ffi_target(
    monkeypatch, function_name, expected_target, expected_shape
):
    import recovar.cuda_backproject as cuda_backproject

    call = {}

    def fake_ffi_call(target, out_type, **options):
        call.update(target=target, out_type=out_type, options=options)
        return lambda *_args: jnp.zeros(out_type.shape, out_type.dtype)

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    function = getattr(cuda_backproject, function_name).__wrapped__
    reference_shape = (1, 2, 5)
    shifted_shape = (1, 3, 5) if "rectangular" in function_name else reference_shape
    actual = function(
        jnp.zeros(reference_shape, dtype=jnp.complex128),
        jnp.zeros(shifted_shape, dtype=jnp.complex128),
        jnp.ones((1, 5), dtype=jnp.float64),
        jnp.arange(5, dtype=jnp.int32),
    )

    assert actual.shape == expected_shape
    assert actual.dtype == jnp.float64
    assert call["target"] == expected_target


@pytest.mark.parametrize(
    "function_name",
    [
        "relion_fine_diff2_rectangular_f32",
        "relion_fine_diff2_pairs_f32",
        "relion_fine_diff2_rectangular_f64",
        "relion_fine_diff2_pairs_f64",
    ],
)
def test_relion_fine_diff2_fails_closed_without_gpu(monkeypatch, function_name):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    function = getattr(cuda_backproject, function_name).__wrapped__
    is_f64 = function_name.endswith("f64")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        function(
            jnp.zeros((1, 1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.zeros((1, 1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float64 if is_f64 else jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_relion_coarse_diff2_fails_closed_without_gpu(monkeypatch, dtype):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    is_f64 = dtype == jnp.float64
    function = (
        cuda_backproject.relion_coarse_diff2_rectangular_f64
        if is_f64
        else cuda_backproject.relion_coarse_diff2_rectangular_f32
    )
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        function.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.zeros((1, 29, 2), dtype=jnp.complex128 if is_f64 else jnp.complex64),
            jnp.ones((1, 2), dtype=dtype),
            jnp.zeros((1,), dtype=dtype),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    "reference_shape,shifted_shape,weight_shape,expected_route,expected_shape",
    [
        ((2, 3, 1, 7), (2, 1, 4, 7), (2, 1, 1, 7), "rectangular", (2, 3, 4)),
        ((3, 1, 7), (1, 4, 7), (1, 1, 7), "rectangular", (3, 4)),
        ((2, 5, 7), (2, 5, 7), (2, 1, 7), "pairs", (2, 5)),
    ],
)
def test_sparse_pass2_fused_flag_routes_supported_operand_layouts(
    monkeypatch,
    reference_shape,
    shifted_shape,
    weight_shape,
    expected_route,
    expected_shape,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_fine_diff2_sum,
    )

    routes = []

    def rectangular(reference, shifted_image, weight, full_to_compact):
        routes.append(
            (
                "rectangular",
                reference.shape,
                shifted_image.shape,
                weight.shape,
                full_to_compact.shape,
            )
        )
        return jnp.zeros(
            (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
            dtype=jnp.float32,
        )

    def pairs(reference, shifted_image, weight, full_to_compact):
        routes.append(
            (
                "pairs",
                reference.shape,
                shifted_image.shape,
                weight.shape,
                full_to_compact.shape,
            )
        )
        return jnp.zeros(reference.shape[:2], dtype=jnp.float32)

    monkeypatch.setenv("RECOVAR_RELION_FINE_DIFF2_FUSED_FFI", "1")
    monkeypatch.setattr(
        cuda_backproject,
        "relion_fine_diff2_rectangular_f32",
        rectangular,
    )
    monkeypatch.setattr(cuda_backproject, "relion_fine_diff2_pairs_f32", pairs)

    actual = _relion_cuda_fine_diff2_sum(
        jnp.zeros(reference_shape, dtype=jnp.complex64),
        jnp.zeros(shifted_shape, dtype=jnp.complex64),
        jnp.ones(weight_shape, dtype=jnp.float32),
        jnp.arange(7, dtype=jnp.int32),
    )

    assert actual.shape == expected_shape
    assert routes[0][0] == expected_route
    assert routes[0][-1] == (7,)


def test_sparse_pass2_fused_flag_routes_float64_to_f64_ffi(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_fine_diff2_sum,
    )

    calls = []

    def rectangular(reference, shifted_image, weight, full_to_compact):
        calls.append((reference.dtype, shifted_image.dtype, weight.dtype))
        return jnp.zeros(
            (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
            dtype=jnp.float64,
        )

    monkeypatch.setenv("RECOVAR_RELION_FINE_DIFF2_FUSED_FFI", "1")
    monkeypatch.setattr(
        cuda_backproject,
        "relion_fine_diff2_rectangular_f64",
        rectangular,
    )
    actual = _relion_cuda_fine_diff2_sum(
        jnp.zeros((2, 3, 1, 7), dtype=jnp.complex128),
        jnp.zeros((2, 1, 4, 7), dtype=jnp.complex128),
        jnp.ones((2, 1, 1, 7), dtype=jnp.float64),
        jnp.arange(7, dtype=jnp.int32),
    )

    assert actual.shape == (2, 3, 4)
    assert actual.dtype == jnp.float64
    assert calls == [(jnp.complex128, jnp.complex128, jnp.float64)]
