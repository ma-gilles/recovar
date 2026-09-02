"""Focused tests for RELION's fused fine-Gaussian CUDA FFI."""

from itertools import permutations
from pathlib import Path

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


def _off_grid_so3_rotations() -> np.ndarray:
    """Deterministic proper rotations with all three model axes active."""

    matrices = []
    for angle_x, angle_y, angle_z in (
        (0.37, -0.52, 0.19),
        (-0.91, 0.43, 1.17),
        (1.20, -0.73, -0.44),
        (-0.28, -1.01, 0.66),
    ):
        cosine_x, sine_x = np.cos(angle_x), np.sin(angle_x)
        cosine_y, sine_y = np.cos(angle_y), np.sin(angle_y)
        cosine_z, sine_z = np.cos(angle_z), np.sin(angle_z)
        rotation_x = np.asarray(
            [[1, 0, 0], [0, cosine_x, -sine_x], [0, sine_x, cosine_x]],
            dtype=np.float64,
        )
        rotation_y = np.asarray(
            [[cosine_y, 0, sine_y], [0, 1, 0], [-sine_y, 0, cosine_y]],
            dtype=np.float64,
        )
        rotation_z = np.asarray(
            [[cosine_z, -sine_z, 0], [sine_z, cosine_z, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        matrices.append(np.asarray(rotation_z @ rotation_y @ rotation_x, np.float32))
    return np.stack(matrices)


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


def test_relion_fused_translate_cuda_source_pins_native_block_topology():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    assert "constexpr int kRelionFineDiff2TranslationCapacity = 7;" in source
    assert "constexpr int kRelionFineDiff2Ref3dJobChunk = 4;" in source
    assert "relion_fine_diff2_fused_translate_rectangular_f32_kernel" in source
    assert "relion_score_translate_f32(" in source
    assert "translation_offset * kRelionFineDiff2BlockSize" in source
    assert "lane_sums[lane_index] = relion_fine_diff2_update_f32(" in source


def test_relion_fine_native_texture_source_pins_production_topology():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    kernel_name = source.index(
        "relion_fine_diff2_native_texture_rectangular_f32_kernel"
    )
    start = source.rindex("__global__", 0, kernel_name)
    block = source[start : source.index("cudaError_t", start)]
    assert "__launch_bounds__(kRelionFineDiff2BlockSize)" in block
    assert "kRelionFineDiff2Ref3dJobChunk" in block
    assert "relion_coarse_project_texture_f32(" in block
    assert "relion_score_translate_f32(" in block
    assert "relion_fine_diff2_update_f32(" in block
    assert "for (int width = kRelionFineDiff2BlockSize / 2" in block
    assert "initial_diff2[0]" in block

    launcher_start = source.index(
        "launch_relion_fine_diff2_native_texture_rectangular_f32"
    )
    launcher = source[launcher_start : source.index("__global__", launcher_start)]
    assert "float* particle_output = nullptr" in launcher
    assert "for (int64_t batch = 0; batch < batch_size; ++batch)" in launcher
    assert "eulers + batch * rotation_count * 9" in launcher
    assert "image + batch * compact_pixel_count" in launcher
    assert "weight + batch * compact_pixel_count" in launcher
    assert "initial_diff2 + batch" in launcher
    assert "output + batch * hypotheses_per_particle" in launcher
    assert "cudaMemcpyDeviceToDevice" in launcher

    assert "RelionFineDiff2NativeTextureRectangularF32Impl" in source
    assert "RelionFineDiff2NativeTextureRectangularF32," in source


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


def test_relion_coarse_diff2_cuda_source_pins_serial_particle_diagnostic():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index("launch_relion_coarse_diff2_rectangular_f32")
    block = source[start : source.index(
        "launch_relion_coarse_diff2_fused_translate_rectangular_f32", start
    )]
    assert "if (serial_particle_launches)" in block
    assert "for (int64_t batch = 0; batch < batch_size; ++batch)" in block
    assert "static_cast<unsigned int>(rotation_blocks)" in block
    assert "shifted_image + image_offset" in block
    assert "output + output_offset" in block

    iteration_loop = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "em"
        / "dense_single_volume"
        / "iteration_loop.py"
    ).read_text()
    assert (
        "RECOVAR_K1_COARSE_GAUSSIAN_FORCE_FULL_ROTATION_GRID_DIAGNOSTIC"
        in iteration_loop
    )
    assert '"RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"' in iteration_loop
    assert "and not native_texture_scoring" in iteration_loop
    assert "return plan.image_batch_size, int(n_rot)" in iteration_loop

    fused_start = source.index(
        "relion_coarse_diff2_fused_translate_rectangular_f32_kernel"
    )
    fused_block = source[fused_start : source.index("cudaError_t", fused_start)]
    assert "relion_coarse_score_translate_f32(" in fused_block
    assert "thread % translation_count" in fused_block
    assert "thread / translation_count" in fused_block
    assert "atomicAdd(" in fused_block


def test_relion_coarse_normalized_cc_source_pins_native_tree_and_atomics():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index("relion_coarse_normalized_cc_pairs_f32_kernel")
    block = source[start : source.index("cudaError_t", start)]
    assert "packed_pixel += kRelionCoarseDiff2BlockSize" in block
    assert "reference_value.x * image_value.x" in block
    assert "reference_value.y * image_value.y" in block
    assert "reference_value.x * reference_value.x" in block
    assert "reference_value.y * reference_value.y" in block
    assert "for (int width = kRelionCoarseDiff2BlockSize / 2" in block
    assert "sqrtf(" in block
    assert "atomicAdd(&output[candidate], contribution)" in block

    texture_start = source.index(
        "relion_coarse_normalized_cc_native_texture_pairs_f32_kernel"
    )
    texture_block = source[texture_start : source.index("cudaError_t", texture_start)]
    assert "relion_coarse_project_texture_f32(" in texture_block
    assert "relion_coarse_score_translate_f32(" in texture_block
    assert "translation_angles[candidate * 2]" in texture_block
    assert "numerator_weight[operand_index]" in texture_block
    assert "half_weights[compact_pixel]" not in texture_block
    assert "packed_pixel += kRelionCoarseDiff2BlockSize" in texture_block
    assert "for (int width = kRelionCoarseDiff2BlockSize / 2" in texture_block
    assert "sqrtf(" in texture_block
    assert "candidate_output[1] = numerator_lanes[0]" in texture_block
    assert "candidate_output[2] = norm_lanes[0]" in texture_block
    assert "atomicAdd(&candidate_output[0], contribution)" in texture_block
    assert "launch_relion_coarse_normalized_cc_native_texture_pairs_f32" in source
    assert "RelionCoarseNormalizedCcNativeTexturePairsF32" in source


def test_relion_coarse_native_texture_source_pins_fused_projection_topology():
    source = (
        Path(__file__).resolve().parents[2]
        / "recovar"
        / "cuda"
        / "cuda_backproject.cu"
    ).read_text()

    start = source.index(
        "relion_coarse_diff2_native_texture_rectangular_f32_kernel"
    )
    block = source[start : source.index("cudaError_t", start)]
    assert "shared_eulers[kRelionCoarseEulersPerBlock * 9]" in block
    assert "relion_coarse_project_texture_f32(" in block
    assert "relion_coarse_score_translate_f32(" in block
    assert "pixel_in_chunk += active_lanes" in block
    assert "atomicAdd(" in block
    assert "const int padded_max_r" in source
    assert "projector_max_r * padding_factor" in source
    assert "for (int64_t batch = 0; batch < batch_size; ++batch)" in source
    assert "image + batch * compact_pixel_count" in source
    assert "weight + batch * compact_pixel_count" in source
    assert "float* particle_output = nullptr" in source
    assert "relion_coarse_diff2_zero_f32_kernel<<<" in source
    assert "relion_coarse_diff2_add_initial_f32_kernel<<<" in source
    assert "particle_output," in source
    assert "output + batch * hypotheses_per_particle" in source
    assert "cudaMemcpyDeviceToDevice" in source
    assert "static_cast<unsigned int>(rotation_blocks)" in source
    assert "fill_relion_half_texture_surface_f32_kernel" in source
    assert "cudaCreateChannelDesc<float2>()" in source
    assert "cudaArraySurfaceLoadStore" in source
    assert "surf3Dwrite(scaled, surface" in source
    assert "tex3D<float2>(" in source
    assert "project_relion_half_texture_f32_kernel" in source
    assert "RelionProjectorHalfTextureF32" in source
    assert "validate_relion_half_texture_geometry(" in source
    assert "max_padded_radius_with_int_square = 46340" in source

    cleanup_start = source.index(
        "cudaError_t synchronize_and_destroy_relion_half_texture_f32("
    )
    cleanup_block = source[
        cleanup_start : source.index(
            "cudaError_t create_relion_half_texture_f32(", cleanup_start
        )
    ]
    assert "cudaStreamSynchronize(stream)" in cleanup_block
    assert "if (err == cudaSuccess) err = sync_err" in cleanup_block
    assert "destroy_relion_half_texture_f32(texture)" in cleanup_block
    assert "if (err == cudaSuccess) err = cleanup_err" in cleanup_block

    create_start = source.index("cudaError_t create_relion_half_texture_f32(")
    create_block = source[
        create_start : source.index("template <bool HALF_IMG>", create_start)
    ]
    assert create_block.count("synchronize_and_destroy_relion_half_texture_f32(") == 2
    assert "cudaGetLastError()" in create_block

    launcher_bounds = (
        (
            "launch_relion_projector_half_texture_f32",
            "/* Match RELION's Wavg A2 topology",
        ),
        (
            "launch_relion_coarse_diff2_native_texture_rectangular_f32",
            "launch_relion_coarse_normalized_cc_native_texture_pairs_f32",
        ),
        (
            "launch_relion_coarse_normalized_cc_native_texture_pairs_f32",
            "__global__ __launch_bounds__(kRelionFineDiff2BlockSize)",
        ),
        (
            "launch_relion_fine_diff2_native_texture_rectangular_f32",
            "__global__ void relion_normalize_f32_kernel",
        ),
    )
    for launcher_name, end_marker in launcher_bounds:
        native_start = source.index(launcher_name)
        native_block = source[native_start : source.index(end_marker, native_start)]
        assert "synchronize_and_destroy_relion_half_texture_f32(" in native_block

    direct_kernel_start = source.index("project_relion_half_texture_f32_kernel(")
    direct_kernel = source[
        direct_kernel_start : source.index(
            "cudaError_t launch_relion_projector_half_texture_f32", direct_kernel_start
        )
    ]
    coarse_projector_start = source.index("relion_coarse_project_texture_f32(")
    coarse_projector = source[
        coarse_projector_start : source.index(
            "/* Bounded normalized-CC replay", coarse_projector_start
        )
    ]
    for projector_block in (direct_kernel, coarse_projector):
        assert "!isfinite(radius_squared)" in projector_block
        assert "radius_squared >= 2147483648.0f" in projector_block
        assert "static_cast<int>(radius_squared)" in projector_block

    assert source.count("padded_image_max_r * padded_image_max_r") >= 5
    assert source.count("projector_max_r < current_size / 2") >= 3
    assert source.count("projector_max_r < image_h / 2") >= 2

    launcher_start = source.index(
        "launch_relion_coarse_diff2_native_texture_rectangular_f32"
    )
    launcher = source[launcher_start : source.index(
        "launch_relion_coarse_normalized_cc_native_texture_pairs_f32",
        launcher_start,
    )]
    assert "create_relion_half_texture_f32(" in launcher
    assert "fill_relion_texture_compact_kernel" not in launcher
    assert "cudaMalloc3DArray(&array_real" not in launcher

    from recovar.em.dense_single_volume.helpers import significance

    significance_source = Path(significance.__file__).read_text()
    # Only the guarded K=1/K-class significance implementation owns this
    # native-texture route.  The legacy helper must not reference its local
    # enable flag (that stale duplicate raised NameError on ordinary callers).
    assert significance_source.count("rotation_block_size = n_rot") == 1
    assert "one particle per " in significance_source
    assert "full orientation grid (%d rotations)" in significance_source
    assert "relion_projector_half_to_texture_full(relion_projector_half[0])" not in (
        significance_source
    )
    assert (
        "coarse_gaussian_projector_half = select_relion_projector_half_for_class("
        in significance_source
    )

    projection_source = (
        Path(significance.__file__).resolve().parent / "projection.py"
    ).read_text()
    texture_start = projection_source.index("def _project_relion_projector_texture(")
    texture_block = projection_source[
        texture_start : projection_source.index(
            "def compute_relion_projector_projections_block(",
            texture_start,
        )
    ]
    assert "relion_projector_half_texture_f32(" in texture_block
    assert "relion_projector_half_to_texture_full(" not in texture_block

    repo_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "scripts/analyze_k1_exact_ppref_coarse_boundary.py",
        "scripts/analyze_k1_coarse_map_score_counterfactual.py",
        "scripts/analyze_k1_exact_ppref_coarse_v1_boundary.py",
        "scripts/analyze_k1_native_texture_fine_counterfactual.py",
    ):
        script_source = (repo_root / relative_path).read_text()
        assert "relion_projector_half_to_texture_full" not in script_source
        assert "projector_scale" in script_source


def test_relion_projector_singleton_selection_is_shape_only_across_handoffs():
    from recovar.em.dense_single_volume import k_class, local_em_engine
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed
    from recovar.em.dense_single_volume.helpers.projection import (
        select_relion_projector_half_for_class,
    )

    projector = np.arange(1 * 5 * 5 * 3, dtype=np.float32).reshape(1, 5, 5, 3).astype(np.complex64)
    selected = select_relion_projector_half_for_class(projector, 0, 1)
    assert selected.shape == (5, 5, 3)
    assert np.shares_memory(selected, projector)
    np.testing.assert_array_equal(selected, projector.reshape(5, 5, 3))

    projector_jax = jnp.asarray(projector)
    stablehlo = str(
        jax.jit(
            lambda value: select_relion_projector_half_for_class(value, 0, 1)
        )
        .lower(projector_jax)
        .compiler_ir(dialect="stablehlo")
    )
    assert "stablehlo.reshape" in stablehlo
    assert "stablehlo.slice" not in stablehlo
    assert "stablehlo.dynamic_slice" not in stablehlo
    with pytest.raises(ValueError, match="before eager device transfer"):
        select_relion_projector_half_for_class(projector_jax, 0, 1)

    two_classes = np.concatenate((projector, projector + 100), axis=0)
    selected_second = select_relion_projector_half_for_class(two_classes, 1, 2)
    assert np.shares_memory(selected_second, two_classes)
    np.testing.assert_array_equal(
        selected_second,
        two_classes[1],
    )

    local_kwargs = k_class._local_engine_kwargs_for_class(
        {"relion_projector_half": projector},
        0,
        1,
    )
    assert local_kwargs["relion_projector_half"].shape == (5, 5, 3)
    assert np.shares_memory(local_kwargs["relion_projector_half"], projector)

    local_source = Path(local_em_engine.__file__).read_text()
    sparse_source = Path(sparse_pass2_bucketed.__file__).read_text()
    k_class_source = Path(k_class.__file__).read_text()
    assert "relion_projector_half_big_jit = relion_projector_half_big_jit[0]" not in local_source
    assert "relion_projector_half = relion_projector_half[0]" not in local_source
    assert "relion_projector_half[class_index]" not in sparse_source
    assert "relion_projector_half = relion_projector_half[None, ...]" not in sparse_source
    assert "projector_half_arr[class_index]" not in k_class_source
    assert (
        "relion_projector_half = jnp.asarray(\n"
        "                _select_projector_half_for_class("
        in sparse_source[
            sparse_source.index("def compute_k_class_pass2_stats_sparse_fused(") :
        ]
    )
    assert local_source.count("_select_projector_half_for_class(") >= 3
    assert sparse_source.count("_select_projector_half_for_class(") >= 6


def test_k1_coarse_gaussian_flag_honors_scoped_default_and_explicit_opt_out(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_GAUSSIAN_FFI", raising=False)
    assert not significance._k1_coarse_gaussian_ffi_enabled()
    assert significance._k1_coarse_gaussian_ffi_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_FFI", "0")
    assert not significance._k1_coarse_gaussian_ffi_enabled(default=True)
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

    k_class_source = (
        Path(significance.__file__).resolve().parent.parent / "k_class.py"
    ).read_text()
    assert 'engine_kwargs.get("preserve_bpref_particle_order", False)' in k_class_source
    assert "relion_coarse_gaussian_default=" in k_class_source


def test_k1_coarse_native_texture_flag_honors_default_and_opt_out(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv(
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE",
        raising=False,
    )
    assert not significance._k1_coarse_gaussian_native_texture_enabled()
    assert significance._k1_coarse_gaussian_native_texture_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE", "1")
    assert significance._k1_coarse_gaussian_native_texture_enabled()
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE", "0")
    assert not significance._k1_coarse_gaussian_native_texture_enabled(default=True)

    source = Path(significance.__file__).read_text()
    start = source.index("coarse_gaussian_native_texture_requested =")
    selection = source[start : source.index("relion_f32_coarse_support_requested", start)]
    assert "_k1_coarse_gaussian_native_texture_enabled(" in selection
    assert "default=False" in selection


def test_k1_coarse_gaussian_exact_operand_flags_honor_default_and_opt_out(monkeypatch):
    from recovar.em.dense_single_volume.helpers import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", raising=False)
    assert not significance._k1_coarse_gaussian_sincosf_enabled()
    assert significance._k1_coarse_gaussian_sincosf_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", "0")
    assert not significance._k1_coarse_gaussian_sincosf_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", "1")
    assert significance._k1_coarse_gaussian_sincosf_enabled()

    monkeypatch.delenv(
        "RECOVAR_K1_COARSE_GAUSSIAN_FINE_TREE_DIAGNOSTIC",
        raising=False,
    )
    assert not significance._k1_coarse_gaussian_fine_tree_enabled()
    monkeypatch.setenv(
        "RECOVAR_K1_COARSE_GAUSSIAN_FINE_TREE_DIAGNOSTIC",
        "1",
    )
    assert significance._k1_coarse_gaussian_fine_tree_enabled()

    monkeypatch.delenv(
        "RECOVAR_K1_COARSE_GAUSSIAN_SERIAL_PARTICLE_LAUNCHES_DIAGNOSTIC",
        raising=False,
    )
    assert not significance._k1_coarse_gaussian_serial_particle_launches_enabled()
    monkeypatch.setenv(
        "RECOVAR_K1_COARSE_GAUSSIAN_SERIAL_PARTICLE_LAUNCHES_DIAGNOSTIC",
        "1",
    )
    assert significance._k1_coarse_gaussian_serial_particle_launches_enabled()

    monkeypatch.delenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", raising=False)
    assert not significance._k1_relion_exact_coarse_operands_enabled()
    assert significance._k1_relion_exact_coarse_operands_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", "0")
    assert not significance._k1_relion_exact_coarse_operands_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", "1")
    assert significance._k1_relion_exact_coarse_operands_enabled()

    source = Path(significance.__file__).read_text()
    assert "coarse_gaussian_sincosf_enabled and not coarse_gaussian_ffi_enabled" in source
    assert "relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled" in source
    assert "production half-image preprocessing path" in source


def test_compact_projection_window_positions_map_full_indices_to_compact_rows():
    from recovar.em.dense_single_volume.helpers.significance import (
        _compact_projection_window_positions,
    )

    compact = np.asarray([20, 21, 25, 26, 10, 11], dtype=np.int32)
    window = np.asarray([10, 20, 26, 11], dtype=np.int32)

    np.testing.assert_array_equal(
        _compact_projection_window_positions(compact, window),
        [4, 0, 3, 5],
    )
    with pytest.raises(ValueError, match="absent from the compact projection"):
        _compact_projection_window_positions(compact, [10, 99])


def test_exact_relion_ctf_source_defaults_to_dataset_star(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_exact_ctf_source_star,
    )

    dataset_star = tmp_path / "particles.star"
    explicit_star = tmp_path / "override.star"
    monkeypatch.delenv("RECOVAR_K1_RELION_EXACT_CTF_STAR", raising=False)
    assert _relion_exact_ctf_source_star(
        SimpleNamespace(particles_file=str(dataset_star)),
    ) == dataset_star.resolve()

    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_CTF_STAR", str(explicit_star))
    assert _relion_exact_ctf_source_star(
        SimpleNamespace(particles_file=str(dataset_star)),
    ) == explicit_star.resolve()

    monkeypatch.delenv("RECOVAR_K1_RELION_EXACT_CTF_STAR", raising=False)
    with pytest.raises(ValueError, match="STAR-backed dataset"):
        _relion_exact_ctf_source_star(SimpleNamespace(particles_file="particles.mrcs"))


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
def test_relion_half_texture_projection_uses_native_rotated_image_radius_cutoff(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The rounded outer shell must use RELION's float32/int cutoff."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    projector_max_r = 23
    projector_size = 2 * projector_max_r + 3
    projector = np.ones(
        (projector_size, projector_size, projector_max_r + 2),
        dtype=np.complex64,
    )
    # These two valid float32 rotations straddle RELION's integer-truncated
    # cutoff for source (ky, kx)=(10, 1), whose exact radius squared is 101.
    # They are frozen from the admitted EMPIAR-10076 K=4 native operand panel.
    rotations = np.asarray(
        [
            [
                [-0.60339195, -0.20407803, -0.7708893],
                [0.3038262, -0.952619, 0.014376025],
                [-0.7372976, -0.22554199, 0.6368069],
            ],
            [
                [0.3366111, 0.59114784, -0.73296463],
                [-0.88786924, 0.45852897, -0.037939373],
                [0.31365776, 0.6635476, 0.6792079],
            ],
        ],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        projected = cuda_backproject.relion_projector_half_texture_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            current_size=20,
            padding_factor=1,
            projector_max_r=projector_max_r,
        )
    projected = np.asarray(projected).reshape(2, 20, 11)
    assert projected[0, 0, 1] != 0
    assert projected[1, 0, 1] == 0


@pytest.mark.gpu
def test_relion_half_texture_projection_matches_legacy_full_staging_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The compact production projector must preserve every projected bit."""

    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
        relion_projector_half_to_texture_full,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(191)
    current_size = 16
    padding_factor = 2
    projector_max_r = 7
    projector_size = 31
    projector_half = (
        rng.normal(0, 0.02, (projector_size, projector_size, 16))
        + 1j * rng.normal(0, 0.02, (projector_size, projector_size, 16))
    ).astype(np.complex64)
    rotations = _off_grid_so3_rotations()
    image_coordinates = np.stack(
        np.meshgrid(
            np.arange(-current_size // 2 + 1, current_size // 2 + 1),
            np.arange(current_size // 2 + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    model_coordinates = np.einsum(
        "rij,pj->rpi",
        rotations[:, :, :2],
        image_coordinates[:, ::-1],
    ) * np.float32(padding_factor)
    assert np.any(model_coordinates[..., 0] < 0)
    assert np.any(model_coordinates[..., 0] > 0)
    assert np.any(np.abs(model_coordinates[..., 2]) > 0.25)
    assert np.any(
        np.abs(model_coordinates[..., 2] - np.rint(model_coordinates[..., 2]))
        > 0.05
    )
    assert np.any(
        np.sum(model_coordinates * model_coordinates, axis=-1)
        > (projector_max_r * padding_factor) ** 2
    )

    with jax.default_device(gpu_device):
        projector_half_jax = jnp.asarray(projector_half)
        rotations_jax = jnp.asarray(rotations)
        projector_full = relion_projector_half_to_texture_full(projector_half_jax)
        legacy = cuda_backproject.project(
            projector_full.reshape(-1),
            rotations_jax,
            image_shape=(current_size, current_size),
            volume_shape=(projector_size,) * 3,
            order=1,
            half_volume=False,
            half_image=True,
            max_r=float(projector_max_r),
            relion_texture_interp=True,
        )
        compact = cuda_backproject.relion_projector_half_texture_f32(
            projector_half_jax,
            rotations_jax,
            current_size=current_size,
            padding_factor=padding_factor,
            projector_max_r=projector_max_r,
        )
        native_scale = np.float32(-(current_size**2))
        legacy_native_scaled = cuda_backproject.project(
            (projector_full * native_scale).reshape(-1),
            rotations_jax,
            image_shape=(current_size, current_size),
            volume_shape=(projector_size,) * 3,
            order=1,
            half_volume=False,
            half_image=True,
            max_r=float(projector_max_r),
            relion_texture_interp=True,
        )
        compact_native_scaled = cuda_backproject.relion_projector_half_texture_f32(
            projector_half_jax,
            rotations_jax,
            current_size=current_size,
            padding_factor=padding_factor,
            projector_max_r=projector_max_r,
            projector_scale=float(native_scale),
        )
        production, production_abs2 = compute_relion_projector_projections_block(
            projector_half_jax,
            rotations_jax,
            (current_size, current_size),
            r_max=projector_max_r,
            padding_factor=padding_factor,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=current_size,
            relion_texture_interp=True,
        )
        legacy_scaled = legacy * np.float32(-(current_size**2))
        legacy_abs2 = jnp.abs(legacy_scaled) ** 2

    np.testing.assert_array_equal(
        np.asarray(compact).view(np.uint32),
        np.asarray(legacy).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_native_scaled).view(np.uint32),
        np.asarray(legacy_native_scaled).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(production).view(np.uint32),
        np.asarray(legacy_scaled).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(production_abs2).view(np.uint32),
        np.asarray(legacy_abs2).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_half_texture_projection_is_bitwise_invariant_to_host_support_crop(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Compacting PPref to every consumed square pixel must preserve bits."""

    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.fourier_window import (
        make_fourier_window_spec,
    )
    from recovar.em.dense_single_volume.helpers.projection import (
        compact_relion_projector_half_for_centered_indices,
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_shape = (16, 16)
    current_size = 6
    padding_factor = 2
    projector_r_max = 7
    padded_r_max = projector_r_max * padding_factor
    projector_size = 2 * (padded_r_max + 1) + 1
    rng = np.random.default_rng(193)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
        + 1j * rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
    ).astype(np.complex64)
    window = make_fourier_window_spec(
        image_shape,
        current_size,
        image_shape[0] * (image_shape[1] // 2 + 1),
        include_recon_window=False,
        score_square=True,
        score_include_dc=True,
    )
    compact, compact_r_max = compact_relion_projector_half_for_centered_indices(
        projector,
        window.score_indices_np,
        image_shape,
        r_max=projector_r_max,
        padding_factor=padding_factor,
    )
    assert compact_r_max == 5
    assert compact.shape == (23, 23, 12)
    rotations = _off_grid_so3_rotations()

    common = dict(
        image_shape=image_shape,
        padding_factor=padding_factor,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=window.score_indices_np,
        relion_texture_interp=True,
    )
    with jax.default_device(gpu_device):
        full_projection, full_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            r_max=projector_r_max,
            **common,
        )
        compact_projection, compact_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(compact),
            jnp.asarray(rotations),
            r_max=compact_r_max,
            **common,
        )

    np.testing.assert_array_equal(
        np.asarray(compact_projection).view(np.uint32),
        np.asarray(full_projection).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(compact_abs2).view(np.uint32),
        np.asarray(full_abs2).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_half_texture_full_even_indexed_projection_matches_full_scatter_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """The full-box Nyquist alias must survive compact indexed projection."""

    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    image_size = 16
    padding_factor = 1
    projector_max_r = image_size // 2
    padded_r_max = projector_max_r * padding_factor
    projector_size = 2 * (padded_r_max + 1) + 1
    rng = np.random.default_rng(194)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
        + 1j
        * rng.normal(0, 0.02, (projector_size, projector_size, padded_r_max + 2))
    ).astype(np.complex64)
    rotations = _off_grid_so3_rotations()
    pixel_indices = np.arange(
        image_size * (image_size // 2 + 1),
        dtype=np.int32,
    )

    common = dict(
        image_shape=(image_size, image_size),
        r_max=projector_max_r,
        padding_factor=padding_factor,
        return_abs2=True,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=image_size,
        relion_texture_interp=True,
    )
    with jax.default_device(gpu_device):
        full_projection, full_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            **common,
        )
        indexed_projection, indexed_abs2 = compute_relion_projector_projections_block(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            pixel_indices=pixel_indices,
            **common,
        )

    np.testing.assert_array_equal(
        np.asarray(indexed_projection).view(np.uint32),
        np.asarray(full_projection).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(indexed_abs2).view(np.uint32),
        np.asarray(full_abs2).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_coarse_native_texture_is_bitwise_batch_context_invariant(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """A particle must score identically alone or inside a larger batch."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(193)
    current_size = 16
    projector_size = 31
    projector_half_width = 16
    batch_size = 3
    rotation_count = 17
    translation_count = 29
    pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, projector_half_width))
        + 1j
        * rng.normal(0, 0.02, (projector_size, projector_size, projector_half_width))
    ).astype(np.complex64)
    rotations = np.repeat(
        np.eye(3, dtype=np.float32)[None, :, :],
        rotation_count,
        axis=0,
    )
    image = (
        rng.normal(0, 0.02, (batch_size, pixel_count))
        + 1j * rng.normal(0, 0.02, (batch_size, pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.uniform(
        -0.5,
        0.5,
        (translation_count, 2),
    ).astype(np.float32)
    weight = rng.uniform(0, 150_000, (batch_size, pixel_count)).astype(
        np.float32
    )
    initial_diff2 = rng.uniform(10_000, 20_000, batch_size).astype(np.float32)
    lookup = np.arange(pixel_count, dtype=np.int32)

    def score(image_arg, weight_arg, initial_arg):
        return cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(image_arg),
            jnp.asarray(translation_angles),
            jnp.asarray(weight_arg),
            jnp.asarray(initial_arg),
            jnp.asarray(lookup),
            current_size,
            2,
            7,
            projector_scale=float(-(current_size**2)),
        )

    with jax.default_device(gpu_device):
        batched = np.asarray(score(image, weight, initial_diff2))
        individual = np.concatenate(
            [
                np.asarray(
                    score(
                        image[index : index + 1],
                        weight[index : index + 1],
                        initial_diff2[index : index + 1],
                    )
                )
                for index in range(batch_size)
            ],
            axis=0,
        )

    np.testing.assert_array_equal(
        batched.view(np.uint32),
        individual.view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_fine_native_texture_is_bitwise_batch_context_invariant(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Native fine scoring must preserve particle-local launch semantics."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(197)
    current_size = 16
    projector_size = 31
    projector_half_width = 16
    batch_size = 3
    rotation_count = 5
    translation_count = 7
    pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0, 0.02, (projector_size, projector_size, projector_half_width))
        + 1j
        * rng.normal(0, 0.02, (projector_size, projector_size, projector_half_width))
    ).astype(np.complex64)
    rotations = np.repeat(
        np.eye(3, dtype=np.float32)[None, None, :, :],
        batch_size * rotation_count,
        axis=0,
    ).reshape(batch_size, rotation_count, 3, 3)
    image = (
        rng.normal(0, 0.02, (batch_size, pixel_count))
        + 1j * rng.normal(0, 0.02, (batch_size, pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.uniform(
        -0.5,
        0.5,
        (translation_count, 2),
    ).astype(np.float32)
    weight = rng.uniform(0, 150_000, (batch_size, pixel_count)).astype(
        np.float32
    )
    initial_diff2 = rng.uniform(10_000, 20_000, batch_size).astype(np.float32)
    lookup = np.arange(pixel_count, dtype=np.int32)

    def score(rotations_arg, image_arg, weight_arg, initial_arg):
        return cuda_backproject.relion_fine_diff2_native_texture_rectangular_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations_arg),
            jnp.asarray(image_arg),
            jnp.asarray(translation_angles),
            jnp.asarray(weight_arg),
            jnp.asarray(initial_arg),
            jnp.asarray(lookup),
            current_size=current_size,
            padding_factor=2,
            projector_max_r=7,
            projector_scale=float(-(current_size**2)),
        )

    with jax.default_device(gpu_device):
        batched = np.asarray(score(rotations, image, weight, initial_diff2))
        individual = np.concatenate(
            [
                np.asarray(
                    score(
                        rotations[index : index + 1],
                        image[index : index + 1],
                        weight[index : index + 1],
                        initial_diff2[index : index + 1],
                    )
                )
                for index in range(batch_size)
            ],
            axis=0,
        )

    np.testing.assert_array_equal(
        batched.view(np.uint32),
        individual.view(np.uint32),
    )


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


@pytest.mark.parametrize(
    "function_name",
    ["relion_fine_diff2_rectangular_f32", "relion_fine_diff2_pairs_f32"],
)
def test_relion_fine_diff2_fails_closed_without_gpu(monkeypatch, function_name):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    function = getattr(cuda_backproject, function_name).__wrapped__
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        function(
            jnp.zeros((1, 1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 1, 2), dtype=jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


def test_relion_fused_translate_fine_diff2_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32.__wrapped__(
            jnp.zeros((1, 1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
        )


def test_relion_coarse_diff2_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_rectangular_f32.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 29, 2), dtype=jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


def test_relion_fused_translate_coarse_diff2_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_fused_translate_rectangular_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
        )


def test_relion_coarse_normalized_cc_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="require a JAX GPU backend"):
        cuda_backproject.relion_coarse_normalized_cc_pairs_f32.__wrapped__(
            jnp.zeros((1, 2, 3), dtype=jnp.complex64),
            jnp.ones((1, 2, 3), dtype=jnp.float32),
            jnp.zeros((1, 2, 3), dtype=jnp.complex64),
            jnp.ones((3,), dtype=jnp.float32),
            jnp.arange(3, dtype=jnp.int32),
        )


def test_relion_coarse_normalized_cc_native_texture_fails_closed_without_gpu(
    monkeypatch,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="require a JAX GPU backend"):
        cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32.__wrapped__(
            jnp.zeros((5, 5, 3), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            1,
            1,
            1,
        )


def test_relion_projector_half_texture_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_projector_half_texture_f32.__wrapped__(
            jnp.zeros((5, 5, 3), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            current_size=2,
            padding_factor=1,
            projector_max_r=1,
        )


def test_relion_coarse_native_texture_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32.__wrapped__(
            jnp.zeros((5, 5, 3), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            1,
            1,
            1,
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
