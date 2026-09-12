"""Focused tests for RELION's fused fine-Gaussian CUDA FFI."""

from decimal import Decimal, localcontext
from itertools import permutations
from pathlib import Path

from helpers.cuda_source import read_cuda_source

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.relion import relion_ctf
from recovar.em.scoring import compact_candidates

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


def _complex_normal_for_test(rng, shape):
    return (
        rng.normal(0.0, 0.02, shape) + 1j * rng.normal(0.0, 0.02, shape)
    ).astype(np.complex64)


def _fine_diff2_update_reference(
    reference,
    shifted,
    weight,
    lane_sum=np.float32(0),
    *,
    prehalf_weight,
):
    """Binary32 oracle for the two mathematically equivalent source orders."""

    diff_real = np.float32(reference.real - shifted.real)
    diff_imag = np.float32(reference.imag - shifted.imag)
    imag_square = np.float32(diff_imag * diff_imag)
    square_sum = _fma32(diff_real, diff_real, imag_square)
    if prehalf_weight:
        staged_weight = np.float32(weight * np.float32(0.5))
        return _fma32(square_sum, staged_weight, lane_sum)
    half_square_sum = np.float32(square_sum * np.float32(0.5))
    return _fma32(half_square_sum, weight, lane_sum)


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
    source = read_cuda_source()

    start = source.index("relion_fine_diff2_update_f32")
    prehalf_start = source.index("relion_fine_diff2_update_prehalf_f32", start)
    block = source[start:prehalf_start]
    prehalf_block = source[prehalf_start : source.index("__global__", prehalf_start)]
    assert "__fsub_rn(reference.x, shifted_image.x)" in block
    assert "__fmul_rn(diff_imag, diff_imag)" in block
    assert "__fmaf_rn(diff_real, diff_real, imag_square)" in block
    assert "__fmul_rn(square_sum, 0.5f)" in block
    assert "__fmaf_rn(half_square_sum, weight, lane_sum)" in block
    assert "__fmaf_rn(square_sum, prehalved_weight, lane_sum)" in prehalf_block
    assert "0.5f" not in prehalf_block
    kernel_start = source.index("void relion_fine_diff2_rectangular_kernel(")
    kernel = source[kernel_start : source.index("__global__", kernel_start)]
    # PR180 shares the body across precision modes; production must still
    # select the explicit float32 update, reduction, and initial-diff2 add.
    assert "if constexpr (std::is_same_v<T, float>)" in kernel
    assert "lane_sum = relion_fine_diff2_update_f32(" in kernel
    assert "lane_sum = relion_fine_diff2_update_f64(" in kernel
    assert "__shared__ T lane_sums[kRelionFineDiff2BlockSize]" in kernel
    assert "if constexpr (ADD_INITIAL)" in kernel
    assert "__fadd_rn(lane_sums[0], initial_diff2[batch])" in kernel
    assert "__dadd_rn(lane_sums[0], initial_diff2[batch])" in kernel


def test_relion_coarse_prehalf_cpu_oracle_pins_equivalent_source_orders():
    rng = np.random.default_rng(71)
    for _ in range(128):
        reference = np.complex64(np.float32(rng.uniform(-2.0, 2.0)) + 1j * np.float32(rng.uniform(-2.0, 2.0)))
        shifted = np.complex64(np.float32(rng.uniform(-2.0, 2.0)) + 1j * np.float32(rng.uniform(-2.0, 2.0)))
        weight = np.float32(rng.uniform(0.25, 16.0))
        lane_sum = np.float32(rng.uniform(0.25, 16.0))
        production = _fine_diff2_update_reference(
            reference,
            shifted,
            weight,
            lane_sum,
            prehalf_weight=False,
        )
        prehalved = _fine_diff2_update_reference(
            reference,
            shifted,
            weight,
            lane_sum,
            prehalf_weight=True,
        )
        assert production.view(np.uint32) == prehalved.view(np.uint32)

    # A normal result built from a subnormal square distinguishes which
    # operand is halved, without relying on an approximate comparison.
    min_subnormal = np.asarray(1, dtype=np.uint32).view(np.float32)[()]
    image = np.complex64(np.sqrt(np.float64(min_subnormal)) + 0j)
    weight = np.float32(2.0**127)
    production = _fine_diff2_update_reference(
        np.complex64(0),
        image,
        weight,
        prehalf_weight=False,
    )
    prehalved = _fine_diff2_update_reference(
        np.complex64(0),
        image,
        weight,
        prehalf_weight=True,
    )
    assert production.view(np.uint32) == np.uint32(0)
    assert prehalved.view(np.uint32) == np.float32(2.0**-23).view(np.uint32)


def test_relion_fused_translate_cuda_source_pins_native_block_topology():
    source = read_cuda_source()

    assert "constexpr int kRelionFineDiff2TranslationCapacity = 7;" in source
    assert "constexpr int kRelionFineDiff2Ref3dJobChunk = 4;" in source
    assert "template <bool FlatRows>" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel<false>" in source
    assert "relion_fine_diff2_fused_translate_rows_f32_kernel<true>" in source
    assert "row_image_ids[row]" in source
    assert "relion_score_translate_f32(" in source
    assert "translation_offset * kRelionFineDiff2BlockSize" in source
    assert "lane_sums[lane_index] = relion_fine_diff2_update_f32(" in source
    assert "initial_diff2[batch]" in source
    assert "runtime_current_size[0]" in source
    assert "logical_full_pixel_count" in source
    assert "RelionFineDiff2FusedTranslateRuntimeFlatRowsF32" in source
    assert "RelionFineDiff2FusedTranslateRuntimeRectangularF32" in source

    pair_start = source.index(
        "relion_fine_diff2_fused_translate_pairs_f32_kernel"
    )
    pair_kernel = source[pair_start : source.index("cudaError_t", pair_start)]
    assert "pair_count + kRelionFineDiff2Ref3dJobChunk - 1" in pair_kernel
    assert "pair_chunk * kRelionFineDiff2Ref3dJobChunk" in pair_kernel
    assert (
        "kRelionFineDiff2BlockSize * kRelionFineDiff2TranslationCapacity"
        in pair_kernel
    )
    assert "image_value = image[image_index]" in pair_kernel
    assert "pixel_weight = weight[image_index]" in pair_kernel

    launch_start = source.index(
        "cudaError_t launch_relion_fine_diff2_fused_translate_pairs_f32"
    )
    launch = source[launch_start : source.index("__global__", launch_start)]
    assert "const int64_t total_blocks = batch_size * pair_chunks;" in launch
    assert "static_cast<unsigned int>(total_blocks)" in launch

    jobs_start = source.index(
        "relion_fine_diff2_fused_translate_jobs_f32_kernel"
    )
    jobs_kernel = source[jobs_start : source.index("cudaError_t", jobs_start)]
    assert "4 * (job_start + offset)" in jobs_kernel
    assert "job_plan[plan_offset + 1]" in jobs_kernel
    assert "job_plan[plan_offset + 3]" in jobs_kernel
    assert "kRelionFineDiff2BlockSize * kJobsPerBlock" in jobs_kernel
    assert "relion_fine_diff2_update_f32(" in jobs_kernel
    assert "initial_diff2[image_rows[offset]]" in jobs_kernel

    jobs_launch_start = source.index(
        "cudaError_t launch_relion_fine_diff2_fused_translate_jobs_f32"
    )
    jobs_launch = source[
        jobs_launch_start : source.index("__global__", jobs_launch_start)
    ]
    assert "job_count + kRelionFineDiff2Ref3dJobChunk - 1" in jobs_launch
    assert "static_cast<unsigned int>(total_blocks)" in jobs_launch


def test_relion_powerclass_cuda_source_pins_native_atomic_topology():
    source = read_cuda_source()

    start = source.index("relion_powerclass_spectrum_highres_f32_kernel")
    block = source[start : source.index("cudaError_t", start)]
    assert "kRelionPowerClassBlockSize = 128" in source
    assert "__float2int_rn(sqrtf(" in block
    assert "atomicAdd(&spectrum[shell], value)" in block
    assert "highres_lanes[tid] += highres_lanes[tid + width]" in block
    assert "atomicAdd(highres_xi2, highres_lanes[0])" in block


def test_relion_coarse_diff2_cuda_source_pins_production_topology():
    source = read_cuda_source()

    start = source.index("relion_coarse_diff2_rotation_block_f32")
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


    # Both original callers and the two runtime-size variants share exactly
    # one production reduction call each; none duplicates the arithmetic.
    callers = (
        "relion_coarse_diff2_rectangular_f32_kernel",
        "relion_coarse_diff2_rotation_blocks_f32_kernel",
        "relion_coarse_diff2_rectangular_runtime_f32_kernel",
        "relion_coarse_diff2_rotation_blocks_runtime_f32_kernel",
    )
    for caller in callers:
        caller_start = source.index("void " + caller + "(")
        caller_block = source[caller_start : source.index("__global__", caller_start)]
        assert caller_block.count("relion_coarse_diff2_rotation_block_f32(") == 1
    # One definition plus the explicitly enumerated callers above.
    assert block.count("relion_coarse_diff2_rotation_block_f32(") == 1 + len(callers)


def test_relion_coarse_normalized_cc_source_pins_native_tree_and_atomics():
    source = read_cuda_source()

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
    source = read_cuda_source()

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
    assert "output + batch * hypotheses_per_batch" in source
    assert "static_cast<unsigned int>(rotation_blocks)" in source


def test_relion_fused_coarse_projector_source_pins_vdam_support_and_segmentation():
    root = Path(__file__).resolve().parents[2]
    cuda_dir = root / "recovar" / "cuda"
    source = read_cuda_source()
    block = (cuda_dir / "relion_coarse_diff2_projector_body.inc").read_text()

    launcher_start = source.index("launch_relion_coarse_diff2_projector_f32_impl")
    launcher = source[launcher_start : source.index("/* Diagnostic", launcher_start)]
    assert "shared_rotations[EULERS_PER_BLOCK * 6]" in block
    assert "shared_references[" in block
    assert "shared_images[kRelionCoarseDiff2BlockSize]" in block
    assert "shared_weights[kRelionCoarseDiff2BlockSize]" in block
    assert "threadIdx.x / kRelionCoarsePrefetchFraction" in block
    assert "threadIdx.x % kRelionCoarsePrefetchFraction" in block
    assert "pixel_in_chunk * EULERS_PER_BLOCK + local_rotation" in block
    assert "relion_score_translate_f32(" in block
    assert "tex3D<float>(" in block
    assert "projector_scale * tex3D<float>(" in block
    assert "RECOVAR_RELION_COARSE_STAGE_WEIGHT(pixel_weight)" in block
    assert "RECOVAR_RELION_COARSE_DIFF2_UPDATE(" in block
    assert "relion_fine_diff2_update_f32(" not in block
    assert "__fmul_rn(pixel_weight, 0.5f)" not in block
    start = source.index("relion_coarse_diff2_projector_f32_kernel")
    template_start = source.rfind("template <", 0, start)
    default_kernel = source[
        template_start:
        source.index("relion_coarse_diff2_projector_prehalf_f32_kernel", start)
    ]
    prehalf_kernel = source[
        source.index("relion_coarse_diff2_projector_prehalf_f32_kernel", start) :
        source.index("launch_relion_coarse_diff2_projector_f32_variant", start)
    ]
    assert "bool SINGLE_LANE_CANONICAL = false>" in default_kernel
    assert "PREHALF_WEIGHT" not in default_kernel
    assert "#define RECOVAR_RELION_COARSE_STAGE_WEIGHT(pixel_weight)\n" in default_kernel
    assert "relion_fine_diff2_update_f32" in default_kernel
    assert "relion_coarse_diff2_projector_body.inc" in default_kernel
    assert "__fmul_rn(pixel_weight, 0.5f)" in prehalf_kernel
    assert "relion_fine_diff2_update_prehalf_f32" in prehalf_kernel
    assert "relion_coarse_diff2_projector_body.inc" in prehalf_kernel
    assert "prehalved coarse weights require native atomic reduction" in source
    assert "const int score_max_r = min(model_max_r, current_size / 2);" in launcher
    assert "(rotation_count / 128) * 128" in launcher
    assert "SINGLE_LANE_CANONICAL" in launcher
    assert "if constexpr (CAPTURE_LANES)" in block
    assert "if constexpr (CANONICAL_REDUCTION)" in block
    assert "shared_lane_partials[" in block
    assert "CANONICAL_REDUCTION && !SINGLE_LANE_CANONICAL" in block
    assert "if constexpr (SINGLE_LANE_CANONICAL)" in block
    assert "translation = static_cast<int>(threadIdx.x);" in block
    assert "active_thread = threadIdx.x < translation_count;" in block
    assert "SINGLE_LANE_CANONICAL ? 1 : active_lanes" in block
    assert "output[output_index] = __fadd_rn(" in block
    assert "threadIdx.x + lane_index * translation_count" in block
    assert "total = __fadd_rn(" in block
    assert "lane_partials[" in block
    assert "kRelionCoarseDiff2BlockSize +" in block

    # The diagnostic delegates validation to the production handler and
    # specializes the same kernel instead of copying fused scoring math.
    capture_start = source.index("RelionCoarseDiff2ProjectorLanesF32Impl")
    capture = source[
        capture_start : source.index("XLA_FFI_DEFINE_HANDLER_SYMBOL(", capture_start)
    ]
    assert "RelionCoarseDiff2ProjectorF32Impl(" in capture
    assert "launch_relion_coarse_diff2_projector_f32<true>(" in capture
    assert "launch_relion_coarse_diff2_projector_prehalf_f32<true>(" in capture
    assert "prehalf_weight" in capture

    from recovar.em.scoring import significance

    significance_source = Path(significance.__file__).read_text()
    # Both guarded routes in the shared K=1/K-class significance
    # implementation preserve RELION's all-rotation launch: the original
    # native-texture diagnostic and InitialModel's fused Gaussian projector.
    # The legacy helper must not add a third copy (its stale duplicate raised
    # NameError on ordinary callers).
    assert significance_source.count("rotation_block_size = n_rot") == 2
    assert "one particle per " in significance_source
    assert "full orientation grid (%d rotations)" in significance_source


def test_relion_coarse_vdam_multistream_source_reuses_production_math():
    source = read_cuda_source()

    helper_start = source.index("constexpr int kRelionVdamWorkerStreams = 8;")
    helper_end = source.index("cudaError_t report_relion_vdam_driver_error", helper_start)
    helpers = source[helper_start:helper_end]
    assert "cudaStreamCreate(&worker_streams[worker])" in helpers
    assert "cudaStreamWaitEvent(" in helpers
    assert "particle % kRelionVdamWorkerStreams" in helpers
    assert "cudaStreamSynchronize(worker_streams[worker])" in helpers

    mstep_start = source.index("launch_relion_vdam_mstep_fused_projector_x_half(")
    mstep_end = source.index("relion_fine_diff2_update_f32", mstep_start)
    mstep_launcher = source[mstep_start:mstep_end]
    assert "initialize_relion_vdam_worker_streams(" in mstep_launcher
    assert "synchronize_relion_vdam_worker_streams(" in mstep_launcher
    assert "destroy_relion_vdam_worker_streams(" in mstep_launcher

    coarse_start = source.index("launch_relion_coarse_diff2_projector_f32_impl(")
    coarse_end = source.index("relion_coarse_diff2_native_texture", coarse_start)
    coarse_launcher = source[coarse_start:coarse_end]
    assert "initialize_relion_vdam_worker_streams(" in coarse_launcher
    assert "dispatch_relion_vdam_round_robin_workers(" in coarse_launcher
    assert "launch_relion_coarse_diff2_projector_f32_variant<" in coarse_launcher
    assert "PREHALF_WEIGHT" in coarse_launcher
    assert "actual_batch_size" in coarse_launcher
    assert "images + particle * compact_pixel_count" in coarse_launcher
    assert "output + particle * output_stride" in coarse_launcher

    handler_start = source.index("RelionCoarseDiff2ProjectorMultistreamF32Impl(")
    handler_end = source.index("XLA_FFI_DEFINE_HANDLER_SYMBOL(", handler_start)
    handler = source[handler_start:handler_end]
    assert "ValidateRelionCoarseDiff2ProjectorF32Operands(" in handler
    assert "launch_relion_coarse_diff2_projector_f32" in handler
    assert "actual_batch_size.untyped_data()" in handler
    assert "cudaMemcpyDeviceToHost" in handler
    assert "kRelionVdamWorkerStreams" in handler
    assert "canonical_reduction != 1" not in handler
    assert "launch_relion_coarse_diff2_projector_f32<false, false>" in handler
    assert "launch_relion_coarse_diff2_projector_f32<false, true>" in handler
    assert "launch_relion_coarse_diff2_projector_f32<false, true, true>" in handler
    assert "launch_relion_coarse_diff2_projector_prehalf_f32(" in handler
    assert "NativeTexture" not in handler
    assert "ProjectorLanes" not in handler

    binding_start = source.index(
        "RelionCoarseDiff2ProjectorMultistreamF32,",
        handler_end,
    )
    binding_end = source.index(");", binding_start)
    binding = source[binding_start:binding_end]
    assert binding.count(".Arg<ffi::AnyBuffer>()") == 8
    assert 'Attr<int64_t>("actual_batch_size")' not in binding
    assert 'Attr<int64_t>("worker_stream_count")' not in binding
    assert 'Attr<int64_t>("prehalf_weight")' in binding

    from recovar import cuda_backproject
    from recovar.em.scoring import significance

    wrapper_source = Path(cuda_backproject.__file__).read_text()
    wrapper_start = wrapper_source.index(
        "def relion_coarse_diff2_projector_multistream_f32("
    )
    decorator_start = wrapper_source.rfind(
        "@functools.partial(",
        0,
        wrapper_start,
    )
    wrapper_end = wrapper_source.index(
        "def relion_coarse_diff2_projector_lanes_f32(",
        wrapper_start,
    )
    wrapper = wrapper_source[decorator_start:wrapper_end]
    decorator = wrapper_source[decorator_start:wrapper_start]
    assert '"actual_batch_size"' not in decorator
    assert '"prehalf_weight"' in decorator
    assert "if not canonical_reduction:" not in wrapper
    assert "actual_batch_size = jnp.asarray(actual_batch_size)" in wrapper
    assert "_TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32" in wrapper
    assert "native_texture" not in wrapper.lower()
    assert "projector_lanes" not in wrapper.lower()
    assert "prehalf_weight=np.int64(bool(prehalf_weight))" in wrapper

    significance_source = Path(significance.__file__).read_text()
    production_start = significance_source.index(
        "def _score_coarse_fused_full_diff2(",
    )
    production_end = significance_source.index(
        "def _project_coarse_gemm_rows(",
        production_start,
    )
    production = significance_source[production_start:production_end]
    assert "relion_coarse_diff2_projector_multistream_f32" in production
    assert "relion_coarse_diff2_projector_f32" in production
    assert "native_texture" not in production.lower()
    assert "projector_lanes" not in production.lower()
    assert 'coarse_projector_kwargs["prehalf_weight"]' in production
    assert "coarse_prehalf_weight_enabled" in production

    score_block_start = significance_source.index("def _score_block(")
    score_block_end = significance_source.index(
        "if coarse_gaussian_score_backend is _CoarseGaussianScoreBackend.NATIVE_TEXTURE:",
        score_block_start,
    )
    score_block = significance_source[score_block_start:score_block_end]
    assert "return -_score_coarse_fused_full_diff2(" in score_block


def test_relion_coarse_prehalf_api_defaults_are_static_and_forwarded():
    import inspect

    from recovar import cuda_backproject

    wrappers = (
        cuda_backproject.relion_coarse_diff2_projector_f32,
        cuda_backproject.relion_coarse_diff2_projector_multistream_f32,
        cuda_backproject.relion_coarse_diff2_projector_lanes_f32,
    )
    source = Path(cuda_backproject.__file__).read_text()
    wrapper_names = [function.__wrapped__.__name__ for function in wrappers]
    wrapper_starts = [source.index(f"def {name}(") for name in wrapper_names]
    wrapper_ends = wrapper_starts[1:] + [
        source.index(
            "def relion_coarse_diff2_native_texture_rectangular_f32(",
            wrapper_starts[-1],
        )
    ]
    for function, start, end in zip(
        wrappers,
        wrapper_starts,
        wrapper_ends,
        strict=True,
    ):
        signature = inspect.signature(function.__wrapped__)
        assert signature.parameters["prehalf_weight"].default is False
        wrapper = source[start:end]
        decorator = source[source.rfind("@functools.partial(", 0, start) : start]
        assert '"prehalf_weight"' in decorator
        assert "prehalf_weight=np.int64(bool(prehalf_weight))" in wrapper

    cuda_source = read_cuda_source()
    handler_names = (
        "RelionCoarseDiff2ProjectorF32,",
        "RelionCoarseDiff2ProjectorMultistreamF32,",
        "RelionCoarseDiff2ProjectorLanesF32,",
    )
    for handler_name in handler_names:
        start = cuda_source.index(handler_name)
        binding = cuda_source[start : cuda_source.index(");", start)]
        assert binding.count('Attr<int64_t>("prehalf_weight")') == 1


def test_relion_coarse_prehalf_shared_body_is_built_packaged_and_stale_checked():
    root = Path(__file__).resolve().parents[2]
    include_name = "relion_coarse_diff2_projector_body.inc"
    makefile = (root / "recovar" / "cuda" / "Makefile").read_text()
    manifest = (root / "MANIFEST.in").read_text()

    build_inputs = ("cuda_backproject.cu", include_name, "noise_residual.cuh", "relion_vdam_mstep.cuh", "relion_scoring.cuh")
    library_rule = next(line for line in makefile.splitlines() if line.startswith("$(LIB):"))
    prerequisites, order_only = library_rule.split(":", 1)[1].split("|", 1)
    assert set(prerequisites.split()) == set(build_inputs)
    assert order_only.split() == ["check-nvcc"]
    for source_name in build_inputs:
        assert f"include recovar/cuda/{source_name}" in manifest

    from recovar import cuda_backproject

    assert cuda_backproject._CUDA_BUILD_SOURCE_NAMES == (
        "noise_residual.cuh",
        "relion_vdam_mstep.cuh",
        "relion_scoring.cuh",
        "cuda_backproject.cu",
        include_name,
        "Makefile",
    )
    python_source = Path(cuda_backproject.__file__).read_text()
    assert "for src_name in _CUDA_BUILD_SOURCE_NAMES:" in python_source


def test_k1_coarse_multistream_workers_are_default_off_and_fail_closed(monkeypatch):
    from recovar.em.scoring import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_MULTISTREAM_WORKERS", raising=False)
    assert significance._k1_coarse_multistream_worker_count() == 0
    assert significance._k1_coarse_multistream_worker_count(default=8) == 8
    monkeypatch.setenv("RECOVAR_K1_COARSE_MULTISTREAM_WORKERS", "8")
    assert significance._k1_coarse_multistream_worker_count() == 8
    for invalid in ("1", "7", "9", "invalid"):
        monkeypatch.setenv("RECOVAR_K1_COARSE_MULTISTREAM_WORKERS", invalid)
        with pytest.raises(ValueError, match="must be 0 or 8"):
            significance._k1_coarse_multistream_worker_count()


def test_k1_coarse_native_atomic_reduction_is_default_off_and_fail_closed(
    monkeypatch,
):
    from recovar.em.scoring import significance

    variable = "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._k1_coarse_native_atomic_reduction_enabled()
    assert significance._k1_coarse_native_atomic_reduction_enabled(default=True)
    monkeypatch.setenv(variable, "1")
    assert significance._k1_coarse_native_atomic_reduction_enabled()
    monkeypatch.setenv(variable, "0")
    assert not significance._k1_coarse_native_atomic_reduction_enabled(default=True)
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(ValueError, match=variable):
        significance._k1_coarse_native_atomic_reduction_enabled()


def test_k1_coarse_prehalf_weight_is_default_off_and_fail_closed(monkeypatch):
    from recovar.em.scoring import significance

    variable = "RECOVAR_K1_COARSE_PREHALF_WEIGHT"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._k1_coarse_prehalf_weight_enabled()
    assert significance._k1_coarse_prehalf_weight_enabled(default=True)
    monkeypatch.setenv(variable, "1")
    assert significance._k1_coarse_prehalf_weight_enabled()
    monkeypatch.setenv(variable, "0")
    assert not significance._k1_coarse_prehalf_weight_enabled(default=True)
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(ValueError, match=variable):
        significance._k1_coarse_prehalf_weight_enabled()


@pytest.mark.parametrize(
    ("translation_count", "expected"),
    [(28, False), (29, True), (30, False), (116, False)],
)
def test_k1_coarse_native_atomic_selection_is_scoped_to_live_t29_gate(
    translation_count,
    expected,
):
    from recovar.em.scoring import significance

    assert (
        significance._k1_coarse_native_atomic_reduction_selected(
            requested=True,
            score_mode="gaussian",
            translation_count=translation_count,
        )
        is expected
    )
    assert not significance._k1_coarse_native_atomic_reduction_selected(
        requested=True,
        score_mode="normalized_cc",
        translation_count=translation_count,
    )
    assert not significance._k1_coarse_native_atomic_reduction_selected(
        requested=False,
        score_mode="gaussian",
        translation_count=translation_count,
    )


def test_k1_coarse_single_lane_canonical_is_default_off_and_fail_closed(
    monkeypatch,
):
    from recovar.em.scoring import significance

    variable = "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._k1_coarse_single_lane_canonical_enabled()
    assert significance._k1_coarse_single_lane_canonical_enabled(default=True)
    monkeypatch.setenv(variable, "1")
    assert significance._k1_coarse_single_lane_canonical_enabled()
    monkeypatch.setenv(variable, "0")
    assert not significance._k1_coarse_single_lane_canonical_enabled(default=True)
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(ValueError, match=variable):
        significance._k1_coarse_single_lane_canonical_enabled()


@pytest.mark.parametrize(
    ("translation_count", "expected"),
    [(29, False), (64, False), (65, True), (116, True), (128, True), (129, False)],
)
def test_k1_coarse_single_lane_selection_falls_back_outside_one_lane_range(
    translation_count,
    expected,
):
    from recovar.em.scoring import significance

    assert (
        significance._k1_coarse_single_lane_canonical_selected(
            requested=True,
            score_mode="gaussian",
            translation_count=translation_count,
        )
        is expected
    )
    assert not significance._k1_coarse_single_lane_canonical_selected(
        requested=True,
        score_mode="normalized_cc",
        translation_count=translation_count,
    )
    assert not significance._k1_coarse_single_lane_canonical_selected(
        requested=False,
        score_mode="gaussian",
        translation_count=translation_count,
    )


@pytest.mark.parametrize("translation_count", [1, 64, 129])
def test_relion_coarse_single_lane_canonical_rejects_unsupported_counts(
    translation_count,
):
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(ValueError, match="requires 65--128 translations"):
        cuda_backproject.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((translation_count, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
            canonical_reduction=True,
            single_lane_canonical=True,
        )


def test_relion_coarse_single_lane_canonical_requires_canonical_reduction():
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(ValueError, match="requires canonical_reduction=True"):
        cuda_backproject.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((116, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
            canonical_reduction=False,
            single_lane_canonical=True,
        )


@pytest.mark.parametrize(
    ("canonical_reduction", "single_lane_canonical"),
    [(True, False), (True, True)],
)
def test_relion_coarse_prehalf_rejects_non_atomic_reductions(
    canonical_reduction,
    single_lane_canonical,
):
    import recovar.cuda_backproject as cuda_backproject

    operands = (
        jnp.zeros((5, 5, 5), dtype=jnp.complex64),
        jnp.eye(3, dtype=jnp.float32)[None, :, :],
        jnp.zeros((1, 1), dtype=jnp.complex64),
        jnp.zeros((116, 2), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.asarray([0], dtype=jnp.int32),
    )
    shared_kwargs = dict(
        current_size=1,
        physical_image_size=1,
        model_max_r=1,
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
        prehalf_weight=True,
    )
    with pytest.raises(ValueError, match="prehalf_weight=True requires"):
        cuda_backproject.relion_coarse_diff2_projector_f32.__wrapped__(
            *operands,
            **shared_kwargs,
        )
    with pytest.raises(ValueError, match="prehalf_weight=True requires"):
        cuda_backproject.relion_coarse_diff2_projector_multistream_f32.__wrapped__(
            *operands,
            actual_batch_size=jnp.asarray(1, dtype=jnp.int32),
            **shared_kwargs,
        )


def test_relion_coarse_multistream_reduction_mode_is_one_static_trace(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    trace_count = 0

    def fake_prepare(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        **kwargs,
    ):
        del projector_full, weight, initial_diff2, full_to_compact, kwargs
        nonlocal trace_count
        trace_count += 1
        compact_rotations = jnp.zeros(
            (rotation_matrices.shape[0], 6),
            dtype=jnp.float32,
        )
        out_type = jax.ShapeDtypeStruct(
            (
                images.shape[0],
                rotation_matrices.shape[0],
                translation_angles.shape[0],
            ),
            jnp.float32,
        )
        return compact_rotations, out_type

    seen_modes = set()

    def fake_ffi_call(target, out_type, **kwargs):
        del kwargs
        assert target == (
            cuda_backproject._TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32
        )

        def invoke(*operands, **attrs):
            seen_modes.add(
                (
                    int(attrs["canonical_reduction"]),
                    int(attrs["prehalf_weight"]),
                )
            )
            actual_batch_size = operands[-1]
            return jnp.zeros(out_type.shape, dtype=out_type.dtype) + (
                actual_batch_size.astype(out_type.dtype) * 0
            )

        return invoke

    monkeypatch.setattr(
        cuda_backproject,
        "_prepare_relion_coarse_diff2_projector_f32",
        fake_prepare,
    )
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    function = cuda_backproject.relion_coarse_diff2_projector_multistream_f32
    function.clear_cache()
    operands = (
        jnp.zeros((5, 5, 5), dtype=jnp.complex64),
        jnp.eye(3, dtype=jnp.float32)[None, :, :],
        jnp.zeros((4, 1), dtype=jnp.complex64),
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.asarray([0], dtype=jnp.int32),
    )
    try:
        for canonical_reduction, prehalf_weight in (
            (True, False),
            (False, False),
            (False, True),
        ):
            for actual_batch_size in (3, 4, 1):
                result = function(
                    *operands,
                    current_size=1,
                    physical_image_size=1,
                    model_max_r=1,
                    actual_batch_size=jnp.asarray(
                        actual_batch_size,
                        dtype=jnp.int32,
                    ),
                    canonical_reduction=canonical_reduction,
                    prehalf_weight=prehalf_weight,
                )
                assert result.shape == (4, 1, 1)
        assert trace_count == 3
        assert function._cache_size() == 3
        assert seen_modes == {(1, 0), (0, 0), (0, 1)}
    finally:
        function.clear_cache()


def test_k1_coarse_gaussian_flag_honors_scoped_default_and_explicit_opt_out(monkeypatch):
    from recovar.em.scoring import significance

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
    assert "coarse_gaussian_projector_full_by_class" in guard
    assert "relion_projector_half[class_index]" in guard
    assert "square_score_indices_np" in guard
    assert "coarse_gaussian_square_layout = _plan_coarse_gaussian_square_layout(" in guard
    planner_start = source.index("def _plan_coarse_gaussian_square_layout(")
    planner = source[planner_start : source.index("\ndef ", planner_start + 1)]
    for assignment in ("logical_indices, logical_count", "physical_indices, physical_count"):
        window_call = planner.split(assignment + " = make_fourier_window_indices_np(", 1)[1].split(")", 1)[0]
        assert "square=True" in window_call
        assert "include_dc=True" in window_call

    k_class_source = (
        Path(significance.__file__).resolve().parent.parent / "classification" / "k_class.py"
    ).read_text()
    assert 'engine_kwargs.get("preserve_bpref_particle_order", False)' in k_class_source
    assert "relion_coarse_gaussian_default=" in k_class_source


def test_k1_coarse_native_texture_flag_honors_default_and_opt_out(monkeypatch):
    from recovar.em.scoring import significance

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
    from recovar.em.scoring import significance

    monkeypatch.delenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", raising=False)
    assert not significance._k1_coarse_gaussian_sincosf_enabled()
    assert significance._k1_coarse_gaussian_sincosf_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", "0")
    assert not significance._k1_coarse_gaussian_sincosf_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF", "1")
    assert significance._k1_coarse_gaussian_sincosf_enabled()

    monkeypatch.delenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", raising=False)
    assert not significance._k1_relion_exact_coarse_operands_enabled()
    assert significance._k1_relion_exact_coarse_operands_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", "0")
    assert not significance._k1_relion_exact_coarse_operands_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS", "1")
    assert significance._k1_relion_exact_coarse_operands_enabled()

    monkeypatch.delenv("RECOVAR_K1_COARSE_FUSED_PROJECTOR", raising=False)
    assert not significance._k1_coarse_fused_projector_enabled()
    assert significance._k1_coarse_fused_projector_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_FUSED_PROJECTOR", "0")
    assert not significance._k1_coarse_fused_projector_enabled(default=True)
    monkeypatch.setenv("RECOVAR_K1_COARSE_FUSED_PROJECTOR", "1")
    assert significance._k1_coarse_fused_projector_enabled()
    assert significance._k1_coarse_fused_projector_supports_padding(1)
    assert not significance._k1_coarse_fused_projector_supports_padding(2)

    monkeypatch.delenv("RECOVAR_RELION_COARSE_CANONICAL_REDUCTION", raising=False)
    assert not significance._relion_coarse_canonical_reduction_enabled()
    assert significance._relion_coarse_canonical_reduction_enabled(default=True)
    monkeypatch.setenv("RECOVAR_RELION_COARSE_CANONICAL_REDUCTION", "1")
    assert significance._relion_coarse_canonical_reduction_enabled()
    monkeypatch.setenv("RECOVAR_RELION_COARSE_CANONICAL_REDUCTION", "0")
    assert not significance._relion_coarse_canonical_reduction_enabled(default=True)

    source = Path(significance.__file__).read_text()
    assert "coarse_gaussian_sincosf_enabled and not coarse_gaussian_ffi_enabled" in source
    assert "relion_coarse_gaussian_default and coarse_gaussian_ffi_enabled" in source
    assert "return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled" in source


    assert "relion_coarse_gaussian_default\n                and coarse_fused_projector_enabled" in source
    # PR179's NumPy preprocessing now supplies the same unshifted operands;
    # the exact-source specialization still requires the CUDA image path.
    numpy_start = source.index("elif use_relion_numpy_preprocess and not relion_cuda_preprocess:")
    numpy_branch = source[numpy_start : source.index("\n        else:", numpy_start)]
    assert "return_unshifted_score_weighted=coarse_gaussian_sincosf_enabled" in numpy_branch
    assert "coarse_gaussian_unshifted_score_weighted" in numpy_branch
    assert "if not relion_cuda_preprocess or relion_preprocess_kwargs is None:" in source
    assert "RELION CUDA image preprocessing" in source
    assert "processed_direct = _process_relion_exact_coarse_half_image(" in source
    assert "exact_operands = _assemble_relion_exact_coarse_gaussian_operands(" in source
    from recovar.em.relion import relion_coarse_operands

    operands_source = Path(relion_coarse_operands.__file__).read_text()
    assembler_start = operands_source.index("def _assemble_relion_exact_coarse_gaussian_operands(")
    assembler = operands_source[assembler_start : operands_source.index("\ndef ", assembler_start + 1)]
    assert "_relion_exact_ctf_half_from_source_star_host(" in assembler
    assert "processed_score * pixel_correction" in assembler
    assert "pixel_indices=score_indices_np" in assembler
    assert "shifted_corrected = translate_fn(" in assembler
    assert "else cuda_backproject.relion_coarse_diff2_projector_f32" in source
    assert "return coarse_projector(" in source
    assert "rotation_block_size = n_rot" in source


def test_compact_projection_window_positions_map_full_indices_to_compact_rows():
    from recovar.em.scoring.significance import _compact_projection_window_positions

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

    from recovar.em.relion.relion_ctf import _relion_exact_ctf_source_star

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


def test_exact_relion_ctf_source_exposes_host_and_shared_device_boundaries(
    monkeypatch,
    tmp_path,
):
    from types import SimpleNamespace

    class Rows:
        def __init__(self, rows):
            self.rows = rows
            self.iloc = self

        def __getitem__(self, index):
            return self.rows[index]

    class RelionBinding:
        @staticmethod
        def get_ctf_image(*_args):
            return np.arange(12, dtype=np.float64).reshape(4, 3)

    source = (tmp_path / "particles.star").resolve()
    cache_key = (str(source), (4, 4))
    particle = {
        "rlnOpticsGroup": 1,
        "rlnDefocusU": 10000.0,
        "rlnDefocusV": 11000.0,
        "rlnDefocusAngle": 12.0,
        "rlnPhaseShift": 3.0,
    }
    optics = {
        "rlnVoltage": 300.0,
        "rlnSphericalAberration": 2.7,
        "rlnAmplitudeContrast": 0.1,
        "rlnImagePixelSize": 1.5,
    }
    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_source_star",
        lambda _dataset: source,
    )
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE,
        cache_key,
        {
            "particles": Rows([particle]),
            "optics": {1: optics},
            "relion_bind": RelionBinding(),
            "images": {},
        },
    )
    dataset = SimpleNamespace(
        original_image_indices_from_local=lambda indices: np.asarray(indices),
    )

    pixel_indices = np.asarray([11, 0, 4, 4], dtype=np.int32)
    compact_result = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset,
        np.asarray([0, 0], dtype=np.int32),
        (4, 4),
        pixel_indices=pixel_indices,
    )
    host_result = relion_ctf._relion_exact_ctf_half_from_source_star_host(
        dataset,
        np.asarray([0], dtype=np.int32),
        (4, 4),
    )
    device_result = relion_ctf._relion_exact_ctf_half_from_source_star(
        dataset,
        np.asarray([0], dtype=np.int32),
        (4, 4),
    )

    assert type(host_result) is np.ndarray
    assert host_result.dtype == np.float64
    assert isinstance(device_result, jax.Array)
    assert device_result.dtype == jnp.float64
    np.testing.assert_array_equal(
        host_result[0],
        -np.fft.fftshift(np.arange(12, dtype=np.float64).reshape(4, 3), axes=0).reshape(-1),
    )
    np.testing.assert_array_equal(np.asarray(device_result), host_result)

    assert compact_result.dtype == np.float64
    np.testing.assert_array_equal(compact_result, host_result[[0, 0]][:, pixel_indices])
    # A compact result must not expose writable aliases of the cached full CTF.
    compact_result[:] = 99.0
    np.testing.assert_array_equal(
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset,
            np.asarray([0], dtype=np.int32),
            (4, 4),
        ),
        host_result,
    )


def test_coarse_gaussian_square_operands_reuse_weighted_score_inputs():
    from recovar.em.relion.relion_coarse_operands import _relion_coarse_gaussian_square_operands

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
    from recovar.em.relion.relion_coarse_operands import _relion_coarse_gaussian_square_operands_sincosf

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
    from recovar.em.relion.relion_coarse_operands import _relion_coarse_gaussian_square_operands_sincosf

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
    from recovar.em.relion.relion_coarse_operands import _relion_coarse_gaussian_square_operands_sincosf
    from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32

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
def test_relion_coarse_diff2_rotation_blocks_matches_atomic_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(41)
    batch_size, rotation_count, translation_count = 2, 17, 29
    compact_pixel_count, full_pixel_count = 67, 83
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
    rotation_block_ids = np.asarray(
        [[1, 0, -1, 99, -2], [0, 1, -1, -2, 99]],
        dtype=np.int32,
    )

    with jax.default_device(gpu_device):
        actual = np.asarray(
            cuda_backproject.relion_coarse_diff2_rotation_blocks_f32(
                jnp.asarray(reference),
                jnp.asarray(shifted),
                jnp.asarray(weight),
                jnp.asarray(initial_diff2),
                jnp.asarray(rotation_block_ids),
                jnp.asarray(lookup),
            )
        )

    assert actual.shape == (batch_size, 5, 16, translation_count)
    assert np.all(np.isposinf(actual[:, 2]))
    assert np.all(np.isnan(actual[:, 3:]))
    for batch, source_blocks in enumerate(((1, 0), (0, 1))):
        for selected_block, source_block in enumerate(source_blocks):
            for rotation_offset in range(16):
                rotation = source_block * 16 + rotation_offset
                if rotation >= rotation_count:
                    assert np.all(
                        np.isposinf(actual[batch, selected_block, rotation_offset])
                    )
                    continue
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
                        actual[
                            batch,
                            selected_block,
                            rotation_offset,
                            translation,
                        ].view(np.uint32)
                    )
                    assert actual_bits in possible


@pytest.mark.gpu
def test_relion_coarse_vdam_projector_lane_capture_matches_atomic_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(46)
    current_size = 8
    model_max_r = 1
    projector_size = 2 * model_max_r + 3
    rotation_count = 128
    translation_count = 29
    compact_pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0.0, 0.02, (projector_size,) * 3)
        + 1j * rng.normal(0.0, 0.02, (projector_size,) * 3)
    ).astype(np.complex64)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], rotation_count, axis=0)
    images = (
        rng.normal(0.0, 0.02, (1, compact_pixel_count))
        + 1j * rng.normal(0.0, 0.02, (1, compact_pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.uniform(-0.2, 0.2, (translation_count, 2)).astype(
        np.float32
    )
    weight = rng.uniform(0.1, 3.0, images.shape).astype(np.float32)
    initial_diff2 = np.asarray([10.0], dtype=np.float32)
    lookup = np.arange(compact_pixel_count, dtype=np.int32)

    with jax.default_device(gpu_device):
        captured, lanes = cuda_backproject.relion_coarse_diff2_projector_lanes_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
        )
        production = cuda_backproject.relion_coarse_diff2_projector_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
        )
        canonical = cuda_backproject.relion_coarse_diff2_projector_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
            canonical_reduction=True,
        )

    captured_np = np.asarray(captured)
    production_np = np.asarray(production)
    canonical_np = np.asarray(canonical)
    lanes_np = np.asarray(lanes)
    assert lanes_np.shape == (1, rotation_count, 128)
    np.testing.assert_array_equal(
        lanes_np[:, :, 116:].view(np.uint32),
        np.zeros((1, rotation_count, 12), dtype=np.uint32),
    )

    for rotation in (0, 100, 101, 127):
        for translation in range(translation_count):
            thread_ids = translation + np.arange(4) * translation_count
            partials = lanes_np[0, rotation, thread_ids]
            possible = set()
            for order in permutations(range(4)):
                total = initial_diff2[0]
                for lane in order:
                    total = np.add(total, partials[lane], dtype=np.float32)
                possible.add(int(total.view(np.uint32)))
            assert int(captured_np[0, rotation, translation].view(np.uint32)) in possible
            assert int(production_np[0, rotation, translation].view(np.uint32)) in possible
            canonical_total = initial_diff2[0]
            for lane in range(4):
                canonical_total = np.add(
                    canonical_total,
                    partials[lane],
                    dtype=np.float32,
                )
            assert canonical_np[0, rotation, translation] == canonical_total


@pytest.mark.gpu
def test_relion_coarse_vdam_prehalf_source_order_across_dispatchers(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    current_size = 1
    model_max_r = 1
    projector_size = 2 * model_max_r + 3
    translation_count = 29
    projector = np.zeros((projector_size,) * 3, dtype=np.complex64)
    rotations = np.eye(3, dtype=np.float32)[None, :, :]
    min_subnormal = np.asarray(1, dtype=np.uint32).view(np.float32)[()]
    image_value = np.float32(np.sqrt(np.float64(min_subnormal)))
    active_images = np.asarray([[image_value + 0j]], dtype=np.complex64)
    active_weight = np.asarray([[np.float32(2.0**127)]], dtype=np.float32)
    active_initial = np.zeros((1,), dtype=np.float32)
    translation_angles = np.zeros((translation_count, 2), dtype=np.float32)
    lookup = np.asarray([0], dtype=np.int32)
    padding_initial = np.asarray([-12345.75], dtype=np.float32)
    physical_images = np.concatenate([active_images, np.full((1, 1), np.complex64(np.nan + 1j * np.nan))])
    physical_weight = np.concatenate([active_weight, np.full((1, 1), np.nan, dtype=np.float32)])
    physical_initial = np.concatenate([active_initial, padding_initial])

    shared_operands = (
        jnp.asarray(projector),
        jnp.asarray(rotations),
        jnp.asarray(active_images),
        jnp.asarray(translation_angles),
        jnp.asarray(active_weight),
        jnp.asarray(active_initial),
        jnp.asarray(lookup),
    )
    shared_kwargs = dict(
        current_size=current_size,
        physical_image_size=current_size,
        model_max_r=model_max_r,
    )
    with jax.default_device(gpu_device):
        production = cuda_backproject.relion_coarse_diff2_projector_f32(
            *shared_operands,
            **shared_kwargs,
        )
        explicit_production = cuda_backproject.relion_coarse_diff2_projector_f32(
            *shared_operands,
            prehalf_weight=False,
            **shared_kwargs,
        )
        prehalved = cuda_backproject.relion_coarse_diff2_projector_f32(
            *shared_operands,
            prehalf_weight=True,
            **shared_kwargs,
        )
        captured, lanes = cuda_backproject.relion_coarse_diff2_projector_lanes_f32(
            *shared_operands,
            prehalf_weight=True,
            **shared_kwargs,
        )
        dispatched = cuda_backproject.relion_coarse_diff2_projector_multistream_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(physical_images),
            jnp.asarray(translation_angles),
            jnp.asarray(physical_weight),
            jnp.asarray(physical_initial),
            jnp.asarray(lookup),
            actual_batch_size=jnp.asarray(1, dtype=jnp.int32),
            canonical_reduction=False,
            prehalf_weight=True,
            **shared_kwargs,
        )

    production_np = np.asarray(production)
    explicit_production_np = np.asarray(explicit_production)
    prehalved_np = np.asarray(prehalved)
    captured_np = np.asarray(captured)
    lanes_np = np.asarray(lanes)
    dispatched_np = np.asarray(dispatched)
    expected = np.float32(2.0**-23)
    expected_active = np.full(
        (1, 1, translation_count),
        expected,
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        production_np.view(np.uint32),
        explicit_production_np.view(np.uint32),
    )
    np.testing.assert_array_equal(
        production_np.view(np.uint32),
        np.zeros(production_np.shape, dtype=np.uint32),
    )
    for output in (prehalved_np, captured_np, dispatched_np[:1]):
        np.testing.assert_array_equal(
            output.view(np.uint32),
            expected_active.view(np.uint32),
        )
    np.testing.assert_array_equal(
        lanes_np[0, 0, :translation_count].view(np.uint32),
        np.full(translation_count, expected, dtype=np.float32).view(np.uint32),
    )
    np.testing.assert_array_equal(
        lanes_np[0, 0, translation_count:].view(np.uint32),
        np.zeros(128 - translation_count, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        dispatched_np[1:].view(np.uint32),
        np.broadcast_to(
            padding_initial[:, None, None],
            (1, 1, translation_count),
        ).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_coarse_vdam_multistream_atomic_stays_in_lane_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(53)
    current_size = 8
    model_max_r = 1
    projector_size = 2 * model_max_r + 3
    actual_batch_size = 2
    physical_batch_size = 4
    rotation_count = 129
    translation_count = 29
    compact_pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0.0, 0.02, (projector_size,) * 3)
        + 1j * rng.normal(0.0, 0.02, (projector_size,) * 3)
    ).astype(np.complex64)
    angles = np.linspace(-np.pi, np.pi, rotation_count, endpoint=False)
    rotations = np.zeros((rotation_count, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.cos(angles)
    rotations[:, 0, 1] = -np.sin(angles)
    rotations[:, 1, 0] = np.sin(angles)
    rotations[:, 1, 1] = np.cos(angles)
    rotations[:, 2, 2] = 1.0
    active_images = (
        rng.normal(0.0, 0.02, (actual_batch_size, compact_pixel_count))
        + 1j * rng.normal(0.0, 0.02, (actual_batch_size, compact_pixel_count))
    ).astype(np.complex64)
    active_weight = rng.uniform(
        0.1,
        3.0,
        (actual_batch_size, compact_pixel_count),
    ).astype(np.float32)
    active_initial = rng.uniform(5.0, 15.0, actual_batch_size).astype(np.float32)
    padding_initial = np.asarray([-12345.75, -0.0], dtype=np.float32)
    padding_count = physical_batch_size - actual_batch_size
    images = np.concatenate(
        [
            active_images,
            np.full(
                (padding_count, compact_pixel_count),
                np.complex64(np.nan + 1j * np.nan),
            ),
        ],
        axis=0,
    )
    weight = np.concatenate(
        [
            active_weight,
            np.full(
                (padding_count, compact_pixel_count),
                np.nan,
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    initial_diff2 = np.concatenate([active_initial, padding_initial])
    translation_angles = rng.uniform(
        -0.2,
        0.2,
        (translation_count, 2),
    ).astype(np.float32)
    lookup = np.arange(compact_pixel_count, dtype=np.int32)

    def multistream():
        return cuda_backproject.relion_coarse_diff2_projector_multistream_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
            actual_batch_size=jnp.asarray(actual_batch_size, dtype=jnp.int32),
            canonical_reduction=False,
        )

    with jax.default_device(gpu_device):
        serial, lanes = cuda_backproject.relion_coarse_diff2_projector_lanes_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(active_images),
            jnp.asarray(translation_angles),
            jnp.asarray(active_weight),
            jnp.asarray(active_initial),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
        )
        dispatched = multistream()
        repeated = multistream()

        serial_np = np.asarray(serial)
        lanes_np = np.asarray(lanes)
        dispatched_np = np.asarray(dispatched)
        repeated_np = np.asarray(repeated)

    assert lanes_np.shape == (actual_batch_size, rotation_count, 128)
    np.testing.assert_array_equal(
        lanes_np[:, :, 116:].view(np.uint32),
        np.zeros(
            (actual_batch_size, rotation_count, 12),
            dtype=np.uint32,
        ),
    )
    for batch in range(actual_batch_size):
        for rotation in (0, 100, 127, 128):
            for translation in range(translation_count):
                thread_ids = translation + np.arange(4) * translation_count
                partials = lanes_np[batch, rotation, thread_ids]
                possible = set()
                for order in permutations(range(4)):
                    total = active_initial[batch]
                    for lane in order:
                        total = np.add(total, partials[lane], dtype=np.float32)
                    possible.add(int(total.view(np.uint32)))
                for output in (serial_np, dispatched_np, repeated_np):
                    output_bits = int(
                        output[batch, rotation, translation].view(np.uint32)
                    )
                    assert output_bits in possible

    expected_padding = np.broadcast_to(
        padding_initial[:, None, None],
        (padding_count, rotation_count, translation_count),
    )
    for output in (dispatched_np, repeated_np):
        np.testing.assert_array_equal(
            output[actual_batch_size:].view(np.uint32),
            expected_padding.view(np.uint32),
        )
        serial_flat = serial_np.reshape(actual_batch_size, -1)
        output_flat = output[:actual_batch_size].reshape(actual_batch_size, -1)
        np.testing.assert_array_equal(
            np.argmin(output_flat, axis=1),
            np.argmin(serial_flat, axis=1),
        )
        support_cutoff = np.partition(serial_flat, 31, axis=1)[:, 31:32]
        np.testing.assert_array_equal(
            output_flat <= support_cutoff,
            serial_flat <= support_cutoff,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("translation_count", "single_lane_canonical"),
    [(13, False), (116, True)],
    ids=("generic_four_plus_lanes", "single_lane_canonical"),
)
def test_relion_coarse_vdam_multistream_skips_poisoned_padding_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    translation_count,
    single_lane_canonical,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(52)
    current_size = 8
    model_max_r = 1
    projector_size = 2 * model_max_r + 3
    physical_batch_size = 12
    actual_batch_size = 9
    rotation_count = 129
    compact_pixel_count = current_size * (current_size // 2 + 1)
    projector = (
        rng.normal(0.0, 0.02, (projector_size,) * 3)
        + 1j * rng.normal(0.0, 0.02, (projector_size,) * 3)
    ).astype(np.complex64)
    angles = np.linspace(-np.pi, np.pi, rotation_count, endpoint=False)
    rotations = np.zeros((rotation_count, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.cos(angles)
    rotations[:, 0, 1] = -np.sin(angles)
    rotations[:, 1, 0] = np.sin(angles)
    rotations[:, 1, 1] = np.cos(angles)
    rotations[:, 2, 2] = 1.0
    active_images = (
        rng.normal(0.0, 0.02, (actual_batch_size, compact_pixel_count))
        + 1j * rng.normal(0.0, 0.02, (actual_batch_size, compact_pixel_count))
    ).astype(np.complex64)
    active_weight = rng.uniform(
        0.1,
        3.0,
        (actual_batch_size, compact_pixel_count),
    ).astype(np.float32)
    active_initial = rng.uniform(5.0, 15.0, actual_batch_size).astype(np.float32)
    padding_count = physical_batch_size - actual_batch_size
    images = np.concatenate(
        [
            active_images,
            np.full(
                (padding_count, compact_pixel_count),
                np.complex64(np.nan + 1j * np.nan),
            ),
        ],
        axis=0,
    )
    weight = np.concatenate(
        [
            active_weight,
            np.full(
                (padding_count, compact_pixel_count),
                np.nan,
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    padding_initial = np.asarray(
        [-12345.75, 6789.125, -0.0],
        dtype=np.float32,
    )
    initial_diff2 = np.concatenate([active_initial, padding_initial])
    translation_angles = rng.uniform(
        -0.2,
        0.2,
        (translation_count, 2),
    ).astype(np.float32)
    lookup = np.arange(compact_pixel_count, dtype=np.int32)

    def multistream(actual):
        return cuda_backproject.relion_coarse_diff2_projector_multistream_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
            actual_batch_size=jnp.asarray(actual, dtype=jnp.int32),
            canonical_reduction=True,
            single_lane_canonical=single_lane_canonical,
        )

    with jax.default_device(gpu_device):
        generic_serial = cuda_backproject.relion_coarse_diff2_projector_f32(
            jnp.asarray(projector),
            jnp.asarray(rotations),
            jnp.asarray(active_images),
            jnp.asarray(translation_angles),
            jnp.asarray(active_weight),
            jnp.asarray(active_initial),
            jnp.asarray(lookup),
            current_size=current_size,
            physical_image_size=current_size,
            model_max_r=model_max_r,
            canonical_reduction=True,
        )
        selected_serial = (
            cuda_backproject.relion_coarse_diff2_projector_f32(
                jnp.asarray(projector),
                jnp.asarray(rotations),
                jnp.asarray(active_images),
                jnp.asarray(translation_angles),
                jnp.asarray(active_weight),
                jnp.asarray(active_initial),
                jnp.asarray(lookup),
                current_size=current_size,
                physical_image_size=current_size,
                model_max_r=model_max_r,
                canonical_reduction=True,
                single_lane_canonical=True,
            )
            if single_lane_canonical
            else generic_serial
        )
        dispatched = multistream(actual_batch_size)
        repeated = multistream(actual_batch_size)

        generic_serial_np = np.asarray(generic_serial)
        selected_serial_np = np.asarray(selected_serial)
        dispatched_np = np.asarray(dispatched)
        repeated_np = np.asarray(repeated)
        np.testing.assert_array_equal(
            selected_serial_np.view(np.uint32),
            generic_serial_np.view(np.uint32),
        )
        np.testing.assert_array_equal(
            dispatched_np[:actual_batch_size].view(np.uint32),
            selected_serial_np.view(np.uint32),
        )
        np.testing.assert_array_equal(
            repeated_np.view(np.uint32),
            dispatched_np.view(np.uint32),
        )
        expected_padding = np.broadcast_to(
            padding_initial[:, None, None],
            (padding_count, rotation_count, translation_count),
        )
        np.testing.assert_array_equal(
            dispatched_np[actual_batch_size:].view(np.uint32),
            expected_padding.view(np.uint32),
        )

        serial_flat = generic_serial_np.reshape(actual_batch_size, -1)
        dispatched_flat = dispatched_np[:actual_batch_size].reshape(
            actual_batch_size,
            -1,
        )
        np.testing.assert_array_equal(
            np.argmin(dispatched_flat, axis=1),
            np.argmin(serial_flat, axis=1),
        )
        support_cutoff = np.partition(serial_flat, 31, axis=1)[:, 31:32]
        np.testing.assert_array_equal(
            dispatched_flat <= support_cutoff,
            serial_flat <= support_cutoff,
        )

        for invalid in (0, physical_batch_size + 1):
            with pytest.raises(
                ValueError,
                match="actual_batch_size",
            ):
                np.asarray(multistream(invalid))


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
    initial_diff2 = np.asarray([0.022644043], dtype=np.float32)
    expected = np.add(
        _production_reference(reference, shifted, weight, lookup),
        initial_diff2[0],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_rectangular_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(shifted[None, None, :]),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
            initial_diff2=jnp.asarray(initial_diff2),
        )

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray([[[expected]]], dtype=np.float32).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_fused_translate_fine_diff2_adds_highres_in_native_order(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    reference, image, weight, compact_lookup = _operands()
    current_size = 32
    lookup = np.full(current_size * (current_size // 2 + 1), -1, dtype=np.int32)
    lookup[: compact_lookup.size] = compact_lookup
    initial_diff2 = np.asarray([0.022644043], dtype=np.float32)
    expected = np.add(
        _production_reference(reference, image, weight, lookup),
        initial_diff2[0],
        dtype=np.float32,
    )

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            jnp.asarray(reference[None, None, :]),
            jnp.asarray(image[None, :]),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.asarray(weight[None, :]),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray([[[expected]]], dtype=np.float32).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_runtime_cutoff_fine_diff2_matches_static_paths_and_reuses_compile(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(932)
    physical_size = 32
    physical_pixels = physical_size * (physical_size // 2 + 1)
    translation_angles = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    initial_diff2 = np.asarray([0.022644043], dtype=np.float32)

    def _operands_for_size(logical_size):
        logical_pixels = logical_size * (logical_size // 2 + 1)
        reference = (
            rng.normal(0, 0.02, (1, 2, logical_pixels))
            + 1j * rng.normal(0, 0.02, (1, 2, logical_pixels))
        ).astype(np.complex64)
        image = (
            rng.normal(0, 0.02, (1, logical_pixels))
            + 1j * rng.normal(0, 0.02, (1, logical_pixels))
        ).astype(np.complex64)
        weight = rng.uniform(0, 150_000, (1, logical_pixels)).astype(np.float32)
        lookup = np.arange(logical_pixels, dtype=np.int32)
        pad = physical_pixels - logical_pixels
        return (
            reference,
            image,
            weight,
            lookup,
            np.pad(reference, ((0, 0), (0, 0), (0, pad)), constant_values=np.complex64(7 + 3j)),
            np.pad(image, ((0, 0), (0, pad)), constant_values=np.complex64(5 + 2j)),
            np.pad(weight, ((0, 0), (0, pad)), constant_values=np.float32(1.25e5)),
            # Keep the physical-only rectangle tail deliberately active.  A
            # kernel using physical count/stride would re-issue compact pixel
            # zero and fail the bitwise comparison.
            np.pad(lookup, (0, pad), constant_values=0),
        )

    with jax.default_device(gpu_device):
        runtime_function = (
            cuda_backproject.relion_fine_diff2_fused_translate_runtime_rectangular_f32
        )
        for logical_size in (30, 32):
            reference, image, weight, lookup, *physical = _operands_for_size(logical_size)
            expected = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
                jnp.asarray(reference),
                jnp.asarray(image),
                jnp.asarray(translation_angles),
                jnp.asarray(weight),
                jnp.asarray(lookup),
                jnp.asarray(initial_diff2),
                current_size=logical_size,
            )
            actual = runtime_function(
                jnp.asarray(physical[0]),
                jnp.asarray(physical[1]),
                jnp.asarray(translation_angles),
                jnp.asarray(physical[2]),
                jnp.asarray(physical[3]),
                jnp.asarray(logical_size, dtype=jnp.int32),
                jnp.asarray(initial_diff2),
            )
            expected, actual = jax.block_until_ready((expected, actual))
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint32),
                np.asarray(expected).view(np.uint32),
            )
            cache_size = runtime_function._cache_size()
            if logical_size == 30:
                first_cache_size = cache_size
            else:
                assert cache_size == first_cache_size


@pytest.mark.gpu
def test_relion_flat_rows_skip_invalid_rows_with_positive_infinity(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1933)
    current_size = 16
    pixel_count = current_size * (current_size // 2 + 1)
    row_image_ids = np.asarray([0, -1, 1, -1], dtype=np.int32)
    reference = (
        rng.normal(0, 0.02, (4, pixel_count))
        + 1j * rng.normal(0, 0.02, (4, pixel_count))
    ).astype(np.complex64)
    image = (
        rng.normal(0, 0.02, (2, pixel_count))
        + 1j * rng.normal(0, 0.02, (2, pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.normal(0, 0.2, (5, 2)).astype(np.float32)
    weight = rng.uniform(0, 150_000, (2, pixel_count)).astype(np.float32)
    lookup = np.arange(pixel_count, dtype=np.int32)
    initial_diff2 = np.asarray([0.03125, 0.0625], dtype=np.float32)

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_fine_diff2_fused_translate_flat_rows_f32(
            jnp.asarray(reference),
            jnp.asarray(row_image_ids),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        expected = cuda_backproject.relion_fine_diff2_fused_translate_flat_rows_f32(
            jnp.asarray(reference[[0, 2]]),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        actual, expected = jax.block_until_ready((actual, expected))

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    np.testing.assert_array_equal(actual[[0, 2]].view(np.uint32), expected.view(np.uint32))
    assert np.all(np.isposinf(actual[[1, 3]]))


@pytest.mark.gpu
def test_relion_runtime_flat_rows_match_shared_rectangular_tree_and_reuse_compile(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1932)
    physical_size = 32
    physical_pixels = physical_size * (physical_size // 2 + 1)
    row_image_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
    row_rotation_ids = np.asarray([0, 2, 1, 2], dtype=np.int32)
    translation_angles = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)

    with jax.default_device(gpu_device):
        runtime_function = (
            cuda_backproject.relion_fine_diff2_fused_translate_runtime_flat_rows_f32
        )
        runtime_function.clear_cache()
        for logical_size in (30, 32):
            logical_pixels = logical_size * (logical_size // 2 + 1)
            dense_reference = (
                rng.normal(0, 0.02, (2, 3, logical_pixels))
                + 1j * rng.normal(0, 0.02, (2, 3, logical_pixels))
            ).astype(np.complex64)
            flat_reference = dense_reference[row_image_ids, row_rotation_ids]
            image = (
                rng.normal(0, 0.02, (2, logical_pixels))
                + 1j * rng.normal(0, 0.02, (2, logical_pixels))
            ).astype(np.complex64)
            weight = rng.uniform(0, 150_000, (2, logical_pixels)).astype(np.float32)
            lookup = np.arange(logical_pixels, dtype=np.int32)
            pad = physical_pixels - logical_pixels
            physical_reference = np.pad(
                flat_reference,
                ((0, 0), (0, pad)),
                constant_values=np.complex64(7 + 3j),
            )
            physical_image = np.pad(
                image,
                ((0, 0), (0, pad)),
                constant_values=np.complex64(5 + 2j),
            )
            physical_weight = np.pad(
                weight,
                ((0, 0), (0, pad)),
                constant_values=np.float32(1.25e5),
            )
            physical_lookup = np.pad(lookup, (0, pad), constant_values=0)

            dense_expected = (
                cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
                    jnp.asarray(dense_reference),
                    jnp.asarray(image),
                    jnp.asarray(translation_angles),
                    jnp.asarray(weight),
                    jnp.asarray(lookup),
                    jnp.asarray(initial_diff2),
                    current_size=logical_size,
                )
            )
            static_flat = (
                cuda_backproject.relion_fine_diff2_fused_translate_flat_rows_f32(
                    jnp.asarray(flat_reference),
                    jnp.asarray(row_image_ids),
                    jnp.asarray(image),
                    jnp.asarray(translation_angles),
                    jnp.asarray(weight),
                    jnp.asarray(lookup),
                    jnp.asarray(initial_diff2),
                    current_size=logical_size,
                )
            )
            runtime_flat = runtime_function(
                jnp.asarray(physical_reference),
                jnp.asarray(row_image_ids),
                jnp.asarray(physical_image),
                jnp.asarray(translation_angles),
                jnp.asarray(physical_weight),
                jnp.asarray(physical_lookup),
                jnp.asarray(logical_size, dtype=jnp.int32),
                jnp.asarray(initial_diff2),
            )
            dense_expected, static_flat, runtime_flat = jax.block_until_ready(
                (dense_expected, static_flat, runtime_flat)
            )
            expected_rows = np.asarray(dense_expected)[
                row_image_ids,
                row_rotation_ids,
            ]
            np.testing.assert_array_equal(
                np.asarray(static_flat).view(np.uint32),
                expected_rows.view(np.uint32),
            )
            np.testing.assert_array_equal(
                np.asarray(runtime_flat).view(np.uint32),
                expected_rows.view(np.uint32),
            )
            cache_size = runtime_function._cache_size()
            if logical_size == 30:
                first_cache_size = cache_size
            else:
                assert cache_size == first_cache_size


@pytest.mark.gpu
def test_relion_fused_translate_pairs_match_rectangular_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1934)
    current_size = 16
    pixel_count = current_size * (current_size // 2 + 1)
    batch_size, rotation_count, translation_count = 2, 3, 5
    dense_reference = (
        rng.normal(0, 0.02, (batch_size, rotation_count, pixel_count))
        + 1j * rng.normal(0, 0.02, (batch_size, rotation_count, pixel_count))
    ).astype(np.complex64)
    flat_reference = dense_reference.reshape(-1, pixel_count)
    image = (
        rng.normal(0, 0.02, (batch_size, pixel_count))
        + 1j * rng.normal(0, 0.02, (batch_size, pixel_count))
    ).astype(np.complex64)
    translation_angles = rng.normal(0, 0.2, (translation_count, 2)).astype(
        np.float32
    )
    weight = rng.uniform(0, 150_000, (batch_size, pixel_count)).astype(np.float32)
    lookup = np.arange(pixel_count, dtype=np.int32)
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)
    pair_reference_rows = np.asarray(
        [[0, 2, 0, -1], [4, 5, 3, 3]],
        dtype=np.int32,
    )
    pair_translation_ids = np.asarray(
        [[0, 2, 0, 3], [1, 0, 3, -1]],
        dtype=np.int32,
    )

    with jax.default_device(gpu_device):
        dense = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            jnp.asarray(dense_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        pairs = cuda_backproject.relion_fine_diff2_fused_translate_pairs_f32(
            jnp.asarray(flat_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(pair_reference_rows),
            jnp.asarray(pair_translation_ids),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        dense, pairs = jax.block_until_ready((dense, pairs))

    dense = np.asarray(dense)
    pairs = np.asarray(pairs)
    expected = np.asarray(
        [
            [dense[0, 0, 0], dense[0, 2, 2], dense[0, 0, 0]],
            [dense[1, 1, 1], dense[1, 2, 0], dense[1, 0, 3]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        pairs[:, :3].view(np.uint32),
        expected.view(np.uint32),
    )
    assert np.all(np.isposinf(pairs[:, 3]))


@pytest.mark.gpu
def test_relion_fused_translate_jobs_match_rectangular_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1937)
    logical_size = 16
    logical_pixels = logical_size * (logical_size // 2 + 1)
    physical_size = 18
    physical_pixels = physical_size * (physical_size // 2 + 1)
    batch_size, rotation_count, translation_count = 2, 3, 5
    dense_reference = _complex_normal_for_test(
        rng,
        (batch_size, rotation_count, logical_pixels),
    )
    flat_reference = dense_reference.reshape(-1, logical_pixels)
    image = _complex_normal_for_test(rng, (batch_size, logical_pixels))
    translation_angles = rng.normal(0, 0.2, (translation_count, 2)).astype(
        np.float32
    )
    weight = rng.uniform(0, 150_000, (batch_size, logical_pixels)).astype(
        np.float32
    )
    lookup = np.arange(logical_pixels, dtype=np.int32)
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)
    # Deliberately cross image and rotation runs within four-job blocks.  The
    # final rows model compile-friendly global capacity padding.
    job_plan = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 2, 2, 2],
            [0, 0, 0, 0],
            [-1, -1, -1, -1],
            [1, 4, 1, 1],
            [1, 5, 2, 0],
            [1, 3, 0, 3],
            [1, -1, 2, 4],
            [0, 1, -1, 2],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    valid = (
        (job_plan[:, 0] >= 0)
        & (job_plan[:, 1] >= 0)
        & (job_plan[:, 2] >= 0)
        & (job_plan[:, 3] >= 0)
    )

    physical_reference = np.pad(
        flat_reference,
        ((0, 0), (0, physical_pixels - logical_pixels)),
        constant_values=np.complex64(7 + 3j),
    )
    physical_image = np.pad(
        image,
        ((0, 0), (0, physical_pixels - logical_pixels)),
        constant_values=np.complex64(5 + 2j),
    )
    physical_weight = np.pad(
        weight,
        ((0, 0), (0, physical_pixels - logical_pixels)),
        constant_values=np.float32(1.25e5),
    )
    physical_lookup = np.pad(
        lookup,
        (0, physical_pixels - logical_pixels),
        constant_values=0,
    )

    with jax.default_device(gpu_device):
        dense = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
            jnp.asarray(dense_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=logical_size,
        )
        static_jobs = cuda_backproject.relion_fine_diff2_fused_translate_jobs_f32(
            jnp.asarray(flat_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(job_plan),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=logical_size,
        )
        runtime_jobs = (
            cuda_backproject.relion_fine_diff2_fused_translate_runtime_jobs_f32(
                jnp.asarray(physical_reference),
                jnp.asarray(physical_image),
                jnp.asarray(translation_angles),
                jnp.asarray(physical_weight),
                jnp.asarray(job_plan),
                jnp.asarray(physical_lookup),
                jnp.asarray(logical_size, dtype=jnp.int32),
                jnp.asarray(initial_diff2),
            )
        )
        dense, static_jobs, runtime_jobs = jax.block_until_ready(
            (dense, static_jobs, runtime_jobs)
        )

    dense = np.asarray(dense)
    expected = np.full((job_plan.shape[0],), np.inf, dtype=np.float32)
    expected[valid] = dense[
        job_plan[valid, 0],
        job_plan[valid, 2],
        job_plan[valid, 3],
    ]
    np.testing.assert_array_equal(
        np.asarray(static_jobs).view(np.uint32),
        expected.view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(runtime_jobs).view(np.uint32),
        expected.view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_fused_translate_pairs_preserve_source_order_posterior_and_ties(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2.sparse_pass2_posterior import _relion_f32_fine_posterior

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1936)
    current_size = 16
    pixel_count = current_size * (current_size // 2 + 1)
    batch_size, rotation_count, translation_count = 2, 3, 4
    # Equal reference rows and translation angles deliberately make every
    # admitted candidate tie at the significance cutoff.
    one_reference = _complex_normal_for_test(rng, (batch_size, 1, pixel_count))
    dense_reference = np.repeat(one_reference, rotation_count, axis=1)
    flat_reference = dense_reference.reshape(-1, pixel_count)
    image = _complex_normal_for_test(rng, (batch_size, pixel_count))
    translation_angles = np.zeros((translation_count, 2), dtype=np.float32)
    weight = rng.uniform(0, 150_000, (batch_size, pixel_count)).astype(np.float32)
    lookup = np.arange(pixel_count, dtype=np.int32)
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)
    source_rotation_rows = np.repeat(
        np.arange(rotation_count, dtype=np.int32),
        translation_count,
    )
    source_translation_ids = np.tile(
        np.arange(translation_count, dtype=np.int32),
        rotation_count,
    )
    pair_reference_rows = np.stack(
        [
            batch * rotation_count + source_rotation_rows
            for batch in range(batch_size)
        ]
    ).astype(np.int32)
    pair_translation_ids = np.broadcast_to(
        source_translation_ids,
        pair_reference_rows.shape,
    ).copy()
    admitted = np.ones(pair_reference_rows.shape, dtype=bool)
    # Exercise row and translation sentinels independently, at their exact
    # source-order positions, while keeping the rectangular candidate length.
    pair_reference_rows[0, 2] = -1
    admitted[0, 2] = False
    pair_translation_ids[1, 5] = -1
    admitted[1, 5] = False

    candidate_mask = admitted.reshape(
        batch_size,
        rotation_count,
        translation_count,
    )
    compact_pairs = compact_candidates.build_compact_pair_index_arrays(
        candidate_mask
    )
    reference_lookup = np.arange(
        batch_size * rotation_count,
        dtype=np.int32,
    ).reshape(batch_size, rotation_count)
    compact_jobs = (
        compact_candidates.build_compact_fine_job_plan_from_pair_arrays(
            compact_pairs,
            reference_lookup,
        )
    )
    job_plan = compact_jobs["job_plan"]
    valid_job_count = int(compact_jobs["valid_job_count"])

    with jax.default_device(gpu_device):
        dense_costs = (
            cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
                jnp.asarray(dense_reference),
                jnp.asarray(image),
                jnp.asarray(translation_angles),
                jnp.asarray(weight),
                jnp.asarray(lookup),
                jnp.asarray(initial_diff2),
                current_size=current_size,
            )
        )
        pair_costs = cuda_backproject.relion_fine_diff2_fused_translate_pairs_f32(
            jnp.asarray(flat_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(pair_reference_rows),
            jnp.asarray(pair_translation_ids),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        job_costs = cuda_backproject.relion_fine_diff2_fused_translate_jobs_f32(
            jnp.asarray(flat_reference),
            jnp.asarray(image),
            jnp.asarray(translation_angles),
            jnp.asarray(weight),
            jnp.asarray(job_plan),
            jnp.asarray(lookup),
            jnp.asarray(initial_diff2),
            current_size=current_size,
        )
        dense_costs, pair_costs, job_costs = jax.block_until_ready(
            (dense_costs, pair_costs, job_costs)
        )
        dense_scores = -jnp.asarray(dense_costs)
        dense_scores = jnp.where(
            jnp.asarray(admitted.reshape(dense_scores.shape)),
            dense_scores,
            -jnp.inf,
        )
        pair_scores = -jnp.asarray(pair_costs)
        dense_posterior = _relion_f32_fine_posterior(
            dense_scores,
            adaptive_fraction=0.5,
        )
        pair_posterior = _relion_f32_fine_posterior(
            pair_scores,
            adaptive_fraction=0.5,
        )
        valid_job_plan = job_plan[:valid_job_count]
        job_scattered = jnp.full_like(dense_costs, jnp.inf).at[
            jnp.asarray(valid_job_plan[:, 0]),
            jnp.asarray(valid_job_plan[:, 2]),
            jnp.asarray(valid_job_plan[:, 3]),
        ].set(jnp.asarray(job_costs[:valid_job_count]))
        job_posterior = _relion_f32_fine_posterior(
            -job_scattered,
            adaptive_fraction=0.5,
        )
        dense_posterior, pair_posterior, job_posterior = jax.block_until_ready(
            (dense_posterior, pair_posterior, job_posterior)
        )

    expected_costs = np.asarray(dense_costs).reshape(batch_size, -1)
    expected_costs = np.where(admitted, expected_costs, np.float32(np.inf))
    np.testing.assert_array_equal(
        np.asarray(pair_costs).view(np.uint32),
        expected_costs.view(np.uint32),
    )
    valid_costs = np.asarray(pair_costs)[admitted]
    for batch in range(batch_size):
        batch_costs = np.asarray(pair_costs)[batch, admitted[batch]]
        assert np.unique(batch_costs.view(np.uint32)).size == 1
    assert valid_costs.size == admitted.sum()
    expected_job_costs = np.asarray(dense_costs)[candidate_mask]
    np.testing.assert_array_equal(
        np.asarray(job_costs[:valid_job_count]).view(np.uint32),
        expected_job_costs.view(np.uint32),
    )
    assert np.all(np.isposinf(np.asarray(job_costs[valid_job_count:])))
    np.testing.assert_array_equal(np.asarray(pair_posterior[2]), admitted)
    for dense_value, pair_value in zip(dense_posterior, pair_posterior):
        dense_array = np.asarray(dense_value).reshape(-1)
        pair_array = np.asarray(pair_value).reshape(-1)
        assert dense_array.dtype == pair_array.dtype
        assert dense_array.tobytes() == pair_array.tobytes()
    for dense_value, job_value in zip(dense_posterior, job_posterior):
        dense_array = np.asarray(dense_value).reshape(-1)
        job_array = np.asarray(job_value).reshape(-1)
        assert dense_array.dtype == job_array.dtype
        assert dense_array.tobytes() == job_array.tobytes()


@pytest.mark.gpu
def test_relion_runtime_fused_translate_pairs_reuse_physical_compile(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(1935)
    physical_size = 32
    physical_pixels = physical_size * (physical_size // 2 + 1)
    translation_angles = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    pair_reference_rows = np.asarray([[0, 2], [3, 1]], dtype=np.int32)
    pair_translation_ids = np.asarray([[0, 3], [2, 1]], dtype=np.int32)
    job_plan = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 2, 2, 3],
            [1, 3, 0, 2],
            [1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    initial_diff2 = np.asarray([0.022644043, 0.03125], dtype=np.float32)

    with jax.default_device(gpu_device):
        runtime_function = (
            cuda_backproject.relion_fine_diff2_fused_translate_runtime_pairs_f32
        )
        runtime_jobs_function = (
            cuda_backproject.relion_fine_diff2_fused_translate_runtime_jobs_f32
        )
        runtime_function.clear_cache()
        runtime_jobs_function.clear_cache()
        for logical_size in (30, 32):
            logical_pixels = logical_size * (logical_size // 2 + 1)
            reference = (
                rng.normal(0, 0.02, (4, logical_pixels))
                + 1j * rng.normal(0, 0.02, (4, logical_pixels))
            ).astype(np.complex64)
            image = (
                rng.normal(0, 0.02, (2, logical_pixels))
                + 1j * rng.normal(0, 0.02, (2, logical_pixels))
            ).astype(np.complex64)
            weight = rng.uniform(0, 150_000, (2, logical_pixels)).astype(np.float32)
            lookup = np.arange(logical_pixels, dtype=np.int32)
            pad = physical_pixels - logical_pixels
            physical_reference = np.pad(
                reference,
                ((0, 0), (0, pad)),
                constant_values=np.complex64(7 + 3j),
            )
            physical_image = np.pad(
                image,
                ((0, 0), (0, pad)),
                constant_values=np.complex64(5 + 2j),
            )
            physical_weight = np.pad(
                weight,
                ((0, 0), (0, pad)),
                constant_values=np.float32(1.25e5),
            )
            physical_lookup = np.pad(lookup, (0, pad), constant_values=0)
            expected = cuda_backproject.relion_fine_diff2_fused_translate_pairs_f32(
                jnp.asarray(reference),
                jnp.asarray(image),
                jnp.asarray(translation_angles),
                jnp.asarray(weight),
                jnp.asarray(pair_reference_rows),
                jnp.asarray(pair_translation_ids),
                jnp.asarray(lookup),
                jnp.asarray(initial_diff2),
                current_size=logical_size,
            )
            actual = runtime_function(
                jnp.asarray(physical_reference),
                jnp.asarray(physical_image),
                jnp.asarray(translation_angles),
                jnp.asarray(physical_weight),
                jnp.asarray(pair_reference_rows),
                jnp.asarray(pair_translation_ids),
                jnp.asarray(physical_lookup),
                jnp.asarray(logical_size, dtype=jnp.int32),
                jnp.asarray(initial_diff2),
            )
            expected, actual = jax.block_until_ready((expected, actual))
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint32),
                np.asarray(expected).view(np.uint32),
            )
            expected_jobs = cuda_backproject.relion_fine_diff2_fused_translate_jobs_f32(
                jnp.asarray(reference),
                jnp.asarray(image),
                jnp.asarray(translation_angles),
                jnp.asarray(weight),
                jnp.asarray(job_plan),
                jnp.asarray(lookup),
                jnp.asarray(initial_diff2),
                current_size=logical_size,
            )
            actual_jobs = runtime_jobs_function(
                jnp.asarray(physical_reference),
                jnp.asarray(physical_image),
                jnp.asarray(translation_angles),
                jnp.asarray(physical_weight),
                jnp.asarray(job_plan),
                jnp.asarray(physical_lookup),
                jnp.asarray(logical_size, dtype=jnp.int32),
                jnp.asarray(initial_diff2),
            )
            expected_jobs, actual_jobs = jax.block_until_ready(
                (expected_jobs, actual_jobs)
            )
            np.testing.assert_array_equal(
                np.asarray(actual_jobs).view(np.uint32),
                np.asarray(expected_jobs).view(np.uint32),
            )
            cache_size = runtime_function._cache_size()
            jobs_cache_size = runtime_jobs_function._cache_size()
            if logical_size == 30:
                first_cache_size = cache_size
                first_jobs_cache_size = jobs_cache_size
            else:
                assert cache_size == first_cache_size
                assert jobs_cache_size == first_jobs_cache_size


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


def test_relion_coarse_diff2_rotation_blocks_fails_closed_without_gpu(
    monkeypatch,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_rotation_blocks_f32.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 29, 2), dtype=jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )


def test_relion_coarse_diff2_rotation_blocks_rejects_noninteger_ids():
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(TypeError, match="rotation_block_ids must be int32"):
        cuda_backproject.relion_coarse_diff2_rotation_blocks_f32.__wrapped__(
            jnp.zeros((1, 2), dtype=jnp.complex64),
            jnp.zeros((1, 29, 2), dtype=jnp.complex64),
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
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
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            1,
            1,
            1,
        )


def test_relion_coarse_native_texture_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_native_texture_rectangular_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
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


def test_relion_coarse_vdam_projector_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_projector_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
        )


def test_relion_coarse_vdam_projector_lane_capture_fails_closed_without_gpu(
    monkeypatch,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_coarse_diff2_projector_lanes_f32.__wrapped__(
            jnp.zeros((5, 5, 5), dtype=jnp.complex64),
            jnp.eye(3, dtype=jnp.float32)[None, :, :],
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            current_size=1,
            physical_image_size=1,
            model_max_r=1,
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
    from recovar.em.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_diff2_sum

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
    from recovar.em.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_diff2_sum

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


@pytest.mark.gpu
def test_relion_powerclass_highres_matches_single_block_tree_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(31)
    xdim, ydim = 5, 8
    image = (
        rng.normal(0.0, 0.2, xdim * ydim)
        + 1j * rng.normal(0.0, 0.2, xdim * ydim)
    ).astype(np.complex64)
    resolution_limit = 3
    lanes = np.zeros(128, dtype=np.float32)
    for voxel, value in enumerate(image):
        x = voxel % xdim
        y = voxel // xdim
        y = y if y < xdim else y - ydim
        shell = int(np.rint(np.sqrt(np.float32(x * x + y * y))))
        if shell <= 0 or shell >= xdim or (x == 0 and y < 0):
            continue
        imag_square = np.float32(value.imag * value.imag)
        power = _fma32(value.real, value.real, imag_square)
        if shell >= resolution_limit:
            lanes[voxel] = power
    for width in (64, 32, 16, 8, 4, 2, 1):
        lanes[:width] = np.add(lanes[:width], lanes[width : 2 * width], dtype=np.float32)

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_powerclass_spectrum_highres_f32(
            jnp.asarray(image[None, :]),
            xdim=xdim,
            ydim=ydim,
            resolution_limit=resolution_limit,
        )

    np.testing.assert_array_equal(
        np.asarray(actual)[0, -1].view(np.uint32),
        lanes[0].view(np.uint32),
    )



@pytest.mark.gpu
def test_relion_wavg_sequential_triplet_matches_jax_loop_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2.sparse_pass2_wavg import _relion_wavg_sequential_triplet_terms_jax

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(47)
    batch_size, rotation_count, translation_count, pixel_count = 3, 7, 11, 37
    projections = (
        rng.normal(0.0, 0.8, (batch_size, rotation_count, pixel_count))
        + 1j * rng.normal(0.0, 0.8, (batch_size, rotation_count, pixel_count))
    ).astype(np.complex64)
    raw_ctf = rng.normal(0.0, 0.7, (batch_size, pixel_count)).astype(np.float32)
    scale = rng.uniform(0.25, 2.0, batch_size).astype(np.float32)
    shifted = (
        rng.normal(0.0, 1.1, (batch_size, translation_count, pixel_count))
        + 1j * rng.normal(0.0, 1.1, (batch_size, translation_count, pixel_count))
    ).astype(np.complex64)
    posterior = rng.uniform(
        0.0,
        1.0,
        (batch_size, rotation_count, translation_count),
    ).astype(np.float32)
    posterior[:, :, ::5] = 0.0

    with jax.default_device(gpu_device):
        operands = tuple(
            jnp.asarray(value)
            for value in (projections, raw_ctf, scale, shifted, posterior)
        )
        expected = _relion_wavg_sequential_triplet_terms_jax(*operands)
        actual = cuda_backproject.relion_wavg_sequential_triplet_f32(*operands)
        expected, actual = jax.block_until_ready((expected, actual))

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
    )



def test_relion_runtime_cutoff_fine_diff2_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_fine_diff2_fused_translate_runtime_rectangular_f32.__wrapped__(
            jnp.zeros((1, 1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )



def test_relion_runtime_flat_rows_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_fine_diff2_fused_translate_runtime_flat_rows_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )



def test_relion_runtime_pairs_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_fine_diff2_fused_translate_runtime_pairs_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )



def test_relion_runtime_jobs_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_fine_diff2_fused_translate_runtime_jobs_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 4), dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        )



def test_relion_powerclass_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_powerclass_spectrum_highres_f32.__wrapped__(
            jnp.zeros((1, 40), dtype=jnp.complex64),
            xdim=5,
            ydim=8,
            resolution_limit=3,
        )


@pytest.mark.parametrize(
    "bad_indices",
    [
        np.asarray([-1], dtype=np.int32),
        np.asarray([12], dtype=np.int32),
        np.asarray([1.5], dtype=np.float64),
        np.asarray([[1]], dtype=np.int32),
        np.asarray([True], dtype=bool),
    ],
)
def test_exact_ctf_compact_indices_reject_invalid_host_geometry(monkeypatch, tmp_path, bad_indices):
    from types import SimpleNamespace

    source = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _: source)
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(source), (4, 4)), {"images": {0: np.ones(12, dtype=np.float64)}}
    )
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: indices)
    with pytest.raises(ValueError):
        relion_ctf._relion_exact_ctf_half_from_source_star_host(
            dataset,
            np.asarray([0]),
            (4, 4),
            pixel_indices=bad_indices,
        )


def test_exact_ctf_compact_indices_never_materialize_device_inputs(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class DeviceOnly:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Unexpected device-to-host materialization")

    source = (tmp_path / "particles.star").resolve()
    monkeypatch.setattr(relion_ctf, "_relion_exact_ctf_source_star", lambda _: source)
    monkeypatch.setitem(
        relion_ctf._RELION_EXACT_CTF_SOURCE_CACHE, (str(source), (4, 4)), {"images": {0: np.ones(12, dtype=np.float64)}}
    )
    dataset = SimpleNamespace(original_image_indices_from_local=lambda indices: indices)
    for indices in (DeviceOnly(), jnp.asarray([0], dtype=jnp.int32)):
        with pytest.raises(TypeError, match="host NumPy array"):
            relion_ctf._relion_exact_ctf_half_from_source_star_host(
                dataset,
                np.asarray([0]),
                (4, 4),
                pixel_indices=indices,
            )
