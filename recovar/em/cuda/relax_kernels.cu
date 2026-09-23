// EM translation unit of librelax_cuda.so, split verbatim from recovar/cuda/cuda_backproject.cu by the
// relax split (seam S4); the shared device helpers live in recovar/cuda/include/recovar_cuda_common.cuh.
/*
 * CUDA Backprojector / Projector  — v6
 *
 *   - XLA FFI handlers → JIT-compatible inside JAX
 *   - C-linkage API    → ctypes benchmarks / standalone use
 *   - Templated on T   → float (C64) or double (C128)
 *   - On-the-fly freq coords, cz=0 elimination (6-element rotation)
 *   - float2/double2 vectorized complex I/O
 *   - HALF_VOL: half-volume (Hermitian symmetry), halves volume memory
 *   - HALF_IMG: rfft image layout (H × W//2+1), halves pixel count
 *     For backproject, each non-boundary rfft pixel scatters both the
 *     primary value and its Hermitian conjugate at the negated coords.
 *
 * v6 changes:
 *   - CONJ_MODE template parameter for ~2x scatter speedup when
 *     HALF_IMG + HALF_VOL: interior kz (0 < hkz < ic2) get doubled in
 *     the primary scatter (CONJ_MODE=1) and skipped in the conjugate
 *     scatter (CONJ_MODE=2).  This works because for interior kz, the
 *     primary and conjugate scatters land at the same half-volume position
 *     after Hermitian fold, making the conjugate scatter redundant.
 *   - Nyquist fix: kz=-N/2 (Nyquist for even N) is self-conjugate and
 *     scatters directly (no fold/conj) — fixes off-by-one error
 *
 * IMPORTANT: The HALF_VOL scatter (Hermitian fold) is the correct adjoint
 * of the index-based half_volume_to_full_volume in fourier_transform_utils.py.
 * Do NOT use an FFT-based half→full expand; its VJP distributes gradients
 * differently, breaking the CUDA kernel's correctness.
 *
 * Volume: (N0, N1, N2) complex  stored as interleaved T pairs.
 * Half  : (N0, N1, N2/2+1) complex.
 * Images full : (n_images, H*W) complex, row-major (k1/col varies fastest).
 * Images rfft : (n_images, H*(W//2+1)) complex, row-major.
 * Rotations: (n_images, 6) T  — first two rows of 3×3 matrix, row-major.
 *
 * Pixel indexing: row-major — k0_idx = pix / image_w, k1_idx = pix % image_w.
 * This matches NumPy/JAX C-order flatten convention.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <cerrno>
#include <chrono>
#include <climits>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#include "device_scratch.cuh"
#include "noise_residual.cuh"

constexpr char kRelionVdamExactNativePtxEnv[] =
    "RECOVAR_VDAM_EXACT_NATIVE_PTX";
constexpr char kRelionVdamExactWavgPredecessorEnv[] =
    "RECOVAR_VDAM_EXACT_WAVG_PREDECESSOR";
constexpr char kRelionVdamRuntimeBprefWithExactWavgEnv[] =
    "RECOVAR_VDAM_RUNTIME_BPREF_WITH_EXACT_WAVG";
constexpr char kRelionVdamWavgBprefHostGapNsEnv[] =
    "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_NS";
constexpr char kRelionVdamWavgBprefHostGapTraceEnv[] =
    "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE";
constexpr char kRelionVdamWavgBprefHostGapTraceParticleEnv[] =
    "RECOVAR_VDAM_WAVG_BPREF_HOST_GAP_TRACE_PARTICLE_ID";
constexpr char kRelionVdamPreprojectPersistentRotationsEnv[] =
    "RECOVAR_VDAM_PREPROJECT_PERSISTENT_ROTATIONS";
constexpr char kRelionVdamPrecomputePersistentResidualsEnv[] =
    "RECOVAR_VDAM_PRECOMPUTE_PERSISTENT_RESIDUALS";
constexpr char kRelionVdamPrecomputeOrderedResidualsEnv[] =
    "RECOVAR_VDAM_PRECOMPUTE_ORDERED_RESIDUALS";
constexpr char kRelionVdamFixedWarpOrderScatterEnv[] =
    "RECOVAR_VDAM_FIXED_WARP_ORDER_SCATTER";
constexpr char kRelionVdamOrderedScatterCudaGraphEnv[] =
    "RECOVAR_VDAM_ORDERED_SCATTER_CUDA_GRAPH";
constexpr char kRelionVdamExactNativePtxKernel[] =
    "_Z29cuda_kernel_backproject3D_SGDILb0ELb0EEv18AccProjectorKernel"
    "PfS1_S1_S1_S1_S1_S1_S1_mffS1_S1_S1_S1_iifjjjjjjii";
constexpr char kRelionVdamExactWavgKernel[] =
    "_Z16cuda_kernel_wavgILb1ELb1ELb0ELi256EEvPf18AccProjectorKernel"
    "jmS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_mfff";

// RELION's GUI-default InitialModel jobs use --j 8.  Both coarse scoring and
// BPref replay use one blocking CUDA stream per worker.  Keep stream creation,
// parent-stream dependency, synchronization, and cleanup in one shared helper
// so the two paths cannot silently acquire different ownership semantics.
constexpr int kRelionVdamWorkerStreams = 8;

cudaError_t initialize_relion_vdam_worker_streams(
    cudaStream_t parent_stream,
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    cudaEvent_t* inputs_ready)
{
    cudaError_t error = cudaEventCreateWithFlags(
        inputs_ready, cudaEventDisableTiming);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(*inputs_ready, parent_stream);
    if (error != cudaSuccess) return error;
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
    {
        // Ordinary blocking streams match RELION's per-worker class streams.
        error = cudaStreamCreate(&worker_streams[worker]);
        if (error != cudaSuccess) return error;
        error = cudaStreamWaitEvent(
            worker_streams[worker], *inputs_ready, 0);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

cudaError_t synchronize_relion_vdam_worker_streams(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams])
{
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
    {
        const cudaError_t error = cudaStreamSynchronize(worker_streams[worker]);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

template <typename LaunchParticle>
cudaError_t dispatch_relion_vdam_round_robin_workers(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    int64_t particle_count,
    LaunchParticle&& launch_particle)
{
    bool worker_started[kRelionVdamWorkerStreams] = {};
    for (int64_t particle = 0; particle < particle_count; ++particle)
    {
        const int worker = static_cast<int>(
            particle % kRelionVdamWorkerStreams);
        // Match RELION's task distributor: a worker synchronizes its class
        // stream before taking its next particle, while other workers remain
        // independent.
        if (worker_started[worker])
        {
            const cudaError_t error = cudaStreamSynchronize(
                worker_streams[worker]);
            if (error != cudaSuccess) return error;
        }
        const cudaError_t error = launch_particle(particle, worker);
        if (error != cudaSuccess) return error;
        worker_started[worker] = true;
    }
    return synchronize_relion_vdam_worker_streams(worker_streams);
}

void destroy_relion_vdam_worker_streams(
    cudaStream_t worker_streams[kRelionVdamWorkerStreams],
    cudaEvent_t inputs_ready)
{
    for (int worker = 0; worker < kRelionVdamWorkerStreams; ++worker)
        if (worker_streams[worker]) cudaStreamDestroy(worker_streams[worker]);
    if (inputs_ready) cudaEventDestroy(inputs_ready);
}

cudaError_t report_relion_vdam_driver_error(
    const char* operation,
    CUresult result)
{
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    std::fprintf(
        stderr,
        "RECOVAR exact RELION VDAM PTX %s failed: %s (%s)\n",
        operation,
        name == nullptr ? "unknown CUDA driver error" : name,
        description == nullptr ? "no description" : description);
    return cudaErrorUnknown;
}

#include "vdam_trace.cuh"

#include "recovar_cuda_common.cuh"


/* Strict RELION x-half diagnostic: scatter the pre-reduced complex data and
 * real weight through the same neighbor loop. RELION updates model real,
 * model imaginary, and model weight consecutively for each neighbor; keeping
 * those atomics together is the sole semantic difference from invoking the
 * generic complex and real scatter paths separately. */
template <typename T, typename ComplexT, bool CAPTURE_SIGNATURE, bool ACCUMULATE,
          bool SEPARATE_DATA = false>
static __device__ __forceinline__ void scatter_trilinear_relion_fused_x_half(
    ComplexT* __restrict__ data_volume,
    T* __restrict__ data_real_volume,
    T* __restrict__ data_imag_volume,
    T* __restrict__ weight_volume,
    T rk0, T rk1, T rk2,
    T data_re, T data_im, T Fweight,
    T c0, T c1, T c2,
    int N0, int N1, int N2_eff, int stride0, int stride1,
    int signature_base,
    int32_t* __restrict__ signature_neighbor_indices,
    float* __restrict__ signature_neighbor_coefficients,
    int32_t* __restrict__ signature_neighbor_flags)
{
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;
    const int ic0 = (int)c0;
    const int ic1 = (int)c1;
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
    const T g2_full = rk2 + c2;

    /* The caller's compact all-neighbor gate already proves these bounds for
     * RELION BPref shapes; retain this as a defensive array-safety check. */
    if (g0 < T(-1) || g0 >= T(N0) ||
        g1 < T(-1) || g1 >= T(N1) ||
        g2_full < T(-1) || g2_full >= T(N2_full)) return;

    /* RELION forms each interpolation fraction from the rotated coordinate
     * before applying the integer model origin.  Adding the origin first is
     * mathematically equivalent but loses float32 mantissa bits and changes
     * the trilinear coefficients by several ulp. */
    /* RELION's BP.cuh calls floorf for these integer buckets even when
     * XFLOAT is double.  Keep the original T-valued coordinate for the
     * fractional remainder, exactly as the native kernel does. */
    const int r0 = relion_floor_int(rk0);
    const int r1 = relion_floor_int(rk1);
    const int r2 = relion_floor_int(rk2);
    const int b0 = r0 + ic0;
    const int b1 = r1 + ic1;
    const int b2 = r2 + ic2;
    const T f0 = rk0 - T(r0);
    const T f1 = rk1 - T(r1);
    const T f2 = rk2 - T(r2);
    const T w0[2] = {T(1) - f0, f0};
    const T w1[2] = {T(1) - f1, f1};
    const T w2[2] = {T(1) - f2, f2};

    #pragma unroll
    for (int d0 = 0; d0 < 2; d0++) {
        int j0 = b0 + d0;
        if ((unsigned)j0 >= (unsigned)N0) continue;
        #pragma unroll
        for (int d1 = 0; d1 < 2; d1++) {
            int j1 = b1 + d1;
            if ((unsigned)j1 >= (unsigned)N1) continue;
            const T ww = w0[d0] * w1[d1];
            #pragma unroll
            for (int d2 = 0; d2 < 2; d2++) {
                const int signature_slot = d0 * 4 + d1 * 2 + d2;
                const int signature_index = signature_base + signature_slot;
                const int j2 = b2 + d2;
                if ((unsigned)j0 >= (unsigned)N0 ||
                    (unsigned)j1 >= (unsigned)N1 ||
                    (unsigned)j2 >= (unsigned)N2_full) {
                    if constexpr (CAPTURE_SIGNATURE) {
                        signature_neighbor_flags[signature_index] = 8;
                    }
                    continue;
                }
                const int kz = j2 - ic2;
                const T w = ww * w2[d2];
                int sj0 = j0;
                int sj1 = j1;
                int hkz;
                int32_t neighbor_flags = 1;
                T sre = w * data_re;
                T sim = w * data_im;
                if (kz >= 0) {
                    hkz = kz;
                } else if ((N2_full & 1) == 0 && -kz == ic2) {
                    hkz = ic2;
                    neighbor_flags |= 4;
                } else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                    sim = -sim;
                    neighbor_flags |= 2;
                }
                if (hkz > ic2) {
                    if constexpr (CAPTURE_SIGNATURE) {
                        signature_neighbor_flags[signature_index] = 8;
                    }
                    continue;
                }
                const int off = sj0 * stride0 + sj1 * stride1 + hkz;
                if constexpr (CAPTURE_SIGNATURE) {
                    signature_neighbor_indices[signature_index] = off;
                    signature_neighbor_coefficients[signature_index] = (float)w;
                    signature_neighbor_flags[signature_index] = neighbor_flags;
                }
                if constexpr (ACCUMULATE) {
                    if constexpr (SEPARATE_DATA) {
                        atomicAdd(&data_real_volume[off], sre);
                        atomicAdd(&data_imag_volume[off], sim);
                    } else {
                        atomicAdd(&data_volume[off].x, sre);
                        atomicAdd(&data_volume[off].y, sim);
                    }
                    atomicAdd(&weight_volume[off], w * Fweight);
                }
            }
        }
    }
}

/* One invocation corresponds to one RELION particle. blockIdx.x retains the
 * particle-local orientation-row order, while each 128-thread block walks the
 * native current-size FFTW square in serial pixel passes. */
template <typename T, typename ComplexT, bool CAPTURE_SIGNATURE, bool ACCUMULATE>
__global__ void __launch_bounds__(128)
relion_fused_x_half_backproject_kernel(
    ComplexT* __restrict__ data_volume,
    T* __restrict__ weight_volume,
    const ComplexT* __restrict__ data_rows,
    const T* __restrict__ weight_rows,
    const int32_t* __restrict__ pixel_indices,
    const T* __restrict__ rot,
    const int32_t* __restrict__ canonical_rotation_keys,
    const int32_t* __restrict__ signature_row_indices,
    int32_t* __restrict__ signature_rotation_keys,
    int32_t* __restrict__ signature_pixel_indices,
    int32_t* __restrict__ signature_row_flags,
    float* __restrict__ signature_source_values,
    int32_t* __restrict__ signature_neighbor_indices,
    float* __restrict__ signature_neighbor_coefficients,
    int32_t* __restrict__ signature_neighbor_flags,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, T max_r2, int n_source_rows)
{
    __shared__ T R[6];
    const int output_row = (int)blockIdx.x;
    int source_row;
    if constexpr (CAPTURE_SIGNATURE) {
        source_row = (int)signature_row_indices[output_row];
        if ((unsigned)source_row >= (unsigned)n_source_rows) return;
    } else {
        source_row = output_row;
    }
    if (threadIdx.x < 6) R[threadIdx.x] = rot[source_row * 6 + threadIdx.x];
    __syncthreads();

    for (int pix = (int)threadIdx.x; pix < n_pixels; pix += 128) {
        const int orig_pix = (int)pixel_indices[pix];
        const int row_pixel = output_row * n_pixels + pix;
        const int source_row_pixel = source_row * n_pixels + pix;
        int32_t row_flags = 0;
        if constexpr (CAPTURE_SIGNATURE) {
            signature_rotation_keys[row_pixel] = canonical_rotation_keys[source_row];
            signature_pixel_indices[row_pixel] = orig_pix;
            signature_row_flags[row_pixel] = 0;
            #pragma unroll
            for (int value_index = 0; value_index < 6; value_index++) {
                signature_source_values[row_pixel * 6 + value_index] = nanf("");
            }
            #pragma unroll
            for (int slot = 0; slot < 8; slot++) {
                const int signature_index = row_pixel * 8 + slot;
                signature_neighbor_indices[signature_index] = -1;
                signature_neighbor_coefficients[signature_index] = 0.0f;
                signature_neighbor_flags[signature_index] = 8;
            }
        }
        const int k0_idx = orig_pix / image_w;
        const int k1_idx = orig_pix % image_w;

        const T k0_unscaled = (k0_idx < image_w)
            ? T(k0_idx)
            : T(k0_idx - image_h);
        const T k1_unscaled = T(k1_idx);
        const T k0 = k0_unscaled * T(upsampling);
        const T k1 = k1_unscaled * T(upsampling);

        /* RELION omits the redundant negative-y x=0 FFTW row. */
        if (k1_idx == 0 && k0_idx >= image_w) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 1;
            continue;
        }
        if (max_r2 >= 0.0f && k0 * k0 + k1 * k1 > max_r2) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 2;
            continue;
        }

        const T Fweight = weight_rows[source_row_pixel];
        const ComplexT value = data_rows[source_row_pixel];
        if constexpr (CAPTURE_SIGNATURE) {
            signature_source_values[row_pixel * 6 + 0] = value.x;
            signature_source_values[row_pixel * 6 + 1] = value.y;
            signature_source_values[row_pixel * 6 + 2] = Fweight;
        }
        /* Match cuda_kernel_backproject3D's sole outer accumulation gate. */
        if (!(Fweight > 0.0f)) {
            if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 4;
            continue;
        }

        T data_re = value.x;
        T data_im = value.y;
        /* Match RELION cuda_kernel_backproject3D: form matrix-x*source-x
         * before matrix-y*source-y, then apply padding_factor. Both the
         * addend order and delayed scaling are observable at interpolation
         * boundaries. */
        T rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * T(upsampling);
        T rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * T(upsampling);
        T rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * T(upsampling);
        if constexpr (CAPTURE_SIGNATURE) {
            signature_source_values[row_pixel * 6 + 3] = rk0;
            signature_source_values[row_pixel * 6 + 4] = rk1;
            signature_source_values[row_pixel * 6 + 5] = rk2;
        }

        if (max_r2 >= T(0)) {
            const T r2_3d = relion_radius_squared(rk0, rk1, rk2);
            if (r2_3d > max_r2) {
                if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 8;
                continue;
            }
        }
        if (rk2 < T(0)) {
            row_flags |= 16;
            rk0 = -rk0;
            rk1 = -rk1;
            rk2 = -rk2;
            data_im = -data_im;
        }
        if (max_r2 >= T(0)) {
            const int maxR = (int)floor(sqrt(max_r2) + T(0.5));
            if (relion_compact_trilinear_oob<T>(rk2, rk1, rk0, maxR)) {
                if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags | 32;
                continue;
            }
        }
        row_flags |= 64;
        if constexpr (CAPTURE_SIGNATURE) signature_row_flags[row_pixel] = row_flags;

        const int stride1 = N2_eff;
        const int stride0 = N1 * N2_eff;
        scatter_trilinear_relion_fused_x_half<T, ComplexT, CAPTURE_SIGNATURE, ACCUMULATE>(
            data_volume, nullptr, nullptr, weight_volume,
            rk0, rk1, rk2, data_re, data_im, Fweight,
            c0, c1, c2, N0, N1, N2_eff, stride0, stride1,
            row_pixel * 8,
            signature_neighbor_indices,
            signature_neighbor_coefficients,
            signature_neighbor_flags);
    }
}

__global__ void
relion_firstiter_bpref_fused_x_half_kernel(
    const float* __restrict__ image_real,
    const float* __restrict__ image_imag,
    const float* __restrict__ translation_x,
    const float* __restrict__ translation_y,
    const float* __restrict__ posterior,
    const float* __restrict__ minvsigma2,
    const float* __restrict__ ctf,
    unsigned long translation_num,
    float significant_weight,
    float weight_norm,
    const float* __restrict__ eulers,
    float* __restrict__ model_real,
    float* __restrict__ model_imag,
    float* __restrict__ model_weight,
    int max_r2,
    float padding_factor,
    unsigned img_x,
    unsigned img_y,
    unsigned img_xyz,
    unsigned mdl_x,
    unsigned mdl_y,
    int mdl_inity,
    int mdl_initz)
{
    const unsigned tid = threadIdx.x;
    const unsigned img = blockIdx.x;
    const int img_y_half = img_y / 2;
    const int max_r2_vol = max_r2 * padding_factor * padding_factor;

    __shared__ float shared_eulers[9];
    if (tid < 9) shared_eulers[tid] = eulers[img * 9 + tid];
    __syncthreads();

    const int pixel_pass_num = (int)ceilf((float)img_xyz / 128.0f);
    for (unsigned pass = 0; pass < (unsigned)pixel_pass_num; ++pass) {
        const unsigned pixel = pass * 128U + tid;
        if (pixel >= img_xyz) continue;

        int x = pixel % img_x;
        int y = (int)(pixel / img_x);
        if (y > img_y_half) y -= img_y;

        const float pixel_minvsigma2 = __ldg(&minvsigma2[pixel]);
        const float pixel_ctf = __ldg(&ctf[pixel]);
        const float pixel_real = __ldg(&image_real[pixel]);
        const float pixel_imag = __ldg(&image_imag[pixel]);
        float Fweight = 0.0f;
        float real = 0.0f;
        float imag = 0.0f;
        for (unsigned long translation = 0; translation < translation_num; ++translation) {
            float weight = posterior[img * translation_num + translation];
            if (weight >= significant_weight) {
                weight = (weight / weight_norm) * pixel_ctf * pixel_minvsigma2;
                Fweight += weight * pixel_ctf;
                float sine;
                float cosine;
                sincosf(x * translation_x[translation] + y * translation_y[translation],
                        &sine, &cosine);
                const float translated_real = cosine * pixel_real - sine * pixel_imag;
                const float translated_imag = cosine * pixel_imag + sine * pixel_real;
                real += translated_real * weight;
                imag += translated_imag * weight;
            }
        }
        if (!(Fweight > 0.0f)) continue;

        float xp = (shared_eulers[0] * x + shared_eulers[1] * y) * padding_factor;
        float yp = (shared_eulers[3] * x + shared_eulers[4] * y) * padding_factor;
        float zp = (shared_eulers[6] * x + shared_eulers[7] * y) * padding_factor;
        if ((xp * xp + yp * yp + zp * zp) > max_r2_vol) continue;
        if (xp < 0.0f) {
            xp = -xp;
            yp = -yp;
            zp = -zp;
            imag = -imag;
        }

        const int x0 = (int)floorf(xp);
        const float fx = xp - x0;
        const int x1 = x0 + 1;
        int y0 = (int)floorf(yp);
        const float fy = yp - y0;
        y0 -= mdl_inity;
        const int y1 = y0 + 1;
        int z0 = (int)floorf(zp);
        const float fz = zp - z0;
        z0 -= mdl_initz;
        const int z1 = z0 + 1;

        const float mfx = 1.0f - fx;
        const float mfy = 1.0f - fy;
        const float mfz = 1.0f - fz;

#define RELION_ADD_NEIGHBOR(Z, Y, X, COEFF) do { \
        const unsigned offset = (unsigned)(Z) * mdl_x * mdl_y + \
                                (unsigned)(Y) * mdl_x + (unsigned)(X); \
        const float coefficient = (COEFF); \
        atomicAdd(&model_real[offset], coefficient * real); \
        atomicAdd(&model_imag[offset], coefficient * imag); \
        atomicAdd(&model_weight[offset], coefficient * Fweight); \
    } while (0)
        RELION_ADD_NEIGHBOR(z0, y0, x0, mfz * mfy * mfx);
        RELION_ADD_NEIGHBOR(z0, y0, x1, mfz * mfy * fx);
        RELION_ADD_NEIGHBOR(z0, y1, x0, mfz * fy * mfx);
        RELION_ADD_NEIGHBOR(z0, y1, x1, mfz * fy * fx);
        RELION_ADD_NEIGHBOR(z1, y0, x0, fz * mfy * mfx);
        RELION_ADD_NEIGHBOR(z1, y0, x1, fz * mfy * fx);
        RELION_ADD_NEIGHBOR(z1, y1, x0, fz * fy * mfx);
        RELION_ADD_NEIGHBOR(z1, y1, x1, fz * fy * fx);
#undef RELION_ADD_NEIGHBOR
    }
}

/* RELION's CUDA accelerated projector stores the Fourier reference in CUDA
 * texture objects with cudaFilterModeLinear. Hardware texture interpolation is
 * not bit-identical to the manual no_tex3D trilinear path above. This gated
 * diagnostic path mirrors RELION's texture setup for full complex64 volumes.
 *
 * Axes are transposed for the texture array: recovar stores vol[i0,i1,i2] with
 * i2 fastest, while tex3D's x coordinate addresses the fastest dimension.
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
split_complex_float_kernel(
    const float* __restrict__ vol,
    float* __restrict__ real,
    float* __restrict__ imag,
    int n_voxels)
{
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_voxels) return;
    real[i] = vol[2 * i];
    imag[i] = vol[2 * i + 1];
}

__global__ void __launch_bounds__(BLOCK_SIZE)
split_complex_double_to_float_kernel(
    const double* __restrict__ vol,
    float* __restrict__ real,
    float* __restrict__ imag,
    int n_voxels)
{
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_voxels) return;
    real[i] = (float)vol[2 * i];
    imag[i] = (float)vol[2 * i + 1];
}

__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_half_texture_surface_f32_kernel(
    const float2* __restrict__ projector_half,
    cudaSurfaceObject_t surface,
    int64_t texture_voxels,
    int tex_x,
    int tex_y,
    float projector_scale)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    if (index >= texture_voxels) return;

    const int x = static_cast<int>(index % tex_x);
    const int64_t yz = index / tex_x;
    const int y = static_cast<int>(yz % tex_y);
    const int z = static_cast<int>(yz / tex_y);
    const float2 value = projector_half[index];
    const float2 scaled = make_float2(
        __fmul_rn(value.x, projector_scale),
        __fmul_rn(value.y, projector_scale));
    surf3Dwrite(scaled, surface, x * static_cast<int>(sizeof(float2)), y, z);
}

struct RelionHalfTextureF32 {
    cudaArray_t array = nullptr;
    cudaSurfaceObject_t surface = 0;
    cudaTextureObject_t texture = 0;
};

cudaError_t destroy_relion_half_texture_f32(RelionHalfTextureF32* texture)
{
    cudaError_t err = cudaSuccess;
    if (texture->texture) {
        const cudaError_t cleanup_err = cudaDestroyTextureObject(texture->texture);
        if (err == cudaSuccess) err = cleanup_err;
    }
    if (texture->surface) {
        const cudaError_t cleanup_err = cudaDestroySurfaceObject(texture->surface);
        if (err == cudaSuccess) err = cleanup_err;
    }
    if (texture->array) {
        const cudaError_t cleanup_err = cudaFreeArray(texture->array);
        if (err == cudaSuccess) err = cleanup_err;
    }
    texture->texture = 0;
    texture->surface = 0;
    texture->array = nullptr;
    return err;
}

cudaError_t synchronize_and_destroy_relion_half_texture_f32(
    cudaStream_t stream,
    RelionHalfTextureF32* texture,
    cudaError_t first_err)
{
    cudaError_t err = first_err;
    if (texture->texture || texture->surface || texture->array) {
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = sync_err;
    }
    const cudaError_t cleanup_err = destroy_relion_half_texture_f32(texture);
    if (err == cudaSuccess) err = cleanup_err;
    return err;
}

cudaError_t create_relion_half_texture_f32(
    cudaStream_t stream,
    const float2* projector_half,
    int tex_x,
    int tex_y,
    int tex_z,
    float projector_scale,
    RelionHalfTextureF32* texture)
{
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y),
        static_cast<size_t>(tex_z));
    cudaError_t err = cudaMalloc3DArray(
        &texture->array,
        &desc,
        extent,
        cudaArraySurfaceLoadStore);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(resource));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = texture->array;
    err = cudaCreateSurfaceObject(&texture->surface, &resource);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    const int64_t texture_voxels =
        static_cast<int64_t>(tex_x) * tex_y * tex_z;
    const int64_t block_count =
        (texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_relion_half_texture_surface_f32_kernel<<<
        static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            projector_half,
            texture->surface,
            texture_voxels,
            tex_x,
            tex_y,
            projector_scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return synchronize_and_destroy_relion_half_texture_f32(
            stream,
            texture,
            err);
    }

    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    err = cudaCreateTextureObject(
        &texture->texture, &resource, &texture_desc, nullptr);
    if (err != cudaSuccess) {
        /* The surface-fill kernel is asynchronous.  If texture creation
         * fails, wait for that queued writer before destroying its surface
         * and array.  Preserve the first error while still attempting every
         * cleanup operation. */
        return synchronize_and_destroy_relion_half_texture_f32(
            stream,
            texture,
            err);
    }
    return cudaSuccess;
}

/* A supplied host Projector::data slab should not need an equally large JAX
 * device allocation merely to populate the CUDA array required for hardware
 * interpolation.  The persistent owner below uploads that host slab once and
 * reuses its texture across sparse pass-2 buckets.  Handles are monotonic and
 * resolved through a guarded registry: a compiled FFI executable can retain a
 * stale handle after explicit destruction, but it can never dereference freed
 * storage or alias a later owner. */
struct PersistentRelionHalfTextureF32 {
    RelionHalfTextureF32 texture;
    int tex_x = 0;
    int tex_y = 0;
    int tex_z = 0;
    int device = -1;

    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool closing = false;
    int active_calls = 0;

    bool acquire_call()
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (closing) return false;
        ++active_calls;
        return true;
    }

    void release_call() noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(state_mutex);
            --active_calls;
            if (active_calls == 0) state_changed.notify_all();
        } catch (...) {
            /* A mutex failure during process teardown must not cross an FFI
             * or destructor boundary. */
        }
    }

    void wait_until_idle()
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        closing = true;
        state_changed.wait(lock, [this] { return active_calls == 0; });
    }

    cudaError_t destroy_owned_texture() noexcept
    {
        int original_device = -1;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err == cudaSuccess && original_device != device)
            err = cudaSetDevice(device);
        if (err == cudaSuccess &&
            (texture.texture || texture.surface || texture.array))
            err = cudaDeviceSynchronize();
        const cudaError_t cleanup_err =
            destroy_relion_half_texture_f32(&texture);
        if (err == cudaSuccess) err = cleanup_err;
        if (original_device >= 0 && original_device != device) {
            const cudaError_t restore_err = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore_err;
        }
        return err;
    }

    ~PersistentRelionHalfTextureF32()
    {
        (void)destroy_owned_texture();
    }
};

struct PersistentRelionHalfTextureF32CallGuard {
    std::shared_ptr<PersistentRelionHalfTextureF32> owner;
    bool active = false;

    explicit PersistentRelionHalfTextureF32CallGuard(
        std::shared_ptr<PersistentRelionHalfTextureF32> value)
        noexcept
        : owner(std::move(value))
    {
        try {
            active = owner && owner->acquire_call();
        } catch (...) {
            active = false;
        }
    }

    ~PersistentRelionHalfTextureF32CallGuard()
    {
        if (active) owner->release_call();
    }
};

static std::mutex persistent_relion_half_texture_f32_mutex;
static std::unordered_map<
    uint64_t,
    std::shared_ptr<PersistentRelionHalfTextureF32>>
    persistent_relion_half_texture_f32_registry;
static std::atomic<uint64_t> persistent_relion_half_texture_f32_next_handle{1};

cudaError_t create_relion_half_texture_f32_from_host(
    const float2* projector_half_host,
    int tex_x,
    int tex_y,
    int tex_z,
    RelionHalfTextureF32* texture)
{
    if (!projector_half_host || !texture ||
        tex_x <= 0 || tex_y <= 0 || tex_z <= 0)
        return cudaErrorInvalidValue;

    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y),
        static_cast<size_t>(tex_z));
    cudaError_t err = cudaMalloc3DArray(
        &texture->array,
        &desc,
        extent,
        cudaArrayDefault);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaMemcpy3DParms copy_params;
    memset(&copy_params, 0, sizeof(copy_params));
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float2*>(projector_half_host),
        static_cast<size_t>(tex_x) * sizeof(float2),
        static_cast<size_t>(tex_x),
        static_cast<size_t>(tex_y));
    copy_params.dstArray = texture->array;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyHostToDevice;
    /* The blocking copy is the ownership boundary.  Python may release its
     * host view immediately after the create call returns successfully. */
    err = cudaMemcpy3D(&copy_params);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }

    cudaResourceDesc resource;
    cudaTextureDesc texture_desc;
    memset(&resource, 0, sizeof(resource));
    memset(&texture_desc, 0, sizeof(texture_desc));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = texture->array;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    err = cudaCreateTextureObject(
        &texture->texture,
        &resource,
        &texture_desc,
        nullptr);
    if (err != cudaSuccess) {
        (void)destroy_relion_half_texture_f32(texture);
        return err;
    }
    return cudaSuccess;
}

extern "C" int recovar_relion_persistent_half_texture_f32_create(
    const void* projector_half_host,
    int tex_x,
    int tex_y,
    int tex_z,
    int device,
    float projector_scale,
    uint64_t* owner_handle)
{
    if (!projector_half_host || !owner_handle || projector_scale != 1.0f ||
        !std::isfinite(projector_scale) ||
        tex_x <= 0 || tex_y <= 0 || tex_z <= 0 || device < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    *owner_handle = 0;

    try {
        auto owner = std::make_shared<PersistentRelionHalfTextureF32>();
        owner->tex_x = tex_x;
        owner->tex_y = tex_y;
        owner->tex_z = tex_z;
        owner->device = device;

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (device >= device_count)
            return static_cast<int>(cudaErrorInvalidDevice);
        int original_device = -1;
        err = cudaGetDevice(&original_device);
        if (err != cudaSuccess) return static_cast<int>(err);
        if (original_device != device) err = cudaSetDevice(device);
        if (err == cudaSuccess) {
            err = create_relion_half_texture_f32_from_host(
                static_cast<const float2*>(projector_half_host),
                tex_x,
                tex_y,
                tex_z,
                &owner->texture);
        }
        if (original_device != device) {
            const cudaError_t restore_err = cudaSetDevice(original_device);
            if (err == cudaSuccess) err = restore_err;
        }
        if (err != cudaSuccess) return static_cast<int>(err);

        const uint64_t handle =
            persistent_relion_half_texture_f32_next_handle.fetch_add(
                1, std::memory_order_relaxed);
        if (handle == 0 ||
            handle > static_cast<uint64_t>(
                std::numeric_limits<int64_t>::max()))
            return static_cast<int>(cudaErrorInvalidValue);
        {
            std::lock_guard<std::mutex> lock(
                persistent_relion_half_texture_f32_mutex);
            const auto inserted =
                persistent_relion_half_texture_f32_registry.emplace(
                    handle, owner);
            if (!inserted.second)
                return static_cast<int>(cudaErrorInvalidResourceHandle);
        }
        *owner_handle = handle;
        return static_cast<int>(cudaSuccess);
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}

extern "C" int recovar_relion_persistent_half_texture_f32_destroy(
    uint64_t owner_handle)
{
    if (owner_handle == 0)
        return static_cast<int>(cudaErrorInvalidResourceHandle);
    try {
        std::shared_ptr<PersistentRelionHalfTextureF32> owner;
        {
            std::lock_guard<std::mutex> lock(
                persistent_relion_half_texture_f32_mutex);
            const auto found =
                persistent_relion_half_texture_f32_registry.find(owner_handle);
            if (found == persistent_relion_half_texture_f32_registry.end())
                return static_cast<int>(cudaErrorInvalidResourceHandle);
            owner = found->second;
            persistent_relion_half_texture_f32_registry.erase(found);
        }
        owner->wait_until_idle();
        return static_cast<int>(owner->destroy_owned_texture());
    } catch (const std::bad_alloc&) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    } catch (...) {
        return static_cast<int>(cudaErrorUnknown);
    }
}



template <bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_relion_half_texture_f32_kernel(
    cudaTextureObject_t texture,
    float2* __restrict__ image,
    const float* __restrict__ rotations,
    int n_pixels,
    int image_h,
    int image_w,
    int tex_y_init,
    int tex_z_init,
    int padding_factor,
    int max_r2_padded)
{
    __shared__ float rotation[6];

    const int image_index = blockIdx.x;
    const int pixel = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (threadIdx.x < 6)
        rotation[threadIdx.x] = rotations[image_index * 6 + threadIdx.x];
    __syncthreads();
    if (pixel >= n_pixels) return;

    const int row = pixel / image_w;
    const int column = pixel % image_w;
    const float source_y = static_cast<float>(
        row == 0 ? image_h / 2 : row - image_h / 2);
    const float source_x = HALF_IMG
        ? static_cast<float>(column)
        : static_cast<float>(column - image_w / 2);

    const float model_x =
        (rotation[3] * source_x + rotation[0] * source_y) *
        static_cast<float>(padding_factor);
    const float model_y =
        (rotation[4] * source_x + rotation[1] * source_y) *
        static_cast<float>(padding_factor);
    const float model_z =
        (rotation[5] * source_x + rotation[2] * source_y) *
        static_cast<float>(padding_factor);
    const int64_t output_index =
        static_cast<int64_t>(image_index) * n_pixels + pixel;
    const float radius_squared =
        model_x * model_x + model_y * model_y + model_z * model_z;
    if (!isfinite(radius_squared) || radius_squared >= 2147483648.0f ||
        static_cast<int>(radius_squared) > max_r2_padded) {
        image[output_index] = make_float2(0.0f, 0.0f);
        return;
    }

    float texture_x = model_x;
    float texture_y = model_y;
    float texture_z = model_z;
    float imag_sign = 1.0f;
    if (texture_x < 0.0f) {
        texture_x = -texture_x;
        texture_y = -texture_y;
        texture_z = -texture_z;
        imag_sign = -1.0f;
    }
    float2 value = tex3D<float2>(
        texture,
        texture_x + 0.5f,
        texture_y - static_cast<float>(tex_y_init) + 0.5f,
        texture_z - static_cast<float>(tex_z_init) + 0.5f);
    value.y *= imag_sign;
    image[output_index] = value;
}

cudaError_t launch_relion_projector_half_texture_f32(
    cudaStream_t stream,
    const float2* projector_half,
    float2* image,
    const float* rotations,
    int64_t n_images,
    int64_t image_h,
    int64_t image_w,
    int padding_factor,
    int projector_max_r,
    float projector_scale)
{
    if (n_images == 0 || image_h == 0 || image_w == 0) return cudaSuccess;
    const int padded_max_r = projector_max_r * padding_factor;
    const int image_max_r =
        projector_max_r < image_h / 2 ? projector_max_r : image_h / 2;
    const int padded_image_max_r = image_max_r * padding_factor;
    const int tex_x = padded_max_r + 2;
    const int tex_y = 2 * padded_max_r + 3;
    const int tex_z = 2 * padded_max_r + 3;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t n_pixels = image_h * image_w;
    RelionHalfTextureF32 projector_texture;
    cudaError_t err = create_relion_half_texture_f32(
        stream,
        projector_half,
        tex_x,
        tex_y,
        tex_z,
        projector_scale,
        &projector_texture);
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 grid(
            static_cast<unsigned int>(n_images),
            static_cast<unsigned int>((n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE));
        dim3 block(BLOCK_SIZE);
        project_relion_half_texture_f32_kernel<true><<<grid, block, 0, stream>>>(
            projector_texture.texture,
            image,
            rotations,
            static_cast<int>(n_pixels),
            static_cast<int>(image_h),
            static_cast<int>(image_w),
            tex_y_init,
            tex_z_init,
            padding_factor,
            padded_image_max_r * padded_image_max_r);
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    }

cleanup:
    return synchronize_and_destroy_relion_half_texture_f32(
        stream,
        &projector_texture,
        err);
}

cudaError_t launch_relion_projector_persistent_half_texture_f32(
    cudaStream_t stream,
    const PersistentRelionHalfTextureF32& owner,
    float2* image,
    const float* rotations,
    int64_t n_images,
    int64_t image_h,
    int64_t image_w,
    int padding_factor,
    int projector_max_r)
{
    if (n_images == 0 || image_h == 0 || image_w == 0) return cudaSuccess;
    if (!owner.texture.texture) return cudaErrorInvalidResourceHandle;

    const int padded_max_r = projector_max_r * padding_factor;
    const int image_max_r =
        projector_max_r < image_h / 2 ? projector_max_r : image_h / 2;
    const int padded_image_max_r = image_max_r * padding_factor;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t n_pixels = image_h * image_w;
    dim3 grid(
        static_cast<unsigned int>(n_images),
        static_cast<unsigned int>((n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE));
    dim3 block(BLOCK_SIZE);
    project_relion_half_texture_f32_kernel<true><<<grid, block, 0, stream>>>(
        owner.texture.texture,
        image,
        rotations,
        static_cast<int>(n_pixels),
        static_cast<int>(image_h),
        static_cast<int>(image_w),
        tex_y_init,
        tex_z_init,
        padding_factor,
        padded_image_max_r * padded_image_max_r);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    return err;
}

/* Diagnostic reproduction of the complete per-pixel atomic issue order in
 * RELION's Wavg kernel.  The final axis is [XA, AA, diff2].  Keeping all three
 * atomics in one thread is important: the diff2 atomic after AA delays the
 * second pixel lane (pixel + blockDim.x) exactly as the native kernel does. */
__global__ void __launch_bounds__(256)
relion_wavg_rotation_atomic_triplet_f32_kernel(
    const float* __restrict__ terms,
    float* __restrict__ output,
    int n_rotations,
    int n_pixels)
{
    const int rotation = blockIdx.x;
    const int batch = blockIdx.y;
    for (int pixel = threadIdx.x; pixel < n_pixels; pixel += blockDim.x) {
        const int64_t input_index =
            ((static_cast<int64_t>(batch) * n_rotations + rotation) * n_pixels + pixel) * 3;
        const int64_t output_index =
            (static_cast<int64_t>(batch) * n_pixels + pixel) * 3;
        atomicAdd(&output[output_index], terms[input_index]);
        atomicAdd(&output[output_index + 1], terms[input_index + 1]);
        atomicAdd(&output[output_index + 2], terms[input_index + 2]);
    }
}

cudaError_t launch_relion_wavg_rotation_atomic_triplet_add_f32(
    cudaStream_t stream,
    const float* terms,
    float* output,
    int64_t batch_size,
    int64_t n_rotations,
    int64_t n_pixels)
{
    dim3 grid(static_cast<unsigned>(n_rotations), static_cast<unsigned>(batch_size));
    dim3 block(256);
    relion_wavg_rotation_atomic_triplet_f32_kernel<<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(n_pixels));
    return cudaGetLastError();
}

/* One body for both term layouts.  ``FLAT_ROWS`` replaces the rectangular
 * ``[batch, rotation]`` block address with a packed row whose image comes from
 * ``row_image_ids``, exactly as the flat-row fine scorer does.  The launcher
 * keeps grid.x = row so a flattened rectangular problem issues the same
 * multiset of per-cell atomic adds under the same linear block index
 * ``rotation + batch * n_rotations``.  The order those adds land in is
 * hardware scheduled in both layouts, so the float32 accumulator is bitwise
 * reproducible only for exactly representable summands. */
template <bool FLAT_ROWS = false>
__global__ void __launch_bounds__(256)
relion_wavg_rotation_atomic_runtime_triplet_f32_kernel(
    const float* __restrict__ terms,
    float* __restrict__ output,
    int n_rotations,
    int pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count,
    const int32_t* __restrict__ row_image_ids = nullptr,
    int64_t row_count = 0,
    int64_t batch_size = 0)
{
    const int rotation = FLAT_ROWS ? 0 : static_cast<int>(blockIdx.x);
    const int64_t row = FLAT_ROWS
        ? static_cast<int64_t>(blockIdx.x)
        : static_cast<int64_t>(blockIdx.y) * n_rotations +
              static_cast<int64_t>(blockIdx.x);
    int64_t batch = FLAT_ROWS ? -1 : static_cast<int64_t>(blockIdx.y);
    const int logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : runtime_logical_pixel_count[0];
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (row == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    if constexpr (FLAT_ROWS) {
        if (row >= row_count) return;
        batch = static_cast<int64_t>(row_image_ids[row]);
        /* Padding rows contribute nothing and read no term. */
        if (batch < 0) return;
        if (batch >= batch_size) {
            /* Fail closed on an out-of-range map, as the runtime kernels do. */
            if (threadIdx.x == 0) output[0] = nanf("");
            return;
        }
    }
    for (int pixel = threadIdx.x;
         pixel < logical_pixel_count;
         pixel += blockDim.x) {
        const int64_t input_index = FLAT_ROWS
            ? (row * pixel_capacity + pixel) * 3
            : ((batch * n_rotations + rotation) * pixel_capacity + pixel) * 3;
        const int64_t output_index =
            (batch * pixel_capacity + pixel) * 3;
        atomicAdd(&output[output_index], terms[input_index]);
        atomicAdd(&output[output_index + 1], terms[input_index + 1]);
        atomicAdd(&output[output_index + 2], terms[input_index + 2]);
    }
}

cudaError_t launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    cudaStream_t stream,
    const float* terms,
    float* output,
    int64_t batch_size,
    int64_t n_rotations,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    dim3 grid(static_cast<unsigned>(n_rotations), static_cast<unsigned>(batch_size));
    dim3 block(256);
    relion_wavg_rotation_atomic_runtime_triplet_f32_kernel<false><<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(pixel_capacity),
        runtime_logical_pixel_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
    cudaStream_t stream,
    const float* terms,
    const int32_t* row_image_ids,
    float* output,
    int64_t batch_size,
    int64_t row_count,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    if (row_count == 0) return cudaSuccess;
    dim3 grid(static_cast<unsigned>(row_count));
    dim3 block(256);
    relion_wavg_rotation_atomic_runtime_triplet_f32_kernel<true><<<grid, block, 0, stream>>>(
        terms,
        output,
        0,
        static_cast<int>(pixel_capacity),
        runtime_logical_pixel_count,
        row_image_ids,
        row_count,
        batch_size);
    return cudaGetLastError();
}

/* Keep RELION's per-rotation, per-pixel Wavg translation reduction inside one
 * CUDA thread.  The corresponding JAX reference deliberately uses a
 * translation-order fori_loop; lowering that loop emits multiple loop-body
 * kernel launches for every local-search bucket.  Explicit round-to-nearest
 * operations here retain the same non-contracted float32 arithmetic. */
__global__ void __launch_bounds__(256)
relion_wavg_sequential_triplet_f32_kernel(
    const float2* __restrict__ projections,
    const float* __restrict__ raw_ctf,
    const float* __restrict__ scale,
    const float2* __restrict__ shifted_images,
    const float* __restrict__ posterior,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count)
{
    const int64_t output_row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_count = batch_size * rotation_count * pixel_count;
    if (output_row >= output_count) return;

    const int64_t pixel = output_row % pixel_count;
    const int64_t batch_rotation = output_row / pixel_count;
    const int64_t rotation = batch_rotation % rotation_count;
    const int64_t batch = batch_rotation / rotation_count;
    const float batch_scale = scale[batch];
    const float ctf_with_scale = __fmul_rn(
        raw_ctf[batch * pixel_count + pixel], batch_scale);
    const float2 projection = projections[output_row];
    const float ref_real = __fmul_rn(projection.x, ctf_with_scale);
    const float ref_imag = __fmul_rn(projection.y, ctf_with_scale);
    const float ref_abs2 = __fadd_rn(
        __fmul_rn(ref_real, ref_real),
        __fmul_rn(ref_imag, ref_imag));

    float xa_raw = 0.0f;
    float aa_raw = 0.0f;
    float diff2 = 0.0f;
    const int64_t posterior_base =
        (batch * rotation_count + rotation) * translation_count;
    const int64_t shifted_base = batch * translation_count * pixel_count;
    for (int64_t translation = 0; translation < translation_count; ++translation)
    {
        const float weight = posterior[posterior_base + translation];
        const float2 translated = shifted_images[
            shifted_base + translation * pixel_count + pixel];
        const float diff_real = __fsub_rn(ref_real, translated.x);
        const float diff_imag = __fsub_rn(ref_imag, translated.y);
        const float diff_abs2 = __fadd_rn(
            __fmul_rn(diff_real, diff_real),
            __fmul_rn(diff_imag, diff_imag));
        const float cross = __fadd_rn(
            __fmul_rn(ref_real, translated.x),
            __fmul_rn(ref_imag, translated.y));
        xa_raw = __fadd_rn(xa_raw, __fmul_rn(weight, cross));
        aa_raw = __fadd_rn(aa_raw, __fmul_rn(weight, ref_abs2));
        diff2 = __fadd_rn(diff2, __fmul_rn(weight, diff_abs2));
    }

    const float safe_scale = fmaxf(batch_scale, 1.0e-30f);
    const int64_t output_base = output_row * 3;
    output[output_base] = __fmul_rn(xa_raw, __frcp_rn(safe_scale));
    output[output_base + 1] = __fmul_rn(
        aa_raw, __frcp_rn(__fmul_rn(safe_scale, safe_scale)));
    output[output_base + 2] = diff2;
}

cudaError_t launch_relion_wavg_sequential_triplet_f32(
    cudaStream_t stream,
    const float2* projections,
    const float* raw_ctf,
    const float* scale,
    const float2* shifted_images,
    const float* posterior,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count)
{
    const int64_t output_count = batch_size * rotation_count * pixel_count;
    if (output_count == 0) return cudaSuccess;
    const int blocks = static_cast<int>((output_count + 255) / 256);
    relion_wavg_sequential_triplet_f32_kernel<<<blocks, 256, 0, stream>>>(
        projections,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        output,
        batch_size,
        rotation_count,
        translation_count,
        pixel_count);
    return cudaGetLastError();
}

template <bool INDEXED_RECTANGLE = false, bool FLAT_ROWS = false>
__global__ void __launch_bounds__(256)
relion_wavg_sequential_runtime_triplet_f32_kernel(
    const float2* __restrict__ projections,
    const float* __restrict__ raw_ctf,
    const float* __restrict__ scale,
    const float2* __restrict__ shifted_images,
    const float* __restrict__ posterior,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count,
    const double* __restrict__ full_ctf = nullptr,
    const int32_t* __restrict__ exact_positions = nullptr,
    const int32_t* __restrict__ recon_indices = nullptr,
    int64_t rectangle_capacity = 0,
    int64_t full_pixel_count = 0,
    const int32_t* __restrict__ invalid = nullptr,
    const int32_t* __restrict__ row_image_ids = nullptr,
    int64_t row_count = 0)
{
    /* ``FLAT_ROWS`` packs the rectangular ``[batch, rotation]`` grid into one
     * ``[row]`` axis whose image address is ``row_image_ids[row]``, the same
     * substitution the flat-row fine scorer makes.  Every arithmetic statement
     * below, the translation order included, is the rectangular body. */
    static_assert(!(INDEXED_RECTANGLE && FLAT_ROWS),
                  "indexed-rectangle and flat-row addressing are exclusive");
    const int64_t rotation = FLAT_ROWS ? 0 : static_cast<int64_t>(blockIdx.x);
    const int64_t row = FLAT_ROWS
        ? static_cast<int64_t>(blockIdx.x)
        : static_cast<int64_t>(blockIdx.y) * rotation_count +
              static_cast<int64_t>(blockIdx.x);
    int64_t batch = FLAT_ROWS ? -1 : static_cast<int64_t>(blockIdx.y);
    if constexpr (FLAT_ROWS) {
        if (row >= row_count) return;
    } else {
        if (batch >= batch_size || rotation >= rotation_count) return;
    }
    if constexpr (INDEXED_RECTANGLE) { if (*invalid) return; }
    const int64_t logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixel_count[0]);
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (row == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    if constexpr (FLAT_ROWS) {
        batch = static_cast<int64_t>(row_image_ids[row]);
        /* Padding rows read nothing and keep the zeros the launcher wrote. */
        if (batch < 0) return;
        if (batch >= batch_size) {
            /* Fail closed on an out-of-range map, as the runtime kernels do. */
            for (int64_t pixel = threadIdx.x;
                 pixel < pixel_capacity;
                 pixel += blockDim.x) {
                const int64_t invalid_base = (row * pixel_capacity + pixel) * 3;
                output[invalid_base] = nanf("");
                output[invalid_base + 1] = nanf("");
                output[invalid_base + 2] = nanf("");
            }
            return;
        }
    }
    const float batch_scale = scale[batch];
    const int64_t posterior_base = FLAT_ROWS
        ? row * translation_count
        : (batch * rotation_count + rotation) * translation_count;
    const int64_t shifted_stride = INDEXED_RECTANGLE ? rectangle_capacity : pixel_capacity;
    const int64_t shifted_base = batch * translation_count * shifted_stride;
    const int64_t projection_base = FLAT_ROWS
        ? row * pixel_capacity
        : (batch * rotation_count + rotation) * pixel_capacity;
    for (int64_t pixel = threadIdx.x;
         pixel < (INDEXED_RECTANGLE ? pixel_capacity : logical_pixel_count);
         pixel += blockDim.x)
    {
        const int64_t stored_pixel = INDEXED_RECTANGLE ? exact_positions[pixel] : pixel;
        const int64_t output_base = INDEXED_RECTANGLE
            ? ((batch * rotation_count + rotation) * rectangle_capacity + stored_pixel) * 3
            : (projection_base + pixel) * 3;
        if constexpr (INDEXED_RECTANGLE) {
            if (pixel >= logical_pixel_count) {
                output[output_base] = 0.0f;
                output[output_base + 1] = 0.0f;
                output[output_base + 2] = 0.0f;
                continue;
            }
        }
        const float ctf_value = INDEXED_RECTANGLE
            ? __double2float_rn(full_ctf[batch * full_pixel_count + recon_indices[pixel]])
            : raw_ctf[batch * pixel_capacity + pixel];
        const float ctf_with_scale = __fmul_rn(ctf_value, batch_scale);
        const float2 projection = projections[projection_base + pixel];
        const float ref_real = __fmul_rn(projection.x, ctf_with_scale);
        const float ref_imag = __fmul_rn(projection.y, ctf_with_scale);
        const float ref_abs2 = __fadd_rn(
            __fmul_rn(ref_real, ref_real),
            __fmul_rn(ref_imag, ref_imag));

        float xa_raw = 0.0f;
        float aa_raw = 0.0f;
        float diff2 = 0.0f;
        for (int64_t translation = 0;
             translation < translation_count;
             ++translation)
        {
            const float weight = posterior[posterior_base + translation];
            const float2 translated = shifted_images[
                shifted_base + translation * shifted_stride + stored_pixel];
            const float diff_real = __fsub_rn(ref_real, translated.x);
            const float diff_imag = __fsub_rn(ref_imag, translated.y);
            const float diff_abs2 = __fadd_rn(
                __fmul_rn(diff_real, diff_real),
                __fmul_rn(diff_imag, diff_imag));
            const float cross = __fadd_rn(
                __fmul_rn(ref_real, translated.x),
                __fmul_rn(ref_imag, translated.y));
            xa_raw = __fadd_rn(xa_raw, __fmul_rn(weight, cross));
            aa_raw = __fadd_rn(aa_raw, __fmul_rn(weight, ref_abs2));
            diff2 = __fadd_rn(diff2, __fmul_rn(weight, diff_abs2));
        }

        const float safe_scale = fmaxf(batch_scale, 1.0e-30f);
        output[output_base] = __fmul_rn(xa_raw, __frcp_rn(safe_scale));
        output[output_base + 1] = __fmul_rn(
            aa_raw, __frcp_rn(__fmul_rn(safe_scale, safe_scale)));
        output[output_base + 2] = diff2;
    }
}

cudaError_t launch_relion_wavg_sequential_runtime_triplet_f32(
    cudaStream_t stream,
    const float2* projections,
    const float* raw_ctf,
    const float* scale,
    const float2* shifted_images,
    const float* posterior,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    const int64_t output_count = batch_size * rotation_count * pixel_capacity;
    if (output_count == 0) return cudaSuccess;
    cudaError_t err = cudaMemsetAsync(
        output,
        0,
        static_cast<size_t>(output_count) * 3 * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    dim3 grid(
        static_cast<unsigned>(rotation_count),
        static_cast<unsigned>(batch_size));
    relion_wavg_sequential_runtime_triplet_f32_kernel<false><<<grid, 256, 0, stream>>>(
        projections,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        output,
        batch_size,
        rotation_count,
        translation_count,
        pixel_capacity,
        runtime_logical_pixel_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_wavg_sequential_runtime_flat_rows_triplet_f32(
    cudaStream_t stream,
    const float2* projections,
    const int32_t* row_image_ids,
    const float* raw_ctf,
    const float* scale,
    const float2* shifted_images,
    const float* posterior,
    float* output,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    const int32_t* runtime_logical_pixel_count)
{
    const int64_t output_count = row_count * pixel_capacity;
    if (output_count == 0) return cudaSuccess;
    /* The zero fill owns the physical pixel tail and every padding row; the
     * kernel then writes only the rows and pixels it is allowed to read. */
    cudaError_t err = cudaMemsetAsync(
        output,
        0,
        static_cast<size_t>(output_count) * 3 * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    dim3 grid(static_cast<unsigned>(row_count));
    relion_wavg_sequential_runtime_triplet_f32_kernel<false, true><<<grid, 256, 0, stream>>>(
        projections,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        output,
        batch_size,
        0,
        translation_count,
        pixel_capacity,
        runtime_logical_pixel_count,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        nullptr,
        row_image_ids,
        row_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_fused_x_half_backproject(
    cudaStream_t stream,
    float2* data_volume,
    float* weight_volume,
    const float2* data_rows,
    const float* weight_rows,
    const int32_t* pixel_indices,
    const float* rot,
    int64_t n_rows,
    int64_t n_pixels,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const float c0 = (float)(N0 / 2);
    const float c1 = (float)(N1 / 2);
    const float c2 = (float)(N2 / 2);
    const float max_r2 = (float)max_r2_x4 / 4.0f;
    dim3 grid((unsigned)n_rows, 1);
    dim3 block(128);
    relion_fused_x_half_backproject_kernel<float, float2, false, true><<<grid, block, 0, stream>>>(
        data_volume, weight_volume, data_rows, weight_rows, pixel_indices, rot,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (int)n_pixels, (int)image_h, (int)image_w,
        (int)N0, (int)N1, N2_eff, c0, c1, c2,
        (int)upsampling, max_r2, (int)n_rows);
    return cudaGetLastError();
}

cudaError_t launch_relion_fused_x_half_backproject_f64(
    cudaStream_t stream,
    double2* data_volume,
    double* weight_volume,
    const double2* data_rows,
    const double* weight_rows,
    const int32_t* pixel_indices,
    const double* rot,
    int64_t n_rows,
    int64_t n_pixels,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const double c0 = double(N0 / 2);
    const double c1 = double(N1 / 2);
    const double c2 = double(N2 / 2);
    const double max_r2 = double(max_r2_x4) / 4.0;
    dim3 grid((unsigned)n_rows, 1);
    dim3 block(128);
    relion_fused_x_half_backproject_kernel<double, double2, false, true><<<grid, block, 0, stream>>>(
        data_volume, weight_volume, data_rows, weight_rows, pixel_indices, rot,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        (int)n_pixels, (int)image_h, (int)image_w,
        (int)N0, (int)N1, N2_eff, c0, c1, c2,
        (int)upsampling, max_r2, (int)n_rows);
    return cudaGetLastError();
}

cudaError_t launch_relion_firstiter_bpref_fused_x_half(
    cudaStream_t stream,
    float* data_volume_real,
    float* data_volume_imag,
    float* weight_volume,
    const float* image_real,
    const float* image_imag,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior,
    const float* translation_x,
    const float* translation_y,
    const float* native_eulers,
    float significant_weight,
    float weight_norm,
    int64_t n_rotations,
    int64_t n_translations,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    dim3 grid((unsigned)n_rotations, 1);
    dim3 block(128);
    relion_firstiter_bpref_fused_x_half_kernel<<<grid, block, 0, stream>>>(
        image_real, image_imag, translation_x, translation_y,
        posterior, minvsigma2, ctf, (unsigned long)n_translations,
        significant_weight, weight_norm, native_eulers,
        data_volume_real, data_volume_imag, weight_volume,
        (int)(max_r2_x4 / (4 * upsampling * upsampling)), (float)upsampling,
        (unsigned)image_w, (unsigned)image_h,
        (unsigned)(image_h * image_w),
        (unsigned)N2_eff, (unsigned)N1,
        -(int)(N1 / 2), -(int)(N0 / 2));
    return cudaGetLastError();
}


cudaError_t launch_relion_fused_x_half_backproject_with_signature(
    cudaStream_t stream,
    float2* data_volume,
    float* weight_volume,
    const float2* data_rows,
    const float* weight_rows,
    const int32_t* pixel_indices,
    const float* rot,
    const int32_t* canonical_rotation_keys,
    const int32_t* signature_row_indices,
    int32_t* signature_rotation_keys,
    int32_t* signature_pixel_indices,
    int32_t* signature_row_flags,
    float* signature_source_values,
    int32_t* signature_neighbor_indices,
    float* signature_neighbor_coefficients,
    int32_t* signature_neighbor_flags,
    float2* accumulator_shadow_data,
    float* accumulator_shadow_weight,
    float2* operand_shadow_data_rows,
    float* operand_shadow_weight_rows,
    int32_t* operand_shadow_pixel_indices,
    float* operand_shadow_rot,
    int32_t* operand_shadow_canonical_rotation_keys,
    int32_t* operand_shadow_signature_row_indices,
    int64_t n_rows,
    int64_t n_signature_rows,
    int64_t n_pixels,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const float c0 = (float)(N0 / 2);
    const float c1 = (float)(N1 / 2);
    const float c2 = (float)(N2 / 2);
    const float max_r2 = (float)max_r2_x4 / 4.0f;
    dim3 signature_grid((unsigned)n_signature_rows, 1);
    dim3 block(128);
    cudaError_t err = launch_relion_fused_x_half_backproject(
        stream, data_volume, weight_volume, data_rows, weight_rows,
        pixel_indices, rot, n_rows, n_pixels, image_h, image_w,
        N0, N1, N2, upsampling, max_r2_x4);
    if (err != cudaSuccess) return err;
    const size_t volume_size = (size_t)N0 * (size_t)N1 * (size_t)N2_eff;
    err = cudaMemcpyAsync(accumulator_shadow_data, data_volume,
                          volume_size * sizeof(float2), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(accumulator_shadow_weight, weight_volume,
                          volume_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_data_rows, data_rows,
                          (size_t)n_rows * (size_t)n_pixels * sizeof(float2),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_weight_rows, weight_rows,
                          (size_t)n_rows * (size_t)n_pixels * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_pixel_indices, pixel_indices,
                          (size_t)n_pixels * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_rot, rot,
                          (size_t)n_rows * 6 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_canonical_rotation_keys, canonical_rotation_keys,
                          (size_t)n_rows * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_signature_row_indices, signature_row_indices,
                          (size_t)n_signature_rows * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    relion_fused_x_half_backproject_kernel<float, float2, true, false><<<signature_grid, block, 0, stream>>>(
        data_volume, weight_volume, data_rows, weight_rows, pixel_indices, rot,
        canonical_rotation_keys, signature_row_indices,
        signature_rotation_keys, signature_pixel_indices, signature_row_flags,
        signature_source_values, signature_neighbor_indices,
        signature_neighbor_coefficients, signature_neighbor_flags,
        (int)n_pixels, (int)image_h, (int)image_w,
        (int)N0, (int)N1, N2_eff, c0, c1, c2,
        (int)upsampling, max_r2, (int)n_rows);
    return cudaGetLastError();
}

/* ================================================================== */
/*                    XLA  FFI  handlers                               */
/* ================================================================== */

namespace {

constexpr int kRelionPreprocessBlockSize = 128;
constexpr int kRelionSoftMaskBlocks = 128;
constexpr int kRelionEulerBlockSize = 128;
constexpr int kRelionTranslateScoreBlockSize = 256;
constexpr int kRelionTranslateBprefBlockSize = 256;
constexpr int kRelionBprefOperandsBlockSize = 128;
constexpr int kRelionCoarseDiff2BlockSize = 128;
constexpr int kRelionCoarseEulersPerBlock = 16;
constexpr int kRelionCoarsePrefetchFraction = 4;
constexpr int kRelionFineDiff2BlockSize = 256;
// The deployed REF3D fine kernel is instantiated with a seven-translation
// shared-memory capacity. Its makeJobsForDiff2Fine call nevertheless uses
// D2F_CHUNK_DATA3D=4 when refIs3D, so native jobs contain at most four
// translations. Preserve both observable constants here.
constexpr int kRelionFineDiff2TranslationCapacity = 7;
constexpr int kRelionFineDiff2Ref3dJobChunk = 4;
constexpr int kRelionPowerClassBlockSize = 128;

template <typename T, bool DoRight>
__global__ void relion_make_scoring_rotations_kernel(
    const T* eulers_deg,
    const T* right_matrix,
    T* scorer_rotations,
    int64_t orientation_count)
{
    int64_t oid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (oid >= orientation_count) return;

    T a = eulers_deg[3 * oid] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T b = eulers_deg[3 * oid + 1] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T g = eulers_deg[3 * oid + 2] * static_cast<T>(3.14159265358979323846) /
          static_cast<T>(180.0);
    T ca, sa, cb, sb, cg, sg, cc, cs, sc, ss;
    T A[9], B[9];
    if constexpr (std::is_same_v<T, float>) {
        sincosf(a, &sa, &ca);
        sincosf(b, &sb, &cb);
        sincosf(g, &sg, &cg);
    } else {
        sincos(a, &sa, &ca);
        sincos(b, &sb, &cb);
        sincos(g, &sg, &cg);
    }
    cc = cb * ca;
    cs = cb * sa;
    sc = sb * ca;
    ss = sb * sa;
    A[0] = cg * cc - sg * sa;
    A[1] = cg * cs + sg * ca;
    A[2] = -cg * sb;
    A[3] = -sg * cc - cg * sa;
    A[4] = -sg * cs + cg * ca;
    A[5] = sg * sb;
    A[6] = sc;
    A[7] = ss;
    A[8] = cb;

    if constexpr (DoRight) {
        for (int i = 0; i < 9; ++i) B[i] = static_cast<T>(0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    B[3 * i + j] += A[3 * i + k] * right_matrix[3 * k + j];
    } else {
        for (int i = 0; i < 9; ++i) B[i] = A[i];
    }

    for (int i = 0; i < 9; ++i) scorer_rotations[9 * oid + i] = B[i];
}

template <typename T, bool DoRight>
cudaError_t launch_relion_make_scoring_rotations(
    cudaStream_t stream,
    const T* eulers_deg,
    const T* right_matrix,
    T* scorer_rotations,
    int64_t orientation_count)
{
    if (orientation_count == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (orientation_count + kRelionEulerBlockSize - 1) / kRelionEulerBlockSize);
    relion_make_scoring_rotations_kernel<T, DoRight>
        <<<blocks, kRelionEulerBlockSize, 0, stream>>>(
            eulers_deg, right_matrix, scorer_rotations, orientation_count);
    return cudaGetLastError();
}

__device__ __forceinline__ float2 relion_score_translate_f32(
    float2 value,
    int x,
    int y,
    float tx,
    float ty)
{
    // Match the PTX emitted for the deployed RELION fine Gaussian kernel.
    // Its phase is one rounded y product followed by x FMA.  The real
    // component is contracted by the H100 driver JIT as cosine*real plus the
    // rounded negative sine*imaginary product.  The imaginary component uses
    // the complementary ordering: a rounded cosine*imaginary product is the
    // addend to sine*real.  Writing both source expressions directly lets
    // newer offline nvcc versions choose a different addend for the
    // imaginary FMA, which changes translated pixels and can move the final
    // diff2 by one binary32 ULP.
    const float phase = __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    const float translated_real = __fmaf_rn(
        cosine, value.x,
        -__fmul_rn(sine, value.y));
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

__device__ __forceinline__ float2 relion_coarse_score_translate_f32(
    float2 value,
    int x,
    int y,
    float tx,
    float ty)
{
    // Match the PTX emitted for RELION's REF3D/DATA2D coarse Gaussian
    // scorer.  Its phase uses a rounded y product followed by an x FMA.
    // Unlike the fine scorer, the real component must retain RELION's source
    // expression so the deployed compiler chooses the production operation
    // sequence in this kernel context. The imaginary component is contracted
    // with the rounded cosine*imaginary product as the addend.
    const float phase = __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    const float translated_real = cosine * value.x - sine * value.y;
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

// RELION's row label for a half image it is not cropping.  ``fftw.h:99-109``
// sets ``ip = (i < XSIZE) ? i : i - YSIZE`` with ``XSIZE`` the half width, so
// the Nyquist row of an uncropped half image is ``+N/2``; recovar's centered
// packed layout stores that same physical row at ``ky = -N/2``.  Every scoring
// kernel in ``relion_scoring.cuh`` already walks RELION's layout and therefore
// derives ``+N/2`` itself (``if (y > cs/2) y -= cs``); the translate kernels
// below take centered indices instead and must convert.  The two labels differ
// only for non-integer shifts, where ``exp(-i*pi*dy)`` and ``exp(+i*pi*dy)``
// are conjugates.  A cropped window never contains that row.
__device__ __forceinline__ int relion_centered_row_to_relion_label(
    int centered_row,
    int image_h)
{
    return centered_row == -(image_h / 2) ? image_h / 2 : centered_row;
}

__global__ void relion_translate_score_f32_kernel(
    const float2* images,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    float tx = translation_angles[2 * translation];
    float ty = translation_angles[2 * translation + 1];
    float2 value = images[image * pixel_count + pixel_row];
    shifted[flat] = relion_score_translate_f32(value, x, y, tx, ty);
}

cudaError_t launch_relion_translate_score_f32(
    cudaStream_t stream,
    const float2* images,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateScoreBlockSize - 1) /
        kRelionTranslateScoreBlockSize);
    relion_translate_score_f32_kernel<<<
        blocks, kRelionTranslateScoreBlockSize, 0, stream>>>(
            images,
            translation_angles,
            pixel_indices,
            shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

__global__ void relion_translate_score_f64_kernel(
    const double2* images,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    double tx = translation_angles[2 * translation];
    double ty = translation_angles[2 * translation + 1];
    double sine;
    double cosine;
    sincos(x * tx + y * ty, &sine, &cosine);

    double2 value = images[image * pixel_count + pixel_row];
    shifted[flat] = make_double2(
        cosine * value.x - sine * value.y,
        cosine * value.y + sine * value.x);
}

cudaError_t launch_relion_translate_score_f64(
    cudaStream_t stream,
    const double2* images,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateScoreBlockSize - 1) /
        kRelionTranslateScoreBlockSize);
    relion_translate_score_f64_kernel<<<
        blocks, kRelionTranslateScoreBlockSize, 0, stream>>>(
            images,
            translation_angles,
            pixel_indices,
            shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

__global__ void relion_translate_bpref_f32_kernel(
    const float2* images,
    const float* weighted_ctf,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    float tx = translation_angles[2 * translation];
    float ty = translation_angles[2 * translation + 1];
    float sine;
    float cosine;
    sincosf(x * tx + y * ty, &sine, &cosine);

    float2 value = images[image * pixel_count + pixel_row];
    float factor = weighted_ctf[image * pixel_count + pixel_row];
    float translated_real = cosine * value.x - sine * value.y;
    // Pin the RELION BPref translatePixel FMA order (captured-bit oracle).
    float translated_imag = __fmaf_rn(
        sine, value.x, __fmul_rn(cosine, value.y));
    weighted_shifted[flat] = make_float2(
        translated_real * factor,
        translated_imag * factor);
}

cudaError_t launch_relion_translate_bpref_f32(
    cudaStream_t stream,
    const float2* images,
    const float* weighted_ctf,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateBprefBlockSize - 1) /
        kRelionTranslateBprefBlockSize);
    relion_translate_bpref_f32_kernel<<<
        blocks, kRelionTranslateBprefBlockSize, 0, stream>>>(
            images,
            weighted_ctf,
            translation_angles,
            pixel_indices,
            weighted_shifted,
            batch_size,
            translation_count,
            pixel_count,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

__global__ void relion_translate_bpref_f64_kernel(
    const double2* images,
    const double* weighted_ctf,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * translation_count * pixel_count;
    if (flat >= total) return;

    int64_t pixel_row = flat % pixel_count;
    int64_t batch_translation = flat / pixel_count;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    int pixel_index = pixel_indices[pixel_row];
    int x = pixel_index % image_half_width;
    int y = relion_centered_row_to_relion_label(
        pixel_index / image_half_width - image_h / 2, image_h);
    double tx = translation_angles[2 * translation];
    double ty = translation_angles[2 * translation + 1];
    double sine;
    double cosine;
    sincos(x * tx + y * ty, &sine, &cosine);

    double2 value = images[image * pixel_count + pixel_row];
    double factor = weighted_ctf[image * pixel_count + pixel_row];
    double translated_real = cosine * value.x - sine * value.y;
    double translated_imag = cosine * value.y + sine * value.x;
    weighted_shifted[flat] = make_double2(
        translated_real * factor,
        translated_imag * factor);
}

cudaError_t launch_relion_translate_bpref_f64(
    cudaStream_t stream,
    const double2* images,
    const double* weighted_ctf,
    const double* translation_angles,
    const int32_t* pixel_indices,
    double2* weighted_shifted,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(
        (total + kRelionTranslateBprefBlockSize - 1) /
        kRelionTranslateBprefBlockSize);
    relion_translate_bpref_f64_kernel<<<
        blocks, kRelionTranslateBprefBlockSize, 0, stream>>>(
            images, weighted_ctf, translation_angles, pixel_indices,
            weighted_shifted, batch_size, translation_count, pixel_count,
            image_h, image_half_width);
    return cudaGetLastError();
}

__global__ void relion_vdam_mstep_sums_f32_kernel(
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const float2* reference,
    float2* numerator_sum,
    float* denominator_sum,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t image_rotation = static_cast<int64_t>(blockIdx.x);
    if (image_rotation >= batch_size * rotation_count) return;
    int64_t image = image_rotation / rotation_count;
    const int passes = ceilf(
        static_cast<float>(pixel_count) /
        static_cast<float>(kRelionBprefOperandsBlockSize));
    for (unsigned pass = 0; pass < static_cast<unsigned>(passes); ++pass)
    {
        int64_t pixel_row =
            static_cast<int64_t>(pass) * kRelionBprefOperandsBlockSize +
            threadIdx.x;
        if (pixel_row >= pixel_count) continue;

        int64_t image_pixel = image * pixel_count + pixel_row;
        int64_t output = image_rotation * pixel_count + pixel_row;
        int pixel_index = pixel_indices[pixel_row];
        int x = pixel_index % image_half_width;
        // RECOVAR's centered half-rFFT stores ky=0 in row image_h/2,
        // exactly as relion_translate_bpref_f32_kernel expects, and its
        // Nyquist row at ky=-N/2 carries RELION's +N/2 label.
        int y = relion_centered_row_to_relion_label(
            pixel_index / image_half_width - image_h / 2, image_h);

        float2 image_value = images[image_pixel];
        float image_ctf = ctf[image_pixel];
        float image_minvsigma2 = minvsigma2[image_pixel];
        float2 reference_value = reference[output];
        // BP.cuh multiplies Fref by CTF once, before its translation loop.
        float reference_real = reference_value.x;
        float reference_imag = reference_value.y;
        reference_real *= image_ctf;
        reference_imag *= image_ctf;

        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        float fweight = 0.0f;
        int64_t posterior_base = image_rotation * translation_count;
        for (int64_t translation = 0; translation < translation_count;
             ++translation)
        {
            // Keep these as separate source statements: this is the exact
            // left-associated order used by cuda_kernel_backproject3D_SGD.
            float weight = posterior_over_weight_norm[
                posterior_base + translation];
            weight = weight * image_ctf * image_minvsigma2;
            fweight += weight * image_ctf;

            float tx = translation_angles[2 * translation];
            float ty = translation_angles[2 * translation + 1];
            float phase = x * tx + y * ty;
            float sine;
            float cosine;
            sincosf(phase, &sine, &cosine);
            float translated_real =
                cosine * image_value.x - sine * image_value.y;
            float translated_imag =
                cosine * image_value.y + sine * image_value.x;
            sum_real += (translated_real - reference_real) * weight;
            sum_imag += (translated_imag - reference_imag) * weight;
        }
        numerator_sum[output] = make_float2(sum_real, sum_imag);
        denominator_sum[output] = fweight;
    }
}

cudaError_t launch_relion_vdam_mstep_sums_f32(
    cudaStream_t stream,
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const float2* reference,
    float2* numerator_sum,
    float* denominator_sum,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    if (batch_size * rotation_count * pixel_count == 0) return cudaSuccess;
    int blocks = static_cast<int>(batch_size * rotation_count);
    relion_vdam_mstep_sums_f32_kernel<<<
        blocks, kRelionBprefOperandsBlockSize, 0, stream>>>(
            images, ctf, minvsigma2, posterior_over_weight_norm,
            translation_angles, pixel_indices, reference, numerator_sum,
            denominator_sum, batch_size, rotation_count, translation_count,
            pixel_count, image_h, image_half_width);
    return cudaGetLastError();
}

__device__ __forceinline__ float2 relion_vdam_project_texture_f32(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    int x,
    int y,
    const float* euler,
    int padding_factor,
    int max_r2_padded,
    int tex_y_init,
    int tex_z_init,
    float projector_scale)
{
    float xp = (euler[0] * x + euler[1] * y) * padding_factor;
    float yp = (euler[3] * x + euler[4] * y) * padding_factor;
    float zp = (euler[6] * x + euler[7] * y) * padding_factor;
    const int r2 = static_cast<int>(xp * xp + yp * yp + zp * zp);
    if (r2 > max_r2_padded) return make_float2(0.0f, 0.0f);
    float imag_sign = 1.0f;
    if (xp < 0.0f)
    {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag_sign = -1.0f;
    }
    return make_float2(
        projector_scale * tex3D<float>(
            tex_real,
            xp + 0.5f,
            yp - static_cast<float>(tex_y_init) + 0.5f,
            zp - static_cast<float>(tex_z_init) + 0.5f),
        projector_scale * imag_sign * tex3D<float>(
            tex_imag,
            xp + 0.5f,
            yp - static_cast<float>(tex_y_init) + 0.5f,
            zp - static_cast<float>(tex_z_init) + 0.5f));
}

}  // namespace

#include "relion_vdam_mstep.cuh"

namespace {

#include "relion_scoring.cuh"

#include "relion_preprocess.cuh"

}  // namespace

ffi::Error RelionMakeScoringRotationsF32Impl(
    cudaStream_t stream,
    int64_t do_right,
    ffi::AnyBuffer eulers_deg,
    ffi::AnyBuffer right_matrix,
    ffi::Result<ffi::AnyBuffer> scorer_rotations)
{
    if (eulers_deg.element_type() != ffi::DataType::F32 ||
        right_matrix.element_type() != ffi::DataType::F32 ||
        scorer_rotations->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: inputs/output must be F32");
    if (do_right != 0 && do_right != 1)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: do_right must be 0 or 1");

    auto euler_dims = eulers_deg.dimensions();
    auto right_dims = right_matrix.dimensions();
    auto output_dims = scorer_rotations->dimensions();
    if (euler_dims.size() != 2 || euler_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: eulers_deg must have shape (N,3)");
    if (right_dims.size() != 2 || right_dims[0] != 3 || right_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: right_matrix must have shape (3,3)");
    if (output_dims.size() != 3 || output_dims[0] != euler_dims[0] ||
        output_dims[1] != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF32: output must have shape (N,3,3)");

    const float* eulers_ptr = static_cast<const float*>(eulers_deg.untyped_data());
    const float* right_ptr = static_cast<const float*>(right_matrix.untyped_data());
    float* output_ptr = static_cast<float*>(scorer_rotations->untyped_data());
    cudaError_t err = do_right
        ? launch_relion_make_scoring_rotations<float, true>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0])
        : launch_relion_make_scoring_rotations<float, false>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionVdamMstepFusedProjectorXHalfCommon(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t pixel_capacity,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int64_t physical_image_size,
    int64_t projector_max_r,
    int64_t projection_padding_factor,
    int64_t reconstruction_group_count,
    int64_t parallel_worker_replay,
    int64_t captured_rotation_replay,
    int64_t serial_rotation_replay,
    int64_t float64_accumulator_replay,
    int64_t reverse_rotation_replay,
    int64_t rotation_replay_stride,
    int64_t native_trace_shape_replay,
    int64_t captured_particle_timing_replay,
    int64_t candidate_trace_active,
    int64_t particle_tail_mask,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer reconstruction_group_ids,
    ffi::AnyBuffer worker_lane_ids,
    ffi::AnyBuffer particle_trace_ids,
    ffi::AnyBuffer rotation_replay_order,
    ffi::AnyBuffer rotation_replay_counts,
    ffi::AnyBuffer particle_start_offsets_ns,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum,
    const ffi::AnyBuffer* runtime_projector_radius = nullptr);

ffi::Error RelionVdamMstepFusedProjectorXHalfImpl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t pixel_capacity,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int64_t physical_image_size,
    int64_t projector_max_r,
    int64_t projection_padding_factor,
    int64_t reconstruction_group_count,
    int64_t parallel_worker_replay,
    int64_t captured_rotation_replay,
    int64_t serial_rotation_replay,
    int64_t float64_accumulator_replay,
    int64_t reverse_rotation_replay,
    int64_t rotation_replay_stride,
    int64_t native_trace_shape_replay,
    int64_t captured_particle_timing_replay,
    int64_t candidate_trace_active,
    int64_t particle_tail_mask,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer reconstruction_group_ids,
    ffi::AnyBuffer worker_lane_ids,
    ffi::AnyBuffer particle_trace_ids,
    ffi::AnyBuffer rotation_replay_order,
    ffi::AnyBuffer rotation_replay_counts,
    ffi::AnyBuffer particle_start_offsets_ns,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    return RelionVdamMstepFusedProjectorXHalfCommon(
        stream, image_h, image_w, pixel_capacity, N0, N1, N2, upsampling,
        max_r2_x4, physical_image_size, projector_max_r,
        projection_padding_factor, reconstruction_group_count,
        parallel_worker_replay, captured_rotation_replay,
        serial_rotation_replay, float64_accumulator_replay,
        reverse_rotation_replay, rotation_replay_stride,
        native_trace_shape_replay, captured_particle_timing_replay,
        candidate_trace_active, particle_tail_mask, projector_full, images, ctf, minvsigma2,
        posterior, translation_angles, eulers, rot, reconstruction_group_ids,
        worker_lane_ids, particle_trace_ids, rotation_replay_order,
        rotation_replay_counts, particle_start_offsets_ns,
        data_real_volume_in, data_imag_volume_in, weight_volume_in, nullptr,
        data_real_volume_out, data_imag_volume_out, weight_volume_out,
        denominator_sum);
}

ffi::Error RelionVdamMstepFusedProjectorRuntimeXHalfImpl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t pixel_capacity,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int64_t physical_image_size,
    int64_t projector_max_r,
    int64_t projection_padding_factor,
    int64_t reconstruction_group_count,
    int64_t parallel_worker_replay,
    int64_t captured_rotation_replay,
    int64_t serial_rotation_replay,
    int64_t float64_accumulator_replay,
    int64_t reverse_rotation_replay,
    int64_t rotation_replay_stride,
    int64_t native_trace_shape_replay,
    int64_t captured_particle_timing_replay,
    int64_t candidate_trace_active,
    int64_t particle_tail_mask,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer reconstruction_group_ids,
    ffi::AnyBuffer worker_lane_ids,
    ffi::AnyBuffer particle_trace_ids,
    ffi::AnyBuffer rotation_replay_order,
    ffi::AnyBuffer rotation_replay_counts,
    ffi::AnyBuffer particle_start_offsets_ns,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    return RelionVdamMstepFusedProjectorXHalfCommon(
        stream, image_h, image_w, pixel_capacity, N0, N1, N2, upsampling,
        max_r2_x4, physical_image_size, projector_max_r,
        projection_padding_factor, reconstruction_group_count,
        parallel_worker_replay, captured_rotation_replay,
        serial_rotation_replay, float64_accumulator_replay,
        reverse_rotation_replay, rotation_replay_stride,
        native_trace_shape_replay, captured_particle_timing_replay,
        candidate_trace_active, particle_tail_mask, projector_full, images, ctf, minvsigma2,
        posterior, translation_angles, eulers, rot, reconstruction_group_ids,
        worker_lane_ids, particle_trace_ids, rotation_replay_order,
        rotation_replay_counts, particle_start_offsets_ns,
        data_real_volume_in, data_imag_volume_in, weight_volume_in,
        &logical_current_size, data_real_volume_out, data_imag_volume_out,
        weight_volume_out, denominator_sum);
}

ffi::Error RelionVdamMstepFusedProjectorCapacityXHalfImpl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t pixel_capacity,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int64_t physical_image_size,
    int64_t projector_max_r,
    int64_t projection_padding_factor,
    int64_t reconstruction_group_count,
    int64_t parallel_worker_replay,
    int64_t captured_rotation_replay,
    int64_t serial_rotation_replay,
    int64_t float64_accumulator_replay,
    int64_t reverse_rotation_replay,
    int64_t rotation_replay_stride,
    int64_t native_trace_shape_replay,
    int64_t captured_particle_timing_replay,
    int64_t candidate_trace_active,
    int64_t particle_tail_mask,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer reconstruction_group_ids,
    ffi::AnyBuffer worker_lane_ids,
    ffi::AnyBuffer particle_trace_ids,
    ffi::AnyBuffer rotation_replay_order,
    ffi::AnyBuffer rotation_replay_counts,
    ffi::AnyBuffer particle_start_offsets_ns,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::AnyBuffer logical_current_size,
    ffi::AnyBuffer logical_projector_radius,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    return RelionVdamMstepFusedProjectorXHalfCommon(
        stream, image_h, image_w, pixel_capacity, N0, N1, N2, upsampling,
        max_r2_x4, physical_image_size, projector_max_r,
        projection_padding_factor, reconstruction_group_count,
        parallel_worker_replay, captured_rotation_replay,
        serial_rotation_replay, float64_accumulator_replay,
        reverse_rotation_replay, rotation_replay_stride,
        native_trace_shape_replay, captured_particle_timing_replay,
        candidate_trace_active, particle_tail_mask, projector_full, images, ctf, minvsigma2,
        posterior, translation_angles, eulers, rot, reconstruction_group_ids,
        worker_lane_ids, particle_trace_ids, rotation_replay_order,
        rotation_replay_counts, particle_start_offsets_ns,
        data_real_volume_in, data_imag_volume_in, weight_volume_in,
        &logical_current_size, data_real_volume_out, data_imag_volume_out,
        weight_volume_out, denominator_sum, &logical_projector_radius);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionMakeScoringRotationsF32, RelionMakeScoringRotationsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("do_right")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionMakeScoringRotationsF64Impl(
    cudaStream_t stream,
    int64_t do_right,
    ffi::AnyBuffer eulers_deg,
    ffi::AnyBuffer right_matrix,
    ffi::Result<ffi::AnyBuffer> scorer_rotations)
{
    if (eulers_deg.element_type() != ffi::DataType::F64 ||
        right_matrix.element_type() != ffi::DataType::F64 ||
        scorer_rotations->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: inputs/output must be F64");
    if (do_right != 0 && do_right != 1)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: do_right must be 0 or 1");

    auto euler_dims = eulers_deg.dimensions();
    auto right_dims = right_matrix.dimensions();
    auto output_dims = scorer_rotations->dimensions();
    if (euler_dims.size() != 2 || euler_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: eulers_deg must have shape (N,3)");
    if (right_dims.size() != 2 || right_dims[0] != 3 || right_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: right_matrix must have shape (3,3)");
    if (output_dims.size() != 3 || output_dims[0] != euler_dims[0] ||
        output_dims[1] != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionMakeScoringRotationsF64: output must have shape (N,3,3)");

    const double* eulers_ptr = static_cast<const double*>(eulers_deg.untyped_data());
    const double* right_ptr = static_cast<const double*>(right_matrix.untyped_data());
    double* output_ptr = static_cast<double*>(scorer_rotations->untyped_data());
    cudaError_t err = do_right
        ? launch_relion_make_scoring_rotations<double, true>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0])
        : launch_relion_make_scoring_rotations<double, false>(
              stream, eulers_ptr, right_ptr, output_ptr, euler_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionMakeScoringRotationsF64, RelionMakeScoringRotationsF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("do_right")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionTranslateScoreF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> shifted)
{
    if (images.element_type() != ffi::DataType::C64 ||
        shifted->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: images/output must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: translation angles must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: images must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF32: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_score_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<float2*>(shifted->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateScoreF32, RelionTranslateScoreF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#if CUB_VERSION < 300000
struct RelionAmpereScanPolicy
{
    using MaxPolicy = typename cub::DeviceScanPolicy<float, cub::Sum>::Policy800;
};

cudaError_t relion_ampere_inclusive_sum_f32(
    void* temporary,
    size_t& temporary_bytes,
    const float* input,
    float* output,
    int count,
    cudaStream_t stream)
{
    using Dispatch = cub::DispatchScan<
        const float*,
        float*,
        cub::Sum,
        cub::NullType,
        int,
        float,
        RelionAmpereScanPolicy>;
    return Dispatch::Dispatch(
        temporary,
        temporary_bytes,
        input,
        output,
        cub::Sum(),
        cub::NullType(),
        count,
        stream);
}
#else
cudaError_t relion_ampere_inclusive_sum_f32(
    void* temporary,
    size_t& temporary_bytes,
    const float* input,
    float* output,
    int count,
    cudaStream_t stream)
{
    // CCCL 3 replaced the policy-dispatch API used above.  Retain a buildable
    // fallback; deployed RELION parity is qualified against CUB 2.x, where
    // the explicit Ampere policy remains available and bitwise-tested.
    return cub::DeviceScan::InclusiveSum(
        temporary, temporary_bytes, input, output, count, stream);
}
#endif
ffi::Error RelionTranslateScoreF64Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> shifted)
{
    if (images.element_type() != ffi::DataType::C128 ||
        shifted->element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: images/output must be C128");
    if (translation_angles.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: translation angles must be F64");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: images must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateScoreF64: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_score_f64(
        stream,
        reinterpret_cast<const double2*>(images.untyped_data()),
        static_cast<const double*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<double2*>(shifted->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateScoreF64, RelionTranslateScoreF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#include "relion_posterior.cuh"

ffi::Error RelionTranslateBprefF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer weighted_ctf,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> weighted_shifted)
{
    if (images.element_type() != ffi::DataType::C64 ||
        weighted_shifted->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: images/output must be C64");
    if (weighted_ctf.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: weighted CTF must be F32");
    if (translation_angles.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: translation angles must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto weighted_ctf_dims = weighted_ctf.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = weighted_shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: images must have shape (B,P)");
    if (weighted_ctf_dims.size() != 2 ||
        weighted_ctf_dims[0] != image_dims[0] ||
        weighted_ctf_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: weighted CTF must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF32: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_bpref_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(weighted_ctf.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<float2*>(weighted_shifted->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateBprefF32, RelionTranslateBprefF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionTranslateBprefF64Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer weighted_ctf,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> weighted_shifted)
{
    if (images.element_type() != ffi::DataType::C128 ||
        weighted_shifted->element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: images/output must be C128");
    if (weighted_ctf.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: weighted CTF must be F64");
    if (translation_angles.element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: translation angles must be F64");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto weighted_ctf_dims = weighted_ctf.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto output_dims = weighted_shifted->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: images must have shape (B,P)");
    if (weighted_ctf_dims.size() != 2 ||
        weighted_ctf_dims[0] != image_dims[0] ||
        weighted_ctf_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: weighted CTF must have shape (B,P)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: translation angles must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: pixel indices must have shape (P,)");
    if (output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] * translation_dims[0] ||
        output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionTranslateBprefF64: output must have shape (B*T,P)");

    cudaError_t err = launch_relion_translate_bpref_f64(
        stream,
        reinterpret_cast<const double2*>(images.untyped_data()),
        static_cast<const double*>(weighted_ctf.untyped_data()),
        static_cast<const double*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<double2*>(weighted_shifted->untyped_data()),
        image_dims[0], translation_dims[0], image_dims[1],
        static_cast<int>(image_h), static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateBprefF64, RelionTranslateBprefF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionVdamMstepSumsF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior_over_weight_norm,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer reference,
    ffi::Result<ffi::AnyBuffer> numerator_sum,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    if (images.element_type() != ffi::DataType::C64 ||
        reference.element_type() != ffi::DataType::C64 ||
        numerator_sum->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: complex inputs/outputs must be C64");
    if (ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior_over_weight_norm.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        denominator_sum->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: scalar inputs/denominator must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: image dimensions must be positive");

    auto image_dims = images.dimensions();
    auto ctf_dims = ctf.dimensions();
    auto noise_dims = minvsigma2.dimensions();
    auto posterior_dims = posterior_over_weight_norm.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto reference_dims = reference.dimensions();
    auto numerator_dims = numerator_sum->dimensions();
    auto denominator_dims = denominator_sum->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: images must have shape (B,P)");
    if (ctf_dims.size() != 2 || noise_dims.size() != 2 ||
        ctf_dims[0] != image_dims[0] || ctf_dims[1] != image_dims[1] ||
        noise_dims[0] != image_dims[0] || noise_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: ctf/minvsigma2 must match images");
    if (posterior_dims.size() != 3 || posterior_dims[0] != image_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: posterior must have shape (B,R,T)");
    if (translation_dims.size() != 2 || translation_dims[1] != 2 ||
        translation_dims[0] != posterior_dims[2])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: translations must have shape (T,2)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: pixel indices must have shape (P,)");
    if (reference_dims.size() != 3 ||
        reference_dims[0] != posterior_dims[0] ||
        reference_dims[1] != posterior_dims[1] ||
        reference_dims[2] != image_dims[1] ||
        numerator_dims.size() != 3 || denominator_dims.size() != 3 ||
        numerator_dims[0] != reference_dims[0] ||
        numerator_dims[1] != reference_dims[1] ||
        numerator_dims[2] != reference_dims[2] ||
        denominator_dims[0] != reference_dims[0] ||
        denominator_dims[1] != reference_dims[1] ||
        denominator_dims[2] != reference_dims[2])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepSumsF32: references/outputs must have shape (B,R,P)");

    cudaError_t err = launch_relion_vdam_mstep_sums_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior_over_weight_norm.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<float2*>(numerator_sum->untyped_data()),
        static_cast<float*>(denominator_sum->untyped_data()),
        image_dims[0], posterior_dims[1], posterior_dims[2], image_dims[1],
        static_cast<int>(image_h), static_cast<int>(image_half_width));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepSumsF32, RelionVdamMstepSumsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

// At most 256 nonempty buckets fit the queue's 256-particle capacity.
// One column's launch arguments stay below the pre-Volta 4 KiB parameter limit.
struct BprefPackColumn {
    const uint32_t* inputs[256];
    int32_t ends[256];
};
static_assert(sizeof(BprefPackColumn) + 64 < 4096);

__device__ int bpref_pack_bucket(const BprefPackColumn& column, int buckets, int particle)
{
    int lo = 0, hi = buckets;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (particle < column.ends[mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

__global__ void bpref_pack_column_kernel(
    BprefPackColumn column, int buckets, int capacity, int64_t row_words,
    uint32_t tail_bits, uint32_t* output)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(capacity) * row_words) return;
    const int particle = static_cast<int>(index / row_words);
    if (particle >= column.ends[buckets - 1]) {
        output[index] = tail_bits;
        return;
    }
    const int bucket = bpref_pack_bucket(column, buckets, particle);
    const int begin = bucket == 0 ? 0 : column.ends[bucket - 1];
    output[index] = column.inputs[bucket][
        static_cast<int64_t>(particle - begin) * row_words + index % row_words];
}

__global__ void bpref_pack_ids_kernel(
    BprefPackColumn column, int buckets, int capacity, int32_t* workers, int32_t* traces)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= capacity) return;
    int local = 0;
    if (particle < column.ends[buckets - 1]) {
        const int bucket = bpref_pack_bucket(column, buckets, particle);
        local = particle - (bucket == 0 ? 0 : column.ends[bucket - 1]);
    }
    workers[particle] = local % 8;
    traces[particle] = local;
}

ffi::Error BprefParticlePackImpl(
    cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets)
{
    const auto invalid = [](const char* message) {
        return ffi::Error::InvalidArgument(std::string("BprefParticlePack: ") + message);
    };
    if (args.size() == 0 || args.size() % 6 != 0 || args.size() / 6 > 256 || rets.size() != 8)
        return invalid("requires six columns and eight results");
    const int buckets = static_cast<int>(args.size() / 6);
    const ffi::DataType types[6] = {ffi::DataType::C64, ffi::DataType::F32,
        ffi::DataType::F32, ffi::DataType::F32, ffi::DataType::F32, ffi::DataType::S32};
    const int ranks[6] = {2, 2, 2, 3, 4, 1};
    BprefPackColumn columns[6] = {};
    void* outputs[8] = {};
    int64_t row_words[6] = {};
    std::vector<int64_t> shapes[6];
    int capacity = 0;
    // Validate the whole ABI before launching any device work.
    for (int field = 0; field < 8; ++field) {
        auto result = rets.get<ffi::AnyBuffer>(field);
        if (!result) return result.error();
        const auto output = **result;
        const auto dims = output.dimensions();
        const int rank = field < 6 ? ranks[field] : 1;
        const auto type = field < 6 ? types[field] : ffi::DataType::S32;
        if (dims.size() != rank || output.element_type() != type ||
            dims[0] <= 0 || dims[0] > 256)
            return invalid("result shape/dtype differs");
        if (field == 0) capacity = static_cast<int>(dims[0]);
        if (dims[0] != capacity) return invalid("result capacities differ");
        outputs[field] = output.untyped_data();
        if (field >= 6) continue;
        shapes[field] = std::vector<int64_t>(dims.begin(), dims.end());
        int64_t words = field == 0 ? 2 : 1;
        for (int axis = 1; axis < rank; ++axis) {
            if (dims[axis] <= 0 ||
                words > std::numeric_limits<int64_t>::max() / capacity / dims[axis])
                return invalid("invalid or overflowing row size");
            words *= dims[axis];
        }
        row_words[field] = words;
        if ((words * capacity - 1) / 256 + 1 > std::numeric_limits<int32_t>::max())
            return invalid("copy grid too large");
        int total = 0;
        for (int bucket = 0; bucket < buckets; ++bucket) {
            auto input = args.get<ffi::AnyBuffer>(field * buckets + bucket);
            if (!input) return input.error();
            const auto in_dims = input->dimensions();
            if (input->element_type() != type || in_dims.size() != rank ||
                in_dims[0] <= 0 || in_dims[0] > capacity - total)
                return invalid("input shape/dtype or particle capacity differs");
            for (int axis = 1; axis < rank; ++axis)
                if (in_dims[axis] != dims[axis]) return invalid("input row shape differs");
            total += static_cast<int>(in_dims[0]);
            if (field != 0 && total != columns[0].ends[bucket])
                return invalid("input particle axes differ");
            columns[field].ends[bucket] = total;
            columns[field].inputs[bucket] = static_cast<const uint32_t*>(input->untyped_data());
        }
    }
    if (shapes[1] != shapes[0] || shapes[2] != shapes[0] ||
        shapes[4][1] != shapes[3][1] || shapes[4][2] != 3 || shapes[4][3] != 3)
        return invalid("image/CTF or rotation/posterior geometry differs");
    for (int field = 0; field < 6; ++field) {
        bpref_pack_column_kernel<<<
            static_cast<unsigned int>((row_words[field] * capacity - 1) / 256 + 1),
            256, 0, stream>>>(columns[field], buckets, capacity, row_words[field],
                field == 5 ? 0xffffffffu : 0u, static_cast<uint32_t*>(outputs[field]));
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    }
    bpref_pack_ids_kernel<<<1, 256, 0, stream>>>(
        columns[0], buckets, capacity, static_cast<int32_t*>(outputs[6]),
        static_cast<int32_t*>(outputs[7]));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BprefParticlePack, BprefParticlePackImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets()
);

ffi::Error RelionVdamMstepDenominatorF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior_over_weight_norm,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    if (ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior_over_weight_norm.element_type() != ffi::DataType::F32 ||
        denominator_sum->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepDenominatorF32: inputs/output must be F32");

    const auto ctf_dims = ctf.dimensions();
    const auto noise_dims = minvsigma2.dimensions();
    const auto posterior_dims = posterior_over_weight_norm.dimensions();
    const auto denominator_dims = denominator_sum->dimensions();
    if (ctf_dims.size() != 2 || ctf_dims[0] <= 0 || ctf_dims[1] <= 0 ||
        noise_dims.size() != 2 || noise_dims[0] != ctf_dims[0] ||
        noise_dims[1] != ctf_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepDenominatorF32: ctf/minvsigma2 must have matching (B,P) shapes");
    if (posterior_dims.size() != 3 || posterior_dims[0] != ctf_dims[0] ||
        posterior_dims[1] <= 0 || posterior_dims[2] <= 0)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepDenominatorF32: posterior must have shape (B,R,T)");
    if (denominator_dims.size() != 3 ||
        denominator_dims[0] != posterior_dims[0] ||
        denominator_dims[1] != posterior_dims[1] ||
        denominator_dims[2] != ctf_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepDenominatorF32: output must have shape (B,R,P)");

    cudaError_t err = launch_relion_vdam_mstep_denominator_f32(
        stream,
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior_over_weight_norm.untyped_data()),
        static_cast<float*>(denominator_sum->untyped_data()),
        ctf_dims[0],
        posterior_dims[1],
        posterior_dims[2],
        ctf_dims[1],
        ctf_dims[1],
        nullptr);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepDenominatorF32, RelionVdamMstepDenominatorF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__device__ int64_t deferred_host_gather_index(int32_t index, int64_t size)
{
    int64_t normalized = index;
    if (normalized < 0) normalized += size;
    return normalized >= 0 && normalized < size ? normalized : -1;
}

__global__ void deferred_host_posterior_pack_kernel(
    const uint32_t* posterior, const uint32_t* sum_t, const int32_t* take,
    const uint8_t* mask, int64_t rows, int64_t packed_rotations,
    int64_t dense_rotations, int64_t translations, uint32_t* packed, uint32_t* packed_sum)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * translations) return;
    const int64_t row = i / translations, translation = i % translations;
    const int64_t rotation = deferred_host_gather_index(take[row], dense_rotations);
    uint32_t value = 0, sum = 0;
    if (mask[row]) {
        if (rotation < 0) {
            value = sum = 0x7fc00000u; // JAX float32 gather fill value.
        } else {
            const int64_t source_row = (row / packed_rotations) * dense_rotations + rotation;
            value = posterior[source_row * translations + translation];
            if (translation == 0) sum = sum_t[source_row];
        }
    }
    packed[i] = value;
    if (translation == 0) packed_sum[row] = sum;
}

__global__ void deferred_host_projection_pack_kernel(
    const uint32_t* projection, const int32_t* take, const uint8_t* mask,
    int64_t rows, int64_t flat_rows, int64_t pixel_words, uint32_t* output)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * pixel_words) return;
    const int64_t row = i / pixel_words, word = i % pixel_words;
    uint32_t value = 0;
    if (mask[row]) {
        const int64_t source_row = deferred_host_gather_index(take[row], flat_rows);
        // JAX complex64 gather fill is NaN + 0j, not NaN + NaNj.
        value = source_row < 0 ? (word % 2 == 0 ? 0x7fc00000u : 0u)
                              : projection[source_row * pixel_words + word];
    }
    output[i] = value;
}

__global__ void deferred_host_image_prefix_kernel(
    const uint2* images, const uint32_t* ctf, const uint32_t* noise, int64_t count,
    uint2* packed_images, uint32_t* packed_ctf, uint32_t* packed_noise)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    packed_images[i] = images[i];
    packed_ctf[i] = ctf[i];
    packed_noise[i] = noise[i];
}

__global__ void deferred_host_denominator_mask_kernel(
    const uint8_t* mask, int64_t count, int64_t pixels, uint32_t* output)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && !mask[i / pixels]) output[i] = 0;
}

ffi::Error DeferredVdamHostPackImpl(
    cudaStream_t stream,
    ffi::AnyBuffer posterior, ffi::AnyBuffer sum_t, ffi::AnyBuffer images,
    ffi::AnyBuffer ctf, ffi::AnyBuffer minvsigma2, ffi::AnyBuffer flat_projection,
    ffi::AnyBuffer take, ffi::AnyBuffer mask, ffi::AnyBuffer flat_take,
    ffi::Result<ffi::AnyBuffer> packed_posterior, ffi::Result<ffi::AnyBuffer> packed_sum,
    ffi::Result<ffi::AnyBuffer> packed_images, ffi::Result<ffi::AnyBuffer> packed_ctf,
    ffi::Result<ffi::AnyBuffer> packed_noise, ffi::Result<ffi::AnyBuffer> packed_projection,
    ffi::Result<ffi::AnyBuffer> denominator)
{
    const auto invalid = [](const char* message) {
        return ffi::Error::InvalidArgument(std::string("DeferredVdamHostPack: ") + message);
    };
    const auto dense = posterior.dimensions(), image_dims = images.dimensions();
    const auto flat = flat_projection.dimensions(), plan = take.dimensions();
    if (dense.size() != 3 || image_dims.size() != 2 || flat.size() != 2 || plan.size() != 2)
        return invalid("input ranks differ");
    const int64_t batch = plan[0], rotations = plan[1], pixels = image_dims[1], translations = dense[2];
    const auto valid_count = [](std::initializer_list<int64_t> shape) {
        const int64_t limit = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 256;
        int64_t count = 1;
        for (int64_t n : shape) {
            if (n <= 0 || n > limit / count) return false;
            count *= n;
        }
        return true;
    };
    if (!valid_count({batch, rotations, translations}) ||
        !valid_count({batch, rotations, pixels, 2}) ||
        !valid_count({dense[0], dense[1], translations}) ||
        !valid_count({image_dims[0], pixels, 2}) || !valid_count({flat[0], pixels, 2}) ||
        dense[0] < batch || image_dims[0] < batch)
        return invalid("invalid batch or overflowing geometry");
    const auto matches = [](ffi::AnyBuffer value, ffi::DataType type,
                            std::initializer_list<int64_t> shape) {
        const auto dims = value.dimensions();
        if (value.element_type() != type || dims.size() != shape.size()) return false;
        auto current = dims.begin();
        for (int64_t n : shape) if (*current++ != n) return false;
        return true;
    };
    using D = ffi::DataType;
    if (!matches(posterior, D::F32, {dense[0], dense[1], translations}) ||
        !matches(sum_t, D::F32, {dense[0], dense[1]}) ||
        !matches(images, D::C64, {image_dims[0], pixels}) ||
        !matches(ctf, D::F32, {image_dims[0], pixels}) ||
        !matches(minvsigma2, D::F32, {image_dims[0], pixels}) ||
        !matches(flat_projection, D::C64, {flat[0], pixels}) ||
        !matches(take, D::S32, {batch, rotations}) ||
        !matches(mask, D::PRED, {batch, rotations}) ||
        !matches(flat_take, D::S32, {batch, rotations}) ||
        !matches(*packed_posterior, D::F32, {batch, rotations, translations}) ||
        !matches(*packed_sum, D::F32, {batch, rotations}) ||
        !matches(*packed_images, D::C64, {batch, pixels}) ||
        !matches(*packed_ctf, D::F32, {batch, pixels}) ||
        !matches(*packed_noise, D::F32, {batch, pixels}) ||
        !matches(*packed_projection, D::C64, {batch, rotations, pixels}) ||
        !matches(*denominator, D::F32, {batch, rotations, pixels}))
        return invalid("input/output shape or dtype differs");
    const auto grid = [](int64_t count) {
        return static_cast<unsigned int>((count - 1) / 256 + 1);
    };
    const int64_t rows = batch * rotations;
    const auto row_mask = static_cast<const uint8_t*>(mask.untyped_data());
    deferred_host_posterior_pack_kernel<<<grid(rows * translations), 256, 0, stream>>>(
        static_cast<const uint32_t*>(posterior.untyped_data()),
        static_cast<const uint32_t*>(sum_t.untyped_data()),
        static_cast<const int32_t*>(take.untyped_data()), row_mask, rows, rotations, dense[1], translations,
        static_cast<uint32_t*>(packed_posterior->untyped_data()),
        static_cast<uint32_t*>(packed_sum->untyped_data()));
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    deferred_host_projection_pack_kernel<<<grid(rows * pixels * 2), 256, 0, stream>>>(
        static_cast<const uint32_t*>(flat_projection.untyped_data()),
        static_cast<const int32_t*>(flat_take.untyped_data()), row_mask, rows, flat[0], pixels * 2,
        static_cast<uint32_t*>(packed_projection->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    deferred_host_image_prefix_kernel<<<grid(batch * pixels), 256, 0, stream>>>(
        static_cast<const uint2*>(images.untyped_data()),
        static_cast<const uint32_t*>(ctf.untyped_data()),
        static_cast<const uint32_t*>(minvsigma2.untyped_data()), batch * pixels,
        static_cast<uint2*>(packed_images->untyped_data()),
        static_cast<uint32_t*>(packed_ctf->untyped_data()),
        static_cast<uint32_t*>(packed_noise->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    // Reuse exactly the original sequential translation arithmetic and launch.
    error = launch_relion_vdam_mstep_denominator_f32(
        stream, static_cast<const float*>(packed_ctf->untyped_data()),
        static_cast<const float*>(packed_noise->untyped_data()),
        static_cast<const float*>(packed_posterior->untyped_data()),
        static_cast<float*>(denominator->untyped_data()),
        batch, rotations, translations, pixels, pixels, nullptr);
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    deferred_host_denominator_mask_kernel<<<grid(rows * pixels), 256, 0, stream>>>(
        row_mask, rows * pixels, pixels, static_cast<uint32_t*>(denominator->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DeferredVdamHostPack, DeferredVdamHostPackImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void noise_pixel_pad_words_kernel(
    const uint32_t* input, const uint32_t* tail_value, int64_t prefix_words,
    int64_t total_words, int tail_words, uint32_t* output)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total_words)
        output[i] = i < prefix_words ? input[i] : (tail_value ? tail_value[i % tail_words] : 0u);
}

ffi::Error NoisePixelPackImpl(
    cudaStream_t stream, int64_t target_batch,
    ffi::AnyBuffer probs, ffi::AnyBuffer projection, ffi::AnyBuffer ctf_probs,
    ffi::AnyBuffer indices, ffi::AnyBuffer spare_index,
    ffi::Result<ffi::AnyBuffer> padded_probs, ffi::Result<ffi::AnyBuffer> padded_projection,
    ffi::Result<ffi::AnyBuffer> padded_ctf_probs, ffi::Result<ffi::AnyBuffer> padded_indices)
{
    const auto invalid = [](const char* message) {
        return ffi::Error::InvalidArgument(std::string("NoisePixelPack: ") + message);
    };
    using D = ffi::DataType;
    const auto pd = probs.dimensions(), rd = projection.dimensions(), cd = ctf_probs.dimensions();
    const auto ids = indices.dimensions();
    if (pd.size() != 3 || rd.size() != 3 || cd.size() != 3 || ids.size() != 1 ||
        spare_index.dimensions().size() != 0)
        return invalid("input ranks differ");
    if (target_batch < pd[0] || pd[0] <= 0 || pd[1] <= 0 || pd[2] <= 0 ||
        rd[0] != pd[0] || rd[1] != pd[1] || rd[2] <= 0 ||
        cd[0] != rd[0] || cd[1] != rd[1] || cd[2] != rd[2] || ids[0] != pd[0])
        return invalid("input geometry differs");
    if ((probs.element_type() != D::F32 && probs.element_type() != D::F64) ||
        (projection.element_type() != D::C64 && projection.element_type() != D::C128) ||
        (ctf_probs.element_type() != D::F32 && ctf_probs.element_type() != D::F64) ||
        (indices.element_type() != D::S32 && indices.element_type() != D::S64) ||
        spare_index.element_type() != indices.element_type())
        return invalid("input dtypes differ");
    const ffi::AnyBuffer inputs[] = {probs, projection, ctf_probs, indices};
    const ffi::AnyBuffer outputs[] = {*padded_probs, *padded_projection, *padded_ctf_probs, *padded_indices};
    int64_t row_words[4];
    const int64_t word_limit = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 256;
    for (int field = 0; field < 4; ++field) {
        const auto in = inputs[field].dimensions(), out = outputs[field].dimensions();
        if (out.size() != in.size() || out[0] != target_batch ||
            outputs[field].element_type() != inputs[field].element_type())
            return invalid("output shape or dtype differs");
        const D type = inputs[field].element_type();
        int64_t words = type == D::C128 ? 4 : (type == D::F64 || type == D::C64 || type == D::S64 ? 2 : 1);
        for (size_t axis = 1; axis < in.size(); ++axis) {
            if (out[axis] != in[axis] || in[axis] <= 0 || in[axis] > word_limit / words)
                return invalid("overflowing or mismatched output geometry");
            words *= in[axis];
        }
        if (target_batch > word_limit / words) return invalid("overflowing output size");
        row_words[field] = words;
    }
    // Every shape/type is validated before any copy. The spare index stays device resident.
    for (int field = 0; field < 4; ++field) {
        const int64_t count = target_batch * row_words[field];
        const auto fill = field == 3 ? static_cast<const uint32_t*>(spare_index.untyped_data()) : nullptr;
        noise_pixel_pad_words_kernel<<<static_cast<unsigned int>((count - 1) / 256 + 1), 256, 0, stream>>>(
            static_cast<const uint32_t*>(inputs[field].untyped_data()), fill,
            pd[0] * row_words[field], count, static_cast<int>(field == 3 ? row_words[field] : 1),
            static_cast<uint32_t*>(outputs[field].untyped_data()));
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NoisePixelPack, NoisePixelPackImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<int64_t>("target_batch")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
);

ffi::Error RelionVdamMstepFusedXHalfImpl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum)
{
    if (images.element_type() != ffi::DataType::C64 ||
        reference.element_type() != ffi::DataType::C64 ||
        ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        rot.element_type() != ffi::DataType::F32 ||
        data_real_volume_in.element_type() != ffi::DataType::F32 ||
        data_imag_volume_in.element_type() != ffi::DataType::F32 ||
        data_real_volume_out->element_type() != ffi::DataType::F32 ||
        data_imag_volume_out->element_type() != ffi::DataType::F32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32 ||
        denominator_sum->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedXHalf: inputs and outputs have invalid dtypes");
    if (image_h <= 0 || image_w != image_h / 2 + 1 ||
        N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedXHalf: invalid x-half geometry");

    const auto image_dims = images.dimensions();
    const auto ctf_dims = ctf.dimensions();
    const auto noise_dims = minvsigma2.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto reference_dims = reference.dimensions();
    const auto rot_dims = rot.dimensions();
    const auto denominator_dims = denominator_sum->dimensions();
    const int64_t pixel_count = image_h * image_w;
    if (image_dims.size() != 2 || image_dims[0] <= 0 || image_dims[1] != pixel_count ||
        ctf_dims.size() != 2 || ctf_dims[0] != image_dims[0] || ctf_dims[1] != pixel_count ||
        noise_dims.size() != 2 || noise_dims[0] != image_dims[0] || noise_dims[1] != pixel_count ||
        posterior_dims.size() != 3 || posterior_dims[0] != image_dims[0] ||
        posterior_dims[1] <= 0 || posterior_dims[2] <= 0 ||
        translation_dims.size() != 2 || translation_dims[0] != posterior_dims[2] || translation_dims[1] != 2 ||
        reference_dims.size() != 3 || reference_dims[0] != image_dims[0] ||
        reference_dims[1] != posterior_dims[1] || reference_dims[2] != pixel_count ||
        rot_dims.size() != 3 || rot_dims[0] != image_dims[0] ||
        rot_dims[1] != posterior_dims[1] || rot_dims[2] != 6 ||
        denominator_dims.size() != 3 || denominator_dims[0] != image_dims[0] ||
        denominator_dims[1] != posterior_dims[1] || denominator_dims[2] != pixel_count)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedXHalf: inconsistent particle/rotation/pixel topology");

    const int64_t volume_size = N0 * N1 * (N2 / 2 + 1);
    const auto data_real_in_dims = data_real_volume_in.dimensions();
    const auto data_imag_in_dims = data_imag_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_real_out_dims = data_real_volume_out->dimensions();
    const auto data_imag_out_dims = data_imag_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_real_in_dims.size() != 1 || data_real_in_dims[0] != volume_size ||
        data_imag_in_dims.size() != 1 || data_imag_in_dims[0] != volume_size ||
        weight_in_dims.size() != 1 || weight_in_dims[0] != volume_size ||
        data_real_out_dims.size() != 1 || data_real_out_dims[0] != volume_size ||
        data_imag_out_dims.size() != 1 || data_imag_out_dims[0] != volume_size ||
        weight_out_dims.size() != 1 || weight_out_dims[0] != volume_size)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedXHalf: accumulator sizes do not match");

    cudaError_t err = launch_relion_vdam_mstep_fused_x_half(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        reinterpret_cast<const float2*>(reference.untyped_data()),
        static_cast<const float*>(rot.untyped_data()),
        static_cast<float*>(data_real_volume_out->untyped_data()),
        static_cast<float*>(data_imag_volume_out->untyped_data()),
        static_cast<float*>(weight_volume_out->untyped_data()),
        static_cast<float*>(denominator_sum->untyped_data()),
        image_dims[0], posterior_dims[1], posterior_dims[2], pixel_count,
        image_h, image_w, N0, N1, N2, upsampling, max_r2_x4);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepFusedXHalf, RelionVdamMstepFusedXHalfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("max_r2_x4")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionVdamMstepFusedProjectorXHalfCommon(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_w,
    int64_t pixel_capacity,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int64_t physical_image_size,
    int64_t projector_max_r,
    int64_t projection_padding_factor,
    int64_t reconstruction_group_count,
    int64_t parallel_worker_replay,
    int64_t captured_rotation_replay,
    int64_t serial_rotation_replay,
    int64_t float64_accumulator_replay,
    int64_t reverse_rotation_replay,
    int64_t rotation_replay_stride,
    int64_t native_trace_shape_replay,
    int64_t captured_particle_timing_replay,
    int64_t candidate_trace_active,
    int64_t particle_tail_mask,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer reconstruction_group_ids,
    ffi::AnyBuffer worker_lane_ids,
    ffi::AnyBuffer particle_trace_ids,
    ffi::AnyBuffer rotation_replay_order,
    ffi::AnyBuffer rotation_replay_counts,
    ffi::AnyBuffer particle_start_offsets_ns,
    ffi::AnyBuffer data_real_volume_in,
    ffi::AnyBuffer data_imag_volume_in,
    ffi::AnyBuffer weight_volume_in,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::Result<ffi::AnyBuffer> data_real_volume_out,
    ffi::Result<ffi::AnyBuffer> data_imag_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> denominator_sum,
    const ffi::AnyBuffer* runtime_projector_radius)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        images.element_type() != ffi::DataType::C64 ||
        data_real_volume_in.element_type() != ffi::DataType::F32 ||
        data_imag_volume_in.element_type() != ffi::DataType::F32 ||
        data_real_volume_out->element_type() != ffi::DataType::F32 ||
        data_imag_volume_out->element_type() != ffi::DataType::F32 ||
        ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        eulers.element_type() != ffi::DataType::F32 ||
        rot.element_type() != ffi::DataType::F32 ||
        reconstruction_group_ids.element_type() != ffi::DataType::S32 ||
        worker_lane_ids.element_type() != ffi::DataType::S32 ||
        particle_trace_ids.element_type() != ffi::DataType::S32 ||
        rotation_replay_order.element_type() != ffi::DataType::S32 ||
        rotation_replay_counts.element_type() != ffi::DataType::S32 ||
        particle_start_offsets_ns.element_type() != ffi::DataType::S32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32 ||
        denominator_sum->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedProjectorXHalf: invalid dtypes");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedProjectorRuntimeXHalf: "
            "logical_current_size must be an S32 scalar");
    const bool capacity_projector = runtime_projector_radius != nullptr;
    if (capacity_projector &&
        (runtime_current_size == nullptr ||
         runtime_projector_radius->element_type() != ffi::DataType::S32 ||
         runtime_projector_radius->dimensions().size() != 0 ||
         projector_max_r != 0 ||
         (projection_padding_factor != 1 && projection_padding_factor != 2) ||
         captured_rotation_replay != 0 || serial_rotation_replay != 0 ||
         float64_accumulator_replay != 0 || reverse_rotation_replay != 0 ||
         rotation_replay_stride != 0 || native_trace_shape_replay != 0 ||
         captured_particle_timing_replay != 0 || candidate_trace_active != 0))
        return ffi::Error::InvalidArgument(
            "BPref projector capacity requires S32 radius, static radius zero, "
            "stable image geometry, padding 1/2 and no replay diagnostics");
    if (image_h <= 0 || image_w != image_h / 2 + 1 ||
        pixel_capacity < image_h * image_w ||
        N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        upsampling <= 0 || max_r2_x4 < 0 || physical_image_size <= 0 ||
        (!capacity_projector && projector_max_r <= 0) || projection_padding_factor <= 0 ||
        reconstruction_group_count <= 0 ||
        (parallel_worker_replay != 0 && parallel_worker_replay != 1) ||
        (captured_rotation_replay != 0 && captured_rotation_replay != 1) ||
        (serial_rotation_replay < 0 || serial_rotation_replay > 2) ||
        (float64_accumulator_replay != 0 && float64_accumulator_replay != 1) ||
        (reverse_rotation_replay != 0 && reverse_rotation_replay != 1) ||
        (native_trace_shape_replay != 0 && native_trace_shape_replay != 1) ||
        (captured_particle_timing_replay != 0 &&
         captured_particle_timing_replay != 1) ||
        (candidate_trace_active != 0 && candidate_trace_active != 1) ||
        rotation_replay_stride < 0 ||
        (captured_rotation_replay != 0 && reverse_rotation_replay != 0) ||
        (captured_rotation_replay != 0 && rotation_replay_stride != 0) ||
        (serial_rotation_replay == 2 && captured_rotation_replay != 0) ||
        (serial_rotation_replay == 2 && reverse_rotation_replay != 0) ||
        (serial_rotation_replay == 2 && rotation_replay_stride != 0) ||
        (captured_particle_timing_replay != 0 && parallel_worker_replay != 0) ||
        (rotation_replay_stride > 0 && serial_rotation_replay == 0) ||
        (rotation_replay_stride > 0 && reverse_rotation_replay != 0))
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedProjectorXHalf: invalid geometry");

    const auto projector_dims = projector_full.dimensions();
    const auto image_dims = images.dimensions();
    const auto ctf_dims = ctf.dimensions();
    const auto noise_dims = minvsigma2.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto euler_dims = eulers.dimensions();
    const auto rot_dims = rot.dimensions();
    const auto reconstruction_group_dims = reconstruction_group_ids.dimensions();
    const auto worker_lane_dims = worker_lane_ids.dimensions();
    const auto particle_trace_dims = particle_trace_ids.dimensions();
    const auto rotation_replay_order_dims = rotation_replay_order.dimensions();
    const auto rotation_replay_count_dims = rotation_replay_counts.dimensions();
    const auto particle_start_offset_dims = particle_start_offsets_ns.dimensions();
    const auto denominator_dims = denominator_sum->dimensions();
    // An explicit empty rank-one F32 result requests accumulator-only work.
    // Every nonempty result must retain the full legacy denominator topology.
    const bool omit_denominator =
        denominator_dims.size() == 1 && denominator_dims[0] == 0;
    if ((particle_tail_mask != 0 && particle_tail_mask != 1) ||
        (particle_tail_mask != 0 &&
         (!omit_denominator || reconstruction_group_count <= 1 ||
          parallel_worker_replay != 0 || capacity_projector ||
          captured_rotation_replay != 0 || serial_rotation_replay != 0 ||
          float64_accumulator_replay != 0 || reverse_rotation_replay != 0 ||
          rotation_replay_stride != 0 || native_trace_shape_replay != 0 ||
          captured_particle_timing_replay != 0 || candidate_trace_active != 0)))
        return ffi::Error::InvalidArgument(
            "BPref particle tail mask requires grouped accumulator-only work "
            "and no parallel/replay/trace or projector capacity");
    const int64_t pixel_count = image_h * image_w;
    const int64_t logical_projector_size =
        2 * projector_max_r * projection_padding_factor + 3;
    const bool direct_half_projector = !capacity_projector &&
        projector_dims.size() == 3 && logical_projector_size >= 5 &&
        projector_dims[0] == logical_projector_size &&
        projector_dims[1] == logical_projector_size &&
        projector_dims[2] == logical_projector_size / 2 + 1;
    const bool valid_projector = capacity_projector
        ? (projector_dims.size() == 3 && projector_dims[0] >= 5 &&
           projector_dims[0] <= 1025 && projector_dims[0] == projector_dims[1] &&
           (projector_dims[0] & 1) == 1 &&
           projector_dims[2] == projector_dims[0] / 2 + 1 &&
           (projector_dims[0] - 3) % (2 * projection_padding_factor) == 0)
        : (direct_half_projector ||
           (projector_dims.size() == 3 && projector_dims[0] > 0 &&
            projector_dims[1] == projector_dims[0] &&
            projector_dims[2] == projector_dims[0]));
    if (!valid_projector ||
        image_dims.size() != 2 || image_dims[0] <= 0 || image_dims[1] != pixel_capacity ||
        ctf_dims.size() != 2 || ctf_dims[0] != image_dims[0] || ctf_dims[1] != pixel_capacity ||
        noise_dims.size() != 2 || noise_dims[0] != image_dims[0] || noise_dims[1] != pixel_capacity ||
        posterior_dims.size() != 3 || posterior_dims[0] != image_dims[0] ||
        posterior_dims[1] <= 0 || posterior_dims[2] <= 0 ||
        translation_dims.size() != 2 || translation_dims[0] != posterior_dims[2] || translation_dims[1] != 2 ||
        euler_dims.size() != 3 || euler_dims[0] != image_dims[0] ||
        euler_dims[1] != posterior_dims[1] || euler_dims[2] != 9 ||
        rot_dims.size() != 3 || rot_dims[0] != image_dims[0] ||
        rot_dims[1] != posterior_dims[1] || rot_dims[2] != 6 ||
        reconstruction_group_dims.size() != 1 ||
        reconstruction_group_dims[0] != image_dims[0] ||
        worker_lane_dims.size() != 1 ||
        worker_lane_dims[0] != image_dims[0] ||
        particle_trace_dims.size() != 1 ||
        particle_trace_dims[0] != image_dims[0] ||
        rotation_replay_order_dims.size() != 2 ||
        rotation_replay_order_dims[0] != image_dims[0] ||
        rotation_replay_order_dims[1] != posterior_dims[1] ||
        rotation_replay_count_dims.size() != 1 ||
        rotation_replay_count_dims[0] != image_dims[0] ||
        particle_start_offset_dims.size() != 1 ||
        particle_start_offset_dims[0] != image_dims[0] ||
        (!omit_denominator &&
         (denominator_dims.size() != 3 || denominator_dims[0] != image_dims[0] ||
          denominator_dims[1] != posterior_dims[1] ||
          denominator_dims[2] != pixel_capacity)))
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedProjectorXHalf: inconsistent topology");

    const int64_t volume_size = N0 * N1 * (N2 / 2 + 1);
    const auto data_real_in_dims = data_real_volume_in.dimensions();
    const auto data_imag_in_dims = data_imag_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_real_out_dims = data_real_volume_out->dimensions();
    const auto data_imag_out_dims = data_imag_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    const bool ungrouped_accumulators = reconstruction_group_count == 1;
    const auto valid_accumulator_dims = [=](auto dims) {
        return ungrouped_accumulators
            ? (dims.size() == 1 && dims[0] == volume_size)
            : (dims.size() == 2 &&
               dims[0] == reconstruction_group_count &&
               dims[1] == volume_size);
    };
    if (!valid_accumulator_dims(data_real_in_dims) ||
        !valid_accumulator_dims(data_imag_in_dims) ||
        !valid_accumulator_dims(weight_in_dims) ||
        !valid_accumulator_dims(data_real_out_dims) ||
        !valid_accumulator_dims(data_imag_out_dims) ||
        !valid_accumulator_dims(weight_out_dims))
        return ffi::Error::InvalidArgument(
            "RelionVdamMstepFusedProjectorXHalf: accumulator sizes do not match");

    cudaError_t err = launch_relion_vdam_mstep_fused_projector_x_half(
        stream,
        reinterpret_cast<const float2*>(projector_full.untyped_data()),
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(eulers.untyped_data()),
        static_cast<const float*>(rot.untyped_data()),
        static_cast<const int32_t*>(reconstruction_group_ids.untyped_data()),
        static_cast<const int32_t*>(worker_lane_ids.untyped_data()),
        static_cast<const int32_t*>(particle_trace_ids.untyped_data()),
        static_cast<const int32_t*>(rotation_replay_order.untyped_data()),
        static_cast<const int32_t*>(rotation_replay_counts.untyped_data()),
        static_cast<const int32_t*>(particle_start_offsets_ns.untyped_data()),
        static_cast<float*>(data_real_volume_out->untyped_data()),
        static_cast<float*>(data_imag_volume_out->untyped_data()),
        static_cast<float*>(weight_volume_out->untyped_data()),
        omit_denominator ? nullptr : static_cast<float*>(denominator_sum->untyped_data()),
        projector_dims[0], image_dims[0], posterior_dims[1], posterior_dims[2],
        pixel_count, pixel_capacity, image_h, image_w, N0, N1, N2,
        upsampling, max_r2_x4,
        static_cast<int>(physical_image_size),
        static_cast<int>(projector_max_r),
        static_cast<int>(projection_padding_factor),
        static_cast<int>(reconstruction_group_count),
        parallel_worker_replay != 0,
        captured_rotation_replay != 0,
        static_cast<int>(serial_rotation_replay),
        float64_accumulator_replay != 0,
        reverse_rotation_replay != 0,
        static_cast<int>(rotation_replay_stride),
        native_trace_shape_replay != 0,
        captured_particle_timing_replay != 0,
        candidate_trace_active != 0,
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()),
        nullptr,
        nullptr,
        nullptr,
        -1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        runtime_projector_radius == nullptr ? nullptr
            : static_cast<const int32_t*>(runtime_projector_radius->untyped_data()),
        particle_tail_mask != 0, direct_half_projector);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepFusedProjectorXHalf,
    RelionVdamMstepFusedProjectorXHalfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("pixel_capacity")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("max_r2_x4")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("projector_max_r")
        .Attr<int64_t>("projection_padding_factor")
        .Attr<int64_t>("reconstruction_group_count")
        .Attr<int64_t>("parallel_worker_replay")
        .Attr<int64_t>("captured_rotation_replay")
        .Attr<int64_t>("serial_rotation_replay")
        .Attr<int64_t>("float64_accumulator_replay")
        .Attr<int64_t>("reverse_rotation_replay")
        .Attr<int64_t>("rotation_replay_stride")
        .Attr<int64_t>("native_trace_shape_replay")
        .Attr<int64_t>("captured_particle_timing_replay")
        .Attr<int64_t>("candidate_trace_active")
        .Attr<int64_t>("particle_tail_mask")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepFusedProjectorRuntimeXHalf,
    RelionVdamMstepFusedProjectorRuntimeXHalfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("pixel_capacity")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("max_r2_x4")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("projector_max_r")
        .Attr<int64_t>("projection_padding_factor")
        .Attr<int64_t>("reconstruction_group_count")
        .Attr<int64_t>("parallel_worker_replay")
        .Attr<int64_t>("captured_rotation_replay")
        .Attr<int64_t>("serial_rotation_replay")
        .Attr<int64_t>("float64_accumulator_replay")
        .Attr<int64_t>("reverse_rotation_replay")
        .Attr<int64_t>("rotation_replay_stride")
        .Attr<int64_t>("native_trace_shape_replay")
        .Attr<int64_t>("captured_particle_timing_replay")
        .Attr<int64_t>("candidate_trace_active")
        .Attr<int64_t>("particle_tail_mask")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionVdamMstepFusedProjectorCapacityXHalf,
    RelionVdamMstepFusedProjectorCapacityXHalfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("pixel_capacity")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("max_r2_x4")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("projector_max_r")
        .Attr<int64_t>("projection_padding_factor")
        .Attr<int64_t>("reconstruction_group_count")
        .Attr<int64_t>("parallel_worker_replay")
        .Attr<int64_t>("captured_rotation_replay")
        .Attr<int64_t>("serial_rotation_replay")
        .Attr<int64_t>("float64_accumulator_replay")
        .Attr<int64_t>("reverse_rotation_replay")
        .Attr<int64_t>("rotation_replay_stride")
        .Attr<int64_t>("native_trace_shape_replay")
        .Attr<int64_t>("captured_particle_timing_replay")
        .Attr<int64_t>("candidate_trace_active")
        .Attr<int64_t>("particle_tail_mask")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2RectangularF32Common(
    cudaStream_t stream,
    const ffi::AnyBuffer* runtime_full_pixel_count,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output,
    bool shared_pretranslated = false)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32 ||
        (runtime_full_pixel_count != nullptr &&
         (runtime_full_pixel_count->element_type() != ffi::DataType::S32 ||
          runtime_full_pixel_count->dimensions().size() != 0)))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: lookup/count must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 2 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 ||
        image_dims[1] > kRelionCoarseDiff2BlockSize ||
        image_dims[2] != reference_dims[1] ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] || lookup_dims[0] <= 0 ||
        initial_dims[0] != image_dims[0] ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != reference_dims[0] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: inconsistent operand shapes");

    const int64_t rotation_blocks =
        (reference_dims[0] + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t block_count = image_dims[0] * rotation_blocks;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF32: block count exceeds CUDA grid");
    cudaError_t err;
    if (shared_pretranslated) {
        err = launch_relion_coarse_diff2_shared_pretranslated_f32(
            stream,
            reinterpret_cast<const float2*>(reference.untyped_data()),
            reinterpret_cast<const float2*>(shifted_image.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            image_dims[0], reference_dims[0], image_dims[1],
            reference_dims[1], lookup_dims[0],
            static_cast<const int32_t*>(
                runtime_full_pixel_count->untyped_data()));
    } else {
        err = launch_relion_coarse_diff2_rectangular_f32(
            stream,
            reinterpret_cast<const float2*>(reference.untyped_data()),
            reinterpret_cast<const float2*>(shifted_image.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            image_dims[0],
            reference_dims[0],
            image_dims[1],
            reference_dims[1],
            lookup_dims[0],
            runtime_full_pixel_count == nullptr
                ? nullptr
                : static_cast<const int32_t*>(
                      runtime_full_pixel_count->untyped_data()));
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionCoarseDiff2RectangularF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RectangularF32Common(
        stream,
        nullptr,
        reference,
        shifted_image,
        weight,
        initial_diff2,
        full_to_compact,
        output);
}

ffi::Error RelionCoarseDiff2RectangularRuntimeF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_full_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RectangularF32Common(
        stream,
        &logical_full_pixel_count,
        reference,
        shifted_image,
        weight,
        initial_diff2,
        full_to_compact,
        output);
}

ffi::Error RelionCoarseDiff2SharedPretranslatedRuntimeF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_full_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RectangularF32Common(
        stream, &logical_full_pixel_count, reference, shifted_image,
        weight, initial_diff2, full_to_compact, output, true);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2SharedPretranslatedRuntimeF32,
    RelionCoarseDiff2SharedPretranslatedRuntimeF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RectangularF32, RelionCoarseDiff2RectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RectangularRuntimeF32,
    RelionCoarseDiff2RectangularRuntimeF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2RotationBlocksF32Common(
    cudaStream_t stream,
    const ffi::AnyBuffer* runtime_full_pixel_count,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer rotation_block_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: weight/initial/output must be F32");
    if (rotation_block_ids.element_type() != ffi::DataType::S32 ||
        full_to_compact.element_type() != ffi::DataType::S32 ||
        (runtime_full_pixel_count != nullptr &&
         (runtime_full_pixel_count->element_type() != ffi::DataType::S32 ||
          runtime_full_pixel_count->dimensions().size() != 0)))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: block IDs/lookup/count must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto block_dims = rotation_block_ids.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 2 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        block_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 4 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 ||
        image_dims[1] > kRelionCoarseDiff2BlockSize ||
        image_dims[2] != reference_dims[1] ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] || initial_dims[0] != image_dims[0] ||
        block_dims[0] != image_dims[0] || block_dims[1] <= 0 ||
        lookup_dims[0] <= 0 || output_dims[0] != image_dims[0] ||
        output_dims[1] != block_dims[1] ||
        output_dims[2] != kRelionCoarseEulersPerBlock ||
        output_dims[3] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: inconsistent operand shapes");

    const int64_t max_grid =
        static_cast<int64_t>(std::numeric_limits<int>::max());
    if (image_dims[0] > max_grid / block_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: block count exceeds CUDA grid");
    const int64_t block_count = image_dims[0] * block_dims[1];
    const int64_t output_count =
        block_count * kRelionCoarseEulersPerBlock * image_dims[1];
    constexpr int initialize_block_size = 256;
    const int64_t initialize_blocks =
        (output_count + initialize_block_size - 1) / initialize_block_size;
    if (initialize_blocks > max_grid)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RotationBlocksF32: output exceeds CUDA grid");

    cudaError_t err = launch_relion_coarse_diff2_rotation_blocks_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(rotation_block_ids.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        block_dims[1],
        reference_dims[0],
        image_dims[1],
        reference_dims[1],
        lookup_dims[0],
        runtime_full_pixel_count == nullptr
            ? nullptr
            : static_cast<const int32_t*>(
                  runtime_full_pixel_count->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionCoarseDiff2RotationBlocksF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer rotation_block_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RotationBlocksF32Common(
        stream,
        nullptr,
        reference,
        shifted_image,
        weight,
        initial_diff2,
        rotation_block_ids,
        full_to_compact,
        output);
}

ffi::Error RelionCoarseDiff2RotationBlocksRuntimeF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer rotation_block_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_full_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionCoarseDiff2RotationBlocksF32Common(
        stream,
        &logical_full_pixel_count,
        reference,
        shifted_image,
        weight,
        initial_diff2,
        rotation_block_ids,
        full_to_compact,
        output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RotationBlocksF32,
    RelionCoarseDiff2RotationBlocksF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RotationBlocksRuntimeF32,
    RelionCoarseDiff2RotationBlocksRuntimeF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error ValidateRelionCoarseDiff2ProjectorF32Operands(
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t canonical_reduction,
    int64_t single_lane_canonical,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        images.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: projector/images must be C64");
    if (rotations.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: rotations/angles/weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: lookup must be S32");

    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (projector_dims.size() != 3 ||
        projector_dims[0] != projector_dims[1] ||
        projector_dims[1] != projector_dims[2] ||
        rotation_dims.size() != 2 || rotation_dims[1] != 6 ||
        image_dims.size() != 2 || angle_dims.size() != 2 ||
        angle_dims[1] != 2 || weight_dims.size() != 2 ||
        initial_dims.size() != 1 || lookup_dims.size() != 1 ||
        output_dims.size() != 3 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 || rotation_dims[0] <= 0 ||
        angle_dims[0] <= 0 ||
        angle_dims[0] > kRelionCoarseDiff2BlockSize ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != image_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != current_size * (current_size / 2 + 1) ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != rotation_dims[0] ||
        output_dims[2] != angle_dims[0] ||
        current_size <= 0 || physical_image_size <= 0 || model_max_r <= 0 ||
        (canonical_reduction != 0 && canonical_reduction != 1) ||
        (single_lane_canonical != 0 && single_lane_canonical != 1) ||
        (prehalf_weight != 0 && prehalf_weight != 1))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: inconsistent operand shapes or attributes");
    if (single_lane_canonical &&
        (canonical_reduction != 1 ||
         angle_dims[0] <= kRelionCoarseDiff2BlockSize / 2))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: single-lane canonical reduction "
            "requires canonical reduction and 65--128 translations");
    if (prehalf_weight && (canonical_reduction || single_lane_canonical))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorF32: prehalved weights require "
            "native atomic reduction");

    return ffi::Error::Success();
}

ffi::Error RelionCoarseDiff2ProjectorF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    int64_t canonical_reduction,
    int64_t single_lane_canonical,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    ffi::Error validation = ValidateRelionCoarseDiff2ProjectorF32Operands(
        current_size,
        physical_image_size,
        model_max_r,
        canonical_reduction,
        single_lane_canonical,
        prehalf_weight,
        projector_full,
        rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        output);
    if (validation.failure()) return validation;
    if (padding_factor <= 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2Projector: padding_factor must be positive");

    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();

    cudaError_t err;
    if (prehalf_weight) {
        err = launch_relion_coarse_diff2_projector_prehalf_f32(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else if (single_lane_canonical) {
        err = launch_relion_coarse_diff2_projector_f32<false, true, true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else if (canonical_reduction) {
        err = launch_relion_coarse_diff2_projector_f32<false, true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else {
        err = launch_relion_coarse_diff2_projector_f32(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2ProjectorF32, RelionCoarseDiff2ProjectorF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("model_max_r")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("canonical_reduction")
        .Attr<int64_t>("single_lane_canonical")
        .Attr<int64_t>("prehalf_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2ProjectorMultistreamF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    int64_t canonical_reduction,
    int64_t single_lane_canonical,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer actual_batch_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    ffi::Error validation = ValidateRelionCoarseDiff2ProjectorF32Operands(
        current_size,
        physical_image_size,
        model_max_r,
        canonical_reduction,
        single_lane_canonical,
        prehalf_weight,
        projector_full,
        rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        output);
    if (validation.failure()) return validation;
    if (padding_factor <= 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2Projector: padding_factor must be positive");
    if (actual_batch_size.element_type() != ffi::DataType::S32 ||
        actual_batch_size.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorMultistreamF32: actual_batch_size "
            "must be a scalar S32 runtime operand");

    // Keep final-batch occupancy out of the XLA compile identity.  This tiny
    // same-stream scalar transfer is ordered after the producing computation;
    // all heavy projector/image operands remain resident on device.
    int32_t actual_batch_size_host = 0;
    cudaError_t err = cudaMemcpyAsync(
        &actual_batch_size_host,
        actual_batch_size.untyped_data(),
        sizeof(actual_batch_size_host),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));

    const auto projector_dims = projector_full.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto image_dims = images.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    if (actual_batch_size_host <= 0 ||
        actual_batch_size_host > image_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorMultistreamF32: actual_batch_size "
            "must be within the physical image batch");

    if (prehalf_weight) {
        err = launch_relion_coarse_diff2_projector_prehalf_f32(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            actual_batch_size_host,
            kRelionVdamWorkerStreams,
            static_cast<int>(padding_factor));
    } else if (single_lane_canonical) {
        err = launch_relion_coarse_diff2_projector_f32<false, true, true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            actual_batch_size_host,
            kRelionVdamWorkerStreams,
            static_cast<int>(padding_factor));
    } else if (canonical_reduction) {
        err = launch_relion_coarse_diff2_projector_f32<false, true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            actual_batch_size_host,
            kRelionVdamWorkerStreams,
            static_cast<int>(padding_factor));
    } else {
        err = launch_relion_coarse_diff2_projector_f32<false, false>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            nullptr,
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            actual_batch_size_host,
            kRelionVdamWorkerStreams,
            static_cast<int>(padding_factor));
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2ProjectorMultistreamF32,
    RelionCoarseDiff2ProjectorMultistreamF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("model_max_r")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("canonical_reduction")
        .Attr<int64_t>("single_lane_canonical")
        .Attr<int64_t>("prehalf_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2ProjectorLanesF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t physical_image_size,
    int64_t model_max_r,
    int64_t padding_factor,
    int64_t prehalf_weight,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer images,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> lane_partials)
{
    // Reuse the production handler for the complete operand/shape contract.
    // The diagnostic then invokes a compile-time specialization of the same
    // fused projector kernel; there is no second scoring implementation.
    ffi::Error validation = RelionCoarseDiff2ProjectorF32Impl(
        stream,
        current_size,
        physical_image_size,
        model_max_r,
        padding_factor,
        0,
        0,
        prehalf_weight,
        projector_full,
        rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        output);
    if (validation.failure()) return validation;
    if (padding_factor <= 0)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2Projector: padding_factor must be positive");

    if (lane_partials->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorLanesF32: lane output must be F32");
    const auto image_dims = images.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto projector_dims = projector_full.dimensions();
    const auto lane_dims = lane_partials->dimensions();
    if (lane_dims.size() != 3 ||
        lane_dims[0] != image_dims[0] ||
        lane_dims[1] != rotation_dims[0] ||
        lane_dims[2] != kRelionCoarseDiff2BlockSize)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2ProjectorLanesF32: lane output must have "
            "shape (batch, rotations, 128)");

    cudaError_t err;
    if (prehalf_weight) {
        err = launch_relion_coarse_diff2_projector_prehalf_f32<true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            static_cast<float*>(lane_partials->untyped_data()),
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    } else {
        err = launch_relion_coarse_diff2_projector_f32<true>(
            stream,
            static_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            static_cast<const float2*>(images.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            static_cast<float*>(lane_partials->untyped_data()),
            image_dims[0],
            rotation_dims[0],
            angle_dims[0],
            image_dims[1],
            current_size,
            projector_dims[0],
            model_max_r,
            -static_cast<float>(physical_image_size * physical_image_size),
            image_dims[0],
            0,
            static_cast<int>(padding_factor));
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2ProjectorLanesF32,
    RelionCoarseDiff2ProjectorLanesF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("physical_image_size")
        .Attr<int64_t>("model_max_r")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("prehalf_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2NativeTextureRectangularF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2NativeTextureRectangularF32: projector/image must be C64");
    if (eulers.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2NativeTextureRectangularF32: eulers/angles/weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2NativeTextureRectangularF32: lookup must be S32");

    const auto projector_dims = projector_full.dimensions();
    const auto euler_dims = eulers.dimensions();
    const auto image_dims = image.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const int64_t expected_full_pixels =
        current_size * (current_size / 2 + 1);
    if (current_size <= 0 || padding_factor <= 0 || projector_max_r <= 0 ||
        projector_dims.size() != 3 || projector_dims[0] <= 0 ||
        projector_dims[1] != projector_dims[0] ||
        projector_dims[2] != projector_dims[0] ||
        euler_dims.size() != 2 || euler_dims[0] <= 0 ||
        euler_dims[1] != 9 || image_dims.size() != 2 ||
        image_dims[0] <= 0 || image_dims[1] <= 0 ||
        angle_dims.size() != 2 || angle_dims[0] <= 0 ||
        angle_dims[0] > kRelionCoarseDiff2BlockSize || angle_dims[1] != 2 ||
        weight_dims.size() != 2 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != image_dims[1] || initial_dims.size() != 1 ||
        initial_dims[0] != image_dims[0] || lookup_dims.size() != 1 ||
        lookup_dims[0] != expected_full_pixels || output_dims.size() != 3 ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != euler_dims[0] ||
        output_dims[2] != angle_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2NativeTextureRectangularF32: inconsistent operand shapes");

    const int64_t rotation_blocks =
        (euler_dims[0] + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    if (rotation_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2NativeTextureRectangularF32: block count exceeds CUDA grid");
    cudaError_t err =
        launch_relion_coarse_diff2_native_texture_rectangular_f32(
            stream,
            reinterpret_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(eulers.untyped_data()),
            reinterpret_cast<const float2*>(image.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            projector_dims[0],
            image_dims[0],
            euler_dims[0],
            angle_dims[0],
            image_dims[1],
            lookup_dims[0],
            static_cast<int>(current_size),
            static_cast<int>(padding_factor),
            static_cast<int>(projector_max_r));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2NativeTextureRectangularF32,
    RelionCoarseDiff2NativeTextureRectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseNormalizedCcPairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer score_weight,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer half_weights,
    ffi::AnyBuffer packed_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (shifted_image.element_type() != ffi::DataType::C64 ||
        reference.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: image/reference must be C64");
    if (score_weight.element_type() != ffi::DataType::F32 ||
        half_weights.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: weights/output must be F32");
    if (packed_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: lookup must be S32");

    const auto image_dims = shifted_image.dimensions();
    const auto score_weight_dims = score_weight.dimensions();
    const auto reference_dims = reference.dimensions();
    const auto half_weight_dims = half_weights.dimensions();
    const auto lookup_dims = packed_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (image_dims.size() != 3 || image_dims[0] <= 0 ||
        image_dims[1] <= 0 || image_dims[2] <= 0 ||
        score_weight_dims.size() != 3 || reference_dims.size() != 3 ||
        score_weight_dims[0] != image_dims[0] ||
        score_weight_dims[1] != image_dims[1] ||
        score_weight_dims[2] != image_dims[2] ||
        reference_dims[0] != image_dims[0] ||
        reference_dims[1] != image_dims[1] ||
        reference_dims[2] != image_dims[2] ||
        half_weight_dims.size() != 1 ||
        half_weight_dims[0] != image_dims[2] || lookup_dims.size() != 1 ||
        lookup_dims[0] <= 0 || output_dims.size() != 2 ||
        output_dims[0] != image_dims[0] || output_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: inconsistent operand shapes");

    const int64_t candidate_count = image_dims[0] * image_dims[1];
    if (candidate_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcPairsF32: candidate count exceeds CUDA grid");
    cudaError_t err = launch_relion_coarse_normalized_cc_pairs_f32(
        stream,
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(score_weight.untyped_data()),
        reinterpret_cast<const float2*>(reference.untyped_data()),
        static_cast<const float*>(half_weights.untyped_data()),
        static_cast<const int32_t*>(packed_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        candidate_count,
        image_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseNormalizedCcPairsF32,
    RelionCoarseNormalizedCcPairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseNormalizedCcNativeTexturePairsF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    ffi::AnyBuffer projector_full,
    ffi::AnyBuffer eulers,
    ffi::AnyBuffer unshifted_image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer score_weight,
    ffi::AnyBuffer numerator_weight,
    ffi::AnyBuffer half_weights,
    ffi::AnyBuffer packed_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_full.element_type() != ffi::DataType::C64 ||
        unshifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: projector/image must be C64");
    if (eulers.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        score_weight.element_type() != ffi::DataType::F32 ||
        numerator_weight.element_type() != ffi::DataType::F32 ||
        half_weights.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: eulers/weights/output must be F32");
    if (packed_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: lookup must be S32");

    const auto projector_dims = projector_full.dimensions();
    const auto euler_dims = eulers.dimensions();
    const auto image_dims = unshifted_image.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto score_weight_dims = score_weight.dimensions();
    const auto numerator_weight_dims = numerator_weight.dimensions();
    const auto half_weight_dims = half_weights.dimensions();
    const auto lookup_dims = packed_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const int64_t expected_packed_pixels =
        current_size * (current_size / 2 + 1);
    if (current_size <= 0 || padding_factor <= 0 || projector_max_r <= 0 ||
        projector_dims.size() != 3 || projector_dims[0] <= 0 ||
        projector_dims[1] != projector_dims[0] ||
        projector_dims[2] != projector_dims[0] ||
        euler_dims.size() != 2 || euler_dims[0] <= 0 ||
        euler_dims[1] != 9 || image_dims.size() != 2 ||
        image_dims[0] != euler_dims[0] || image_dims[1] <= 0 ||
        angle_dims.size() != 2 || angle_dims[0] != image_dims[0] ||
        angle_dims[1] != 2 ||
        score_weight_dims.size() != 2 ||
        score_weight_dims[0] != image_dims[0] ||
        score_weight_dims[1] != image_dims[1] ||
        numerator_weight_dims.size() != 2 ||
        numerator_weight_dims[0] != image_dims[0] ||
        numerator_weight_dims[1] != image_dims[1] ||
        half_weight_dims.size() != 1 ||
        half_weight_dims[0] != image_dims[1] ||
        lookup_dims.size() != 1 ||
        lookup_dims[0] != expected_packed_pixels ||
        output_dims.size() != 2 || output_dims[0] != image_dims[0] ||
        output_dims[1] != 3)
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: inconsistent operand shapes");
    if (image_dims[0] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseNormalizedCcNativeTexturePairsF32: candidate count exceeds CUDA grid");

    cudaError_t err =
        launch_relion_coarse_normalized_cc_native_texture_pairs_f32(
            stream,
            reinterpret_cast<const float2*>(projector_full.untyped_data()),
            static_cast<const float*>(eulers.untyped_data()),
            reinterpret_cast<const float2*>(unshifted_image.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(score_weight.untyped_data()),
            static_cast<const float*>(numerator_weight.untyped_data()),
            static_cast<const float*>(half_weights.untyped_data()),
            static_cast<const int32_t*>(packed_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            projector_dims[0],
            image_dims[0],
            image_dims[1],
            lookup_dims[0],
            static_cast<int>(current_size),
            static_cast<int>(padding_factor),
            static_cast<int>(projector_max_r));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseNormalizedCcNativeTexturePairsF32,
    RelionCoarseNormalizedCcNativeTexturePairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionCoarseDiff2RectangularF64Impl(
    cudaStream_t stream, ffi::AnyBuffer reference, ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight, ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact, ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C128 ||
        shifted_image.element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF64: reference/image must be C128");
    if (weight.element_type() != ffi::DataType::F64 ||
        initial_diff2.element_type() != ffi::DataType::F64 ||
        output->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF64: weight/initial/output must be F64");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF64: lookup must be S32");
    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 2 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 || output_dims.size() != 3 ||
        reference_dims[0] <= 0 || reference_dims[1] <= 0 ||
        image_dims[0] <= 0 || image_dims[1] <= 0 ||
        image_dims[1] > kRelionCoarseDiff2BlockSize ||
        image_dims[2] != reference_dims[1] ||
        weight_dims[0] != image_dims[0] || weight_dims[1] != reference_dims[1] ||
        initial_dims[0] != image_dims[0] || lookup_dims[0] <= 0 ||
        output_dims[0] != image_dims[0] || output_dims[1] != reference_dims[0] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2RectangularF64: inconsistent operand shapes");
    cudaError_t err = launch_relion_coarse_diff2_rectangular_f64(
        stream, reinterpret_cast<const double2*>(reference.untyped_data()),
        reinterpret_cast<const double2*>(shifted_image.untyped_data()),
        static_cast<const double*>(weight.untyped_data()),
        static_cast<const double*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<double*>(output->untyped_data()), image_dims[0],
        reference_dims[0], image_dims[1], reference_dims[1], lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2RectangularF64, RelionCoarseDiff2RectangularF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        initial_dims[0] != reference_dims[0] ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular<float, float2, true>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        image_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularF32, RelionFineDiff2RectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularMaskedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer candidate_mask,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: lookup must be S32");
    if (candidate_mask.element_type() != ffi::DataType::PRED &&
        candidate_mask.element_type() != ffi::DataType::U8 &&
        candidate_mask.element_type() != ffi::DataType::S8)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: candidate_mask must be PRED/U8/S8");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto mask_dims = candidate_mask.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 || mask_dims.size() != 3 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        initial_dims[0] != reference_dims[0] ||
        mask_dims[0] != reference_dims[0] ||
        mask_dims[1] != reference_dims[1] ||
        mask_dims[2] != image_dims[1] ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularMaskedF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular_masked<float, float2>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<const uint8_t*>(candidate_mask.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        image_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularMaskedF32, RelionFineDiff2RectangularMaskedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionPowerClassSpectrumHighresF32Impl(
    cudaStream_t stream,
    int64_t xdim,
    int64_t ydim,
    int64_t resolution_limit,
    ffi::AnyBuffer image,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (image.element_type() != ffi::DataType::C64 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionPowerClassSpectrumHighresF32: image/output must be C64/F32");
    const auto image_dims = image.dimensions();
    const auto output_dims = output->dimensions();
    if (image_dims.size() != 2 || output_dims.size() != 2 ||
        image_dims[0] <= 0 || image_dims[1] <= 0 ||
        xdim <= 0 || ydim <= 0 || image_dims[1] != xdim * ydim ||
        output_dims[0] != image_dims[0] || output_dims[1] != xdim + 1 ||
        resolution_limit < 0 || resolution_limit > xdim)
        return ffi::Error::InvalidArgument(
            "RelionPowerClassSpectrumHighresF32: inconsistent dimensions");
    if (image_dims[0] > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        image_dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        xdim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        ydim > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionPowerClassSpectrumHighresF32: dimensions exceed CUDA limits");
    cudaError_t err = launch_relion_powerclass_spectrum_highres_f32(
        stream,
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        static_cast<int>(image_dims[1]),
        static_cast<int>(xdim),
        static_cast<int>(xdim),
        static_cast<int>(ydim),
        static_cast<int>(resolution_limit));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPowerClassSpectrumHighresF32,
    RelionPowerClassSpectrumHighresF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("xdim")
        .Attr<int64_t>("ydim")
        .Attr<int64_t>("resolution_limit")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionPowerClassSpectrumHighresRuntimeF32Impl(
    cudaStream_t stream,
    int64_t xdim,
    int64_t ydim,
    ffi::AnyBuffer image,
    ffi::AnyBuffer resolution_limit,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (image.element_type() != ffi::DataType::C64 ||
        output->element_type() != ffi::DataType::F32 ||
        resolution_limit.element_type() != ffi::DataType::S32 ||
        resolution_limit.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionPowerClassSpectrumHighresRuntimeF32: invalid buffers");
    const auto image_dims = image.dimensions();
    const auto output_dims = output->dimensions();
    if (image_dims.size() != 2 || output_dims.size() != 2 ||
        image_dims[0] <= 0 || image_dims[1] <= 0 || xdim <= 0 || ydim <= 0 ||
        image_dims[1] != xdim * ydim || output_dims[0] != image_dims[0] ||
        output_dims[1] != xdim + 1 ||
        image_dims[0] > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        image_dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        xdim > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        ydim > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionPowerClassSpectrumHighresRuntimeF32: inconsistent dimensions");
    cudaError_t err = launch_relion_powerclass_spectrum_highres_runtime_f32(
        stream,
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        static_cast<int>(image_dims[1]),
        static_cast<int>(xdim),
        static_cast<int>(xdim),
        static_cast<int>(ydim),
        static_cast<const int32_t*>(resolution_limit.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPowerClassSpectrumHighresRuntimeF32,
    RelionPowerClassSpectrumHighresRuntimeF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("xdim")
        .Attr<int64_t>("ydim")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateRectangularF32Common(
    cudaStream_t stream,
    int64_t current_size,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: reference/image must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: angles/weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: lookup must be S32");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRuntimeRectangularF32: "
            "logical_current_size must be an S32 scalar");
    if (runtime_current_size == nullptr &&
        (current_size <= 0 || current_size > std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: invalid current_size");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = image.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const bool lookup_shape_valid = lookup_dims.size() == 1 && lookup_dims[0] > 0;
    const int64_t expected_full_pixels = runtime_current_size == nullptr
        ? current_size * (current_size / 2 + 1)
        : (lookup_shape_valid ? lookup_dims[0] : 0);
    if (reference_dims.size() != 3 || image_dims.size() != 2 ||
        translation_dims.size() != 2 || translation_dims[1] != 2 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        !lookup_shape_valid ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] != reference_dims[2] ||
        translation_dims[0] <= 0 ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] ||
        initial_dims[0] != reference_dims[0] ||
        lookup_dims[0] != expected_full_pixels ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != translation_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: inconsistent operand shapes");

    const int64_t translation_chunks =
        (translation_dims[0] + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks =
        reference_dims[0] * reference_dims[1] * translation_chunks;
    if (total_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRectangularF32: block count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_fused_translate_rectangular_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        translation_dims[0],
        reference_dims[2],
        lookup_dims[0],
        static_cast<int>(current_size),
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFineDiff2FusedTranslateRectangularF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateRectangularF32Common(
        stream, current_size, nullptr, reference, image, translation_angles,
        weight, initial_diff2, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRectangularF32,
    RelionFineDiff2FusedTranslateRectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateFlatRowsF32Common(
    cudaStream_t stream,
    int64_t current_size,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: reference/image must be C64");
    if (row_image_ids.element_type() != ffi::DataType::S32 ||
        full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: row ids/lookup must be S32");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: angles/weight/initial/output must be F32");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRuntimeFlatRowsF32: "
            "logical_current_size must be an S32 scalar");
    if (runtime_current_size == nullptr &&
        (current_size <= 0 || current_size > std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: invalid current_size");

    const auto reference_dims = reference.dimensions();
    const auto row_image_dims = row_image_ids.dimensions();
    const auto image_dims = image.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const bool lookup_shape_valid = lookup_dims.size() == 1 && lookup_dims[0] > 0;
    const int64_t expected_full_pixels = runtime_current_size == nullptr
        ? current_size * (current_size / 2 + 1)
        : (lookup_shape_valid ? lookup_dims[0] : 0);
    if (reference_dims.size() != 2 || row_image_dims.size() != 1 ||
        image_dims.size() != 2 || translation_dims.size() != 2 ||
        translation_dims[1] != 2 || weight_dims.size() != 2 ||
        initial_dims.size() != 1 || !lookup_shape_valid ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || row_image_dims[0] != reference_dims[0] ||
        image_dims[0] <= 0 || image_dims[1] != reference_dims[1] ||
        translation_dims[0] <= 0 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != expected_full_pixels ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != translation_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: inconsistent operand shapes");

    const int64_t translation_chunks =
        (translation_dims[0] + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks = reference_dims[0] * translation_chunks;
    if (total_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateFlatRowsF32: block count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_fused_translate_flat_rows_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        reference_dims[0],
        translation_dims[0],
        reference_dims[1],
        lookup_dims[0],
        static_cast<int>(current_size),
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFineDiff2FusedTranslateFlatRowsF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateFlatRowsF32Common(
        stream, current_size, nullptr, reference, row_image_ids, image,
        translation_angles, weight, initial_diff2, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateFlatRowsF32,
    RelionFineDiff2FusedTranslateFlatRowsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateRuntimeFlatRowsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateFlatRowsF32Common(
        stream, 0, &logical_current_size, reference, row_image_ids, image,
        translation_angles, weight, initial_diff2, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimeFlatRowsF32,
    RelionFineDiff2FusedTranslateRuntimeFlatRowsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslatePairsF32Common(
    cudaStream_t stream,
    int64_t current_size,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer pair_reference_rows,
    ffi::AnyBuffer pair_translation_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: reference/image must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: angles/weight/initial/output must be F32");
    if (pair_reference_rows.element_type() != ffi::DataType::S32 ||
        pair_translation_ids.element_type() != ffi::DataType::S32 ||
        full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: pair ids/lookup must be S32");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRuntimePairsF32: "
            "logical_current_size must be an S32 scalar");
    if (runtime_current_size == nullptr &&
        (current_size <= 0 || current_size > std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: invalid current_size");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = image.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto pair_reference_dims = pair_reference_rows.dimensions();
    const auto pair_translation_dims = pair_translation_ids.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const bool lookup_shape_valid = lookup_dims.size() == 1 && lookup_dims[0] > 0;
    const int64_t expected_full_pixels = runtime_current_size == nullptr
        ? current_size * (current_size / 2 + 1)
        : (lookup_shape_valid ? lookup_dims[0] : 0);
    if (reference_dims.size() != 2 || image_dims.size() != 2 ||
        translation_dims.size() != 2 || translation_dims[1] != 2 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        pair_reference_dims.size() != 2 || pair_translation_dims.size() != 2 ||
        !lookup_shape_valid || output_dims.size() != 2 ||
        reference_dims[0] <= 0 || reference_dims[1] <= 0 ||
        image_dims[0] <= 0 || image_dims[1] != reference_dims[1] ||
        translation_dims[0] <= 0 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        pair_reference_dims[0] != image_dims[0] ||
        pair_reference_dims[1] <= 0 ||
        pair_translation_dims[0] != pair_reference_dims[0] ||
        pair_translation_dims[1] != pair_reference_dims[1] ||
        lookup_dims[0] != expected_full_pixels ||
        output_dims[0] != pair_reference_dims[0] ||
        output_dims[1] != pair_reference_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: inconsistent operand shapes");

    const int64_t total_hypotheses =
        pair_reference_dims[0] * pair_reference_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslatePairsF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_fused_translate_pairs_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(pair_reference_rows.untyped_data()),
        static_cast<const int32_t*>(pair_translation_ids.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        reference_dims[0],
        pair_reference_dims[1],
        translation_dims[0],
        reference_dims[1],
        lookup_dims[0],
        static_cast<int>(current_size),
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFineDiff2FusedTranslatePairsF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer pair_reference_rows,
    ffi::AnyBuffer pair_translation_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslatePairsF32Common(
        stream, current_size, nullptr, reference, image, translation_angles,
        weight, initial_diff2, pair_reference_rows, pair_translation_ids,
        full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslatePairsF32,
    RelionFineDiff2FusedTranslatePairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateRuntimePairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer pair_reference_rows,
    ffi::AnyBuffer pair_translation_ids,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslatePairsF32Common(
        stream, 0, &logical_current_size, reference, image,
        translation_angles, weight, initial_diff2, pair_reference_rows,
        pair_translation_ids, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimePairsF32,
    RelionFineDiff2FusedTranslateRuntimePairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateJobsF32Common(
    cudaStream_t stream,
    int64_t current_size,
    const ffi::AnyBuffer* runtime_current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer job_plan,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: reference/image must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: angles/weight/initial/output must be F32");
    if (job_plan.element_type() != ffi::DataType::S32 ||
        full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: job plan/lookup must be S32");
    if (runtime_current_size != nullptr &&
        (runtime_current_size->element_type() != ffi::DataType::S32 ||
         runtime_current_size->dimensions().size() != 0))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateRuntimeJobsF32: "
            "logical_current_size must be an S32 scalar");
    if (runtime_current_size == nullptr &&
        (current_size <= 0 || current_size > std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: invalid current_size");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = image.dimensions();
    const auto translation_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto job_dims = job_plan.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const bool lookup_shape_valid = lookup_dims.size() == 1 && lookup_dims[0] > 0;
    const int64_t expected_full_pixels = runtime_current_size == nullptr
        ? current_size * (current_size / 2 + 1)
        : (lookup_shape_valid ? lookup_dims[0] : 0);
    if (reference_dims.size() != 2 || image_dims.size() != 2 ||
        translation_dims.size() != 2 || translation_dims[1] != 2 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        job_dims.size() != 2 || job_dims[0] <= 0 || job_dims[1] != 4 ||
        !lookup_shape_valid || output_dims.size() != 1 ||
        reference_dims[0] <= 0 || reference_dims[1] <= 0 ||
        image_dims[0] <= 0 || image_dims[1] != reference_dims[1] ||
        translation_dims[0] <= 0 || weight_dims[0] != image_dims[0] ||
        weight_dims[1] != reference_dims[1] ||
        initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != expected_full_pixels || output_dims[0] != job_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: inconsistent operand shapes");

    const int64_t total_blocks =
        (job_dims[0] + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    if (total_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2FusedTranslateJobsF32: block count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_fused_translate_jobs_f32(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(image.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(initial_diff2.untyped_data()),
        static_cast<const int32_t*>(job_plan.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        image_dims[0],
        reference_dims[0],
        job_dims[0],
        translation_dims[0],
        reference_dims[1],
        lookup_dims[0],
        static_cast<int>(current_size),
        runtime_current_size == nullptr
            ? nullptr
            : static_cast<const int32_t*>(runtime_current_size->untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFineDiff2FusedTranslateJobsF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer job_plan,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateJobsF32Common(
        stream, current_size, nullptr, reference, image, translation_angles,
        weight, initial_diff2, job_plan, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateJobsF32,
    RelionFineDiff2FusedTranslateJobsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateRuntimeJobsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer job_plan,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateJobsF32Common(
        stream, 0, &logical_current_size, reference, image,
        translation_angles, weight, initial_diff2, job_plan,
        full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimeJobsF32,
    RelionFineDiff2FusedTranslateRuntimeJobsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2FusedTranslateRuntimeRectangularF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer image,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer initial_diff2,
    ffi::AnyBuffer full_to_compact,
    ffi::AnyBuffer logical_current_size,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionFineDiff2FusedTranslateRectangularF32Common(
        stream, 0, &logical_current_size, reference, image,
        translation_angles, weight, initial_diff2, full_to_compact, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2FusedTranslateRuntimeRectangularF32,
    RelionFineDiff2FusedTranslateRuntimeRectangularF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2PairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C64 ||
        shifted_image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: reference/image must be C64");
    if (weight.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: weight/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] != reference_dims[1] ||
        image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: inconsistent operand shapes");

    const int64_t total_hypotheses = reference_dims[0] * reference_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF32: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_pairs<float, float2>(
        stream,
        reinterpret_cast<const float2*>(reference.untyped_data()),
        reinterpret_cast<const float2*>(shifted_image.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        reference_dims[0],
        reference_dims[1],
        reference_dims[2],
        lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2PairsF32, RelionFineDiff2PairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2RectangularF64Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C128 ||
        shifted_image.element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: reference/image must be C128");
    if (weight.element_type() != ffi::DataType::F64 ||
        output->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: weight/output must be F64");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 3 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] <= 0 || image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1] ||
        output_dims[2] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: inconsistent operand shapes");

    const int64_t total_hypotheses =
        reference_dims[0] * reference_dims[1] * image_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2RectangularF64: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_rectangular<double, double2, false>(
        stream,
        reinterpret_cast<const double2*>(reference.untyped_data()),
        reinterpret_cast<const double2*>(shifted_image.untyped_data()),
        static_cast<const double*>(weight.untyped_data()),
        nullptr,
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<double*>(output->untyped_data()),
        reference_dims[0], reference_dims[1], image_dims[1],
        reference_dims[2], lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2RectangularF64, RelionFineDiff2RectangularF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFineDiff2PairsF64Impl(
    cudaStream_t stream,
    ffi::AnyBuffer reference,
    ffi::AnyBuffer shifted_image,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer full_to_compact,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (reference.element_type() != ffi::DataType::C128 ||
        shifted_image.element_type() != ffi::DataType::C128)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: reference/image must be C128");
    if (weight.element_type() != ffi::DataType::F64 ||
        output->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: weight/output must be F64");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = shifted_image.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    if (reference_dims.size() != 3 || image_dims.size() != 3 ||
        weight_dims.size() != 2 || lookup_dims.size() != 1 ||
        output_dims.size() != 2 || reference_dims[0] <= 0 ||
        reference_dims[1] <= 0 || reference_dims[2] <= 0 ||
        image_dims[0] != reference_dims[0] ||
        image_dims[1] != reference_dims[1] ||
        image_dims[2] != reference_dims[2] ||
        weight_dims[0] != reference_dims[0] ||
        weight_dims[1] != reference_dims[2] || lookup_dims[0] <= 0 ||
        output_dims[0] != reference_dims[0] ||
        output_dims[1] != reference_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: inconsistent operand shapes");

    const int64_t total_hypotheses = reference_dims[0] * reference_dims[1];
    if (total_hypotheses > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionFineDiff2PairsF64: hypothesis count exceeds CUDA grid");
    cudaError_t err = launch_relion_fine_diff2_pairs<double, double2>(
        stream,
        reinterpret_cast<const double2*>(reference.untyped_data()),
        reinterpret_cast<const double2*>(shifted_image.untyped_data()),
        static_cast<const double*>(weight.untyped_data()),
        static_cast<const int32_t*>(full_to_compact.untyped_data()),
        static_cast<double*>(output->untyped_data()),
        reference_dims[0], reference_dims[1], reference_dims[2], lookup_dims[0]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFineDiff2PairsF64, RelionFineDiff2PairsF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionPreprocessRealF32ImplWithReduction(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out,
    int reduction_mode)
{
    if (invalid_count_out->element_type() != ffi::DataType::S32 ||
        invalid_count_out->dimensions().size() != 1 || invalid_count_out->dimensions()[0] != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: invalid_count must be S32 with shape (1,)");
    if (host_check != 0 && host_check != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: host_check must be 0 or 1");
    if (images.element_type() != ffi::DataType::F32 ||
        normalization_factors.element_type() != ffi::DataType::F32 ||
        normalized_shifted_out->element_type() != ffi::DataType::F32 ||
        masked_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: images/factors/outputs must be F32");
    if (integer_shifts.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: shifts must be S32");
    auto image_dims = images.dimensions();
    auto factor_dims = normalization_factors.dimensions();
    auto shift_dims = integer_shifts.dimensions();
    auto normshift_dims = normalized_shifted_out->dimensions();
    auto masked_dims = masked_out->dimensions();
    if (image_dims.size() != 3 || image_dims[0] <= 0 || image_dims[1] <= 0 || image_dims[1] != image_dims[2])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: images must have shape (batch,D,D)");
    if (factor_dims.size() != 1 || factor_dims[0] != image_dims[0])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: factors must have shape (batch,)");
    if (shift_dims.size() != 2 || shift_dims[0] != image_dims[0] || shift_dims[1] != 2)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: shifts must have shape (batch,2)");
    if (normshift_dims.size() != 3 || masked_dims.size() != 3 ||
        normshift_dims[0] != image_dims[0] || normshift_dims[1] != image_dims[1] ||
        normshift_dims[2] != image_dims[2] || masked_dims[0] != image_dims[0] ||
        masked_dims[1] != image_dims[1] || masked_dims[2] != image_dims[2])
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: output shapes must match images");
    if (!(radius > 0.0f) || !(cosine_width > 0.0f) ||
        !std::isfinite(radius) || !std::isfinite(cosine_width))
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: radius/width must be finite and positive");
    if (apply_mask != 0 && apply_mask != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: apply_mask must be 0 or 1");

    RelionPreprocessScratchLayout layout;
    cudaError_t err = relion_preprocess_scratch_layout(image_dims[0], reduction_mode, &layout);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    // XLA provisions the soft-mask workspace as an output buffer sized by the
    // Python wrapper from the same layout constants; fail closed if it is
    // smaller than this build's layout (for example a larger CUB temporary).
    if (workspace_out->element_type() != ffi::DataType::U8 || workspace_out->dimensions().size() != 1)
        return ffi::Error::InvalidArgument("RelionPreprocessRealF32: workspace must be U8 with shape (bytes,)");
    size_t workspace_bytes = static_cast<size_t>(workspace_out->dimensions()[0]);
    if (apply_mask != 0 && workspace_bytes < layout.total_bytes)
        return ffi::Error::InvalidArgument(
            "RelionPreprocessRealF32: workspace too small: need " + std::to_string(layout.total_bytes) +
            " bytes, got " + std::to_string(workspace_bytes));
    void* scratch_ptr = apply_mask != 0 ? workspace_out->untyped_data() : nullptr;
    err = launch_relion_preprocess_real_f32(
        stream,
        static_cast<const float*>(images.untyped_data()),
        static_cast<const float*>(normalization_factors.untyped_data()),
        static_cast<const int32_t*>(integer_shifts.untyped_data()),
        static_cast<float*>(normalized_shifted_out->untyped_data()),
        static_cast<float*>(masked_out->untyped_data()),
        image_dims[0], static_cast<int>(image_dims[1]), static_cast<int>(image_dims[2]),
        radius, cosine_width, apply_mask != 0, reduction_mode, host_check != 0,
        static_cast<int32_t*>(invalid_count_out->untyped_data()), scratch_ptr, layout);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionPreprocessRealF32Impl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 0);
}

ffi::Error RelionPreprocessRealF32NativeLaneImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 1);
}

ffi::Error RelionPreprocessRealF32NativeAtomicImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    int64_t host_check,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    ffi::Result<ffi::AnyBuffer> workspace_out,
    ffi::Result<ffi::AnyBuffer> invalid_count_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, host_check, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, workspace_out, invalid_count_out, 2);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32, RelionPreprocessRealF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32NativeLane, RelionPreprocessRealF32NativeLaneImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32NativeAtomic, RelionPreprocessRealF32NativeAtomicImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Attr<int64_t>("host_check")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionFusedXHalfBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer data_rows,
    ffi::AnyBuffer weight_rows,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer data_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out)
{
    const bool use_f64 = data_rows.element_type() == ffi::DataType::C128;
    const ffi::DataType complex_type = use_f64 ? ffi::DataType::C128 : ffi::DataType::C64;
    const ffi::DataType real_type = use_f64 ? ffi::DataType::F64 : ffi::DataType::F32;
    if (data_rows.element_type() != complex_type ||
        data_volume_in.element_type() != complex_type ||
        data_volume_out->element_type() != complex_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: data rows and volumes must share complex64 or complex128 dtype");
    if (weight_rows.element_type() != real_type ||
        weight_volume_in.element_type() != real_type ||
        weight_volume_out->element_type() != real_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: weight dtype must match the complex precision");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: pixel indices must be int32");
    if (rot.element_type() != real_type)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rotation dtype must match the accumulator precision");
    if (order != 1 || half_volume != 1 || half_image != 1)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: requires order=1 and half image/volume");
    if (N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: volume must be positive, cubic, and odd-sized");
    if (image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: image attrs must describe a native FFTW half square");
    if (upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: requires positive upsampling and an explicit radius");

    const auto data_row_dims = data_rows.dimensions();
    const auto weight_row_dims = weight_rows.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    const auto data_in_dims = data_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_out_dims = data_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_row_dims.size() != 2 || weight_row_dims.size() != 2 ||
        data_row_dims[0] <= 0 || data_row_dims[1] <= 0 ||
        weight_row_dims[0] != data_row_dims[0] ||
        weight_row_dims[1] != data_row_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: data/weight rows must have matching nonempty rank-2 shapes");
    if (pixel_dims.size() != 1 || pixel_dims[0] != data_row_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: pixel index length must match row width");
    if (rot_dims.size() != 2 || rot_dims[0] != data_row_dims[0] || rot_dims[1] != 6)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rotations must have shape (n_rows, 6)");
    if (data_row_dims[1] != image_h * image_w)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: rows must contain the full native FFTW square");

    const int64_t expected_volume_size = N0 * N1 * (N2 / 2 + 1);
    if (data_in_dims.size() != 1 || weight_in_dims.size() != 1 ||
        data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_in_dims[0] != expected_volume_size ||
        weight_in_dims[0] != expected_volume_size ||
        data_out_dims[0] != expected_volume_size ||
        weight_out_dims[0] != expected_volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackproject: accumulator sizes do not match the half volume");

    cudaError_t err;
    if (use_f64) {
        err = launch_relion_fused_x_half_backproject_f64(
            stream,
            reinterpret_cast<double2*>(data_volume_out->untyped_data()),
            static_cast<double*>(weight_volume_out->untyped_data()),
            reinterpret_cast<const double2*>(data_rows.untyped_data()),
            static_cast<const double*>(weight_rows.untyped_data()),
            static_cast<const int32_t*>(pixel_indices.untyped_data()),
            static_cast<const double*>(rot.untyped_data()),
            data_row_dims[0], data_row_dims[1], image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
    } else {
        err = launch_relion_fused_x_half_backproject(
            stream,
            reinterpret_cast<float2*>(data_volume_out->untyped_data()),
            static_cast<float*>(weight_volume_out->untyped_data()),
            reinterpret_cast<const float2*>(data_rows.untyped_data()),
            static_cast<const float*>(weight_rows.untyped_data()),
            static_cast<const int32_t*>(pixel_indices.untyped_data()),
            static_cast<const float*>(rot.untyped_data()),
            data_row_dims[0], data_row_dims[1], image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionFusedXHalfBackprojectParticleGridImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4, int64_t n_particles, int64_t rows_per_particle,
    ffi::AnyBuffer data_rows,
    ffi::AnyBuffer weight_rows,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer data_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out)
{
    if (data_rows.element_type() != ffi::DataType::C64 ||
        data_volume_in.element_type() != ffi::DataType::C64 ||
        data_volume_out->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: data rows and volumes must be complex64");
    if (weight_rows.element_type() != ffi::DataType::F32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: weight rows and volumes must be float32");
    if (pixel_indices.element_type() != ffi::DataType::S32 ||
        rot.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: indices/rotations have invalid dtypes");
    if (order != 1 || half_volume != 1 || half_image != 1 ||
        N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h ||
        upsampling <= 0 || max_r2_x4 < 0 ||
        n_particles <= 0 || rows_per_particle <= 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: invalid strict x-half attributes");

    const auto data_dims = data_rows.dimensions();
    const auto weight_dims = weight_rows.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    const int64_t n_rows = n_particles * rows_per_particle;
    if (data_dims.size() != 2 || data_dims[0] != n_rows || data_dims[1] <= 0 ||
        weight_dims.size() != 2 || weight_dims[0] != n_rows ||
        weight_dims[1] != data_dims[1] ||
        pixel_dims.size() != 1 || pixel_dims[0] != data_dims[1] ||
        rot_dims.size() != 2 || rot_dims[0] != n_rows || rot_dims[1] != 6 ||
        data_dims[1] != image_h * image_w)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: inconsistent row topology");

    const int64_t volume_size = N0 * N1 * (N2 / 2 + 1);
    const auto data_in_dims = data_volume_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_out_dims = data_volume_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_in_dims.size() != 1 || data_in_dims[0] != volume_size ||
        weight_in_dims.size() != 1 || weight_in_dims[0] != volume_size ||
        data_out_dims.size() != 1 || data_out_dims[0] != volume_size ||
        weight_out_dims.size() != 1 || weight_out_dims[0] != volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectParticleGrid: accumulator sizes do not match");

    auto* data_out = reinterpret_cast<float2*>(data_volume_out->untyped_data());
    auto* weight_out = static_cast<float*>(weight_volume_out->untyped_data());
    const auto* data = reinterpret_cast<const float2*>(data_rows.untyped_data());
    const auto* weight = static_cast<const float*>(weight_rows.untyped_data());
    const auto* indices = static_cast<const int32_t*>(pixel_indices.untyped_data());
    const auto* rotations = static_cast<const float*>(rot.untyped_data());
    const int64_t n_pixels = data_dims[1];
    const int64_t row_stride = rows_per_particle * n_pixels;
    const int64_t rotation_stride = rows_per_particle * 6;
    for (int64_t particle = 0; particle < n_particles; ++particle) {
        cudaError_t err = launch_relion_fused_x_half_backproject(
            stream, data_out, weight_out,
            data + particle * row_stride,
            weight + particle * row_stride,
            indices,
            rotations + particle * rotation_stride,
            rows_per_particle, n_pixels, image_h, image_w,
            N0, N1, N2, upsampling, max_r2_x4);
        if (err != cudaSuccess)
            return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error RelionFirstiterBprefFusedXHalfImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    float significant_weight,
    float weight_norm,
    ffi::AnyBuffer image_real,
    ffi::AnyBuffer image_imag,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_x,
    ffi::AnyBuffer translation_y,
    ffi::AnyBuffer native_eulers,
    ffi::AnyBuffer data_volume_real_in,
    ffi::AnyBuffer data_volume_imag_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_real_out,
    ffi::Result<ffi::AnyBuffer> data_volume_imag_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out)
{
    if (image_real.element_type() != ffi::DataType::F32 ||
        image_imag.element_type() != ffi::DataType::F32 ||
        ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        translation_x.element_type() != ffi::DataType::F32 ||
        translation_y.element_type() != ffi::DataType::F32 ||
        native_eulers.element_type() != ffi::DataType::F32 ||
        data_volume_real_in.element_type() != ffi::DataType::F32 ||
        data_volume_imag_in.element_type() != ffi::DataType::F32 ||
        data_volume_real_out->element_type() != ffi::DataType::F32 ||
        data_volume_imag_out->element_type() != ffi::DataType::F32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: exact native operands must be float32");
    if (order != 1 || half_volume != 1 || half_image != 1)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: requires order=1 and half image/volume");
    if (N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: volume must be positive, cubic, and odd-sized");
    if (image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: image attrs must describe a native FFTW half square");
    if (upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: requires positive upsampling and an explicit radius");

    const auto image_real_dims = image_real.dimensions();
    const auto image_imag_dims = image_imag.dimensions();
    const auto ctf_dims = ctf.dimensions();
    const auto minv_dims = minvsigma2.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto translation_x_dims = translation_x.dimensions();
    const auto translation_y_dims = translation_y.dimensions();
    const auto euler_dims = native_eulers.dimensions();
    const int64_t n_pixels = image_h * image_w;
    if (image_real_dims.size() != 1 || image_imag_dims.size() != 1 ||
        ctf_dims.size() != 1 || minv_dims.size() != 1 ||
        image_real_dims[0] != n_pixels || image_imag_dims[0] != n_pixels ||
        ctf_dims[0] != n_pixels || minv_dims[0] != n_pixels)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: image/CTF/noise rows must match the dense FFTW square");
    if (posterior_dims.size() != 2 || posterior_dims[0] <= 0 || posterior_dims[1] <= 0)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: posterior must be a nonempty rotation/translation matrix");
    if (translation_x_dims.size() != 1 || translation_y_dims.size() != 1 ||
        translation_x_dims[0] != posterior_dims[1] ||
        translation_y_dims[0] != posterior_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: split translation arrays have wrong shape");
    if (euler_dims.size() != 2 || euler_dims[0] != posterior_dims[0] || euler_dims[1] != 9)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: native Euler matrices must have shape (n_rotations, 9)");
    if (!std::isfinite(significant_weight) || !std::isfinite(weight_norm) ||
        weight_norm <= 0.0f)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: scalar attributes are invalid");

    const int64_t expected_volume_size = N0 * N1 * (N2 / 2 + 1);
    const auto data_real_in_dims = data_volume_real_in.dimensions();
    const auto data_imag_in_dims = data_volume_imag_in.dimensions();
    const auto weight_in_dims = weight_volume_in.dimensions();
    const auto data_real_out_dims = data_volume_real_out->dimensions();
    const auto data_imag_out_dims = data_volume_imag_out->dimensions();
    const auto weight_out_dims = weight_volume_out->dimensions();
    if (data_real_in_dims.size() != 1 || data_imag_in_dims.size() != 1 ||
        weight_in_dims.size() != 1 || data_real_out_dims.size() != 1 ||
        data_imag_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_real_in_dims[0] != expected_volume_size ||
        data_imag_in_dims[0] != expected_volume_size ||
        weight_in_dims[0] != expected_volume_size ||
        data_real_out_dims[0] != expected_volume_size ||
        data_imag_out_dims[0] != expected_volume_size ||
        weight_out_dims[0] != expected_volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFirstiterBprefFusedXHalf: accumulator sizes do not match the half volume");

    cudaError_t err = launch_relion_firstiter_bpref_fused_x_half(
        stream,
        static_cast<float*>(data_volume_real_out->untyped_data()),
        static_cast<float*>(data_volume_imag_out->untyped_data()),
        static_cast<float*>(weight_volume_out->untyped_data()),
        static_cast<const float*>(image_real.untyped_data()),
        static_cast<const float*>(image_imag.untyped_data()),
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<const float*>(translation_x.untyped_data()),
        static_cast<const float*>(translation_y.untyped_data()),
        static_cast<const float*>(native_eulers.untyped_data()),
        significant_weight,
        weight_norm,
        posterior_dims[0], posterior_dims[1], image_h, image_w,
        N0, N1, N2, upsampling, max_r2_x4);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}


ffi::Error RelionFusedXHalfBackprojectSignatureImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer data_rows,
    ffi::AnyBuffer weight_rows,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer canonical_rotation_keys,
    ffi::AnyBuffer signature_row_indices,
    ffi::AnyBuffer data_volume_in,
    ffi::AnyBuffer weight_volume_in,
    ffi::Result<ffi::AnyBuffer> data_volume_out,
    ffi::Result<ffi::AnyBuffer> weight_volume_out,
    ffi::Result<ffi::AnyBuffer> signature_rotation_keys,
    ffi::Result<ffi::AnyBuffer> signature_pixel_indices,
    ffi::Result<ffi::AnyBuffer> signature_row_flags,
    ffi::Result<ffi::AnyBuffer> signature_source_values,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_indices,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_coefficients,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_flags,
    ffi::Result<ffi::AnyBuffer> accumulator_shadow_data,
    ffi::Result<ffi::AnyBuffer> accumulator_shadow_weight,
    ffi::Result<ffi::AnyBuffer> operand_shadow_data_rows,
    ffi::Result<ffi::AnyBuffer> operand_shadow_weight_rows,
    ffi::Result<ffi::AnyBuffer> operand_shadow_pixel_indices,
    ffi::Result<ffi::AnyBuffer> operand_shadow_rot,
    ffi::Result<ffi::AnyBuffer> operand_shadow_canonical_rotation_keys,
    ffi::Result<ffi::AnyBuffer> operand_shadow_signature_row_indices)
{
    if (data_rows.element_type() != ffi::DataType::C64 ||
        data_volume_in.element_type() != ffi::DataType::C64 ||
        data_volume_out->element_type() != ffi::DataType::C64 ||
        accumulator_shadow_data->element_type() != ffi::DataType::C64 ||
        operand_shadow_data_rows->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: data rows and volumes must be complex64");
    if (weight_rows.element_type() != ffi::DataType::F32 ||
        weight_volume_in.element_type() != ffi::DataType::F32 ||
        weight_volume_out->element_type() != ffi::DataType::F32 ||
        signature_source_values->element_type() != ffi::DataType::F32 ||
        signature_neighbor_coefficients->element_type() != ffi::DataType::F32 ||
        accumulator_shadow_weight->element_type() != ffi::DataType::F32 ||
        operand_shadow_weight_rows->element_type() != ffi::DataType::F32 ||
        operand_shadow_rot->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: weights/signature floats must be float32");
    if (pixel_indices.element_type() != ffi::DataType::S32 ||
        canonical_rotation_keys.element_type() != ffi::DataType::S32 ||
        signature_row_indices.element_type() != ffi::DataType::S32 ||
        signature_rotation_keys->element_type() != ffi::DataType::S32 ||
        signature_pixel_indices->element_type() != ffi::DataType::S32 ||
        signature_row_flags->element_type() != ffi::DataType::S32 ||
        signature_neighbor_indices->element_type() != ffi::DataType::S32 ||
        signature_neighbor_flags->element_type() != ffi::DataType::S32 ||
        operand_shadow_pixel_indices->element_type() != ffi::DataType::S32 ||
        operand_shadow_canonical_rotation_keys->element_type() != ffi::DataType::S32 ||
        operand_shadow_signature_row_indices->element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: signature keys/indices/flags must be int32");
    if (rot.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: rotations must be float32");
    if (order != 1 || half_volume != 1 || half_image != 1 ||
        N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        image_h <= 0 || image_w != image_h / 2 + 1 || full_image_w != image_h ||
        upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: invalid strict x-half attributes");

    const auto data_dims = data_rows.dimensions();
    const auto weight_dims = weight_rows.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    if (data_dims.size() != 2 || data_dims[0] <= 0 || data_dims[1] <= 0 ||
        weight_dims.size() != 2 || weight_dims[0] != data_dims[0] ||
        weight_dims[1] != data_dims[1] ||
        pixel_dims.size() != 1 || pixel_dims[0] != data_dims[1] ||
        rot_dims.size() != 2 || rot_dims[0] != data_dims[0] || rot_dims[1] != 6 ||
        canonical_rotation_keys.dimensions().size() != 1 ||
        canonical_rotation_keys.dimensions()[0] != data_dims[0] ||
        signature_row_indices.dimensions().size() != 1 ||
        signature_row_indices.dimensions()[0] <= 0 ||
        data_dims[1] != image_h * image_w)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: operand shapes are inconsistent");

    const int64_t n_rows = data_dims[0];
    const int64_t n_signature_rows = signature_row_indices.dimensions()[0];
    const int64_t n_pixels = data_dims[1];
    const int64_t expected_volume_size = N0 * N1 * (N2 / 2 + 1);
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (n_rows > int_max || n_signature_rows > n_rows || n_pixels > int_max ||
        n_signature_rows > int_max / n_pixels / 8 || expected_volume_size > int_max)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: signature indexing exceeds signed int32 bounds");
    if (data_volume_in.dimensions().size() != 1 ||
        weight_volume_in.dimensions().size() != 1 ||
        data_volume_out->dimensions().size() != 1 ||
        weight_volume_out->dimensions().size() != 1 ||
        data_volume_in.dimensions()[0] != expected_volume_size ||
        weight_volume_in.dimensions()[0] != expected_volume_size ||
        data_volume_out->dimensions()[0] != expected_volume_size ||
        weight_volume_out->dimensions()[0] != expected_volume_size ||
        accumulator_shadow_data->dimensions().size() != 1 ||
        accumulator_shadow_weight->dimensions().size() != 1 ||
        accumulator_shadow_data->dimensions()[0] != expected_volume_size ||
        accumulator_shadow_weight->dimensions()[0] != expected_volume_size)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: accumulator shapes do not match");

    const auto has_shape = [](const auto& dims, int64_t a, int64_t b, int64_t c) {
        if (c > 0) return dims.size() == 3 && dims[0] == a && dims[1] == b && dims[2] == c;
        return dims.size() == 2 && dims[0] == a && dims[1] == b;
    };
    if (!has_shape(signature_rotation_keys->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_pixel_indices->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_row_flags->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_source_values->dimensions(), n_signature_rows, n_pixels, 6) ||
        !has_shape(signature_neighbor_indices->dimensions(), n_signature_rows, n_pixels, 8) ||
        !has_shape(signature_neighbor_coefficients->dimensions(), n_signature_rows, n_pixels, 8) ||
        !has_shape(signature_neighbor_flags->dimensions(), n_signature_rows, n_pixels, 8))
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: output signature shapes are inconsistent");
    if (!has_shape(operand_shadow_data_rows->dimensions(), n_rows, n_pixels, 0) ||
        !has_shape(operand_shadow_weight_rows->dimensions(), n_rows, n_pixels, 0) ||
        operand_shadow_pixel_indices->dimensions().size() != 1 ||
        operand_shadow_pixel_indices->dimensions()[0] != n_pixels ||
        !has_shape(operand_shadow_rot->dimensions(), n_rows, 6, 0) ||
        operand_shadow_canonical_rotation_keys->dimensions().size() != 1 ||
        operand_shadow_canonical_rotation_keys->dimensions()[0] != n_rows ||
        operand_shadow_signature_row_indices->dimensions().size() != 1 ||
        operand_shadow_signature_row_indices->dimensions()[0] != n_signature_rows)
        return ffi::Error::InvalidArgument(
            "RelionFusedXHalfBackprojectSignature: operand shadow shapes are inconsistent");

    cudaError_t err = launch_relion_fused_x_half_backproject_with_signature(
        stream,
        reinterpret_cast<float2*>(data_volume_out->untyped_data()),
        static_cast<float*>(weight_volume_out->untyped_data()),
        reinterpret_cast<const float2*>(data_rows.untyped_data()),
        static_cast<const float*>(weight_rows.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        static_cast<const float*>(rot.untyped_data()),
        static_cast<const int32_t*>(canonical_rotation_keys.untyped_data()),
        static_cast<const int32_t*>(signature_row_indices.untyped_data()),
        static_cast<int32_t*>(signature_rotation_keys->untyped_data()),
        static_cast<int32_t*>(signature_pixel_indices->untyped_data()),
        static_cast<int32_t*>(signature_row_flags->untyped_data()),
        static_cast<float*>(signature_source_values->untyped_data()),
        static_cast<int32_t*>(signature_neighbor_indices->untyped_data()),
        static_cast<float*>(signature_neighbor_coefficients->untyped_data()),
        static_cast<int32_t*>(signature_neighbor_flags->untyped_data()),
        reinterpret_cast<float2*>(accumulator_shadow_data->untyped_data()),
        static_cast<float*>(accumulator_shadow_weight->untyped_data()),
        reinterpret_cast<float2*>(operand_shadow_data_rows->untyped_data()),
        static_cast<float*>(operand_shadow_weight_rows->untyped_data()),
        static_cast<int32_t*>(operand_shadow_pixel_indices->untyped_data()),
        static_cast<float*>(operand_shadow_rot->untyped_data()),
        static_cast<int32_t*>(operand_shadow_canonical_rotation_keys->untyped_data()),
        static_cast<int32_t*>(operand_shadow_signature_row_indices->untyped_data()),
        n_rows, n_signature_rows, n_pixels, image_h, image_w, N0, N1, N2, upsampling, max_r2_x4);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFusedXHalfBackproject, RelionFusedXHalfBackprojectImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Arg<ffi::AnyBuffer>()           /* data_rows       */
        .Arg<ffi::AnyBuffer>()           /* weight_rows     */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices   */
        .Arg<ffi::AnyBuffer>()           /* rot             */
        .Arg<ffi::AnyBuffer>()           /* data_volume_in  */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFusedXHalfBackprojectParticleGrid,
    RelionFusedXHalfBackprojectParticleGridImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Attr<int64_t>("n_particles")
        .Attr<int64_t>("rows_per_particle")
        .Arg<ffi::AnyBuffer>()           /* data_rows */
        .Arg<ffi::AnyBuffer>()           /* weight_rows */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot */
        .Arg<ffi::AnyBuffer>()           /* data_volume_in */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFirstiterBprefFusedXHalf, RelionFirstiterBprefFusedXHalfImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Attr<float>("significant_weight")
        .Attr<float>("weight_norm")
        .Arg<ffi::AnyBuffer>()           /* image_real */
        .Arg<ffi::AnyBuffer>()           /* image_imag */
        .Arg<ffi::AnyBuffer>()           /* ctf */
        .Arg<ffi::AnyBuffer>()           /* minvsigma2 */
        .Arg<ffi::AnyBuffer>()           /* posterior */
        .Arg<ffi::AnyBuffer>()           /* translation_x */
        .Arg<ffi::AnyBuffer>()           /* translation_y */
        .Arg<ffi::AnyBuffer>()           /* native_eulers */
        .Arg<ffi::AnyBuffer>()           /* data_volume_real_in */
        .Arg<ffi::AnyBuffer>()           /* data_volume_imag_in */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_real_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* data_volume_imag_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionFusedXHalfBackprojectSignature, RelionFusedXHalfBackprojectSignatureImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("N0")
        .Attr<int64_t>("N1")
        .Attr<int64_t>("N2")
        .Attr<int64_t>("upsampling")
        .Attr<int64_t>("order")
        .Attr<int64_t>("half_volume")
        .Attr<int64_t>("half_image")
        .Attr<int64_t>("full_image_w")
        .Attr<int64_t>("max_r2_x4")
        .Arg<ffi::AnyBuffer>()           /* data_rows */
        .Arg<ffi::AnyBuffer>()           /* weight_rows */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot */
        .Arg<ffi::AnyBuffer>()           /* canonical_rotation_keys */
        .Arg<ffi::AnyBuffer>()           /* signature_row_indices */
        .Arg<ffi::AnyBuffer>()           /* data_volume_in */
        .Arg<ffi::AnyBuffer>()           /* weight_volume_in */
        .Ret<ffi::AnyBuffer>()           /* data_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* weight_volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* signature_rotation_keys */
        .Ret<ffi::AnyBuffer>()           /* signature_pixel_indices */
        .Ret<ffi::AnyBuffer>()           /* signature_row_flags */
        .Ret<ffi::AnyBuffer>()           /* signature_source_values */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_indices */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_coefficients */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_flags */
        .Ret<ffi::AnyBuffer>()           /* accumulator_shadow_data */
        .Ret<ffi::AnyBuffer>()           /* accumulator_shadow_weight */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_data_rows */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_weight_rows */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_pixel_indices */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_rot */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_canonical_rotation_keys */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_signature_row_indices */
);

struct RelionHalfTextureGeometry {
    int64_t padded_max_r;
    int64_t projector_size;
    int64_t projector_half_x;
    int64_t image_pixels;
};

bool validate_relion_half_texture_geometry(
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    RelionHalfTextureGeometry* geometry)
{
    const int64_t int_max =
        static_cast<int64_t>(std::numeric_limits<int>::max());
    /* Every launcher stores the squared padded radius in an int and forms
     * texture y/z as 2 * padded_max_r + 3.  Validate against the tighter
     * radius-square limit before evaluating either product. */
    constexpr int64_t max_padded_radius_with_int_square = 46340;
    if (current_size <= 0 || current_size > int_max ||
        padding_factor <= 0 || padding_factor > int_max ||
        projector_max_r <= 0 || projector_max_r > int_max ||
        projector_max_r >
            max_padded_radius_with_int_square / padding_factor)
        return false;

    const int64_t padded_max_r = projector_max_r * padding_factor;
    if (padded_max_r > (int_max - 3) / 2) return false;
    const int64_t image_half_x = current_size / 2 + 1;
    if (current_size > int_max / image_half_x) return false;

    geometry->padded_max_r = padded_max_r;
    geometry->projector_size = 2 * padded_max_r + 3;
    geometry->projector_half_x = padded_max_r + 2;
    geometry->image_pixels = current_size * image_half_x;
    return true;
}

ffi::Error RelionProjectorHalfTextureF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    float projector_scale,
    ffi::AnyBuffer projector_half,
    ffi::AnyBuffer rotations,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projector_half.element_type() != ffi::DataType::C64 ||
        output->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: projector/output must be C64");
    if (rotations.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: rotations must be F32");

    const auto projector_dims = projector_half.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto output_dims = output->dimensions();
    RelionHalfTextureGeometry geometry;
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (!validate_relion_half_texture_geometry(
            current_size, padding_factor, projector_max_r, &geometry) ||
        !std::isfinite(projector_scale) ||
        projector_dims.size() != 3 ||
        projector_dims[0] != geometry.projector_size ||
        projector_dims[1] != geometry.projector_size ||
        projector_dims[2] != geometry.projector_half_x ||
        rotation_dims.size() != 2 || rotation_dims[0] <= 0 ||
        rotation_dims[0] > int_max || rotation_dims[1] != 6 ||
        output_dims.size() != 2 || output_dims[0] != rotation_dims[0] ||
        output_dims[1] != geometry.image_pixels)
        return ffi::Error::InvalidArgument(
            "RelionProjectorHalfTextureF32: inconsistent operand shapes");

    cudaError_t err = launch_relion_projector_half_texture_f32(
        stream,
        reinterpret_cast<const float2*>(projector_half.untyped_data()),
        reinterpret_cast<float2*>(output->untyped_data()),
        static_cast<const float*>(rotations.untyped_data()),
        rotation_dims[0],
        current_size,
        current_size / 2 + 1,
        static_cast<int>(padding_factor),
        static_cast<int>(projector_max_r),
        projector_scale);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionProjectorHalfTextureF32,
    RelionProjectorHalfTextureF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Attr<float>("projector_scale")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error RelionProjectorPersistentHalfTextureF32Impl(
    cudaStream_t stream,
    int64_t current_size,
    int64_t padding_factor,
    int64_t projector_max_r,
    ffi::AnyBuffer owner_handle_buffer,
    ffi::AnyBuffer rotations,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (owner_handle_buffer.element_type() != ffi::DataType::U64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: owner handle must be U64");
    if (rotations.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: rotations must be F32");
    if (output->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: output must be C64");

    const auto handle_dims = owner_handle_buffer.dimensions();
    const auto rotation_dims = rotations.dimensions();
    const auto output_dims = output->dimensions();
    RelionHalfTextureGeometry geometry;
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (handle_dims.size() != 0 ||
        !validate_relion_half_texture_geometry(
            current_size, padding_factor, projector_max_r, &geometry) ||
        rotation_dims.size() != 2 || rotation_dims[0] <= 0 ||
        rotation_dims[0] > int_max || rotation_dims[1] != 6 ||
        output_dims.size() != 2 || output_dims[0] != rotation_dims[0] ||
        output_dims[1] != geometry.image_pixels)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: inconsistent operands");

    uint64_t owner_handle = 0;
    cudaError_t err = cudaMemcpyAsync(
        &owner_handle,
        owner_handle_buffer.untyped_data(),
        sizeof(owner_handle),
        cudaMemcpyDeviceToHost,
        stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));

    std::shared_ptr<PersistentRelionHalfTextureF32> owner;
    try {
        std::lock_guard<std::mutex> lock(
            persistent_relion_half_texture_f32_mutex);
        const auto found =
            persistent_relion_half_texture_f32_registry.find(owner_handle);
        if (found == persistent_relion_half_texture_f32_registry.end())
            return ffi::Error::InvalidArgument(
                "RelionProjectorPersistentHalfTextureF32: owner handle is not live");
        owner = found->second;
    } catch (...) {
        return ffi::Error::Internal(
            "RelionProjectorPersistentHalfTextureF32: registry lookup failed");
    }
    PersistentRelionHalfTextureF32CallGuard call_guard(owner);
    if (!call_guard.active)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: owner is closing");
    int current_device = -1;
    err = cudaGetDevice(&current_device);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    if (owner->tex_x != geometry.projector_half_x ||
        owner->tex_y != geometry.projector_size ||
        owner->tex_z != geometry.projector_size ||
        owner->device != current_device)
        return ffi::Error::InvalidArgument(
            "RelionProjectorPersistentHalfTextureF32: texture geometry/device mismatch");

    err = launch_relion_projector_persistent_half_texture_f32(
        stream,
        *owner,
        reinterpret_cast<float2*>(output->untyped_data()),
        static_cast<const float*>(rotations.untyped_data()),
        rotation_dims[0],
        current_size,
        current_size / 2 + 1,
        static_cast<int>(padding_factor),
        static_cast<int>(projector_max_r));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionProjectorPersistentHalfTextureF32,
    RelionProjectorPersistentHalfTextureF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("current_size")
        .Attr<int64_t>("padding_factor")
        .Attr<int64_t>("projector_max_r")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error ProjectRelionHalfRuntimeCore(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::Result<ffi::AnyBuffer> output, const int32_t* image_radius)
{
    const auto h = half.dimensions();
    const auto r = rot.dimensions();
    const auto o = output->dimensions();
    if (half.element_type() != ffi::DataType::C64 ||
        rot.element_type() != ffi::DataType::F32 ||
        radius.element_type() != ffi::DataType::S32 ||
        output->element_type() != ffi::DataType::C64 ||
        radius.dimensions().size() != 0)
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: require C64 half, F32 rotations, scalar S32 radius and C64 output");
    if (padding_factor != 1 && padding_factor != 2)
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: padding must be 1 or 2");
    if (h.size() != 3 || h[0] < 5 || h[0] != h[1] || h[0] % 2 != 1 ||
        h[2] != h[0] / 2 + 1 || (h[0] - 3) % (2 * padding_factor) != 0 ||
        h[0] > 1025 || r.size() != 2 || r[1] != 6 || r[0] <= 0 || r[0] > 65535 ||
        image_h <= 0 || image_h != image_w || image_h % 2 != 0 || image_h > 4096 ||
        o.size() != 2 || o[0] != r[0] || o[1] != image_h * (image_w / 2 + 1) ||
        r[0] * o[1] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ProjectRelionHalfRuntime: invalid capacity, rotation or image geometry");
    cudaError_t err = launch_project_texture_float<true>(
        stream, static_cast<const float*>(half.untyped_data()),
        static_cast<float*>(output->untyped_data()), static_cast<const float*>(rot.untyped_data()),
        r[0], o[1], image_h, image_w / 2 + 1,
        h[0], h[1], h[2], padding_factor, 1, image_w, -1,
        static_cast<const int32_t*>(radius.untyped_data()), image_radius);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error ProjectRelionHalfRuntimeImpl(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::Result<ffi::AnyBuffer> output)
{
    return ProjectRelionHalfRuntimeCore(
        stream, image_h, image_w, padding_factor, half, rot, radius, output, nullptr);
}

ffi::Error ProjectRelionHalfImageRadiusImpl(
    cudaStream_t stream, int64_t image_h, int64_t image_w,
    int64_t padding_factor,
    ffi::AnyBuffer half, ffi::AnyBuffer rot, ffi::AnyBuffer radius,
    ffi::AnyBuffer image_radius, ffi::Result<ffi::AnyBuffer> output)
{
    if (image_radius.element_type() != ffi::DataType::S32 ||
        image_radius.dimensions().size() != 0)
        return ffi::Error::InvalidArgument("ProjectRelionHalfImageRadius: image radius must be scalar S32");
    return ProjectRelionHalfRuntimeCore(
        stream, image_h, image_w, padding_factor, half, rot, radius, output,
        static_cast<const int32_t*>(image_radius.untyped_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfImageRadius, ProjectRelionHalfImageRadiusImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("padding_factor")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectRelionHalfRuntime, ProjectRelionHalfRuntimeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_w")
        .Attr<int64_t>("padding_factor")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

/* ================================================================== */
/*       RELION BPref x=0 + point-group symmetry finalisation          */
/* ================================================================== */

template <typename T>
struct RelionSymmetryComplex {
    T real;
    T imag;
};

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_lerp(
    T fraction,
    RelionSymmetryComplex<T> low,
    RelionSymmetryComplex<T> high)
{
    return {
        low.real + (high.real - low.real) * fraction,
        low.imag + (high.imag - low.imag) * fraction,
    };
}

template <typename T>
static __device__ __forceinline__ T relion_symmetry_lerp(
    T fraction, T low, T high)
{
    return low + (high - low) * fraction;
}

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data(
    const T* __restrict__ data,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    RelionSymmetryComplex<T> value = {data[2 * index], data[2 * index + 1]};
    if (x_index != 0)
        return value;

    /* BackProjector::enforceHermitianSymmetry sums each x=0 pair and does
     * not divide by two.  BPref grids are odd, so the partner of direct
     * index i is N-1-i.  The (z=0,y=0) center is deliberately untouched. */
    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    value.real += data[2 * partner_index];
    value.imag -= data[2 * partner_index + 1];
    return value;
}

template <typename T>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data_split(
    const T* __restrict__ data_real,
    const T* __restrict__ data_imag,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    RelionSymmetryComplex<T> value = {data_real[index], data_imag[index]};
    if (x_index != 0)
        return value;

    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    value.real += data_real[partner_index];
    value.imag -= data_imag[partner_index];
    return value;
}

template <typename T, bool SPLIT_INPUT>
static __device__ __forceinline__ RelionSymmetryComplex<T>
relion_symmetry_read_data_layout(
    const T* __restrict__ data,
    const T* __restrict__ data_imag,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    if constexpr (SPLIT_INPUT)
        return relion_symmetry_read_data_split(
            data, data_imag, z_index, y_index, x_index, full_z, full_y, x_half);
    return relion_symmetry_read_data(
        data, z_index, y_index, x_index, full_z, full_y, x_half);
}

template <typename T>
static __device__ __forceinline__ T relion_symmetry_read_weight(
    const T* __restrict__ weight,
    int z_index,
    int y_index,
    int x_index,
    int full_z,
    int full_y,
    int x_half)
{
    const int64_t index =
        (static_cast<int64_t>(z_index) * full_y + y_index) * x_half + x_index;
    T value = weight[index];
    if (x_index != 0)
        return value;
    const int partner_z = full_z - 1 - z_index;
    const int partner_y = full_y - 1 - y_index;
    if (partner_z == z_index && partner_y == y_index)
        return value;
    const int64_t partner_index =
        (static_cast<int64_t>(partner_z) * full_y + partner_y) * x_half;
    return value + weight[partner_index];
}

template <typename T, bool SPLIT_INPUT, bool RANGED_OUTPUT>
__global__ void __launch_bounds__(BLOCK_SIZE)
relion_point_group_symmetrise_bpref_kernel(
    const T* __restrict__ data,
    const T* __restrict__ data_imag,
    const T* __restrict__ weight,
    const T* __restrict__ right_operators,
    const int64_t* __restrict__ range_start,
    T* __restrict__ data_out,
    T* __restrict__ weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count,
    int64_t output_count,
    int64_t voxel_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count)
        return;
    int64_t direct_index = output_index;
    if constexpr (RANGED_OUTPUT)
        direct_index = range_start[0] + output_index;
    if (direct_index < 0 || direct_index >= voxel_count) {
        data_out[2 * output_index] = static_cast<T>(0);
        data_out[2 * output_index + 1] = static_cast<T>(0);
        weight_out[output_index] = static_cast<T>(0);
        return;
    }

    const int x_half = full_x / 2 + 1;
    const int x_index = static_cast<int>(direct_index % x_half);
    const int64_t yz_index = direct_index / x_half;
    const int y_index = static_cast<int>(yz_index % full_y);
    const int z_index = static_cast<int>(yz_index / full_y);

    const int x_logical = x_index;
    const int y_logical = y_index - full_y / 2;
    const int z_logical = z_index - full_z / 2;

    /* RELION initialises sum_data/sum_weight by copying the x=0-enforced
     * source.  Keep that identity contribution even outside rmax2. */
    RelionSymmetryComplex<T> data_sum = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
        data, data_imag, z_index, y_index, x_index, full_z, full_y, x_half);
    T weight_sum = relion_symmetry_read_weight(
        weight, z_index, y_index, x_index, full_z, full_y, x_half);

    const int64_t radius_squared =
        static_cast<int64_t>(x_logical) * x_logical +
        static_cast<int64_t>(y_logical) * y_logical +
        static_cast<int64_t>(z_logical) * z_logical;
    const int64_t support_squared =
        static_cast<int64_t>(support_radius) * support_radius;

    if (radius_squared <= support_squared) {
        const T x = static_cast<T>(x_logical);
        const T y = static_cast<T>(y_logical);
        const T z = static_cast<T>(z_logical);

        /* Operator zero is identity and was copied above.  Stream through
         * RELION's remaining SymList order without rotated-volume copies. */
        for (int operator_index = 1; operator_index < operator_count; ++operator_index) {
            const T* R = right_operators + static_cast<int64_t>(operator_index) * 9;
            T xp = x * R[0] + y * R[1] + z * R[2];
            T yp = x * R[3] + y * R[4] + z * R[5];
            T zp = x * R[6] + y * R[7] + z * R[8];

            bool conjugate_sample = false;
            if (xp < static_cast<T>(0)) {
                xp = -xp;
                yp = -yp;
                zp = -zp;
                conjugate_sample = true;
            }

            const int x0 = floor_int(xp);
            const int y0 = floor_int(yp) + full_y / 2;
            const int z0 = floor_int(zp) + full_z / 2;
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const int z1 = z0 + 1;
            const T fx = xp - static_cast<T>(x0);
            const T fy = yp - static_cast<T>(y0 - full_y / 2);
            const T fz = zp - static_cast<T>(z0 - full_z / 2);

            const RelionSymmetryComplex<T> d000 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y0, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d001 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y0, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d010 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y1, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d011 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z0, y1, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d100 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y0, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d101 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y0, x1, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d110 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y1, x0, full_z, full_y, x_half);
            const RelionSymmetryComplex<T> d111 = relion_symmetry_read_data_layout<T, SPLIT_INPUT>(
                data, data_imag, z1, y1, x1, full_z, full_y, x_half);

            const RelionSymmetryComplex<T> dx00 = relion_symmetry_lerp(fx, d000, d001);
            const RelionSymmetryComplex<T> dx01 = relion_symmetry_lerp(fx, d100, d101);
            const RelionSymmetryComplex<T> dx10 = relion_symmetry_lerp(fx, d010, d011);
            const RelionSymmetryComplex<T> dx11 = relion_symmetry_lerp(fx, d110, d111);
            const RelionSymmetryComplex<T> dxy0 = relion_symmetry_lerp(fy, dx00, dx10);
            const RelionSymmetryComplex<T> dxy1 = relion_symmetry_lerp(fy, dx01, dx11);
            RelionSymmetryComplex<T> sample = relion_symmetry_lerp(fz, dxy0, dxy1);
            if (conjugate_sample)
                sample.imag = -sample.imag;
            data_sum.real += sample.real;
            data_sum.imag += sample.imag;

            const T w000 = relion_symmetry_read_weight(
                weight, z0, y0, x0, full_z, full_y, x_half);
            const T w001 = relion_symmetry_read_weight(
                weight, z0, y0, x1, full_z, full_y, x_half);
            const T w010 = relion_symmetry_read_weight(
                weight, z0, y1, x0, full_z, full_y, x_half);
            const T w011 = relion_symmetry_read_weight(
                weight, z0, y1, x1, full_z, full_y, x_half);
            const T w100 = relion_symmetry_read_weight(
                weight, z1, y0, x0, full_z, full_y, x_half);
            const T w101 = relion_symmetry_read_weight(
                weight, z1, y0, x1, full_z, full_y, x_half);
            const T w110 = relion_symmetry_read_weight(
                weight, z1, y1, x0, full_z, full_y, x_half);
            const T w111 = relion_symmetry_read_weight(
                weight, z1, y1, x1, full_z, full_y, x_half);
            const T wx00 = relion_symmetry_lerp(fx, w000, w001);
            const T wx01 = relion_symmetry_lerp(fx, w100, w101);
            const T wx10 = relion_symmetry_lerp(fx, w010, w011);
            const T wx11 = relion_symmetry_lerp(fx, w110, w111);
            const T wxy0 = relion_symmetry_lerp(fy, wx00, wx10);
            const T wxy1 = relion_symmetry_lerp(fy, wx01, wx11);
            weight_sum += relion_symmetry_lerp(fz, wxy0, wxy1);
        }
    }

    data_out[2 * output_index] = data_sum.real;
    data_out[2 * output_index + 1] = data_sum.imag;
    weight_out[output_index] = weight_sum;
}

template <typename T>
static cudaError_t launch_relion_point_group_symmetrise_bpref(
    cudaStream_t stream,
    const T* data,
    const T* weight,
    const T* right_operators,
    T* data_out,
    T* weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count)
{
    const int64_t voxel_count =
        static_cast<int64_t>(full_z) * full_y * (full_x / 2 + 1);
    const int64_t block_count = (voxel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relion_point_group_symmetrise_bpref_kernel<T, false, false>
        <<<static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            data,
            nullptr,
            weight,
            right_operators,
            nullptr,
            data_out,
            weight_out,
            full_z,
            full_y,
            full_x,
            support_radius,
            operator_count,
            voxel_count,
            voxel_count);
    return cudaGetLastError();
}

template <typename T, bool SPLIT_INPUT = true>
static cudaError_t launch_relion_point_group_symmetrise_bpref_split_range(
    cudaStream_t stream,
    const T* data_real,
    const T* data_imag,
    const T* weight,
    const T* right_operators,
    const int64_t* range_start,
    T* data_out,
    T* weight_out,
    int full_z,
    int full_y,
    int full_x,
    int support_radius,
    int operator_count,
    int64_t output_count)
{
    const int64_t voxel_count =
        static_cast<int64_t>(full_z) * full_y * (full_x / 2 + 1);
    const int64_t block_count = (output_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relion_point_group_symmetrise_bpref_kernel<T, SPLIT_INPUT, true>
        <<<static_cast<unsigned int>(block_count), BLOCK_SIZE, 0, stream>>>(
            data_real,
            data_imag,
            weight,
            right_operators,
            range_start,
            data_out,
            weight_out,
            full_z,
            full_y,
            full_x,
            support_radius,
            operator_count,
            output_count,
            voxel_count);
    return cudaGetLastError();
}

ffi::Error RelionPointGroupSymmetriseBprefImpl(
    cudaStream_t stream,
    int64_t full_z,
    int64_t full_y,
    int64_t full_x,
    int64_t support_radius,
    ffi::AnyBuffer data,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer right_operators,
    ffi::Result<ffi::AnyBuffer> data_out,
    ffi::Result<ffi::AnyBuffer> weight_out)
{
    const auto data_dims = data.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto operator_dims = right_operators.dimensions();
    const auto data_out_dims = data_out->dimensions();
    const auto weight_out_dims = weight_out->dimensions();
    if (data_dims.size() != 1 || weight_dims.size() != 1 ||
        data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_dims[0] != weight_dims[0] || data_dims[0] != data_out_dims[0] ||
        data_dims[0] != weight_out_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: data and weight must be matching flat buffers");
    if (operator_dims.size() != 3 || operator_dims[0] < 1 ||
        operator_dims[1] != 3 || operator_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: operators must have shape (n,3,3)");
    if (full_z <= 0 || full_y <= 0 || full_x <= 0 ||
        full_z != full_y || full_z != full_x || (full_x % 2) == 0)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: full BPref dimensions must be equal positive odd values");
    if (full_z > std::numeric_limits<int>::max() ||
        operator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: dimensions exceed CUDA integer bounds");
    const int64_t expected_voxels = full_z * full_y * (full_x / 2 + 1);
    if (data_dims[0] != expected_voxels)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: flat accumulator size does not match full dimensions");
    const int64_t maximum_supported_radius = full_x / 2 - 1;
    if (support_radius < 0 || support_radius > maximum_supported_radius)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: support radius must leave one interpolation voxel");

    cudaError_t err;
    if (data.element_type() == ffi::DataType::C64 &&
        data_out->element_type() == ffi::DataType::C64 &&
        weight.element_type() == ffi::DataType::F32 &&
        weight_out->element_type() == ffi::DataType::F32 &&
        right_operators.element_type() == ffi::DataType::F32) {
        err = launch_relion_point_group_symmetrise_bpref<float>(
            stream,
            static_cast<const float*>(data.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(right_operators.untyped_data()),
            static_cast<float*>(data_out->untyped_data()),
            static_cast<float*>(weight_out->untyped_data()),
            static_cast<int>(full_z),
            static_cast<int>(full_y),
            static_cast<int>(full_x),
            static_cast<int>(support_radius),
            static_cast<int>(operator_dims[0]));
    } else if (data.element_type() == ffi::DataType::C128 &&
               data_out->element_type() == ffi::DataType::C128 &&
               weight.element_type() == ffi::DataType::F64 &&
               weight_out->element_type() == ffi::DataType::F64 &&
               right_operators.element_type() == ffi::DataType::F64) {
        err = launch_relion_point_group_symmetrise_bpref<double>(
            stream,
            static_cast<const double*>(data.untyped_data()),
            static_cast<const double*>(weight.untyped_data()),
            static_cast<const double*>(right_operators.untyped_data()),
            static_cast<double*>(data_out->untyped_data()),
            static_cast<double*>(weight_out->untyped_data()),
            static_cast<int>(full_z),
            static_cast<int>(full_y),
            static_cast<int>(full_x),
            static_cast<int>(support_radius),
            static_cast<int>(operator_dims[0]));
    } else {
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBpref: expected C64/F32/F32 or C128/F64/F64 buffers");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPointGroupSymmetriseBpref, RelionPointGroupSymmetriseBprefImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("full_z")
        .Attr<int64_t>("full_y")
        .Attr<int64_t>("full_x")
        .Attr<int64_t>("support_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionPointGroupSymmetriseBprefSplitRangeImpl(
    cudaStream_t stream,
    int64_t full_z,
    int64_t full_y,
    int64_t full_x,
    int64_t support_radius,
    ffi::AnyBuffer data_real,
    ffi::AnyBuffer data_imag,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer right_operators,
    ffi::AnyBuffer range_start,
    ffi::Result<ffi::AnyBuffer> data_out,
    ffi::Result<ffi::AnyBuffer> weight_out)
{
    const auto data_real_dims = data_real.dimensions();
    const auto data_imag_dims = data_imag.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto operator_dims = right_operators.dimensions();
    const auto range_start_dims = range_start.dimensions();
    const auto data_out_dims = data_out->dimensions();
    const auto weight_out_dims = weight_out->dimensions();
    if (data_real_dims.size() != 1 || data_imag_dims.size() != 1 ||
        weight_dims.size() != 1 ||
        data_real_dims[0] != data_imag_dims[0] ||
        data_real_dims[0] != weight_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: split inputs must be matching flat buffers");
    if (data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_out_dims[0] <= 0 || data_out_dims[0] != weight_out_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: outputs must be matching nonempty flat ranges");
    if (range_start_dims.size() != 1 || range_start_dims[0] != 1 ||
        range_start.element_type() != ffi::DataType::S64)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: range_start must be one int64 value");
    if (operator_dims.size() != 3 || operator_dims[0] < 1 ||
        operator_dims[1] != 3 || operator_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: operators must have shape (n,3,3)");
    if (full_z <= 0 || full_y <= 0 || full_x <= 0 ||
        full_z != full_y || full_z != full_x || (full_x % 2) == 0)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: full BPref dimensions must be equal positive odd values");
    if (full_z > std::numeric_limits<int>::max() ||
        operator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: dimensions exceed CUDA integer bounds");
    const int64_t expected_voxels = full_z * full_y * (full_x / 2 + 1);
    if (data_real_dims[0] != expected_voxels || data_out_dims[0] > expected_voxels)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: input or range size does not match full dimensions");
    const int64_t maximum_supported_radius = full_x / 2 - 1;
    if (support_radius < 0 || support_radius > maximum_supported_radius)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: support radius must leave one interpolation voxel");
    if (data_real.element_type() != ffi::DataType::F32 ||
        data_imag.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        right_operators.element_type() != ffi::DataType::F32 ||
        data_out->element_type() != ffi::DataType::C64 ||
        weight_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefSplitRange: expected F32/F32/F32/F32/S64 inputs and C64/F32 outputs");

    cudaError_t err = launch_relion_point_group_symmetrise_bpref_split_range<float>(
        stream,
        static_cast<const float*>(data_real.untyped_data()),
        static_cast<const float*>(data_imag.untyped_data()),
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(right_operators.untyped_data()),
        static_cast<const int64_t*>(range_start.untyped_data()),
        static_cast<float*>(data_out->untyped_data()),
        static_cast<float*>(weight_out->untyped_data()),
        static_cast<int>(full_z),
        static_cast<int>(full_y),
        static_cast<int>(full_x),
        static_cast<int>(support_radius),
        static_cast<int>(operator_dims[0]),
        static_cast<int64_t>(data_out_dims[0]));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPointGroupSymmetriseBprefSplitRange,
    RelionPointGroupSymmetriseBprefSplitRangeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("full_z")
        .Attr<int64_t>("full_y")
        .Attr<int64_t>("full_x")
        .Attr<int64_t>("support_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionPointGroupSymmetriseBprefComplexRangeImpl(
    cudaStream_t stream,
    int64_t full_z,
    int64_t full_y,
    int64_t full_x,
    int64_t support_radius,
    ffi::AnyBuffer data_real,
    ffi::AnyBuffer weight,
    ffi::AnyBuffer right_operators,
    ffi::AnyBuffer range_start,
    ffi::Result<ffi::AnyBuffer> data_out,
    ffi::Result<ffi::AnyBuffer> weight_out)
{
    const auto data_real_dims = data_real.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto operator_dims = right_operators.dimensions();
    const auto range_start_dims = range_start.dimensions();
    const auto data_out_dims = data_out->dimensions();
    const auto weight_out_dims = weight_out->dimensions();
    if (data_real_dims.size() != 1 ||
        weight_dims.size() != 1 ||
        data_real_dims[0] != weight_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: complex/weight inputs must be matching flat buffers");
    if (data_out_dims.size() != 1 || weight_out_dims.size() != 1 ||
        data_out_dims[0] <= 0 || data_out_dims[0] != weight_out_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: outputs must be matching nonempty flat ranges");
    if (range_start_dims.size() != 1 || range_start_dims[0] != 1 ||
        range_start.element_type() != ffi::DataType::S64)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: range_start must be one int64 value");
    if (operator_dims.size() != 3 || operator_dims[0] < 1 ||
        operator_dims[1] != 3 || operator_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: operators must have shape (n,3,3)");
    if (full_z <= 0 || full_y <= 0 || full_x <= 0 ||
        full_z != full_y || full_z != full_x || (full_x % 2) == 0)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: full BPref dimensions must be equal positive odd values");
    if (full_z > std::numeric_limits<int>::max() ||
        operator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: dimensions exceed CUDA integer bounds");
    const int64_t expected_voxels = full_z * full_y * (full_x / 2 + 1);
    if (data_real_dims[0] != expected_voxels || data_out_dims[0] > expected_voxels)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: input or range size does not match full dimensions");
    const int64_t maximum_supported_radius = full_x / 2 - 1;
    if (support_radius < 0 || support_radius > maximum_supported_radius)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: support radius must leave one interpolation voxel");
    if (data_real.element_type() != ffi::DataType::C64 ||
        weight.element_type() != ffi::DataType::F32 ||
        right_operators.element_type() != ffi::DataType::F32 ||
        data_out->element_type() != ffi::DataType::C64 ||
        weight_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionPointGroupSymmetriseBprefComplexRange: expected C64/F32/F32/S64 inputs and C64/F32 outputs");

    cudaError_t err = launch_relion_point_group_symmetrise_bpref_split_range<float, false>(
        stream,
        static_cast<const float*>(data_real.untyped_data()),
        nullptr,
        static_cast<const float*>(weight.untyped_data()),
        static_cast<const float*>(right_operators.untyped_data()),
        static_cast<const int64_t*>(range_start.untyped_data()),
        static_cast<float*>(data_out->untyped_data()),
        static_cast<float*>(weight_out->untyped_data()),
        static_cast<int>(full_z),
        static_cast<int>(full_y),
        static_cast<int>(full_x),
        static_cast<int>(support_radius),
        static_cast<int>(operator_dims[0]),
        static_cast<int64_t>(data_out_dims[0]));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPointGroupSymmetriseBprefComplexRange,
    RelionPointGroupSymmetriseBprefComplexRangeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("full_z")
        .Attr<int64_t>("full_y")
        .Attr<int64_t>("full_x")
        .Attr<int64_t>("support_radius")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgRotationAtomicTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer accumulator_in,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicTripletAddF32: need F32 buffers");
    const auto dims = terms.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 4 || accumulator_dims.size() != 3 || output_dims.size() != 3 ||
        dims[3] != 3 || accumulator_dims[2] != 3 ||
        output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        accumulator_dims[0] != dims[0] || accumulator_dims[1] != dims[2])
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicTripletAddF32: expected terms[B,R,P,3] "
            "and accumulator[B,P,3]");
    if (dims[0] <= 0 || dims[0] > 65535 || dims[1] <= 0 ||
        dims[1] > std::numeric_limits<int>::max() || dims[2] <= 0 ||
        dims[2] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicTripletAddF32: dimensions exceed CUDA grid bounds");
    cudaError_t err = launch_relion_wavg_rotation_atomic_triplet_add_f32(
        stream,
        static_cast<const float*>(terms.untyped_data()),
        static_cast<float*>(accumulator_out->untyped_data()),
        dims[0],
        dims[1],
        dims[2]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicTripletAddF32,
    RelionWavgRotationAtomicTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgRotationAtomicRuntimeTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer accumulator_in,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeTripletAddF32: invalid buffers");
    const auto dims = terms.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 4 || accumulator_dims.size() != 3 ||
        output_dims.size() != 3 || dims[3] != 3 ||
        accumulator_dims[2] != 3 || output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        accumulator_dims[0] != dims[0] || accumulator_dims[1] != dims[2] ||
        dims[0] <= 0 || dims[0] > 65535 || dims[1] <= 0 ||
        dims[1] > std::numeric_limits<int>::max() || dims[2] <= 0 ||
        dims[2] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeTripletAddF32: inconsistent topology");
    cudaError_t err = launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(
        stream,
        static_cast<const float*>(terms.untyped_data()),
        static_cast<float*>(accumulator_out->untyped_data()),
        dims[0],
        dims[1],
        dims[2],
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicRuntimeTripletAddF32,
    RelionWavgRotationAtomicRuntimeTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgSequentialTripletF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer projections,
    ffi::AnyBuffer raw_ctf,
    ffi::AnyBuffer scale,
    ffi::AnyBuffer shifted_images,
    ffi::AnyBuffer posterior,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projections.element_type() != ffi::DataType::C64 ||
        raw_ctf.element_type() != ffi::DataType::F32 ||
        scale.element_type() != ffi::DataType::F32 ||
        shifted_images.element_type() != ffi::DataType::C64 ||
        posterior.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialTripletF32: expected C64 projections/shifted "
            "images and F32 CTF/scale/posterior/output");

    const auto projection_dims = projections.dimensions();
    const auto ctf_dims = raw_ctf.dimensions();
    const auto scale_dims = scale.dimensions();
    const auto shifted_dims = shifted_images.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto output_dims = output->dimensions();
    if (projection_dims.size() != 3 || ctf_dims.size() != 2 ||
        scale_dims.size() != 1 || shifted_dims.size() != 3 ||
        posterior_dims.size() != 3 || output_dims.size() != 4 ||
        output_dims[3] != 3)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialTripletF32: expected projections[B,R,P], "
            "CTF[B,P], scale[B], shifted[B,T,P], posterior[B,R,T], and "
            "output[B,R,P,3]");

    const int64_t batch_size = projection_dims[0];
    const int64_t rotation_count = projection_dims[1];
    const int64_t pixel_count = projection_dims[2];
    const int64_t translation_count = shifted_dims[1];
    if (batch_size <= 0 || rotation_count <= 0 || translation_count <= 0 ||
        pixel_count <= 0 ||
        ctf_dims[0] != batch_size || ctf_dims[1] != pixel_count ||
        scale_dims[0] != batch_size ||
        shifted_dims[0] != batch_size || shifted_dims[2] != pixel_count ||
        posterior_dims[0] != batch_size ||
        posterior_dims[1] != rotation_count ||
        posterior_dims[2] != translation_count ||
        output_dims[0] != batch_size ||
        output_dims[1] != rotation_count ||
        output_dims[2] != pixel_count)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialTripletF32: input/output dimensions do not match");

    const int64_t output_count = batch_size * rotation_count * pixel_count;
    if (output_count > static_cast<int64_t>(std::numeric_limits<int>::max()) * 256)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialTripletF32: output exceeds CUDA grid bounds");
    cudaError_t err = launch_relion_wavg_sequential_triplet_f32(
        stream,
        reinterpret_cast<const float2*>(projections.untyped_data()),
        static_cast<const float*>(raw_ctf.untyped_data()),
        static_cast<const float*>(scale.untyped_data()),
        reinterpret_cast<const float2*>(shifted_images.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        batch_size,
        rotation_count,
        translation_count,
        pixel_count);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgSequentialTripletF32,
    RelionWavgSequentialTripletF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgSequentialRuntimeTripletF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer projections,
    ffi::AnyBuffer raw_ctf,
    ffi::AnyBuffer scale,
    ffi::AnyBuffer shifted_images,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projections.element_type() != ffi::DataType::C64 ||
        raw_ctf.element_type() != ffi::DataType::F32 ||
        scale.element_type() != ffi::DataType::F32 ||
        shifted_images.element_type() != ffi::DataType::C64 ||
        posterior.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeTripletF32: invalid buffers");
    const auto projection_dims = projections.dimensions();
    const auto ctf_dims = raw_ctf.dimensions();
    const auto scale_dims = scale.dimensions();
    const auto shifted_dims = shifted_images.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto output_dims = output->dimensions();
    if (projection_dims.size() != 3 || ctf_dims.size() != 2 ||
        scale_dims.size() != 1 || shifted_dims.size() != 3 ||
        posterior_dims.size() != 3 || output_dims.size() != 4 ||
        output_dims[3] != 3)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeTripletF32: invalid ranks");
    const int64_t batch_size = projection_dims[0];
    const int64_t rotation_count = projection_dims[1];
    const int64_t pixel_capacity = projection_dims[2];
    const int64_t translation_count = shifted_dims[1];
    if (batch_size <= 0 || batch_size > 65535 || rotation_count <= 0 ||
        rotation_count > std::numeric_limits<int>::max() ||
        translation_count <= 0 || pixel_capacity <= 0 ||
        pixel_capacity > std::numeric_limits<int>::max() ||
        ctf_dims[0] != batch_size || ctf_dims[1] != pixel_capacity ||
        scale_dims[0] != batch_size || shifted_dims[0] != batch_size ||
        shifted_dims[2] != pixel_capacity || posterior_dims[0] != batch_size ||
        posterior_dims[1] != rotation_count ||
        posterior_dims[2] != translation_count || output_dims[0] != batch_size ||
        output_dims[1] != rotation_count || output_dims[2] != pixel_capacity)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeTripletF32: inconsistent topology");
    cudaError_t err = launch_relion_wavg_sequential_runtime_triplet_f32(
        stream,
        reinterpret_cast<const float2*>(projections.untyped_data()),
        static_cast<const float*>(raw_ctf.untyped_data()),
        static_cast<const float*>(scale.untyped_data()),
        reinterpret_cast<const float2*>(shifted_images.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        batch_size,
        rotation_count,
        translation_count,
        pixel_capacity,
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgSequentialRuntimeTripletF32,
    RelionWavgSequentialRuntimeTripletF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

// Flat-row Wavg: projections[Q,P] and posterior[Q,T] are packed candidate rows,
// row_image_ids[Q] carries each row's image, and negative ids mark padding.
ffi::Error RelionWavgSequentialRuntimeFlatRowsTripletF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer projections,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer raw_ctf,
    ffi::AnyBuffer scale,
    ffi::AnyBuffer shifted_images,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (projections.element_type() != ffi::DataType::C64 ||
        raw_ctf.element_type() != ffi::DataType::F32 ||
        scale.element_type() != ffi::DataType::F32 ||
        shifted_images.element_type() != ffi::DataType::C64 ||
        posterior.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: invalid buffers");
    const auto projection_dims = projections.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto ctf_dims = raw_ctf.dimensions();
    const auto scale_dims = scale.dimensions();
    const auto shifted_dims = shifted_images.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto output_dims = output->dimensions();
    if (projection_dims.size() != 2 || row_dims.size() != 1 ||
        ctf_dims.size() != 2 || scale_dims.size() != 1 ||
        shifted_dims.size() != 3 || posterior_dims.size() != 2 ||
        output_dims.size() != 3 || output_dims[2] != 3)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: invalid ranks");
    const int64_t row_count = projection_dims[0];
    const int64_t pixel_capacity = projection_dims[1];
    const int64_t batch_size = ctf_dims[0];
    const int64_t translation_count = shifted_dims[1];
    if (row_count <= 0 || row_count > std::numeric_limits<int>::max() ||
        batch_size <= 0 || translation_count <= 0 || pixel_capacity <= 0 ||
        pixel_capacity > std::numeric_limits<int>::max() ||
        row_dims[0] != row_count || ctf_dims[1] != pixel_capacity ||
        scale_dims[0] != batch_size || shifted_dims[0] != batch_size ||
        shifted_dims[2] != pixel_capacity || posterior_dims[0] != row_count ||
        posterior_dims[1] != translation_count ||
        output_dims[0] != row_count || output_dims[1] != pixel_capacity)
        return ffi::Error::InvalidArgument(
            "RelionWavgSequentialRuntimeFlatRowsTripletF32: inconsistent topology");
    cudaError_t err = launch_relion_wavg_sequential_runtime_flat_rows_triplet_f32(
        stream,
        reinterpret_cast<const float2*>(projections.untyped_data()),
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const float*>(raw_ctf.untyped_data()),
        static_cast<const float*>(scale.untyped_data()),
        reinterpret_cast<const float2*>(shifted_images.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        batch_size,
        row_count,
        translation_count,
        pixel_capacity,
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgSequentialRuntimeFlatRowsTripletF32,
    RelionWavgSequentialRuntimeFlatRowsTripletF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer accumulator_in,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32: invalid buffers");
    const auto dims = terms.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 3 || row_dims.size() != 1 ||
        accumulator_dims.size() != 3 || output_dims.size() != 3 ||
        dims[2] != 3 || accumulator_dims[2] != 3 ||
        output_dims[0] != accumulator_dims[0] ||
        output_dims[1] != accumulator_dims[1] ||
        output_dims[2] != accumulator_dims[2] ||
        row_dims[0] != dims[0] || accumulator_dims[1] != dims[1] ||
        dims[0] <= 0 || dims[0] > std::numeric_limits<int>::max() ||
        dims[1] <= 0 || dims[1] > std::numeric_limits<int>::max() ||
        accumulator_dims[0] <= 0 ||
        accumulator_dims[0] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32: inconsistent topology");
    cudaError_t err =
        launch_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
            stream,
            static_cast<const float*>(terms.untyped_data()),
            static_cast<const int32_t*>(row_image_ids.untyped_data()),
            static_cast<float*>(accumulator_out->untyped_data()),
            accumulator_dims[0],
            dims[0],
            dims[1],
            static_cast<const int32_t*>(logical_pixel_count.untyped_data()));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32,
    RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

// Native Wavg prefix: preserve the exact translation arithmetic above and the
// separate native-order rotation-atomic launch. No scalar readback or sync.
__global__ void validate_wavg_prefix_maps_kernel(
    const int32_t* exact_positions, const int32_t* recon_indices,
    const int32_t* logical_exact, const int32_t* logical_rectangle,
    int64_t exact, int64_t rectangle, int64_t full_pixels,
    int32_t* owners, int32_t* invalid)
{
    const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t ne = *logical_exact, nr = *logical_rectangle;
    if (ne < 0 || ne > exact || nr < 0 || nr > rectangle) {
        if (pixel == 0) atomicExch(invalid, 1);
        return;
    }
    if (pixel >= exact) return;
    const int32_t position = exact_positions[pixel];
    const int32_t index = recon_indices[pixel];
    if (position < 0 || position >= rectangle || index < 0 || index >= full_pixels ||
        (pixel < ne ? position >= nr : position < nr)) {
        atomicExch(invalid, 1);
        return;
    }
    if (atomicCAS(owners + position, -1, static_cast<int32_t>(pixel)) != -1)
        atomicExch(invalid, 1);
}

__global__ void initialize_wavg_prefix_rectangle_kernel(
    const float* image_power, float* rectangle_terms, int64_t count)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    rectangle_terms[3 * i] = 0.0f;
    rectangle_terms[3 * i + 1] = 0.0f;
    rectangle_terms[3 * i + 2] = image_power[i];
}

__global__ void invalidate_wavg_prefix_output_kernel(
    const int32_t* invalid, float* output, int64_t count)
{
    if (!*invalid) return;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = nanf("");
}

ffi::Error RelionWavgNativePrefixCommon(
    cudaStream_t stream, ffi::AnyBuffer raw_rectangle, ffi::AnyBuffer image_power,
    ffi::AnyBuffer projections, ffi::AnyBuffer full_ctf, ffi::AnyBuffer scale,
    ffi::AnyBuffer posterior, ffi::AnyBuffer exact_positions,
    ffi::AnyBuffer recon_indices, ffi::AnyBuffer logical_exact,
    ffi::AnyBuffer logical_rectangle, ffi::AnyBuffer output,
    float* debug_terms)
{
    if (raw_rectangle.element_type() != ffi::DataType::C64 ||
        image_power.element_type() != ffi::DataType::F32 ||
        projections.element_type() != ffi::DataType::C64 ||
        full_ctf.element_type() != ffi::DataType::F64 ||
        scale.element_type() != ffi::DataType::F32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        exact_positions.element_type() != ffi::DataType::S32 ||
        recon_indices.element_type() != ffi::DataType::S32 ||
        logical_exact.element_type() != ffi::DataType::S32 ||
        logical_rectangle.element_type() != ffi::DataType::S32 ||
        output.element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("RelionWavgNativePrefix: operand dtype mismatch");
    const auto p = projections.dimensions(), raw = raw_rectangle.dimensions();
    const auto power = image_power.dimensions(), ctf = full_ctf.dimensions();
    const auto sc = scale.dimensions(), post = posterior.dimensions();
    const auto ep = exact_positions.dimensions(), ri = recon_indices.dimensions();
    const auto out = output.dimensions();
    if (p.size() != 3 || raw.size() != 3 || power.size() != 3 || ctf.size() != 2 ||
        sc.size() != 1 || post.size() != 3 || ep.size() != 1 || ri.size() != 1 ||
        logical_exact.dimensions().size() != 0 ||
        logical_rectangle.dimensions().size() != 0 || out.size() != 3)
        return ffi::Error::InvalidArgument("RelionWavgNativePrefix: operand rank mismatch");
    const int64_t B = p[0], R = p[1], Pe = p[2], T = raw[1], Pr = raw[2], F = ctf[1];
    if (B <= 0 || B > 65535 || R <= 0 || R > std::numeric_limits<int>::max() ||
        Pe <= 0 || Pe > std::numeric_limits<int>::max() || T <= 0 ||
        T > std::numeric_limits<int>::max() || Pr <= 0 ||
        Pr > std::numeric_limits<int>::max() || F <= 0 || F > std::numeric_limits<int>::max() ||
        raw[0] != B || power[0] != B || power[1] != R || power[2] != Pr ||
        ctf[0] != B || sc[0] != B || post[0] != B || post[1] != R || post[2] != T ||
        ep[0] != Pe || ri[0] != Pe || out[0] != B || out[1] != Pr || out[2] != 3)
        return ffi::Error::InvalidArgument("RelionWavgNativePrefix: inconsistent geometry");
    const int64_t max_threads = static_cast<int64_t>(std::numeric_limits<int>::max()) * 256;
    if (B * R > max_threads / Pr || B > max_threads / (Pr * 3))
        return ffi::Error::InvalidArgument("RelionWavgNativePrefix: launch extent overflow");
    const int64_t terms_count = B * R * Pr;
    const size_t terms_bytes = static_cast<size_t>(terms_count) * 3 * sizeof(float);
    const size_t map_bytes = (static_cast<size_t>(Pr) + 1) * sizeof(int32_t);
    void* allocation = nullptr;
    cudaError_t err = cudaMallocAsync(&allocation, (debug_terms ? 0 : terms_bytes) + map_bytes, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("RelionWavgNativePrefix allocation: ") + cudaGetErrorString(err));
    float* terms = debug_terms ? debug_terms : static_cast<float*>(allocation);
    int32_t* owners = reinterpret_cast<int32_t*>(
        static_cast<char*>(allocation) + (debug_terms ? 0 : terms_bytes));
    int32_t* invalid = owners + Pr;
    auto finish = [&](cudaError_t result) {
        const cudaError_t free_error = cudaFreeAsync(allocation, stream);
        if (result == cudaSuccess) result = free_error;
        return result == cudaSuccess ? ffi::Error::Success()
            : ffi::Error::Internal(std::string("RelionWavgNativePrefix CUDA: ") + cudaGetErrorString(result));
    };
    err = cudaMemsetAsync(owners, 0xff, static_cast<size_t>(Pr) * sizeof(int32_t), stream);
    if (err != cudaSuccess) return finish(err);
    err = cudaMemsetAsync(invalid, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return finish(err);
    validate_wavg_prefix_maps_kernel<<<static_cast<unsigned>((Pe + 255) / 256), 256, 0, stream>>>(
        static_cast<const int32_t*>(exact_positions.untyped_data()),
        static_cast<const int32_t*>(recon_indices.untyped_data()),
        static_cast<const int32_t*>(logical_exact.untyped_data()),
        static_cast<const int32_t*>(logical_rectangle.untyped_data()), Pe, Pr, F, owners, invalid);
    err = cudaGetLastError();
    if (err != cudaSuccess) return finish(err);
    initialize_wavg_prefix_rectangle_kernel<<<static_cast<unsigned>((terms_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(image_power.untyped_data()), terms, terms_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return finish(err);
    dim3 grid(static_cast<unsigned>(R), static_cast<unsigned>(B));
    relion_wavg_sequential_runtime_triplet_f32_kernel<true><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const float2*>(projections.untyped_data()), nullptr,
        static_cast<const float*>(scale.untyped_data()),
        reinterpret_cast<const float2*>(raw_rectangle.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()), terms, B, R, T, Pe,
        static_cast<const int32_t*>(logical_exact.untyped_data()),
        static_cast<const double*>(full_ctf.untyped_data()),
        static_cast<const int32_t*>(exact_positions.untyped_data()),
        static_cast<const int32_t*>(recon_indices.untyped_data()), Pr, F, invalid);
    err = cudaGetLastError();
    if (err != cudaSuccess) return finish(err);
    auto* result = static_cast<float*>(output.untyped_data());
    err = cudaMemsetAsync(result, 0, static_cast<size_t>(B * Pr * 3) * sizeof(float), stream);
    if (err != cudaSuccess) return finish(err);
    err = launch_relion_wavg_rotation_atomic_runtime_triplet_add_f32(
        stream, terms, result, B, R, Pr,
        static_cast<const int32_t*>(logical_rectangle.untyped_data()));
    if (err != cudaSuccess) return finish(err);
    invalidate_wavg_prefix_output_kernel<<<static_cast<unsigned>((B * Pr * 3 + 255) / 256), 256, 0, stream>>>(
        invalid, result, B * Pr * 3);
    return finish(cudaGetLastError());
}

ffi::Error RelionWavgNativePrefixF32Impl(
    cudaStream_t stream, ffi::AnyBuffer raw, ffi::AnyBuffer power, ffi::AnyBuffer proj,
    ffi::AnyBuffer ctf, ffi::AnyBuffer scale, ffi::AnyBuffer posterior,
    ffi::AnyBuffer positions, ffi::AnyBuffer indices, ffi::AnyBuffer ne, ffi::AnyBuffer nr,
    ffi::Result<ffi::AnyBuffer> output)
{
    return RelionWavgNativePrefixCommon(stream, raw, power, proj, ctf, scale,
        posterior, positions, indices, ne, nr, *output, nullptr);
}

ffi::Error RelionWavgNativePrefixDebugF32Impl(
    cudaStream_t stream, ffi::AnyBuffer raw, ffi::AnyBuffer power, ffi::AnyBuffer proj,
    ffi::AnyBuffer ctf, ffi::AnyBuffer scale, ffi::AnyBuffer posterior,
    ffi::AnyBuffer positions, ffi::AnyBuffer indices, ffi::AnyBuffer ne, ffi::AnyBuffer nr,
    ffi::Result<ffi::AnyBuffer> output, ffi::Result<ffi::AnyBuffer> debug)
{
    const auto p = proj.dimensions(), r = raw.dimensions(), d = debug->dimensions();
    if (debug->element_type() != ffi::DataType::F32 || p.size() != 3 || r.size() != 3 ||
        d.size() != 4 || d[0] != p[0] || d[1] != p[1] || d[2] != r[2] || d[3] != 3)
        return ffi::Error::InvalidArgument("RelionWavgNativePrefixDebug: output geometry differs");
    return RelionWavgNativePrefixCommon(stream, raw, power, proj, ctf, scale,
        posterior, positions, indices, ne, nr, *output,
        static_cast<float*>(debug->untyped_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RelionWavgNativePrefixF32, RelionWavgNativePrefixF32Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(RelionWavgNativePrefixDebugF32, RelionWavgNativePrefixDebugF32Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());


static __device__ __forceinline__ float dual_weighted_fma(
    float weight, float value, float accumulator)
{
    return __fmaf_rn(weight, value, accumulator);
}

static __device__ __forceinline__ double dual_weighted_fma(
    float weight, double value, double accumulator)
{
    return __fma_rn(static_cast<double>(weight), value, accumulator);
}

template <typename T>
__global__ void dual_weighted_sums_f32_kernel(
    const float* probabilities,
    const vec2_t<T>* values,
    vec2_t<T>* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_count = batch_size * rotation_count * pixel_count;
    if (output_index >= output_count) return;
    const int64_t pixel = output_index % pixel_count;
    const int64_t batch_rotation = output_index / pixel_count;
    const int64_t rotation = batch_rotation % rotation_count;
    const int64_t batch = batch_rotation / rotation_count;
    const int64_t probability_base =
        (batch * rotation_count + rotation) * translation_count;
    const int64_t value_base = batch * translation_count * pixel_count + pixel;
    T sum_real = static_cast<T>(0);
    T sum_imag = static_cast<T>(0);
    for (int64_t translation = 0; translation < translation_count; ++translation) {
        const float weight = probabilities[probability_base + translation];
        if (weight == 0.0f) continue;
        const vec2_t<T> value = values[value_base + translation * pixel_count];
        sum_real = dual_weighted_fma(weight, value.x, sum_real);
        sum_imag = dual_weighted_fma(weight, value.y, sum_imag);
    }
    output[output_index] = make_v2(sum_real, sum_imag);
}

template <typename T>
cudaError_t launch_dual_weighted_sums_f32(
    cudaStream_t stream,
    const float* probabilities,
    const vec2_t<T>* first_values,
    const vec2_t<T>* second_values,
    vec2_t<T>* first_output,
    vec2_t<T>* second_output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t first_pixel_count,
    int64_t second_pixel_count)
{
    constexpr int threads = 256;
    const int64_t first_count = batch_size * rotation_count * first_pixel_count;
    const int64_t second_count = batch_size * rotation_count * second_pixel_count;
    const int first_blocks = static_cast<int>((first_count + threads - 1) / threads);
    const int second_blocks = static_cast<int>((second_count + threads - 1) / threads);
    dual_weighted_sums_f32_kernel<T><<<first_blocks, threads, 0, stream>>>(
        probabilities, first_values, first_output,
        batch_size, rotation_count, translation_count, first_pixel_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    dual_weighted_sums_f32_kernel<T><<<second_blocks, threads, 0, stream>>>(
        probabilities, second_values, second_output,
        batch_size, rotation_count, translation_count, second_pixel_count);
    return cudaGetLastError();
}

ffi::Error DualWeightedSumsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer probabilities,
    ffi::AnyBuffer first_values,
    ffi::AnyBuffer second_values,
    ffi::Result<ffi::AnyBuffer> first_output,
    ffi::Result<ffi::AnyBuffer> second_output)
{
    const auto value_type = first_values.element_type();
    if (probabilities.element_type() != ffi::DataType::F32 ||
        (value_type != ffi::DataType::C64 && value_type != ffi::DataType::C128) ||
        second_values.element_type() != value_type ||
        first_output->element_type() != value_type ||
        second_output->element_type() != value_type)
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsF32: expected F32 probabilities and matching C64/C128 values/outputs");
    const auto probability_dims = probabilities.dimensions();
    const auto first_dims = first_values.dimensions();
    const auto second_dims = second_values.dimensions();
    const auto first_output_dims = first_output->dimensions();
    const auto second_output_dims = second_output->dimensions();
    if (probability_dims.size() != 3 || first_dims.size() != 3 || second_dims.size() != 3 ||
        first_output_dims.size() != 3 || second_output_dims.size() != 3 ||
        probability_dims[0] <= 0 || probability_dims[1] <= 0 || probability_dims[2] <= 0 ||
        first_dims[0] != probability_dims[0] || first_dims[1] != probability_dims[2] ||
        second_dims[0] != probability_dims[0] || second_dims[1] != probability_dims[2] ||
        first_dims[2] <= 0 || second_dims[2] <= 0 ||
        first_output_dims[0] != probability_dims[0] ||
        first_output_dims[1] != probability_dims[1] ||
        first_output_dims[2] != first_dims[2] ||
        second_output_dims[0] != probability_dims[0] ||
        second_output_dims[1] != probability_dims[1] ||
        second_output_dims[2] != second_dims[2])
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsF32: inconsistent [B,R,T], [B,T,P], and [B,R,P] shapes");
    const int64_t largest_count = probability_dims[0] * probability_dims[1] *
        std::max(first_dims[2], second_dims[2]);
    if (largest_count > static_cast<int64_t>(std::numeric_limits<int>::max()) * 256)
        return ffi::Error::InvalidArgument("DualWeightedSumsF32: output exceeds CUDA grid bounds");

    cudaError_t err;
    if (value_type == ffi::DataType::C64) {
        err = launch_dual_weighted_sums_f32<float>(
            stream, static_cast<const float*>(probabilities.untyped_data()),
            reinterpret_cast<const float2*>(first_values.untyped_data()),
            reinterpret_cast<const float2*>(second_values.untyped_data()),
            reinterpret_cast<float2*>(first_output->untyped_data()),
            reinterpret_cast<float2*>(second_output->untyped_data()),
            probability_dims[0], probability_dims[1], probability_dims[2],
            first_dims[2], second_dims[2]);
    } else {
        err = launch_dual_weighted_sums_f32<double>(
            stream, static_cast<const float*>(probabilities.untyped_data()),
            reinterpret_cast<const double2*>(first_values.untyped_data()),
            reinterpret_cast<const double2*>(second_values.untyped_data()),
            reinterpret_cast<double2*>(first_output->untyped_data()),
            reinterpret_cast<double2*>(second_output->untyped_data()),
            probability_dims[0], probability_dims[1], probability_dims[2],
            first_dims[2], second_dims[2]);
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DualWeightedSumsF32, DualWeightedSumsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

/* Pair-sparse twin of dual_weighted_sums_f32_kernel.  The dense kernel loops
 * over every translation of every (batch, rotation) row and skips zero
 * probabilities; for compact K-class pairs the dense (B, R, T) table is ~1 %
 * populated and 73 % of the rows are padding, so it spent its time reading
 * zeros.  Here each row owns the contiguous pair range
 * [row_offsets[b, r], row_offsets[b, r + 1]) of pairs sorted by
 * (rotation row, translation); iterating that range in order performs the
 * same fma sequence as the dense loop over ascending translations, so the
 * outputs are bit-identical when every (row, translation) pair is unique. */
template <typename T>
__global__ void dual_weighted_sums_pairs_f32_kernel(
    const float* pair_probabilities,
    const int32_t* pair_translations,
    const int32_t* row_offsets,
    const vec2_t<T>* values,
    vec2_t<T>* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pair_count,
    int64_t pixel_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_count = batch_size * rotation_count * pixel_count;
    if (output_index >= output_count) return;
    const int64_t pixel = output_index % pixel_count;
    const int64_t batch_rotation = output_index / pixel_count;
    const int64_t rotation = batch_rotation % rotation_count;
    const int64_t batch = batch_rotation / rotation_count;
    const int64_t offset_base = batch * (rotation_count + 1) + rotation;
    int64_t start = row_offsets[offset_base];
    int64_t end = row_offsets[offset_base + 1];
    if (start < 0) start = 0;
    if (end > pair_count) end = pair_count;
    const int64_t pair_base = batch * pair_count;
    const int64_t value_base = batch * translation_count * pixel_count + pixel;
    T sum_real = static_cast<T>(0);
    T sum_imag = static_cast<T>(0);
    for (int64_t pair = start; pair < end; ++pair) {
        const float weight = pair_probabilities[pair_base + pair];
        if (weight == 0.0f) continue;
        const int64_t translation = pair_translations[pair_base + pair];
        if (translation < 0 || translation >= translation_count) continue;
        const vec2_t<T> value = values[value_base + translation * pixel_count];
        sum_real = dual_weighted_fma(weight, value.x, sum_real);
        sum_imag = dual_weighted_fma(weight, value.y, sum_imag);
    }
    output[output_index] = make_v2(sum_real, sum_imag);
}

template <typename T>
cudaError_t launch_dual_weighted_sums_pairs_f32(
    cudaStream_t stream,
    const float* pair_probabilities,
    const int32_t* pair_translations,
    const int32_t* row_offsets,
    const vec2_t<T>* first_values,
    const vec2_t<T>* second_values,
    vec2_t<T>* first_output,
    vec2_t<T>* second_output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pair_count,
    int64_t first_pixel_count,
    int64_t second_pixel_count)
{
    constexpr int threads = 256;
    const int64_t first_count = batch_size * rotation_count * first_pixel_count;
    const int64_t second_count = batch_size * rotation_count * second_pixel_count;
    const int first_blocks = static_cast<int>((first_count + threads - 1) / threads);
    const int second_blocks = static_cast<int>((second_count + threads - 1) / threads);
    dual_weighted_sums_pairs_f32_kernel<T><<<first_blocks, threads, 0, stream>>>(
        pair_probabilities, pair_translations, row_offsets, first_values, first_output,
        batch_size, rotation_count, translation_count, pair_count, first_pixel_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    dual_weighted_sums_pairs_f32_kernel<T><<<second_blocks, threads, 0, stream>>>(
        pair_probabilities, pair_translations, row_offsets, second_values, second_output,
        batch_size, rotation_count, translation_count, pair_count, second_pixel_count);
    return cudaGetLastError();
}

ffi::Error DualWeightedSumsPairsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer pair_probabilities,
    ffi::AnyBuffer pair_translations,
    ffi::AnyBuffer row_offsets,
    ffi::AnyBuffer first_values,
    ffi::AnyBuffer second_values,
    ffi::Result<ffi::AnyBuffer> first_output,
    ffi::Result<ffi::AnyBuffer> second_output)
{
    const auto value_type = first_values.element_type();
    if (pair_probabilities.element_type() != ffi::DataType::F32 ||
        pair_translations.element_type() != ffi::DataType::S32 ||
        row_offsets.element_type() != ffi::DataType::S32 ||
        (value_type != ffi::DataType::C64 && value_type != ffi::DataType::C128) ||
        second_values.element_type() != value_type ||
        first_output->element_type() != value_type ||
        second_output->element_type() != value_type)
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsPairsF32: expected F32 pair probabilities, S32 pair "
            "translations and row offsets, and matching C64/C128 values/outputs");
    const auto probability_dims = pair_probabilities.dimensions();
    const auto translation_dims = pair_translations.dimensions();
    const auto offset_dims = row_offsets.dimensions();
    const auto first_dims = first_values.dimensions();
    const auto second_dims = second_values.dimensions();
    const auto first_output_dims = first_output->dimensions();
    const auto second_output_dims = second_output->dimensions();
    if (probability_dims.size() != 2 || translation_dims.size() != 2 ||
        offset_dims.size() != 2 || first_dims.size() != 3 || second_dims.size() != 3 ||
        first_output_dims.size() != 3 || second_output_dims.size() != 3 ||
        probability_dims[0] <= 0 || probability_dims[1] <= 0 ||
        translation_dims[0] != probability_dims[0] ||
        translation_dims[1] != probability_dims[1] ||
        offset_dims[0] != probability_dims[0] || offset_dims[1] < 2 ||
        first_dims[0] != probability_dims[0] || second_dims[0] != probability_dims[0] ||
        first_dims[1] != second_dims[1] || first_dims[1] <= 0 ||
        first_dims[2] <= 0 || second_dims[2] <= 0 ||
        first_output_dims[0] != probability_dims[0] ||
        first_output_dims[1] != offset_dims[1] - 1 ||
        first_output_dims[2] != first_dims[2] ||
        second_output_dims[0] != probability_dims[0] ||
        second_output_dims[1] != offset_dims[1] - 1 ||
        second_output_dims[2] != second_dims[2])
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsPairsF32: inconsistent [B,P], [B,R+1], [B,T,N] and [B,R,N] shapes");
    const int64_t batch_size = probability_dims[0];
    const int64_t pair_count = probability_dims[1];
    const int64_t rotation_count = offset_dims[1] - 1;
    const int64_t translation_count = first_dims[1];
    const int64_t largest_count =
        batch_size * rotation_count * std::max(first_dims[2], second_dims[2]);
    if (largest_count > static_cast<int64_t>(std::numeric_limits<int>::max()) * 256)
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsPairsF32: output exceeds CUDA grid bounds");
    cudaError_t err;
    if (value_type == ffi::DataType::C64) {
        err = launch_dual_weighted_sums_pairs_f32<float>(
            stream,
            static_cast<const float*>(pair_probabilities.untyped_data()),
            static_cast<const int32_t*>(pair_translations.untyped_data()),
            static_cast<const int32_t*>(row_offsets.untyped_data()),
            reinterpret_cast<const float2*>(first_values.untyped_data()),
            reinterpret_cast<const float2*>(second_values.untyped_data()),
            reinterpret_cast<float2*>(first_output->untyped_data()),
            reinterpret_cast<float2*>(second_output->untyped_data()),
            batch_size, rotation_count, translation_count, pair_count,
            first_dims[2], second_dims[2]);
    } else {
        err = launch_dual_weighted_sums_pairs_f32<double>(
            stream,
            static_cast<const float*>(pair_probabilities.untyped_data()),
            static_cast<const int32_t*>(pair_translations.untyped_data()),
            static_cast<const int32_t*>(row_offsets.untyped_data()),
            reinterpret_cast<const double2*>(first_values.untyped_data()),
            reinterpret_cast<const double2*>(second_values.untyped_data()),
            reinterpret_cast<double2*>(first_output->untyped_data()),
            reinterpret_cast<double2*>(second_output->untyped_data()),
            batch_size, rotation_count, translation_count, pair_count,
            first_dims[2], second_dims[2]);
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DualWeightedSumsPairsF32, DualWeightedSumsPairsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

// Flat real-row form of the pair-sparse dual weighted sums: one output row per
// listed (batch, rotation) pair instead of the padded [B, R, N] layout. At
// 100k/256 K=4 about 71 % of the padded rows are padding (mean local rotation
// count 559 in buckets of up to a few thousand rows), and the padded kernel spent
// most of its threads and all of its output bandwidth on them. Each listed row
// accumulates its pairs in the same order with the same fma as the padded
// kernel, so out[f] is bit-identical to the padded output at (row_batch[f],
// row_rotation[f]).
template <typename T>
__global__ void dual_weighted_sums_pairs_rows_f32_kernel(
    const float* pair_probabilities,
    const int32_t* pair_translations,
    const int32_t* row_offsets,
    const int32_t* row_batch,
    const int32_t* row_rotation,
    const vec2_t<T>* values,
    vec2_t<T>* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pair_count,
    int64_t pixel_count,
    int64_t row_count,
    int64_t pixel_blocks)
{
    const int64_t flat_row = static_cast<int64_t>(blockIdx.x) / pixel_blocks;
    const int64_t pixel =
        (static_cast<int64_t>(blockIdx.x) % pixel_blocks) * blockDim.x + threadIdx.x;
    if (flat_row >= row_count || pixel >= pixel_count) return;
    const int64_t batch = row_batch[flat_row];
    const int64_t rotation = row_rotation[flat_row];
    T sum_real = static_cast<T>(0);
    T sum_imag = static_cast<T>(0);
    if (batch >= 0 && batch < batch_size && rotation >= 0 && rotation < rotation_count) {
        const int64_t offset_base = batch * (rotation_count + 1) + rotation;
        int64_t start = row_offsets[offset_base];
        int64_t end = row_offsets[offset_base + 1];
        if (start < 0) start = 0;
        if (end > pair_count) end = pair_count;
        const int64_t pair_base = batch * pair_count;
        const int64_t value_base = batch * translation_count * pixel_count + pixel;
        for (int64_t pair = start; pair < end; ++pair) {
            const float weight = pair_probabilities[pair_base + pair];
            if (weight == 0.0f) continue;
            const int64_t translation = pair_translations[pair_base + pair];
            if (translation < 0 || translation >= translation_count) continue;
            const vec2_t<T> value = values[value_base + translation * pixel_count];
            sum_real = dual_weighted_fma(weight, value.x, sum_real);
            sum_imag = dual_weighted_fma(weight, value.y, sum_imag);
        }
    }
    output[flat_row * pixel_count + pixel] = make_v2(sum_real, sum_imag);
}

template <typename T>
cudaError_t launch_dual_weighted_sums_pairs_rows_f32(
    cudaStream_t stream,
    const float* pair_probabilities,
    const int32_t* pair_translations,
    const int32_t* row_offsets,
    const int32_t* row_batch,
    const int32_t* row_rotation,
    const vec2_t<T>* first_values,
    const vec2_t<T>* second_values,
    vec2_t<T>* first_output,
    vec2_t<T>* second_output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pair_count,
    int64_t row_count,
    int64_t first_pixel_count,
    int64_t second_pixel_count)
{
    constexpr int threads = 256;
    if (row_count <= 0) return cudaSuccess;
    const int64_t first_pixel_blocks = (first_pixel_count + threads - 1) / threads;
    const int64_t second_pixel_blocks = (second_pixel_count + threads - 1) / threads;
    const int64_t first_blocks = row_count * first_pixel_blocks;
    const int64_t second_blocks = row_count * second_pixel_blocks;
    if (first_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        second_blocks > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return cudaErrorInvalidConfiguration;
    dual_weighted_sums_pairs_rows_f32_kernel<T><<<static_cast<int>(first_blocks), threads, 0, stream>>>(
        pair_probabilities, pair_translations, row_offsets, row_batch, row_rotation,
        first_values, first_output, batch_size, rotation_count, translation_count,
        pair_count, first_pixel_count, row_count, first_pixel_blocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    dual_weighted_sums_pairs_rows_f32_kernel<T><<<static_cast<int>(second_blocks), threads, 0, stream>>>(
        pair_probabilities, pair_translations, row_offsets, row_batch, row_rotation,
        second_values, second_output, batch_size, rotation_count, translation_count,
        pair_count, second_pixel_count, row_count, second_pixel_blocks);
    return cudaGetLastError();
}

ffi::Error DualWeightedSumsPairsRowsF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer pair_probabilities,
    ffi::AnyBuffer pair_translations,
    ffi::AnyBuffer row_offsets,
    ffi::AnyBuffer row_batch,
    ffi::AnyBuffer row_rotation,
    ffi::AnyBuffer first_values,
    ffi::AnyBuffer second_values,
    ffi::Result<ffi::AnyBuffer> first_output,
    ffi::Result<ffi::AnyBuffer> second_output)
{
    const auto value_type = first_values.element_type();
    if (pair_probabilities.element_type() != ffi::DataType::F32 ||
        pair_translations.element_type() != ffi::DataType::S32 ||
        row_offsets.element_type() != ffi::DataType::S32 ||
        row_batch.element_type() != ffi::DataType::S32 ||
        row_rotation.element_type() != ffi::DataType::S32 ||
        (value_type != ffi::DataType::C64 && value_type != ffi::DataType::C128) ||
        second_values.element_type() != value_type ||
        first_output->element_type() != value_type ||
        second_output->element_type() != value_type)
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsPairsRowsF32: expected F32 pair probabilities, S32 pair "
            "translations, row offsets and row lists, and matching C64/C128 values/outputs");
    const auto probability_dims = pair_probabilities.dimensions();
    const auto translation_dims = pair_translations.dimensions();
    const auto offset_dims = row_offsets.dimensions();
    const auto row_batch_dims = row_batch.dimensions();
    const auto row_rotation_dims = row_rotation.dimensions();
    const auto first_dims = first_values.dimensions();
    const auto second_dims = second_values.dimensions();
    const auto first_output_dims = first_output->dimensions();
    const auto second_output_dims = second_output->dimensions();
    if (probability_dims.size() != 2 || translation_dims.size() != 2 ||
        offset_dims.size() != 2 || row_batch_dims.size() != 1 || row_rotation_dims.size() != 1 ||
        first_dims.size() != 3 || second_dims.size() != 3 ||
        first_output_dims.size() != 2 || second_output_dims.size() != 2 ||
        probability_dims[0] <= 0 || probability_dims[1] <= 0 ||
        translation_dims[0] != probability_dims[0] ||
        translation_dims[1] != probability_dims[1] ||
        offset_dims[0] != probability_dims[0] || offset_dims[1] < 2 ||
        row_rotation_dims[0] != row_batch_dims[0] ||
        first_dims[0] != probability_dims[0] || second_dims[0] != probability_dims[0] ||
        first_dims[1] != second_dims[1] || first_dims[1] <= 0 ||
        first_dims[2] <= 0 || second_dims[2] <= 0 ||
        first_output_dims[0] != row_batch_dims[0] ||
        first_output_dims[1] != first_dims[2] ||
        second_output_dims[0] != row_batch_dims[0] ||
        second_output_dims[1] != second_dims[2])
        return ffi::Error::InvalidArgument(
            "DualWeightedSumsPairsRowsF32: inconsistent [B,P], [B,R+1], [F], [B,T,N] and [F,N] shapes");
    const int64_t batch_size = probability_dims[0];
    const int64_t pair_count = probability_dims[1];
    const int64_t rotation_count = offset_dims[1] - 1;
    const int64_t translation_count = first_dims[1];
    const int64_t row_count = row_batch_dims[0];
    cudaError_t err;
    if (value_type == ffi::DataType::C64) {
        err = launch_dual_weighted_sums_pairs_rows_f32<float>(
            stream,
            static_cast<const float*>(pair_probabilities.untyped_data()),
            static_cast<const int32_t*>(pair_translations.untyped_data()),
            static_cast<const int32_t*>(row_offsets.untyped_data()),
            static_cast<const int32_t*>(row_batch.untyped_data()),
            static_cast<const int32_t*>(row_rotation.untyped_data()),
            reinterpret_cast<const float2*>(first_values.untyped_data()),
            reinterpret_cast<const float2*>(second_values.untyped_data()),
            reinterpret_cast<float2*>(first_output->untyped_data()),
            reinterpret_cast<float2*>(second_output->untyped_data()),
            batch_size, rotation_count, translation_count, pair_count, row_count,
            first_dims[2], second_dims[2]);
    } else {
        err = launch_dual_weighted_sums_pairs_rows_f32<double>(
            stream,
            static_cast<const float*>(pair_probabilities.untyped_data()),
            static_cast<const int32_t*>(pair_translations.untyped_data()),
            static_cast<const int32_t*>(row_offsets.untyped_data()),
            static_cast<const int32_t*>(row_batch.untyped_data()),
            static_cast<const int32_t*>(row_rotation.untyped_data()),
            reinterpret_cast<const double2*>(first_values.untyped_data()),
            reinterpret_cast<const double2*>(second_values.untyped_data()),
            reinterpret_cast<double2*>(first_output->untyped_data()),
            reinterpret_cast<double2*>(second_output->untyped_data()),
            batch_size, rotation_count, translation_count, pair_count, row_count,
            first_dims[2], second_dims[2]);
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DualWeightedSumsPairsRowsF32, DualWeightedSumsPairsRowsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

// Fused sparse pass-2 posterior handlers (needs relion_ampere_inclusive_sum_f32).
#include "sparse_pass2_posterior.cuh"

// Flat-row translate-and-sum for the device-resident pass-2 M-step
// (needs relion_score_translate_f32's phase and rotation).
#include "relion_translate_sum.cuh"
