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

struct VdamCandidateBlockTraceHeader
{
    char magic[16];
    std::uint32_t schema_version;
    std::uint32_t header_size;
    std::uint32_t record_size;
    std::uint32_t iteration;
    std::uint64_t record_count;
    std::uint64_t capacity;
    std::uint64_t reserved0;
    std::uint64_t reserved1;
};

struct VdamCandidateBlockTraceRecord
{
    std::uint64_t launch_sequence;
    std::int64_t particle_id;
    std::uint64_t block_start_globaltimer;
    std::uint64_t first_atomic_globaltimer;
    std::uint64_t block_end_globaltimer;
    std::uint32_t orientation_row;
    std::int32_t worker_id;
    std::int32_t class_id;
    std::uint32_t sm_id;
    std::uint32_t image_count;
    std::uint32_t iteration;
    std::uint32_t flags;
    std::uint32_t reserved;
};

static_assert(sizeof(VdamCandidateBlockTraceHeader) == 64,
              "candidate VDAM block trace header must be 64 bytes");
static_assert(sizeof(VdamCandidateBlockTraceRecord) == 72,
              "candidate VDAM block trace record must be 72 bytes");

class VdamCandidateBlockTraceWriter
{
public:
    bool requested()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return requested_;
    }

    bool healthy()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return !requested_ || healthy_;
    }

    std::uint32_t iteration()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        return header_.iteration;
    }

    bool reserve(std::uint64_t record_count, std::uint64_t* launch_sequence)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked();
        if (!requested_ || !healthy_ || record_count == 0 ||
            reserved_records_ > header_.capacity ||
            record_count > header_.capacity - reserved_records_)
            return false;
        *launch_sequence = next_launch_++;
        reserved_records_ += record_count;
        return true;
    }

    bool append(const VdamCandidateBlockTraceRecord* records, std::uint64_t record_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!requested_ || !healthy_ || records == nullptr || record_count == 0)
            return false;
        output_.seekp(0, std::ios::end);
        output_.write(
            reinterpret_cast<const char*>(records),
            static_cast<std::streamsize>(record_count * sizeof(*records)));
        written_records_ += record_count;
        header_.record_count = written_records_;
        output_.seekp(0, std::ios::beg);
        output_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
        output_.flush();
        healthy_ = static_cast<bool>(output_);
        return healthy_;
    }

private:
    void initialize_locked()
    {
        if (initialized_) return;
        initialized_ = true;
        const char* path = std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE");
        if (path == nullptr || path[0] == '\0') return;
        requested_ = true;
        const char* iteration_text =
            std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_ITER");
        const char* capacity_text =
            std::getenv("RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPACITY");
        if (iteration_text == nullptr || iteration_text[0] == '\0') return;
        errno = 0;
        char* iteration_end = nullptr;
        const unsigned long iteration =
            std::strtoul(iteration_text, &iteration_end, 10);
        if (errno != 0 || iteration_end == iteration_text || *iteration_end != '\0' ||
            iteration == 0 || iteration > std::numeric_limits<std::uint32_t>::max())
            return;
        std::uint64_t capacity = 1000000;
        if (capacity_text != nullptr && capacity_text[0] != '\0')
        {
            errno = 0;
            char* capacity_end = nullptr;
            const unsigned long long parsed =
                std::strtoull(capacity_text, &capacity_end, 10);
            if (errno != 0 || capacity_end == capacity_text || *capacity_end != '\0' ||
                parsed == 0)
                return;
            capacity = static_cast<std::uint64_t>(parsed);
        }
        const char magic[16] = {
            'R', 'E', 'L', 'I', 'O', 'N', '_', 'V', 'D', 'A', 'M', '_', 'B', 'T', '1', '\0'};
        std::memcpy(header_.magic, magic, sizeof(magic));
        header_.schema_version = 1;
        header_.header_size = sizeof(header_);
        header_.record_size = sizeof(VdamCandidateBlockTraceRecord);
        header_.iteration = static_cast<std::uint32_t>(iteration);
        header_.record_count = 0;
        header_.capacity = capacity;
        header_.reserved0 = 0;
        header_.reserved1 = 0;
        output_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!output_) return;
        output_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
        output_.flush();
        healthy_ = static_cast<bool>(output_);
    }

    std::mutex mutex_;
    bool initialized_ = false;
    bool requested_ = false;
    bool healthy_ = false;
    std::fstream output_;
    VdamCandidateBlockTraceHeader header_ = {};
    std::uint64_t next_launch_ = 0;
    std::uint64_t reserved_records_ = 0;
    std::uint64_t written_records_ = 0;
};

static VdamCandidateBlockTraceWriter& vdam_candidate_block_trace_writer()
{
    static VdamCandidateBlockTraceWriter writer;
    return writer;
}

static __device__ __forceinline__ std::uint64_t vdam_candidate_globaltimer()
{
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ __forceinline__ std::uint32_t vdam_candidate_smid()
{
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

/* ================================================================== */
/*                     Type helpers                                    */
/* ================================================================== */

template <typename T> struct Vec2;
template <> struct Vec2<float>  { using type = float2; };
template <> struct Vec2<double> { using type = double2; };
template <typename T> using vec2_t = typename Vec2<T>::type;

static __device__ __forceinline__ float2  make_v2(float  a, float  b) { return make_float2(a, b); }
static __device__ __forceinline__ double2 make_v2(double a, double b) { return make_double2(a, b); }

static __device__ __forceinline__ int floor_int(float  x) { return (int)floorf(x); }
static __device__ __forceinline__ int floor_int(double x) { return (int)floor(x); }
/* RELION's CUDA BP kernels call floorf() even when XFLOAT is double.  The
 * conversion to float happens only for the integer bucket; the fractional
 * remainder is still formed from the original XFLOAT coordinate. */
static __device__ __forceinline__ int relion_floor_int(float  x) { return (int)floorf(x); }
static __device__ __forceinline__ int relion_floor_int(double x) { return (int)floorf((float)x); }
static __device__ __forceinline__ int round_int(float  x) { return (int)rintf(x); }
static __device__ __forceinline__ int round_int(double x) { return (int)rint(x); }

/* ================================================================== */
/*                 Cubic B-spline basis function                       */
/* ================================================================== */

/* Evaluate the cubic B-spline basis function B3(t).
 * B3(t) is non-zero only for |t| < 2:
 *   |t| < 1:  4 - 6t² + 3|t|³
 *   1 ≤ |t| < 2:  (2 - |t|)³
 *
 * Note: this matches the JAX _cubic_basis function in cubic_interpolation.py.
 */
template <typename T>
static __device__ __forceinline__ T cubic_basis(T t) {
    T at = (t >= (T)0) ? t : -t;
    if (at >= (T)2) return (T)0;
    if (at >= (T)1) {
        T u = (T)2 - at;
        return u * u * u;
    }
    return (T)4 - (T)6 * at * at + (T)3 * at * at * at;
}

/* Modular wrap for periodic boundary: result in [0, N). */
static __device__ __forceinline__ int wrap_mod(int x, int N) {
    int r = x % N;
    return r < 0 ? r + N : r;
}

/* Recover the full last-axis size from its packed half-spectrum size.
 *
 * Standard RECOVAR grids have an even last axis, so N2 = 2*(N2_eff-1).
 * RELION BackProjector accumulators are odd and cubic; preserve their final
 * centered z plane when that shape is unambiguous from N0/N1/N2_eff.
 * Arbitrary odd rectangular last axes remain unsupported because the packed
 * shape alone cannot distinguish full sizes 2*N2_eff-2 and 2*N2_eff-1.
 */
static __device__ __forceinline__ int full_z_size_from_half(
    int N0, int N1, int N2_eff)
{
    const bool odd_cubic =
        (N0 & 1) && N0 == N1 && N2_eff == N0 / 2 + 1;
    return odd_cubic ? N0 : 2 * (N2_eff - 1);
}

/* Match RELION BP.cuh's compiled radius predicate in its physical axis order.
 * RECOVAR's backprojection coordinates rk2/rk1/rk0 correspond to RELION's
 * physical x/y/z. The explicit round-to-nearest operations are observable at
 * the exact outer rim: reassociating the sum can flip r2 > max_r2 by one ulp. */
static __device__ __forceinline__ float relion_radius_squared(
    float rk0, float rk1, float rk2)
{
    const float y2 = __fmul_rn(rk1, rk1);
    const float xy2 = __fmaf_rn(rk2, rk2, y2);
    return __fmaf_rn(rk0, rk0, xy2);
}

static __device__ __forceinline__ double relion_radius_squared(
    double rk0, double rk1, double rk2)
{
    const double y2 = __dmul_rn(rk1, rk1);
    const double xy2 = __fma_rn(rk2, rk2, y2);
    return __fma_rn(rk0, rk0, xy2);
}

#define BLOCK_SIZE 256

/* ================================================================== */
/*   Device helpers: scatter one value into volume at rotated coords   */
/* ================================================================== */

template <typename T>
static __device__ __forceinline__ bool relion_compact_trilinear_oob(
    T relion_x, T relion_y, T relion_z, int maxR)
{
    /* RELION BackProjector::backproject2Dto3D accumulates into a compact
     * Fourier box sized x=maxR+2, y/z=2*maxR+3 with STARTINGY/Z=-(maxR+1).
     * For linear interpolation it drops the entire source pixel if any of the
     * eight neighbors would leave that compact box. RECOVAR's normal scatter
     * clips neighbors independently in the full padded box; RELION parity must
     * reproduce the all-or-nothing compact-boundary skip. */
    const int x0 = relion_floor_int(relion_x);
    const int y0 = relion_floor_int(relion_y) + maxR + 1;
    const int z0 = relion_floor_int(relion_z) + maxR + 1;
    const int xdim = maxR + 2;
    const int ydim = 2 * maxR + 3;
    return x0 < 0 || x0 + 1 >= xdim ||
           y0 < 0 || y0 + 1 >= ydim ||
           z0 < 0 || z0 + 1 >= ydim;
}

/* scatter_nearest: atomicAdd one value at the nearest voxel.
 *
 * HALF_VOL: Hermitian fold approach.  Voxels with kz >= 0 scatter
 * directly.  Voxels with kz < 0 are folded to the Hermitian partner
 * at ((N0-i0)%N0, (N1-i1)%N1, |kz|) with conjugated value.
 * This is the correct adjoint of half_volume_to_full_volume (expand).
 *
 * CONJ_MODE (only when HALF_VOL):
 *   0 = normal scatter
 *   1 = double interior kz (0 < hkz < ic2) — primary scatter with
 *       HALF_IMG optimization (accounts for conjugate partner)
 *   2 = boundary only — skip interior kz, scatter only kz=0 and
 *       Nyquist (conjugate scatter with HALF_IMG optimization)
 *
 * REAL_DATA: when true, vol stores 1 float per voxel (not 2).
 *   Only val_re is used; val_im is ignored.  Hermitian fold does NOT
 *   negate (conj(real) = real).  Offset skips the *2 complex stride.
 */
template <typename T, bool HALF_VOL, int CONJ_MODE = 0, bool REAL_DATA = false>
static __device__ __forceinline__ void scatter_nearest(
    T* __restrict__ vol,
    T rk0, T rk1, T rk2, T val_re, T val_im,
    T c0, T c1, T c2,
    int N0, int N1, int N2_eff, int stride0, int stride1)
{
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;

    if (HALF_VOL) {
        const int ic2 = (int)c2;
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        const T g2_full = rk2 + c2;
        int i0 = round_int(g0);
        int i1 = round_int(g1);
        const int i2 = round_int(g2_full);
        if ((unsigned)i0 >= (unsigned)N0 ||
            (unsigned)i1 >= (unsigned)N1 ||
            (unsigned)i2 >= (unsigned)N2_full) return;
        const int kz = i2 - ic2;
        int hkz;
        if (kz >= 0) {
            hkz = kz;
        } else if ((N2_full & 1) == 0 && -kz == ic2) {
            /* Nyquist (kz = -N/2 = +N/2): self-conjugate, scatter directly */
            hkz = ic2;
        } else {
            /* Fold to Hermitian partner in centered (fftshift) convention:
             * shifted[j] = Y[(j - N//2) % N], Hermitian u' = (N - u) % N,
             * partner(j) = (N - j + 2*(N//2)) % N.
             * Even N: 2*(N//2) = N   => partner(j) = (N - j) % N
             * Odd N:  2*(N//2) = N-1 => partner(j) = (N - 1 - j) % N
             * General: partner(j) = (N - (N & 1) - j) % N.
             * NOTE: the sign on (N & 1) is MINUS, not plus. */
            i0 = (N0 - (N0 & 1) - i0) % N0;
            i1 = (N1 - (N1 & 1) - i1) % N1;
            hkz = -kz;
            if (!REAL_DATA) val_im = -val_im;  /* conj(real) = real */
        }
        if (hkz > ic2) return;  /* out of half-vol bounds */
        /* CONJ_MODE 2: only scatter to boundary columns (kz=0, Nyquist) */
        if (CONJ_MODE == 2 && hkz > 0 && hkz < ic2) return;
        /* CONJ_MODE 1: double interior kz to account for conjugate partner */
        if (CONJ_MODE == 1 && hkz > 0 && hkz < ic2) {
            val_re *= (T)2;
            if (!REAL_DATA) val_im *= (T)2;
        }
        if (REAL_DATA) {
            const int off = i0 * stride0 + i1 * stride1 + hkz;
            atomicAdd(&vol[off], val_re);
        } else {
            const int off = (i0 * stride0 + i1 * stride1 + hkz) * 2;
            atomicAdd(&vol[off],     val_re);
            atomicAdd(&vol[off + 1], val_im);
        }
        return;
    }

    /* Non-HALF_VOL path */
    const T g2 = rk2 + c2;
    const int i0 = round_int(g0);
    const int i1 = round_int(g1);
    const int i2 = round_int(g2);
    if ((unsigned)i0 >= (unsigned)N0 ||
        (unsigned)i1 >= (unsigned)N1 ||
        (unsigned)i2 >= (unsigned)N2_eff) return;
    if (REAL_DATA) {
        const int off = i0 * stride0 + i1 * stride1 + i2;
        atomicAdd(&vol[off], val_re);
    } else {
        const int off = (i0 * stride0 + i1 * stride1 + i2) * 2;
        atomicAdd(&vol[off],     val_re);
        atomicAdd(&vol[off + 1], val_im);
    }
}

/* scatter_trilinear: atomicAdd one value at 8 trilinear neighbors.
 *
 * HALF_VOL: Hermitian fold approach.  For each trilinear neighbor,
 * if kz >= 0, scatter w*val directly.  If kz < 0, fold to the
 * Hermitian partner ((N0-j0)%N0, (N1-j1)%N1, |kz|) and scatter
 * w*conj(val).  This is the correct adjoint of expand (half→full).
 *
 * CONJ_MODE: same as scatter_nearest (0=normal, 1=double interior, 2=boundary only)
 * REAL_DATA: same as scatter_nearest (1 float/voxel, no conj, no *2 offset)
 */
template <typename T, bool HALF_VOL, int CONJ_MODE = 0, bool REAL_DATA = false>
static __device__ __forceinline__ void scatter_trilinear(
    T* __restrict__ vol,
    T rk0, T rk1, T rk2, T val_re, T val_im,
    T c0, T c1, T c2,
    int N0, int N1, int N2_eff, int stride0, int stride1,
    bool relion_floorf_quirk = false)
{
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;

    if (HALF_VOL) {
        const int ic2 = (int)c2;
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        const T g2_full = rk2 + c2;

        if (g0 < (T)-1 || g0 >= (T)N0 ||
            g1 < (T)-1 || g1 >= (T)N1 ||
            g2_full < (T)-1 || g2_full >= (T)N2_full) return;

        /* RELION applies floorf to the unshifted XFLOAT coordinates and only
         * then adds the integer compact-volume origin.  Preserve that order:
         * casting (rk + origin) to float is not equivalent near cancellation. */
        const int b0 = relion_floorf_quirk
            ? relion_floor_int(rk0) + (int)c0
            : floor_int(g0);
        const int b1 = relion_floorf_quirk
            ? relion_floor_int(rk1) + (int)c1
            : floor_int(g1);
        const int b2 = relion_floorf_quirk
            ? relion_floor_int(rk2) + (int)c2
            : floor_int(g2_full);
        const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2_full - (T)b2;
        const T w0[2] = {(T)1 - f0, f0};
        const T w1[2] = {(T)1 - f1, f1};
        const T w2[2] = {(T)1 - f2, f2};

        /* Per-neighbor Hermitian fold: kz >= 0 direct, kz < 0 fold+conj.
         * This correctly implements the adjoint of half_volume_to_full_volume. */
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
                    const int j2 = b2 + d2;
                    if ((unsigned)j2 >= (unsigned)N2_full) continue;
                    const int kz = j2 - ic2;
                    const T w = ww * w2[d2];
                    int sj0 = j0, sj1 = j1;
                    int hkz;
                    T sre = w * val_re;
                    T sim = REAL_DATA ? (T)0 : w * val_im;
                    if (kz >= 0) {
                        hkz = kz;
                    } else if ((N2_full & 1) == 0 && -kz == ic2) {
                        /* Nyquist: self-conjugate, scatter directly */
                        hkz = ic2;
                    } else {
                        /* Fold to Hermitian partner in centered convention:
                         * partner(j) = (N - (N & 1) - j) % N.
                         * See scatter_nearest comment for derivation. */
                        sj0 = (N0 - (N0 & 1) - j0) % N0;
                        sj1 = (N1 - (N1 & 1) - j1) % N1;
                        hkz = -kz;
                        if (!REAL_DATA) sim = -sim;  /* conj(real) = real */
                    }
                    if (hkz > ic2) continue;  /* out of half-vol bounds */
                    /* CONJ_MODE 2: only scatter to boundary columns (kz=0, Nyquist) */
                    if (CONJ_MODE == 2 && hkz > 0 && hkz < ic2) continue;
                    /* CONJ_MODE 1: double interior kz to account for conjugate partner */
                    if (CONJ_MODE == 1 && hkz > 0 && hkz < ic2) {
                        sre *= (T)2;
                        if (!REAL_DATA) sim *= (T)2;
                    }
                    if (REAL_DATA) {
                        const int off = sj0 * stride0 + sj1 * stride1 + hkz;
                        atomicAdd(&vol[off], sre);
                    } else {
                        const int off = (sj0 * stride0 + sj1 * stride1 + hkz) * 2;
                        atomicAdd(&vol[off],     sre);
                        atomicAdd(&vol[off + 1], sim);
                    }
                }
            }
        }
        return;
    }

    /* Non-HALF_VOL path */
    const T g2 = rk2 + c2;

    if (g0 < (T)-1 || g0 >= (T)N0 ||
        g1 < (T)-1 || g1 >= (T)N1 ||
        g2 < (T)-1 || g2 >= (T)N2_eff) return;

    const int b0 = floor_int(g0);
    const int b1 = floor_int(g1);
    const int b2 = floor_int(g2);
    const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2 - (T)b2;
    const T w0[2] = {(T)1 - f0, f0};
    const T w1[2] = {(T)1 - f1, f1};
    const T w2[2] = {(T)1 - f2, f2};

    #pragma unroll
    for (int d0 = 0; d0 < 2; d0++) {
        const int j0 = b0 + d0;
        if ((unsigned)j0 >= (unsigned)N0) continue;
        #pragma unroll
        for (int d1 = 0; d1 < 2; d1++) {
            const int j1 = b1 + d1;
            if ((unsigned)j1 >= (unsigned)N1) continue;
            const T ww = w0[d0] * w1[d1];
            #pragma unroll
            for (int d2 = 0; d2 < 2; d2++) {
                const int j2 = b2 + d2;
                if ((unsigned)j2 >= (unsigned)N2_eff) continue;
                const T w = ww * w2[d2];
                if (REAL_DATA) {
                    const int off = j0 * stride0 + j1 * stride1 + j2;
                    atomicAdd(&vol[off], w * val_re);
                } else {
                    const int off = (j0 * stride0 + j1 * stride1 + j2) * 2;
                    atomicAdd(&vol[off],     w * val_re);
                    atomicAdd(&vol[off + 1], w * val_im);
                }
            }
        }
    }
}

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

/* ================================================================== */
/*                  Backproject kernel                                 */
/* ================================================================== */

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG, bool REAL_DATA = false>
__global__ void __launch_bounds__(BLOCK_SIZE)
backproject_kernel(
    T*       __restrict__ vol,
    const T* __restrict__ img,
    const T* __restrict__ rot,   /* (n_images, 6) */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* On-the-fly frequency coords — row-major pixel layout */
    const int k0_idx = pix / image_w;   /* row index */
    const int k1_idx = pix % image_w;   /* col index */

    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        /* rfft: k1 = 0, ups, ..., (W/2)*ups.
         * Use negative Nyquist to match centered full-DFT convention. */
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling     /* Nyquist: -W/2 */
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;  /* full: centered */
    }

    /* Pre-rotation disk check: rotation preserves ||k||, so
     * k0² + k1² == rk0² + rk1² + rk2².  Skip before loading R. */
    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    /* Rotate  (cz=0  →  only 6 elements) */
    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    /* Load pixel — scalar for REAL_DATA, complex pair otherwise */
    T val_re, val_im;
    if (REAL_DATA) {
        val_re = img[img_idx * n_pixels + pix];
        val_im = (T)0;
    } else {
        using V2 = vec2_t<T>;
        V2 px = reinterpret_cast<const V2*>(img)[img_idx * n_pixels + pix];
        val_re = px.x;
        val_im = px.y;
    }

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

    /* ── CONJ_MODE optimization for HALF_IMG + HALF_VOL backprojection ──
     *
     * For rfft half-images scattered into a half-volume, each non-boundary
     * rfft pixel generates TWO scatters: primary at rotated(k0,k1) and
     * conjugate at rotated(-k0,-k1) with conj(val).
     *
     * Key insight: when the conjugate coords satisfy crk = -rk (which is
     * true for all pixels EXCEPT k0_idx==0 with even H), the conjugate
     * scatter lands at the same half-volume position as the primary after
     * Hermitian fold, for interior kz (0 < hkz < ic2).  So we can:
     *   - CONJ_MODE=1 on primary: double interior kz weight
     *   - CONJ_MODE=2 on conjugate: skip interior kz (only boundary)
     * This eliminates ~all conjugate scatter work → ~2x speedup.
     *
     * The optimization does NOT apply when:
     *   (a) Boundary rfft pixels (k1_idx==0 or Nyquist): no conjugate
     *       scatter exists, so doubling the primary would be wrong.
     *   (b) k0_idx==0 with even H: the Nyquist row's conjugate uses
     *       crk = rot @ (k0, -k1) ≠ -rk, so scatters land at different
     *       half-vol positions.  Must use normal scatter for both.
     *
     * IMPORTANT: Do NOT replace this with full→half volume conversion
     * (e.g. backproject to full volume then contract).  That loses both
     * the memory savings and the ~2x scatter speedup.
     */

    /* Determine if CONJ_MODE optimization applies to this pixel.
     * True when: (1) this is a non-boundary rfft pixel with a conjugate
     * scatter, (2) crk = -rk (not the k0 Nyquist special case), AND
     * (3) BOTH primary (rk+c) and conjugate (-rk+c) scatter positions
     * are within full-volume bounds.
     *
     * Why (3) is needed: if the primary is OOB, CONJ_MODE=1 doubling
     * never fires, but CONJ_MODE=2 still skips the conjugate's interior
     * kz → contribution lost.  Conversely, if the conjugate is OOB,
     * CONJ_MODE=1 doubles the primary but the conjugate can't match →
     * phantom contribution.  Disabling conj_opt when either is OOB
     * makes both fall back to normal (CONJ_MODE=0) scatter. */
    bool conj_opt = HALF_IMG && HALF_VOL
        && (k1_idx > 0 && k1_idx * 2 != full_image_w)    /* non-boundary */
        && !(k0_idx == 0 && (image_h & 1) == 0);         /* not Nyquist row */

    if (conj_opt) {
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        if (ORDER == 0) {
            /* Nearest: both round(rk+c) and round(-rk+c) must be in [0,N). */
            const int pi0 = round_int(rk0+c0), pi1 = round_int(rk1+c1);
            const int pi2 = round_int(rk2+c2);
            const int ci0 = round_int(-rk0+c0), ci1 = round_int(-rk1+c1);
            const int ci2 = round_int(-rk2+c2);
            if ((unsigned)pi0 >= (unsigned)N0 || (unsigned)pi1 >= (unsigned)N1 ||
                (unsigned)pi2 >= (unsigned)N2_full ||
                (unsigned)ci0 >= (unsigned)N0 || (unsigned)ci1 >= (unsigned)N1 ||
                (unsigned)ci2 >= (unsigned)N2_full)
                conj_opt = false;
        } else {
            /* Trilinear: all 8 neighbors of both primary and conjugate must
             * be within [0, N-1].  g in [0, N-1] ensures floor(g) >= 0 and
             * floor(g)+1 <= N-1, so no trilinear neighbor is OOB.
             * (At g = N-1 exactly, neighbor j+1 = N gets weight 0 → harmless.) */
            const T pg0 = rk0+c0, pg1 = rk1+c1, pg2 = rk2+c2;
            const T cg0 = -rk0+c0, cg1 = -rk1+c1, cg2 = -rk2+c2;
            if (pg0 < (T)0 || pg0 > (T)(N0-1) ||
                pg1 < (T)0 || pg1 > (T)(N1-1) ||
                pg2 < (T)0 || pg2 > (T)(N2_full-1) ||
                cg0 < (T)0 || cg0 > (T)(N0-1) ||
                cg1 < (T)0 || cg1 > (T)(N1-1) ||
                cg2 < (T)0 || cg2 > (T)(N2_full-1))
                conj_opt = false;
        }
    }

    /* Primary scatter */
    if (ORDER == 0) {
        if (conj_opt)
            scatter_nearest<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                        c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        else
            scatter_nearest<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                         c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
    } else {
        if (conj_opt)
            scatter_trilinear<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                          c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        else
            scatter_trilinear<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                           c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
    }

    /* Conjugate scatter for rfft non-boundary pixels.
     * Boundary: k1_idx == 0  or  k1_idx == full_image_w/2 (Nyquist, even W).
     * For non-boundary pixels, scatter conj(value) at rotated(-k0, -k1).
     * For REAL_DATA: conj(real) = real, so conjugate value = same value. */
    if (HALF_IMG) {
        if (k1_idx > 0 && k1_idx * 2 != full_image_w) {
            T crk0, crk1, crk2;
            if (k0_idx == 0 && (image_h & 1) == 0) {
                const T neg_k1 = -k1;
                crk0 = k0 * R[0] + neg_k1 * R[3];
                crk1 = k0 * R[1] + neg_k1 * R[4];
                crk2 = k0 * R[2] + neg_k1 * R[5];
            } else {
                crk0 = -rk0;
                crk1 = -rk1;
                crk2 = -rk2;
            }
            /* For REAL_DATA: conjugate value is val_re (same), no -val_im needed */
            const T conj_im = REAL_DATA ? (T)0 : -val_im;
            if (ORDER == 0) {
                /* conj_opt: skip interior kz (already doubled in primary).
                 * !conj_opt && HALF_VOL: normal scatter (Nyquist row special case).
                 * !HALF_VOL: full-volume scatter (no fold needed). */
                if (conj_opt)
                    scatter_nearest<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_nearest<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_nearest<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                              val_re, conj_im,
                                              c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            } else {
                if (conj_opt)
                    scatter_trilinear<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_trilinear<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_trilinear<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            }
        }
    }
}

/* Local exact path only: duplicate the dense backproject kernel so the
 * original dense entrypoint stays byte-for-byte unchanged. The only semantic
 * difference is that image samples are stored compactly and mapped back to the
 * original flattened image grid through pixel_indices[pix]. */
template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG, bool REAL_DATA = false>
__global__ void __launch_bounds__(BLOCK_SIZE)
backproject_indexed_kernel(
    T*       __restrict__ vol,
    const T* __restrict__ img,
    const int32_t* __restrict__ pixel_indices,
    const T* __restrict__ rot,   /* (n_images, 6) */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    T max_r2,
    int relion_fold_x)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int orig_pix = (int)pixel_indices[pix];

    /* On-the-fly frequency coords — row-major pixel layout. pixel_indices
     * references the original flattened image/half-image grid, while img uses
     * the compact local ordering. */
    const int k0_idx = orig_pix / image_w;   /* row index */
    const int k1_idx = orig_pix % image_w;   /* col index */

    T k0;
    T k0_unscaled = (T)0;
    if (relion_fold_x && HALF_IMG) {
        /* RELION iterates FFTW half-images in native row order:
         * i=0..N/2 are nonnegative y, then i=N/2+1..N-1 are negative y.
         * Do not use RECOVAR's centered row convention in this mode. */
        k0_unscaled = (k0_idx < image_w)
                      ? (T)k0_idx
                      : (T)(k0_idx - image_h);
        k0 = k0_unscaled * upsampling;
    } else {
        k0 = (T)(k0_idx - image_h / 2) * upsampling;
    }
    T k1;
    T k1_unscaled = (T)0;
    if (HALF_IMG) {
        if (relion_fold_x) {
            k1_unscaled = (T)k1_idx;
            k1 = k1_unscaled * upsampling;
        } else {
            k1 = (k1_idx * 2 == full_image_w)
                 ? (T)(-k1_idx) * upsampling
                 : (T)(k1_idx)  * upsampling;
        }
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    if (relion_fold_x && HALF_IMG && HALF_VOL && k1_idx == 0 && k0_idx >= image_w) {
        /* RELION's FFTW half-plane stores x=0 twice: once for positive rows
         * and once for negative rows.  BackProjector::backproject2Dto3D skips
         * the negative-row duplicate. */
        return;
    }

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    T rk0, rk1, rk2;
    if (relion_fold_x && HALF_IMG) {
        /* Match RELION cuda_kernel_backproject3D arithmetic exactly: form
         * matrix-x*source-x first, add matrix-y*source-y, then apply
         * padding_factor. Reversing the addends changes CUDA's contracted FMA
         * and can move exact-integer interpolation coordinates by one ulp. */
        rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (T)upsampling;
        rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (T)upsampling;
        rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (T)upsampling;
    } else {
        rk0 = k0 * R[0] + k1 * R[3];
        rk1 = k0 * R[1] + k1 * R[4];
        rk2 = k0 * R[2] + k1 * R[5];
    }

    if (relion_fold_x && HALF_IMG && HALF_VOL && max_r2 >= (T)0) {
        /* RELION's backproject2Dto3D repeats the radius cutoff after the
         * source pixel has been rotated into 3-D. Mathematically this is
         * redundant for an exactly orthonormal matrix, but at the outer shell
         * it changes inclusion for roundoff-level boundary pixels. */
        const T r2_3d = relion_radius_squared(rk0, rk1, rk2);
        if (r2_3d > max_r2) return;
    }

    T val_re, val_im;
    if (REAL_DATA) {
        val_re = img[img_idx * n_pixels + pix];
        val_im = (T)0;
    } else {
        using V2 = vec2_t<T>;
        V2 px = reinterpret_cast<const V2*>(img)[img_idx * n_pixels + pix];
        val_re = px.x;
        val_im = px.y;
    }

    const bool relion_half_backproject = relion_fold_x && HALF_IMG && HALF_VOL;

    /* RELION's BackProjector iterates an FFTW half-image and stores only one
     * Hermitian half of the 3-D Fourier volume.  It omits duplicated x=0 rows
     * for negative y in the 2-D FFTW layout, folds the stored 3-D half-axis
     * coordinate before trilinear interpolation, and does not emit a separate
     * conjugate rFFT scatter.  RECOVAR's default path remains the adjoint of
     * its half_image_to_full_image expansion; this source-level RELION mode is
     * env-gated while validating M-step parity. */
    if (relion_half_backproject && rk2 < (T)0) {
        rk0 = -rk0;
        rk1 = -rk1;
        rk2 = -rk2;
        if (!REAL_DATA) val_im = -val_im;
    }
    if (relion_fold_x && HALF_IMG && !HALF_VOL && rk2 < (T)0) {
        rk0 = -rk0;
        rk1 = -rk1;
        rk2 = -rk2;
        if (!REAL_DATA) val_im = -val_im;
    }

    if (relion_half_backproject && ORDER == 1 && max_r2 >= (T)0) {
        const int maxR = (int)floor(sqrt((double)max_r2) + 0.5);
        if (relion_compact_trilinear_oob<T>(rk2, rk1, rk0, maxR)) return;
    }

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

    bool conj_opt = HALF_IMG && HALF_VOL && !relion_half_backproject
        && (k1_idx > 0 && k1_idx * 2 != full_image_w)
        && !(k0_idx == 0 && (image_h & 1) == 0);

    if (conj_opt) {
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        if (ORDER == 0) {
            const int pi0 = round_int(rk0+c0), pi1 = round_int(rk1+c1);
            const int pi2 = round_int(rk2+c2);
            const int ci0 = round_int(-rk0+c0), ci1 = round_int(-rk1+c1);
            const int ci2 = round_int(-rk2+c2);
            if ((unsigned)pi0 >= (unsigned)N0 || (unsigned)pi1 >= (unsigned)N1 ||
                (unsigned)pi2 >= (unsigned)N2_full ||
                (unsigned)ci0 >= (unsigned)N0 || (unsigned)ci1 >= (unsigned)N1 ||
                (unsigned)ci2 >= (unsigned)N2_full)
                conj_opt = false;
        } else {
            const T pg0 = rk0+c0, pg1 = rk1+c1, pg2 = rk2+c2;
            const T cg0 = -rk0+c0, cg1 = -rk1+c1, cg2 = -rk2+c2;
            if (pg0 < (T)0 || pg0 > (T)(N0-1) ||
                pg1 < (T)0 || pg1 > (T)(N1-1) ||
                pg2 < (T)0 || pg2 > (T)(N2_full-1) ||
                cg0 < (T)0 || cg0 > (T)(N0-1) ||
                cg1 < (T)0 || cg1 > (T)(N1-1) ||
                cg2 < (T)0 || cg2 > (T)(N2_full-1))
                conj_opt = false;
        }
    }

    if (ORDER == 0) {
        if (conj_opt)
            scatter_nearest<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                        c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        else
            scatter_nearest<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                         c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
    } else {
        if (conj_opt)
            scatter_trilinear<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                          c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        else
            scatter_trilinear<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                           c0, c1, c2, N0, N1, N2_eff, stride0, stride1,
                                           relion_half_backproject);
    }

    if (HALF_IMG && !relion_half_backproject) {
        if (k1_idx > 0 && k1_idx * 2 != full_image_w) {
            T crk0, crk1, crk2;
            if (relion_fold_x && !HALF_VOL) {
                crk0 = -rk0;
                crk1 = -rk1;
                crk2 = -rk2;
            } else if (k0_idx == 0 && (image_h & 1) == 0) {
                const T neg_k1 = -k1;
                crk0 = k0 * R[0] + neg_k1 * R[3];
                crk1 = k0 * R[1] + neg_k1 * R[4];
                crk2 = k0 * R[2] + neg_k1 * R[5];
            } else {
                crk0 = -rk0;
                crk1 = -rk1;
                crk2 = -rk2;
            }
            const T conj_im = REAL_DATA ? (T)0 : -val_im;
            if (ORDER == 0) {
                if (conj_opt)
                    scatter_nearest<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_nearest<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_nearest<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                              val_re, conj_im,
                                              c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            } else {
                if (conj_opt)
                    scatter_trilinear<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_trilinear<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_trilinear<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            }
        }
    }
}

/* Diagnostic companion for the ordinary indexed production kernel above.
 * It receives a strictly increasing subset of source rows and writes unique
 * signature slots without atomics. Coordinate, fold, compact-support, and
 * trilinear expressions deliberately mirror backproject_indexed_kernel and
 * scatter_trilinear<float,true,0,false>, including formation of fractions
 * after adding the integer volume origin. */
__global__ void __launch_bounds__(BLOCK_SIZE)
backproject_indexed_signature_kernel(
    const float2* __restrict__ img,
    const int32_t* __restrict__ pixel_indices,
    const float* __restrict__ rot,
    const int32_t* __restrict__ canonical_rotation_keys,
    const int32_t* __restrict__ signature_row_indices,
    int32_t* __restrict__ signature_rotation_keys,
    int32_t* __restrict__ signature_pixel_indices,
    int32_t* __restrict__ signature_row_flags,
    float* __restrict__ signature_source_values,
    int32_t* __restrict__ signature_neighbor_indices,
    float* __restrict__ signature_neighbor_coefficients,
    int32_t* __restrict__ signature_neighbor_flags,
    int n_signature_rows, int n_source_rows, int n_pixels,
    int image_h, int image_w,
    int N0, int N1, int N2_eff,
    float c0, float c1, float c2,
    int upsampling, float max_r2)
{
    __shared__ float R[6];
    const int output_row = (int)blockIdx.x;
    const int source_row = (int)signature_row_indices[output_row];
    const int pix = (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;
    if ((unsigned)source_row >= (unsigned)n_source_rows) return;
    if (threadIdx.x < 6) R[threadIdx.x] = rot[source_row * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int row_pixel = output_row * n_pixels + pix;
    const int source_row_pixel = source_row * n_pixels + pix;
    const int orig_pix = (int)pixel_indices[pix];
    signature_rotation_keys[row_pixel] = canonical_rotation_keys[source_row];
    signature_pixel_indices[row_pixel] = orig_pix;
    signature_row_flags[row_pixel] = 0;
    #pragma unroll
    for (int value_index = 0; value_index < 5; ++value_index)
        signature_source_values[row_pixel * 5 + value_index] = nanf("");
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int out = row_pixel * 8 + slot;
        signature_neighbor_indices[out] = -1;
        signature_neighbor_coefficients[out] = 0.0f;
        signature_neighbor_flags[out] = 8;
    }

    const int k0_idx = orig_pix / image_w;
    const int k1_idx = orig_pix % image_w;
    const float k0_unscaled = (k0_idx < image_w)
        ? (float)k0_idx
        : (float)(k0_idx - image_h);
    const float k1_unscaled = (float)k1_idx;
    const float k0 = k0_unscaled * (float)upsampling;
    const float k1 = k1_unscaled * (float)upsampling;

    if (k1_idx == 0 && k0_idx >= image_w) {
        signature_row_flags[row_pixel] = 1;
        return;
    }
    if (max_r2 >= 0.0f && k0 * k0 + k1 * k1 > max_r2) {
        signature_row_flags[row_pixel] = 2;
        return;
    }

    const float2 source_value = img[source_row_pixel];
    float val_re = source_value.x;
    float val_im = source_value.y;
    float rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (float)upsampling;
    float rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (float)upsampling;
    float rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (float)upsampling;
    signature_source_values[row_pixel * 5 + 0] = val_re;
    signature_source_values[row_pixel * 5 + 1] = val_im;
    signature_source_values[row_pixel * 5 + 2] = rk0;
    signature_source_values[row_pixel * 5 + 3] = rk1;
    signature_source_values[row_pixel * 5 + 4] = rk2;

    if (max_r2 >= 0.0f && relion_radius_squared(rk0, rk1, rk2) > max_r2) {
        signature_row_flags[row_pixel] = 8;
        return;
    }
    int32_t row_flags = 0;
    if (rk2 < 0.0f) {
        row_flags |= 16;
        rk0 = -rk0;
        rk1 = -rk1;
        rk2 = -rk2;
        val_im = -val_im;
    }
    if (max_r2 >= 0.0f) {
        const int maxR = (int)floor(sqrt((double)max_r2) + 0.5);
        if (relion_compact_trilinear_oob<float>(rk2, rk1, rk0, maxR)) {
            signature_row_flags[row_pixel] = row_flags | 32;
            return;
        }
    }
    signature_row_flags[row_pixel] = row_flags | 64;

    const float g0 = rk0 + c0;
    const float g1 = rk1 + c1;
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
    const float g2_full = rk2 + c2;
    if (g0 < -1.0f || g0 >= (float)N0 ||
        g1 < -1.0f || g1 >= (float)N1 ||
        g2_full < -1.0f || g2_full >= (float)N2_full)
        return;
    const int b0 = floor_int(g0);
    const int b1 = floor_int(g1);
    const int b2 = floor_int(g2_full);
    const float f0 = g0 - (float)b0;
    const float f1 = g1 - (float)b1;
    const float f2 = g2_full - (float)b2;
    const float w0[2] = {1.0f - f0, f0};
    const float w1[2] = {1.0f - f1, f1};
    const float w2[2] = {1.0f - f2, f2};
    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;
    #pragma unroll
    for (int d0 = 0; d0 < 2; ++d0) {
        const int j0 = b0 + d0;
        #pragma unroll
        for (int d1 = 0; d1 < 2; ++d1) {
            const int j1 = b1 + d1;
            const float ww = w0[d0] * w1[d1];
            #pragma unroll
            for (int d2 = 0; d2 < 2; ++d2) {
                const int slot = d0 * 4 + d1 * 2 + d2;
                const int out = row_pixel * 8 + slot;
                const int j2 = b2 + d2;
                if ((unsigned)j0 >= (unsigned)N0 ||
                    (unsigned)j1 >= (unsigned)N1 ||
                    (unsigned)j2 >= (unsigned)N2_full)
                    continue;
                const int kz = j2 - ic2;
                int sj0 = j0;
                int sj1 = j1;
                int hkz;
                int32_t neighbor_flags = 1;
                if (kz >= 0) {
                    hkz = kz;
                } else if ((N2_full & 1) == 0 && -kz == ic2) {
                    hkz = ic2;
                    neighbor_flags |= 4;
                } else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                    neighbor_flags |= 2;
                }
                if (hkz > ic2) continue;
                signature_neighbor_indices[out] = sj0 * stride0 + sj1 * stride1 + hkz;
                signature_neighbor_coefficients[out] = ww * w2[d2];
                signature_neighbor_flags[out] = neighbor_flags;
            }
        }
    }
}

/* Batched indexed backprojection: same semantics as
 * backproject_indexed_kernel, but scatter a small batch of images into
 * matching independent volumes while reusing pixel coordinates and rotations.
 */
template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG, bool REAL_DATA = false,
          bool RELION_BLOCK_TOPOLOGY = false>
__global__ void __launch_bounds__(BLOCK_SIZE)
batch_backproject_indexed_kernel(
    T*       __restrict__ vols,
    const T* __restrict__ imgs,
    const int32_t* __restrict__ pixel_indices,
    const T* __restrict__ rot,   /* (n_images, 6) */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    int vol_stride,
    int n_images,
    int batch_size,
    T max_r2,
    int relion_fold_x)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix_start = RELION_BLOCK_TOPOLOGY
        ? (int)threadIdx.x
        : (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    const int pixel_stride = RELION_BLOCK_TOPOLOGY ? 128 : n_pixels;
    for (int pix = pix_start; pix < n_pixels; pix += pixel_stride) {

    const int orig_pix = (int)pixel_indices[pix];
    const int k0_idx = orig_pix / image_w;
    const int k1_idx = orig_pix % image_w;

    T k0;
    T k0_unscaled = (T)0;
    if (relion_fold_x && HALF_IMG) {
        k0_unscaled = (k0_idx < image_w)
                      ? (T)k0_idx
                      : (T)(k0_idx - image_h);
        k0 = k0_unscaled * upsampling;
    } else {
        k0 = (T)(k0_idx - image_h / 2) * upsampling;
    }
    T k1;
    T k1_unscaled = (T)0;
    if (HALF_IMG) {
        if (relion_fold_x) {
            k1_unscaled = (T)k1_idx;
            k1 = k1_unscaled * upsampling;
        } else {
            k1 = (k1_idx * 2 == full_image_w)
                 ? (T)(-k1_idx) * upsampling
                 : (T)(k1_idx)  * upsampling;
        }
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    if (relion_fold_x && HALF_IMG && HALF_VOL && k1_idx == 0 && k0_idx >= image_w) {
        /* RELION's FFTW half-plane stores x=0 twice: once for positive rows
         * and once for negative rows.  BackProjector::backproject2Dto3D skips
         * the negative-row duplicate. */
        continue;
    }

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) continue;

    T rk0, rk1, rk2;
    if (relion_fold_x && HALF_IMG) {
        /* RELION forms matrix-x*source-x before matrix-y*source-y, then pads. */
        rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (T)upsampling;
        rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (T)upsampling;
        rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (T)upsampling;
    } else {
        rk0 = k0 * R[0] + k1 * R[3];
        rk1 = k0 * R[1] + k1 * R[4];
        rk2 = k0 * R[2] + k1 * R[5];
    }

    if (relion_fold_x && HALF_IMG && HALF_VOL && max_r2 >= (T)0) {
        const T r2_3d = relion_radius_squared(rk0, rk1, rk2);
        if (r2_3d > max_r2) continue;
    }

    const bool relion_half_backproject = relion_fold_x && HALF_IMG && HALF_VOL;
    const bool fold_full_negative_z = relion_fold_x && HALF_IMG && !HALF_VOL && rk2 < (T)0;
    const bool fold_half_negative_z = relion_half_backproject && rk2 < (T)0;
    if (fold_half_negative_z || fold_full_negative_z) {
        rk0 = -rk0;
        rk1 = -rk1;
        rk2 = -rk2;
    }

    if (relion_half_backproject && ORDER == 1 && max_r2 >= (T)0) {
        const int maxR = (int)floor(sqrt((double)max_r2) + 0.5);
        if (relion_compact_trilinear_oob<T>(rk2, rk1, rk0, maxR)) continue;
    }

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;
    const int img_stride = n_images * n_pixels;
    const int vol_bytes_stride = REAL_DATA ? vol_stride : vol_stride * 2;

    bool conj_opt = HALF_IMG && HALF_VOL && !relion_half_backproject
        && (k1_idx > 0 && k1_idx * 2 != full_image_w)
        && !(k0_idx == 0 && (image_h & 1) == 0);

    if (conj_opt) {
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        if (ORDER == 0) {
            const int pi0 = round_int(rk0+c0), pi1 = round_int(rk1+c1);
            const int pi2 = round_int(rk2+c2);
            const int ci0 = round_int(-rk0+c0), ci1 = round_int(-rk1+c1);
            const int ci2 = round_int(-rk2+c2);
            if ((unsigned)pi0 >= (unsigned)N0 || (unsigned)pi1 >= (unsigned)N1 ||
                (unsigned)pi2 >= (unsigned)N2_full ||
                (unsigned)ci0 >= (unsigned)N0 || (unsigned)ci1 >= (unsigned)N1 ||
                (unsigned)ci2 >= (unsigned)N2_full)
                conj_opt = false;
        } else {
            const T pg0 = rk0+c0, pg1 = rk1+c1, pg2 = rk2+c2;
            const T cg0 = -rk0+c0, cg1 = -rk1+c1, cg2 = -rk2+c2;
            if (pg0 < (T)0 || pg0 > (T)(N0-1) ||
                pg1 < (T)0 || pg1 > (T)(N1-1) ||
                pg2 < (T)0 || pg2 > (T)(N2_full-1) ||
                cg0 < (T)0 || cg0 > (T)(N0-1) ||
                cg1 < (T)0 || cg1 > (T)(N1-1) ||
                cg2 < (T)0 || cg2 > (T)(N2_full-1))
                conj_opt = false;
        }
    }

    for (int b = 0; b < batch_size; b++) {
        T* vol = vols + b * vol_bytes_stride;

        T val_re, val_im;
        if (REAL_DATA) {
            val_re = imgs[(b * img_stride) + img_idx * n_pixels + pix];
            val_im = (T)0;
        } else {
            using V2 = vec2_t<T>;
            V2 px = reinterpret_cast<const V2*>(imgs)[(b * img_stride) + img_idx * n_pixels + pix];
            val_re = px.x;
            val_im = (fold_half_negative_z || fold_full_negative_z) ? -px.y : px.y;
        }
        if (RELION_BLOCK_TOPOLOGY && val_re == (T)0 && (REAL_DATA || val_im == (T)0))
            continue;

        if (ORDER == 0) {
            if (conj_opt)
                scatter_nearest<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                            c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            else
                scatter_nearest<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                             c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        } else {
            if (conj_opt)
                scatter_trilinear<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                              c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            else
                scatter_trilinear<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                               c0, c1, c2, N0, N1, N2_eff, stride0, stride1,
                                               relion_half_backproject);
        }

        if (HALF_IMG && !relion_half_backproject) {
            if (k1_idx > 0 && k1_idx * 2 != full_image_w) {
                T crk0, crk1, crk2;
                if (relion_fold_x && !HALF_VOL) {
                    crk0 = -rk0;
                    crk1 = -rk1;
                    crk2 = -rk2;
                } else if (k0_idx == 0 && (image_h & 1) == 0) {
                    const T neg_k1 = -k1;
                    crk0 = k0 * R[0] + neg_k1 * R[3];
                    crk1 = k0 * R[1] + neg_k1 * R[4];
                    crk2 = k0 * R[2] + neg_k1 * R[5];
                } else {
                    crk0 = -rk0;
                    crk1 = -rk1;
                    crk2 = -rk2;
                }
                const T conj_im = REAL_DATA ? (T)0 : -val_im;
                if (ORDER == 0) {
                    if (conj_opt)
                        scatter_nearest<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                    val_re, conj_im,
                                                    c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                    else if (HALF_VOL)
                        scatter_nearest<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                    val_re, conj_im,
                                                    c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                    else
                        scatter_nearest<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                } else {
                    if (conj_opt)
                        scatter_trilinear<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                      val_re, conj_im,
                                                      c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                    else if (HALF_VOL)
                        scatter_trilinear<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                      val_re, conj_im,
                                                      c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                    else
                        scatter_trilinear<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                    val_re, conj_im,
                                                    c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                }
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

/* ================================================================== */
/*                    Project kernel                                   */
/* ================================================================== */

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG, bool INDEXED>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_kernel(
    const T* __restrict__ vol,
    T*       __restrict__ img,
    const T* __restrict__ rot,
    const int32_t* __restrict__ pixel_indices,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* Row-major pixel layout */
    const int orig_pix = INDEXED ? (int)pixel_indices[pix] : pix;
    const int k0_idx = orig_pix / image_w;   /* row index */
    const int k1_idx = orig_pix % image_w;   /* col index */
    T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    using V2 = vec2_t<T>;
    V2* img2 = reinterpret_cast<V2*>(img);
    const int img_off = img_idx * n_pixels + pix;

    /* Pre-rotation disk check: rotation preserves ||k||. */
    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) {
        img2[img_off] = make_v2((T)0, (T)0);
        return;
    }

    T rk0 = k0 * R[0] + k1 * R[3];
    T rk1 = k0 * R[1] + k1 * R[4];
    T rk2 = k0 * R[2] + k1 * R[5];

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

    /* ── HALF_VOL: per-neighbor Hermitian read from half-volume ──────
     *
     * Use the full centered-volume coordinate system for bounds checks
     * (matching full-volume behavior).  For each trilinear neighbor,
     * convert the centered z index to half-volume kz.  Neighbors with
     * kz >= 0 read directly from the half-volume; neighbors with kz < 0
     * read the Hermitian partner at (-kx, -ky, -kz) and conjugate.
     */
    if (HALF_VOL) {
        const T g0 = rk0 + c0;
        const T g1 = rk1 + c1;
        /* Recover the actual full z dimension so odd cubic RELION grids keep
         * their final centered plane and even rectangular grids keep N2. */
        const int ic2 = (int)c2;          /* N2/2 */
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        const T g2_full = rk2 + c2;

        if (ORDER == 0) {
            const int i0 = round_int(g0);
            const int i1 = round_int(g1);
            const int i2 = round_int(g2_full);
            if ((unsigned)i0 >= (unsigned)N0 ||
                (unsigned)i1 >= (unsigned)N1 ||
                (unsigned)i2 >= (unsigned)N2_full) {
                img2[img_off] = make_v2((T)0, (T)0);
                return;
            }
            /* Convert centered index i2 to half-volume kz */
            const int kz = i2 - ic2;
            int ri, rj, rk;
            bool cj = false;
            if (kz >= 0) {
                ri = i0; rj = i1; rk = kz;
            } else {
                /* Hermitian partner: partner(j) = (N - (N & 1) - j) % N */
                ri = (N0 - (N0 & 1) - i0) % N0;
                rj = (N1 - (N1 & 1) - i1) % N1;
                rk = -kz;
                cj = true;
            }
            const int off = ri * stride0 + rj * stride1 + rk;
            V2 v = __ldg(&reinterpret_cast<const V2*>(vol)[off]);
            if (cj) v.y = -v.y;
            img2[img_off] = v;
            return;
        }

        /* ──── cubic HALF_VOL (ORDER==3, periodic wrap) ──── */
        if (ORDER == 3) {
            /* Periodic cubic: g = rk + c - 1 (the -1 shift for periodic convention).
             * All indices wrap periodically, so no OOB checks needed. */
            const T cg0 = rk0 + c0 - (T)1;
            const T cg1 = rk1 + c1 - (T)1;
            const T cg2_full = rk2 + c2 - (T)1;

            const int cb0 = floor_int(cg0);
            const int cb1 = floor_int(cg1);
            const int cb2 = floor_int(cg2_full);
            const T cf0 = cg0 - (T)cb0;
            const T cf1 = cg1 - (T)cb1;
            const T cf2 = cg2_full - (T)cb2;

            T sum_re = 0, sum_im = 0;
            const V2* vol2 = reinterpret_cast<const V2*>(vol);

            for (int d0 = 0; d0 < 4; d0++) {
                const int j0 = wrap_mod(cb0 + d0, N0);
                const T bw0 = cubic_basis(cf0 - (T)d0 + (T)1);
                for (int d1 = 0; d1 < 4; d1++) {
                    const int j1 = wrap_mod(cb1 + d1, N1);
                    const T bw01 = bw0 * cubic_basis(cf1 - (T)d1 + (T)1);
                    for (int d2 = 0; d2 < 4; d2++) {
                        const int j2_full = wrap_mod(cb2 + d2, N2_full);
                        const T w = bw01 * cubic_basis(cf2 - (T)d2 + (T)1);
                        const int kz = j2_full - ic2;
                        int ri = j0, rj = j1;
                        int hkz;
                        bool cj = false;
                        if (kz >= 0) {
                            hkz = kz;
                        } else if ((N2_full & 1) == 0 && -kz == ic2) {
                            /* Nyquist: self-conjugate */
                            hkz = ic2;
                        } else {
                            ri = (N0 - (N0 & 1) - j0) % N0;
                            rj = (N1 - (N1 & 1) - j1) % N1;
                            hkz = -kz;
                            cj = true;
                        }
                        if (hkz <= ic2) {
                            const int off = ri * stride0 + rj * stride1 + hkz;
                            V2 v = __ldg(&vol2[off]);
                            if (cj) v.y = -v.y;
                            sum_re += w * v.x;
                            sum_im += w * v.y;
                        }
                    }
                }
            }
            img2[img_off] = make_v2(sum_re, sum_im);
            return;
        }

        /* ──── trilinear HALF_VOL ──── */
        if (g0 < (T)-1 || g0 >= (T)N0 ||
            g1 < (T)-1 || g1 >= (T)N1 ||
            g2_full < (T)-1 || g2_full >= (T)N2_full) {
            img2[img_off] = make_v2((T)0, (T)0);
            return;
        }

        const int b0 = floor_int(g0);
        const int b1 = floor_int(g1);
        const int b2 = floor_int(g2_full);
        const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2_full - (T)b2;
        const T w0[2] = {(T)1 - f0, f0};
        const T w1[2] = {(T)1 - f1, f1};
        const T w2[2] = {(T)1 - f2, f2};

        T sum_re = 0, sum_im = 0;
        const V2* vol2 = reinterpret_cast<const V2*>(vol);

        const bool all_in = (b0 >= 0 && b0 + 1 < N0 &&
                             b1 >= 0 && b1 + 1 < N1 &&
                             b2 >= 0 && b2 + 1 < N2_full);

        if (all_in && b2 >= ic2) {
            /* Fast path: all in-bounds, all kz >= 0 — direct reads.
             * Prefetch all 8 neighbors so the compiler pipelines loads. */
            const int kz0 = b2 - ic2;
            const V2 v000 = __ldg(&vol2[b0*stride0 + b1*stride1 + kz0]);
            const V2 v001 = __ldg(&vol2[b0*stride0 + b1*stride1 + kz0 + 1]);
            const V2 v010 = __ldg(&vol2[b0*stride0 + (b1+1)*stride1 + kz0]);
            const V2 v011 = __ldg(&vol2[b0*stride0 + (b1+1)*stride1 + kz0 + 1]);
            const V2 v100 = __ldg(&vol2[(b0+1)*stride0 + b1*stride1 + kz0]);
            const V2 v101 = __ldg(&vol2[(b0+1)*stride0 + b1*stride1 + kz0 + 1]);
            const V2 v110 = __ldg(&vol2[(b0+1)*stride0 + (b1+1)*stride1 + kz0]);
            const V2 v111 = __ldg(&vol2[(b0+1)*stride0 + (b1+1)*stride1 + kz0 + 1]);
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    const T ww = w0[d0] * w1[d1];
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const T w = ww * w2[d2];
                        const V2& v = (d0 == 0)
                            ? ((d1 == 0) ? (d2 == 0 ? v000 : v001) : (d2 == 0 ? v010 : v011))
                            : ((d1 == 0) ? (d2 == 0 ? v100 : v101) : (d2 == 0 ? v110 : v111));
                        sum_re += w * v.x;
                        sum_im += w * v.y;
                    }
                }
            }
        } else if (all_in && b2 + 1 < ic2) {
            /* Fast path: all in-bounds, all kz < 0 — Hermitian partner reads.
             * Since weights are real, conj(Σ w·v) = Σ w·conj(v),
             * so we sum normally then negate imaginary. */
            /* partner(j) = (N - (N & 1) - j) % N */
            const int r0_0 = (N0 - (N0 & 1) - b0) % N0,     r0_1 = (N0 - (N0 & 1) - b0 - 1) % N0;
            const int r1_0 = (N1 - (N1 & 1) - b1) % N1,     r1_1 = (N1 - (N1 & 1) - b1 - 1) % N1;
            const int rk0  = ic2 - b2,            rk1  = rk0 - 1;
            const V2 v000 = __ldg(&vol2[r0_0*stride0 + r1_0*stride1 + rk0]);
            const V2 v001 = __ldg(&vol2[r0_0*stride0 + r1_0*stride1 + rk1]);
            const V2 v010 = __ldg(&vol2[r0_0*stride0 + r1_1*stride1 + rk0]);
            const V2 v011 = __ldg(&vol2[r0_0*stride0 + r1_1*stride1 + rk1]);
            const V2 v100 = __ldg(&vol2[r0_1*stride0 + r1_0*stride1 + rk0]);
            const V2 v101 = __ldg(&vol2[r0_1*stride0 + r1_0*stride1 + rk1]);
            const V2 v110 = __ldg(&vol2[r0_1*stride0 + r1_1*stride1 + rk0]);
            const V2 v111 = __ldg(&vol2[r0_1*stride0 + r1_1*stride1 + rk1]);
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    const T ww = w0[d0] * w1[d1];
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const T w = ww * w2[d2];
                        const V2& v = (d0 == 0)
                            ? ((d1 == 0) ? (d2 == 0 ? v000 : v001) : (d2 == 0 ? v010 : v011))
                            : ((d1 == 0) ? (d2 == 0 ? v100 : v101) : (d2 == 0 ? v110 : v111));
                        sum_re += w * v.x;
                        sum_im += w * v.y;
                    }
                }
            }
            sum_im = -sum_im;  /* conjugate the result */
        } else {
            /* Slow path: boundary or mixed kz (b2 = ic2-1) */
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                const int j0 = b0 + d0;
                if ((unsigned)j0 >= (unsigned)N0) continue;
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    const int j1 = b1 + d1;
                    if ((unsigned)j1 >= (unsigned)N1) continue;
                    const T ww = w0[d0] * w1[d1];
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const int j2 = b2 + d2;
                        if ((unsigned)j2 >= (unsigned)N2_full) continue;
                        const int kz = j2 - ic2;
                        const T w = ww * w2[d2];
                        int ri, rj, rk;
                        bool cj = false;
                        if (kz >= 0) {
                            ri = j0; rj = j1; rk = kz;
                        } else {
                            /* partner(j) = (N - (N & 1) - j) % N */
                            ri = (N0 - (N0 & 1) - j0) % N0;
                            rj = (N1 - (N1 & 1) - j1) % N1;
                            rk = -kz;
                            cj = true;
                        }
                        const int off = ri * stride0 + rj * stride1 + rk;
                        V2 v = __ldg(&vol2[off]);
                        if (cj) v.y = -v.y;
                        sum_re += w * v.x;
                        sum_im += w * v.y;
                    }
                }
            }
        }
        img2[img_off] = make_v2(sum_re, sum_im);
        return;
    }

    /* ── Non-HALF_VOL path (unchanged) ───────────────────────────── */
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;
    const T g2 = rk2 + c2;

    if (ORDER == 0) {
        const int i0 = round_int(g0);
        const int i1 = round_int(g1);
        const int i2 = round_int(g2);
        if ((unsigned)i0 >= (unsigned)N0 ||
            (unsigned)i1 >= (unsigned)N1 ||
            (unsigned)i2 >= (unsigned)N2_eff) {
            img2[img_off] = make_v2((T)0, (T)0);
            return;
        }
        const int off = i0 * stride0 + i1 * stride1 + i2;
        V2 v = __ldg(&reinterpret_cast<const V2*>(vol)[off]);
        img2[img_off] = v;
        return;
    }

    /* ──── cubic (full volume, ORDER==3, periodic wrap) ──── */
    if (ORDER == 3) {
        /* Periodic cubic: g = rk + c - 1 (the -1 shift for periodic convention).
         * All indices wrap periodically, so no OOB checks needed. */
        const T cg0 = rk0 + c0 - (T)1;
        const T cg1 = rk1 + c1 - (T)1;
        const T cg2 = rk2 + c2 - (T)1;

        const int cb0 = floor_int(cg0);
        const int cb1 = floor_int(cg1);
        const int cb2 = floor_int(cg2);
        const T cf0 = cg0 - (T)cb0;
        const T cf1 = cg1 - (T)cb1;
        const T cf2 = cg2 - (T)cb2;

        T sum_re = 0, sum_im = 0;
        const V2* vol2 = reinterpret_cast<const V2*>(vol);

        for (int d0 = 0; d0 < 4; d0++) {
            const int j0 = wrap_mod(cb0 + d0, N0);
            const T bw0 = cubic_basis(cf0 - (T)d0 + (T)1);
            for (int d1 = 0; d1 < 4; d1++) {
                const int j1 = wrap_mod(cb1 + d1, N1);
                const T bw01 = bw0 * cubic_basis(cf1 - (T)d1 + (T)1);
                for (int d2 = 0; d2 < 4; d2++) {
                    const int j2 = wrap_mod(cb2 + d2, N2_eff);
                    const T w = bw01 * cubic_basis(cf2 - (T)d2 + (T)1);
                    const int off = j0 * stride0 + j1 * stride1 + j2;
                    V2 v = __ldg(&vol2[off]);
                    sum_re += w * v.x;
                    sum_im += w * v.y;
                }
            }
        }
        img2[img_off] = make_v2(sum_re, sum_im);
        return;
    }

    /* ──── trilinear (full volume) ──── */
    if (g0 < (T)-1 || g0 >= (T)N0 ||
        g1 < (T)-1 || g1 >= (T)N1 ||
        g2 < (T)-1 || g2 >= (T)N2_eff) {
        img2[img_off] = make_v2((T)0, (T)0);
        return;
    }

    const int b0 = floor_int(g0);
    const int b1 = floor_int(g1);
    const int b2 = floor_int(g2);
    const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2 - (T)b2;
    const T w0[2] = {(T)1 - f0, f0};
    const T w1[2] = {(T)1 - f1, f1};
    const T w2[2] = {(T)1 - f2, f2};

    T sum_re = 0, sum_im = 0;

    /* Fast path: all 8 neighbors in-bounds (true for ~95% of pixels). */
    if (b0 >= 0 && b0 + 1 < N0 &&
        b1 >= 0 && b1 + 1 < N1 &&
        b2 >= 0 && b2 + 1 < N2_eff) {
        const V2* vol2 = reinterpret_cast<const V2*>(vol);
        /* Prefetch all 8 neighbors — compiler can pipeline the loads. */
        const V2 v000 = __ldg(&vol2[b0 * stride0 + b1 * stride1 + b2]);
        const V2 v001 = __ldg(&vol2[b0 * stride0 + b1 * stride1 + b2 + 1]);
        const V2 v010 = __ldg(&vol2[b0 * stride0 + (b1+1) * stride1 + b2]);
        const V2 v011 = __ldg(&vol2[b0 * stride0 + (b1+1) * stride1 + b2 + 1]);
        const V2 v100 = __ldg(&vol2[(b0+1) * stride0 + b1 * stride1 + b2]);
        const V2 v101 = __ldg(&vol2[(b0+1) * stride0 + b1 * stride1 + b2 + 1]);
        const V2 v110 = __ldg(&vol2[(b0+1) * stride0 + (b1+1) * stride1 + b2]);
        const V2 v111 = __ldg(&vol2[(b0+1) * stride0 + (b1+1) * stride1 + b2 + 1]);
        /* Trilinear combination */
        #pragma unroll
        for (int d0 = 0; d0 < 2; d0++) {
            #pragma unroll
            for (int d1 = 0; d1 < 2; d1++) {
                const T ww = w0[d0] * w1[d1];
                #pragma unroll
                for (int d2 = 0; d2 < 2; d2++) {
                    const T w = ww * w2[d2];
                    const V2& v = (d0 == 0)
                        ? ((d1 == 0) ? (d2 == 0 ? v000 : v001) : (d2 == 0 ? v010 : v011))
                        : ((d1 == 0) ? (d2 == 0 ? v100 : v101) : (d2 == 0 ? v110 : v111));
                    sum_re += w * v.x;
                    sum_im += w * v.y;
                }
            }
        }
    } else {
        /* Boundary path: check each neighbor. */
        #pragma unroll
        for (int d0 = 0; d0 < 2; d0++) {
            const int j0 = b0 + d0;
            if ((unsigned)j0 >= (unsigned)N0) continue;
            #pragma unroll
            for (int d1 = 0; d1 < 2; d1++) {
                const int j1 = b1 + d1;
                if ((unsigned)j1 >= (unsigned)N1) continue;
                const T ww = w0[d0] * w1[d1];
                #pragma unroll
                for (int d2 = 0; d2 < 2; d2++) {
                    const int j2 = b2 + d2;
                    if ((unsigned)j2 >= (unsigned)N2_eff) continue;
                    const T w = ww * w2[d2];
                    const int off = j0 * stride0 + j1 * stride1 + j2;
                    V2 v = __ldg(&reinterpret_cast<const V2*>(vol)[off]);
                    sum_re += w * v.x;
                    sum_im += w * v.y;
                }
            }
        }
    }

    img2[img_off] = make_v2(sum_re, sum_im);
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

template <typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_texture_compact_kernel(
    const T* __restrict__ vol,
    float* __restrict__ real,
    float* __restrict__ imag,
    int texX, int texY, int texZ,
    int yinit, int zinit,
    int N0, int N1, int N2)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int n = texX * texY * texZ;
    if (idx >= n) return;

    const int x = idx % texX;
    const int yidx = (idx / texX) % texY;
    const int zidx = idx / (texX * texY);
    const int y = yidx + yinit;
    const int z = zidx + zinit;

    const int i0 = N0 / 2 + x;
    const int i1 = N1 / 2 + y;
    const int i2 = N2 / 2 + z;

    float re = 0.0f;
    float im = 0.0f;
    if ((unsigned)i0 < (unsigned)N0 && (unsigned)i1 < (unsigned)N1 && (unsigned)i2 < (unsigned)N2) {
        using V2 = vec2_t<T>;
        const V2 v = reinterpret_cast<const V2*>(vol)[i0 * N1 * N2 + i1 * N2 + i2];
        re = (float)v.x;
        im = (float)v.y;
    }
    real[idx] = re;
    imag[idx] = im;
}

/* Physical half storage is [z,y,x>=0]. Stage the logical texture at its
 * original origin, retaining ghost texels and the old sampling coordinates. */
__global__ void __launch_bounds__(BLOCK_SIZE)
fill_relion_texture_capacity_kernel(
    const float2* __restrict__ vol, float* real, float* imag,
    const int32_t* logical_radius, int upsampling,
    int texX, int texY, int texZ,
    int storageX = 0, int storageY = 0, int storageZ = 0)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= texX * texY * texZ) return;
    const int x = idx % texX;
    const int y = (idx / texX) % texY;
    const int z = idx / (texX * texY);
    const int sourceX = storageX > 0 ? storageX : texX;
    const int sourceY = storageY > 0 ? storageY : texY;
    const int sourceZ = storageZ > 0 ? storageZ : texZ;
    const int radius = *logical_radius;
    float2 value = make_float2(0.0f, 0.0f);
    if (radius >= 0 && radius <= (sourceY / 2 - 1) / upsampling) {
        const int R = radius * upsampling;
        if (x < R + 2 && y < 2 * R + 3 && z < 2 * R + 3) {
            const int iy = sourceY / 2 + y - (R + 1);
            const int iz = sourceZ / 2 + z - (R + 1);
            value = vol[(iz * sourceY + iy) * sourceX + x];
        }
    }
    real[idx] = value.x;
    imag[idx] = value.y;
}

template <bool HALF_IMG, bool RUNTIME_RADIUS = false>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_texture_kernel(
    cudaTextureObject_t texReal,
    cudaTextureObject_t texImag,
    float* __restrict__ img,
    const float* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int tex_yinit, int tex_zinit,
    int upsampling, int full_image_w,
    int maxR2_padded,
    const int32_t* logical_radius = nullptr, int capacity_radius = 0,
    const int32_t* image_radius = nullptr)
{
    __shared__ float R[6];

    const int img_idx = blockIdx.x;
    const int pix = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    /* RELION keeps both even-box Nyquist axes positive in its accelerated
     * half-spectrum projector.  This differs intentionally from RECOVAR's
     * generic centered-grid convention used by the non-texture kernels. */
    const float k0_unscaled = (float)(
        k0_idx == 0 ? image_h / 2 : k0_idx - image_h / 2);
    float k1_unscaled;
    if (HALF_IMG) {
        k1_unscaled = (float)k1_idx;
    } else {
        k1_unscaled = (float)(k1_idx - image_w / 2);
    }

    float2* img2 = reinterpret_cast<float2*>(img);
    const int img_off = img_idx * n_pixels + pix;
    if constexpr (RUNTIME_RADIUS) {
        const int radius = *logical_radius;
        if (radius < 0 || radius > capacity_radius / upsampling) {
            img2[img_off] = make_float2(nanf(""), nanf(""));
            return;
        }
        const int padded_radius = radius * upsampling;
        maxR2_padded = padded_radius * padded_radius;
        tex_yinit = -(padded_radius + 1);
        tex_zinit = -(padded_radius + 1);
    }

    if (image_radius != nullptr) {
        const int radius = *image_radius;
        if (radius < 0 || radius > image_h / 2) {
            img2[img_off] = make_float2(nanf(""), nanf(""));
            return;
        }
        const int padded_radius = radius * upsampling;
        // AccProjectorKernel::makeKernel clips at the smaller image/model
        // radius. Texture staging still uses the original model extent.
        maxR2_padded = min(maxR2_padded, padded_radius * padded_radius);
    }

    /* Match RELION AccProjectorKernel source order exactly under RECOVAR's
     * compact row-swapped R mapping: matrix-x*source-x is the first addend.
     * Reversing the addends changes CUDA's contracted FMA association and can
     * cross a texture interpolation fraction-bin boundary. */
    const float rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (float)upsampling;
    const float rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (float)upsampling;
    const float rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (float)upsampling;

    if ((int)(rk0 * rk0 + rk1 * rk1 + rk2 * rk2) > maxR2_padded) {
        img2[img_off] = make_float2(0.0f, 0.0f);
        return;
    }

    float xp = rk0;
    float yp = rk1;
    float zp = rk2;
    float imag_sign = 1.0f;
    if (xp < 0.0f) {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag_sign = -1.0f;
    }

    /* Stage and sample the same compact half-Fourier texture layout as
     * RELION: texture x is nonnegative model-x, y/z start at mdlInitY/Z. */
    const float re = tex3D<float>(texReal, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
    const float im = imag_sign * tex3D<float>(texImag, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
    img2[img_off] = make_float2(re, im);
}

/* Match RELION's Wavg A2 topology: one block per orientation, one thread per
 * pixel lane, and a float32 atomic add over orientations. The input has
 * already completed the per-translation accumulation for each orientation. */
__global__ void __launch_bounds__(256)
relion_wavg_rotation_atomic_f32_kernel(
    const float* __restrict__ terms,
    float* __restrict__ output,
    int n_rotations,
    int n_pixels)
{
    const int rotation = blockIdx.x;
    const int batch = blockIdx.y;
    for (int pixel = threadIdx.x; pixel < n_pixels; pixel += blockDim.x) {
        const int64_t input_index =
            (static_cast<int64_t>(batch) * n_rotations + rotation) * n_pixels + pixel;
        atomicAdd(&output[static_cast<int64_t>(batch) * n_pixels + pixel], terms[input_index]);
    }
}

cudaError_t launch_relion_wavg_rotation_atomic_f32(
    cudaStream_t stream,
    const float* terms,
    float* output,
    int64_t batch_size,
    int64_t n_rotations,
    int64_t n_pixels)
{
    cudaError_t err = cudaMemsetAsync(
        output,
        0,
        static_cast<size_t>(batch_size * n_pixels) * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    dim3 grid(static_cast<unsigned>(n_rotations), static_cast<unsigned>(batch_size));
    dim3 block(256);
    relion_wavg_rotation_atomic_f32_kernel<<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(n_pixels));
    return cudaGetLastError();
}

cudaError_t launch_relion_wavg_rotation_atomic_add_f32(
    cudaStream_t stream,
    const float* terms,
    float* output,
    int64_t batch_size,
    int64_t n_rotations,
    int64_t n_pixels)
{
    dim3 grid(static_cast<unsigned>(n_rotations), static_cast<unsigned>(batch_size));
    dim3 block(256);
    relion_wavg_rotation_atomic_f32_kernel<<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(n_pixels));
    return cudaGetLastError();
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

__global__ void __launch_bounds__(256)
relion_wavg_rotation_atomic_runtime_triplet_f32_kernel(
    const float* __restrict__ terms,
    float* __restrict__ output,
    int n_rotations,
    int pixel_capacity,
    const int32_t* __restrict__ runtime_logical_pixel_count)
{
    const int rotation = blockIdx.x;
    const int batch = blockIdx.y;
    const int logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : runtime_logical_pixel_count[0];
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (rotation == 0 && batch == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    for (int pixel = threadIdx.x;
         pixel < logical_pixel_count;
         pixel += blockDim.x) {
        const int64_t input_index =
            ((static_cast<int64_t>(batch) * n_rotations + rotation) *
                 pixel_capacity + pixel) * 3;
        const int64_t output_index =
            (static_cast<int64_t>(batch) * pixel_capacity + pixel) * 3;
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
    relion_wavg_rotation_atomic_runtime_triplet_f32_kernel<<<grid, block, 0, stream>>>(
        terms,
        output,
        static_cast<int>(n_rotations),
        static_cast<int>(pixel_capacity),
        runtime_logical_pixel_count);
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

template <bool INDEXED_RECTANGLE = false>
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
    const int32_t* __restrict__ invalid = nullptr)
{
    const int64_t rotation = blockIdx.x;
    const int64_t batch = blockIdx.y;
    if (batch >= batch_size || rotation >= rotation_count) return;
    if constexpr (INDEXED_RECTANGLE) { if (*invalid) return; }
    const int64_t logical_pixel_count = runtime_logical_pixel_count == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixel_count[0]);
    if (logical_pixel_count < 0 || logical_pixel_count > pixel_capacity) {
        if (rotation == 0 && batch == 0 && threadIdx.x == 0)
            output[0] = nanf("");
        return;
    }
    const float batch_scale = scale[batch];
    const int64_t posterior_base =
        (batch * rotation_count + rotation) * translation_count;
    const int64_t shifted_stride = INDEXED_RECTANGLE ? rectangle_capacity : pixel_capacity;
    const int64_t shifted_base = batch * translation_count * shifted_stride;
    const int64_t projection_base =
        (batch * rotation_count + rotation) * pixel_capacity;
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

template <bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_texture_double_kernel(
    cudaTextureObject_t texReal,
    cudaTextureObject_t texImag,
    double* __restrict__ img,
    const double* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int tex_yinit, int tex_zinit,
    int upsampling, int full_image_w,
    int maxR2_padded)
{
    __shared__ float R[6];

    const int img_idx = blockIdx.x;
    const int pix = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = (float)rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    /* Keep the coordinate convention identical to the C64 texture path. */
    const float k0_unscaled = (float)(
        k0_idx == 0 ? image_h / 2 : k0_idx - image_h / 2);
    float k1_unscaled;
    if (HALF_IMG) {
        k1_unscaled = (float)k1_idx;
    } else {
        k1_unscaled = (float)(k1_idx - image_w / 2);
    }

    double2* img2 = reinterpret_cast<double2*>(img);
    const int img_off = img_idx * n_pixels + pix;

    /* Keep the exact RELION source operand order used by the C64 path. */
    const float rk0 = (R[3] * k1_unscaled + R[0] * k0_unscaled) * (float)upsampling;
    const float rk1 = (R[4] * k1_unscaled + R[1] * k0_unscaled) * (float)upsampling;
    const float rk2 = (R[5] * k1_unscaled + R[2] * k0_unscaled) * (float)upsampling;

    if ((int)(rk0 * rk0 + rk1 * rk1 + rk2 * rk2) > maxR2_padded) {
        img2[img_off] = make_double2(0.0, 0.0);
        return;
    }

    float xp = rk0;
    float yp = rk1;
    float zp = rk2;
    float imag_sign = 1.0f;
    if (xp < 0.0f) {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag_sign = -1.0f;
    }

    const float re = tex3D<float>(texReal, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
    const float im = imag_sign * tex3D<float>(texImag, xp + 0.5f, yp - (float)tex_yinit + 0.5f, zp - (float)tex_zinit + 0.5f);
    img2[img_off] = make_double2((double)re, (double)im);
}

/* ================================================================== */
/*                  Launch dispatchers                                 */
/* ================================================================== */

/* Dispatch macro over (ORDER, HALF_VOL, HALF_IMG) — 8 combinations */

template <typename T>
cudaError_t launch_backproject(
    cudaStream_t s, T* vol, const T* img, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t real_data = 0, int64_t max_r2_x4 = -1)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BP(O, HV, HI, RD) \
        backproject_kernel<T, O, HV, HI, RD><<<grid, block, 0, s>>>( \
            vol, img, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, max_r2)

    int key = (real_data ? 8 : 0) | (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    /* complex data */
    case  0: BP(0, false, false, false); break;
    case  1: BP(0, false, true,  false); break;
    case  2: BP(0, true,  false, false); break;
    case  3: BP(0, true,  true,  false); break;
    case  4: BP(1, false, false, false); break;
    case  5: BP(1, false, true,  false); break;
    case  6: BP(1, true,  false, false); break;
    case  7: BP(1, true,  true,  false); break;
    /* real data */
    case  8: BP(0, false, false, true); break;
    case  9: BP(0, false, true,  true); break;
    case 10: BP(0, true,  false, true); break;
    case 11: BP(0, true,  true,  true); break;
    case 12: BP(1, false, false, true); break;
    case 13: BP(1, false, true,  true); break;
    case 14: BP(1, true,  false, true); break;
    case 15: BP(1, true,  true,  true); break;
    }
    #undef BP
    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_backproject_indexed(
    cudaStream_t s, T* vol, const T* img, const int32_t* pixel_indices, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t real_data = 0, int64_t max_r2_x4 = -1,
    int64_t relion_fold_x = 0,
    int64_t relion_block_topology = 0)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BPI(O, HV, HI, RD) \
        backproject_indexed_kernel<T, O, HV, HI, RD><<<grid, block, 0, s>>>( \
            vol, img, pixel_indices, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, max_r2, (int)relion_fold_x)

    int key = (real_data ? 8 : 0) | (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    if (!relion_block_topology) switch (key) {
    case  0: BPI(0, false, false, false); break;
    case  1: BPI(0, false, true,  false); break;
    case  2: BPI(0, true,  false, false); break;
    case  3: BPI(0, true,  true,  false); break;
    case  4: BPI(1, false, false, false); break;
    case  5: BPI(1, false, true,  false); break;
    case  6: BPI(1, true,  false, false); break;
    case  7: BPI(1, true,  true,  false); break;
    case  8: BPI(0, false, false, true); break;
    case  9: BPI(0, false, true,  true); break;
    case 10: BPI(0, true,  false, true); break;
    case 11: BPI(0, true,  true,  true); break;
    case 12: BPI(1, false, false, true); break;
    case 13: BPI(1, false, true,  true); break;
    case 14: BPI(1, true,  false, true); break;
    case 15: BPI(1, true,  true,  true); break;
    }
    #undef BPI
    if (relion_block_topology) {
        const int vol_stride = (int)N0 * (int)N1 * N2_eff;
        dim3 relion_grid((int)n_images, 1);
        dim3 relion_block(128);
        #define RBPI(O, HV, HI, RD) \
            batch_backproject_indexed_kernel<T, O, HV, HI, RD, true><<<relion_grid, relion_block, 0, s>>>( \
                vol, img, pixel_indices, rot, (int)n_pixels, (int)ih, (int)iw, \
                (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
                vol_stride, (int)n_images, 1, max_r2, (int)relion_fold_x)
        switch (key) {
        case  0: RBPI(0, false, false, false); break;
        case  1: RBPI(0, false, true,  false); break;
        case  2: RBPI(0, true,  false, false); break;
        case  3: RBPI(0, true,  true,  false); break;
        case  4: RBPI(1, false, false, false); break;
        case  5: RBPI(1, false, true,  false); break;
        case  6: RBPI(1, true,  false, false); break;
        case  7: RBPI(1, true,  true,  false); break;
        case  8: RBPI(0, false, false, true); break;
        case  9: RBPI(0, false, true,  true); break;
        case 10: RBPI(0, true,  false, true); break;
        case 11: RBPI(0, true,  true,  true); break;
        case 12: RBPI(1, false, false, true); break;
        case 13: RBPI(1, false, true,  true); break;
        case 14: RBPI(1, true,  false, true); break;
        case 15: RBPI(1, true,  true,  true); break;
        }
        #undef RBPI
    }
    return cudaGetLastError();
}

cudaError_t launch_backproject_indexed_with_signature(
    cudaStream_t stream,
    float* volume,
    const float* images,
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
    float* accumulator_shadow,
    float* operand_shadow_images,
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
    cudaError_t err = launch_backproject_indexed<float>(
        stream, volume, images, pixel_indices, rot,
        n_rows, n_pixels, image_h, image_w, N0, N1, N2,
        upsampling, 1, 1, 1, image_h, 0, max_r2_x4, 1, 0);
    if (err != cudaSuccess) return err;
    const int N2_eff = (int)(N2 / 2 + 1);
    const size_t volume_size = (size_t)N0 * (size_t)N1 * (size_t)N2_eff;
    err = cudaMemcpyAsync(accumulator_shadow, volume,
                          volume_size * sizeof(float2), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(operand_shadow_images, images,
                          (size_t)n_rows * (size_t)n_pixels * sizeof(float2),
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

    const float c0 = (float)(N0 / 2);
    const float c1 = (float)(N1 / 2);
    const float c2 = (float)(N2 / 2);
    const float max_r2 = (float)max_r2_x4 / 4.0f;
    dim3 grid((unsigned)n_signature_rows,
              ((unsigned)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    backproject_indexed_signature_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(images), pixel_indices, rot,
        canonical_rotation_keys, signature_row_indices,
        signature_rotation_keys, signature_pixel_indices, signature_row_flags,
        signature_source_values, signature_neighbor_indices,
        signature_neighbor_coefficients, signature_neighbor_flags,
        (int)n_signature_rows, (int)n_rows, (int)n_pixels,
        (int)image_h, (int)image_w, (int)N0, (int)N1, N2_eff,
        c0, c1, c2, (int)upsampling, max_r2);
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

template <typename T>
cudaError_t launch_batch_backproject_indexed(
    cudaStream_t s, T* vols, const T* imgs, const int32_t* pixel_indices, const T* rot,
    int64_t batch_size, int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t real_data = 0, int64_t max_r2_x4 = -1,
    int64_t relion_fold_x = 0,
    int64_t relion_block_topology = 0)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BBPI(O, HV, HI, RD) \
        batch_backproject_indexed_kernel<T, O, HV, HI, RD><<<grid, block, 0, s>>>( \
            vols, imgs, pixel_indices, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
            vol_stride, (int)n_images, (int)batch_size, max_r2, (int)relion_fold_x)

    int key = (real_data ? 8 : 0) | (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    if (!relion_block_topology) switch (key) {
    case  0: BBPI(0, false, false, false); break;
    case  1: BBPI(0, false, true,  false); break;
    case  2: BBPI(0, true,  false, false); break;
    case  3: BBPI(0, true,  true,  false); break;
    case  4: BBPI(1, false, false, false); break;
    case  5: BBPI(1, false, true,  false); break;
    case  6: BBPI(1, true,  false, false); break;
    case  7: BBPI(1, true,  true,  false); break;
    case  8: BBPI(0, false, false, true); break;
    case  9: BBPI(0, false, true,  true); break;
    case 10: BBPI(0, true,  false, true); break;
    case 11: BBPI(0, true,  true,  true); break;
    case 12: BBPI(1, false, false, true); break;
    case 13: BBPI(1, false, true,  true); break;
    case 14: BBPI(1, true,  false, true); break;
    case 15: BBPI(1, true,  true,  true); break;
    }
    #undef BBPI
    if (relion_block_topology) {
        dim3 relion_grid((int)n_images, 1);
        dim3 relion_block(128);
        #define RBBPI(O, HV, HI, RD) \
            batch_backproject_indexed_kernel<T, O, HV, HI, RD, true><<<relion_grid, relion_block, 0, s>>>( \
                vols, imgs, pixel_indices, rot, (int)n_pixels, (int)ih, (int)iw, \
                (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
                vol_stride, (int)n_images, (int)batch_size, max_r2, (int)relion_fold_x)
        switch (key) {
        case  0: RBBPI(0, false, false, false); break;
        case  1: RBBPI(0, false, true,  false); break;
        case  2: RBBPI(0, true,  false, false); break;
        case  3: RBBPI(0, true,  true,  false); break;
        case  4: RBBPI(1, false, false, false); break;
        case  5: RBBPI(1, false, true,  false); break;
        case  6: RBBPI(1, true,  false, false); break;
        case  7: RBBPI(1, true,  true,  false); break;
        case  8: RBBPI(0, false, false, true); break;
        case  9: RBBPI(0, false, true,  true); break;
        case 10: RBBPI(0, true,  false, true); break;
        case 11: RBBPI(0, true,  true,  true); break;
        case 12: RBBPI(1, false, false, true); break;
        case 13: RBBPI(1, false, true,  true); break;
        case 14: RBBPI(1, true,  false, true); break;
        case 15: RBBPI(1, true,  true,  true); break;
        }
        #undef RBBPI
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_project(
    cudaStream_t s, const T* vol, T* img, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t max_r2_x4 = -1)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define PJ(O, HV, HI) \
        project_kernel<T, O, HV, HI, false><<<grid, block, 0, s>>>( \
            vol, img, rot, nullptr, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, max_r2)

    /* order_code: 0→0, 1→1, 3→2.  key = (order_code << 2) | (half_vol << 1) | half_img */
    int order_code = (order == 3) ? 2 : (int)order;
    int key = (order_code << 2) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case  0: PJ(0, false, false); break;
    case  1: PJ(0, false, true);  break;
    case  2: PJ(0, true,  false); break;
    case  3: PJ(0, true,  true);  break;
    case  4: PJ(1, false, false); break;
    case  5: PJ(1, false, true);  break;
    case  6: PJ(1, true,  false); break;
    case  7: PJ(1, true,  true);  break;
    /* ORDER=3 (cubic, periodic wrap) — project only, no backproject */
    case  8: PJ(3, false, false); break;
    case  9: PJ(3, false, true);  break;
    case 10: PJ(3, true,  false); break;
    case 11: PJ(3, true,  true);  break;
    }
    #undef PJ
    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_project_indexed(
    cudaStream_t s, const T* vol, T* img, const int32_t* pixel_indices, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t max_r2_x4 = -1)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define PJI(O, HV, HI) \
        project_kernel<T, O, HV, HI, true><<<grid, block, 0, s>>>( \
            vol, img, rot, pixel_indices, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, max_r2)

    int order_code = (order == 3) ? 2 : (int)order;
    int key = (order_code << 2) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case  0: PJI(0, false, false); break;
    case  1: PJI(0, false, true);  break;
    case  2: PJI(0, true,  false); break;
    case  3: PJI(0, true,  true);  break;
    case  4: PJI(1, false, false); break;
    case  5: PJI(1, false, true);  break;
    case  6: PJI(1, true,  false); break;
    case  7: PJI(1, true,  true);  break;
    case  8: PJI(3, false, false); break;
    case  9: PJI(3, false, true);  break;
    case 10: PJI(3, true,  false); break;
    case 11: PJI(3, true,  true);  break;
    }
    #undef PJI
    return cudaGetLastError();
}

template <bool CAPACITY_HALF = false>
cudaError_t launch_project_texture_float(
    cudaStream_t s, const float* vol, float* img, const float* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t half_img,
    int64_t full_iw, int64_t max_r2_x4 = -1,
    const int32_t* logical_radius = nullptr,
    const int32_t* image_radius = nullptr)
{
    const float max_r2 = max_r2_x4 < 0 ? (float)((N0 / 2 - 1) * (N0 / 2 - 1)) : (float)max_r2_x4 / 4.0f;
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int texX = CAPACITY_HALF ? (int)N2 : maxR + 2;
    const int texY = CAPACITY_HALF ? (int)N1 : 2 * maxR + 3;
    const int texZ = CAPACITY_HALF ? (int)N0 : 2 * maxR + 3;
    const int texYInit = -(maxR + 1);
    const int texZInit = -(maxR + 1);
    const int n_voxels = texX * texY * texZ;
    float *real = nullptr, *imag = nullptr;
    cudaArray_t arrReal = nullptr, arrImag = nullptr;
    cudaTextureObject_t texReal = 0, texImag = 0;

    cudaError_t err = cudaMalloc((void**)&real, n_voxels * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&imag, n_voxels * sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if constexpr (CAPACITY_HALF) {
            fill_relion_texture_capacity_kernel<<<grid, block, 0, s>>>(
                reinterpret_cast<const float2*>(vol), real, imag,
                logical_radius, (int)ups, texX, texY, texZ);
        } else {
            fill_relion_texture_compact_kernel<float><<<grid, block, 0, s>>>(
                vol, real, imag, texX, texY, texZ, texYInit, texZInit, (int)N0, (int)N1, (int)N2);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)texZ);
        err = cudaMalloc3DArray(&arrReal, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc3DArray(&arrImag, &desc, extent);
        if (err != cudaSuccess) goto cleanup;

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.dstArray = arrReal;
        copyParams.srcPtr = make_cudaPitchedPtr(real, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        err = cudaMemcpy3DAsync(&copyParams, s);
        if (err != cudaSuccess) goto cleanup;
        copyParams.dstArray = arrImag;
        copyParams.srcPtr = make_cudaPitchedPtr(imag, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        err = cudaMemcpy3DAsync(&copyParams, s);
        if (err != cudaSuccess) goto cleanup;

        cudaResourceDesc resReal, resImag;
        cudaTextureDesc texDesc;
        memset(&resReal, 0, sizeof(resReal));
        memset(&resImag, 0, sizeof(resImag));
        memset(&texDesc, 0, sizeof(texDesc));
        resReal.resType = cudaResourceTypeArray;
        resReal.res.array.array = arrReal;
        resImag.resType = cudaResourceTypeArray;
        resImag.res.array.array = arrImag;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(&texReal, &resReal, &texDesc, nullptr);
        if (err != cudaSuccess) goto cleanup;
        err = cudaCreateTextureObject(&texImag, &resImag, &texDesc, nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

    {
        dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        if (half_img) {
            project_texture_kernel<true, CAPACITY_HALF><<<grid, block, 0, s>>>(
                texReal, texImag, img, rot, (int)n_pixels, (int)ih, (int)iw,
                texYInit, texZInit, (int)ups, (int)full_iw, maxR * maxR, logical_radius, maxR, image_radius);
        } else {
            project_texture_kernel<false, CAPACITY_HALF><<<grid, block, 0, s>>>(
                texReal, texImag, img, rot, (int)n_pixels, (int)ih, (int)iw,
                texYInit, texZInit, (int)ups, (int)full_iw, maxR * maxR, logical_radius, maxR, image_radius);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        err = cudaStreamSynchronize(s);
    }

cleanup:
    if (texReal) cudaDestroyTextureObject(texReal);
    if (texImag) cudaDestroyTextureObject(texImag);
    if (arrReal) cudaFreeArray(arrReal);
    if (arrImag) cudaFreeArray(arrImag);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

cudaError_t launch_project_texture_double(
    cudaStream_t s, const double* vol, double* img, const double* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t half_img,
    int64_t full_iw, int64_t max_r2_x4 = -1)
{
    const float max_r2 = max_r2_x4 < 0 ? (float)((N0 / 2 - 1) * (N0 / 2 - 1)) : (float)max_r2_x4 / 4.0f;
    const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
    const int texX = maxR + 2;
    const int texY = 2 * maxR + 3;
    const int texZ = 2 * maxR + 3;
    const int texYInit = -(maxR + 1);
    const int texZInit = -(maxR + 1);
    const int n_voxels = texX * texY * texZ;
    float *real = nullptr, *imag = nullptr;
    cudaArray_t arrReal = nullptr, arrImag = nullptr;
    cudaTextureObject_t texReal = 0, texImag = 0;

    cudaError_t err = cudaMalloc((void**)&real, n_voxels * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&imag, n_voxels * sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((n_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fill_relion_texture_compact_kernel<double><<<grid, block, 0, s>>>(
            vol, real, imag, texX, texY, texZ, texYInit, texZInit, (int)N0, (int)N1, (int)N2);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent((size_t)texX, (size_t)texY, (size_t)texZ);
        err = cudaMalloc3DArray(&arrReal, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc3DArray(&arrImag, &desc, extent);
        if (err != cudaSuccess) goto cleanup;

        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.dstArray = arrReal;
        copyParams.srcPtr = make_cudaPitchedPtr(real, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        err = cudaMemcpy3DAsync(&copyParams, s);
        if (err != cudaSuccess) goto cleanup;
        copyParams.dstArray = arrImag;
        copyParams.srcPtr = make_cudaPitchedPtr(imag, (size_t)texX * sizeof(float), (size_t)texX, (size_t)texY);
        err = cudaMemcpy3DAsync(&copyParams, s);
        if (err != cudaSuccess) goto cleanup;

        cudaResourceDesc resReal, resImag;
        cudaTextureDesc texDesc;
        memset(&resReal, 0, sizeof(resReal));
        memset(&resImag, 0, sizeof(resImag));
        memset(&texDesc, 0, sizeof(texDesc));
        resReal.resType = cudaResourceTypeArray;
        resReal.res.array.array = arrReal;
        resImag.resType = cudaResourceTypeArray;
        resImag.res.array.array = arrImag;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(&texReal, &resReal, &texDesc, nullptr);
        if (err != cudaSuccess) goto cleanup;
        err = cudaCreateTextureObject(&texImag, &resImag, &texDesc, nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

    {
        dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        if (half_img) {
            project_texture_double_kernel<true><<<grid, block, 0, s>>>(
                texReal, texImag, img, rot, (int)n_pixels, (int)ih, (int)iw,
                texYInit, texZInit, (int)ups, (int)full_iw, maxR * maxR);
        } else {
            project_texture_double_kernel<false><<<grid, block, 0, s>>>(
                texReal, texImag, img, rot, (int)n_pixels, (int)ih, (int)iw,
                texYInit, texZInit, (int)ups, (int)full_iw, maxR * maxR);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        err = cudaStreamSynchronize(s);
    }

cleanup:
    if (texReal) cudaDestroyTextureObject(texReal);
    if (texImag) cudaDestroyTextureObject(texImag);
    if (arrReal) cudaFreeArray(arrReal);
    if (arrImag) cudaFreeArray(arrImag);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

/* ================================================================== */
/*              Batched kernels  (batch of volumes)                     */
/* ================================================================== */
/*
 * Same pixel-level logic as the single-volume kernels, but with an
 * extra batch dimension via blockIdx.z.
 *
 *   vols:  (batch, vol_elements * 2)  — contiguous batch of volumes
 *   imgs:  (batch, n_images, n_pixels * 2)  — per-batch images
 *   rot :  (n_images, 6)  — shared across all batches
 *
 * Grid: (n_images, ceil(n_pixels/BLOCK_SIZE), batch_size)
 */

/*
 * Batched kernels with inner-loop over batch dimension.
 *
 * Grid: (n_images, ceil(n_pixels/BLOCK_SIZE))  — same as single-volume.
 * Each block loops over batch_size volumes, reusing rotation coordinates.
 * This gives much better cache locality: the same spatial region of each
 * volume is accessed in a tight loop, keeping working sets in L2 cache.
 */

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG, bool REAL_DATA = false>
__global__ void __launch_bounds__(BLOCK_SIZE)
batch_backproject_kernel(
    T*       __restrict__ vols,
    const T* __restrict__ imgs,
    const T* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    int vol_stride,    /* N0*N1*N2_eff (complex elements for complex, real for REAL_DATA) */
    int n_images,
    int batch_size,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* Compute rotation-dependent coords once, reuse across batch (row-major) */
    const int k0_idx = pix / image_w;   /* row index */
    const int k1_idx = pix % image_w;   /* col index */
    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    /* Pre-rotation disk check: rotation preserves ||k||. */
    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;
    const int img_stride = n_images * n_pixels;  /* elements between batch slices */
    using V2 = vec2_t<T>;

    /* Conjugate scatter coords (computed once for HALF_IMG) */
    T crk0, crk1, crk2;
    bool do_conj_scatter = false;
    if (HALF_IMG && k1_idx > 0 && k1_idx * 2 != full_image_w) {
        do_conj_scatter = true;
        if (k0_idx == 0 && (image_h & 1) == 0) {
            const T neg_k1 = -k1;
            crk0 = k0 * R[0] + neg_k1 * R[3];
            crk1 = k0 * R[1] + neg_k1 * R[4];
            crk2 = k0 * R[2] + neg_k1 * R[5];
        } else {
            crk0 = -rk0; crk1 = -rk1; crk2 = -rk2;
        }
    }

    /* CONJ_MODE optimization: same logic as backproject_kernel.
     * Only applies when crk = -rk (true for all non-boundary pixels
     * EXCEPT k0_idx==0 with even H where crk ≠ -rk), AND when both
     * primary and conjugate positions are within volume bounds.
     * See backproject_kernel comments for detailed explanation. */
    bool conj_opt = HALF_IMG && HALF_VOL
        && (k1_idx > 0 && k1_idx * 2 != full_image_w)
        && !(k0_idx == 0 && (image_h & 1) == 0);

    if (conj_opt) {
        const int N2_full = full_z_size_from_half(N0, N1, N2_eff);
        if (ORDER == 0) {
            const int pi0 = round_int(rk0+c0), pi1 = round_int(rk1+c1);
            const int pi2 = round_int(rk2+c2);
            const int ci0 = round_int(-rk0+c0), ci1 = round_int(-rk1+c1);
            const int ci2 = round_int(-rk2+c2);
            if ((unsigned)pi0 >= (unsigned)N0 || (unsigned)pi1 >= (unsigned)N1 ||
                (unsigned)pi2 >= (unsigned)N2_full ||
                (unsigned)ci0 >= (unsigned)N0 || (unsigned)ci1 >= (unsigned)N1 ||
                (unsigned)ci2 >= (unsigned)N2_full)
                conj_opt = false;
        } else {
            const T pg0 = rk0+c0, pg1 = rk1+c1, pg2 = rk2+c2;
            const T cg0 = -rk0+c0, cg1 = -rk1+c1, cg2 = -rk2+c2;
            if (pg0 < (T)0 || pg0 > (T)(N0-1) ||
                pg1 < (T)0 || pg1 > (T)(N1-1) ||
                pg2 < (T)0 || pg2 > (T)(N2_full-1) ||
                cg0 < (T)0 || cg0 > (T)(N0-1) ||
                cg1 < (T)0 || cg1 > (T)(N1-1) ||
                cg2 < (T)0 || cg2 > (T)(N2_full-1))
                conj_opt = false;
        }
    }

    /* Volume stride: REAL_DATA uses 1 T per voxel, complex uses 2 */
    const int vol_bytes_stride = REAL_DATA ? vol_stride : vol_stride * 2;

    /* Inner loop over batch — same coords, different volumes and images */
    for (int b = 0; b < batch_size; b++) {
        T* vol = vols + b * vol_bytes_stride;

        /* Load pixel — scalar for REAL_DATA, complex pair for complex */
        T val_re, val_im;
        if (REAL_DATA) {
            val_re = imgs[(b * img_stride) + img_idx * n_pixels + pix];
            val_im = (T)0;
        } else {
            V2 px = reinterpret_cast<const V2*>(imgs)[(b * img_stride) + img_idx * n_pixels + pix];
            val_re = px.x;
            val_im = px.y;
        }

        if (ORDER == 0) {
            if (conj_opt)
                scatter_nearest<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                            c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            else
                scatter_nearest<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                             c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        } else {
            if (conj_opt)
                scatter_trilinear<T, true, 1, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                              c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            else
                scatter_trilinear<T, HALF_VOL, 0, REAL_DATA>(vol, rk0, rk1, rk2, val_re, val_im,
                                               c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        }

        if (do_conj_scatter) {
            const T conj_im = REAL_DATA ? (T)0 : -val_im;
            if (ORDER == 0) {
                if (conj_opt)
                    scatter_nearest<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_nearest<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_nearest<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                              val_re, conj_im,
                                              c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            } else {
                if (conj_opt)
                    scatter_trilinear<T, true, 2, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else if (HALF_VOL)
                    scatter_trilinear<T, true, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                  val_re, conj_im,
                                                  c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
                else
                    scatter_trilinear<T, false, 0, REAL_DATA>(vol, crk0, crk1, crk2,
                                                val_re, conj_im,
                                                c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            }
        }
    }
}

/* Batched launch dispatchers */

template <typename T>
cudaError_t launch_batch_backproject(
    cudaStream_t s, T* vols, const T* imgs, const T* rot,
    int64_t batch_size, int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t real_data = 0, int64_t max_r2_x4 = -1)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BBP(O, HV, HI, RD) \
        batch_backproject_kernel<T, O, HV, HI, RD><<<grid, block, 0, s>>>( \
            vols, imgs, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
            vol_stride, (int)n_images, (int)batch_size, max_r2)

    int key = (real_data ? 8 : 0) | (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    /* complex data */
    case  0: BBP(0, false, false, false); break;
    case  1: BBP(0, false, true,  false); break;
    case  2: BBP(0, true,  false, false); break;
    case  3: BBP(0, true,  true,  false); break;
    case  4: BBP(1, false, false, false); break;
    case  5: BBP(1, false, true,  false); break;
    case  6: BBP(1, true,  false, false); break;
    case  7: BBP(1, true,  true,  false); break;
    /* real data */
    case  8: BBP(0, false, false, true); break;
    case  9: BBP(0, false, true,  true); break;
    case 10: BBP(0, true,  false, true); break;
    case 11: BBP(0, true,  true,  true); break;
    case 12: BBP(1, false, false, true); break;
    case 13: BBP(1, false, true,  true); break;
    case 14: BBP(1, true,  false, true); break;
    case 15: BBP(1, true,  true,  true); break;
    }
    #undef BBP
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
    int y = pixel_index / image_half_width - image_h / 2;
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
    int y = pixel_index / image_half_width - image_h / 2;
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
    int y = pixel_index / image_half_width - image_h / 2;
    float tx = translation_angles[2 * translation];
    float ty = translation_angles[2 * translation + 1];
    float sine;
    float cosine;
    sincosf(x * tx + y * ty, &sine, &cosine);

    float2 value = images[image * pixel_count + pixel_row];
    float factor = weighted_ctf[image * pixel_count + pixel_row];
    float translated_real = cosine * value.x - sine * value.y;
    float translated_imag = cosine * value.y + sine * value.x;
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
    int y = pixel_index / image_half_width - image_h / 2;
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

template <int PhaseMode, int TranslateMode>
__global__ void relion_bpref_operands_f32_kernel(
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* numerator,
    float* denominator,
    float2* translated,
    float* weighted_ctf,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width)
{
    int64_t batch_translation = static_cast<int64_t>(blockIdx.x);
    if (batch_translation >= batch_size * translation_count) return;
    int64_t translation = batch_translation % translation_count;
    int64_t image = batch_translation / translation_count;
    const int passes = ceilf(
        static_cast<float>(pixel_count) /
        static_cast<float>(kRelionBprefOperandsBlockSize));
    for (unsigned pass = 0; pass < static_cast<unsigned>(passes); ++pass)
    {
        int64_t pixel_row =
            static_cast<int64_t>(pass) * kRelionBprefOperandsBlockSize +
            threadIdx.x;
        if (pixel_row >= pixel_count) continue;
        int64_t flat = batch_translation * pixel_count + pixel_row;
        int64_t image_pixel = image * pixel_count + pixel_row;
        int pixel_index = pixel_indices[pixel_row];
        int x = pixel_index % image_half_width;
        int y = pixel_index / image_half_width;
        if (y > image_h / 2) y -= image_h;
        float tx = translation_angles[2 * translation];
        float ty = translation_angles[2 * translation + 1];
        float phase;
        if constexpr (PhaseMode == 0)
            phase = x * tx + y * ty;
        else if constexpr (PhaseMode == 1)
            phase = __fadd_rn(__fmul_rn(static_cast<float>(x), tx),
                              __fmul_rn(static_cast<float>(y), ty));
        else if constexpr (PhaseMode == 2)
            phase = __fmaf_rn(static_cast<float>(x), tx,
                              __fmul_rn(static_cast<float>(y), ty));
        else
            phase = __fmaf_rn(static_cast<float>(y), ty,
                              __fmul_rn(static_cast<float>(x), tx));
        float sine;
        float cosine;
        sincosf(phase, &sine, &cosine);

        // Preserve the source statement order in RELION BP.cuh. This
        // primitive accepts native-unit inputs, so RECOVAR normalization is
        // outside the comparison boundary.
        float weight = posterior_over_weight_norm[
            image * translation_count + translation];
        weight = weight * ctf[image_pixel] * minvsigma2[image_pixel];
        weighted_ctf[flat] = weight;
        denominator[flat] = weight * ctf[image_pixel];

        float2 value = images[image_pixel];
        float translated_real;
        float translated_imag;
        if constexpr (TranslateMode == 0)
        {
            translated_real = cosine * value.x - sine * value.y;
            translated_imag = cosine * value.y + sine * value.x;
        }
        else if constexpr (TranslateMode == 1)
        {
            translated_real = __fsub_rn(__fmul_rn(cosine, value.x),
                                        __fmul_rn(sine, value.y));
            translated_imag = __fadd_rn(__fmul_rn(cosine, value.y),
                                        __fmul_rn(sine, value.x));
        }
        else if constexpr (TranslateMode == 2)
        {
            translated_real = __fmaf_rn(cosine, value.x,
                                        -__fmul_rn(sine, value.y));
            translated_imag = __fmaf_rn(cosine, value.y,
                                        __fmul_rn(sine, value.x));
        }
        else if constexpr (TranslateMode == 3)
        {
            translated_real = __fmaf_rn(-sine, value.y,
                                        __fmul_rn(cosine, value.x));
            translated_imag = __fmaf_rn(sine, value.x,
                                        __fmul_rn(cosine, value.y));
        }
        else if constexpr (TranslateMode == 4)
        {
            // RELION's coarse Gaussian scorer uses this mixed contraction:
            // two rounded products for the real component, but an FMA for
            // the imaginary component. Keep it as an explicit diagnostic
            // variant because BPref and fine scoring use other pairings.
            translated_real = __fsub_rn(__fmul_rn(cosine, value.x),
                                        __fmul_rn(sine, value.y));
            translated_imag = __fmaf_rn(sine, value.x,
                                        __fmul_rn(cosine, value.y));
        }
        else
        {
            translated_real = cosine * value.x - sine * value.y;
            translated_imag = __fmaf_rn(sine, value.x,
                                        __fmul_rn(cosine, value.y));
        }
        translated[flat] = make_float2(translated_real, translated_imag);
        numerator[flat] = make_float2(
            translated_real * weight,
            translated_imag * weight);
    }
}

cudaError_t launch_relion_bpref_operands_f32(
    cudaStream_t stream,
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const int32_t* pixel_indices,
    float2* numerator,
    float* denominator,
    float2* translated,
    float* weighted_ctf,
    int64_t batch_size,
    int64_t translation_count,
    int64_t pixel_count,
    int image_h,
    int image_half_width,
    int arithmetic_variant)
{
    int64_t total = batch_size * translation_count * pixel_count;
    if (total == 0) return cudaSuccess;
    int blocks = static_cast<int>(batch_size * translation_count);
    #define RELION_BPREF_VARIANT(PHASE, TRANSLATE)                         \
        relion_bpref_operands_f32_kernel<PHASE, TRANSLATE><<<              \
            blocks, kRelionBprefOperandsBlockSize, 0, stream>>>(           \
            images, ctf, minvsigma2, posterior_over_weight_norm,           \
            translation_angles, pixel_indices, numerator, denominator,     \
            translated, weighted_ctf,                                      \
            batch_size, translation_count, pixel_count, image_h,           \
            image_half_width)
    switch (arithmetic_variant)
    {
    case 0: RELION_BPREF_VARIANT(0, 0); break;
    case 1: RELION_BPREF_VARIANT(0, 1); break;
    case 2: RELION_BPREF_VARIANT(0, 2); break;
    case 3: RELION_BPREF_VARIANT(0, 3); break;
    case 4: RELION_BPREF_VARIANT(1, 0); break;
    case 5: RELION_BPREF_VARIANT(1, 1); break;
    case 6: RELION_BPREF_VARIANT(1, 2); break;
    case 7: RELION_BPREF_VARIANT(1, 3); break;
    case 8: RELION_BPREF_VARIANT(2, 0); break;
    case 9: RELION_BPREF_VARIANT(2, 1); break;
    case 10: RELION_BPREF_VARIANT(2, 2); break;
    case 11: RELION_BPREF_VARIANT(2, 3); break;
    case 12: RELION_BPREF_VARIANT(3, 0); break;
    case 13: RELION_BPREF_VARIANT(3, 1); break;
    case 14: RELION_BPREF_VARIANT(3, 2); break;
    case 15: RELION_BPREF_VARIANT(3, 3); break;
    case 16: RELION_BPREF_VARIANT(0, 4); break;
    case 17: RELION_BPREF_VARIANT(1, 4); break;
    case 18: RELION_BPREF_VARIANT(2, 4); break;
    case 19: RELION_BPREF_VARIANT(3, 4); break;
    case 20: RELION_BPREF_VARIANT(0, 5); break;
    case 21: RELION_BPREF_VARIANT(1, 5); break;
    case 22: RELION_BPREF_VARIANT(2, 5); break;
    case 23: RELION_BPREF_VARIANT(3, 5); break;
    default: return cudaErrorInvalidValue;
    }
    #undef RELION_BPREF_VARIANT
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
        // exactly as relion_translate_bpref_f32_kernel expects.
        int y = pixel_index / image_half_width - image_h / 2;

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

__global__ void relion_normalize_f32_kernel(
    const float* images,
    const float* normalization_factors,
    float* normalized,
    int64_t pixels_per_image,
    int64_t total_pixels)
{
    int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= total_pixels) return;
    int64_t image = pixel / pixels_per_image;
    normalized[pixel] = images[pixel] * normalization_factors[image];
}

__global__ void relion_translate2d_f32_kernel(
    const float* normalized,
    const int32_t* shifts,
    float* shifted,
    int64_t pixels_per_image,
    int image_h,
    int image_w,
    int64_t total_pixels)
{
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_pixels) return;
    int64_t image = flat / pixels_per_image;
    int pixel = static_cast<int>(flat - image * pixels_per_image);
    int x = pixel % image_w;
    int y = pixel / image_w;
    int xp = x + shifts[2 * image];
    int yp = y + shifts[2 * image + 1];
    if (xp >= 0 && xp < image_w && yp >= 0 && yp < image_h) {
        int64_t out = image * pixels_per_image + static_cast<int64_t>(yp) * image_w + xp;
        shifted[out] = normalized[flat];
    }
}

__global__ void relion_softmask_background_f32_kernel(
    const float* image,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* block_sum,
    float* block_sum_bg)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float partial_sum = 0.0f;
    float partial_sum_bg = 0.0f;
    int64_t passes = (image_size + kRelionPreprocessBlockSize * gridDim.x - 1) /
                     (kRelionPreprocessBlockSize * gridDim.x);
    int64_t texel = static_cast<int64_t>(bid) * kRelionPreprocessBlockSize * passes + tid;

    for (int64_t pass = 0; pass < passes; ++pass, texel += kRelionPreprocessBlockSize) {
        if (texel >= image_size) continue;
        float value = __ldg(&image[texel]);
        int y = static_cast<int>(texel / image_w) - yinit;
        int x = static_cast<int>(texel % image_w) - xinit;
        float r = sqrtf(static_cast<float>(x * x + y * y));
        if (r < radius) continue;
        if (r > radius_p) {
            partial_sum += 1.0f;
            partial_sum_bg += value;
        } else {
            float raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width);
            partial_sum += raisedcos;
            partial_sum_bg += raisedcos * value;
        }
    }

    // Preserve the original 128-block pixel parallelism, but give every block
    // a unique output slot.  Two fixed CUB trees (block-local here, then
    // device-wide below) replace the schedule-dependent atomicAdd into shared
    // lane slots.
    using BlockReduce = cub::BlockReduce<float, kRelionPreprocessBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    float reduced_sum = BlockReduce(reduce_storage).Sum(partial_sum);
    __syncthreads();
    float reduced_sum_bg = BlockReduce(reduce_storage).Sum(partial_sum_bg);
    if (tid == 0) {
        block_sum[bid] = reduced_sum;
        block_sum_bg[bid] = reduced_sum_bg;
    }
}

__global__ void relion_softmask_background_lane_partials_f32_kernel(
    const float* image,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* block_lane_sum,
    float* block_lane_sum_bg)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float partial_sum = 0.0f;
    float partial_sum_bg = 0.0f;
    int64_t passes = (image_size + kRelionPreprocessBlockSize * gridDim.x - 1) /
                     (kRelionPreprocessBlockSize * gridDim.x);
    int64_t texel = static_cast<int64_t>(bid) * kRelionPreprocessBlockSize * passes + tid;

    for (int64_t pass = 0; pass < passes; ++pass, texel += kRelionPreprocessBlockSize) {
        if (texel >= image_size) continue;
        float value = __ldg(&image[texel]);
        int y = static_cast<int>(texel / image_w) - yinit;
        int x = static_cast<int>(texel % image_w) - xinit;
        float r = sqrtf(static_cast<float>(x * x + y * y));
        if (r < radius) continue;
        if (r > radius_p) {
            partial_sum += 1.0f;
            partial_sum_bg += value;
        } else {
            float raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width);
            partial_sum += raisedcos;
            partial_sum_bg += raisedcos * value;
        }
    }

    int output = bid * kRelionPreprocessBlockSize + tid;
    block_lane_sum[output] = partial_sum;
    block_lane_sum_bg[output] = partial_sum_bg;
}

__global__ void relion_softmask_background_native_atomic_f32_kernel(
    const float* image,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* lane_sum,
    float* lane_sum_bg)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float partial_sum = 0.0f;
    float partial_sum_bg = 0.0f;
    int64_t passes = (image_size + kRelionPreprocessBlockSize * gridDim.x - 1) /
                     (kRelionPreprocessBlockSize * gridDim.x);
    int64_t texel = static_cast<int64_t>(bid) * kRelionPreprocessBlockSize * passes + tid;

    for (int64_t pass = 0; pass < passes; ++pass, texel += kRelionPreprocessBlockSize) {
        if (texel >= image_size) continue;
        float value = __ldg(&image[texel]);
        int y = static_cast<int>(texel / image_w) - yinit;
        int x = static_cast<int>(texel % image_w) - xinit;
        float r = sqrtf(static_cast<float>(x * x + y * y));
        if (r < radius) continue;
        if (r > radius_p) {
            partial_sum += 1.0f;
            partial_sum_bg += value;
        } else {
            float raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width);
            partial_sum += raisedcos;
            partial_sum_bg += raisedcos * value;
        }
    }

    // Match deployed RELION: 128 blocks atomically accumulate into one slot
    // per lane, followed by the same 128-value CUB reduction.
    atomicAdd(&lane_sum[tid], partial_sum);
    atomicAdd(&lane_sum_bg[tid], partial_sum_bg);
}

__global__ void relion_softmask_finalize_lane_partials_f32_kernel(
    const float* block_lane_sum,
    const float* block_lane_sum_bg,
    float* lane_sum,
    float* lane_sum_bg)
{
    int tid = threadIdx.x;
    volatile float total = 0.0f;
    volatile float total_bg = 0.0f;
    for (int block = 0; block < kRelionSoftMaskBlocks; ++block) {
        int input = block * kRelionPreprocessBlockSize + tid;
        total = total + block_lane_sum[input];
        total_bg = total_bg + block_lane_sum_bg[input];
    }
    lane_sum[tid] = total;
    lane_sum_bg[tid] = total_bg;
}

__global__ void relion_cosine_fill_f32_kernel(
    float* image,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float bg_value)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int64_t passes = (image_size + kRelionPreprocessBlockSize * gridDim.x - 1) /
                     (kRelionPreprocessBlockSize * gridDim.x);
    int64_t texel = static_cast<int64_t>(bid) * kRelionPreprocessBlockSize * passes + tid;

    for (int64_t pass = 0; pass < passes; ++pass, texel += kRelionPreprocessBlockSize) {
        if (texel >= image_size) continue;
        float value = __ldg(&image[texel]);
        int y = static_cast<int>(texel / image_w) - yinit;
        int x = static_cast<int>(texel % image_w) - xinit;
        float r = sqrtf(static_cast<float>(x * x + y * y));
        if (r < radius) continue;
        if (r > radius_p) {
            value = bg_value;
        } else {
            float raisedcos = 0.5f + 0.5f * cospif((radius_p - r) / cosine_width);
            value = value * (1.0f - raisedcos) + bg_value * raisedcos;
        }
        image[texel] = value;
    }
}

cudaError_t launch_relion_preprocess_real_f32(
    cudaStream_t stream,
    const float* images,
    const float* normalization_factors,
    const int32_t* shifts,
    float* normalized_shifted,
    float* masked,
    int64_t batch_size,
    int image_h,
    int image_w,
    float radius,
    float cosine_width,
    bool apply_mask,
    int reduction_mode)
{
    int64_t pixels_per_image = static_cast<int64_t>(image_h) * image_w;
    int64_t total_pixels = batch_size * pixels_per_image;
    size_t image_bytes = static_cast<size_t>(total_pixels) * sizeof(float);
    int blocks = static_cast<int>((total_pixels + kRelionPreprocessBlockSize - 1) /
                                  kRelionPreprocessBlockSize);

    // `masked` is temporary normalized storage until translation completes.
    relion_normalize_f32_kernel<<<blocks, kRelionPreprocessBlockSize, 0, stream>>>(
        images, normalization_factors, masked, pixels_per_image, total_pixels);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(normalized_shifted, 0, image_bytes, stream);
    if (err != cudaSuccess) return err;
    relion_translate2d_f32_kernel<<<blocks, kRelionPreprocessBlockSize, 0, stream>>>(
        masked, shifts, normalized_shifted, pixels_per_image, image_h, image_w, total_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(masked, normalized_shifted, image_bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess || !apply_mask) return err;

    constexpr int kRelionSoftMaskLanePartials =
        kRelionSoftMaskBlocks * kRelionPreprocessBlockSize;
    bool deterministic_lane_reduction = reduction_mode == 1;
    bool native_atomic_reduction = reduction_mode == 2;
    int primary_count = deterministic_lane_reduction
        ? kRelionSoftMaskLanePartials
        : (native_atomic_reduction ? kRelionPreprocessBlockSize : kRelionSoftMaskBlocks);
    int reduction_input_count = deterministic_lane_reduction
        ? kRelionPreprocessBlockSize
        : primary_count;
    size_t reduction_storage_count = 2 * static_cast<size_t>(primary_count);
    if (deterministic_lane_reduction)
        reduction_storage_count += 2 * kRelionPreprocessBlockSize;

    float* reduction_storage = nullptr;
    float* reduce_values = nullptr;
    void* reduce_temp = nullptr;
    size_t reduce_temp_bytes = 0;
    err = cudaMalloc(
        reinterpret_cast<void**>(&reduction_storage),
        reduction_storage_count * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(reinterpret_cast<void**>(&reduce_values), 2 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(reduction_storage);
        return err;
    }
    float* reduction_input = deterministic_lane_reduction
        ? reduction_storage + 2 * primary_count
        : reduction_storage;
    err = cub::DeviceReduce::Sum(
        nullptr, reduce_temp_bytes, reduction_input, reduce_values,
        reduction_input_count, stream);
    if (err == cudaSuccess)
        err = cudaMalloc(&reduce_temp, reduce_temp_bytes == 0 ? 1 : reduce_temp_bytes);
    if (err != cudaSuccess) {
        cudaFree(reduce_values);
        cudaFree(reduction_storage);
        return err;
    }

    float radius_p = radius + cosine_width;
    for (int64_t image = 0; image < batch_size; ++image) {
        float* primary_sum = reduction_storage;
        float* primary_sum_bg = reduction_storage + primary_count;
        float* sum_input = primary_sum;
        float* sum_input_bg = primary_sum_bg;
        float* image_ptr = masked + image * pixels_per_image;
        if (deterministic_lane_reduction) {
            float* lane_sum = reduction_storage + 2 * primary_count;
            float* lane_sum_bg = lane_sum + kRelionPreprocessBlockSize;
            relion_softmask_background_lane_partials_f32_kernel<<<
                kRelionSoftMaskBlocks, kRelionPreprocessBlockSize, 0, stream>>>(
                image_ptr, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
                radius, radius_p, cosine_width, primary_sum, primary_sum_bg);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;
            relion_softmask_finalize_lane_partials_f32_kernel<<<
                1, kRelionPreprocessBlockSize, 0, stream>>>(
                primary_sum, primary_sum_bg, lane_sum, lane_sum_bg);
            sum_input = lane_sum;
            sum_input_bg = lane_sum_bg;
        } else if (native_atomic_reduction) {
            err = cudaMemsetAsync(
                reduction_storage,
                0,
                2 * kRelionPreprocessBlockSize * sizeof(float),
                stream);
            if (err != cudaSuccess) break;
            relion_softmask_background_native_atomic_f32_kernel<<<
                kRelionSoftMaskBlocks, kRelionPreprocessBlockSize, 0, stream>>>(
                image_ptr, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
                radius, radius_p, cosine_width, primary_sum, primary_sum_bg);
        } else {
            relion_softmask_background_f32_kernel<<<
                kRelionSoftMaskBlocks, kRelionPreprocessBlockSize, 0, stream>>>(
                image_ptr, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
                radius, radius_p, cosine_width, primary_sum, primary_sum_bg);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
        err = cub::DeviceReduce::Sum(
            reduce_temp, reduce_temp_bytes, sum_input, reduce_values,
            reduction_input_count, stream);
        if (err != cudaSuccess) break;
        err = cub::DeviceReduce::Sum(
            reduce_temp, reduce_temp_bytes, sum_input_bg, reduce_values + 1,
            reduction_input_count, stream);
        if (err != cudaSuccess) break;
        float host_sums[2];
        err = cudaMemcpyAsync(
            host_sums, reduce_values, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;
        if (!(host_sums[0] > 0.0f) || !std::isfinite(host_sums[0]) || !std::isfinite(host_sums[1])) {
            err = cudaErrorInvalidValue;
            break;
        }
        float bg_value = host_sums[1] / host_sums[0];
        relion_cosine_fill_f32_kernel<<<
            kRelionSoftMaskBlocks, kRelionPreprocessBlockSize, 0, stream>>>(
            image_ptr, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
            radius, radius_p, cosine_width, bg_value);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
    }

    cudaError_t free_temp_err = cudaFree(reduce_temp);
    cudaError_t free_values_err = cudaFree(reduce_values);
    cudaError_t free_storage_err = cudaFree(reduction_storage);
    if (err != cudaSuccess) return err;
    if (free_temp_err != cudaSuccess) return free_temp_err;
    if (free_values_err != cudaSuccess) return free_values_err;
    return free_storage_err;
}

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

ffi::Error RelionCubSortScanF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must be F32");

    auto input_dims = values.dimensions();
    auto sorted_dims = sorted->dimensions();
    auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 1 || input_dims[0] < 1 ||
        sorted_dims.size() != 1 || sorted_dims[0] != input_dims[0] ||
        cumulative_dims.size() != 1 || cumulative_dims[0] != input_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must have the same nonempty 1-D shape");

    const int64_t count = input_dims[0];
    if (count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: vector is too large for CUB's item count");

    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 sort query: ") + cudaGetErrorString(err));
    err = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 scan query: ") + cudaGetErrorString(err));

    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
    err = cudaMalloc(&temporary, temporary_bytes);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaMalloc: ") + cudaGetErrorString(err));

    err = cub::DeviceRadixSort::SortKeys(
        temporary, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err == cudaSuccess)
        err = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    cudaError_t free_error = cudaFree(temporary);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 execute: ") + cudaGetErrorString(err));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaFree: ") + cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

ffi::Error RelionCubSortScanBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must be F32");

    const auto input_dims = values.dimensions();
    const auto sorted_dims = sorted->dimensions();
    const auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 2 || input_dims[0] < 1 || input_dims[1] < 1 ||
        sorted_dims.size() != 2 || sorted_dims[0] != input_dims[0] ||
        sorted_dims[1] != input_dims[1] ||
        cumulative_dims.size() != 2 || cumulative_dims[0] != input_dims[0] ||
        cumulative_dims[1] != input_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must have the same nonempty 2-D shape");
    if (input_dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: row is too large for CUB's item count");

    const int64_t row_count = input_dims[0];
    const int count = static_cast<int>(input_dims[1]);
    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());

    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, count,
        0, sizeof(float) * 8, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 sort query: ") +
            cudaGetErrorString(error));
    error = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, count, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 scan query: ") +
            cudaGetErrorString(error));

    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(
        1, std::max(sort_bytes, scan_bytes));
    error = cudaMallocAsync(&temporary, temporary_bytes, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaMallocAsync: ") +
            cudaGetErrorString(error));

    for (int64_t row = 0; row < row_count && error == cudaSuccess; ++row)
    {
        const int64_t offset = row * static_cast<int64_t>(count);
        error = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, input_ptr + offset, sorted_ptr + offset,
            count, 0, sizeof(float) * 8, stream);
        if (error == cudaSuccess)
            error = relion_ampere_inclusive_sum_f32(
                temporary, scan_bytes, sorted_ptr + offset,
                cumulative_ptr + offset, count, stream);
    }
    const cudaError_t free_error = cudaFreeAsync(temporary, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 execute: ") +
            cudaGetErrorString(error));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaFreeAsync: ") +
            cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

// Explicit, unused score-to-support transaction. Keep the exact existing CUB
// sort and Ampere scan, but compile the surrounding row algebra once in CUDA.
// Support capacity is B*N: threshold ties are never truncated to maxsig.
struct CoarsePosteriorFiniteMax {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void coarse_posterior_row_max_f32(
    const float* scores, const float* raw_max, const int32_t* actual,
    float* maxima, int rows, int count)
{
    const int row = blockIdx.x;
    float best = -CUDART_INF_F;
    if (row < *actual && *actual <= rows) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const float value = __fadd_rn(scores[int64_t(row)*count+i], -raw_max[row]);
            if (isfinite(value)) best = fmaxf(best, value);
        }
    }
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const float result = Reduce(temp).Reduce(best, CoarsePosteriorFiniteMax());
    if (threadIdx.x == 0) maxima[row] = result;
}

__global__ void coarse_posterior_weights_f32(
    const float* scores, const float* raw_max, const float* maxima,
    float* weights, int count, int64_t size)
{
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= size) return;
    const int row = index/count;
    const float score = __fadd_rn(scores[index], -raw_max[row]);
    const float best = maxima[row];
    const float add = __fsub_rn(50.0f, isfinite(best) ? best : 0.0f);
    const float exponent = __fadd_rn(score, add);
    weights[index] = !isfinite(score) || !isfinite(best) || exponent < -88.0f
        ? 0.0f : expf(exponent);
}

struct CoarsePosteriorReduction {
    float probability, best_score;
    int winner, best, count;
};

struct CoarsePosteriorCombine {
    __device__ CoarsePosteriorReduction operator()(
        CoarsePosteriorReduction a, CoarsePosteriorReduction b) const
    {
        if (b.probability > a.probability ||
            (b.probability == a.probability && b.winner < a.winner)) {
            a.probability = b.probability; a.winner = b.winner;
        }
        // jnp.argmax picks the first NaN, otherwise the first maximum.
        if ((!isnan(a.best_score) && isnan(b.best_score)) ||
            (!isnan(a.best_score) && !isnan(b.best_score) &&
             (b.best_score > a.best_score ||
              (b.best_score == a.best_score && b.best < a.best))) ||
            (isnan(a.best_score) && isnan(b.best_score) && b.best < a.best)) {
            a.best_score = b.best_score; a.best = b.best;
        }
        a.count += b.count;
        return a;
    }
};

__device__ int coarse_posterior_upper_bound(const float* values, int count, float target)
{
    int low = 0, high = count;
    while (low < high) {
        const int middle = low + (high-low)/2;
        if (target < values[middle]) high = middle;
        else low = middle+1;
    }
    return low;
}

__global__ void coarse_posterior_finish_f32(
    const float* scores, const float* weights, const float* sorted,
    const float* cumulative, const float* maxima, const int32_t* actual,
    uint8_t* mask, float* statistics, int32_t* indices,
    int rows, int count, float fraction, int maxsig)
{
    const int row = blockIdx.x;
    const int64_t offset = int64_t(row)*count;
    __shared__ float total, threshold;
    __shared__ int has_mass, cutoff;
    if (threadIdx.x == 0) {
        total = cumulative[offset+count-1];
        has_mass = row < *actual && *actual > 0 && *actual <= rows &&
            isfinite(maxima[row]) && isfinite(total) && total > 0.0f;
        const float target = float(__dmul_rn(__dsub_rn(1.0, double(fraction)), double(total)));
        int rank = coarse_posterior_upper_bound(cumulative+offset, count, target);
        const int first_positive = coarse_posterior_upper_bound(sorted+offset, count, 0.0f);
        rank = min(max(rank, first_positive), count-1);
        if (maxsig > 0) rank = max(rank, count-maxsig);
        threshold = sorted[offset+rank];
        cutoff = has_mass ? count-rank : 0;
    }
    __syncthreads();
    CoarsePosteriorReduction value{0.0f, -CUDART_INF_F, count, count, 0};
    const CoarsePosteriorCombine combine;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float weight = weights[offset+i];
        const bool significant = has_mass && weight > 0.0f && weight >= threshold;
        mask[offset+i] = significant;
        // Preserve float/float division even where rounding creates a tie.
        const float probability = has_mass ? __fdiv_rn(weight, total) : 0.0f;
        value = combine(value, {probability, scores[offset+i], i, i, int(significant)});
    }
    using Reduce = cub::BlockReduce<CoarsePosteriorReduction, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const auto result = Reduce(temp).Reduce(value, combine);
    if (threadIdx.x == 0) {
        const bool valid_count = *actual > 0 && *actual <= rows;
        statistics[4*row+0] = valid_count ? result.best_score : CUDART_NAN_F;
        statistics[4*row+1] = result.probability;
        statistics[4*row+2] = total;
        statistics[4*row+3] = threshold;
        indices[4*row+0] = result.best;
        indices[4*row+1] = result.winner;
        indices[4*row+2] = result.count;
        indices[4*row+3] = cutoff;
    }
}

ffi::Error RelionCoarsePosteriorTransactionF32Impl(
    cudaStream_t stream, ffi::AnyBuffer scores, ffi::AnyBuffer raw_max,
    ffi::AnyBuffer actual, float fraction, int64_t maxsig,
    ffi::Result<ffi::AnyBuffer> statistics, ffi::Result<ffi::AnyBuffer> indices,
    ffi::Result<ffi::AnyBuffer> support, ffi::Result<ffi::AnyBuffer> support_count)
{
    const auto dims = scores.dimensions();
    if (scores.element_type() != ffi::DataType::F32 || dims.size() != 2 ||
        dims[0] < 1 || dims[1] < 1 || dims[0] > INT_MAX/dims[1] ||
        raw_max.element_type() != ffi::DataType::F32 ||
        raw_max.dimensions().size() != 1 || raw_max.dimensions()[0] != dims[0] ||
        actual.element_type() != ffi::DataType::S32 || actual.dimensions().size() != 0 ||
        !std::isfinite(fraction) || fraction <= 0 || fraction > 1 || maxsig < 0 || maxsig > INT_MAX)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid scores, maxima, count or policy");
    const int rows = dims[0], count = dims[1], size = rows*count;
    if (statistics->element_type() != ffi::DataType::F32 ||
        indices->element_type() != ffi::DataType::S32 ||
        support->element_type() != ffi::DataType::S32 ||
        support_count->element_type() != ffi::DataType::S32 ||
        statistics->dimensions().size() != 2 || statistics->dimensions()[0] != rows || statistics->dimensions()[1] != 4 ||
        indices->dimensions().size() != 2 || indices->dimensions()[0] != rows || indices->dimensions()[1] != 4 ||
        support->dimensions().size() != 1 || support->dimensions()[0] != size ||
        support_count->dimensions().size() != 0)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid output shapes/dtypes");
    // Three score-size scratch arrays and flags; no host reads or host stream
    // synchronization. All work and temporary lifetimes use the FFI stream.
    void* storage = nullptr;
    cudaError_t error = cudaMallocAsync(&storage, size_t(size)*13+size_t(rows)*4, stream);
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    float* weights = static_cast<float*>(storage);
    float* sorted = weights+size;
    float* cumulative = sorted+size;
    float* maxima = cumulative+size;
    uint8_t* mask = reinterpret_cast<uint8_t*>(maxima+rows);
    auto* out_support = static_cast<int32_t*>(support->untyped_data());
    auto* out_count = static_cast<int32_t*>(support_count->untyped_data());
    thrust::counting_iterator<int32_t> positions(0);
    size_t sort_bytes=0, scan_bytes=0, select_bytes=0;
    error = cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, weights, sorted, count, 0, 32, stream);
    if (error == cudaSuccess) error = relion_ampere_inclusive_sum_f32(nullptr, scan_bytes, sorted, cumulative, count, stream);
    if (error == cudaSuccess) error = cub::DeviceSelect::Flagged(nullptr, select_bytes, positions, mask, out_support, out_count, size, stream);
    void* temporary = nullptr;
    if (error == cudaSuccess) error = cudaMallocAsync(&temporary, std::max(size_t(1), std::max(select_bytes, std::max(sort_bytes, scan_bytes))), stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(out_support, 0xff, size_t(size)*sizeof(int32_t), stream);
    if (error == cudaSuccess) {
        coarse_posterior_row_max_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()),
            static_cast<const int32_t*>(actual.untyped_data()), maxima, rows, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        coarse_posterior_weights_f32<<<(int64_t(size)+255)/256,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()), maxima, weights, count, size);
        error = cudaGetLastError();
    }
    for (int row=0; row<rows && error==cudaSuccess; ++row) {
        const int64_t offset = int64_t(row)*count;
        error = cub::DeviceRadixSort::SortKeys(temporary, sort_bytes, weights+offset, sorted+offset, count, 0, 32, stream);
        if (error == cudaSuccess) error = relion_ampere_inclusive_sum_f32(temporary, scan_bytes, sorted+offset, cumulative+offset, count, stream);
    }
    if (error == cudaSuccess) {
        coarse_posterior_finish_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), weights, sorted, cumulative, maxima,
            static_cast<const int32_t*>(actual.untyped_data()), mask,
            static_cast<float*>(statistics->untyped_data()), static_cast<int32_t*>(indices->untyped_data()),
            rows, count, fraction, maxsig);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cub::DeviceSelect::Flagged(temporary, select_bytes, positions, mask, out_support, out_count, size, stream);
    const auto free_temp = temporary ? cudaFreeAsync(temporary, stream) : cudaSuccess;
    const auto free_storage = cudaFreeAsync(storage, stream);
    if (error == cudaSuccess) error = free_temp;
    if (error == cudaSuccess) error = free_storage;
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarsePosteriorTransactionF32, RelionCoarsePosteriorTransactionF32Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<float>("fraction").Attr<int64_t>("maxsig")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
);

struct RelionPositiveF32
{
    __device__ __forceinline__ bool operator()(const float& value) const
    {
        return value > 0.0f;
    }
};

cudaError_t relion_cub_positive_sort_scan_f32(
    cudaStream_t stream,
    const float* input,
    float* sorted,
    float* cumulative,
    int count)
{
    float* filtered = nullptr;
    int* selected_count_device = nullptr;
    void* temporary = nullptr;
    cudaError_t err = cudaSuccess;

    do
    {
        err = cudaMalloc(reinterpret_cast<void**>(&filtered), count * sizeof(float));
        if (err != cudaSuccess) break;
        err = cudaMalloc(reinterpret_cast<void**>(&selected_count_device), sizeof(int));
        if (err != cudaSuccess) break;

        size_t select_bytes = 0;
        size_t sort_bytes = 0;
        size_t scan_bytes = 0;
        err = cub::DeviceSelect::If(
            nullptr, select_bytes, input, filtered, selected_count_device,
            count, RelionPositiveF32(), stream);
        if (err != cudaSuccess) break;
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, sort_bytes, filtered, sorted, count,
            0, sizeof(float) * 8, stream);
        if (err != cudaSuccess) break;
        err = relion_ampere_inclusive_sum_f32(
            nullptr, scan_bytes, sorted, cumulative, count, stream);
        if (err != cudaSuccess) break;

        const size_t temporary_bytes = std::max<size_t>(
            1, std::max(select_bytes, std::max(sort_bytes, scan_bytes)));
        err = cudaMalloc(&temporary, temporary_bytes);
        if (err != cudaSuccess) break;
        err = cudaMemsetAsync(sorted, 0, count * sizeof(float), stream);
        if (err != cudaSuccess) break;
        err = cudaMemsetAsync(cumulative, 0, count * sizeof(float), stream);
        if (err != cudaSuccess) break;

        err = cub::DeviceSelect::If(
            temporary, select_bytes, input, filtered, selected_count_device,
            count, RelionPositiveF32(), stream);
        if (err != cudaSuccess) break;
        int selected_count = 0;
        err = cudaMemcpyAsync(
            &selected_count, selected_count_device, sizeof(int),
            cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;
        if (selected_count < 0 || selected_count > count)
        {
            err = cudaErrorInvalidValue;
            break;
        }
        if (selected_count == 0) break;

        const int output_offset = count - selected_count;
        err = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, filtered, sorted + output_offset,
            selected_count, 0, sizeof(float) * 8, stream);
        if (err != cudaSuccess) break;
        err = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted + output_offset,
            cumulative + output_offset, selected_count, stream);
    } while (false);

    const cudaError_t temporary_free_error =
        temporary == nullptr ? cudaSuccess : cudaFree(temporary);
    const cudaError_t selected_count_free_error =
        selected_count_device == nullptr ? cudaSuccess : cudaFree(selected_count_device);
    const cudaError_t filtered_free_error =
        filtered == nullptr ? cudaSuccess : cudaFree(filtered);
    if (err != cudaSuccess) return err;
    if (temporary_free_error != cudaSuccess) return temporary_free_error;
    if (selected_count_free_error != cudaSuccess) return selected_count_free_error;
    return filtered_free_error;
}

ffi::Error RelionCubPositiveSortScanF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: input and outputs must be F32");

    auto input_dims = values.dimensions();
    auto sorted_dims = sorted->dimensions();
    auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 1 || input_dims[0] < 1 ||
        sorted_dims.size() != 1 || sorted_dims[0] != input_dims[0] ||
        cumulative_dims.size() != 1 || cumulative_dims[0] != input_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: input and outputs must have the same nonempty 1-D shape");

    const int64_t count = input_dims[0];
    if (count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: vector is too large for CUB's item count");

    const cudaError_t err = relion_cub_positive_sort_scan_f32(
        stream,
        static_cast<const float*>(values.untyped_data()),
        static_cast<float*>(sorted->untyped_data()),
        static_cast<float*>(cumulative->untyped_data()),
        static_cast<int>(count));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubPositiveSortScanF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

__global__ void relion_exponentiate_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float exponent = values[index] + add[0];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto add_dims = add.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        add_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: values/output must be matching nonempty 1-D arrays and add a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_exponentiate_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(add.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateF32, RelionExponentiateF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_exponentiate_batched_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float exponent = values[index] + add[index / row_size];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto add_dims = add.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        add_dims.size() != 1 || add_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: expected values/output (B,N) and add (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: launch grid exceeds CUDA limit");
    relion_exponentiate_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(add.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateBatchedF32, RelionExponentiateBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[0];
}

ffi::Error RelionDivideF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto divisor_dims = divisor.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        divisor_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: values/output must be matching nonempty 1-D arrays and divisor a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_divide_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(divisor.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideF32, RelionDivideF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_batched_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[index / row_size];
}

ffi::Error RelionDivideBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto divisor_dims = divisor.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        divisor_dims.size() != 1 || divisor_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: expected values/output (B,N) and divisor (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: launch grid exceeds CUDA limit");
    relion_divide_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(divisor.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideBatchedF32, RelionDivideBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanF32, RelionCubSortScanF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanBatchedF32, RelionCubSortScanBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubPositiveSortScanF32, RelionCubPositiveSortScanF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

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

ffi::Error RelionBprefOperandsF32Impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    int64_t arithmetic_variant,
    ffi::AnyBuffer images,
    ffi::AnyBuffer ctf,
    ffi::AnyBuffer minvsigma2,
    ffi::AnyBuffer posterior_over_weight_norm,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::Result<ffi::AnyBuffer> numerator,
    ffi::Result<ffi::AnyBuffer> denominator,
    ffi::Result<ffi::AnyBuffer> translated,
    ffi::Result<ffi::AnyBuffer> weighted_ctf)
{
    if (images.element_type() != ffi::DataType::C64 ||
        numerator->element_type() != ffi::DataType::C64 ||
        translated->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: complex inputs/outputs must be C64");
    if (ctf.element_type() != ffi::DataType::F32 ||
        minvsigma2.element_type() != ffi::DataType::F32 ||
        posterior_over_weight_norm.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        denominator->element_type() != ffi::DataType::F32 ||
        weighted_ctf->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: scalar inputs/denominator must be F32");
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: pixel indices must be S32");
    if (image_h <= 0 || image_half_width <= 0)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: image dimensions must be positive");
    if (arithmetic_variant < 0 || arithmetic_variant > 23)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: arithmetic variant must be in [0,23]");

    auto image_dims = images.dimensions();
    auto ctf_dims = ctf.dimensions();
    auto noise_dims = minvsigma2.dimensions();
    auto posterior_dims = posterior_over_weight_norm.dimensions();
    auto translation_dims = translation_angles.dimensions();
    auto pixel_dims = pixel_indices.dimensions();
    auto numerator_dims = numerator->dimensions();
    auto denominator_dims = denominator->dimensions();
    auto translated_dims = translated->dimensions();
    auto weighted_ctf_dims = weighted_ctf->dimensions();
    if (image_dims.size() != 2)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: images must have shape (B,P)");
    if (ctf_dims.size() != 2 || noise_dims.size() != 2 ||
        ctf_dims[0] != image_dims[0] || ctf_dims[1] != image_dims[1] ||
        noise_dims[0] != image_dims[0] || noise_dims[1] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: ctf/minvsigma2 must match images");
    if (translation_dims.size() != 2 || translation_dims[1] != 2)
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: translation angles must have shape (T,2)");
    if (posterior_dims.size() != 2 || posterior_dims[0] != image_dims[0] ||
        posterior_dims[1] != translation_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: posterior must have shape (B,T)");
    if (pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: pixel indices must have shape (P,)");
    if (numerator_dims.size() != 2 || denominator_dims.size() != 2 ||
        translated_dims.size() != 2 || weighted_ctf_dims.size() != 2 ||
        numerator_dims[0] != image_dims[0] * translation_dims[0] ||
        numerator_dims[1] != image_dims[1] ||
        denominator_dims[0] != numerator_dims[0] ||
        denominator_dims[1] != numerator_dims[1] ||
        translated_dims[0] != numerator_dims[0] ||
        translated_dims[1] != numerator_dims[1] ||
        weighted_ctf_dims[0] != numerator_dims[0] ||
        weighted_ctf_dims[1] != numerator_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionBprefOperandsF32: outputs must have shape (B*T,P)");

    cudaError_t err = launch_relion_bpref_operands_f32(
        stream,
        reinterpret_cast<const float2*>(images.untyped_data()),
        static_cast<const float*>(ctf.untyped_data()),
        static_cast<const float*>(minvsigma2.untyped_data()),
        static_cast<const float*>(posterior_over_weight_norm.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        reinterpret_cast<float2*>(numerator->untyped_data()),
        static_cast<float*>(denominator->untyped_data()),
        reinterpret_cast<float2*>(translated->untyped_data()),
        static_cast<float*>(weighted_ctf->untyped_data()),
        image_dims[0],
        translation_dims[0],
        image_dims[1],
        static_cast<int>(image_h),
        static_cast<int>(image_half_width),
        static_cast<int>(arithmetic_variant));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionBprefOperandsF32, RelionBprefOperandsF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Attr<int64_t>("arithmetic_variant")
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
    const bool valid_projector = capacity_projector
        ? (projector_dims.size() == 3 && projector_dims[0] >= 5 &&
           projector_dims[0] <= 1025 && projector_dims[0] == projector_dims[1] &&
           (projector_dims[0] & 1) == 1 &&
           projector_dims[2] == projector_dims[0] / 2 + 1 &&
           (projector_dims[0] - 3) % (2 * projection_padding_factor) == 0)
        : (projector_dims.size() == 3 && projector_dims[0] > 0 &&
           projector_dims[1] == projector_dims[0] &&
           projector_dims[2] == projector_dims[0]);
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
        particle_tail_mask != 0);
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
        err = launch_relion_coarse_diff2_fused_translate_rectangular_f32<false>(
            stream,
            reinterpret_cast<const float2*>(reference.untyped_data()),
            reinterpret_cast<const float2*>(shifted_image.untyped_data()),
            nullptr,
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            image_dims[0], reference_dims[0], image_dims[1],
            reference_dims[1], lookup_dims[0], 0,
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

ffi::Error RelionCoarseDiff2FusedTranslateRectangularF32Impl(
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
    if (reference.element_type() != ffi::DataType::C64 ||
        image.element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2FusedTranslateRectangularF32: reference/image must be C64");
    if (translation_angles.element_type() != ffi::DataType::F32 ||
        weight.element_type() != ffi::DataType::F32 ||
        initial_diff2.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2FusedTranslateRectangularF32: angles/weight/initial/output must be F32");
    if (full_to_compact.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2FusedTranslateRectangularF32: lookup must be S32");

    const auto reference_dims = reference.dimensions();
    const auto image_dims = image.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto weight_dims = weight.dimensions();
    const auto initial_dims = initial_diff2.dimensions();
    const auto lookup_dims = full_to_compact.dimensions();
    const auto output_dims = output->dimensions();
    const int64_t expected_full_pixels =
        current_size * (current_size / 2 + 1);
    if (current_size <= 0 || reference_dims.size() != 2 ||
        image_dims.size() != 2 || angle_dims.size() != 2 ||
        weight_dims.size() != 2 || initial_dims.size() != 1 ||
        lookup_dims.size() != 1 || output_dims.size() != 3 ||
        reference_dims[0] <= 0 || reference_dims[1] <= 0 ||
        image_dims[0] <= 0 || image_dims[1] != reference_dims[1] ||
        angle_dims[0] <= 0 ||
        angle_dims[0] > kRelionCoarseDiff2BlockSize || angle_dims[1] != 2 ||
        weight_dims[0] != image_dims[0] ||
        weight_dims[1] != image_dims[1] || initial_dims[0] != image_dims[0] ||
        lookup_dims[0] != expected_full_pixels ||
        output_dims[0] != image_dims[0] ||
        output_dims[1] != reference_dims[0] ||
        output_dims[2] != angle_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2FusedTranslateRectangularF32: inconsistent operand shapes");

    const int64_t rotation_blocks =
        (reference_dims[0] + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t block_count = image_dims[0] * rotation_blocks;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCoarseDiff2FusedTranslateRectangularF32: block count exceeds CUDA grid");
    cudaError_t err =
        launch_relion_coarse_diff2_fused_translate_rectangular_f32(
            stream,
            reinterpret_cast<const float2*>(reference.untyped_data()),
            reinterpret_cast<const float2*>(image.untyped_data()),
            static_cast<const float*>(translation_angles.untyped_data()),
            static_cast<const float*>(weight.untyped_data()),
            static_cast<const float*>(initial_diff2.untyped_data()),
            static_cast<const int32_t*>(full_to_compact.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            image_dims[0],
            reference_dims[0],
            angle_dims[0],
            reference_dims[1],
            lookup_dims[0],
            static_cast<int>(current_size));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarseDiff2FusedTranslateRectangularF32,
    RelionCoarseDiff2FusedTranslateRectangularF32Impl,
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
            0);
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
            0);
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
            0);
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
            0);
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
            kRelionVdamWorkerStreams);
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
            kRelionVdamWorkerStreams);
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
            kRelionVdamWorkerStreams);
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
            kRelionVdamWorkerStreams);
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
            0);
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
            0);
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
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out,
    int reduction_mode)
{
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

    cudaError_t err = launch_relion_preprocess_real_f32(
        stream,
        static_cast<const float*>(images.untyped_data()),
        static_cast<const float*>(normalization_factors.untyped_data()),
        static_cast<const int32_t*>(integer_shifts.untyped_data()),
        static_cast<float*>(normalized_shifted_out->untyped_data()),
        static_cast<float*>(masked_out->untyped_data()),
        image_dims[0], static_cast<int>(image_dims[1]), static_cast<int>(image_dims[2]),
        radius, cosine_width, apply_mask != 0, reduction_mode);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error RelionPreprocessRealF32Impl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, 0);
}

ffi::Error RelionPreprocessRealF32NativeLaneImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, 1);
}

ffi::Error RelionPreprocessRealF32NativeAtomicImpl(
    cudaStream_t stream,
    float radius,
    float cosine_width,
    int64_t apply_mask,
    ffi::AnyBuffer images,
    ffi::AnyBuffer normalization_factors,
    ffi::AnyBuffer integer_shifts,
    ffi::Result<ffi::AnyBuffer> normalized_shifted_out,
    ffi::Result<ffi::AnyBuffer> masked_out)
{
    return RelionPreprocessRealF32ImplWithReduction(
        stream, radius, cosine_width, apply_mask, images, normalization_factors,
        integer_shifts, normalized_shifted_out, masked_out, 2);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionPreprocessRealF32, RelionPreprocessRealF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("radius")
        .Attr<float>("cosine_width")
        .Attr<int64_t>("apply_mask")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
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
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
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
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

ffi::Error BackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer img,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer /*vol_in*/,
    ffi::Result<ffi::AnyBuffer> vol_out)
{
    const int64_t n_images = rot.dimensions()[0];
    const int64_t n_pixels = image_h * image_w;
    void*       vol_ptr = vol_out->untyped_data();
    const void* img_ptr = img.untyped_data();
    const void* rot_ptr = rot.untyped_data();

    cudaError_t err;
    switch (img.element_type()) {
    case ffi::DataType::C64:
        err = launch_backproject<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4);
        break;
    case ffi::DataType::C128:
        err = launch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4);
        break;
    case ffi::DataType::F32:
        err = launch_backproject<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4);
        break;
    case ffi::DataType::F64:
        err = launch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4);
        break;
    default:
        return ffi::Error::InvalidArgument("backproject: images must be C64, C128, F32, or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error BackprojectIndexedImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    int64_t relion_fold_x,
    int64_t relion_block_topology,
    ffi::AnyBuffer img,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer /*vol_in*/,
    ffi::Result<ffi::AnyBuffer> vol_out)
{
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("backproject_indexed: pixel_indices must be int32");

    const int64_t n_images = rot.dimensions()[0];
    const int64_t n_pixels = pixel_indices.dimensions()[0];
    void*       vol_ptr = vol_out->untyped_data();
    const void* img_ptr = img.untyped_data();
    const void* pix_ptr = pixel_indices.untyped_data();
    const void* rot_ptr = rot.untyped_data();

    cudaError_t err;
    switch (img.element_type()) {
    case ffi::DataType::C64:
        err = launch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4,
            relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::C128:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4,
            relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::F32:
        err = launch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4,
            relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::F64:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4,
            relion_fold_x, relion_block_topology);
        break;
    default:
        return ffi::Error::InvalidArgument("backproject_indexed: images must be C64, C128, F32, or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

ffi::Error BackprojectIndexedSignatureImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    int64_t relion_fold_x,
    int64_t relion_block_topology,
    ffi::AnyBuffer images,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer canonical_rotation_keys,
    ffi::AnyBuffer signature_row_indices,
    ffi::AnyBuffer volume_in,
    ffi::Result<ffi::AnyBuffer> volume_out,
    ffi::Result<ffi::AnyBuffer> signature_rotation_keys,
    ffi::Result<ffi::AnyBuffer> signature_pixel_indices,
    ffi::Result<ffi::AnyBuffer> signature_row_flags,
    ffi::Result<ffi::AnyBuffer> signature_source_values,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_indices,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_coefficients,
    ffi::Result<ffi::AnyBuffer> signature_neighbor_flags,
    ffi::Result<ffi::AnyBuffer> accumulator_shadow,
    ffi::Result<ffi::AnyBuffer> operand_shadow_images,
    ffi::Result<ffi::AnyBuffer> operand_shadow_pixel_indices,
    ffi::Result<ffi::AnyBuffer> operand_shadow_rot,
    ffi::Result<ffi::AnyBuffer> operand_shadow_canonical_rotation_keys,
    ffi::Result<ffi::AnyBuffer> operand_shadow_signature_row_indices)
{
    if (images.element_type() != ffi::DataType::C64 ||
        volume_in.element_type() != ffi::DataType::C64 ||
        volume_out->element_type() != ffi::DataType::C64 ||
        accumulator_shadow->element_type() != ffi::DataType::C64 ||
        operand_shadow_images->element_type() != ffi::DataType::C64)
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: images/volumes/shadows must be complex64");
    if (rot.element_type() != ffi::DataType::F32 ||
        signature_source_values->element_type() != ffi::DataType::F32 ||
        signature_neighbor_coefficients->element_type() != ffi::DataType::F32 ||
        operand_shadow_rot->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: rotations/source/coefficients must be float32");
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
            "BackprojectIndexedSignature: signature indices/keys/flags must be int32");
    if (order != 1 || half_volume != 1 || half_image != 1 ||
        relion_fold_x != 1 || relion_block_topology != 0)
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: requires ordinary order-1 RELION x-half topology");
    if (N0 <= 0 || N0 != N1 || N1 != N2 || (N2 & 1) == 0 ||
        image_h <= 0 || image_w <= 0 || full_image_w != image_h ||
        upsampling <= 0 || max_r2_x4 < 0)
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: invalid image/volume/radius attributes");

    const auto image_dims = images.dimensions();
    const auto pixel_dims = pixel_indices.dimensions();
    const auto rot_dims = rot.dimensions();
    const auto key_dims = canonical_rotation_keys.dimensions();
    const auto selected_dims = signature_row_indices.dimensions();
    if (image_dims.size() != 2 || image_dims[0] <= 0 || image_dims[1] <= 0 ||
        pixel_dims.size() != 1 || pixel_dims[0] != image_dims[1] ||
        rot_dims.size() != 2 || rot_dims[0] != image_dims[0] || rot_dims[1] != 6 ||
        key_dims.size() != 1 || key_dims[0] != image_dims[0] ||
        selected_dims.size() != 1 || selected_dims[0] <= 0 ||
        selected_dims[0] > image_dims[0])
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: inconsistent row/pixel/rotation shapes");
    const int64_t n_rows = image_dims[0];
    const int64_t n_pixels = image_dims[1];
    const int64_t n_signature_rows = selected_dims[0];
    const int64_t volume_size = N0 * N1 * (N2 / 2 + 1);

    auto has_shape = [](auto dims, int64_t d0, int64_t d1, int64_t d2) {
        if (d2 > 0)
            return dims.size() == 3 && dims[0] == d0 && dims[1] == d1 && dims[2] == d2;
        if (d1 > 0)
            return dims.size() == 2 && dims[0] == d0 && dims[1] == d1;
        return dims.size() == 1 && dims[0] == d0;
    };
    if (!has_shape(volume_in.dimensions(), volume_size, 0, 0) ||
        !has_shape(volume_out->dimensions(), volume_size, 0, 0) ||
        !has_shape(accumulator_shadow->dimensions(), volume_size, 0, 0) ||
        !has_shape(signature_rotation_keys->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_pixel_indices->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_row_flags->dimensions(), n_signature_rows, n_pixels, 0) ||
        !has_shape(signature_source_values->dimensions(), n_signature_rows, n_pixels, 5) ||
        !has_shape(signature_neighbor_indices->dimensions(), n_signature_rows, n_pixels, 8) ||
        !has_shape(signature_neighbor_coefficients->dimensions(), n_signature_rows, n_pixels, 8) ||
        !has_shape(signature_neighbor_flags->dimensions(), n_signature_rows, n_pixels, 8) ||
        !has_shape(operand_shadow_images->dimensions(), n_rows, n_pixels, 0) ||
        !has_shape(operand_shadow_pixel_indices->dimensions(), n_pixels, 0, 0) ||
        !has_shape(operand_shadow_rot->dimensions(), n_rows, 6, 0) ||
        !has_shape(operand_shadow_canonical_rotation_keys->dimensions(), n_rows, 0, 0) ||
        !has_shape(operand_shadow_signature_row_indices->dimensions(), n_signature_rows, 0, 0))
        return ffi::Error::InvalidArgument(
            "BackprojectIndexedSignature: output/shadow shapes are inconsistent");

    cudaError_t err = launch_backproject_indexed_with_signature(
        stream,
        static_cast<float*>(volume_out->untyped_data()),
        static_cast<const float*>(images.untyped_data()),
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
        static_cast<float*>(accumulator_shadow->untyped_data()),
        static_cast<float*>(operand_shadow_images->untyped_data()),
        static_cast<int32_t*>(operand_shadow_pixel_indices->untyped_data()),
        static_cast<float*>(operand_shadow_rot->untyped_data()),
        static_cast<int32_t*>(operand_shadow_canonical_rotation_keys->untyped_data()),
        static_cast<int32_t*>(operand_shadow_signature_row_indices->untyped_data()),
        n_rows, n_signature_rows, n_pixels,
        image_h, image_w, N0, N1, N2, upsampling, max_r2_x4);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

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
    Backproject, BackprojectImpl,
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
        .Arg<ffi::AnyBuffer>()           /* img    */
        .Arg<ffi::AnyBuffer>()           /* rot    */
        .Arg<ffi::AnyBuffer>()           /* vol_in */
        .Ret<ffi::AnyBuffer>()           /* vol_out (aliased with vol_in) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BackprojectIndexed, BackprojectIndexedImpl,
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
        .Attr<int64_t>("relion_fold_x")
        .Attr<int64_t>("relion_block_topology")
        .Arg<ffi::AnyBuffer>()           /* img           */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot           */
        .Arg<ffi::AnyBuffer>()           /* vol_in        */
        .Ret<ffi::AnyBuffer>()           /* vol_out (aliased with vol_in) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BackprojectIndexedSignature, BackprojectIndexedSignatureImpl,
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
        .Attr<int64_t>("relion_fold_x")
        .Attr<int64_t>("relion_block_topology")
        .Arg<ffi::AnyBuffer>()           /* images */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot */
        .Arg<ffi::AnyBuffer>()           /* canonical_rotation_keys */
        .Arg<ffi::AnyBuffer>()           /* signature_row_indices */
        .Arg<ffi::AnyBuffer>()           /* volume_in */
        .Ret<ffi::AnyBuffer>()           /* volume_out (aliased) */
        .Ret<ffi::AnyBuffer>()           /* signature_rotation_keys */
        .Ret<ffi::AnyBuffer>()           /* signature_pixel_indices */
        .Ret<ffi::AnyBuffer>()           /* signature_row_flags */
        .Ret<ffi::AnyBuffer>()           /* signature_source_values */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_indices */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_coefficients */
        .Ret<ffi::AnyBuffer>()           /* signature_neighbor_flags */
        .Ret<ffi::AnyBuffer>()           /* accumulator_shadow */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_images */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_pixel_indices */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_rot */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_canonical_rotation_keys */
        .Ret<ffi::AnyBuffer>()           /* operand_shadow_signature_row_indices */
);

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

ffi::Error ProjectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    int64_t relion_texture_interp,
    ffi::AnyBuffer vol,
    ffi::AnyBuffer rot,
    ffi::Result<ffi::AnyBuffer> img_out)
{
    const int64_t n_images = rot.dimensions()[0];
    const int64_t n_pixels = image_h * image_w;
    const void* vol_ptr = vol.untyped_data();
    const void* rot_ptr = rot.untyped_data();
    void*       img_ptr = img_out->untyped_data();

    cudaError_t err;
    switch (vol.element_type()) {
    case ffi::DataType::C64:
        if (relion_texture_interp && order == 1 && !half_volume) {
            err = launch_project_texture_float(
                stream, (const float*)vol_ptr, (float*)img_ptr, (const float*)rot_ptr,
                n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
                half_image, full_image_w, max_r2_x4);
        } else {
            err = launch_project<float>(
                stream, (const float*)vol_ptr, (float*)img_ptr, (const float*)rot_ptr,
                n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
                order, half_volume, half_image, full_image_w, max_r2_x4);
        }
        break;
    case ffi::DataType::C128:
        if (relion_texture_interp && order == 1 && !half_volume) {
            err = launch_project_texture_double(
                stream, (const double*)vol_ptr, (double*)img_ptr, (const double*)rot_ptr,
                n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
                half_image, full_image_w, max_r2_x4);
        } else {
            err = launch_project<double>(
                stream, (const double*)vol_ptr, (double*)img_ptr, (const double*)rot_ptr,
                n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
                order, half_volume, half_image, full_image_w, max_r2_x4);
        }
        break;
    default:
        return ffi::Error::InvalidArgument("project: volume must be C64 or C128");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

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

ffi::Error ProjectIndexedImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer vol,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::Result<ffi::AnyBuffer> img_out)
{
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("project_indexed: pixel_indices must be int32");

    const int64_t n_images = rot.dimensions()[0];
    const int64_t n_pixels = pixel_indices.dimensions()[0];
    const void* vol_ptr = vol.untyped_data();
    const void* pix_ptr = pixel_indices.untyped_data();
    const void* rot_ptr = rot.untyped_data();
    void*       img_ptr = img_out->untyped_data();

    cudaError_t err;
    switch (vol.element_type()) {
    case ffi::DataType::C64:
        err = launch_project_indexed<float>(
            stream, (const float*)vol_ptr, (float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, max_r2_x4);
        break;
    case ffi::DataType::C128:
        err = launch_project_indexed<double>(
            stream, (const double*)vol_ptr, (double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, max_r2_x4);
        break;
    default:
        return ffi::Error::InvalidArgument("project_indexed: volume must be C64 or C128");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Project, ProjectImpl,
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
        .Attr<int64_t>("relion_texture_interp")
        .Arg<ffi::AnyBuffer>()           /* vol     */
        .Arg<ffi::AnyBuffer>()           /* rot     */
        .Ret<ffi::AnyBuffer>()           /* img_out */
);

ffi::Error RelionWavgRotationAtomicF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("RelionWavgRotationAtomicF32: need F32 buffers");
    const auto dims = terms.dimensions();
    const auto output_dims = output->dimensions();
    if (dims.size() != 3 || output_dims.size() != 2 ||
        output_dims[0] != dims[0] || output_dims[1] != dims[2])
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicF32: expected terms[B,R,P] and output[B,P]");
    if (dims[0] <= 0 || dims[0] > 65535 || dims[1] <= 0 ||
        dims[1] > std::numeric_limits<int>::max() || dims[2] <= 0 ||
        dims[2] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("RelionWavgRotationAtomicF32: dimensions exceed CUDA grid bounds");
    cudaError_t err = launch_relion_wavg_rotation_atomic_f32(
        stream,
        static_cast<const float*>(terms.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        dims[0],
        dims[1],
        dims[2]);
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionWavgRotationAtomicF32, RelionWavgRotationAtomicF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

ffi::Error RelionWavgRotationAtomicAddF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer terms,
    ffi::AnyBuffer accumulator_in,
    ffi::Result<ffi::AnyBuffer> accumulator_out)
{
    if (terms.element_type() != ffi::DataType::F32 ||
        accumulator_in.element_type() != ffi::DataType::F32 ||
        accumulator_out->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument("RelionWavgRotationAtomicAddF32: need F32 buffers");
    const auto dims = terms.dimensions();
    const auto accumulator_dims = accumulator_in.dimensions();
    const auto output_dims = accumulator_out->dimensions();
    if (dims.size() != 3 || accumulator_dims.size() != 2 || output_dims.size() != 2 ||
        output_dims[0] != accumulator_dims[0] || output_dims[1] != accumulator_dims[1] ||
        accumulator_dims[0] != dims[0] || accumulator_dims[1] != dims[2])
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicAddF32: expected terms[B,R,P] and accumulator[B,P]");
    if (dims[0] <= 0 || dims[0] > 65535 || dims[1] <= 0 ||
        dims[1] > std::numeric_limits<int>::max() || dims[2] <= 0 ||
        dims[2] > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionWavgRotationAtomicAddF32: dimensions exceed CUDA grid bounds");
    cudaError_t err = launch_relion_wavg_rotation_atomic_add_f32(
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
    RelionWavgRotationAtomicAddF32, RelionWavgRotationAtomicAddF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProjectIndexed, ProjectIndexedImpl,
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
        .Arg<ffi::AnyBuffer>()           /* vol           */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot           */
        .Ret<ffi::AnyBuffer>()           /* img_out       */
);


/* ── Batched FFI handlers ────────────────────────────────────────── */

ffi::Error BatchBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    ffi::AnyBuffer imgs,       /* (batch, n_images, n_pixels) */
    ffi::AnyBuffer rot,        /* (n_images, 6) */
    ffi::AnyBuffer /*vols_in*/,
    ffi::Result<ffi::AnyBuffer> vols_out)
{
    /* vols shape: (batch, vol_flat_size).  imgs shape: (batch, n_images, n_pixels). */
    const int64_t batch_size = vols_out->dimensions()[0];
    const int64_t n_images   = rot.dimensions()[0];
    const int64_t n_pixels   = image_h * image_w;
    void*       vol_ptr = vols_out->untyped_data();
    const void* img_ptr = imgs.untyped_data();
    const void* rot_ptr = rot.untyped_data();

    cudaError_t err;
    switch (imgs.element_type()) {
    case ffi::DataType::C64:
        err = launch_batch_backproject<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const float*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4);
        break;
    case ffi::DataType::C128:
        err = launch_batch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4);
        break;
    case ffi::DataType::F32:
        err = launch_batch_backproject<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const float*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4);
        break;
    case ffi::DataType::F64:
        err = launch_batch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4);
        break;
    default:
        return ffi::Error::InvalidArgument("batch_backproject: images must be C64, C128, F32, or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BatchBackproject, BatchBackprojectImpl,
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
        .Arg<ffi::AnyBuffer>()           /* imgs     */
        .Arg<ffi::AnyBuffer>()           /* rot      */
        .Arg<ffi::AnyBuffer>()           /* vols_in  */
        .Ret<ffi::AnyBuffer>()           /* vols_out (aliased) */
);

ffi::Error BatchBackprojectIndexedImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    int64_t relion_fold_x,
    int64_t relion_block_topology,
    ffi::AnyBuffer imgs,          /* (batch, n_images, n_pixels) */
    ffi::AnyBuffer pixel_indices, /* (n_pixels,) */
    ffi::AnyBuffer rot,           /* (n_images, 6) */
    ffi::AnyBuffer /*vols_in*/,
    ffi::Result<ffi::AnyBuffer> vols_out)
{
    if (pixel_indices.element_type() != ffi::DataType::S32)
        return ffi::Error::InvalidArgument("batch_backproject_indexed: pixel_indices must be int32");

    const int64_t batch_size = vols_out->dimensions()[0];
    const int64_t n_images   = rot.dimensions()[0];
    const int64_t n_pixels   = pixel_indices.dimensions()[0];
    void*       vol_ptr = vols_out->untyped_data();
    const void* img_ptr = imgs.untyped_data();
    const void* pix_ptr = pixel_indices.untyped_data();
    const void* rot_ptr = rot.untyped_data();

    cudaError_t err;
    switch (imgs.element_type()) {
    case ffi::DataType::C64:
        err = launch_batch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/0,
            max_r2_x4, relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::C128:
        err = launch_batch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/0,
            max_r2_x4, relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::F32:
        err = launch_batch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/1,
            max_r2_x4, relion_fold_x, relion_block_topology);
        break;
    case ffi::DataType::F64:
        err = launch_batch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, /*real_data=*/1,
            max_r2_x4, relion_fold_x, relion_block_topology);
        break;
    default:
        return ffi::Error::InvalidArgument("batch_backproject_indexed: images must be C64, C128, F32, or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BatchBackprojectIndexed, BatchBackprojectIndexedImpl,
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
        .Attr<int64_t>("relion_fold_x")
        .Attr<int64_t>("relion_block_topology")
        .Arg<ffi::AnyBuffer>()           /* imgs          */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot           */
        .Arg<ffi::AnyBuffer>()           /* vols_in       */
        .Ret<ffi::AnyBuffer>()           /* vols_out (aliased) */
);

/* ================================================================== */
/*              C-linkage API  (ctypes / benchmarks)                   */
/* ================================================================== */

extern "C" {

int backproject_c(
    float* vol, const float* img, const float* rot,
    int n_images, int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2, int upsampling,
    float center, int order, int half_volume, int half_image,
    int full_image_w, cudaStream_t s)
{
    return launch_backproject<float>(
        s, vol, img, rot, n_images, n_pixels, image_h, image_w,
        N0, N1, N2, upsampling, order, half_volume, half_image, full_image_w)
        != cudaSuccess ? -1 : 0;
}

int project_c(
    const float* vol, float* img, const float* rot,
    int n_images, int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2, int upsampling,
    float center, int order, int half_volume, int half_image,
    int full_image_w, cudaStream_t s)
{
    return launch_project<float>(
        s, vol, img, rot, n_images, n_pixels, image_h, image_w,
        N0, N1, N2, upsampling, order, half_volume, half_image, full_image_w)
        != cudaSuccess ? -1 : 0;
}

float benchmark_backproject_c(
    float* vol, const float* img, const float* rot,
    int n_images, int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2, int upsampling,
    float center, int order, int half_volume, int half_image,
    int full_image_w, int n_iters)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    backproject_c(vol,img,rot,n_images,n_pixels,image_h,image_w,
                  N0,N1,N2,upsampling,center,order,half_volume,
                  half_image,full_image_w,0);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; i++)
        backproject_c(vol,img,rot,n_images,n_pixels,image_h,image_w,
                      N0,N1,N2,upsampling,center,order,half_volume,
                      half_image,full_image_w,0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

float benchmark_project_c(
    const float* vol, float* img, const float* rot,
    int n_images, int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2, int upsampling,
    float center, int order, int half_volume, int half_image,
    int full_image_w, int n_iters)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    project_c(vol,img,rot,n_images,n_pixels,image_h,image_w,
              N0,N1,N2,upsampling,center,order,half_volume,
              half_image,full_image_w,0);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; i++)
        project_c(vol,img,rot,n_images,n_pixels,image_h,image_w,
                  N0,N1,N2,upsampling,center,order,half_volume,
                  half_image,full_image_w,0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

} /* extern "C" — close to allow template definitions */

/* =========================================================================
 * Interleaved batch backproject — output layout (n_voxels, batch_size)
 * instead of (batch_size, n_voxels).  All batch entries for the same voxel
 * are contiguous in memory, giving ~30× better L2 cache utilization when
 * batch_size is large (e.g., 210 PPCA upper-tri channels).
 *
 * REAL_DATA only (float atomicAdd).  HALF_VOL + trilinear (ORDER=1).
 * Simplified: no CONJ_MODE optimization, no HALF_IMG support.
 * ========================================================================= */

template <typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
batch_backproject_interleaved_kernel(
    T*       __restrict__ vols,       /* output: (vol_stride, batch_size) interleaved */
    const T* __restrict__ imgs,       /* input:  (batch_size, n_images, n_pixels)     */
    const T* __restrict__ rot,        /* (n_images, 6)                                */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling,
    int vol_stride,    /* N0 * N1 * N2_eff */
    int n_images,
    int batch_size,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* Compute freq coords (row-major, same as standard kernel) */
    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    const T k1 = (T)(k1_idx - image_w / 2) * upsampling;

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;
    const T g2 = rk2 + c2;

    /* HALF_VOL trilinear scatter with interleaved output */
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);

    if (g0 < (T)-1 || g0 >= (T)N0 ||
        g1 < (T)-1 || g1 >= (T)N1 ||
        g2 < (T)-1 || g2 >= (T)N2_full) return;

    const int b0 = floor_int(g0);
    const int b1 = floor_int(g1);
    const int b2 = floor_int(g2);
    const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2 - (T)b2;
    const T w0[2] = {(T)1 - f0, f0};
    const T w1[2] = {(T)1 - f1, f1};
    const T w2[2] = {(T)1 - f2, f2};

    const int spatial_stride1 = N2_eff;
    const int spatial_stride0 = N1 * N2_eff;
    const int img_stride = n_images * n_pixels;

    /* For each trilinear neighbor, scatter ALL batch entries with one
     * contiguous write burst (batch entries are adjacent in memory). */
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
                const int j2 = b2 + d2;
                if ((unsigned)j2 >= (unsigned)N2_full) continue;
                const int kz = j2 - ic2;
                const T w = ww * w2[d2];

                /* Hermitian fold for half-volume */
                int sj0 = j0, sj1 = j1;
                int hkz;
                if (kz >= 0) {
                    hkz = kz;
                } else if ((N2_full & 1) == 0 && -kz == ic2) {
                    hkz = ic2;
                } else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                    /* Real data: no conjugation needed */
                }
                if (hkz > ic2) continue;

                /* Interleaved offset: voxel_idx * batch_size + b */
                const int voxel_idx = sj0 * spatial_stride0 + sj1 * spatial_stride1 + hkz;
                T* dst = vols + voxel_idx * batch_size;

                /* Inner loop over batch — writes are contiguous in memory! */
                for (int b = 0; b < batch_size; b++) {
                    T val = imgs[b * img_stride + img_idx * n_pixels + pix];
                    atomicAdd(&dst[b], w * val);
                }
            }
        }
    }
}


template <typename T>
cudaError_t launch_batch_backproject_interleaved(
    cudaStream_t s, T* vols, const T* imgs, const T* rot,
    int64_t batch_size, int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t max_r2_x4 = -1)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;

    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    batch_backproject_interleaved_kernel<T><<<grid, block, 0, s>>>(
        vols, imgs, rot, (int)n_pixels, (int)ih, (int)iw,
        (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups,
        vol_stride, (int)n_images, (int)batch_size, max_r2);

    return cudaGetLastError();
}


/* =========================================================================
 * Fused backproject: reads base_images (n_images, n_pixels) and
 * weight_matrix (n_images, n_channels) separately, computes
 * val = base_images[n][pix] * weight_matrix[n][ch] inside the kernel.
 *
 * Eliminates the (n_channels, n_images, n_pixels) intermediate tensor
 * which is ~3.4 GB at 256³ with 70ch × 200 images.
 *
 * Input reads:  base_images (200×65K×4 = 50 MB) + weights (200×210×4 = 168 KB)
 * vs current:   before_chunk (70×200×65K×4 = 3.4 GB)
 * = 68× less input bandwidth.
 *
 * Output: (n_voxels_half, n_channels) interleaved layout.
 * ========================================================================= */

template <typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_backproject_kernel(
    T*       __restrict__ vols,          /* (vol_stride * n_channels) interleaved */
    const T* __restrict__ base_images,   /* (n_images, n_pixels) e.g. ctf²       */
    const T* __restrict__ weight_matrix, /* (n_images, n_channels) e.g. smz_tri  */
    const T* __restrict__ rot,           /* (n_images, 6)                         */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling,
    int n_images,
    int n_channels,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    const T k1 = (T)(k1_idx - image_w / 2) * upsampling;

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    const T g0 = rk0 + c0, g1 = rk1 + c1, g2 = rk2 + c2;
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);

    if (g0 < (T)-1 || g0 >= (T)N0 ||
        g1 < (T)-1 || g1 >= (T)N1 ||
        g2 < (T)-1 || g2 >= (T)N2_full) return;

    const int b0 = floor_int(g0), b1 = floor_int(g1), b2 = floor_int(g2);
    const T f0 = g0-(T)b0, f1 = g1-(T)b1, f2 = g2-(T)b2;
    const T w0[2] = {(T)1-f0, f0}, w1[2] = {(T)1-f1, f1}, w2[2] = {(T)1-f2, f2};

    const int spatial_stride1 = N2_eff;
    const int spatial_stride0 = N1 * N2_eff;

    /* Load base pixel value ONCE */
    const T base_val = base_images[img_idx * n_pixels + pix];

    /* Pointer to this image's weight row: weight_matrix[img_idx, :] */
    const T* wt_row = weight_matrix + img_idx * n_channels;

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
                const int j2 = b2 + d2;
                if ((unsigned)j2 >= (unsigned)N2_full) continue;
                const int kz = j2 - ic2;
                const T trilinear_w = ww * w2[d2];

                int sj0 = j0, sj1 = j1;
                int hkz;
                if (kz >= 0) { hkz = kz; }
                else if ((N2_full & 1) == 0 && -kz == ic2) { hkz = ic2; }
                else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                }
                if (hkz > ic2) continue;

                const int voxel_idx = sj0 * spatial_stride0 + sj1 * spatial_stride1 + hkz;
                T* dst = vols + voxel_idx * n_channels;
                const T weighted_base = trilinear_w * base_val;

                /* Inner loop: multiply base by per-channel weight, scatter.
                 * wt_row is tiny (~840 bytes for 210 channels) → L1 cached.
                 * dst is contiguous for all channels → L2 coalesced. */
                for (int ch = 0; ch < n_channels; ch++) {
                    atomicAdd(&dst[ch], weighted_base * wt_row[ch]);
                }
            }
        }
    }
}


template <typename T>
cudaError_t launch_fused_backproject(
    cudaStream_t s, T* vols, const T* base_images, const T* weight_matrix,
    const T* rot,
    int64_t n_images, int64_t n_pixels, int64_t n_channels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t max_r2_x4 = -1)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const T c0 = (T)(N0/2), c1 = (T)(N1/2), c2 = (T)(N2/2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;

    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    fused_backproject_kernel<T><<<grid, block, 0, s>>>(
        vols, base_images, weight_matrix, rot,
        (int)n_pixels, (int)ih, (int)iw,
        (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups,
        (int)n_images, (int)n_channels, max_r2);

    return cudaGetLastError();
}


/* XLA FFI handler for fused backproject */
static ffi::Error FusedBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w, int64_t vol_n0, int64_t vol_n1,
    int64_t vol_n2, int64_t upsampling, int64_t max_r2_x4,
    ffi::AnyBuffer base_images,    /* (n_images, n_pixels) */
    ffi::AnyBuffer weight_matrix,  /* (n_images, n_channels) */
    ffi::AnyBuffer rotations,      /* (n_images, 6) */
    ffi::AnyBuffer /*vols_in*/,
    ffi::Result<ffi::AnyBuffer> vols_out)
{
    void* out_ptr = vols_out->untyped_data();
    auto vol_dtype = vols_out->element_type();

    int64_t n_images = base_images.dimensions()[0];
    int64_t n_pixels = base_images.dimensions()[1];
    int64_t n_channels = weight_matrix.dimensions()[1];

    cudaError_t err;
    if (vol_dtype == ffi::F32) {
        err = launch_fused_backproject<float>(
            stream, (float*)out_ptr,
            (const float*)base_images.untyped_data(),
            (const float*)weight_matrix.untyped_data(),
            (const float*)rotations.untyped_data(),
            n_images, n_pixels, n_channels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else if (vol_dtype == ffi::F64) {
        err = launch_fused_backproject<double>(
            stream, (double*)out_ptr,
            (const double*)base_images.untyped_data(),
            (const double*)weight_matrix.untyped_data(),
            (const double*)rotations.untyped_data(),
            n_images, n_pixels, n_channels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else {
        return ffi::Error::InvalidArgument("FusedBackproject: need F32 or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedBackproject,
                              FusedBackprojectImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("vol_n0")
                                  .Attr<int64_t>("vol_n1")
                                  .Attr<int64_t>("vol_n2")
                                  .Attr<int64_t>("upsampling")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Arg<ffi::AnyBuffer>()   /* base_images */
                                  .Arg<ffi::AnyBuffer>()   /* weight_matrix */
                                  .Arg<ffi::AnyBuffer>()   /* rotations */
                                  .Arg<ffi::AnyBuffer>()   /* vols_in (aliased) */
                                  .Ret<ffi::AnyBuffer>()); /* vols_out */


/* =========================================================================
 * Per-image backproject: output layout (n_voxels_half, n_images).
 *
 * Each image writes to its own "column" in the output volume — atomicAdds
 * from the SAME image rarely collide (sparse scatter), and different
 * images never collide (different columns).
 *
 * The output is then reduced via GEMM: (n_voxels, n_images) @ (n_images, n_channels)
 * to produce the final (n_voxels, n_channels) LHS.
 * ========================================================================= */

template <typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
per_image_backproject_kernel(
    T*       __restrict__ vols,          /* (vol_stride, n_images) interleaved */
    const T* __restrict__ base_images,   /* (n_images, n_pixels) e.g. ctf²    */
    const T* __restrict__ rot,           /* (n_images, 6)                      */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling,
    int n_images,
    T max_r2)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    const T k1 = (T)(k1_idx - image_w / 2) * upsampling;

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    const T g0 = rk0 + c0, g1 = rk1 + c1, g2 = rk2 + c2;
    const int ic2 = (int)c2;
    const int N2_full = full_z_size_from_half(N0, N1, N2_eff);

    if (g0 < (T)-1 || g0 >= (T)N0 ||
        g1 < (T)-1 || g1 >= (T)N1 ||
        g2 < (T)-1 || g2 >= (T)N2_full) return;

    const int b0 = floor_int(g0), b1 = floor_int(g1), b2 = floor_int(g2);
    const T f0 = g0-(T)b0, f1 = g1-(T)b1, f2 = g2-(T)b2;
    const T w0[2] = {(T)1-f0, f0}, w1[2] = {(T)1-f1, f1}, w2[2] = {(T)1-f2, f2};

    const int spatial_stride1 = N2_eff;
    const int spatial_stride0 = N1 * N2_eff;

    const T base_val = base_images[img_idx * n_pixels + pix];

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
                const int j2 = b2 + d2;
                if ((unsigned)j2 >= (unsigned)N2_full) continue;
                const int kz = j2 - ic2;
                const T trilinear_w = ww * w2[d2];

                int sj0 = j0, sj1 = j1;
                int hkz;
                if (kz >= 0) { hkz = kz; }
                else if ((N2_full & 1) == 0 && -kz == ic2) { hkz = ic2; }
                else {
                    sj0 = (N0 - (N0 & 1) - j0) % N0;
                    sj1 = (N1 - (N1 & 1) - j1) % N1;
                    hkz = -kz;
                }
                if (hkz > ic2) continue;

                const int voxel_idx = sj0 * spatial_stride0 + sj1 * spatial_stride1 + hkz;
                /* Per-image slot: atomicAdd only competes with the ~8 neighbors
                 * from the same image's other pixels — near-zero contention. */
                atomicAdd(&vols[voxel_idx * n_images + img_idx], trilinear_w * base_val);
            }
        }
    }
}


template <typename T>
cudaError_t launch_per_image_backproject(
    cudaStream_t s, T* vols, const T* base_images, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t max_r2_x4 = -1)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const T c0 = (T)(N0/2), c1 = (T)(N1/2), c2 = (T)(N2/2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;

    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    per_image_backproject_kernel<T><<<grid, block, 0, s>>>(
        vols, base_images, rot,
        (int)n_pixels, (int)ih, (int)iw,
        (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups,
        (int)n_images, max_r2);

    return cudaGetLastError();
}


static ffi::Error PerImageBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w, int64_t vol_n0, int64_t vol_n1,
    int64_t vol_n2, int64_t upsampling, int64_t max_r2_x4,
    ffi::AnyBuffer base_images,  /* (n_images, n_pixels) */
    ffi::AnyBuffer rotations,    /* (n_images, 6) */
    ffi::AnyBuffer /*vols_in*/,
    ffi::Result<ffi::AnyBuffer> vols_out)
{
    void* out_ptr = vols_out->untyped_data();
    auto vol_dtype = vols_out->element_type();

    int64_t n_images = base_images.dimensions()[0];
    int64_t n_pixels = base_images.dimensions()[1];

    cudaError_t err;
    if (vol_dtype == ffi::F32) {
        err = launch_per_image_backproject<float>(
            stream, (float*)out_ptr,
            (const float*)base_images.untyped_data(),
            (const float*)rotations.untyped_data(),
            n_images, n_pixels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else if (vol_dtype == ffi::F64) {
        err = launch_per_image_backproject<double>(
            stream, (double*)out_ptr,
            (const double*)base_images.untyped_data(),
            (const double*)rotations.untyped_data(),
            n_images, n_pixels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else {
        return ffi::Error::InvalidArgument("PerImageBackproject: need F32 or F64");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PerImageBackproject,
                              PerImageBackprojectImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("vol_n0")
                                  .Attr<int64_t>("vol_n1")
                                  .Attr<int64_t>("vol_n2")
                                  .Attr<int64_t>("upsampling")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());


extern "C" {

/* XLA FFI handler for interleaved batch backproject */
static ffi::Error BatchBackprojectInterleavedImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w, int64_t vol_n0, int64_t vol_n1,
    int64_t vol_n2, int64_t upsampling, int64_t max_r2_x4,
    ffi::AnyBuffer images,
    ffi::AnyBuffer rotations,
    ffi::AnyBuffer /*vols_in*/,
    ffi::Result<ffi::AnyBuffer> vols_out)
{
    /* vols_in is aliased to vols_out via input_output_aliases={2:0} */
    void* out_ptr = vols_out->untyped_data();
    auto vol_dtype = vols_out->element_type();

    auto dims = images.dimensions();
    int64_t batch_size = dims[0];
    int64_t n_images = dims[1];
    int64_t n_pixels = dims[2];

    cudaError_t err;
    if (vol_dtype == ffi::F32) {
        err = launch_batch_backproject_interleaved<float>(
            stream,
            static_cast<float*>(out_ptr),
            static_cast<const float*>(images.untyped_data()),
            static_cast<const float*>(rotations.untyped_data()),
            batch_size, n_images, n_pixels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else if (vol_dtype == ffi::F64) {
        err = launch_batch_backproject_interleaved<double>(
            stream,
            static_cast<double*>(out_ptr),
            static_cast<const double*>(images.untyped_data()),
            static_cast<const double*>(rotations.untyped_data()),
            batch_size, n_images, n_pixels,
            image_h, image_w, vol_n0, vol_n1, vol_n2,
            upsampling, max_r2_x4);
    } else {
        return ffi::Error::InvalidArgument(
            "BatchBackprojectInterleaved: unsupported dtype (need F32 or F64)");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(BatchBackprojectInterleaved,
                              BatchBackprojectInterleavedImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("image_h")
                                  .Attr<int64_t>("image_w")
                                  .Attr<int64_t>("vol_n0")
                                  .Attr<int64_t>("vol_n1")
                                  .Attr<int64_t>("vol_n2")
                                  .Attr<int64_t>("upsampling")
                                  .Attr<int64_t>("max_r2_x4")
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());


} /* extern "C" */
