/*
 * CUDA Backprojector / Projector  — v5
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
 * v5 changes:
 *   - Per-axis centers (correct for non-cubic volumes)
 *   - Conjugate scatter uses -rk shortcut (saves 6 FMA for 99.9% of pixels)
 *   - __launch_bounds__ for better register allocation
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

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

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
static __device__ __forceinline__ int round_int(float  x) { return (int)rintf(x); }
static __device__ __forceinline__ int round_int(double x) { return (int)rint(x); }

#define BLOCK_SIZE 256

/* ================================================================== */
/*   Device helpers: scatter one value into volume at rotated coords   */
/* ================================================================== */

/* scatter_nearest: atomicAdd one complex value at the nearest voxel.
 *
 * HALF_VOL: uses full centered-volume coords for bounds checking.
 * Restrict approach: only scatter voxels with kz >= 0 (kept half),
 * drop kz < 0.  Equivalent to scattering into full 3D grid then
 * restricting to rfft3 storage.  Halves atomicAdd count.
 */
template <typename T, bool HALF_VOL>
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
        const int N2_full = 2 * ic2;
        const T g2_full = rk2 + c2;
        const int i0 = round_int(g0);
        const int i1 = round_int(g1);
        const int i2 = round_int(g2_full);
        if ((unsigned)i0 >= (unsigned)N0 ||
            (unsigned)i1 >= (unsigned)N1 ||
            (unsigned)i2 >= (unsigned)N2_full) return;
        const int kz = i2 - ic2;
        if (kz < 0) return;  /* drop kz < 0: restrict */
        const int off = (i0 * stride0 + i1 * stride1 + kz) * 2;
        atomicAdd(&vol[off],     val_re);
        atomicAdd(&vol[off + 1], val_im);
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
    const int off = (i0 * stride0 + i1 * stride1 + i2) * 2;
    atomicAdd(&vol[off],     val_re);
    atomicAdd(&vol[off + 1], val_im);
}

/* scatter_trilinear: atomicAdd one complex value at 8 trilinear neighbors.
 *
 * HALF_VOL: uses full centered-volume coords for bounds checking.
 * Restrict approach: only scatter neighbors with kz >= 0, drop kz < 0.
 * Equivalent to scattering into full 3D grid then restricting to rfft3
 * storage.  Halves the atomicAdd count vs Hermitian folding.
 */
template <typename T, bool HALF_VOL>
static __device__ __forceinline__ void scatter_trilinear(
    T* __restrict__ vol,
    T rk0, T rk1, T rk2, T val_re, T val_im,
    T c0, T c1, T c2,
    int N0, int N1, int N2_eff, int stride0, int stride1)
{
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;

    if (HALF_VOL) {
        const int ic2 = (int)c2;
        const int N2_full = 2 * ic2;
        const T g2_full = rk2 + c2;

        if (g0 < (T)-1 || g0 >= (T)N0 ||
            g1 < (T)-1 || g1 >= (T)N1 ||
            g2_full < (T)-1 || g2_full >= (T)N2_full) return;

        const int b0 = floor_int(g0);
        const int b1 = floor_int(g1);
        const int b2 = floor_int(g2_full);
        const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2_full - (T)b2;
        const T w0[2] = {(T)1 - f0, f0};
        const T w1[2] = {(T)1 - f1, f1};
        const T w2[2] = {(T)1 - f2, f2};

        /* Restrict approach: only scatter neighbors with kz >= 0.
         * Neighbors with kz < 0 are dropped — equivalent to scattering
         * into the full 3D grid then restricting to the kz >= 0 half.
         * This halves the atomicAdd count compared to Hermitian folding. */
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
                    if (kz < 0) continue;  /* drop kz < 0: restrict */
                    const T w = ww * w2[d2];
                    const int off = (j0 * stride0 + j1 * stride1 + kz) * 2;
                    atomicAdd(&vol[off],     w * val_re);
                    atomicAdd(&vol[off + 1], w * val_im);
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
                const int off = (j0 * stride0 + j1 * stride1 + j2) * 2;
                atomicAdd(&vol[off],     w * val_re);
                atomicAdd(&vol[off + 1], w * val_im);
            }
        }
    }
}

/* ================================================================== */
/*                  Backproject kernel                                 */
/* ================================================================== */

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
backproject_kernel(
    T*       __restrict__ vol,
    const T* __restrict__ img,
    const T* __restrict__ rot,   /* (n_images, 6) */
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w)
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

    /* Rotate  (cz=0  →  only 6 elements) */
    const T rk0 = k0 * R[0] + k1 * R[3];
    const T rk1 = k0 * R[1] + k1 * R[4];
    const T rk2 = k0 * R[2] + k1 * R[5];

    /* Load pixel (vectorized) */
    using V2 = vec2_t<T>;
    V2 px = reinterpret_cast<const V2*>(img)[img_idx * n_pixels + pix];
    const T val_re = px.x;
    const T val_im = px.y;

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

    /* Primary scatter */
    if (ORDER == 0) {
        scatter_nearest<T, HALF_VOL>(vol, rk0, rk1, rk2, val_re, val_im,
                                     c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
    } else {
        scatter_trilinear<T, HALF_VOL>(vol, rk0, rk1, rk2, val_re, val_im,
                                       c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
    }

    /* Conjugate scatter for rfft non-boundary pixels.
     * Boundary: k1_idx == 0  or  k1_idx == full_image_w/2 (Nyquist, even W).
     * For non-boundary pixels, scatter conj(value) at rotated(-k0, -k1).
     * Uses the same scatter function as the primary — HALF_VOL bounds
     * checking naturally handles negative g2 (no fold needed). */
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
            if (ORDER == 0) {
                scatter_nearest<T, HALF_VOL>(vol, crk0, crk1, crk2,
                                             val_re, -val_im,
                                             c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            } else {
                scatter_trilinear<T, HALF_VOL>(vol, crk0, crk1, crk2,
                                               val_re, -val_im,
                                               c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            }
        }
    }
}

/* ================================================================== */
/*                    Project kernel                                   */
/* ================================================================== */

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_kernel(
    const T* __restrict__ vol,
    T*       __restrict__ img,
    const T* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* Row-major pixel layout */
    const int k0_idx = pix / image_w;   /* row index */
    const int k1_idx = pix % image_w;   /* col index */
    T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    T rk0 = k0 * R[0] + k1 * R[3];
    T rk1 = k0 * R[1] + k1 * R[4];
    T rk2 = k0 * R[2] + k1 * R[5];

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

    using V2 = vec2_t<T>;
    V2* img2 = reinterpret_cast<V2*>(img);
    const int img_off = img_idx * n_pixels + pix;

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
        /* N2_full = 2 * c2 for even N;  recover it for the full-vol
         * z coordinate (g2_full) and its bounds check. */
        const int ic2 = (int)c2;          /* N2/2 */
        const int N2_full = 2 * ic2;      /* == N2 for even N2 */
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
                ri = (N0 - i0) % N0;
                rj = (N1 - i1) % N1;
                rk = -kz;
                cj = true;
            }
            const int off = ri * stride0 + rj * stride1 + rk;
            V2 v = __ldg(&reinterpret_cast<const V2*>(vol)[off]);
            if (cj) v.y = -v.y;
            img2[img_off] = v;
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
            const int r0_0 = (N0 - b0) % N0,     r0_1 = (N0 - b0 - 1) % N0;
            const int r1_0 = (N1 - b1) % N1,     r1_1 = (N1 - b1 - 1) % N1;
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
                            ri = (N0 - j0) % N0;
                            rj = (N1 - j1) % N1;
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
    int64_t full_iw)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BP(O, HV, HI) \
        backproject_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vol, img, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw)

    int key = (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case 0: BP(0, false, false); break;
    case 1: BP(0, false, true);  break;
    case 2: BP(0, true,  false); break;
    case 3: BP(0, true,  true);  break;
    case 4: BP(1, false, false); break;
    case 5: BP(1, false, true);  break;
    case 6: BP(1, true,  false); break;
    case 7: BP(1, true,  true);  break;
    }
    #undef BP
    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_project(
    cudaStream_t s, const T* vol, T* img, const T* rot,
    int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define PJ(O, HV, HI) \
        project_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vol, img, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw)

    int key = (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case 0: PJ(0, false, false); break;
    case 1: PJ(0, false, true);  break;
    case 2: PJ(0, true,  false); break;
    case 3: PJ(0, true,  true);  break;
    case 4: PJ(1, false, false); break;
    case 5: PJ(1, false, true);  break;
    case 6: PJ(1, true,  false); break;
    case 7: PJ(1, true,  true);  break;
    }
    #undef PJ
    return cudaGetLastError();
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

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
batch_backproject_kernel(
    T*       __restrict__ vols,
    const T* __restrict__ imgs,
    const T* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    int vol_stride,    /* N0*N1*N2_eff (complex elements) */
    int n_images,
    int batch_size)
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

    /* Inner loop over batch — same coords, different volumes and images */
    for (int b = 0; b < batch_size; b++) {
        T* vol = vols + b * vol_stride * 2;
        V2 px = reinterpret_cast<const V2*>(imgs)[(b * img_stride) + img_idx * n_pixels + pix];

        if (ORDER == 0)
            scatter_nearest<T, HALF_VOL>(vol, rk0, rk1, rk2, px.x, px.y,
                                         c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        else
            scatter_trilinear<T, HALF_VOL>(vol, rk0, rk1, rk2, px.x, px.y,
                                           c0, c1, c2, N0, N1, N2_eff, stride0, stride1);

        if (do_conj_scatter) {
            if (ORDER == 0)
                scatter_nearest<T, HALF_VOL>(vol, crk0, crk1, crk2, px.x, -px.y,
                                             c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
            else
                scatter_trilinear<T, HALF_VOL>(vol, crk0, crk1, crk2, px.x, -px.y,
                                               c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
        }
    }
}

template <typename T, int ORDER, bool HALF_VOL, bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
batch_project_kernel(
    const T* __restrict__ vols,
    T*       __restrict__ imgs,
    const T* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int N0, int N1, int N2_eff,
    T c0, T c1, T c2,
    int upsampling, int full_image_w,
    int vol_stride,
    int n_images,
    int batch_size)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    /* Compute rotation-dependent coords once (row-major) */
    const int k0_idx = pix / image_w;   /* row index */
    const int k1_idx = pix % image_w;   /* col index */
    T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    T rk0 = k0 * R[0] + k1 * R[3];
    T rk1 = k0 * R[1] + k1 * R[4];
    T rk2 = k0 * R[2] + k1 * R[5];

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;
    const int img_stride = n_images * n_pixels;
    using V2 = vec2_t<T>;

    /* ── HALF_VOL: per-neighbor Hermitian read (precompute once, reuse) ── */
    if (HALF_VOL) {
        const T g0 = rk0 + c0;
        const T g1 = rk1 + c1;
        const int ic2 = (int)c2;
        const int N2_full = 2 * ic2;
        const T g2_full = rk2 + c2;

        if (ORDER == 0) {
            const int i0 = round_int(g0);
            const int i1 = round_int(g1);
            const int i2 = round_int(g2_full);
            const bool oob = ((unsigned)i0 >= (unsigned)N0 ||
                              (unsigned)i1 >= (unsigned)N1 ||
                              (unsigned)i2 >= (unsigned)N2_full);
            int ri = 0, rj = 0, rk = 0;
            bool cj = false;
            if (!oob) {
                const int kz = i2 - ic2;
                if (kz >= 0) {
                    ri = i0; rj = i1; rk = kz;
                } else {
                    ri = (N0 - i0) % N0;
                    rj = (N1 - i1) % N1;
                    rk = -kz;
                    cj = true;
                }
            }
            const int voff = ri * stride0 + rj * stride1 + rk;
            for (int b = 0; b < batch_size; b++) {
                V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
                if (oob) { out[pix] = make_v2((T)0, (T)0); continue; }
                V2 v = __ldg(&reinterpret_cast<const V2*>(vols + b * vol_stride * 2)[voff]);
                if (cj) v.y = -v.y;
                out[pix] = v;
            }
            return;
        }

        /* trilinear HALF_VOL — precompute neighbor info, reuse across batch */
        const bool oob = (g0 < (T)-1 || g0 >= (T)N0 ||
                          g1 < (T)-1 || g1 >= (T)N1 ||
                          g2_full < (T)-1 || g2_full >= (T)N2_full);

        if (oob) {
            for (int b = 0; b < batch_size; b++) {
                V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
                out[pix] = make_v2((T)0, (T)0);
            }
            return;
        }

        const int bb0 = floor_int(g0);
        const int bb1 = floor_int(g1);
        const int bb2 = floor_int(g2_full);
        const T f0 = g0 - (T)bb0, f1 = g1 - (T)bb1, f2 = g2_full - (T)bb2;
        const T wt0[2] = {(T)1 - f0, f0};
        const T wt1[2] = {(T)1 - f1, f1};
        const T wt2[2] = {(T)1 - f2, f2};

        const bool all_in = (bb0 >= 0 && bb0 + 1 < N0 &&
                             bb1 >= 0 && bb1 + 1 < N1 &&
                             bb2 >= 0 && bb2 + 1 < N2_full);

        if (all_in && bb2 >= ic2) {
            /* Fast path: all kz >= 0 — precompute 8 offsets + weights */
            const int kz0 = bb2 - ic2;
            int off[8]; T wt[8];
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const int idx = d0*4 + d1*2 + d2;
                        off[idx] = (bb0+d0)*stride0 + (bb1+d1)*stride1 + kz0+d2;
                        wt[idx] = wt0[d0] * wt1[d1] * wt2[d2];
                    }
                }
            }
            for (int b = 0; b < batch_size; b++) {
                V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
                const V2* vol2 = reinterpret_cast<const V2*>(vols + b * vol_stride * 2);
                T sr = 0, si = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    V2 v = __ldg(&vol2[off[i]]);
                    sr += wt[i] * v.x;
                    si += wt[i] * v.y;
                }
                out[pix] = make_v2(sr, si);
            }
        } else if (all_in && bb2 + 1 < ic2) {
            /* Fast path: all kz < 0 — Hermitian partner reads, conjugate sum */
            const int r0[2] = {(N0 - bb0) % N0, (N0 - bb0 - 1) % N0};
            const int r1[2] = {(N1 - bb1) % N1, (N1 - bb1 - 1) % N1};
            const int rk0 = ic2 - bb2, rk1 = rk0 - 1;
            int off[8]; T wt[8];
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const int idx = d0*4 + d1*2 + d2;
                        off[idx] = r0[d0]*stride0 + r1[d1]*stride1 + (d2 == 0 ? rk0 : rk1);
                        wt[idx] = wt0[d0] * wt1[d1] * wt2[d2];
                    }
                }
            }
            for (int b = 0; b < batch_size; b++) {
                V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
                const V2* vol2 = reinterpret_cast<const V2*>(vols + b * vol_stride * 2);
                T sr = 0, si = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    V2 v = __ldg(&vol2[off[i]]);
                    sr += wt[i] * v.x;
                    si += wt[i] * v.y;
                }
                out[pix] = make_v2(sr, -si);  /* conjugate the result */
            }
        } else {
            /* Slow path: boundary or mixed kz — variable-length neighbor table */
            struct { int off; T w; bool cj; } nbr[8];
            int n_nbr = 0;
            #pragma unroll
            for (int d0 = 0; d0 < 2; d0++) {
                const int j0 = bb0 + d0;
                if ((unsigned)j0 >= (unsigned)N0) continue;
                #pragma unroll
                for (int d1 = 0; d1 < 2; d1++) {
                    const int j1 = bb1 + d1;
                    if ((unsigned)j1 >= (unsigned)N1) continue;
                    const T ww = wt0[d0] * wt1[d1];
                    #pragma unroll
                    for (int d2 = 0; d2 < 2; d2++) {
                        const int j2 = bb2 + d2;
                        if ((unsigned)j2 >= (unsigned)N2_full) continue;
                        const int kz = j2 - ic2;
                        int ri, rj, rkk;
                        bool cjj = false;
                        if (kz >= 0) {
                            ri = j0; rj = j1; rkk = kz;
                        } else {
                            ri = (N0 - j0) % N0;
                            rj = (N1 - j1) % N1;
                            rkk = -kz;
                            cjj = true;
                        }
                        nbr[n_nbr].off = ri * stride0 + rj * stride1 + rkk;
                        nbr[n_nbr].w   = ww * wt2[d2];
                        nbr[n_nbr].cj  = cjj;
                        n_nbr++;
                    }
                }
            }
            for (int b = 0; b < batch_size; b++) {
                V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
                const V2* vol2 = reinterpret_cast<const V2*>(vols + b * vol_stride * 2);
                T sr = 0, si = 0;
                for (int i = 0; i < n_nbr; i++) {
                    V2 v = __ldg(&vol2[nbr[i].off]);
                    if (nbr[i].cj) v.y = -v.y;
                    sr += nbr[i].w * v.x;
                    si += nbr[i].w * v.y;
                }
                out[pix] = make_v2(sr, si);
            }
        }
        return;
    }

    /* ── Non-HALF_VOL path (unchanged) ───────────────────────────── */
    const T g0 = rk0 + c0;
    const T g1 = rk1 + c1;
    const T g2 = rk2 + c2;

    for (int b = 0; b < batch_size; b++) {
        const T* vol = vols + b * vol_stride * 2;
        V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;

        if (ORDER == 0) {
            const int i0 = round_int(g0);
            const int i1 = round_int(g1);
            const int i2 = round_int(g2);
            if ((unsigned)i0 >= (unsigned)N0 ||
                (unsigned)i1 >= (unsigned)N1 ||
                (unsigned)i2 >= (unsigned)N2_eff) {
                out[pix] = make_v2((T)0, (T)0);
                continue;
            }
            const int off = i0 * stride0 + i1 * stride1 + i2;
            V2 v = __ldg(&reinterpret_cast<const V2*>(vol)[off]);
            out[pix] = v;
            continue;
        }

        /* trilinear */
        if (g0 < (T)-1 || g0 >= (T)N0 ||
            g1 < (T)-1 || g1 >= (T)N1 ||
            g2 < (T)-1 || g2 >= (T)N2_eff) {
            out[pix] = make_v2((T)0, (T)0);
            continue;
        }

        const int b0 = floor_int(g0);
        const int b1 = floor_int(g1);
        const int b2 = floor_int(g2);
        const T f0 = g0 - (T)b0, f1 = g1 - (T)b1, f2 = g2 - (T)b2;
        const T w0[2] = {(T)1 - f0, f0};
        const T w1[2] = {(T)1 - f1, f1};
        const T w2[2] = {(T)1 - f2, f2};

        T sum_re = 0, sum_im = 0;

        if (b0 >= 0 && b0 + 1 < N0 &&
            b1 >= 0 && b1 + 1 < N1 &&
            b2 >= 0 && b2 + 1 < N2_eff) {
            const V2* vol2 = reinterpret_cast<const V2*>(vol);
            const V2 v000 = __ldg(&vol2[b0 * stride0 + b1 * stride1 + b2]);
            const V2 v001 = __ldg(&vol2[b0 * stride0 + b1 * stride1 + b2 + 1]);
            const V2 v010 = __ldg(&vol2[b0 * stride0 + (b1+1) * stride1 + b2]);
            const V2 v011 = __ldg(&vol2[b0 * stride0 + (b1+1) * stride1 + b2 + 1]);
            const V2 v100 = __ldg(&vol2[(b0+1) * stride0 + b1 * stride1 + b2]);
            const V2 v101 = __ldg(&vol2[(b0+1) * stride0 + b1 * stride1 + b2 + 1]);
            const V2 v110 = __ldg(&vol2[(b0+1) * stride0 + (b1+1) * stride1 + b2]);
            const V2 v111 = __ldg(&vol2[(b0+1) * stride0 + (b1+1) * stride1 + b2 + 1]);
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

        out[pix] = make_v2(sum_re, sum_im);
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
    int64_t full_iw)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BBP(O, HV, HI) \
        batch_backproject_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vols, imgs, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
            vol_stride, (int)n_images, (int)batch_size)

    int key = (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case 0: BBP(0, false, false); break;
    case 1: BBP(0, false, true);  break;
    case 2: BBP(0, true,  false); break;
    case 3: BBP(0, true,  true);  break;
    case 4: BBP(1, false, false); break;
    case 5: BBP(1, false, true);  break;
    case 6: BBP(1, true,  false); break;
    case 7: BBP(1, true,  true);  break;
    }
    #undef BBP
    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_batch_project(
    cudaStream_t s, const T* vols, T* imgs, const T* rot,
    int64_t batch_size, int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BPJ(O, HV, HI) \
        batch_project_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vols, imgs, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
            vol_stride, (int)n_images, (int)batch_size)

    int key = (order ? 4 : 0) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case 0: BPJ(0, false, false); break;
    case 1: BPJ(0, false, true);  break;
    case 2: BPJ(0, true,  false); break;
    case 3: BPJ(0, true,  true);  break;
    case 4: BPJ(1, false, false); break;
    case 5: BPJ(1, false, true);  break;
    case 6: BPJ(1, true,  false); break;
    case 7: BPJ(1, true,  true);  break;
    }
    #undef BPJ
    return cudaGetLastError();
}


/* ================================================================== */
/*                    XLA  FFI  handlers                               */
/* ================================================================== */

ffi::Error BackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
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
            order, half_volume, half_image, full_image_w);
        break;
    case ffi::DataType::C128:
        err = launch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w);
        break;
    default:
        return ffi::Error::InvalidArgument("backproject: images must be C64 or C128");
    }
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
        .Arg<ffi::AnyBuffer>()           /* img    */
        .Arg<ffi::AnyBuffer>()           /* rot    */
        .Arg<ffi::AnyBuffer>()           /* vol_in */
        .Ret<ffi::AnyBuffer>()           /* vol_out (aliased with vol_in) */
);

ffi::Error ProjectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
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
        err = launch_project<float>(
            stream, (const float*)vol_ptr, (float*)img_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w);
        break;
    case ffi::DataType::C128:
        err = launch_project<double>(
            stream, (const double*)vol_ptr, (double*)img_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w);
        break;
    default:
        return ffi::Error::InvalidArgument("project: volume must be C64 or C128");
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
        .Arg<ffi::AnyBuffer>()           /* vol     */
        .Arg<ffi::AnyBuffer>()           /* rot     */
        .Ret<ffi::AnyBuffer>()           /* img_out */
);


/* ── Batched FFI handlers ────────────────────────────────────────── */

ffi::Error BatchBackprojectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
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
            upsampling, order, half_volume, half_image, full_image_w);
        break;
    case ffi::DataType::C128:
        err = launch_batch_backproject<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w);
        break;
    default:
        return ffi::Error::InvalidArgument("batch_backproject: images must be C64 or C128");
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
        .Arg<ffi::AnyBuffer>()           /* imgs     */
        .Arg<ffi::AnyBuffer>()           /* rot      */
        .Arg<ffi::AnyBuffer>()           /* vols_in  */
        .Ret<ffi::AnyBuffer>()           /* vols_out (aliased) */
);

ffi::Error BatchProjectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    ffi::AnyBuffer vols,
    ffi::AnyBuffer rot,
    ffi::Result<ffi::AnyBuffer> imgs_out)
{
    const int64_t batch_size = vols.dimensions()[0];
    const int64_t n_images   = rot.dimensions()[0];
    const int64_t n_pixels   = image_h * image_w;
    const void* vol_ptr = vols.untyped_data();
    const void* rot_ptr = rot.untyped_data();
    void*       img_ptr = imgs_out->untyped_data();

    cudaError_t err;
    switch (vols.element_type()) {
    case ffi::DataType::C64:
        err = launch_batch_project<float>(
            stream, (const float*)vol_ptr, (float*)img_ptr, (const float*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w);
        break;
    case ffi::DataType::C128:
        err = launch_batch_project<double>(
            stream, (const double*)vol_ptr, (double*)img_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w);
        break;
    default:
        return ffi::Error::InvalidArgument("batch_project: volumes must be C64 or C128");
    }
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BatchProject, BatchProjectImpl,
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
        .Arg<ffi::AnyBuffer>()           /* vols     */
        .Arg<ffi::AnyBuffer>()           /* rot      */
        .Ret<ffi::AnyBuffer>()           /* imgs_out */
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

} /* extern "C" */
