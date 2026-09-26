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

#include "include/device_scratch.cuh"

#include "include/recovar_cuda_common.cuh"

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

/* Form flattened volume addresses in 64 bits.  A box-800 RELION x-half
 * accumulator has 1603*1603*802 = 2,060,826,418 complex voxels, which fits
 * in a signed 32-bit spatial index.  Its interleaved scalar offset does not:
 * the imaginary component can reach 4,121,652,835.  Keep this address-only
 * arithmetic separate from the interpolation arithmetic so float operation
 * order and atomic accumulation topology remain unchanged. */
static __device__ __forceinline__ int64_t volume_spatial_offset(
    int i0, int i1, int i2, int stride0, int stride1)
{
    return static_cast<int64_t>(i0) * stride0
         + static_cast<int64_t>(i1) * stride1
         + static_cast<int64_t>(i2);
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
            const int64_t off = volume_spatial_offset(i0, i1, hkz, stride0, stride1);
            atomicAdd(&vol[off], val_re);
        } else {
            const int64_t off =
                volume_spatial_offset(i0, i1, hkz, stride0, stride1) * 2;
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
        const int64_t off = volume_spatial_offset(i0, i1, i2, stride0, stride1);
        atomicAdd(&vol[off], val_re);
    } else {
        const int64_t off =
            volume_spatial_offset(i0, i1, i2, stride0, stride1) * 2;
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
                        const int64_t off =
                            volume_spatial_offset(sj0, sj1, hkz, stride0, stride1);
                        atomicAdd(&vol[off], sre);
                    } else {
                        const int64_t off =
                            volume_spatial_offset(sj0, sj1, hkz, stride0, stride1) * 2;
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
                    const int64_t off =
                        volume_spatial_offset(j0, j1, j2, stride0, stride1);
                    atomicAdd(&vol[off], w * val_re);
                } else {
                    const int64_t off =
                        volume_spatial_offset(j0, j1, j2, stride0, stride1) * 2;
                    atomicAdd(&vol[off],     w * val_re);
                    atomicAdd(&vol[off + 1], w * val_im);
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
    int relion_fold_x,
    int skip_zero_values = 0,
    const int64_t* __restrict__ runtime_max_r2_x4 = nullptr)
{
    __shared__ T R[6];

    const int img_idx = blockIdx.x;
    const int pix     = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    /* A runtime radius (quarter-pixel^2 in padded units, as the attribute)
     * replaces the attribute: every cutoff below reads max_r2, so one program
     * serves every radius up to the volume's capacity. */
    if (runtime_max_r2_x4 != nullptr) {
        const int64_t x4 = *runtime_max_r2_x4;
        max_r2 = x4 < 0 ? (T)-1 : (T)x4 / (T)4;
    }

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    if (skip_zero_values) {
        /* Opt-in: sparse pass-2 M-step rows are padded to bucket size and
         * rows whose posterior mass was pruned are entirely zero.  Scattering
         * zeros adds exactly +0.0 to every touched voxel, so skipping them
         * only removes atomics (RELION's cuda_kernel_backproject3D likewise
         * skips pixels with Fweight == 0). */
        if (REAL_DATA) {
            if (img[img_idx * n_pixels + pix] == (T)0) return;
        } else {
            using V2z = vec2_t<T>;
            const V2z pz = reinterpret_cast<const V2z*>(img)[img_idx * n_pixels + pix];
            if (pz.x == (T)0 && pz.y == (T)0) return;
        }
    }

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
    const int64_t vol_scalar_stride = REAL_DATA
        ? static_cast<int64_t>(vol_stride)
        : static_cast<int64_t>(vol_stride) * 2;

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
        T* vol = vols + b * vol_scalar_stride;

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
    int64_t relion_block_topology = 0,
    int64_t skip_zero_values = 0,
    const int64_t* runtime_max_r2_x4 = nullptr)
{
    if (runtime_max_r2_x4 != nullptr && relion_block_topology)
        return cudaErrorInvalidValue;
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
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, max_r2, (int)relion_fold_x, \
            (int)skip_zero_values, runtime_max_r2_x4)

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

    cudaError_t err = recovar::scratch_alloc((void**)&real, n_voxels * sizeof(float), s);
    if (err != cudaSuccess) goto cleanup;
    err = recovar::scratch_alloc((void**)&imag, n_voxels * sizeof(float), s);
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
    if (real) recovar::scratch_free(real, s);
    if (imag) recovar::scratch_free(imag, s);
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
    const int64_t vol_scalar_stride = REAL_DATA
        ? static_cast<int64_t>(vol_stride)
        : static_cast<int64_t>(vol_stride) * 2;

    /* Inner loop over batch — same coords, different volumes and images */
    for (int b = 0; b < batch_size; b++) {
        T* vol = vols + b * vol_scalar_stride;

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

}
namespace {
}

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

static ffi::Error BackprojectIndexedCommon(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
    int64_t relion_fold_x,
    int64_t relion_block_topology,
    int64_t skip_zero_values,
    ffi::AnyBuffer img,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::Result<ffi::AnyBuffer> vol_out,
    const ffi::AnyBuffer* runtime_max_r2_x4 = nullptr)
{
    const int64_t* runtime_radius = nullptr;
    if (runtime_max_r2_x4 != nullptr) {
        if (runtime_max_r2_x4->element_type() != ffi::DataType::S64 ||
            runtime_max_r2_x4->dimensions().size() != 0)
            return ffi::Error::InvalidArgument("backproject_indexed: the runtime radius must be an int64 scalar");
        if (relion_block_topology)
            return ffi::Error::InvalidArgument(
                "backproject_indexed: a runtime radius is not implemented for the RELION block topology");
        runtime_radius = static_cast<const int64_t*>(runtime_max_r2_x4->untyped_data());
    }
    if (skip_zero_values && relion_block_topology)
        return ffi::Error::InvalidArgument("backproject_indexed: skip_zero_values is not implemented for the RELION block topology");
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
            relion_fold_x, relion_block_topology, skip_zero_values, runtime_radius);
        break;
    case ffi::DataType::C128:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4,
            relion_fold_x, relion_block_topology, skip_zero_values, runtime_radius);
        break;
    case ffi::DataType::F32:
        err = launch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4,
            relion_fold_x, relion_block_topology, skip_zero_values, runtime_radius);
        break;
    case ffi::DataType::F64:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4,
            relion_fold_x, relion_block_topology, skip_zero_values, runtime_radius);
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

#define RECOVAR_BACKPROJECT_INDEXED_ATTRS \
    int64_t image_h, int64_t image_w, \
    int64_t N0, int64_t N1, int64_t N2, \
    int64_t upsampling, int64_t order, \
    int64_t half_volume, int64_t half_image, int64_t full_image_w, \
    int64_t max_r2_x4, \
    int64_t relion_fold_x, \
    int64_t relion_block_topology

ffi::Error BackprojectIndexedImpl(
    cudaStream_t stream,
    RECOVAR_BACKPROJECT_INDEXED_ATTRS,
    ffi::AnyBuffer img,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer /*vol_in*/,
    ffi::Result<ffi::AnyBuffer> vol_out)
{
    return BackprojectIndexedCommon(
        stream, image_h, image_w, N0, N1, N2, upsampling, order, half_volume, half_image,
        full_image_w, max_r2_x4, relion_fold_x, relion_block_topology, /*skip_zero_values=*/0,
        img, pixel_indices, rot, vol_out);
}

/* Opt-in variant: identical ABI, but pixels whose value is exactly zero are
 * not scattered (RECOVAR_BACKPROJECT_SKIP_ZERO=1). */
ffi::Error BackprojectIndexedSkipZeroImpl(
    cudaStream_t stream,
    RECOVAR_BACKPROJECT_INDEXED_ATTRS,
    ffi::AnyBuffer img,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer /*vol_in*/,
    ffi::Result<ffi::AnyBuffer> vol_out)
{
    return BackprojectIndexedCommon(
        stream, image_h, image_w, N0, N1, N2, upsampling, order, half_volume, half_image,
        full_image_w, max_r2_x4, relion_fold_x, relion_block_topology, /*skip_zero_values=*/1,
        img, pixel_indices, rot, vol_out);
}

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

/* The same kernel with the radius as a device scalar (int64 max_r2_x4 in padded
 * units), so the program is not keyed on it: a pass whose volume is a stable
 * capacity class clips each call at RELION's logical radius. The max_r2_x4
 * attribute gives the capacity and is ignored by the kernel. */
ffi::Error BackprojectIndexedRuntimeRadiusImpl(
    cudaStream_t stream,
    RECOVAR_BACKPROJECT_INDEXED_ATTRS,
    ffi::AnyBuffer img,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer rot,
    ffi::AnyBuffer /*vol_in*/,
    ffi::AnyBuffer runtime_max_r2_x4,
    ffi::Result<ffi::AnyBuffer> vol_out)
{
    return BackprojectIndexedCommon(
        stream, image_h, image_w, N0, N1, N2, upsampling, order, half_volume, half_image,
        full_image_w, max_r2_x4, relion_fold_x, relion_block_topology, /*skip_zero_values=*/0,
        img, pixel_indices, rot, vol_out, &runtime_max_r2_x4);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BackprojectIndexedRuntimeRadius, BackprojectIndexedRuntimeRadiusImpl,
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
        .Arg<ffi::AnyBuffer>()           /* img              */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices    */
        .Arg<ffi::AnyBuffer>()           /* rot              */
        .Arg<ffi::AnyBuffer>()           /* vol_in           */
        .Arg<ffi::AnyBuffer>()           /* runtime max_r2_x4, int64 scalar */
        .Ret<ffi::AnyBuffer>()           /* vol_out (aliased with vol_in) */
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BackprojectIndexedSkipZero, BackprojectIndexedSkipZeroImpl,
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