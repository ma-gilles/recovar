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
    const int x0 = floor_int(relion_x);
    const int y0 = floor_int(relion_y) + maxR + 1;
    const int z0 = floor_int(relion_z) + maxR + 1;
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
        const int N2_full = 2 * ic2;
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
        } else if (-kz == ic2) {
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
                    } else if (-kz == ic2) {
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
        const int ic2 = (int)c2;
        const int N2_full = 2 * ic2;
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

    const T k0 = (T)(k0_idx - image_h / 2) * upsampling;
    T k1;
    if (HALF_IMG) {
        k1 = (k1_idx * 2 == full_image_w)
             ? (T)(-k1_idx) * upsampling
             : (T)(k1_idx)  * upsampling;
    } else {
        k1 = (T)(k1_idx - image_w / 2) * upsampling;
    }

    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) return;

    T rk0 = k0 * R[0] + k1 * R[3];
    T rk1 = k0 * R[1] + k1 * R[4];
    T rk2 = k0 * R[2] + k1 * R[5];

    if (relion_fold_x && HALF_IMG && HALF_VOL && max_r2 >= (T)0) {
        /* RELION's backproject2Dto3D repeats the radius cutoff after the
         * source pixel has been rotated into 3-D. Mathematically this is
         * redundant for an exactly orthonormal matrix, but at the outer shell
         * it changes inclusion for roundoff-level boundary pixels. */
        const double r2_3d =
            (double)rk0 * (double)rk0 +
            (double)rk1 * (double)rk1 +
            (double)rk2 * (double)rk2;
        if (r2_3d > (double)max_r2) return;
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
        const int ic2 = (int)c2;
        const int N2_full = 2 * ic2;
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
                                           c0, c1, c2, N0, N1, N2_eff, stride0, stride1);
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
                        } else if (-kz == ic2) {
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

template <bool HALF_IMG>
__global__ void __launch_bounds__(BLOCK_SIZE)
project_texture_kernel(
    cudaTextureObject_t texReal,
    cudaTextureObject_t texImag,
    float* __restrict__ img,
    const float* __restrict__ rot,
    int n_pixels, int image_h, int image_w,
    int tex_yinit, int tex_zinit,
    int upsampling, int full_image_w,
    int maxR2_padded)
{
    __shared__ float R[6];

    const int img_idx = blockIdx.x;
    const int pix = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.x < 6) R[threadIdx.x] = rot[img_idx * 6 + threadIdx.x];
    __syncthreads();
    if (pix >= n_pixels) return;

    const int k0_idx = pix / image_w;
    const int k1_idx = pix % image_w;
    const float k0_unscaled = (float)(k0_idx - image_h / 2);
    float k1_unscaled;
    if (HALF_IMG) {
        k1_unscaled = (k1_idx * 2 == full_image_w)
             ? (float)(-k1_idx)
             : (float)(k1_idx);
    } else {
        k1_unscaled = (float)(k1_idx - image_w / 2);
    }

    float2* img2 = reinterpret_cast<float2*>(img);
    const int img_off = img_idx * n_pixels + pix;

    /* Match RELION AccProjectorKernel arithmetic: rotate integer image
     * coordinates first, then multiply by padding_factor. Scaling the image
     * coordinates before the dot product changes CUDA texture fractions by a
     * few ulps and is visible in borderline per-particle Pmax comparisons. */
    const float rk0 = (k0_unscaled * R[0] + k1_unscaled * R[3]) * (float)upsampling;
    const float rk1 = (k0_unscaled * R[1] + k1_unscaled * R[4]) * (float)upsampling;
    const float rk2 = (k0_unscaled * R[2] + k1_unscaled * R[5]) * (float)upsampling;

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
    const float k0_unscaled = (float)(k0_idx - image_h / 2);
    float k1_unscaled;
    if (HALF_IMG) {
        k1_unscaled = (k1_idx * 2 == full_image_w)
             ? (float)(-k1_idx)
             : (float)(k1_idx);
    } else {
        k1_unscaled = (float)(k1_idx - image_w / 2);
    }

    double2* img2 = reinterpret_cast<double2*>(img);
    const int img_off = img_idx * n_pixels + pix;

    /* Match RELION AccProjectorKernel arithmetic: rotate integer image
     * coordinates first, then multiply by padding_factor. */
    const float rk0 = (k0_unscaled * R[0] + k1_unscaled * R[3]) * (float)upsampling;
    const float rk1 = (k0_unscaled * R[1] + k1_unscaled * R[4]) * (float)upsampling;
    const float rk2 = (k0_unscaled * R[2] + k1_unscaled * R[5]) * (float)upsampling;

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
    int64_t relion_fold_x = 0)
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
    switch (key) {
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
        project_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vol, img, rot, (int)n_pixels, (int)ih, (int)iw, \
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

cudaError_t launch_project_texture_float(
    cudaStream_t s, const float* vol, float* img, const float* rot,
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
        fill_relion_texture_compact_kernel<float><<<grid, block, 0, s>>>(
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
            project_texture_kernel<true><<<grid, block, 0, s>>>(
                texReal, texImag, img, rot, (int)n_pixels, (int)ih, (int)iw,
                texYInit, texZInit, (int)ups, (int)full_iw, maxR * maxR);
        } else {
            project_texture_kernel<false><<<grid, block, 0, s>>>(
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
        const int ic2 = (int)c2;
        const int N2_full = 2 * ic2;
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
    int batch_size,
    T max_r2)
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

    using V2 = vec2_t<T>;
    const int img_stride = n_images * n_pixels;

    /* Pre-rotation disk check: rotation preserves ||k||. */
    if (max_r2 >= (T)0 && k0 * k0 + k1 * k1 > max_r2) {
        for (int b = 0; b < batch_size; b++) {
            V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
            out[pix] = make_v2((T)0, (T)0);
        }
        return;
    }

    T rk0 = k0 * R[0] + k1 * R[3];
    T rk1 = k0 * R[1] + k1 * R[4];
    T rk2 = k0 * R[2] + k1 * R[5];

    const int stride1 = N2_eff;
    const int stride0 = N1 * N2_eff;

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
                    /* partner(j) = (N - (N & 1) - j) % N */
                    ri = (N0 - (N0 & 1) - i0) % N0;
                    rj = (N1 - (N1 & 1) - i1) % N1;
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

        /* ──── cubic HALF_VOL batch (ORDER==3, periodic wrap) ──── */
        if (ORDER == 3) {
            const T cg0 = rk0 + c0 - (T)1;
            const T cg1 = rk1 + c1 - (T)1;
            const T cg2_full = rk2 + c2 - (T)1;
            const int cb0 = floor_int(cg0);
            const int cb1 = floor_int(cg1);
            const int cb2 = floor_int(cg2_full);
            const T cf0 = cg0 - (T)cb0;
            const T cf1 = cg1 - (T)cb1;
            const T cf2 = cg2_full - (T)cb2;

            /* Precompute 64 neighbor offsets/weights/conj flags */
            struct { int off; T w; bool cj; } nbr[64];
            int n_nbr = 0;
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
                        bool cjj = false;
                        if (kz >= 0) {
                            hkz = kz;
                        } else if (-kz == ic2) {
                            hkz = ic2;
                        } else {
                            ri = (N0 - (N0 & 1) - j0) % N0;
                            rj = (N1 - (N1 & 1) - j1) % N1;
                            hkz = -kz;
                            cjj = true;
                        }
                        if (hkz <= ic2) {
                            nbr[n_nbr].off = ri * stride0 + rj * stride1 + hkz;
                            nbr[n_nbr].w = w;
                            nbr[n_nbr].cj = cjj;
                            n_nbr++;
                        }
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
            /* partner(j) = (N - (N & 1) - j) % N */
            const int r0[2] = {(N0 - (N0 & 1) - bb0) % N0, (N0 - (N0 & 1) - bb0 - 1) % N0};
            const int r1[2] = {(N1 - (N1 & 1) - bb1) % N1, (N1 - (N1 & 1) - bb1 - 1) % N1};
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
                            /* partner(j) = (N - (N & 1) - j) % N */
                            ri = (N0 - (N0 & 1) - j0) % N0;
                            rj = (N1 - (N1 & 1) - j1) % N1;
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

    /* ── Non-HALF_VOL path ───────────────────────────────────────── */

    /* ──── cubic non-HALF_VOL batch (ORDER==3, periodic wrap) ──── */
    if (ORDER == 3) {
        const T cg0 = rk0 + c0 - (T)1;
        const T cg1 = rk1 + c1 - (T)1;
        const T cg2 = rk2 + c2 - (T)1;
        const int cb0 = floor_int(cg0);
        const int cb1 = floor_int(cg1);
        const int cb2 = floor_int(cg2);
        const T cf0 = cg0 - (T)cb0;
        const T cf1 = cg1 - (T)cb1;
        const T cf2 = cg2 - (T)cb2;

        /* Precompute 64 neighbor offsets + weights */
        int off[64]; T wt[64];
        int n_nbr = 0;
        for (int d0 = 0; d0 < 4; d0++) {
            const int j0 = wrap_mod(cb0 + d0, N0);
            const T bw0 = cubic_basis(cf0 - (T)d0 + (T)1);
            for (int d1 = 0; d1 < 4; d1++) {
                const int j1 = wrap_mod(cb1 + d1, N1);
                const T bw01 = bw0 * cubic_basis(cf1 - (T)d1 + (T)1);
                for (int d2 = 0; d2 < 4; d2++) {
                    const int j2 = wrap_mod(cb2 + d2, N2_eff);
                    off[n_nbr] = j0 * stride0 + j1 * stride1 + j2;
                    wt[n_nbr] = bw01 * cubic_basis(cf2 - (T)d2 + (T)1);
                    n_nbr++;
                }
            }
        }
        for (int b = 0; b < batch_size; b++) {
            const V2* vol2 = reinterpret_cast<const V2*>(vols + b * vol_stride * 2);
            V2* out = reinterpret_cast<V2*>(imgs) + b * img_stride + img_idx * n_pixels;
            T sr = 0, si = 0;
            for (int i = 0; i < 64; i++) {
                V2 v = __ldg(&vol2[off[i]]);
                sr += wt[i] * v.x;
                si += wt[i] * v.y;
            }
            out[pix] = make_v2(sr, si);
        }
        return;
    }

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

template <typename T>
cudaError_t launch_batch_project(
    cudaStream_t s, const T* vols, T* imgs, const T* rot,
    int64_t batch_size, int64_t n_images, int64_t n_pixels,
    int64_t ih, int64_t iw,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t ups, int64_t order, int64_t half_vol, int64_t half_img,
    int64_t full_iw, int64_t max_r2_x4 = -1)
{
    const int N2_eff = half_vol ? (int)(N2 / 2 + 1) : (int)N2;
    const int vol_stride = (int)N0 * (int)N1 * N2_eff;
    const T c0 = (T)(N0 / 2);
    const T c1 = (T)(N1 / 2);
    const T c2 = (T)(N2 / 2);
    const T max_r2 = max_r2_x4 < 0 ? (T)-1 : (T)max_r2_x4 / (T)4;
    dim3 grid((int)n_images, ((int)n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    #define BPJ(O, HV, HI) \
        batch_project_kernel<T, O, HV, HI><<<grid, block, 0, s>>>( \
            vols, imgs, rot, (int)n_pixels, (int)ih, (int)iw, \
            (int)N0, (int)N1, N2_eff, c0, c1, c2, (int)ups, (int)full_iw, \
            vol_stride, (int)n_images, (int)batch_size, max_r2)

    int order_code = (order == 3) ? 2 : (int)order;
    int key = (order_code << 2) | (half_vol ? 2 : 0) | (half_img ? 1 : 0);
    switch (key) {
    case  0: BPJ(0, false, false); break;
    case  1: BPJ(0, false, true);  break;
    case  2: BPJ(0, true,  false); break;
    case  3: BPJ(0, true,  true);  break;
    case  4: BPJ(1, false, false); break;
    case  5: BPJ(1, false, true);  break;
    case  6: BPJ(1, true,  false); break;
    case  7: BPJ(1, true,  true);  break;
    /* ORDER=3 (cubic, periodic wrap) */
    case  8: BPJ(3, false, false); break;
    case  9: BPJ(3, false, true);  break;
    case 10: BPJ(3, true,  false); break;
    case 11: BPJ(3, true,  true);  break;
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
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4, relion_fold_x);
        break;
    case ffi::DataType::C128:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/0, max_r2_x4, relion_fold_x);
        break;
    case ffi::DataType::F32:
        err = launch_backproject_indexed<float>(
            stream, (float*)vol_ptr, (const float*)img_ptr, (const int32_t*)pix_ptr, (const float*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4, relion_fold_x);
        break;
    case ffi::DataType::F64:
        err = launch_backproject_indexed<double>(
            stream, (double*)vol_ptr, (const double*)img_ptr, (const int32_t*)pix_ptr, (const double*)rot_ptr,
            n_images, n_pixels, image_h, image_w, N0, N1, N2, upsampling,
            order, half_volume, half_image, full_image_w, /*real_data=*/1, max_r2_x4, relion_fold_x);
        break;
    default:
        return ffi::Error::InvalidArgument("backproject_indexed: images must be C64, C128, F32, or F64");
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
        .Arg<ffi::AnyBuffer>()           /* img           */
        .Arg<ffi::AnyBuffer>()           /* pixel_indices */
        .Arg<ffi::AnyBuffer>()           /* rot           */
        .Arg<ffi::AnyBuffer>()           /* vol_in        */
        .Ret<ffi::AnyBuffer>()           /* vol_out (aliased with vol_in) */
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

ffi::Error BatchProjectImpl(
    cudaStream_t stream,
    int64_t image_h, int64_t image_w,
    int64_t N0, int64_t N1, int64_t N2,
    int64_t upsampling, int64_t order,
    int64_t half_volume, int64_t half_image, int64_t full_image_w,
    int64_t max_r2_x4,
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
            upsampling, order, half_volume, half_image, full_image_w, max_r2_x4);
        break;
    case ffi::DataType::C128:
        err = launch_batch_project<double>(
            stream, (const double*)vol_ptr, (double*)img_ptr, (const double*)rot_ptr,
            batch_size, n_images, n_pixels, image_h, image_w, N0, N1, N2,
            upsampling, order, half_volume, half_image, full_image_w, max_r2_x4);
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
        .Attr<int64_t>("max_r2_x4")
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
