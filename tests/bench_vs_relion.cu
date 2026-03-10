/*
 * Standalone benchmark: recovar vs RELION-style backproject/project kernels.
 *
 * Reimplements RELION 5.0.1's cuda_kernel_backproject3D (from BP.cuh) as a
 * standalone kernel with no RELION dependencies.  Compares against a
 * reimplementation of recovar's backproject kernel.
 *
 * Three variants are benchmarked:
 *   1. "recovar"     — our interleaved-complex, 1-thread-per-pixel kernel
 *   2. "relion_pure"  — RELION's algorithm stripped to pure interpolation
 *                       (no CTF, no translations, no weights)
 *   3. "relion_full"  — RELION's full algorithm with CTF=1, 1 translation,
 *                       weight=1, Minvsigma2=1 (the branch overhead is real)
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 \
 *        -gencode arch=compute_90,code=sm_90 \
 *        -o bench_vs_relion bench_vs_relion.cu
 *
 * Run:
 *   ./bench_vs_relion [N=128] [n_images=1000] [n_iters=50]
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <functional>

/* ================================================================== */
/*  RELION-style kernel (from BP.cuh lines 175-403)                   */
/*  Reimplemented with no RELION dependencies.                        */
/*  Template: PURE_INTERP strips CTF/translation/weight overhead.     */
/* ================================================================== */

#define RELION_BP_BLOCK_SIZE 128

template <bool PURE_INTERP>
__global__ void relion_backproject3D(
    const float* __restrict__ g_img_real,
    const float* __restrict__ g_img_imag,
    const float* __restrict__ g_eulers,       /* 9 floats per image */
    float* __restrict__ g_model_real,
    float* __restrict__ g_model_imag,
    float* __restrict__ g_model_weight,
    /* Only used when !PURE_INTERP: */
    const float* __restrict__ g_ctfs,         /* per-pixel CTF */
    const float* __restrict__ g_Minvsigma2s,  /* per-pixel noise weight */
    const float* __restrict__ g_weights,      /* per-image weight */
    float significant_weight,
    float weight_norm,
    int max_r2,
    float padding_factor,
    unsigned img_x,     /* rfft width: N/2+1 */
    unsigned img_y,     /* N */
    unsigned img_xy,    /* img_x * img_y */
    unsigned mdl_x,     /* model rfft width */
    unsigned mdl_y,     /* model N */
    int mdl_inity,      /* -(N/2) */
    int mdl_initz)      /* -(N/2) */
{
    unsigned tid = threadIdx.x;
    unsigned img = blockIdx.x;

    int img_y_half = img_y / 2;
    int max_r2_vol = (int)(max_r2 * padding_factor * padding_factor);

    __shared__ float s_eulers[9];
    if (tid < 9)
        s_eulers[tid] = g_eulers[img * 9 + tid];
    __syncthreads();

    int pixel_pass_num = (int)ceilf((float)img_xy / (float)RELION_BP_BLOCK_SIZE);

    for (int pass = 0; pass < pixel_pass_num; pass++)
    {
        unsigned pixel = pass * RELION_BP_BLOCK_SIZE + tid;
        if (pixel >= img_xy)
            continue;

        int x = pixel % img_x;
        int y = (int)floorf((float)pixel / (float)img_x);
        if (y > img_y_half)
            y -= img_y;

        float real, imag, Fweight;

        if (PURE_INTERP)
        {
            /* Pure interpolation: just read pixel value directly */
            real = g_img_real[img * img_xy + pixel];
            imag = g_img_imag[img * img_xy + pixel];
            Fweight = 1.0f;
        }
        else
        {
            /* Full RELION logic: CTF, Minvsigma2, weight, translation */
            float minvsigma2 = __ldg(&g_Minvsigma2s[pixel]);
            float ctf = __ldg(&g_ctfs[pixel]);
            float img_real_val = __ldg(&g_img_real[img * img_xy + pixel]);
            float img_imag_val = __ldg(&g_img_imag[img * img_xy + pixel]);

            Fweight = 0.0f;
            real = 0.0f;
            imag = 0.0f;

            /* Single translation (no shift), weight = g_weights[img] */
            float weight = g_weights[img];
            if (weight >= significant_weight)
            {
                weight = (weight / weight_norm) * ctf * minvsigma2;
                Fweight += weight * ctf;
                /* No translation shift: temp = img directly */
                real += img_real_val * weight;
                imag += img_imag_val * weight;
            }
        }

        if (Fweight > 0.0f)
        {
            float xp = (s_eulers[0] * x + s_eulers[1] * y) * padding_factor;
            float yp = (s_eulers[3] * x + s_eulers[4] * y) * padding_factor;
            float zp = (s_eulers[6] * x + s_eulers[7] * y) * padding_factor;

            if ((xp * xp + yp * yp + zp * zp) > (float)max_r2_vol)
                continue;

            /* Hermitian: store only xp >= 0 half */
            if (xp < 0.0f)
            {
                xp = -xp;
                yp = -yp;
                zp = -zp;
                imag = -imag;
            }

            int x0 = (int)floorf(xp);
            float fx = xp - x0;
            int x1 = x0 + 1;

            int y0 = (int)floorf(yp);
            float fy = yp - y0;
            y0 -= mdl_inity;
            int y1 = y0 + 1;

            int z0 = (int)floorf(zp);
            float fz = zp - z0;
            z0 -= mdl_initz;
            int z1 = z0 + 1;

            float mfx = 1.0f - fx;
            float mfy = 1.0f - fy;
            float mfz = 1.0f - fz;

            /* 8 neighbors × 3 arrays = 24 atomicAdds */
            #define RELION_SCATTER(iz, iy, ix, w) \
            { \
                int idx = (iz) * mdl_x * mdl_y + (iy) * mdl_x + (ix); \
                atomicAdd(&g_model_real[idx],   (w) * real); \
                atomicAdd(&g_model_imag[idx],   (w) * imag); \
                atomicAdd(&g_model_weight[idx], (w) * Fweight); \
            }

            RELION_SCATTER(z0, y0, x0, mfz * mfy * mfx);
            RELION_SCATTER(z0, y0, x1, mfz * mfy *  fx);
            RELION_SCATTER(z0, y1, x0, mfz *  fy * mfx);
            RELION_SCATTER(z0, y1, x1, mfz *  fy *  fx);
            RELION_SCATTER(z1, y0, x0,  fz * mfy * mfx);
            RELION_SCATTER(z1, y0, x1,  fz * mfy *  fx);
            RELION_SCATTER(z1, y1, x0,  fz *  fy * mfx);
            RELION_SCATTER(z1, y1, x1,  fz *  fy *  fx);
            #undef RELION_SCATTER
        }
    }
}


/* ================================================================== */
/*  RELION-style forward projection kernel                            */
/*  (from acc_projector.h / acc_projector_impl.h)                     */
/*  Reimplemented: trilinear interpolation from separate real/imag.   */
/* ================================================================== */

#define RELION_FP_BLOCK_SIZE 256

__global__ void relion_project3D(
    const float* __restrict__ g_model_real,
    const float* __restrict__ g_model_imag,
    const float* __restrict__ g_eulers,       /* 9 floats per image */
    float* __restrict__ g_img_real,
    float* __restrict__ g_img_imag,
    int max_r2,
    float padding_factor,
    unsigned img_x,     /* rfft width: N/2+1 */
    unsigned img_y,     /* N */
    unsigned img_xy,    /* img_x * img_y */
    unsigned mdl_x,     /* model rfft width */
    unsigned mdl_y,     /* model N */
    unsigned mdl_z,     /* model N */
    int mdl_inity,
    int mdl_initz)
{
    unsigned tid = threadIdx.x;
    unsigned img = blockIdx.x;

    int img_y_half = img_y / 2;
    int max_r2_vol = (int)(max_r2 * padding_factor * padding_factor);

    __shared__ float s_eulers[9];
    if (tid < 9)
        s_eulers[tid] = g_eulers[img * 9 + tid];
    __syncthreads();

    int pixel_pass_num = (int)ceilf((float)img_xy / (float)RELION_FP_BLOCK_SIZE);

    for (int pass = 0; pass < pixel_pass_num; pass++)
    {
        unsigned pixel = pass * RELION_FP_BLOCK_SIZE + tid;
        if (pixel >= img_xy)
            continue;

        int x = pixel % img_x;
        int y = (int)floorf((float)pixel / (float)img_x);
        if (y > img_y_half)
            y -= img_y;

        float xp = (s_eulers[0] * x + s_eulers[1] * y) * padding_factor;
        float yp = (s_eulers[3] * x + s_eulers[4] * y) * padding_factor;
        float zp = (s_eulers[6] * x + s_eulers[7] * y) * padding_factor;

        float real_out = 0.0f, imag_out = 0.0f;

        if ((xp * xp + yp * yp + zp * zp) <= (float)max_r2_vol)
        {
            bool is_neg_x = (xp < 0.0f);
            if (is_neg_x) { xp = -xp; yp = -yp; zp = -zp; }

            int x0 = (int)floorf(xp);
            float fx = xp - x0;
            int x1 = x0 + 1;

            int y0 = (int)floorf(yp);
            float fy = yp - y0;
            y0 -= mdl_inity;
            int y1 = y0 + 1;

            int z0 = (int)floorf(zp);
            float fz = zp - z0;
            z0 -= mdl_initz;
            int z1 = z0 + 1;

            float mfx = 1.0f - fx;
            float mfy = 1.0f - fy;
            float mfz = 1.0f - fz;

            #define RELION_GATHER(iz, iy, ix, w) \
            { \
                int idx = (iz) * mdl_x * mdl_y + (iy) * mdl_x + (ix); \
                real_out += (w) * g_model_real[idx]; \
                imag_out += (w) * g_model_imag[idx]; \
            }

            RELION_GATHER(z0, y0, x0, mfz * mfy * mfx);
            RELION_GATHER(z0, y0, x1, mfz * mfy *  fx);
            RELION_GATHER(z0, y1, x0, mfz *  fy * mfx);
            RELION_GATHER(z0, y1, x1, mfz *  fy *  fx);
            RELION_GATHER(z1, y0, x0,  fz * mfy * mfx);
            RELION_GATHER(z1, y0, x1,  fz * mfy *  fx);
            RELION_GATHER(z1, y1, x0,  fz *  fy * mfx);
            RELION_GATHER(z1, y1, x1,  fz *  fy *  fx);
            #undef RELION_GATHER

            if (is_neg_x) imag_out = -imag_out;
        }

        g_img_real[img * img_xy + pixel] = real_out;
        g_img_imag[img * img_xy + pixel] = imag_out;
    }
}


/* ================================================================== */
/*  Recovar-style kernels (from cuda_backproject.cu)                  */
/*  Reimplemented standalone.  Full-volume, full-image, no CONJ_MODE. */
/* ================================================================== */

#define RECOVAR_BLOCK_SIZE 256

__global__ void recovar_backproject(
    float* __restrict__ vol,          /* interleaved complex: [2*(N0*N1*N2)] */
    const float* __restrict__ imgs,   /* interleaved complex: [n_images * n_pix * 2] */
    const float* __restrict__ rot,    /* (n_images, 6): first 2 rows of 3x3 */
    int n_images,
    int n_pixels,
    int image_h,
    int image_w,
    int N0, int N1, int N2,
    int upsampling,
    float max_r2)
{
    int gid = blockIdx.x * RECOVAR_BLOCK_SIZE + threadIdx.x;
    int total = n_images * n_pixels;
    if (gid >= total) return;

    int img_idx = gid / n_pixels;
    int pix     = gid % n_pixels;

    /* Load compact rotation (6 floats: first 2 rows of 3x3) */
    const float* R = rot + img_idx * 6;
    float R0 = R[0], R1 = R[1], R2 = R[2];
    float R3 = R[3], R4 = R[4], R5 = R[5];

    /* Frequency coordinates */
    int k0_idx = pix / image_w;
    int k1_idx = pix % image_w;
    float k0 = (float)(k0_idx - image_h / 2) * upsampling;

    /* Handle Nyquist for k1 */
    float k1;
    if (k1_idx * 2 == image_w)
        k1 = (float)(-(image_w / 2)) * upsampling;  /* Nyquist */
    else if (k1_idx > image_w / 2)
        k1 = (float)(k1_idx - image_w) * upsampling;
    else
        k1 = (float)(k1_idx) * upsampling;

    /* Rotated coordinates (cz=0 plane) */
    float rk0 = k0 * R0 + k1 * R3;
    float rk1 = k0 * R1 + k1 * R4;
    float rk2 = k0 * R2 + k1 * R5;

    /* Sphere clipping */
    if (max_r2 >= 0.0f && rk0 * rk0 + rk1 * rk1 + rk2 * rk2 > max_r2)
        return;

    /* Image value */
    float val_re = imgs[(img_idx * n_pixels + pix) * 2 + 0];
    float val_im = imgs[(img_idx * n_pixels + pix) * 2 + 1];

    /* Map to grid indices */
    float gx = rk0 + (float)(N0 / 2);
    float gy = rk1 + (float)(N1 / 2);
    float gz = rk2 + (float)(N2 / 2);

    int x0 = (int)floorf(gx); float fx = gx - x0; int x1 = x0 + 1;
    int y0 = (int)floorf(gy); float fy = gy - y0; int y1 = y0 + 1;
    int z0 = (int)floorf(gz); float fz = gz - z0; int z1 = z0 + 1;

    /* Bounds check */
    if (x0 < 0 || x1 >= N0 || y0 < 0 || y1 >= N1 || z0 < 0 || z1 >= N2)
        return;

    float mfx = 1.0f - fx, mfy = 1.0f - fy, mfz = 1.0f - fz;

    /* 8 neighbors × 2 floats (real, imag) = 16 atomicAdds */
    #define RECOVAR_SCATTER(ix, iy, iz, w) \
    { \
        int idx = ((ix) * N1 * N2 + (iy) * N2 + (iz)) * 2; \
        atomicAdd(&vol[idx + 0], (w) * val_re); \
        atomicAdd(&vol[idx + 1], (w) * val_im); \
    }

    RECOVAR_SCATTER(x0, y0, z0, mfx * mfy * mfz);
    RECOVAR_SCATTER(x0, y0, z1, mfx * mfy *  fz);
    RECOVAR_SCATTER(x0, y1, z0, mfx *  fy * mfz);
    RECOVAR_SCATTER(x0, y1, z1, mfx *  fy *  fz);
    RECOVAR_SCATTER(x1, y0, z0,  fx * mfy * mfz);
    RECOVAR_SCATTER(x1, y0, z1,  fx * mfy *  fz);
    RECOVAR_SCATTER(x1, y1, z0,  fx *  fy * mfz);
    RECOVAR_SCATTER(x1, y1, z1,  fx *  fy *  fz);
    #undef RECOVAR_SCATTER
}


__global__ void recovar_project(
    const float* __restrict__ vol,    /* interleaved complex */
    float* __restrict__ imgs,         /* interleaved complex */
    const float* __restrict__ rot,
    int n_images,
    int n_pixels,
    int image_h,
    int image_w,
    int N0, int N1, int N2,
    int upsampling,
    float max_r2)
{
    int gid = blockIdx.x * RECOVAR_BLOCK_SIZE + threadIdx.x;
    int total = n_images * n_pixels;
    if (gid >= total) return;

    int img_idx = gid / n_pixels;
    int pix     = gid % n_pixels;

    const float* R = rot + img_idx * 6;
    float R0 = R[0], R1 = R[1], R2 = R[2];
    float R3 = R[3], R4 = R[4], R5 = R[5];

    int k0_idx = pix / image_w;
    int k1_idx = pix % image_w;
    float k0 = (float)(k0_idx - image_h / 2) * upsampling;
    float k1;
    if (k1_idx * 2 == image_w)
        k1 = (float)(-(image_w / 2)) * upsampling;
    else if (k1_idx > image_w / 2)
        k1 = (float)(k1_idx - image_w) * upsampling;
    else
        k1 = (float)(k1_idx) * upsampling;

    float rk0 = k0 * R0 + k1 * R3;
    float rk1 = k0 * R1 + k1 * R4;
    float rk2 = k0 * R2 + k1 * R5;

    int out_idx = (img_idx * n_pixels + pix) * 2;

    if (max_r2 >= 0.0f && rk0 * rk0 + rk1 * rk1 + rk2 * rk2 > max_r2)
    {
        imgs[out_idx + 0] = 0.0f;
        imgs[out_idx + 1] = 0.0f;
        return;
    }

    float gx = rk0 + (float)(N0 / 2);
    float gy = rk1 + (float)(N1 / 2);
    float gz = rk2 + (float)(N2 / 2);

    int x0 = (int)floorf(gx); float fx = gx - x0; int x1 = x0 + 1;
    int y0 = (int)floorf(gy); float fy = gy - y0; int y1 = y0 + 1;
    int z0 = (int)floorf(gz); float fz = gz - z0; int z1 = z0 + 1;

    float val_re = 0.0f, val_im = 0.0f;

    if (x0 >= 0 && x1 < N0 && y0 >= 0 && y1 < N1 && z0 >= 0 && z1 < N2)
    {
        float mfx = 1.0f - fx, mfy = 1.0f - fy, mfz = 1.0f - fz;

        #define RECOVAR_GATHER(ix, iy, iz, w) \
        { \
            int idx = ((ix) * N1 * N2 + (iy) * N2 + (iz)) * 2; \
            val_re += (w) * vol[idx + 0]; \
            val_im += (w) * vol[idx + 1]; \
        }

        RECOVAR_GATHER(x0, y0, z0, mfx * mfy * mfz);
        RECOVAR_GATHER(x0, y0, z1, mfx * mfy *  fz);
        RECOVAR_GATHER(x0, y1, z0, mfx *  fy * mfz);
        RECOVAR_GATHER(x0, y1, z1, mfx *  fy *  fz);
        RECOVAR_GATHER(x1, y0, z0,  fx * mfy * mfz);
        RECOVAR_GATHER(x1, y0, z1,  fx * mfy *  fz);
        RECOVAR_GATHER(x1, y1, z0,  fx *  fy * mfz);
        RECOVAR_GATHER(x1, y1, z1,  fx *  fy *  fz);
        #undef RECOVAR_GATHER
    }

    imgs[out_idx + 0] = val_re;
    imgs[out_idx + 1] = val_im;
}


/* ================================================================== */
/*  Recovar-style half-image backproject (rfft layout, like RELION)    */
/*  This is the apples-to-apples comparison kernel.                   */
/* ================================================================== */

__global__ void recovar_backproject_half(
    float* __restrict__ vol,          /* interleaved complex: [2*(N0*N1*N2)] */
    const float* __restrict__ imgs,   /* interleaved complex, rfft: [n_images * H*(W/2+1) * 2] */
    const float* __restrict__ rot,    /* (n_images, 6) */
    int n_images,
    int n_pixels,       /* H * (W/2+1) */
    int image_h,
    int image_w_half,   /* W/2+1 */
    int full_image_w,   /* W (original full width) */
    int N0, int N1, int N2,
    int upsampling,
    float max_r2)
{
    int gid = blockIdx.x * RECOVAR_BLOCK_SIZE + threadIdx.x;
    int total = n_images * n_pixels;
    if (gid >= total) return;

    int img_idx = gid / n_pixels;
    int pix     = gid % n_pixels;

    const float* R = rot + img_idx * 6;
    float R0 = R[0], R1 = R[1], R2 = R[2];
    float R3 = R[3], R4 = R[4], R5 = R[5];

    int k0_idx = pix / image_w_half;
    int k1_idx = pix % image_w_half;

    /* rfft layout: k0 centered, k1 = 0..W/2 */
    float k0 = (float)(k0_idx - image_h / 2) * upsampling;
    float k1 = (float)(k1_idx) * upsampling;

    float rk0 = k0 * R0 + k1 * R3;
    float rk1 = k0 * R1 + k1 * R4;
    float rk2 = k0 * R2 + k1 * R5;

    if (max_r2 >= 0.0f && rk0 * rk0 + rk1 * rk1 + rk2 * rk2 > max_r2)
        return;

    float val_re = imgs[(img_idx * n_pixels + pix) * 2 + 0];
    float val_im = imgs[(img_idx * n_pixels + pix) * 2 + 1];

    float gx = rk0 + (float)(N0 / 2);
    float gy = rk1 + (float)(N1 / 2);
    float gz = rk2 + (float)(N2 / 2);

    int x0 = (int)floorf(gx); float fx = gx - x0; int x1 = x0 + 1;
    int y0 = (int)floorf(gy); float fy = gy - y0; int y1 = y0 + 1;
    int z0 = (int)floorf(gz); float fz = gz - z0; int z1 = z0 + 1;

    if (x0 < 0 || x1 >= N0 || y0 < 0 || y1 >= N1 || z0 < 0 || z1 >= N2)
        return;

    float mfx = 1.0f - fx, mfy = 1.0f - fy, mfz = 1.0f - fz;

    #define RECOVAR_HALF_SCATTER(ix, iy, iz, w) \
    { \
        int idx = ((ix) * N1 * N2 + (iy) * N2 + (iz)) * 2; \
        atomicAdd(&vol[idx + 0], (w) * val_re); \
        atomicAdd(&vol[idx + 1], (w) * val_im); \
    }

    RECOVAR_HALF_SCATTER(x0, y0, z0, mfx * mfy * mfz);
    RECOVAR_HALF_SCATTER(x0, y0, z1, mfx * mfy *  fz);
    RECOVAR_HALF_SCATTER(x0, y1, z0, mfx *  fy * mfz);
    RECOVAR_HALF_SCATTER(x0, y1, z1, mfx *  fy *  fz);
    RECOVAR_HALF_SCATTER(x1, y0, z0,  fx * mfy * mfz);
    RECOVAR_HALF_SCATTER(x1, y0, z1,  fx * mfy *  fz);
    RECOVAR_HALF_SCATTER(x1, y1, z0,  fx *  fy * mfz);
    RECOVAR_HALF_SCATTER(x1, y1, z1,  fx *  fy *  fz);
    #undef RECOVAR_HALF_SCATTER
}


__global__ void recovar_project_half(
    const float* __restrict__ vol,
    float* __restrict__ imgs,
    const float* __restrict__ rot,
    int n_images,
    int n_pixels,
    int image_h,
    int image_w_half,
    int full_image_w,
    int N0, int N1, int N2,
    int upsampling,
    float max_r2)
{
    int gid = blockIdx.x * RECOVAR_BLOCK_SIZE + threadIdx.x;
    int total = n_images * n_pixels;
    if (gid >= total) return;

    int img_idx = gid / n_pixels;
    int pix     = gid % n_pixels;

    const float* R = rot + img_idx * 6;
    float R0 = R[0], R1 = R[1], R2 = R[2];
    float R3 = R[3], R4 = R[4], R5 = R[5];

    int k0_idx = pix / image_w_half;
    int k1_idx = pix % image_w_half;
    float k0 = (float)(k0_idx - image_h / 2) * upsampling;
    float k1 = (float)(k1_idx) * upsampling;

    float rk0 = k0 * R0 + k1 * R3;
    float rk1 = k0 * R1 + k1 * R4;
    float rk2 = k0 * R2 + k1 * R5;

    int out_idx = (img_idx * n_pixels + pix) * 2;

    if (max_r2 >= 0.0f && rk0 * rk0 + rk1 * rk1 + rk2 * rk2 > max_r2)
    {
        imgs[out_idx + 0] = 0.0f;
        imgs[out_idx + 1] = 0.0f;
        return;
    }

    float gx = rk0 + (float)(N0 / 2);
    float gy = rk1 + (float)(N1 / 2);
    float gz = rk2 + (float)(N2 / 2);

    int x0 = (int)floorf(gx); float fx = gx - x0; int x1 = x0 + 1;
    int y0 = (int)floorf(gy); float fy = gy - y0; int y1 = y0 + 1;
    int z0 = (int)floorf(gz); float fz = gz - z0; int z1 = z0 + 1;

    float val_re = 0.0f, val_im = 0.0f;
    if (x0 >= 0 && x1 < N0 && y0 >= 0 && y1 < N1 && z0 >= 0 && z1 < N2)
    {
        float mfx = 1.0f - fx, mfy = 1.0f - fy, mfz = 1.0f - fz;
        #define RECOVAR_HALF_GATHER(ix, iy, iz, w) \
        { \
            int idx = ((ix) * N1 * N2 + (iy) * N2 + (iz)) * 2; \
            val_re += (w) * vol[idx + 0]; \
            val_im += (w) * vol[idx + 1]; \
        }
        RECOVAR_HALF_GATHER(x0, y0, z0, mfx * mfy * mfz);
        RECOVAR_HALF_GATHER(x0, y0, z1, mfx * mfy *  fz);
        RECOVAR_HALF_GATHER(x0, y1, z0, mfx *  fy * mfz);
        RECOVAR_HALF_GATHER(x0, y1, z1, mfx *  fy *  fz);
        RECOVAR_HALF_GATHER(x1, y0, z0,  fx * mfy * mfz);
        RECOVAR_HALF_GATHER(x1, y0, z1,  fx * mfy *  fz);
        RECOVAR_HALF_GATHER(x1, y1, z0,  fx *  fy * mfz);
        RECOVAR_HALF_GATHER(x1, y1, z1,  fx *  fy *  fz);
        #undef RECOVAR_HALF_GATHER
    }
    imgs[out_idx + 0] = val_re;
    imgs[out_idx + 1] = val_im;
}


/* ================================================================== */
/*  Helper: random rotations                                          */
/* ================================================================== */

static void random_rotation_9(float* out, unsigned seed)
{
    /* Simple Gram-Schmidt to get a random rotation */
    srand(seed);
    auto randf = []() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };

    float a[3] = {randf(), randf(), randf()};
    float na = sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    a[0]/=na; a[1]/=na; a[2]/=na;

    float b[3] = {randf(), randf(), randf()};
    float dot = b[0]*a[0]+b[1]*a[1]+b[2]*a[2];
    b[0]-=dot*a[0]; b[1]-=dot*a[1]; b[2]-=dot*a[2];
    float nb = sqrtf(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
    b[0]/=nb; b[1]/=nb; b[2]/=nb;

    float c[3] = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};

    /* Row-major 3x3 */
    out[0]=a[0]; out[1]=a[1]; out[2]=a[2];
    out[3]=b[0]; out[4]=b[1]; out[5]=b[2];
    out[6]=c[0]; out[7]=c[1]; out[8]=c[2];
}


/* ================================================================== */
/*  Benchmark driver                                                  */
/* ================================================================== */

struct BenchResult {
    float ms_total;
    int n_iters;
    float ms_per_call() const { return ms_total / n_iters; }
};

static BenchResult bench_gpu(int n_warmup, int n_iters, std::function<void()> fn)
{
    for (int i = 0; i < n_warmup; i++) fn();
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; i++) fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return {ms, n_iters};
}


int main(int argc, char** argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 128;
    int n_images = (argc > 2) ? atoi(argv[2]) : 1000;
    int n_iters  = (argc > 3) ? atoi(argv[3]) : 50;
    int padding  = (argc > 4) ? atoi(argv[4]) : 1;   /* padding_factor */

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    printf("\nN=%d, n_images=%d, n_iters=%d, padding=%d\n", N, n_images, n_iters, padding);

    /* Dimensions */
    int img_N = N;                      /* image size */
    int vol_N = N * padding;            /* volume size (padded) */
    int img_x = img_N / 2 + 1;         /* rfft width (RELION uses rfft images) */
    int img_y = img_N;
    int img_xy = img_x * img_y;
    int mdl_x = vol_N / 2 + 1;         /* model rfft width */
    int mdl_y = vol_N;
    int mdl_z = vol_N;
    int mdl_inity = -(vol_N / 2);
    int mdl_initz = -(vol_N / 2);
    int max_r = img_N / 2 - 1;
    int max_r2 = max_r * max_r;

    /* For recovar: full-spectrum images (N*N pixels, interleaved complex) */
    int recovar_n_pixels = img_N * img_N;
    float recovar_max_r2 = (float)max_r2;

    printf("RELION: img_xy=%d (rfft), vol=%dx%dx%d, max_r=%d\n",
           img_xy, mdl_x, mdl_y, mdl_z, max_r);
    printf("Recovar: n_pixels=%d (full), vol=%dx%dx%d, max_r=%.0f\n",
           recovar_n_pixels, vol_N, vol_N, vol_N, sqrtf(recovar_max_r2));

    /* Allocate host data */
    size_t relion_model_size = (size_t)mdl_x * mdl_y * mdl_z;
    size_t recovar_vol_size = (size_t)vol_N * vol_N * vol_N;

    float* h_eulers_9 = (float*)malloc(n_images * 9 * sizeof(float));
    float* h_eulers_6 = (float*)malloc(n_images * 6 * sizeof(float));
    float* h_img_real = (float*)malloc((size_t)n_images * img_xy * sizeof(float));
    float* h_img_imag = (float*)malloc((size_t)n_images * img_xy * sizeof(float));
    float* h_imgs_interleaved = (float*)malloc((size_t)n_images * recovar_n_pixels * 2 * sizeof(float));
    float* h_vol_real = (float*)malloc(relion_model_size * sizeof(float));
    float* h_vol_imag = (float*)malloc(relion_model_size * sizeof(float));
    float* h_vol_interleaved = (float*)malloc(recovar_vol_size * 2 * sizeof(float));
    float* h_ctfs = (float*)malloc(img_xy * sizeof(float));
    float* h_Minvsigma2 = (float*)malloc(img_xy * sizeof(float));
    float* h_weights = (float*)malloc(n_images * sizeof(float));

    /* Initialize data */
    srand(42);
    for (int i = 0; i < n_images; i++) {
        random_rotation_9(h_eulers_9 + i * 9, 42 + i);
        /* Extract first 2 rows for recovar compact format */
        for (int j = 0; j < 6; j++)
            h_eulers_6[i * 6 + j] = h_eulers_9[i * 9 + j];
    }

    for (size_t i = 0; i < (size_t)n_images * img_xy; i++) {
        h_img_real[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_img_imag[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (size_t i = 0; i < (size_t)n_images * recovar_n_pixels * 2; i++) {
        h_imgs_interleaved[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (size_t i = 0; i < relion_model_size; i++) {
        h_vol_real[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_vol_imag[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (size_t i = 0; i < recovar_vol_size * 2; i++) {
        h_vol_interleaved[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < img_xy; i++) {
        h_ctfs[i] = 1.0f;
        h_Minvsigma2[i] = 1.0f;
    }
    for (int i = 0; i < n_images; i++) {
        h_weights[i] = 1.0f;
    }

    /* Recovar half-image data */
    int recovar_half_w = img_N / 2 + 1;
    int recovar_half_pixels = img_N * recovar_half_w;
    float* h_imgs_half = (float*)malloc((size_t)n_images * recovar_half_pixels * 2 * sizeof(float));
    for (size_t i = 0; i < (size_t)n_images * recovar_half_pixels * 2; i++)
        h_imgs_half[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    /* Allocate device memory */
    float *d_eulers_9, *d_eulers_6;
    float *d_img_real, *d_img_imag, *d_imgs_interleaved;
    float *d_vol_real, *d_vol_imag, *d_vol_weight, *d_vol_interleaved;
    float *d_ctfs, *d_Minvsigma2, *d_weights;
    /* For project output */
    float *d_out_real, *d_out_imag, *d_out_interleaved;
    float *d_imgs_half, *d_out_half;

    cudaMalloc(&d_eulers_9, n_images * 9 * sizeof(float));
    cudaMalloc(&d_eulers_6, n_images * 6 * sizeof(float));
    cudaMalloc(&d_img_real, (size_t)n_images * img_xy * sizeof(float));
    cudaMalloc(&d_img_imag, (size_t)n_images * img_xy * sizeof(float));
    cudaMalloc(&d_imgs_interleaved, (size_t)n_images * recovar_n_pixels * 2 * sizeof(float));
    cudaMalloc(&d_vol_real, relion_model_size * sizeof(float));
    cudaMalloc(&d_vol_imag, relion_model_size * sizeof(float));
    cudaMalloc(&d_vol_weight, relion_model_size * sizeof(float));
    cudaMalloc(&d_vol_interleaved, recovar_vol_size * 2 * sizeof(float));
    cudaMalloc(&d_ctfs, img_xy * sizeof(float));
    cudaMalloc(&d_Minvsigma2, img_xy * sizeof(float));
    cudaMalloc(&d_weights, n_images * sizeof(float));
    cudaMalloc(&d_out_real, (size_t)n_images * img_xy * sizeof(float));
    cudaMalloc(&d_out_imag, (size_t)n_images * img_xy * sizeof(float));
    cudaMalloc(&d_out_interleaved, (size_t)n_images * recovar_n_pixels * 2 * sizeof(float));
    cudaMalloc(&d_imgs_half, (size_t)n_images * recovar_half_pixels * 2 * sizeof(float));
    cudaMalloc(&d_out_half, (size_t)n_images * recovar_half_pixels * 2 * sizeof(float));

    /* Copy to device */
    cudaMemcpy(d_eulers_9, h_eulers_9, n_images * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eulers_6, h_eulers_6, n_images * 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_real, h_img_real, (size_t)n_images * img_xy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_imag, h_img_imag, (size_t)n_images * img_xy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgs_interleaved, h_imgs_interleaved, (size_t)n_images * recovar_n_pixels * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vol_real, h_vol_real, relion_model_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vol_imag, h_vol_imag, relion_model_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_vol_weight, 0, relion_model_size * sizeof(float));
    cudaMemcpy(d_vol_interleaved, h_vol_interleaved, recovar_vol_size * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctfs, h_ctfs, img_xy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Minvsigma2, h_Minvsigma2, img_xy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, n_images * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgs_half, h_imgs_half, (size_t)n_images * recovar_half_pixels * 2 * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n");
    printf("=================================================================\n");
    printf("  BACKPROJECT BENCHMARK\n");
    printf("=================================================================\n");

    /* ---- RELION pure interp backproject ---- */
    auto relion_pure_bp = [&]() {
        cudaMemset(d_vol_real, 0, relion_model_size * sizeof(float));
        cudaMemset(d_vol_imag, 0, relion_model_size * sizeof(float));
        cudaMemset(d_vol_weight, 0, relion_model_size * sizeof(float));
        relion_backproject3D<true><<<n_images, RELION_BP_BLOCK_SIZE>>>(
            d_img_real, d_img_imag, d_eulers_9,
            d_vol_real, d_vol_imag, d_vol_weight,
            d_ctfs, d_Minvsigma2, d_weights,
            0.0f, 1.0f,
            max_r2, (float)padding,
            img_x, img_y, img_xy,
            mdl_x, mdl_y, mdl_inity, mdl_initz);
    };
    auto r1 = bench_gpu(3, n_iters, relion_pure_bp);
    printf("RELION pure interp BP:  %8.2f ms/call  (rfft images, 24 atomicAdds/px)\n", r1.ms_per_call());

    /* ---- RELION full backproject ---- */
    auto relion_full_bp = [&]() {
        cudaMemset(d_vol_real, 0, relion_model_size * sizeof(float));
        cudaMemset(d_vol_imag, 0, relion_model_size * sizeof(float));
        cudaMemset(d_vol_weight, 0, relion_model_size * sizeof(float));
        relion_backproject3D<false><<<n_images, RELION_BP_BLOCK_SIZE>>>(
            d_img_real, d_img_imag, d_eulers_9,
            d_vol_real, d_vol_imag, d_vol_weight,
            d_ctfs, d_Minvsigma2, d_weights,
            0.5f, 1.0f,
            max_r2, (float)padding,
            img_x, img_y, img_xy,
            mdl_x, mdl_y, mdl_inity, mdl_initz);
    };
    auto r2 = bench_gpu(3, n_iters, relion_full_bp);
    printf("RELION full BP:         %8.2f ms/call  (+ CTF/weight overhead)\n", r2.ms_per_call());

    /* ---- Recovar backproject ---- */
    int total_threads_bp = n_images * recovar_n_pixels;
    int grid_bp = (total_threads_bp + RECOVAR_BLOCK_SIZE - 1) / RECOVAR_BLOCK_SIZE;
    auto recovar_bp = [&]() {
        cudaMemset(d_vol_interleaved, 0, recovar_vol_size * 2 * sizeof(float));
        recovar_backproject<<<grid_bp, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_imgs_interleaved, d_eulers_6,
            n_images, recovar_n_pixels, img_N, img_N,
            vol_N, vol_N, vol_N, padding,
            recovar_max_r2);
    };
    auto r3 = bench_gpu(3, n_iters, recovar_bp);
    printf("Recovar BP:             %8.2f ms/call  (full images, 16 atomicAdds/px)\n", r3.ms_per_call());

    /* ---- Recovar BP without max_r ---- */
    auto recovar_bp_nomr = [&]() {
        cudaMemset(d_vol_interleaved, 0, recovar_vol_size * 2 * sizeof(float));
        recovar_backproject<<<grid_bp, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_imgs_interleaved, d_eulers_6,
            n_images, recovar_n_pixels, img_N, img_N,
            vol_N, vol_N, vol_N, padding,
            -1.0f);  /* max_r disabled */
    };
    auto r4 = bench_gpu(3, n_iters, recovar_bp_nomr);
    printf("Recovar BP (no max_r):  %8.2f ms/call  (no sphere clip)\n", r4.ms_per_call());

    printf("\n  Speedups (BP):\n");
    printf("    Recovar vs RELION pure:   %.2fx\n", r1.ms_per_call() / r3.ms_per_call());
    printf("    Recovar vs RELION full:   %.2fx\n", r2.ms_per_call() / r3.ms_per_call());
    printf("    max_r speedup (recovar):  %.2fx\n", r4.ms_per_call() / r3.ms_per_call());

    printf("\n  Pixels/call:\n");
    printf("    RELION:  %d images × %d rfft_pixels = %.1fM pixels\n",
           n_images, img_xy, (float)n_images * img_xy / 1e6);
    printf("    Recovar: %d images × %d full_pixels = %.1fM pixels\n",
           n_images, recovar_n_pixels, (float)n_images * recovar_n_pixels / 1e6);
    printf("    (Recovar processes %.1fx more pixels due to full-spectrum images)\n",
           (float)recovar_n_pixels / img_xy);


    printf("\n=================================================================\n");
    printf("  PROJECT (FORWARD) BENCHMARK\n");
    printf("=================================================================\n");

    /* ---- RELION project ---- */
    auto relion_proj = [&]() {
        relion_project3D<<<n_images, RELION_FP_BLOCK_SIZE>>>(
            d_vol_real, d_vol_imag, d_eulers_9,
            d_out_real, d_out_imag,
            max_r2, (float)padding,
            img_x, img_y, img_xy,
            mdl_x, mdl_y, mdl_z, mdl_inity, mdl_initz);
    };
    auto p1 = bench_gpu(3, n_iters, relion_proj);
    printf("RELION project:         %8.2f ms/call  (rfft images)\n", p1.ms_per_call());

    /* ---- Recovar project ---- */
    int total_threads_proj = n_images * recovar_n_pixels;
    int grid_proj = (total_threads_proj + RECOVAR_BLOCK_SIZE - 1) / RECOVAR_BLOCK_SIZE;
    auto recovar_proj = [&]() {
        recovar_project<<<grid_proj, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_out_interleaved, d_eulers_6,
            n_images, recovar_n_pixels, img_N, img_N,
            vol_N, vol_N, vol_N, padding,
            recovar_max_r2);
    };
    auto p2 = bench_gpu(3, n_iters, recovar_proj);
    printf("Recovar project:        %8.2f ms/call  (full images, max_r)\n", p2.ms_per_call());

    /* ---- Recovar project no max_r ---- */
    auto recovar_proj_nomr = [&]() {
        recovar_project<<<grid_proj, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_out_interleaved, d_eulers_6,
            n_images, recovar_n_pixels, img_N, img_N,
            vol_N, vol_N, vol_N, padding,
            -1.0f);
    };
    auto p3 = bench_gpu(3, n_iters, recovar_proj_nomr);
    printf("Recovar project (no mr):%8.2f ms/call  (no sphere clip)\n", p3.ms_per_call());

    printf("\n  Speedups (project):\n");
    printf("    Recovar vs RELION:        %.2fx\n", p1.ms_per_call() / p2.ms_per_call());
    printf("    max_r speedup (recovar):  %.2fx\n", p3.ms_per_call() / p2.ms_per_call());

    printf("\n  Effective throughput:\n");
    float relion_bp_tput = (float)n_images * img_xy / (r1.ms_per_call() / 1000.0f) / 1e9;
    float recovar_bp_tput = (float)n_images * recovar_n_pixels / (r3.ms_per_call() / 1000.0f) / 1e9;
    float relion_proj_tput = (float)n_images * img_xy / (p1.ms_per_call() / 1000.0f) / 1e9;
    float recovar_proj_tput = (float)n_images * recovar_n_pixels / (p2.ms_per_call() / 1000.0f) / 1e9;
    printf("    RELION  BP throughput:    %.2f Gpix/s\n", relion_bp_tput);
    printf("    Recovar BP throughput:    %.2f Gpix/s\n", recovar_bp_tput);
    printf("    RELION  proj throughput:  %.2f Gpix/s\n", relion_proj_tput);
    printf("    Recovar proj throughput:  %.2f Gpix/s\n", recovar_proj_tput);

    printf("\n=================================================================\n");
    printf("  APPLES-TO-APPLES: RELION rfft vs Recovar half-image\n");
    printf("  (same pixel count: %d pixels/image)\n", recovar_half_pixels);
    printf("=================================================================\n");

    int total_half_bp = n_images * recovar_half_pixels;
    int grid_half_bp = (total_half_bp + RECOVAR_BLOCK_SIZE - 1) / RECOVAR_BLOCK_SIZE;

    /* ---- Recovar half-image backproject with max_r ---- */
    auto recovar_half_bp = [&]() {
        cudaMemset(d_vol_interleaved, 0, recovar_vol_size * 2 * sizeof(float));
        recovar_backproject_half<<<grid_half_bp, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_imgs_half, d_eulers_6,
            n_images, recovar_half_pixels, img_N, recovar_half_w, img_N,
            vol_N, vol_N, vol_N, padding,
            recovar_max_r2);
    };
    auto rh1 = bench_gpu(3, n_iters, recovar_half_bp);
    printf("Recovar half BP (max_r): %7.2f ms/call  (16 atomicAdds/px)\n", rh1.ms_per_call());
    printf("RELION  pure BP:         %7.2f ms/call  (24 atomicAdds/px)\n", r1.ms_per_call());
    printf("  => Recovar half vs RELION pure: %.2fx faster\n",
           r1.ms_per_call() / rh1.ms_per_call());
    printf("  => Recovar half vs RELION full: %.2fx faster\n",
           r2.ms_per_call() / rh1.ms_per_call());

    /* ---- Recovar half-image project with max_r ---- */
    auto recovar_half_proj = [&]() {
        recovar_project_half<<<grid_half_bp, RECOVAR_BLOCK_SIZE>>>(
            d_vol_interleaved, d_out_half, d_eulers_6,
            n_images, recovar_half_pixels, img_N, recovar_half_w, img_N,
            vol_N, vol_N, vol_N, padding,
            recovar_max_r2);
    };
    auto ph1 = bench_gpu(3, n_iters, recovar_half_proj);
    printf("\nRecovar half proj (max_r):%7.2f ms/call\n", ph1.ms_per_call());
    printf("RELION  proj:             %7.2f ms/call\n", p1.ms_per_call());
    printf("  => Recovar half vs RELION: %.2fx faster\n",
           p1.ms_per_call() / ph1.ms_per_call());

    /* Also benchmark max_r sweep for recovar backproject */
    printf("\n=================================================================\n");
    printf("  MAX_R SWEEP (Recovar backproject)\n");
    printf("=================================================================\n");

    float r_values[] = {(float)(N/2), (float)(N/2-1), (float)(N/4), (float)(N/8), (float)(N/16)};
    int n_r = sizeof(r_values) / sizeof(r_values[0]);
    for (int ri = 0; ri < n_r; ri++) {
        float mr = r_values[ri];
        if (mr < 2) continue;
        float mr2 = mr * mr;
        float area_frac = (3.14159f * mr2) / (float)(img_N * img_N);

        auto bp_sweep = [&]() {
            cudaMemset(d_vol_interleaved, 0, recovar_vol_size * 2 * sizeof(float));
            recovar_backproject<<<grid_bp, RECOVAR_BLOCK_SIZE>>>(
                d_vol_interleaved, d_imgs_interleaved, d_eulers_6,
                n_images, recovar_n_pixels, img_N, img_N,
                vol_N, vol_N, vol_N, padding,
                mr2);
        };
        auto rs = bench_gpu(3, n_iters, bp_sweep);
        printf("  max_r=%5.0f  area_frac=%.3f  %8.2f ms/call  speedup_vs_none=%.2fx  (predicted=%.2fx)\n",
               mr, area_frac, rs.ms_per_call(),
               r4.ms_per_call() / rs.ms_per_call(),
               1.0f / area_frac);
    }

    /* Cleanup */
    cudaFree(d_eulers_9); cudaFree(d_eulers_6);
    cudaFree(d_img_real); cudaFree(d_img_imag); cudaFree(d_imgs_interleaved);
    cudaFree(d_vol_real); cudaFree(d_vol_imag); cudaFree(d_vol_weight);
    cudaFree(d_vol_interleaved);
    cudaFree(d_ctfs); cudaFree(d_Minvsigma2); cudaFree(d_weights);
    cudaFree(d_out_real); cudaFree(d_out_imag); cudaFree(d_out_interleaved);
    cudaFree(d_imgs_half); cudaFree(d_out_half);

    free(h_eulers_9); free(h_eulers_6);
    free(h_img_real); free(h_img_imag); free(h_imgs_interleaved);
    free(h_vol_real); free(h_vol_imag); free(h_vol_interleaved);
    free(h_ctfs); free(h_Minvsigma2); free(h_weights);
    free(h_imgs_half);

    return 0;
}
