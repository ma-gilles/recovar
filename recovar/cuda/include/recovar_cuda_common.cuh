// Device helpers and RELION-mode texture projection shared by recovar's pipeline library
// (recovar/cuda/cuda_backproject.cu) and the EM library (recovar/em/cuda/relax_kernels.cu).
// Extracted verbatim from the single translation unit by the relax split (seam S4); both
// libraries include this one implementation. Requires <cuda_runtime.h> and the standard
// headers the translation units include first.
#ifndef RECOVAR_CUDA_COMMON_CUH
#define RECOVAR_CUDA_COMMON_CUH


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

    cudaError_t err = recovar::scratch_alloc((void**)&real, n_voxels * sizeof(float), s);
    if (err != cudaSuccess) goto cleanup;
    err = recovar::scratch_alloc((void**)&imag, n_voxels * sizeof(float), s);
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
    if (real) recovar::scratch_free(real, s);
    if (imag) recovar::scratch_free(imag, s);
    return err;
}

#endif  // RECOVAR_CUDA_COMMON_CUH
