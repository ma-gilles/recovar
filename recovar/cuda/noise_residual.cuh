// Unused native residual-statistics experiment. Compact outputs, explicit scratch.
#pragma once

namespace recovar_noise_residual {
constexpr int kPixels = 32;
constexpr int kRows = 8;
constexpr int kRowTile = 256;
constexpr int kImageTile = 8192;

template <typename T, typename C>
__device__ C cross_term(C proj, C summed) {
    if (summed.x == T(0) && summed.y == T(0)) return C{T(0), T(0)};
    return C{proj.x * summed.x + proj.y * summed.y,
             proj.y * summed.x - proj.x * summed.y};
}

template <typename T>
__device__ double a2_term(T abs2, T ctf, double variance) {
    if (ctf == T(0)) return 0.0;
    const double raw = static_cast<double>(ctf) * variance;
    return static_cast<double>(abs2) * raw;
}

template <typename T, typename C>
__global__ void pixel_partials(const C* proj, const T* abs2, const C* summed,
                               const T* ctf, const double* variance,
                               int64_t rows, int64_t pixels,
                               double* out_a2, C* out_cross) {
    const int px = threadIdx.x, lane = threadIdx.y;
    const int64_t pixel = int64_t(blockIdx.x) * kPixels + px;
    const int64_t begin = int64_t(blockIdx.y) * kRowTile;
    double a2 = 0.0;
    C cross{T(0), T(0)};
    if (pixel < pixels) {
        for (int64_t row = begin + lane; row < rows && row < begin + kRowTile; row += kRows) {
            const int64_t index = row * pixels + pixel;
            a2 += a2_term(abs2[index], ctf[index], variance[pixel]);
            C term = cross_term<T>(proj[index], summed[index]);
            cross.x += term.x;
            cross.y += term.y;
        }
    }
    __shared__ double sa[kRows][kPixels];
    __shared__ C sc[kRows][kPixels];
    sa[lane][px] = a2;
    sc[lane][px] = cross;
    __syncthreads();
    for (int step = kRows / 2; step; step /= 2) {
        if (lane < step) {
            sa[lane][px] += sa[lane + step][px];
            sc[lane][px].x += sc[lane + step][px].x;
            sc[lane][px].y += sc[lane + step][px].y;
        }
        __syncthreads();
    }
    if (lane == 0 && pixel < pixels) {
        const int64_t index = int64_t(blockIdx.y) * pixels + pixel;
        out_a2[index] = sa[0][px];
        out_cross[index] = sc[0][px];
    }
}

template <typename T, typename C>
__global__ void finish_pixels(const double* partial_a2, const C* partial_cross,
                              int64_t tiles, int64_t pixels,
                              double* out_a2, C* out_cross) {
    const int px = threadIdx.x, lane = threadIdx.y;
    const int64_t pixel = int64_t(blockIdx.x) * kPixels + px;
    double a2 = 0.0;
    C cross{T(0), T(0)};
    if (pixel < pixels) {
        for (int64_t tile = lane; tile < tiles; tile += kRows) {
            const int64_t index = tile * pixels + pixel;
            a2 += partial_a2[index];
            cross.x += partial_cross[index].x;
            cross.y += partial_cross[index].y;
        }
    }
    __shared__ double sa[kRows][kPixels];
    __shared__ C sc[kRows][kPixels];
    sa[lane][px] = a2;
    sc[lane][px] = cross;
    __syncthreads();
    for (int step = kRows / 2; step; step /= 2) {
        if (lane < step) {
            sa[lane][px] += sa[lane + step][px];
            sc[lane][px].x += sc[lane + step][px].x;
            sc[lane][px].y += sc[lane + step][px].y;
        }
        __syncthreads();
    }
    if (lane == 0 && pixel < pixels) {
        out_a2[pixel] = sa[0][px];
        out_cross[pixel] = sc[0][px];
    }
}

template <typename T, typename C>
__global__ void image_partials(const C* proj, const T* abs2, const C* summed,
                               const T* ctf, const double* variance,
                               const bool* scale_mask, bool compute_scale,
                               int64_t poses, int64_t pixels, int64_t tiles,
                               double* partial) {
    const int lane = threadIdx.x;
    const int64_t count = poses * pixels;
    const int64_t begin = int64_t(blockIdx.y) * kImageTile;
    double values[4] = {0.0, 0.0, 0.0, 0.0};
    for (int64_t i = begin + lane; i < count && i < begin + kImageTile; i += 256) {
        const int64_t index = int64_t(blockIdx.x) * count + i;
        const int64_t pixel = i % pixels;
        const double nv = variance[pixel];
        C cross = cross_term<T>(proj[index], summed[index]);
        const double a2 = a2_term(abs2[index], ctf[index], nv);
        const double xa = nv * static_cast<double>(cross.x);
        values[0] += a2;
        values[1] += xa;
        if (compute_scale) {
            values[2] += scale_mask[pixel] ? a2 : 0.0;
            // Preserve the original unguarded variance * masked-zero behavior.
            values[3] += nv * static_cast<double>(scale_mask[pixel] ? cross.x : T(0));
        }
    }
    __shared__ double sums[4][256];
    for (int j = 0; j < 4; ++j) sums[j][lane] = values[j];
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step)
            for (int j = 0; j < 4; ++j) sums[j][lane] += sums[j][lane + step];
        __syncthreads();
    }
    if (lane == 0)
        for (int j = 0; j < 4; ++j)
            partial[(int64_t(blockIdx.x) * tiles + blockIdx.y) * 4 + j] = sums[j][0];
}

__global__ void finish_images(const double* partial, int64_t tiles,
                              double* a2, double* xa, double* scale_a2, double* scale_xa) {
    const int lane = threadIdx.x;
    __shared__ double sums[4][256];
    for (int j = 0; j < 4; ++j) {
        double value = 0.0;
        for (int64_t tile = lane; tile < tiles; tile += 256)
            value += partial[(int64_t(blockIdx.x) * tiles + tile) * 4 + j];
        sums[j][lane] = value;
    }
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (lane < step)
            for (int j = 0; j < 4; ++j) sums[j][lane] += sums[j][lane + step];
        __syncthreads();
    }
    if (lane == 0) {
        a2[blockIdx.x] = sums[0][0];
        xa[blockIdx.x] = sums[1][0];
        scale_a2[blockIdx.x] = sums[2][0];
        scale_xa[blockIdx.x] = sums[3][0];
    }
}

bool shape(const ffi::AnyBuffer& value, ffi::DataType type,
           std::initializer_list<int64_t> dimensions) {
    const auto actual = value.dimensions();
    if (value.element_type() != type || actual.size() != dimensions.size()) return false;
    int i = 0;
    for (int64_t n : dimensions) if (actual[i++] != n) return false;
    return true;
}

template <typename T, typename C>
cudaError_t launch(cudaStream_t stream, const ffi::AnyBuffer& proj,
                   const ffi::AnyBuffer& abs2, const ffi::AnyBuffer& summed,
                   const ffi::AnyBuffer& ctf, const ffi::AnyBuffer& variance,
                   const ffi::AnyBuffer& mask, bool scale, int64_t b, int64_t r, int64_t p,
                   double* pixel_a2, C* pixel_cross, double* image_a2, double* image_xa,
                   double* scale_a2, double* scale_xa,
                   double* scratch_a2, C* scratch_cross, double* scratch_image) {
    const int64_t row_tiles = (b*r + kRowTile-1)/kRowTile;
    const int64_t image_tiles = (r*p + kImageTile-1)/kImageTile;
    const auto* pr = static_cast<const C*>(proj.untyped_data());
    const auto* aa = static_cast<const T*>(abs2.untyped_data());
    const auto* su = static_cast<const C*>(summed.untyped_data());
    const auto* ct = static_cast<const T*>(ctf.untyped_data());
    const auto* nv = static_cast<const double*>(variance.untyped_data());
    pixel_partials<T,C><<<dim3((p+31)/32,row_tiles),dim3(32,8),0,stream>>>(
        pr,aa,su,ct,nv,b*r,p,scratch_a2,scratch_cross);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    finish_pixels<T,C><<<dim3((p+31)/32),dim3(32,8),0,stream>>>(
        scratch_a2,scratch_cross,row_tiles,p,pixel_a2,pixel_cross);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    image_partials<T,C><<<dim3(b,image_tiles),256,0,stream>>>(
        pr,aa,su,ct,nv,static_cast<const bool*>(mask.untyped_data()),scale,
        r,p,image_tiles,scratch_image);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    finish_images<<<dim3(b),256,0,stream>>>(scratch_image,image_tiles,
        image_a2,image_xa,scale_a2,scale_xa);
    return cudaGetLastError();
}

ffi::Error run(cudaStream_t stream, int64_t compute_scale,
               ffi::AnyBuffer proj, ffi::AnyBuffer abs2, ffi::AnyBuffer summed,
               ffi::AnyBuffer ctf, ffi::AnyBuffer variance, ffi::AnyBuffer mask,
               ffi::Result<ffi::AnyBuffer> pixel_a2, ffi::Result<ffi::AnyBuffer> pixel_cross,
               ffi::Result<ffi::AnyBuffer> image_a2, ffi::Result<ffi::AnyBuffer> image_xa,
               ffi::Result<ffi::AnyBuffer> scale_a2, ffi::Result<ffi::AnyBuffer> scale_xa,
               ffi::Result<ffi::AnyBuffer> scratch_a2, ffi::Result<ffi::AnyBuffer> scratch_cross,
               ffi::Result<ffi::AnyBuffer> scratch_image) {
    const auto dims = proj.dimensions();
    if (dims.size() != 3 || (compute_scale != 0 && compute_scale != 1))
        return ffi::Error::InvalidArgument("NoiseResidual: expected B,R,P and boolean scale policy");
    const int64_t b=dims[0], r=dims[1], p=dims[2];
    const bool double_input = proj.element_type() == ffi::DataType::C128;
    const auto real = double_input ? ffi::DataType::F64 : ffi::DataType::F32;
    const auto complex = double_input ? ffi::DataType::C128 : ffi::DataType::C64;
    // Bound products before multiplication and respect the grid.y limit.
    if (b <= 0 || r <= 0 || p <= 0 || b > 65535 || r > 65535 || p > 65535 ||
        b*r > int64_t(65535)*kRowTile || r*p > int64_t(65535)*kImageTile)
        return ffi::Error::InvalidArgument("NoiseResidual: invalid or overflowing launch geometry");
    const int64_t rt=(b*r+kRowTile-1)/kRowTile, it=(r*p+kImageTile-1)/kImageTile;
    if (!shape(proj,complex,{b,r,p}) || !shape(abs2,real,{b,r,p}) ||
        !shape(summed,complex,{b,r,p}) || !shape(ctf,real,{b,r,p}) ||
        !shape(variance,ffi::DataType::F64,{p}) || !shape(mask,ffi::DataType::PRED,{p}) ||
        !shape(*pixel_a2,ffi::DataType::F64,{p}) || !shape(*pixel_cross,complex,{p}) ||
        !shape(*image_a2,ffi::DataType::F64,{b}) || !shape(*image_xa,ffi::DataType::F64,{b}) ||
        !shape(*scale_a2,ffi::DataType::F64,{b}) || !shape(*scale_xa,ffi::DataType::F64,{b}) ||
        !shape(*scratch_a2,ffi::DataType::F64,{rt,p}) || !shape(*scratch_cross,complex,{rt,p}) ||
        !shape(*scratch_image,ffi::DataType::F64,{b,it,4}))
        return ffi::Error::InvalidArgument("NoiseResidual: input/output shape or dtype mismatch");
    cudaError_t error;
    if (double_input)
        error=launch<double,double2>(stream,proj,abs2,summed,ctf,variance,mask,compute_scale,b,r,p,
            static_cast<double*>(pixel_a2->untyped_data()),static_cast<double2*>(pixel_cross->untyped_data()),
            static_cast<double*>(image_a2->untyped_data()),static_cast<double*>(image_xa->untyped_data()),
            static_cast<double*>(scale_a2->untyped_data()),static_cast<double*>(scale_xa->untyped_data()),
            static_cast<double*>(scratch_a2->untyped_data()),static_cast<double2*>(scratch_cross->untyped_data()),
            static_cast<double*>(scratch_image->untyped_data()));
    else
        error=launch<float,float2>(stream,proj,abs2,summed,ctf,variance,mask,compute_scale,b,r,p,
            static_cast<double*>(pixel_a2->untyped_data()),static_cast<float2*>(pixel_cross->untyped_data()),
            static_cast<double*>(image_a2->untyped_data()),static_cast<double*>(image_xa->untyped_data()),
            static_cast<double*>(scale_a2->untyped_data()),static_cast<double*>(scale_xa->untyped_data()),
            static_cast<double*>(scratch_a2->untyped_data()),static_cast<float2*>(scratch_cross->untyped_data()),
            static_cast<double*>(scratch_image->untyped_data()));
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    return ffi::Error::Success();
}
}  // namespace recovar_noise_residual

XLA_FFI_DEFINE_HANDLER_SYMBOL(NoiseResidualStatistics, recovar_noise_residual::run,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<int64_t>("compute_scale")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
