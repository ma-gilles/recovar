// Flat-row translate-and-sum for RECOVAR's device-resident K=1 pass-2 M-step.
//
// Replaces the XLA reduce fusion of
// recovar/em/sparse_pass2/resident_pass2.py::_resident_block_weighted_sums,
// which gathers a ``[rows, translations, pixels]`` shifted tile per M-step
// block (2.3 GB at the hp3 state) and contracts it with the posterior.  This
// kernel keeps only the per-image unshifted operands and applies RELION's
// translation inside the reduction, exactly as the fused-translate fine
// scorer does, so no ``[rows, T, P]`` tile is ever materialised:
//
//   summed[r, p]        = sum_t posterior[r, t] * shift_t(recon_image[id[r], p])
//   summed_masked[r, p] = sum_t posterior[r, t] * shift_t(noise_image[id[r], p])
//   probs_sum_t[r]      = sum_t posterior[r, t]
//
// Addressing follows the flat-row family (``relion_fine_diff2_fused_translate_
// runtime_flat_rows_f32`` and ``relion_wavg_sequential_runtime_flat_rows_
// triplet_f32``): every row carries its own image id and a negative id marks
// padding.  ``x``/``y`` come from RECOVAR's centered packed-half pixel index,
// the convention ``relion_translate_score_f32_kernel`` uses to build the very
// tile this kernel replaces; the fine scorer derives the same ``(x, y)`` from
// its own RELION full-pixel rectangle instead.
//
// Arithmetic contract.  The phase and the complex rotation are
// ``relion_score_translate_f32`` split in two, so that one ``sincosf`` per
// (pixel, translation) serves every row a block owns.  Both halves are written
// with the same rounding intrinsics as the original, which are not contractible,
// so the split emits the same operations.  The reduction over translations is
// sequential in increasing ``t`` with a rounded multiply followed by a rounded
// add, the order RELION's Wavg accumulation uses.
//
// This header is included from cuda_backproject.cu after the anonymous
// namespace that defines relion_score_translate_f32.

namespace recovar_relion_translate_sum {

constexpr int kBlockThreads = 256;
constexpr int kMaxSharedBytes = 32 * 1024;

// The phase of relion_score_translate_f32, verbatim: one rounded y product
// followed by an x FMA.
__device__ __forceinline__ float translate_phase_f32(
    int x, int y, float tx, float ty)
{
    return __fmaf_rn(
        static_cast<float>(x), tx,
        __fmul_rn(static_cast<float>(y), ty));
}

// The complex rotation of relion_score_translate_f32, verbatim.
__device__ __forceinline__ float2 translate_rotate_f32(
    float2 value, float sine, float cosine)
{
    const float translated_real = __fmaf_rn(
        cosine, value.x,
        -__fmul_rn(sine, value.y));
    const float translated_imag = __fmaf_rn(
        sine, value.x,
        __fmul_rn(cosine, value.y));
    return make_float2(translated_real, translated_imag);
}

template <int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(kBlockThreads)
translate_sum_flat_rows_f32_kernel(
    const float2* __restrict__ recon_image,          // [B, P]
    const float2* __restrict__ noise_image,          // [B, P]
    const int32_t* __restrict__ row_image_ids,       // [Q]
    const float* __restrict__ posterior,             // [Q, T]
    const float* __restrict__ translation_angles,    // [T, 2]
    const int32_t* __restrict__ pixel_indices,       // [P]
    const int32_t* __restrict__ runtime_valid_rows,  // scalar or null
    const int32_t* __restrict__ runtime_logical_pixels,  // scalar or null
    float2* __restrict__ summed,                     // [Q, P]
    float2* __restrict__ summed_masked,              // [Q, P]
    float* __restrict__ probs_sum_t,                 // [Q]
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width)
{
    const int64_t row_base =
        static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK;
    if (row_base >= row_count) return;

    const int64_t valid_rows = runtime_valid_rows == nullptr
        ? row_count
        : static_cast<int64_t>(runtime_valid_rows[0]);
    const int64_t logical_pixels = runtime_logical_pixels == nullptr
        ? pixel_capacity
        : static_cast<int64_t>(runtime_logical_pixels[0]);
    if (valid_rows < 0 || valid_rows > row_count ||
        logical_pixels < 0 || logical_pixels > pixel_capacity) {
        // Fail closed exactly as the runtime flat-row kernels do.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            summed[0] = make_float2(nanf(""), nanf(""));
            summed_masked[0] = make_float2(nanf(""), nanf(""));
            probs_sum_t[0] = nanf("");
        }
        return;
    }
    // Rows past the valid count keep the zeros the launcher wrote.
    if (row_base >= valid_rows) return;

    int32_t image_ids[ROWS_PER_BLOCK];
    bool live[ROWS_PER_BLOCK];
    bool any_live = false;
#pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        const int64_t row = row_base + i;
        image_ids[i] = -1;
        live[i] = false;
        if (row < valid_rows) {
            const int32_t id = row_image_ids[row];
            if (id >= static_cast<int32_t>(batch_size)) {
                // Out-of-range map: fail closed on this row's outputs.
                for (int64_t pixel = threadIdx.x;
                     pixel < pixel_capacity;
                     pixel += kBlockThreads) {
                    const int64_t base = row * pixel_capacity + pixel;
                    summed[base] = make_float2(nanf(""), nanf(""));
                    summed_masked[base] = make_float2(nanf(""), nanf(""));
                }
                if (threadIdx.x == 0) probs_sum_t[row] = nanf("");
                continue;
            }
            // A negative id is padding: it reads nothing and stays zero.
            if (id >= 0) {
                image_ids[i] = id;
                live[i] = true;
                any_live = true;
            }
        }
    }
    if (!any_live) return;

    // Shared copies of the two block-uniform tables.  Without them the inner
    // translation loop re-reads them once per pixel pass.
    extern __shared__ float translate_sum_shared[];
    float* shared_angles = translate_sum_shared;                     // [2T]
    float* shared_posterior = shared_angles + 2 * translation_count;  // [R, T]
    for (int64_t index = threadIdx.x;
         index < 2 * translation_count;
         index += kBlockThreads) {
        shared_angles[index] = translation_angles[index];
    }
    for (int64_t index = threadIdx.x;
         index < ROWS_PER_BLOCK * translation_count;
         index += kBlockThreads) {
        const int64_t i = index / translation_count;
        const int64_t t = index - i * translation_count;
        const int64_t row = row_base + i;
        // Padding rows past the valid count are never read again, but a
        // defined zero keeps the shared tile free of stale values.
        shared_posterior[index] = row < valid_rows
            ? posterior[row * translation_count + t]
            : 0.0f;
    }
    __syncthreads();

    // probs_sum_t is the same sequential translation sum, one row per lane.
    if (threadIdx.x < ROWS_PER_BLOCK) {
        const int i = threadIdx.x;
        if (live[i]) {
            float mass = 0.0f;
            for (int64_t t = 0; t < translation_count; ++t) {
                mass = __fadd_rn(
                    mass, shared_posterior[i * translation_count + t]);
            }
            probs_sum_t[row_base + i] = mass;
        }
    }

    for (int64_t pixel = threadIdx.x;
         pixel < logical_pixels;
         pixel += kBlockThreads)
    {
        const int pixel_index = pixel_indices[pixel];
        const int x = pixel_index % image_half_width;
        const int y = pixel_index / image_half_width - image_h / 2;

        float2 recon_value[ROWS_PER_BLOCK];
        float2 noise_value[ROWS_PER_BLOCK];
        float2 recon_acc[ROWS_PER_BLOCK];
        float2 noise_acc[ROWS_PER_BLOCK];
#pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            recon_acc[i] = make_float2(0.0f, 0.0f);
            noise_acc[i] = make_float2(0.0f, 0.0f);
            recon_value[i] = make_float2(0.0f, 0.0f);
            noise_value[i] = make_float2(0.0f, 0.0f);
            if (live[i]) {
                const int64_t base =
                    static_cast<int64_t>(image_ids[i]) * pixel_capacity + pixel;
                recon_value[i] = recon_image[base];
                noise_value[i] = noise_image[base];
            }
        }

        for (int64_t translation = 0;
             translation < translation_count;
             ++translation)
        {
            const float tx = shared_angles[2 * translation];
            const float ty = shared_angles[2 * translation + 1];
            const float phase = translate_phase_f32(x, y, tx, ty);
            float sine;
            float cosine;
            sincosf(phase, &sine, &cosine);
#pragma unroll
            for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                if (!live[i]) continue;
                const float weight =
                    shared_posterior[i * translation_count + translation];
                const float2 recon_shifted =
                    translate_rotate_f32(recon_value[i], sine, cosine);
                const float2 noise_shifted =
                    translate_rotate_f32(noise_value[i], sine, cosine);
                recon_acc[i].x = __fadd_rn(
                    recon_acc[i].x, __fmul_rn(weight, recon_shifted.x));
                recon_acc[i].y = __fadd_rn(
                    recon_acc[i].y, __fmul_rn(weight, recon_shifted.y));
                noise_acc[i].x = __fadd_rn(
                    noise_acc[i].x, __fmul_rn(weight, noise_shifted.x));
                noise_acc[i].y = __fadd_rn(
                    noise_acc[i].y, __fmul_rn(weight, noise_shifted.y));
            }
        }

#pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            if (!live[i]) continue;
            const int64_t base = (row_base + i) * pixel_capacity + pixel;
            summed[base] = recon_acc[i];
            summed_masked[base] = noise_acc[i];
        }
    }
}

inline int64_t shared_bytes_for(int rows_per_block, int64_t translation_count)
{
    return static_cast<int64_t>(
        (2 + rows_per_block) * translation_count * sizeof(float));
}

// Rows a block owns.  More rows share one sincosf but cost registers; the
// launcher keeps the shared tables inside kMaxSharedBytes and never asks for
// more rows than the problem has.
inline int choose_rows_per_block(
    int requested, int64_t row_count, int64_t translation_count)
{
    int rows = requested > 0 ? requested : 4;
    if (rows != 1 && rows != 2 && rows != 4 && rows != 8) rows = 4;
    while (rows > 1 && static_cast<int64_t>(rows) > row_count) rows /= 2;
    while (rows > 1 &&
           shared_bytes_for(rows, translation_count) > kMaxSharedBytes) {
        rows /= 2;
    }
    return rows;
}

template <int ROWS_PER_BLOCK>
cudaError_t launch_templated(
    cudaStream_t stream,
    const float2* recon_image,
    const float2* noise_image,
    const int32_t* row_image_ids,
    const float* posterior,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const int32_t* runtime_valid_rows,
    const int32_t* runtime_logical_pixels,
    float2* summed,
    float2* summed_masked,
    float* probs_sum_t,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width)
{
    const int64_t blocks =
        (row_count + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const size_t shared = static_cast<size_t>(
        shared_bytes_for(ROWS_PER_BLOCK, translation_count));
    translate_sum_flat_rows_f32_kernel<ROWS_PER_BLOCK>
        <<<static_cast<unsigned>(blocks), kBlockThreads, shared, stream>>>(
            recon_image,
            noise_image,
            row_image_ids,
            posterior,
            translation_angles,
            pixel_indices,
            runtime_valid_rows,
            runtime_logical_pixels,
            summed,
            summed_masked,
            probs_sum_t,
            batch_size,
            row_count,
            translation_count,
            pixel_capacity,
            image_h,
            image_half_width);
    return cudaGetLastError();
}

inline cudaError_t launch(
    cudaStream_t stream,
    const float2* recon_image,
    const float2* noise_image,
    const int32_t* row_image_ids,
    const float* posterior,
    const float* translation_angles,
    const int32_t* pixel_indices,
    const int32_t* runtime_valid_rows,
    const int32_t* runtime_logical_pixels,
    float2* summed,
    float2* summed_masked,
    float* probs_sum_t,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t pixel_capacity,
    int image_h,
    int image_half_width,
    int requested_rows_per_block)
{
    if (row_count == 0 || pixel_capacity == 0) return cudaSuccess;
    // The zero fill owns every padding row, every zero-mass row and the
    // physical pixel tail; the kernel writes only what it is allowed to read.
    cudaError_t err = cudaMemsetAsync(
        summed,
        0,
        static_cast<size_t>(row_count) * pixel_capacity * sizeof(float2),
        stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(
        summed_masked,
        0,
        static_cast<size_t>(row_count) * pixel_capacity * sizeof(float2),
        stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(
        probs_sum_t,
        0,
        static_cast<size_t>(row_count) * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    if (translation_count == 0) return cudaSuccess;

    const int rows = choose_rows_per_block(
        requested_rows_per_block, row_count, translation_count);
    if (shared_bytes_for(rows, translation_count) > kMaxSharedBytes)
        return cudaErrorInvalidValue;

#define RECOVAR_TRANSLATE_SUM_DISPATCH(R)                    \
    case R:                                                  \
        return launch_templated<R>(                          \
            stream, recon_image, noise_image, row_image_ids, \
            posterior, translation_angles, pixel_indices,    \
            runtime_valid_rows, runtime_logical_pixels,      \
            summed, summed_masked, probs_sum_t, batch_size,  \
            row_count, translation_count, pixel_capacity,    \
            image_h, image_half_width)

    switch (rows) {
        RECOVAR_TRANSLATE_SUM_DISPATCH(1);
        RECOVAR_TRANSLATE_SUM_DISPATCH(2);
        RECOVAR_TRANSLATE_SUM_DISPATCH(4);
        RECOVAR_TRANSLATE_SUM_DISPATCH(8);
        default:
            return cudaErrorInvalidValue;
    }
#undef RECOVAR_TRANSLATE_SUM_DISPATCH
}

ffi::Error impl(
    cudaStream_t stream,
    int64_t image_h,
    int64_t image_half_width,
    int64_t rows_per_block,
    ffi::AnyBuffer recon_image,
    ffi::AnyBuffer noise_image,
    ffi::AnyBuffer row_image_ids,
    ffi::AnyBuffer posterior,
    ffi::AnyBuffer translation_angles,
    ffi::AnyBuffer pixel_indices,
    ffi::AnyBuffer n_valid_rows,
    ffi::AnyBuffer logical_pixel_count,
    ffi::Result<ffi::AnyBuffer> summed,
    ffi::Result<ffi::AnyBuffer> summed_masked,
    ffi::Result<ffi::AnyBuffer> probs_sum_t)
{
    if (recon_image.element_type() != ffi::DataType::C64 ||
        noise_image.element_type() != ffi::DataType::C64 ||
        row_image_ids.element_type() != ffi::DataType::S32 ||
        posterior.element_type() != ffi::DataType::F32 ||
        translation_angles.element_type() != ffi::DataType::F32 ||
        pixel_indices.element_type() != ffi::DataType::S32 ||
        n_valid_rows.element_type() != ffi::DataType::S32 ||
        logical_pixel_count.element_type() != ffi::DataType::S32 ||
        summed->element_type() != ffi::DataType::C64 ||
        summed_masked->element_type() != ffi::DataType::C64 ||
        probs_sum_t->element_type() != ffi::DataType::F32 ||
        n_valid_rows.dimensions().size() != 0 ||
        logical_pixel_count.dimensions().size() != 0)
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid buffers");

    const auto recon_dims = recon_image.dimensions();
    const auto noise_dims = noise_image.dimensions();
    const auto row_dims = row_image_ids.dimensions();
    const auto posterior_dims = posterior.dimensions();
    const auto angle_dims = translation_angles.dimensions();
    const auto index_dims = pixel_indices.dimensions();
    const auto summed_dims = summed->dimensions();
    const auto masked_dims = summed_masked->dimensions();
    const auto mass_dims = probs_sum_t->dimensions();
    if (recon_dims.size() != 2 || noise_dims.size() != 2 ||
        row_dims.size() != 1 || posterior_dims.size() != 2 ||
        angle_dims.size() != 2 || angle_dims[1] != 2 ||
        index_dims.size() != 1 || summed_dims.size() != 2 ||
        masked_dims.size() != 2 || mass_dims.size() != 1)
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid ranks");

    const int64_t batch_size = recon_dims[0];
    const int64_t pixel_capacity = recon_dims[1];
    const int64_t row_count = row_dims[0];
    const int64_t translation_count = angle_dims[0];
    if (batch_size <= 0 || pixel_capacity <= 0 || row_count <= 0 ||
        translation_count <= 0 ||
        pixel_capacity > std::numeric_limits<int>::max() ||
        noise_dims[0] != batch_size || noise_dims[1] != pixel_capacity ||
        posterior_dims[0] != row_count ||
        posterior_dims[1] != translation_count ||
        index_dims[0] != pixel_capacity ||
        summed_dims[0] != row_count || summed_dims[1] != pixel_capacity ||
        masked_dims[0] != row_count || masked_dims[1] != pixel_capacity ||
        mass_dims[0] != row_count)
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: inconsistent topology");
    if (image_h <= 0 || image_half_width <= 0 ||
        image_h > std::numeric_limits<int>::max() ||
        image_half_width > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: invalid image dimensions");

    const int64_t rows_per_grid_block = rows_per_block > 0 ? rows_per_block : 1;
    const int64_t blocks_needed =
        (row_count + rows_per_grid_block - 1) / rows_per_grid_block;
    if (blocks_needed > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument(
            "RelionTranslateSumFlatRowsF32: row count exceeds the grid");

    cudaError_t err = launch(
        stream,
        reinterpret_cast<const float2*>(recon_image.untyped_data()),
        reinterpret_cast<const float2*>(noise_image.untyped_data()),
        static_cast<const int32_t*>(row_image_ids.untyped_data()),
        static_cast<const float*>(posterior.untyped_data()),
        static_cast<const float*>(translation_angles.untyped_data()),
        static_cast<const int32_t*>(pixel_indices.untyped_data()),
        static_cast<const int32_t*>(n_valid_rows.untyped_data()),
        static_cast<const int32_t*>(logical_pixel_count.untyped_data()),
        reinterpret_cast<float2*>(summed->untyped_data()),
        reinterpret_cast<float2*>(summed_masked->untyped_data()),
        static_cast<float*>(probs_sum_t->untyped_data()),
        batch_size,
        row_count,
        translation_count,
        pixel_capacity,
        static_cast<int>(image_h),
        static_cast<int>(image_half_width),
        static_cast<int>(rows_per_block));
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

}  // namespace recovar_relion_translate_sum

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionTranslateSumFlatRowsF32,
    recovar_relion_translate_sum::impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("image_h")
        .Attr<int64_t>("image_half_width")
        .Attr<int64_t>("rows_per_block")
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
        .Ret<ffi::AnyBuffer>());
