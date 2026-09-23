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
    const float* images,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* block_sums,
    int64_t block_sum_stride)
{
    // One image per blockIdx.y.  Every block keeps the exact texel range and
    // addition order of the per-image launch, so batching changes no bits.
    const float* image = images + static_cast<int64_t>(blockIdx.y) * image_size;
    float* block_sum = block_sums + static_cast<int64_t>(blockIdx.y) * block_sum_stride;
    float* block_sum_bg = block_sum + kRelionSoftMaskBlocks;
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
    const float* images,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* block_lane_sums,
    int64_t block_lane_sum_stride)
{
    const float* image = images + static_cast<int64_t>(blockIdx.y) * image_size;
    float* block_lane_sum =
        block_lane_sums + static_cast<int64_t>(blockIdx.y) * block_lane_sum_stride;
    float* block_lane_sum_bg =
        block_lane_sum + kRelionSoftMaskBlocks * kRelionPreprocessBlockSize;
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
    const float* images,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    float* lane_sums,
    int64_t lane_sum_stride)
{
    const float* image = images + static_cast<int64_t>(blockIdx.y) * image_size;
    float* lane_sum = lane_sums + static_cast<int64_t>(blockIdx.y) * lane_sum_stride;
    float* lane_sum_bg = lane_sum + kRelionPreprocessBlockSize;
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
    const float* block_lane_sums,
    int64_t block_lane_sum_stride,
    float* lane_sums,
    int64_t lane_sum_stride)
{
    // One block per image (blockIdx.x); the per-lane serial block sum is the
    // native observer's tree and must stay a volatile serial loop.
    const float* block_lane_sum =
        block_lane_sums + static_cast<int64_t>(blockIdx.x) * block_lane_sum_stride;
    const float* block_lane_sum_bg =
        block_lane_sum + kRelionSoftMaskBlocks * kRelionPreprocessBlockSize;
    float* lane_sum = lane_sums + static_cast<int64_t>(blockIdx.x) * lane_sum_stride;
    float* lane_sum_bg = lane_sum + kRelionPreprocessBlockSize;
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
    float* images,
    int64_t image_size,
    int image_w,
    int image_h,
    int xinit,
    int yinit,
    float radius,
    float radius_p,
    float cosine_width,
    const float* background_sums)
{
    // One image per blockIdx.y.  The background value is the same IEEE
    // float32 quotient the host used to form, now taken from the device-side
    // CUB sums so the fill needs no per-image host round trip.  A
    // non-positive or non-finite weight sum writes NaN here; the launcher's
    // host check turns that case into the same error as before.
    float* image = images + static_cast<int64_t>(blockIdx.y) * image_size;
    float weight_sum = background_sums[2 * blockIdx.y];
    float weighted_bg = background_sums[2 * blockIdx.y + 1];
    float bg_value = (weight_sum > 0.0f && isfinite(weight_sum) && isfinite(weighted_bg))
        ? weighted_bg / weight_sum
        : __int_as_float(0x7fc00000);
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

__global__ void relion_softmask_count_invalid_backgrounds_kernel(
    const float* background_sums,
    int64_t batch_size,
    int32_t* invalid_count)
{
    // Single block: count images whose background weight is non-positive or
    // whose sums are non-finite.  Consumed by the deferred host check.
    int local = 0;
    for (int64_t image = threadIdx.x; image < batch_size; image += blockDim.x) {
        float weight_sum = background_sums[2 * image];
        float weighted_bg = background_sums[2 * image + 1];
        if (!(weight_sum > 0.0f) || !isfinite(weight_sum) || !isfinite(weighted_bg)) ++local;
    }
    using BlockReduce = cub::BlockReduce<int, kRelionPreprocessBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    int total = BlockReduce(reduce_storage).Sum(local);
    if (threadIdx.x == 0) invalid_count[0] = total;
}

// Scratch layout for launch_relion_preprocess_real_f32 (floats, then CUB temp bytes).
struct RelionPreprocessScratchLayout {
    int primary_count;          // per-image entries in each primary sum array
    int reduction_input_count;  // values fed to each CUB device sum
    int64_t per_image_storage;  // floats of reduction storage per image
    int64_t storage_floats;     // batch * per_image_storage
    int64_t values_floats;      // 2 * batch (weight sum, weighted background)
    size_t reduce_temp_bytes;   // CUB temporary storage (one reduction at a time)
    size_t total_bytes;         // storage + values + temp with 256-byte alignment
};

cudaError_t relion_preprocess_scratch_layout(
    int64_t batch_size,
    int reduction_mode,
    RelionPreprocessScratchLayout* layout)
{
    constexpr int kRelionSoftMaskLanePartials =
        kRelionSoftMaskBlocks * kRelionPreprocessBlockSize;
    bool deterministic_lane_reduction = reduction_mode == 1;
    bool native_atomic_reduction = reduction_mode == 2;
    layout->primary_count = deterministic_lane_reduction
        ? kRelionSoftMaskLanePartials
        : (native_atomic_reduction ? kRelionPreprocessBlockSize : kRelionSoftMaskBlocks);
    layout->reduction_input_count = deterministic_lane_reduction
        ? kRelionPreprocessBlockSize
        : layout->primary_count;
    layout->per_image_storage = 2 * static_cast<int64_t>(layout->primary_count);
    if (deterministic_lane_reduction)
        layout->per_image_storage += 2 * kRelionPreprocessBlockSize;
    layout->storage_floats = batch_size * layout->per_image_storage;
    layout->values_floats = 2 * batch_size;
    layout->reduce_temp_bytes = 0;
    cudaError_t err = cub::DeviceReduce::Sum(
        nullptr, layout->reduce_temp_bytes, static_cast<float*>(nullptr),
        static_cast<float*>(nullptr), layout->reduction_input_count, nullptr);
    if (err != cudaSuccess) return err;
    if (layout->reduce_temp_bytes == 0) layout->reduce_temp_bytes = 1;
    auto align256 = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    layout->total_bytes =
        align256(static_cast<size_t>(layout->storage_floats) * sizeof(float)) +
        align256(static_cast<size_t>(layout->values_floats) * sizeof(float)) +
        align256(layout->reduce_temp_bytes);
    return cudaSuccess;
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
    int reduction_mode,
    bool host_check,
    int32_t* invalid_count,
    void* scratch,
    const RelionPreprocessScratchLayout& layout)
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
    if (err != cudaSuccess) return err;
    if (!apply_mask) return cudaMemsetAsync(invalid_count, 0, sizeof(int32_t), stream);

    // The soft-mask background weight is a pure function of the geometry:
    // the farthest texel from the centre is (0, 0).  RELION's mask has no
    // exterior when even that texel lies inside ``radius``; fail closed here
    // instead of reading the reduced weight back on the host per image.
    {
        int xinit = image_w / 2;
        int yinit = image_h / 2;
        float r_max = sqrtf(static_cast<float>(xinit * xinit + yinit * yinit));
        if (!(r_max > radius)) return cudaErrorInvalidValue;
    }

    bool deterministic_lane_reduction = reduction_mode == 1;
    bool native_atomic_reduction = reduction_mode == 2;
    if (scratch == nullptr) return cudaErrorInvalidValue;
    auto align256 = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    char* scratch_bytes = static_cast<char*>(scratch);
    float* reduction_storage = reinterpret_cast<float*>(scratch_bytes);
    scratch_bytes += align256(static_cast<size_t>(layout.storage_floats) * sizeof(float));
    float* reduce_values = reinterpret_cast<float*>(scratch_bytes);
    scratch_bytes += align256(static_cast<size_t>(layout.values_floats) * sizeof(float));
    void* reduce_temp = scratch_bytes;
    size_t reduce_temp_bytes = layout.reduce_temp_bytes;
    const int primary_count = layout.primary_count;
    const int reduction_input_count = layout.reduction_input_count;
    const int64_t per_image_storage = layout.per_image_storage;

    float radius_p = radius + cosine_width;
    dim3 mask_grid(kRelionSoftMaskBlocks, static_cast<unsigned int>(batch_size));
    if (deterministic_lane_reduction) {
        // Lane partials live after the two lane-sum arrays inside each
        // image's storage: [lane_sum | lane_sum_bg | partials | partials_bg].
        relion_softmask_background_lane_partials_f32_kernel<<<
            mask_grid, kRelionPreprocessBlockSize, 0, stream>>>(
            masked, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
            radius, radius_p, cosine_width,
            reduction_storage + 2 * kRelionPreprocessBlockSize, per_image_storage);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        relion_softmask_finalize_lane_partials_f32_kernel<<<
            static_cast<unsigned int>(batch_size), kRelionPreprocessBlockSize, 0, stream>>>(
            reduction_storage + 2 * kRelionPreprocessBlockSize, per_image_storage,
            reduction_storage, per_image_storage);
    } else if (native_atomic_reduction) {
        err = cudaMemsetAsync(
            reduction_storage,
            0,
            static_cast<size_t>(layout.storage_floats) * sizeof(float),
            stream);
        if (err != cudaSuccess) return err;
        relion_softmask_background_native_atomic_f32_kernel<<<
            mask_grid, kRelionPreprocessBlockSize, 0, stream>>>(
            masked, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
            radius, radius_p, cosine_width, reduction_storage, per_image_storage);
    } else {
        relion_softmask_background_f32_kernel<<<
            mask_grid, kRelionPreprocessBlockSize, 0, stream>>>(
            masked, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
            radius, radius_p, cosine_width, reduction_storage, per_image_storage);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // The final device-wide CUB sum keeps the accepted per-image tree: one
    // DeviceReduce::Sum over the same input count per image, stream-ordered.
    for (int64_t image = 0; image < batch_size; ++image) {
        float* sum_input = reduction_storage + image * per_image_storage;
        float* sum_input_bg = sum_input + (deterministic_lane_reduction
            ? kRelionPreprocessBlockSize
            : primary_count);
        err = cub::DeviceReduce::Sum(
            reduce_temp, reduce_temp_bytes, sum_input, reduce_values + 2 * image,
            reduction_input_count, stream);
        if (err != cudaSuccess) return err;
        err = cub::DeviceReduce::Sum(
            reduce_temp, reduce_temp_bytes, sum_input_bg, reduce_values + 2 * image + 1,
            reduction_input_count, stream);
        if (err != cudaSuccess) return err;
    }
    relion_cosine_fill_f32_kernel<<<mask_grid, kRelionPreprocessBlockSize, 0, stream>>>(
        masked, pixels_per_image, image_w, image_h, image_w / 2, image_h / 2,
        radius, radius_p, cosine_width, reduce_values);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // The invalid-image count is always produced on the device so the
    // deferred check (host_check == false) can read it later without a
    // per-call synchronization.
    relion_softmask_count_invalid_backgrounds_kernel<<<1, kRelionPreprocessBlockSize, 0, stream>>>(
        reduce_values, batch_size, invalid_count);
    err = cudaGetLastError();
    if (err != cudaSuccess || !host_check) return err;

    // Default failure semantics are those of the per-image launcher: a
    // non-positive or non-finite background weight, or a non-finite weighted
    // background, aborts the call.  One read-back per call replaces one per
    // image.  Deferring it (RECOVAR_RELION_PREPROCESS_DEFERRED_CHECK) moves
    // the same check to the caller's drain point; NaN fills the affected
    // exterior in the meantime.
    std::vector<float> host_sums(static_cast<size_t>(2 * batch_size));
    err = cudaMemcpyAsync(
        host_sums.data(), reduce_values, host_sums.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;
    for (int64_t image = 0; image < batch_size; ++image) {
        float weight_sum = host_sums[2 * image];
        float weighted_bg = host_sums[2 * image + 1];
        if (!(weight_sum > 0.0f) || !std::isfinite(weight_sum) || !std::isfinite(weighted_bg))
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
