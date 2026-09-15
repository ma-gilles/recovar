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
