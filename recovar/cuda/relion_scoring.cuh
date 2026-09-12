__device__ __forceinline__ float relion_fine_diff2_update_f32(
    float2 reference,
    float2 shifted_image,
    float weight,
    float lane_sum)
{
    const float diff_real = __fsub_rn(reference.x, shifted_image.x);
    const float diff_imag = __fsub_rn(reference.y, shifted_image.y);
    const float imag_square = __fmul_rn(diff_imag, diff_imag);
    const float square_sum = __fmaf_rn(diff_real, diff_real, imag_square);
    const float half_square_sum = __fmul_rn(square_sum, 0.5f);
    return __fmaf_rn(half_square_sum, weight, lane_sum);
}

__device__ __forceinline__ double relion_fine_diff2_update_f64(
    double2 reference,
    double2 shifted_image,
    double weight,
    double lane_sum)
{
    const double diff_real = __dsub_rn(reference.x, shifted_image.x);
    const double diff_imag = __dsub_rn(reference.y, shifted_image.y);
    const double imag_square = __dmul_rn(diff_imag, diff_imag);
    const double square_sum = __fma_rn(diff_real, diff_real, imag_square);
    const double half_square_sum = __dmul_rn(square_sum, 0.5);
    return __fma_rn(half_square_sum, weight, lane_sum);
}

__device__ __forceinline__ float relion_fine_diff2_update_prehalf_f32(
    float2 reference,
    float2 shifted_image,
    float prehalved_weight,
    float lane_sum)
{
    const float diff_real = __fsub_rn(reference.x, shifted_image.x);
    const float diff_imag = __fsub_rn(reference.y, shifted_image.y);
    const float imag_square = __fmul_rn(diff_imag, diff_imag);
    const float square_sum = __fmaf_rn(diff_real, diff_real, imag_square);
    return __fmaf_rn(square_sum, prehalved_weight, lane_sum);
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_normalized_cc_pairs_f32_kernel(
    const float2* shifted_image,
    const float* score_weight,
    const float2* reference,
    const float* half_weights,
    const int32_t* packed_to_compact,
    float* output,
    int64_t candidate_count,
    int64_t compact_pixel_count,
    int64_t packed_pixel_count)
{
    const int64_t candidate = static_cast<int64_t>(blockIdx.x);
    if (candidate >= candidate_count) return;
    const int tid = threadIdx.x;
    float numerator = 0.0f;
    float norm = 0.0f;
    for (int64_t packed_pixel = tid;
         packed_pixel < packed_pixel_count;
         packed_pixel += kRelionCoarseDiff2BlockSize) {
        const int32_t compact_pixel = packed_to_compact[packed_pixel];
        if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
        const int64_t operand_index =
            candidate * compact_pixel_count + compact_pixel;
        const float2 image_value = shifted_image[operand_index];
        const float2 reference_value = reference[operand_index];
        const float hermitian_weight = half_weights[compact_pixel];
        numerator +=
            (reference_value.x * image_value.x +
             reference_value.y * image_value.y) *
            hermitian_weight;
        norm +=
            (reference_value.x * reference_value.x +
             reference_value.y * reference_value.y) *
            score_weight[operand_index] * hermitian_weight;
    }

    __shared__ float numerator_lanes[kRelionCoarseDiff2BlockSize];
    __shared__ float norm_lanes[kRelionCoarseDiff2BlockSize];
    numerator_lanes[tid] = numerator;
    norm_lanes[tid] = norm;
    __syncthreads();
    for (int width = kRelionCoarseDiff2BlockSize / 2; width > 0; width /= 2) {
        if (tid < width) {
            numerator_lanes[tid] += numerator_lanes[tid + width];
            norm_lanes[tid] += norm_lanes[tid + width];
        }
        __syncthreads();
    }

    if (tid == 0) output[candidate] = 0.0f;
    __syncthreads();
    const float contribution = numerator_lanes[0] /
        (static_cast<float>(kRelionCoarseDiff2BlockSize) *
         sqrtf(fmaxf(norm_lanes[0], 1e-30f)));
    atomicAdd(&output[candidate], contribution);
}

cudaError_t launch_relion_coarse_normalized_cc_pairs_f32(
    cudaStream_t stream,
    const float2* shifted_image,
    const float* score_weight,
    const float2* reference,
    const float* half_weights,
    const int32_t* packed_to_compact,
    float* output,
    int64_t candidate_count,
    int64_t compact_pixel_count,
    int64_t packed_pixel_count)
{
    if (candidate_count == 0) return cudaSuccess;
    relion_coarse_normalized_cc_pairs_f32_kernel<<<
        static_cast<unsigned int>(candidate_count),
        kRelionCoarseDiff2BlockSize,
        0,
        stream>>>(
            shifted_image,
            score_weight,
            reference,
            half_weights,
            packed_to_compact,
            output,
            candidate_count,
            compact_pixel_count,
            packed_pixel_count);
    return cudaGetLastError();
}

__device__ __forceinline__ void relion_coarse_diff2_rotation_block_f32(
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch,
    int64_t rotation_start,
    int64_t output_rotation_start,
    int64_t rotation_count,
    int64_t output_rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int translation = threadIdx.x % translation_count;
    const int lane = threadIdx.x / translation_count;
    const int active_lanes = kRelionCoarseDiff2BlockSize / translation_count;
    float lane_sums[kRelionCoarseEulersPerBlock] = {0.0f};

    if (lane < active_lanes) {
        constexpr int pixels_per_chunk =
            kRelionCoarseDiff2BlockSize / kRelionCoarsePrefetchFraction;
        for (int64_t chunk_start = 0;
             chunk_start < full_pixel_count;
             chunk_start += pixels_per_chunk) {
            for (int pixel_in_chunk = lane;
                 pixel_in_chunk < pixels_per_chunk;
                 pixel_in_chunk += active_lanes) {
                const int64_t full_pixel = chunk_start + pixel_in_chunk;
                if (full_pixel >= full_pixel_count) break;
                const int32_t compact_pixel = full_to_compact[full_pixel];
                if (compact_pixel < 0 || compact_pixel >= compact_pixel_count)
                    continue;
                const int64_t image_index =
                    (batch * translation_count + translation) *
                        compact_pixel_count +
                    compact_pixel;
                const int64_t weight_index =
                    batch * compact_pixel_count + compact_pixel;
                #pragma unroll
                for (int rotation_offset = 0;
                     rotation_offset < kRelionCoarseEulersPerBlock;
                     ++rotation_offset) {
                    const int64_t rotation = rotation_start + rotation_offset;
                    if (rotation >= rotation_count) continue;
                    const int64_t reference_index =
                        rotation * compact_pixel_count + compact_pixel;
                    lane_sums[rotation_offset] = relion_fine_diff2_update_f32(
                        reference[reference_index],
                        shifted_image[image_index],
                        weight[weight_index],
                        lane_sums[rotation_offset]);
                }
            }
        }
    }

    // RELION issues one atomic add per thread, including zero-valued inactive
    // lanes. Keeping that write topology lets CUDA choose the same legal lane
    // order as the production coarse scorer.
    #pragma unroll
    for (int rotation_offset = 0;
         rotation_offset < kRelionCoarseEulersPerBlock;
         ++rotation_offset) {
        const int64_t rotation = rotation_start + rotation_offset;
        if (rotation >= rotation_count) continue;
        const int64_t output_rotation =
            output_rotation_start + rotation_offset;
        atomicAdd(
            &output[(batch * output_rotation_count + output_rotation) *
                        translation_count +
                    translation],
            lane_sums[rotation_offset]);
    }
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_rectangular_f32_kernel(
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / rotation_blocks;
    const int64_t rotation_start =
        (flat_block - batch * rotation_blocks) * kRelionCoarseEulersPerBlock;
    if (batch >= batch_size) return;

    relion_coarse_diff2_rotation_block_f32(
        reference,
        shifted_image,
        weight,
        full_to_compact,
        output,
        batch,
        rotation_start,
        rotation_start,
        rotation_count,
        rotation_count,
        translation_count,
        compact_pixel_count,
        full_pixel_count);
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_rectangular_runtime_f32_kernel(
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t full_pixel_capacity,
    const int32_t* runtime_full_pixel_count)
{
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / rotation_blocks;
    const int64_t rotation_start =
        (flat_block - batch * rotation_blocks) * kRelionCoarseEulersPerBlock;
    if (batch >= batch_size) return;
    const int64_t full_pixel_count =
        static_cast<int64_t>(runtime_full_pixel_count[0]);
    if (full_pixel_count < 0 || full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < translation_count) {
            #pragma unroll
            for (int rotation_offset = 0;
                 rotation_offset < kRelionCoarseEulersPerBlock;
                 ++rotation_offset) {
                const int64_t rotation = rotation_start + rotation_offset;
                if (rotation >= rotation_count) continue;
                atomicAdd(
                    &output[(batch * rotation_count + rotation) *
                                translation_count + threadIdx.x],
                    __int_as_float(0x7fc00000));
            }
        }
        return;
    }

    relion_coarse_diff2_rotation_block_f32(
        reference,
        shifted_image,
        weight,
        full_to_compact,
        output,
        batch,
        rotation_start,
        rotation_start,
        rotation_count,
        rotation_count,
        translation_count,
        full_pixel_count,
        full_pixel_count);
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_rotation_blocks_f32_kernel(
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const int32_t* rotation_block_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t selected_block_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / selected_block_count;
    const int64_t selected_block =
        flat_block - batch * selected_block_count;
    if (batch >= batch_size) return;

    const int32_t source_block =
        rotation_block_ids[batch * selected_block_count + selected_block];
    const int64_t available_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    if (source_block < 0 || source_block >= available_blocks) return;

    relion_coarse_diff2_rotation_block_f32(
        reference,
        shifted_image,
        weight,
        full_to_compact,
        output,
        batch,
        static_cast<int64_t>(source_block) * kRelionCoarseEulersPerBlock,
        selected_block * kRelionCoarseEulersPerBlock,
        rotation_count,
        selected_block_count * kRelionCoarseEulersPerBlock,
        translation_count,
        compact_pixel_count,
        full_pixel_count);
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_rotation_blocks_runtime_f32_kernel(
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const int32_t* rotation_block_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t selected_block_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t full_pixel_capacity,
    const int32_t* runtime_full_pixel_count)
{
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / selected_block_count;
    const int64_t selected_block =
        flat_block - batch * selected_block_count;
    if (batch >= batch_size) return;

    const int32_t source_block =
        rotation_block_ids[batch * selected_block_count + selected_block];
    const int64_t available_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    if (source_block < 0 || source_block >= available_blocks) return;
    const int64_t full_pixel_count =
        static_cast<int64_t>(runtime_full_pixel_count[0]);
    if (full_pixel_count < 0 || full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < translation_count) {
            #pragma unroll
            for (int rotation_offset = 0;
                 rotation_offset < kRelionCoarseEulersPerBlock;
                 ++rotation_offset) {
                const int64_t source_rotation =
                    static_cast<int64_t>(source_block) *
                        kRelionCoarseEulersPerBlock + rotation_offset;
                if (source_rotation >= rotation_count) continue;
                const int64_t output_rotation =
                    selected_block * kRelionCoarseEulersPerBlock +
                    rotation_offset;
                atomicAdd(
                    &output[(batch * selected_block_count *
                                 kRelionCoarseEulersPerBlock +
                             output_rotation) * translation_count +
                            threadIdx.x],
                    __int_as_float(0x7fc00000));
            }
        }
        return;
    }

    relion_coarse_diff2_rotation_block_f32(
        reference,
        shifted_image,
        weight,
        full_to_compact,
        output,
        batch,
        static_cast<int64_t>(source_block) * kRelionCoarseEulersPerBlock,
        selected_block * kRelionCoarseEulersPerBlock,
        rotation_count,
        selected_block_count * kRelionCoarseEulersPerBlock,
        translation_count,
        full_pixel_count,
        full_pixel_count);
}

template <bool FUSED_TRANSLATION>
__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_fused_translate_rectangular_f32_kernel(
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    int current_size,
    const int32_t* runtime_full_pixel_count)
{
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / rotation_blocks;
    const int64_t rotation_start =
        (flat_block - batch * rotation_blocks) * kRelionCoarseEulersPerBlock;
    if (batch >= batch_size) return;

    const int64_t full_pixel_count = runtime_full_pixel_count == nullptr
        ? full_pixel_capacity
        : static_cast<int64_t>(runtime_full_pixel_count[0]);
    if (full_pixel_count < 0 || full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < translation_count) {
            for (int offset = 0; offset < kRelionCoarseEulersPerBlock; ++offset) {
                const int64_t rotation = rotation_start + offset;
                if (rotation < rotation_count)
                    output[(batch * rotation_count + rotation) *
                               translation_count + threadIdx.x] =
                        __int_as_float(0x7fc00000);
            }
        }
        return;
    }

    constexpr int pixels_per_chunk =
        kRelionCoarseDiff2BlockSize / kRelionCoarsePrefetchFraction;
    __shared__ float2 shared_reference[
        pixels_per_chunk * kRelionCoarseEulersPerBlock];
    __shared__ float2 shared_image[kRelionCoarseDiff2BlockSize];
    __shared__ float shared_weight[kRelionCoarseDiff2BlockSize];

    const int thread = threadIdx.x;
    const int translation = thread % translation_count;
    const int lane = thread / translation_count;
    const int active_lanes = kRelionCoarseDiff2BlockSize / translation_count;
    float tx = 0.0f;
    float ty = 0.0f;
    if constexpr (FUSED_TRANSLATION) {
        tx = translation_angles[2 * translation];
        ty = translation_angles[2 * translation + 1];
    }
    const int current_half_width = current_size / 2 + 1;
    float lane_sums[kRelionCoarseEulersPerBlock] = {0.0f};

    const int64_t padded_pixel_count =
        ((full_pixel_count + kRelionCoarseDiff2BlockSize - 1) /
         kRelionCoarseDiff2BlockSize) *
        kRelionCoarseDiff2BlockSize;
    for (int64_t chunk_start = 0;
         chunk_start < padded_pixel_count;
         chunk_start += pixels_per_chunk) {
        __syncthreads();

        const int64_t reference_full_pixel =
            chunk_start + thread / kRelionCoarsePrefetchFraction;
        const int32_t reference_compact_pixel =
            reference_full_pixel < full_pixel_count
                ? full_to_compact[reference_full_pixel]
                : -1;
        for (int rotation_offset = thread % kRelionCoarsePrefetchFraction;
             rotation_offset < kRelionCoarseEulersPerBlock;
             rotation_offset += kRelionCoarsePrefetchFraction) {
            const int64_t rotation = rotation_start + rotation_offset;
            float2 value = make_float2(0.0f, 0.0f);
            if (rotation < rotation_count && reference_compact_pixel >= 0 &&
                reference_compact_pixel < compact_pixel_count) {
                value = reference[
                    rotation * compact_pixel_count + reference_compact_pixel];
            }
            shared_reference[
                (thread / kRelionCoarsePrefetchFraction) *
                    kRelionCoarseEulersPerBlock +
                rotation_offset] = value;
        }

        if (chunk_start % kRelionCoarseDiff2BlockSize == 0) {
            const int64_t image_full_pixel = chunk_start + thread;
            const int32_t image_compact_pixel =
                image_full_pixel < full_pixel_count
                    ? full_to_compact[image_full_pixel]
                    : -1;
            float2 image_value = make_float2(0.0f, 0.0f);
            float weight_value = 0.0f;
            if (image_compact_pixel >= 0 &&
                image_compact_pixel < compact_pixel_count) {
                if constexpr (FUSED_TRANSLATION)
                    image_value = image[
                        batch * compact_pixel_count + image_compact_pixel];
                weight_value = weight[
                    batch * compact_pixel_count + image_compact_pixel];
            }
            shared_image[thread] = image_value;
            shared_weight[thread] = weight_value;
        }

        __syncthreads();

        if (lane < active_lanes) {
            for (int pixel_in_chunk = lane;
                 pixel_in_chunk < pixels_per_chunk;
                 pixel_in_chunk += active_lanes) {
                const int64_t full_pixel = chunk_start + pixel_in_chunk;
                if (full_pixel >= full_pixel_count) break;
                const int32_t compact_pixel = full_to_compact[full_pixel];
                if (compact_pixel < 0 || compact_pixel >= compact_pixel_count)
                    continue;
                const int shared_pixel =
                    pixel_in_chunk + static_cast<int>(chunk_start %
                                                      kRelionCoarseDiff2BlockSize);
                float2 shifted;
                if constexpr (FUSED_TRANSLATION) {
                    const int x = static_cast<int>(full_pixel % current_half_width);
                    int y = static_cast<int>(full_pixel / current_half_width);
                    if (y > current_size / 2) y -= current_size;
                    shifted = relion_coarse_score_translate_f32(
                        shared_image[shared_pixel], x, y, tx, ty);
                } else {
                    // Preserve physical row strides: logical-prefix packing
                    // copies are unnecessary when only traversal is bounded.
                    shifted = image[(batch * translation_count + translation) *
                                        compact_pixel_count + compact_pixel];
                }
                const float pixel_weight = shared_weight[shared_pixel];
                #pragma unroll
                for (int rotation_offset = 0;
                     rotation_offset < kRelionCoarseEulersPerBlock;
                     ++rotation_offset) {
                    const int64_t rotation = rotation_start + rotation_offset;
                    if (rotation >= rotation_count) continue;
                    lane_sums[rotation_offset] = relion_fine_diff2_update_f32(
                        shared_reference[
                            pixel_in_chunk * kRelionCoarseEulersPerBlock +
                            rotation_offset],
                        shifted,
                        pixel_weight,
                        lane_sums[rotation_offset]);
                }
            }
        }
    }

    #pragma unroll
    for (int rotation_offset = 0;
         rotation_offset < kRelionCoarseEulersPerBlock;
         ++rotation_offset) {
        const int64_t rotation = rotation_start + rotation_offset;
        if (rotation >= rotation_count) continue;
        atomicAdd(
            &output[(batch * rotation_count + rotation) * translation_count +
                    translation],
            lane_sums[rotation_offset]);
    }
}

__device__ __forceinline__ float2 relion_coarse_project_texture_f32(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    int x,
    int y,
    const float* euler,
    int padding_factor,
    int max_r2_padded,
    int tex_y_init,
    int tex_z_init)
{
    // AccProjectorKernel::project3Dmodel(x, y, e0, e1, e3, e4, e6, e7).
    float xp = (euler[0] * x + euler[1] * y) * padding_factor;
    float yp = (euler[3] * x + euler[4] * y) * padding_factor;
    float zp = (euler[6] * x + euler[7] * y) * padding_factor;
    const int r2 = static_cast<int>(xp * xp + yp * yp + zp * zp);
    if (r2 > max_r2_padded) return make_float2(0.0f, 0.0f);

    float imag_sign = 1.0f;
    if (xp < 0.0f) {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag_sign = -1.0f;
    }
    const float real = tex3D<float>(
        tex_real,
        xp + 0.5f,
        yp - static_cast<float>(tex_y_init) + 0.5f,
        zp - static_cast<float>(tex_z_init) + 0.5f);
    const float imag = imag_sign * tex3D<float>(
        tex_imag,
        xp + 0.5f,
        yp - static_cast<float>(tex_y_init) + 0.5f,
        zp - static_cast<float>(tex_z_init) + 0.5f);
    return make_float2(real, imag);
}

/* Bounded normalized-CC replay for candidate pairs.  Projection and scoring
 * deliberately share one CUDA kernel so the reference samples have the same
 * texture interpolation and float32 contraction boundaries as RELION's
 * production coarse kernel. */
__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_normalized_cc_native_texture_pairs_f32_kernel(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    const float* eulers,
    const float2* unshifted_image,
    const float* translation_angles,
    const float* score_weight,
    const float* numerator_weight,
    const float* half_weights,
    const int32_t* packed_to_compact,
    float* output,
    int64_t candidate_count,
    int64_t compact_pixel_count,
    int64_t packed_pixel_count,
    int current_size,
    int padding_factor,
    int max_r2_padded,
    int tex_y_init,
    int tex_z_init)
{
    const int64_t candidate = static_cast<int64_t>(blockIdx.x);
    if (candidate >= candidate_count) return;
    const int tid = threadIdx.x;
    const int current_half_width = current_size / 2 + 1;
    const float* euler = eulers + candidate * 9;
    const float tx = translation_angles[candidate * 2];
    const float ty = translation_angles[candidate * 2 + 1];
    float numerator = 0.0f;
    float norm = 0.0f;
    for (int64_t packed_pixel = tid;
         packed_pixel < packed_pixel_count;
         packed_pixel += kRelionCoarseDiff2BlockSize) {
        const int32_t compact_pixel = packed_to_compact[packed_pixel];
        if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
        const int x = static_cast<int>(packed_pixel % current_half_width);
        int y = static_cast<int>(packed_pixel / current_half_width);
        if (y > current_size / 2) y -= current_size;
        const float2 reference_value = relion_coarse_project_texture_f32(
            tex_real,
            tex_imag,
            x,
            y,
            euler,
            padding_factor,
            max_r2_padded,
            tex_y_init,
            tex_z_init);
        const int64_t operand_index =
            candidate * compact_pixel_count + compact_pixel;
        const float2 image_value = relion_coarse_score_translate_f32(
            unshifted_image[operand_index], x, y, tx, ty);
        const float correction = score_weight[operand_index];
        const float numerator_correction = numerator_weight[operand_index];
        // RELION's packed coarse-CC kernel visits every stored pixel once;
        // it does not apply a separate half-spectrum multiplicity.  Keeping a
        // runtime multiply by an all-one array changes the compiler's operand
        // contraction at one-ULP ties even though the mathematical value is
        // unchanged.  Match the native source expression directly here.
        numerator +=
            (reference_value.x * image_value.x +
             reference_value.y * image_value.y) *
            numerator_correction;
        norm +=
            (reference_value.x * reference_value.x +
             reference_value.y * reference_value.y) *
            correction;
    }

    __shared__ float numerator_lanes[kRelionCoarseDiff2BlockSize];
    __shared__ float norm_lanes[kRelionCoarseDiff2BlockSize];
    numerator_lanes[tid] = numerator;
    norm_lanes[tid] = norm;
    __syncthreads();
    for (int width = kRelionCoarseDiff2BlockSize / 2; width > 0; width /= 2) {
        if (tid < width) {
            numerator_lanes[tid] += numerator_lanes[tid + width];
            norm_lanes[tid] += norm_lanes[tid + width];
        }
        __syncthreads();
    }

    float* candidate_output = output + candidate * 3;
    if (tid == 0) {
        candidate_output[0] = 0.0f;
        candidate_output[1] = numerator_lanes[0];
        candidate_output[2] = norm_lanes[0];
    }
    __syncthreads();
    const float contribution = numerator_lanes[0] /
        (static_cast<float>(kRelionCoarseDiff2BlockSize) *
         sqrtf(fmaxf(norm_lanes[0], 1e-30f)));
    atomicAdd(&candidate_output[0], contribution);
}

/* VDAM compatibility path for RELION's historical InitialModel coarse
 * projector.  Keep the compact six-component rotation layout and the
 * 128-orientation main segment plus one-orientation tail: marginal adaptive
 * parents depend on this exact arithmetic and launch topology. */
__global__ void relion_coarse_diff2_initialize_f32_kernel(
    const float* initial_diff2,
    float* output,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t output_count);

template <
    int EULERS_PER_BLOCK,
    bool CAPTURE_LANES = false,
    bool CANONICAL_REDUCTION = false,
    bool SINGLE_LANE_CANONICAL = false>
__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_projector_f32_kernel(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int rotation_offset,
    int rotation_count,
    int output_rotation_count,
    int batch_size,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int tex_yinit,
    int tex_zinit,
    int model_max_r2,
    float projector_scale)
#define RECOVAR_RELION_COARSE_STAGE_WEIGHT(pixel_weight)
#define RECOVAR_RELION_COARSE_DIFF2_UPDATE relion_fine_diff2_update_f32
#include "relion_coarse_diff2_projector_body.inc"
#undef RECOVAR_RELION_COARSE_DIFF2_UPDATE
#undef RECOVAR_RELION_COARSE_STAGE_WEIGHT

template <int EULERS_PER_BLOCK, bool CAPTURE_LANES = false>
__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_projector_prehalf_f32_kernel(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int rotation_offset,
    int rotation_count,
    int output_rotation_count,
    int batch_size,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int tex_yinit,
    int tex_zinit,
    int model_max_r2,
    float projector_scale)
#define CANONICAL_REDUCTION false
#define SINGLE_LANE_CANONICAL false
#define RECOVAR_RELION_COARSE_STAGE_WEIGHT(pixel_weight) \
    pixel_weight = __fmul_rn(pixel_weight, 0.5f);
#define RECOVAR_RELION_COARSE_DIFF2_UPDATE \
    relion_fine_diff2_update_prehalf_f32
#include "relion_coarse_diff2_projector_body.inc"
#undef RECOVAR_RELION_COARSE_DIFF2_UPDATE
#undef RECOVAR_RELION_COARSE_STAGE_WEIGHT
#undef SINGLE_LANE_CANONICAL
#undef CANONICAL_REDUCTION

template <
    int EULERS_PER_BLOCK,
    bool CAPTURE_LANES,
    bool CANONICAL_REDUCTION,
    bool SINGLE_LANE_CANONICAL,
    bool PREHALF_WEIGHT>
void launch_relion_coarse_diff2_projector_f32_variant(
    int blocks,
    cudaStream_t stream,
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int rotation_offset,
    int rotation_count,
    int output_rotation_count,
    int batch_size,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int tex_yinit,
    int tex_zinit,
    int model_max_r2,
    float projector_scale)
{
    static_assert(
        !PREHALF_WEIGHT || (!CANONICAL_REDUCTION && !SINGLE_LANE_CANONICAL),
        "prehalved coarse weights require native atomic reduction");
    if constexpr (PREHALF_WEIGHT) {
        relion_coarse_diff2_projector_prehalf_f32_kernel<
            EULERS_PER_BLOCK,
            CAPTURE_LANES><<<
            blocks, kRelionCoarseDiff2BlockSize, 0, stream>>>(
                tex_real,
                tex_imag,
                rotations,
                images,
                translation_angles,
                weight,
                full_to_compact,
                output,
                lane_partials,
                rotation_offset,
                rotation_count,
                output_rotation_count,
                batch_size,
                translation_count,
                compact_pixel_count,
                current_size,
                tex_yinit,
                tex_zinit,
                model_max_r2,
                projector_scale);
    } else {
        relion_coarse_diff2_projector_f32_kernel<
            EULERS_PER_BLOCK,
            CAPTURE_LANES,
            CANONICAL_REDUCTION,
            SINGLE_LANE_CANONICAL><<<
            blocks, kRelionCoarseDiff2BlockSize, 0, stream>>>(
                tex_real,
                tex_imag,
                rotations,
                images,
                translation_angles,
                weight,
                full_to_compact,
                output,
                lane_partials,
                rotation_offset,
                rotation_count,
                output_rotation_count,
                batch_size,
                translation_count,
                compact_pixel_count,
                current_size,
                tex_yinit,
                tex_zinit,
                model_max_r2,
                projector_scale);
    }
}

template <
    bool CAPTURE_LANES = false,
    bool CANONICAL_REDUCTION = false,
    bool SINGLE_LANE_CANONICAL = false,
    bool PREHALF_WEIGHT = false>
cudaError_t launch_relion_coarse_diff2_projector_f32_impl(
    cudaStream_t stream,
    const float2* projector_full,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int batch_size,
    int rotation_count,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int projector_size,
    int model_max_r,
    float projector_scale,
    int actual_batch_size,
    int worker_stream_count)
{
    const int output_count = batch_size * rotation_count * translation_count;
    constexpr int initialize_block_size = 256;
    relion_coarse_diff2_initialize_f32_kernel<<<
        (output_count + initialize_block_size - 1) / initialize_block_size,
        initialize_block_size,
        0,
        stream>>>(
            initial_diff2,
            output,
            rotation_count,
            translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const int tex_x = model_max_r + 2;
    const int tex_y = 2 * model_max_r + 3;
    const int tex_z = 2 * model_max_r + 3;
    const int tex_yinit = -(model_max_r + 1);
    const int tex_zinit = -(model_max_r + 1);
    const int score_max_r = min(model_max_r, current_size / 2);
    const int voxel_count = tex_x * tex_y * tex_z;
    float* real = nullptr;
    float* imag = nullptr;
    cudaArray_t array_real = nullptr;
    cudaArray_t array_imag = nullptr;
    cudaTextureObject_t texture_real = 0;
    cudaTextureObject_t texture_imag = 0;
    cudaStream_t worker_streams[kRelionVdamWorkerStreams] = {};
    cudaEvent_t worker_inputs_ready = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&real), voxel_count * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(reinterpret_cast<void**>(&imag), voxel_count * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    fill_relion_texture_compact_kernel<float><<<
        (voxel_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE,
        0,
        stream>>>(
            reinterpret_cast<const float*>(projector_full),
            real,
            imag,
            tex_x,
            tex_y,
            tex_z,
            tex_yinit,
            tex_zinit,
            projector_size,
            projector_size,
            projector_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    {
        cudaChannelFormatDesc desc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(tex_x, tex_y, tex_z);
        err = cudaMalloc3DArray(&array_real, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc3DArray(&array_imag, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        cudaMemcpy3DParms copy = {0};
        copy.extent = extent;
        copy.kind = cudaMemcpyDeviceToDevice;
        copy.srcPtr = make_cudaPitchedPtr(
            real, static_cast<size_t>(tex_x) * sizeof(float), tex_x, tex_y);
        copy.dstArray = array_real;
        err = cudaMemcpy3DAsync(&copy, stream);
        if (err != cudaSuccess) goto cleanup;
        copy.srcPtr = make_cudaPitchedPtr(
            imag, static_cast<size_t>(tex_x) * sizeof(float), tex_x, tex_y);
        copy.dstArray = array_imag;
        err = cudaMemcpy3DAsync(&copy, stream);
        if (err != cudaSuccess) goto cleanup;

        cudaResourceDesc resource_real = {};
        cudaResourceDesc resource_imag = {};
        cudaTextureDesc texture_desc = {};
        resource_real.resType = cudaResourceTypeArray;
        resource_real.res.array.array = array_real;
        resource_imag.resType = cudaResourceTypeArray;
        resource_imag.res.array.array = array_imag;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(
            &texture_real, &resource_real, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
        err = cudaCreateTextureObject(
            &texture_imag, &resource_imag, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

    if (worker_stream_count == 0) {
        const int main_rotation_count = (rotation_count / 128) * 128;
        if (main_rotation_count > 0) {
            const int blocks =
                batch_size * (main_rotation_count / kRelionCoarseEulersPerBlock);
            launch_relion_coarse_diff2_projector_f32_variant<
                kRelionCoarseEulersPerBlock,
                CAPTURE_LANES,
                CANONICAL_REDUCTION,
                SINGLE_LANE_CANONICAL,
                PREHALF_WEIGHT>(
                    blocks,
                    stream,
                    texture_real, texture_imag, rotations, images,
                    translation_angles, weight, full_to_compact, output,
                    lane_partials,
                    0, main_rotation_count, rotation_count, batch_size, translation_count,
                    compact_pixel_count, current_size, tex_yinit, tex_zinit,
                    score_max_r * score_max_r, projector_scale);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        const int tail_count = rotation_count - main_rotation_count;
        if (tail_count > 0) {
            launch_relion_coarse_diff2_projector_f32_variant<
                1,
                CAPTURE_LANES,
                CANONICAL_REDUCTION,
                SINGLE_LANE_CANONICAL,
                PREHALF_WEIGHT>(
                    batch_size * tail_count,
                    stream,
                    texture_real, texture_imag, rotations, images,
                    translation_angles, weight, full_to_compact, output,
                    lane_partials,
                    main_rotation_count, tail_count, rotation_count, batch_size,
                    translation_count, compact_pixel_count, current_size,
                    tex_yinit, tex_zinit, score_max_r * score_max_r,
                    projector_scale);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        err = cudaStreamSynchronize(stream);
    } else {
        // This path changes only particle scheduling.  Texture ownership,
        // projection, translation, lane arithmetic, selected lane reduction,
        // class->rotation->translation output layout all remain in the shared
        // production kernel above.  Synthetic padded image rows are initialized
        // but are never scored.
        err = initialize_relion_vdam_worker_streams(
            stream, worker_streams, &worker_inputs_ready);
        if (err != cudaSuccess) goto cleanup;
        const int main_rotation_count = (rotation_count / 128) * 128;
        const int tail_count = rotation_count - main_rotation_count;
        const int64_t output_stride =
            static_cast<int64_t>(rotation_count) * translation_count;
        const int64_t lane_stride =
            static_cast<int64_t>(rotation_count) *
            kRelionCoarseDiff2BlockSize;
        err = dispatch_relion_vdam_round_robin_workers(
            worker_streams,
            actual_batch_size,
            [&](int64_t particle, int worker) -> cudaError_t {
                const float2* particle_images =
                    images + particle * compact_pixel_count;
                const float* particle_weight =
                    weight + particle * compact_pixel_count;
                float* particle_output = output + particle * output_stride;
                float* particle_lane_partials = lane_partials == nullptr
                    ? nullptr
                    : lane_partials + particle * lane_stride;
                if (main_rotation_count > 0) {
                    launch_relion_coarse_diff2_projector_f32_variant<
                        kRelionCoarseEulersPerBlock,
                        CAPTURE_LANES,
                        CANONICAL_REDUCTION,
                        SINGLE_LANE_CANONICAL,
                        PREHALF_WEIGHT>(
                        main_rotation_count / kRelionCoarseEulersPerBlock,
                        worker_streams[worker],
                            texture_real, texture_imag, rotations,
                            particle_images, translation_angles,
                            particle_weight, full_to_compact, particle_output,
                            particle_lane_partials,
                            0, main_rotation_count, rotation_count, 1,
                            translation_count, compact_pixel_count,
                            current_size, tex_yinit, tex_zinit,
                            score_max_r * score_max_r, projector_scale);
                    cudaError_t launch_error = cudaGetLastError();
                    if (launch_error != cudaSuccess) return launch_error;
                }
                if (tail_count > 0) {
                    launch_relion_coarse_diff2_projector_f32_variant<
                        1,
                        CAPTURE_LANES,
                        CANONICAL_REDUCTION,
                        SINGLE_LANE_CANONICAL,
                        PREHALF_WEIGHT>(
                        tail_count,
                        worker_streams[worker],
                            texture_real, texture_imag, rotations,
                            particle_images, translation_angles,
                            particle_weight, full_to_compact, particle_output,
                            particle_lane_partials,
                            main_rotation_count, tail_count, rotation_count, 1,
                            translation_count, compact_pixel_count,
                            current_size, tex_yinit, tex_zinit,
                            score_max_r * score_max_r, projector_scale);
                    const cudaError_t launch_error = cudaGetLastError();
                    if (launch_error != cudaSuccess) return launch_error;
                }
                return cudaSuccess;
            });
    }

cleanup:
    destroy_relion_vdam_worker_streams(
        worker_streams, worker_inputs_ready);
    if (texture_real) cudaDestroyTextureObject(texture_real);
    if (texture_imag) cudaDestroyTextureObject(texture_imag);
    if (array_real) cudaFreeArray(array_real);
    if (array_imag) cudaFreeArray(array_imag);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

template <
    bool CAPTURE_LANES = false,
    bool CANONICAL_REDUCTION = false,
    bool SINGLE_LANE_CANONICAL = false>
cudaError_t launch_relion_coarse_diff2_projector_f32(
    cudaStream_t stream,
    const float2* projector_full,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int batch_size,
    int rotation_count,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int projector_size,
    int model_max_r,
    float projector_scale,
    int actual_batch_size,
    int worker_stream_count)
{
    return launch_relion_coarse_diff2_projector_f32_impl<
        CAPTURE_LANES,
        CANONICAL_REDUCTION,
        SINGLE_LANE_CANONICAL,
        false>(
            stream,
            projector_full,
            rotations,
            images,
            translation_angles,
            weight,
            initial_diff2,
            full_to_compact,
            output,
            lane_partials,
            batch_size,
            rotation_count,
            translation_count,
            compact_pixel_count,
            current_size,
            projector_size,
            model_max_r,
            projector_scale,
            actual_batch_size,
            worker_stream_count);
}

template <bool CAPTURE_LANES = false>
cudaError_t launch_relion_coarse_diff2_projector_prehalf_f32(
    cudaStream_t stream,
    const float2* projector_full,
    const float* rotations,
    const float2* images,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    float* lane_partials,
    int batch_size,
    int rotation_count,
    int translation_count,
    int compact_pixel_count,
    int current_size,
    int projector_size,
    int model_max_r,
    float projector_scale,
    int actual_batch_size,
    int worker_stream_count)
{
    return launch_relion_coarse_diff2_projector_f32_impl<
        CAPTURE_LANES,
        false,
        false,
        true>(
            stream,
            projector_full,
            rotations,
            images,
            translation_angles,
            weight,
            initial_diff2,
            full_to_compact,
            output,
            lane_partials,
            batch_size,
            rotation_count,
            translation_count,
            compact_pixel_count,
            current_size,
            projector_size,
            model_max_r,
            projector_scale,
            actual_batch_size,
            worker_stream_count);
}

/* Diagnostic reproduction of RELION's complete REF3D/DATA2D coarse kernel.
 * Unlike relion_coarse_diff2_fused_translate_rectangular_f32_kernel, this
 * kernel performs the texture projection in the same thread that stages the
 * reference for scoring.  The shared arrays and loop topology mirror
 * cuda_kernel_diff2_coarse<true, false, 128, 16, 4> in the pinned RELION
 * source. */
__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_native_texture_rectangular_f32_kernel(
    cudaTextureObject_t tex_real,
    cudaTextureObject_t tex_imag,
    const float* eulers,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    int padding_factor,
    int max_r2_padded,
    int tex_y_init,
    int tex_z_init)
{
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / rotation_blocks;
    const int64_t rotation_start =
        (flat_block - batch * rotation_blocks) * kRelionCoarseEulersPerBlock;
    if (batch >= batch_size) return;

    constexpr int pixels_per_chunk =
        kRelionCoarseDiff2BlockSize / kRelionCoarsePrefetchFraction;
    __shared__ float shared_eulers[kRelionCoarseEulersPerBlock * 9];
    __shared__ float2 shared_reference[
        pixels_per_chunk * kRelionCoarseEulersPerBlock];
    __shared__ float2 shared_image[kRelionCoarseDiff2BlockSize];
    __shared__ float shared_corr[kRelionCoarseDiff2BlockSize];

    const int thread = threadIdx.x;
    for (int index = thread;
         index < kRelionCoarseEulersPerBlock * 9;
         index += kRelionCoarseDiff2BlockSize) {
        const int64_t rotation = rotation_start + index / 9;
        shared_eulers[index] = rotation < rotation_count
            ? eulers[rotation * 9 + index % 9]
            : 0.0f;
    }

    const int translation = thread % translation_count;
    const int lane = thread / translation_count;
    const int active_lanes = kRelionCoarseDiff2BlockSize / translation_count;
    const float tx = translation_angles[2 * translation];
    const float ty = translation_angles[2 * translation + 1];
    const int current_half_width = current_size / 2 + 1;
    float lane_sums[kRelionCoarseEulersPerBlock] = {0.0f};

    const int64_t padded_pixel_count =
        ((full_pixel_count + kRelionCoarseDiff2BlockSize - 1) /
         kRelionCoarseDiff2BlockSize) *
        kRelionCoarseDiff2BlockSize;
    for (int64_t chunk_start = 0;
         chunk_start < padded_pixel_count;
         chunk_start += pixels_per_chunk) {
        __syncthreads();

        const int64_t reference_full_pixel =
            chunk_start + thread / kRelionCoarsePrefetchFraction;
        const int x = static_cast<int>(reference_full_pixel % current_half_width);
        int y = static_cast<int>(reference_full_pixel / current_half_width);
        if (y > current_size / 2) y -= current_size;
        for (int rotation_offset = thread % kRelionCoarsePrefetchFraction;
             rotation_offset < kRelionCoarseEulersPerBlock;
             rotation_offset += kRelionCoarsePrefetchFraction) {
            const int64_t rotation = rotation_start + rotation_offset;
            float2 value = make_float2(0.0f, 0.0f);
            if (reference_full_pixel < full_pixel_count &&
                rotation < rotation_count) {
                value = relion_coarse_project_texture_f32(
                    tex_real,
                    tex_imag,
                    x,
                    y,
                    &shared_eulers[rotation_offset * 9],
                    padding_factor,
                    max_r2_padded,
                    tex_y_init,
                    tex_z_init);
            }
            shared_reference[
                (thread / kRelionCoarsePrefetchFraction) *
                    kRelionCoarseEulersPerBlock +
                rotation_offset] = value;
        }

        if (chunk_start % kRelionCoarseDiff2BlockSize == 0) {
            const int64_t image_full_pixel = chunk_start + thread;
            const int32_t compact_pixel = image_full_pixel < full_pixel_count
                ? full_to_compact[image_full_pixel]
                : -1;
            float2 image_value = make_float2(0.0f, 0.0f);
            float corr_value = 0.0f;
            if (compact_pixel >= 0 && compact_pixel < compact_pixel_count) {
                image_value = image[
                    batch * compact_pixel_count + compact_pixel];
                corr_value = weight[
                    batch * compact_pixel_count + compact_pixel];
            }
            shared_image[thread] = image_value;
            shared_corr[thread] = corr_value;
        }

        __syncthreads();

        if (lane < active_lanes) {
            for (int pixel_in_chunk = lane;
                 pixel_in_chunk < pixels_per_chunk;
                 pixel_in_chunk += active_lanes) {
                const int64_t full_pixel = chunk_start + pixel_in_chunk;
                if (full_pixel >= full_pixel_count) break;
                const int32_t compact_pixel = full_to_compact[full_pixel];
                if (compact_pixel < 0 || compact_pixel >= compact_pixel_count)
                    continue;
                const int score_x = static_cast<int>(
                    full_pixel % current_half_width);
                int score_y = static_cast<int>(
                    full_pixel / current_half_width);
                if (score_y > current_size / 2) score_y -= current_size;
                const int shared_pixel =
                    pixel_in_chunk + static_cast<int>(
                        chunk_start % kRelionCoarseDiff2BlockSize);
                const float2 shifted = relion_coarse_score_translate_f32(
                    shared_image[shared_pixel], score_x, score_y, tx, ty);
                const float pixel_weight = shared_corr[shared_pixel];
                #pragma unroll
                for (int rotation_offset = 0;
                     rotation_offset < kRelionCoarseEulersPerBlock;
                     ++rotation_offset) {
                    const int64_t rotation = rotation_start + rotation_offset;
                    if (rotation >= rotation_count) continue;
                    lane_sums[rotation_offset] = relion_fine_diff2_update_f32(
                        shared_reference[
                            pixel_in_chunk * kRelionCoarseEulersPerBlock +
                            rotation_offset],
                        shifted,
                        pixel_weight,
                        lane_sums[rotation_offset]);
                }
            }
        }
    }

    #pragma unroll
    for (int rotation_offset = 0;
         rotation_offset < kRelionCoarseEulersPerBlock;
         ++rotation_offset) {
        const int64_t rotation = rotation_start + rotation_offset;
        if (rotation >= rotation_count) continue;
        atomicAdd(
            &output[(batch * rotation_count + rotation) * translation_count +
                    translation],
            lane_sums[rotation_offset]);
    }
}

__global__ void relion_coarse_diff2_initialize_f32_kernel(
    const float* initial_diff2,
    float* output,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t output_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) return;
    const int64_t hypotheses_per_batch = rotation_count * translation_count;
    output[output_index] = initial_diff2[output_index / hypotheses_per_batch];
}

__global__ void relion_coarse_diff2_rotation_blocks_initialize_f32_kernel(
    const float* initial_diff2,
    const int32_t* rotation_block_ids,
    float* output,
    int64_t selected_block_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t output_count)
{
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) return;

    const int64_t output_rotation = output_index / translation_count;
    const int64_t output_rotations_per_batch =
        selected_block_count * kRelionCoarseEulersPerBlock;
    const int64_t batch = output_rotation / output_rotations_per_batch;
    const int64_t batch_output_rotation =
        output_rotation - batch * output_rotations_per_batch;
    const int64_t selected_block =
        batch_output_rotation / kRelionCoarseEulersPerBlock;
    const int64_t rotation_offset =
        batch_output_rotation -
        selected_block * kRelionCoarseEulersPerBlock;
    const int32_t source_block =
        rotation_block_ids[batch * selected_block_count + selected_block];
    const int64_t available_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t source_rotation =
        static_cast<int64_t>(source_block) * kRelionCoarseEulersPerBlock +
        rotation_offset;
    if (source_block == -1 ||
        (source_block >= 0 && source_block < available_blocks &&
         source_rotation >= rotation_count)) {
        output[output_index] = __int_as_float(0x7f800000);
    } else if (source_block < 0 || source_block >= available_blocks) {
        output[output_index] = __int_as_float(0x7fc00000);
    } else {
        output[output_index] = initial_diff2[batch];
    }
}

cudaError_t launch_relion_coarse_diff2_rectangular_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    const int32_t* runtime_full_pixel_count)
{
    const int64_t output_count =
        batch_size * rotation_count * translation_count;
    if (output_count == 0) return cudaSuccess;
    constexpr int initialize_block_size = 256;
    const int64_t initialize_blocks =
        (output_count + initialize_block_size - 1) / initialize_block_size;
    relion_coarse_diff2_initialize_f32_kernel<<<
        static_cast<unsigned int>(initialize_blocks),
        initialize_block_size,
        0,
        stream>>>(
            initial_diff2,
            output,
            rotation_count,
            translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t block_count = batch_size * rotation_blocks;
    if (runtime_full_pixel_count == nullptr) {
        relion_coarse_diff2_rectangular_f32_kernel<<<
            static_cast<unsigned int>(block_count),
            kRelionCoarseDiff2BlockSize,
            0,
            stream>>>(
                reference,
                shifted_image,
                weight,
                full_to_compact,
                output,
                batch_size,
                rotation_count,
                translation_count,
                compact_pixel_count,
                full_pixel_capacity);
    } else {
        relion_coarse_diff2_rectangular_runtime_f32_kernel<<<
            static_cast<unsigned int>(block_count),
            kRelionCoarseDiff2BlockSize,
            0,
            stream>>>(
                reference,
                shifted_image,
                weight,
                full_to_compact,
                output,
                batch_size,
                rotation_count,
                translation_count,
                full_pixel_capacity,
                runtime_full_pixel_count);
    }
    return cudaGetLastError();
}

cudaError_t launch_relion_coarse_diff2_rotation_blocks_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* shifted_image,
    const float* weight,
    const float* initial_diff2,
    const int32_t* rotation_block_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t selected_block_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    const int32_t* runtime_full_pixel_count)
{
    const int64_t output_count =
        batch_size * selected_block_count * kRelionCoarseEulersPerBlock *
        translation_count;
    if (output_count == 0) return cudaSuccess;
    constexpr int initialize_block_size = 256;
    const int64_t initialize_blocks =
        (output_count + initialize_block_size - 1) / initialize_block_size;
    relion_coarse_diff2_rotation_blocks_initialize_f32_kernel<<<
        static_cast<unsigned int>(initialize_blocks),
        initialize_block_size,
        0,
        stream>>>(
            initial_diff2,
            rotation_block_ids,
            output,
            selected_block_count,
            rotation_count,
            translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const int64_t block_count = batch_size * selected_block_count;
    if (runtime_full_pixel_count == nullptr) {
        relion_coarse_diff2_rotation_blocks_f32_kernel<<<
            static_cast<unsigned int>(block_count),
            kRelionCoarseDiff2BlockSize,
            0,
            stream>>>(
                reference,
                shifted_image,
                weight,
                rotation_block_ids,
                full_to_compact,
                output,
                batch_size,
                selected_block_count,
                rotation_count,
                translation_count,
                compact_pixel_count,
                full_pixel_capacity);
    } else {
        relion_coarse_diff2_rotation_blocks_runtime_f32_kernel<<<
            static_cast<unsigned int>(block_count),
            kRelionCoarseDiff2BlockSize,
            0,
            stream>>>(
                reference,
                shifted_image,
                weight,
                rotation_block_ids,
                full_to_compact,
                output,
                batch_size,
                selected_block_count,
                rotation_count,
                translation_count,
                full_pixel_capacity,
                runtime_full_pixel_count);
    }
    return cudaGetLastError();
}

template <bool FUSED_TRANSLATION = true>
cudaError_t launch_relion_coarse_diff2_fused_translate_rectangular_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    const int32_t* runtime_full_pixel_count = nullptr)
{
    const int64_t output_count =
        batch_size * rotation_count * translation_count;
    if (output_count == 0) return cudaSuccess;
    constexpr int initialize_block_size = 256;
    const int64_t initialize_blocks =
        (output_count + initialize_block_size - 1) / initialize_block_size;
    relion_coarse_diff2_initialize_f32_kernel<<<
        static_cast<unsigned int>(initialize_blocks),
        initialize_block_size,
        0,
        stream>>>(
            initial_diff2,
            output,
            rotation_count,
            translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) /
        kRelionCoarseEulersPerBlock;
    const int64_t block_count = batch_size * rotation_blocks;
    relion_coarse_diff2_fused_translate_rectangular_f32_kernel<FUSED_TRANSLATION><<<
        static_cast<unsigned int>(block_count),
        kRelionCoarseDiff2BlockSize,
        0,
        stream>>>(
            reference,
            image,
            translation_angles,
            weight,
            full_to_compact,
            output,
            batch_size,
            rotation_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count,
            current_size,
            runtime_full_pixel_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_coarse_diff2_native_texture_rectangular_f32(
    cudaStream_t stream,
    const float2* projector_full,
    const float* eulers,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    int64_t projector_size,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    int padding_factor,
    int projector_max_r)
{
    const int64_t output_count =
        batch_size * rotation_count * translation_count;
    if (output_count == 0) return cudaSuccess;

    constexpr int initialize_block_size = 256;
    const int64_t initialize_blocks =
        (output_count + initialize_block_size - 1) / initialize_block_size;
    relion_coarse_diff2_initialize_f32_kernel<<<
        static_cast<unsigned int>(initialize_blocks),
        initialize_block_size,
        0,
        stream>>>(
            initial_diff2,
            output,
            rotation_count,
            translation_count,
            output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const int padded_max_r = static_cast<int>(floorf(
        static_cast<float>(projector_max_r * padding_factor) + 0.5f));
    const int tex_x = padded_max_r + 2;
    const int tex_y = 2 * padded_max_r + 3;
    const int tex_z = 2 * padded_max_r + 3;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t texture_voxels =
        static_cast<int64_t>(tex_x) * tex_y * tex_z;
    float* real = nullptr;
    float* imag = nullptr;
    cudaArray_t array_real = nullptr;
    cudaArray_t array_imag = nullptr;
    cudaTextureObject_t texture_real = 0;
    cudaTextureObject_t texture_imag = 0;

    err = cudaMalloc(
        reinterpret_cast<void**>(&real),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(
        reinterpret_cast<void**>(&imag),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 block(BLOCK_SIZE);
        dim3 grid(static_cast<unsigned int>(
            (texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE));
        fill_relion_texture_compact_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(projector_full),
            real,
            imag,
            tex_x,
            tex_y,
            tex_z,
            tex_y_init,
            tex_z_init,
            static_cast<int>(projector_size),
            static_cast<int>(projector_size),
            static_cast<int>(projector_size));
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc(
            32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y),
            static_cast<size_t>(tex_z));
        err = cudaMalloc3DArray(&array_real, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc3DArray(&array_imag, &desc, extent);
        if (err != cudaSuccess) goto cleanup;

        cudaMemcpy3DParms copy_params = {0};
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyDeviceToDevice;
        copy_params.srcPtr = make_cudaPitchedPtr(
            real,
            static_cast<size_t>(tex_x) * sizeof(float),
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y));
        copy_params.dstArray = array_real;
        err = cudaMemcpy3DAsync(&copy_params, stream);
        if (err != cudaSuccess) goto cleanup;
        copy_params.srcPtr = make_cudaPitchedPtr(
            imag,
            static_cast<size_t>(tex_x) * sizeof(float),
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y));
        copy_params.dstArray = array_imag;
        err = cudaMemcpy3DAsync(&copy_params, stream);
        if (err != cudaSuccess) goto cleanup;

        cudaResourceDesc resource_real;
        cudaResourceDesc resource_imag;
        cudaTextureDesc texture_desc;
        memset(&resource_real, 0, sizeof(resource_real));
        memset(&resource_imag, 0, sizeof(resource_imag));
        memset(&texture_desc, 0, sizeof(texture_desc));
        resource_real.resType = cudaResourceTypeArray;
        resource_real.res.array.array = array_real;
        resource_imag.resType = cudaResourceTypeArray;
        resource_imag.res.array.array = array_imag;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(
            &texture_real, &resource_real, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
        err = cudaCreateTextureObject(
            &texture_imag, &resource_imag, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

    {
        const int64_t rotation_blocks =
            (rotation_count + kRelionCoarseEulersPerBlock - 1) /
            kRelionCoarseEulersPerBlock;
        const int64_t hypotheses_per_batch =
            rotation_count * translation_count;
        const int max_r2_padded = padded_max_r * padded_max_r;
        // RELION launches one complete orientation grid for each SPA particle.
        // Keep that launch scope: combining particles in one grid changes the
        // scheduling order of the four float32 atomic contributions and can
        // move hypotheses across the adaptive-significance cutoff.
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            relion_coarse_diff2_native_texture_rectangular_f32_kernel<<<
                static_cast<unsigned int>(rotation_blocks),
                kRelionCoarseDiff2BlockSize,
                0,
                stream>>>(
                    texture_real,
                    texture_imag,
                    eulers,
                    image + batch * compact_pixel_count,
                    translation_angles,
                    weight + batch * compact_pixel_count,
                    full_to_compact,
                    output + batch * hypotheses_per_batch,
                    1,
                    rotation_count,
                    translation_count,
                    compact_pixel_count,
                    full_pixel_count,
                    current_size,
                    padding_factor,
                    max_r2_padded,
                    tex_y_init,
                    tex_z_init);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        err = cudaStreamSynchronize(stream);
    }

cleanup:
    if (texture_real) cudaDestroyTextureObject(texture_real);
    if (texture_imag) cudaDestroyTextureObject(texture_imag);
    if (array_real) cudaFreeArray(array_real);
    if (array_imag) cudaFreeArray(array_imag);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

cudaError_t launch_relion_coarse_normalized_cc_native_texture_pairs_f32(
    cudaStream_t stream,
    const float2* projector_full,
    const float* eulers,
    const float2* unshifted_image,
    const float* translation_angles,
    const float* score_weight,
    const float* numerator_weight,
    const float* half_weights,
    const int32_t* packed_to_compact,
    float* output,
    int64_t projector_size,
    int64_t candidate_count,
    int64_t compact_pixel_count,
    int64_t packed_pixel_count,
    int current_size,
    int padding_factor,
    int projector_max_r)
{
    if (candidate_count == 0) return cudaSuccess;

    const int padded_max_r = static_cast<int>(floorf(
        static_cast<float>(projector_max_r * padding_factor) + 0.5f));
    const int tex_x = padded_max_r + 2;
    const int tex_y = 2 * padded_max_r + 3;
    const int tex_z = 2 * padded_max_r + 3;
    const int tex_y_init = -(padded_max_r + 1);
    const int tex_z_init = -(padded_max_r + 1);
    const int64_t texture_voxels =
        static_cast<int64_t>(tex_x) * tex_y * tex_z;
    float* real = nullptr;
    float* imag = nullptr;
    cudaArray_t array_real = nullptr;
    cudaArray_t array_imag = nullptr;
    cudaTextureObject_t texture_real = 0;
    cudaTextureObject_t texture_imag = 0;
    cudaError_t err = cudaMalloc(
        reinterpret_cast<void**>(&real),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(
        reinterpret_cast<void**>(&imag),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    {
        dim3 block(BLOCK_SIZE);
        dim3 grid(static_cast<unsigned int>(
            (texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE));
        fill_relion_texture_compact_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(projector_full),
            real,
            imag,
            tex_x,
            tex_y,
            tex_z,
            tex_y_init,
            tex_z_init,
            static_cast<int>(projector_size),
            static_cast<int>(projector_size),
            static_cast<int>(projector_size));
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc(
            32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y),
            static_cast<size_t>(tex_z));
        err = cudaMalloc3DArray(&array_real, &desc, extent);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc3DArray(&array_imag, &desc, extent);
        if (err != cudaSuccess) goto cleanup;

        cudaMemcpy3DParms copy_params = {0};
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyDeviceToDevice;
        copy_params.srcPtr = make_cudaPitchedPtr(
            real,
            static_cast<size_t>(tex_x) * sizeof(float),
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y));
        copy_params.dstArray = array_real;
        err = cudaMemcpy3DAsync(&copy_params, stream);
        if (err != cudaSuccess) goto cleanup;
        copy_params.srcPtr = make_cudaPitchedPtr(
            imag,
            static_cast<size_t>(tex_x) * sizeof(float),
            static_cast<size_t>(tex_x),
            static_cast<size_t>(tex_y));
        copy_params.dstArray = array_imag;
        err = cudaMemcpy3DAsync(&copy_params, stream);
        if (err != cudaSuccess) goto cleanup;

        cudaResourceDesc resource_real;
        cudaResourceDesc resource_imag;
        cudaTextureDesc texture_desc;
        memset(&resource_real, 0, sizeof(resource_real));
        memset(&resource_imag, 0, sizeof(resource_imag));
        memset(&texture_desc, 0, sizeof(texture_desc));
        resource_real.resType = cudaResourceTypeArray;
        resource_real.res.array.array = array_real;
        resource_imag.resType = cudaResourceTypeArray;
        resource_imag.res.array.array = array_imag;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.addressMode[2] = cudaAddressModeClamp;
        err = cudaCreateTextureObject(
            &texture_real, &resource_real, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
        err = cudaCreateTextureObject(
            &texture_imag, &resource_imag, &texture_desc, nullptr);
        if (err != cudaSuccess) goto cleanup;
    }

    relion_coarse_normalized_cc_native_texture_pairs_f32_kernel<<<
        static_cast<unsigned int>(candidate_count),
        kRelionCoarseDiff2BlockSize,
        0,
        stream>>>(
            texture_real,
            texture_imag,
            eulers,
            unshifted_image,
            translation_angles,
            score_weight,
            numerator_weight,
            half_weights,
            packed_to_compact,
            output,
            candidate_count,
            compact_pixel_count,
            packed_pixel_count,
            current_size,
            padding_factor,
            padded_max_r * padded_max_r,
            tex_y_init,
            tex_z_init);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

cleanup:
    if (texture_real) cudaDestroyTextureObject(texture_real);
    if (texture_imag) cudaDestroyTextureObject(texture_imag);
    if (array_real) cudaFreeArray(array_real);
    if (array_imag) cudaFreeArray(array_imag);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

__global__ __launch_bounds__(kRelionCoarseDiff2BlockSize)
void relion_coarse_diff2_rectangular_f64_kernel(
    const double2* reference, const double2* shifted_image,
    const double* weight, const int32_t* full_to_compact, double* output,
    int64_t batch_size, int64_t rotation_count, int64_t translation_count,
    int64_t compact_pixel_count, int64_t full_pixel_count)
{
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) / kRelionCoarseEulersPerBlock;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / rotation_blocks;
    const int64_t rotation_start =
        (flat_block - batch * rotation_blocks) * kRelionCoarseEulersPerBlock;
    if (batch >= batch_size) return;
    const int translation = threadIdx.x % translation_count;
    const int lane = threadIdx.x / translation_count;
    const int active_lanes = kRelionCoarseDiff2BlockSize / translation_count;
    double lane_sums[kRelionCoarseEulersPerBlock] = {0.0};
    if (lane < active_lanes) {
        constexpr int pixels_per_chunk =
            kRelionCoarseDiff2BlockSize / kRelionCoarsePrefetchFraction;
        for (int64_t chunk_start = 0; chunk_start < full_pixel_count;
             chunk_start += pixels_per_chunk) {
            for (int pixel_in_chunk = lane; pixel_in_chunk < pixels_per_chunk;
                 pixel_in_chunk += active_lanes) {
                const int64_t full_pixel = chunk_start + pixel_in_chunk;
                if (full_pixel >= full_pixel_count) break;
                const int32_t compact_pixel = full_to_compact[full_pixel];
                if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
                const int64_t image_index =
                    (batch * translation_count + translation) * compact_pixel_count + compact_pixel;
                const int64_t weight_index = batch * compact_pixel_count + compact_pixel;
                #pragma unroll
                for (int rotation_offset = 0; rotation_offset < kRelionCoarseEulersPerBlock;
                     ++rotation_offset) {
                    const int64_t rotation = rotation_start + rotation_offset;
                    if (rotation >= rotation_count) continue;
                    const int64_t reference_index = rotation * compact_pixel_count + compact_pixel;
                    lane_sums[rotation_offset] = relion_fine_diff2_update_f64(
                        reference[reference_index], shifted_image[image_index],
                        weight[weight_index], lane_sums[rotation_offset]);
                }
            }
        }
    }
    #pragma unroll
    for (int rotation_offset = 0; rotation_offset < kRelionCoarseEulersPerBlock;
         ++rotation_offset) {
        const int64_t rotation = rotation_start + rotation_offset;
        if (rotation >= rotation_count) continue;
        atomicAdd(&output[(batch * rotation_count + rotation) * translation_count + translation],
                  lane_sums[rotation_offset]);
    }
}

__global__ void relion_coarse_diff2_initialize_f64_kernel(
    const double* initial_diff2, double* output, int64_t rotation_count,
    int64_t translation_count, int64_t output_count)
{
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) return;
    output[output_index] = initial_diff2[output_index / (rotation_count * translation_count)];
}

cudaError_t launch_relion_coarse_diff2_rectangular_f64(
    cudaStream_t stream, const double2* reference, const double2* shifted_image,
    const double* weight, const double* initial_diff2, const int32_t* full_to_compact,
    double* output, int64_t batch_size, int64_t rotation_count,
    int64_t translation_count, int64_t compact_pixel_count, int64_t full_pixel_count)
{
    const int64_t output_count = batch_size * rotation_count * translation_count;
    if (output_count == 0) return cudaSuccess;
    constexpr int initialize_block_size = 256;
    relion_coarse_diff2_initialize_f64_kernel<<<
        static_cast<unsigned int>((output_count + initialize_block_size - 1) / initialize_block_size),
        initialize_block_size, 0, stream>>>(initial_diff2, output, rotation_count,
                                             translation_count, output_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const int64_t rotation_blocks =
        (rotation_count + kRelionCoarseEulersPerBlock - 1) / kRelionCoarseEulersPerBlock;
    relion_coarse_diff2_rectangular_f64_kernel<<<
        static_cast<unsigned int>(batch_size * rotation_blocks),
        kRelionCoarseDiff2BlockSize, 0, stream>>>(
            reference, shifted_image, weight, full_to_compact, output,
            batch_size, rotation_count, translation_count,
            compact_pixel_count, full_pixel_count);
    return cudaGetLastError();
}

template <typename T, typename ComplexT, bool ADD_INITIAL>
__global__ __launch_bounds__(kRelionFineDiff2BlockSize)
void relion_fine_diff2_rectangular_kernel(
    const ComplexT* reference,
    const ComplexT* shifted_image,
    const T* weight,
    const T* initial_diff2,
    const int32_t* full_to_compact,
    T* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t hypothesis = static_cast<int64_t>(blockIdx.x);
    const int64_t hypotheses_per_batch = rotation_count * translation_count;
    const int64_t total_hypotheses = batch_size * hypotheses_per_batch;
    if (hypothesis >= total_hypotheses) return;

    const int64_t batch = hypothesis / hypotheses_per_batch;
    const int64_t batch_hypothesis = hypothesis - batch * hypotheses_per_batch;
    const int64_t rotation = batch_hypothesis / translation_count;
    const int64_t translation = batch_hypothesis - rotation * translation_count;
    T lane_sum = static_cast<T>(0);
    for (int64_t full_pixel = threadIdx.x;
         full_pixel < full_pixel_count;
         full_pixel += kRelionFineDiff2BlockSize) {
        const int32_t compact_pixel = full_to_compact[full_pixel];
        if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
        const int64_t reference_index =
            (batch * rotation_count + rotation) * compact_pixel_count + compact_pixel;
        const int64_t image_index =
            (batch * translation_count + translation) * compact_pixel_count + compact_pixel;
        const int64_t weight_index = batch * compact_pixel_count + compact_pixel;
        if constexpr (std::is_same_v<T, float>)
            lane_sum = relion_fine_diff2_update_f32(
                reference[reference_index], shifted_image[image_index],
                weight[weight_index], lane_sum);
        else
            lane_sum = relion_fine_diff2_update_f64(
                reference[reference_index], shifted_image[image_index],
                weight[weight_index], lane_sum);
    }

    __shared__ T lane_sums[kRelionFineDiff2BlockSize];
    lane_sums[threadIdx.x] = lane_sum;
    __syncthreads();
    for (int width = kRelionFineDiff2BlockSize / 2; width > 0; width /= 2) {
        if (threadIdx.x < width)
            if constexpr (std::is_same_v<T, float>)
                lane_sums[threadIdx.x] = __fadd_rn(
                    lane_sums[threadIdx.x], lane_sums[threadIdx.x + width]);
            else
                lane_sums[threadIdx.x] = __dadd_rn(
                    lane_sums[threadIdx.x], lane_sums[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // VDAM includes sum_init in the float32 transaction. The diagnostic
        // double FFI keeps its original sum-only contract.
        if constexpr (ADD_INITIAL) {
            if constexpr (std::is_same_v<T, float>)
                output[hypothesis] = __fadd_rn(lane_sums[0], initial_diff2[batch]);
            else
                output[hypothesis] = __dadd_rn(lane_sums[0], initial_diff2[batch]);
        } else {
            output[hypothesis] = lane_sums[0];
        }
    }
}

template <typename T, typename ComplexT>
__global__ __launch_bounds__(kRelionFineDiff2BlockSize)
void relion_fine_diff2_pairs_kernel(
    const ComplexT* reference,
    const ComplexT* shifted_image,
    const T* weight,
    const int32_t* full_to_compact,
    T* output,
    int64_t batch_size,
    int64_t pair_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t hypothesis = static_cast<int64_t>(blockIdx.x);
    const int64_t total_hypotheses = batch_size * pair_count;
    if (hypothesis >= total_hypotheses) return;

    const int64_t batch = hypothesis / pair_count;
    T lane_sum = static_cast<T>(0);
    for (int64_t full_pixel = threadIdx.x;
         full_pixel < full_pixel_count;
         full_pixel += kRelionFineDiff2BlockSize) {
        const int32_t compact_pixel = full_to_compact[full_pixel];
        if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
        const int64_t operand_index = hypothesis * compact_pixel_count + compact_pixel;
        const int64_t weight_index = batch * compact_pixel_count + compact_pixel;
        if constexpr (std::is_same_v<T, float>)
            lane_sum = relion_fine_diff2_update_f32(
                reference[operand_index], shifted_image[operand_index],
                weight[weight_index], lane_sum);
        else
            lane_sum = relion_fine_diff2_update_f64(
                reference[operand_index], shifted_image[operand_index],
                weight[weight_index], lane_sum);
    }

    __shared__ T lane_sums[kRelionFineDiff2BlockSize];
    lane_sums[threadIdx.x] = lane_sum;
    __syncthreads();
    for (int width = kRelionFineDiff2BlockSize / 2; width > 0; width /= 2) {
        if (threadIdx.x < width)
            if constexpr (std::is_same_v<T, float>)
                lane_sums[threadIdx.x] = __fadd_rn(
                    lane_sums[threadIdx.x], lane_sums[threadIdx.x + width]);
            else
                lane_sums[threadIdx.x] = __dadd_rn(
                    lane_sums[threadIdx.x], lane_sums[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) output[hypothesis] = lane_sums[0];
}

template <bool FlatRows>
__global__ __launch_bounds__(kRelionFineDiff2BlockSize)
void relion_fine_diff2_fused_translate_rows_f32_kernel(
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* row_image_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t row_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    int current_size,
    const int32_t* runtime_current_size)
{
    const int64_t translation_chunks =
        (translation_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t row = flat_block / translation_chunks;
    const int64_t translation_chunk = flat_block % translation_chunks;
    if (row >= row_count) return;
    const int64_t translation_start =
        translation_chunk * kRelionFineDiff2Ref3dJobChunk;
    const int translation_in_chunk = static_cast<int>(min(
        static_cast<int64_t>(kRelionFineDiff2Ref3dJobChunk),
        translation_count - translation_start));
    const int64_t batch = FlatRows
        ? static_cast<int64_t>(row_image_ids[row])
        : row / rotation_count;
    if (batch < 0 || batch >= batch_size) {
        if (threadIdx.x < translation_in_chunk) {
            const int64_t translation = translation_start + threadIdx.x;
            output[row * translation_count + translation] =
                __int_as_float(0x7f800000);
        }
        return;
    }
    const int logical_current_size = runtime_current_size == nullptr
        ? current_size
        : runtime_current_size[0];
    const int64_t logical_full_pixel_count =
        static_cast<int64_t>(logical_current_size) *
        (logical_current_size / 2 + 1);
    if (logical_current_size <= 0 || (logical_current_size & 1) != 0 ||
        logical_full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < translation_in_chunk) {
            const int64_t translation = translation_start + threadIdx.x;
            const int64_t output_index = row * translation_count + translation;
            output[output_index] = nanf("");
        }
        return;
    }
    __shared__ float lane_sums[
        kRelionFineDiff2BlockSize * kRelionFineDiff2TranslationCapacity];
    for (int translation_offset = 0;
         translation_offset < translation_in_chunk;
         ++translation_offset) {
        lane_sums[translation_offset * kRelionFineDiff2BlockSize + threadIdx.x] =
            0.0f;
    }

    const int current_half_width = logical_current_size / 2 + 1;
    const int pass_count = static_cast<int>(
        (logical_full_pixel_count + kRelionFineDiff2BlockSize - 1) /
        kRelionFineDiff2BlockSize);
    for (int pass = 0; pass < pass_count; ++pass) {
        const int64_t full_pixel =
            static_cast<int64_t>(pass) * kRelionFineDiff2BlockSize + threadIdx.x;
        if (full_pixel < logical_full_pixel_count) {
            const int32_t compact_pixel = full_to_compact[full_pixel];
            if (compact_pixel >= 0 && compact_pixel < compact_pixel_count) {
                const int x = static_cast<int>(full_pixel % current_half_width);
                int y = static_cast<int>(full_pixel / current_half_width);
                if (y > logical_current_size / 2) y -= logical_current_size;
                const int64_t reference_index =
                    row * compact_pixel_count + compact_pixel;
                const int64_t image_index =
                    batch * compact_pixel_count + compact_pixel;
                const int64_t weight_index =
                    batch * compact_pixel_count + compact_pixel;
                const float2 image_value = image[image_index];
                const float2 reference_value = reference[reference_index];
                const float pixel_weight = weight[weight_index];
                for (int translation_offset = 0;
                     translation_offset < translation_in_chunk;
                     ++translation_offset) {
                    const int64_t translation =
                        translation_start + translation_offset;
                    const float tx = translation_angles[2 * translation];
                    const float ty = translation_angles[2 * translation + 1];
                    const float2 shifted = relion_score_translate_f32(
                        image_value, x, y, tx, ty);
                    const int lane_index =
                        translation_offset * kRelionFineDiff2BlockSize +
                        threadIdx.x;
                    lane_sums[lane_index] = relion_fine_diff2_update_f32(
                        reference_value,
                        shifted,
                        pixel_weight,
                        lane_sums[lane_index]);
                }
            }
        }
        __syncthreads();
    }

    for (int width = kRelionFineDiff2BlockSize / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            for (int translation_offset = 0;
                 translation_offset < translation_in_chunk;
                 ++translation_offset) {
                const int lane_index =
                    translation_offset * kRelionFineDiff2BlockSize + threadIdx.x;
                lane_sums[lane_index] = __fadd_rn(
                    lane_sums[lane_index], lane_sums[lane_index + width]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < translation_in_chunk) {
        const int64_t translation = translation_start + threadIdx.x;
        const int64_t output_index = row * translation_count + translation;
        output[output_index] = __fadd_rn(
            lane_sums[threadIdx.x * kRelionFineDiff2BlockSize],
            initial_diff2[batch]);
    }
}

__global__ __launch_bounds__(kRelionFineDiff2BlockSize)
void relion_fine_diff2_fused_translate_pairs_f32_kernel(
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* pair_reference_rows,
    const int32_t* pair_translation_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t reference_row_count,
    int64_t pair_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    int current_size,
    const int32_t* runtime_current_size)
{
    // RELION's makeJobsForDiff2Fine groups a short source-ordered translation
    // run into one block.  The compact pair ABI already stores candidates in
    // that source order, so consume four adjacent pairs per block without
    // materializing another job layout.  A chunk may cross a rotation run;
    // each lane keeps its own reference id, which preserves arbitrary compact
    // masks while still sharing the image/weight fetch and launch overhead.
    const int64_t pair_chunks =
        (pair_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = flat_block / pair_chunks;
    const int64_t pair_chunk = flat_block % pair_chunks;
    if (batch >= batch_size) return;
    const int64_t pair_start =
        pair_chunk * kRelionFineDiff2Ref3dJobChunk;
    const int pairs_in_chunk = static_cast<int>(min(
        static_cast<int64_t>(kRelionFineDiff2Ref3dJobChunk),
        pair_count - pair_start));

    int32_t reference_rows[kRelionFineDiff2TranslationCapacity];
    int32_t translations[kRelionFineDiff2TranslationCapacity];
    bool valid_pairs[kRelionFineDiff2TranslationCapacity];
    bool any_valid_pair = false;
    #pragma unroll
    for (int pair_offset = 0;
         pair_offset < kRelionFineDiff2TranslationCapacity;
         ++pair_offset) {
        const bool in_chunk = pair_offset < pairs_in_chunk;
        const int64_t hypothesis =
            batch * pair_count + pair_start + pair_offset;
        const int32_t reference_row =
            in_chunk ? pair_reference_rows[hypothesis] : -1;
        const int32_t translation =
            in_chunk ? pair_translation_ids[hypothesis] : -1;
        const bool valid =
            in_chunk && reference_row >= 0 &&
            reference_row < reference_row_count && translation >= 0 &&
            translation < translation_count;
        reference_rows[pair_offset] = reference_row;
        translations[pair_offset] = translation;
        valid_pairs[pair_offset] = valid;
        any_valid_pair = any_valid_pair || valid;
    }
    if (!any_valid_pair) {
        if (threadIdx.x < pairs_in_chunk) {
            const int64_t hypothesis =
                batch * pair_count + pair_start + threadIdx.x;
            output[hypothesis] = __int_as_float(0x7f800000);
        }
        return;
    }

    const int logical_current_size = runtime_current_size == nullptr
        ? current_size
        : runtime_current_size[0];
    const int64_t logical_full_pixel_count =
        static_cast<int64_t>(logical_current_size) *
        (logical_current_size / 2 + 1);
    if (logical_current_size <= 0 || (logical_current_size & 1) != 0 ||
        logical_full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < pairs_in_chunk) {
            const int pair_offset = threadIdx.x;
            const int64_t hypothesis =
                batch * pair_count + pair_start + pair_offset;
            output[hypothesis] = valid_pairs[pair_offset]
                ? nanf("")
                : __int_as_float(0x7f800000);
        }
        return;
    }

    __shared__ float lane_sums[
        kRelionFineDiff2BlockSize * kRelionFineDiff2TranslationCapacity];
    float pair_sums[kRelionFineDiff2TranslationCapacity];
    #pragma unroll
    for (int pair_offset = 0;
         pair_offset < kRelionFineDiff2TranslationCapacity;
         ++pair_offset) {
        pair_sums[pair_offset] = 0.0f;
    }
    const int current_half_width = logical_current_size / 2 + 1;
    const int pass_count = static_cast<int>(
        (logical_full_pixel_count + kRelionFineDiff2BlockSize - 1) /
        kRelionFineDiff2BlockSize);
    for (int pass = 0; pass < pass_count; ++pass) {
        const int64_t full_pixel =
            static_cast<int64_t>(pass) * kRelionFineDiff2BlockSize +
            threadIdx.x;
        if (full_pixel < logical_full_pixel_count) {
            const int32_t compact_pixel = full_to_compact[full_pixel];
            if (compact_pixel >= 0 && compact_pixel < compact_pixel_count) {
                const int x = static_cast<int>(full_pixel % current_half_width);
                int y = static_cast<int>(full_pixel / current_half_width);
                if (y > logical_current_size / 2) y -= logical_current_size;
                const int64_t image_index =
                    batch * compact_pixel_count + compact_pixel;
                const float2 image_value = image[image_index];
                const float pixel_weight = weight[image_index];
                #pragma unroll
                for (int pair_offset = 0;
                     pair_offset < kRelionFineDiff2TranslationCapacity;
                     ++pair_offset) {
                    if (pair_offset >= pairs_in_chunk ||
                        !valid_pairs[pair_offset])
                        continue;
                    const int64_t reference_index =
                        static_cast<int64_t>(reference_rows[pair_offset]) *
                            compact_pixel_count +
                        compact_pixel;
                    const int32_t translation = translations[pair_offset];
                    const float2 shifted = relion_score_translate_f32(
                        image_value,
                        x,
                        y,
                        translation_angles[2 * translation],
                        translation_angles[2 * translation + 1]);
                    pair_sums[pair_offset] = relion_fine_diff2_update_f32(
                        reference[reference_index],
                        shifted,
                        pixel_weight,
                        pair_sums[pair_offset]);
                }
            }
        }
    }
    #pragma unroll
    for (int pair_offset = 0;
         pair_offset < kRelionFineDiff2TranslationCapacity;
         ++pair_offset) {
        if (pair_offset < pairs_in_chunk && valid_pairs[pair_offset]) {
            lane_sums[
                pair_offset * kRelionFineDiff2BlockSize + threadIdx.x] =
                pair_sums[pair_offset];
        }
    }
    __syncthreads();
    for (int width = kRelionFineDiff2BlockSize / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            #pragma unroll
            for (int pair_offset = 0;
                 pair_offset < kRelionFineDiff2TranslationCapacity;
                 ++pair_offset) {
                if (pair_offset >= pairs_in_chunk ||
                    !valid_pairs[pair_offset])
                    continue;
                const int lane_index =
                    pair_offset * kRelionFineDiff2BlockSize + threadIdx.x;
                lane_sums[lane_index] = __fadd_rn(
                    lane_sums[lane_index], lane_sums[lane_index + width]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < pairs_in_chunk) {
        const int pair_offset = threadIdx.x;
        const int64_t hypothesis =
            batch * pair_count + pair_start + pair_offset;
        output[hypothesis] = valid_pairs[pair_offset]
            ? __fadd_rn(
                  lane_sums[pair_offset * kRelionFineDiff2BlockSize],
                  initial_diff2[batch])
            : __int_as_float(0x7f800000);
    }
}

/* Compact source-ordered fine jobs.  Each row is
 * (image, projected-reference row, dense rotation row, translation).  The
 * dense rotation field is carried for the caller's scatter and deliberately
 * ignored here.  Packing across the whole microbatch avoids the B * P_max
 * expansion of the transitional pair ABI while retaining RELION's four-job
 * block grouping and exact 256-lane accumulation tree. */
__global__ __launch_bounds__(kRelionFineDiff2BlockSize)
void relion_fine_diff2_fused_translate_jobs_f32_kernel(
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* job_plan,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t reference_row_count,
    int64_t job_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_capacity,
    int current_size,
    const int32_t* runtime_current_size)
{
    constexpr int kJobsPerBlock = kRelionFineDiff2Ref3dJobChunk;
    const int64_t job_start =
        static_cast<int64_t>(blockIdx.x) * kJobsPerBlock;
    if (job_start >= job_count) return;
    const int jobs_in_block = static_cast<int>(min(
        static_cast<int64_t>(kJobsPerBlock), job_count - job_start));

    int32_t image_rows[kJobsPerBlock];
    int32_t reference_rows[kJobsPerBlock];
    int32_t translations[kJobsPerBlock];
    bool valid_jobs[kJobsPerBlock];
    bool any_valid_job = false;
    #pragma unroll
    for (int offset = 0; offset < kJobsPerBlock; ++offset) {
        const bool in_block = offset < jobs_in_block;
        const int64_t plan_offset = 4 * (job_start + offset);
        const int32_t image_row = in_block ? job_plan[plan_offset] : -1;
        const int32_t reference_row =
            in_block ? job_plan[plan_offset + 1] : -1;
        const int32_t rotation_row =
            in_block ? job_plan[plan_offset + 2] : -1;
        const int32_t translation =
            in_block ? job_plan[plan_offset + 3] : -1;
        const bool valid =
            in_block && image_row >= 0 && image_row < batch_size &&
            reference_row >= 0 && reference_row < reference_row_count &&
            rotation_row >= 0 &&
            translation >= 0 && translation < translation_count;
        image_rows[offset] = image_row;
        reference_rows[offset] = reference_row;
        translations[offset] = translation;
        valid_jobs[offset] = valid;
        any_valid_job = any_valid_job || valid;
    }
    if (!any_valid_job) {
        if (threadIdx.x < jobs_in_block)
            output[job_start + threadIdx.x] = __int_as_float(0x7f800000);
        return;
    }

    const int logical_current_size = runtime_current_size == nullptr
        ? current_size
        : runtime_current_size[0];
    const int64_t logical_full_pixel_count =
        static_cast<int64_t>(logical_current_size) *
        (logical_current_size / 2 + 1);
    if (logical_current_size <= 0 || (logical_current_size & 1) != 0 ||
        logical_full_pixel_count > full_pixel_capacity) {
        if (threadIdx.x < jobs_in_block) {
            const int offset = threadIdx.x;
            output[job_start + offset] = valid_jobs[offset]
                ? nanf("")
                : __int_as_float(0x7f800000);
        }
        return;
    }

    float job_sums[kJobsPerBlock];
    #pragma unroll
    for (int offset = 0; offset < kJobsPerBlock; ++offset)
        job_sums[offset] = 0.0f;
    const int current_half_width = logical_current_size / 2 + 1;
    const int pass_count = static_cast<int>(
        (logical_full_pixel_count + kRelionFineDiff2BlockSize - 1) /
        kRelionFineDiff2BlockSize);
    for (int pass = 0; pass < pass_count; ++pass) {
        const int64_t full_pixel =
            static_cast<int64_t>(pass) * kRelionFineDiff2BlockSize +
            threadIdx.x;
        if (full_pixel >= logical_full_pixel_count) continue;
        const int32_t compact_pixel = full_to_compact[full_pixel];
        if (compact_pixel < 0 || compact_pixel >= compact_pixel_count) continue;
        const int x = static_cast<int>(full_pixel % current_half_width);
        int y = static_cast<int>(full_pixel / current_half_width);
        if (y > logical_current_size / 2) y -= logical_current_size;
        #pragma unroll
        for (int offset = 0; offset < kJobsPerBlock; ++offset) {
            if (offset >= jobs_in_block || !valid_jobs[offset]) continue;
            const int64_t operand_index =
                static_cast<int64_t>(image_rows[offset]) *
                    compact_pixel_count + compact_pixel;
            const int64_t reference_index =
                static_cast<int64_t>(reference_rows[offset]) *
                    compact_pixel_count + compact_pixel;
            const int32_t translation = translations[offset];
            const float2 shifted = relion_score_translate_f32(
                image[operand_index],
                x,
                y,
                translation_angles[2 * translation],
                translation_angles[2 * translation + 1]);
            job_sums[offset] = relion_fine_diff2_update_f32(
                reference[reference_index],
                shifted,
                weight[operand_index],
                job_sums[offset]);
        }
    }

    __shared__ float lane_sums[
        kRelionFineDiff2BlockSize * kJobsPerBlock];
    #pragma unroll
    for (int offset = 0; offset < kJobsPerBlock; ++offset) {
        if (offset < jobs_in_block && valid_jobs[offset])
            lane_sums[offset * kRelionFineDiff2BlockSize + threadIdx.x] =
                job_sums[offset];
    }
    __syncthreads();
    for (int width = kRelionFineDiff2BlockSize / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            #pragma unroll
            for (int offset = 0; offset < kJobsPerBlock; ++offset) {
                if (offset >= jobs_in_block || !valid_jobs[offset]) continue;
                const int lane =
                    offset * kRelionFineDiff2BlockSize + threadIdx.x;
                lane_sums[lane] = __fadd_rn(
                    lane_sums[lane], lane_sums[lane + width]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < jobs_in_block) {
        const int offset = threadIdx.x;
        output[job_start + offset] = valid_jobs[offset]
            ? __fadd_rn(
                  lane_sums[offset * kRelionFineDiff2BlockSize],
                  initial_diff2[image_rows[offset]])
            : __int_as_float(0x7f800000);
    }
}

template <typename T, typename ComplexT, bool ADD_INITIAL>
cudaError_t launch_relion_fine_diff2_rectangular(
    cudaStream_t stream,
    const ComplexT* reference,
    const ComplexT* shifted_image,
    const T* weight,
    const T* initial_diff2,
    const int32_t* full_to_compact,
    T* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t total_hypotheses = batch_size * rotation_count * translation_count;
    if (total_hypotheses == 0) return cudaSuccess;
    relion_fine_diff2_rectangular_kernel<T, ComplexT, ADD_INITIAL><<<
        static_cast<unsigned int>(total_hypotheses),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            shifted_image,
            weight,
            initial_diff2,
            full_to_compact,
            output,
            batch_size,
            rotation_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count);
    return cudaGetLastError();
}

__global__ __launch_bounds__(kRelionPowerClassBlockSize)
void relion_powerclass_spectrum_highres_f32_kernel(
    const float2* image,
    float* spectrum,
    int image_size,
    int spectrum_size,
    int xdim,
    int ydim,
    int resolution_limit,
    float* highres_xi2)
{
    const int tid = threadIdx.x;
    const int voxel = tid + static_cast<int>(blockIdx.x) * kRelionPowerClassBlockSize;
    __shared__ float highres_lanes[kRelionPowerClassBlockSize];
    highres_lanes[tid] = 0.0f;

    if (voxel < image_size) {
        const int x = voxel % xdim;
        int y = (voxel - x) / xdim;
        y = y < xdim ? y : y - ydim;
        const int radius_squared = x * x + y * y;
        const bool coordinates_in_range = !(x == 0 && y < 0);
        const int shell = __float2int_rn(sqrtf(static_cast<float>(radius_squared)));
        if (shell > 0 && shell < spectrum_size && coordinates_in_range) {
            const float value =
                image[voxel].x * image[voxel].x +
                image[voxel].y * image[voxel].y;
            atomicAdd(&spectrum[shell], value);
            if (shell >= resolution_limit)
                highres_lanes[tid] = value;
        }
    }

    __syncthreads();
    for (int width = kRelionPowerClassBlockSize / 2; width > 0; width /= 2) {
        if (tid < width)
            highres_lanes[tid] += highres_lanes[tid + width];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(highres_xi2, highres_lanes[0]);
}

cudaError_t launch_relion_powerclass_spectrum_highres_f32(
    cudaStream_t stream,
    const float2* image,
    float* spectrum_and_highres,
    int64_t batch_size,
    int image_size,
    int spectrum_size,
    int xdim,
    int ydim,
    int resolution_limit)
{
    const int output_stride = spectrum_size + 1;
    cudaError_t err = cudaMemsetAsync(
        spectrum_and_highres,
        0,
        static_cast<size_t>(batch_size) * output_stride * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    const int block_count =
        (image_size + kRelionPowerClassBlockSize - 1) /
        kRelionPowerClassBlockSize;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        float* output_row = spectrum_and_highres + batch * output_stride;
        relion_powerclass_spectrum_highres_f32_kernel<<<
            block_count,
            kRelionPowerClassBlockSize,
            0,
            stream>>>(
                image + batch * image_size,
                output_row,
                image_size,
                spectrum_size,
                xdim,
                ydim,
                resolution_limit,
                output_row + spectrum_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

__global__ __launch_bounds__(kRelionPowerClassBlockSize)
void relion_powerclass_spectrum_highres_runtime_f32_kernel(
    const float2* image,
    float* spectrum,
    int image_size,
    int spectrum_size,
    int xdim,
    int ydim,
    const int32_t* __restrict__ runtime_resolution_limit,
    float* highres_xi2)
{
    const int tid = threadIdx.x;
    const int voxel = tid + static_cast<int>(blockIdx.x) * kRelionPowerClassBlockSize;
    __shared__ float highres_lanes[kRelionPowerClassBlockSize];
    highres_lanes[tid] = 0.0f;
    const int active_resolution_limit = runtime_resolution_limit[0];
    if (active_resolution_limit < 0 || active_resolution_limit > spectrum_size) {
        if (tid == 0) atomicExch(highres_xi2, nanf(""));
        return;
    }

    if (voxel < image_size) {
        const int x = voxel % xdim;
        int y = (voxel - x) / xdim;
        y = y < xdim ? y : y - ydim;
        const int radius_squared = x * x + y * y;
        const bool coordinates_in_range = !(x == 0 && y < 0);
        const int shell = __float2int_rn(sqrtf(static_cast<float>(radius_squared)));
        if (shell > 0 && shell < spectrum_size && coordinates_in_range) {
            const float value =
                image[voxel].x * image[voxel].x +
                image[voxel].y * image[voxel].y;
            atomicAdd(&spectrum[shell], value);
            if (shell >= active_resolution_limit)
                highres_lanes[tid] = value;
        }
    }

    __syncthreads();
    for (int width = kRelionPowerClassBlockSize / 2; width > 0; width /= 2) {
        if (tid < width)
            highres_lanes[tid] += highres_lanes[tid + width];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(highres_xi2, highres_lanes[0]);
}

cudaError_t launch_relion_powerclass_spectrum_highres_runtime_f32(
    cudaStream_t stream,
    const float2* image,
    float* spectrum_and_highres,
    int64_t batch_size,
    int image_size,
    int spectrum_size,
    int xdim,
    int ydim,
    const int32_t* runtime_resolution_limit)
{
    const int output_stride = spectrum_size + 1;
    cudaError_t err = cudaMemsetAsync(
        spectrum_and_highres,
        0,
        static_cast<size_t>(batch_size) * output_stride * sizeof(float),
        stream);
    if (err != cudaSuccess) return err;
    const int block_count =
        (image_size + kRelionPowerClassBlockSize - 1) /
        kRelionPowerClassBlockSize;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        float* output_row = spectrum_and_highres + batch * output_stride;
        relion_powerclass_spectrum_highres_runtime_f32_kernel<<<
            block_count,
            kRelionPowerClassBlockSize,
            0,
            stream>>>(
                image + batch * image_size,
                output_row,
                image_size,
                spectrum_size,
                xdim,
                ydim,
                runtime_resolution_limit,
                output_row + spectrum_size);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template <typename T, typename ComplexT>
cudaError_t launch_relion_fine_diff2_pairs(
    cudaStream_t stream,
    const ComplexT* reference,
    const ComplexT* shifted_image,
    const T* weight,
    const int32_t* full_to_compact,
    T* output,
    int64_t batch_size,
    int64_t pair_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count)
{
    const int64_t total_hypotheses = batch_size * pair_count;
    if (total_hypotheses == 0) return cudaSuccess;
    relion_fine_diff2_pairs_kernel<T, ComplexT><<<
        static_cast<unsigned int>(total_hypotheses),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            shifted_image,
            weight,
            full_to_compact,
            output,
            batch_size,
            pair_count,
            compact_pixel_count,
            full_pixel_count);
    return cudaGetLastError();
}

cudaError_t launch_relion_fine_diff2_fused_translate_rectangular_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    const int32_t* runtime_current_size)
{
    const int64_t translation_chunks =
        (translation_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks =
        batch_size * rotation_count * translation_chunks;
    if (total_blocks == 0) return cudaSuccess;
    relion_fine_diff2_fused_translate_rows_f32_kernel<false><<<
        static_cast<unsigned int>(total_blocks),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            image,
            translation_angles,
            weight,
            initial_diff2,
            nullptr,
            full_to_compact,
            output,
            batch_size,
            rotation_count,
            batch_size * rotation_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count,
            current_size,
            runtime_current_size);
    return cudaGetLastError();
}

cudaError_t launch_relion_fine_diff2_fused_translate_flat_rows_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* row_image_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t row_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    const int32_t* runtime_current_size)
{
    const int64_t translation_chunks =
        (translation_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks = row_count * translation_chunks;
    if (total_blocks == 0) return cudaSuccess;
    relion_fine_diff2_fused_translate_rows_f32_kernel<true><<<
        static_cast<unsigned int>(total_blocks),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            image,
            translation_angles,
            weight,
            initial_diff2,
            row_image_ids,
            full_to_compact,
            output,
            batch_size,
            0,
            row_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count,
            current_size,
            runtime_current_size);
    return cudaGetLastError();
}

cudaError_t launch_relion_fine_diff2_fused_translate_pairs_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* pair_reference_rows,
    const int32_t* pair_translation_ids,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t reference_row_count,
    int64_t pair_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    const int32_t* runtime_current_size)
{
    const int64_t pair_chunks =
        (pair_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    const int64_t total_blocks = batch_size * pair_chunks;
    if (total_blocks == 0) return cudaSuccess;
    relion_fine_diff2_fused_translate_pairs_f32_kernel<<<
        static_cast<unsigned int>(total_blocks),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            image,
            translation_angles,
            weight,
            initial_diff2,
            pair_reference_rows,
            pair_translation_ids,
            full_to_compact,
            output,
            batch_size,
            reference_row_count,
            pair_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count,
            current_size,
            runtime_current_size);
    return cudaGetLastError();
}

cudaError_t launch_relion_fine_diff2_fused_translate_jobs_f32(
    cudaStream_t stream,
    const float2* reference,
    const float2* image,
    const float* translation_angles,
    const float* weight,
    const float* initial_diff2,
    const int32_t* job_plan,
    const int32_t* full_to_compact,
    float* output,
    int64_t batch_size,
    int64_t reference_row_count,
    int64_t job_count,
    int64_t translation_count,
    int64_t compact_pixel_count,
    int64_t full_pixel_count,
    int current_size,
    const int32_t* runtime_current_size)
{
    const int64_t total_blocks =
        (job_count + kRelionFineDiff2Ref3dJobChunk - 1) /
        kRelionFineDiff2Ref3dJobChunk;
    if (total_blocks == 0) return cudaSuccess;
    relion_fine_diff2_fused_translate_jobs_f32_kernel<<<
        static_cast<unsigned int>(total_blocks),
        kRelionFineDiff2BlockSize,
        0,
        stream>>>(
            reference,
            image,
            translation_angles,
            weight,
            initial_diff2,
            job_plan,
            full_to_compact,
            output,
            batch_size,
            reference_row_count,
            job_count,
            translation_count,
            compact_pixel_count,
            full_pixel_count,
            current_size,
            runtime_current_size);
    return cudaGetLastError();
}
