/* Shared body for the default RELION coarse projector and the optional
 * prehalved-weight specialization.  The including wrapper must define
 * RECOVAR_RELION_COARSE_STAGE_WEIGHT(weight) and
 * RECOVAR_RELION_COARSE_DIFF2_UPDATE before including this file. */
{
    static_assert(
        !SINGLE_LANE_CANONICAL || CANONICAL_REDUCTION,
        "single-lane coarse reduction requires canonical reduction");
    const int rotation_blocks =
        (rotation_count + EULERS_PER_BLOCK - 1) / EULERS_PER_BLOCK;
    const int flat_block = static_cast<int>(blockIdx.x);
    const int batch = flat_block / rotation_blocks;
    const int block_in_batch = flat_block - batch * rotation_blocks;
    const int rotation_start =
        rotation_offset + block_in_batch * EULERS_PER_BLOCK;
    if (batch >= batch_size) return;

    constexpr int pixels_per_chunk =
        kRelionCoarseDiff2BlockSize / kRelionCoarsePrefetchFraction;
    __shared__ float shared_rotations[EULERS_PER_BLOCK * 6];
    __shared__ float2 shared_references[
        pixels_per_chunk * EULERS_PER_BLOCK];
    __shared__ float2 shared_images[kRelionCoarseDiff2BlockSize];
    __shared__ float shared_weights[kRelionCoarseDiff2BlockSize];
    __shared__ float shared_lane_partials[
        CANONICAL_REDUCTION && !SINGLE_LANE_CANONICAL
        ? EULERS_PER_BLOCK * kRelionCoarseDiff2BlockSize
        : 1];
    for (int index = threadIdx.x; index < EULERS_PER_BLOCK * 6; index += blockDim.x) {
        const int local_rotation = index / 6;
        const int component = index - local_rotation * 6;
        const int rotation = rotation_start + local_rotation;
        shared_rotations[index] =
            rotation < rotation_offset + rotation_count
            ? rotations[rotation * 6 + component]
            : 0.0f;
    }
    __syncthreads();

    int translation;
    int lane;
    int active_lanes;
    bool active_thread;
    if constexpr (SINGLE_LANE_CANONICAL) {
        translation = static_cast<int>(threadIdx.x);
        lane = 0;
        active_lanes = 1;
        active_thread = threadIdx.x < translation_count;
    } else {
        translation = threadIdx.x % translation_count;
        lane = threadIdx.x / translation_count;
        active_lanes = kRelionCoarseDiff2BlockSize / translation_count;
        active_thread = lane < active_lanes;
    }
    const int current_half_width = current_size / 2 + 1;
    const int current_max_r = current_size / 2;
    const int full_pixel_count = current_size * current_half_width;
    const int padded_pixel_count =
        ((full_pixel_count + kRelionCoarseDiff2BlockSize - 1) /
         kRelionCoarseDiff2BlockSize) *
        kRelionCoarseDiff2BlockSize;
    float tx;
    float ty;
    if constexpr (SINGLE_LANE_CANONICAL) {
        // Threads above the runtime translation count do no scoring, and must
        // not read past the compact translation-angle array.
        tx = active_thread ? translation_angles[2 * translation] : 0.0f;
        ty = active_thread ? translation_angles[2 * translation + 1] : 0.0f;
    } else {
        tx = translation_angles[2 * translation];
        ty = translation_angles[2 * translation + 1];
    }
    float lane_sums[EULERS_PER_BLOCK] = {0.0f};

    // RELION projects each pixel/orientation pair once per block, stages the
    // result in shared memory, and reuses it for every translation lane.  The
    // previous fused implementation repeated the same texture projection in
    // every translation lane, which dominated InitialModel's coarse pass.
    for (int chunk_start = 0;
         chunk_start < padded_pixel_count;
         chunk_start += pixels_per_chunk) {
        __syncthreads();

        const int reference_full_pixel =
            chunk_start + threadIdx.x / kRelionCoarsePrefetchFraction;
        const int reference_x = reference_full_pixel % current_half_width;
        const int reference_native_y = reference_full_pixel / current_half_width;
        const int reference_y = reference_native_y > current_max_r
            ? reference_native_y - current_size
            : reference_native_y;
        for (int local_rotation =
                 threadIdx.x % kRelionCoarsePrefetchFraction;
             local_rotation < EULERS_PER_BLOCK;
             local_rotation += kRelionCoarsePrefetchFraction) {
            const int rotation = rotation_start + local_rotation;
            float2 reference = make_float2(0.0f, 0.0f);
            if (reference_full_pixel < full_pixel_count &&
                rotation < rotation_offset + rotation_count) {
                const float* R = &shared_rotations[local_rotation * 6];
                const float rk0 = R[3] * static_cast<float>(reference_x)
                    + R[0] * static_cast<float>(reference_y);
                const float rk1 = R[4] * static_cast<float>(reference_x)
                    + R[1] * static_cast<float>(reference_y);
                const float rk2 = R[5] * static_cast<float>(reference_x)
                    + R[2] * static_cast<float>(reference_y);
                if (static_cast<int>(rk0 * rk0 + rk1 * rk1 + rk2 * rk2)
                    <= model_max_r2) {
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
                    reference.x = projector_scale * tex3D<float>(
                        tex_real,
                        xp + 0.5f,
                        yp - static_cast<float>(tex_yinit) + 0.5f,
                        zp - static_cast<float>(tex_zinit) + 0.5f);
                    reference.y = projector_scale * imag_sign * tex3D<float>(
                        tex_imag,
                        xp + 0.5f,
                        yp - static_cast<float>(tex_yinit) + 0.5f,
                        zp - static_cast<float>(tex_zinit) + 0.5f);
                }
            }
            shared_references[
                (threadIdx.x / kRelionCoarsePrefetchFraction) *
                    EULERS_PER_BLOCK +
                local_rotation] = reference;
        }

        if (chunk_start % kRelionCoarseDiff2BlockSize == 0) {
            const int image_full_pixel = chunk_start + threadIdx.x;
            const int compact_pixel = image_full_pixel < full_pixel_count
                ? full_to_compact[image_full_pixel]
                : -1;
            float2 image_value = make_float2(0.0f, 0.0f);
            float pixel_weight = 0.0f;
            if (compact_pixel >= 0 && compact_pixel < compact_pixel_count) {
                image_value =
                    images[batch * compact_pixel_count + compact_pixel];
                pixel_weight =
                    weight[batch * compact_pixel_count + compact_pixel];
                RECOVAR_RELION_COARSE_STAGE_WEIGHT(pixel_weight)
            }
            shared_images[threadIdx.x] = image_value;
            shared_weights[threadIdx.x] = pixel_weight;
        }

        __syncthreads();

        if (active_thread) {
            const int first_pixel_in_chunk =
                SINGLE_LANE_CANONICAL ? 0 : lane;
            const int pixel_stride =
                SINGLE_LANE_CANONICAL ? 1 : active_lanes;
            for (int pixel_in_chunk = first_pixel_in_chunk;
                 pixel_in_chunk < pixels_per_chunk;
                 pixel_in_chunk += pixel_stride) {
                const int full_pixel = chunk_start + pixel_in_chunk;
                if (full_pixel >= full_pixel_count) break;
                const int compact_pixel = full_to_compact[full_pixel];
                if (compact_pixel < 0 || compact_pixel >= compact_pixel_count)
                    continue;
                const int x = full_pixel % current_half_width;
                const int native_y = full_pixel / current_half_width;
                const int y = native_y > current_max_r
                    ? native_y - current_size
                    : native_y;
                const int shared_pixel =
                    pixel_in_chunk +
                    chunk_start % kRelionCoarseDiff2BlockSize;
                const float2 shifted = relion_score_translate_f32(
                    shared_images[shared_pixel], x, y, tx, ty);
                const float pixel_weight = shared_weights[shared_pixel];

                #pragma unroll
                for (int local_rotation = 0;
                     local_rotation < EULERS_PER_BLOCK;
                     ++local_rotation) {
                    const int rotation = rotation_start + local_rotation;
                    if (rotation >= rotation_offset + rotation_count) continue;
                    lane_sums[local_rotation] = RECOVAR_RELION_COARSE_DIFF2_UPDATE(
                        shared_references[
                            pixel_in_chunk * EULERS_PER_BLOCK + local_rotation],
                        shifted,
                        pixel_weight,
                        lane_sums[local_rotation]);
                }
            }
        }
    }

    #pragma unroll
    for (int local_rotation = 0;
         local_rotation < EULERS_PER_BLOCK;
         ++local_rotation) {
        const int rotation = rotation_start + local_rotation;
        if (rotation >= rotation_offset + rotation_count) continue;
        if constexpr (CAPTURE_LANES) {
            lane_partials[
                (static_cast<int64_t>(batch) * output_rotation_count + rotation) *
                    kRelionCoarseDiff2BlockSize +
                threadIdx.x] = lane_sums[local_rotation];
        }
        if constexpr (SINGLE_LANE_CANONICAL) {
            // For 65--128 translations, active_lanes is exactly one: thread
            // ``translation`` is the sole contributor to that output.  Keep
            // RELION's initial-value-then-lane addition order without staging
            // an otherwise unused 8 KiB canonical-reduction buffer.
            if (threadIdx.x < translation_count) {
                const int output_index =
                    (batch * output_rotation_count + rotation) *
                        translation_count + threadIdx.x;
                output[output_index] = __fadd_rn(
                    output[output_index], lane_sums[local_rotation]);
            }
        } else if constexpr (CANONICAL_REDUCTION) {
            shared_lane_partials[
                local_rotation * kRelionCoarseDiff2BlockSize + threadIdx.x] =
                    lane_sums[local_rotation];
        } else {
            atomicAdd(
                &output[(batch * output_rotation_count + rotation) *
                        translation_count + translation],
                lane_sums[local_rotation]);
        }
    }

    if constexpr (CANONICAL_REDUCTION && !SINGLE_LANE_CANONICAL) {
        __syncthreads();
        if (threadIdx.x < translation_count) {
            #pragma unroll
            for (int local_rotation = 0;
                 local_rotation < EULERS_PER_BLOCK;
                 ++local_rotation) {
                const int rotation = rotation_start + local_rotation;
                if (rotation >= rotation_offset + rotation_count) continue;
                const int output_index =
                    (batch * output_rotation_count + rotation) *
                        translation_count + threadIdx.x;
                float total = output[output_index];
                for (int lane_index = 0;
                     lane_index < active_lanes;
                     ++lane_index) {
                    total = __fadd_rn(
                        total,
                        shared_lane_partials[
                            local_rotation * kRelionCoarseDiff2BlockSize +
                            threadIdx.x + lane_index * translation_count]);
                }
                output[output_index] = total;
            }
        }
    }
}
