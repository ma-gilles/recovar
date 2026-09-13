namespace {

/* Preserve RELION's by-value projector layout and method body for the VDAM
 * SGD discriminator.  The texture payload is pre-scaled into RELION's frame,
 * so projection itself has the same arithmetic and control flow as BP.cuh. */
struct RelionVdamProjectorKernel
{
    int mdlX, mdlXY, mdlZ;
    int imgX, imgY, imgZ;
    int mdlInitY, mdlInitZ;
    int maxR, maxR2, maxR2_padded;
    float padding_factor;
    cudaTextureObject_t mdlReal;
    cudaTextureObject_t mdlImag;

    __device__ __forceinline__ void project3Dmodel(
        int x,
        int y,
        float e0,
        float e1,
        float e3,
        float e4,
        float e6,
        float e7,
        float& real,
        float& imag)
    {
        float xp = (e0 * x + e1 * y) * padding_factor;
        float yp = (e3 * x + e4 * y) * padding_factor;
        float zp = (e6 * x + e7 * y) * padding_factor;
        int r2 = xp * xp + yp * yp + zp * zp;
        if (r2 <= maxR2_padded)
        {
            if (xp < 0.0f)
            {
                xp = -xp;
                yp = -yp;
                zp = -zp;
                yp -= mdlInitY;
                zp -= mdlInitZ;
                real = tex3D<float>(mdlReal, xp + 0.5f, yp + 0.5f, zp + 0.5f);
                imag = -tex3D<float>(mdlImag, xp + 0.5f, yp + 0.5f, zp + 0.5f);
            }
            else
            {
                yp -= mdlInitY;
                zp -= mdlInitZ;
                real = tex3D<float>(mdlReal, xp + 0.5f, yp + 0.5f, zp + 0.5f);
                imag = tex3D<float>(mdlImag, xp + 0.5f, yp + 0.5f, zp + 0.5f);
            }
        }
        else
        {
            real = 0.0f;
            imag = 0.0f;
        }
    }
};

static_assert(sizeof(RelionVdamProjectorKernel) == 64,
              "RELION AccProjectorKernel ABI must remain 64 bytes");
static_assert(alignof(RelionVdamProjectorKernel) == 8,
              "RELION AccProjectorKernel ABI must remain 8-byte aligned");

struct RelionVdamImageGeometry
{
    unsigned x;
    unsigned y;
    unsigned pixels;
    bool valid;
};

__device__ __forceinline__ RelionVdamImageGeometry
relion_vdam_runtime_image_geometry(
    unsigned physical_x,
    unsigned physical_y,
    unsigned physical_pixels,
    const int32_t* runtime_current_size)
{
    if (runtime_current_size == nullptr)
        return {physical_x, physical_y, physical_pixels, true};
    const int current_size = runtime_current_size[0];
    const int64_t logical_pixels = static_cast<int64_t>(current_size) *
        (current_size / 2 + 1);
    const bool valid = current_size > 0 && (current_size & 1) == 0 &&
        current_size <= static_cast<int>(physical_y) &&
        current_size / 2 + 1 <= static_cast<int>(physical_x) &&
        logical_pixels <= static_cast<int64_t>(physical_pixels);
    return {
        valid ? static_cast<unsigned>(current_size / 2 + 1) : 0U,
        valid ? static_cast<unsigned>(current_size) : 0U,
        valid ? static_cast<unsigned>(logical_pixels) : 0U,
        valid,
    };
}

__global__ void relion_vdam_native_project_f32_kernel(
    RelionVdamProjectorKernel projector,
    const float* eulers,
    float2* references,
    unsigned image_x,
    unsigned image_y,
    unsigned image_xyz,
    unsigned rotation_count,
    const int32_t* runtime_current_size)
{
    const unsigned image = blockIdx.x;
    if (image >= rotation_count) return;
    const RelionVdamImageGeometry logical =
        relion_vdam_runtime_image_geometry(
            image_x, image_y, image_xyz, runtime_current_size);
    if (!logical.valid) return;
    __shared__ float shared_eulers[9];
    if (threadIdx.x < 9)
        shared_eulers[threadIdx.x] = eulers[image * 9 + threadIdx.x];
    __syncthreads();
    const int image_y_half = logical.y / 2;
    for (unsigned pixel = threadIdx.x; pixel < logical.pixels; pixel += blockDim.x)
    {
        const int x = pixel % logical.x;
        int y = static_cast<int>(pixel / logical.x);
        if (y > image_y_half) y -= logical.y;
        float real = 0.0f;
        float imag = 0.0f;
        projector.project3Dmodel(
            x,
            y,
            shared_eulers[0],
            shared_eulers[1],
            shared_eulers[3],
            shared_eulers[4],
            shared_eulers[6],
            shared_eulers[7],
            real,
            imag);
        references[image * image_xyz + pixel] = make_float2(real, imag);
    }
}

__global__ void relion_vdam_scale_texture_f32_kernel(
    float* real,
    float* imag,
    int64_t count,
    float scale)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    real[index] *= scale;
    imag[index] *= scale;
}

__global__ void relion_vdam_split_translations_f32_kernel(
    const float* translation_angles,
    float* translation_x,
    float* translation_y,
    int64_t translation_count)
{
    const int64_t translation =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (translation >= translation_count) return;
    translation_x[translation] = translation_angles[2 * translation];
    translation_y[translation] = translation_angles[2 * translation + 1];
}

__device__ __forceinline__ void relion_vdam_translate_pixel_f32(
    int x,
    int y,
    float tx,
    float ty,
    float& real,
    float& imag,
    float& translated_real,
    float& translated_imag)
{
    float sine;
    float cosine;
    sincosf(x * tx + y * ty, &sine, &cosine);
    translated_real = cosine * real - sine * imag;
    translated_imag = cosine * imag + sine * real;
}

__device__ __forceinline__ void relion_vdam_native_residual_f32(
    unsigned image,
    unsigned pixel,
    int x,
    int y,
    float reference_real,
    float reference_imag,
    const float* image_real,
    const float* image_imag,
    const float* translation_x,
    const float* translation_y,
    const float* weights,
    const float* minvsigma2s,
    const float* ctfs,
    unsigned long translation_count,
    float significant_weight,
    float weight_norm,
    float& real,
    float& imag,
    float& Fweight)
{
    float minvsigma2 = __ldg(&minvsigma2s[pixel]);
    float ctf = __ldg(&ctfs[pixel]);
    float pixel_real = __ldg(&image_real[pixel]);
    float pixel_imag = __ldg(&image_imag[pixel]);
    Fweight = 0.0f;
    real = 0.0f;
    imag = 0.0f;
    reference_real *= ctf;
    reference_imag *= ctf;
    float translated_real;
    float translated_imag;
    for (unsigned long translation = 0; translation < translation_count;
         ++translation)
    {
        float weight = weights[image * translation_count + translation];
        if (weight >= significant_weight)
        {
            weight = (weight / weight_norm) * ctf * minvsigma2;
            Fweight += weight * ctf;
            relion_vdam_translate_pixel_f32(
                x,
                y,
                translation_x[translation],
                translation_y[translation],
                pixel_real,
                pixel_imag,
                translated_real,
                translated_imag);
            real += (translated_real - reference_real) * weight;
            imag += (translated_imag - reference_imag) * weight;
        }
    }
}

__global__ void relion_vdam_native_residual_f32_kernel(
    float2* references_and_residuals,
    float* residual_weights,
    const float* image_real,
    const float* image_imag,
    const float* translation_x,
    const float* translation_y,
    const float* weights,
    const float* minvsigma2s,
    const float* ctfs,
    unsigned long translation_count,
    float significant_weight,
    float weight_norm,
    unsigned image_x,
    unsigned image_y,
    unsigned image_xyz,
    unsigned rotation_count,
    const int32_t* runtime_current_size)
{
    const unsigned image = blockIdx.x;
    if (image >= rotation_count) return;
    const RelionVdamImageGeometry logical =
        relion_vdam_runtime_image_geometry(
            image_x, image_y, image_xyz, runtime_current_size);
    if (!logical.valid) return;
    const int image_y_half = logical.y / 2;
    for (unsigned pixel = threadIdx.x; pixel < logical.pixels; pixel += blockDim.x)
    {
        const int x = pixel % logical.x;
        int y = static_cast<int>(pixel / logical.x);
        if (y > image_y_half) y -= logical.y;
        const unsigned output = image * image_xyz + pixel;
        const float2 reference = references_and_residuals[output];
        float real;
        float imag;
        float Fweight;
        relion_vdam_native_residual_f32(
            image,
            pixel,
            x,
            y,
            reference.x,
            reference.y,
            image_real,
            image_imag,
            translation_x,
            translation_y,
            weights,
            minvsigma2s,
            ctfs,
            translation_count,
            significant_weight,
            weight_norm,
            real,
            imag,
            Fweight);
        references_and_residuals[output] = make_float2(real, imag);
        residual_weights[output] = Fweight;
    }
}

template <
    typename Accumulator,
    bool CapturedOrder,
    bool Trace,
    bool PersistentSerialRotations,
    bool FixedWarpOrderScatter>
__global__ void relion_vdam_native_sgd_f32_kernel(
    RelionVdamProjectorKernel projector,
    const float2* preprojected_references,
    const float2* precomputed_residuals,
    const float* precomputed_residual_weights,
    float* image_real,
    float* image_imag,
    float* translation_x,
    float* translation_y,
    float* translation_z,
    float* weights,
    float* minvsigma2s,
    float* ctfs,
    unsigned long translation_count,
    float significant_weight,
    float weight_norm,
    float* eulers,
    Accumulator* model_real,
    Accumulator* model_imag,
    Accumulator* model_weight,
    int max_r,
    int max_r2,
    float padding_factor,
    unsigned image_x,
    unsigned image_y,
    unsigned image_z,
    unsigned image_xyz,
    unsigned model_x,
    unsigned model_y,
    int model_init_y,
    int model_init_z,
    unsigned rotation_count,
    const int32_t* rotation_replay_order,
    VdamCandidateBlockTraceRecord* trace_records,
    std::uint64_t trace_launch_sequence,
    std::int64_t trace_particle_id,
    std::int32_t trace_worker_id,
    std::uint32_t trace_iteration,
    const int32_t* runtime_current_size)
{
    unsigned tid = threadIdx.x;
    const RelionVdamImageGeometry logical =
        relion_vdam_runtime_image_geometry(
            image_x, image_y, image_xyz, runtime_current_size);
    if (!logical.valid) return;
    __shared__ int trace_first_atomic_claimed;
    int image_y_half = logical.y / 2;
    if (runtime_current_size != nullptr)
    {
        const int logical_max_r = static_cast<int>(
            static_cast<float>(logical.y / 2) * padding_factor);
        max_r2 = logical_max_r * logical_max_r;
    }
    int max_r2_volume = max_r2 * padding_factor * padding_factor;
    __shared__ float shared_eulers[9];
    float Fweight;
    float real;
    float imag;
    const unsigned physical_image_begin =
        PersistentSerialRotations ? 0 : blockIdx.x;
    const unsigned physical_image_end = PersistentSerialRotations
        ? rotation_count
        : physical_image_begin + 1;
    for (unsigned physical_image = physical_image_begin;
         physical_image < physical_image_end;
         ++physical_image)
    {
        const unsigned image = CapturedOrder
            ? static_cast<unsigned>(rotation_replay_order[physical_image])
            : physical_image;
        VdamCandidateBlockTraceRecord* trace_record = nullptr;
        if constexpr (Trace)
        {
            if (trace_records != nullptr)
            {
                trace_record = trace_records + physical_image;
                if (tid == 0)
                {
                    trace_first_atomic_claimed = 0;
                    trace_record->launch_sequence = trace_launch_sequence;
                    trace_record->particle_id = trace_particle_id;
                    trace_record->block_start_globaltimer = vdam_candidate_globaltimer();
                    trace_record->first_atomic_globaltimer = 0;
                    trace_record->block_end_globaltimer = 0;
                    trace_record->orientation_row = image;
                    trace_record->worker_id = trace_worker_id;
                    trace_record->class_id = 0;
                    trace_record->sm_id = vdam_candidate_smid();
                    trace_record->image_count = rotation_count;
                    trace_record->iteration = trace_iteration;
                    trace_record->flags = std::uint32_t(1U << 2);
                    trace_record->reserved = 0;
                }
            }
        }
        if (tid < 9) shared_eulers[tid] = eulers[image * 9 + tid];
        __syncthreads();

        int pixel_pass_count = ceilf(static_cast<float>(logical.pixels) / 128.0f);
        for (unsigned pass = 0;
             pass < static_cast<unsigned>(pixel_pass_count);
             ++pass)
        {
            unsigned pixel = pass * 128 + tid;
            bool scatter_pixel = pixel < logical.pixels;
            int x = 0;
            int y = 0;
            int x0 = 0;
            int x1 = 0;
            int y0 = 0;
            int y1 = 0;
            int z0 = 0;
            int z1 = 0;
            float fx = 0.0f;
            float fy = 0.0f;
            float fz = 0.0f;
            float mfx = 0.0f;
            float mfy = 0.0f;
            float mfz = 0.0f;
            if (scatter_pixel)
            {
                x = pixel % logical.x;
                y = static_cast<int>(pixel / logical.x);
                if (y > image_y_half) y -= logical.y;

                if (precomputed_residuals != nullptr)
                {
                    const unsigned residual_index = image * image_xyz + pixel;
                    const float2 residual = precomputed_residuals[residual_index];
                    real = residual.x;
                    imag = residual.y;
                    Fweight = precomputed_residual_weights[residual_index];
                }
                else
                {
                    float reference_real;
                    float reference_imag;
                    if (preprojected_references != nullptr)
                    {
                        const float2 reference =
                            preprojected_references[image * image_xyz + pixel];
                        reference_real = reference.x;
                        reference_imag = reference.y;
                    }
                    else
                    {
                        reference_real = 0.0f;
                        reference_imag = 0.0f;
                        projector.project3Dmodel(
                            x,
                            y,
                            shared_eulers[0],
                            shared_eulers[1],
                            shared_eulers[3],
                            shared_eulers[4],
                            shared_eulers[6],
                            shared_eulers[7],
                            reference_real,
                            reference_imag);
                    }
                    relion_vdam_native_residual_f32(
                        image,
                        pixel,
                        x,
                        y,
                        reference_real,
                        reference_imag,
                        image_real,
                        image_imag,
                        translation_x,
                        translation_y,
                        weights,
                        minvsigma2s,
                        ctfs,
                        translation_count,
                        significant_weight,
                        weight_norm,
                        real,
                        imag,
                        Fweight);
                }

                scatter_pixel = Fweight > 0.0f;
                if (scatter_pixel)
                {
                    float xp =
                        (shared_eulers[0] * x + shared_eulers[1] * y) *
                        padding_factor;
                    float yp =
                        (shared_eulers[3] * x + shared_eulers[4] * y) *
                        padding_factor;
                    float zp =
                        (shared_eulers[6] * x + shared_eulers[7] * y) *
                        padding_factor;
                    scatter_pixel =
                        (xp * xp + yp * yp + zp * zp) <= max_r2_volume;
                    if (scatter_pixel)
                    {
                        if (xp < 0.0f)
                        {
                            xp = -xp;
                            yp = -yp;
                            zp = -zp;
                            imag = -imag;
                        }
                        // Native trace marks entry to interpolation, not the
                        // later scatter (which may wait for an ordered warp).
                        if constexpr (Trace)
                            if (trace_record != nullptr && trace_first_atomic_claimed == 0 &&
                                atomicCAS(&trace_first_atomic_claimed, 0, 1) == 0)
                                trace_record->first_atomic_globaltimer = vdam_candidate_globaltimer();
                        x0 = floorf(xp);
                        fx = xp - x0;
                        x1 = x0 + 1;
                        y0 = floorf(yp);
                        fy = yp - y0;
                        y0 -= model_init_y;
                        y1 = y0 + 1;
                        z0 = floorf(zp);
                        fz = zp - z0;
                        z0 -= model_init_z;
                        z1 = z0 + 1;
                        mfx = 1.0f - fx;
                        mfy = 1.0f - fy;
                        mfz = 1.0f - fz;
                    }
                }
            }

#define RELION_VDAM_NATIVE_ATOMIC_TRIPLET(Z, Y, X, COEFFICIENT)                    \
    atomicAdd(&model_real[(Z) * model_x * model_y + (Y) * model_x + (X)],          \
              static_cast<Accumulator>((COEFFICIENT) * real));                      \
    atomicAdd(&model_imag[(Z) * model_x * model_y + (Y) * model_x + (X)],          \
              static_cast<Accumulator>((COEFFICIENT) * imag));                      \
    atomicAdd(&model_weight[(Z) * model_x * model_y + (Y) * model_x + (X)],        \
              static_cast<Accumulator>((COEFFICIENT) * Fweight))
#define RELION_VDAM_NATIVE_SCATTER_PIXEL()                                          \
    float dd000 = mfz * mfy * mfx;                                                  \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z0, y0, x0, dd000);                           \
    float dd001 = mfz * mfy * fx;                                                   \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z0, y0, x1, dd001);                           \
    float dd010 = mfz * fy * mfx;                                                   \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z0, y1, x0, dd010);                           \
    float dd011 = mfz * fy * fx;                                                    \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z0, y1, x1, dd011);                           \
    float dd100 = fz * mfy * mfx;                                                   \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z1, y0, x0, dd100);                           \
    float dd101 = fz * mfy * fx;                                                    \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z1, y0, x1, dd101);                           \
    float dd110 = fz * fy * mfx;                                                    \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z1, y1, x0, dd110);                           \
    float dd111 = fz * fy * fx;                                                     \
    RELION_VDAM_NATIVE_ATOMIC_TRIPLET(z1, y1, x1, dd111)

            if constexpr (FixedWarpOrderScatter)
            {
                // Preserve the parallel preparation used by the mature EM
                // path, but make the contended reduction order independent
                // of the scheduler. A block has exactly four 32-thread warps.
                for (unsigned active_warp = 0; active_warp < 4; ++active_warp)
                {
                    if (scatter_pixel && tid / 32 == active_warp)
                    {
                        RELION_VDAM_NATIVE_SCATTER_PIXEL();
                    }
                    __syncthreads();
                }
            }
            else if (scatter_pixel)
            {
                RELION_VDAM_NATIVE_SCATTER_PIXEL();
            }
#undef RELION_VDAM_NATIVE_SCATTER_PIXEL
#undef RELION_VDAM_NATIVE_ATOMIC_TRIPLET
        }
    if constexpr (Trace)
    {
        __syncthreads();
        if (trace_record != nullptr && tid == 0)
        {
            if (trace_first_atomic_claimed == 0)
                trace_record->flags |= std::uint32_t(1U << 3);
            trace_record->block_end_globaltimer = vdam_candidate_globaltimer();
        }
    }
        // A kernel boundary orders the ordinary one-block replay.  Preserve
        // that orientation-to-orientation order inside the persistent block
        // before shared Euler and trace state are reused.
        __syncthreads();
    }
}

template <typename Output, typename Input>
__global__ void relion_vdam_cast_accumulator_kernel(
    const Input* input,
    Output* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<Output>(input[index]);
}

__global__ void relion_vdam_denominator_after_sgd_f32_kernel(
    const float* ctf,
    const float* minvsigma2,
    const float* posterior,
    float* denominator,
    int64_t particle_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t logical_pixel_count,
    int64_t pixel_capacity,
    const int32_t* runtime_current_size)
{
    const int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool valid_runtime_size = true;
    if (runtime_current_size != nullptr)
    {
        const int current_size = runtime_current_size[0];
        const int64_t runtime_pixel_count = static_cast<int64_t>(current_size) *
            (current_size / 2 + 1);
        valid_runtime_size = current_size > 0 && (current_size & 1) == 0 &&
            runtime_pixel_count <= pixel_capacity;
        logical_pixel_count = valid_runtime_size ? runtime_pixel_count : 0;
    }
    if (!valid_runtime_size)
    {
        const int64_t capacity_count =
            particle_count * rotation_count * pixel_capacity;
        if (output < capacity_count) denominator[output] = nanf("");
        return;
    }
    const int64_t logical_output_count =
        particle_count * rotation_count * logical_pixel_count;
    if (output >= logical_output_count) return;
    const int64_t pixel = output % logical_pixel_count;
    const int64_t particle_rotation = output / logical_pixel_count;
    const int64_t particle = particle_rotation / rotation_count;
    const int64_t rotation = particle_rotation % rotation_count;
    const float pixel_ctf = ctf[particle * pixel_capacity + pixel];
    const float pixel_minvsigma2 = minvsigma2[
        particle * pixel_capacity + pixel];
    float Fweight = 0.0f;
    const int64_t posterior_base =
        (particle * rotation_count + rotation) * translation_count;
    for (int64_t translation = 0; translation < translation_count; ++translation)
    {
        const float posterior_value = posterior[posterior_base + translation];
        if (posterior_value > 0.0f)
        {
            const float weight = posterior_value * pixel_ctf * pixel_minvsigma2;
            Fweight += weight * pixel_ctf;
        }
    }
    denominator[particle_rotation * pixel_capacity + pixel] = Fweight;
}

cudaError_t launch_relion_vdam_mstep_denominator_f32(
    cudaStream_t stream,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior,
    float* denominator,
    int64_t particle_count,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t logical_pixel_count,
    int64_t pixel_capacity,
    const int32_t* runtime_current_size)
{
    const int64_t denominator_capacity =
        particle_count * rotation_count * pixel_capacity;
    const int64_t launched_denominator_count = runtime_current_size == nullptr
        ? particle_count * rotation_count * logical_pixel_count
        : denominator_capacity;
    if (denominator_capacity == 0) return cudaSuccess;
    cudaError_t err = cudaMemsetAsync(
        denominator,
        0,
        static_cast<size_t>(denominator_capacity) * sizeof(float),
        stream);
    if (err != cudaSuccess || launched_denominator_count == 0) return err;
    constexpr int block_size = 256;
    relion_vdam_denominator_after_sgd_f32_kernel<<<
        static_cast<unsigned int>(
            (launched_denominator_count + block_size - 1) / block_size),
        block_size,
        0,
        stream>>>(
        ctf,
        minvsigma2,
        posterior,
        denominator,
        particle_count,
        rotation_count,
        translation_count,
        logical_pixel_count,
        pixel_capacity,
        runtime_current_size);
    return cudaGetLastError();
}

template <bool INLINE_PROJECTOR>
__global__ void relion_vdam_mstep_fused_x_half_kernel(
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const float2* reference,
    cudaTextureObject_t projector_tex_real,
    cudaTextureObject_t projector_tex_imag,
    const float* projector_eulers,
    const float* rot,
    float* data_real_volume,
    float* data_imag_volume,
    float* weight_volume,
    float* denominator_sum,
    int rotation_count,
    int translation_count,
    int pixel_count,
    int image_h,
    int image_w,
    int N0,
    int N1,
    int N2_eff,
    float c0,
    float c1,
    float c2,
    int upsampling,
    float max_r2,
    int projector_padding_factor,
    int projector_max_r2_padded,
    int projector_tex_y_init,
    int projector_tex_z_init,
    float projector_scale)
{
    const int rotation = (int)blockIdx.x;
    if (rotation >= rotation_count) return;
    /* The inline specialization uses RELION's same nine Euler entries for
     * projection and backprojection. The preprojected specialization keeps
     * the established six-entry compact mapping in the same allocation. */
    __shared__ float R[9];
    if constexpr (INLINE_PROJECTOR)
    {
        if (threadIdx.x < 9)
            R[threadIdx.x] = projector_eulers[rotation * 9 + threadIdx.x];
    }
    else if (threadIdx.x < 6)
    {
        R[threadIdx.x] = rot[rotation * 6 + threadIdx.x];
    }
    __syncthreads();

    for (int pixel = (int)threadIdx.x; pixel < pixel_count; pixel += 128)
    {
        const int x = pixel % image_w;
        const int row = pixel / image_w;
        const int y = row > image_h / 2 ? row - image_h : row;
        const float image_ctf = ctf[pixel];
        const float image_minvsigma2 = minvsigma2[pixel];
        const float2 image_value = images[pixel];
        float2 reference_value;
        if constexpr (INLINE_PROJECTOR)
        {
            reference_value = relion_vdam_project_texture_f32(
                projector_tex_real,
                projector_tex_imag,
                x,
                y,
                R,
                projector_padding_factor,
                projector_max_r2_padded,
                projector_tex_y_init,
                projector_tex_z_init,
                projector_scale);
        }
        else
        {
            reference_value = reference[rotation * pixel_count + pixel];
        }
        const float reference_real = reference_value.x * image_ctf;
        const float reference_imag = reference_value.y * image_ctf;
        float data_real = 0.0f;
        float data_imag = 0.0f;
        float Fweight = 0.0f;
        const int posterior_base = rotation * translation_count;
        for (int translation = 0; translation < translation_count; ++translation)
        {
            float weight = posterior_over_weight_norm[posterior_base + translation];
            weight = weight * image_ctf * image_minvsigma2;
            Fweight += weight * image_ctf;
            const float tx = translation_angles[2 * translation];
            const float ty = translation_angles[2 * translation + 1];
            const float phase = x * tx + y * ty;
            float sine;
            float cosine;
            sincosf(phase, &sine, &cosine);
            const float translated_real = cosine * image_value.x - sine * image_value.y;
            const float translated_imag = cosine * image_value.y + sine * image_value.x;
            data_real += (translated_real - reference_real) * weight;
            data_imag += (translated_imag - reference_imag) * weight;
        }
        denominator_sum[rotation * pixel_count + pixel] = Fweight;
        if (!(Fweight > 0.0f)) continue;

        const float y_unscaled = (float)y;
        const float x_unscaled = (float)x;
        float rk0;
        float rk1;
        float rk2;
        if constexpr (INLINE_PROJECTOR)
        {
            rk0 = (R[6] * x_unscaled + R[7] * y_unscaled) * (float)upsampling;
            rk1 = (R[3] * x_unscaled + R[4] * y_unscaled) * (float)upsampling;
            rk2 = (R[0] * x_unscaled + R[1] * y_unscaled) * (float)upsampling;
        }
        else
        {
            rk0 = (R[3] * x_unscaled + R[0] * y_unscaled) * (float)upsampling;
            rk1 = (R[4] * x_unscaled + R[1] * y_unscaled) * (float)upsampling;
            rk2 = (R[5] * x_unscaled + R[2] * y_unscaled) * (float)upsampling;
        }
        if (max_r2 >= 0.0f && relion_radius_squared(rk0, rk1, rk2) > max_r2) continue;
        if (rk2 < 0.0f)
        {
            rk0 = -rk0;
            rk1 = -rk1;
            rk2 = -rk2;
            data_imag = -data_imag;
        }
        if (max_r2 >= 0.0f)
        {
            const int maxR = (int)floorf(sqrtf(max_r2) + 0.5f);
            if (relion_compact_trilinear_oob<float>(rk2, rk1, rk0, maxR)) continue;
        }
        const int stride1 = N2_eff;
        const int stride0 = N1 * N2_eff;
        scatter_trilinear_relion_fused_x_half<float, float2, false, true, true>(
            nullptr, data_real_volume, data_imag_volume, weight_volume,
            rk0, rk1, rk2, data_real, data_imag, Fweight,
            c0, c1, c2, N0, N1, N2_eff, stride0, stride1,
            0, nullptr, nullptr, nullptr);
    }
}

cudaError_t launch_relion_vdam_mstep_fused_x_half(
    cudaStream_t stream,
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const float2* reference,
    const float* rot,
    float* data_real_volume,
    float* data_imag_volume,
    float* weight_volume,
    float* denominator_sum,
    int64_t n_particles,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4)
{
    const int N2_eff = (int)(N2 / 2 + 1);
    const float c0 = (float)(N0 / 2);
    const float c1 = (float)(N1 / 2);
    const float c2 = (float)(N2 / 2);
    const float max_r2 = (float)max_r2_x4 / 4.0f;
    const int64_t image_stride = pixel_count;
    const int64_t posterior_stride = rotation_count * translation_count;
    const int64_t reference_stride = rotation_count * pixel_count;
    const int64_t rotation_stride = rotation_count * 6;
    const int64_t denominator_stride = rotation_count * pixel_count;
    for (int64_t particle = 0; particle < n_particles; ++particle)
    {
        relion_vdam_mstep_fused_x_half_kernel<false><<<rotation_count, 128, 0, stream>>>(
            images + particle * image_stride,
            ctf + particle * image_stride,
            minvsigma2 + particle * image_stride,
            posterior_over_weight_norm + particle * posterior_stride,
            translation_angles,
            reference + particle * reference_stride,
            0,
            0,
            nullptr,
            rot + particle * rotation_stride,
            data_real_volume,
            data_imag_volume,
            weight_volume,
            denominator_sum + particle * denominator_stride,
            (int)rotation_count,
            (int)translation_count,
            (int)pixel_count,
            (int)image_h,
            (int)image_w,
            (int)N0,
            (int)N1,
            N2_eff,
            c0,
            c1,
            c2,
            (int)upsampling,
            max_r2,
            1,
            0,
            0,
            0,
            1.0f);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

cudaError_t launch_relion_vdam_mstep_fused_projector_x_half(
    cudaStream_t stream,
    const float2* projector_full,
    const float2* images,
    const float* ctf,
    const float* minvsigma2,
    const float* posterior_over_weight_norm,
    const float* translation_angles,
    const float* projector_eulers,
    const float* rot,
    const int32_t* reconstruction_group_ids,
    const int32_t* worker_lane_ids,
    const int32_t* particle_trace_ids,
    const int32_t* rotation_replay_order,
    const int32_t* rotation_replay_counts,
    const int32_t* particle_start_offsets_ns,
    float* data_real_volume,
    float* data_imag_volume,
    float* weight_volume,
    float* denominator_sum,
    int64_t projector_size,
    int64_t n_particles,
    int64_t rotation_count,
    int64_t translation_count,
    int64_t pixel_count,
    int64_t pixel_capacity,
    int64_t image_h,
    int64_t image_w,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    int64_t upsampling,
    int64_t max_r2_x4,
    int physical_image_size,
    int projector_max_r,
    int projection_padding_factor,
    int reconstruction_group_count,
    bool parallel_worker_replay,
    bool captured_rotation_replay,
    int serial_rotation_replay_mode,
    bool float64_accumulator_replay,
    bool reverse_rotation_replay,
    int rotation_replay_stride,
    bool native_trace_shape_replay,
    bool captured_particle_timing_replay,
    bool candidate_trace_active,
    const int32_t* runtime_current_size,
    float* quiesced_prelaunch_data_real,
    float* quiesced_prelaunch_data_imag,
    float* quiesced_prelaunch_weight,
    std::int64_t quiesced_prelaunch_target_particle_id,
    std::int32_t* quiesced_prelaunch_found,
    std::int32_t* quiesced_prelaunch_particle_row,
    std::int32_t* quiesced_prelaunch_worker_lane,
    std::int32_t* quiesced_prelaunch_reconstruction_group,
    const int32_t* runtime_projector_radius = nullptr,
    bool particle_tail_mask = false)
{
    const bool serial_rotation_replay = serial_rotation_replay_mode != 0;
    const bool persistent_serial_rotation_replay =
        serial_rotation_replay_mode == 2;
    int padded_max_r = static_cast<int>(floorf(
        static_cast<float>(projector_max_r * projection_padding_factor) + 0.5f));
    int tex_x = padded_max_r + 2;
    int tex_y = 2 * padded_max_r + 3;
    int tex_z = 2 * padded_max_r + 3;
    int tex_y_init = -(padded_max_r + 1);
    int tex_z_init = -(padded_max_r + 1);
    int64_t texture_voxels = static_cast<int64_t>(tex_x) * tex_y * tex_z;
    float* real = nullptr;
    float* imag = nullptr;
    float* image_real = nullptr;
    float* image_imag = nullptr;
    float* translation_x = nullptr;
    float* translation_y = nullptr;
    float* wavg_dummy_outputs = nullptr;
    float2* preprojected_references = nullptr;
    float* precomputed_residual_weights = nullptr;
    float* ordered_scatter_graph_eulers = nullptr;
    cudaGraph_t* ordered_scatter_graphs = nullptr;
    cudaGraphExec_t* ordered_scatter_graph_execs = nullptr;
    cudaArray_t array_real = nullptr;
    cudaArray_t array_imag = nullptr;
    cudaTextureObject_t texture_real = 0;
    cudaTextureObject_t texture_imag = 0;
    // --pool controls how many particles are read into the outer pool and does
    // not set the shared GUI-default worker-stream count above.
    cudaStream_t particle_streams[kRelionVdamWorkerStreams] = {};
    cudaEvent_t particle_inputs_ready = nullptr;
    int32_t* logical_projector_radius_host = nullptr;
    int32_t* reconstruction_groups_host = nullptr;
    int32_t* worker_lanes_host = nullptr;
    int32_t* particle_trace_ids_host = nullptr;
    int32_t* rotation_replay_order_host = nullptr;
    int32_t* rotation_replay_counts_host = nullptr;
    int32_t* particle_start_offsets_ns_host = nullptr;
    VdamCandidateBlockTraceRecord* candidate_trace_records = nullptr;
    VdamCandidateBlockTraceWriter* candidate_trace_writer =
        &vdam_candidate_block_trace_writer();
    const bool candidate_trace_requested =
        candidate_trace_active && candidate_trace_writer->requested();
    const bool quiesced_prelaunch_capture_requested =
        quiesced_prelaunch_target_particle_id >= 0;
    const bool device_trace_requested =
        candidate_trace_requested || native_trace_shape_replay;
    const char* exact_native_ptx_path =
        std::getenv(kRelionVdamExactNativePtxEnv);
    const bool exact_native_ptx_requested =
        exact_native_ptx_path != nullptr && exact_native_ptx_path[0] != '\0';
    const char* preproject_persistent_value =
        std::getenv(kRelionVdamPreprojectPersistentRotationsEnv);
    const char* precompute_persistent_residuals_value =
        std::getenv(kRelionVdamPrecomputePersistentResidualsEnv);
    const char* precompute_ordered_residuals_value =
        std::getenv(kRelionVdamPrecomputeOrderedResidualsEnv);
    const bool precompute_persistent_residuals_requested =
        precompute_persistent_residuals_value != nullptr &&
        precompute_persistent_residuals_value[0] != '\0' &&
        std::strcmp(precompute_persistent_residuals_value, "0") != 0;
    const bool precompute_ordered_residuals_requested =
        precompute_ordered_residuals_value != nullptr &&
        precompute_ordered_residuals_value[0] != '\0' &&
        std::strcmp(precompute_ordered_residuals_value, "0") != 0;
    const char* fixed_warp_order_scatter_value =
        std::getenv(kRelionVdamFixedWarpOrderScatterEnv);
    const bool fixed_warp_order_scatter_requested =
        fixed_warp_order_scatter_value != nullptr &&
        fixed_warp_order_scatter_value[0] != '\0' &&
        std::strcmp(fixed_warp_order_scatter_value, "0") != 0;
    const char* ordered_scatter_cuda_graph_value =
        std::getenv(kRelionVdamOrderedScatterCudaGraphEnv);
    const bool ordered_scatter_cuda_graph_requested =
        ordered_scatter_cuda_graph_value != nullptr &&
        ordered_scatter_cuda_graph_value[0] != '\0' &&
        std::strcmp(ordered_scatter_cuda_graph_value, "0") != 0;
    const bool precompute_residuals_requested =
        precompute_persistent_residuals_requested ||
        precompute_ordered_residuals_requested;
    const bool preproject_persistent_only_requested =
        preproject_persistent_value != nullptr &&
        preproject_persistent_value[0] != '\0' &&
        std::strcmp(preproject_persistent_value, "0") != 0;
    const bool preproject_persistent_requested =
        precompute_residuals_requested ||
        preproject_persistent_only_requested;
    const char* exact_wavg_predecessor_value =
        std::getenv(kRelionVdamExactWavgPredecessorEnv);
    const bool exact_wavg_predecessor_requested =
        exact_wavg_predecessor_value != nullptr &&
        exact_wavg_predecessor_value[0] != '\0' &&
        std::strcmp(exact_wavg_predecessor_value, "0") != 0;
    const char* runtime_bpref_value =
        std::getenv(kRelionVdamRuntimeBprefWithExactWavgEnv);
    const bool runtime_bpref_with_exact_wavg_requested =
        runtime_bpref_value != nullptr && runtime_bpref_value[0] != '\0' &&
        std::strcmp(runtime_bpref_value, "0") != 0;
    const char* wavg_bpref_host_gap_value =
        std::getenv(kRelionVdamWavgBprefHostGapNsEnv);
    const bool wavg_bpref_host_gap_requested =
        wavg_bpref_host_gap_value != nullptr &&
        wavg_bpref_host_gap_value[0] != '\0';
    long long wavg_bpref_host_gap_ns = 0;
    if (wavg_bpref_host_gap_requested)
    {
        char* gap_end = nullptr;
        errno = 0;
        wavg_bpref_host_gap_ns = std::strtoll(
            wavg_bpref_host_gap_value, &gap_end, 10);
        if (errno != 0 || gap_end == wavg_bpref_host_gap_value ||
            gap_end == nullptr || gap_end[0] != '\0' ||
            wavg_bpref_host_gap_ns < 0)
            return cudaErrorInvalidValue;
    }
    const char* wavg_bpref_host_gap_trace_path =
        std::getenv(kRelionVdamWavgBprefHostGapTraceEnv);
    const bool wavg_bpref_host_gap_trace_requested =
        wavg_bpref_host_gap_trace_path != nullptr &&
        wavg_bpref_host_gap_trace_path[0] != '\0';
    long long wavg_bpref_host_gap_trace_particle = -1;
    if (wavg_bpref_host_gap_trace_requested)
    {
        const char* trace_particle_value =
            std::getenv(kRelionVdamWavgBprefHostGapTraceParticleEnv);
        if (trace_particle_value == nullptr || trace_particle_value[0] == '\0')
            return cudaErrorInvalidValue;
        char* trace_end = nullptr;
        errno = 0;
        wavg_bpref_host_gap_trace_particle = std::strtoll(
            trace_particle_value, &trace_end, 10);
        if (errno != 0 || trace_end == trace_particle_value ||
            trace_end == nullptr || trace_end[0] != '\0' ||
            wavg_bpref_host_gap_trace_particle < 0)
            return cudaErrorInvalidValue;
    }
    CUcontext exact_native_ptx_context = nullptr;
    CUmodule exact_native_ptx_module = nullptr;
    CUfunction exact_native_ptx_kernel = nullptr;
    CUfunction exact_wavg_kernel = nullptr;
    const std::uint32_t candidate_trace_iteration =
        candidate_trace_requested ? candidate_trace_writer->iteration() : 0;
    double* data_real_volume_f64 = nullptr;
    double* data_imag_volume_f64 = nullptr;
    double* weight_volume_f64 = nullptr;
    if ((candidate_trace_requested && !candidate_trace_writer->healthy()) ||
        (candidate_trace_requested && native_trace_shape_replay) ||
        (device_trace_requested && serial_rotation_replay) ||
        (captured_particle_timing_replay && parallel_worker_replay) ||
        (quiesced_prelaunch_capture_requested &&
         (quiesced_prelaunch_data_real == nullptr ||
          quiesced_prelaunch_data_imag == nullptr ||
          quiesced_prelaunch_weight == nullptr ||
          quiesced_prelaunch_found == nullptr ||
          quiesced_prelaunch_particle_row == nullptr ||
          quiesced_prelaunch_worker_lane == nullptr ||
          quiesced_prelaunch_reconstruction_group == nullptr)))
        return cudaErrorInvalidValue;
    // The extracted native entry has RELION's ordinary Ref3D ABI.  Keep the
    // discriminator fail-closed instead of silently mixing it with RECOVAR's
    // diagnostic-only remapping, tracing, or double-accumulator variants.
    if (exact_native_ptx_requested &&
        (captured_rotation_replay || serial_rotation_replay ||
         float64_accumulator_replay || device_trace_requested ||
         reverse_rotation_replay || rotation_replay_stride > 0))
        return cudaErrorInvalidValue;
    // Extracted RELION PTX receives logical image geometry through host ABI
    // values.  A device-only runtime cutoff cannot safely rewrite those
    // arguments, so keep that diagnostic route explicitly unsupported.
    if (runtime_current_size != nullptr &&
        (exact_native_ptx_requested || exact_wavg_predecessor_requested))
        return cudaErrorInvalidValue;
    if (exact_wavg_predecessor_requested && !exact_native_ptx_requested)
        return cudaErrorInvalidValue;
    if (runtime_bpref_with_exact_wavg_requested &&
        !exact_wavg_predecessor_requested)
        return cudaErrorInvalidValue;
    if (wavg_bpref_host_gap_requested && !exact_wavg_predecessor_requested)
        return cudaErrorInvalidValue;
    if (wavg_bpref_host_gap_trace_requested &&
        !exact_wavg_predecessor_requested)
        return cudaErrorInvalidValue;
    if (preproject_persistent_only_requested &&
        !persistent_serial_rotation_replay)
        return cudaErrorInvalidValue;
    if (precompute_persistent_residuals_requested &&
        !persistent_serial_rotation_replay)
        return cudaErrorInvalidValue;
    if (precompute_ordered_residuals_requested &&
        (!serial_rotation_replay || persistent_serial_rotation_replay))
        return cudaErrorInvalidValue;
    if (fixed_warp_order_scatter_requested &&
        (!precompute_ordered_residuals_requested ||
         persistent_serial_rotation_replay || parallel_worker_replay ||
         exact_native_ptx_requested || device_trace_requested))
        return cudaErrorInvalidValue;
    // CUDA Graph replay is an optimization of one qualified source topology,
    // not a new replay mode.  Keep it fail-closed so every captured node is the
    // existing one-block fixed-warp scatter, in the existing padded row order.
    if (ordered_scatter_cuda_graph_requested &&
        (!serial_rotation_replay || persistent_serial_rotation_replay ||
         !precompute_ordered_residuals_requested ||
         !fixed_warp_order_scatter_requested || parallel_worker_replay ||
         captured_rotation_replay || float64_accumulator_replay ||
         reverse_rotation_replay || rotation_replay_stride != 0 ||
         device_trace_requested || captured_particle_timing_replay ||
         quiesced_prelaunch_capture_requested || exact_native_ptx_requested ||
         exact_wavg_predecessor_requested ||
         runtime_bpref_with_exact_wavg_requested ||
         wavg_bpref_host_gap_requested ||
         wavg_bpref_host_gap_trace_requested))
        return cudaErrorInvalidValue;
    if (preproject_persistent_requested &&
        (parallel_worker_replay || captured_rotation_replay ||
         reverse_rotation_replay || rotation_replay_stride > 0))
        return cudaErrorInvalidValue;
    // The capacity route retains the native logical texture extent/clamp and
    // 64-byte projector struct. Read its independent radius with the same
    // group/lane/count metadata synchronization, earlier only on this route.
    // No particle, rotation, translation or scatter kernel is launched here.
    const bool capacity_projector = runtime_projector_radius != nullptr;
    if (capacity_projector &&
        (runtime_current_size == nullptr || projector_max_r != 0 ||
         captured_rotation_replay || serial_rotation_replay ||
         float64_accumulator_replay || reverse_rotation_replay ||
         rotation_replay_stride != 0 || native_trace_shape_replay ||
         captured_particle_timing_replay || candidate_trace_active ||
         quiesced_prelaunch_capture_requested || exact_native_ptx_requested ||
         exact_wavg_predecessor_requested || preproject_persistent_requested ||
         fixed_warp_order_scatter_requested || ordered_scatter_cuda_graph_requested ||
         runtime_bpref_with_exact_wavg_requested || wavg_bpref_host_gap_requested ||
         wavg_bpref_host_gap_trace_requested))
        return cudaErrorInvalidValue;
    if (particle_tail_mask &&
        (denominator_sum != nullptr || reconstruction_group_count <= 1 ||
         parallel_worker_replay || capacity_projector ||
         captured_rotation_replay || serial_rotation_replay ||
         float64_accumulator_replay || reverse_rotation_replay ||
         rotation_replay_stride != 0 || native_trace_shape_replay ||
         captured_particle_timing_replay || candidate_trace_active ||
         quiesced_prelaunch_capture_requested || exact_native_ptx_requested ||
         exact_wavg_predecessor_requested || preproject_persistent_requested ||
         fixed_warp_order_scatter_requested || ordered_scatter_cuda_graph_requested ||
         runtime_bpref_with_exact_wavg_requested || wavg_bpref_host_gap_requested ||
         wavg_bpref_host_gap_trace_requested))
        return cudaErrorInvalidValue;
    cudaError_t err = cudaSuccess;
    if (capacity_projector)
    {
        err = cudaMallocHost(reinterpret_cast<void**>(&logical_projector_radius_host),
                             sizeof(int32_t));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMallocHost(reinterpret_cast<void**>(&reconstruction_groups_host),
                             static_cast<size_t>(n_particles) * sizeof(int32_t));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMallocHost(reinterpret_cast<void**>(&worker_lanes_host),
                             static_cast<size_t>(n_particles) * sizeof(int32_t));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMallocHost(reinterpret_cast<void**>(&rotation_replay_counts_host),
                             static_cast<size_t>(n_particles) * sizeof(int32_t));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(logical_projector_radius_host, runtime_projector_radius,
                              sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(reconstruction_groups_host, reconstruction_group_ids,
                              static_cast<size_t>(n_particles) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(worker_lanes_host, worker_lane_ids,
                              static_cast<size_t>(n_particles) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(rotation_replay_counts_host, rotation_replay_counts,
                              static_cast<size_t>(n_particles) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;
        projector_max_r = *logical_projector_radius_host;
        if (projector_max_r <= 0 ||
            projector_max_r > (projector_size - 3) / (2 * projection_padding_factor))
        {
            err = cudaErrorInvalidValue;
            goto cleanup;
        }
        padded_max_r = static_cast<int>(floorf(
            static_cast<float>(projector_max_r * projection_padding_factor) + 0.5f));
        tex_x = padded_max_r + 2;
        tex_y = tex_z = 2 * padded_max_r + 3;
        tex_y_init = tex_z_init = -(padded_max_r + 1);
        texture_voxels = static_cast<int64_t>(tex_x) * tex_y * tex_z;
    }
    err = cudaMalloc(
        reinterpret_cast<void**>(&real),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(
        reinterpret_cast<void**>(&imag),
        static_cast<size_t>(texture_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    if (capacity_projector)
    {
        fill_relion_texture_capacity_kernel<<<
            static_cast<unsigned int>((texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE, 0, stream>>>(
                projector_full, real, imag, runtime_projector_radius,
                projection_padding_factor, tex_x, tex_y, tex_z,
                static_cast<int>(projector_size / 2 + 1),
                static_cast<int>(projector_size), static_cast<int>(projector_size));
    }
    else
    {
        fill_relion_texture_compact_kernel<float><<<
            static_cast<unsigned int>((texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE,
            0,
            stream>>>(
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
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    {
        const float projector_scale = -static_cast<float>(
            physical_image_size * physical_image_size);
        relion_vdam_scale_texture_f32_kernel<<<
            static_cast<unsigned int>((texture_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE,
            0,
            stream>>>(real, imag, texture_voxels, projector_scale);
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

    {
        const float max_r2 = static_cast<float>(max_r2_x4) / 4.0f;
        const int projector_max_r2_padded = padded_max_r * padded_max_r;
        if (pixel_capacity < pixel_count)
        {
            err = cudaErrorInvalidValue;
            goto cleanup;
        }
        const int64_t image_stride = pixel_capacity;
        const int64_t posterior_stride = rotation_count * translation_count;
        const int64_t euler_stride = rotation_count * 9;
        const int64_t image_value_count = n_particles * pixel_capacity;
        const int64_t accumulator_stride = N0 * N1 * (N2 / 2 + 1);
        const int64_t accumulator_count =
            static_cast<int64_t>(reconstruction_group_count) * accumulator_stride;
        const int model_x = static_cast<int>(N2 / 2 + 1);
        const int model_y = static_cast<int>(N1);
        const int model_init_y = -static_cast<int>(N1 / 2);
        const int model_init_z = -static_cast<int>(N0 / 2);
        const float significant_weight = std::numeric_limits<float>::min();
        const float weight_norm = 1.0f;

        err = cudaMalloc(
            reinterpret_cast<void**>(&image_real),
            static_cast<size_t>(image_value_count) * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(
            reinterpret_cast<void**>(&image_imag),
            static_cast<size_t>(image_value_count) * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(
            reinterpret_cast<void**>(&translation_x),
            static_cast<size_t>(translation_count) * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(
            reinterpret_cast<void**>(&translation_y),
            static_cast<size_t>(translation_count) * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
        if (preproject_persistent_requested)
        {
            const size_t reference_count =
                static_cast<size_t>(rotation_count) *
                static_cast<size_t>(pixel_count);
            if (rotation_count > 0 &&
                reference_count / static_cast<size_t>(rotation_count) !=
                    static_cast<size_t>(pixel_count))
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            err = cudaMalloc(
                reinterpret_cast<void**>(&preprojected_references),
                reference_count * sizeof(float2));
            if (err != cudaSuccess) goto cleanup;
            if (precompute_residuals_requested)
            {
                err = cudaMalloc(
                    reinterpret_cast<void**>(&precomputed_residual_weights),
                    reference_count * sizeof(float));
                if (err != cudaSuccess) goto cleanup;
            }
        }
        if (ordered_scatter_cuda_graph_requested)
        {
            const size_t euler_value_count =
                static_cast<size_t>(rotation_count) * 9;
            if (rotation_count > 0 &&
                euler_value_count / static_cast<size_t>(rotation_count) != 9)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            err = cudaMalloc(
                reinterpret_cast<void**>(&ordered_scatter_graph_eulers),
                euler_value_count * sizeof(float));
            if (err != cudaSuccess) goto cleanup;
            ordered_scatter_graphs = static_cast<cudaGraph_t*>(std::calloc(
                static_cast<size_t>(reconstruction_group_count),
                sizeof(cudaGraph_t)));
            ordered_scatter_graph_execs = static_cast<cudaGraphExec_t*>(std::calloc(
                static_cast<size_t>(reconstruction_group_count),
                sizeof(cudaGraphExec_t)));
            if (ordered_scatter_graphs == nullptr ||
                ordered_scatter_graph_execs == nullptr)
            {
                err = cudaErrorMemoryAllocation;
                goto cleanup;
            }
        }
        if (exact_wavg_predecessor_requested)
        {
            // RELION's wdiff2 arrays are worker-local and reused by successive
            // particles.  Preserve that ownership topology while discarding
            // the diagnostic predecessor's numerical outputs.
            constexpr int kWavgOutputCount = 3;
            const size_t wavg_output_count = static_cast<size_t>(
                kRelionVdamWorkerStreams) * kWavgOutputCount * pixel_count;
            err = cudaMalloc(
                reinterpret_cast<void**>(&wavg_dummy_outputs),
                wavg_output_count * sizeof(float));
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(
                wavg_dummy_outputs, 0, wavg_output_count * sizeof(float), stream);
            if (err != cudaSuccess) goto cleanup;
        }

        split_complex_float_kernel<<<
            static_cast<unsigned int>((image_value_count + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE,
            0,
            stream>>>(
                reinterpret_cast<const float*>(images),
                image_real,
                image_imag,
                static_cast<int>(image_value_count));
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        relion_vdam_split_translations_f32_kernel<<<
            static_cast<unsigned int>((translation_count + BLOCK_SIZE - 1) / BLOCK_SIZE),
            BLOCK_SIZE,
            0,
            stream>>>(translation_angles, translation_x, translation_y, translation_count);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        if (!capacity_projector)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&reconstruction_groups_host),
                static_cast<size_t>(n_particles) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (float64_accumulator_replay)
        {
            const size_t accumulator_bytes =
                static_cast<size_t>(accumulator_count) * sizeof(double);
            err = cudaMalloc(reinterpret_cast<void**>(&data_real_volume_f64), accumulator_bytes);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMalloc(reinterpret_cast<void**>(&data_imag_volume_f64), accumulator_bytes);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMalloc(reinterpret_cast<void**>(&weight_volume_f64), accumulator_bytes);
            if (err != cudaSuccess) goto cleanup;
            const unsigned int cast_blocks = static_cast<unsigned int>(
                (accumulator_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
            relion_vdam_cast_accumulator_kernel<double, float><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                data_real_volume, data_real_volume_f64, accumulator_count);
            relion_vdam_cast_accumulator_kernel<double, float><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                data_imag_volume, data_imag_volume_f64, accumulator_count);
            relion_vdam_cast_accumulator_kernel<double, float><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                weight_volume, weight_volume_f64, accumulator_count);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&worker_lanes_host),
                static_cast<size_t>(n_particles) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (candidate_trace_requested || wavg_bpref_host_gap_trace_requested ||
            quiesced_prelaunch_capture_requested)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&particle_trace_ids_host),
                static_cast<size_t>(n_particles) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (device_trace_requested)
        {
            err = cudaMalloc(
                reinterpret_cast<void**>(&candidate_trace_records),
                static_cast<size_t>(n_particles * rotation_count) *
                    sizeof(VdamCandidateBlockTraceRecord));
            if (err != cudaSuccess) goto cleanup;
        }
        if (captured_rotation_replay)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&rotation_replay_order_host),
                static_cast<size_t>(n_particles * rotation_count) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&rotation_replay_counts_host),
                static_cast<size_t>(n_particles) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (captured_particle_timing_replay)
        {
            err = cudaMallocHost(
                reinterpret_cast<void**>(&particle_start_offsets_ns_host),
                static_cast<size_t>(n_particles) * sizeof(int32_t));
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaMemcpyAsync(
                reconstruction_groups_host,
                reconstruction_group_ids,
                static_cast<size_t>(n_particles) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaMemcpyAsync(
                worker_lanes_host,
                worker_lane_ids,
                static_cast<size_t>(n_particles) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (candidate_trace_requested || wavg_bpref_host_gap_trace_requested ||
            quiesced_prelaunch_capture_requested)
        {
            err = cudaMemcpyAsync(
                particle_trace_ids_host,
                particle_trace_ids,
                static_cast<size_t>(n_particles) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (captured_rotation_replay)
        {
            err = cudaMemcpyAsync(
                rotation_replay_order_host,
                rotation_replay_order,
                static_cast<size_t>(n_particles * rotation_count) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaMemcpyAsync(
                rotation_replay_counts_host,
                rotation_replay_counts,
                static_cast<size_t>(n_particles) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (captured_particle_timing_replay)
        {
            err = cudaMemcpyAsync(
                particle_start_offsets_ns_host,
                particle_start_offsets_ns,
                static_cast<size_t>(n_particles) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream);
            if (err != cudaSuccess) goto cleanup;
        }
        if (!capacity_projector)
        {
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) goto cleanup;
        }
        // The existing metadata copy covers physical capacity. Only a trailing
        // -1 group suffix is padding; the original loop below validates every
        // active row, rejecting holes and other negative/out-of-range IDs.
        // No padded row reaches worker scheduling or scientific scatter.
        if (particle_tail_mask)
        {
            while (n_particles > 0 &&
                   reconstruction_groups_host[n_particles - 1] == -1)
                --n_particles;
            if (n_particles == 0)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
        }
        const int32_t preproject_worker_lane =
            n_particles > 0 ? worker_lanes_host[0] : -1;
        for (int64_t particle = 0; particle < n_particles; ++particle)
        {
            if (reconstruction_groups_host[particle] < 0 ||
                reconstruction_groups_host[particle] >= reconstruction_group_count)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            if (worker_lanes_host[particle] < 0 ||
                worker_lanes_host[particle] >= kRelionVdamWorkerStreams)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            // A single preprojection buffer is reused particle by particle.
            // Different worker streams could overwrite it while an earlier
            // persistent scatter still reads it, so this optimization only
            // accepts the single-lane serial-particle contract.
            if (preproject_persistent_requested &&
                worker_lanes_host[particle] != preproject_worker_lane)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            if ((candidate_trace_requested ||
                 wavg_bpref_host_gap_trace_requested ||
                 quiesced_prelaunch_capture_requested) &&
                particle_trace_ids_host[particle] < 0)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            if (rotation_replay_counts_host[particle] <= 0 ||
                rotation_replay_counts_host[particle] > rotation_count)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            if (captured_particle_timing_replay &&
                particle_start_offsets_ns_host[particle] < 0)
            {
                err = cudaErrorInvalidValue;
                goto cleanup;
            }
            if (captured_rotation_replay)
            {
                std::vector<unsigned char> seen(static_cast<size_t>(rotation_count), 0);
                for (int64_t launch = 0; launch < rotation_count; ++launch)
                {
                    const int32_t rotation = rotation_replay_order_host[
                        particle * rotation_count + launch];
                    if (rotation < 0 || rotation >= rotation_count || seen[rotation])
                    {
                        err = cudaErrorInvalidValue;
                        goto cleanup;
                    }
                    seen[rotation] = 1;
                }
            }
        }

        if (exact_native_ptx_requested)
        {
            CUresult driver_result = cuCtxGetCurrent(&exact_native_ptx_context);
            if (driver_result != CUDA_SUCCESS || exact_native_ptx_context == nullptr)
            {
                err = report_relion_vdam_driver_error(
                    "cuCtxGetCurrent", driver_result);
                goto cleanup;
            }
            driver_result = cuModuleLoad(
                &exact_native_ptx_module, exact_native_ptx_path);
            if (driver_result != CUDA_SUCCESS)
            {
                err = report_relion_vdam_driver_error(
                    "cuModuleLoad", driver_result);
                goto cleanup;
            }
            driver_result = cuModuleGetFunction(
                &exact_native_ptx_kernel,
                exact_native_ptx_module,
                kRelionVdamExactNativePtxKernel);
            if (driver_result != CUDA_SUCCESS)
            {
                err = report_relion_vdam_driver_error(
                    "cuModuleGetFunction", driver_result);
                goto cleanup;
            }
            if (exact_wavg_predecessor_requested)
            {
                driver_result = cuModuleGetFunction(
                    &exact_wavg_kernel,
                    exact_native_ptx_module,
                    kRelionVdamExactWavgKernel);
                if (driver_result != CUDA_SUCCESS)
                {
                    err = report_relion_vdam_driver_error(
                        "cuModuleGetFunction(wavg)", driver_result);
                    goto cleanup;
                }
            }
        }

        // RELION's task distributor hands one particle at a time to each of
        // its --j workers.  Each worker launches into its own blocking class
        // stream and synchronizes that stream before requesting another task.
        // Keep setup on XLA's stream, then reproduce the one-in-flight task per
        // worker topology on ordinary (blocking) CUDA streams.
        err = initialize_relion_vdam_worker_streams(
            stream, particle_streams, &particle_inputs_ready);
        if (err != cudaSuccess) goto cleanup;

        RelionVdamProjectorKernel projector{
            tex_x,
            tex_x * tex_y,
            tex_z,
            static_cast<int>(image_w),
            static_cast<int>(image_h),
            1,
            tex_y_init,
            tex_z_init,
            projector_max_r,
            projector_max_r * projector_max_r,
            projector_max_r2_padded,
            static_cast<float>(projection_padding_factor),
            texture_real,
            texture_imag,
        };
        std::vector<long long> wavg_bpref_intrinsic_gap_ns(
            static_cast<size_t>(n_particles), -1);
        std::vector<long long> wavg_bpref_effective_gap_ns(
            static_cast<size_t>(n_particles), -1);
        std::vector<long long> wavg_host_enqueue_ns(
            static_cast<size_t>(n_particles), -1);
        std::vector<long long> bpref_host_enqueue_ns(
            static_cast<size_t>(n_particles), -1);
        std::vector<long long> wavg_to_bpref_return_ns(
            static_cast<size_t>(n_particles), -1);
        int quiesced_prelaunch_capture_count = 0;
        std::shared_mutex quiesced_prelaunch_launch_gate;
        const auto launch_particle_with_accumulators = [&](
            int64_t particle,
            int lane,
            auto* accumulator_real,
            auto* accumulator_imag,
            auto* accumulator_weight) {
            using Accumulator =
                std::remove_pointer_t<decltype(accumulator_real)>;
            const int64_t accumulator_offset =
                static_cast<int64_t>(reconstruction_groups_host[particle]) *
                accumulator_stride;
            std::uint64_t trace_launch_sequence = 0;
            const int64_t particle_rotation_count =
                rotation_replay_counts_host[particle];
            std::chrono::steady_clock::time_point exact_wavg_return_time;
            const bool trace_this_gap_particle =
                wavg_bpref_host_gap_trace_requested &&
                particle_trace_ids_host[particle] ==
                    wavg_bpref_host_gap_trace_particle;
            if (candidate_trace_requested)
            {
                if (!candidate_trace_writer->reserve(
                        static_cast<std::uint64_t>(particle_rotation_count),
                        &trace_launch_sequence))
                    return cudaErrorInvalidValue;
                const cudaError_t clear_error = cudaMemsetAsync(
                    candidate_trace_records + particle * rotation_count,
                    0,
                    static_cast<size_t>(particle_rotation_count) *
                        sizeof(VdamCandidateBlockTraceRecord),
                    particle_streams[lane]);
                if (clear_error != cudaSuccess) return clear_error;
            }
            if (exact_wavg_predecessor_requested)
            {
                // RELION queues Wavg and BPref consecutively on the same class
                // stream.  Its Wavg outputs are irrelevant to BPref, but the
                // predecessor changes the device scheduler state seen by the
                // atomic backprojection.  Launch the exact embedded RELION
                // Ref3D/Data2D/CTF-corrected entry with the same particle
                // operands and worker-local scratch, without an intervening
                // synchronization.
                CUresult driver_result =
                    cuCtxSetCurrent(exact_native_ptx_context);
                if (driver_result != CUDA_SUCCESS)
                    return report_relion_vdam_driver_error(
                        "cuCtxSetCurrent(wavg)", driver_result);
                float* eulers_arg = const_cast<float*>(
                    projector_eulers + particle * euler_stride);
                unsigned image_size_arg = static_cast<unsigned>(pixel_count);
                unsigned long orientation_count_arg =
                    static_cast<unsigned long>(particle_rotation_count);
                float* image_real_arg = image_real + particle * image_stride;
                float* image_imag_arg = image_imag + particle * image_stride;
                float* translation_x_arg = translation_x;
                float* translation_y_arg = translation_y;
                float* translation_z_arg = nullptr;
                float* weights_arg = const_cast<float*>(
                    posterior_over_weight_norm + particle * posterior_stride);
                float* ctf_arg = const_cast<float*>(
                    ctf + particle * image_stride);
                const int64_t lane_output_offset =
                    static_cast<int64_t>(lane) * 3 * pixel_count;
                float* wdiff2_parts_arg =
                    wavg_dummy_outputs + lane_output_offset;
                float* wdiff2_aa_arg = wdiff2_parts_arg + pixel_count;
                float* wdiff2_xa_arg = wdiff2_aa_arg + pixel_count;
                unsigned long translation_count_arg =
                    static_cast<unsigned long>(translation_count);
                float weight_norm_arg = weight_norm;
                float significant_weight_arg = significant_weight;
                float part_scale_arg = 1.0f;
                void* wavg_parameters[] = {
                    &eulers_arg,
                    &projector,
                    &image_size_arg,
                    &orientation_count_arg,
                    &image_real_arg,
                    &image_imag_arg,
                    &translation_x_arg,
                    &translation_y_arg,
                    &translation_z_arg,
                    &weights_arg,
                    &ctf_arg,
                    &wdiff2_parts_arg,
                    &wdiff2_aa_arg,
                    &wdiff2_xa_arg,
                    &translation_count_arg,
                    &weight_norm_arg,
                    &significant_weight_arg,
                    &part_scale_arg,
                };
                constexpr unsigned kWavgBlockSize = 256;
                constexpr unsigned kWavgSharedBytes =
                    (3 * kWavgBlockSize + 9) * sizeof(float);
                std::chrono::steady_clock::time_point wavg_enqueue_start;
                if (trace_this_gap_particle)
                    wavg_enqueue_start = std::chrono::steady_clock::now();
                driver_result = cuLaunchKernel(
                    exact_wavg_kernel,
                    static_cast<unsigned>(particle_rotation_count), 1, 1,
                    kWavgBlockSize, 1, 1,
                    kWavgSharedBytes,
                    reinterpret_cast<CUstream>(particle_streams[lane]),
                    wavg_parameters,
                    nullptr);
                if (driver_result != CUDA_SUCCESS)
                    return report_relion_vdam_driver_error(
                        "cuLaunchKernel(wavg)", driver_result);
                if (wavg_bpref_host_gap_requested ||
                    wavg_bpref_host_gap_trace_requested)
                    exact_wavg_return_time = std::chrono::steady_clock::now();
                if (trace_this_gap_particle)
                    wavg_host_enqueue_ns[particle] =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            exact_wavg_return_time - wavg_enqueue_start).count();
            }
            if (quiesced_prelaunch_capture_requested &&
                particle_trace_ids_host[particle] ==
                    quiesced_prelaunch_target_particle_id)
            {
                if constexpr (!std::is_same_v<Accumulator, float>)
                {
                    return cudaErrorInvalidValue;
                }
                else
                {
                    if (quiesced_prelaunch_capture_count != 0)
                        return cudaErrorInvalidValue;
                    // RELION copies a worker-private accumulator after
                    // synchronizing that worker's class stream. RECOVAR's
                    // current accumulator is shared across worker streams, so
                    // a lane-only copy would race with other workers. Quiesce
                    // all streams and label the diagnostic explicitly.
                    cudaError_t capture_error = cudaDeviceSynchronize();
                    if (capture_error != cudaSuccess) return capture_error;
                    const size_t capture_bytes =
                        static_cast<size_t>(accumulator_stride) * sizeof(float);
                    capture_error = cudaMemcpy(
                        quiesced_prelaunch_data_real,
                        accumulator_real + accumulator_offset,
                        capture_bytes,
                        cudaMemcpyDeviceToHost);
                    if (capture_error != cudaSuccess) return capture_error;
                    capture_error = cudaMemcpy(
                        quiesced_prelaunch_data_imag,
                        accumulator_imag + accumulator_offset,
                        capture_bytes,
                        cudaMemcpyDeviceToHost);
                    if (capture_error != cudaSuccess) return capture_error;
                    capture_error = cudaMemcpy(
                        quiesced_prelaunch_weight,
                        accumulator_weight + accumulator_offset,
                        capture_bytes,
                        cudaMemcpyDeviceToHost);
                    if (capture_error != cudaSuccess) return capture_error;
                    *quiesced_prelaunch_found = 1;
                    *quiesced_prelaunch_particle_row =
                        static_cast<std::int32_t>(particle);
                    *quiesced_prelaunch_worker_lane =
                        static_cast<std::int32_t>(lane);
                    *quiesced_prelaunch_reconstruction_group =
                        reconstruction_groups_host[particle];
                    ++quiesced_prelaunch_capture_count;
                }
            }
            const auto launch_precomputed_fixed_warp_scatter = [&](
                const float* scatter_eulers,
                int64_t rotation_offset,
                unsigned scatter_rotation_count) -> cudaError_t {
                const int64_t ordered_operand_offset =
                    rotation_offset * static_cast<int64_t>(pixel_count);
                relion_vdam_native_sgd_f32_kernel<
                    Accumulator,
                    false,
                    false,
                    false,
                    true><<<
                    1, 128, 0, particle_streams[lane]>>>(
                    projector,
                    nullptr,
                    preprojected_references + ordered_operand_offset,
                    precomputed_residual_weights + ordered_operand_offset,
                    image_real + particle * image_stride,
                    image_imag + particle * image_stride,
                    translation_x,
                    translation_y,
                    nullptr,
                    const_cast<float*>(
                        posterior_over_weight_norm + particle * posterior_stride +
                        rotation_offset * translation_count),
                    const_cast<float*>(minvsigma2 + particle * image_stride),
                    const_cast<float*>(ctf + particle * image_stride),
                    static_cast<unsigned long>(translation_count),
                    significant_weight,
                    weight_norm,
                    const_cast<float*>(scatter_eulers + rotation_offset * 9),
                    accumulator_real + accumulator_offset,
                    accumulator_imag + accumulator_offset,
                    accumulator_weight + accumulator_offset,
                    static_cast<int>(sqrtf(max_r2) + 0.5f),
                    static_cast<int>(max_r2),
                    static_cast<float>(upsampling),
                    static_cast<unsigned>(image_w),
                    static_cast<unsigned>(image_h),
                    1,
                    static_cast<unsigned>(pixel_count),
                    static_cast<unsigned>(model_x),
                    static_cast<unsigned>(model_y),
                    model_init_y,
                    model_init_z,
                    scatter_rotation_count,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    static_cast<std::int32_t>(lane),
                    0,
                    runtime_current_size);
                return cudaGetLastError();
            };
            if (ordered_scatter_cuda_graph_requested)
            {
                // The graph keeps every ordinary launch boundary as a separate
                // node.  Only particle-varying Euler bytes are staged; ordered
                // residual buffers and the selected group's accumulators are
                // already stable for the lifetime of this callback.
                cudaError_t graph_error = cudaMemcpyAsync(
                    ordered_scatter_graph_eulers,
                    projector_eulers + particle * euler_stride,
                    static_cast<size_t>(euler_stride) * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    particle_streams[lane]);
                if (graph_error != cudaSuccess) return graph_error;

                relion_vdam_native_project_f32_kernel<<<
                    rotation_count,
                    128,
                    0,
                    particle_streams[lane]>>>(
                        projector,
                        ordered_scatter_graph_eulers,
                        preprojected_references,
                        static_cast<unsigned>(image_w),
                        static_cast<unsigned>(image_h),
                        static_cast<unsigned>(pixel_count),
                        static_cast<unsigned>(rotation_count),
                        runtime_current_size);
                graph_error = cudaGetLastError();
                if (graph_error != cudaSuccess) return graph_error;
                relion_vdam_native_residual_f32_kernel<<<
                    rotation_count,
                    128,
                    0,
                    particle_streams[lane]>>>(
                        preprojected_references,
                        precomputed_residual_weights,
                        image_real + particle * image_stride,
                        image_imag + particle * image_stride,
                        translation_x,
                        translation_y,
                        const_cast<float*>(
                            posterior_over_weight_norm +
                            particle * posterior_stride),
                        const_cast<float*>(minvsigma2 + particle * image_stride),
                        const_cast<float*>(ctf + particle * image_stride),
                        static_cast<unsigned long>(translation_count),
                        significant_weight,
                        weight_norm,
                        static_cast<unsigned>(image_w),
                        static_cast<unsigned>(image_h),
                        static_cast<unsigned>(pixel_count),
                        static_cast<unsigned>(rotation_count),
                        runtime_current_size);
                graph_error = cudaGetLastError();
                if (graph_error != cudaSuccess) return graph_error;

                const int reconstruction_group =
                    reconstruction_groups_host[particle];
                if (ordered_scatter_graph_execs[reconstruction_group] == nullptr)
                {
                    graph_error = cudaStreamBeginCapture(
                        particle_streams[lane],
                        cudaStreamCaptureModeThreadLocal);
                    if (graph_error != cudaSuccess) return graph_error;

                    cudaError_t captured_launch_error = cudaSuccess;
                    for (int64_t rotation_offset = 0;
                         rotation_offset < rotation_count;
                         ++rotation_offset)
                    {
                        // This kernel neither traces nor loops over the count.
                        // Use padded R so the graph is reusable for unequal
                        // per-particle valid-row masks.
                        captured_launch_error =
                            launch_precomputed_fixed_warp_scatter(
                                ordered_scatter_graph_eulers,
                                rotation_offset,
                                static_cast<unsigned>(rotation_count));
                        if (captured_launch_error != cudaSuccess) break;
                    }

                    cudaGraph_t captured_graph = nullptr;
                    const cudaError_t end_capture_error = cudaStreamEndCapture(
                        particle_streams[lane], &captured_graph);
                    if (captured_launch_error != cudaSuccess)
                    {
                        if (captured_graph != nullptr)
                            cudaGraphDestroy(captured_graph);
                        return captured_launch_error;
                    }
                    if (end_capture_error != cudaSuccess)
                    {
                        if (captured_graph != nullptr)
                            cudaGraphDestroy(captured_graph);
                        return end_capture_error;
                    }
                    size_t captured_node_count = 0;
                    graph_error = cudaGraphGetNodes(
                        captured_graph, nullptr, &captured_node_count);
                    if (graph_error != cudaSuccess ||
                        captured_node_count != static_cast<size_t>(rotation_count))
                    {
                        cudaGraphDestroy(captured_graph);
                        return graph_error != cudaSuccess
                            ? graph_error
                            : cudaErrorInvalidValue;
                    }
                    ordered_scatter_graphs[reconstruction_group] = captured_graph;
                    graph_error = cudaGraphInstantiate(
                        &ordered_scatter_graph_execs[reconstruction_group],
                        captured_graph,
                        nullptr,
                        nullptr,
                        0);
                    if (graph_error != cudaSuccess) return graph_error;
                }
                return cudaGraphLaunch(
                    ordered_scatter_graph_execs[reconstruction_group],
                    particle_streams[lane]);
            }
            const int64_t launch_count =
                serial_rotation_replay && !persistent_serial_rotation_replay
                    ? rotation_count
                    : 1;
            for (int64_t launch = 0; launch < launch_count; ++launch)
            {
                int64_t rotation_offset =
                    serial_rotation_replay && !persistent_serial_rotation_replay
                    ? (reverse_rotation_replay ? rotation_count - 1 - launch : launch)
                    : 0;
                if (captured_rotation_replay && serial_rotation_replay)
                    rotation_offset = rotation_replay_order_host[
                        particle * rotation_count + launch];
                if (serial_rotation_replay && rotation_replay_stride > 0)
                {
                    // Approximate a native fixed-SM work queue: each logical
                    // SM consumes block indices sm, sm + stride, ... before
                    // the next logical SM is serialized.  The mapping covers
                    // every rotation exactly once even for a partial tail.
                    const int64_t stride = std::min<int64_t>(
                        rotation_replay_stride, rotation_count);
                    const int64_t short_count = rotation_count / stride;
                    const int64_t long_lanes = rotation_count % stride;
                    const int64_t long_launch_count =
                        long_lanes * (short_count + 1);
                    int64_t logical_lane;
                    int64_t lane_wave;
                    if (launch < long_launch_count)
                    {
                        logical_lane = launch / (short_count + 1);
                        lane_wave = launch % (short_count + 1);
                    }
                    else
                    {
                        const int64_t short_launch = launch - long_launch_count;
                        logical_lane = long_lanes + short_launch / short_count;
                        lane_wave = short_launch % short_count;
                    }
                    rotation_offset = logical_lane + lane_wave * stride;
                }
                const int64_t grid_rotations = serial_rotation_replay
                    ? 1
                    : particle_rotation_count;
            const auto launch_sgd = [&](auto captured_order_tag, auto trace_tag) {
                    constexpr bool use_captured_order =
                        decltype(captured_order_tag)::value;
                    constexpr bool use_trace = decltype(trace_tag)::value;
                    if (exact_native_ptx_requested &&
                        !runtime_bpref_with_exact_wavg_requested)
                    {
                        if constexpr (!std::is_same_v<Accumulator, float>)
                        {
                            return cudaErrorInvalidValue;
                        }
                        else
                        {
                            CUresult driver_result =
                                cuCtxSetCurrent(exact_native_ptx_context);
                            if (driver_result != CUDA_SUCCESS)
                                return report_relion_vdam_driver_error(
                                    "cuCtxSetCurrent", driver_result);

                            float* image_real_arg =
                                image_real + particle * image_stride;
                            float* image_imag_arg =
                                image_imag + particle * image_stride;
                            float* translation_x_arg = translation_x;
                            float* translation_y_arg = translation_y;
                            float* translation_z_arg = nullptr;
                            float* weights_arg = const_cast<float*>(
                                posterior_over_weight_norm +
                                particle * posterior_stride +
                                rotation_offset * translation_count);
                            float* minvsigma2_arg = const_cast<float*>(
                                minvsigma2 + particle * image_stride);
                            float* ctf_arg = const_cast<float*>(
                                ctf + particle * image_stride);
                            unsigned long translation_count_arg =
                                static_cast<unsigned long>(translation_count);
                            float significant_weight_arg = significant_weight;
                            float weight_norm_arg = weight_norm;
                            float* eulers_arg = const_cast<float*>(
                                projector_eulers + particle * euler_stride +
                                rotation_offset * 9);
                            float* accumulator_real_arg =
                                accumulator_real + accumulator_offset;
                            float* accumulator_imag_arg =
                                accumulator_imag + accumulator_offset;
                            float* accumulator_weight_arg =
                                accumulator_weight + accumulator_offset;
                            int max_r_arg =
                                static_cast<int>(sqrtf(max_r2) + 0.5f);
                            int max_r2_arg = static_cast<int>(max_r2);
                            float padding_factor_arg =
                                static_cast<float>(upsampling);
                            unsigned image_x_arg =
                                static_cast<unsigned>(image_w);
                            unsigned image_y_arg =
                                static_cast<unsigned>(image_h);
                            unsigned image_z_arg = 1;
                            unsigned image_xyz_arg =
                                static_cast<unsigned>(pixel_count);
                            unsigned model_x_arg =
                                static_cast<unsigned>(model_x);
                            unsigned model_y_arg =
                                static_cast<unsigned>(model_y);
                            int model_init_y_arg = model_init_y;
                            int model_init_z_arg = model_init_z;
                            void* kernel_parameters[] = {
                                &projector,
                                &image_real_arg,
                                &image_imag_arg,
                                &translation_x_arg,
                                &translation_y_arg,
                                &translation_z_arg,
                                &weights_arg,
                                &minvsigma2_arg,
                                &ctf_arg,
                                &translation_count_arg,
                                &significant_weight_arg,
                                &weight_norm_arg,
                                &eulers_arg,
                                &accumulator_real_arg,
                                &accumulator_imag_arg,
                                &accumulator_weight_arg,
                                &max_r_arg,
                                &max_r2_arg,
                                &padding_factor_arg,
                                &image_x_arg,
                                &image_y_arg,
                                &image_z_arg,
                                &image_xyz_arg,
                                &model_x_arg,
                                &model_y_arg,
                                &model_init_y_arg,
                                &model_init_z_arg,
                            };
                            const bool trace_this_gap = trace_this_gap_particle;
                            if (trace_this_gap)
                                wavg_bpref_intrinsic_gap_ns[particle] =
                                    std::chrono::duration_cast<
                                        std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() -
                                            exact_wavg_return_time).count();
                            if (wavg_bpref_host_gap_requested)
                            {
                                const auto target = exact_wavg_return_time +
                                    std::chrono::nanoseconds(
                                        wavg_bpref_host_gap_ns);
                                constexpr auto spin_guard =
                                    std::chrono::microseconds(50);
                                const auto now = std::chrono::steady_clock::now();
                                if (now + spin_guard < target)
                                    std::this_thread::sleep_until(
                                        target - spin_guard);
                                while (std::chrono::steady_clock::now() < target) {}
                            }
                            if (trace_this_gap)
                                wavg_bpref_effective_gap_ns[particle] =
                                    std::chrono::duration_cast<
                                        std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() -
                                            exact_wavg_return_time).count();
                            std::chrono::steady_clock::time_point
                                bpref_enqueue_start;
                            if (trace_this_gap)
                                bpref_enqueue_start =
                                    std::chrono::steady_clock::now();
                            driver_result = cuLaunchKernel(
                                exact_native_ptx_kernel,
                                static_cast<unsigned>(grid_rotations), 1, 1,
                                128, 1, 1,
                                0,
                                reinterpret_cast<CUstream>(particle_streams[lane]),
                                kernel_parameters,
                                nullptr);
                            if (driver_result != CUDA_SUCCESS)
                                return report_relion_vdam_driver_error(
                                    "cuLaunchKernel", driver_result);
                            if (trace_this_gap)
                            {
                                const auto bpref_enqueue_end =
                                    std::chrono::steady_clock::now();
                                bpref_host_enqueue_ns[particle] =
                                    std::chrono::duration_cast<
                                        std::chrono::nanoseconds>(
                                            bpref_enqueue_end -
                                            bpref_enqueue_start).count();
                                wavg_to_bpref_return_ns[particle] =
                                    std::chrono::duration_cast<
                                        std::chrono::nanoseconds>(
                                            bpref_enqueue_end -
                                            exact_wavg_return_time).count();
                            }
                            return cudaSuccess;
                        }
                    }
                    const bool trace_runtime_gap =
                        runtime_bpref_with_exact_wavg_requested &&
                        trace_this_gap_particle;
                    if (trace_runtime_gap)
                        wavg_bpref_intrinsic_gap_ns[particle] =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() -
                                exact_wavg_return_time).count();
                    if (runtime_bpref_with_exact_wavg_requested &&
                        wavg_bpref_host_gap_requested)
                    {
                        const auto target = exact_wavg_return_time +
                            std::chrono::nanoseconds(wavg_bpref_host_gap_ns);
                        constexpr auto spin_guard =
                            std::chrono::microseconds(50);
                        const auto now = std::chrono::steady_clock::now();
                        if (now + spin_guard < target)
                            std::this_thread::sleep_until(target - spin_guard);
                        while (std::chrono::steady_clock::now() < target) {}
                    }
                    if (trace_runtime_gap)
                        wavg_bpref_effective_gap_ns[particle] =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() -
                                exact_wavg_return_time).count();
                    std::chrono::steady_clock::time_point
                        runtime_bpref_enqueue_start;
                    if (trace_runtime_gap)
                        runtime_bpref_enqueue_start =
                            std::chrono::steady_clock::now();
                    // The launch-serialized path enters this lambda once per
                    // rotation. Materialize the shared projection/residual
                    // buffer only before its first ordered scatter launch;
                    // all later launches are on the same stream.
                    const bool materialize_ordered_operands =
                        preproject_persistent_requested &&
                        (persistent_serial_rotation_replay ||
                         !serial_rotation_replay || rotation_offset == 0);
                    if (materialize_ordered_operands)
                    {
                        const int64_t operand_rotation_count =
                            precompute_ordered_residuals_requested
                                ? rotation_count
                                : particle_rotation_count;
                        relion_vdam_native_project_f32_kernel<<<
                            operand_rotation_count,
                            128,
                            0,
                            particle_streams[lane]>>>(
                            projector,
                            projector_eulers + particle * euler_stride,
                            preprojected_references,
                            static_cast<unsigned>(image_w),
                            static_cast<unsigned>(image_h),
                            static_cast<unsigned>(pixel_count),
                            static_cast<unsigned>(operand_rotation_count),
                            runtime_current_size);
                        const cudaError_t projection_error = cudaGetLastError();
                        if (projection_error != cudaSuccess)
                            return projection_error;
                        if (precompute_residuals_requested)
                        {
                            relion_vdam_native_residual_f32_kernel<<<
                                operand_rotation_count,
                                128,
                                0,
                                particle_streams[lane]>>>(
                                preprojected_references,
                                precomputed_residual_weights,
                                image_real + particle * image_stride,
                                image_imag + particle * image_stride,
                                translation_x,
                                translation_y,
                                const_cast<float*>(
                                    posterior_over_weight_norm +
                                    particle * posterior_stride),
                                const_cast<float*>(
                                    minvsigma2 + particle * image_stride),
                                const_cast<float*>(
                                    ctf + particle * image_stride),
                                static_cast<unsigned long>(translation_count),
                                significant_weight,
                                weight_norm,
                                static_cast<unsigned>(image_w),
                                static_cast<unsigned>(image_h),
                                static_cast<unsigned>(pixel_count),
                                static_cast<unsigned>(operand_rotation_count),
                                runtime_current_size);
                            const cudaError_t residual_error = cudaGetLastError();
                            if (residual_error != cudaSuccess)
                                return residual_error;
                        }
                    }
                    const auto launch_runtime_sgd = [&](
                        auto persistent_tag,
                        auto fixed_warp_order_tag) -> cudaError_t {
                        constexpr bool persistent_serial =
                            decltype(persistent_tag)::value;
                        constexpr bool fixed_warp_order =
                            decltype(fixed_warp_order_tag)::value;
                        // Host dispatch cannot combine persistent serial execution
                        // with captured-order or tracing template parameters.
                        if constexpr (persistent_serial && (use_captured_order || use_trace))
                        {
                            return cudaErrorInvalidValue;
                        }
                        else if constexpr (fixed_warp_order)
                        {
                            if constexpr (
                                use_captured_order || use_trace ||
                                persistent_serial)
                                return cudaErrorInvalidValue;
                            return launch_precomputed_fixed_warp_scatter(
                                projector_eulers + particle * euler_stride,
                                rotation_offset,
                                static_cast<unsigned>(particle_rotation_count));
                        }
                        else
                        {
                            const int64_t ordered_operand_offset =
                                persistent_serial
                                    ? 0
                                    : rotation_offset *
                                        static_cast<int64_t>(pixel_count);
                            relion_vdam_native_sgd_f32_kernel<
                                Accumulator,
                                use_captured_order,
                                use_trace,
                                persistent_serial,
                                fixed_warp_order><<<
                                grid_rotations, 128, 0, particle_streams[lane]>>>(
                                projector,
                                precompute_residuals_requested
                                    ? nullptr
                                    : preprojected_references,
                                precompute_residuals_requested
                                    ? preprojected_references +
                                        ordered_operand_offset
                                    : nullptr,
                                precompute_residuals_requested
                                    ? precomputed_residual_weights +
                                        ordered_operand_offset
                                    : nullptr,
                                image_real + particle * image_stride,
                                image_imag + particle * image_stride,
                                translation_x,
                                translation_y,
                                nullptr,
                                const_cast<float*>(
                                    posterior_over_weight_norm +
                                    particle * posterior_stride +
                                    rotation_offset * translation_count),
                                const_cast<float*>(
                                    minvsigma2 + particle * image_stride),
                                const_cast<float*>(
                                    ctf + particle * image_stride),
                                static_cast<unsigned long>(translation_count),
                                significant_weight,
                                weight_norm,
                                const_cast<float*>(
                                    projector_eulers +
                                    particle * euler_stride +
                                    rotation_offset * 9),
                                accumulator_real + accumulator_offset,
                                accumulator_imag + accumulator_offset,
                                accumulator_weight + accumulator_offset,
                                static_cast<int>(sqrtf(max_r2) + 0.5f),
                                static_cast<int>(max_r2),
                                static_cast<float>(upsampling),
                                static_cast<unsigned>(image_w),
                                static_cast<unsigned>(image_h),
                                1,
                                static_cast<unsigned>(pixel_count),
                                static_cast<unsigned>(model_x),
                                static_cast<unsigned>(model_y),
                                model_init_y,
                                model_init_z,
                                static_cast<unsigned>(particle_rotation_count),
                                use_captured_order
                                    ? rotation_replay_order + particle * rotation_count
                                    : nullptr,
                                use_trace
                                    ? candidate_trace_records + particle * rotation_count
                                    : nullptr,
                                trace_launch_sequence,
                                use_trace
                                    ? (candidate_trace_requested
                                        ? static_cast<std::int64_t>(
                                            particle_trace_ids_host[particle])
                                        : particle)
                                    : 0,
                                static_cast<std::int32_t>(lane),
                                candidate_trace_iteration,
                                runtime_current_size);
                            return cudaGetLastError();
                        }
                    };
                    cudaError_t runtime_launch_error = cudaSuccess;
                    if (fixed_warp_order_scatter_requested)
                        runtime_launch_error =
                            launch_runtime_sgd(std::false_type{}, std::true_type{});
                    else if (persistent_serial_rotation_replay)
                        runtime_launch_error =
                            launch_runtime_sgd(std::true_type{}, std::false_type{});
                    else
                        runtime_launch_error =
                            launch_runtime_sgd(std::false_type{}, std::false_type{});
                    if (trace_runtime_gap)
                    {
                        const auto runtime_bpref_enqueue_end =
                            std::chrono::steady_clock::now();
                        bpref_host_enqueue_ns[particle] =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                runtime_bpref_enqueue_end -
                                runtime_bpref_enqueue_start).count();
                        wavg_to_bpref_return_ns[particle] =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                runtime_bpref_enqueue_end -
                                exact_wavg_return_time).count();
                    }
                    return runtime_launch_error;
                };
                const bool use_captured_order =
                    captured_rotation_replay && !serial_rotation_replay;
                cudaError_t launch_error = cudaSuccess;
                if (use_captured_order && device_trace_requested)
                    launch_error = launch_sgd(std::true_type{}, std::true_type{});
                else if (use_captured_order)
                    launch_error = launch_sgd(std::true_type{}, std::false_type{});
                else if (device_trace_requested)
                    launch_error = launch_sgd(std::false_type{}, std::true_type{});
                else
                    launch_error = launch_sgd(std::false_type{}, std::false_type{});
                if (launch_error != cudaSuccess) return launch_error;
            }
            return cudaSuccess;
        };
        const auto launch_particle = [&](int64_t particle, int lane) {
            if (float64_accumulator_replay)
                return launch_particle_with_accumulators(
                    particle, lane, data_real_volume_f64, data_imag_volume_f64,
                    weight_volume_f64);
            return launch_particle_with_accumulators(
                particle, lane, data_real_volume, data_imag_volume, weight_volume);
        };
        if (parallel_worker_replay)
        {
            // The captured owner map fixes RELION's task-to-worker assignment.
            // Issue each worker's exact particle chain from an independent host
            // thread so cross-stream launch timing is no longer serialized by
            // RECOVAR's controller thread.
            cudaError_t lane_errors[kRelionVdamWorkerStreams] = {};
            std::thread worker_threads[kRelionVdamWorkerStreams];
            for (int lane = 0; lane < kRelionVdamWorkerStreams; ++lane)
            {
                worker_threads[lane] = std::thread([&, lane]() {
                    bool lane_started = false;
                    for (int64_t particle = 0; particle < n_particles; ++particle)
                    {
                        if (worker_lanes_host[particle] != lane) continue;
                        if (lane_started)
                        {
                            lane_errors[lane] =
                                cudaStreamSynchronize(particle_streams[lane]);
                            if (lane_errors[lane] != cudaSuccess) return;
                        }
                        if (quiesced_prelaunch_capture_requested &&
                            particle_trace_ids_host[particle] ==
                                quiesced_prelaunch_target_particle_id)
                        {
                            std::unique_lock<std::shared_mutex> launch_lock(
                                quiesced_prelaunch_launch_gate);
                            lane_errors[lane] = launch_particle(particle, lane);
                        }
                        else if (quiesced_prelaunch_capture_requested)
                        {
                            std::shared_lock<std::shared_mutex> launch_lock(
                                quiesced_prelaunch_launch_gate);
                            lane_errors[lane] = launch_particle(particle, lane);
                        }
                        else
                        {
                            lane_errors[lane] = launch_particle(particle, lane);
                        }
                        if (lane_errors[lane] != cudaSuccess) return;
                        lane_started = true;
                    }
                });
            }
            for (int lane = 0; lane < kRelionVdamWorkerStreams; ++lane)
                worker_threads[lane].join();
            for (int lane = 0; lane < kRelionVdamWorkerStreams; ++lane)
            {
                if (lane_errors[lane] != cudaSuccess)
                {
                    err = lane_errors[lane];
                    goto cleanup;
                }
            }
        }
        else
        {
            bool lane_started[kRelionVdamWorkerStreams] = {};
            const auto particle_timing_epoch = std::chrono::steady_clock::now();
            for (int64_t particle = 0; particle < n_particles; ++particle)
            {
                const int lane = worker_lanes_host[particle];
                if (lane_started[lane])
                {
                    err = cudaStreamSynchronize(particle_streams[lane]);
                    if (err != cudaSuccess) goto cleanup;
                }
                if (captured_particle_timing_replay)
                {
                    // Pace host launches with the native first-block offsets.
                    // Waiting here avoids consuming SMs with delay kernels and
                    // keeps particle operands, owners, and launch order fixed.
                    const auto target = particle_timing_epoch +
                        std::chrono::nanoseconds(
                            particle_start_offsets_ns_host[particle]);
                    constexpr auto spin_guard = std::chrono::microseconds(50);
                    const auto now = std::chrono::steady_clock::now();
                    if (now + spin_guard < target)
                        std::this_thread::sleep_until(target - spin_guard);
                    while (std::chrono::steady_clock::now() < target) {}
                }
                err = launch_particle(particle, lane);
                if (err != cudaSuccess) goto cleanup;
                lane_started[lane] = true;
            }
        }
        err = synchronize_relion_vdam_worker_streams(particle_streams);
        if (err != cudaSuccess) goto cleanup;
        if (wavg_bpref_host_gap_trace_requested)
        {
            bool found = false;
            for (int64_t particle = 0; particle < n_particles; ++particle)
                found = found || wavg_bpref_intrinsic_gap_ns[particle] >= 0;
            // One host-replay callback covers one packed bucket.  Most
            // callbacks do not contain the requested global particle ID, so
            // only the owning callback appends a record.
            if (found)
            {
                std::ofstream trace(
                    wavg_bpref_host_gap_trace_path, std::ios::app);
                if (!trace)
                {
                    err = cudaErrorInvalidValue;
                    goto cleanup;
                }
                if (trace.tellp() == 0)
                    trace << "particle\ttrace_particle_id\tworker_lane"
                          << "\twavg_host_enqueue_ns"
                          << "\tintrinsic_gap_ns\teffective_gap_ns"
                          << "\tbpref_host_enqueue_ns"
                          << "\twavg_to_bpref_return_ns"
                          << "\ttarget_gap_ns\n";
                for (int64_t particle = 0; particle < n_particles; ++particle)
                {
                    if (wavg_bpref_intrinsic_gap_ns[particle] < 0) continue;
                    trace << particle << '\t'
                          << particle_trace_ids_host[particle] << '\t'
                          << worker_lanes_host[particle] << '\t'
                          << wavg_host_enqueue_ns[particle] << '\t'
                          << wavg_bpref_intrinsic_gap_ns[particle] << '\t'
                          << wavg_bpref_effective_gap_ns[particle] << '\t'
                          << bpref_host_enqueue_ns[particle] << '\t'
                          << wavg_to_bpref_return_ns[particle] << '\t'
                          << (wavg_bpref_host_gap_requested
                                  ? wavg_bpref_host_gap_ns
                                  : -1)
                          << '\n';
                }
                trace.close();
                if (!trace)
                {
                    err = cudaErrorInvalidValue;
                    goto cleanup;
                }
            }
        }
        if (candidate_trace_requested)
        {
            // Keep tracing passive: all particle streams complete before one
            // bulk device-to-host copy.  Synchronizing and copying inside
            // launch_particle would serialize otherwise concurrent lanes and
            // make the measured device chronology self-perturbing.
            std::vector<VdamCandidateBlockTraceRecord> host_records(
                static_cast<size_t>(n_particles * rotation_count));
            err = cudaMemcpy(
                host_records.data(),
                candidate_trace_records,
                host_records.size() * sizeof(VdamCandidateBlockTraceRecord),
                cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) goto cleanup;
            for (int64_t particle = 0; particle < n_particles; ++particle)
            {
                const auto record_count = static_cast<std::uint64_t>(
                    rotation_replay_counts_host[particle]);
                if (!candidate_trace_writer->append(
                        host_records.data() + particle * rotation_count,
                        record_count))
                {
                    err = cudaErrorInvalidValue;
                    goto cleanup;
                }
            }
        }
        if (float64_accumulator_replay)
        {
            const unsigned int cast_blocks = static_cast<unsigned int>(
                (accumulator_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
            relion_vdam_cast_accumulator_kernel<float, double><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                data_real_volume_f64, data_real_volume, accumulator_count);
            relion_vdam_cast_accumulator_kernel<float, double><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                data_imag_volume_f64, data_imag_volume, accumulator_count);
            relion_vdam_cast_accumulator_kernel<float, double><<<
                cast_blocks, BLOCK_SIZE, 0, stream>>>(
                weight_volume_f64, weight_volume, accumulator_count);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
        // Accumulator-only callers do not consume this separately computed
        // denominator. All worker/scatter completion above is unchanged.
        if (denominator_sum != nullptr)
        {
            err = launch_relion_vdam_mstep_denominator_f32(
                stream,
                ctf,
                minvsigma2,
                posterior_over_weight_norm,
                denominator_sum,
                n_particles,
                rotation_count,
                translation_count,
                pixel_count,
                pixel_capacity,
                runtime_current_size);
            if (err != cudaSuccess) goto cleanup;
        }
        err = cudaStreamSynchronize(stream);
    }

cleanup:
    if (exact_native_ptx_module != nullptr)
    {
        CUresult driver_result = cuCtxSetCurrent(exact_native_ptx_context);
        if (driver_result == CUDA_SUCCESS)
            driver_result = cuCtxSynchronize();
        if (driver_result == CUDA_SUCCESS)
            driver_result = cuModuleUnload(exact_native_ptx_module);
        if (driver_result != CUDA_SUCCESS && err == cudaSuccess)
            err = report_relion_vdam_driver_error(
                "module cleanup", driver_result);
    }
    if (ordered_scatter_graph_execs != nullptr)
    {
        for (int group = 0; group < reconstruction_group_count; ++group)
            if (ordered_scatter_graph_execs[group] != nullptr)
                cudaGraphExecDestroy(ordered_scatter_graph_execs[group]);
        std::free(ordered_scatter_graph_execs);
    }
    if (ordered_scatter_graphs != nullptr)
    {
        for (int group = 0; group < reconstruction_group_count; ++group)
            if (ordered_scatter_graphs[group] != nullptr)
                cudaGraphDestroy(ordered_scatter_graphs[group]);
        std::free(ordered_scatter_graphs);
    }
    destroy_relion_vdam_worker_streams(
        particle_streams, particle_inputs_ready);
    if (logical_projector_radius_host) cudaFreeHost(logical_projector_radius_host);
    if (reconstruction_groups_host) cudaFreeHost(reconstruction_groups_host);
    if (worker_lanes_host) cudaFreeHost(worker_lanes_host);
    if (particle_trace_ids_host) cudaFreeHost(particle_trace_ids_host);
    if (rotation_replay_order_host) cudaFreeHost(rotation_replay_order_host);
    if (rotation_replay_counts_host) cudaFreeHost(rotation_replay_counts_host);
    if (particle_start_offsets_ns_host) cudaFreeHost(particle_start_offsets_ns_host);
    if (candidate_trace_records) cudaFree(candidate_trace_records);
    if (data_real_volume_f64) cudaFree(data_real_volume_f64);
    if (data_imag_volume_f64) cudaFree(data_imag_volume_f64);
    if (weight_volume_f64) cudaFree(weight_volume_f64);
    if (texture_real) cudaDestroyTextureObject(texture_real);
    if (texture_imag) cudaDestroyTextureObject(texture_imag);
    if (array_real) cudaFreeArray(array_real);
    if (array_imag) cudaFreeArray(array_imag);
    if (image_real) cudaFree(image_real);
    if (image_imag) cudaFree(image_imag);
    if (translation_x) cudaFree(translation_x);
    if (translation_y) cudaFree(translation_y);
    if (wavg_dummy_outputs) cudaFree(wavg_dummy_outputs);
    if (precomputed_residual_weights) cudaFree(precomputed_residual_weights);
    if (preprojected_references) cudaFree(preprojected_references);
    if (ordered_scatter_graph_eulers) cudaFree(ordered_scatter_graph_eulers);
    if (real) cudaFree(real);
    if (imag) cudaFree(imag);
    return err;
}

}  // namespace

// Host-only ABI used by the clean-process VDAM discriminator.  It deliberately
// accepts already-materialized dense operands: the parent JAX process retains
// responsibility for scoring and packing, while a fresh CUDA-runtime process
// owns texture construction, worker streams, the exact PTX launch, and atomic
// accumulation.  This path is diagnostic-only and is not registered with XLA.
struct RelionVdamExactHostReplayArguments
{
    const float2* projector_full;
    const float2* images;
    const float* ctf;
    const float* minvsigma2;
    const float* posterior_over_weight_norm;
    const float* translation_angles;
    const float* projector_eulers;
    const float* compact_rotations;
    const std::int32_t* reconstruction_group_ids;
    const std::int32_t* worker_lane_ids;
    const std::int32_t* particle_trace_ids;
    const std::int32_t* rotation_replay_order;
    const std::int32_t* rotation_replay_counts;
    const std::int32_t* particle_start_offsets_ns;
    float* data_real_volume;
    float* data_imag_volume;
    float* weight_volume;
    float* denominator_sum;
    float* quiesced_prelaunch_data_real;
    float* quiesced_prelaunch_data_imag;
    float* quiesced_prelaunch_weight;
    std::int32_t* quiesced_prelaunch_found;
    std::int32_t* quiesced_prelaunch_particle_row;
    std::int32_t* quiesced_prelaunch_worker_lane;
    std::int32_t* quiesced_prelaunch_reconstruction_group;
    std::int64_t projector_size;
    std::int64_t n_particles;
    std::int64_t rotation_count;
    std::int64_t translation_count;
    std::int64_t pixel_count;
    std::int64_t image_h;
    std::int64_t image_w;
    std::int64_t volume_n0;
    std::int64_t volume_n1;
    std::int64_t volume_n2;
    std::int64_t upsampling;
    std::int64_t max_r2_x4;
    std::int32_t physical_image_size;
    std::int32_t projector_max_r;
    std::int32_t projection_padding_factor;
    std::int32_t reconstruction_group_count;
    std::int32_t parallel_worker_replay;
    std::int64_t quiesced_prelaunch_target_particle_id;
};

extern "C" int recovar_relion_vdam_exact_native_host_replay(
    const RelionVdamExactHostReplayArguments* arguments)
{
    if (arguments == nullptr ||
        arguments->projector_full == nullptr ||
        arguments->images == nullptr ||
        arguments->ctf == nullptr ||
        arguments->minvsigma2 == nullptr ||
        arguments->posterior_over_weight_norm == nullptr ||
        arguments->translation_angles == nullptr ||
        arguments->projector_eulers == nullptr ||
        arguments->compact_rotations == nullptr ||
        arguments->reconstruction_group_ids == nullptr ||
        arguments->worker_lane_ids == nullptr ||
        arguments->particle_trace_ids == nullptr ||
        arguments->rotation_replay_order == nullptr ||
        arguments->rotation_replay_counts == nullptr ||
        arguments->particle_start_offsets_ns == nullptr ||
        arguments->data_real_volume == nullptr ||
        arguments->data_imag_volume == nullptr ||
        arguments->weight_volume == nullptr ||
        arguments->denominator_sum == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    if (arguments->projector_size <= 0 ||
        arguments->n_particles <= 0 ||
        arguments->rotation_count <= 0 ||
        arguments->translation_count <= 0 ||
        arguments->pixel_count <= 0 ||
        arguments->image_h <= 0 ||
        arguments->image_w <= 0 ||
        arguments->image_h * arguments->image_w != arguments->pixel_count ||
        arguments->volume_n0 <= 0 ||
        arguments->volume_n1 <= 0 ||
        arguments->volume_n2 <= 0 ||
        arguments->reconstruction_group_count <= 0 ||
        (arguments->parallel_worker_replay != 0 &&
         arguments->parallel_worker_replay != 1))
        return static_cast<int>(cudaErrorInvalidValue);
    const bool quiesced_prelaunch_capture_requested =
        arguments->quiesced_prelaunch_target_particle_id >= 0;
    if (quiesced_prelaunch_capture_requested &&
        (arguments->quiesced_prelaunch_data_real == nullptr ||
         arguments->quiesced_prelaunch_data_imag == nullptr ||
         arguments->quiesced_prelaunch_weight == nullptr ||
         arguments->quiesced_prelaunch_found == nullptr ||
         arguments->quiesced_prelaunch_particle_row == nullptr ||
         arguments->quiesced_prelaunch_worker_lane == nullptr ||
         arguments->quiesced_prelaunch_reconstruction_group == nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    const char* exact_ptx = std::getenv(kRelionVdamExactNativePtxEnv);
    if (exact_ptx == nullptr || exact_ptx[0] == '\0')
        return static_cast<int>(cudaErrorInvalidValue);

    const std::int64_t projector_count =
        arguments->projector_size * arguments->projector_size *
        arguments->projector_size;
    const std::int64_t image_count =
        arguments->n_particles * arguments->pixel_count;
    const std::int64_t posterior_count =
        arguments->n_particles * arguments->rotation_count *
        arguments->translation_count;
    const std::int64_t rotation_count =
        arguments->n_particles * arguments->rotation_count;
    const std::int64_t accumulator_stride =
        arguments->volume_n0 * arguments->volume_n1 *
        (arguments->volume_n2 / 2 + 1);
    const std::int64_t accumulator_count =
        arguments->reconstruction_group_count * accumulator_stride;
    const std::int64_t denominator_count =
        arguments->n_particles * arguments->rotation_count *
        arguments->pixel_count;

    cudaStream_t stream = nullptr;
    float2* device_projector = nullptr;
    float2* device_images = nullptr;
    float* device_ctf = nullptr;
    float* device_minvsigma2 = nullptr;
    float* device_posterior = nullptr;
    float* device_translations = nullptr;
    float* device_eulers = nullptr;
    float* device_compact_rotations = nullptr;
    std::int32_t* device_reconstruction_groups = nullptr;
    std::int32_t* device_worker_lanes = nullptr;
    std::int32_t* device_particle_trace_ids = nullptr;
    std::int32_t* device_rotation_order = nullptr;
    std::int32_t* device_rotation_counts = nullptr;
    std::int32_t* device_particle_offsets = nullptr;
    float* device_data_real = nullptr;
    float* device_data_imag = nullptr;
    float* device_weight = nullptr;
    float* device_denominator = nullptr;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) goto cleanup_host_replay;

#define RECOVAR_HOST_REPLAY_ALLOC_COPY(device_pointer, host_pointer, count)       \
    do {                                                                          \
        error = cudaMalloc(                                                       \
            reinterpret_cast<void**>(&(device_pointer)),                         \
            static_cast<std::size_t>(count) * sizeof(*(device_pointer)));         \
        if (error != cudaSuccess) goto cleanup_host_replay;                       \
        error = cudaMemcpyAsync(                                                  \
            (device_pointer),                                                     \
            (host_pointer),                                                       \
            static_cast<std::size_t>(count) * sizeof(*(device_pointer)),          \
            cudaMemcpyHostToDevice,                                               \
            stream);                                                              \
        if (error != cudaSuccess) goto cleanup_host_replay;                       \
    } while (false)

    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_projector, arguments->projector_full, projector_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_images, arguments->images, image_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(device_ctf, arguments->ctf, image_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_minvsigma2, arguments->minvsigma2, image_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_posterior, arguments->posterior_over_weight_norm, posterior_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_translations, arguments->translation_angles,
        arguments->translation_count * 2);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_eulers, arguments->projector_eulers, rotation_count * 9);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_compact_rotations, arguments->compact_rotations, rotation_count * 6);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_reconstruction_groups, arguments->reconstruction_group_ids,
        arguments->n_particles);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_worker_lanes, arguments->worker_lane_ids, arguments->n_particles);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_particle_trace_ids, arguments->particle_trace_ids,
        arguments->n_particles);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_rotation_order, arguments->rotation_replay_order, rotation_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_rotation_counts, arguments->rotation_replay_counts,
        arguments->n_particles);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_particle_offsets, arguments->particle_start_offsets_ns,
        arguments->n_particles);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_data_real, arguments->data_real_volume, accumulator_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_data_imag, arguments->data_imag_volume, accumulator_count);
    RECOVAR_HOST_REPLAY_ALLOC_COPY(
        device_weight, arguments->weight_volume, accumulator_count);
    error = cudaMalloc(
        reinterpret_cast<void**>(&device_denominator),
        static_cast<std::size_t>(denominator_count) * sizeof(float));
    if (error != cudaSuccess) goto cleanup_host_replay;
    error = cudaMemsetAsync(
        device_denominator,
        0,
        static_cast<std::size_t>(denominator_count) * sizeof(float),
        stream);
    if (error != cudaSuccess) goto cleanup_host_replay;

    error = launch_relion_vdam_mstep_fused_projector_x_half(
        stream,
        device_projector,
        device_images,
        device_ctf,
        device_minvsigma2,
        device_posterior,
        device_translations,
        device_eulers,
        device_compact_rotations,
        device_reconstruction_groups,
        device_worker_lanes,
        device_particle_trace_ids,
        device_rotation_order,
        device_rotation_counts,
        device_particle_offsets,
        device_data_real,
        device_data_imag,
        device_weight,
        device_denominator,
        arguments->projector_size,
        arguments->n_particles,
        arguments->rotation_count,
        arguments->translation_count,
        arguments->pixel_count,
        arguments->pixel_count,
        arguments->image_h,
        arguments->image_w,
        arguments->volume_n0,
        arguments->volume_n1,
        arguments->volume_n2,
        arguments->upsampling,
        arguments->max_r2_x4,
        arguments->physical_image_size,
        arguments->projector_max_r,
        arguments->projection_padding_factor,
        arguments->reconstruction_group_count,
        arguments->parallel_worker_replay != 0,
        false,
        false,
        false,
        false,
        0,
        false,
        false,
        false,
        nullptr,
        arguments->quiesced_prelaunch_data_real,
        arguments->quiesced_prelaunch_data_imag,
        arguments->quiesced_prelaunch_weight,
        arguments->quiesced_prelaunch_target_particle_id,
        arguments->quiesced_prelaunch_found,
        arguments->quiesced_prelaunch_particle_row,
        arguments->quiesced_prelaunch_worker_lane,
        arguments->quiesced_prelaunch_reconstruction_group);
    if (error != cudaSuccess) goto cleanup_host_replay;

#define RECOVAR_HOST_REPLAY_COPY_OUTPUT(host_pointer, device_pointer, count)      \
    do {                                                                          \
        error = cudaMemcpyAsync(                                                  \
            (host_pointer),                                                       \
            (device_pointer),                                                     \
            static_cast<std::size_t>(count) * sizeof(*(device_pointer)),          \
            cudaMemcpyDeviceToHost,                                               \
            stream);                                                              \
        if (error != cudaSuccess) goto cleanup_host_replay;                       \
    } while (false)

    RECOVAR_HOST_REPLAY_COPY_OUTPUT(
        arguments->data_real_volume, device_data_real, accumulator_count);
    RECOVAR_HOST_REPLAY_COPY_OUTPUT(
        arguments->data_imag_volume, device_data_imag, accumulator_count);
    RECOVAR_HOST_REPLAY_COPY_OUTPUT(
        arguments->weight_volume, device_weight, accumulator_count);
    RECOVAR_HOST_REPLAY_COPY_OUTPUT(
        arguments->denominator_sum, device_denominator, denominator_count);
    error = cudaStreamSynchronize(stream);

cleanup_host_replay:
#undef RECOVAR_HOST_REPLAY_ALLOC_COPY
#undef RECOVAR_HOST_REPLAY_COPY_OUTPUT
    if (device_projector) cudaFree(device_projector);
    if (device_images) cudaFree(device_images);
    if (device_ctf) cudaFree(device_ctf);
    if (device_minvsigma2) cudaFree(device_minvsigma2);
    if (device_posterior) cudaFree(device_posterior);
    if (device_translations) cudaFree(device_translations);
    if (device_eulers) cudaFree(device_eulers);
    if (device_compact_rotations) cudaFree(device_compact_rotations);
    if (device_reconstruction_groups) cudaFree(device_reconstruction_groups);
    if (device_worker_lanes) cudaFree(device_worker_lanes);
    if (device_particle_trace_ids) cudaFree(device_particle_trace_ids);
    if (device_rotation_order) cudaFree(device_rotation_order);
    if (device_rotation_counts) cudaFree(device_rotation_counts);
    if (device_particle_offsets) cudaFree(device_particle_offsets);
    if (device_data_real) cudaFree(device_data_real);
    if (device_data_imag) cudaFree(device_data_imag);
    if (device_weight) cudaFree(device_weight);
    if (device_denominator) cudaFree(device_denominator);
    if (stream) cudaStreamDestroy(stream);
    if (error != cudaSuccess)
        std::fprintf(
            stderr,
            "RECOVAR exact RELION VDAM clean-process replay failed: %s\n",
            cudaGetErrorString(error));
    return static_cast<int>(error);
}

