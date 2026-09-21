// RELION float32 posterior primitives and FFI handlers.
// Included once at global scope after the shared Ampere scan policy.
// Keep the operation order and CUB reduction topology unchanged.

ffi::Error RelionCubSortScanF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must be F32");

    auto input_dims = values.dimensions();
    auto sorted_dims = sorted->dimensions();
    auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 1 || input_dims[0] < 1 ||
        sorted_dims.size() != 1 || sorted_dims[0] != input_dims[0] ||
        cumulative_dims.size() != 1 || cumulative_dims[0] != input_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: input and outputs must have the same nonempty 1-D shape");

    const int64_t count = input_dims[0];
    if (count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanF32: vector is too large for CUB's item count");

    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    cudaError_t err = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 sort query: ") + cudaGetErrorString(err));
    err = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 scan query: ") + cudaGetErrorString(err));

    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
    err = recovar::scratch_alloc(&temporary, temporary_bytes, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaMalloc: ") + cudaGetErrorString(err));

    err = cub::DeviceRadixSort::SortKeys(
        temporary, sort_bytes, input_ptr, sorted_ptr, static_cast<int>(count),
        0, sizeof(float) * 8, stream);
    if (err == cudaSuccess)
        err = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted_ptr, cumulative_ptr, static_cast<int>(count), stream);
    cudaError_t free_error = recovar::scratch_free(temporary, stream);
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 execute: ") + cudaGetErrorString(err));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanF32 cudaFree: ") + cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

ffi::Error RelionCubSortScanBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must be F32");

    const auto input_dims = values.dimensions();
    const auto sorted_dims = sorted->dimensions();
    const auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 2 || input_dims[0] < 1 || input_dims[1] < 1 ||
        sorted_dims.size() != 2 || sorted_dims[0] != input_dims[0] ||
        sorted_dims[1] != input_dims[1] ||
        cumulative_dims.size() != 2 || cumulative_dims[0] != input_dims[0] ||
        cumulative_dims[1] != input_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: input and outputs must have the same nonempty 2-D shape");
    if (input_dims[1] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubSortScanBatchedF32: row is too large for CUB's item count");

    const int64_t row_count = input_dims[0];
    const int count = static_cast<int>(input_dims[1]);
    const float* input_ptr = static_cast<const float*>(values.untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());

    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, input_ptr, sorted_ptr, count,
        0, sizeof(float) * 8, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 sort query: ") +
            cudaGetErrorString(error));
    error = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, count, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 scan query: ") +
            cudaGetErrorString(error));

    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(
        1, std::max(sort_bytes, scan_bytes));
    error = cudaMallocAsync(&temporary, temporary_bytes, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaMallocAsync: ") +
            cudaGetErrorString(error));

    for (int64_t row = 0; row < row_count && error == cudaSuccess; ++row)
    {
        const int64_t offset = row * static_cast<int64_t>(count);
        error = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, input_ptr + offset, sorted_ptr + offset,
            count, 0, sizeof(float) * 8, stream);
        if (error == cudaSuccess)
            error = relion_ampere_inclusive_sum_f32(
                temporary, scan_bytes, sorted_ptr + offset,
                cumulative_ptr + offset, count, stream);
    }
    const cudaError_t free_error = cudaFreeAsync(temporary, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 execute: ") +
            cudaGetErrorString(error));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubSortScanBatchedF32 cudaFreeAsync: ") +
            cudaGetErrorString(free_error));
    return ffi::Error::Success();
}

// Opt-in grouped coarse score-to-support transaction. Keep the existing CUB
// sort and Ampere scan; coarse_publication.py selects this CUDA backend.
// Support capacity is B*N: threshold ties are never truncated to maxsig.
struct CoarsePosteriorFiniteMax {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void coarse_posterior_row_max_f32(
    const float* scores, const float* raw_max, const int32_t* actual,
    float* maxima, int rows, int count)
{
    const int row = blockIdx.x;
    float best = -CUDART_INF_F;
    if (row < *actual && *actual <= rows) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const float value = __fadd_rn(scores[int64_t(row)*count+i], -raw_max[row]);
            if (isfinite(value)) best = fmaxf(best, value);
        }
    }
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const float result = Reduce(temp).Reduce(best, CoarsePosteriorFiniteMax());
    if (threadIdx.x == 0) maxima[row] = result;
}

__global__ void coarse_posterior_weights_f32(
    const float* scores, const float* raw_max, const float* maxima,
    float* weights, int count, int64_t size)
{
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= size) return;
    const int row = index/count;
    const float score = __fadd_rn(scores[index], -raw_max[row]);
    const float best = maxima[row];
    const float add = __fsub_rn(50.0f, isfinite(best) ? best : 0.0f);
    const float exponent = __fadd_rn(score, add);
    weights[index] = !isfinite(score) || !isfinite(best) || exponent < -88.0f
        ? 0.0f : expf(exponent);
}

struct CoarsePosteriorReduction {
    float probability, best_score;
    int winner, best, count;
};

struct CoarsePosteriorCombine {
    __device__ CoarsePosteriorReduction operator()(
        CoarsePosteriorReduction a, CoarsePosteriorReduction b) const
    {
        if (b.probability > a.probability ||
            (b.probability == a.probability && b.winner < a.winner)) {
            a.probability = b.probability; a.winner = b.winner;
        }
        // jnp.argmax picks the first NaN, otherwise the first maximum.
        if ((!isnan(a.best_score) && isnan(b.best_score)) ||
            (!isnan(a.best_score) && !isnan(b.best_score) &&
             (b.best_score > a.best_score ||
              (b.best_score == a.best_score && b.best < a.best))) ||
            (isnan(a.best_score) && isnan(b.best_score) && b.best < a.best)) {
            a.best_score = b.best_score; a.best = b.best;
        }
        a.count += b.count;
        return a;
    }
};

__device__ int coarse_posterior_upper_bound(const float* values, int count, float target)
{
    int low = 0, high = count;
    while (low < high) {
        const int middle = low + (high-low)/2;
        if (target < values[middle]) high = middle;
        else low = middle+1;
    }
    return low;
}

__global__ void coarse_posterior_finish_f32(
    const float* scores, const float* weights, const float* sorted,
    const float* cumulative, const float* maxima, const int32_t* actual,
    uint8_t* mask, float* statistics, int32_t* indices,
    int rows, int count, float fraction, int maxsig)
{
    const int row = blockIdx.x;
    const int64_t offset = int64_t(row)*count;
    __shared__ float total, threshold;
    __shared__ int has_mass, cutoff;
    if (threadIdx.x == 0) {
        total = cumulative[offset+count-1];
        has_mass = row < *actual && *actual > 0 && *actual <= rows &&
            isfinite(maxima[row]) && isfinite(total) && total > 0.0f;
        const float target = float(__dmul_rn(__dsub_rn(1.0, double(fraction)), double(total)));
        int rank = coarse_posterior_upper_bound(cumulative+offset, count, target);
        const int first_positive = coarse_posterior_upper_bound(sorted+offset, count, 0.0f);
        rank = min(max(rank, first_positive), count-1);
        if (maxsig > 0) rank = max(rank, count-maxsig);
        threshold = sorted[offset+rank];
        cutoff = has_mass ? count-rank : 0;
    }
    __syncthreads();
    CoarsePosteriorReduction value{0.0f, -CUDART_INF_F, count, count, 0};
    const CoarsePosteriorCombine combine;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float weight = weights[offset+i];
        const bool significant = has_mass && weight > 0.0f && weight >= threshold;
        mask[offset+i] = significant;
        // Preserve float/float division even where rounding creates a tie.
        const float probability = has_mass ? __fdiv_rn(weight, total) : 0.0f;
        value = combine(value, {probability, scores[offset+i], i, i, int(significant)});
    }
    using Reduce = cub::BlockReduce<CoarsePosteriorReduction, 256>;
    __shared__ typename Reduce::TempStorage temp;
    const auto result = Reduce(temp).Reduce(value, combine);
    if (threadIdx.x == 0) {
        const bool valid_count = *actual > 0 && *actual <= rows;
        statistics[4*row+0] = valid_count ? result.best_score : CUDART_NAN_F;
        statistics[4*row+1] = result.probability;
        statistics[4*row+2] = total;
        statistics[4*row+3] = threshold;
        indices[4*row+0] = result.best;
        indices[4*row+1] = result.winner;
        indices[4*row+2] = result.count;
        indices[4*row+3] = cutoff;
    }
}

ffi::Error RelionCoarsePosteriorTransactionF32Impl(
    cudaStream_t stream, ffi::AnyBuffer scores, ffi::AnyBuffer raw_max,
    ffi::AnyBuffer actual, float fraction, int64_t maxsig,
    ffi::Result<ffi::AnyBuffer> statistics, ffi::Result<ffi::AnyBuffer> indices,
    ffi::Result<ffi::AnyBuffer> support, ffi::Result<ffi::AnyBuffer> support_count)
{
    const auto dims = scores.dimensions();
    if (scores.element_type() != ffi::DataType::F32 || dims.size() != 2 ||
        dims[0] < 1 || dims[1] < 1 || dims[0] > INT_MAX/dims[1] ||
        raw_max.element_type() != ffi::DataType::F32 ||
        raw_max.dimensions().size() != 1 || raw_max.dimensions()[0] != dims[0] ||
        actual.element_type() != ffi::DataType::S32 || actual.dimensions().size() != 0 ||
        !std::isfinite(fraction) || fraction <= 0 || fraction > 1 || maxsig < 0 || maxsig > INT_MAX)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid scores, maxima, count or policy");
    const int rows = dims[0], count = dims[1], size = rows*count;
    if (statistics->element_type() != ffi::DataType::F32 ||
        indices->element_type() != ffi::DataType::S32 ||
        support->element_type() != ffi::DataType::S32 ||
        support_count->element_type() != ffi::DataType::S32 ||
        statistics->dimensions().size() != 2 || statistics->dimensions()[0] != rows || statistics->dimensions()[1] != 4 ||
        indices->dimensions().size() != 2 || indices->dimensions()[0] != rows || indices->dimensions()[1] != 4 ||
        support->dimensions().size() != 1 || support->dimensions()[0] != size ||
        support_count->dimensions().size() != 0)
        return ffi::Error::InvalidArgument("Coarse posterior: invalid output shapes/dtypes");
    // Three score-size scratch arrays and flags; no host reads or host stream
    // synchronization. All work and temporary lifetimes use the FFI stream.
    void* storage = nullptr;
    cudaError_t error = cudaMallocAsync(&storage, size_t(size)*13+size_t(rows)*4, stream);
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
    float* weights = static_cast<float*>(storage);
    float* sorted = weights+size;
    float* cumulative = sorted+size;
    float* maxima = cumulative+size;
    uint8_t* mask = reinterpret_cast<uint8_t*>(maxima+rows);
    auto* out_support = static_cast<int32_t*>(support->untyped_data());
    auto* out_count = static_cast<int32_t*>(support_count->untyped_data());
    thrust::counting_iterator<int32_t> positions(0);
    size_t sort_bytes=0, scan_bytes=0, select_bytes=0;
    error = cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, weights, sorted, count, 0, 32, stream);
    if (error == cudaSuccess) error = relion_ampere_inclusive_sum_f32(nullptr, scan_bytes, sorted, cumulative, count, stream);
    if (error == cudaSuccess) error = cub::DeviceSelect::Flagged(nullptr, select_bytes, positions, mask, out_support, out_count, size, stream);
    void* temporary = nullptr;
    if (error == cudaSuccess) error = cudaMallocAsync(&temporary, std::max(size_t(1), std::max(select_bytes, std::max(sort_bytes, scan_bytes))), stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(out_support, 0xff, size_t(size)*sizeof(int32_t), stream);
    if (error == cudaSuccess) {
        coarse_posterior_row_max_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()),
            static_cast<const int32_t*>(actual.untyped_data()), maxima, rows, count);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        coarse_posterior_weights_f32<<<(int64_t(size)+255)/256,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), static_cast<const float*>(raw_max.untyped_data()), maxima, weights, count, size);
        error = cudaGetLastError();
    }
    for (int row=0; row<rows && error==cudaSuccess; ++row) {
        const int64_t offset = int64_t(row)*count;
        error = cub::DeviceRadixSort::SortKeys(temporary, sort_bytes, weights+offset, sorted+offset, count, 0, 32, stream);
        if (error == cudaSuccess) error = relion_ampere_inclusive_sum_f32(temporary, scan_bytes, sorted+offset, cumulative+offset, count, stream);
    }
    if (error == cudaSuccess) {
        coarse_posterior_finish_f32<<<rows,256,0,stream>>>(
            static_cast<const float*>(scores.untyped_data()), weights, sorted, cumulative, maxima,
            static_cast<const int32_t*>(actual.untyped_data()), mask,
            static_cast<float*>(statistics->untyped_data()), static_cast<int32_t*>(indices->untyped_data()),
            rows, count, fraction, maxsig);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cub::DeviceSelect::Flagged(temporary, select_bytes, positions, mask, out_support, out_count, size, stream);
    const auto free_temp = temporary ? cudaFreeAsync(temporary, stream) : cudaSuccess;
    const auto free_storage = cudaFreeAsync(storage, stream);
    if (error == cudaSuccess) error = free_temp;
    if (error == cudaSuccess) error = free_storage;
    return error == cudaSuccess ? ffi::Error::Success() : ffi::Error::Internal(cudaGetErrorString(error));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCoarsePosteriorTransactionF32, RelionCoarsePosteriorTransactionF32Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Attr<float>("fraction").Attr<int64_t>("maxsig")
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
);

struct RelionPositiveF32
{
    __device__ __forceinline__ bool operator()(const float& value) const
    {
        return value > 0.0f;
    }
};

cudaError_t relion_cub_positive_sort_scan_f32(
    cudaStream_t stream,
    const float* input,
    float* sorted,
    float* cumulative,
    int count)
{
    float* filtered = nullptr;
    int* selected_count_device = nullptr;
    void* temporary = nullptr;
    cudaError_t err = cudaSuccess;

    do
    {
        err = recovar::scratch_alloc(reinterpret_cast<void**>(&filtered), count * sizeof(float), stream);
        if (err != cudaSuccess) break;
        err = recovar::scratch_alloc(reinterpret_cast<void**>(&selected_count_device), sizeof(int), stream);
        if (err != cudaSuccess) break;

        size_t select_bytes = 0;
        size_t sort_bytes = 0;
        size_t scan_bytes = 0;
        err = cub::DeviceSelect::If(
            nullptr, select_bytes, input, filtered, selected_count_device,
            count, RelionPositiveF32(), stream);
        if (err != cudaSuccess) break;
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, sort_bytes, filtered, sorted, count,
            0, sizeof(float) * 8, stream);
        if (err != cudaSuccess) break;
        err = relion_ampere_inclusive_sum_f32(
            nullptr, scan_bytes, sorted, cumulative, count, stream);
        if (err != cudaSuccess) break;

        const size_t temporary_bytes = std::max<size_t>(
            1, std::max(select_bytes, std::max(sort_bytes, scan_bytes)));
        err = recovar::scratch_alloc(&temporary, temporary_bytes, stream);
        if (err != cudaSuccess) break;
        err = cudaMemsetAsync(sorted, 0, count * sizeof(float), stream);
        if (err != cudaSuccess) break;
        err = cudaMemsetAsync(cumulative, 0, count * sizeof(float), stream);
        if (err != cudaSuccess) break;

        err = cub::DeviceSelect::If(
            temporary, select_bytes, input, filtered, selected_count_device,
            count, RelionPositiveF32(), stream);
        if (err != cudaSuccess) break;
        int selected_count = 0;
        err = cudaMemcpyAsync(
            &selected_count, selected_count_device, sizeof(int),
            cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;
        if (selected_count < 0 || selected_count > count)
        {
            err = cudaErrorInvalidValue;
            break;
        }
        if (selected_count == 0) break;

        const int output_offset = count - selected_count;
        err = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, filtered, sorted + output_offset,
            selected_count, 0, sizeof(float) * 8, stream);
        if (err != cudaSuccess) break;
        err = relion_ampere_inclusive_sum_f32(
            temporary, scan_bytes, sorted + output_offset,
            cumulative + output_offset, selected_count, stream);
    } while (false);

    const cudaError_t temporary_free_error =
        temporary == nullptr ? cudaSuccess : recovar::scratch_free(temporary, stream);
    const cudaError_t selected_count_free_error =
        selected_count_device == nullptr ? cudaSuccess : recovar::scratch_free(selected_count_device, stream);
    const cudaError_t filtered_free_error =
        filtered == nullptr ? cudaSuccess : recovar::scratch_free(filtered, stream);
    if (err != cudaSuccess) return err;
    if (temporary_free_error != cudaSuccess) return temporary_free_error;
    if (selected_count_free_error != cudaSuccess) return selected_count_free_error;
    return filtered_free_error;
}

ffi::Error RelionCubPositiveSortScanF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative)
{
    if (values.element_type() != ffi::DataType::F32 ||
        sorted->element_type() != ffi::DataType::F32 ||
        cumulative->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: input and outputs must be F32");

    auto input_dims = values.dimensions();
    auto sorted_dims = sorted->dimensions();
    auto cumulative_dims = cumulative->dimensions();
    if (input_dims.size() != 1 || input_dims[0] < 1 ||
        sorted_dims.size() != 1 || sorted_dims[0] != input_dims[0] ||
        cumulative_dims.size() != 1 || cumulative_dims[0] != input_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: input and outputs must have the same nonempty 1-D shape");

    const int64_t count = input_dims[0];
    if (count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionCubPositiveSortScanF32: vector is too large for CUB's item count");

    const cudaError_t err = relion_cub_positive_sort_scan_f32(
        stream,
        static_cast<const float*>(values.untyped_data()),
        static_cast<float*>(sorted->untyped_data()),
        static_cast<float*>(cumulative->untyped_data()),
        static_cast<int>(count));
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionCubPositiveSortScanF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

__global__ void relion_exponentiate_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float exponent = values[index] + add[0];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto add_dims = add.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        add_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateF32: values/output must be matching nonempty 1-D arrays and add a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_exponentiate_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(add.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateF32, RelionExponentiateF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_exponentiate_batched_f32_kernel(
    const float* values,
    const float* add,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float exponent = values[index] + add[index / row_size];
    output[index] = exponent < -88.0f ? 0.0f : expf(exponent);
}

ffi::Error RelionExponentiateBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer add,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        add.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto add_dims = add.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        add_dims.size() != 1 || add_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: expected values/output (B,N) and add (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionExponentiateBatchedF32: launch grid exceeds CUDA limit");
    relion_exponentiate_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(add.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionExponentiateBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionExponentiateBatchedF32, RelionExponentiateBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[0];
}

ffi::Error RelionDivideF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: inputs and output must be F32");

    auto value_dims = values.dimensions();
    auto divisor_dims = divisor.dimensions();
    auto output_dims = output->dimensions();
    if (value_dims.size() != 1 || value_dims[0] < 1 ||
        divisor_dims.size() != 0 || output_dims.size() != 1 ||
        output_dims[0] != value_dims[0])
        return ffi::Error::InvalidArgument(
            "RelionDivideF32: values/output must be matching nonempty 1-D arrays and divisor a scalar");

    const int64_t count = value_dims[0];
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    relion_divide_f32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(values.untyped_data()),
        static_cast<const float*>(divisor.untyped_data()),
        static_cast<float*>(output->untyped_data()),
        count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideF32: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideF32, RelionDivideF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

__global__ void relion_divide_batched_f32_kernel(
    const float* values,
    const float* divisor,
    float* output,
    int64_t row_size,
    int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = values[index] / divisor[index / row_size];
}

ffi::Error RelionDivideBatchedF32Impl(
    cudaStream_t stream,
    ffi::AnyBuffer values,
    ffi::AnyBuffer divisor,
    ffi::Result<ffi::AnyBuffer> output)
{
    if (values.element_type() != ffi::DataType::F32 ||
        divisor.element_type() != ffi::DataType::F32 ||
        output->element_type() != ffi::DataType::F32)
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: inputs and output must be F32");

    const auto value_dims = values.dimensions();
    const auto divisor_dims = divisor.dimensions();
    const auto output_dims = output->dimensions();
    if (value_dims.size() != 2 || value_dims[0] < 1 || value_dims[1] < 1 ||
        divisor_dims.size() != 1 || divisor_dims[0] != value_dims[0] ||
        output_dims.size() != 2 || output_dims[0] != value_dims[0] ||
        output_dims[1] != value_dims[1])
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: expected values/output (B,N) and divisor (B,)");

    const int64_t count = value_dims[0] * value_dims[1];
    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "RelionDivideBatchedF32: launch grid exceeds CUDA limit");
    relion_divide_batched_f32_kernel<<<
        static_cast<int>(block_count), threads, 0, stream>>>(
            static_cast<const float*>(values.untyped_data()),
            static_cast<const float*>(divisor.untyped_data()),
            static_cast<float*>(output->untyped_data()),
            value_dims[1], count);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("RelionDivideBatchedF32: ") +
            cudaGetErrorString(error));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionDivideBatchedF32, RelionDivideBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanF32, RelionCubSortScanF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubSortScanBatchedF32, RelionCubSortScanBatchedF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RelionCubPositiveSortScanF32, RelionCubPositiveSortScanF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

