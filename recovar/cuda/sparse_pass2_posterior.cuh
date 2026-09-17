// Fused sparse pass-2 posterior for RECOVAR's RELION-parity K=1 path.
//
// Two runtime-shaped XLA FFI handlers replace the per-shape XLA glue of
// recovar/em/sparse_pass2/sparse_pass2_posterior.py:
//
//   SparsePass2LogZF64      <-> _logsumexp_pass2_bucket_score_only
//   SparsePass2PosteriorF32 <-> _normalize_pass2_bucket_with_log_z followed by
//                               _relion_f32_fine_posterior
//
// Every per-element formula is copied from the XLA path.  Order-independent
// reductions (max, argmax with smallest index, mask counts) are exact.  The
// RELION significance boundary reuses the same cub::DeviceRadixSort::SortKeys
// and pinned Ampere inclusive scan as RelionCubSortScanBatchedF32Impl, so the
// float32 sum_weight and threshold are bitwise identical to that path.  The
// only order-dependent quantity is the float64 log-sum-exp of the log-Z
// handler, which uses a fixed block tree instead of XLA's reduction tree.
//
// This header is included from cuda_backproject.cu after
// relion_ampere_inclusive_sum_f32 is defined.

namespace recovar_sparse_pass2_posterior {

constexpr int kRowThreads = 1024;

__device__ __forceinline__ bool better_candidate(
    float value, int64_t index, float best_value, int64_t best_index)
{
    // jnp.argmax picks the smallest index among equal maxima.
    return value > best_value || (value == best_value && index < best_index);
}

// Block-wide finite max with smallest-index argmax.  All threads must call it.
__device__ void block_finite_argmax(
    const float* row, int64_t n, float* shared_values, int64_t* shared_indices,
    float& best_value, int64_t& best_index)
{
    float local_best = -CUDART_INF_F;
    int64_t local_index = INT64_MAX;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = row[i];
        if (isfinite(v) && better_candidate(v, i, local_best, local_index))
        {
            local_best = v;
            local_index = i;
        }
    }
    shared_values[threadIdx.x] = local_best;
    shared_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            const float other = shared_values[threadIdx.x + stride];
            const int64_t other_index = shared_indices[threadIdx.x + stride];
            if (better_candidate(other, other_index, shared_values[threadIdx.x],
                                 shared_indices[threadIdx.x]))
            {
                shared_values[threadIdx.x] = other;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    best_value = shared_values[0];
    best_index = shared_indices[0];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// SparsePass2LogZF64: per-row log-sum-exp of float32 scores in float64.
// ---------------------------------------------------------------------------
__global__ void log_z_kernel(const float* scores, int64_t n, double* log_z)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    __shared__ double shared_sums[kRowThreads];
    const float* row = scores + static_cast<int64_t>(blockIdx.x) * n;
    float best;
    int64_t best_index;
    block_finite_argmax(row, n, shared_values, shared_indices, best, best_index);
    const bool has_finite = isfinite(best);
    // XLA: shifted = where(has_finite, scores - safe_best, -inf) in float32,
    // then exp in float64; non-finite scores were replaced by -inf first.
    const float safe_best = has_finite ? best : 0.0f;
    double local_sum = 0.0;
    if (has_finite)
    {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        {
            const float v = row[i];
            const float shifted = isfinite(v) ? (v - safe_best) : -CUDART_INF_F;
            const double term = exp(static_cast<double>(shifted));
            local_sum += isfinite(term) ? term : 0.0;
        }
    }
    shared_sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const double sum_exp = shared_sums[0];
        const bool has_mass = has_finite && sum_exp > 0.0 && isfinite(sum_exp);
        log_z[blockIdx.x] = has_mass
            ? static_cast<double>(safe_best) + log(sum_exp)
            : -CUDART_INF;
    }
}

inline bool row_geometry(const ffi::Span<const int64_t> dims, int64_t& rows, int64_t& n)
{
    if (dims.size() < 2) return false;
    rows = dims[0];
    n = 1;
    for (size_t i = 1; i < dims.size(); ++i) n *= dims[i];
    return rows >= 1 && n >= 1;
}

inline bool same_dims(const ffi::Span<const int64_t> a, const ffi::Span<const int64_t> b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

ffi::Error log_z_impl(
    cudaStream_t stream,
    ffi::AnyBuffer scores,
    ffi::Result<ffi::AnyBuffer> log_z)
{
    if (scores.element_type() != ffi::DataType::F32 || log_z->element_type() != ffi::DataType::F64)
        return ffi::Error::InvalidArgument("SparsePass2LogZF64: scores must be F32 and log_z F64");
    int64_t rows = 0, n = 0;
    if (!row_geometry(scores.dimensions(), rows, n))
        return ffi::Error::InvalidArgument("SparsePass2LogZF64: scores must be a nonempty (B, ...) array");
    if (log_z->dimensions().size() != 1 || log_z->dimensions()[0] != rows)
        return ffi::Error::InvalidArgument("SparsePass2LogZF64: log_z must have shape (B,)");
    if (rows > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument("SparsePass2LogZF64: too many rows for one launch grid");
    log_z_kernel<<<static_cast<int>(rows), kRowThreads, 0, stream>>>(
        static_cast<const float*>(scores.untyped_data()), n,
        static_cast<double*>(log_z->untyped_data()));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2LogZF64: ") + cudaGetErrorString(error));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// SparsePass2PosteriorF32: fused normalization + RELION float32 fine posterior.
// ---------------------------------------------------------------------------
struct RowState
{
    float best;            // finite max or -inf
    float exponent_add;    // 50 - safe_best (RELION: 50 - weights_max)
    double safe_log_z;     // external log-Z when finite together with best, else 0
    float safe_sum_weight; // sum_weight when has_mass, else 1
    float threshold;       // RELION significance cutoff
    int has_finite;        // isfinite(best)
    int has_finite_norm;   // isfinite(best) && isfinite(log_z)
    int has_mass;          // has_finite && isfinite(sum_weight) && sum_weight > 0
    int keep_all;
};

__global__ void row_max_kernel(
    const float* scores, const double* log_z, int64_t n, RowState* state,
    double* log_z_out, float* best_log_score, int64_t* best_argmax)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    const int64_t row_index = blockIdx.x;
    const float* row = scores + row_index * n;
    float best;
    int64_t best_index;
    block_finite_argmax(row, n, shared_values, shared_indices, best, best_index);
    if (threadIdx.x != 0) return;
    const bool has_finite = isfinite(best);
    const double external_log_z = log_z[row_index];
    const bool has_finite_norm = has_finite && isfinite(external_log_z);
    RowState s;
    s.best = best;
    s.exponent_add = 50.0f - (has_finite ? best : 0.0f);
    s.safe_log_z = has_finite_norm ? external_log_z : 0.0;
    s.safe_sum_weight = 1.0f;
    s.threshold = 0.0f;
    s.has_finite = has_finite;
    s.has_finite_norm = has_finite_norm;
    s.has_mass = 0;
    s.keep_all = 0;
    state[row_index] = s;
    // _normalize_pass2_bucket_with_log_z outputs.
    log_z_out[row_index] = s.safe_log_z;
    best_log_score[row_index] = has_finite_norm ? best : -CUDART_INF_F;
    best_argmax[row_index] = has_finite_norm ? best_index : 0;
}

__global__ void exponentiate_kernel(
    const float* scores, const RowState* state, int64_t n, int64_t count,
    float* raw_weights, double* probs)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const RowState s = state[index / n];
    const float v = scores[index];
    const bool finite = isfinite(v);
    const float finite_score = finite ? v : -CUDART_INF_F;
    // relion_exponentiate_batched_f32_kernel: expf(value + add), zero below -88.
    const float exponent = finite_score + s.exponent_add;
    raw_weights[index] = exponent < -88.0f ? 0.0f : expf(exponent);
    // _normalize_pass2_bucket_with_log_z: exp(scores - safe_log_z) in float64,
    // zero when the row has no finite normalizer or the value overflows.
    double p = 0.0;
    if (s.has_finite_norm)
    {
        p = exp(static_cast<double>(finite_score) - s.safe_log_z);
        if (!isfinite(p)) p = 0.0;
    }
    probs[index] = p;
}

__global__ void threshold_kernel(
    const float* sorted, const float* cumulative, const float* external_sum_weight,
    int64_t rows, int64_t n, float adaptive_fraction, int keep_all, int use_external_sum_weight,
    RowState* state, float* sum_weight_out, float* threshold_out)
{
    const int64_t row_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_index >= rows) return;
    RowState s = state[row_index];
    const float* row_cumulative = cumulative + row_index * n;
    const float* row_sorted = sorted + row_index * n;
    const float fine_sum_weight = row_cumulative[n - 1];
    const float sum_weight = use_external_sum_weight
        ? external_sum_weight[row_index]
        : fine_sum_weight;
    const bool has_mass = s.has_finite && isfinite(sum_weight) && sum_weight > 0.0f;
    float threshold = 0.0f;
    if (!keep_all)
    {
        // _relion_cuda_f32_tail_target: float32 fraction widened to float64,
        // multiplied by the float64 widening of the float32 fine sum, narrowed.
        const float tail_target = static_cast<float>(
            (1.0 - static_cast<double>(adaptive_fraction)) * static_cast<double>(fine_sum_weight));
        // jnp.searchsorted(row, target, side="right") with JAX's default scan
        // method: ceil(log2(n + 1)) halving steps on [low, high) using
        // unsigned (low + high) / 2, moving left when target < row[mid].
        int32_t low = 0;
        int32_t high = static_cast<int32_t>(n);
        const int levels = static_cast<int>(ceil(log2(static_cast<double>(n) + 1.0)));
        for (int level = 0; level < levels; ++level)
        {
            const uint32_t mid_u = (static_cast<uint32_t>(low) + static_cast<uint32_t>(high)) / 2u;
            int32_t mid = static_cast<int32_t>(mid_u);
            int32_t clamped = mid < 0 ? 0 : (mid > static_cast<int32_t>(n) - 1 ? static_cast<int32_t>(n) - 1 : mid);
            const bool go_left = tail_target < row_cumulative[clamped];
            if (go_left) high = mid; else low = mid;
        }
        int32_t threshold_index = high;
        if (threshold_index > static_cast<int32_t>(n) - 1) threshold_index = static_cast<int32_t>(n) - 1;
        if (threshold_index < 0) threshold_index = 0;
        threshold = row_sorted[threshold_index];
    }
    s.has_mass = has_mass;
    s.safe_sum_weight = has_mass ? sum_weight : 1.0f;
    s.threshold = threshold;
    s.keep_all = keep_all;
    state[row_index] = s;
    sum_weight_out[row_index] = sum_weight;
    threshold_out[row_index] = threshold;
}

__global__ void normalize_kernel(
    const float* scores, const float* raw_weights, const RowState* state, int64_t n,
    float* normalized, float* reconstruction, bool* mask,
    int32_t* n_significant, float* max_posterior)
{
    __shared__ int shared_counts[kRowThreads];
    __shared__ float shared_max[kRowThreads];
    const int64_t row_index = blockIdx.x;
    const RowState s = state[row_index];
    const int64_t base = row_index * n;
    int local_count = 0;
    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float raw = raw_weights[base + i];
        // relion_divide_f32_kernel: IEEE float32 division by the row sum.
        const float weight = raw / s.safe_sum_weight;
        const bool finite = isfinite(scores[base + i]);
        const bool significant = s.has_mass && finite &&
            (s.keep_all ? (raw > 0.0f) : (raw >= s.threshold));
        const float kept = significant ? weight : 0.0f;
        normalized[base + i] = weight;
        reconstruction[base + i] = kept;
        mask[base + i] = significant;
        local_count += significant ? 1 : 0;
        local_max = kept > local_max ? kept : local_max;
    }
    shared_counts[threadIdx.x] = local_count;
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
            const float other = shared_max[threadIdx.x + stride];
            shared_max[threadIdx.x] = other > shared_max[threadIdx.x] ? other : shared_max[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        n_significant[row_index] = shared_counts[0];
        max_posterior[row_index] = shared_max[0];
    }
}

ffi::Error posterior_impl(
    cudaStream_t stream,
    float adaptive_fraction,
    int64_t keep_all,
    int64_t use_external_sum_weight,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer log_z,
    ffi::AnyBuffer external_sum_weight,
    ffi::Result<ffi::AnyBuffer> log_z_out,
    ffi::Result<ffi::AnyBuffer> best_log_score,
    ffi::Result<ffi::AnyBuffer> best_argmax,
    ffi::Result<ffi::AnyBuffer> max_posterior,
    ffi::Result<ffi::AnyBuffer> probs,
    ffi::Result<ffi::AnyBuffer> normalized,
    ffi::Result<ffi::AnyBuffer> reconstruction,
    ffi::Result<ffi::AnyBuffer> mask,
    ffi::Result<ffi::AnyBuffer> n_significant,
    ffi::Result<ffi::AnyBuffer> sum_weight,
    ffi::Result<ffi::AnyBuffer> threshold,
    ffi::Result<ffi::AnyBuffer> raw_weights,
    ffi::Result<ffi::AnyBuffer> sorted,
    ffi::Result<ffi::AnyBuffer> cumulative,
    ffi::Result<ffi::AnyBuffer> row_state)
{
    int64_t rows = 0, n = 0;
    if (scores.element_type() != ffi::DataType::F32 || !row_geometry(scores.dimensions(), rows, n))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: scores must be a nonempty F32 (B, ...) array");
    if (rows > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: rows or row length exceed CUDA/CUB limits");
    const int64_t count = rows * n;
    const auto vector_ok = [&](const ffi::AnyBuffer& b, ffi::DataType type) {
        return b.element_type() == type && b.dimensions().size() == 1 && b.dimensions()[0] == rows;
    };
    const auto full_ok = [&](const ffi::AnyBuffer& b, ffi::DataType type) {
        return b.element_type() == type && same_dims(b.dimensions(), scores.dimensions());
    };
    if (!vector_ok(log_z, ffi::DataType::F64) || !vector_ok(external_sum_weight, ffi::DataType::F32))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: log_z must be F64 (B,) and external_sum_weight F32 (B,)");
    if (!vector_ok(*log_z_out, ffi::DataType::F64) || !vector_ok(*best_log_score, ffi::DataType::F32) ||
        !vector_ok(*best_argmax, ffi::DataType::S64) || !vector_ok(*max_posterior, ffi::DataType::F32) ||
        !vector_ok(*n_significant, ffi::DataType::S32) || !vector_ok(*sum_weight, ffi::DataType::F32) ||
        !vector_ok(*threshold, ffi::DataType::F32))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: per-row outputs must have shape (B,) with the documented dtypes");
    if (!full_ok(*probs, ffi::DataType::F64) || !full_ok(*normalized, ffi::DataType::F32) ||
        !full_ok(*reconstruction, ffi::DataType::F32) || !full_ok(*mask, ffi::DataType::PRED) ||
        !full_ok(*raw_weights, ffi::DataType::F32) || !full_ok(*sorted, ffi::DataType::F32) ||
        !full_ok(*cumulative, ffi::DataType::F32))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: full outputs must match the scores shape with the documented dtypes");
    if (row_state->element_type() != ffi::DataType::U8 || row_state->dimensions().size() != 2 ||
        row_state->dimensions()[0] != rows || row_state->dimensions()[1] != static_cast<int64_t>(sizeof(RowState)))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: row_state scratch must be U8 (B, sizeof(RowState))");

    const float* scores_ptr = static_cast<const float*>(scores.untyped_data());
    RowState* state_ptr = static_cast<RowState*>(row_state->untyped_data());
    float* raw_ptr = static_cast<float*>(raw_weights->untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());

    row_max_kernel<<<static_cast<int>(rows), kRowThreads, 0, stream>>>(
        scores_ptr, static_cast<const double*>(log_z.untyped_data()), n, state_ptr,
        static_cast<double*>(log_z_out->untyped_data()),
        static_cast<float*>(best_log_score->untyped_data()),
        static_cast<int64_t*>(best_argmax->untyped_data()));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 row max: ") + cudaGetErrorString(error));

    constexpr int threads = 256;
    const int64_t block_count = (count + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument("SparsePass2PosteriorF32: launch grid exceeds CUDA limit");
    exponentiate_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
        scores_ptr, state_ptr, n, count, raw_ptr, static_cast<double*>(probs->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 exponentiate: ") + cudaGetErrorString(error));

    // Same CUB radix sort and pinned Ampere inclusive scan as
    // RelionCubSortScanBatchedF32Impl, row by row on the caller's stream.
    const int row_count = static_cast<int>(n);
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, raw_ptr, sorted_ptr, row_count, 0, sizeof(float) * 8, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 sort query: ") + cudaGetErrorString(error));
    error = relion_ampere_inclusive_sum_f32(nullptr, scan_bytes, sorted_ptr, cumulative_ptr, row_count, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 scan query: ") + cudaGetErrorString(error));
    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
    error = cudaMallocAsync(&temporary, temporary_bytes, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 cudaMallocAsync: ") + cudaGetErrorString(error));
    for (int64_t row = 0; row < rows && error == cudaSuccess; ++row)
    {
        const int64_t offset = row * n;
        error = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, raw_ptr + offset, sorted_ptr + offset, row_count,
            0, sizeof(float) * 8, stream);
        if (error == cudaSuccess)
            error = relion_ampere_inclusive_sum_f32(
                temporary, scan_bytes, sorted_ptr + offset, cumulative_ptr + offset, row_count, stream);
    }
    const cudaError_t free_error = cudaFreeAsync(temporary, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 sort/scan: ") + cudaGetErrorString(error));
    if (free_error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 cudaFreeAsync: ") + cudaGetErrorString(free_error));

    constexpr int row_threads = 128;
    const int row_blocks = static_cast<int>((rows + row_threads - 1) / row_threads);
    threshold_kernel<<<row_blocks, row_threads, 0, stream>>>(
        sorted_ptr, cumulative_ptr, static_cast<const float*>(external_sum_weight.untyped_data()),
        rows, n, adaptive_fraction, static_cast<int>(keep_all != 0), static_cast<int>(use_external_sum_weight != 0),
        state_ptr, static_cast<float*>(sum_weight->untyped_data()), static_cast<float*>(threshold->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 threshold: ") + cudaGetErrorString(error));

    normalize_kernel<<<static_cast<int>(rows), kRowThreads, 0, stream>>>(
        scores_ptr, raw_ptr, state_ptr, n,
        static_cast<float*>(normalized->untyped_data()),
        static_cast<float*>(reconstruction->untyped_data()),
        static_cast<bool*>(mask->untyped_data()),
        static_cast<int32_t*>(n_significant->untyped_data()),
        static_cast<float*>(max_posterior->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(std::string("SparsePass2PosteriorF32 normalize: ") + cudaGetErrorString(error));
    return ffi::Error::Success();
}

}  // namespace recovar_sparse_pass2_posterior

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2LogZF64, recovar_sparse_pass2_posterior::log_z_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2PosteriorF32, recovar_sparse_pass2_posterior::posterior_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("adaptive_fraction")
        .Attr<int64_t>("keep_all")
        .Attr<int64_t>("use_external_sum_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
