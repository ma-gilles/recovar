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
// The same two handlers also exist in a segmented form for the device-resident
// pass-2 data model, in which image i owns a contiguous run of a flat cell
// array instead of a padded rectangular row:
//
//   SparsePass2SegmentedLogZF64      <-> SparsePass2LogZF64 per segment
//   SparsePass2SegmentedPosteriorF32 <-> SparsePass2PosteriorF32 per segment
//
// Both call the shared bodies below on the segment's slice, so flattening a
// rectangular row into a segment reproduces the rectangular outputs bitwise.
// See the segmented section for the one host synchronization it needs.
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
// Shared body of the per-row float64 log-sum-exp.  The rectangular and the
// segmented kernel differ only in how they locate their row, so both inline
// this one copy of the arithmetic.
__device__ __forceinline__ void log_z_body(
    const float* row, int64_t n, double* log_z_slot,
    float* shared_values, int64_t* shared_indices, double* shared_sums)
{
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
        *log_z_slot = has_mass
            ? static_cast<double>(safe_best) + log(sum_exp)
            : -CUDART_INF;
    }
}

__global__ void log_z_kernel(const float* scores, int64_t n, double* log_z)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    __shared__ double shared_sums[kRowThreads];
    log_z_body(scores + static_cast<int64_t>(blockIdx.x) * n, n, log_z + blockIdx.x,
               shared_values, shared_indices, shared_sums);
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

// Shared body of the row-max / normalization state.  Every thread of the block
// must call it; only thread 0 writes.
__device__ __forceinline__ void row_max_body(
    const float* row, int64_t n, double external_log_z,
    float* shared_values, int64_t* shared_indices,
    RowState* state_slot, double* log_z_slot, float* best_log_score_slot,
    int64_t* best_argmax_slot)
{
    float best;
    int64_t best_index;
    block_finite_argmax(row, n, shared_values, shared_indices, best, best_index);
    if (threadIdx.x != 0) return;
    const bool has_finite = isfinite(best);
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
    *state_slot = s;
    // _normalize_pass2_bucket_with_log_z outputs.
    *log_z_slot = s.safe_log_z;
    *best_log_score_slot = has_finite_norm ? best : -CUDART_INF_F;
    *best_argmax_slot = has_finite_norm ? best_index : 0;
}

__global__ void row_max_kernel(
    const float* scores, const double* log_z, int64_t n, RowState* state,
    double* log_z_out, float* best_log_score, int64_t* best_argmax)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    const int64_t row_index = blockIdx.x;
    row_max_body(scores + row_index * n, n, log_z[row_index], shared_values, shared_indices,
                 state + row_index, log_z_out + row_index, best_log_score + row_index,
                 best_argmax + row_index);
}

// Shared per-cell exponentiation of one candidate score.
__device__ __forceinline__ void exponentiate_cell(
    const RowState& s, float value, float* raw_slot, double* prob_slot)
{
    const bool finite = isfinite(value);
    const float finite_score = finite ? value : -CUDART_INF_F;
    // relion_exponentiate_batched_f32_kernel: expf(value + add), zero below -88.
    const float exponent = finite_score + s.exponent_add;
    *raw_slot = exponent < -88.0f ? 0.0f : expf(exponent);
    // _normalize_pass2_bucket_with_log_z: exp(scores - safe_log_z) in float64,
    // zero when the row has no finite normalizer or the value overflows.
    double p = 0.0;
    if (s.has_finite_norm)
    {
        p = exp(static_cast<double>(finite_score) - s.safe_log_z);
        if (!isfinite(p)) p = 0.0;
    }
    *prob_slot = p;
}

__global__ void exponentiate_kernel(
    const float* scores, const RowState* state, int64_t n, int64_t count,
    float* raw_weights, double* probs)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    exponentiate_cell(state[index / n], scores[index], raw_weights + index, probs + index);
}

// Shared RELION significance boundary for one already sorted/scanned row.
// ``n == 0`` is the empty-segment case of the segmented handler; the
// rectangular handler always passes ``n >= 1`` and reaches the same code.
__device__ __forceinline__ void threshold_body(
    const float* row_sorted, const float* row_cumulative, int64_t n,
    float external_sum_weight, float adaptive_fraction, int keep_all,
    int use_external_sum_weight, RowState& s,
    float* sum_weight_slot, float* threshold_slot)
{
    const float fine_sum_weight = n > 0 ? row_cumulative[n - 1] : 0.0f;
    const float sum_weight = use_external_sum_weight
        ? external_sum_weight
        : fine_sum_weight;
    const bool has_mass = s.has_finite && isfinite(sum_weight) && sum_weight > 0.0f;
    float threshold = 0.0f;
    if (!keep_all && n > 0)
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
    *sum_weight_slot = sum_weight;
    *threshold_slot = threshold;
}

__global__ void threshold_kernel(
    const float* sorted, const float* cumulative, const float* external_sum_weight,
    int64_t rows, int64_t n, float adaptive_fraction, int keep_all, int use_external_sum_weight,
    RowState* state, float* sum_weight_out, float* threshold_out)
{
    const int64_t row_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_index >= rows) return;
    RowState s = state[row_index];
    threshold_body(sorted + row_index * n, cumulative + row_index * n, n,
                   external_sum_weight[row_index], adaptive_fraction, keep_all,
                   use_external_sum_weight, s, sum_weight_out + row_index,
                   threshold_out + row_index);
    state[row_index] = s;
}

// Shared normalization / pruning body.  All pointers are already offset to the
// start of the row or segment; every thread of the block must call it.
__device__ __forceinline__ void normalize_body(
    const float* row_scores, const float* row_raw, const RowState& s, int64_t n,
    float* row_normalized, float* row_reconstruction, bool* row_mask,
    int32_t* n_significant_slot, float* max_posterior_slot,
    int* shared_counts, float* shared_max)
{
    int local_count = 0;
    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float raw = row_raw[i];
        // relion_divide_f32_kernel: IEEE float32 division by the row sum.
        const float weight = raw / s.safe_sum_weight;
        const bool finite = isfinite(row_scores[i]);
        const bool significant = s.has_mass && finite &&
            (s.keep_all ? (raw > 0.0f) : (raw >= s.threshold));
        const float kept = significant ? weight : 0.0f;
        row_normalized[i] = weight;
        row_reconstruction[i] = kept;
        row_mask[i] = significant;
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
        *n_significant_slot = shared_counts[0];
        *max_posterior_slot = shared_max[0];
    }
}

__global__ void normalize_kernel(
    const float* scores, const float* raw_weights, const RowState* state, int64_t n,
    float* normalized, float* reconstruction, bool* mask,
    int32_t* n_significant, float* max_posterior)
{
    __shared__ int shared_counts[kRowThreads];
    __shared__ float shared_max[kRowThreads];
    const int64_t row_index = blockIdx.x;
    const int64_t base = row_index * n;
    normalize_body(scores + base, raw_weights + base, state[row_index], n,
                   normalized + base, reconstruction + base, mask + base,
                   n_significant + row_index, max_posterior + row_index,
                   shared_counts, shared_max);
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


// ---------------------------------------------------------------------------
// Segmented variants.
//
// The rectangular handlers above own a dense [B, M] score block: image i owns
// row i and every image pays for the widest row.  The segmented handlers take
// the same scores as one flat array of cells in which image i owns the
// contiguous run [segment_offsets[i], segment_offsets[i + 1]).  Cells inside a
// run are the CSR rows x translations of the device-resident pass-2 data model,
// so a chunk no longer pads every image to a common row count.
//
// Every formula is the shared body of the rectangular path, called on the
// segment's slice, so flattening a rectangular row into a segment reproduces
// the rectangular outputs bitwise.  Two consequences of that choice:
//
//   * The significance boundary keeps the per-segment cub::DeviceRadixSort::
//     SortKeys plus relion_ampere_inclusive_sum_f32 of the rectangular path.
//     cub::DeviceSegmentedRadixSort could replace the sort, but CUB has no
//     segmented inclusive scan whose float32 summation reproduces the
//     single-segment decoupled-lookback order, and a differing cumulative
//     array moves the searchsorted boundary.  Since the scan must run once per
//     segment with a host-side length, the sort uses the same loop, which makes
//     the sorted key order identical by construction instead of by argument.
//   * The host therefore needs the segment lengths: the posterior handler
//     copies segment_offsets back and synchronizes the stream once per call.
//     This is the one host round trip left in the device-resident chunk body,
//     and it cannot be removed while the significance boundary is CUB's
//     decoupled-lookback float32 scan: that scan's summation order depends on
//     the segment length, sum_weight is cumulative[n - 1] and the threshold is
//     a searchsorted over the same array, so a device-side segmented scan (a
//     ScanByKey, or one scan over the whole chunk) moves both and breaks the
//     bitwise contract with the rectangular handler and with RELION.  The
//     launch geometry is capacity-only: every kernel is launched at the static
//     segment count, the scratch is sized at the static cell count, and
//     n_valid_images is read only on the device, so nothing on the host
//     depends on the chunk's occupancy.  To keep the stall off the device, the
//     handler enqueues the max and exponentiate kernels, and the CUB scratch
//     allocation, before it synchronizes, so the device is working on this
//     call's own kernels while the host waits.  The log-Z handler needs no
//     lengths on the host and stays device-resident.
//
// Empty segments and images at or beyond n_valid_images produce exactly the
// values the rectangular path produces for an all -inf row.  Cells covered by
// no segment are written as zero / false.
// ---------------------------------------------------------------------------

struct SegmentState
{
    RowState row;
    int valid;      // segment index < n_valid_images
    int reserved;   // explicit padding, keeps begin 8-byte aligned
    int64_t begin;  // first cell of the segment, clamped into [0, n_cells]
    int64_t n;      // cell count, 0 for empty or invalid segments
};

// Last index i with offsets[i] <= cell, or -1 when no segment covers the cell.
// Empty segments are never returned: offsets[i] == offsets[i + 1] cannot
// bracket a cell.
__device__ __forceinline__ int find_segment(
    const int32_t* offsets, int n_segments, int64_t cell)
{
    int low = 0;
    int high = n_segments + 1;
    while (low < high)
    {
        const int mid = (low + high) >> 1;
        if (static_cast<int64_t>(offsets[mid]) <= cell) low = mid + 1; else high = mid;
    }
    const int index = low - 1;
    if (index < 0 || index >= n_segments) return -1;
    return index;
}

__device__ __forceinline__ void segment_extent(
    const int32_t* offsets, int segment, int valid_count, int64_t n_cells,
    int64_t& begin, int64_t& n)
{
    // Defensive clamping: a malformed offset table must not read out of bounds.
    int64_t first = static_cast<int64_t>(offsets[segment]);
    int64_t last = static_cast<int64_t>(offsets[segment + 1]);
    if (first < 0) first = 0;
    if (first > n_cells) first = n_cells;
    if (last < first) last = first;
    if (last > n_cells) last = n_cells;
    begin = first;
    n = segment < valid_count ? last - first : 0;
}

__device__ __forceinline__ int clamp_valid_count(int32_t n_valid_images, int n_segments)
{
    if (n_valid_images < 0) return 0;
    if (n_valid_images > n_segments) return n_segments;
    return static_cast<int>(n_valid_images);
}

__global__ void segmented_log_z_kernel(
    const float* scores, const int32_t* offsets, const int32_t* n_valid_images,
    int n_segments, int64_t n_cells, double* log_z)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    __shared__ double shared_sums[kRowThreads];
    const int segment = blockIdx.x;
    const int valid_count = clamp_valid_count(*n_valid_images, n_segments);
    int64_t begin = 0, n = 0;
    segment_extent(offsets, segment, valid_count, n_cells, begin, n);
    log_z_body(scores + begin, n, log_z + segment, shared_values, shared_indices, shared_sums);
}

__global__ void segmented_max_kernel(
    const float* scores, const int32_t* offsets, const int32_t* n_valid_images,
    const double* log_z, int n_segments, int64_t n_cells, SegmentState* state,
    double* log_z_out, float* best_log_score, int64_t* best_argmax)
{
    __shared__ float shared_values[kRowThreads];
    __shared__ int64_t shared_indices[kRowThreads];
    const int segment = blockIdx.x;
    const int valid_count = clamp_valid_count(*n_valid_images, n_segments);
    int64_t begin = 0, n = 0;
    segment_extent(offsets, segment, valid_count, n_cells, begin, n);
    row_max_body(scores + begin, n, log_z[segment], shared_values, shared_indices,
                 &state[segment].row, log_z_out + segment, best_log_score + segment,
                 best_argmax + segment);
    if (threadIdx.x == 0)
    {
        state[segment].valid = segment < valid_count ? 1 : 0;
        state[segment].reserved = 0;
        state[segment].begin = begin;
        state[segment].n = n;
    }
}

__global__ void segmented_exponentiate_kernel(
    const float* scores, const int32_t* offsets, const SegmentState* state,
    int n_segments, int64_t n_cells, float* raw_weights, double* probs,
    float* sorted, float* cumulative, float* normalized, float* reconstruction,
    bool* mask)
{
    __shared__ int shared_segment;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (threadIdx.x == 0)
    {
        int64_t last = first + blockDim.x - 1;
        if (last > n_cells - 1) last = n_cells - 1;
        const int head = find_segment(offsets, n_segments, first);
        const int tail = find_segment(offsets, n_segments, last);
        shared_segment = head == tail ? head : -2;
    }
    __syncthreads();
    const int64_t index = first + threadIdx.x;
    if (index >= n_cells) return;
    int segment = shared_segment;
    if (segment == -2) segment = find_segment(offsets, n_segments, index);
    bool covered = false;
    if (segment >= 0)
    {
        const int64_t begin = state[segment].begin;
        covered = state[segment].valid != 0 && index >= begin &&
                  index < begin + state[segment].n;
    }
    if (!covered)
    {
        // Padding cells, empty segments and images past n_valid_images behave
        // like an all -inf rectangular row: zero weight, zero probability, no
        // significant candidate.  normalized/reconstruction/mask are written
        // here because the normalize kernel only walks live segments.
        raw_weights[index] = 0.0f;
        probs[index] = 0.0;
        sorted[index] = 0.0f;
        cumulative[index] = 0.0f;
        normalized[index] = 0.0f;
        reconstruction[index] = 0.0f;
        mask[index] = false;
        return;
    }
    exponentiate_cell(state[segment].row, scores[index], raw_weights + index, probs + index);
}

__global__ void segmented_threshold_kernel(
    const float* sorted, const float* cumulative, const float* external_sum_weight,
    int n_segments, float adaptive_fraction, int keep_all, int use_external_sum_weight,
    SegmentState* state, float* sum_weight_out, float* threshold_out)
{
    const int segment = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (segment >= n_segments) return;
    SegmentState s = state[segment];
    threshold_body(sorted + s.begin, cumulative + s.begin, s.n,
                   external_sum_weight[segment], adaptive_fraction, keep_all,
                   use_external_sum_weight, s.row, sum_weight_out + segment,
                   threshold_out + segment);
    state[segment] = s;
}

__global__ void segmented_normalize_kernel(
    const float* scores, const float* raw_weights, const SegmentState* state,
    float* normalized, float* reconstruction, bool* mask,
    int32_t* n_significant, float* max_posterior)
{
    __shared__ int shared_counts[kRowThreads];
    __shared__ float shared_max[kRowThreads];
    const int segment = blockIdx.x;
    // Read the extent as scalars and keep the RowState reference in global
    // memory, as the rectangular kernel does.  Copying SegmentState into a
    // local and passing its address puts the record in local memory, which the
    // inner loop then reloads once per cell; on a one-block 7.3M-cell segment
    // that alone cost ~1.9 ms.
    const int64_t begin = state[segment].begin;
    const int64_t n = state[segment].n;
    normalize_body(scores + begin, raw_weights + begin, state[segment].row, n,
                   normalized + begin, reconstruction + begin, mask + begin,
                   n_significant + segment, max_posterior + segment,
                   shared_counts, shared_max);
}

inline bool flat_geometry(const ffi::Span<const int64_t> dims, int64_t& count)
{
    count = 1;
    for (size_t i = 0; i < dims.size(); ++i) count *= dims[i];
    return dims.size() >= 1 && count >= 1;
}

inline bool segmented_vector_ok(
    const ffi::AnyBuffer& buffer, ffi::DataType type, int64_t length)
{
    return buffer.element_type() == type && buffer.dimensions().size() == 1 &&
           buffer.dimensions()[0] == length;
}

inline bool segmented_flat_ok(
    const ffi::AnyBuffer& buffer, ffi::DataType type, int64_t count)
{
    int64_t total = 0;
    return buffer.element_type() == type && flat_geometry(buffer.dimensions(), total) &&
           total == count;
}

inline bool scalar_ok(const ffi::AnyBuffer& buffer, ffi::DataType type)
{
    int64_t total = 1;
    for (size_t i = 0; i < buffer.dimensions().size(); ++i) total *= buffer.dimensions()[i];
    return buffer.element_type() == type && total == 1;
}

ffi::Error segmented_log_z_impl(
    cudaStream_t stream,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer segment_offsets,
    ffi::AnyBuffer n_valid_images,
    ffi::Result<ffi::AnyBuffer> log_z)
{
    int64_t n_cells = 0;
    if (scores.element_type() != ffi::DataType::F32 || !flat_geometry(scores.dimensions(), n_cells))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: scores must be a nonempty F32 array");
    if (n_cells > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: cell count exceeds the int32 offset range");
    if (segment_offsets.element_type() != ffi::DataType::S32 ||
        segment_offsets.dimensions().size() != 1 || segment_offsets.dimensions()[0] < 2)
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: segment_offsets must be S32 with shape (n_segments + 1,)");
    const int64_t n_segments = segment_offsets.dimensions()[0] - 1;
    if (n_segments > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: too many segments for one launch grid");
    if (!scalar_ok(n_valid_images, ffi::DataType::S32))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: n_valid_images must be a single S32 value");
    if (!segmented_vector_ok(*log_z, ffi::DataType::F64, n_segments))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedLogZF64: log_z must be F64 with shape (n_segments,)");
    segmented_log_z_kernel<<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        static_cast<const float*>(scores.untyped_data()),
        static_cast<const int32_t*>(segment_offsets.untyped_data()),
        static_cast<const int32_t*>(n_valid_images.untyped_data()),
        static_cast<int>(n_segments), n_cells,
        static_cast<double*>(log_z->untyped_data()));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedLogZF64: ") + cudaGetErrorString(error));
    return ffi::Error::Success();
}

ffi::Error segmented_posterior_impl(
    cudaStream_t stream,
    float adaptive_fraction,
    int64_t keep_all,
    int64_t use_external_sum_weight,
    ffi::AnyBuffer scores,
    ffi::AnyBuffer segment_offsets,
    ffi::AnyBuffer n_valid_images,
    ffi::AnyBuffer log_z,
    ffi::AnyBuffer external_sum_weight,
    ffi::Result<ffi::AnyBuffer> log_z_out,
    ffi::Result<ffi::AnyBuffer> best_log_score,
    ffi::Result<ffi::AnyBuffer> best_cell_index,
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
    ffi::Result<ffi::AnyBuffer> segment_state)
{
    int64_t n_cells = 0;
    if (scores.element_type() != ffi::DataType::F32 || !flat_geometry(scores.dimensions(), n_cells))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: scores must be a nonempty F32 array");
    if (n_cells > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: cell count exceeds the int32 offset range");
    if (segment_offsets.element_type() != ffi::DataType::S32 ||
        segment_offsets.dimensions().size() != 1 || segment_offsets.dimensions()[0] < 2)
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: segment_offsets must be S32 with shape (n_segments + 1,)");
    const int64_t n_segments = segment_offsets.dimensions()[0] - 1;
    if (n_segments > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: too many segments for one launch grid");
    if (!scalar_ok(n_valid_images, ffi::DataType::S32))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: n_valid_images must be a single S32 value");
    if (!segmented_vector_ok(log_z, ffi::DataType::F64, n_segments) ||
        !segmented_vector_ok(external_sum_weight, ffi::DataType::F32, n_segments))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: log_z must be F64 (n_segments,) and external_sum_weight F32 (n_segments,)");
    if (!segmented_vector_ok(*log_z_out, ffi::DataType::F64, n_segments) ||
        !segmented_vector_ok(*best_log_score, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(*best_cell_index, ffi::DataType::S64, n_segments) ||
        !segmented_vector_ok(*max_posterior, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(*n_significant, ffi::DataType::S32, n_segments) ||
        !segmented_vector_ok(*sum_weight, ffi::DataType::F32, n_segments) ||
        !segmented_vector_ok(*threshold, ffi::DataType::F32, n_segments))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: per-segment outputs must have shape (n_segments,) with the documented dtypes");
    if (!segmented_flat_ok(*probs, ffi::DataType::F64, n_cells) ||
        !segmented_flat_ok(*normalized, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(*reconstruction, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(*mask, ffi::DataType::PRED, n_cells) ||
        !segmented_flat_ok(*raw_weights, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(*sorted, ffi::DataType::F32, n_cells) ||
        !segmented_flat_ok(*cumulative, ffi::DataType::F32, n_cells))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: per-cell outputs must hold n_cells elements with the documented dtypes");
    if (segment_state->element_type() != ffi::DataType::U8 ||
        segment_state->dimensions().size() != 2 ||
        segment_state->dimensions()[0] != n_segments ||
        segment_state->dimensions()[1] != static_cast<int64_t>(sizeof(SegmentState)))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: segment_state scratch must be U8 (n_segments, sizeof(SegmentState))");

    const float* scores_ptr = static_cast<const float*>(scores.untyped_data());
    const int32_t* offsets_ptr = static_cast<const int32_t*>(segment_offsets.untyped_data());
    SegmentState* state_ptr = static_cast<SegmentState*>(segment_state->untyped_data());
    float* raw_ptr = static_cast<float*>(raw_weights->untyped_data());
    float* sorted_ptr = static_cast<float*>(sorted->untyped_data());
    float* cumulative_ptr = static_cast<float*>(cumulative->untyped_data());

    // Stage 1 and 2 need nothing from the host: launch them, and reserve the
    // CUB scratch at the static cell count, before the offsets readback, so the
    // device works on this call's own kernels while the host waits for it.
    segmented_max_kernel<<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        scores_ptr, offsets_ptr,
        static_cast<const int32_t*>(n_valid_images.untyped_data()),
        static_cast<const double*>(log_z.untyped_data()),
        static_cast<int>(n_segments), n_cells, state_ptr,
        static_cast<double*>(log_z_out->untyped_data()),
        static_cast<float*>(best_log_score->untyped_data()),
        static_cast<int64_t*>(best_cell_index->untyped_data()));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 segment max: ") + cudaGetErrorString(error));

    constexpr int threads = 256;
    const int64_t block_count = (n_cells + threads - 1) / threads;
    if (block_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        return ffi::Error::InvalidArgument(
            "SparsePass2SegmentedPosteriorF32: launch grid exceeds CUDA limit");
    segmented_exponentiate_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
        scores_ptr, offsets_ptr, state_ptr, static_cast<int>(n_segments), n_cells,
        raw_ptr, static_cast<double*>(probs->untyped_data()), sorted_ptr, cumulative_ptr,
        static_cast<float*>(normalized->untyped_data()),
        static_cast<float*>(reconstruction->untyped_data()),
        static_cast<bool*>(mask->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 exponentiate: ") + cudaGetErrorString(error));

    // Scratch for the per-segment sort and scan, sized at the capacity: a
    // query at n_cells bounds every segment of this chunk, so the allocation
    // and its query are the same for every chunk of a capacity class instead
    // of following the longest segment of this one.
    const int capacity_count = static_cast<int>(n_cells);
    size_t sort_bytes = 0;
    size_t scan_bytes = 0;
    error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, raw_ptr, sorted_ptr, capacity_count, 0, sizeof(float) * 8, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 sort query: ") + cudaGetErrorString(error));
    error = relion_ampere_inclusive_sum_f32(
        nullptr, scan_bytes, sorted_ptr, cumulative_ptr, capacity_count, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 scan query: ") + cudaGetErrorString(error));
    void* temporary = nullptr;
    const size_t temporary_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
    error = cudaMallocAsync(&temporary, temporary_bytes, stream);
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 cudaMallocAsync: ") + cudaGetErrorString(error));

    // The one host round trip: the per-segment CUB calls take their item count
    // on the host, and that count is what fixes the float32 scan order the
    // significance boundary is defined by.  n_valid_images is NOT read back:
    // segments at or past it carry no cells on the device (segment_extent
    // clamps them), and sorting a range the device treats as empty writes only
    // into scratch that the threshold body never reads for that segment.
    std::vector<int32_t> host_offsets(static_cast<size_t>(n_segments) + 1, 0);
    error = cudaMemcpyAsync(host_offsets.data(), offsets_ptr,
                            host_offsets.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(temporary, stream);
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 offset readback: ") + cudaGetErrorString(error));
    }
    int64_t previous = 0;
    for (size_t i = 0; i < host_offsets.size(); ++i)
    {
        const int64_t offset = host_offsets[i];
        if (offset < previous || offset > n_cells)
        {
            cudaFreeAsync(temporary, stream);
            return ffi::Error::InvalidArgument(
                "SparsePass2SegmentedPosteriorF32: segment_offsets must be nondecreasing within [0, n_cells]");
        }
        previous = offset;
    }

    // Same CUB radix sort and pinned Ampere inclusive scan as the rectangular
    // handler, once per nonempty segment on the caller's stream.
    for (int64_t segment = 0; segment < n_segments && error == cudaSuccess; ++segment)
    {
        const int64_t begin = host_offsets[static_cast<size_t>(segment)];
        const int64_t length = host_offsets[static_cast<size_t>(segment) + 1] - begin;
        if (length <= 0) continue;
        const int segment_count = static_cast<int>(length);
        error = cub::DeviceRadixSort::SortKeys(
            temporary, sort_bytes, raw_ptr + begin, sorted_ptr + begin, segment_count,
            0, sizeof(float) * 8, stream);
        if (error == cudaSuccess)
            error = relion_ampere_inclusive_sum_f32(
                temporary, scan_bytes, sorted_ptr + begin, cumulative_ptr + begin,
                segment_count, stream);
    }
    {
        const cudaError_t free_error = cudaFreeAsync(temporary, stream);
        if (error != cudaSuccess)
            return ffi::Error::Internal(
                std::string("SparsePass2SegmentedPosteriorF32 sort/scan: ") + cudaGetErrorString(error));
        if (free_error != cudaSuccess)
            return ffi::Error::Internal(
                std::string("SparsePass2SegmentedPosteriorF32 cudaFreeAsync: ") + cudaGetErrorString(free_error));
    }

    constexpr int segment_threads = 128;
    const int segment_blocks = static_cast<int>((n_segments + segment_threads - 1) / segment_threads);
    segmented_threshold_kernel<<<segment_blocks, segment_threads, 0, stream>>>(
        sorted_ptr, cumulative_ptr,
        static_cast<const float*>(external_sum_weight.untyped_data()),
        static_cast<int>(n_segments), adaptive_fraction,
        static_cast<int>(keep_all != 0), static_cast<int>(use_external_sum_weight != 0),
        state_ptr, static_cast<float*>(sum_weight->untyped_data()),
        static_cast<float*>(threshold->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 threshold: ") + cudaGetErrorString(error));

    segmented_normalize_kernel<<<static_cast<int>(n_segments), kRowThreads, 0, stream>>>(
        scores_ptr, raw_ptr, state_ptr,
        static_cast<float*>(normalized->untyped_data()),
        static_cast<float*>(reconstruction->untyped_data()),
        static_cast<bool*>(mask->untyped_data()),
        static_cast<int32_t*>(n_significant->untyped_data()),
        static_cast<float*>(max_posterior->untyped_data()));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return ffi::Error::Internal(
            std::string("SparsePass2SegmentedPosteriorF32 normalize: ") + cudaGetErrorString(error));
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2SegmentedLogZF64, recovar_sparse_pass2_posterior::segmented_log_z_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SparsePass2SegmentedPosteriorF32, recovar_sparse_pass2_posterior::segmented_posterior_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("adaptive_fraction")
        .Attr<int64_t>("keep_all")
        .Attr<int64_t>("use_external_sum_weight")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
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
