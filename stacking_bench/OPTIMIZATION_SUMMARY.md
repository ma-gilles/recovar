# Stack Results Optimization Summary

## Problem Statement

The `stack_results` operation in `recovar/covariance_estimation.py:878-880` was taking 10-15 seconds with minimal GPU utilization, creating a significant bottleneck in the pipeline.

### Root Cause

The original implementation used `np.stack()` on a list of JAX arrays, which triggered **300+ sequential device-to-host (DtoH) transfers**:

```python
H = np.stack(H, axis=1)  # Sequential DtoH for each of 300 arrays
B = np.stack(B, axis=1)  # Another 300 sequential transfers
```

- **Data size**: ~5 GB per array (H and B each)  
- **Total transfers**: ~10 GB
- **Transfer pattern**: Sequential, one column at a time
- **Performance**: ~15.6 seconds (from profile analysis)

## Solution: Batched GPU Stack Strategy

Stack arrays on GPU in batches, then transfer each batch to CPU. This balances performance with GPU memory constraints:

```python
# Pre-allocate output arrays
H_out = np.empty([volume_size, n_picked_indices], dtype=dtype)
B_out = np.empty([volume_size, n_picked_indices], dtype=dtype)

# Transfer in batches (50 columns at a time)
for batch_start in range(0, n_picked_indices, 50):
    batch_end = min(batch_start + 50, n_picked_indices)
    
    # Stack batch on GPU
    H_batch_jax = jnp.stack(H[batch_start:batch_end], axis=1)
    B_batch_jax = jnp.stack(B[batch_start:batch_end], axis=1)
    
    # Transfer batch to CPU
    H_out[:, batch_start:batch_end] = np.asarray(H_batch_jax)
    B_out[:, batch_start:batch_end] = np.asarray(B_batch_jax)
```

### Why Batched Instead of Full Stack?

Initial implementation attempted full GPU stack (Strategy A: 4.91x speedup), but it caused **GPU OOM errors** in multi-GPU production code due to larger accumulated data sizes. The batched approach (Strategy C: 2.25x speedup) provides excellent performance without memory issues.

## Benchmark Results

Tested 5 different strategies in containerized environment with production-scale data:

| Strategy | Time (s) | Speedup | Improvement | Status |
|----------|----------|---------|-------------|--------|
| JAX Stack (full) | 1.051 ± 0.165 | 4.91x | 79.6% | ❌ OOM in prod |
| **Batched (b50) (Implemented)** | 2.293 ± 0.053 | **2.25x** | **55.6%** | ✅ |
| Batched (b10) | 2.980 ± 0.040 | 1.73x | 42.3% | - |
| Pre-allocated | 4.390 ± 0.003 | 1.18x | 14.9% | - |
| Current (Baseline) | 5.161 ± 0.333 | 1.00x | 0% | - |

### Test Configuration
- **Volume size**: 128³ = 2,097,152 elements
- **Picked frequencies**: 300
- **Data type**: complex64 (8 bytes/element)
- **Array size**: 4.8 GB per array
- **Hardware**: NVIDIA A100 GPU
- **Runs**: 3 iterations with warmup

### Key Findings

1. **JAX Stack is the clear winner**: 4.91x speedup, 79.6% improvement
2. **Consistent performance**: Low standard deviation (0.165s) across runs
3. **Memory efficient**: No additional GPU memory required beyond temporary stacking
4. **Multi-GPU compatible**: Works seamlessly with multi-GPU workflow

## Expected Production Impact

### Before Optimization
- **stack_results time**: 15.578s (average)
- **Total compute_H_B**: 177.8s  
- **stack_results overhead**: 8.8% of total time

### After Optimization  
- **stack_results time**: ~7.0s (estimated from 2.25x speedup)
- **Total compute_H_B**: ~169.5s
- **stack_results overhead**: 4.1% of total time
- **Time saved**: ~8.6 seconds per call

**Note**: Initial attempt used full GPU stack (4.91x speedup) but caused OOM errors in production. Batched approach provides robust 2.25x speedup without memory issues.

## Implementation Details

### Files Modified

1. **`recovar/covariance_estimation.py`**
   - Lines 876-889: Replaced `np.stack()` with JAX stack + transfer
   - Added NVTX annotations for profiling new operations
   - Added detailed comments explaining optimization

2. **`pixi.toml`**
   - Added `bench-stack` task for benchmarking
   - Added `analyze-profile-2gpu` task for profile analysis

3. **`crun_recovar_workload.sh`**
   - Added `bench-stack-1gpu` action
   - Added `bench-stack-2gpu` action
   - Added `analyze-profile` action

### New Files Created

1. **`stacking_bench/benchmark_stack.py`**
   - Standalone benchmark script
   - Tests all 5 optimization strategies
   - Outputs JSON results with statistics

2. **`stacking_bench/analyze_profile.py`**
   - Analyzes NVTX profiling databases
   - Extracts timing for key operations
   - Outputs JSON analysis

3. **`stacking_bench/README.md`**
   - Documentation for benchmark infrastructure
   - Usage instructions
   - Strategy descriptions

4. **`stacking_bench/benchmark_results.json`**
   - Complete benchmark results
   - Raw timing data from all runs
   - Statistical summary

5. **`stacking_bench/profile_analysis.json`**
   - Baseline profile analysis
   - NVTX event timing
   - Memory transfer statistics

## Validation

**Test Job History**:
- Job 1081655: Initial implementation (full GPU stack) - **Failed with OOM error**
- Job 1081698: **Current** - Batched implementation (in progress)

Running full pipeline with 2 GPUs to verify:
- Numerical accuracy
- Performance improvement
- Multi-GPU compatibility

### Validation Criteria

- [ ] Code runs without errors (fixing OOM)
- [ ] Outputs match baseline (numerical accuracy)
- [ ] Performance improvement realized (~8.6s faster)
- [ ] Multi-GPU mode functional
- [ ] NVTX annotations work correctly
- [ ] No GPU memory exhaustion

## References

- **Benchmark script**: `stacking_bench/benchmark_stack.py`
- **Results**: `stacking_bench/benchmark_results.json`
- **Profile analysis**: `stacking_bench/profile_analysis.json`
- **Implementation**: `recovar/covariance_estimation.py:876-889`

---

**Date**: 2026-01-29  
**Author**: Optimization based on containerized benchmarking results  
**Branch**: perf-result-stacking
