# Stack Optimization Benchmarks

This directory contains benchmarking tools to test optimization strategies for the `stack_results` operation in `recovar/covariance_estimation.py:878-880`.

## Problem

The current implementation uses `np.stack()` on a list of JAX arrays, which triggers 300+ sequential device-to-host transfers taking 10-15 seconds with minimal GPU utilization.

## Contents

- `benchmark_stack.py` - Standalone benchmark comparing different stacking strategies
- `analyze_profile.py` - Extract timing data from existing profiling databases
- `README.md` - This file

## Usage

### Running Benchmarks (Containerized)

Submit benchmark jobs using the crun workflow:

```bash
# 1 GPU benchmark
./crun_recovar_workload.sh bench-stack-1gpu

# 2 GPU benchmark
./crun_recovar_workload.sh bench-stack-2gpu

# Monitor job output
tail -f scripts/output/slurm-*.out

# View results
cat stacking_bench/benchmark_results.json
```

### Analyzing Existing Profiles

```bash
# Analyze profiling database
./crun_recovar_workload.sh analyze-profile

# View analysis
cat stacking_bench/profile_analysis.json
```

### Local Testing (Development)

If running locally with GPU access:

```bash
# Install dependencies
pixi install
pixi run install-recovar

# Run benchmark
pixi run bench-stack

# Analyze profile
pixi run analyze-profile-2gpu
```

## Strategies Tested

1. **Current (Baseline)**: `np.stack()` on list of JAX arrays
   - Sequential device-to-host transfers
   - Expected: 10-15 seconds

2. **JAX Stack**: Stack on GPU, then single transfer
   - Parallel GPU stacking operation
   - One bulk DtoH transfer
   - Expected: 2-4 seconds (70-80% improvement)
   - ⚠️ Can cause OOM with large arrays

3. **Pre-allocated**: Pre-allocate NumPy array, assign columns
   - Explicit memory allocation
   - Sequential but controlled transfers
   - Expected: 6-9 seconds (30-50% improvement)

4. **Batched (10)**: Transfer in batches of 10 columns
   - Balance between memory and transfer efficiency
   - Expected: 5-7 seconds (50-60% improvement)

5. **Batched (50)**: Transfer in batches of 50 columns ✅ **IMPLEMENTED**
   - Larger batches for fewer transfers
   - Expected: 4-6 seconds (60-70% improvement)
   - Memory safe for production

6. **Async Transfer**: Overlapped CPU/GPU operations
   - Background thread handles transfers
   - Can overlap with other CPU work
   - Expected: Similar to batched, depends on overlap opportunities

## Advanced Testing

### Test with Production-Scale Data
```bash
# Test with larger arrays (256^3 volume)
./crun_recovar_workload.sh bench-stack-large

# Test async strategy
./crun_recovar_workload.sh bench-stack-async
```

### CLI Options
```bash
# Custom volume size and picked frequencies
python stacking_bench/benchmark_stack.py \
  --volume-size 16777216 \
  --n-picked 300 \
  --test-async \
  --skip-oom-risk \
  --runs 3
```

## Data Characteristics

- **Volume size**: 128³ = 2,097,152 elements
- **Picked frequencies**: ~300 (from `sampling_n_cols` config)
- **Array size**: ~5 GB per array (H or B)
- **Total transfer**: ~10 GB (H + B)

## Next Steps

1. Run containerized benchmarks
2. Analyze results and choose best strategy
3. Implement chosen strategy in `recovar/covariance_estimation.py`
4. Profile optimized version
5. Validate numerical accuracy

## Related Files

- [`recovar/covariance_estimation.py:878-880`](../recovar/covariance_estimation.py) - Current implementation
- [`pixi.toml`](../pixi.toml) - Task definitions
- [`crun_recovar_workload.sh`](../crun_recovar_workload.sh) - Job submission script
