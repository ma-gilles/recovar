# RECOVAR Performance Profiling Results Summary

## Completed Profiling Jobs

### 256x300k Dataset (Large Scale)

| Job ID | GPUs | Runtime | Status | Storage | Lazy Loading | Profile Size |
|--------|------|---------|--------|---------|--------------|--------------|
| 4355455 | 1 | 6h 15m | ✅ COMPLETED | /tigress | Enabled | 9.8 GB |
| 4355456 | 2 | 4h 32m | ✅ COMPLETED | /tigress | Enabled | 10 GB |
| 4355457 | 4 | 9h 45m | ✅ COMPLETED | /tigress | Enabled | 11 GB |

**Note**: The 4-GPU run took longer than expected (9h45m vs 4h32m for 2-GPU), suggesting potential scaling inefficiencies or different workload characteristics.

### 128x100k Dataset (Test Scale)

| Job ID | GPUs | Runtime | Status | Storage |
|--------|------|---------|--------|---------|
| 4333613 | 1 | 49m 30s | ✅ COMPLETED | /scratch |
| 4341486 | 1 | 55m 12s | ✅ COMPLETED | /scratch |
| 4341487 | 2 | 52m 22s | ✅ COMPLETED | /scratch |
| 4341488 | 4 | 54m 0s | ✅ COMPLETED | /scratch |

### Pending/Failed Jobs

| Job ID | GPUs | Status | Issue |
|--------|------|--------|-------|
| 4548283 | 1 | ❌ FAILED | IndentationError (fixed) |
| 4548284 | 1 | ❌ FAILED | IndentationError (fixed) |

**Status**: Code fixes applied. Jobs ready for resubmission.

---

## Detailed Analysis: 1 GPU (256-300k dataset, Job 4355455)

### Total Runtime
~6 hours 15 minutes

## Top Time Consumers (by NVTX ranges)

| Time % | Function | Total Time | Instances | Avg Time |
|--------|----------|------------|-----------|----------|
| 12.0% | `principal_components:estimate_principal_components` | 13.6s | 1 | 13.6s |
| 8.1% | `data_io:collate_to_jax` | 9.2s | 218,914 | 42ms |
| 8.1% | `data_io:numpy_to_jax_transfer` | 9.2s | 218,914 | 42ms |
| 7.6% | `compute_regularized_covariance_columns_in_batch` | 8.7s | 1 | 8.7s |
| 7.4% | `compute_both_H_B` | 8.4s | 3 | 2.8s |
| 7.4% | `compute_H_B_in_volume_batch` | 8.4s | 6 | 1.4s |
| 6.9% | `compute_H_B:compute_H_B` | 7.9s | 6 | 1.3s |
| 6.1% | `compute_H_B:frequency_loop` | 6.9s | 5,922 | 1.2s |
| 5.6% | `compute_H_B:accumulate_H_B` | 6.3s | 592,200 | 11ms |
| 3.3% | `embedding:get_per_image_embedding` | 3.7s | 10 | 374ms |
| 3.1% | `embedding:get_coords_in_basis_and_contrast` | 3.5s | 19 | 184ms |
| 3.1% | `principal_components:get_cov_svds` | 3.5s | 1 | 3.5s |

## IO Performance Analysis

### Data Loading Operations
- **`data_io:MRCLoader._load`**: 0.1% (151s total, 2 instances) - ~75s per load
- **`data_io:disk_read_150000_images`**: 0.1% (151s total, 2 instances) - ~75s per read
- **`data_io:random_access_read`**: 0.1% (151s total, 2 instances) - ~75s per read
- **`data_io:ParticleImageDataset.__getitem__`**: 0.1% (142s total, 5.4M instances) - ~26μs per item
- **`data_io:load_from_source`**: 0.1% (113s total, 5.4M instances) - ~21μs per item

### Data Transfer Operations (CPU↔GPU)
- **`data_io:collate_to_jax`**: **8.1%** (9.2s total, 218,914 instances) - **42ms avg per transfer**
- **`data_io:numpy_to_jax_transfer`**: **8.1%** (9.2s total, 218,914 instances) - **42ms avg per transfer**

**Key Finding**: The `collate_to_jax` and `numpy_to_jax_transfer` operations are major bottlenecks, taking 16.2% combined time with ~42ms per transfer. This suggests frequent small transfers that could benefit from batching or async transfer.

## Memory Transfer Summary

| Operation | Time % | Total Time | Count | Avg Time | Total Data |
|-----------|--------|------------|-------|----------|------------|
| Host-to-Device | 48.5% | 119.2s | 7.3M | 16.4μs | 2.3 TB |
| Device-to-Device | 41.5% | 102.1s | 6.4M | 16.0μs | 80.4 TB |
| Device-to-Host | 10.0% | 24.5s | 211k | 116.2μs | 614 GB |

**Key Findings**:
- **Host-to-Device transfers dominate** (48.5% of memory transfer time)
- Very high frequency of transfers (7.3M H2D, 6.4M D2D operations)
- Average transfer sizes are relatively small (avg ~0.3MB for H2D)
- Total data movement: ~83 TB (mostly D2D)

## Recommendations

### IO Optimization Opportunities
1. **Batch data transfers**: The 218,914 instances of `collate_to_jax` with 42ms each suggest frequent small transfers. Consider batching.
2. **Async transfers**: Overlap data loading with computation using CUDA streams.
3. **Reduce transfer frequency**: The 7.3M H2D transfers suggest many small operations that could be combined.

### Compute Optimization Opportunities
1. **`compute_H_B:accumulate_H_B`**: 5.6% time, 592k instances - could benefit from parallelization
2. **`principal_components:estimate_principal_components`**: 12% time, single instance - largest single bottleneck
3. **`compute_regularized_covariance_columns_in_batch`**: 7.6% time, single instance - good candidate for multi-GPU

## Files Generated
- `nvtx_summary.txt` - Full NVTX range breakdown
- `gpu_kernel_summary.txt` - GPU kernel performance
- `cuda_api_summary.txt` - CUDA API call statistics
- `memory_transfer_summary.txt` - Memory transfer details

---

## Executive Summary

### Key Performance Insights

1. **IO Bottleneck Identified**: 16.2% of total time spent on CPU↔GPU data transfers
   - 218,914 transfer operations averaging 42ms each
   - High frequency suggests batching opportunities

2. **Memory Transfer Patterns**:
   - 7.3M Host-to-Device transfers (48.5% of transfer time)
   - 6.4M Device-to-Device transfers (41.5% of transfer time)
   - Total data movement: ~83 TB

3. **Top Compute Bottlenecks**:
   - `principal_components:estimate_principal_components`: 12.0% (single instance)
   - `compute_regularized_covariance_columns_in_batch`: 7.6% (single instance)
   - `compute_H_B:accumulate_H_B`: 5.6% (592k instances - parallelization candidate)

4. **Multi-GPU Scaling**:
   - 2-GPU: 4h32m (1.4x speedup vs 1-GPU's 6h15m)
   - 4-GPU: 9h45m (slower than 2-GPU - scaling inefficiency detected)

### Profile File Locations

**Primary profiles (256-300k dataset)**:
```
/tigress/CRYOEM/singerlab/mg6942/recovar_profiling/data/data-256-300000/test_dataset/
├── recovar_1gpu_profile_4355455.nsys-rep (9.8GB)
├── recovar_2gpu_profile_4355456.nsys-rep (10GB)
└── recovar_4gpu_profile_4355457.nsys-rep (11GB)
```

**Reduced-size profiles (for GUI compatibility)**:
- Job 4548264: 245KB (failed due to code error, ready for resubmission)
- Job 4548284: 245KB (failed due to code error, ready for resubmission)

### GUI Access Status

**Issue**: Nsight Systems GUI reports version mismatch when opening profiles.

**Workarounds**:
1. Use matching version: `/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/nsys-ui`
2. Download profiles and open locally with Nsight Systems 2025.3.2+
3. Use command-line analysis (all stats available via `nsys stats`)

**Recent attempt**: Reduced profile size by disabling CPU sampling (`--sample=none`) to improve GUI compatibility. Jobs ready for resubmission after code fixes.

### Next Steps

1. ✅ **Code fixes applied**: Indentation errors in `simulator.py` fixed
2. 🔄 **Resubmit reduced-size profiling jobs** (with `--sample=none` for GUI compatibility)
3. 📊 **Analyze multi-GPU scaling inefficiency** (4-GPU slower than 2-GPU)
4. 🔍 **Deep dive into IO bottlenecks** (218k transfers, 42ms avg)
5. 🎯 **Test lazy loading impact** (compare enabled vs disabled)
6. 💾 **Compare storage performance** (scratch vs tigress)
