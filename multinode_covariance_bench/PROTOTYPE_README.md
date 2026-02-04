# Frequency-Parallel Prototype

## Overview

This prototype validates the frequency-parallel multi-node approach before full integration into the RECOVAR pipeline.

**Goal**: Prove that splitting covariance computation across nodes by frequency achieves near-linear speedup.

## Files

- `prototype_frequency_parallel.py` - Main prototype script
- `test_prototype.sh` - Automated test script
- `PROTOTYPE_PLAN.md` - Detailed implementation plan

## Quick Start

### Prerequisites

1. Dataset available at `data-128-100000/test_dataset/`
2. RECOVAR environment activated (pixi or conda)
3. JAX with GPU support

### Run the Test

```bash
# Run full automated test (recommended)
bash test_prototype.sh
```

This will:
1. Run baseline (1 node, all frequencies)
2. Run 2 nodes in parallel (frequency-split)
3. Concatenate results
4. Validate outputs match
5. Report timing and speedup

**Expected output**:
- Baseline: ~X seconds
- 2 nodes: ~X/2 seconds (each runs in parallel)
- Speedup: ~2× (near-linear scaling)

## Manual Testing

### Step 1: Run Baseline

```bash
python prototype_frequency_parallel.py \
    --node-rank 0 \
    --n-nodes 1 \
    --output-dir prototype_output/baseline \
    --n-images 1000
```

### Step 2: Run 2 Nodes (Simulated Multi-Node)

In terminal 1:
```bash
python prototype_frequency_parallel.py \
    --node-rank 0 \
    --n-nodes 2 \
    --output-dir prototype_output/freq_parallel \
    --n-images 1000
```

In terminal 2 (simultaneously):
```bash
python prototype_frequency_parallel.py \
    --node-rank 1 \
    --n-nodes 2 \
    --output-dir prototype_output/freq_parallel \
    --n-images 1000
```

### Step 3: Concatenate Results

After both nodes complete:
```bash
python prototype_frequency_parallel.py \
    --concatenate \
    --n-nodes 2 \
    --output-dir prototype_output/freq_parallel
```

### Step 4: Validate

```python
import numpy as np

# Load baseline
baseline = np.load('prototype_output/baseline/node000_result.npz')
print("Baseline shape:", baseline['covariance_cols'].shape)

# Load concatenated
concat = np.load('prototype_output/freq_parallel/concatenated_result.npz')
print("Concatenated shape:", concat['covariance_cols'].shape)

# Should match!
assert baseline['covariance_cols'].shape == concat['covariance_cols'].shape
print("✓ Shapes match!")
```

## What the Prototype Tests

### Core Logic
- ✅ Frequency splitting across nodes
- ✅ Independent computation per node
- ✅ Concatenation of results
- ✅ Continuity validation (no gaps/overlaps)

### Performance
- ✅ Near-linear speedup (2× with 2 nodes)
- ✅ Minimal overhead from splitting
- ✅ Parallel efficiency > 85%

### Correctness
- ✅ Output shapes match baseline
- ✅ Frequency indices are continuous
- ✅ No data corruption

## Success Criteria

**Functional** ✓:
- Both nodes complete without errors
- Concatenation produces correct shape
- No frequency gaps or overlaps

**Performance** ✓:
- 2 nodes complete in ~50% of baseline time
- Parallel efficiency > 85%

**Correctness** ✓:
- Output shapes match baseline
- Frequency coverage is complete

## Interpreting Results

### Good Results
```
Baseline (1 node):     100s
Parallel (2 nodes):    52s
Speedup:               1.92×
Parallel efficiency:   96%
```
→ **Proceed with full implementation!**

### Acceptable Results
```
Baseline (1 node):     100s
Parallel (2 nodes):    60s
Speedup:               1.67×
Parallel efficiency:   83%
```
→ Some overhead, but still good. Investigate if needed.

### Poor Results
```
Baseline (1 node):     100s
Parallel (2 nodes):    90s
Speedup:               1.11×
Parallel efficiency:   55%
```
→ **Stop and debug!** Something is wrong.

## Common Issues

### Issue: "Dataset directory not found"
**Solution**: Update `--dataset-dir` to point to your dataset location.

### Issue: "Missing result from node X"
**Solution**: Ensure both nodes completed before concatenating. Check logs for errors.

### Issue: "Shape mismatch"
**Solution**: Frequency splitting logic may be incorrect. Check node output files.

### Issue: Poor speedup (< 1.5×)
**Possible causes**:
- Dataset too small (overhead dominates)
- Disk I/O bottleneck
- GPU not being used
- Frequency split unbalanced

## Next Steps

### If Prototype Succeeds ✓
1. Review timing results
2. Proceed with full implementation (see `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md`)
3. Integrate into `covariance_estimation.py`
4. Add SLURM support
5. Full testing with 2 nodes × 2 GPUs

### If Prototype Fails ✗
1. Check error messages in logs
2. Verify dataset is correct
3. Test with smaller dataset (--n-images 100)
4. Debug frequency splitting logic
5. Iterate on prototype before full implementation

## Parameters

```
--node-rank INT        Node rank (0-indexed, required for computation)
--n-nodes INT          Total number of nodes (default: 2)
--concatenate          Concatenate mode (post-processing)
--dataset-dir PATH     Dataset directory (default: data-128-100000/test_dataset)
--output-dir PATH      Output directory (default: prototype_output)
--image-size INT       Image size (default: 128)
--n-images INT         Number of images (default: 1000)
--freq-radius FLOAT    Frequency radius (default: 0.95)
--gpu-memory FLOAT     GPU memory in GB (default: 32)
```

## Architecture

```
Node 0                          Node 1
├─ Load dataset                 ├─ Load dataset
├─ Get frequencies [0:6k]       ├─ Get frequencies [6k:12k]
├─ Compute covariance           ├─ Compute covariance
└─ Save node000_result.npz      └─ Save node001_result.npz
         ↓                               ↓
         └───────────────┬───────────────┘
                         ↓
                  Concatenate
                         ↓
              concatenated_result.npz
```

## Timing Breakdown

The prototype measures:
- **Total time**: End-to-end including data loading
- **Compute time**: Just covariance computation
- **Overhead**: Data loading, frequency splitting, saving

**Expected breakdown**:
- Compute: 80-90% of total time
- Overhead: 10-20% of total time

## Output Files

```
prototype_output/
├── baseline/
│   └── node000_result.npz       # Baseline (1 node, all frequencies)
└── freq_parallel/
    ├── node000_result.npz       # Node 0 (first half of frequencies)
    ├── node001_result.npz       # Node 1 (second half of frequencies)
    └── concatenated_result.npz  # Combined result
```

Each `.npz` file contains:
- `covariance_cols`: Covariance matrix columns
- `picked_frequencies`: Frequency indices
- `fscs`: Fourier Shell Correlation curves
- `freq_start`, `freq_end`: Index range
- `node_rank`: Node identifier
- `compute_time`: Computation time in seconds

## Integration Points

The prototype validates these integration points for full implementation:

1. **Frequency splitting** (`split_frequencies()`)
   - Will be added to `covariance_estimation.py`

2. **Subset computation** (`compute_covariance_subset()`)
   - Calls existing `compute_regularized_covariance_columns()`
   - Just passes frequency subset parameter

3. **Concatenation** (`concatenate_results()`)
   - Will become `concatenate_covariance.py` script

4. **Validation** (continuity checks)
   - Will be part of post-processing

## Estimated Time

- **Setup**: 5 minutes
- **Run test**: 10-30 minutes (depends on dataset size)
- **Analysis**: 5 minutes
- **Total**: 20-40 minutes

Much faster than full implementation (40 hours)!

## Questions?

See `PROTOTYPE_PLAN.md` for detailed implementation notes.
See `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md` for full implementation plan.
