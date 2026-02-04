# Multi-Node Covariance Benchmark

Prototype and implementation plan for scaling RECOVAR's covariance computation across multiple nodes using frequency-level parallelism.

## Quick Start

### Run the Prototype

```bash
cd multinode_covariance_bench
bash test_prototype.sh
```

This validates the frequency-parallel approach by:
1. Running baseline (1 node, all frequencies)
2. Running 2 nodes in parallel (frequency-split)
3. Concatenating results
4. Validating correctness and measuring speedup

**Expected**: ~2× speedup with 2 nodes

### Analyze Results

```bash
python analyze_prototype_results.py prototype_output_TIMESTAMP
```

## Files

| File | Purpose |
|------|---------|
| `README.md` | This file - quick start guide |
| `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md` | **Main implementation plan** - complete guide with code examples |
| `PROTOTYPE_README.md` | Detailed prototype documentation and usage |
| `prototype_frequency_parallel.py` | Working prototype script |
| `test_prototype.sh` | Automated test script |
| `analyze_prototype_results.py` | Results analysis and validation |

## Architecture

### Hybrid Approach: Frequency-Parallel + Multi-GPU

```
Across nodes:  Frequency-level parallelism (each node = different columns)
Within nodes:  Image-level parallelism (multi-GPU, existing code)
```

**Key benefit**: No inter-node communication during compute!

### Why This Works

```python
# Covariance column k is a sum over ALL images:
H[:, freq_k] = sum_over_all_images(contribution_to_freq_k)

# Frequency-parallel: Each node computes different columns
Node 0: H[:, 0:6k]   = sum_over_all_images(...)  # First half
Node 1: H[:, 6k:12k] = sum_over_all_images(...)  # Second half

# Results are independent → just concatenate!
H_complete = np.concatenate([H_node0, H_node1], axis=1)
```

**No reduction needed** - this is the key advantage.

## Performance Expectations

### Testing Environment (2 GPUs)

| Configuration | Speedup | Time (50k images) |
|--------------|---------|-------------------|
| 1 node, 1 GPU | 1× | 100 min |
| 1 node, 2 GPUs | 1.9× | 53 min |
| **2 nodes, 4 GPUs** | **~3.7×** | **~27 min** |

### Production Projections (8 GPUs)

| Configuration | Speedup | Time (50k images) |
|--------------|---------|-------------------|
| 1 node, 8 GPUs | 7.6× | 13 min |
| 4 nodes, 32 GPUs | ~28× | 3.6 min |
| 8 nodes, 64 GPUs | ~56× | 1.8 min |

## Workflow

### Phase 1: Prototype (Current - 1-2 hours)

```bash
# Run prototype
bash test_prototype.sh

# Analyze results
python analyze_prototype_results.py prototype_output_*

# Validate:
# - Speedup ~2× with 2 nodes
# - Outputs match baseline
# - Efficiency > 85%
```

### Phase 2: Full Implementation (2 weeks, 40 hours)

If prototype succeeds, follow `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md`:

**Week 1**: Core infrastructure (~20 hours)
- Add CLI arguments
- Modify covariance_estimation.py
- Create concatenation script
- Add SLURM support

**Week 2**: Testing & benchmarking (~20 hours)
- Unit tests
- Integration tests
- Validation tests
- Performance benchmarks

### Phase 3: Production Deployment (Future)

After validation with 2 nodes × 2 GPUs:
- Scale to 4 nodes × 8 GPUs
- Scale to 8 nodes × 8 GPUs
- Production deployment

## Testing Configuration

All testing constrained to:
- **Max 2 nodes**
- **2 GPUs per node**
- **Baseline**: 1 node, 2 GPUs

This validates the scaling principle without requiring a large cluster.

## Success Criteria

### Prototype ✓
- [ ] Script runs without errors
- [ ] ~2× speedup with 2 nodes
- [ ] Outputs match baseline
- [ ] Efficiency > 85%

### Full Implementation
- [ ] Integrated into pipeline.py
- [ ] SLURM auto-detection works
- [ ] 2 nodes × 2 GPUs: ~1.9× speedup vs 1 node
- [ ] Outputs match baseline (rtol=1e-5)
- [ ] All tests pass

## Key Design Decisions

### 1. Frequency-Parallel (Not Image-Parallel)
**Why**: No inter-node communication needed. Image-parallel requires expensive all-reduce.

### 2. SLURM-Native Synchronization
**Why**: Simple, robust, no custom distributed framework needed.

### 3. Minimal Code Changes
**Why**: Backward compatible, low risk. ~1,235 lines total (310 new + 255 modified).

### 4. Post-Processing Concatenation
**Why**: Decouples compute from post-processing. Easy to debug and validate.

## Common Issues

**"Dataset directory not found"**
→ Update paths in test_prototype.sh

**"Poor speedup (< 1.5×)"**
→ Check dataset size, verify GPU usage, check disk I/O

**"Shape mismatch"**
→ Debug frequency splitting logic, check node output files

## Next Steps

1. **Run prototype**: `bash test_prototype.sh`
2. **Analyze results**: `python analyze_prototype_results.py prototype_output_*`
3. **If successful**: Follow `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md`
4. **If unsuccessful**: Debug and iterate on prototype

## Documentation

- `README.md` (this file) - Overview and quick start
- `HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md` - Complete implementation guide with code
- `PROTOTYPE_README.md` - Detailed prototype usage and troubleshooting

## References

- Main RECOVAR code: `../recovar/`
- Test dataset: `../data-128-100000/test_dataset/`
- Existing pipeline: `../recovar/commands/pipeline.py`
- Covariance code: `../recovar/covariance_estimation.py`

---

**Status**: Prototype ready for testing  
**Last updated**: 2026-02-03
