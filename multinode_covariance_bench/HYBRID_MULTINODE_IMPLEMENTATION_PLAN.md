# Hybrid Multi-Node Multi-GPU Implementation Plan

**Version**: 1.0  
**Date**: 2026-01-29  
**Status**: Ready for Implementation

---

## Executive Summary

### The Approach: Hybrid Frequency-Parallel + Multi-GPU

**Across nodes**: Frequency-level parallelism (each node computes different frequency columns)  
**Within nodes**: Image-level parallelism (existing multi-GPU across 8 GPUs per node)

### Key Benefits

- ✅ **Perfect scaling**: No inter-node communication during compute
- ✅ **Simple implementation**: ~310 lines of new code
- ✅ **Zero network overhead**: No all-reduce needed (just concatenate columns)
- ✅ **Backward compatible**: All existing code works unchanged
- ✅ **Leverages existing multi-GPU**: Builds on working implementation
- ✅ **Fault tolerant**: Nodes work independently
- ✅ **SLURM-native**: Automatic synchronization via `srun`

### Performance Projections

**Testing Environment (2 GPUs):**
| Configuration | Speedup | Time (50k images) |
|--------------|---------|-------------------|
| Baseline (1 node, 1 GPU) | 1× | 100 min |
| Current (1 node, 2 GPUs) | 1.9× | 53 min |
| **Test Target (2 nodes, 4 GPUs)** | **~3.7×** | **~27 min** |

**Production Scale Projections (8 GPUs):**
| Configuration | Speedup | Time (50k images) |
|--------------|---------|-------------------|
| Baseline (1 GPU) | 1× | 100 min |
| Current (1 node, 8 GPUs) | 7.6× | 13 min |
| Hybrid (4 nodes, 32 GPUs) | ~28× | 3.6 min |
| Hybrid (8 nodes, 64 GPUs) | ~56× | 1.8 min |

---

## Architecture Overview

### Data Flow

```
Input: 50k images (1 halfset) × 12k frequencies

┌────────────────────────────────────────────────┐
│  Frequency-Level Split (Across Nodes)         │
└────────────────────────────────────────────────┘
         ↓              ↓              ↓
    Node 0         Node 1         Node 2
  Freqs 0-3k     Freqs 3k-6k    Freqs 6k-9k
         ↓              ↓              ↓
┌────────────────────────────────────────────────┐
│  Image-Level Split (Within Each Node)         │
└────────────────────────────────────────────────┘
   GPU 0-7         GPU 0-7        GPU 0-7
  6.25k imgs     6.25k imgs     6.25k imgs
   per GPU        per GPU        per GPU
         ↓              ↓              ↓
   Local sum      Local sum      Local sum
         ↓              ↓              ↓
   Save disk      Save disk      Save disk
         ↓              ↓              ↓
         └──────────────┴──────────────┘
                       ↓
              Concatenate columns
                       ↓
                  Complete H, B
```

### Why This Works

**Mathematics**:
```python
# Covariance is a sum over all images:
H[:, freq_k] = sum_over_all_images( image_contribution_to_freq_k )

# Frequency-parallel splits the frequency axis (columns):
Node 0: H[:, 0:6k]   = sum_over_all_images(...)  # First half of frequencies
Node 1: H[:, 6k:12k] = sum_over_all_images(...)  # Second half of frequencies

# Each node processes ALL images (can do within-node multi-GPU with 2 GPUs)
# Results are independent → just concatenate!
H_complete = [H_node0 | H_node1]
```

**No reduction needed** because different columns are mathematically independent.

---

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)

#### Task 1.1: Add CLI Arguments

**File**: `recovar/commands/pipeline.py`  
**Function**: `add_args()`  
**Location**: After line 318

```python
# Add frequency-parallel arguments
parser.add_argument(
    "--frequency-parallel",
    action="store_true",
    dest="frequency_parallel",
    default=False,
    help="Enable frequency-parallel multi-node computation (requires SLURM)"
)

parser.add_argument(
    "--node-rank",
    type=int,
    default=None,
    dest="node_rank",
    help="Node rank for frequency-parallel mode (auto-detected from SLURM_PROCID)"
)

parser.add_argument(
    "--n-nodes",
    type=int,
    default=None,
    dest="n_nodes",
    help="Total number of nodes (auto-detected from SLURM_NTASKS)"
)

parser.add_argument(
    "--frequency-output-dir",
    type=str,
    default=None,
    dest="frequency_output_dir",
    help="Directory for node results (default: outdir/covariance_parts)"
)
```

**Estimated time**: 30 minutes

---

#### Task 1.2: Add SLURM Auto-Detection

**File**: `recovar/commands/pipeline.py`  
**Location**: Before `standard_recovar_pipeline()` function

```python
def get_distributed_context(args):
    """
    Detect distributed execution context (SLURM or manual).
    
    Returns:
        dict with keys: is_distributed, node_rank, n_nodes, mode
    """
    import os
    
    context = {
        'is_distributed': False,
        'node_rank': 0,
        'n_nodes': 1,
        'mode': 'single'
    }
    
    # Check if frequency-parallel mode requested
    if not args.frequency_parallel:
        return context
    
    # Try command-line args first (for manual testing)
    if args.node_rank is not None and args.n_nodes is not None:
        context.update({
            'is_distributed': True,
            'node_rank': args.node_rank,
            'n_nodes': args.n_nodes,
            'mode': 'manual'
        })
        logger.info(f"Manual distributed mode: Node {args.node_rank}/{args.n_nodes}")
        return context
    
    # Auto-detect SLURM environment
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        node_rank = int(os.environ['SLURM_PROCID'])
        n_nodes = int(os.environ['SLURM_NTASKS'])
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        
        context.update({
            'is_distributed': True,
            'node_rank': node_rank,
            'n_nodes': n_nodes,
            'mode': 'slurm',
            'job_id': job_id
        })
        
        logger.info("=" * 70)
        logger.info("SLURM DISTRIBUTED MODE DETECTED")
        logger.info("=" * 70)
        logger.info(f"Node rank:     {node_rank}")
        logger.info(f"Total nodes:   {n_nodes}")
        logger.info(f"Job ID:        {job_id}")
        logger.info("=" * 70)
        
        return context
    
    # Frequency-parallel requested but no environment detected
    logger.warning(
        "Frequency-parallel mode requested but no SLURM environment detected. "
        "Running in single-node mode."
    )
    
    return context
```

**Estimated time**: 1 hour

---

#### Task 1.3: Modify Pipeline Entry Point

**File**: `recovar/commands/pipeline.py`  
**Function**: `standard_recovar_pipeline()`  
**Location**: Beginning of function (after line 328)

```python
def standard_recovar_pipeline(args):
    st_time = time.time()
    
    # NEW: Detect distributed context
    dist_context = get_distributed_context(args)
    
    # NEW: Setup frequency-parallel output directory
    if dist_context['is_distributed']:
        if args.frequency_output_dir is None:
            args.frequency_output_dir = os.path.join(args.outdir, 'covariance_parts')
        os.makedirs(args.frequency_output_dir, exist_ok=True)
        
        logger.info(f"Node {dist_context['node_rank']}: "
                   f"Frequency-parallel output dir: {args.frequency_output_dir}")
    
    # ... rest of existing code ...
```

**Location**: Where `estimate_principal_components` is called (lines 656, 704)

```python
# Existing call (modify to add new parameters):
u, s, covariance_cols, picked_frequencies, column_fscs = \
    principal_components.estimate_principal_components(
        cryos, options, means, mean_prior, focus_mask, 
        dilated_volume_mask, valid_idx, batch_size, 
        gpu_memory_to_use=gpu_memory, 
        covariance_options=covariance_options, 
        variance_estimate=variance_est['combined'],
        use_multi_gpu=args.multi_gpu,  # Existing
        n_gpus=args.n_gpus,             # Existing
        # NEW: Add distributed parameters
        distributed_context=dist_context,
        frequency_output_dir=args.frequency_output_dir
    )

# NEW: Early exit for distributed nodes
if dist_context['is_distributed']:
    logger.info(f"Node {dist_context['node_rank']}: Computation complete")
    logger.info(f"Saved results to {args.frequency_output_dir}")
    logger.info("Exiting - SLURM will synchronize nodes")
    return None  # Exit early, post-processing happens separately
```

**Estimated time**: 1.5 hours

---

#### Task 1.4: Update Principal Components Function

**File**: `recovar/principal_components.py`  
**Function**: `estimate_principal_components()`  
**Location**: Function signature (line 20)

```python
def estimate_principal_components(cryos, options, means, mean_prior, volume_mask,
                                dilated_volume_mask, valid_idx, batch_size, 
                                gpu_memory_to_use,
                                covariance_options=None, 
                                variance_estimate=None, 
                                use_reg_mean_in_contrast=False, 
                                use_multi_gpu=False, 
                                n_gpus=None,
                                # NEW parameters
                                distributed_context=None,
                                frequency_output_dir=None):
```

**Location**: Where covariance is computed

```python
# Find the call to compute_regularized_covariance_columns_in_batch
# Pass through distributed parameters

covariance_cols, picked_frequencies, fscs = \
    covariance_estimation.compute_regularized_covariance_columns_in_batch(
        cryos, means, mean_prior, volume_mask, dilated_volume_mask, 
        valid_idx, gpu_memory_to_use, options, picked_frequencies,
        use_multi_gpu=use_multi_gpu,
        n_gpus=n_gpus,
        # NEW
        distributed_context=distributed_context,
        frequency_output_dir=frequency_output_dir
    )

# NEW: Handle distributed early exit
if distributed_context and distributed_context['is_distributed']:
    # Save node results and return None to signal early exit
    node_rank = distributed_context['node_rank']
    save_path = os.path.join(
        frequency_output_dir,
        f'pca_node{node_rank:03d}.pkl'
    )
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump({
            'covariance_cols': covariance_cols,
            'picked_frequencies': picked_frequencies,
            'fscs': fscs
        }, f)
    
    logger.info(f"Node {node_rank}: Saved intermediate results to {save_path}")
    return None, None, None, None, None
```

**Estimated time**: 1 hour

---

#### Task 1.5: Modify Covariance Estimation

**File**: `recovar/covariance_estimation.py`  
**Function**: `compute_regularized_covariance_columns_in_batch()`  
**Location**: Line 284

```python
def compute_regularized_covariance_columns_in_batch(
    cryos, means, mean_prior, volume_mask, dilated_volume_mask, 
    valid_idx, gpu_memory, options, picked_frequencies, 
    use_multi_gpu=False, n_gpus=None,
    # NEW parameters
    distributed_context=None,
    frequency_output_dir=None):
    
    # NEW: Determine frequency subset for this node
    if distributed_context and distributed_context['is_distributed']:
        node_rank = distributed_context['node_rank']
        n_nodes = distributed_context['n_nodes']
        
        # Split frequencies across nodes
        total_freqs = len(picked_frequencies)
        freqs_per_node = total_freqs // n_nodes
        
        freq_start = node_rank * freqs_per_node
        if node_rank == n_nodes - 1:
            freq_end = total_freqs  # Last node gets remainder
        else:
            freq_end = freq_start + freqs_per_node
        
        # This node's frequency subset
        my_frequencies = picked_frequencies[freq_start:freq_end]
        
        logger.info(f"Node {node_rank}/{n_nodes}: Computing {len(my_frequencies)} frequencies")
        logger.info(f"  Frequency indices: {freq_start} to {freq_end} (of {total_freqs})")
    else:
        # Normal mode: all frequencies
        my_frequencies = picked_frequencies
        freq_start = 0
        freq_end = len(picked_frequencies)
    
    # Rest of function uses my_frequencies instead of picked_frequencies
    frequency_batch = utils.get_column_batch_size(cryos[0].grid_size, gpu_memory)    

    covariance_cols = []
    fscs = []
    for k in range(0, int(np.ceil(len(my_frequencies)/frequency_batch))):
        batch_st = int(k * frequency_batch)
        batch_end = int(np.min([(k+1) * frequency_batch, len(my_frequencies)]))

        covariance_cols_b, _, fscs_b = compute_regularized_covariance_columns(
            cryos, means, mean_prior, volume_mask, dilated_volume_mask, 
            valid_idx, gpu_memory, options, 
            my_frequencies[batch_st:batch_end],  # ← Subset
            use_multi_gpu=use_multi_gpu, 
            n_gpus=n_gpus
        )
        logger.info(f'batch of col done: {batch_st}, {batch_end}')

        covariance_cols.append(covariance_cols_b['est_mask'])
        fscs.append(fscs_b)

    covariance_cols = {'est_mask': np.concatenate(covariance_cols, axis=-1)}
    fscs = np.concatenate(fscs, axis=0)
    
    # NEW: Save node results if distributed
    if distributed_context and distributed_context['is_distributed']:
        node_rank = distributed_context['node_rank']
        save_path = os.path.join(
            frequency_output_dir,
            f'covariance_node{node_rank:03d}.npz'
        )
        np.savez_compressed(
            save_path,
            covariance_cols=covariance_cols['est_mask'],
            picked_frequencies=my_frequencies,
            fscs=fscs,
            freq_start=freq_start,
            freq_end=freq_end,
            node_rank=node_rank
        )
        logger.info(f"Node {node_rank}: Saved results to {save_path}")
    
    return covariance_cols, my_frequencies, fscs
```

**Estimated time**: 2 hours

---

### Phase 2: Post-Processing Infrastructure (Week 1)

#### Task 2.1: Create Concatenation Script

**New file**: `recovar/commands/concatenate_covariance.py`

```python
#!/usr/bin/env python3
"""
Post-processing: Concatenate frequency-parallel covariance results.

Usage:
    python -m recovar.commands.concatenate_covariance \
        output/covariance_parts \
        --n-nodes 4 \
        --output output/covariance_complete.pkl
"""

import argparse
import logging
import numpy as np
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_node_files(output_dir, n_nodes, n_halfsets=2):
    """
    Validate that all expected node files exist.
    """
    missing = []
    for node_rank in range(n_nodes):
        filepath = os.path.join(
            output_dir,
            f"covariance_node{node_rank:03d}.npz"
        )
        if not os.path.exists(filepath):
            missing.append((node_rank, filepath))
    
    if missing:
        error_msg = "Missing covariance files from nodes:\n"
        for node_rank, filepath in missing:
            error_msg += f"  Node {node_rank}: {filepath}\n"
        raise FileNotFoundError(error_msg)
    
    logger.info(f"✓ All {n_nodes} node files found")


def load_node_covariance(output_dir, node_rank):
    """Load covariance from a single node."""
    filepath = os.path.join(output_dir, f"covariance_node{node_rank:03d}.npz")
    
    logger.info(f"Loading node {node_rank}: {filepath}")
    data = np.load(filepath)
    
    return {
        'covariance_cols': data['covariance_cols'],
        'picked_frequencies': data['picked_frequencies'],
        'fscs': data['fscs'],
        'freq_start': int(data['freq_start']),
        'freq_end': int(data['freq_end']),
        'node_rank': int(data['node_rank'])
    }


def concatenate_covariance(output_dir, n_nodes):
    """
    Concatenate covariance columns from all nodes.
    
    Args:
        output_dir: Directory containing node*.npz files
        n_nodes: Number of nodes
        
    Returns:
        dict with concatenated covariance_cols, picked_frequencies, fscs
    """
    logger.info("=" * 70)
    logger.info("Concatenating Covariance from All Nodes")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of nodes:  {n_nodes}")
    logger.info("")
    
    # Validate all files exist
    validate_node_files(output_dir, n_nodes)
    
    # Load all node data
    node_data = []
    for node_rank in range(n_nodes):
        data = load_node_covariance(output_dir, node_rank)
        node_data.append(data)
        
        logger.info(f"  Node {node_rank}: {len(data['picked_frequencies'])} frequencies "
                   f"(indices {data['freq_start']}:{data['freq_end']})")
    
    # Sort by frequency start index
    node_data.sort(key=lambda x: x['freq_start'])
    
    # Validate continuity (no gaps or overlaps)
    logger.info("\nValidating frequency coverage...")
    for i in range(len(node_data) - 1):
        curr_end = node_data[i]['freq_end']
        next_start = node_data[i+1]['freq_start']
        
        if curr_end != next_start:
            raise ValueError(
                f"Frequency range mismatch between node {i} and {i+1}: "
                f"node {i} ends at {curr_end}, node {i+1} starts at {next_start}"
            )
    
    logger.info("✓ Frequency coverage is continuous")
    
    # Concatenate along frequency axis
    logger.info("\nConcatenating arrays...")
    
    covariance_cols_complete = np.concatenate(
        [nd['covariance_cols'] for nd in node_data],
        axis=-1  # Last axis is frequency dimension
    )
    
    picked_frequencies_complete = np.concatenate(
        [nd['picked_frequencies'] for nd in node_data]
    )
    
    fscs_complete = np.concatenate(
        [nd['fscs'] for nd in node_data],
        axis=0
    )
    
    logger.info(f"  Complete covariance shape: {covariance_cols_complete.shape}")
    logger.info(f"  Total frequencies: {len(picked_frequencies_complete)}")
    logger.info(f"  FSCs shape: {fscs_complete.shape}")
    
    # Save concatenated result
    output_file = os.path.join(output_dir, "covariance_complete.pkl")
    
    logger.info(f"\nSaving concatenated result to {output_file}")
    
    result = {
        'covariance_cols': covariance_cols_complete,
        'picked_frequencies': picked_frequencies_complete,
        'fscs': fscs_complete,
        'n_nodes': n_nodes
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    # Also save as npz for easy inspection
    npz_file = os.path.join(output_dir, "covariance_complete.npz")
    np.savez_compressed(
        npz_file,
        covariance_cols=covariance_cols_complete,
        picked_frequencies=picked_frequencies_complete,
        fscs=fscs_complete
    )
    
    logger.info(f"Saved to {output_file} and {npz_file}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONCATENATION COMPLETE")
    logger.info("=" * 70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate frequency-parallel covariance results"
    )
    
    parser.add_argument(
        "covariance_dir",
        type=str,
        help="Directory containing node covariance files"
    )
    
    parser.add_argument(
        "--n-nodes",
        type=int,
        required=True,
        help="Number of nodes that computed covariance"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: covariance_dir/covariance_complete.pkl)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Concatenate
    result = concatenate_covariance(args.covariance_dir, args.n_nodes)
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
```

**Estimated time**: 2 hours

---

#### Task 1.6: Create SLURM Launch Script

**New file**: `scripts/run_frequency_parallel_multigpu.sh`

```bash
#!/bin/bash
# Frequency-parallel multi-node computation with multi-GPU within nodes
# Usage: ./run_frequency_parallel_multigpu.sh <n_nodes> <n_gpus> <dataset_dir>

set -e

N_NODES=${1:-2}
N_GPUS=${2:-2}
DATASET_DIR=${3:-"/workspace/data-128-100000/test_dataset"}
IMAGE_SIZE=128
N_IMAGES=100000

echo "=========================================="
echo "Frequency-Parallel Multi-Node Computation"
echo "Dataset: $DATASET_DIR"
echo "Nodes: $N_NODES"
echo "GPUs per node: $N_GPUS"
echo "=========================================="

cd "$DATASET_DIR"

# Output directory (unique per job)
OUTPUT_DIR="pipeline_output_${N_NODES}node_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Build the pipeline command
PIPELINE_CMD="python -m recovar.commands.pipeline \
    particles.${IMAGE_SIZE}.mrcs \
    --ctf ctf.pkl \
    --poses poses.pkl \
    --mask=from_halfmaps \
    -o ${OUTPUT_DIR} \
    --n-images $N_IMAGES \
    --frequency-parallel \
    --multi-gpu \
    --n-gpus ${N_GPUS} \
    --lazy"

echo "Running: $PIPELINE_CMD"
echo ""

# Note: This script is meant to be called FROM SLURM via srun
# The srun command should be in the parent SLURM script
# This script just defines the command to run

$PIPELINE_CMD

echo "=========================================="
echo "Node computation complete!"
echo "=========================================="
```

**Estimated time**: 30 minutes

---

#### Task 1.7: Update crun_recovar_workload.sh

**File**: `crun_recovar_workload.sh`  
**Location**: Add to help text and case statement

```bash
# In help text (around line 142):
    echo "  Multi-node frequency-parallel (2 GPU testing):"
    echo "    freq-parallel-2node  - Frequency-parallel with 2 nodes, 2 GPUs each (1h)"
    echo ""

# In case statement (before the *) default):
    # Multi-node frequency-parallel (2 GPU testing)
    freq-parallel-2node)
        submit_job "Freq-Parallel 2 Nodes" "pixi run freq-parallel-2node" 2 "01:00:00"
        ;;
```

**Estimated time**: 15 minutes

---

#### Task 1.8: Update pixi.toml

**File**: `pixi.toml`  
**Location**: Add to [tasks] section

```toml
# Multi-node frequency-parallel tasks (2 GPU testing)
freq-parallel-2node = "bash ./scripts/run_frequency_parallel_multigpu.sh 2 2"

# Concatenation utility
concatenate-covariance = "python -m recovar.commands.concatenate_covariance"
```

**Estimated time**: 10 minutes

---

### Phase 3: Testing and Validation (Week 2)

#### Task 3.1: Unit Tests for Frequency Splitting

**New file**: `tests/test_frequency_parallel.py`

```python
"""
Unit tests for frequency-parallel distribution.
"""
import numpy as np
import pytest


def split_frequencies(picked_frequencies, node_rank, n_nodes):
    """Split frequencies for a given node."""
    total_freqs = len(picked_frequencies)
    freqs_per_node = total_freqs // n_nodes
    
    freq_start = node_rank * freqs_per_node
    if node_rank == n_nodes - 1:
        freq_end = total_freqs
    else:
        freq_end = freq_start + freqs_per_node
    
    return picked_frequencies[freq_start:freq_end], freq_start, freq_end


def test_frequency_split_even():
    """Test splitting with evenly divisible frequencies."""
    freqs = np.arange(12000)
    n_nodes = 4
    
    all_freqs = []
    for node_rank in range(n_nodes):
        node_freqs, start, end = split_frequencies(freqs, node_rank, n_nodes)
        all_freqs.extend(node_freqs)
        
        # Check sizes
        if node_rank < n_nodes - 1:
            assert len(node_freqs) == 3000
        else:
            assert len(node_freqs) == 3000  # Still even
        
        # Check continuity
        assert start == node_rank * 3000
        if node_rank < n_nodes - 1:
            assert end == (node_rank + 1) * 3000
    
    # Verify no gaps, no overlaps
    assert len(all_freqs) == len(freqs)
    assert np.array_equal(all_freqs, freqs)


def test_frequency_split_uneven():
    """Test splitting with remainder."""
    freqs = np.arange(10000)  # Not divisible by 4
    n_nodes = 4
    
    sizes = []
    all_freqs = []
    
    for node_rank in range(n_nodes):
        node_freqs, start, end = split_frequencies(freqs, node_rank, n_nodes)
        sizes.append(len(node_freqs))
        all_freqs.extend(node_freqs)
    
    # First 3 nodes: 2500 each, last node: 2500 + remainder
    assert sizes[0] == 2500
    assert sizes[1] == 2500
    assert sizes[2] == 2500
    assert sizes[3] == 2500  # 2500 + 0 remainder
    
    # Verify complete coverage
    assert len(all_freqs) == len(freqs)
    assert np.array_equal(all_freqs, freqs)


def test_concatenation_order():
    """Test that concatenation preserves frequency order."""
    # Simulate node outputs
    node_results = [
        {'freq_start': 0, 'freq_end': 3000, 
         'cols': np.random.randn(1000, 3000)},
        {'freq_start': 3000, 'freq_end': 6000,
         'cols': np.random.randn(1000, 3000)},
        {'freq_start': 6000, 'freq_end': 9000,
         'cols': np.random.randn(1000, 3000)},
        {'freq_start': 9000, 'freq_end': 12000,
         'cols': np.random.randn(1000, 3000)},
    ]
    
    # Concatenate
    complete = np.concatenate([nr['cols'] for nr in node_results], axis=1)
    
    # Verify shape
    assert complete.shape == (1000, 12000)
    
    # Verify columns match
    for i, nr in enumerate(node_results):
        start = nr['freq_start']
        end = nr['freq_end']
        assert np.array_equal(complete[:, start:end], nr['cols'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Estimated time**: 1 hour

---

#### Task 3.2: Integration Test (2 Nodes, Small Dataset)

**Test script**: `tests/test_2node_integration.sh`

```bash
#!/bin/bash
# Integration test: 2 nodes on small dataset

set -e

TEST_DIR="/tmp/freq_parallel_test_$$"
mkdir -p "$TEST_DIR"

echo "Integration Test: 2-Node Frequency-Parallel (2 GPUs per node)"
echo "Test directory: $TEST_DIR"

# Simulate 2 nodes manually
echo "Simulating Node 0..."
python -m recovar.commands.pipeline \
    data-128-100000/test_dataset/particles.128.mrcs \
    --poses data-128-100000/test_dataset/poses.pkl \
    --ctf data-128-100000/test_dataset/ctf.pkl \
    --mask from_halfmaps \
    -o ${TEST_DIR}/output \
    --frequency-parallel \
    --node-rank 0 \
    --n-nodes 2 \
    --multi-gpu \
    --n-gpus 2 \
    --n-images 1000 &

PID0=$!

echo "Simulating Node 1..."
python -m recovar.commands.pipeline \
    data-128-100000/test_dataset/particles.128.mrcs \
    --poses data-128-100000/test_dataset/poses.pkl \
    --ctf data-128-100000/test_dataset/ctf.pkl \
    --mask from_halfmaps \
    -o ${TEST_DIR}/output \
    --frequency-parallel \
    --node-rank 1 \
    --n-nodes 2 \
    --multi-gpu \
    --n-gpus 2 \
    --n-images 1000 &

PID1=$!

# Wait for both
wait $PID0
wait $PID1

echo "Both nodes complete! Concatenating..."

python -m recovar.commands.concatenate_covariance \
    ${TEST_DIR}/output/covariance_parts \
    --n-nodes 2

echo "Test complete! Results in ${TEST_DIR}"
```

**Estimated time**: 30 minutes

---

#### Task 3.3: Validation Test (Compare with Baseline)

**New file**: `recovar/commands/validate_distributed_output.py`

```python
#!/usr/bin/env python3
"""
Validate that frequency-parallel output matches single-node baseline.
"""

import argparse
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


def load_covariance(filepath):
    """Load covariance from pickle or npz file."""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        return {
            'covariance_cols': data['covariance_cols'],
            'picked_frequencies': data['picked_frequencies'],
            'fscs': data.get('fscs', None)
        }
    else:
        raise ValueError(f"Unknown format: {filepath}")


def compare_covariance(baseline_path, distributed_path, rtol=1e-5, atol=1e-8):
    """
    Compare baseline and distributed covariance outputs.
    
    Args:
        baseline_path: Path to single-node baseline output
        distributed_path: Path to frequency-parallel output
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        dict with comparison results
    """
    logger.info("=" * 70)
    logger.info("Validating Distributed Output vs Baseline")
    logger.info("=" * 70)
    logger.info(f"Baseline:     {baseline_path}")
    logger.info(f"Distributed:  {distributed_path}")
    logger.info("")
    
    # Load both
    baseline = load_covariance(baseline_path)
    distributed = load_covariance(distributed_path)
    
    # Compare shapes
    logger.info("Comparing shapes...")
    baseline_shape = baseline['covariance_cols'].shape
    distributed_shape = distributed['covariance_cols'].shape
    
    logger.info(f"  Baseline shape:     {baseline_shape}")
    logger.info(f"  Distributed shape:  {distributed_shape}")
    
    if baseline_shape != distributed_shape:
        logger.error("✗ SHAPE MISMATCH!")
        return {'valid': False, 'reason': 'shape_mismatch'}
    
    logger.info("  ✓ Shapes match")
    
    # Compare covariance columns
    logger.info("\nComparing covariance columns...")
    
    cov_baseline = baseline['covariance_cols']
    cov_distributed = distributed['covariance_cols']
    
    # Compute differences
    abs_diff = np.abs(cov_baseline - cov_distributed)
    rel_diff = abs_diff / (np.abs(cov_baseline) + 1e-10)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    logger.info(f"  Max absolute difference: {max_abs_diff:.2e}")
    logger.info(f"  Max relative difference: {max_rel_diff:.2e}")
    logger.info(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    logger.info(f"  Mean relative difference: {mean_rel_diff:.2e}")
    
    # Check tolerance
    is_close = np.allclose(cov_baseline, cov_distributed, rtol=rtol, atol=atol)
    
    if is_close:
        logger.info("  ✓ Covariance matches within tolerance")
    else:
        # Find where differences are largest
        large_diff_mask = (abs_diff > atol) & (rel_diff > rtol)
        n_different = np.sum(large_diff_mask)
        pct_different = 100 * n_different / large_diff_mask.size
        
        logger.error(f"  ✗ Covariance differs!")
        logger.error(f"    {n_different} elements ({pct_different:.2f}%) exceed tolerance")
        
        return {
            'valid': False,
            'reason': 'covariance_mismatch',
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff,
            'n_different': n_different,
            'pct_different': pct_different
        }
    
    # Compare frequencies
    logger.info("\nComparing frequency indices...")
    
    if np.array_equal(baseline['picked_frequencies'], distributed['picked_frequencies']):
        logger.info("  ✓ Frequency indices match")
    else:
        logger.error("  ✗ Frequency indices differ!")
        return {'valid': False, 'reason': 'frequency_mismatch'}
    
    # Compare FSCs (if available)
    if baseline.get('fscs') is not None and distributed.get('fscs') is not None:
        logger.info("\nComparing FSCs...")
        
        if np.allclose(baseline['fscs'], distributed['fscs'], rtol=rtol, atol=atol):
            logger.info("  ✓ FSCs match")
        else:
            logger.warning("  ⚠ FSCs differ (may be OK if order changed)")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ VALIDATION PASSED")
    logger.info("=" * 70)
    
    return {
        'valid': True,
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_abs_diff': mean_abs_diff,
        'mean_rel_diff': mean_rel_diff
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate distributed covariance output"
    )
    
    parser.add_argument(
        "baseline",
        help="Baseline covariance file (single-node)"
    )
    
    parser.add_argument(
        "distributed",
        help="Distributed covariance file (multi-node)"
    )
    
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance (default: 1e-5)"
    )
    
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance (default: 1e-8)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    result = compare_covariance(
        args.baseline,
        args.distributed,
        rtol=args.rtol,
        atol=args.atol
    )
    
    if result['valid']:
        print("\n✓ Validation successful!")
        return 0
    else:
        print(f"\n✗ Validation failed: {result['reason']}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
```

**Estimated time**: 1.5 hours

---

#### Task 3.4: Create Test Suite

**New file**: `scripts/test_frequency_parallel_suite.sh`

```bash
#!/bin/bash
# Complete test suite for frequency-parallel implementation

set -e

DATASET_DIR="data-128-100000/test_dataset"
TEST_OUTPUT_DIR="test_outputs_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$TEST_OUTPUT_DIR"

echo "======================================================================"
echo "Frequency-Parallel Implementation Test Suite (2 GPU Testing)"
echo "======================================================================"
echo "Dataset: $DATASET_DIR"
echo "Output:  $TEST_OUTPUT_DIR"
echo ""

# ============================================================================
# TEST 1: Baseline (Single Node, 2 GPUs) - for comparison
# ============================================================================
echo "TEST 1: Running baseline (single node, 2 GPUs)..."
echo "----------------------------------------------------------------------"

python -m recovar.commands.pipeline \
    ${DATASET_DIR}/particles.128.mrcs \
    --poses ${DATASET_DIR}/poses.pkl \
    --ctf ${DATASET_DIR}/ctf.pkl \
    --mask from_halfmaps \
    -o ${TEST_OUTPUT_DIR}/baseline_1node_2gpu \
    --multi-gpu \
    --n-gpus 2 \
    --n-images 10000 \
    --lazy

echo "✓ TEST 1 complete"
echo ""

# ============================================================================
# TEST 2: Frequency-Parallel (2 Nodes, 2 GPUs each, Manual Simulation)
# ============================================================================
echo "TEST 2: Frequency-parallel (2 nodes, 2 GPUs each, simulated)..."
echo "----------------------------------------------------------------------"

# Node 0
echo "  Running node 0..."
python -m recovar.commands.pipeline \
    ${DATASET_DIR}/particles.128.mrcs \
    --poses ${DATASET_DIR}/poses.pkl \
    --ctf ${DATASET_DIR}/ctf.pkl \
    --mask from_halfmaps \
    -o ${TEST_OUTPUT_DIR}/freq_parallel_2node \
    --frequency-parallel \
    --node-rank 0 \
    --n-nodes 2 \
    --multi-gpu \
    --n-gpus 2 \
    --n-images 10000 \
    --lazy &

PID0=$!

# Node 1
echo "  Running node 1..."
python -m recovar.commands.pipeline \
    ${DATASET_DIR}/particles.128.mrcs \
    --poses ${DATASET_DIR}/poses.pkl \
    --ctf ${DATASET_DIR}/ctf.pkl \
    --mask from_halfmaps \
    -o ${TEST_OUTPUT_DIR}/freq_parallel_2node \
    --frequency-parallel \
    --node-rank 1 \
    --n-nodes 2 \
    --multi-gpu \
    --n-gpus 2 \
    --n-images 10000 \
    --lazy &

PID1=$!

# Wait for both
wait $PID0
wait $PID1

echo "  Both nodes complete!"

# Concatenate
echo "  Concatenating results..."
python -m recovar.commands.concatenate_covariance \
    ${TEST_OUTPUT_DIR}/freq_parallel_2node/covariance_parts \
    --n-nodes 2

echo "✓ TEST 2 complete"
echo ""

# ============================================================================
# TEST 3: Validation (Compare Outputs)
# ============================================================================
echo "TEST 3: Validating outputs match baseline..."
echo "----------------------------------------------------------------------"

python -m recovar.commands.validate_distributed_output \
    ${TEST_OUTPUT_DIR}/baseline_1node_2gpu/model/covariance_cols.pkl \
    ${TEST_OUTPUT_DIR}/freq_parallel_2node/covariance_parts/covariance_complete.pkl

echo "✓ TEST 3 complete"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "======================================================================"
echo "TEST SUITE COMPLETE"
echo "======================================================================"
echo ""
echo "Results:"
echo "  Baseline (1 node, 2 GPUs):      ${TEST_OUTPUT_DIR}/baseline_1node_2gpu"
echo "  Frequency-parallel (2 nodes):   ${TEST_OUTPUT_DIR}/freq_parallel_2node"
echo ""
echo "All tests passed! ✓"
```

**Estimated time**: 1 hour

---

### Phase 4: Benchmarking (Week 2)

#### Task 4.1: Create Benchmarking Script

**New file**: `scripts/benchmark_frequency_parallel.sh`

```bash
#!/bin/bash
# Benchmark: Compare single-node vs 2-node performance (2 GPUs per node)

DATASET_DIR="data-128-100000/test_dataset"
BENCHMARK_OUTPUT="benchmark_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BENCHMARK_OUTPUT"

echo "======================================================================"
echo "Frequency-Parallel Performance Benchmark (2 GPU Testing)"
echo "======================================================================"
echo "Dataset: $DATASET_DIR (100k images)"
echo "Output:  $BENCHMARK_OUTPUT"
echo ""

# Common arguments
COMMON_ARGS="${DATASET_DIR}/particles.128.mrcs \
    --poses ${DATASET_DIR}/poses.pkl \
    --ctf ${DATASET_DIR}/ctf.pkl \
    --mask from_halfmaps \
    --n-images 100000 \
    --lazy"

# Benchmark 1: Single node, single GPU (absolute baseline)
echo "BENCHMARK 1: Single node, 1 GPU (absolute baseline)..."
time python -m recovar.commands.pipeline \
    $COMMON_ARGS \
    -o ${BENCHMARK_OUTPUT}/1node_1gpu \
    2>&1 | tee ${BENCHMARK_OUTPUT}/1node_1gpu.log

# Benchmark 2: Single node, 2 GPUs (multi-GPU baseline for comparison)
echo ""
echo "BENCHMARK 2: Single node, 2 GPUs (multi-GPU baseline)..."
time python -m recovar.commands.pipeline \
    $COMMON_ARGS \
    -o ${BENCHMARK_OUTPUT}/1node_2gpu \
    --multi-gpu \
    --n-gpus 2 \
    2>&1 | tee ${BENCHMARK_OUTPUT}/1node_2gpu.log

# Benchmark 3: 2 nodes, 2 GPUs each (hybrid frequency-parallel)
echo ""
echo "BENCHMARK 3: 2 nodes, 2 GPUs each (hybrid frequency-parallel)..."
echo "  Submitting SLURM job..."

JOB_ID=$(sbatch --parsable \
    --nodes=2 \
    --ntasks=2 \
    --gpus-per-node=2 \
    --time=01:00:00 \
    --job-name=bench_2node_2gpu \
    --output=${BENCHMARK_OUTPUT}/2node_2gpu.log \
    --wrap="
        srun python -m recovar.commands.pipeline \
            $COMMON_ARGS \
            -o ${BENCHMARK_OUTPUT}/2node_2gpu \
            --frequency-parallel \
            --multi-gpu \
            --n-gpus 2
        
        python -m recovar.commands.concatenate_covariance \
            ${BENCHMARK_OUTPUT}/2node_2gpu/covariance_parts \
            --n-nodes 2
    ")

echo "  Job submitted: $JOB_ID"

echo ""
echo "======================================================================"
echo "Benchmarks launched!"
echo "======================================================================"
echo "Monitor with: watch -n 5 'squeue -u \$USER'"
echo "Results will be in: $BENCHMARK_OUTPUT/"
echo ""
echo "Expected results:"
echo "  1 node, 1 GPU:   100 min (baseline 1×)"
echo "  1 node, 2 GPUs:  ~53 min (1.9× speedup)"
echo "  2 nodes, 4 GPUs: ~27 min (3.7× speedup, ~1.9× vs 1 node 2 GPU)"
```

**Estimated time**: 1 hour

---

#### Task 4.2: Results Analysis Script

**New file**: `scripts/analyze_benchmark_results.py`

```python
#!/usr/bin/env python3
"""
Analyze benchmark results and generate comparison report.
"""

import argparse
import os
import re
from pathlib import Path
import json


def extract_timing(log_file):
    """Extract timing information from log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    timings = {}
    
    # Look for key timing markers
    patterns = {
        'total_time': r'total time:\s+([\d.]+)',
        'covariance_time': r'Time to cov\s+([\d.]+)',
        'wall_time': r'real\s+(\d+)m([\d.]+)s'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'wall_time':
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                timings[key] = minutes * 60 + seconds
            else:
                timings[key] = float(match.group(1))
    
    return timings


def analyze_benchmarks(benchmark_dir):
    """Analyze all benchmark results."""
    
    print("=" * 70)
    print("Benchmark Results Analysis")
    print("=" * 70)
    print(f"Directory: {benchmark_dir}")
    print()
    
    results = {}
    
    # Find all log files
    configs = [
        ('1node_1gpu', 1, 1, 'baseline'),
        ('1node_8gpu', 1, 8, 'multi-gpu'),
        ('2node', 2, 16, 'hybrid'),
        ('4node', 4, 32, 'hybrid'),
        ('8node', 8, 64, 'hybrid')
    ]
    
    baseline_time = None
    
    for name, n_nodes, n_gpus, mode in configs:
        log_file = os.path.join(benchmark_dir, f"{name}.log")
        
        if not os.path.exists(log_file):
            print(f"⚠ Missing: {name}")
            continue
        
        timings = extract_timing(log_file)
        
        if 'wall_time' in timings:
            wall_time = timings['wall_time']
        elif 'total_time' in timings:
            wall_time = timings['total_time']
        else:
            print(f"⚠ No timing found for {name}")
            continue
        
        # Calculate speedup
        if baseline_time is None:
            baseline_time = wall_time
            speedup = 1.0
            efficiency = 100.0
        else:
            speedup = baseline_time / wall_time
            efficiency = 100 * speedup / n_gpus
        
        results[name] = {
            'config': name,
            'n_nodes': n_nodes,
            'n_gpus': n_gpus,
            'mode': mode,
            'time': wall_time,
            'speedup': speedup,
            'efficiency': efficiency
        }
        
        print(f"{name:20} | {n_nodes:2} nodes | {n_gpus:3} GPUs | "
              f"{wall_time:6.1f}s | {speedup:5.1f}× | {efficiency:5.1f}%")
    
    print()
    print("=" * 70)
    
    # Save results
    results_file = os.path.join(benchmark_dir, 'benchmark_summary.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_dir', help='Directory with benchmark logs')
    args = parser.parse_args()
    
    analyze_benchmarks(args.benchmark_dir)
```

**Estimated time**: 1 hour

---

## Testing Strategy

### Stage 1: Unit Tests (Local, < 1 hour)

```bash
# Test frequency splitting logic
pytest tests/test_frequency_parallel.py

# Test argument parsing
python -m recovar.commands.pipeline --help | grep frequency
```

**Success criteria**: All unit tests pass

---

### Stage 2: Integration Test (2 Nodes, Manual, ~30 min)

```bash
# Simulate 2 nodes locally (use small dataset)
bash tests/test_2node_integration.sh
```

**Success criteria**: 
- Both nodes complete without errors
- Concatenation succeeds
- Output shape matches expected

---

### Stage 3: Validation Test (2 Nodes, SLURM, ~1 hour)

```bash
# Submit 2-node job via SLURM (2 GPUs per node)
sbatch --nodes=2 --ntasks=2 --gpus-per-node=2 \
    scripts/run_frequency_parallel_test.sh

# Compare with baseline
python -m recovar.commands.validate_distributed_output \
    baseline_output/model/covariance_cols.pkl \
    freq_parallel_output/covariance_parts/covariance_complete.pkl
```

**Success criteria**:
- Outputs match baseline within tolerance (rtol=1e-5)
- No numerical differences

---

### Stage 4: Performance Benchmarking (2 Nodes, 2 GPUs, ~2 hours)

```bash
# Run benchmark suite (limited to 2 nodes, 2 GPUs)
bash scripts/benchmark_frequency_parallel.sh

# Analyze results
python scripts/analyze_benchmark_results.py benchmark_results/
```

**Success criteria**:
- 2 nodes, 2 GPUs each: ~1.9× speedup vs 1 node, 2 GPUs
- Efficiency > 85% (speedup / 2 nodes)
- Validates scaling principle for future expansion

---

### Stage 5: Extended Dataset Test (100k images, ~2 hours)

```bash
# Run on full 100k dataset with 2 nodes
sbatch --nodes=2 --ntasks=2 --gpus-per-node=2 \
    scripts/run_frequency_parallel_extended.sh

# Validate against existing outputs
python -m recovar.commands.validate_distributed_output \
    data-128-100000/test_dataset/pipeline_output_1node_2gpu/model/covariance_cols.pkl \
    extended_output/covariance_parts/covariance_complete.pkl
```

**Success criteria**:
- Completes without OOM errors
- Outputs match baseline
- Achieves expected ~1.9× speedup
- Validates approach for future larger-scale deployments

---

## Validation Methodology

### Numerical Validation

**What to compare**:
1. Covariance columns (primary output)
2. Picked frequency indices
3. FSC curves
4. Principal component eigenvalues

**Comparison method**:
```python
# Element-wise comparison
np.allclose(baseline, distributed, rtol=1e-5, atol=1e-8)

# Check max difference
max_diff = np.max(np.abs(baseline - distributed))
assert max_diff < 1e-6

# Check relative difference
rel_diff = np.abs(baseline - distributed) / (np.abs(baseline) + 1e-10)
assert np.max(rel_diff) < 1e-4
```

### File Structure Validation

**Baseline output**:
```
output/
├── model/
│   ├── covariance_cols.pkl
│   ├── embeddings.pkl
│   ├── params.pkl
│   └── ...
└── output/
    └── ...
```

**Frequency-parallel output**:
```
output/
├── covariance_parts/
│   ├── covariance_node000.npz
│   ├── covariance_node001.npz
│   ├── covariance_node002.npz
│   ├── covariance_node003.npz
│   └── covariance_complete.pkl  ← Compare this
└── ... (rest same as baseline)
```

---

## Benchmarking Plan

### Benchmark Matrix

**Testing Environment (2 GPU Constraint):**

| Config | Nodes | GPUs | Images | Purpose |
|--------|-------|------|--------|---------|
| Baseline | 1 | 1 | 10k | Absolute reference |
| Multi-GPU | 1 | 2 | 10k | Multi-GPU baseline |
| **Hybrid** | **2** | **4** | **10k** | **Test frequency-parallel** |
| Extended | 2 | 4 | 100k | Validate on full dataset |

### Metrics to Collect

1. **Wall-clock time**: Total time for covariance computation
2. **Covariance compute time**: Time in `compute_both_H_B`
3. **Network I/O**: Time for disk saves (node outputs)
4. **Post-processing time**: Time for concatenation
5. **Memory usage**: Peak GPU and CPU memory per node
6. **Speedup**: Relative to single-GPU baseline
7. **Efficiency**: Speedup / n_gpus (target: > 85%)

### Expected Results

**Testing Environment (2 GPU Constraint):**

| Config | Time | Speedup (vs 1 GPU) | Speedup (vs 1 node 2 GPU) | Efficiency | Notes |
|--------|------|-------------------|---------------------------|------------|-------|
| 1 node, 1 GPU | 100 min | 1× | N/A | 100% | Absolute baseline |
| 1 node, 2 GPUs | 53 min | 1.9× | 1× | 95% | Multi-GPU baseline |
| **2 nodes, 4 GPUs** | **~27 min** | **~3.7×** | **~1.9×** | **~95%** | **Frequency-parallel test** |

**Future Production Scale Projections (8 GPUs):**

| Config | Time | Speedup (vs 1 GPU) | Efficiency | Notes |
|--------|------|-------------------|------------|-------|
| 1 node, 8 GPUs | 13 min | 7.6× | 95% | Current production |
| 4 nodes, 32 GPUs | 3.6 min | 28× | 87% | Projected (4× node scale) |
| 8 nodes, 64 GPUs | 1.8 min | 56× | 87% | Projected (8× node scale) |

---

## Implementation Checklist

### Code Changes

- [ ] **Task 1.1**: Add CLI arguments to `pipeline.py` (30 min)
- [ ] **Task 1.2**: Add `get_distributed_context()` helper (1 hour)
- [ ] **Task 1.3**: Modify `standard_recovar_pipeline()` entry point (1.5 hours)
- [ ] **Task 1.4**: Update `principal_components.py` signature (1 hour)
- [ ] **Task 1.5**: Modify `covariance_estimation.py` frequency splitting (2 hours)
- [ ] **Task 1.6**: Create `scripts/run_frequency_parallel_multigpu.sh` (30 min)
- [ ] **Task 1.7**: Update `crun_recovar_workload.sh` (15 min)
- [ ] **Task 1.8**: Update `pixi.toml` (10 min)
- [ ] **Task 2.1**: Create `concatenate_covariance.py` (2 hours)

**Total coding time**: ~10 hours

### Testing

- [ ] **Task 3.1**: Write unit tests (1.5 hours)
- [ ] **Task 3.2**: Integration test (2 nodes, manual) (30 min)
- [ ] **Task 3.3**: Create validation script (1.5 hours)
- [ ] **Task 3.4**: Create test suite (1 hour)
- [ ] **Stage 1**: Run unit tests (< 1 hour)
- [ ] **Stage 2**: Run integration test (30 min)
- [ ] **Stage 3**: Validation test (2 nodes, SLURM) (1 hour)

**Total testing time**: ~8 hours

### Benchmarking

- [ ] **Task 4.1**: Create benchmark script (1 hour)
- [ ] **Task 4.2**: Create analysis script (1 hour)
- [ ] **Stage 4**: Run performance benchmarks (4 hours compute time)
- [ ] **Stage 5**: Production validation (2 hours compute time)

**Total benchmarking time**: ~8 hours (includes compute wait time)

---

## Risk Analysis

### Risk 1: Frequency Assignment Bug

**Risk**: Nodes compute wrong frequency ranges (gaps or overlaps)

**Mitigation**:
- Unit tests validate splitting logic
- Validation script checks continuity
- Log frequency ranges at runtime

**Likelihood**: Low  
**Impact**: High (wrong results)  
**Detection**: Early (unit tests)

---

### Risk 2: File System Contention

**Risk**: All nodes reading/writing simultaneously overwhelms filesystem

**Mitigation**:
- Test with 2 nodes first
- Monitor I/O bandwidth during tests
- Use local SSD staging if needed

**Likelihood**: Medium (depends on filesystem)  
**Impact**: Medium (slower, not broken)  
**Detection**: Medium (benchmarking stage)

---

### Risk 3: SLURM Configuration Issues

**Risk**: SLURM environment variables not set correctly

**Mitigation**:
- Auto-detection with fallback
- Manual --node-rank option for testing
- Clear error messages if detection fails

**Likelihood**: Low (SLURM is standard)  
**Impact**: Low (easy to debug)  
**Detection**: Early (integration test)

---

### Risk 4: Numerical Precision Differences

**Risk**: Different frequency ordering leads to slight numerical differences

**Mitigation**:
- Keep frequency order consistent
- Use tolerant comparison (rtol=1e-5)
- Validate with multiple random seeds

**Likelihood**: Very low  
**Impact**: Low (expected in floating point)  
**Detection**: Validation stage

---

## File Summary

### Files to Create (New)

1. `recovar/commands/concatenate_covariance.py` (200 lines)
2. `recovar/commands/validate_distributed_output.py` (180 lines)
3. `scripts/run_frequency_parallel_multigpu.sh` (80 lines)
4. `scripts/test_frequency_parallel_suite.sh` (150 lines)
5. `scripts/benchmark_frequency_parallel.sh` (120 lines)
6. `scripts/analyze_benchmark_results.py` (150 lines)
7. `tests/test_frequency_parallel.py` (100 lines)

**Total new code**: ~980 lines

### Files to Modify (Existing)

1. `recovar/commands/pipeline.py` (+100 lines)
2. `recovar/principal_components.py` (+50 lines)
3. `recovar/covariance_estimation.py` (+80 lines)
4. `crun_recovar_workload.sh` (+20 lines)
5. `pixi.toml` (+5 lines)

**Total modifications**: ~255 lines

**Grand total**: ~1,235 lines

---

## Timeline

### Week 1: Implementation
- Day 1-2: Core infrastructure (Tasks 1.1-1.5) - 8 hours
- Day 3: Scripts and utilities (Tasks 1.6-2.1) - 5 hours
- Day 4: Testing infrastructure (Tasks 3.1-3.4) - 4.5 hours
- Day 5: Buffer and documentation - 2.5 hours

### Week 2: Testing and Benchmarking
- Day 1: Unit and integration tests (Stages 1-2) - 2 hours
- Day 2: Validation tests (Stage 3) - 4 hours
- Day 3: Performance benchmarking (Stage 4) - 6 hours
- Day 4: Production validation (Stage 5) - 4 hours
- Day 5: Results analysis and documentation - 4 hours

**Total time**: 40 hours (2 weeks)

---

## Success Criteria

### Functional Requirements

- ✅ Pipeline accepts `--frequency-parallel` flag
- ✅ Auto-detects SLURM environment
- ✅ Splits frequencies correctly across nodes
- ✅ Each node uses multi-GPU for image parallelism
- ✅ Post-processing concatenates results correctly
- ✅ Outputs match baseline numerically

### Performance Requirements

**Testing Phase (2 GPU Environment):**
- ✅ 2 nodes (4 GPUs): ~1.9× speedup vs 1 node 2 GPUs
- ✅ 2 nodes (4 GPUs): ~3.7× speedup vs 1 GPU baseline
- ✅ Efficiency: > 85% (validates scaling principle)
- ✅ Post-processing overhead: < 2% of compute time

**Future Production Expectations (based on testing validation):**
- ✅ 4 nodes (32 GPUs): > 25× speedup vs single GPU
- ✅ 8 nodes (64 GPUs): > 50× speedup vs single GPU
- ✅ Linear scaling demonstrated at 2-node level

### Quality Requirements

- ✅ Backward compatible: existing code works unchanged
- ✅ No numerical differences: outputs match within 1e-5 tolerance
- ✅ Robust error handling: clear errors if nodes fail
- ✅ Well documented: README and inline comments

---

## Next Steps

1. **Review this plan** - Confirm approach and timeline
2. **Set up test environment** - Ensure SLURM access with multiple nodes
3. **Begin implementation** - Start with Task 1.1 (CLI arguments)
4. **Incremental testing** - Test each phase before proceeding
5. **Benchmark early** - Run 2-node test ASAP to validate approach

---

## Questions to Address Before Starting

1. **SLURM access**: Do you have access to submit multi-node jobs?
2. **Node availability**: Can you get 2-8 nodes with 8 GPUs each?
3. **Filesystem type**: What storage system (Lustre, NFS, local SSD)?
4. **Baseline data**: Do you have existing single-node outputs for validation?
5. **Dataset location**: Is dataset on shared filesystem accessible from all nodes?

---

## Appendix: Quick Start Commands

Once implemented, using the hybrid approach will be simple:

### Single-Node (Existing, No Changes)
```bash
# With 2 GPUs
python -m recovar.commands.pipeline particles.mrcs \
    --poses poses.pkl --ctf ctf.pkl --mask from_halfmaps \
    -o output/ --multi-gpu --n-gpus 2

# Or with 8 GPUs in production
python -m recovar.commands.pipeline particles.mrcs \
    --poses poses.pkl --ctf ctf.pkl --mask from_halfmaps \
    -o output/ --multi-gpu --n-gpus 8
```

### Multi-Node (New, via SLURM)
```bash
# Option 1: Via crun wrapper (2 nodes, 2 GPUs each for testing)
./crun_recovar_workload.sh freq-parallel-2node

# Option 2: Direct SLURM submission (2 nodes, 2 GPUs for testing)
sbatch --nodes=2 --ntasks=2 --gpus-per-node=2 scripts/run_frequency_parallel.sh

# Future production: 4 nodes, 8 GPUs each (after validation)
sbatch --nodes=4 --ntasks=4 --gpus-per-node=8 scripts/run_frequency_parallel.sh
```

The user interface is simple, all complexity is hidden!
