#!/usr/bin/env python3
"""
Analyze and compare prototype results.

Usage:
    python analyze_prototype_results.py prototype_output_20260129_123456
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def load_result(filepath):
    """Load a result file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.load(filepath)
    return {
        'covariance_cols': data['covariance_cols'],
        'picked_frequencies': data['picked_frequencies'],
        'fscs': data.get('fscs', None),
        'freq_start': int(data.get('freq_start', 0)),
        'freq_end': int(data.get('freq_end', len(data['picked_frequencies']))),
        'node_rank': int(data.get('node_rank', 0)),
        'compute_time': float(data.get('compute_time', 0)),
        'n_nodes': int(data.get('n_nodes', 1))
    }


def analyze_results(output_dir):
    """Analyze prototype results."""
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("PROTOTYPE RESULTS ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load baseline
    baseline_file = output_dir / 'baseline' / 'node000_result.npz'
    
    if not baseline_file.exists():
        print(f"ERROR: Baseline file not found: {baseline_file}")
        return False
    
    print("Loading baseline...")
    baseline = load_result(baseline_file)
    
    print(f"  File: {baseline_file.name}")
    print(f"  Covariance shape: {baseline['covariance_cols'].shape}")
    print(f"  Frequencies: {len(baseline['picked_frequencies'])}")
    print(f"  Compute time: {baseline['compute_time']:.2f}s")
    print()
    
    # Load frequency-parallel results
    freq_parallel_dir = output_dir / 'freq_parallel'
    
    if not freq_parallel_dir.exists():
        print(f"ERROR: Frequency-parallel directory not found: {freq_parallel_dir}")
        return False
    
    # Load concatenated result
    concat_file = freq_parallel_dir / 'concatenated_result.npz'
    
    if not concat_file.exists():
        print(f"ERROR: Concatenated file not found: {concat_file}")
        print("Make sure to run concatenation after both nodes complete.")
        return False
    
    print("Loading frequency-parallel results...")
    concat = load_result(concat_file)
    n_nodes = concat['n_nodes']
    
    print(f"  File: {concat_file.name}")
    print(f"  Covariance shape: {concat['covariance_cols'].shape}")
    print(f"  Frequencies: {len(concat['picked_frequencies'])}")
    print(f"  Number of nodes: {n_nodes}")
    print()
    
    # Load individual node results
    print(f"Loading individual node results ({n_nodes} nodes)...")
    node_results = []
    
    for node_rank in range(n_nodes):
        node_file = freq_parallel_dir / f'node{node_rank:03d}_result.npz'
        
        if not node_file.exists():
            print(f"  WARNING: Node {node_rank} file not found: {node_file.name}")
            continue
        
        node_data = load_result(node_file)
        node_results.append(node_data)
        
        print(f"  Node {node_rank}:")
        print(f"    Shape: {node_data['covariance_cols'].shape}")
        print(f"    Frequencies: {len(node_data['picked_frequencies'])} "
              f"[{node_data['freq_start']}:{node_data['freq_end']}]")
        print(f"    Compute time: {node_data['compute_time']:.2f}s")
    
    print()
    
    # Validation
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    all_valid = True
    
    # Check shapes
    print("Checking shapes...")
    if baseline['covariance_cols'].shape == concat['covariance_cols'].shape:
        print("  ✓ Shapes match")
    else:
        print(f"  ✗ Shape mismatch!")
        print(f"    Baseline: {baseline['covariance_cols'].shape}")
        print(f"    Concatenated: {concat['covariance_cols'].shape}")
        all_valid = False
    
    # Check frequency counts
    print("Checking frequency counts...")
    if len(baseline['picked_frequencies']) == len(concat['picked_frequencies']):
        print("  ✓ Frequency counts match")
    else:
        print(f"  ✗ Frequency count mismatch!")
        print(f"    Baseline: {len(baseline['picked_frequencies'])}")
        print(f"    Concatenated: {len(concat['picked_frequencies'])}")
        all_valid = False
    
    # Check frequency indices
    print("Checking frequency indices...")
    if np.array_equal(baseline['picked_frequencies'], concat['picked_frequencies']):
        print("  ✓ Frequency indices match exactly")
    else:
        # Check if they're the same set (just different order)
        if set(baseline['picked_frequencies']) == set(concat['picked_frequencies']):
            print("  ⚠ Frequency indices match (different order)")
        else:
            print("  ✗ Frequency indices differ!")
            all_valid = False
    
    # Check numerical accuracy (if shapes match)
    if baseline['covariance_cols'].shape == concat['covariance_cols'].shape:
        print("Checking numerical accuracy...")
        
        abs_diff = np.abs(baseline['covariance_cols'] - concat['covariance_cols'])
        rel_diff = abs_diff / (np.abs(baseline['covariance_cols']) + 1e-10)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        
        # Tolerance check
        rtol = 1e-5
        atol = 1e-8
        
        if np.allclose(baseline['covariance_cols'], concat['covariance_cols'], 
                       rtol=rtol, atol=atol):
            print(f"  ✓ Numerical accuracy within tolerance (rtol={rtol}, atol={atol})")
        else:
            n_different = np.sum((abs_diff > atol) & (rel_diff > rtol))
            pct_different = 100 * n_different / abs_diff.size
            print(f"  ⚠ Some differences exceed tolerance:")
            print(f"    {n_different} elements ({pct_different:.2f}%)")
            print(f"    This may be acceptable for prototype validation")
    
    print()
    
    # Performance analysis
    print("=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    baseline_time = baseline['compute_time']
    
    if node_results:
        node_times = [nr['compute_time'] for nr in node_results]
        max_node_time = max(node_times)
        total_node_time = sum(node_times)
        
        print(f"Baseline (1 node, all frequencies):")
        print(f"  Compute time: {baseline_time:.2f}s")
        print()
        
        print(f"Frequency-parallel ({n_nodes} nodes):")
        for i, nr in enumerate(node_results):
            print(f"  Node {i} compute time: {nr['compute_time']:.2f}s")
        
        print(f"  Max node time (wall-clock): {max_node_time:.2f}s")
        print(f"  Total compute time: {total_node_time:.2f}s")
        print()
        
        # Calculate speedup
        if max_node_time > 0:
            speedup = baseline_time / max_node_time
            efficiency = (speedup / n_nodes) * 100
            
            print(f"Speedup analysis:")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  Expected (ideal): {n_nodes}×")
            print(f"  Parallel efficiency: {efficiency:.1f}%")
            print()
            
            # Interpret results
            if speedup >= n_nodes * 0.9:
                print("  ✓ EXCELLENT: Near-linear scaling!")
                print("    Frequency-parallel approach is working very well.")
            elif speedup >= n_nodes * 0.75:
                print("  ✓ GOOD: Significant speedup with acceptable overhead.")
                print("    Frequency-parallel approach is working well.")
            elif speedup >= n_nodes * 0.5:
                print("  ⚠ ACCEPTABLE: Speedup achieved but with noticeable overhead.")
                print("    Consider investigating overhead sources.")
            else:
                print("  ✗ POOR: Speedup less than expected.")
                print("    May need to debug or optimize further.")
                all_valid = False
        
        # Load balance analysis
        print()
        print("Load balance:")
        min_time = min(node_times)
        max_time = max(node_times)
        imbalance = (max_time - min_time) / max_time * 100
        
        print(f"  Min node time: {min_time:.2f}s")
        print(f"  Max node time: {max_time:.2f}s")
        print(f"  Imbalance: {imbalance:.1f}%")
        
        if imbalance < 10:
            print("  ✓ Well balanced")
        elif imbalance < 25:
            print("  ⚠ Moderate imbalance")
        else:
            print("  ✗ Significant imbalance - check frequency splitting")
    
    print()
    print("=" * 70)
    
    if all_valid:
        print("✓ PROTOTYPE VALIDATION PASSED")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review performance results above")
        print("  2. Proceed with full implementation")
        print("  3. See HYBRID_MULTINODE_IMPLEMENTATION_PLAN.md")
        return True
    else:
        print("✗ PROTOTYPE VALIDATION FAILED")
        print("=" * 70)
        print()
        print("Issues detected - please review errors above.")
        print("Debug and iterate on prototype before full implementation.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prototype results"
    )
    
    parser.add_argument(
        'output_dir',
        help='Prototype output directory'
    )
    
    args = parser.parse_args()
    
    success = analyze_results(args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
