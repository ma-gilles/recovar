#!/usr/bin/env python3
"""
Benchmark different array stacking strategies for compute_H_B optimization.

Simulates the workload from recovar/covariance_estimation.py:878-880 to test
optimization strategies in isolation before implementing in production.

Usage:
    python benchmark_stack.py --output results.json --runs 3
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
import json
from typing import List, Dict

# Real workload parameters from production code
# Default: 128^3 volume, 300 picked frequencies (can be overridden via CLI)
DEFAULT_VOLUME_SIZE = 128**3  # 2,097,152 elements
DEFAULT_N_PICKED = 300        # From covariance_options['sampling_n_cols']
DTYPE = np.complex64  # 8 bytes per element

def create_test_data(volume_size: int, n_picked: int) -> List:
    """Create JAX arrays on GPU simulating accumulated H/B arrays."""
    print(f"Creating test data: {n_picked} arrays of size {volume_size:,}")
    H_list = []
    for i in range(n_picked):
        # Simulate accumulated arrays with variation to prevent optimization
        arr = jnp.ones(volume_size, dtype=DTYPE) * (i + 1.0 + 0.1j)
        H_list.append(arr)
    # Block until all arrays are on GPU
    jax.block_until_ready(H_list[-1])
    return H_list

def benchmark_current_approach(H_list: List, volume_size: int, n_picked: int) -> Dict:
    """Strategy 0: Current - np.stack on list of JAX arrays (BASELINE)"""
    print("\n=== Benchmark: Current Approach (np.stack) ===")
    
    start = time.perf_counter()
    H = np.stack(H_list, axis=1)
    end = time.perf_counter()
    
    return {
        "name": "current_np_stack",
        "time_seconds": end - start,
        "shape": list(H.shape),
        "size_mb": float(H.nbytes / (1024**2))
    }

def benchmark_jax_stack(H_list: List, volume_size: int, n_picked: int) -> Dict:
    """Strategy A: JAX stack on GPU + single bulk transfer"""
    print("\n=== Benchmark: JAX Stack (GPU) + Single Transfer ===")
    
    start = time.perf_counter()
    H_jax = jnp.stack(H_list, axis=1)  # Stack on GPU (parallel)
    jax.block_until_ready(H_jax)        # Ensure GPU work completes
    H = np.asarray(H_jax)               # Single DtoH transfer
    end = time.perf_counter()
    
    return {
        "name": "jax_stack_single_transfer",
        "time_seconds": end - start,
        "shape": list(H.shape),
        "size_mb": float(H.nbytes / (1024**2))
    }

def benchmark_preallocated(H_list: List, volume_size: int, n_picked: int) -> Dict:
    """Strategy B: Pre-allocated NumPy + sequential direct assignment"""
    print("\n=== Benchmark: Pre-allocated + Direct Assignment ===")
    
    H = np.empty([volume_size, n_picked], dtype=DTYPE)
    
    start = time.perf_counter()
    for k in range(n_picked):
        H[:, k] = jax.device_get(H_list[k])
    end = time.perf_counter()
    
    return {
        "name": "preallocated_direct",
        "time_seconds": end - start,
        "shape": list(H.shape),
        "size_mb": float(H.nbytes / (1024**2))
    }

def benchmark_batched_transfer(H_list: List, volume_size: int, n_picked: int, batch_size: int = 10) -> Dict:
    """Strategy C: Batched GPU stack + batched transfers"""
    print(f"\n=== Benchmark: Batched Transfer (batch_size={batch_size}) ===")
    
    H = np.empty([volume_size, n_picked], dtype=DTYPE)
    
    start = time.perf_counter()
    for batch_start in range(0, n_picked, batch_size):
        batch_end = min(batch_start + batch_size, n_picked)
        H_batch = jnp.stack(H_list[batch_start:batch_end], axis=1)
        jax.block_until_ready(H_batch)
        H[:, batch_start:batch_end] = np.asarray(H_batch)
    end = time.perf_counter()
    
    return {
        "name": f"batched_transfer_b{batch_size}",
        "time_seconds": end - start,
        "shape": list(H.shape),
        "size_mb": float(H.nbytes / (1024**2))
    }

def benchmark_async_transfer(H_list: List, volume_size: int, n_picked: int) -> Dict:
    """Strategy D: Async transfers with overlapped CPU/GPU operations"""
    print("\n=== Benchmark: Async Transfer (Overlapped) ===")
    
    import threading
    import queue
    
    H = np.empty([volume_size, n_picked], dtype=DTYPE)
    transfer_queue = queue.Queue(maxsize=2)  # Pipeline with 2-deep buffer
    
    def transfer_worker():
        """Background thread to handle transfers"""
        while True:
            item = transfer_queue.get()
            if item is None:  # Sentinel to stop
                break
            k, arr_jax = item
            H[:, k] = np.asarray(arr_jax)
            transfer_queue.task_done()
    
    start = time.perf_counter()
    
    # Start background transfer thread
    worker = threading.Thread(target=transfer_worker, daemon=True)
    worker.start()
    
    # Queue transfers (this can overlap with GPU work)
    for k in range(n_picked):
        transfer_queue.put((k, H_list[k]))
    
    # Wait for all transfers to complete
    transfer_queue.join()
    transfer_queue.put(None)  # Stop worker
    worker.join()
    
    end = time.perf_counter()
    
    return {
        "name": "async_transfer",
        "time_seconds": end - start,
        "shape": list(H.shape),
        "size_mb": float(H.nbytes / (1024**2))
    }

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark array stacking strategies for compute_H_B optimization"
    )
    parser.add_argument("--output", default="benchmark_results.json", 
                       help="Output JSON file for results")
    parser.add_argument("--warmup", type=int, default=1, 
                       help="Number of warmup runs (default: 1)")
    parser.add_argument("--runs", type=int, default=3, 
                       help="Number of benchmark runs (default: 3)")
    parser.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE,
                       help=f"Volume size in elements (default: {DEFAULT_VOLUME_SIZE:,})")
    parser.add_argument("--n-picked", type=int, default=DEFAULT_N_PICKED,
                       help=f"Number of picked frequencies (default: {DEFAULT_N_PICKED})")
    parser.add_argument("--skip-oom-risk", action="store_true",
                       help="Skip strategies that may cause OOM (e.g., full JAX stack)")
    parser.add_argument("--test-async", action="store_true",
                       help="Include async transfer strategy in benchmarks")
    args = parser.parse_args()
    
    volume_size = args.volume_size
    n_picked = args.n_picked
    
    print("=" * 70)
    print("Array Stacking Optimization Benchmark")
    print("=" * 70)
    print(f"Volume size: {volume_size:,} elements")
    print(f"Picked frequencies: {n_picked}")
    print(f"Data type: {DTYPE}")
    print(f"Single array size: {volume_size * n_picked * 8 / (1024**2):.1f} MB")
    print(f"Total (H + B): {volume_size * n_picked * 8 * 2 / (1024**2):.1f} MB")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    if args.skip_oom_risk:
        print("⚠️  Skipping OOM-risk strategies (full JAX stack)")
    
    # Warmup to ensure JIT compilation doesn't affect results
    print(f"\n{'='*70}")
    print(f"Warmup Phase ({args.warmup} runs)")
    print(f"{'='*70}")
    for i in range(args.warmup):
        print(f"\nWarmup run {i + 1}/{args.warmup}")
        H_list = create_test_data(volume_size, n_picked)
        _ = benchmark_current_approach(H_list, volume_size, n_picked)
        del H_list
    
    # Run benchmarks
    results = []
    for run in range(args.runs):
        print(f"\n{'='*70}")
        print(f"Benchmark Run {run + 1}/{args.runs}")
        print(f"{'='*70}")
        
        H_list = create_test_data(volume_size, n_picked)
        
        # Test all strategies
        results.append(benchmark_current_approach(H_list, volume_size, n_picked))
        
        # Conditionally test JAX stack (may OOM on large arrays)
        if not args.skip_oom_risk:
            try:
                results.append(benchmark_jax_stack(H_list, volume_size, n_picked))
            except Exception as e:
                print(f"⚠️  JAX stack failed (likely OOM): {e}")
                results.append({
                    "name": "jax_stack_single_transfer",
                    "time_seconds": float('inf'),
                    "shape": [volume_size, n_picked],
                    "size_mb": volume_size * n_picked * 8 / (1024**2),
                    "error": str(e)
                })
        
        results.append(benchmark_preallocated(H_list, volume_size, n_picked))
        results.append(benchmark_batched_transfer(H_list, volume_size, n_picked, batch_size=10))
        results.append(benchmark_batched_transfer(H_list, volume_size, n_picked, batch_size=50))
        
        # Optionally test async strategy
        if args.test_async:
            results.append(benchmark_async_transfer(H_list, volume_size, n_picked))
        
        del H_list
    
    # Aggregate and analyze results
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    
    strategies = {}
    for result in results:
        name = result["name"]
        if name not in strategies:
            strategies[name] = []
        strategies[name].append(result["time_seconds"])
    
    summary = []
    baseline_time = np.mean(strategies["current_np_stack"])
    
    print(f"\nBaseline (current_np_stack): {baseline_time:.3f}s")
    print("\n" + "-" * 70)
    
    for name, times in strategies.items():
        mean_time = np.mean(times)
        std_time = np.std(times)
        speedup = baseline_time / mean_time
        improvement_pct = (1 - mean_time / baseline_time) * 100
        
        summary.append({
            "strategy": name,
            "mean_time_s": float(mean_time),
            "std_time_s": float(std_time),
            "speedup": float(speedup),
            "improvement_pct": float(improvement_pct)
        })
        
        print(f"\n{name}:")
        print(f"  Time: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {improvement_pct:+.1f}%")
    
    # Save results to JSON
    output_data = {
        "config": {
            "volume_size": volume_size,
            "n_picked": n_picked,
            "dtype": str(DTYPE),
            "warmup_runs": args.warmup,
            "benchmark_runs": args.runs,
            "skip_oom_risk": args.skip_oom_risk,
            "test_async": args.test_async,
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()]
        },
        "raw_results": results,
        "summary": summary,
        "recommendation": min(summary, key=lambda x: x["mean_time_s"]) if summary else None
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}")
    
    # Print recommendation
    if len(summary) > 1:
        best = min(summary, key=lambda x: x["mean_time_s"])
        print(f"\n✓ RECOMMENDATION: {best['strategy']}")
        print(f"  Expected improvement: {best['improvement_pct']:.1f}%")
        print(f"  Expected time: {best['mean_time_s']:.3f}s (baseline: {baseline_time:.3f}s)")

if __name__ == "__main__":
    main()
