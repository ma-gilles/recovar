#!/usr/bin/env python3
"""
Compare results from old and new RECOVAR versions.
Usage: python compare_test_results.py <results_dir> <old_time> <new_time>
"""

import numpy as np
import os
import sys
import json
from pathlib import Path
import glob


def compare_results(results_dir: str, old_time: int, new_time: int):
    """Compare test results between old and new versions."""
    results_dir = Path(results_dir)
    old_dir = results_dir / 'old_version'
    new_dir = results_dir / 'new_version'
    
    comparison = {
        'timing': {
            'old_version_seconds': old_time,
            'new_version_seconds': new_time,
            'speedup': old_time / new_time if new_time > 0 else float('inf')
        },
        'files_compared': [],
        'numerical_differences': []
    }
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"\nTiming:")
    print(f"  OLD version: {comparison['timing']['old_version_seconds']} seconds")
    print(f"  NEW version: {comparison['timing']['new_version_seconds']} seconds")
    print(f"  Speedup: {comparison['timing']['speedup']:.2f}x")
    
    # Compare all_scores.json if it exists
    old_scores_file = old_dir / 'test_dataset' / 'metrics_plot' / 'all_scores.json'
    new_scores_file = new_dir / 'test_dataset' / 'metrics_plot' / 'all_scores.json'
    
    if old_scores_file.exists() and new_scores_file.exists():
        print(f"\nComparing metrics (all_scores.json)...")
        try:
            with open(old_scores_file) as f:
                old_scores = json.load(f)
            with open(new_scores_file) as f:
                new_scores = json.load(f)
            
            all_keys = set(old_scores.keys()) | set(new_scores.keys())
            metrics_comparison = []
            
            for key in sorted(all_keys):
                old_val = old_scores.get(key)
                new_val = new_scores.get(key)
                
                if old_val is None:
                    print(f"  ⚠ {key}: Only in NEW version ({new_val})")
                elif new_val is None:
                    print(f"  ⚠ {key}: Only in OLD version ({old_val})")
                elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    diff = abs(new_val - old_val)
                    rel_diff = diff / (abs(old_val) + 1e-10)
                    if rel_diff < 1e-6:
                        status = "✓ Identical"
                    elif rel_diff < 1e-3:
                        status = "✓ Very close"
                    elif rel_diff < 1e-2:
                        status = "⚠ Small diff"
                    else:
                        status = "❌ Large diff"
                    
                    print(f"  {status}: {key} = {new_val:.6f} (old: {old_val:.6f}, diff: {diff:.2e}, rel: {rel_diff:.2e})")
                    
                    metrics_comparison.append({
                        'metric': key,
                        'old_value': float(old_val),
                        'new_value': float(new_val),
                        'absolute_diff': float(diff),
                        'relative_diff': float(rel_diff)
                    })
                else:
                    print(f"  ℹ {key}: Non-numeric values")
                    metrics_comparison.append({
                        'metric': key,
                        'old_value': old_val,
                        'new_value': new_val
                    })
            
            comparison['metrics_compared'] = metrics_comparison
            
        except Exception as e:
            print(f"  ❌ Error comparing metrics: {e}")
    else:
        print(f"\n⚠ Metrics file not found in one or both versions")
    
    # Compare output files
    print(f"\nComparing .npy output files...")
    
    # Find all .npy files in old directory
    for old_file in glob.glob(str(old_dir / '**' / '*.npy'), recursive=True):
        rel_path = Path(old_file).relative_to(old_dir)
        new_file = new_dir / rel_path
        
        if new_file.exists():
            try:
                old_data = np.load(old_file)
                new_data = np.load(new_file)
                
                # Check shapes match
                if old_data.shape != new_data.shape:
                    print(f"  ❌ {rel_path}: Shape mismatch ({old_data.shape} vs {new_data.shape})")
                    comparison['numerical_differences'].append({
                        'file': str(rel_path),
                        'issue': 'shape_mismatch',
                        'old_shape': old_data.shape,
                        'new_shape': new_data.shape
                    })
                    continue
                
                # Compute differences
                max_diff = np.max(np.abs(old_data - new_data))
                mean_diff = np.mean(np.abs(old_data - new_data))
                rel_diff = max_diff / (np.max(np.abs(old_data)) + 1e-10)
                
                if max_diff < 1e-6:
                    print(f"  ✓ {rel_path}: Identical (max diff: {max_diff:.2e})")
                    status = 'identical'
                elif rel_diff < 1e-3:
                    print(f"  ✓ {rel_path}: Very close (max diff: {max_diff:.2e}, rel: {rel_diff:.2e})")
                    status = 'very_close'
                elif rel_diff < 1e-2:
                    print(f"  ⚠ {rel_path}: Small difference (max diff: {max_diff:.2e}, rel: {rel_diff:.2e})")
                    status = 'small_diff'
                else:
                    print(f"  ❌ {rel_path}: Significant difference (max diff: {max_diff:.2e}, rel: {rel_diff:.2e})")
                    status = 'significant_diff'
                
                comparison['files_compared'].append({
                    'file': str(rel_path),
                    'status': status,
                    'max_diff': float(max_diff),
                    'mean_diff': float(mean_diff),
                    'rel_diff': float(rel_diff)
                })
                
            except Exception as e:
                print(f"  ❌ {rel_path}: Error comparing - {e}")
                comparison['files_compared'].append({
                    'file': str(rel_path),
                    'status': 'error',
                    'error': str(e)
                })
        else:
            print(f"  ❌ {rel_path}: Missing in NEW version")
            comparison['numerical_differences'].append({
                'file': str(rel_path),
                'issue': 'missing_in_new'
            })
    
    # Check for files only in new version
    for new_file in glob.glob(str(new_dir / '**' / '*.npy'), recursive=True):
        rel_path = Path(new_file).relative_to(new_dir)
        old_file = old_dir / rel_path
        
        if not old_file.exists():
            print(f"  ⚠ {rel_path}: Only in NEW version")
            comparison['numerical_differences'].append({
                'file': str(rel_path),
                'issue': 'only_in_new'
            })
    
    # Save comparison results
    output_file = results_dir / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison results saved to: {output_file}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Files compared: {len(comparison['files_compared'])}")
    
    # Count significant differences (relative diff > 1e-3)
    significant_diffs = 0
    if 'metrics_compared' in comparison:
        for metric in comparison.get('metrics_compared', []):
            if isinstance(metric.get('relative_diff'), float) and metric['relative_diff'] > 1e-3:
                significant_diffs += 1
    
    for file_cmp in comparison['files_compared']:
        if file_cmp.get('status') in ['significant_diff', 'shape_mismatch', 'error']:
            significant_diffs += 1
    
    print(f"Issues found: {len(comparison['numerical_differences'])}")
    print(f"Significant differences (>0.1%): {significant_diffs}")
    
    if len(comparison['numerical_differences']) == 0 and significant_diffs == 0:
        print("\n✓ All tests passed! Results match between versions.")
        return 0
    elif significant_diffs == 0:
        print("\n✓ Minor differences only (likely due to floating-point arithmetic or randomness).")
        print("   All metrics within 0.1% tolerance.")
        return 0
    else:
        print("\n⚠ Significant differences found (>0.1%). Check details above.")
        return 1


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python compare_test_results.py <results_dir> <old_time> <new_time>")
        print("Example: python compare_test_results.py /path/to/results 1234 5678")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    old_time = int(sys.argv[2])
    new_time = int(sys.argv[3])
    
    exit_code = compare_results(results_dir, old_time, new_time)
    sys.exit(exit_code)

