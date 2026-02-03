#!/usr/bin/env python3
"""
Analyze NVTX profiling data from recovar_2gpu_profile.sqlite.

Extracts timing information for stack_results and related operations
to understand the current performance baseline.

Usage:
    python analyze_profile.py path/to/profile.sqlite --output analysis.json
"""

import sqlite3
import argparse
import json
from pathlib import Path
from typing import Dict

def analyze_nvtx_events(db_path: str, output_json: str = None) -> Dict:
    """Extract NVTX timing for stack_results and related operations."""
    
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("NVTX Profile Analysis")
    print("=" * 70)
    print(f"Database: {db_path}\n")
    
    # Query for timing of key operations
    cursor.execute("""
        SELECT 
            s.value as event_name,
            COUNT(*) as count,
            MIN((n.end - n.start)/1e9) as min_s,
            MAX((n.end - n.start)/1e9) as max_s,
            AVG((n.end - n.start)/1e9) as avg_s,
            SUM((n.end - n.start)/1e9) as total_s
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE s.value IN ('stack_results', 'accumulate_H_B', 'frequency_loop', 
                          'compute_H_B', 'cleanup_batch', 'transfer_to_cpu')
        GROUP BY s.value
        ORDER BY avg_s DESC
    """)
    
    results = {}
    print("{:<30} {:>8} {:>12} {:>12} {:>12}".format(
        "Event", "Count", "Avg (s)", "Min (s)", "Max (s)"
    ))
    print("-" * 70)
    
    for row in cursor.fetchall():
        name, count, min_s, max_s, avg_s, total_s = row
        results[name] = {
            "count": count,
            "min_seconds": float(min_s) if min_s else 0.0,
            "max_seconds": float(max_s) if max_s else 0.0,
            "avg_seconds": float(avg_s) if avg_s else 0.0,
            "total_seconds": float(total_s) if total_s else 0.0
        }
        print("{:<30} {:>8} {:>12.3f} {:>12.3f} {:>12.3f}".format(
            name[:28], count, avg_s, min_s, max_s
        ))
    
    # Query for memory copy operations
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            SUM(bytes)/1e9 as total_gb,
            AVG(bytes)/1e6 as avg_mb,
            SUM((end - start)/1e9) as total_time_s
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 2  -- DtoH transfers
    """)
    
    memcpy_row = cursor.fetchone()
    if memcpy_row and memcpy_row[0]:
        count, total_gb, avg_mb, total_time = memcpy_row
        results["dtoh_memcpy"] = {
            "count": count,
            "total_gb": float(total_gb) if total_gb else 0.0,
            "avg_mb_per_transfer": float(avg_mb) if avg_mb else 0.0,
            "total_time_s": float(total_time) if total_time else 0.0
        }
        print(f"\nDtoH Memory Transfers:")
        print(f"  Count: {count}")
        print(f"  Total data: {total_gb:.2f} GB")
        print(f"  Avg per transfer: {avg_mb:.2f} MB")
        print(f"  Total time: {total_time:.3f}s")
    
    conn.close()
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NVTX profiling data from SQLite database"
    )
    parser.add_argument("db_path", help="Path to .sqlite profiling database")
    parser.add_argument("--output", help="Output JSON file (optional)")
    args = parser.parse_args()
    
    try:
        analyze_nvtx_events(args.db_path, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
