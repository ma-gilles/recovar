#!/usr/bin/env python3
"""
Concatenate covariance results from multi-node frequency-parallel computation.

This script loads per-node covariance results and combines them into a single
coherent covariance matrix that can be used for PCA and downstream analysis.

Usage:
    python -m recovar.commands.concatenate_covariance <covariance_dir> [--n-nodes N]
    
Example:
    python -m recovar.commands.concatenate_covariance covariance_parts --n-nodes 2
"""

import argparse
import logging
import os
import pickle
import sys
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def find_covariance_files(covariance_dir):
    """
    Find all covariance_nodeXXX.pkl files in the directory.
    
    Returns:
        List of (node_rank, filepath) tuples sorted by node_rank
    """
    pattern = "covariance_node*.pkl"
    files = []
    
    for filepath in Path(covariance_dir).glob(pattern):
        # Extract node number from filename (e.g., covariance_node000.pkl -> 0)
        filename = filepath.name
        node_str = filename.replace("covariance_node", "").replace(".pkl", "")
        try:
            node_rank = int(node_str)
            files.append((node_rank, str(filepath)))
        except ValueError:
            logger.warning(f"Skipping file with invalid node number: {filename}")
    
    # Sort by node_rank
    files.sort(key=lambda x: x[0])
    
    return files


def load_node_result(filepath):
    """Load covariance result from a single node."""
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    
    required_keys = ['covariance_cols', 'picked_frequencies', 
                     'picked_frequencies_global_indices', 'column_fscs',
                     'node_rank', 'n_nodes', 'freq_start', 'freq_end']
    
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing required key '{key}' in {filepath}")
    
    return result


def validate_node_results(node_results):
    """
    Validate that node results are compatible for concatenation.
    
    Checks:
    - All nodes have same n_nodes value
    - Node ranks are sequential from 0 to n_nodes-1
    - Frequency ranges are contiguous (no gaps or overlaps)
    """
    if not node_results:
        raise ValueError("No node results to validate")
    
    n_nodes = node_results[0]['n_nodes']
    expected_node_ranks = set(range(n_nodes))
    actual_node_ranks = {r['node_rank'] for r in node_results}
    
    # Check node count consistency
    for result in node_results:
        if result['n_nodes'] != n_nodes:
            raise ValueError(
                f"Inconsistent n_nodes: expected {n_nodes}, "
                f"got {result['n_nodes']} for node {result['node_rank']}"
            )
    
    # Check we have all nodes
    if actual_node_ranks != expected_node_ranks:
        missing = expected_node_ranks - actual_node_ranks
        raise ValueError(
            f"Missing results from nodes: {sorted(missing)}\n"
            f"Found results from nodes: {sorted(actual_node_ranks)}"
        )
    
    # Check frequency range continuity
    # Sort by freq_start to check ranges
    sorted_results = sorted(node_results, key=lambda r: r['freq_start'])
    
    for i in range(len(sorted_results) - 1):
        curr = sorted_results[i]
        next_result = sorted_results[i + 1]
        
        if curr['freq_end'] != next_result['freq_start']:
            raise ValueError(
                f"Frequency range gap detected:\n"
                f"  Node {curr['node_rank']}: frequencies [{curr['freq_start']}:{curr['freq_end']}]\n"
                f"  Node {next_result['node_rank']}: frequencies [{next_result['freq_start']}:{next_result['freq_end']}]\n"
                f"  Gap/Overlap: {curr['freq_end']} != {next_result['freq_start']}"
            )
    
    logger.info("✓ Node results validation passed")
    return True


def concatenate_results(node_results):
    """
    Concatenate covariance results from multiple nodes.
    
    Returns:
        Dictionary with concatenated results
    """
    # Sort by frequency start index to ensure correct order
    sorted_results = sorted(node_results, key=lambda r: r['freq_start'])
    
    # Extract and concatenate covariance columns
    covariance_cols_list = [r['covariance_cols']['est_mask'] for r in sorted_results]
    concatenated_covariance = np.concatenate(covariance_cols_list, axis=-1)
    
    # Concatenate frequencies
    picked_frequencies_list = [r['picked_frequencies'] for r in sorted_results]
    concatenated_frequencies = np.concatenate(picked_frequencies_list)
    
    # Concatenate FSCs
    column_fscs_list = [r['column_fscs'] for r in sorted_results]
    concatenated_fscs = np.concatenate(column_fscs_list, axis=0)
    
    # Build global frequency indices
    global_indices_list = [r['picked_frequencies_global_indices'] for r in sorted_results]
    concatenated_global_indices = np.concatenate(global_indices_list)
    
    # Log statistics
    total_freqs = concatenated_frequencies.size
    logger.info(f"Concatenated {len(node_results)} node results:")
    for r in sorted_results:
        n_freqs = r['picked_frequencies'].size
        logger.info(
            f"  Node {r['node_rank']}: frequencies [{r['freq_start']}:{r['freq_end']}] "
            f"({n_freqs} freqs)"
        )
    logger.info(f"Total frequencies: {total_freqs}")
    logger.info(f"Covariance shape: {concatenated_covariance.shape}")
    
    return {
        'covariance_cols': {'est_mask': concatenated_covariance},
        'picked_frequencies': concatenated_frequencies,
        'picked_frequencies_global_indices': concatenated_global_indices,
        'column_fscs': concatenated_fscs,
        'n_nodes': node_results[0]['n_nodes'],
        'node_results': [
            {
                'node_rank': r['node_rank'],
                'freq_start': r['freq_start'],
                'freq_end': r['freq_end'],
                'n_freqs': r['picked_frequencies'].size
            }
            for r in sorted_results
        ]
    }


def save_concatenated_result(result, output_file):
    """Save concatenated result to file."""
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    logger.info(f"Saved concatenated result to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multi-node covariance results"
    )
    
    parser.add_argument(
        "covariance_dir",
        type=str,
        help="Directory containing per-node covariance files (covariance_nodeXXX.pkl)"
    )
    
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=None,
        help="Expected number of nodes (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: <covariance_dir>/covariance_complete.pkl)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate node results without concatenating"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("=" * 70)
    logger.info("CONCATENATE MULTI-NODE COVARIANCE RESULTS")
    logger.info("=" * 70)
    logger.info(f"Covariance directory: {args.covariance_dir}")
    
    # Check directory exists
    if not os.path.isdir(args.covariance_dir):
        logger.error(f"Directory not found: {args.covariance_dir}")
        sys.exit(1)
    
    # Find covariance files
    logger.info("\nSearching for covariance files...")
    node_files = find_covariance_files(args.covariance_dir)
    
    if not node_files:
        logger.error(f"No covariance_nodeXXX.pkl files found in {args.covariance_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(node_files)} node result files:")
    for node_rank, filepath in node_files:
        logger.info(f"  Node {node_rank}: {os.path.basename(filepath)}")
    
    # Check expected number of nodes
    if args.n_nodes is not None and len(node_files) != args.n_nodes:
        logger.error(
            f"Expected {args.n_nodes} nodes, but found {len(node_files)} files"
        )
        sys.exit(1)
    
    # Load all node results
    logger.info("\nLoading node results...")
    node_results = []
    for node_rank, filepath in node_files:
        try:
            result = load_node_result(filepath)
            node_results.append(result)
            logger.info(f"✓ Loaded node {node_rank}: {result['picked_frequencies'].size} frequencies")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            sys.exit(1)
    
    # Validate results
    logger.info("\nValidating node results...")
    try:
        validate_node_results(node_results)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION COMPLETE - No errors found")
        logger.info("=" * 70)
        return
    
    # Concatenate results
    logger.info("\nConcatenating results...")
    concatenated = concatenate_results(node_results)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.covariance_dir, "covariance_complete.pkl")
    
    # Save result
    logger.info("\nSaving concatenated result...")
    save_concatenated_result(concatenated, output_file)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CONCATENATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total nodes: {concatenated['n_nodes']}")
    logger.info(f"Total frequencies: {concatenated['picked_frequencies'].size}")
    logger.info(f"Covariance shape: {concatenated['covariance_cols']['est_mask'].shape}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Continue pipeline: python -m recovar.commands.continue_from_covariance \\")
    logger.info(f"       --covariance-dir {args.covariance_dir} \\")
    logger.info("       --output-dir <output_directory>")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
