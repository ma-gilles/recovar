#!/usr/bin/env python3
"""
Minimal prototype for frequency-parallel covariance computation.

This prototype validates the core frequency-splitting approach before
full integration into the RECOVAR pipeline.

Usage:
    # Run baseline (1 node, all frequencies)
    python prototype_frequency_parallel.py --node-rank 0 --n-nodes 1

    # Run node 0 (first half of frequencies)
    python prototype_frequency_parallel.py --node-rank 0 --n-nodes 2

    # Run node 1 (second half of frequencies, in parallel)
    python prototype_frequency_parallel.py --node-rank 1 --n-nodes 2

    # After both complete, concatenate:
    python prototype_frequency_parallel.py --concatenate --n-nodes 2
"""

import argparse
import numpy as np
import pickle
import time
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def split_frequencies(picked_frequencies, node_rank, n_nodes):
    """
    Split frequencies across nodes.
    
    This is the CORE LOGIC that will be integrated into covariance_estimation.py.
    
    Args:
        picked_frequencies: Array of frequency indices to compute
        node_rank: This node's rank (0-indexed)
        n_nodes: Total number of nodes
        
    Returns:
        node_freqs: Frequency subset for this node
        freq_start: Starting index in full array
        freq_end: Ending index in full array
    """
    total_freqs = len(picked_frequencies)
    freqs_per_node = total_freqs // n_nodes
    
    freq_start = node_rank * freqs_per_node
    if node_rank == n_nodes - 1:
        freq_end = total_freqs  # Last node gets remainder
    else:
        freq_end = freq_start + freqs_per_node
    
    node_freqs = picked_frequencies[freq_start:freq_end]
    
    logger.info("=" * 70)
    logger.info(f"FREQUENCY SPLIT - Node {node_rank}/{n_nodes}")
    logger.info("=" * 70)
    logger.info(f"Total frequencies: {total_freqs}")
    logger.info(f"This node: {len(node_freqs)} frequencies ({len(node_freqs)/total_freqs*100:.1f}%)")
    logger.info(f"Index range: [{freq_start}:{freq_end}]")
    logger.info("=" * 70)
    
    return node_freqs, freq_start, freq_end


def load_dataset(args):
    """
    Load dataset using existing RECOVAR infrastructure.
    """
    logger.info("Loading dataset...")
    
    dataset_dir = Path(args.dataset_dir)
    particles_file = dataset_dir / f"particles.{args.image_size}.mrcs"
    poses_file = dataset_dir / "poses.pkl"
    ctf_file = dataset_dir / "ctf.pkl"
    
    logger.info(f"  Particles: {particles_file}")
    logger.info(f"  Poses: {poses_file}")
    logger.info(f"  CTF: {ctf_file}")
    logger.info(f"  N images: {args.n_images}")
    
    # Import RECOVAR modules
    from recovar import dataset as recovar_dataset
    
    # Create a simple args-like object for load_dataset_from_args
    class DatasetArgs:
        def __init__(self, particles, poses, ctf, n_images):
            self.particles = str(particles)
            self.poses = str(poses)
            self.ctf = str(ctf)
            self.n_images = n_images
            self.lazy = True
            self.ind_split = None
            self.halfsets = None  # Let RECOVAR auto-detect halfsets
            self.datadir = None
            self.tilt_series = False
            self.tilt_series_ctf = 'cryoem'
            self.ind = None
            self.tilt_ind = None
            self.ntilts = None
            self.strip_prefix = None
            # Additional attributes from make_dataset_loader_dict
            self.padding = 0
            self.correct_contrast = False
            self.contrast_params = None
            self.keep_in_memory = False
            self.noise_model = None
            self.noise_level = None
            self.uninvert_data = "automatic"
            self.premultiplied_ctf = False
            self.angle_per_tilt = None
            self.dose_per_tilt = None
    
    dataset_args = DatasetArgs(particles_file, poses_file, ctf_file, args.n_images)
    
    # Use existing RECOVAR dataset loading function
    cryos = recovar_dataset.load_dataset_from_args(dataset_args, lazy=True)
    
    logger.info(f"✓ Dataset loaded: {cryos[0].n_images} images, grid size {cryos[0].grid_size}")
    
    return cryos


def get_picked_frequencies(volume_shape, radius=0.95):
    """
    Get list of frequency indices to compute.
    
    Uses existing RECOVAR frequency selection logic.
    """
    from recovar import covariance_core
    
    picked_frequencies = np.array(
        covariance_core.get_picked_frequencies(
            volume_shape, 
            radius=radius, 
            use_half=True
        )
    )
    
    logger.info(f"Picked {len(picked_frequencies)} frequencies (radius={radius})")
    
    return picked_frequencies


def compute_covariance_subset(cryos, frequencies, args):
    """
    Compute covariance for a subset of frequencies.
    
    This calls existing RECOVAR covariance computation functions,
    but limits computation to the specified frequency subset.
    
    This is the KEY INTEGRATION POINT with existing code.
    """
    logger.info(f"Computing covariance for {len(frequencies)} frequencies...")
    
    # For prototype, we'll call the existing covariance computation
    # but with a limited frequency set
    
    # Import necessary modules
    from recovar import covariance_estimation, homogeneous, noise
    from recovar.fourier_transform_utils import fourier_transform_utils
    import jax.numpy as jnp
    
    ftu = fourier_transform_utils(jnp)
    
    # Get basic parameters
    cryo = cryos[0]
    volume_shape = cryo.volume_shape
    volume_mask = np.ones(volume_shape, dtype=bool)
    
    # Compute mean (simplified - in real pipeline this comes from earlier stage)
    logger.info("Computing mean volume...")
    mean_estimate = homogeneous.get_mean(
        cryos,
        None,  # mean_prior
        volume_mask,
        None,  # valid_idx (use all)
        gpu_memory=args.gpu_memory,
        disc_type='linear_interp'
    )
    
    means = {
        'combined': mean_estimate['combined'],
        'est_mask': mean_estimate['est_mask'],
        'lhs': mean_estimate.get('lhs', None)
    }
    
    # Estimate noise (simplified)
    logger.info("Estimating noise...")
    noise_var_used = noise.make_radial_noise(cryo.image_shape, 1.0)
    variance_estimate = {'combined': noise_var_used}
    
    # Now compute covariance for our frequency subset
    logger.info(f"Computing covariance columns for {len(frequencies)} frequencies...")
    
    from recovar.covariance_estimation import compute_regularized_covariance_columns
    
    # Get default options
    from recovar.covariance_estimation import get_default_covariance_computation_options
    options = get_default_covariance_computation_options()
    
    # Compute covariance
    covariance_cols, _, fscs = compute_regularized_covariance_columns(
        cryos=cryos,
        means=means,
        mean_prior=None,
        volume_mask=volume_mask,
        dilated_volume_mask=volume_mask,
        valid_idx=None,
        gpu_memory=args.gpu_memory,
        options=options,
        picked_frequencies=frequencies,  # ← Our subset!
        use_multi_gpu=False,
        n_gpus=1
    )
    
    logger.info(f"✓ Covariance computed: shape {covariance_cols['est_mask'].shape}")
    
    return covariance_cols['est_mask'], fscs


def run_node_computation(args):
    """
    Run covariance computation for this node's frequency subset.
    """
    logger.info("=" * 70)
    logger.info(f"NODE {args.node_rank} COMPUTATION START")
    logger.info("=" * 70)
    
    start_time_total = time.time()
    
    # Load dataset
    cryos = load_dataset(args)
    
    # Get full frequency list
    volume_shape = cryos[0].volume_shape
    picked_frequencies = get_picked_frequencies(volume_shape, radius=args.freq_radius)
    
    # Split frequencies for this node
    node_freqs, freq_start, freq_end = split_frequencies(
        picked_frequencies, 
        args.node_rank, 
        args.n_nodes
    )
    
    # Compute covariance for this node's frequencies
    start_time_compute = time.time()
    
    covariance_cols, fscs = compute_covariance_subset(
        cryos,
        node_freqs,
        args
    )
    
    compute_time = time.time() - start_time_compute
    logger.info(f"✓ Computation time: {compute_time:.2f} seconds")
    
    # Save node result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"node{args.node_rank:03d}_result.npz"
    
    logger.info(f"Saving result to {result_file}")
    np.savez_compressed(
        result_file,
        covariance_cols=covariance_cols,
        picked_frequencies=node_freqs,
        fscs=fscs,
        freq_start=freq_start,
        freq_end=freq_end,
        node_rank=args.node_rank,
        compute_time=compute_time,
        n_nodes=args.n_nodes
    )
    
    total_time = time.time() - start_time_total
    
    logger.info("=" * 70)
    logger.info(f"NODE {args.node_rank} COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Compute time: {compute_time:.2f} seconds")
    logger.info(f"Overhead: {total_time - compute_time:.2f} seconds")
    logger.info("=" * 70)
    
    return result_file


def concatenate_results(args):
    """
    Concatenate results from all nodes and validate.
    """
    logger.info("=" * 70)
    logger.info("CONCATENATING NODE RESULTS")
    logger.info("=" * 70)
    
    output_dir = Path(args.output_dir)
    
    # Load all node results
    logger.info(f"Loading results from {args.n_nodes} nodes...")
    node_results = []
    
    for node_rank in range(args.n_nodes):
        result_file = output_dir / f"node{node_rank:03d}_result.npz"
        
        if not result_file.exists():
            raise FileNotFoundError(
                f"Missing result from node {node_rank}: {result_file}\n"
                f"Make sure all nodes have completed before concatenating."
            )
        
        logger.info(f"  Loading node {node_rank}: {result_file.name}")
        data = np.load(result_file)
        
        node_results.append({
            'covariance_cols': data['covariance_cols'],
            'picked_frequencies': data['picked_frequencies'],
            'fscs': data['fscs'],
            'freq_start': int(data['freq_start']),
            'freq_end': int(data['freq_end']),
            'node_rank': int(data['node_rank']),
            'compute_time': float(data['compute_time'])
        })
        
        logger.info(f"    Shape: {data['covariance_cols'].shape}, "
                   f"Frequencies: {len(data['picked_frequencies'])}, "
                   f"Time: {data['compute_time']:.2f}s")
    
    # Sort by frequency start (should already be sorted, but be safe)
    node_results.sort(key=lambda x: x['freq_start'])
    
    # Validate continuity (no gaps or overlaps)
    logger.info("\nValidating frequency coverage...")
    for i in range(len(node_results) - 1):
        curr_end = node_results[i]['freq_end']
        next_start = node_results[i+1]['freq_start']
        
        if curr_end != next_start:
            raise ValueError(
                f"Frequency range mismatch!\n"
                f"  Node {i} ends at {curr_end}\n"
                f"  Node {i+1} starts at {next_start}\n"
                f"  Gap/overlap of {next_start - curr_end} frequencies"
            )
        
        logger.info(f"  Node {i} → Node {i+1}: continuous [{curr_end}]")
    
    logger.info("  ✓ No gaps or overlaps - coverage is continuous")
    
    # Concatenate arrays
    logger.info("\nConcatenating arrays...")
    
    covariance_complete = np.concatenate(
        [nr['covariance_cols'] for nr in node_results],
        axis=1  # Concatenate along frequency axis (columns)
    )
    
    frequencies_complete = np.concatenate(
        [nr['picked_frequencies'] for nr in node_results]
    )
    
    fscs_complete = np.concatenate(
        [nr['fscs'] for nr in node_results],
        axis=0
    )
    
    logger.info(f"  Complete covariance shape: {covariance_complete.shape}")
    logger.info(f"  Total frequencies: {len(frequencies_complete)}")
    logger.info(f"  FSCs shape: {fscs_complete.shape}")
    
    # Save concatenated result
    concat_file = output_dir / "concatenated_result.npz"
    logger.info(f"\nSaving concatenated result to {concat_file}")
    
    np.savez_compressed(
        concat_file,
        covariance_cols=covariance_complete,
        picked_frequencies=frequencies_complete,
        fscs=fscs_complete,
        n_nodes=args.n_nodes
    )
    
    # Print timing summary
    logger.info("\n" + "=" * 70)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 70)
    
    node_times = [nr['compute_time'] for nr in node_results]
    max_time = max(node_times)
    total_time = sum(node_times)
    
    logger.info("Node compute times:")
    for nr in node_results:
        logger.info(f"  Node {nr['node_rank']}: {nr['compute_time']:.2f}s")
    
    logger.info(f"\nParallel execution:")
    logger.info(f"  Wall-clock time (max): {max_time:.2f}s")
    logger.info(f"  Total compute time: {total_time:.2f}s")
    logger.info(f"  Speedup: {total_time/max_time:.2f}×")
    logger.info(f"  Parallel efficiency: {(total_time/max_time/args.n_nodes)*100:.1f}%")
    
    if args.n_nodes > 1:
        logger.info(f"\nScaling analysis:")
        logger.info(f"  Expected speedup (ideal): {args.n_nodes}×")
        logger.info(f"  Actual speedup: {total_time/max_time:.2f}×")
        logger.info(f"  Efficiency: {(total_time/max_time/args.n_nodes)*100:.1f}%")
    
    logger.info("=" * 70)
    logger.info("CONCATENATION COMPLETE ✓")
    logger.info("=" * 70)
    
    return concat_file


def main():
    parser = argparse.ArgumentParser(
        description="Frequency-parallel covariance prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline (1 node, all frequencies)
  python prototype_frequency_parallel.py --node-rank 0 --n-nodes 1
  
  # Run 2 nodes in parallel (in separate terminals/processes)
  python prototype_frequency_parallel.py --node-rank 0 --n-nodes 2 &
  python prototype_frequency_parallel.py --node-rank 1 --n-nodes 2 &
  wait
  
  # Concatenate results
  python prototype_frequency_parallel.py --concatenate --n-nodes 2
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--node-rank',
        type=int,
        default=None,
        help='Node rank (0-indexed). Required for computation mode.'
    )
    
    parser.add_argument(
        '--n-nodes',
        type=int,
        default=2,
        help='Total number of nodes (default: 2)'
    )
    
    parser.add_argument(
        '--concatenate',
        action='store_true',
        help='Concatenate results from all nodes (post-processing mode)'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/workspace/data-128-100000/test_dataset',
        help='Dataset directory (default: /workspace/data-128-100000/test_dataset)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/workspace/multinode_covariance_bench/prototype_output',
        help='Output directory (default: /workspace/multinode_covariance_bench/prototype_output)'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=128,
        help='Image size (default: 128)'
    )
    
    parser.add_argument(
        '--n-images',
        type=int,
        default=1000,
        help='Number of images to process (default: 1000)'
    )
    
    parser.add_argument(
        '--freq-radius',
        type=float,
        default=0.95,
        help='Frequency radius for picking (default: 0.95)'
    )
    
    parser.add_argument(
        '--gpu-memory',
        type=float,
        default=32,
        help='GPU memory in GB (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.concatenate:
        # Concatenate mode
        concatenate_results(args)
    elif args.node_rank is not None:
        # Node computation mode
        if args.node_rank < 0 or args.node_rank >= args.n_nodes:
            parser.error(f"node-rank must be in range [0, {args.n_nodes-1}]")
        run_node_computation(args)
    else:
        parser.error("Must specify either --node-rank (for computation) or --concatenate (for post-processing)")


if __name__ == '__main__':
    main()
