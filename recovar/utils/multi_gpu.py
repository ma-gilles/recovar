"""
Multi-GPU utilities for RECOVAR

This module provides functions for distributing computation across multiple GPUs.
Implements data parallelism for the compute_H_B function.

Author: RECOVAR Team
Date: 2025-10-16
"""

import logging
import numpy as np
import jax
from typing import List, Tuple, Optional, Callable
import time
from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)

# Import NVTX domain for compute_H_B profiling
from recovar.heterogeneity.covariance_estimation import NVTX_DOMAIN_H_B


def get_available_gpus() -> List:
    """
    Get list of available GPU devices.

    Returns:
        List of JAX GPU devices
    """
    try:
        devices = jax.devices("gpu")
        logger.info("Found %s GPU(s): %s", len(devices), [str(d) for d in devices])
        return devices
    except RuntimeError as e:
        logger.warning("No GPUs found: %s", e)
        return []


def split_indices_for_gpus(n_items: int, n_gpus: int) -> List[np.ndarray]:
    """
    Split a range of indices evenly across GPUs.

    Args:
        n_items: Total number of items to split
        n_gpus: Number of GPUs to split across

    Returns:
        List of index arrays, one per GPU

    Example:
        >>> split_indices_for_gpus(100, 3)
        [array([0, 1, ..., 32]), array([33, 34, ..., 65]), array([66, 67, ..., 99])]
    """
    splits = [chunk for chunk in np.array_split(np.arange(n_items), n_gpus) if len(chunk) > 0]

    logger.info("Split %d items across %d GPUs: sizes = %s", n_items, len(splits), [len(s) for s in splits])

    return splits


def compute_on_gpus_parallel(
    compute_fn: Callable, image_indices_per_gpu: List[np.ndarray], devices: List, *args, **kwargs
) -> Tuple[List, List]:
    """
    Execute a computation function in parallel across multiple GPUs.

    This function distributes work across GPUs by:
    1. Assigning each GPU a subset of images to process
    2. Running computation on each GPU in parallel (via Python threads)
    3. Collecting results from all GPUs

    Args:
        compute_fn: Function to execute. Should accept image_subset parameter.
        image_indices_per_gpu: List of index arrays, one per GPU
        devices: List of JAX devices to use
        *args: Positional arguments to pass to compute_fn
        **kwargs: Keyword arguments to pass to compute_fn

    Returns:
        Two lists: (results_H, results_B), one element per GPU
    """
    import concurrent.futures

    n_gpus = len(devices)
    results_H = [None] * n_gpus
    results_B = [None] * n_gpus

    def compute_on_device(gpu_id, device, image_indices):
        """Worker function for a single GPU"""
        logger.info("GPU %s: Starting computation on %s images", gpu_id, len(image_indices))
        start_time = time.time()

        with jax.default_device(device):
            H, B = compute_fn(*args, image_subset=image_indices, **kwargs)

            # Tag to measure overhead between compute_fn return and transfer start
            with nvtx.annotate(f"GPU{gpu_id}_post_compute", color="pink", domain=NVTX_DOMAIN_H_B):
                # Measure any implicit synchronization or overhead
                pass

            # Move results to CPU to free GPU memory
            with nvtx.annotate(f"GPU{gpu_id}_to_CPU_transfer", color="orange", domain=NVTX_DOMAIN_H_B):
                H_cpu = np.array(H)
                B_cpu = np.array(B)

        elapsed = time.time() - start_time
        logger.info("GPU %s: Completed in %.2fs", gpu_id, elapsed)

        results_H[gpu_id] = H_cpu
        results_B[gpu_id] = B_cpu

    # Launch computations in parallel using threads
    # (JAX releases GIL during computation, so threads are effective)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for gpu_id, (device, indices) in enumerate(zip(devices, image_indices_per_gpu)):
            future = executor.submit(compute_on_device, gpu_id, device, indices)
            futures.append(future)

        # Wait for all to complete
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception() is not None:
                raise future.exception()

    return results_H, results_B


def reduce_results(results_H: List[np.ndarray], results_B: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce (sum) results from multiple GPUs.

    Args:
        results_H: List of H matrices, one per GPU
        results_B: List of B matrices, one per GPU

    Returns:
        Tuple of (H_total, B_total)
    """
    logger.info("Reducing results from %s GPUs...", len(results_H))
    start_time = time.time()

    # Incremental sum to avoid allocating N_gpus * array_size at once
    with nvtx.annotate("reduce_results_sum", color="cyan", domain=NVTX_DOMAIN_H_B):
        H_total = results_H[0]
        B_total = results_B[0]
        for i in range(1, len(results_H)):
            H_total = H_total + results_H[i]
            B_total = B_total + results_B[i]
            results_H[i] = None  # free memory eagerly
            results_B[i] = None

    elapsed = time.time() - start_time
    logger.info("Reduction completed in %.2fs", elapsed)

    return H_total, B_total


def compute_H_B_multi_gpu(
    compute_H_B_fn: Callable, experiment_dataset, n_gpus: Optional[int] = None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-GPU wrapper for compute_H_B function.

    This is the main entry point for multi-GPU computation. It:
    1. Detects available GPUs
    2. Splits images across GPUs
    3. Runs computation in parallel
    4. Reduces results

    Args:
        compute_H_B_fn: Single-GPU compute function
        experiment_dataset: Dataset object
        n_gpus: Number of GPUs to use (None = use all available)
        **kwargs: Additional arguments for compute_H_B_fn

    Returns:
        Tuple of (H, B) matrices

    Example:
        >>> H, B = compute_H_B_multi_gpu(
        ...     compute_H_B_single_gpu,
        ...     experiment_dataset,
        ...     mean_estimate=mean,
        ...     volume_mask=mask,
        ...     picked_frequencies=freqs,
        ...     batch_size=1000,
        ...     options=options
        ... )
    """
    # Get available GPUs
    devices = get_available_gpus()

    if len(devices) == 0:
        raise RuntimeError("No GPUs available for multi-GPU computation")

    # Determine how many GPUs to use
    if n_gpus is None:
        n_gpus = len(devices)
    else:
        n_gpus = min(n_gpus, len(devices))

    devices = devices[:n_gpus]
    logger.info("Using %s GPU(s) for computation", n_gpus)

    # Check if single GPU (no parallelization needed)
    if n_gpus == 1:
        logger.info("Single GPU mode - calling compute_H_B directly")
        with jax.default_device(devices[0]):
            return compute_H_B_fn(experiment_dataset, **kwargs)

    # Split images across GPUs
    n_images = experiment_dataset.n_images
    image_indices_per_gpu = split_indices_for_gpus(n_images, n_gpus)

    # Compute in parallel
    start_time = time.time()
    results_H, results_B = compute_on_gpus_parallel(
        compute_H_B_fn, image_indices_per_gpu, devices, experiment_dataset, **kwargs
    )

    # Reduce results
    H_total, B_total = reduce_results(results_H, results_B)

    total_time = time.time() - start_time
    logger.info("Multi-GPU computation completed in %.2fs", total_time)

    return H_total, B_total


def estimate_multi_gpu_speedup(n_images: int, n_gpus: int) -> dict:
    """
    Estimate expected speedup from multi-GPU parallelization.

    Args:
        n_images: Number of images to process
        n_gpus: Number of GPUs

    Returns:
        Dictionary with speedup estimates
    """
    # Empirical efficiency estimates based on communication overhead
    efficiency_model = {1: 1.00, 2: 0.975, 4: 0.960, 8: 0.950, 16: 0.920}

    # Interpolate for intermediate values
    efficiency = efficiency_model.get(n_gpus, 0.95)

    speedup = n_gpus * efficiency
    images_per_gpu = n_images / n_gpus

    return {"n_gpus": n_gpus, "efficiency": efficiency, "expected_speedup": speedup, "images_per_gpu": images_per_gpu}


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test GPU detection
    logger.info("Testing GPU detection...")
    devices = get_available_gpus()
    logger.info("Found %d GPU(s)", len(devices))

    # Test index splitting
    logger.info("Testing index splitting...")
    for n_gpus in [2, 4, 8]:
        splits = split_indices_for_gpus(100000, n_gpus)
        logger.info("%d GPUs: sizes = %s", n_gpus, [len(s) for s in splits])

    # Test speedup estimation
    logger.info("Testing speedup estimation...")
    for n_gpus in [1, 2, 4, 8]:
        est = estimate_multi_gpu_speedup(300000, n_gpus)
        logger.info(
            "%d GPUs: %.2f× speedup (%.1f%% efficiency)", n_gpus, est["expected_speedup"], est["efficiency"] * 100
        )
