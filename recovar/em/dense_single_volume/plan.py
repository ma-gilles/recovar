"""Memory planner for dense single-volume EM.

Consolidates the scattered ad-hoc batch-size multipliers from
e_step.py and m_step.py into a single, testable function.

Each constant is annotated with its origin in the original code.
"""

import logging

from recovar import utils

from .types import DenseEMPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ad-hoc multipliers, documented and centralized.
# These preserve the exact batch sizes of the original code.
# ---------------------------------------------------------------------------
_PROJ_MULT = 5  # e_step.py:55 — slicing is cheap, use larger batches
_DOT_MULT = 10  # e_step.py:71 — dot products use less memory
_NORM_MULT = 3  # e_step.py:162 — norm computation is lighter
_PROB_DIV = 10  # e_step.py:181 — softmax is memory-intensive
_MSTEP_IMG_MULT = 20  # m_step.py:255 — backprojection accumulates into one vol
_MSTEP_ROT_DIV = 5  # m_step.py:263 — rotation block divisor


def plan_em_iteration(
    grid_size: int,
    n_rotations: int,
    n_translations: int,
    memory_to_use_gb: float = 128.0,
    projection_size_gb: float = 0.0,
) -> DenseEMPlan:
    """Compute all batch sizes for one dense EM iteration.

    Parameters match the original scattered calculations exactly so that
    the refactored path produces identical batch partitions.

    Args:
        grid_size: Image grid side length (e.g. 256).
        n_rotations: Number of candidate rotations.
        n_translations: Number of candidate translations.
        memory_to_use_gb: Memory budget for outer image batch (GB).
        projection_size_gb: Size of precomputed projections (GB).
            Used to reduce available memory for later batches.

    Returns:
        DenseEMPlan with all batch sizes.
    """
    gpu_memory = utils.get_gpu_memory_total()
    base_batch = utils.get_image_batch_size(grid_size, gpu_memory)

    # E-step: projection precompute (rotation batching)
    projection_batch = utils.safe_batch_size(base_batch * _PROJ_MULT)

    # E-step: dot products (image batching, accounting for translation inner loop)
    remaining_memory = gpu_memory - projection_size_gb
    dot_base = utils.get_image_batch_size(grid_size, max(remaining_memory, 0.01))
    dot_product_batch = utils.safe_batch_size(dot_base / n_translations * _DOT_MULT)

    # E-step: CTF norms (image batching)
    norm_batch = utils.safe_batch_size(utils.get_image_batch_size(grid_size, max(remaining_memory, 0.01)) * _NORM_MULT)

    # E-step: softmax normalization (image batching)
    prob_batch = utils.safe_batch_size(projection_batch // _PROB_DIV)

    # Outer image batch (from iterations.py:27)
    total_hidden = n_rotations * n_translations
    image_batch = max(1, int(memory_to_use_gb * 1e9 / (total_hidden * 8)))

    # M-step: image batching
    mstep_image_batch = utils.safe_batch_size(
        utils.get_image_batch_size(grid_size, gpu_memory) // n_translations * _MSTEP_IMG_MULT
    )

    # M-step: rotation block size
    mstep_rotation_batch = max(1, n_rotations // _MSTEP_ROT_DIV)

    plan = DenseEMPlan(
        projection_batch=projection_batch,
        dot_product_batch=dot_product_batch,
        norm_batch=norm_batch,
        prob_batch=prob_batch,
        image_batch=image_batch,
        mstep_image_batch=mstep_image_batch,
        mstep_rotation_batch=mstep_rotation_batch,
    )

    logger.info(
        "DenseEMPlan: proj_batch=%d, dot_batch=%d, norm_batch=%d, "
        "prob_batch=%d, img_batch=%d, mstep_img=%d, mstep_rot=%d",
        *plan,
    )

    return plan
