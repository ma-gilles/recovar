"""Projection cache: precompute P_r mu for all rotations.

At the start of each EM iteration, every candidate rotation is projected
once.  This converts the 3D volume problem into a 2D batched comparison
problem and is the core dense-grid precompute.
"""

import logging

import jax.numpy as jnp
import numpy as np

from recovar import core, utils

logger = logging.getLogger(__name__)


def precompute_projections(
    volume: np.ndarray,
    rotations: np.ndarray,
    image_shape: tuple,
    volume_shape: tuple,
    disc_type: str,
    batch_size: int,
) -> jnp.ndarray:
    """Slice volume at all rotations, returning (n_rot, image_size) complex64.

    Verbatim extraction of E_with_precompute lines 57-64.
    """
    image_size = int(np.prod(image_shape))
    n_rotations = rotations.shape[0]
    projections = np.zeros((n_rotations, image_size), dtype=np.complex64)
    for rot_indices in utils.index_batch_iter(n_rotations, batch_size):
        projections[rot_indices] = core.slice_volume(
            volume, rotations[rot_indices], image_shape, volume_shape, disc_type
        )
    return jnp.asarray(projections)
