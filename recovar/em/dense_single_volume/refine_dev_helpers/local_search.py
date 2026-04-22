"""Local angular search grid partitioning and padding helpers.

These functions handle the mechanics of grouping images by their prior
orientations and building GPU-friendly rotation grids for exact local
angular search.  They are called exclusively by ``_run_grouped_local_search_em``
in ``refine.py``.
"""

import healpy as hp
import numpy as np

from recovar import utils
from recovar.em.sampling import (
    get_local_rotation_grid_fast,
    rotation_grid_n_in_planes,
)


def _local_search_chunk_size(image_batch_size: int) -> int:
    """Return the seed group size for exact local-search packing.

    We start from moderately sized orientation-sorted image groups and split
    them recursively until the exact union of local rotations stays within
    the hard cap enforced by `_local_search_max_union_rotations`.
    """
    return int(max(1, min(int(image_batch_size), 64)))


def _local_search_rotation_block_size(local_rotation_count: int, rotation_block_size: int) -> int:
    """Bucket local-search rotation blocks to reduce JIT shape churn.

    For small exact local neighborhoods, padding up to the next power of two
    is much cheaper than recompiling for every distinct candidate count. For
    larger supports we fall back to the caller's cap so the engine still
    tiles large neighborhoods across multiple blocks.
    """
    if local_rotation_count <= 0:
        return 1
    if local_rotation_count >= rotation_block_size:
        return int(rotation_block_size)
    bucket = 1 << max(int(local_rotation_count - 1).bit_length(), 4)
    return int(min(bucket, rotation_block_size))


def _local_search_engine_rotation_block_size(rotation_block_size: int) -> int:
    """Cap the exact local-search engine block size.

    Local search already reduces the candidate set per image from the full
    HEALPix grid down to a few thousand rotations. Reusing the dense-search
    5k rotation tile size here creates oversized XLA kernels whose compile
    time dominates the first local-search iteration. A 1k cap keeps the
    candidate set exact while making the compiled score kernels much smaller.
    """
    return int(max(64, min(int(rotation_block_size), 1024)))


def _local_search_max_union_rotations(rotation_block_size: int) -> int:
    """Return the largest exact local-search union allowed in one engine call."""
    return int(4 * _local_search_engine_rotation_block_size(rotation_block_size))


def _prior_rotations_to_relion_eulers(prior_rotations: np.ndarray) -> np.ndarray:
    """Convert prior rotations to RELION Euler angles."""
    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        return np.asarray(prior_rotations, dtype=np.float32).reshape(-1, 3)
    if prior_rotations.ndim == 3 and prior_rotations.shape[1:] == (3, 3):
        return utils.R_to_relion(np.asarray(prior_rotations, dtype=np.float32), degrees=True).astype(np.float32)
    raise ValueError(f"prior_rotations must have shape (n,3) or (n,3,3), got {prior_rotations.shape}")


def _local_search_sort_order(prior_rotations: np.ndarray, healpix_order: int) -> np.ndarray:
    """Sort images by coarse viewing direction and psi to improve local support overlap."""
    prior_eulers = _prior_rotations_to_relion_eulers(prior_rotations)
    prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True).astype(np.float32)
    prior_dir_vecs = np.asarray(prior_rotation_mats[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm
    dir_pixels = hp.vec2pix(
        2 ** int(healpix_order),
        prior_dir_vecs[:, 0],
        prior_dir_vecs[:, 1],
        prior_dir_vecs[:, 2],
    )
    n_psi = rotation_grid_n_in_planes(healpix_order)
    psi_step = 360.0 / float(n_psi)
    psi_bins = np.floor(np.mod(prior_eulers[:, 2], 360.0) / psi_step + 0.5).astype(np.int64) % n_psi
    return np.lexsort((psi_bins, dir_pixels)).astype(np.int64)


def _partition_local_search_groups(
    prior_rotations: np.ndarray,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    image_batch_size: int,
    rotation_block_size: int,
    grid_metadata: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build exact local-search groups whose rotation unions stay below a hard cap."""
    n_images = int(np.asarray(prior_rotations).shape[0])
    if n_images == 0:
        return []

    seed_group_size = _local_search_chunk_size(image_batch_size)
    max_union_rotations = _local_search_max_union_rotations(rotation_block_size)
    processing_order = _local_search_sort_order(prior_rotations, healpix_order)
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def _split_group(group_image_indices: np.ndarray) -> None:
        local_indices, local_log_prior = get_local_rotation_grid_fast(
            np.asarray(prior_rotations)[group_image_indices],
            sigma_rot,
            sigma_psi,
            healpix_order,
            sigma_cutoff=3.0,
            per_image=True,
            grid_metadata=grid_metadata,
        )
        if group_image_indices.shape[0] <= 1 or int(local_indices.shape[0]) <= max_union_rotations:
            groups.append(
                (
                    np.asarray(group_image_indices, dtype=np.int64),
                    np.asarray(local_indices, dtype=np.int64),
                    np.asarray(local_log_prior, dtype=np.float32),
                )
            )
            return
        mid = max(1, group_image_indices.shape[0] // 2)
        _split_group(group_image_indices[:mid])
        _split_group(group_image_indices[mid:])

    for start in range(0, n_images, seed_group_size):
        _split_group(processing_order[start : start + seed_group_size])

    return groups


def _pad_local_search_rotations(
    local_rotations: np.ndarray,
    local_log_prior: np.ndarray,
    rotation_block_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pad one exact local neighborhood to a compile-friendly bucket.

    The padded rotations are masked out with a ``-1e30`` log-prior so the
    posterior over the real candidates is unchanged. This lets the local
    path reuse a small number of compiled ``run_em_v2`` shapes instead of
    recompiling for every distinct neighborhood size.
    """
    local_rotations = np.asarray(local_rotations, dtype=np.float32).reshape(-1, 3, 3)
    local_log_prior = np.asarray(local_log_prior, dtype=np.float32)
    actual_count = int(local_rotations.shape[0])
    block_size = _local_search_rotation_block_size(actual_count, int(rotation_block_size))
    if block_size <= actual_count:
        return local_rotations, local_log_prior, actual_count, block_size

    pad_count = block_size - actual_count
    padded_rotations = np.concatenate(
        [
            local_rotations,
            np.broadcast_to(np.eye(3, dtype=np.float32), (pad_count, 3, 3)).copy(),
        ],
        axis=0,
    )
    pad_shape = local_log_prior.shape[:-1] + (pad_count,)
    padded_log_prior = np.concatenate(
        [
            local_log_prior,
            np.full(pad_shape, -1e30, dtype=np.float32),
        ],
        axis=-1,
    )
    return padded_rotations, padded_log_prior, actual_count, block_size
