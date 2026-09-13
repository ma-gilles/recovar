"""Materialized NumPy significance masks for lazy K-class mask regression tests.

This reference expands the complete per-image fine grid. Production uses lazy
block masks to avoid that allocation. Keep their mask-building algorithms
separate; the shared complement container only describes input indices.
"""

import numpy as np

from recovar.em.scoring.significant_samples import ComplementSignificantSampleIndices


def _build_fine_grid_significance_mask(
    significant_sample_indices_per_class,
    n_rot_coarse: int,
    n_trans_coarse: int,
    n_rot_fine: int,
    n_trans_fine: int,
    rot_oversampling_factor: int,
    trans_oversampling_factor: int,
    rot_parent_map: np.ndarray,
    trans_parent_map: np.ndarray,
    n_images: int,
) -> np.ndarray:
    """Expand pass-1 coarse significance to a per-image fine-grid mask.

    For each image, ``significant_sample_indices_per_class[i]`` holds the
    flat coarse pose indices ``r_coarse * n_trans_coarse + t_coarse`` that
    survived adaptive_fraction pruning at the coarse grid (or ``None``
    when every coarse pose was significant).

    Each coarse pose ``(r_coarse, t_coarse)`` expands to
    ``rot_oversampling_factor * trans_oversampling_factor`` fine poses,
    where the parent of fine rotation ``r_fine`` is
    ``rot_parent_map[r_fine]`` and likewise for translations.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_images, n_rot_fine, n_trans_fine)
        True at fine pose positions whose coarse parent was significant.

    Mirrors RELION's pass-2 significance mask in
    ``ml_optimiser.cpp::expectationOneParticle`` (line 5022 onward), where
    only oversampled children of pass-1 significant coarse samples are
    evaluated in pass-2.
    """
    if rot_parent_map.shape != (n_rot_fine,):
        raise ValueError(
            f"rot_parent_map must have shape ({n_rot_fine},), got {rot_parent_map.shape}",
        )
    if trans_parent_map.shape != (n_trans_fine,):
        raise ValueError(
            f"trans_parent_map must have shape ({n_trans_fine},), got {trans_parent_map.shape}",
        )

    mask = np.zeros((n_images, n_rot_fine, n_trans_fine), dtype=bool)
    for image_index in range(n_images):
        sig = significant_sample_indices_per_class[image_index]
        if sig is None:
            mask[image_index] = True
            continue
        if isinstance(sig, ComplementSignificantSampleIndices):
            excluded = np.asarray(sig.excluded_indices, dtype=np.int64).reshape(-1)
            if excluded.size == 0:
                mask[image_index] = True
                continue
            coarse_rot_idx = excluded // n_trans_coarse
            coarse_trans_idx = excluded % n_trans_coarse
            coarse_pair = np.ones((n_rot_coarse, n_trans_coarse), dtype=bool)
            coarse_pair[coarse_rot_idx, coarse_trans_idx] = False
            mask[image_index] = coarse_pair[rot_parent_map][:, trans_parent_map]
            continue
        sig = np.asarray(sig, dtype=np.int64)
        if sig.size == 0:
            continue
        coarse_rot_idx = sig // n_trans_coarse
        coarse_trans_idx = sig % n_trans_coarse
        coarse_pair = np.zeros((n_rot_coarse, n_trans_coarse), dtype=bool)
        coarse_pair[coarse_rot_idx, coarse_trans_idx] = True
        # Broadcast parent significance to the fine grid.
        mask[image_index] = coarse_pair[rot_parent_map][:, trans_parent_map]
    return mask
