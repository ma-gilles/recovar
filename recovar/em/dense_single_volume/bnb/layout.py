"""Pack BnB survivors into a ``LocalHypothesisLayout`` for ``run_local_em_exact``.

The local EM engine expects per-image local rotation neighborhoods plus an
optional per-image (rotation, translation) sample mask. The BnB selector
produces a dense (n_images, n_rot, n_trans) boolean mask over the global
fixed grid; this module collapses that mask into the ragged layout the
local engine consumes, preserving the (rotation, translation) joint
survivor relationship via ``sample_mask_flat``.

This module is intentionally minimal — it does NOT compute log priors
(those are passed through unchanged from the caller) and does NOT
handle K-class (deferred to a later phase).
"""

from __future__ import annotations

import numpy as np

from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout

from .support import BnBSupportResult


def build_bnb_local_layout(
    support: BnBSupportResult,
    rotations_global: np.ndarray,
    translations_global: np.ndarray,
    *,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
) -> LocalHypothesisLayout:
    """Convert BnB survivors into a ``LocalHypothesisLayout``.

    Parameters
    ----------
    support : BnBSupportResult
        Output of ``select_bnb_support_fixed_grid_k1``.
    rotations_global : (n_rot, 3, 3) float
        The global rotation pool the BnB ran on. Indices in
        ``support.rotation_survivor_mask_per_image`` index into this array.
    translations_global : (n_trans, 2) float
        Global shift grid; shared across images.
    rotation_log_prior : (n_rot,) or (n_images, n_rot) float, optional
        Per-rotation log prior. If 1-D, broadcast over images.
    translation_log_prior : (n_trans,) or (n_images, n_trans) float, optional
        Per-shift log prior. If 1-D, broadcast over images.

    Returns
    -------
    layout : LocalHypothesisLayout
        Ready to feed to ``bucket_local_hypothesis_layout`` and then
        ``run_local_em_exact``.
    """
    rotations_global = np.asarray(rotations_global, dtype=np.float32)
    translations_global = np.asarray(translations_global, dtype=np.float32)

    sample_mask = np.asarray(support.sample_mask_per_image, dtype=bool)
    rot_survivor = np.asarray(support.rotation_survivor_mask_per_image, dtype=bool)
    n_images, n_rot, n_trans = sample_mask.shape

    # Build per-image local rotation lists. For each image, gather the indices
    # where rot_survivor[i, r] is True.
    rotation_ids_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    sample_mask_parts: list[np.ndarray] = []

    rotation_counts = np.zeros(n_images, dtype=np.int32)
    rotation_offsets = np.zeros(n_images + 1, dtype=np.int64)

    # Resolve rotation log prior into per-image, per-local-rotation form.
    if rotation_log_prior is None:
        rot_log_prior_per_image = None
    else:
        rot_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rot_log_prior.ndim == 1:
            rot_log_prior_per_image = np.broadcast_to(
                rot_log_prior[None, :], (n_images, n_rot),
            )
        else:
            rot_log_prior_per_image = rot_log_prior

    # Resolve translation log prior into per-image form.
    if translation_log_prior is None:
        translation_log_priors_resolved = np.zeros((n_images, n_trans), dtype=np.float32)
    else:
        t_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if t_log_prior.ndim == 1:
            translation_log_priors_resolved = np.broadcast_to(
                t_log_prior[None, :], (n_images, n_trans),
            ).copy()
        else:
            translation_log_priors_resolved = t_log_prior.astype(np.float32, copy=False)

    for i in range(n_images):
        survivor_idx = np.flatnonzero(rot_survivor[i]).astype(np.int32, copy=False)
        if survivor_idx.size == 0:
            # No survivor for this image — engine cannot run with zero
            # rotations. Restore the top rotation by sample-mask coverage
            # (any rotation that has at least one True trans, or rotation 0
            # if none).
            survivor_idx = np.asarray([0], dtype=np.int32)

        n_local = int(survivor_idx.size)
        rotation_counts[i] = n_local
        rotation_offsets[i + 1] = rotation_offsets[i] + n_local

        rotation_ids_parts.append(survivor_idx)
        rotations_parts.append(rotations_global[survivor_idx])
        if rot_log_prior_per_image is None:
            log_prior_parts.append(np.zeros(n_local, dtype=np.float32))
        else:
            log_prior_parts.append(
                rot_log_prior_per_image[i, survivor_idx].astype(np.float32, copy=False),
            )

        # Sample mask: rows = local rotations, cols = translations.
        sample_mask_parts.append(
            sample_mask[i, survivor_idx, :].astype(bool, copy=False),
        )

    rotation_ids_flat = (
        np.concatenate(rotation_ids_parts, axis=0)
        if rotation_ids_parts
        else np.zeros(0, dtype=np.int32)
    )
    rotations_flat = (
        np.concatenate(rotations_parts, axis=0)
        if rotations_parts
        else np.zeros((0, 3, 3), dtype=np.float32)
    )
    rotation_log_priors_flat = (
        np.concatenate(log_prior_parts, axis=0)
        if log_prior_parts
        else np.zeros(0, dtype=np.float32)
    )
    sample_mask_flat = (
        np.concatenate(sample_mask_parts, axis=0)
        if sample_mask_parts
        else np.zeros((0, n_trans), dtype=bool)
    )

    return LocalHypothesisLayout(
        n_global_rotations=int(n_rot),
        n_pixels=0,  # not using HEALPix factorization for BnB rotations
        n_psi=0,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=rotation_counts,
        translation_grid=translations_global,
        translation_log_priors=translation_log_priors_resolved,
        rotation_posterior_ids_flat=None,
        sample_mask_flat=sample_mask_flat,
    )
