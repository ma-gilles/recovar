"""Shared translation-prior center helpers for dense EM paths."""

from __future__ import annotations

import numpy as np


def validate_translation_prior_centers(translation_prior_centers, *, n_images: int, n_dims: int):
    """Validate shape and return ``translation_prior_centers`` as an array.

    Preserves the caller's own dtype rather than forcing float32: this is a
    pure validation/reshape step (no new arithmetic), and the input already
    carries whatever precision the caller's prior-center construction chose
    (float64 under double-precision scoring). RELION's own offset-prior
    center is never narrowed (RFLOAT, see ``relion_translation_prior_center``).
    """
    if translation_prior_centers is None:
        return None
    centers = np.asarray(translation_prior_centers)
    if centers.ndim == 1:
        if centers.shape != (int(n_dims),):
            raise ValueError(
                "translation_prior_centers must have shape "
                f"({int(n_dims)},), got {centers.shape}",
            )
    elif centers.ndim == 2:
        if centers.shape != (int(n_images), int(n_dims)):
            raise ValueError(
                "translation_prior_centers must have shape "
                f"({int(n_images)}, {int(n_dims)}) when image-specific, got {centers.shape}",
            )
    else:
        raise ValueError(
            f"translation_prior_centers must be 1D or 2D, got {centers.ndim} dimensions",
        )
    return centers


def translation_prior_centers_for_images(translation_prior_centers, image_indices, *, batch_size: int | None = None):
    """Select/broadcast prior centers per image, preserving the input dtype."""
    if translation_prior_centers is None:
        return None
    centers = np.asarray(translation_prior_centers)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    if centers.ndim == 1:
        rows = int(image_indices.size if batch_size is None else batch_size)
        return np.broadcast_to(centers[None, :], (rows, centers.shape[0]))
    return centers[image_indices]


def translation_sqdist_angstrom(translations, centers, voxel_size: float):
    """Squared Angstrom distance feeding RELION's offset-prior sufficient
    statistic (``wsum_sigma2_offset``). Preserves ``translations``' own
    dtype -- RELION's host offset arithmetic is RFLOAT, never narrowed.
    """
    if centers is None:
        return None
    voxel = float(voxel_size if voxel_size > 0 else 1.0)
    return np.sum(
        ((np.asarray(translations)[None, :, :] - centers[:, None, :]) * voxel) ** 2,
        axis=-1,
        dtype=np.float64,
    )


def expand_fine_translation_prior(
    translation_log_prior_np, fine_translation_parent, *, n_images, n_fine_trans, dtype
):
    """Gather coarse priors into an image-by-fine-translation host array.

    Callers retain the already converted coarse prior and its precision policy.
    Shared (1D) priors broadcast over images; image-specific (2D) rows retain
    their order. Parent IDs may repeat. Preserve NumPy's gather/copy semantics
    and the read-only broadcast view for shared priors of the requested dtype.
    """
    if translation_log_prior_np.ndim == 1:
        fine_tp = translation_log_prior_np[fine_translation_parent]
        fine_translation_prior_2d = np.broadcast_to(fine_tp, (n_images, n_fine_trans)).astype(
            dtype, copy=False
        )
    elif translation_log_prior_np.ndim == 2:
        fine_translation_prior_2d = translation_log_prior_np[:, fine_translation_parent].astype(
            dtype, copy=False
        )
    else:
        raise ValueError(
            f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions",
        )

    return fine_translation_prior_2d
