"""Shared translation-prior center helpers for dense EM paths."""

from __future__ import annotations

import numpy as np


def validate_translation_prior_centers(
    translation_prior_centers,
    *,
    n_images: int,
    n_dims: int,
    max_image_index: int | None = None,
):
    if translation_prior_centers is None:
        return None
    centers = np.asarray(translation_prior_centers, dtype=np.float32)
    if centers.ndim == 1:
        if centers.shape != (int(n_dims),):
            raise ValueError(
                "translation_prior_centers must have shape "
                f"({int(n_dims)},), got {centers.shape}",
            )
    elif centers.ndim == 2:
        valid_shape = centers.shape == (int(n_images), int(n_dims))
        if max_image_index is not None:
            valid_shape = valid_shape or (
                centers.shape[1] == int(n_dims) and centers.shape[0] > int(max_image_index)
            )
        if not valid_shape:
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
    if translation_prior_centers is None:
        return None
    centers = np.asarray(translation_prior_centers, dtype=np.float32)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    if centers.ndim == 1:
        rows = int(image_indices.size if batch_size is None else batch_size)
        return np.broadcast_to(centers[None, :], (rows, centers.shape[0]))
    return centers[image_indices]


def translation_sqdist_angstrom(translations, centers, voxel_size: float):
    if centers is None:
        return None
    voxel = float(voxel_size if voxel_size > 0 else 1.0)
    return np.sum(
        ((np.asarray(translations, dtype=np.float32)[None, :, :] - centers[:, None, :]) * voxel) ** 2,
        axis=-1,
        dtype=np.float64,
    )
