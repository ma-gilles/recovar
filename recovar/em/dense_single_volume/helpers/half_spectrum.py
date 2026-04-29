"""Half-spectrum weights and shell-index helpers shared by dense EM engines."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


def make_half_image_weights(image_shape):
    """Return Hermitian weights for half-spectrum inner products."""

    height, width = image_shape
    weights = 2.0 * jnp.ones((height, width // 2 + 1), dtype=jnp.float32)
    weights = weights.at[:, 0].set(1.0)
    weights = weights.at[:, -1].set(1.0)
    return weights.reshape(-1)


def make_scoring_half_image_weights(image_shape, *, relion_half_sum: bool):
    """Return half-spectrum weights for likelihood scoring.

    RELION scores the packed rfft half-plane with unit weights rather than
    Hermitian weights. Keep this convention centralized so dense/local/sparse
    scoring cannot drift apart.
    """

    height, width = image_shape
    if relion_half_sum:
        return jnp.ones(height * (width // 2 + 1), dtype=jnp.float32)
    return make_half_image_weights(image_shape)


def make_shell_indices_half(image_shape):
    """Return half-spectrum radial shell indices in packed-rfft layout."""

    radii = fourier_transform_utils.get_grid_of_radial_distances_real(
        image_shape,
        voxel_size=1,
        scaled=False,
        frequency_shift=0,
        rounded=True,
    )
    return radii.reshape(-1).astype(jnp.int32)


def half_spectrum_dc_index(image_shape) -> int:
    """Return the flat packed-rfft index of the DC pixel."""
    shell_indices = np.asarray(make_shell_indices_half(image_shape), dtype=np.int32)
    dc_indices = np.flatnonzero(shell_indices == 0)
    if dc_indices.size != 1:
        raise ValueError(f"Expected exactly one half-spectrum DC pixel, found {dc_indices.size}")
    return int(dc_indices[0])


def make_relion_noise_shell_indices_half(image_shape):
    """Return RELION's non-redundant half-plane shell indices for noise sums."""

    height, width = int(image_shape[0]), int(image_shape[1])
    n_shells = height // 2 + 1
    shell_indices = np.asarray(make_shell_indices_half(image_shape), dtype=np.int32).reshape(
        height,
        width // 2 + 1,
    )
    coords = np.asarray(
        fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
            image_shape,
            voxel_size=1,
            scaled=False,
        ),
    ).reshape(height, width // 2 + 1, 2)
    kx = np.rint(coords[..., 0]).astype(np.int32)
    ky = np.rint(coords[..., 1]).astype(np.int32)
    keep = shell_indices < n_shells
    keep &= ~((kx == 0) & (ky < 0))
    shell_indices = np.where(keep, shell_indices, n_shells)
    return jnp.asarray(shell_indices.reshape(-1), dtype=jnp.int32)


def bin_shell_values_np(values, shell_indices, n_shells):
    """Bin per-pixel values into integer shell indices on host."""

    return np.bincount(
        np.asarray(shell_indices, dtype=np.int64),
        weights=np.asarray(values, dtype=np.float64),
        minlength=int(n_shells),
    )[: int(n_shells)]
