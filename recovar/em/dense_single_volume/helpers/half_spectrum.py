"""Half-spectrum weights and shell-index helpers shared by dense EM engines."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


@dataclass(frozen=True)
class _HostHalfSpectrumPlan:
    """Immutable shape-only arrays shared by dense/local EM and VDAM."""

    hermitian_weights: np.ndarray
    relion_scoring_weights: np.ndarray
    relion_cc_scoring_weights: np.ndarray
    shell_indices: np.ndarray
    relion_noise_shell_indices: np.ndarray
    dc_index: int


def _readonly(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


@functools.lru_cache(maxsize=None)
def _host_half_spectrum_plan(image_shape):
    """Construct exact static half-spectrum geometry without eager JAX calls."""

    height, width = image_shape
    half_width = width // 2 + 1

    hermitian_weights = np.full((height, half_width), 2.0, dtype=np.float32)
    hermitian_weights[:, 0] = 1.0
    hermitian_weights[:, -1] = 1.0

    relion_scoring_weights = np.ones((height, half_width), dtype=np.float32)
    if height > 2:
        relion_scoring_weights[1 : (height + 1) // 2, 0] = 0.0
    relion_cc_scoring_weights = np.ones((height, half_width), dtype=np.float32)

    vertical_grid = np.arange(-(height // 2), height - height // 2, dtype=np.float32)
    packed_grid = np.arange(0, half_width, dtype=np.float32)
    radial_sq = np.zeros((height, half_width), dtype=np.float32)
    radial_sq = radial_sq + vertical_grid[:, None] ** 2
    radial_sq = radial_sq + packed_grid[None, :] ** 2
    shell_indices = np.rint(np.sqrt(radial_sq)).astype(np.int32).reshape(-1)

    coords = fourier_transform_utils.get_k_coordinate_of_each_pixel_half_np(
        image_shape,
        voxel_size=1,
        scaled=False,
    ).reshape(height, half_width, 2)
    kx = np.rint(coords[..., 0]).astype(np.int32)
    ky = np.rint(coords[..., 1]).astype(np.int32)
    n_shells = height // 2 + 1
    shell_grid = shell_indices.reshape(height, half_width)
    vertical_nyquist = (height % 2 == 0) & (ky == -(height // 2))
    redundant_x0 = (kx == 0) & (ky < 0) & ~vertical_nyquist
    keep = (shell_grid < n_shells) & ~redundant_x0
    relion_noise_shell_indices = np.where(keep, shell_grid, n_shells).astype(
        np.int32,
        copy=False,
    )

    dc_indices = np.flatnonzero(shell_indices == 0)
    if dc_indices.size != 1:
        raise ValueError(f"Expected exactly one half-spectrum DC pixel, found {dc_indices.size}")

    return _HostHalfSpectrumPlan(
        hermitian_weights=_readonly(hermitian_weights.reshape(-1)),
        relion_scoring_weights=_readonly(relion_scoring_weights.reshape(-1)),
        relion_cc_scoring_weights=_readonly(relion_cc_scoring_weights.reshape(-1)),
        shell_indices=_readonly(shell_indices),
        relion_noise_shell_indices=_readonly(relion_noise_shell_indices.reshape(-1)),
        dc_index=int(dc_indices[0]),
    )


def _normalize_image_shape(image_shape):
    image_shape = tuple(int(size) for size in image_shape)
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    if any(size <= 0 for size in image_shape):
        raise ValueError(f"image_shape entries must be positive, got {image_shape}")
    return image_shape


def make_half_image_weights(image_shape):
    """Return Hermitian weights for half-spectrum inner products."""

    plan = _host_half_spectrum_plan(_normalize_image_shape(image_shape))
    return jnp.asarray(plan.hermitian_weights)


def make_scoring_half_image_weights(
    image_shape,
    *,
    relion_half_sum: bool,
    exclude_relion_redundant_x0: bool = True,
):
    """Return half-spectrum weights for likelihood scoring.

    RELION scores its non-redundant FFTW half-plane with unit weights rather
    than Hermitian weights.  RECOVAR's centered packed layout also contains
    the conjugate ``kx=0, ky<0`` rows; RELION's Gaussian likelihood leaves
    those redundant rows at zero.  Mask them here so full-size
    dense/local/sparse Gaussian scoring cannot count the axis twice.  The
    ``ky=-N/2`` boundary is retained because RELION represents it as
    ``+N/2``.

    RELION's first-iteration normalized-CC CUDA kernels are different: they
    iterate over every pixel in the rectangular FFTW crop, including both
    sides of the centered ``kx=0`` axis.  Those callers must pass
    ``exclude_relion_redundant_x0=False``.
    """

    image_shape = _normalize_image_shape(image_shape)
    plan = _host_half_spectrum_plan(image_shape)
    if relion_half_sum:
        weights = plan.relion_scoring_weights if exclude_relion_redundant_x0 else plan.relion_cc_scoring_weights
        return jnp.asarray(weights)
    return jnp.asarray(plan.hermitian_weights)


def make_shell_indices_half(image_shape):
    """Return half-spectrum radial shell indices in packed-rfft layout."""

    plan = _host_half_spectrum_plan(_normalize_image_shape(image_shape))
    return jnp.asarray(plan.shell_indices)


def half_spectrum_dc_index(image_shape) -> int:
    """Return the flat packed-rfft index of the DC pixel."""
    plan = _host_half_spectrum_plan(_normalize_image_shape(image_shape))
    return plan.dc_index


def make_relion_noise_shell_indices_half(image_shape):
    """Return RELION's non-redundant half-plane shell indices for noise sums."""

    plan = _host_half_spectrum_plan(_normalize_image_shape(image_shape))
    return jnp.asarray(plan.relion_noise_shell_indices)


def mask_relion_noise_shell_indices_to_current_window(
    shell_indices,
    image_shape,
    current_size,
    current_window_indices,
):
    """Drop low-shell pixels outside RELION's asymmetric current FFT crop.

    A rounded radial shell mask alone retains a few pixels on both even-size
    boundary rows (for example ``ky=+/-28`` at current size 56). RELION's
    ``windowFourierTransform`` rectangle retains only the positive boundary
    row in RECOVAR's centered layout. Pixels on the opposite boundary are
    neither current-image residuals nor high-shell extensions, because their
    rounded shell is still the current cutoff.
    """

    if current_size is None or int(current_size) >= int(image_shape[0]):
        return jnp.asarray(shell_indices, dtype=jnp.int32)
    image_shape = _normalize_image_shape(image_shape)
    shell_indices_np = np.asarray(shell_indices, dtype=np.int32)
    current_window_mask = np.zeros(shell_indices_np.shape, dtype=bool)
    current_window_mask[np.asarray(current_window_indices, dtype=np.int32)] = True
    current_window_mask[half_spectrum_dc_index(image_shape)] = True
    shell_cutoff = int(current_size) // 2
    sentinel = int(image_shape[0]) // 2 + 1
    outside_current_crop = (shell_indices_np <= shell_cutoff) & ~current_window_mask
    masked_shell_indices = np.where(outside_current_crop, sentinel, shell_indices_np).astype(
        np.int32,
        copy=False,
    )
    return jnp.asarray(masked_shell_indices)


def bin_shell_values_jax(values, shell_indices, n_shells):
    """Bin per-pixel values into shell indices, dropping RELION sentinel pixels."""

    shell_count = int(n_shells)
    values = jnp.asarray(values)
    shell_indices = jnp.asarray(shell_indices, dtype=jnp.int32)
    sentinel = jnp.asarray(shell_count, dtype=jnp.int32)
    safe_indices = jnp.where(
        (shell_indices >= 0) & (shell_indices < shell_count),
        shell_indices,
        sentinel,
    )
    bins = jnp.zeros(shell_count + 1, dtype=values.dtype)
    return bins.at[safe_indices].add(values)[:shell_count]


def bin_shell_values_np(values, shell_indices, n_shells):
    """Bin per-pixel values into shell indices, dropping sentinel pixels on host."""

    shell_count = int(n_shells)
    shell_indices_np = np.asarray(shell_indices, dtype=np.int64)
    safe_indices = np.where(
        (shell_indices_np >= 0) & (shell_indices_np < shell_count),
        shell_indices_np,
        shell_count,
    )
    return np.bincount(
        safe_indices,
        weights=np.asarray(values, dtype=np.float64),
        minlength=shell_count + 1,
    )[:shell_count]
