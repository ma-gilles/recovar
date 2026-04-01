"""Coordinate-preserving Fourier windowing for resolution-dependent EM.

Implements RELION's ``rlnCurrentImageSize`` concept: at early iterations,
restrict computations to low-frequency shells.  Instead of passing a smaller
``image_shape`` to slice_volume (which would break the CUDA kernel's
``volume_shape[0] // image_shape[0]`` upsampling factor), we apply a
frequency-radius mask on the original half-spectrum grid and use
gather/scatter to operate on only the unmasked indices.

This gives the same FLOP reduction as actual Fourier cropping while preserving
correct physical frequency spacing.

**Restricted size set**: Only sizes that are divisors of the original image
dimension are allowed (``[32, 64, 128]`` for 128px images).  Non-divisor sizes
break RECOVAR's CUDA scaling rule.

See ``docs/math/plan_relion_parity.md``, Phase 3.
"""

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

# Allowed current_size values for 128px images.
# These are divisors of 128 that keep the CUDA kernel's upsampling factor valid.
ALLOWED_CURRENT_SIZES = [32, 64, 128]


def make_frequency_radius_map_half(image_shape):
    """Return (N_half,) array of frequency radius at each pixel of the half-spectrum.

    Uses the same coordinate system as ``ftu.get_k_coordinate_of_each_pixel_half``:
    unscaled integer frequency indices in the packed half-spectrum layout.

    Parameters
    ----------
    image_shape : tuple (H, W)
        Original real-space image shape.

    Returns
    -------
    radii : jnp.ndarray, shape (N_half,), dtype float32
        Euclidean distance from DC for each half-spectrum pixel.
    """
    # Get (N_half, 2) frequency coordinates in unscaled integer units
    coords = ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False)
    # coords[:, 0] is x (col direction), coords[:, 1] is y (row direction)
    # due to indexing="xy" in meshgrid
    return jnp.sqrt(jnp.sum(coords ** 2, axis=-1))


def make_fourier_window_indices(image_shape, current_size):
    """Return sorted 1D integer indices into the half-spectrum that select
    frequencies within the current resolution shell.

    Parameters
    ----------
    image_shape : tuple (H, W)
        Original real-space image shape.
    current_size : int
        Diameter in pixels (like RELION's rlnCurrentImageSize).
        Frequencies with radius <= current_size // 2 are selected.

    Returns
    -------
    indices : jnp.ndarray of int32
        Sorted indices into the (N_half,) flat half-spectrum array.
        Length varies with current_size.
    """
    r_max = current_size // 2
    radii = make_frequency_radius_map_half(image_shape)
    mask = radii <= r_max
    return jnp.where(mask, size=_max_window_size(image_shape, current_size),
                     fill_value=0)[0]


def _max_window_size(image_shape, current_size):
    """Upper bound on the number of half-spectrum pixels within radius.

    Used as the ``size`` argument for ``jnp.where`` to make the output
    shape static (required for JIT).  We use the actual count computed
    eagerly on the host.
    """
    r_max = current_size // 2
    H, W = image_shape
    # Compute on host with numpy for the size parameter
    # Use the same coordinate computation as make_frequency_radius_map_half
    coords_np = np.array(ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False))
    radii_np = np.sqrt(np.sum(coords_np ** 2, axis=-1))
    return int(np.sum(radii_np <= r_max))


def make_fourier_window_indices_np(image_shape, current_size):
    """NumPy version of make_fourier_window_indices for host-side precomputation.

    This avoids JIT compilation overhead and is suitable for precomputing
    the window indices once before the EM loop.

    Parameters
    ----------
    image_shape : tuple (H, W)
    current_size : int

    Returns
    -------
    indices : np.ndarray of int32, sorted
    n_windowed : int
    """
    r_max = current_size // 2
    coords_np = np.array(ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False))
    radii_np = np.sqrt(np.sum(coords_np ** 2, axis=-1))
    mask = radii_np <= r_max
    indices = np.where(mask)[0].astype(np.int32)
    return indices, len(indices)


def quantize_current_size(cs, allowed=None):
    """Round up ``cs`` to the nearest allowed current_size.

    Parameters
    ----------
    cs : int or float
        Raw current_size value (e.g., from 2 * max_FSC_shell).
    allowed : list of int, optional
        Sorted list of allowed sizes. Defaults to ALLOWED_CURRENT_SIZES.

    Returns
    -------
    int : The smallest allowed size >= cs.
    """
    if allowed is None:
        allowed = ALLOWED_CURRENT_SIZES
    for s in allowed:
        if s >= cs:
            return s
    return allowed[-1]
