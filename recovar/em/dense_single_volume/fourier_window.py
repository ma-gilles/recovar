"""Coordinate-preserving Fourier windowing for resolution-dependent EM.

Implements RELION's ``rlnCurrentImageSize`` concept: at early iterations,
restrict computations to low-frequency shells.  Instead of passing a smaller
``image_shape`` to slice_volume (which would break the CUDA kernel's
``volume_shape[0] // image_shape[0]`` upsampling factor), we apply a
frequency-radius mask on the original half-spectrum grid and use
gather/scatter to operate on only the unmasked indices.

This gives the same FLOP reduction as actual Fourier cropping while preserving
correct physical frequency spacing.

**Quantized size options**: explicit callers may still request a restricted
size set, but the RELION-parity path now allows any even ``current_size`` up
to the original box size because the gather/scatter window does not change the
underlying CUDA image grid.

See ``docs/math/plan_relion_parity.md``, Phase 3.
"""

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

# Representative sizes kept for explicit callers that still want a bounded set.
ALLOWED_CURRENT_SIZES = [16, 24, 32, 48, 64, 80, 96, 104, 112, 120, 128, 160, 192, 224, 256]


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
    return jnp.sqrt(jnp.sum(coords**2, axis=-1))


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
    # Use rounded radii for RELION-compatible shell assignment
    mask = jnp.round(radii).astype(jnp.int32) <= r_max
    return jnp.where(mask, size=_max_window_size(image_shape, current_size), fill_value=0)[0]


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
    radii_np = np.sqrt(np.sum(coords_np**2, axis=-1))
    # Use rounded radii, matching RELION's ROUND() convention
    radii_rounded = np.round(radii_np).astype(np.int32)
    return int(np.sum(radii_rounded <= r_max))


def make_fourier_window_indices_np(image_shape, current_size, square=False):
    """NumPy version of make_fourier_window_indices for host-side precomputation.

    This avoids JIT compilation overhead and is suitable for precomputing
    the window indices once before the EM loop.

    Uses ROUNDED integer radii for the window cutoff, matching RELION's
    convention where ``Mresol_fine[n] = ROUND(sqrt(kp^2 + ip^2))`` and
    pixels with ``ires <= current_size // 2`` are included.

    Parameters
    ----------
    image_shape : tuple (H, W)
    current_size : int
    square : bool, optional
        If True, use a square window of (current_size, current_size//2+1)
        pixels in the centered half-spectrum, matching RELION's Fourier-crop
        convention.  If False (default), use circular windowing with a
        frequency-radius cutoff.

    Returns
    -------
    indices : np.ndarray of int32, sorted
    n_windowed : int
    """
    H, W = image_shape
    if square:
        # RELION square window: (current_size) rows × (current_size//2+1) cols
        # in centered half-spectrum layout (DC at row H//2, col 0).
        cs = current_size
        half_cs = cs // 2
        n_cols = W // 2 + 1  # total columns in half-spectrum
        r_start = H // 2 - half_cs  # first row included
        r_end = H // 2 + half_cs  # last row + 1
        c_end = half_cs + 1  # columns 0..half_cs
        mask_2d = np.zeros((H, n_cols), dtype=bool)
        mask_2d[r_start:r_end, :c_end] = True
        mask = mask_2d.ravel()
    else:
        r_max = current_size // 2
        coords_np = np.array(ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False))
        radii_np = np.sqrt(np.sum(coords_np**2, axis=-1))
        # Use rounded radii for shell assignment, matching RELION's ROUND() convention
        radii_rounded = np.round(radii_np).astype(np.int32)
        mask = radii_rounded <= r_max
    indices = np.where(mask)[0].astype(np.int32)
    return indices, len(indices)


def quantize_current_size(cs, allowed=None, ori_size=None, min_size=16):
    """Quantize ``cs`` to a valid current_size.

    Parameters
    ----------
    cs : int or float
        Raw current_size value (e.g., from 2 * max_FSC_shell).
    allowed : list of int, optional
        Sorted list of allowed sizes. When provided, round up to the
        smallest allowed size >= ``cs``.
    ori_size : int, optional
        Original image box size. When provided and ``allowed`` is None,
        quantize to the nearest even size in ``[min_size, ori_size]`` (matching
        RELION's arbitrary-even current image sizes).
    min_size : int, optional
        Minimum current_size to allow in the ``ori_size`` path. For tiny
        test boxes where ``ori_size < min_size``, the lower bound is reduced
        automatically so those tests can still exercise non-trivial windowing.

    Returns
    -------
    int
        Quantized current_size.
    """
    if allowed is not None:
        for s in allowed:
            if s >= cs:
                return s
        return allowed[-1]

    if ori_size is not None:
        upper = int(ori_size)
        if upper % 2 != 0:
            upper -= 1
        if upper < 2:
            raise ValueError(f"ori_size must allow at least one even size, got {ori_size}")

        lower = int(min_size)
        if upper < lower:
            lower = max(4, upper // 2)
            if lower % 2 != 0:
                lower -= 1
            lower = max(2, lower)

        q = max(lower, int(np.ceil(cs)))
        if q % 2 != 0:
            q += 1
        return min(q, upper)

    if allowed is None:
        allowed = ALLOWED_CURRENT_SIZES
    for s in allowed:
        if s >= cs:
            return s
    return allowed[-1]
