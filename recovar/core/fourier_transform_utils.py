import jax.numpy as jnp
import numpy as np

DEFAULT_FFT_NORM = "backward"

# TODO: some of these functions are built-in numpy/jnp. These should be used instead, or optimized otherwise


def get_1d_frequency_grid(n, voxel_size=1, scaled=False):
    # Equivalent to the old even/odd linspace logic, but cheaper and exact on integer steps.
    half = n // 2
    grid = jnp.arange(-half, n - half, dtype=jnp.float32)
    if scaled:
        grid = grid / (n * voxel_size)
    return grid


def get_1d_frequency_grid_rfft(n, voxel_size=1, scaled=False):
    """Frequency grid for Hermitian-packed real FFT axis.

    Returns non-negative bins `[0, 1, ..., n//2]` (or scaled equivalent).
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    grid = jnp.arange(0, n // 2 + 1, dtype=jnp.float32)
    if scaled:
        grid = grid / (n * voxel_size)
    return grid


def get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="xy")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_k_coordinate_of_each_pixel_3d(image_shape, voxel_size, scaled=True):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled=True):
    """2D packed-spectrum frequency coordinates for real-input FFT.

    `image_shape` is the original real image shape `(H, W)`. The returned
    coordinates enumerate the packed spectrum with shape `(H, W//2 + 1)`.
    """
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    one_d_grids = [
        get_1d_frequency_grid(image_shape[0], voxel_size, scaled),
        get_1d_frequency_grid_rfft(image_shape[1], voxel_size, scaled),
    ]
    grids = jnp.meshgrid(*one_d_grids, indexing="xy")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_k_coordinate_of_each_pixel_3d_real(image_shape, voxel_size, scaled=True):
    """3D packed-spectrum frequency coordinates for real-input FFT.

    `image_shape` is the original real volume shape `(D1, D2, D3)`. The
    returned coordinates enumerate the packed spectrum with shape
    `(D1, D2, D3//2 + 1)`.
    """
    if len(image_shape) != 3:
        raise ValueError(f"image_shape must have 3 dims, got {image_shape}")
    one_d_grids = [
        get_1d_frequency_grid(image_shape[0], voxel_size, scaled),
        get_1d_frequency_grid(image_shape[1], voxel_size, scaled),
        get_1d_frequency_grid_rfft(image_shape[2], voxel_size, scaled),
    ]
    grids = jnp.meshgrid(*one_d_grids, indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_grid_of_radial_distances(image_shape, voxel_size=1, scaled=False, frequency_shift=0, rounded=True):
    # Build squared distances with broadcasting per axis to avoid materializing
    # a full (..., ndim) coordinate stack, which saves GPU memory.
    ndim = len(image_shape)
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]

    shift = jnp.asarray(frequency_shift, dtype=one_d_grids[0].dtype)
    if shift.ndim == 0:
        shift = jnp.full((ndim,), shift, dtype=one_d_grids[0].dtype)

    radial_sq = jnp.zeros(tuple(image_shape), dtype=one_d_grids[0].dtype)
    for axis, (g, s) in enumerate(zip(one_d_grids, shift)):
        shape = [1] * ndim
        shape[axis] = image_shape[axis]
        radial_sq = radial_sq + (g.reshape(shape) - s) ** 2

    radial = jnp.sqrt(radial_sq)
    if rounded and not scaled:
        return jnp.round(radial).astype(jnp.int32)
    return radial


def get_grid_of_radial_distances_real(image_shape, voxel_size=1, scaled=False, frequency_shift=0, rounded=True):
    """Radial distance grid for Hermitian-packed real FFT spectra.

    `image_shape` is the original real-space shape; output shape is the packed
    spectrum shape with last axis `N//2 + 1`.
    """
    image_shape = tuple(int(s) for s in image_shape)
    ndim = len(image_shape)
    if ndim < 1:
        raise ValueError("image_shape must have at least one dimension")
    if any(s <= 0 for s in image_shape):
        raise ValueError(f"image_shape entries must be positive, got {image_shape}")

    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape[:-1]]
    one_d_grids.append(get_1d_frequency_grid_rfft(image_shape[-1], voxel_size, scaled))

    shift = jnp.asarray(frequency_shift, dtype=one_d_grids[0].dtype)
    if shift.ndim == 0:
        shift = jnp.full((ndim,), shift, dtype=one_d_grids[0].dtype)
    elif shift.shape != (ndim,):
        raise ValueError(f"frequency_shift must be scalar or shape ({ndim},), got {shift.shape}")

    radial_shape = tuple(image_shape[:-1]) + (image_shape[-1] // 2 + 1,)
    radial_sq = jnp.zeros(radial_shape, dtype=one_d_grids[0].dtype)
    for axis, (g, s) in enumerate(zip(one_d_grids, shift)):
        shape = [1] * ndim
        shape[axis] = g.shape[0]
        radial_sq = radial_sq + (g.reshape(shape) - s) ** 2

    radial = jnp.sqrt(radial_sq)
    if rounded and not scaled:
        return jnp.round(radial).astype(jnp.int32)
    return radial


def get_dft(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.fftshift(jnp.fft.fft(jnp.fft.fftshift(img, axes=(-1,)), norm=norm), axes=(-1,))


def get_idft(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.ifftshift(img, axes=(-1,)), norm=norm), axes=(-1,))


def get_dft2(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.fftshift(img, axes=(-2, -1)), norm=norm),
        axes=(-2, -1),
    )


def get_idft2(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.ifftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(img, axes=(-2, -1)), norm=norm),
        axes=(-2, -1),
    )


def get_dft3(img, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = jnp.fft.fftshift(img, axes=axes)
    img = jnp.fft.fftn(img, axes=axes, norm=norm)
    img = jnp.fft.fftshift(img, axes=axes)
    return img


def get_idft3(img, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = jnp.fft.ifftshift(img, axes=axes)
    img = jnp.fft.ifftn(img, axes=axes, norm=norm)
    img = jnp.fft.ifftshift(img, axes=axes)
    return img


def get_real_fft_packed_shape(shape):
    """Return Hermitian-packed spectrum shape for a real-valued signal.

    The last axis is reduced from `N` to `N//2 + 1` as in `rfft/rfftn`.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) == 0:
        raise ValueError("shape must have at least one dimension")
    if any(s <= 0 for s in shape):
        raise ValueError(f"shape entries must be positive, got {shape}")
    return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)


def get_real_fft_packed_last_axis_indices(n):
    """Indices in shifted full spectrum that correspond to packed real bins."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    half = n // 2
    if n % 2 == 0:
        # [0, 1, ..., n/2] mapped into shifted axis ordering.
        return jnp.asarray(list(range(half, n)) + [0], dtype=jnp.int32)
    return jnp.asarray(list(range(half, n)), dtype=jnp.int32)


def _half_image_pixel_indices(image_shape):
    """Flat pixel indices into the full ``(H*W,)`` array corresponding to half-image pixels.

    This mirrors the extraction performed by :func:`full_image_to_half_image`:
    reshape ``(H*W,)`` to ``(H, W)``, take columns ``packed_last_idx``, flatten
    to ``(H*(W//2+1),)``.
    """
    H, W = image_shape
    packed_col = get_real_fft_packed_last_axis_indices(W)
    row_idx = jnp.arange(H)[:, None]
    return (row_idx * W + packed_col[None, :]).ravel()


def get_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled=True):
    """Half-image frequency coords consistent with ``full_image_to_half_image``."""
    full = get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled)
    return full[_half_image_pixel_indices(image_shape)]


def get_shifted_conjugate_partner_indices(n):
    """Index of the conjugate-symmetric partner for each bin in a shifted FFT axis.

    For a length-*n* axis after ``fftshift``, bin *i* holds frequency
    ``u = (i + n//2) % n``.  Its conjugate partner is at frequency ``-u % n``,
    which maps back to shifted index ``(-u % n - n//2) % n``.  This function
    returns that mapping as an int32 array of length *n*.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    half = n // 2
    i = jnp.arange(n, dtype=jnp.int32)
    u = (i + half) % n
    u_partner = (-u) % n
    return ((u_partner - half) % n).astype(jnp.int32)


def get_real_fft_memory_saving_ratio(shape):
    """Ratio of packed (rfft) to full spectrum size: ``prod(packed) / prod(shape)``."""
    packed = get_real_fft_packed_shape(shape)
    return int(np.prod(packed)) / int(np.prod(shape))


def _normalize_volume_shape_3d(volume_shape):
    volume_shape = tuple(int(s) for s in volume_shape)
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have 3 dims, got {volume_shape}")
    if any(s <= 0 for s in volume_shape):
        raise ValueError(f"volume_shape entries must be positive, got {volume_shape}")
    return volume_shape


def _normalize_image_shape_2d(image_shape):
    image_shape = tuple(int(s) for s in image_shape)
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    if any(s <= 0 for s in image_shape):
        raise ValueError(f"image_shape entries must be positive, got {image_shape}")
    return image_shape


def image_shape_to_half_image_shape(image_shape):
    """Packed real-spectrum shape for 2D image shape under last-axis rFFT."""
    image_shape = _normalize_image_shape_2d(image_shape)
    return tuple(image_shape[:-1]) + (image_shape[-1] // 2 + 1,)


def volume_shape_to_half_volume_shape(volume_shape):
    """Packed real-spectrum shape for 3D volume shape under last-axis rFFT."""
    volume_shape = _normalize_volume_shape_3d(volume_shape)
    return tuple(volume_shape[:-1]) + (volume_shape[-1] // 2 + 1,)


def _coerce_grid_or_flat(arr, grid_shape, name):
    arr = jnp.asarray(arr)
    grid_shape = tuple(int(s) for s in grid_shape)
    flat_size = int(np.prod(grid_shape))

    if arr.ndim >= len(grid_shape) and tuple(arr.shape[-len(grid_shape) :]) == grid_shape:
        return arr, False

    if arr.ndim >= 1 and int(arr.shape[-1]) == flat_size:
        out_shape = tuple(arr.shape[:-1]) + grid_shape
        return arr.reshape(out_shape), True

    raise ValueError(f"{name} must have trailing shape {grid_shape} or trailing flat size {flat_size}, got {arr.shape}")


def _restore_grid_or_flat(arr_grid, return_flat, n_spatial_dims):
    if not return_flat:
        return arr_grid
    flat_size = int(np.prod(arr_grid.shape[-n_spatial_dims:]))
    return arr_grid.reshape(tuple(arr_grid.shape[:-n_spatial_dims]) + (flat_size,))


def full_image_to_half_image(image, image_shape):
    """Map centered full 2D spectrum to packed Hermitian representation.

    Accepts either trailing grid shape `(..., *image_shape)` or flattened
    trailing axis `(..., prod(image_shape))`. Returns in the same style.
    """
    image_shape = _normalize_image_shape_2d(image_shape)
    image_grid, was_flat = _coerce_grid_or_flat(image, image_shape, name="image")

    packed_last_idx = get_real_fft_packed_last_axis_indices(image_shape[-1])
    half_grid = jnp.take(image_grid, packed_last_idx, axis=-1)
    return _restore_grid_or_flat(half_grid, was_flat, n_spatial_dims=2)


def half_image_to_full_image(half_image, image_shape):
    """Map packed Hermitian 2D spectrum to centered full complex spectrum.

    Uses index-based Hermitian conjugation: non-redundant columns are placed
    at their packed positions, redundant columns are filled as
    ``conj(half[(-i0)%H, |kx|])``.

    IMPORTANT: Do NOT change this to an FFT-based round-trip (e.g. irfft2 → fft2).
    This must stay consistent with half_volume_to_full_volume's approach so that
    VJPs match the CUDA kernel's Hermitian fold scatter.

    Accepts either trailing packed grid shape
    `(..., *image_shape_to_half_image_shape(image_shape))` or flattened
    trailing axis `(..., prod(half_shape))`. Returns in the same style.
    """
    image_shape = _normalize_image_shape_2d(image_shape)
    half_shape = image_shape_to_half_image_shape(image_shape)
    half_grid, was_flat = _coerce_grid_or_flat(half_image, half_shape, name="half_image")

    H, W = image_shape
    ic = W // 2

    # Place non-redundant columns at packed positions
    packed_idx = get_real_fft_packed_last_axis_indices(W)
    full_grid = jnp.zeros(half_grid.shape[:-2] + (H, W), dtype=half_grid.dtype)
    full_grid = full_grid.at[..., :, packed_idx].set(half_grid)

    # Fill redundant columns by Hermitian conjugation
    if W % 2 == 0:
        redundant = jnp.arange(1, ic)
    else:
        redundant = jnp.arange(0, ic)

    if redundant.size > 0:
        # Hermitian partner in centered (fftshift) convention:
        #   shifted[j] = Y[(j - N//2) % N], partner u' = (N - u) % N
        #   => partner(j) = (N - j + 2*(N//2)) % N
        #   Even N: 2*(N//2) = N   => partner(j) = (N - j) % N
        #   Odd N:  2*(N//2) = N-1 => partner(j) = (N - 1 - j) % N
        #   General: partner(j) = (N - (N % 2) - j) % N
        partner_i0 = (H - (H % 2) - jnp.arange(H)) % H
        conj_partner = jnp.conj(jnp.take(half_grid, partner_i0, axis=-2))
        source_cols = ic - redundant
        full_grid = full_grid.at[..., :, redundant].set(conj_partner[..., source_cols])

    return _restore_grid_or_flat(full_grid, was_flat, n_spatial_dims=2)


def full_volume_to_half_volume(volume, volume_shape):
    """Map centered full 3D spectrum to packed Hermitian representation.

    Accepts either trailing grid shape `(..., *volume_shape)` or flattened
    trailing axis `(..., prod(volume_shape))`. Returns in the same style.
    """
    volume_shape = _normalize_volume_shape_3d(volume_shape)
    volume_grid, was_flat = _coerce_grid_or_flat(volume, volume_shape, name="volume")

    packed_last_idx = get_real_fft_packed_last_axis_indices(volume_shape[-1])
    half_grid = jnp.take(volume_grid, packed_last_idx, axis=-1)
    return _restore_grid_or_flat(half_grid, was_flat, n_spatial_dims=3)


def half_volume_to_full_volume(half_volume, volume_shape):
    """Map packed Hermitian 3D spectrum to centered full complex spectrum.

    Uses index-based Hermitian conjugation: non-redundant columns are placed
    at their packed positions, redundant columns (negative last-axis
    frequencies) are filled as ``conj(half[(-i0)%N0, (-i1)%N1, |kz|])``.

    This approach has a simple VJP that matches per-voxel Hermitian folding
    in the CUDA backproject kernel (unlike the FFT round-trip which
    distributes gradients through the FFT chain differently).

    IMPORTANT: Do NOT change this to an FFT-based round-trip (e.g. irfft3 → fft3).
    The VJP of an FFT round-trip distributes gradients differently than
    per-voxel Hermitian folding, breaking the CUDA half-volume backproject
    kernel's correctness.  The CUDA kernel's Hermitian fold scatter is
    specifically designed to be the correct adjoint of THIS index-based expand.

    Accepts either trailing packed grid shape
    `(..., *volume_shape_to_half_volume_shape(volume_shape))` or flattened
    trailing axis `(..., prod(half_shape))`. Returns in the same style.
    """
    volume_shape = _normalize_volume_shape_3d(volume_shape)
    half_shape = volume_shape_to_half_volume_shape(volume_shape)
    half_grid, was_flat = _coerce_grid_or_flat(half_volume, half_shape, name="half_volume")

    N0, N1, N2 = volume_shape
    ic2 = N2 // 2

    # Place non-redundant columns (kz = 0, 1, ..., N2//2) at packed positions
    packed_idx = get_real_fft_packed_last_axis_indices(N2)
    full_grid = jnp.zeros(half_grid.shape[:-3] + (N0, N1, N2), dtype=half_grid.dtype)
    full_grid = full_grid.at[..., :, :, packed_idx].set(half_grid)

    # Fill redundant columns (negative kz) by Hermitian conjugation.
    # For even N2, index 0 (Nyquist) is already placed by packed_idx.
    # For odd N2, index 0 is redundant and needs filling.
    if N2 % 2 == 0:
        redundant = jnp.arange(1, ic2)
    else:
        redundant = jnp.arange(0, ic2)

    if redundant.size > 0:
        # Hermitian partner indices in the centered (fftshift) convention:
        #   shifted[j] = Y[(j - N//2) % N], partner u' = (N - u) % N
        #   => partner(j) = (N - j + 2*(N//2)) % N
        #   Even N: 2*(N//2) = N   => partner(j) = (N - j) % N
        #   Odd N:  2*(N//2) = N-1 => partner(j) = (N - 1 - j) % N
        #   General: partner(j) = (N - (N % 2) - j) % N
        partner_i0 = (N0 - (N0 % 2) - jnp.arange(N0)) % N0
        partner_i1 = (N1 - (N1 % 2) - jnp.arange(N1)) % N1
        conj_partner = jnp.conj(jnp.take(jnp.take(half_grid, partner_i0, axis=-3), partner_i1, axis=-2))
        source_cols = ic2 - redundant
        full_grid = full_grid.at[..., :, :, redundant].set(conj_partner[..., source_cols])

    return _restore_grid_or_flat(full_grid, was_flat, n_spatial_dims=3)


def get_dft2_real(img, norm=DEFAULT_FFT_NORM):
    """Centered 2D FFT for real-valued inputs using Hermitian packing.

    Output shape is `(..., H, W//2 + 1)` with only the non-redundant last-axis
    frequencies stored.
    """
    img = jnp.fft.fftshift(img, axes=(-2, -1))
    img = jnp.fft.rfft2(img, norm=norm)
    # Shift only non-packed axes.
    img = jnp.fft.fftshift(img, axes=(-2,))
    return img


def get_idft2_real(img, image_shape=None, norm=DEFAULT_FFT_NORM):
    """Inverse of `get_dft2_real` returning a real-valued image."""
    img = jnp.fft.ifftshift(img, axes=(-2,))
    if image_shape is None:
        image_shape = (img.shape[-2], 2 * (img.shape[-1] - 1))
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    img = jnp.fft.irfft2(img, s=tuple(int(s) for s in image_shape), norm=norm)
    img = jnp.fft.ifftshift(img, axes=(-2, -1))
    return img


def get_dft3_real(img, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    """Centered 3D FFT for real-valued inputs using Hermitian packing.

    The packed axis is the last transform axis in `axes`.
    """
    axes = tuple(axes)
    if len(axes) != 3:
        raise ValueError(f"axes must have length 3, got {axes}")
    img = jnp.fft.fftshift(img, axes=axes)
    img = jnp.fft.rfftn(img, axes=axes, norm=norm)
    # Shift only non-packed axes (all except the final transform axis).
    img = jnp.fft.fftshift(img, axes=axes[:-1])
    return img


def get_idft3_real(img, volume_shape=None, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    """Inverse of `get_dft3_real` returning a real-valued volume."""
    axes = tuple(axes)
    if len(axes) != 3:
        raise ValueError(f"axes must have length 3, got {axes}")

    img = jnp.fft.ifftshift(img, axes=axes[:-1])

    if volume_shape is None:
        # Infer for the common trailing-axes case.
        if axes != (-3, -2, -1):
            raise ValueError("volume_shape is required when axes != (-3, -2, -1)")
        volume_shape = (img.shape[-3], img.shape[-2], 2 * (img.shape[-1] - 1))
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have 3 dims, got {volume_shape}")

    img = jnp.fft.irfftn(img, s=tuple(int(s) for s in volume_shape), axes=axes, norm=norm)
    img = jnp.fft.ifftshift(img, axes=axes)
    return img
