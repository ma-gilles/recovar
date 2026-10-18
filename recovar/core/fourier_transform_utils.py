import collections
import functools
import threading

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_FFT_NORM = "backward"

# TODO: some of these functions are built-in numpy/jnp. These should be used instead, or optimized otherwise


# --------------------------------------------------- constant-grid memoization ---
#
# The frequency and pixel-coordinate grids below are pure functions of their
# static arguments (shape, voxel spacing, the `scaled` flag and the dtype), yet
# their callers rebuild them once per image batch.  Each rebuild costs a handful
# of eager JAX dispatches, and together they are ~19% of the host dispatches of a
# steady EM iteration (P3-E census, 2026-09-20: 1949 of 10501 at healpix order 3).
#
# They are therefore memoized on their full static argument tuple.  Three
# properties make that safe, and callers must preserve them:
#
# * A cache hit returns THE SAME device array object, not a copy.  JAX arrays are
#   immutable -- `arr.at[...].set(...)` returns a new array and leaves the cached
#   one untouched -- so aliasing cannot corrupt another caller's grid.  The
#   returned array must still be treated as read-only static geometry.
# * None of these grids is ever passed as a donated argument.  The only donated
#   buffers in the repository are the EM M-step accumulators and the projection
#   cache; a cached constant handed to a donating program would be a
#   use-after-donation on the next hit.
# * The bytes are unchanged.  A hit skips the rebuild; a miss runs exactly the
#   expression the unmemoized helper ran, on exactly the arguments it was given.
#
# The key carries, besides the arguments, the two pieces of interpreter state a
# built grid depends on: the thread's `jax.default_device` (so a grid built on
# one device is never handed back inside a `jax.default_device` block selecting
# another, which would silently move downstream work) and `jax_enable_x64`
# (which decides the canonical dtype).  Arguments that are not plain Python or
# NumPy scalars -- a tracer, a device array, anything unhashable -- make the call
# uncacheable and fall through to the unmemoized builder.
#
# The cache is bypassed entirely while a JAX trace is capturing operations.
# Inside `jax.jit` (or vmap/grad) these helpers stage their arange/meshgrid into
# the program being traced rather than producing a concrete array, so serving a
# concrete cached array there would replace staged operations with a jaxpr
# constant and change the HLO that XLA then fuses.  Bypassing keeps every
# compiled program byte-identical to the unmemoized module; the dispatches this
# memoization targets are eager ones by definition, so nothing is lost.

_GRID_CACHE_MAXSIZE = 256
_grid_cache: "collections.OrderedDict[tuple, object]" = collections.OrderedDict()
_grid_cache_lock = threading.Lock()
_UNCACHEABLE = object()
_CACHE_MISS = object()


def _static_scalar_token(value):
    """Hashable, type-faithful cache token for a static scalar argument."""
    if isinstance(value, (bool, int, float, complex, str, np.generic)):
        return (type(value), value)
    return _UNCACHEABLE


def _static_shape_token(shape):
    """Hashable cache token for a static shape argument."""
    if isinstance(shape, (str, bytes)):
        return _UNCACHEABLE
    try:
        entries = tuple(shape)
    except TypeError:
        return _static_scalar_token(shape)
    tokens = tuple(_static_scalar_token(entry) for entry in entries)
    if any(token is _UNCACHEABLE for token in tokens):
        return _UNCACHEABLE
    return tokens


def _static_dtype_token(dtype):
    """Hashable cache token for a dtype argument, canonical across spellings."""
    if dtype is None:
        return (type(None), None)
    try:
        return ("dtype", np.dtype(dtype))
    except TypeError:
        return _UNCACHEABLE


def _grid_cache_context():
    """Interpreter state a built grid depends on besides its arguments."""
    return (jax.config.jax_default_device, jax.config.jax_enable_x64)


def _tracing():
    """True while a JAX trace (jit, vmap, grad, ...) is capturing operations.

    An unrecognised trace type reads as "tracing", which only bypasses the
    cache; it can never serve a staged value.
    """
    return type(jax.core.trace_ctx.trace).__name__ != "EvalTrace"


def _memoized_grid(name, tokens, build):
    """Return ``build()``, memoized on ``(name, interpreter state, tokens)``.

    Falls through to ``build()`` without touching the cache while a JAX trace is
    active, when any token is uncacheable, or if the built value is a tracer.
    """
    if _tracing() or any(token is _UNCACHEABLE for token in tokens):
        return build()
    key = (name, _grid_cache_context()) + tuple(tokens)
    with _grid_cache_lock:
        cached = _grid_cache.get(key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            _grid_cache.move_to_end(key)
            return cached
    value = build()
    if isinstance(value, jax.core.Tracer):  # defensive: never store a staged value
        return value
    with _grid_cache_lock:
        _grid_cache[key] = value
        _grid_cache.move_to_end(key)
        while len(_grid_cache) > _GRID_CACHE_MAXSIZE:
            _grid_cache.popitem(last=False)
    return value


def clear_grid_cache():
    """Drop every memoized constant grid.  Diagnostics and tests only."""
    with _grid_cache_lock:
        _grid_cache.clear()
    _get_k_coordinate_of_each_pixel_half_np_cached.cache_clear()


def grid_cache_size():
    """Number of memoized constant grids currently held."""
    with _grid_cache_lock:
        return len(_grid_cache)


def _build_1d_frequency_grid(n, voxel_size, scaled, dtype):
    # Equivalent to the old even/odd linspace logic, but cheaper and exact on integer steps.
    half = n // 2
    grid = jnp.arange(-half, n - half, dtype=dtype)
    if scaled:
        grid = grid / (jnp.asarray(n, dtype=dtype) * jnp.asarray(voxel_size, dtype=dtype))
    return grid


def get_1d_frequency_grid(n, voxel_size=1, scaled=False, dtype=jnp.float32):
    return _memoized_grid(
        "get_1d_frequency_grid",
        (
            _static_scalar_token(n),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
            _static_dtype_token(dtype),
        ),
        lambda: _build_1d_frequency_grid(n, voxel_size, scaled, dtype),
    )


def _build_1d_frequency_grid_rfft(n, voxel_size, scaled, dtype):
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    grid = jnp.arange(0, n // 2 + 1, dtype=dtype)
    if scaled:
        grid = grid / (jnp.asarray(n, dtype=dtype) * jnp.asarray(voxel_size, dtype=dtype))
    return grid


def get_1d_frequency_grid_rfft(n, voxel_size=1, scaled=False, dtype=jnp.float32):
    """Frequency grid for Hermitian-packed real FFT axis.

    Returns non-negative bins `[0, 1, ..., n//2]` (or scaled equivalent).
    """
    return _memoized_grid(
        "get_1d_frequency_grid_rfft",
        (
            _static_scalar_token(n),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
            _static_dtype_token(dtype),
        ),
        lambda: _build_1d_frequency_grid_rfft(n, voxel_size, scaled, dtype),
    )


def _build_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled, dtype=dtype) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="xy")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True, dtype=jnp.float32):
    return _memoized_grid(
        "get_k_coordinate_of_each_pixel",
        (
            _static_shape_token(image_shape),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
            _static_dtype_token(dtype),
        ),
        lambda: _build_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype),
    )


def _build_k_coordinate_of_each_pixel_3d(image_shape, voxel_size, scaled):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)


def get_k_coordinate_of_each_pixel_3d(image_shape, voxel_size, scaled=True):
    return _memoized_grid(
        "get_k_coordinate_of_each_pixel_3d",
        (
            _static_shape_token(image_shape),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
        ),
        lambda: _build_k_coordinate_of_each_pixel_3d(image_shape, voxel_size, scaled),
    )


def get_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled=True):
    """2D packed-spectrum frequency coordinates for real-input FFT.

    `image_shape` is the original real image shape `(H, W)`. The returned
    coordinates enumerate the packed spectrum with shape `(H, W//2 + 1)`.
    """
    return _memoized_grid(
        "get_k_coordinate_of_each_pixel_real",
        (
            _static_shape_token(image_shape),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
        ),
        lambda: _build_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled),
    )


def _build_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled):
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
    return _memoized_grid(
        "get_k_coordinate_of_each_pixel_3d_real",
        (
            _static_shape_token(image_shape),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
        ),
        lambda: _build_k_coordinate_of_each_pixel_3d_real(image_shape, voxel_size, scaled),
    )


def _build_k_coordinate_of_each_pixel_3d_real(image_shape, voxel_size, scaled):
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


def _build_real_fft_packed_last_axis_indices(n):
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    half = n // 2
    if n % 2 == 0:
        # [0, 1, ..., n/2] mapped into shifted axis ordering.
        return jnp.asarray(list(range(half, n)) + [0], dtype=jnp.int32)
    return jnp.asarray(list(range(half, n)), dtype=jnp.int32)


def get_real_fft_packed_last_axis_indices(n):
    """Indices in shifted full spectrum that correspond to packed real bins."""
    return _memoized_grid(
        "get_real_fft_packed_last_axis_indices",
        (_static_scalar_token(n),),
        lambda: _build_real_fft_packed_last_axis_indices(n),
    )


def _build_half_image_pixel_indices(image_shape):
    H, W = image_shape
    packed_col = get_real_fft_packed_last_axis_indices(W)
    row_idx = jnp.arange(H)[:, None]
    return (row_idx * W + packed_col[None, :]).ravel()


def _half_image_pixel_indices(image_shape):
    """Flat pixel indices into the full ``(H*W,)`` array corresponding to half-image pixels.

    This mirrors the extraction performed by :func:`full_image_to_half_image`:
    reshape ``(H*W,)`` to ``(H, W)``, take columns ``packed_last_idx``, flatten
    to ``(H*(W//2+1),)``.
    """
    return _memoized_grid(
        "_half_image_pixel_indices",
        (_static_shape_token(image_shape),),
        lambda: _build_half_image_pixel_indices(image_shape),
    )


def _build_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled, dtype):
    full = get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype=dtype)
    return full[_half_image_pixel_indices(image_shape)]


def get_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled=True, dtype=jnp.float32):
    """Half-image frequency coords consistent with ``full_image_to_half_image``."""
    return _memoized_grid(
        "get_k_coordinate_of_each_pixel_half",
        (
            _static_shape_token(image_shape),
            _static_scalar_token(voxel_size),
            _static_scalar_token(scaled),
            _static_dtype_token(dtype),
        ),
        lambda: _build_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled, dtype),
    )


@functools.lru_cache(maxsize=None)
def _get_k_coordinate_of_each_pixel_half_np_cached(image_shape):
    """Build immutable packed-half coordinates without dispatching JAX work."""

    height, width = image_shape
    one_d_grids = []
    for size in image_shape:
        half = size // 2
        grid = np.arange(-half, size - half, dtype=np.float32)
        one_d_grids.append(grid)
    grids = np.meshgrid(*one_d_grids, indexing="xy")
    full = np.stack([grid.ravel() for grid in grids], axis=-1)

    half_width = width // 2
    if width % 2 == 0:
        packed_columns = np.asarray(list(range(half_width, width)) + [0], dtype=np.int32)
    else:
        packed_columns = np.arange(half_width, width, dtype=np.int32)
    rows = np.arange(height, dtype=np.int32)[:, None]
    packed_pixel_indices = (rows * width + packed_columns[None, :]).ravel()
    coords = np.asarray(full[packed_pixel_indices], dtype=np.float32)
    coords.setflags(write=False)
    return coords


def get_k_coordinate_of_each_pixel_half_np(image_shape, voxel_size=1, scaled=False):
    """Host-planned unscaled packed-half frequency coordinates.

    This is byte-equivalent to :func:`get_k_coordinate_of_each_pixel_half`,
    including its packed-column ordering, but it avoids compiling a chain of
    eager JAX indexing primitives when the coordinates are immediately used
    by host-side Fourier-window planning.  The returned cached array is
    read-only and must be treated as static geometry.  Scaled coordinates
    remain device arithmetic because NumPy and XLA division are not guaranteed
    to be byte-identical.
    """

    image_shape = tuple(int(size) for size in image_shape)
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    if any(size <= 0 for size in image_shape):
        raise ValueError(f"image_shape entries must be positive, got {image_shape}")
    if scaled:
        raise ValueError("host-packed half coordinates require scaled=False")
    del voxel_size  # Unscaled frequency indices are independent of voxel size.
    return _get_k_coordinate_of_each_pixel_half_np_cached(image_shape)


def get_shifted_conjugate_partner_indices(n):
    """Index of the conjugate-symmetric partner for each bin in a shifted FFT axis.

    For a length-*n* axis after ``fftshift``, bin *i* holds frequency
    ``u = (i + n//2) % n``.  Its conjugate partner is at frequency ``-u % n``,
    which maps back to shifted index ``(-u % n - n//2) % n``.  This function
    returns that mapping as an int32 array of length *n*.
    """
    return _memoized_grid(
        "get_shifted_conjugate_partner_indices",
        (_static_scalar_token(n),),
        lambda: _build_shifted_conjugate_partner_indices(n),
    )


def _build_shifted_conjugate_partner_indices(n):
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
