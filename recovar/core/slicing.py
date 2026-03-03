"""Fourier slice extraction and adjoint (backprojection) operations.

Design: CUDA kernels when available, JAX ``map_coordinates`` as fallback.
There is no separate "trilinear" vs "by_map" distinction — all paths go
through :func:`slice_volume_by_map` / :func:`adjoint_slice_volume_by_map`
with ``disc_type`` selecting the interpolation order.

CUDA custom-VJP wrappers live in :mod:`recovar.core.cuda_ops`; this module
contains only dispatch logic and the JAX fallback implementations.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.geometry import rotations_to_grid_point_coords

logger = logging.getLogger(__name__)


# ── CUDA availability (cached) ────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _check_cuda():
    """Return True if CUDA project/backproject kernels are available."""
    try:
        from recovar.cuda_backproject import cuda_available
        return cuda_available()
    except (ImportError, OSError):
        return False


def _is_complex(arr):
    return jnp.issubdtype(arr.dtype, jnp.complexfloating)


def _is_jvp_tracer(arr):
    """Detect forward-mode (JVP) tracing — custom_vjp doesn't support JVP."""
    try:
        from jax._src.interpreters.ad import JVPTracer
        return isinstance(arr, JVPTracer)
    except (ImportError, AttributeError):
        return False


# ── Nearest-neighbour gather (pre-computed indices) ───────────────────

@jax.jit
def slice_volume_by_nearest(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid]


batch_slice_volume_by_nearest = jax.vmap(slice_volume_by_nearest, (None, 0))


# ── Interpolation order helper ────────────────────────────────────────

def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    if disc_type == "nearest":
        return 0
    if disc_type == "cubic":
        return 3
    raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")


# ── Core JAX map_coordinates engine ──────────────────────────────────

@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order):
    coords, og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    if order == 3:
        from recovar.core import cubic_interpolation
        slices = cubic_interpolation.map_coordinates_with_cubic_spline(
            volume, coords, mode="fill", cval=0.0
        ).reshape(og_shape[:-1]).astype(volume.dtype)
    else:
        slices = jax.scipy.ndimage.map_coordinates(
            volume.reshape(volume_shape), coords, order=order, mode="constant", cval=0.0,
        ).reshape(og_shape[:-1]).astype(volume.dtype)
    return slices


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, decide_order(disc_type))


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _adjoint_slice_volume_by_map_jax(slices, rotation_matrices, image_shape, volume_shape, disc_type):
    volume_size = np.prod(volume_shape)
    order = decide_order(disc_type)
    if order == 3:
        from recovar.core import cubic_interpolation
        def f(volume_flat):
            coeffs = cubic_interpolation.calculate_spline_coefficients(volume_flat.reshape(volume_shape))
            return map_coordinates_on_slices(coeffs, rotation_matrices, image_shape, volume_shape, 3)
    else:
        f = lambda v: _slice_volume_by_map_jax(v, rotation_matrices, image_shape, volume_shape, disc_type)
    _, u = vjp(f, jnp.zeros(volume_size, dtype=slices.dtype))
    return u(slices)[0]


# ── Public API ────────────────────────────────────────────────────────

def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project volume to images via interpolation. CUDA when available, JAX fallback."""
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volume) and not _is_jvp_tracer(volume):
        from recovar.core.cuda_ops import cuda_slice_full
        return cuda_slice_full(volume, rotation_matrices, image_shape, volume_shape, order)
    return _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)


def batch_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a batch of volumes to images. Batched CUDA when available."""
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volumes):
        from recovar.cuda_backproject import batch_project
        return batch_project(volumes, rotation_matrices, image_shape, volume_shape, order=order)
    return jax.vmap(
        lambda v: slice_volume_by_map(v, rotation_matrices, image_shape, volume_shape, disc_type)
    )(volumes)


def slice_volume_by_map_to_half_image(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a full volume to rfft-packed half-spectrum images."""
    # CUDA half_image=True project kernel is not yet numerically validated;
    # project to full images first, then convert to half-spectrum.
    full = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    return fourier_transform_utils.full_image_to_half_image(full, image_shape)


def batch_slice_volume_by_map_to_half_image(volumes, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a batch of full volumes to half-spectrum images."""
    # CUDA half_image=True project kernel is not yet numerically validated;
    # project to full images first, then convert to half-spectrum.
    full = batch_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type)
    return jax.vmap(
        lambda f: fourier_transform_utils.full_image_to_half_image(f, image_shape)
    )(full)


def slice_volume_by_map_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a Hermitian half-volume to images.

    Always expands to full Hermitian format before slicing.  The dedicated
    CUDA half_volume=True kernel exists in cuda_ops but is not yet numerically
    validated (max_err ~75 vs reference); the expand-then-slice path gives
    correct results on all backends.
    """
    full_volume = fourier_transform_utils.half_volume_to_full_volume(half_volume, volume_shape)
    return slice_volume_by_map(full_volume, rotation_matrices, image_shape, volume_shape, disc_type)


def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type,
                                volume=None, half_image=False, half_volume=False):
    """Adjoint slice extraction (backprojection). CUDA when available, JAX VJP fallback.

    Parameters
    ----------
    half_image : if True, *slices* are rfft-packed half-spectrum images.
    half_volume : if True, output uses rfft-packed half-volume layout (CUDA or JAX VJP).
    volume : optional accumulator to add the result into.
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda():
        from recovar.cuda_backproject import backproject as cuda_backproject
        # CUDA half_image=True backproject kernel is not yet numerically validated
        # (max_err ~42 vs reference); expand to full spectrum before dispatching.
        if half_image:
            slices = fourier_transform_utils.half_image_to_full_image(slices, image_shape)
            half_image = False
        if not _is_complex(slices):
            slices = slices.astype(jnp.result_type(slices, jnp.complex64))
        if half_volume:
            # CUDA backproject(half_volume=True) is not numerically validated
            # (max_err ~75 vs reference).  Use full backproject + VJP of
            # half_volume_to_full_volume to get the correct adjoint.
            full_vol_zeros = jnp.zeros(int(np.prod(volume_shape)), dtype=slices.dtype)
            full_grad = cuda_backproject(
                full_vol_zeros, slices, rotation_matrices, image_shape, volume_shape, order=order
            )
            half_vol_size = int(np.prod(
                fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
            ))
            _, vjp_expand = vjp(
                lambda hv: fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
                jnp.zeros(half_vol_size, dtype=slices.dtype),
            )
            result = vjp_expand(full_grad)[0]
            return result if volume is None else result + volume
        if volume is None:
            volume = jnp.zeros(int(np.prod(volume_shape)), dtype=slices.dtype)
        return cuda_backproject(volume, slices, rotation_matrices, image_shape, volume_shape,
                                order=order)
    # JAX fallback
    if half_image:
        slices = fourier_transform_utils.half_image_to_full_image(slices, image_shape)
    if half_volume:
        if order > 1:
            raise NotImplementedError("half_volume=True with cubic requires CUDA kernels")
        half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(half_shape))
        f = lambda v: _slice_volume_by_map_jax(
            fourier_transform_utils.half_volume_to_full_volume(v, volume_shape),
            rotation_matrices, image_shape, volume_shape, disc_type,
        )
        _, u = vjp(f, jnp.zeros(vol_size, dtype=slices.dtype))
        result = u(slices)[0]
    else:
        result = _adjoint_slice_volume_by_map_jax(slices, rotation_matrices, image_shape, volume_shape, disc_type)
    return result if volume is None else result + volume


# ── Backward-compat thin wrappers ─────────────────────────────────────

def slice_volume_by_trilinear_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape):
    """Thin wrapper: slice from half-volume with linear interpolation."""
    return slice_volume_by_map_from_half_volume(
        half_volume, rotation_matrices, image_shape, volume_shape, disc_type="linear_interp"
    )


def batch_slice_volume_by_trilinear(volumes, rotation_matrices, image_shape, volume_shape):
    """Thin wrapper: batch slice with linear interpolation."""
    return batch_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, "linear_interp")


def batch_adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volumes=None):
    """Backproject per-volume images into a batch of volumes with shared rotations.

    Parameters
    ----------
    images : complex, shape ``(batch, n_images, n_pixels)``
    rotation_matrices : shape ``(n_images, 3, 3)``
    volumes : optional, shape ``(batch, vol_flat_size)``

    Returns
    -------
    complex, shape ``(batch, vol_flat_size)``
    """
    batch = images.shape[0]
    if volumes is None:
        volumes = jnp.zeros((batch, int(np.prod(volume_shape))), dtype=images.dtype)
    if _check_cuda() and _is_complex(images):
        from recovar.cuda_backproject import batch_backproject as cuda_batch_backproject
        return cuda_batch_backproject(volumes, images, rotation_matrices, image_shape, volume_shape, order=1)
    return jax.vmap(
        lambda v, im: adjoint_slice_volume_by_trilinear(im, rotation_matrices, image_shape, volume_shape, volume=v)
    )(volumes, images)


def adjoint_slice_volume_by_trilinear_from_weights(images, grid_vec_indices, weights, volume_shape, volume=None):
    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype=images.dtype)
    else:
        volume = jnp.asarray(volume)

    weights *= images.reshape(-1, 1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))
    return volume


# ── Cubic half-volume slicer ──────────────────────────────────────────────

def precompute_cubic_half_coefficients(volume, volume_shape):
    """Precompute cubic B-spline coefficients for a volume.

    Takes a full complex volume (centered-FFT convention) and fits the
    3-D cubic B-spline coefficient array of shape ``(N0+2, N1+2, N2+2)``.

    The "half" in the name refers to typical usage: callers hold the volume
    in half-spectrum (rfft3) format and expand it before calling this
    function.  Full coefficients are returned so that cubic interpolation
    evaluates correctly everywhere (natural-boundary-condition cubic splines
    are NOT Hermitian-symmetric — only the data values are).

    Parameters
    ----------
    volume : complex array, shape ``(N0, N1, N2)`` or ``(N0*N1*N2,)``
        Full centered-FFT volume.
    volume_shape : (N0, N1, N2)

    Returns
    -------
    complex array, shape ``(N0+2, N1+2, N2+2)``
        Full cubic B-spline coefficients, ready for
        :func:`slice_from_cubic_half_coefficients`.
    """
    from recovar.core import cubic_interpolation
    N0, N1, N2 = tuple(int(s) for s in volume_shape)
    volume_grid = jnp.asarray(volume).reshape(N0, N1, N2)
    return cubic_interpolation.calculate_spline_coefficients(volume_grid)


@functools.partial(jax.jit, static_argnums=[2, 3])
def _slice_from_half_cubic_coeffs_jax(coeffs, rotation_matrices, image_shape, volume_shape):
    """Sample rotated central slices from precomputed cubic coefficients.

    Parameters
    ----------
    coeffs : complex array, shape ``(N0+2, N1+2, N2+2)``
        As returned by :func:`precompute_cubic_half_coefficients`.
    rotation_matrices : float array, shape ``(n_images, 3, 3)``
    image_shape : (H, W)  [static]
    volume_shape : (N0, N1, N2)  [static]

    Returns
    -------
    complex array, shape ``(n_images, H*W)``
    """
    from recovar.core import cubic_interpolation

    coords, coords_og_shape = rotations_to_grid_point_coords(
        rotation_matrices, image_shape, volume_shape
    )
    vals = cubic_interpolation.map_coordinates_with_cubic_spline(
        coeffs, coords, mode="fill", cval=0.0
    )
    n_images = rotation_matrices.shape[0]
    H, W = image_shape
    return vals.reshape(n_images, H * W).astype(coeffs.dtype)


def slice_from_cubic_half_coefficients(coeffs, rotation_matrices, image_shape, volume_shape):
    """Project from precomputed cubic coefficients to images.

    Convenience wrapper around :func:`_slice_from_half_cubic_coeffs_jax`.

    Parameters
    ----------
    coeffs : complex array, shape ``(N0+2, N1+2, N2+2)``
        As returned by :func:`precompute_cubic_half_coefficients`.
    rotation_matrices : float array, shape ``(n_images, 3, 3)``
    image_shape : (H, W)
    volume_shape : (N0, N1, N2)

    Returns
    -------
    complex array, shape ``(n_images, H*W)``
    """
    return _slice_from_half_cubic_coeffs_jax(
        jnp.asarray(coeffs), rotation_matrices, image_shape, volume_shape
    )


__all__ = [
    "slice_volume_by_nearest",
    "batch_slice_volume_by_nearest",
    "decide_order",
    "slice_volume_by_map",
    "batch_slice_volume_by_map",
    "slice_volume_by_map_to_half_image",
    "batch_slice_volume_by_map_to_half_image",
    "slice_volume_by_map_from_half_volume",
    "slice_volume_by_trilinear_from_half_volume",
    "batch_slice_volume_by_trilinear",
    "adjoint_slice_volume_by_map",
    "map_coordinates_on_slices",
    "precompute_cubic_half_coefficients",
    "slice_from_cubic_half_coefficients",
]
