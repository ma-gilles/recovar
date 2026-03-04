"""Fourier slice extraction and adjoint (backprojection) operations.

Dispatch rules:
  - GPU + order <= 1 → CUDA kernels (mandatory; error if unavailable)
  - CPU or cubic     → JAX ``map_coordinates`` fallback

Three core public functions handle all volume/image format combinations via
``half_volume`` and ``half_image`` parameters:
  - :func:`slice_volume`          (forward projection)
  - :func:`batch_slice_volume`    (batched forward)
  - :func:`adjoint_slice_volume`  (backprojection)

CUDA custom-VJP wrappers live in :mod:`recovar.core.cuda_ops`.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.core.fourier_transform_utils as ftu
from recovar.core.geometry import rotations_to_grid_point_coords

logger = logging.getLogger(__name__)


# ── Dispatch ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _on_gpu():
    return jax.default_backend() == "gpu"


def _use_cuda(order):
    """Return True if CUDA should be used.  Error if on GPU but CUDA unavailable."""
    if order > 1 or not _on_gpu():
        return False
    from recovar.cuda_backproject import cuda_available
    if not cuda_available():
        raise RuntimeError(
            "CUDA backproject/project kernels required on GPU but not available. "
            "Rebuild the CUDA library or set RECOVAR_DISABLE_CUDA=1 to force JAX fallback."
        )
    return True


def _is_complex(arr):
    return jnp.issubdtype(arr.dtype, jnp.complexfloating)


# ── Interpolation order ──────────────────────────────────────────────

def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    if disc_type == "nearest":
        return 0
    if disc_type == "cubic":
        return 3
    raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")


# ── JAX fallback engine (CPU or cubic) ───────────────────────────────

@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _jax_slice(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project volume to images using JAX map_coordinates."""
    coords, og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    if order == 3:
        from recovar.core import cubic_interpolation
        return cubic_interpolation.map_coordinates_with_cubic_spline(
            volume, coords, mode="fill", cval=0.0
        ).reshape(og_shape[:-1]).astype(volume.dtype)
    return jax.scipy.ndimage.map_coordinates(
        volume.reshape(volume_shape), coords, order=order, mode="constant", cval=0.0,
    ).reshape(og_shape[:-1]).astype(volume.dtype)


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _jax_adjoint_slice(slices, rotation_matrices, image_shape, volume_shape, order):
    """Adjoint of _jax_slice via VJP."""
    volume_size = np.prod(volume_shape)
    if order == 3:
        from recovar.core import cubic_interpolation
        def f(volume_flat):
            coeffs = cubic_interpolation.calculate_spline_coefficients(volume_flat.reshape(volume_shape))
            return _jax_slice(coeffs, rotation_matrices, image_shape, volume_shape, 3)
    else:
        f = lambda v: _jax_slice(v, rotation_matrices, image_shape, volume_shape, order)
    _, u = vjp(f, jnp.zeros(volume_size, dtype=slices.dtype))
    return u(slices)[0]


# ── Nearest-neighbour gather (pre-computed indices) ──────────────────

@jax.jit
def slice_volume_by_nearest(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid]


batch_slice_volume_by_nearest = jax.vmap(slice_volume_by_nearest, (None, 0))


# ── Public API ───────────────────────────────────────────────────────

def slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type,
                 half_volume=False, half_image=False):
    """Project volume to images via interpolation.

    Parameters
    ----------
    half_volume : if True, *volume* is rfft-packed ``(N0*N1*(N2//2+1),)``.
    half_image : if True, output images are rfft-packed ``(n, H*(W//2+1))``.
    """
    order = decide_order(disc_type)
    if _use_cuda(order):
        if not _is_complex(volume):
            volume = volume.astype(jnp.result_type(volume, jnp.complex64))
        from recovar.core.cuda_ops import cuda_project
        try:
            return cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                                order, half_volume, half_image)
        except TypeError:
            pass  # JVP through custom_vjp — fall through to JAX
    # JAX fallback (CPU, cubic, or JVP context)
    if half_volume:
        volume = ftu.half_volume_to_full_volume(volume, volume_shape)
    result = _jax_slice(volume, rotation_matrices, image_shape, volume_shape, order)
    if half_image:
        result = ftu.full_image_to_half_image(result, image_shape)
    return result


def batch_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, disc_type,
                       half_volume=False, half_image=False):
    """Project a batch of volumes to images.

    Parameters
    ----------
    half_volume : if True, *volumes* are rfft-packed half-volumes.
    half_image : if True, output images are rfft-packed.
    """
    order = decide_order(disc_type)
    if _use_cuda(order):
        if not _is_complex(volumes):
            volumes = volumes.astype(jnp.result_type(volumes, jnp.complex64))
        from recovar.cuda_backproject import batch_project
        return batch_project(volumes, rotation_matrices, image_shape, volume_shape,
                             order=order, half_volume=half_volume, half_image=half_image)
    return jax.vmap(
        lambda v: slice_volume(v, rotation_matrices, image_shape, volume_shape,
                               disc_type, half_volume=half_volume, half_image=half_image)
    )(volumes)


def adjoint_slice_volume(slices, rotation_matrices, image_shape, volume_shape, disc_type,
                         volume=None, half_image=False, half_volume=False):
    """Adjoint slice extraction (backprojection).

    Parameters
    ----------
    half_image : if True, *slices* are rfft-packed half-spectrum images.
    half_volume : if True, output uses rfft-packed half-volume layout.
    volume : optional accumulator to add the result into.
    """
    order = decide_order(disc_type)
    if _use_cuda(order):
        from recovar.cuda_backproject import backproject
        # Real inputs stay real for 2x efficiency; promote only if accumulator is complex.
        if not _is_complex(slices) and volume is not None and _is_complex(volume):
            slices = slices.astype(jnp.result_type(slices, jnp.complex64))
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
        if volume is None:
            volume = jnp.zeros(int(np.prod(vol_shape)), dtype=slices.dtype)
        return backproject(volume, slices, rotation_matrices, image_shape, volume_shape,
                           order=order, half_image=half_image, half_volume=half_volume)
    # JAX fallback (CPU or cubic)
    if half_image:
        slices = ftu.half_image_to_full_image(slices, image_shape)
    if half_volume:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        f = lambda v: _jax_slice(
            ftu.half_volume_to_full_volume(v, volume_shape),
            rotation_matrices, image_shape, volume_shape, order,
        )
        _, u = vjp(f, jnp.zeros(int(np.prod(half_shape)), dtype=slices.dtype))
        result = u(slices)[0]
    else:
        result = _jax_adjoint_slice(slices, rotation_matrices, image_shape, volume_shape, order)
    return result if volume is None else result + volume


# ── Standalone utilities ─────────────────────────────────────────────

def adjoint_slice_volume_by_trilinear_from_weights(images, grid_vec_indices, weights, volume_shape, volume=None):
    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype=images.dtype)
    else:
        volume = jnp.asarray(volume)

    weights *= images.reshape(-1, 1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))
    return volume


# ── Cubic half-volume slicer ─────────────────────────────────────────

def precompute_cubic_half_coefficients(volume, volume_shape):
    """Precompute cubic B-spline coefficients for a volume.

    Takes a full complex volume (centered-FFT convention) and fits the
    3-D cubic B-spline coefficient array of shape ``(N0+2, N1+2, N2+2)``.

    Parameters
    ----------
    volume : complex array, shape ``(N0, N1, N2)`` or ``(N0*N1*N2,)``
        Full centered-FFT volume.
    volume_shape : (N0, N1, N2)

    Returns
    -------
    complex array, shape ``(N0+2, N1+2, N2+2)``
    """
    from recovar.core import cubic_interpolation
    N0, N1, N2 = tuple(int(s) for s in volume_shape)
    volume_grid = jnp.asarray(volume).reshape(N0, N1, N2)
    return cubic_interpolation.calculate_spline_coefficients(volume_grid)


@functools.partial(jax.jit, static_argnums=[2, 3])
def _slice_from_half_cubic_coeffs_jax(coeffs, rotation_matrices, image_shape, volume_shape):
    """Sample rotated central slices from precomputed cubic coefficients."""
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
    """Project from precomputed cubic coefficients to images."""
    return _slice_from_half_cubic_coeffs_jax(
        jnp.asarray(coeffs), rotation_matrices, image_shape, volume_shape
    )


__all__ = [
    "decide_order",
    "slice_volume_by_nearest",
    "batch_slice_volume_by_nearest",
    "slice_volume",
    "batch_slice_volume",
    "adjoint_slice_volume",
    "adjoint_slice_volume_by_trilinear_from_weights",
    "precompute_cubic_half_coefficients",
    "slice_from_cubic_half_coefficients",
]
