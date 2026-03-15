"""Fourier slice extraction and adjoint (backprojection) operations.

Dispatch rules:
  - GPU + order 0/1 → CUDA project + CUDA backproject
  - GPU + order 3   → CUDA project + JAX VJP backproject (cubic)
  - CPU + order 0/1 → RELION-style JAX (explicit scatter backproject)
  - CPU + order 3   → JAX ``map_coordinates`` + VJP backproject

Three core public functions handle all volume/image format combinations via
``half_volume`` and ``half_image`` parameters:
  - :func:`slice_volume`          (forward projection)
  - :func:`batch_slice_volume`    (batched forward)
  - :func:`adjoint_slice_volume`  (backprojection)

CUDA custom-VJP wrappers live in :mod:`recovar.core.cuda_ops`.
RELION-style JAX reference: :mod:`recovar.core.relion_interp`.
"""

import functools
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.core.fourier_transform_utils as ftu
from recovar.core.geometry import (
    rotations_to_grid_point_coords,
    _half_image_rotations_to_coords,
)

logger = logging.getLogger(__name__)


# ── Default max_r ────────────────────────────────────────────────────

# Sentinel: means "use the default max_r = image_shape[0]//2 - 1".
# Pass ``max_r=None`` explicitly to disable sphere clipping (old behavior).
_AUTO = object()


def _default_max_r(image_shape):
    """Default sphere-clipping radius: ``image_shape[0] // 2 - 1``.

    RELION clips at ``N // 2`` (Nyquist).  For recovar's full-image
    representation the N//2 frequency bin has no Hermitian conjugate
    partner, so we use ``N // 2 - 1`` to guarantee that half-image and
    full-image results are exactly equivalent and that the Nyquist edge
    is excluded.
    """
    return image_shape[0] // 2 - 1


def _resolve_max_r(max_r, image_shape):
    """Resolve max_r: _AUTO → default, None → None (no clip), number → number."""
    if max_r is _AUTO:
        return _default_max_r(image_shape)
    return max_r


def _cuda_max_r(max_r, image_shape, volume_shape):
    """Scale max_r from image coordinates to volume coordinates for CUDA.

    The CUDA kernel computes pixel frequencies in volume-space coordinates
    (scaled by ``upsampling = volume_shape[0] // image_shape[0]``), so
    ``max_r`` must be scaled to match.  The JAX ``relion_interp`` path
    uses image-space coordinates and needs no scaling.
    """
    if max_r is None:
        return None
    upsampling = volume_shape[0] // image_shape[0]
    return max_r * upsampling


# ── Dispatch ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _on_gpu():
    return jax.default_backend() == "gpu"


def _use_cuda(order):
    """Return True if CUDA project should be used for forward projection."""
    if order not in (0, 1, 3) or not _on_gpu():
        return False
    if os.environ.get("RECOVAR_DISABLE_CUDA", "0") == "1":
        return False
    from recovar.cuda_backproject import cuda_available
    if not cuda_available():
        raise RuntimeError(
            "CUDA backproject/project kernels required on GPU but not available. "
            "Rebuild the CUDA library or set RECOVAR_DISABLE_CUDA=1 to force JAX fallback."
        )
    return True


def _use_cuda_backproject(order):
    """Return True if CUDA backproject should be used (order 0 or 1 only).

    Cubic (order 3) backproject stays on JAX because the 64 atomic scatter
    ops per pixel make the CUDA kernel impractical.
    """
    if order > 1:
        return False
    return _use_cuda(order)


def _is_complex(arr):
    return jnp.issubdtype(arr.dtype, jnp.complexfloating)


def _normalize_volume(volume, volume_shape, half_volume):
    """Validate size and flatten volume to 1-D."""
    volume = jnp.asarray(volume)
    expected_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
    expected_size = int(np.prod(expected_shape))
    if volume.size != expected_size:
        raise ValueError(
            f"volume has {volume.size} elements, expected {expected_size} "
            f"for {'half' if half_volume else 'full'} volume_shape={volume_shape}"
        )
    return volume.ravel()


def _normalize_slices(slices, image_shape, half_image):
    """Validate and flatten slices to ``(n_images, n_pixels)``."""
    slices = jnp.asarray(slices)
    H, W = image_shape
    if half_image:
        grid_shape = (H, W // 2 + 1)
    else:
        grid_shape = (H, W)
    n_pixels = int(np.prod(grid_shape))

    if slices.ndim == 3:
        if tuple(slices.shape[-2:]) != grid_shape:
            raise ValueError(
                f"Expected slice grid shape (n_images, {grid_shape[0]}, {grid_shape[1]}), "
                f"got {tuple(slices.shape)}"
            )
        slices = slices.reshape(slices.shape[0], n_pixels)
    elif slices.ndim == 2:
        if slices.shape[-1] != n_pixels:
            raise ValueError(
                f"Expected slices with {n_pixels} pixels per image, got shape {tuple(slices.shape)}"
            )
    else:
        raise ValueError(
            f"Expected slices with shape (n_images, {n_pixels}) or "
            f"(n_images, {grid_shape[0]}, {grid_shape[1]}), got {tuple(slices.shape)}"
        )
    return slices


# ── Interpolation order ──────────────────────────────────────────────

def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    if disc_type == "nearest":
        return 0
    if disc_type == "cubic":
        return 3
    raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")


# ── JAX interpolation engine ────────────────────────────────────────

@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _jax_slice(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project volume to images using JAX map_coordinates.

    Expects *volume* as a flat 1-D complex array with
    ``prod(volume_shape)`` elements.
    """
    coords, og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    if order == 3:
        from recovar.core import cubic_interpolation
        return cubic_interpolation.map_coordinates_with_cubic_spline(
            volume.reshape(volume_shape), coords, mode="wrap", cval=0.0
        ).reshape(og_shape[:-1])
    return jax.scipy.ndimage.map_coordinates(
        volume.reshape(volume_shape), coords, order=order, mode="constant", cval=0.0,
    ).reshape(og_shape[:-1])


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _jax_slice_half_image(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project volume to half-images (rfft-packed) directly.

    Like ``_jax_slice`` but generates coordinates for only the H*(W//2+1)
    non-redundant pixels, avoiding ~50% of the interpolation work.
    """
    coords, og_shape = _half_image_rotations_to_coords(
        rotation_matrices, image_shape, volume_shape
    )
    if order == 3:
        from recovar.core import cubic_interpolation
        return cubic_interpolation.map_coordinates_with_cubic_spline(
            volume.reshape(volume_shape), coords, mode="wrap", cval=0.0
        ).reshape(og_shape[:-1])
    return jax.scipy.ndimage.map_coordinates(
        volume.reshape(volume_shape), coords, order=order, mode="constant", cval=0.0,
    ).reshape(og_shape[:-1])


# ── Public API ───────────────────────────────────────────────────────

def slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type,
                 half_volume=False, half_image=False, max_r=_AUTO):
    """Project volume to images via interpolation.

    Parameters
    ----------
    volume : array, flat ``(vol_size,)`` or grid-shaped.
        Real or complex.  Promoted to complex for the CUDA kernel.
    half_volume : if True, *volume* is rfft-packed ``(N0*N1*(N2//2+1),)``.
    half_image : if True, output images are rfft-packed ``(n, H*(W//2+1))``.
    max_r : sphere clipping radius.  Default (``_AUTO``) uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable clipping.
    """
    volume = _normalize_volume(volume, volume_shape, half_volume)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)

    if _use_cuda(order):
        # CUDA kernel requires complex input
        if not _is_complex(volume):
            volume = volume.astype(jnp.result_type(volume, jnp.complex64))
        from recovar.core.cuda_ops import cuda_project
        try:
            return cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                                order, half_volume, half_image,
                                _cuda_max_r(max_r, image_shape, volume_shape))
        except TypeError:
            pass  # JVP through custom_vjp not supported — fall through to JAX

    # JAX path — order 0/1: RELION-style trilinear/nearest
    if order <= 1:
        from recovar.core import relion_interp
        return relion_interp.project(
            volume, rotation_matrices, image_shape, volume_shape,
            order=order, half_volume=half_volume, half_image=half_image,
            max_r=max_r,
        )

    # Cubic: the 4×4×4 stencil reads from both Hermitian halves, so
    # half-volumes must be expanded.  (Linear/nearest handle half-volumes
    # natively via per-neighbor Hermitian fold.)
    if half_volume:
        volume = ftu.half_volume_to_full_volume(volume, volume_shape)
    if half_image:
        return _jax_slice_half_image(volume, rotation_matrices, image_shape, volume_shape, order)
    return _jax_slice(volume, rotation_matrices, image_shape, volume_shape, order)


def batch_slice_volume(volumes, rotation_matrices, image_shape, volume_shape, disc_type,
                       half_volume=False, half_image=False, max_r=_AUTO):
    """Project a batch of volumes to images.

    Parameters
    ----------
    volumes : array ``(batch, vol_size)``.
        Real or complex.  Promoted to complex for the CUDA kernel.
    half_volume : if True, *volumes* are rfft-packed half-volumes.
    half_image : if True, output images are rfft-packed.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    volumes = jnp.asarray(volumes)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)
    if _use_cuda(order):
        # CUDA kernel requires complex input
        if not _is_complex(volumes):
            volumes = volumes.astype(jnp.result_type(volumes, jnp.complex64))
        from recovar.cuda_backproject import batch_project
        return batch_project(volumes, rotation_matrices, image_shape, volume_shape,
                             order=order, half_volume=half_volume, half_image=half_image,
                             max_r=_cuda_max_r(max_r, image_shape, volume_shape))
    return jax.vmap(
        lambda v: slice_volume(v, rotation_matrices, image_shape, volume_shape,
                               disc_type, half_volume=half_volume, half_image=half_image,
                               max_r=max_r)
    )(volumes)


def adjoint_slice_volume(slices, rotation_matrices, image_shape, volume_shape, disc_type,
                         volume=None, half_image=False, half_volume=False, max_r=_AUTO):
    """Adjoint slice extraction (backprojection).

    Parameters
    ----------
    slices : array ``(n_images, n_pixels)``.
        Real or complex.  If *volume* is complex, slices are promoted
        to match (and vice versa).
    half_image : if True, *slices* are rfft-packed half-spectrum images.
    half_volume : if True, output uses rfft-packed half-volume layout.
    volume : optional accumulator to add the result into.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    slices = _normalize_slices(slices, image_shape, half_image)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)

    # CUDA backproject (order 0/1 only)
    if _use_cuda_backproject(order):
        from recovar.cuda_backproject import backproject
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
        if volume is None:
            volume = jnp.zeros(int(np.prod(vol_shape)), dtype=slices.dtype)
        # CUDA needs matching dtypes: if either is complex, promote both.
        out_dtype = jnp.result_type(slices, volume)
        slices = slices.astype(out_dtype)
        volume = volume.astype(out_dtype)
        return backproject(volume, slices, rotation_matrices, image_shape, volume_shape,
                           order=order, half_image=half_image, half_volume=half_volume,
                           max_r=_cuda_max_r(max_r, image_shape, volume_shape))

    # JAX order 0/1: RELION-style explicit scatter
    if order <= 1:
        from recovar.core import relion_interp
        result = relion_interp.backproject(
            slices, rotation_matrices, image_shape, volume_shape,
            order=order, half_volume=half_volume, half_image=half_image,
            max_r=max_r,
        )
        return result if volume is None else result + volume

    # Cubic: VJP-based backprojection (differentiates through coeff computation)
    result = _vjp_adjoint_cubic(slices, rotation_matrices, image_shape, volume_shape,
                                half_image=half_image, half_volume=half_volume)
    return result if volume is None else result + volume


def _vjp_adjoint_cubic(slices, rotation_matrices, image_shape, volume_shape,
                       half_image=False, half_volume=False):
    """VJP-based cubic backprojection: gradient of original volume through
    coefficient computation + slice.
    """
    slices = _normalize_slices(slices, image_shape, half_image)
    _slice_fn = _jax_slice_half_image if half_image else _jax_slice

    # Cubic's 4×4×4 stencil reads from both Hermitian halves, so the
    # forward function must expand half→full before interpolating.
    if half_volume:
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(vol_shape))
        def f(v):
            full_v = ftu.half_volume_to_full_volume(v, volume_shape)
            from recovar.core import cubic_interpolation
            coeffs = cubic_interpolation.calculate_spline_coefficients(full_v.reshape(volume_shape))
            return _slice_fn(coeffs, rotation_matrices, image_shape, volume_shape, 3)
    else:
        vol_size = int(np.prod(volume_shape))
        def f(v):
            from recovar.core import cubic_interpolation
            coeffs = cubic_interpolation.calculate_spline_coefficients(v.reshape(volume_shape))
            return _slice_fn(coeffs, rotation_matrices, image_shape, volume_shape, 3)

    _, u = vjp(f, jnp.zeros(vol_size, dtype=slices.dtype))
    return u(slices)[0]


def _jax_adjoint_slice_from_coefficients(slices, rotation_matrices, image_shape, volume_shape,
                                          half_volume=False, half_image=False, max_r=None):
    """VJP of ``_jax_slice`` w.r.t. already-computed spline coefficients.

    Unlike :func:`_vjp_adjoint_cubic` (which differentiates *through*
    coefficient computation), this computes the gradient w.r.t. the
    coefficients directly.

    .. note:: This function IS used — it is called from
       ``cuda_ops._cuda_project_bwd`` for the order=3 (cubic) CUDA
       backward pass.  Do not remove.
    """
    slices = _normalize_slices(slices, image_shape, half_image)
    _slice_fn = _jax_slice_half_image if half_image else _jax_slice

    if half_volume:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        half_size = int(np.prod(half_shape))
        def f(half_coeffs_flat):
            full_coeffs = ftu.half_volume_to_full_volume(
                half_coeffs_flat.reshape(half_shape), volume_shape)
            return _slice_fn(full_coeffs, rotation_matrices, image_shape, volume_shape, 3)
        _, u = vjp(f, jnp.zeros(half_size, dtype=slices.dtype))
        return u(slices)[0].reshape(half_shape)
    else:
        vol_size = int(np.prod(volume_shape))
        def f(coeffs_flat):
            return _slice_fn(coeffs_flat, rotation_matrices, image_shape, volume_shape, 3)
        _, u = vjp(f, jnp.zeros(vol_size, dtype=slices.dtype))
        return u(slices)[0].reshape(volume_shape)


def batch_adjoint_slice_volume(slices, rotation_matrices, image_shape, volume_shape, disc_type,
                               volumes=None, half_image=False, half_volume=False, max_r=_AUTO):
    """Batch backprojection: per-volume image sets to batch of volumes.

    Parameters
    ----------
    slices : shape ``(batch, n_images, n_pixels)``.
        Real or complex.  Promoted to match *volumes* dtype (and vice versa).
    rotation_matrices : shape ``(n_images, 3, 3)`` — shared across batch.
    volumes : optional ``(batch, vol_flat_size)`` accumulators.
    half_image, half_volume : same semantics as ``adjoint_slice_volume``.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    slices = jnp.asarray(slices)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)
    vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
    vol_flat = int(np.prod(vol_shape))
    batch = slices.shape[0]
    if volumes is None:
        volumes = jnp.zeros((batch, vol_flat), dtype=slices.dtype)
    if _use_cuda_backproject(order):
        from recovar.cuda_backproject import batch_backproject
        # CUDA needs matching dtypes: if either is complex, promote both.
        out_dtype = jnp.result_type(slices, volumes)
        slices = slices.astype(out_dtype)
        volumes = volumes.astype(out_dtype)
        return batch_backproject(volumes, slices, rotation_matrices, image_shape, volume_shape,
                                order=order, half_volume=half_volume, half_image=half_image,
                                max_r=_cuda_max_r(max_r, image_shape, volume_shape))
    # JAX fallback: vmap single adjoint
    return jax.vmap(
        lambda sl, vol: adjoint_slice_volume(sl, rotation_matrices, image_shape, volume_shape,
                                             disc_type, volume=vol, half_image=half_image,
                                             half_volume=half_volume, max_r=max_r)
    )(slices, volumes)


# ── Cubic coefficient precompute + direct slicer ─────────────────────

def precompute_cubic_coefficients(volume, volume_shape):
    """Precompute periodic cubic B-spline coefficients for a full volume.

    Uses periodic boundary conditions (FFT-based circulant solve).
    Output has the same shape as input (no boundary padding).
    """
    from recovar.core import cubic_interpolation
    N0, N1, N2 = tuple(int(s) for s in volume_shape)
    volume_grid = jnp.asarray(volume).reshape(N0, N1, N2)
    return cubic_interpolation.calculate_spline_coefficients(volume_grid)


def precompute_cubic_coefficients_half(volume, volume_shape):
    """Precompute periodic cubic coefficients, stored as half-volume.

    Since periodic coefficients preserve Hermitian symmetry, this is lossless.
    """
    coeffs = precompute_cubic_coefficients(volume, volume_shape)
    return ftu.full_volume_to_half_volume(coeffs, volume_shape).reshape(-1)


def slice_from_cubic_coefficients(coeffs, rotation_matrices, image_shape, volume_shape,
                                   half_image=False):
    """Project from precomputed periodic cubic coefficients to images.

    Parameters
    ----------
    coeffs : precomputed spline coefficients from :func:`precompute_cubic_coefficients`.
        Can be full-volume (N0,N1,N2) or half-volume (N0*N1*(N2//2+1),) layout.
    half_image : if True, output images are rfft-packed ``(n, H*(W//2+1))``.
    """
    from recovar.core import cubic_interpolation
    coeffs = jnp.asarray(coeffs)

    # Detect half-volume layout by size
    half_vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    full_vol_size = int(np.prod(volume_shape))
    is_half = (coeffs.size == half_vol_size and coeffs.size != full_vol_size)

    # Cubic's 4×4×4 stencil reads from both Hermitian halves, so
    # half-volume coefficients must be expanded to full before interpolation.
    if is_half:
        coeffs = ftu.half_volume_to_full_volume(coeffs, volume_shape).reshape(volume_shape)
    else:
        coeffs = coeffs.reshape(volume_shape)

    # Choose coordinate grid
    if half_image:
        coords, og_shape = _half_image_rotations_to_coords(
            rotation_matrices, image_shape, volume_shape
        )
        n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    else:
        coords, og_shape = rotations_to_grid_point_coords(
            rotation_matrices, image_shape, volume_shape
        )
        n_pixels = image_shape[0] * image_shape[1]

    vals = cubic_interpolation.map_coordinates_with_cubic_spline(
        coeffs, coords, mode="wrap", cval=0.0
    )
    n_images = rotation_matrices.shape[0]
    return vals.reshape(n_images, n_pixels)


__all__ = [
    "decide_order",
    "slice_volume",
    "batch_slice_volume",
    "adjoint_slice_volume",
    "batch_adjoint_slice_volume",
    "precompute_cubic_coefficients",
    "precompute_cubic_coefficients_half",
    "slice_from_cubic_coefficients",
    "_AUTO",
    "_default_max_r",
]
