"""Fourier slice extraction and adjoint (backprojection) operations.

Dispatch rules:
  - GPU + order 0/1 → CUDA project + CUDA backproject
  - GPU + order 3   → CUDA project + JAX VJP backproject (cubic)
  - CPU + order 0/1 → RELION-style JAX (explicit scatter backproject)
  - CPU + order 3   → JAX ``map_coordinates`` + VJP backproject

Three core public functions handle all volume/image format combinations via
explicit :class:`Volume` / :class:`CubicVolume` inputs plus the optional
``half_image`` parameter:
  - :func:`slice_volume`          (forward projection)
  - :func:`batch_slice_volume`    (batched forward)
  - :func:`adjoint_slice_volume`  (backprojection)

CUDA custom-VJP wrappers live in :mod:`recovar.core.cuda_ops`.
RELION-style JAX reference: :mod:`recovar.core.relion_interp`.
"""

import functools
import logging
import os

import equinox as eqx
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


class Volume(eqx.Module):
    """Thin wrapper for non-cubic projection inputs."""

    values: jax.Array
    disc_type: str = eqx.field(static=True, default="linear_interp")
    half_volume: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if self.disc_type == "cubic":
            raise ValueError(
                "Use to_cubic(...) for raw cubic inputs or CubicVolume.from_coeffs(...) for precomputed coefficients"
            )
        if self.disc_type not in {"nearest", "linear_interp"}:
            raise ValueError(
                "Volume disc_type must be 'nearest' or 'linear_interp'; use to_cubic(...) for cubic inputs"
            )

    @property
    def array(self):
        return self.values

    def replace_array(self, array):
        return type(self)(
            values=jnp.asarray(array),
            disc_type=self.disc_type,
            half_volume=self.half_volume,
        )


_CUBIC_VOLUME_INIT_TOKEN = object()


class CubicVolume(eqx.Module):
    """Thin wrapper for precomputed cubic spline coefficients.

    Use :func:`to_cubic` for raw sampled Fourier volumes. Construct
    :class:`CubicVolume` via :meth:`from_coeffs` when coefficients are already available.
    """

    coeffs: jax.Array
    half_volume: bool = eqx.field(static=True, default=False)
    disc_type: str = eqx.field(static=True, default="cubic")
    _init_token: object | None = eqx.field(static=True, default=None, repr=False)

    def __check_init__(self):
        if self.disc_type != "cubic":
            raise ValueError("CubicVolume must use disc_type='cubic'")
        if self._init_token is not _CUBIC_VOLUME_INIT_TOKEN:
            raise TypeError("Use CubicVolume.from_coeffs(...) for precomputed cubic coefficients")

    @property
    def array(self):
        return self.coeffs

    @classmethod
    def from_coeffs(cls, coeffs, *, half_volume=False):
        return cls(
            coeffs=jnp.asarray(coeffs),
            half_volume=half_volume,
            _init_token=_CUBIC_VOLUME_INIT_TOKEN,
        )

    def replace_array(self, array):
        return type(self).from_coeffs(
            array,
            half_volume=self.half_volume,
        )


_VOLUME_TYPES = (Volume, CubicVolume)


def _require_volume_object(volume, *, function_name):
    if isinstance(volume, _VOLUME_TYPES):
        return volume
    raise TypeError(
        f"{function_name} requires a Volume or CubicVolume; wrap sampled arrays with Volume(...) "
        "and raw cubic arrays with to_cubic(...). Use CubicVolume.from_coeffs(...) only for precomputed coefficients."
    )




def _wrap_projection_array(array, *, disc_type, half_volume=False):
    if disc_type == "cubic":
        return CubicVolume.from_coeffs(array, half_volume=half_volume)
    return Volume(array, disc_type=disc_type, half_volume=half_volume)


def _zeros_projection_volume(volume_shape, *, disc_type, dtype, half_volume=False, batch_ndim=None):
    expected_shape = _expected_volume_shape(volume_shape, half_volume)
    flat = int(np.prod(expected_shape))
    shape = (flat,) if batch_ndim is None else (*batch_ndim, flat)
    return _wrap_projection_array(jnp.zeros(shape, dtype=dtype), disc_type=disc_type, half_volume=half_volume)


def _expected_volume_shape(volume_shape, half_volume):
    return ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else tuple(int(s) for s in volume_shape)


def _match_volume_array_layout(volume, expected_shape):
    expected_rank = len(expected_shape)
    expected_size = int(np.prod(expected_shape))

    if volume.ndim == 1:
        if volume.size != expected_size:
            raise ValueError(f"Expected flat volume with {expected_size} elements, got shape {tuple(volume.shape)}")
        return "single_flat"

    if tuple(volume.shape) == expected_shape:
        return "single_grid"

    if volume.ndim == 2:
        if volume.shape[-1] != expected_size:
            raise ValueError(
                f"Expected batched flat volumes with trailing size {expected_size}, got shape {tuple(volume.shape)}"
            )
        return "batch_flat"

    if volume.ndim == expected_rank + 1 and tuple(volume.shape[-expected_rank:]) == expected_shape:
        return "batch_grid"

    raise ValueError(
        f"Unsupported volume shape {tuple(volume.shape)} for expected per-volume shape {expected_shape}"
    )


def _volume_array_layout(volume, volume_shape, half_volume=None):
    volume = jnp.asarray(volume)
    if half_volume is not None:
        half_volume = bool(half_volume)
        expected_shape = _expected_volume_shape(volume_shape, half_volume)
        return _match_volume_array_layout(volume, expected_shape), half_volume

    matches = []
    for candidate in (False, True):
        expected_shape = _expected_volume_shape(volume_shape, candidate)
        try:
            layout = _match_volume_array_layout(volume, expected_shape)
        except ValueError:
            continue
        matches.append((layout, candidate))

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(
            f"Could not infer volume layout from shape {tuple(volume.shape)} for volume_shape={tuple(volume_shape)}"
        )
    raise ValueError(
        f"Ambiguous volume layout for shape {tuple(volume.shape)} and volume_shape={tuple(volume_shape)}; "
        "pass a Volume with explicit half_volume metadata"
    )


def _projection_volume(volume, volume_shape, *, function_name):
    wrapped_volume = _require_volume_object(volume, function_name=function_name)
    _match_volume_array_layout(
        jnp.asarray(wrapped_volume.array),
        _expected_volume_shape(volume_shape, wrapped_volume.half_volume),
    )
    return wrapped_volume


def _calculate_spline_coefficients(volume):
    from recovar.core import cubic_interpolation

    return cubic_interpolation.calculate_spline_coefficients(volume)


_batch_calculate_spline_coefficients = jax.vmap(_calculate_spline_coefficients, in_axes=0, out_axes=0)


def _precompute_cubic_single(volume, volume_shape, half_volume):
    if half_volume:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        full_volume = ftu.half_volume_to_full_volume(volume.reshape(half_shape), volume_shape)
    else:
        full_volume = volume.reshape(volume_shape)

    coeffs = _calculate_spline_coefficients(full_volume.reshape(volume_shape))
    if half_volume:
        coeffs = ftu.full_volume_to_half_volume(coeffs, volume_shape)
    return coeffs.reshape(-1)


def _precompute_cubic_batch(volumes, volume_shape, half_volume):
    volumes = jnp.asarray(volumes)
    if half_volume:
        half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
        full_volumes = jax.vmap(
            lambda hv: ftu.half_volume_to_full_volume(hv.reshape(half_shape), volume_shape),
            in_axes=0,
            out_axes=0,
        )(volumes.reshape((-1, *half_shape)))
    else:
        full_volumes = volumes.reshape((-1, *volume_shape))

    coeffs = _batch_calculate_spline_coefficients(full_volumes)
    if half_volume:
        coeffs = jax.vmap(
            lambda c: ftu.full_volume_to_half_volume(c, volume_shape),
            in_axes=0,
            out_axes=0,
        )(coeffs)
    return coeffs.reshape(volumes.shape[0], -1)


def _reshape_cubic_coeffs(coeffs, layout, expected_shape):
    if layout == "single_flat":
        return coeffs.reshape(-1)
    if layout == "single_grid":
        return coeffs.reshape(expected_shape)
    if layout == "batch_flat":
        return coeffs.reshape(coeffs.shape[0], -1)
    if layout == "batch_grid":
        return coeffs.reshape(coeffs.shape[0], *expected_shape)
    raise ValueError(f"Unsupported layout {layout!r}")


def _resolve_half_image(half_image, half_volume):
    if half_image is None:
        return bool(half_volume)
    return bool(half_image)


def to_cubic(volume, volume_shape, half_volume=None):
    """Convert raw volumes into cubic spline-coefficient representation.

    Parameters
    ----------
    volume : array, Volume, or CubicVolume
        Raw full/half sampled values, optionally batched. If already a
        :class:`CubicVolume`, it is returned unchanged after layout validation.
        Use ``CubicVolume.from_coeffs(...)`` only for arrays that already store
        spline coefficients.
    volume_shape : tuple[int, int, int]
        Full-grid volume shape.
    half_volume : bool | None
        Whether *volume* uses the Hermitian-packed half-volume layout. If
        ``volume`` is already a :class:`CubicVolume`, ``None`` means "use the
        stored layout".
    """
    if isinstance(volume, CubicVolume):
        if half_volume is not None and bool(half_volume) != volume.half_volume:
            raise ValueError(
                f"half_volume={half_volume!r} does not match CubicVolume.half_volume={volume.half_volume!r}"
            )
        return volume
    if isinstance(volume, Volume):
        if half_volume is not None and bool(half_volume) != volume.half_volume:
            raise ValueError(
                f"half_volume={half_volume!r} does not match Volume.half_volume={volume.half_volume!r}"
            )
        values = volume.array
        half_volume = volume.half_volume
        layout = _match_volume_array_layout(values, _expected_volume_shape(volume_shape, half_volume))
    else:
        values = volume
        layout, half_volume = _volume_array_layout(values, volume_shape, half_volume=half_volume)

    expected_shape = _expected_volume_shape(volume_shape, half_volume)
    values = jnp.asarray(values)

    if layout == "single_flat":
        coeffs = _precompute_cubic_single(values, volume_shape, half_volume)
    elif layout == "single_grid":
        coeffs = _precompute_cubic_single(values.reshape(-1), volume_shape, half_volume)
    elif layout == "batch_flat":
        coeffs = _precompute_cubic_batch(values, volume_shape, half_volume)
    else:
        coeffs = _precompute_cubic_batch(values.reshape(values.shape[0], -1), volume_shape, half_volume)

    return CubicVolume.from_coeffs(
        _reshape_cubic_coeffs(coeffs, layout, expected_shape),
        half_volume=half_volume,
    )


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
                f"Expected slice grid shape (n_images, {grid_shape[0]}, {grid_shape[1]}), got {tuple(slices.shape)}"
            )
        slices = slices.reshape(slices.shape[0], n_pixels)
    elif slices.ndim == 2:
        if slices.shape[-1] != n_pixels:
            raise ValueError(f"Expected slices with {n_pixels} pixels per image, got shape {tuple(slices.shape)}")
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
        volume.reshape(volume_shape),
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    ).reshape(og_shape[:-1])


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _jax_slice_half_image(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project volume to half-images (rfft-packed) directly.

    Like ``_jax_slice`` but generates coordinates for only the H*(W//2+1)
    non-redundant pixels, avoiding ~50% of the interpolation work.
    """
    coords, og_shape = _half_image_rotations_to_coords(rotation_matrices, image_shape, volume_shape)
    if order == 3:
        from recovar.core import cubic_interpolation

        return cubic_interpolation.map_coordinates_with_cubic_spline(
            volume.reshape(volume_shape), coords, mode="wrap", cval=0.0
        ).reshape(og_shape[:-1])
    return jax.scipy.ndimage.map_coordinates(
        volume.reshape(volume_shape),
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    ).reshape(og_shape[:-1])


# ── Public API ───────────────────────────────────────────────────────


def slice_volume(
    volume,
    rotation_matrices,
    image_shape,
    volume_shape,
    half_image=None,
    max_r=_AUTO,
):
    """Project volume to images via interpolation.

    Parameters
    ----------
    volume : Volume or CubicVolume.
        Projection input. Real inputs are promoted to complex for the CUDA kernel.
    half_image : bool | None
        If True, output images are rfft-packed ``(n, H*(W//2+1))``. ``None``
        defaults to ``volume.half_volume``.
    max_r : sphere clipping radius.  Default (``_AUTO``) uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable clipping.
    """
    wrapped_volume = _projection_volume(
        volume,
        volume_shape,
        function_name="slice_volume",
    )
    disc_type = wrapped_volume.disc_type
    half_volume = wrapped_volume.half_volume
    half_image = _resolve_half_image(half_image, half_volume)
    volume = _normalize_volume(wrapped_volume.array, volume_shape, half_volume)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)

    if _use_cuda(order):
        # CUDA kernel requires complex input
        if not _is_complex(volume):
            volume = volume.astype(jnp.result_type(volume, jnp.complex64))
        from recovar.core.cuda_ops import cuda_project

        try:
            return cuda_project(
                volume,
                rotation_matrices,
                image_shape,
                volume_shape,
                order,
                half_volume,
                half_image,
                _cuda_max_r(max_r, image_shape, volume_shape),
            )
        except TypeError:
            pass  # JVP through custom_vjp not supported — fall through to JAX

    # JAX path — order 0/1: RELION-style trilinear/nearest
    if order <= 1:
        from recovar.core import relion_interp

        return relion_interp.project(
            volume,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
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


def batch_slice_volume(
    volumes,
    rotation_matrices,
    image_shape,
    volume_shape,
    half_image=None,
    max_r=_AUTO,
):
    """Project a batch of volumes to images.

    Parameters
    ----------
    volumes : Volume or CubicVolume.
        Batched projection inputs. Real inputs are promoted to complex for the CUDA kernel.
    half_image : bool | None
        If True, output images are rfft-packed. ``None`` defaults to
        ``volumes.half_volume``.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    wrapped_volume = _projection_volume(
        volumes,
        volume_shape,
        function_name="batch_slice_volume",
    )
    disc_type = wrapped_volume.disc_type
    half_volume = wrapped_volume.half_volume
    half_image = _resolve_half_image(half_image, half_volume)
    volumes = jnp.asarray(wrapped_volume.array)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)
    if _use_cuda(order):
        # CUDA kernel requires complex input
        if not _is_complex(volumes):
            volumes = volumes.astype(jnp.result_type(volumes, jnp.complex64))
        from recovar.cuda_backproject import batch_project

        return batch_project(
            volumes,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
            max_r=_cuda_max_r(max_r, image_shape, volume_shape),
        )
    return jax.vmap(
        lambda v: slice_volume(
            wrapped_volume.replace_array(v),
            rotation_matrices,
            image_shape,
            volume_shape,
            half_image=half_image,
            max_r=max_r,
        )
    )(volumes)


def adjoint_slice_volume(
    slices,
    rotation_matrices,
    image_shape,
    volume_shape,
    *,
    like,
    half_image=None,
    max_r=_AUTO,
):
    """Adjoint slice extraction (backprojection).

    Parameters
    ----------
    slices : array ``(n_images, n_pixels)``.
        Real or complex. If ``like.array`` is complex, slices are promoted
        to match (and vice versa).
    like : Volume or CubicVolume
        Template/accumulator that fixes the output representation. The returned
        array lives in the same representation as ``like`` and is added into
        its stored array.
    half_image : bool | None
        If True, *slices* are rfft-packed half-spectrum images. ``None``
        defaults to ``like.half_volume``.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    wrapped_volume = _projection_volume(like, volume_shape, function_name="adjoint_slice_volume")
    disc_type = wrapped_volume.disc_type
    half_volume = wrapped_volume.half_volume
    seed_array = jnp.asarray(wrapped_volume.array)
    output_shape = seed_array.shape
    volume = _normalize_volume(seed_array, volume_shape, half_volume)
    half_image = _resolve_half_image(half_image, half_volume)
    slices = _normalize_slices(slices, image_shape, half_image)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)

    # CUDA backproject (order 0/1 only)
    if _use_cuda_backproject(order):
        from recovar.cuda_backproject import backproject

        # CUDA needs matching dtypes: if either is complex, promote both.
        out_dtype = jnp.result_type(slices, volume)
        slices = slices.astype(out_dtype)
        volume = volume.astype(out_dtype)
        result = backproject(
            volume,
            slices,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_image=half_image,
            half_volume=half_volume,
            max_r=_cuda_max_r(max_r, image_shape, volume_shape),
        )
        return jnp.asarray(result).reshape(output_shape)

    # JAX order 0/1: RELION-style explicit scatter
    if order <= 1:
        from recovar.core import relion_interp

        result = relion_interp.backproject(
            slices,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
        return jnp.asarray(result).reshape(output_shape) + seed_array

    # Cubic adjoints are representation-preserving: CubicVolume accumulates in
    # coefficient space.
    result = _jax_adjoint_slice_from_coefficients(
        slices,
        rotation_matrices,
        image_shape,
        volume_shape,
        half_volume=half_volume,
        half_image=half_image,
        max_r=max_r,
    )
    return jnp.asarray(result).reshape(output_shape) + seed_array


def _vjp_adjoint_cubic(slices, rotation_matrices, image_shape, volume_shape, half_image=False, half_volume=False):
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


def _jax_adjoint_slice_from_coefficients(
    slices, rotation_matrices, image_shape, volume_shape, half_volume=False, half_image=False, max_r=None
):
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
            full_coeffs = ftu.half_volume_to_full_volume(half_coeffs_flat.reshape(half_shape), volume_shape)
            return _slice_fn(full_coeffs, rotation_matrices, image_shape, volume_shape, 3)

        _, u = vjp(f, jnp.zeros(half_size, dtype=slices.dtype))
        return u(slices)[0].reshape(half_shape)
    else:
        vol_size = int(np.prod(volume_shape))

        def f(coeffs_flat):
            return _slice_fn(coeffs_flat, rotation_matrices, image_shape, volume_shape, 3)

        _, u = vjp(f, jnp.zeros(vol_size, dtype=slices.dtype))
        return u(slices)[0].reshape(volume_shape)


def batch_adjoint_slice_volume(
    slices,
    rotation_matrices,
    image_shape,
    volume_shape,
    *,
    like,
    half_image=None,
    max_r=_AUTO,
):
    """Batch backprojection: per-volume image sets to batch of volumes.

    Parameters
    ----------
    slices : shape ``(batch, n_images, n_pixels)``.
        Real or complex.  Promoted to match *volumes* dtype (and vice versa).
    rotation_matrices : shape ``(n_images, 3, 3)`` — shared across batch.
    like : Volume or CubicVolume
        Batched template/accumulator fixing the output representation. The
        returned arrays are added into its stored arrays.
    half_image : same semantics as ``adjoint_slice_volume``.
    max_r : sphere clipping radius.  Default uses
        ``image_shape[0]//2 - 1``.  Pass ``None`` to disable.
    """
    wrapped_volume = _projection_volume(like, volume_shape, function_name="batch_adjoint_slice_volume")
    disc_type = wrapped_volume.disc_type
    half_volume = wrapped_volume.half_volume
    volumes_array = jnp.asarray(wrapped_volume.array)
    output_shape = volumes_array.shape
    volumes = volumes_array.reshape(volumes_array.shape[0], -1) if volumes_array.ndim > 2 else volumes_array
    half_image = _resolve_half_image(half_image, half_volume)
    slices = jnp.asarray(slices)
    max_r = _resolve_max_r(max_r, image_shape)
    order = decide_order(disc_type)
    if _use_cuda_backproject(order):
        from recovar.cuda_backproject import batch_backproject

        # CUDA needs matching dtypes: if either is complex, promote both.
        out_dtype = jnp.result_type(slices, volumes)
        slices = slices.astype(out_dtype)
        volumes = volumes.astype(out_dtype)
        result = batch_backproject(
            volumes,
            slices,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
            max_r=_cuda_max_r(max_r, image_shape, volume_shape),
        )
        return jnp.asarray(result).reshape(output_shape)
    # JAX fallback: vmap single adjoint
    return jax.vmap(
        lambda sl, vol: adjoint_slice_volume(
            sl,
            rotation_matrices,
            image_shape,
            volume_shape,
            like=wrapped_volume.replace_array(vol),
            half_image=half_image,
            max_r=max_r,
        )
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


def slice_from_cubic_coefficients(coeffs, rotation_matrices, image_shape, volume_shape, half_image=False):
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
    is_half = coeffs.size == half_vol_size and coeffs.size != full_vol_size

    # Cubic's 4×4×4 stencil reads from both Hermitian halves, so
    # half-volume coefficients must be expanded to full before interpolation.
    if is_half:
        coeffs = ftu.half_volume_to_full_volume(coeffs, volume_shape).reshape(volume_shape)
    else:
        coeffs = coeffs.reshape(volume_shape)

    # Choose coordinate grid
    if half_image:
        coords, og_shape = _half_image_rotations_to_coords(rotation_matrices, image_shape, volume_shape)
        n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    else:
        coords, og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
        n_pixels = image_shape[0] * image_shape[1]

    vals = cubic_interpolation.map_coordinates_with_cubic_spline(coeffs, coords, mode="wrap", cval=0.0)
    n_images = rotation_matrices.shape[0]
    return vals.reshape(n_images, n_pixels)


__all__ = [
    "Volume",
    "CubicVolume",
    "to_cubic",
    "decide_order",
    "slice_volume",
    "batch_slice_volume",
    "adjoint_slice_volume",
    "batch_adjoint_slice_volume",
    "_AUTO",
    "_default_max_r",
]
