"""Fourier slice extraction and adjoint (backprojection) operations."""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.geometry import (
    get_stencil,
    get_unrotated_plane_grid_points,
    rotations_to_grid_point_coords,
)
from recovar.core.indexing import vol_indices_to_vec_indices

logger = logging.getLogger(__name__)

# ── CUDA acceleration (optional) ──────────────────────────────────────

def _check_cuda():
    """Return True if CUDA project/backproject kernels are available."""
    try:
        from recovar.cuda_backproject import cuda_available
        return cuda_available()
    except (ImportError, OSError):
        return False


@jax.jit
def slice_volume_by_nearest(volume_vec, plane_indices_on_grid):
    return volume_vec[plane_indices_on_grid]


batch_slice_volume_by_nearest = jax.vmap(slice_volume_by_nearest, (None, 0))


def decide_order(disc_type):
    if disc_type == "linear_interp":
        return 1
    if disc_type == "nearest":
        return 0
    if disc_type == "cubic":
        return 3
    raise ValueError("disc_type must be 'linear_interp', 'nearest', or 'cubic'")


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order)


def _is_complex(arr):
    """Check if array has complex dtype (works with tracers too)."""
    return jnp.issubdtype(arr.dtype, jnp.complexfloating)


def _is_tracer(arr):
    return isinstance(arr, jax.core.Tracer)


def _is_jvp_tracer(arr):
    """Detect forward-mode (JVP) tracing — custom_vjp doesn't support JVP."""
    try:
        from jax._src.interpreters.ad import JVPTracer
        return isinstance(arr, JVPTracer)
    except (ImportError, AttributeError):
        return False


# ── CUDA projection with custom VJP (so VJP-based callers work) ─────
@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _slice_volume_by_map_cuda(volume, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=order)


def _slice_cuda_fwd(volume, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    result = cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=order)
    return result, (rotation_matrices,)


def _slice_cuda_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    vol_size = 1
    for s in volume_shape:
        vol_size *= s
    volume = jnp.zeros(vol_size, dtype=g.dtype)
    adj = cuda_backproject(volume, g, rotation_matrices, image_shape, volume_shape, order=order)
    return (adj, jnp.zeros_like(rotation_matrices))


_slice_volume_by_map_cuda.defvjp(_slice_cuda_fwd, _slice_cuda_bwd)


# ── CUDA half-volume projection with custom VJP ─────────────────────
@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _slice_from_half_volume_cuda(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                        order=order, half_volume=True, half_image=False)


def _slice_half_vol_fwd(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    result = cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                          order=order, half_volume=True, half_image=False)
    return result, (rotation_matrices,)


def _slice_half_vol_bwd(image_shape, volume_shape, order, res, g):
    """VJP backward for half-volume project.

    Uses full-vol backproject + VJP(expand) to correctly handle the kz=0 and
    kz=Nyquist boundary slices where 2D Hermitian fold is needed.
    """
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    vol_size = 1
    for s in volume_shape:
        vol_size *= s
    # Backproject into full volume
    full_vol = jnp.zeros(vol_size, dtype=g.dtype)
    full_grad = cuda_backproject(full_vol, g, rotation_matrices, image_shape, volume_shape, order=order)
    # Fold to half via VJP of expand
    half_vol_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    _, vjp_expand = vjp(
        lambda hv: fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
        jnp.zeros(half_vol_size, dtype=g.dtype),
    )
    adj = vjp_expand(full_grad)[0]
    return (adj, jnp.zeros_like(rotation_matrices))


_slice_from_half_volume_cuda.defvjp(_slice_half_vol_fwd, _slice_half_vol_bwd)


# ── CUDA full-volume → half-image projection with custom VJP ────────
@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _slice_volume_by_map_to_half_image_cuda(volume, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                        order=order, half_image=True)


def _slice_to_half_image_cuda_fwd(volume, rotation_matrices, image_shape, volume_shape, order):
    result = _slice_volume_by_map_to_half_image_cuda(volume, rotation_matrices, image_shape, volume_shape, order)
    return result, (rotation_matrices,)


def _slice_to_half_image_cuda_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    volume = jnp.zeros(int(np.prod(volume_shape)), dtype=g.dtype)
    adj = cuda_backproject(volume, g, rotation_matrices, image_shape, volume_shape,
                           order=order, half_image=True)
    return (adj, jnp.zeros_like(rotation_matrices))


_slice_volume_by_map_to_half_image_cuda.defvjp(
    _slice_to_half_image_cuda_fwd, _slice_to_half_image_cuda_bwd
)


def slice_volume_by_map_to_half_image(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a full volume directly to half-image (rfft-packed) format.

    On the CUDA path the kernel only computes the non-redundant half of the
    output frequencies.  On the JAX path the existing
    :func:`_slice_volume_by_map_to_half_image_jax` samples the half
    coordinates directly, avoiding a full-image intermediate.
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volume) and not _is_jvp_tracer(volume):
        return _slice_volume_by_map_to_half_image_cuda(volume, rotation_matrices, image_shape, volume_shape, order)
    return _slice_volume_by_map_to_half_image_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)


def batch_slice_volume_by_map_to_half_image(volumes, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a batch of full volumes to half-spectrum (rfft-packed) images.

    Uses the batched CUDA kernel (with ``half_image=True``) when available.
    Falls back to vmap over :func:`slice_volume_by_map_to_half_image`, which for
    cubic (order>1) does a full slice followed by half extraction.

    Parameters
    ----------
    volumes : complex, shape ``(batch, vol_flat_size)``
    rotation_matrices : real, shape ``(n_images, 3, 3)``
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volumes):
        from recovar.cuda_backproject import batch_project
        return batch_project(volumes, rotation_matrices, image_shape, volume_shape,
                             order=order, half_image=True)
    return jax.vmap(
        lambda v: slice_volume_by_map_to_half_image(v, rotation_matrices, image_shape, volume_shape, disc_type)
    )(volumes)


def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volume) and not _is_jvp_tracer(volume):
        return _slice_volume_by_map_cuda(volume, rotation_matrices, image_shape, volume_shape, order)
    if order <= 1 and not _is_jvp_tracer(volume):
        logger.warning(
            "slice_volume_by_map: CUDA kernel not available (cuda=%s) — "
            "falling back to slower JAX implementation.",
            _check_cuda(),
        )
    return _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)


def batch_slice_volume_by_map(volumes, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project a batch of volumes to images.

    Uses the batched CUDA kernel when available (avoids slow vmap-of-ffi_call).
    Falls back to vmap over :func:`slice_volume_by_map` otherwise.

    Parameters
    ----------
    volumes : complex, shape ``(batch, vol_flat_size)``
    rotation_matrices : real, shape ``(n_images, 3, 3)``
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volumes):
        from recovar.cuda_backproject import batch_project
        return batch_project(volumes, rotation_matrices, image_shape, volume_shape, order=order)
    return jax.vmap(
        lambda v: slice_volume_by_map(v, rotation_matrices, image_shape, volume_shape, disc_type)
    )(volumes)


def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type,
                                volume=None, half_image=False, half_volume=False):
    """Adjoint slice extraction (backprojection) via map_coordinates or CUDA.

    Parameters
    ----------
    half_image : bool
        If True, *slices* are rfft-packed half-spectrum images with shape
        ``(n, H * (W // 2 + 1))``.  The CUDA kernel scatters each pixel
        and its Hermitian conjugate automatically.
    half_volume : bool
        If True, the output volume uses rfft-packed layout with shape
        ``(N0 * N1 * (N2 // 2 + 1),)``.
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda():
        from recovar.cuda_backproject import backproject as cuda_backproject
        if not _is_complex(slices):
            slices = slices.astype(jnp.result_type(slices, jnp.complex64))
        if volume is None:
            vol_shape = (
                fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
                if half_volume else volume_shape
            )
            volume = jnp.zeros(np.prod(vol_shape), dtype=slices.dtype)
        return cuda_backproject(volume, slices, rotation_matrices, image_shape, volume_shape,
                                order=order, half_image=half_image, half_volume=half_volume)
    logger.warning(
        "adjoint_slice_volume_by_map: CUDA kernel not available (order=%d, cuda=%s) — "
        "falling back to slower JAX implementation.",
        order, _check_cuda(),
    )
    # JAX fallback: expand half-images once (CUDA handles this natively via Hermitian scatter)
    if half_image:
        slices = fourier_transform_utils.half_image_to_full_image(slices, image_shape)
    if order > 1:
        if half_volume:
            raise NotImplementedError("half_volume=True with cubic interpolation requires CUDA kernels")
        result = _adjoint_slice_volume_by_map_jax(slices, rotation_matrices, image_shape, volume_shape, disc_type)
    else:
        result = _adjoint_slice_volume_by_map_half_jax(
            slices, rotation_matrices, image_shape, volume_shape, disc_type, half_image=False, half_volume=half_volume
        )
    return result if volume is None else result + volume


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _adjoint_slice_volume_by_map_jax(slices, rotation_matrices, image_shape, volume_shape, disc_type):
    volume_size = np.prod(volume_shape)
    order = decide_order(disc_type)
    if order == 3:
        # For cubic, slice_volume_by_map expects pre-computed spline coefficients (shape N+2 per dim).
        # The VJP must be defined over the flat raw volume space, so we compute the coefficients
        # inside the function-to-differentiate so the gradient flows back to the flat volume.
        from recovar.core import cubic_interpolation

        def f(volume_flat):
            coeffs = cubic_interpolation.calculate_spline_coefficients(volume_flat.reshape(volume_shape))
            return map_coordinates_on_slices(coeffs, rotation_matrices, image_shape, volume_shape, 3)
    else:
        f = lambda volume: _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    _, u = vjp(f, jnp.zeros(volume_size, dtype=slices.dtype))
    return u(slices)[0]


@functools.partial(jax.jit, static_argnums=0)
def summed_adjoint_slice_by_nearest(volume_size, image_vecs, plane_indices_on_grids, volume_vec=None):
    if volume_vec is None:
        volume_vec = jnp.zeros(volume_size, dtype=image_vecs.dtype)
    volume_vec = volume_vec.at[plane_indices_on_grids.reshape(-1)].add((image_vecs).reshape(-1))
    return volume_vec


batch_over_vol_summed_adjoint_slice_by_nearest = jax.vmap(
    summed_adjoint_slice_by_nearest, in_axes=(None, -1, None, -1), out_axes=(-1)
)
nosummed_adjoint_slice_by_nearest = jax.vmap(summed_adjoint_slice_by_nearest, in_axes=(None, 0, 0))


def _coerce_half_image_to_flat(arr, image_shape, name):
    image_shape = tuple(int(s) for s in image_shape)
    half_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    n_half = int(np.prod(half_shape))
    arr = jnp.asarray(arr)
    if arr.ndim >= 2 and tuple(arr.shape[-2:]) == half_shape:
        return arr.reshape(tuple(arr.shape[:-2]) + (n_half,)), True
    if arr.ndim >= 1 and int(arr.shape[-1]) == n_half:
        return arr, False
    raise ValueError(
        f"{name} must have trailing shape {half_shape} or trailing flat size {n_half}, got {arr.shape}"
    )


def _coerce_half_volume_to_flat(arr, volume_shape, name):
    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    n_half = int(np.prod(half_shape))
    arr = jnp.asarray(arr)
    if arr.ndim >= 3 and tuple(arr.shape[-3:]) == half_shape:
        return arr.reshape(tuple(arr.shape[:-3]) + (n_half,)), True
    if arr.ndim >= 1 and int(arr.shape[-1]) == n_half:
        return arr, False
    raise ValueError(
        f"{name} must have trailing shape {half_shape} or trailing flat size {n_half}, got {arr.shape}"
    )


def _reshape_slice_coords(rotation_matrices, image_shape, volume_shape):
    """Return rotated slice coordinates reshaped as (n_images, W, H, 3)."""
    coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    n_images = rotation_matrices.shape[0]
    H, W = image_shape
    return coords.T.reshape(n_images, W, H, 3)


def _packed_half_image_coords(rotation_matrices, image_shape, volume_shape):
    """Return packed-half-image coordinates as (3, n_images * H * (W//2+1))."""
    coords_grid = _reshape_slice_coords(rotation_matrices, image_shape, volume_shape)
    packed_last_idx = fourier_transform_utils.get_real_fft_packed_last_axis_indices(image_shape[1])
    coords_half = jnp.take(coords_grid, packed_last_idx, axis=1)
    return coords_half.reshape(-1, 3).T


def _map_coordinates_volume_from_coords(volume_grid, coords, order):
    if order == 3:
        from recovar.core import cubic_interpolation

        return cubic_interpolation.map_coordinates_with_cubic_spline(
            volume_grid, coords, mode="fill", cval=0.0
        )
    return jax.scipy.ndimage.map_coordinates(
        volume_grid,
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    )


def _sample_half_volume_on_full_coords(half_volume_flat, rotation_matrices, image_shape, volume_shape, order, half_image):
    """Sample Hermitian half-volume on rotated slice coords without full expansion."""
    N0, N1, N2 = volume_shape
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_grid = half_volume_flat.reshape(half_shape)

    coords_grid = _reshape_slice_coords(rotation_matrices, image_shape, volume_shape)
    if half_image:
        packed_last_idx = fourier_transform_utils.get_real_fft_packed_last_axis_indices(image_shape[1])
        coords_grid = jnp.take(coords_grid, packed_last_idx, axis=1)

    coords = coords_grid.reshape(-1, 3).T
    packed_z = fourier_transform_utils.get_real_fft_packed_last_axis_indices(N2).astype(jnp.int32)
    partner0 = fourier_transform_utils.get_shifted_conjugate_partner_indices(N0).astype(jnp.int32)
    partner1 = fourier_transform_utils.get_shifted_conjugate_partner_indices(N1).astype(jnp.int32)
    partner2 = fourier_transform_utils.get_shifted_conjugate_partner_indices(N2).astype(jnp.int32)
    z_to_half = -jnp.ones((N2,), dtype=jnp.int32)
    z_to_half = z_to_half.at[packed_z].set(jnp.arange(half_shape[-1], dtype=jnp.int32))
    half_edge_z = jnp.zeros((half_shape[-1],), dtype=bool)
    half_edge_z = half_edge_z.at[0].set(True)
    if N2 % 2 == 0:
        half_edge_z = half_edge_z.at[half_shape[-1] - 1].set(True)

    def gather_full_value(ix, iy, iz):
        hz = z_to_half[iz]
        in_half = hz >= 0
        hz_safe = jnp.where(in_half, hz, 0)
        ixp = partner0[ix]
        iyp = partner1[iy]
        izp = partner2[iz]
        hzp = z_to_half[izp]
        hzp_safe = jnp.where(hzp >= 0, hzp, 0)
        direct_raw = half_grid[ix, iy, hz_safe]
        direct_sym = 0.5 * (direct_raw + jnp.conj(half_grid[ixp, iyp, hz_safe]))
        direct = jnp.where(half_edge_z[hz_safe], direct_sym, direct_raw)
        mirrored = jnp.conj(half_grid[ixp, iyp, hzp_safe])
        return jnp.where(in_half, direct, mirrored)

    x, y, z = coords[0], coords[1], coords[2]
    if order == 0:
        ix = jnp.rint(x).astype(jnp.int32)
        iy = jnp.rint(y).astype(jnp.int32)
        iz = jnp.rint(z).astype(jnp.int32)
        valid = (ix >= 0) & (ix < N0) & (iy >= 0) & (iy < N1) & (iz >= 0) & (iz < N2)
        ix = jnp.clip(ix, 0, N0 - 1)
        iy = jnp.clip(iy, 0, N1 - 1)
        iz = jnp.clip(iz, 0, N2 - 1)
        vals = jnp.where(valid, gather_full_value(ix, iy, iz), 0.0 + 0.0j)
    else:
        b0 = jnp.floor(x).astype(jnp.int32)
        b1 = jnp.floor(y).astype(jnp.int32)
        b2 = jnp.floor(z).astype(jnp.int32)
        f0 = x - b0
        f1 = y - b1
        f2 = z - b2
        w0 = (1.0 - f0, f0)
        w1 = (1.0 - f1, f1)
        w2 = (1.0 - f2, f2)

        vals = jnp.zeros_like(x, dtype=half_volume_flat.dtype)
        for d0 in (0, 1):
            i0 = b0 + d0
            v0 = (i0 >= 0) & (i0 < N0)
            i0s = jnp.clip(i0, 0, N0 - 1)
            for d1 in (0, 1):
                i1 = b1 + d1
                v1 = v0 & (i1 >= 0) & (i1 < N1)
                i1s = jnp.clip(i1, 0, N1 - 1)
                ww = w0[d0] * w1[d1]
                for d2 in (0, 1):
                    i2 = b2 + d2
                    valid = v1 & (i2 >= 0) & (i2 < N2)
                    i2s = jnp.clip(i2, 0, N2 - 1)
                    contrib = gather_full_value(i0s, i1s, i2s)
                    vals = vals + jnp.where(valid, ww * w2[d2] * contrib, 0.0 + 0.0j)

    n_images = rotation_matrices.shape[0]
    H, W = image_shape
    out_w = W // 2 + 1 if half_image else W
    return vals.reshape(n_images, H * out_w).astype(half_volume_flat.dtype)


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def _slice_volume_by_map_to_half_image_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    if order > 1:
        # Cubic: no dedicated half-coord cubic kernel — slice to full then extract half.
        full = _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)
        return fourier_transform_utils.full_image_to_half_image(full, image_shape)
    coords_half = _packed_half_image_coords(rotation_matrices, image_shape, volume_shape)
    sampled = _map_coordinates_volume_from_coords(volume.reshape(volume_shape), coords_half, order)
    H, W = image_shape
    n_images = rotation_matrices.shape[0]
    return sampled.reshape(n_images, H * (W // 2 + 1)).astype(volume.dtype)


@functools.partial(jax.jit, static_argnums=[2, 3, 4, 5])
def _slice_volume_by_map_from_half_volume_jax(half_volume, rotation_matrices, image_shape, volume_shape, disc_type, half_image):
    order = decide_order(disc_type)
    if order > 1:
        raise NotImplementedError("half-volume projection only supports nearest/linear on the direct JAX path")
    return _sample_half_volume_on_full_coords(
        half_volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        order=order,
        half_image=half_image,
    )


@functools.partial(jax.jit, static_argnums=[2, 3, 4, 5, 6])
def _adjoint_slice_volume_by_map_half_jax(
    slices, rotation_matrices, image_shape, volume_shape, disc_type, half_image, half_volume
):
    order = decide_order(disc_type)
    if order > 1:
        raise NotImplementedError("half-spectrum adjoint only supports nearest/linear on the direct JAX path")

    if half_volume:
        half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(half_shape))
        f = lambda v: _slice_volume_by_map_from_half_volume_jax(
            v, rotation_matrices, image_shape, volume_shape, disc_type, half_image
        )
    elif half_image:
        vol_size = int(np.prod(volume_shape))
        f = lambda v: _slice_volume_by_map_to_half_image_jax(
            v, rotation_matrices, image_shape, volume_shape, disc_type
        )
    else:
        vol_size = int(np.prod(volume_shape))
        f = lambda v: _slice_volume_by_map_jax(v, rotation_matrices, image_shape, volume_shape, disc_type)

    _, u = vjp(f, jnp.zeros(vol_size, dtype=slices.dtype))
    return u(slices)[0]




@functools.partial(jax.jit, static_argnums=0)
def sum_adj_forward_model(volume_size, images, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return summed_adjoint_slice_by_nearest(
        volume_size, images * jnp.conj(CTF_val_on_grid_stacked), plane_indices_on_grid_stacked
    )


@jax.jit
def forward_model(volume_vec, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return batch_slice_volume_by_nearest(volume_vec, plane_indices_on_grid_stacked) * CTF_val_on_grid_stacked


def get_trilinear_weights_and_vol_indices(grid_coords, volume_shape):
    lower_points_ndim = grid_coords.ndim - 1
    stencil = get_stencil(grid_coords.shape[-1])
    all_grid_points = jnp.floor(grid_coords)[..., None, :] + stencil.reshape(
        [*(lower_points_ndim * [1]), stencil.shape[0], stencil.shape[1]]
    )

    all_weights = (1 - jnp.abs(grid_coords[..., None, :] - all_grid_points)).astype(jnp.float32)
    all_weights = jnp.where(all_weights > 0, all_weights, 0)
    all_weights = jnp.prod(all_weights, axis=-1)

    vol_shape = jnp.asarray(volume_shape)
    good_points = jnp.all((all_grid_points >= 0) * (all_grid_points < vol_shape), axis=-1)
    all_weights *= good_points
    return all_grid_points.astype(jnp.int32), all_weights


def slice_volume_by_trilinear(volume, rotation_matrices, image_shape, volume_shape):
    if _check_cuda() and _is_complex(volume):
        return _slice_volume_by_trilinear_cuda(volume, rotation_matrices, image_shape, volume_shape)
    return _slice_volume_by_trilinear_jax(volume, rotation_matrices, image_shape, volume_shape)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _slice_volume_by_trilinear_cuda(volume, rotation_matrices, image_shape, volume_shape):
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=1)


def _slice_trilinear_cuda_fwd(volume, rotation_matrices, image_shape, volume_shape):
    result = _slice_volume_by_trilinear_cuda(volume, rotation_matrices, image_shape, volume_shape)
    return result, (rotation_matrices,)


def _slice_trilinear_cuda_bwd(image_shape, volume_shape, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    vol_size = 1
    for s in volume_shape:
        vol_size *= s
    volume = jnp.zeros(vol_size, dtype=g.dtype)
    adj = cuda_backproject(volume, g, rotation_matrices, image_shape, volume_shape, order=1)
    return (adj, jnp.zeros_like(rotation_matrices))


_slice_volume_by_trilinear_cuda.defvjp(_slice_trilinear_cuda_fwd, _slice_trilinear_cuda_bwd)


def slice_volume_by_trilinear_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape):
    """Project from a packed half volume (Hermitian last-axis) to full images.

    Uses direct half-volume projection without expanding to a full volume.
    """
    return slice_volume_by_map_from_half_volume(
        half_volume, rotation_matrices, image_shape, volume_shape, disc_type="linear_interp"
    )


def slice_volume_by_map_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project from a packed half volume to full images via map_coordinates.

    Uses direct Hermitian sampling from the packed half-volume for nearest/linear.
    Falls back to half->full expansion for cubic.
    """
    order = decide_order(disc_type)
    half_volume_flat, _ = _coerce_half_volume_to_flat(half_volume, volume_shape, name="half_volume")
    if order <= 1 and _check_cuda() and _is_complex(half_volume_flat) and not _is_jvp_tracer(half_volume_flat):
        return _slice_from_half_volume_cuda(half_volume_flat, rotation_matrices, image_shape, volume_shape, order)
    if order > 1:
        full_volume = fourier_transform_utils.half_volume_to_full_volume(half_volume, volume_shape)
        return slice_volume_by_map(full_volume, rotation_matrices, image_shape, volume_shape, disc_type)
    return _slice_volume_by_map_from_half_volume_jax(
        half_volume_flat, rotation_matrices, image_shape, volume_shape, disc_type, half_image=False
    )


def _slice_volume_by_trilinear_jax(volume, rotation_matrices, image_shape, volume_shape):
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)
    sliced_volume = jnp.sum(volume[grid_vec_indices.reshape(-1)].reshape(grid_vec_indices.shape) * weights, axis=-1)
    return sliced_volume.reshape(grid_coords_og_shape[:-1]).astype(volume.dtype)


def adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape,
                                      volume=None, half_image=False, half_volume=False):
    """Adjoint trilinear slicing with optional half-image/half-volume support.

    When CUDA is available, dispatches to the CUDA kernel via
    :func:`adjoint_slice_volume_by_map` which supports half_image/half_volume.
    Falls back to a direct JAX scatter implementation for the full-spectrum case.
    """
    if half_image or half_volume or _check_cuda():
        return adjoint_slice_volume_by_map(
            images, rotation_matrices, image_shape, volume_shape, "linear_interp",
            volume=volume, half_image=half_image, half_volume=half_volume,
        )
    return _adjoint_slice_volume_by_trilinear_jax(images, rotation_matrices, image_shape, volume_shape, volume)


def _adjoint_slice_volume_by_trilinear_jax(images, rotation_matrices, image_shape, volume_shape, volume=None):
    grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype=images.dtype)

    weights *= images.reshape(-1, 1)
    volume = volume.at[grid_vec_indices.reshape(-1)].add(weights.reshape(-1))
    return volume


def adjoint_slice_volume_by_trilinear_from_half_images(
    half_images, rotation_matrices, image_shape, volume_shape, volume=None,
    half_volume=False,
):
    """Adjoint trilinear slicing from packed real-FFT image spectra.

    Uses CUDA ``half_image=True`` when available (no expansion needed).
    JAX fallback also operates directly on packed half-images.

    Parameters
    ----------
    half_volume : bool
        If True, output volume uses rfft-packed layout.
    """
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    return adjoint_slice_volume_by_map(
        half_images_flat, rotation_matrices, image_shape, volume_shape, "linear_interp",
        volume=volume, half_image=True, half_volume=half_volume,
    )




def batch_slice_volume_by_trilinear(volumes, rotation_matrices, image_shape, volume_shape):
    """Project a batch of volumes to images with shared rotations.

    Uses a single CUDA kernel launch that loops over the batch dimension
    internally, reusing rotation coordinates across volumes.

    Parameters
    ----------
    volumes : complex, shape ``(batch, vol_flat_size)``
    rotation_matrices : shape ``(n_images, 3, 3)``

    Returns
    -------
    complex, shape ``(batch, n_images, n_pixels)``
    """
    if _check_cuda() and _is_complex(volumes):
        from recovar.cuda_backproject import batch_project as cuda_batch_project
        return cuda_batch_project(volumes, rotation_matrices, image_shape, volume_shape, order=1)
    return jax.vmap(
        lambda v: slice_volume_by_trilinear(v, rotation_matrices, image_shape, volume_shape)
    )(volumes)


def batch_adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volumes=None):
    """Backproject per-volume images into a batch of volumes with shared rotations.

    Uses a single CUDA kernel launch that loops over the batch dimension
    internally, reusing rotation coordinates across volumes.

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


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order):
    batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape = rotations_to_grid_point_coords(
        rotation_matrices, image_shape, volume_shape
    )
    if order == 3:
        from recovar.core import cubic_interpolation

        slices = cubic_interpolation.map_coordinates_with_cubic_spline(
            volume, batch_grid_pt_vec_ind_of_images, mode="fill", cval=0.0
        ).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1]).astype(volume.dtype)
    else:
        slices = jax.scipy.ndimage.map_coordinates(
            volume.reshape(volume_shape),
            batch_grid_pt_vec_ind_of_images,
            order=order,
            mode="constant",
            cval=0.0,
        ).reshape(batch_grid_pt_vec_ind_of_images_og_shape[:-1]).astype(volume.dtype)
    return slices


__all__ = [
    "slice_volume_by_nearest",
    "batch_slice_volume_by_nearest",
    "decide_order",
    "slice_volume_by_map",
    "batch_slice_volume_by_map",
    "batch_slice_volume_by_map_to_half_image",
    "slice_volume_by_map_from_half_volume",
    "adjoint_slice_volume_by_map",
    "summed_adjoint_slice_by_nearest",
    "batch_over_vol_summed_adjoint_slice_by_nearest",
    "nosummed_adjoint_slice_by_nearest",
    "sum_adj_forward_model",
    "forward_model",
    "get_trilinear_weights_and_vol_indices",
    "slice_volume_by_trilinear",
    "slice_volume_by_trilinear_from_half_volume",
    "adjoint_slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear_from_half_images",
    "adjoint_slice_volume_by_trilinear_from_weights",
    "batch_slice_volume_by_trilinear",
    "batch_adjoint_slice_volume_by_trilinear",
    "map_coordinates_on_slices",
]
