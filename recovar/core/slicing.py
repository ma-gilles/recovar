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
_cuda_available = None  # tri-state: None = not checked, True/False = result


def _check_cuda():
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available
    try:
        # Only use CUDA if we actually have a GPU backend
        if not any(d.platform == "gpu" for d in jax.devices()):
            _cuda_available = False
            return False
        from recovar.cuda_backproject import _ensure_ffi  # noqa: F401
        _ensure_ffi()
        _cuda_available = True
        logger.info("CUDA backproject/project kernels enabled")
    except Exception as e:
        _cuda_available = False
        logger.debug("CUDA backproject not available: %s", e)
    return _cuda_available


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



def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(volume):
        return _slice_volume_by_map_cuda(volume, rotation_matrices, image_shape, volume_shape, order)
    return _slice_volume_by_map_jax(volume, rotation_matrices, image_shape, volume_shape, disc_type)


def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda() and _is_complex(slices):
        from recovar.cuda_backproject import backproject as cuda_backproject
        volume = jnp.zeros(np.prod(volume_shape), dtype=slices.dtype)
        return cuda_backproject(volume, slices, rotation_matrices, image_shape, volume_shape, order=order)
    return _adjoint_slice_volume_by_map_jax(slices, rotation_matrices, image_shape, volume_shape, disc_type)


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


def _supports_direct_half_symmetry(image_shape):
    return int(image_shape[-1]) % 2 == 0


@functools.lru_cache(maxsize=None)
def _half_image_geometry_maps(image_shape):
    image_shape = tuple(int(s) for s in image_shape)
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    h, w = image_shape
    half = w // 2
    if w % 2 == 0:
        packed_cols = np.asarray(list(range(half, w)) + [0], dtype=np.int32)
    else:
        packed_cols = np.asarray(list(range(half, w)), dtype=np.int32)
    w_half = packed_cols.size

    # Flat full-pixel indices corresponding to packed representation columns.
    packed_full_pixel_indices = (np.arange(h, dtype=np.int32)[:, None] * w + packed_cols[None, :]).reshape(-1)

    # For each full pixel (r, c), map to source packed pixel index and whether conjugation is needed.
    col_to_half = -np.ones((w,), dtype=np.int32)
    col_to_half[packed_cols] = np.arange(w_half, dtype=np.int32)
    def _shifted_conjugate_partner_indices_np(n):
        n_half = n // 2
        idx = np.arange(n, dtype=np.int64)
        unshifted = (idx + n_half) % n
        unshifted_partner = (-unshifted) % n
        shifted_partner = (unshifted_partner - n_half) % n
        return shifted_partner.astype(np.int32)

    row_partner = _shifted_conjugate_partner_indices_np(h)
    col_partner = _shifted_conjugate_partner_indices_np(w)

    rows = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))
    cols = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    packed_pos = col_to_half[cols]
    in_packed = packed_pos >= 0

    partner_rows = row_partner[rows]
    partner_cols = col_partner[cols]
    partner_pos = col_to_half[partner_cols]
    if np.any((~in_packed) & (partner_pos < 0)):
        bad_col = int(cols[(~in_packed) & (partner_pos < 0)][0])
        bad_partner = int(partner_cols[(~in_packed) & (partner_pos < 0)][0])
        raise ValueError(
            f"Internal error: partner column {bad_partner} of column {bad_col} is not in packed columns."
        )

    source_rows = np.where(in_packed, rows, partner_rows)
    source_pos = np.where(in_packed, packed_pos, partner_pos).astype(np.int32)
    source_half_flat_idx = (source_rows * w_half + source_pos).reshape(-1).astype(np.int32)
    source_conjugate = (~in_packed).reshape(-1)

    packed_rows = np.repeat(np.arange(h, dtype=np.int32), w_half)
    packed_cols_tiled = np.tile(packed_cols, h)
    packed_partner_cols = col_partner[packed_cols_tiled]
    packed_partner_full_pixel_indices = row_partner[packed_rows] * w + packed_partner_cols
    packed_partner_in_packed = col_to_half[packed_partner_cols] >= 0
    packed_needs_explicit_partner_term = np.logical_not(packed_partner_in_packed)

    return (
        packed_full_pixel_indices,
        source_half_flat_idx,
        source_conjugate,
        packed_partner_full_pixel_indices,
        packed_needs_explicit_partner_term,
    )


@functools.lru_cache(maxsize=None)
def _half_plane_unrotated_and_partner_coords(image_shape, volume_shape):
    image_shape = tuple(int(s) for s in image_shape)
    volume_shape = tuple(int(s) for s in volume_shape)
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must have 2 dims, got {image_shape}")
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have 3 dims, got {volume_shape}")
    if not _supports_direct_half_symmetry(image_shape):
        raise ValueError(
            f"Direct packed-half geometry requires even image width, got image_shape={image_shape}"
        )

    upsampling = int(volume_shape[0] // image_shape[0])
    full_plane = get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor=upsampling)
    (
        packed_full_pixel_indices,
        _,
        _,
        packed_partner_full_pixel_indices,
        packed_needs_explicit_partner_term,
    ) = _half_image_geometry_maps(image_shape)
    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    packed_partner_full_pixel_indices = jnp.asarray(packed_partner_full_pixel_indices, dtype=jnp.int32)
    half_plane = jnp.take(full_plane, packed_full_pixel_indices, axis=0)
    partner_plane = jnp.take(full_plane, packed_partner_full_pixel_indices, axis=0)

    partner_mask = jnp.asarray(packed_needs_explicit_partner_term, dtype=bool)
    return half_plane.astype(jnp.float32), partner_plane.astype(jnp.float32), partner_mask


def _rotate_unrotated_plane_coords(unrotated_plane_coords, rotation_matrices, volume_shape):
    rotation_matrices = jnp.asarray(rotation_matrices)
    rotated = jnp.einsum(
        "pc,ncd->npd",
        jnp.asarray(unrotated_plane_coords),
        rotation_matrices,
        precision=jax.lax.Precision.HIGHEST,
    )
    return rotated + (volume_shape[0] // 2)


def _rotated_half_and_partner_coords(rotation_matrices, image_shape, volume_shape):
    image_shape = tuple(int(s) for s in image_shape)
    volume_shape = tuple(int(s) for s in volume_shape)
    half_plane, partner_plane, partner_mask = _half_plane_unrotated_and_partner_coords(
        image_shape, volume_shape
    )
    rotated_primary = _rotate_unrotated_plane_coords(half_plane, rotation_matrices, volume_shape)
    rotated_partner = _rotate_unrotated_plane_coords(partner_plane, rotation_matrices, volume_shape)
    return rotated_primary, rotated_partner, partner_mask


def _nearest_indices_and_valid_mask_from_rotated_coords(rotated_coords, volume_shape):
    rounded = jax.lax.round(rotated_coords).astype(jnp.int32)
    vol_shape = jnp.asarray(volume_shape, dtype=jnp.int32)
    valid = jnp.all((rounded >= 0) & (rounded < vol_shape[None, None, :]), axis=-1)
    nearest_indices = vol_indices_to_vec_indices(rounded, volume_shape)
    return nearest_indices, valid


def _half_plane_nearest_indices_and_masks(rotation_matrices, image_shape, volume_shape):
    primary_coords, partner_coords, partner_mask_1d = _rotated_half_and_partner_coords(
        rotation_matrices, image_shape, volume_shape
    )
    primary_indices, primary_valid = _nearest_indices_and_valid_mask_from_rotated_coords(
        primary_coords, volume_shape
    )
    partner_indices, partner_valid = _nearest_indices_and_valid_mask_from_rotated_coords(
        partner_coords, volume_shape
    )
    partner_mask = jnp.broadcast_to(
        partner_mask_1d[None, :],
        primary_valid.shape,
    )
    return primary_indices, primary_valid, partner_indices, partner_valid, partner_mask


@functools.lru_cache(maxsize=None)
def _full_volume_to_half_index_map(volume_shape):
    volume_shape = tuple(int(s) for s in volume_shape)
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have 3 dims, got {volume_shape}")
    d0, d1, d2 = volume_shape
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    d2_half = half_shape[-1]
    half = d2 // 2
    if d2 % 2 == 0:
        packed_z = np.asarray(list(range(half, d2)) + [0], dtype=np.int32)
    else:
        packed_z = np.asarray(list(range(half, d2)), dtype=np.int32)
    z_to_half = -np.ones((d2,), dtype=np.int32)
    z_to_half[packed_z] = np.arange(d2_half, dtype=np.int32)

    n_xy = d0 * d1
    xy = np.arange(n_xy, dtype=np.int32)
    mapping = -np.ones((d0 * d1 * d2,), dtype=np.int32)

    full_idx = (xy[:, None] * d2 + packed_z[None, :]).reshape(-1)
    half_pos = z_to_half[packed_z]
    half_idx = (xy[:, None] * d2_half + half_pos[None, :]).reshape(-1)
    mapping[full_idx] = half_idx
    return mapping


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


def _half_image_sizes(image_shape):
    image_shape = tuple(int(s) for s in image_shape)
    n_full = int(np.prod(image_shape))
    n_half = int(np.prod(fourier_transform_utils.image_shape_to_half_image_shape(image_shape)))
    return n_full, n_half


def _coerce_full_plane_indices(image_shape, plane_indices_on_grid_stacked):
    n_full, _ = _half_image_sizes(image_shape)
    arr = jnp.asarray(plane_indices_on_grid_stacked)
    if arr.ndim < 1 or int(arr.shape[-1]) != n_full:
        raise ValueError(
            f"plane_indices_on_grid_stacked must have trailing size {n_full} (full grid), got {arr.shape}"
        )
    return arr


def split_full_plane_indices_for_half(image_shape, plane_indices_on_grid_stacked):
    """Split full-grid plane indices into packed-primary and packed-partner indices."""
    full_plane_indices = _coerce_full_plane_indices(image_shape, plane_indices_on_grid_stacked)
    packed_full_pixel_indices, _, _, packed_partner_full_pixel_indices, _ = _half_image_geometry_maps(
        tuple(int(s) for s in image_shape)
    )
    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    packed_partner_full_pixel_indices = jnp.asarray(packed_partner_full_pixel_indices, dtype=jnp.int32)
    primary_indices = jnp.take(full_plane_indices, packed_full_pixel_indices, axis=-1)
    partner_indices = jnp.take(full_plane_indices, packed_partner_full_pixel_indices, axis=-1)
    return primary_indices, partner_indices


def _coerce_half_plane_indices_primary(image_shape, plane_indices_on_grid_stacked):
    _, n_half = _half_image_sizes(image_shape)
    if isinstance(plane_indices_on_grid_stacked, (tuple, list)):
        if len(plane_indices_on_grid_stacked) != 2:
            raise ValueError(
                "When passing half-plane indices as tuple/list, use (primary_indices, partner_indices)."
            )
        primary = jnp.asarray(plane_indices_on_grid_stacked[0])
        if primary.ndim < 1 or int(primary.shape[-1]) != n_half:
            raise ValueError(
                f"primary half-plane indices must have trailing size {n_half}, got {primary.shape}"
            )
        return primary

    arr = jnp.asarray(plane_indices_on_grid_stacked)
    if arr.ndim < 1:
        raise ValueError(f"plane_indices_on_grid_stacked must be at least 1D, got {arr.shape}")
    trailing = int(arr.shape[-1])
    if trailing == n_half:
        return arr
    n_full, _ = _half_image_sizes(image_shape)
    if trailing == n_full:
        primary, _ = split_full_plane_indices_for_half(image_shape, arr)
        return primary
    raise ValueError(
        f"plane_indices_on_grid_stacked must have trailing size {n_half} (half) or {n_full} (full), got {arr.shape}"
    )


def _coerce_half_plane_indices_with_partner(image_shape, plane_indices_on_grid_stacked):
    _, n_half = _half_image_sizes(image_shape)
    if isinstance(plane_indices_on_grid_stacked, (tuple, list)):
        if len(plane_indices_on_grid_stacked) != 2:
            raise ValueError(
                "When passing half-plane indices as tuple/list, use (primary_indices, partner_indices)."
            )
        primary = jnp.asarray(plane_indices_on_grid_stacked[0])
        partner = jnp.asarray(plane_indices_on_grid_stacked[1])
        if primary.ndim < 1 or int(primary.shape[-1]) != n_half:
            raise ValueError(
                f"primary half-plane indices must have trailing size {n_half}, got {primary.shape}"
            )
        if partner.ndim < 1 or int(partner.shape[-1]) != n_half:
            raise ValueError(
                f"partner half-plane indices must have trailing size {n_half}, got {partner.shape}"
            )
        return primary, partner

    arr = jnp.asarray(plane_indices_on_grid_stacked)
    if arr.ndim < 1:
        raise ValueError(f"plane_indices_on_grid_stacked must be at least 1D, got {arr.shape}")
    n_full, _ = _half_image_sizes(image_shape)
    trailing = int(arr.shape[-1])
    if trailing == n_full:
        return split_full_plane_indices_for_half(image_shape, arr)
    if trailing == n_half:
        raise ValueError(
            "Half-plane indices for adjoint must include partner indices; pass tuple "
            "(primary_indices, partner_indices), or pass full-grid indices."
        )
    raise ValueError(
        f"plane_indices_on_grid_stacked must have trailing size {n_full} (full), "
        f"or tuple of half-plane indices with trailing size {n_half}, got {arr.shape}"
    )


def _restore_half_image_from_flat(arr_flat, image_shape, return_grid):
    if not return_grid:
        return arr_flat
    half_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    return arr_flat.reshape(tuple(arr_flat.shape[:-1]) + half_shape)


def _half_products_to_full_contribution_terms(
    half_images_flat, half_ctf_flat, image_shape, primary_indices, partner_indices
):
    (
        _,
        _,
        _,
        _,
        packed_needs_explicit_partner_term,
    ) = _half_image_geometry_maps(tuple(int(s) for s in image_shape))

    packed_needs_explicit_partner_term = jnp.asarray(packed_needs_explicit_partner_term, dtype=bool)

    primary_values = half_images_flat * jnp.conj(half_ctf_flat)

    partner_values = jnp.conj(half_images_flat) * half_ctf_flat
    partner_mask = packed_needs_explicit_partner_term.reshape(
        (1,) * (partner_values.ndim - 1) + (partner_values.shape[-1],)
    )
    partner_values = jnp.where(partner_mask, partner_values, 0)
    partner_indices = jnp.where(partner_mask, partner_indices, 0)
    return primary_indices, primary_values, partner_indices, partner_values


def _volume_shapes_and_sizes(volume_shape):
    volume_shape = tuple(int(s) for s in volume_shape)
    full_volume_size = int(np.prod(volume_shape))
    half_volume_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    return volume_shape, full_volume_size, half_volume_shape, half_volume_size


def _normalize_full_volume_seed(volume, full_volume_size):
    if volume is None:
        return None
    volume = jnp.asarray(volume)
    if int(volume.shape[-1]) != full_volume_size:
        raise ValueError(f"volume must have trailing size {full_volume_size}, got {volume.shape}")
    return volume


def _normalize_half_volume_seed(volume, volume_shape, full_volume_size, half_volume_size):
    if volume is None:
        return None
    volume = jnp.asarray(volume)
    trailing = int(volume.shape[-1])
    if trailing == half_volume_size:
        return volume
    if trailing == full_volume_size:
        return fourier_transform_utils.full_volume_to_half_volume(volume, volume_shape)
    raise ValueError(
        f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
    )


def _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size):
    if volume is None:
        return None
    volume = jnp.asarray(volume)
    trailing = int(volume.shape[-1])
    if trailing == full_volume_size:
        return volume
    if trailing == half_volume_size:
        return fourier_transform_utils.half_volume_to_full_volume(volume, volume_shape)
    raise ValueError(
        f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
    )


def _mask_mapped_half_indices(mapped_indices, values):
    valid = mapped_indices >= 0
    safe_indices = jnp.where(valid, mapped_indices, 0)
    safe_values = jnp.where(valid, values, 0)
    return safe_indices, safe_values


def _accumulate_full_index_terms_into_half_volume(
    volume_shape,
    primary_indices,
    primary_values,
    partner_indices,
    partner_values,
    volume=None,
):
    volume_shape, full_volume_size, _, half_volume_size = _volume_shapes_and_sizes(volume_shape)
    full_to_half_map = jnp.asarray(_full_volume_to_half_index_map(volume_shape), dtype=jnp.int32)

    mapped_primary_indices = jnp.take(full_to_half_map, primary_indices, axis=0)
    safe_primary_indices, safe_primary_values = _mask_mapped_half_indices(mapped_primary_indices, primary_values)

    mapped_partner_indices = jnp.take(full_to_half_map, partner_indices, axis=0)
    safe_partner_indices, safe_partner_values = _mask_mapped_half_indices(mapped_partner_indices, partner_values)

    half_seed = _normalize_half_volume_seed(volume, volume_shape, full_volume_size, half_volume_size)
    out = summed_adjoint_slice_by_nearest(
        half_volume_size, safe_primary_values, safe_primary_indices, volume_vec=half_seed
    )
    return summed_adjoint_slice_by_nearest(
        half_volume_size, safe_partner_values, safe_partner_indices, volume_vec=out
    )


def _adjoint_map_from_half_via_full(half_images_flat, rotation_matrices, image_shape, volume_shape, disc_type, volume=None):
    full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
    out = adjoint_slice_volume_by_map(full_images_flat, rotation_matrices, image_shape, volume_shape, disc_type)
    return out if volume is None else out + volume


@functools.partial(jax.jit, static_argnums=0)
def sum_adj_forward_model(volume_size, images, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return summed_adjoint_slice_by_nearest(
        volume_size, images * jnp.conj(CTF_val_on_grid_stacked), plane_indices_on_grid_stacked
    )


@jax.jit
def forward_model(volume_vec, CTF_val_on_grid_stacked, plane_indices_on_grid_stacked):
    return batch_slice_volume_by_nearest(volume_vec, plane_indices_on_grid_stacked) * CTF_val_on_grid_stacked


@functools.partial(jax.jit, static_argnums=[2])
def forward_model_from_half_ctf(volume_vec, half_CTF_val_on_grid_stacked, image_shape, plane_indices_on_grid_stacked):
    """Forward model in packed half-spectrum layout, equivalent to mapped full output."""
    half_ctf_flat, return_grid = _coerce_half_image_to_flat(
        half_CTF_val_on_grid_stacked, image_shape, name="half_CTF_val_on_grid_stacked"
    )
    if not _supports_direct_half_symmetry(image_shape):
        plane_indices_full = _coerce_full_plane_indices(image_shape, plane_indices_on_grid_stacked)
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        full_images_flat = forward_model(volume_vec, full_ctf_flat, plane_indices_full)
        half_images_flat = fourier_transform_utils.full_image_to_half_image(full_images_flat, image_shape)
        return _restore_half_image_from_flat(half_images_flat, image_shape, return_grid)
    plane_indices_half = _coerce_half_plane_indices_primary(image_shape, plane_indices_on_grid_stacked)
    half_images_flat = batch_slice_volume_by_nearest(volume_vec, plane_indices_half) * half_ctf_flat
    return _restore_half_image_from_flat(half_images_flat, image_shape, return_grid)


@functools.partial(jax.jit, static_argnums=[0, 3])
def sum_adj_forward_model_from_half(
    volume_size, half_images, half_CTF_val_on_grid_stacked, image_shape, plane_indices_on_grid_stacked
):
    """Adjoint accumulation in packed half-spectrum layout, equivalent to full-case adjoint."""
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    half_ctf_flat, _ = _coerce_half_image_to_flat(
        half_CTF_val_on_grid_stacked, image_shape, name="half_CTF_val_on_grid_stacked"
    )
    if not _supports_direct_half_symmetry(image_shape):
        plane_indices_full = _coerce_full_plane_indices(image_shape, plane_indices_on_grid_stacked)
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        return sum_adj_forward_model(volume_size, full_images_flat, full_ctf_flat, plane_indices_full)
    primary_indices, partner_indices = _coerce_half_plane_indices_with_partner(
        image_shape, plane_indices_on_grid_stacked
    )
    primary_indices, primary_values, partner_indices, partner_values = _half_products_to_full_contribution_terms(
        half_images_flat, half_ctf_flat, image_shape, primary_indices, partner_indices
    )
    volume = summed_adjoint_slice_by_nearest(volume_size, primary_values, primary_indices)
    return summed_adjoint_slice_by_nearest(volume_size, partner_values, partner_indices, volume_vec=volume)


@functools.partial(jax.jit, static_argnums=[0, 3])
def sum_adj_forward_model_from_half_to_half(
    volume_shape, half_images, half_CTF_val_on_grid_stacked, image_shape, plane_indices_on_grid_stacked
):
    """Adjoint accumulation from packed half images to packed half volumes.

    Equivalent to full adjoint followed by full->half selection, but computed
    directly without constructing a full volume array.
    """
    volume_shape, full_volume_size, _, _ = _volume_shapes_and_sizes(volume_shape)
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    half_ctf_flat, _ = _coerce_half_image_to_flat(
        half_CTF_val_on_grid_stacked, image_shape, name="half_CTF_val_on_grid_stacked"
    )
    if not _supports_direct_half_symmetry(image_shape):
        plane_indices_full = _coerce_full_plane_indices(image_shape, plane_indices_on_grid_stacked)
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        full_volume = sum_adj_forward_model(
            full_volume_size, full_images_flat, full_ctf_flat, plane_indices_full
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)
    primary_indices, partner_indices = _coerce_half_plane_indices_with_partner(
        image_shape, plane_indices_on_grid_stacked
    )
    primary_indices, primary_values, partner_indices, partner_values = _half_products_to_full_contribution_terms(
        half_images_flat, half_ctf_flat, image_shape, primary_indices, partner_indices
    )
    return _accumulate_full_index_terms_into_half_volume(
        volume_shape,
        primary_indices,
        primary_values,
        partner_indices,
        partner_values,
        volume=None,
    )


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
        from recovar.cuda_backproject import project as cuda_project
        return cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=1)
    return _slice_volume_by_trilinear_jax(volume, rotation_matrices, image_shape, volume_shape)


def slice_volume_by_trilinear_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape):
    """Project from a packed half volume (Hermitian last-axis) to full images.

    Expands to full volume then projects via CUDA or JAX.
    """
    full_volume = fourier_transform_utils.half_volume_to_full_volume(half_volume, volume_shape)
    return slice_volume_by_trilinear(full_volume, rotation_matrices, image_shape, volume_shape)


def slice_volume_by_map_from_half_volume(half_volume, rotation_matrices, image_shape, volume_shape, disc_type):
    """Project from a packed half volume to full images via map_coordinates.

    Expands to full volume then projects via CUDA or JAX.
    """
    full_volume = fourier_transform_utils.half_volume_to_full_volume(half_volume, volume_shape)
    return slice_volume_by_map(full_volume, rotation_matrices, image_shape, volume_shape, disc_type)


def _slice_volume_by_trilinear_jax(volume, rotation_matrices, image_shape, volume_shape):
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)
    sliced_volume = jnp.sum(volume[grid_vec_indices.reshape(-1)].reshape(grid_vec_indices.shape) * weights, axis=-1)
    return sliced_volume.reshape(grid_coords_og_shape[:-1]).astype(volume.dtype)


def adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volume=None):
    if _check_cuda() and _is_complex(images):
        from recovar.cuda_backproject import backproject as cuda_backproject
        if volume is None:
            volume = jnp.zeros(np.prod(volume_shape), dtype=images.dtype)
        # Ensure images are 2D (n_images, n_pixels) for the CUDA kernel
        imgs_2d = images.reshape(-1, np.prod(image_shape))
        return cuda_backproject(volume, imgs_2d, rotation_matrices, image_shape, volume_shape, order=1)
    return _adjoint_slice_volume_by_trilinear_jax(images, rotation_matrices, image_shape, volume_shape, volume)


def adjoint_slice_volume_by_trilinear_to_half_volume(images, rotation_matrices, image_shape, volume_shape, volume=None):
    """Backproject full images directly into a packed half volume.

    Note: CUDA half_volume=True cannot be used with full images because both a pixel
    and its Hermitian conjugate scatter to the same half-volume voxel (2x counting).
    We backproject to full volume via CUDA, then extract the half.
    """
    volume_shape, full_volume_size, half_volume_shape, half_volume_size = _volume_shapes_and_sizes(volume_shape)
    full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
    full_volume = adjoint_slice_volume_by_trilinear(
        images, rotation_matrices, image_shape, volume_shape, volume=full_seed
    )
    return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)


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
    half_images, rotation_matrices, image_shape, volume_shape, volume=None
):
    """Adjoint trilinear slicing from packed real-FFT image spectra."""
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    if _check_cuda():
        # Expand half → full images (JAX layout), then use CUDA full-image backproject.
        # The CUDA half_image mode uses a different pixel layout (column-major) than
        # JAX's packed half format (row-major), so we expand to full for compatibility.
        full_images = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        return adjoint_slice_volume_by_trilinear(
            full_images, rotation_matrices, image_shape, volume_shape, volume=volume
        )
    if not _supports_direct_half_symmetry(image_shape):
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        return adjoint_slice_volume_by_trilinear(
            full_images_flat, rotation_matrices, image_shape, volume_shape, volume=volume
        )
    half_images_2d = half_images_flat.reshape(-1, half_images_flat.shape[-1])
    n_images = int(half_images_2d.shape[0])
    n_half = int(half_images_2d.shape[-1])

    primary_coords, partner_coords, partner_mask_1d = _rotated_half_and_partner_coords(
        rotation_matrices, image_shape, volume_shape
    )
    packed_points, packed_weights = get_trilinear_weights_and_vol_indices(primary_coords.reshape(-1, 3), volume_shape)
    packed_indices = vol_indices_to_vec_indices(packed_points, volume_shape)
    partner_points, partner_weights = get_trilinear_weights_and_vol_indices(
        partner_coords.reshape(-1, 3), volume_shape
    )
    partner_indices = vol_indices_to_vec_indices(partner_points, volume_shape)

    packed_values = half_images_2d.reshape(-1)
    partner_mask_flat = jnp.broadcast_to(partner_mask_1d[None, :], (n_images, n_half)).reshape(-1)
    partner_values = jnp.where(partner_mask_flat, jnp.conj(half_images_2d).reshape(-1), 0)

    volume = adjoint_slice_volume_by_trilinear_from_weights(
        packed_values, packed_indices, packed_weights, volume_shape, volume=volume
    )
    return adjoint_slice_volume_by_trilinear_from_weights(
        partner_values, partner_indices, partner_weights, volume_shape, volume=volume
    )


def adjoint_slice_volume_by_trilinear_from_half_images_to_half_volume(
    half_images, rotation_matrices, image_shape, volume_shape, volume=None
):
    """Adjoint trilinear slicing from packed half images to packed half volumes."""
    volume_shape, full_volume_size, half_volume_shape, half_volume_size = _volume_shapes_and_sizes(volume_shape)
    if _check_cuda():
        # Expand half → full images, backproject via CUDA to full volume, then extract half.
        # Cannot use CUDA half_volume=True with full images (Hermitian pair double-counting).
        half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
        full_images = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
        full_volume = adjoint_slice_volume_by_trilinear(
            full_images, rotation_matrices, image_shape, volume_shape, volume=full_seed
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)
    if not _supports_direct_half_symmetry(image_shape):
        full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
        full_volume = adjoint_slice_volume_by_trilinear_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, volume=full_seed
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)

    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    half_images_2d = half_images_flat.reshape(-1, half_images_flat.shape[-1])
    n_images = int(half_images_2d.shape[0])
    n_half = int(half_images_2d.shape[-1])

    primary_coords, partner_coords, partner_mask_1d = _rotated_half_and_partner_coords(
        rotation_matrices, image_shape, volume_shape
    )
    packed_points, packed_weights = get_trilinear_weights_and_vol_indices(primary_coords.reshape(-1, 3), volume_shape)
    packed_indices = vol_indices_to_vec_indices(packed_points, volume_shape)
    partner_points, partner_weights = get_trilinear_weights_and_vol_indices(
        partner_coords.reshape(-1, 3), volume_shape
    )
    partner_indices = vol_indices_to_vec_indices(partner_points, volume_shape)

    packed_values = half_images_2d.reshape(-1)
    partner_mask_flat = jnp.broadcast_to(partner_mask_1d[None, :], (n_images, n_half)).reshape(-1)
    partner_values = jnp.where(partner_mask_flat, jnp.conj(half_images_2d).reshape(-1), 0)
    packed_weighted_vals = packed_weights * packed_values.reshape(-1, 1)
    partner_weighted_vals = partner_weights * partner_values.reshape(-1, 1)
    return _accumulate_full_index_terms_into_half_volume(
        volume_shape,
        packed_indices,
        packed_weighted_vals,
        partner_indices,
        partner_weighted_vals,
        volume=volume,
    )


def adjoint_slice_volume_by_map_from_half_images(
    half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=None
):
    """Adjoint map-based slicing from packed real-FFT image spectra.

    Uses the same geometry and interpolation order as `adjoint_slice_volume_by_map`.
    """
    order = decide_order(disc_type)
    if order <= 1 and _check_cuda():
        # Expand half → full, then use CUDA full-image backproject
        half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
        return _adjoint_map_from_half_via_full(
            half_images_flat, rotation_matrices, image_shape, volume_shape, disc_type, volume=volume
        )
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    volume_shape, full_volume_size, _, _ = _volume_shapes_and_sizes(volume_shape)
    volume = _normalize_full_volume_seed(volume, full_volume_size)
    if not _supports_direct_half_symmetry(image_shape):
        return _adjoint_map_from_half_via_full(
            half_images_flat, rotation_matrices, image_shape, volume_shape, disc_type, volume=volume
        )

    if disc_type == "nearest":
        primary_indices, primary_valid, partner_indices, partner_valid, partner_mask = _half_plane_nearest_indices_and_masks(
            rotation_matrices, image_shape, volume_shape
        )
        primary_values = jnp.where(primary_valid, half_images_flat, 0)
        primary_indices = jnp.where(primary_valid, primary_indices, 0)
        partner_active = jnp.logical_and(partner_valid, partner_mask)
        partner_values = jnp.where(partner_active, jnp.conj(half_images_flat), 0)
        partner_indices = jnp.where(partner_active, partner_indices, 0)
        out = summed_adjoint_slice_by_nearest(full_volume_size, primary_values, primary_indices, volume_vec=volume)
        return summed_adjoint_slice_by_nearest(full_volume_size, partner_values, partner_indices, volume_vec=out)

    if disc_type == "linear_interp":
        return adjoint_slice_volume_by_trilinear_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, volume=volume
        )

    return _adjoint_map_from_half_via_full(
        half_images_flat, rotation_matrices, image_shape, volume_shape, disc_type, volume=volume
    )


def adjoint_slice_volume_by_map_from_half_images_to_half_volume(
    half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=None
):
    """Adjoint map-based slicing from packed half images to packed half volumes."""
    order = decide_order(disc_type)
    volume_shape, full_volume_size, half_volume_shape, half_volume_size = _volume_shapes_and_sizes(volume_shape)
    if order <= 1 and _check_cuda():
        # Expand half → full images, backproject via CUDA to full volume, then extract half.
        # Cannot use CUDA half_volume=True with full images (Hermitian pair double-counting).
        half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
        full_images = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
        full_volume = adjoint_slice_volume_by_map(
            full_images, rotation_matrices, image_shape, volume_shape, disc_type
        )
        if full_seed is not None:
            full_volume = full_volume + full_seed
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)
    if not _supports_direct_half_symmetry(image_shape):
        full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
        full_volume = adjoint_slice_volume_by_map_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=full_seed
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)

    if disc_type == "nearest":
        half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
        primary_indices, primary_valid, partner_indices, partner_valid, partner_mask = _half_plane_nearest_indices_and_masks(
            rotation_matrices, image_shape, volume_shape
        )
        primary_values = jnp.where(primary_valid, half_images_flat, 0)
        primary_indices = jnp.where(primary_valid, primary_indices, 0)
        partner_active = jnp.logical_and(partner_valid, partner_mask)
        partner_values = jnp.where(partner_active, jnp.conj(half_images_flat), 0)
        partner_indices = jnp.where(partner_active, partner_indices, 0)
        return _accumulate_full_index_terms_into_half_volume(
            volume_shape,
            primary_indices,
            primary_values,
            partner_indices,
            partner_values,
            volume=volume,
        )

    if disc_type == "linear_interp":
        return adjoint_slice_volume_by_trilinear_from_half_images_to_half_volume(
            half_images, rotation_matrices, image_shape, volume_shape, volume=volume
        )

    full_seed = _normalize_full_volume_seed_from_any(volume, volume_shape, full_volume_size, half_volume_size)
    full_volume = adjoint_slice_volume_by_map_from_half_images(
        half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=full_seed
    )
    return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)


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
    "slice_volume_by_map_from_half_volume",
    "adjoint_slice_volume_by_map",
    "summed_adjoint_slice_by_nearest",
    "batch_over_vol_summed_adjoint_slice_by_nearest",
    "nosummed_adjoint_slice_by_nearest",
    "sum_adj_forward_model",
    "forward_model",
    "forward_model_from_half_ctf",
    "split_full_plane_indices_for_half",
    "sum_adj_forward_model_from_half",
    "sum_adj_forward_model_from_half_to_half",
    "get_trilinear_weights_and_vol_indices",
    "slice_volume_by_trilinear",
    "slice_volume_by_trilinear_from_half_volume",
    "adjoint_slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear_to_half_volume",
    "adjoint_slice_volume_by_trilinear_from_half_images",
    "adjoint_slice_volume_by_trilinear_from_half_images_to_half_volume",
    "adjoint_slice_volume_by_map_from_half_images",
    "adjoint_slice_volume_by_map_from_half_images_to_half_volume",
    "adjoint_slice_volume_by_trilinear_from_weights",
    "batch_slice_volume_by_trilinear",
    "batch_adjoint_slice_volume_by_trilinear",
    "map_coordinates_on_slices",
]
