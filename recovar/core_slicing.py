import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.fourier_transform_utils as fourier_transform_utils
from recovar.core_geometry import (
    get_stencil,
    rotations_to_grid_point_coords,
)
from recovar.core_indexing import vol_indices_to_vec_indices


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
def slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type):
    order = decide_order(disc_type)
    return map_coordinates_on_slices(volume, rotation_matrices, image_shape, volume_shape, order)


@functools.partial(jax.jit, static_argnums=[2, 3, 4])
def adjoint_slice_volume_by_map(slices, rotation_matrices, image_shape, volume_shape, disc_type):
    volume_size = np.prod(volume_shape)
    f = lambda volume: slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
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


def _restore_half_image_from_flat(arr_flat, image_shape, return_grid):
    if not return_grid:
        return arr_flat
    half_shape = fourier_transform_utils.image_shape_to_half_image_shape(image_shape)
    return arr_flat.reshape(tuple(arr_flat.shape[:-1]) + half_shape)


def _reconstruct_full_image_flat_from_half(half_flat, image_shape):
    image_shape = tuple(int(s) for s in image_shape)
    if not _supports_direct_half_symmetry(image_shape):
        return fourier_transform_utils.half_image_to_full_image(half_flat, image_shape)
    _, source_half_flat_idx, source_conjugate, _, _ = _half_image_geometry_maps(tuple(int(s) for s in image_shape))
    source_half_flat_idx = jnp.asarray(source_half_flat_idx, dtype=jnp.int32)
    source_conjugate = jnp.asarray(source_conjugate, dtype=bool)

    full_flat = jnp.take(half_flat, source_half_flat_idx, axis=-1)
    conj_mask = source_conjugate.reshape((1,) * (full_flat.ndim - 1) + (full_flat.shape[-1],))
    return jnp.where(conj_mask, jnp.conj(full_flat), full_flat)


def _half_products_to_full_contribution_terms(
    half_images_flat, half_ctf_flat, image_shape, plane_indices_on_grid_stacked
):
    (
        packed_full_pixel_indices,
        _,
        _,
        packed_partner_full_pixel_indices,
        packed_needs_explicit_partner_term,
    ) = _half_image_geometry_maps(tuple(int(s) for s in image_shape))

    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    packed_partner_full_pixel_indices = jnp.asarray(packed_partner_full_pixel_indices, dtype=jnp.int32)
    packed_needs_explicit_partner_term = jnp.asarray(packed_needs_explicit_partner_term, dtype=bool)

    primary_indices = jnp.take(plane_indices_on_grid_stacked, packed_full_pixel_indices, axis=-1)
    primary_values = half_images_flat * jnp.conj(half_ctf_flat)

    partner_indices = jnp.take(plane_indices_on_grid_stacked, packed_partner_full_pixel_indices, axis=-1)
    partner_values = jnp.conj(half_images_flat) * half_ctf_flat
    partner_mask = packed_needs_explicit_partner_term.reshape(
        (1,) * (partner_values.ndim - 1) + (partner_values.shape[-1],)
    )
    partner_values = jnp.where(partner_mask, partner_values, 0)
    partner_indices = jnp.where(partner_mask, partner_indices, 0)
    return primary_indices, primary_values, partner_indices, partner_values


def _half_values_to_full_contribution_terms(
    half_values_flat, image_shape, plane_indices_on_grid_stacked, valid_mask_full=None
):
    (
        packed_full_pixel_indices,
        _,
        _,
        packed_partner_full_pixel_indices,
        packed_needs_explicit_partner_term,
    ) = _half_image_geometry_maps(tuple(int(s) for s in image_shape))
    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    packed_partner_full_pixel_indices = jnp.asarray(packed_partner_full_pixel_indices, dtype=jnp.int32)
    packed_needs_explicit_partner_term = jnp.asarray(packed_needs_explicit_partner_term, dtype=bool)

    primary_indices = jnp.take(plane_indices_on_grid_stacked, packed_full_pixel_indices, axis=-1)
    primary_values = half_values_flat
    if valid_mask_full is not None:
        valid_primary = jnp.take(valid_mask_full, packed_full_pixel_indices, axis=-1)
        primary_values = jnp.where(valid_primary, primary_values, 0)
        primary_indices = jnp.where(valid_primary, primary_indices, 0)

    partner_indices = jnp.take(plane_indices_on_grid_stacked, packed_partner_full_pixel_indices, axis=-1)
    partner_values = jnp.conj(half_values_flat)
    partner_mask = packed_needs_explicit_partner_term.reshape(
        (1,) * (partner_values.ndim - 1) + (partner_values.shape[-1],)
    )
    if valid_mask_full is not None:
        valid_partner = jnp.take(valid_mask_full, packed_partner_full_pixel_indices, axis=-1)
        partner_mask = jnp.logical_and(partner_mask, valid_partner)
    partner_values = jnp.where(partner_mask, partner_values, 0)
    partner_indices = jnp.where(partner_mask, partner_indices, 0)
    return primary_indices, primary_values, partner_indices, partner_values


def _nearest_plane_indices_and_valid_mask(rotation_matrices, image_shape, volume_shape):
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    rounded_grid = jax.lax.round(grid_coords.T).astype(jnp.int32)
    vol_shape = jnp.asarray(volume_shape, dtype=jnp.int32)
    valid_flat = jnp.all((rounded_grid >= 0) & (rounded_grid < vol_shape[None, :]), axis=-1)
    nearest_indices_flat = vol_indices_to_vec_indices(rounded_grid, volume_shape)
    nearest_indices = nearest_indices_flat.reshape(grid_coords_og_shape[:-1])
    valid_mask = valid_flat.reshape(grid_coords_og_shape[:-1])
    return nearest_indices, valid_mask


def _get_packed_and_partner_image_rows(image_shape, n_images):
    (
        packed_full_pixel_indices,
        _,
        _,
        packed_partner_full_pixel_indices,
        packed_needs_explicit_partner_term,
    ) = _half_image_geometry_maps(tuple(int(s) for s in image_shape))
    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    packed_partner_full_pixel_indices = jnp.asarray(packed_partner_full_pixel_indices, dtype=jnp.int32)
    packed_needs_explicit_partner_term = jnp.asarray(packed_needs_explicit_partner_term, dtype=bool)

    n_pixels = int(np.prod(image_shape))
    image_offsets = (jnp.arange(n_images, dtype=jnp.int32) * n_pixels)[:, None]
    packed_rows = (image_offsets + packed_full_pixel_indices[None, :]).reshape(-1)
    partner_rows = (image_offsets + packed_partner_full_pixel_indices[None, :]).reshape(-1)
    partner_mask_flat = jnp.broadcast_to(
        packed_needs_explicit_partner_term[None, :],
        (n_images, packed_needs_explicit_partner_term.size),
    ).reshape(-1)
    return packed_rows, partner_rows, partner_mask_flat


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
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        full_images_flat = forward_model(volume_vec, full_ctf_flat, plane_indices_on_grid_stacked)
        half_images_flat = fourier_transform_utils.full_image_to_half_image(full_images_flat, image_shape)
        return _restore_half_image_from_flat(half_images_flat, image_shape, return_grid)
    packed_full_pixel_indices, _, _, _, _ = _half_image_geometry_maps(tuple(int(s) for s in image_shape))
    packed_full_pixel_indices = jnp.asarray(packed_full_pixel_indices, dtype=jnp.int32)
    plane_indices_half = jnp.take(plane_indices_on_grid_stacked, packed_full_pixel_indices, axis=-1)
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
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        return sum_adj_forward_model(volume_size, full_images_flat, full_ctf_flat, plane_indices_on_grid_stacked)
    primary_indices, primary_values, partner_indices, partner_values = _half_products_to_full_contribution_terms(
        half_images_flat, half_ctf_flat, image_shape, plane_indices_on_grid_stacked
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
    volume_shape = tuple(int(s) for s in volume_shape)
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    half_ctf_flat, _ = _coerce_half_image_to_flat(
        half_CTF_val_on_grid_stacked, image_shape, name="half_CTF_val_on_grid_stacked"
    )
    if not _supports_direct_half_symmetry(image_shape):
        volume_size = int(np.prod(volume_shape))
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        full_ctf_flat = fourier_transform_utils.half_image_to_full_image(half_ctf_flat, image_shape)
        full_volume = sum_adj_forward_model(volume_size, full_images_flat, full_ctf_flat, plane_indices_on_grid_stacked)
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)
    primary_indices, primary_values, partner_indices, partner_values = _half_products_to_full_contribution_terms(
        half_images_flat, half_ctf_flat, image_shape, plane_indices_on_grid_stacked
    )

    full_to_half_map = jnp.asarray(_full_volume_to_half_index_map(volume_shape), dtype=jnp.int32)
    target_primary_indices = jnp.take(full_to_half_map, primary_indices, axis=0)
    valid_primary = target_primary_indices >= 0
    safe_primary_indices = jnp.where(valid_primary, target_primary_indices, 0)
    safe_primary_values = jnp.where(valid_primary, primary_values, 0)

    target_partner_indices = jnp.take(full_to_half_map, partner_indices, axis=0)
    valid_partner = target_partner_indices >= 0
    safe_partner_indices = jnp.where(valid_partner, target_partner_indices, 0)
    safe_partner_values = jnp.where(valid_partner, partner_values, 0)

    half_volume_size = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)))
    half_volume = summed_adjoint_slice_by_nearest(half_volume_size, safe_primary_values, safe_primary_indices)
    return summed_adjoint_slice_by_nearest(
        half_volume_size,
        safe_partner_values,
        safe_partner_indices,
        volume_vec=half_volume,
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
    grid_coords, grid_coords_og_shape = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)
    sliced_volume = jnp.sum(volume[grid_vec_indices.reshape(-1)].reshape(grid_vec_indices.shape) * weights, axis=-1)
    return sliced_volume.reshape(grid_coords_og_shape[:-1]).astype(volume.dtype)


def adjoint_slice_volume_by_trilinear(images, rotation_matrices, image_shape, volume_shape, volume=None):
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
    if not _supports_direct_half_symmetry(image_shape):
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        return adjoint_slice_volume_by_trilinear(
            full_images_flat, rotation_matrices, image_shape, volume_shape, volume=volume
        )
    half_images_2d = half_images_flat.reshape(-1, half_images_flat.shape[-1])
    n_images = int(half_images_2d.shape[0])
    grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    packed_rows, partner_rows, partner_mask_flat = _get_packed_and_partner_image_rows(image_shape, n_images)
    packed_indices = jnp.take(grid_vec_indices, packed_rows, axis=0)
    packed_weights = jnp.take(weights, packed_rows, axis=0)
    partner_indices = jnp.take(grid_vec_indices, partner_rows, axis=0)
    partner_weights = jnp.take(weights, partner_rows, axis=0)

    packed_values = half_images_2d.reshape(-1)
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
    volume_shape = tuple(int(s) for s in volume_shape)
    full_volume_size = int(np.prod(volume_shape))
    half_volume_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    if not _supports_direct_half_symmetry(image_shape):
        full_seed = None
        if volume is not None:
            volume = jnp.asarray(volume)
            if int(volume.shape[-1]) == half_volume_size:
                full_seed = fourier_transform_utils.half_volume_to_full_volume(volume, volume_shape)
            elif int(volume.shape[-1]) == full_volume_size:
                full_seed = volume
            else:
                raise ValueError(
                    f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
                )
        full_volume = adjoint_slice_volume_by_trilinear_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, volume=full_seed
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)

    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    half_images_2d = half_images_flat.reshape(-1, half_images_flat.shape[-1])
    n_images = int(half_images_2d.shape[0])

    grid_coords, _ = rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape)
    grid_points, weights = get_trilinear_weights_and_vol_indices(grid_coords.T, volume_shape)
    grid_vec_indices = vol_indices_to_vec_indices(grid_points, volume_shape)

    packed_rows, partner_rows, partner_mask_flat = _get_packed_and_partner_image_rows(image_shape, n_images)
    packed_indices = jnp.take(grid_vec_indices, packed_rows, axis=0)
    packed_weights = jnp.take(weights, packed_rows, axis=0)
    partner_indices = jnp.take(grid_vec_indices, partner_rows, axis=0)
    partner_weights = jnp.take(weights, partner_rows, axis=0)

    packed_values = half_images_2d.reshape(-1)
    partner_values = jnp.where(partner_mask_flat, jnp.conj(half_images_2d).reshape(-1), 0)
    packed_weighted_vals = packed_weights * packed_values.reshape(-1, 1)
    partner_weighted_vals = partner_weights * partner_values.reshape(-1, 1)

    full_to_half_map = jnp.asarray(_full_volume_to_half_index_map(tuple(int(s) for s in volume_shape)), dtype=jnp.int32)
    mapped_packed_indices = jnp.take(full_to_half_map, packed_indices, axis=0)
    valid_packed = mapped_packed_indices >= 0
    safe_packed_indices = jnp.where(valid_packed, mapped_packed_indices, 0)
    safe_packed_vals = jnp.where(valid_packed, packed_weighted_vals, 0)

    mapped_partner_indices = jnp.take(full_to_half_map, partner_indices, axis=0)
    valid_partner = mapped_partner_indices >= 0
    safe_partner_indices = jnp.where(valid_partner, mapped_partner_indices, 0)
    safe_partner_vals = jnp.where(valid_partner, partner_weighted_vals, 0)

    if volume is not None:
        volume = jnp.asarray(volume)
        if int(volume.shape[-1]) == full_volume_size:
            volume = fourier_transform_utils.full_volume_to_half_volume(volume, volume_shape)
        elif int(volume.shape[-1]) != half_volume_size:
            raise ValueError(
                f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
            )

    half_volume = summed_adjoint_slice_by_nearest(
        half_volume_size,
        safe_packed_vals,
        safe_packed_indices,
        volume_vec=volume,
    )
    return summed_adjoint_slice_by_nearest(
        half_volume_size,
        safe_partner_vals,
        safe_partner_indices,
        volume_vec=half_volume,
    )


def adjoint_slice_volume_by_map_from_half_images(
    half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=None
):
    """Adjoint map-based slicing from packed real-FFT image spectra.

    Uses the same geometry and interpolation order as `adjoint_slice_volume_by_map`.
    """
    half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
    volume_shape = tuple(int(s) for s in volume_shape)
    volume_size = int(np.prod(volume_shape))
    if volume is not None:
        volume = jnp.asarray(volume)
        if int(volume.shape[-1]) != volume_size:
            raise ValueError(f"volume must have trailing size {volume_size}, got {volume.shape}")
    if not _supports_direct_half_symmetry(image_shape):
        full_images_flat = fourier_transform_utils.half_image_to_full_image(half_images_flat, image_shape)
        out = adjoint_slice_volume_by_map(
            full_images_flat, rotation_matrices, image_shape, volume_shape, disc_type
        )
        return out if volume is None else out + volume

    if disc_type == "nearest":
        plane_indices_on_grid_stacked, valid_mask_full = _nearest_plane_indices_and_valid_mask(
            rotation_matrices, image_shape, volume_shape
        )
        primary_indices, primary_values, partner_indices, partner_values = _half_values_to_full_contribution_terms(
            half_images_flat, image_shape, plane_indices_on_grid_stacked, valid_mask_full=valid_mask_full
        )
        out = summed_adjoint_slice_by_nearest(volume_size, primary_values, primary_indices, volume_vec=volume)
        return summed_adjoint_slice_by_nearest(volume_size, partner_values, partner_indices, volume_vec=out)

    if disc_type == "linear_interp":
        return adjoint_slice_volume_by_trilinear_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, volume=volume
        )

    full_images_flat = _reconstruct_full_image_flat_from_half(half_images_flat, image_shape)
    return adjoint_slice_volume_by_map(
        full_images_flat, rotation_matrices, image_shape, volume_shape, disc_type
    ) if volume is None else adjoint_slice_volume_by_map(
        full_images_flat, rotation_matrices, image_shape, volume_shape, disc_type
    ) + volume


def adjoint_slice_volume_by_map_from_half_images_to_half_volume(
    half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=None
):
    """Adjoint map-based slicing from packed half images to packed half volumes."""
    volume_shape = tuple(int(s) for s in volume_shape)
    full_volume_size = int(np.prod(volume_shape))
    half_volume_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    if not _supports_direct_half_symmetry(image_shape):
        full_seed = None
        if volume is not None:
            volume = jnp.asarray(volume)
            if int(volume.shape[-1]) == half_volume_size:
                full_seed = fourier_transform_utils.half_volume_to_full_volume(volume, volume_shape)
            elif int(volume.shape[-1]) == full_volume_size:
                full_seed = volume
            else:
                raise ValueError(
                    f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
                )
        full_volume = adjoint_slice_volume_by_map_from_half_images(
            half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=full_seed
        )
        return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)

    if disc_type == "nearest":
        half_images_flat, _ = _coerce_half_image_to_flat(half_images, image_shape, name="half_images")
        plane_indices_on_grid_stacked, valid_mask_full = _nearest_plane_indices_and_valid_mask(
            rotation_matrices, image_shape, volume_shape
        )
        primary_indices, primary_values, partner_indices, partner_values = _half_values_to_full_contribution_terms(
            half_images_flat, image_shape, plane_indices_on_grid_stacked, valid_mask_full=valid_mask_full
        )
        full_to_half_map = jnp.asarray(_full_volume_to_half_index_map(volume_shape), dtype=jnp.int32)

        mapped_primary_indices = jnp.take(full_to_half_map, primary_indices, axis=0)
        valid_primary = mapped_primary_indices >= 0
        safe_primary_indices = jnp.where(valid_primary, mapped_primary_indices, 0)
        safe_primary_values = jnp.where(valid_primary, primary_values, 0)

        mapped_partner_indices = jnp.take(full_to_half_map, partner_indices, axis=0)
        valid_partner = mapped_partner_indices >= 0
        safe_partner_indices = jnp.where(valid_partner, mapped_partner_indices, 0)
        safe_partner_values = jnp.where(valid_partner, partner_values, 0)

        if volume is not None:
            volume = jnp.asarray(volume)
            if int(volume.shape[-1]) == full_volume_size:
                volume = fourier_transform_utils.full_volume_to_half_volume(volume, volume_shape)
            elif int(volume.shape[-1]) != half_volume_size:
                raise ValueError(
                    f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
                )

        out = summed_adjoint_slice_by_nearest(
            half_volume_size, safe_primary_values, safe_primary_indices, volume_vec=volume
        )
        return summed_adjoint_slice_by_nearest(
            half_volume_size, safe_partner_values, safe_partner_indices, volume_vec=out
        )

    if disc_type == "linear_interp":
        return adjoint_slice_volume_by_trilinear_from_half_images_to_half_volume(
            half_images, rotation_matrices, image_shape, volume_shape, volume=volume
        )

    full_seed = None
    if volume is not None:
        volume = jnp.asarray(volume)
        if int(volume.shape[-1]) == half_volume_size:
            full_seed = fourier_transform_utils.half_volume_to_full_volume(volume, volume_shape)
        elif int(volume.shape[-1]) == full_volume_size:
            full_seed = volume
        else:
            raise ValueError(
                f"volume must have trailing size {half_volume_size} (half) or {full_volume_size} (full), got {volume.shape}"
            )

    full_volume = adjoint_slice_volume_by_map_from_half_images(
        half_images, rotation_matrices, image_shape, volume_shape, disc_type, volume=full_seed
    )
    return fourier_transform_utils.full_volume_to_half_volume(full_volume, volume_shape)


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
        from recovar import cubic_interpolation

        slices = cubic_interpolation.map_coordinates_with_cubic_spline(
            volume.reshape(volume_shape), batch_grid_pt_vec_ind_of_images, mode="fill", cval=0.0
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
    "adjoint_slice_volume_by_map",
    "summed_adjoint_slice_by_nearest",
    "batch_over_vol_summed_adjoint_slice_by_nearest",
    "nosummed_adjoint_slice_by_nearest",
    "sum_adj_forward_model",
    "forward_model",
    "forward_model_from_half_ctf",
    "sum_adj_forward_model_from_half",
    "sum_adj_forward_model_from_half_to_half",
    "get_trilinear_weights_and_vol_indices",
    "slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear_from_half_images",
    "adjoint_slice_volume_by_trilinear_from_half_images_to_half_volume",
    "adjoint_slice_volume_by_map_from_half_images",
    "adjoint_slice_volume_by_map_from_half_images_to_half_volume",
    "adjoint_slice_volume_by_trilinear_from_weights",
    "map_coordinates_on_slices",
]
