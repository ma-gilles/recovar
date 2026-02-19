import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

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


def adjoint_slice_volume_by_trilinear_from_weights(images, grid_vec_indices, weights, volume_shape, volume=None):
    if volume is None:
        volume = jnp.zeros(np.prod(volume_shape), dtype=images.dtype)

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
    "adjoint_slice_volume_by_map",
    "summed_adjoint_slice_by_nearest",
    "batch_over_vol_summed_adjoint_slice_by_nearest",
    "nosummed_adjoint_slice_by_nearest",
    "sum_adj_forward_model",
    "forward_model",
    "get_trilinear_weights_and_vol_indices",
    "slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear",
    "adjoint_slice_volume_by_trilinear_from_weights",
    "map_coordinates_on_slices",
]
