import functools

import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.indexing import vol_indices_to_vec_indices


@jax.jit
def round_to_int(array):
    return jax.lax.round(array).astype(jnp.int32)


@functools.partial(jax.jit, static_argnums=[1])
def find_frequencies_within_grid_dist(coords, max_grid_dist: int):
    dim = coords.shape[-1]
    if dim not in (2, 3):
        raise ValueError(f"Unsupported coordinate dimension: {dim}")

    axis = jnp.arange(-max_grid_dist, max_grid_dist + 1, dtype=jnp.int32)
    if dim == 2:
        n0, n1 = jnp.meshgrid(axis, axis, indexing="ij")
        neighbors = jnp.stack([n0, n1], axis=-1).reshape(-1, 2)
    else:
        n0, n1, n2 = jnp.meshgrid(axis, axis, axis, indexing="ij")
        neighbors = jnp.stack([n0, n1, n2], axis=-1).reshape(-1, 3)

    coords_ndim = coords.ndim
    neighbors = neighbors.reshape((coords_ndim - 1) * [1] + list(neighbors.shape))
    neighbors = neighbors + coords[..., None, :]
    if coords.dtype.kind == "f":
        neighbors = round_to_int(neighbors)
    return neighbors


batch_find_frequencies_within_grid_dist = jax.vmap(find_frequencies_within_grid_dist, in_axes=(0, None))
batch_batch_find_frequencies_within_grid_dist = jax.vmap(batch_find_frequencies_within_grid_dist, in_axes=(0, None))


def get_unrotated_plane_grid_points(image_shape, three_d_upsampling_factor=1):
    unrotated_plane_indices = fourier_transform_utils.get_k_coordinate_of_each_pixel(
        image_shape, voxel_size=1, scaled=False
    )
    unrotated_plane_indices = jnp.pad(unrotated_plane_indices, ((0, 0), (0, 1)))
    return unrotated_plane_indices * three_d_upsampling_factor


def get_unrotated_plane_coords(image_shape, voxel_size, scaled=True):
    plane_coords = fourier_transform_utils.get_k_coordinate_of_each_pixel(
        image_shape, voxel_size=voxel_size, scaled=scaled
    )
    plane_coords = jnp.pad(plane_coords, ((0, 0), (0, 1)))
    return plane_coords


def get_nearest_gridpoint_indices(rotation_matrix, image_shape, volume_shape):
    rotated_plane = get_gridpoint_coords(rotation_matrix, image_shape, volume_shape)
    rotated_indices = round_to_int(rotated_plane)
    return vol_indices_to_vec_indices(rotated_indices, volume_shape)


@functools.partial(jax.jit, static_argnums=[1, 2])
def get_gridpoint_coords(rotation_matrix, image_shape, volume_shape):
    three_d_upsampling_factor = volume_shape[0] // image_shape[0]
    unrotated_plane_indices = get_unrotated_plane_grid_points(
        image_shape, three_d_upsampling_factor=three_d_upsampling_factor
    )
    rotated_plane = jnp.matmul(unrotated_plane_indices, rotation_matrix, precision=jax.lax.Precision.HIGHEST)
    rotated_coords = rotated_plane + (volume_shape[0] // 2)
    return rotated_coords


batch_get_nearest_gridpoint_indices = jax.vmap(get_nearest_gridpoint_indices, in_axes=(0, None, None))
batch_get_gridpoint_coords = jax.vmap(get_gridpoint_coords, in_axes=(0, None, None))


def rotations_to_grid_point_coords(rotation_matrices, image_shape, volume_shape):
    batch_grid_pt_vec_ind_of_images = batch_get_gridpoint_coords(rotation_matrices, image_shape, volume_shape)
    batch_grid_pt_vec_ind_of_images_og_shape = batch_grid_pt_vec_ind_of_images.shape
    batch_grid_pt_vec_ind_of_images = batch_grid_pt_vec_ind_of_images.reshape(-1, 3).T
    return batch_grid_pt_vec_ind_of_images, batch_grid_pt_vec_ind_of_images_og_shape


_STENCIL_2D = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
_STENCIL_3D = jnp.array(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=jnp.int32,
)


def get_stencil(dim):
    if dim == 2:
        return _STENCIL_2D
    if dim == 3:
        return _STENCIL_3D
    raise ValueError(f"Unsupported stencil dimension: {dim}")


@jax.jit
def translate_single_image(image, translation, lattice):
    phase_shift = jnp.exp(-2j * jnp.pi * (lattice @ translation))
    return image * phase_shift


batch_translate = jax.vmap(translate_single_image, in_axes=(0, 0, None))


@functools.partial(jax.jit, static_argnums=2)
def translate_images(image, translation, image_shape):
    twod_lattice = get_unrotated_plane_coords(image_shape, voxel_size=1, scaled=True)[:, :2]
    return batch_translate(image, translation, twod_lattice)


batch_trans_translate_images = jax.vmap(translate_images, in_axes=(None, -2, None), out_axes=-2)


__all__ = [
    "round_to_int",
    "find_frequencies_within_grid_dist",
    "batch_find_frequencies_within_grid_dist",
    "batch_batch_find_frequencies_within_grid_dist",
    "get_unrotated_plane_grid_points",
    "get_unrotated_plane_coords",
    "get_nearest_gridpoint_indices",
    "get_gridpoint_coords",
    "batch_get_nearest_gridpoint_indices",
    "batch_get_gridpoint_coords",
    "rotations_to_grid_point_coords",
    "get_stencil",
    "translate_single_image",
    "batch_translate",
    "translate_images",
    "batch_trans_translate_images",
]
