import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnums=[1])
def vol_indices_to_vec_indices(vol_indices, vol_shape):
    og_shape = vol_indices.shape
    vol_indices = vol_indices.reshape(-1, vol_indices.shape[-1])
    vec_indices = jnp.ravel_multi_index(vol_indices.T, vol_shape, mode="clip").reshape(og_shape[:-1])
    return vec_indices


def vec_indices_to_vol_indices(vec_indices, vol_shape):
    vol_indices = jnp.unravel_index(vec_indices, vol_shape)
    return jnp.stack(vol_indices, axis=-1)


def vol_indices_to_frequencies(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid = ((vol_shape - (vol_shape % 2 == 1)) // 2).astype(int)
    return vol_indices - mid_grid


def frequencies_to_vol_indices(vol_indices, vol_shape):
    vol_shape = jnp.array(vol_shape)
    mid_grid = ((vol_shape - (vol_shape % 2 == 1)) // 2).astype(int)
    return vol_indices + mid_grid


def vec_indices_to_frequencies(vec_indices, vol_shape):
    return vol_indices_to_frequencies(vec_indices_to_vol_indices(vec_indices, vol_shape), vol_shape)


def frequencies_to_vec_indices(frequencies, vol_shape):
    return vol_indices_to_vec_indices(frequencies_to_vol_indices(frequencies, vol_shape), vol_shape)


def check_frequencies_in_bound(frequencies, grid_size):
    return jnp.all((frequencies >= -grid_size / 2) & (frequencies < grid_size / 2), axis=-1)


def check_vol_indices_in_bound(vol_indices, grid_size):
    return jnp.all((vol_indices >= 0) & (vol_indices < grid_size), axis=-1)


def check_vec_indices_in_bound(vec_indices, grid_size):
    return ((vec_indices < grid_size**3) * (vec_indices >= 0)).astype(bool)


def distance_to_max_grid_dist(dist):
    return np.ceil(dist).astype(int)


__all__ = [
    "vol_indices_to_vec_indices",
    "vec_indices_to_vol_indices",
    "vol_indices_to_frequencies",
    "frequencies_to_vol_indices",
    "vec_indices_to_frequencies",
    "frequencies_to_vec_indices",
    "check_frequencies_in_bound",
    "check_vol_indices_in_bound",
    "check_vec_indices_in_bound",
    "distance_to_max_grid_dist",
]

