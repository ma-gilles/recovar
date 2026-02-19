import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

from recovar.core_geometry import translate_images
from recovar.core_slicing import adjoint_slice_volume_by_trilinear, slice_volume_by_map


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def forward_model_from_map(
    volume,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    CTF_fun,
    disc_type,
    skip_ctf=False,
):
    slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    if not skip_ctf:
        slices = slices * CTF_fun(CTF_params, image_shape, voxel_size)
    return slices


batch_forward_model_from_map = jax.vmap(
    forward_model_from_map, in_axes=(0, 0, 0, None, None, None, None, None, None)
)


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def forward_model_from_map_and_return_adjoint(
    volume,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    CTF_fun,
    disc_type,
    skip_ctf=False,
):
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    slices, f_adj = vjp(f, volume)
    return slices, f_adj


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def adjoint_forward_model_from_map(
    slices,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    CTF_fun,
    disc_type,
    skip_ctf=False,
):
    volume_size = np.prod(volume_shape)
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    _, u = vjp(f, jnp.zeros(volume_size, dtype=slices.dtype))
    return u(slices)[0]


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def adjoint_forward_model_from_trilinear(
    slices,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    CTF_fun,
    disc_type,
    skip_ctf=False,
):
    if not skip_ctf:
        slices = slices * CTF_fun(CTF_params, image_shape, voxel_size)
    return adjoint_slice_volume_by_trilinear(slices, rotation_matrices, image_shape, volume_shape, volume=None)


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 9], static_argnames=["skip_ctf"])
def compute_A_t_Av_forward_model_from_map(
    volume,
    CTF_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    CTF_fun,
    disc_type,
    noise_variance,
    skip_ctf=False,
):
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    y, u = vjp(f, volume)
    return u(y / noise_variance)


batch_translate_images = jax.vmap(translate_images, in_axes=(0, 0, None))


__all__ = [
    "forward_model_from_map",
    "batch_forward_model_from_map",
    "forward_model_from_map_and_return_adjoint",
    "adjoint_forward_model_from_map",
    "adjoint_forward_model_from_trilinear",
    "compute_A_t_Av_forward_model_from_map",
    "batch_translate_images",
]

