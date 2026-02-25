"""Forward model and adjoint operations for cryo-EM image formation.

Provides both Equinox-based APIs (``forward_model``, ``adjoint_forward_model``, etc.)
that take a :class:`~recovar.configs.ForwardModelConfig` as first argument, and
legacy wrapper functions that preserve the old positional-argument signatures for
backward compatibility during migration.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

from recovar.configs import ForwardModelConfig
from recovar.core_geometry import translate_images
from recovar.core_slicing import adjoint_slice_volume_by_trilinear, slice_volume_by_map


# ============================================================================
# New Equinox-based API
# ============================================================================


@eqx.filter_jit
def forward_model(
    config: ForwardModelConfig,
    volume: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Project volume into images via slice-and-CTF forward model."""
    slices = slice_volume_by_map(
        volume, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
    )
    if not skip_ctf:
        slices = slices * config.compute_ctf(ctf_params)
    return slices


@eqx.filter_jit
def forward_model_and_adjoint(
    config: ForwardModelConfig,
    volume: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
):
    """Forward model plus its VJP (adjoint) closure."""
    f = lambda v: forward_model(config, v, ctf_params, rotation_matrices, skip_ctf)
    slices, f_adj = vjp(f, volume)
    return slices, f_adj


@eqx.filter_jit
def adjoint_forward_model(
    config: ForwardModelConfig,
    slices: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Adjoint of the forward model (back-projection via VJP)."""
    f = lambda v: forward_model(config, v, ctf_params, rotation_matrices, skip_ctf)
    _, u = vjp(f, jnp.zeros(config.volume_size, dtype=slices.dtype))
    return u(slices)[0]


@eqx.filter_jit
def adjoint_forward_model_trilinear(
    config: ForwardModelConfig,
    slices: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Adjoint via trilinear interpolation (direct, not VJP-based)."""
    if not skip_ctf:
        slices = slices * config.compute_ctf(ctf_params)
    return adjoint_slice_volume_by_trilinear(
        slices, rotation_matrices, config.image_shape, config.volume_shape, volume=None
    )


@eqx.filter_jit
def compute_AtAv(
    config: ForwardModelConfig,
    volume: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    noise_variance: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Compute A^T (A v / noise_variance) for normal equations."""
    f = lambda v: forward_model(config, v, ctf_params, rotation_matrices, skip_ctf)
    y, u = vjp(f, volume)
    return u(y / noise_variance)


# ============================================================================
# Legacy wrappers (backward-compatible, to be removed after full migration)
# ============================================================================


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def forward_model_from_map(
    volume, CTF_params, rotation_matrices,
    image_shape, volume_shape, voxel_size, CTF_fun, disc_type,
    skip_ctf=False,
):
    """Legacy wrapper — prefer :func:`forward_model` with ForwardModelConfig."""
    slices = slice_volume_by_map(volume, rotation_matrices, image_shape, volume_shape, disc_type)
    if not skip_ctf:
        slices = slices * CTF_fun(CTF_params, image_shape, voxel_size)
    return slices


batch_forward_model_from_map = jax.vmap(
    forward_model_from_map, in_axes=(0, 0, 0, None, None, None, None, None, None)
)


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def forward_model_from_map_and_return_adjoint(
    volume, CTF_params, rotation_matrices,
    image_shape, volume_shape, voxel_size, CTF_fun, disc_type,
    skip_ctf=False,
):
    """Legacy wrapper — prefer :func:`forward_model_and_adjoint`."""
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    slices, f_adj = vjp(f, volume)
    return slices, f_adj


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def adjoint_forward_model_from_map(
    slices, CTF_params, rotation_matrices,
    image_shape, volume_shape, voxel_size, CTF_fun, disc_type,
    skip_ctf=False,
):
    """Legacy wrapper — prefer :func:`adjoint_forward_model`."""
    volume_size = np.prod(volume_shape)
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    _, u = vjp(f, jnp.zeros(volume_size, dtype=slices.dtype))
    return u(slices)[0]


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 8], static_argnames=["skip_ctf"])
def adjoint_forward_model_from_trilinear(
    slices, CTF_params, rotation_matrices,
    image_shape, volume_shape, voxel_size, CTF_fun, disc_type,
    skip_ctf=False,
):
    """Legacy wrapper — prefer :func:`adjoint_forward_model_trilinear`."""
    if not skip_ctf:
        slices = slices * CTF_fun(CTF_params, image_shape, voxel_size)
    return adjoint_slice_volume_by_trilinear(slices, rotation_matrices, image_shape, volume_shape, volume=None)


@functools.partial(jax.jit, static_argnums=[3, 4, 6, 7, 9], static_argnames=["skip_ctf"])
def compute_A_t_Av_forward_model_from_map(
    volume, CTF_params, rotation_matrices,
    image_shape, volume_shape, voxel_size, CTF_fun, disc_type,
    noise_variance, skip_ctf=False,
):
    """Legacy wrapper — prefer :func:`compute_AtAv`."""
    f = lambda v: forward_model_from_map(
        v, CTF_params, rotation_matrices, image_shape, volume_shape, voxel_size, CTF_fun, disc_type, skip_ctf
    )
    y, u = vjp(f, volume)
    return u(y / noise_variance)


batch_translate_images = jax.vmap(translate_images, in_axes=(0, 0, None))


__all__ = [
    # New API
    "forward_model",
    "forward_model_and_adjoint",
    "adjoint_forward_model",
    "adjoint_forward_model_trilinear",
    "compute_AtAv",
    # Legacy API (backward-compatible)
    "forward_model_from_map",
    "batch_forward_model_from_map",
    "forward_model_from_map_and_return_adjoint",
    "adjoint_forward_model_from_map",
    "adjoint_forward_model_from_trilinear",
    "compute_A_t_Av_forward_model_from_map",
    "batch_translate_images",
]
