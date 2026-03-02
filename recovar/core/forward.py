"""Forward model and adjoint operations for cryo-EM image formation.

Provides Equinox-based APIs (``forward_model``, ``adjoint_forward_model``, etc.)
that take a :class:`~recovar.configs.ForwardModelConfig` as first argument.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vjp

from recovar.core.configs import ForwardModelConfig
from recovar.core.geometry import translate_images
from recovar.core.slicing import (
    adjoint_slice_volume_by_map,
    adjoint_slice_volume_by_trilinear,
    slice_volume_by_map,
)


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
    volume: jax.Array = None,
    half_image: bool = False,
    half_volume: bool = False,
) -> jax.Array:
    """Adjoint of the forward model (direct back-projection).

    Uses :func:`adjoint_slice_volume_by_map` which dispatches to CUDA
    when available, avoiding the VJP-through-FFI overhead.

    Parameters
    ----------
    volume : optional accumulator array to add into (avoids fresh allocation).
    half_image : if True, *slices* are rfft-packed half-spectrum images.
        CTF is computed in half-spectrum format when ``skip_ctf=False``.
    half_volume : if True, output volume uses rfft-packed layout
        ``(N0 * N1 * (N2 // 2 + 1),)``.  Only supported with CUDA.
    """
    if not skip_ctf:
        ctf = config.compute_ctf_half(ctf_params) if half_image else config.compute_ctf(ctf_params)
        slices = slices * ctf
    return adjoint_slice_volume_by_map(
        slices, rotation_matrices, config.image_shape, config.volume_shape, config.disc_type,
        volume=volume, half_image=half_image, half_volume=half_volume,
    )


@eqx.filter_jit
def adjoint_forward_model_trilinear(
    config: ForwardModelConfig,
    slices: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
    volume: jax.Array = None,
    half_image: bool = False,
    half_volume: bool = False,
) -> jax.Array:
    """Adjoint via trilinear interpolation (direct, not VJP-based).

    Parameters
    ----------
    volume : optional accumulator array to add into.
    half_image : if True, *slices* are rfft-packed half-spectrum images.
    half_volume : if True, output volume uses rfft-packed layout (CUDA only).
    """
    if not skip_ctf:
        ctf = config.compute_ctf_half(ctf_params) if half_image else config.compute_ctf(ctf_params)
        slices = slices * ctf
    return adjoint_slice_volume_by_trilinear(
        slices, rotation_matrices, config.image_shape, config.volume_shape,
        volume=volume, half_image=half_image, half_volume=half_volume,
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


batch_translate_images = jax.vmap(translate_images, in_axes=(0, 0, None))


__all__ = [
    "forward_model",
    "forward_model_and_adjoint",
    "adjoint_forward_model",
    "adjoint_forward_model_trilinear",
    "compute_AtAv",
    "batch_translate_images",
]
