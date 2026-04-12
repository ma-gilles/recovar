"""Forward model and adjoint operations for cryo-EM image formation.

Provides Equinox-based APIs (``forward_model``, ``adjoint_forward_model``, etc.)
that take a :class:`~recovar.configs.ForwardModelConfig` as first argument.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vjp

logger = logging.getLogger(__name__)

from recovar.core.configs import ForwardModelConfig
from recovar.core.geometry import translate_images
from recovar.core.slicing import (
    _coerce_volume,
    _projection_volume,
    adjoint_slice_volume,
    slice_volume,
)


@eqx.filter_jit
def forward_model(
    config: ForwardModelConfig,
    volume,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
    half_image: bool | None = None,
) -> jax.Array:
    """Project volume into images via slice-and-CTF forward model.

    Parameters
    ----------
    volume : Volume or CubicVolume
        Projection input. Cubic inputs must be passed explicitly as a
        ``CubicVolume`` produced by ``to_cubic(...)`` or ``CubicVolume(...)``.
    half_image : bool | None
        If True, return rfft-packed half-spectrum images and use
        ``config.compute_ctf_half`` for CTF, roughly halving memory and compute.
        ``None`` defaults to ``volume.half_volume``.
    """
    volume = _projection_volume(
        volume,
        config.volume_shape,
        disc_type=config.disc_type,
        function_name="forward_model",
    )
    if half_image is None:
        half_image = volume.half_volume

    slices = slice_volume(volume, rotation_matrices, config.image_shape, config.volume_shape, half_image=half_image)
    if not skip_ctf:
        slices = slices * config.compute_ctf(ctf_params, half_image=half_image)
    return slices


@eqx.filter_jit
def forward_model_and_adjoint(
    config: ForwardModelConfig,
    volume,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
):
    """Forward model plus its VJP (adjoint) closure."""
    volume = _projection_volume(
        volume,
        config.volume_shape,
        disc_type=config.disc_type,
        function_name="forward_model_and_adjoint",
    )
    f = lambda values: forward_model(config, volume.with_values(values), ctf_params, rotation_matrices, skip_ctf)
    slices, f_adj = vjp(f, volume.values)
    return slices, f_adj


@eqx.filter_jit
def adjoint_forward_model(
    config: ForwardModelConfig,
    slices: jax.Array,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    skip_ctf: bool = False,
    volume=None,
    half_image: bool | None = None,
) -> jax.Array:
    """Adjoint of the forward model (direct back-projection).

    Uses :func:`adjoint_slice_volume` which dispatches to CUDA
    when available, avoiding the VJP-through-FFI overhead.

    Parameters
    ----------
    volume : optional accumulator array or Volume to add into.
        When a :class:`~recovar.core.slicing.Volume` is supplied, its
        ``half_volume`` layout controls the output layout.
    half_image : if True, *slices* are rfft-packed half-spectrum images.
        CTF is computed in half-spectrum format when ``skip_ctf=False``.
    """
    volume_obj = None if volume is None else _coerce_volume(volume, config.disc_type, config.volume_shape)
    if half_image is None:
        half_image = volume_obj.half_volume if volume_obj is not None else False

    if not skip_ctf:
        slices = slices * config.compute_ctf(ctf_params, half_image=half_image)
    adjoint_kwargs = dict(volume=volume_obj, half_image=half_image)
    if volume_obj is None:
        adjoint_kwargs["disc_type"] = config.disc_type
    return adjoint_slice_volume(slices, rotation_matrices, config.image_shape, config.volume_shape, **adjoint_kwargs)


@eqx.filter_jit
def compute_AtAv(
    config: ForwardModelConfig,
    volume,
    ctf_params: jax.Array,
    rotation_matrices: jax.Array,
    noise_variance: jax.Array,
    skip_ctf: bool = False,
) -> jax.Array:
    """Compute A^T (A v / noise_variance) for normal equations."""
    volume = _projection_volume(
        volume,
        config.volume_shape,
        disc_type=config.disc_type,
        function_name="compute_AtAv",
    )
    f = lambda values: forward_model(config, volume.with_values(values), ctf_params, rotation_matrices, skip_ctf)
    y, u = vjp(f, volume.values)
    return u(y / noise_variance)


batch_translate_images = jax.vmap(translate_images, in_axes=(0, 0, None))


__all__ = [
    "forward_model",
    "forward_model_and_adjoint",
    "adjoint_forward_model",
    "compute_AtAv",
    "batch_translate_images",
]
