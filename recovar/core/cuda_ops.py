"""CUDA-backed Fourier slice operations with JAX custom_vjp rules.

Exposed wrappers:
  ``cuda_slice_full``                        – full volume → full image
  ``cuda_slice_to_half_image``               – full volume → half image (rfft)
  ``cuda_slice_from_half_vol``               – half volume (rfft) → full image
  ``cuda_slice_from_half_vol_to_half_image`` – half volume (rfft) → half image (rfft)

Each wrapper defines a ``custom_vjp`` that uses the CUDA backproject kernel
for the backward pass.  For half-volume backward, the VJP goes through
full-volume backproject + VJP of ``half_volume_to_full_volume``.

Import this module into ``slicing.py``; do not import ``cuda_backproject``
directly from user-facing modules.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import vjp

import recovar.core.fourier_transform_utils as fourier_transform_utils


# ── full volume → full image ──────────────────────────────────────────

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def cuda_slice_full(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project full volume to full images via CUDA kernel."""
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=order)


def _cuda_slice_full_fwd(volume, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    out = cuda_project(volume, rotation_matrices, image_shape, volume_shape, order=order)
    return out, (rotation_matrices,)


def _cuda_slice_full_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    volume = jnp.zeros(int(np.prod(volume_shape)), dtype=g.dtype)
    grad_vol = cuda_backproject(volume, g, rotation_matrices, image_shape, volume_shape, order=order)
    return grad_vol, jnp.zeros_like(rotation_matrices)


cuda_slice_full.defvjp(_cuda_slice_full_fwd, _cuda_slice_full_bwd)


# ── full volume → half image (rfft) ──────────────────────────────────

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def cuda_slice_to_half_image(volume, rotation_matrices, image_shape, volume_shape, order):
    """Project full volume to rfft-packed half images via CUDA kernel."""
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                        order=order, half_image=True)


def _cuda_slice_to_half_image_fwd(volume, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    out = cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                       order=order, half_image=True)
    return out, (rotation_matrices,)


def _cuda_slice_to_half_image_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    volume = jnp.zeros(int(np.prod(volume_shape)), dtype=g.dtype)
    grad_vol = cuda_backproject(volume, g, rotation_matrices, image_shape, volume_shape,
                                order=order, half_image=True)
    return grad_vol, jnp.zeros_like(rotation_matrices)


cuda_slice_to_half_image.defvjp(_cuda_slice_to_half_image_fwd, _cuda_slice_to_half_image_bwd)


# ── half volume (rfft) → full image ──────────────────────────────────

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def cuda_slice_from_half_vol(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    """Project rfft-packed half volume to full images via CUDA kernel."""
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                        order=order, half_volume=True, half_image=False)


def _cuda_slice_from_half_vol_fwd(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    out = cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                       order=order, half_volume=True, half_image=False)
    return out, (rotation_matrices,)


def _cuda_slice_from_half_vol_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    # Backproject to full-volume gradient, then chain through half→full expansion VJP.
    full_vol = jnp.zeros(int(np.prod(volume_shape)), dtype=g.dtype)
    full_grad = cuda_backproject(full_vol, g, rotation_matrices, image_shape, volume_shape, order=order)
    half_vol_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    _, vjp_expand = vjp(
        lambda hv: fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
        jnp.zeros(half_vol_size, dtype=g.dtype),
    )
    return vjp_expand(full_grad)[0], jnp.zeros_like(rotation_matrices)


cuda_slice_from_half_vol.defvjp(_cuda_slice_from_half_vol_fwd, _cuda_slice_from_half_vol_bwd)


# ── half volume (rfft) → half image (rfft) ───────────────────────────

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def cuda_slice_from_half_vol_to_half_image(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    """Project rfft-packed half volume to rfft-packed half images via CUDA kernel."""
    from recovar.cuda_backproject import project as cuda_project
    return cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                        order=order, half_volume=True, half_image=True)


def _cuda_slice_from_half_vol_to_half_image_fwd(half_volume_flat, rotation_matrices, image_shape, volume_shape, order):
    from recovar.cuda_backproject import project as cuda_project
    out = cuda_project(half_volume_flat, rotation_matrices, image_shape, volume_shape,
                       order=order, half_volume=True, half_image=True)
    return out, (rotation_matrices,)


def _cuda_slice_from_half_vol_to_half_image_bwd(image_shape, volume_shape, order, res, g):
    from recovar.cuda_backproject import backproject as cuda_backproject
    (rotation_matrices,) = res
    # Backproject half-images into full volume, then fold to half via VJP of expand.
    full_vol = jnp.zeros(int(np.prod(volume_shape)), dtype=g.dtype)
    full_grad = cuda_backproject(full_vol, g, rotation_matrices, image_shape, volume_shape,
                                 order=order, half_image=True)
    half_vol_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    _, vjp_expand = vjp(
        lambda hv: fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape),
        jnp.zeros(half_vol_size, dtype=g.dtype),
    )
    return vjp_expand(full_grad)[0], jnp.zeros_like(rotation_matrices)


cuda_slice_from_half_vol_to_half_image.defvjp(
    _cuda_slice_from_half_vol_to_half_image_fwd, _cuda_slice_from_half_vol_to_half_image_bwd
)


__all__ = [
    "cuda_slice_full",
    "cuda_slice_to_half_image",
    "cuda_slice_from_half_vol",
    "cuda_slice_from_half_vol_to_half_image",
]
