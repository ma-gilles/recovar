"""CUDA-backed Fourier slice operations with JAX custom_vjp rules.

The ``cuda_slice_full`` wrapper covers the validated code path:
  full volume → full image (forward/backward both verified numerically).

The CUDA kernels for half_image=True (forward project) and half_volume=True
(backward project) are not yet numerically validated and are therefore not
exposed here.  Use ``slice_volume_by_map_to_half_image`` (which projects to
full then extracts half) and ``adjoint_slice_volume_by_map(half_volume=True)``
(which uses full backproject + VJP of the expand function) instead.

Import this module into ``slicing.py``; do not import ``cuda_backproject``
directly from user-facing modules.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


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


__all__ = [
    "cuda_slice_full",
]
