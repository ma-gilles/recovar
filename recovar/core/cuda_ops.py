"""CUDA-backed Fourier slice projection with JAX custom_vjp.

Provides ``cuda_project`` — a single unified projection function that handles
all four volume/image format combinations (full/half volume × full/half image)
via ``half_volume`` and ``half_image`` parameters.

The custom_vjp backward pass calls the CUDA backproject kernel **directly**
with matching half_volume/half_image flags.  For half-volume outputs, the
backward uses Hermitian-fold scatter with CONJ_MODE optimization (~2x scatter
speedup for HALF_IMG + HALF_VOL: doubles primary weights on interior kz and
skips redundant conjugate scatters).

IMPORTANT: Do NOT replace the backward with full-volume backproject + contract.
That would lose half-volume memory savings and the ~2x CONJ_MODE speedup.

Import from ``slicing.py``; do not import ``cuda_backproject`` directly from
user-facing modules.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def cuda_project(volume, rotation_matrices, image_shape, volume_shape, order,
                 half_volume, half_image):
    """Project volume to images via CUDA kernel (all half-vol/half-img combos)."""
    from recovar.cuda_backproject import project
    return project(volume, rotation_matrices, image_shape, volume_shape,
                   order=order, half_volume=half_volume, half_image=half_image)


def _cuda_project_fwd(volume, rotation_matrices, image_shape, volume_shape, order,
                      half_volume, half_image):
    out = cuda_project(volume, rotation_matrices, image_shape, volume_shape,
                       order, half_volume, half_image)
    return out, (rotation_matrices,)


def _cuda_project_bwd(image_shape, volume_shape, order, half_volume, half_image, res, g):
    from recovar.cuda_backproject import backproject
    (rotation_matrices,) = res
    if half_volume:
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    else:
        vol_shape = volume_shape
    vol = jnp.zeros(int(np.prod(vol_shape)), dtype=g.dtype)
    grad_vol = backproject(vol, g, rotation_matrices, image_shape, volume_shape,
                           order=order, half_volume=half_volume, half_image=half_image)
    return grad_vol, jnp.zeros_like(rotation_matrices)


cuda_project.defvjp(_cuda_project_fwd, _cuda_project_bwd)


__all__ = ["cuda_project"]
