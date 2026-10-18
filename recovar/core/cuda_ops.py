"""CUDA-backed Fourier slice projection with JAX custom_vjp.

Provides ``cuda_project`` — a single unified projection function that handles
all four volume/image format combinations (full/half volume × full/half image)
via ``half_volume`` and ``half_image`` parameters.

The custom_vjp backward pass is the exact transpose of the forward map and calls
the CUDA backproject kernel **directly** into the matching volume layout.  The
kernel's default half-volume result is the *reconstruction* convention (the
full-volume backprojection restricted to the stored kz >= 0 voxels), which is not
a transpose; the backward asks for ``hermitian_fold=True``, the Hermitian-fold
scatter that is the exact transpose of ``half_volume_to_full_volume``, so interior
kz entries carry twice the full-volume value.  A half-image cotangent is placed
at its packed pixels of a zero full image (the transpose of the packing) and
scattered as a full image: the ``half_image=True`` kernel also scatters the
Hermitian partner of every interior packed pixel (the adjoint of the *full*-image
operator, which is what reconstruction wants), and would double count in a vjp.

IMPORTANT: Do NOT replace the backward with full-volume backproject + contract.
That would lose the half-volume memory savings.

Import from ``slicing.py``; do not import ``cuda_backproject`` directly from
user-facing modules.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7))
def cuda_project(volume, rotation_matrices, image_shape, volume_shape, order, half_volume, half_image, max_r=None):
    """Project volume to images via CUDA kernel (all half-vol/half-img combos)."""
    from recovar.cuda_backproject import project

    return project(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        order=order,
        half_volume=half_volume,
        half_image=half_image,
        max_r=max_r,
    )


def _cuda_project_fwd(volume, rotation_matrices, image_shape, volume_shape, order, half_volume, half_image, max_r):
    out = cuda_project(volume, rotation_matrices, image_shape, volume_shape, order, half_volume, half_image, max_r)
    return out, (rotation_matrices,)


def _cuda_project_bwd(image_shape, volume_shape, order, half_volume, half_image, max_r, res, g):
    (rotation_matrices,) = res
    if order == 3:
        # Cubic backproject not supported in CUDA (64 atomic ops per pixel).
        # VJP of _jax_slice w.r.t. coefficients (not original volume).
        from recovar.core import slicing

        grad_vol = slicing._jax_adjoint_slice_from_coefficients(
            g, rotation_matrices, image_shape, volume_shape, half_volume=half_volume, half_image=half_image, max_r=max_r
        )
        return grad_vol, jnp.zeros_like(rotation_matrices)
    from recovar.cuda_backproject import backproject

    if half_volume:
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    else:
        vol_shape = volume_shape
    vol = jnp.zeros(int(np.prod(vol_shape)), dtype=g.dtype)
    if half_image:
        # Transpose of the packing: each packed pixel's cotangent goes to its own
        # full-grid pixel once; the Hermitian partner pixels are never read by the
        # forward map and stay zero.
        pixel_indices = ftu._half_image_pixel_indices(image_shape)
        g = jnp.zeros((g.shape[0], int(np.prod(image_shape))), dtype=g.dtype).at[:, pixel_indices].set(g)
    grad_vol = backproject(
        vol,
        g,
        rotation_matrices,
        image_shape,
        volume_shape,
        order=order,
        half_volume=half_volume,
        half_image=False,
        max_r=max_r,
        hermitian_fold=half_volume,
    )
    return grad_vol, jnp.zeros_like(rotation_matrices)


cuda_project.defvjp(_cuda_project_fwd, _cuda_project_bwd)


__all__ = ["cuda_project"]
