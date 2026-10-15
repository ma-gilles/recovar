"""CUDA-backed Fourier slice projection with JAX custom_vjp.

Provides ``cuda_project`` — a single unified projection function that handles
all four volume/image format combinations (full/half volume × full/half image)
via ``half_volume`` and ``half_image`` parameters.

The custom_vjp backward pass is the exact transpose of the forward map and calls
the CUDA backproject kernels **directly** into the matching volume layout.  The
kernels' default half-volume result is the *reconstruction* convention (the
full-volume backprojection restricted to the stored kz >= 0 voxels), which is not
a transpose; the backward asks for ``hermitian_fold=True``, the Hermitian-fold
scatter that is the exact transpose of ``half_volume_to_full_volume``, so interior
kz entries carry twice the full-volume value.  For half-image cotangents each
packed pixel is scattered once, through ``backproject_indexed``: the dense
``half_image=True`` kernel also scatters the Hermitian partner of every interior
packed pixel (the adjoint of the *full*-image operator, which is what
reconstruction wants), and would double count in a vjp.

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


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def cuda_project(
    volume,
    rotation_matrices,
    image_shape,
    volume_shape,
    order,
    half_volume,
    half_image,
    max_r=None,
    relion_texture_interp=False,
):
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
        relion_texture_interp=relion_texture_interp,
    )


def _cuda_project_fwd(
    volume,
    rotation_matrices,
    image_shape,
    volume_shape,
    order,
    half_volume,
    half_image,
    max_r,
    relion_texture_interp,
):
    out = cuda_project(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        order,
        half_volume,
        half_image,
        max_r,
        relion_texture_interp,
    )
    return out, (rotation_matrices,)


def _cuda_project_bwd(image_shape, volume_shape, order, half_volume, half_image, max_r, relion_texture_interp, res, g):
    _ = relion_texture_interp
    (rotation_matrices,) = res
    if order == 3:
        # Cubic backproject not supported in CUDA (64 atomic ops per pixel).
        # VJP of _jax_slice w.r.t. coefficients (not original volume).
        from recovar.core import slicing

        grad_vol = slicing._jax_adjoint_slice_from_coefficients(
            g, rotation_matrices, image_shape, volume_shape, half_volume=half_volume, half_image=half_image, max_r=max_r
        )
        return grad_vol, jnp.zeros_like(rotation_matrices)
    from recovar.cuda_backproject import backproject, backproject_indexed

    if half_volume:
        vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    else:
        vol_shape = volume_shape
    vol = jnp.zeros(int(np.prod(vol_shape)), dtype=g.dtype)
    # The transpose of the half-volume expansion, not the reconstruction convention of the default.
    kwargs = dict(order=order, half_volume=half_volume, half_image=False, max_r=max_r, hermitian_fold=half_volume)
    if half_image:
        # For half images, cuda_backproject.backproject expands the half image to full image
        # before scattering - resulting in double-counting the gradient w.r.t. to each internal frequency
        pixel_indices = ftu._half_image_pixel_indices(image_shape)
        grad_vol = backproject_indexed(vol, g, pixel_indices, rotation_matrices, image_shape, volume_shape, **kwargs)
    else:
        grad_vol = backproject(vol, g, rotation_matrices, image_shape, volume_shape, **kwargs)
    return grad_vol, jnp.zeros_like(rotation_matrices)


cuda_project.defvjp(_cuda_project_fwd, _cuda_project_bwd)


__all__ = ["cuda_project"]
