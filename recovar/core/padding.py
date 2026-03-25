"""Real-space zero-padding and unpadding for Fourier oversampling."""

import functools
import logging

import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as fourier_transform_utils

logger = logging.getLogger(__name__)


# PADDING FUNCTIONS


@functools.partial(jax.jit, static_argnums=[1, 2])
def padded_dft(images, image_size, padding: int):
    n_images = images.shape[0]
    images_shape = images.shape[-2:]
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    images_big = jnp.zeros_like(images, shape=[n_images, padded_image_x, padded_image_y])
    images_big = images_big.at[
        ..., padding // 2 : images_shape[0] + padding // 2, padding // 2 : images_shape[1] + padding // 2
    ].set(images)
    padded_image_size = padded_image_x * padded_image_y
    return fourier_transform_utils.get_dft2(images_big).reshape([n_images, padded_image_size])


@functools.partial(jax.jit, static_argnums=[1, 2])
def padded_rfft(images, image_size, padding: int):
    """Pad real-space images and compute real FFT → half-spectrum output.

    Like :func:`padded_dft` but uses rfft2, producing flattened half-spectrum
    images of shape ``(n_images, H * (W // 2 + 1))``.
    """
    if images.dtype == jnp.float16:
        images = images.astype(jnp.float32)
    images_big = pad_images_spatial_domain(images, padding)
    return fourier_transform_utils.get_dft2_real(images_big).reshape([images.shape[0], -1])


def pad_images_spatial_domain(images, padding):
    n_images = images.shape[0]
    images_shape = images.shape[-2:]
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    images_big = jnp.zeros_like(images, shape=[n_images, padded_image_x, padded_image_y])
    images_big = images_big.at[
        ..., padding // 2 : images_shape[0] + padding // 2, padding // 2 : images_shape[1] + padding // 2
    ].set(images)
    return images_big


def pad_volume_spatial_domain(images, padding):
    images_shape = images.shape
    padded_image_x = images_shape[0] + padding
    padded_image_y = images_shape[1] + padding
    padded_image_z = images_shape[2] + padding
    images_big = jnp.zeros_like(images, shape=[padded_image_x, padded_image_y, padded_image_z])
    images_big = images_big.at[
        padding // 2 : images_shape[0] + padding // 2,
        padding // 2 : images_shape[1] + padding // 2,
        padding // 2 : images_shape[2] + padding // 2,
    ].set(images)
    return images_big


def unpad_volume_spatial_domain(volume, padding):
    return volume[
        padding // 2 : volume.shape[0] - padding // 2,
        padding // 2 : volume.shape[1] - padding // 2,
        padding // 2 : volume.shape[2] - padding // 2,
    ]


def unpad_volume_fourier_domain(volume, padded_image_shape, padding):
    volume = volume.reshape(list(padded_image_shape))
    volume = fourier_transform_utils.get_idft3(volume)
    unpadded_volume = unpad_volume_spatial_domain(volume, padding)
    unpadded_volume = fourier_transform_utils.get_dft3(unpadded_volume).reshape(-1)
    return unpadded_volume


def pad_images_fourier_domain(images, image_shape, padding):
    images = images.reshape([-1] + list(image_shape))
    images = fourier_transform_utils.get_idft2(images)
    padded_images = pad_images_spatial_domain(images, padding)
    return fourier_transform_utils.get_dft2(padded_images).reshape([images.shape[0], -1])


def unpad_images_spatial_domain(images, padding):
    return images[..., padding // 2 : images.shape[-2] - padding // 2, padding // 2 : images.shape[-1] - padding // 2]


def unpad_images_fourier_domain(images, padded_image_shape, padding):
    images = images.reshape([images.shape[0]] + list(padded_image_shape))
    images = fourier_transform_utils.get_idft2(images)
    unpadded_images = unpad_images_spatial_domain(images, padding)
    unpadded_images = fourier_transform_utils.get_dft2(unpadded_images).reshape([images.shape[0], -1])
    return unpadded_images


### PROJECTIONS BY NUFFT
