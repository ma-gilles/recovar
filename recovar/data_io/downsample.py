"""
2D image downsampling via Fourier cropping.

Uses NumPy FFT (CPU-only) to keep GPU memory free for the reconstruction
pipeline.  The algorithm is:

    real-space image → FFT → fftshift → crop centre → ifftshift → IFFT → real

This is numerically equivalent to ``cryodrgn downsample``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def downsample_images(
    images: np.ndarray,
    target_D: int,
) -> np.ndarray:
    """Downsample 2D images by Fourier cropping.

    Args:
        images: Real-space images, shape ``(D, D)`` or ``(N, D, D)``.
        target_D: Target box size in pixels.  Must be even and ``<= D``.

    Returns:
        Downsampled images with the last two dimensions equal to *target_D*.
    """
    if images.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got ndim={images.ndim}")

    D = images.shape[-1]
    if images.shape[-2] != D:
        raise ValueError(f"Images must be square, got shape {images.shape[-2:]}")
    if target_D > D:
        raise ValueError(f"target_D ({target_D}) must be <= image size ({D})")
    if target_D < 1:
        raise ValueError("target_D must be >= 1")
    if target_D % 2 != 0:
        raise ValueError(f"target_D must be even, got {target_D}")

    # Identity case
    if target_D == D:
        return images.copy()

    # FFT → shift so DC is in the centre → crop → unshift → IFFT
    ft = np.fft.fftshift(
        np.fft.fft2(images, axes=(-2, -1)),
        axes=(-2, -1),
    )

    crop_start = (D - target_D) // 2
    crop_end = crop_start + target_D
    ft_cropped = ft[..., crop_start:crop_end, crop_start:crop_end]

    result = np.fft.ifft2(
        np.fft.ifftshift(ft_cropped, axes=(-2, -1)),
        axes=(-2, -1),
    ).real

    # Compensate for FFT normalization: fft2 sums D² terms but ifft2
    # divides by target_D², so the result is scaled by (D/target_D)².
    # Multiply by (target_D/D)² to preserve pixel values (e.g. mean).
    result *= (target_D / D) ** 2

    return result.astype(images.dtype)


def downsample_images_batch(
    images: np.ndarray,
    target_D: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """Downsample a large stack in batches to limit peak memory.

    Args:
        images: ``(N, D, D)`` image stack.
        target_D: Target box size.
        batch_size: Number of images per batch.

    Returns:
        ``(N, target_D, target_D)`` downsampled stack.
    """
    if images.ndim != 3:
        raise ValueError(f"Expected 3D array (N, D, D), got ndim={images.ndim}")

    N = images.shape[0]
    out = np.empty((N, target_D, target_D), dtype=images.dtype)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        out[start:end] = downsample_images(images[start:end], target_D)

    return out
