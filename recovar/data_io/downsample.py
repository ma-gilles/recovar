"""
2D image downsampling via Fourier cropping.

Supports both CPU (NumPy FFT) and GPU (JAX FFT) paths.  The GPU path is
selected automatically when a GPU is available, giving ~50-100x speedup
on large datasets.

The algorithm is:

    real-space image -> FFT -> fftshift -> crop centre -> ifftshift -> IFFT -> real

This is numerically equivalent to ``cryodrgn downsample``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def downsample_images(
    images: np.ndarray,
    target_D: int,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Downsample 2D images by Fourier cropping.

    Args:
        images: Real-space images, shape ``(D, D)`` or ``(N, D, D)``.
        target_D: Target box size in pixels.  Must be even and ``<= D``.
        use_gpu: If True, use JAX GPU FFT. If None, auto-detect.

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

    if use_gpu is None:
        use_gpu = _gpu_available()

    if use_gpu:
        return _downsample_gpu(images, target_D)
    return _downsample_cpu(images, target_D)


def get_downsample_batch_size(orig_D: int, gpu_memory_gb: float) -> int:
    """Compute adaptive batch size for GPU downsampling.

    Each image needs ~4x its size during FFT (input float32 + complex64 output
    + shifted + cropped). We use 50% of available GPU memory.

    Args:
        orig_D: Original image box size in pixels.
        gpu_memory_gb: Available GPU memory in GB.

    Returns:
        Batch size (clamped to [64, 16384]).
    """
    bytes_per_image = orig_D * orig_D * 4  # float32 input
    # ~4x overhead: input + complex FFT + shifted + cropped
    bytes_per_image_total = bytes_per_image * 4
    usable_bytes = gpu_memory_gb * 1e9 * 0.5
    batch = int(usable_bytes / max(bytes_per_image_total, 1))
    return max(64, min(16384, batch))


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """Check if a JAX GPU device is available."""
    try:
        import jax
        return any(d.platform == 'gpu' for d in jax.devices())
    except (ImportError, RuntimeError):
        return False


def _downsample_gpu(images: np.ndarray, target_D: int) -> np.ndarray:
    """Downsample via JAX GPU FFT."""
    import jax.numpy as jnp

    D = images.shape[-1]
    crop_start = (D - target_D) // 2
    crop_end = crop_start + target_D
    scale = (target_D / D) ** 2

    images_jax = jnp.asarray(images)

    ft = jnp.fft.fftshift(
        jnp.fft.fft2(images_jax, axes=(-2, -1)),
        axes=(-2, -1),
    )
    ft_cropped = ft[..., crop_start:crop_end, crop_start:crop_end]
    result = jnp.fft.ifft2(
        jnp.fft.ifftshift(ft_cropped, axes=(-2, -1)),
        axes=(-2, -1),
    ).real * scale

    return np.asarray(result).astype(images.dtype)


def _downsample_cpu(images: np.ndarray, target_D: int) -> np.ndarray:
    """Downsample via NumPy CPU FFT."""
    D = images.shape[-1]
    crop_start = (D - target_D) // 2
    crop_end = crop_start + target_D
    scale = (target_D / D) ** 2

    ft = np.fft.fftshift(
        np.fft.fft2(images, axes=(-2, -1)),
        axes=(-2, -1),
    )
    ft_cropped = ft[..., crop_start:crop_end, crop_start:crop_end]
    result = np.fft.ifft2(
        np.fft.ifftshift(ft_cropped, axes=(-2, -1)),
        axes=(-2, -1),
    ).real * scale

    return result.astype(images.dtype)
