"""Layout conversion functions between three Fourier volume representations.

The three layouts are:

1. **Raw FFTW half-complex** -- what ``np.fft.rfftn`` produces.
   Shape ``(N, N, N//2+1)`` complex128, DC at ``[0, 0, 0]``.

2. **RELION Projector-centered** -- RELION's internal ``Projector::data``.
   Shape ``(pad_size, pad_size, pad_size//2+1)`` complex128.
   X axis left-anchored (half-complex, x >= 0).
   Y, Z axes centered (``yinit = zinit = -pad_size // 2``).

3. **recovar centered full-complex** -- ``fftshift(fftn(real_vol))``.
   Shape ``(N, N, N)`` complex128, DC at ``[N//2, N//2, N//2]``.

Additionally, the real-space conventions differ by a transpose + sign flip:
``vol_recovar = -np.transpose(vol_relion, (2, 1, 0))``.

Implementation note
-------------------
Several conversions go through real space (irfftn/rfftn round-trip) to avoid
error-prone Hermitian index bookkeeping.  This is algebraically exact for
real-valued signals and costs one extra FFT pair, which is negligible for the
offline conversion use case these functions serve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_relion_pad_size(ori_size: int, padding_factor: int = 1) -> int:
    """Compute RELION's ``pad_size`` for a given ``ori_size`` and padding factor.

    .. code-block:: text

        r_max    = ori_size // 2
        pad_size = 2 * (round(pf * r_max) + 1) + 1

    Parameters
    ----------
    ori_size : int
        Original (unpadded) box size.
    padding_factor : int
        RELION padding factor (typically 1 or 2).

    Returns
    -------
    int
        The padded box size used by RELION's Projector.
    """
    r_max = ori_size // 2
    return 2 * (round(padding_factor * r_max) + 1) + 1


def _centerfftbysign(N: int) -> NDArray[np.floating]:
    """Return the ``(-1)^(i+j+k)`` sign array for an ``(N, N, N)`` volume."""
    idx = np.arange(N)
    iz, iy, ix = np.meshgrid(idx, idx, idx, indexing="ij")
    return np.where((iz + iy + ix) % 2 == 0, 1.0, -1.0)


# ---------------------------------------------------------------------------
# FFTW half-complex  <-->  RELION Projector-centered
# ---------------------------------------------------------------------------


def fftw_half_to_relion_projector(
    fftw_half: NDArray[np.complexfloating],
    padding_factor: int = 1,
) -> NDArray[np.complexfloating]:
    """Convert raw FFTW ``rfftn`` output to RELION's Projector-centered layout.

    RELION applies *CenterFFTbySign* (multiply real-space volume by
    ``(-1)^(i+j+k)``) **before** the FFT.  This shifts DC to the center of
    the Y and Z dimensions of the half-complex output.  The x-axis remains
    left-anchored at 0.

    The conversion goes through real space to avoid Hermitian bookkeeping:

    1. ``irfftn`` to recover real space.
    2. Multiply by ``(-1)^(i+j+k)`` (CenterFFTbySign).
    3. ``rfftn`` to get the centered half-complex.
    4. Zero-pad into the ``pad_size`` array if ``padding_factor > 1``.

    Parameters
    ----------
    fftw_half : complex array, shape ``(N, N, N//2+1)``
        Raw ``np.fft.rfftn`` output (DC at ``[0, 0, 0]``).
    padding_factor : int
        RELION padding factor (1 or 2).

    Returns
    -------
    proj_data : complex array, shape ``(pad_size, pad_size, pad_size//2+1)``
        RELION ``Projector::data`` layout.
    """
    N = fftw_half.shape[0]
    assert fftw_half.shape == (N, N, N // 2 + 1), f"Expected shape (N, N, N//2+1), got {fftw_half.shape}"

    pad_size = compute_relion_pad_size(N, padding_factor)

    # Recover real-space volume, apply CenterFFTbySign, re-transform.
    vol = np.fft.irfftn(fftw_half, s=(N, N, N))
    sign = _centerfftbysign(N)
    csign_half = np.fft.rfftn(vol * sign)
    # csign_half is in FFTW physical layout: DC at [0,0,0], negative
    # frequencies at high Y,Z indices.  RELION's copy loop (projector.cpp:509)
    # remaps FFTW physical → Xmipp logical (centered Y,Z, left-anchored X).
    # We replicate this with fftshift on the Y (axis 1) and Z (axis 0) dims.
    centered_half = np.fft.fftshift(csign_half, axes=(0, 1))

    # Embed into the pad_size array (zero-pad if padding_factor > 1).
    proj_data = np.zeros(
        (pad_size, pad_size, pad_size // 2 + 1),
        dtype=fftw_half.dtype,
    )
    offset_yz = (pad_size - N) // 2
    n_x = min(N // 2 + 1, pad_size // 2 + 1)
    proj_data[
        offset_yz : offset_yz + N,
        offset_yz : offset_yz + N,
        :n_x,
    ] = centered_half[:, :, :n_x]

    return proj_data


def relion_projector_to_fftw_half(
    proj_data: NDArray[np.complexfloating],
    ori_size: int,
    padding_factor: int = 1,
) -> NDArray[np.complexfloating]:
    """Inverse of :func:`fftw_half_to_relion_projector`.

    1. Extract the ``ori_size``-sized block from the projector array.
    2. ``irfftn`` to recover ``vol * sign`` in real space.
    3. Multiply by ``(-1)^(i+j+k)`` to undo the sign (it is its own inverse).
    4. ``rfftn`` to get the standard FFTW half-complex output.

    Parameters
    ----------
    proj_data : complex array, shape ``(pad_size, pad_size, pad_size//2+1)``
        RELION ``Projector::data`` layout.
    ori_size : int
        Original box size *N*.
    padding_factor : int
        RELION padding factor used when the projector was built.

    Returns
    -------
    fftw_half : complex array, shape ``(ori_size, ori_size, ori_size//2+1)``
        Raw ``np.fft.rfftn``-compatible output.
    """
    N = ori_size
    pad_size = compute_relion_pad_size(N, padding_factor)
    assert proj_data.shape == (pad_size, pad_size, pad_size // 2 + 1), (
        f"Expected shape ({pad_size}, {pad_size}, {pad_size // 2 + 1}), got {proj_data.shape}"
    )

    # Extract the N-sized centered half-complex block.
    offset_yz = (pad_size - N) // 2
    n_x = min(N // 2 + 1, pad_size // 2 + 1)
    centered_half = np.zeros((N, N, N // 2 + 1), dtype=proj_data.dtype)
    centered_half[:, :, :n_x] = proj_data[
        offset_yz : offset_yz + N,
        offset_yz : offset_yz + N,
        :n_x,
    ]

    # Undo the Y,Z centering (inverse of fftshift in forward path).
    csign_half = np.fft.ifftshift(centered_half, axes=(0, 1))

    # Recover real space (this gives vol * sign), undo the sign, re-transform.
    vol_signed = np.fft.irfftn(csign_half, s=(N, N, N))
    sign = _centerfftbysign(N)
    vol = vol_signed * sign
    fftw_half = np.fft.rfftn(vol)

    return fftw_half


# ---------------------------------------------------------------------------
# FFTW half-complex  <-->  recovar centered full-complex
# ---------------------------------------------------------------------------


def fftw_half_to_recovar_centered(
    fftw_half: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Convert FFTW ``rfftn`` half-complex to recovar's centered full-complex.

    Steps:

    1. Expand half-complex to full complex via Hermitian symmetry
       (implemented as ``irfftn`` then ``fftn``).
    2. ``fftshift`` to center DC at ``[N//2, N//2, N//2]``.

    Parameters
    ----------
    fftw_half : complex array, shape ``(N, N, N//2+1)``
        Raw ``np.fft.rfftn`` output.

    Returns
    -------
    centered : complex array, shape ``(N, N, N)``
        recovar centered full-complex volume.
    """
    N = fftw_half.shape[0]
    assert fftw_half.shape == (N, N, N // 2 + 1), f"Expected shape (N, N, N//2+1), got {fftw_half.shape}"

    # Expand half-complex to full complex by exploiting Hermitian symmetry.
    # irfftn recovers real space exactly, then fftn gives the full complex
    # Fourier volume.  Algebraically equivalent to filling negative-x
    # frequencies via F[-kz,-ky,-kx] = conj(F[kz,ky,kx]).
    full = np.fft.fftn(np.fft.irfftn(fftw_half, s=(N, N, N)))

    # fftshift to center DC at [N//2, N//2, N//2].
    centered = np.fft.fftshift(full)
    return centered


def recovar_centered_to_fftw_half(
    centered: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Inverse of :func:`fftw_half_to_recovar_centered`.

    Steps:

    1. ``ifftshift`` to move DC back to the corner.
    2. Take half-complex slice (x >= 0 only).

    Parameters
    ----------
    centered : complex array, shape ``(N, N, N)``
        recovar centered full-complex volume.

    Returns
    -------
    fftw_half : complex array, shape ``(N, N, N//2+1)``
        Raw ``np.fft.rfftn``-compatible output.
    """
    N = centered.shape[0]
    assert centered.shape == (N, N, N), f"Expected shape (N, N, N), got {centered.shape}"

    # ifftshift to move DC from center back to corner.
    corner = np.fft.ifftshift(centered)

    # Take half-complex (x >= 0).
    fftw_half = corner[:, :, : N // 2 + 1].copy()
    return fftw_half


# ---------------------------------------------------------------------------
# RELION Projector-centered  <-->  recovar centered full-complex  (composite)
# ---------------------------------------------------------------------------


def relion_projector_to_recovar_centered(
    proj_data: NDArray[np.complexfloating],
    ori_size: int,
    padding_factor: int = 1,
) -> NDArray[np.complexfloating]:
    """Composite: RELION Projector-centered --> recovar centered full-complex.

    Parameters
    ----------
    proj_data : complex array, shape ``(pad_size, pad_size, pad_size//2+1)``
    ori_size : int
        Original box size *N*.
    padding_factor : int
        RELION padding factor.

    Returns
    -------
    centered : complex array, shape ``(ori_size, ori_size, ori_size)``
    """
    fftw_half = relion_projector_to_fftw_half(proj_data, ori_size, padding_factor)
    return fftw_half_to_recovar_centered(fftw_half)


def recovar_centered_to_relion_projector(
    centered: NDArray[np.complexfloating],
    padding_factor: int = 1,
) -> NDArray[np.complexfloating]:
    """Composite: recovar centered full-complex --> RELION Projector-centered.

    Parameters
    ----------
    centered : complex array, shape ``(N, N, N)``
    padding_factor : int
        RELION padding factor.

    Returns
    -------
    proj_data : complex array, shape ``(pad_size, pad_size, pad_size//2+1)``
    """
    fftw_half = recovar_centered_to_fftw_half(centered)
    return fftw_half_to_relion_projector(fftw_half, padding_factor)


# ---------------------------------------------------------------------------
# Real-space convention conversions
# ---------------------------------------------------------------------------


def relion_real_to_recovar_real(
    vol_relion: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convert RELION real-space volume to recovar real-space convention.

    ``vol_recovar = -np.transpose(vol_relion, (2, 1, 0))``

    Parameters
    ----------
    vol_relion : real array, shape ``(N, N, N)``

    Returns
    -------
    vol_recovar : real array, shape ``(N, N, N)``
    """
    return -np.transpose(vol_relion, (2, 1, 0))


def recovar_real_to_relion_real(
    vol_recovar: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Inverse of :func:`relion_real_to_recovar_real`.

    The transform is an involution (applying it twice returns the original).

    Parameters
    ----------
    vol_recovar : real array, shape ``(N, N, N)``

    Returns
    -------
    vol_relion : real array, shape ``(N, N, N)``
    """
    return -np.transpose(vol_recovar, (2, 1, 0))
